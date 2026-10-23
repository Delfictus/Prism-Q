//! Runtime-native Thermodynamic Salience Hysteresis Stage II observer.
//!
//! This module is observational only. It consumes runtime scientific streams and measured
//! `ScientificUncertaintyTensor` references, emits transition evidence objects, and never exposes
//! force, timestep, sampling, energy, or trajectory mutation controls.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const RUNTIME_ORIGIN: &str = "RUNTIME_INLINE_TSH_STAGE_II_OBSERVER";

// ═══════════════════════════════════════════════════════════════════════════
// Scale-free spectral recurrence — UNITS AND INVARIANCE
// ═══════════════════════════════════════════════════════════════════════════
//
// Read this before changing the metric.
//
// The quantity this gate needs is STRUCTURAL: do two runtime streams recur
// with the same shape along their index axis? Magnitude statistics cannot
// answer that. `mean_abs` of a force series carries kJ/mol/nm; `mean_abs` of a
// KCC descriptor is a dimensionless per-residue score. Differencing them is
// dimensionally meaningless, and because the two live on different scales the
// difference is dominated by whichever happens to be larger — such a metric
// measures "how big is the force file", not "do these two recur alike".
//
// The metric implemented here is the normalized autocorrelation signature:
//
//     r_k = Σ_i (x_i − μ)(x_{i+k} − μ) / Σ_i (x_i − μ)²        for k = 1..K
//
// PROPERTIES the gate depends on:
//   * UNITS — dimensionless. Numerator and denominator are second moments of
//     the same quantity, so every physical unit cancels exactly. Force and KCC
//     signatures are therefore comparable without a unit conversion existing.
//   * INVARIANCE — r_k is unchanged by x → a·x + b for every real a ≠ 0 and
//     every real b. Re-expressing a force field from kJ/mol/nm to kcal/mol/Å,
//     or re-zeroing a descriptor, leaves the signature bit-identical up to
//     floating-point rounding. This affine invariance is the scale-freedom the
//     comparison requires: SHAPE is compared, MAGNITUDE is not.
//   * RANGE — r_k ∈ [−1, 1].
//   * MEANING — by Wiener–Khinchin the autocorrelation sequence is the inverse
//     Fourier transform of the power spectral density, so {r_1..r_K} is the
//     low-frequency band of the normalized power spectrum. Agreement of the
//     r-vectors is agreement of that band. "Spectral recurrence" therefore
//     names what is actually computed. It is a BAND-LIMITED spectral statistic
//     and the receipts never claim a full spectrum.
//
// The comparison is the mean absolute signature difference, halved so that the
// per-lag maximum separation of 2 maps to 1:
//
//     delta = (1/K) · Σ_k |r_k^a − r_k^b| / 2                  ∈ [0, 1]
//     score = 1 − delta                                        ∈ [0, 1]
//
// `score` spans its entire range: identical shape → 1.0, exactly opposed shape
// → 0.0, so a threshold anywhere in (0, 1] is reachable from both sides. The
// superseded form `1 / (1 + |mean_abs_force − mean_abs_kcc|)` over an unbounded
// raw-magnitude delta could only ever reach ~1e-3 against a 0.5 threshold: the
// gate was arithmetically unreachable for every possible input, and no choice
// of threshold could repair it because the defect was in the metric.
//
// REALIZABILITY — the 0.0 end of that range is a property of the vector
// metric, not of real signals. `score = 0` needs r_k^b = −r_k^a at every lag,
// and a lag-wise sign flip of a real signature is generally not itself a real
// signature: an autocorrelation sequence is positive semidefinite, so
// r_1 ≈ −1 forces r_2 ≈ +1. A strict ±1 alternation is therefore NOT opposed
// to a smooth series — it agrees with it at every even lag and lands on the
// metric midpoint, score = 0.5 + O(1/n). Realizable opposition is an
// oscillation whose half-period falls inside the band: period ≈ 12 samples
// puts r_6 ≈ −1 and scores ≈ 0.35 against a ramp, which is also where a scan
// over sinusoid periods against a ramp bottoms out for K = 8. The failing side
// of the 0.5 default is therefore reachable with margin.
//
// When either signature is undefined — a constant series, too few samples, an
// absent artifact, or a declared on-disk schema this build cannot decode —
// there is NO measurement. No number is substituted, because a substituted
// number would pass as a measurement. The reason travels out of band in
// `spectral_signature_status`, and the numeric fields carry a non-finite value
// that serializes to JSON `null` rather than to any value inside [0, 1].

/// Number of autocorrelation lags forming the band-limited spectral signature.
///
/// Eight is a documented cut, not a tuned parameter: it is the low-frequency
/// band of the normalized power spectral density (see the module note above).
pub const SPECTRAL_SIGNATURE_LAGS: usize = 8;

/// Documented source-literal default for the spectral-recurrence gate.
///
/// A score of 0.5 is the midpoint of the metric's full [0, 1] range: it admits
/// signatures whose mean per-lag disagreement is at most one unit of the
/// two-unit maximum. Policy may override it; see
/// [`crate::tsh_consensus_gate::resolve_spectral_recurrence_threshold`].
pub const DEFAULT_SPECTRAL_RECURRENCE_THRESHOLD: f64 = 0.5;

/// Minimum decoded samples before any lag is considered estimable.
const SPECTRAL_MIN_SAMPLES: usize = SPECTRAL_SIGNATURE_LAGS * 4;

/// A series whose centred sum of squares is at or below this fraction of its
/// raw sum of squares is constant to within floating-point noise. Normalized
/// autocorrelation is 0/0 there, and 0/0 is not a measurement.
const SPECTRAL_DEGENERATE_VARIANCE_RATIO: f64 = 1e-12;

/// Whole-file decode budget. Files at or under this size are read in full and
/// the receipt records `whole_file`.
const SPECTRAL_FULL_READ_BUDGET_BYTES: u64 = 64 * 1024 * 1024;

/// Total bytes decoded when a file exceeds the whole-file budget.
const SPECTRAL_SAMPLE_BUDGET_BYTES: u64 = 16 * 1024 * 1024;

/// Size of each contiguous block in a strided sample. Blocks are contiguous so
/// that lag products inside a block reflect real adjacency in the file.
const SPECTRAL_SAMPLE_BLOCK_BYTES: u64 = 256 * 1024;

/// TSH runtime observer configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TshRuntimeConfig {
    /// PRISM run identifier.
    pub run_id: String,
    /// Target or topology stem.
    pub target_id: String,
    /// Minimum distinct seeds required before a persistent transition object may enter the cache.
    pub min_consensus_seeds: usize,
    /// Stable topology hash before the observed transition interval.
    pub topology_hash_before: String,
    /// Stable topology hash after the observed transition interval.
    pub topology_hash_after: String,
    /// Baseline physics signature used for non-interference comparison.
    pub baseline_physics_signature: String,
    /// Enabled-run physics signature used for non-interference comparison.
    pub enabled_physics_signature: String,
}

impl TshRuntimeConfig {
    /// Build a minimal config for a runtime row.
    pub fn new(run_id: impl Into<String>, target_id: impl Into<String>) -> Self {
        Self {
            run_id: run_id.into(),
            target_id: target_id.into(),
            min_consensus_seeds: 2,
            topology_hash_before: "topology:unknown".to_string(),
            topology_hash_after: "topology:unknown".to_string(),
            baseline_physics_signature: "not_measured".to_string(),
            enabled_physics_signature: "not_measured".to_string(),
        }
    }
}

/// First-class reference to a measured ScientificUncertaintyTensor record.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScientificUncertaintyRef {
    /// Tensor identifier.
    pub tensor_id: String,
    /// Observable name, for example `force`, `pocket_volume`, or `contact_probability`.
    pub observable: String,
    /// Observable family.
    pub observable_family: String,
    /// Measured sigma used for normalization.
    pub sigma: f64,
    /// Number of measured source samples.
    pub n_samples: u64,
    /// Measurement status.
    pub measurement_status: String,
    /// Source status.
    pub source_status: String,
    /// Source family. `NoiseFloor` is rejected as primary gate evidence.
    pub source_family: String,
    /// Source artifact hash.
    pub source_sha256: String,
    /// Source references.
    pub source_refs: Vec<String>,
    /// Source schema kind.
    pub schema_kind: String,
}

impl ScientificUncertaintyRef {
    /// Validate that this uncertainty ref is measured scientific uncertainty evidence, not legacy
    /// NoiseFloor or prior-only evidence.
    pub fn validate_for_tsh(&self) -> Result<()> {
        let schema = self.schema_kind.to_ascii_lowercase();
        if !schema.contains("uncertainty") || schema.contains("noise_floor") {
            bail!("TSH requires ScientificUncertaintyTensor schema evidence");
        }
        if self.source_family.eq_ignore_ascii_case("NoiseFloor")
            || self
                .source_family
                .eq_ignore_ascii_case("NoiseFloorLegacyAdapter")
        {
            bail!("legacy NoiseFloor cannot normalize TSH salience");
        }
        if !matches!(
            self.measurement_status.as_str(),
            "measured" | "derived_from_measured"
        ) {
            bail!("TSH uncertainty ref is not measured-derived");
        }
        if self.n_samples == 0 {
            bail!("TSH uncertainty ref has n_samples=0");
        }
        if !self.sigma.is_finite() || self.sigma <= 0.0 {
            bail!("TSH uncertainty ref sigma must be finite and positive");
        }
        if self.source_sha256.trim().is_empty() || self.source_refs.is_empty() {
            bail!("TSH uncertainty ref lacks source hashes/refs");
        }
        Ok(())
    }
}

/// One runtime observable transition interval prepared for TSH salience evaluation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TshObservableInput {
    /// Exact per-interval tensor IDs; empty retains unambiguous legacy resolution.
    #[serde(default)]
    pub uncertainty_tensor_refs: Vec<String>,
    /// Runtime stream id.
    pub stream_id: String,
    /// Runtime seed id.
    pub seed_id: String,
    /// Epoch id.
    pub epoch_id: String,
    /// Chronology event id.
    pub chronology_event_id: String,
    /// Chronology reference.
    pub chronology_ref: String,
    /// First step in interval.
    pub step_start: u64,
    /// Last step in interval.
    pub step_end: u64,
    /// Start time in ps.
    pub time_start_ps: f64,
    /// End time in ps.
    pub time_end_ps: f64,
    /// Participating residue set hash.
    pub residue_set_hash: String,
    /// Spectral signature hash.
    pub spectral_signature_hash: String,
    /// Force delta.
    pub force_delta: f64,
    /// Pocket volume delta.
    pub pocket_volume_delta: f64,
    /// Contact probability delta.
    pub contact_delta: f64,
    /// Lag, in samples, at which the contact-support autocorrelation peaks.
    /// Non-finite when the signature is undefined.
    pub causal_lag: f64,
    /// Normalized autocorrelation at `causal_lag`. Dimensionless, [-1, 1] when
    /// measured, non-finite when the signature is undefined.
    pub lag_corr_peak: f64,
    /// Mean absolute per-lag difference between the two normalized
    /// autocorrelation signatures, halved. Dimensionless and invariant under
    /// `x -> a*x + b` on either signal; [0, 1] when measured, non-finite
    /// otherwise. See the module note on units and invariance.
    pub spectral_signature_delta: f64,
    /// `measured`, or the specific reason `spectral_signature_delta` is not a
    /// measurement. Out of band from the numeric field on purpose.
    #[serde(default = "spectral_status_unset")]
    pub spectral_signature_status: String,
    /// Number of autocorrelation lags compared. Zero when not measured.
    #[serde(default)]
    pub spectral_signature_lags: usize,
    /// How the force artifact was read and which declared schema was honoured.
    #[serde(default)]
    pub force_signal_provenance: BTreeMap<String, String>,
    /// How the contact-support artifact was read and which declared schema was
    /// honoured.
    #[serde(default)]
    pub contact_signal_provenance: BTreeMap<String, String>,
    /// Source artifact references.
    pub source_refs: Vec<String>,
    /// Combined source sha256.
    pub source_sha256: String,
}

/// Deserialization default for records written before the spectral status field
/// existed. Never treated as a measurement.
fn spectral_status_unset() -> String {
    "not_measurable_status_absent_from_record".to_string()
}

/// Name of the implemented spectral metric.
fn spectral_metric_kind() -> String {
    format!("normalized_autocorrelation_signature_l1_lags_{SPECTRAL_SIGNATURE_LAGS}")
}

/// Units of the spectral metric.
fn spectral_metric_units() -> String {
    "dimensionless_ratio_of_second_moments".to_string()
}

/// The invariance the spectral metric guarantees.
fn spectral_metric_invariance() -> String {
    "invariant_under_affine_rescaling_x_to_a_x_plus_b_for_a_nonzero".to_string()
}

/// Whether the TSH salience producer ran at all, and what it found.
///
/// C6 no-producer marker. `salience_event_count == 0` is ambiguous on its own,
/// because zero events is a legitimate scientific outcome. This report resolves
/// the ambiguity with a typed status that lives outside the count's value
/// space, so a training consumer never has to infer wiring from a number.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SalienceProducerReport {
    /// Schema kind.
    pub schema_kind: String,
    /// Schema version.
    pub schema_version: u32,
    /// One of [`SALIENCE_PRODUCER_NOT_WIRED`],
    /// [`SALIENCE_PRODUCER_WIRED_NO_INPUTS`],
    /// [`SALIENCE_PRODUCER_WIRED_ZERO_EVENTS`],
    /// [`SALIENCE_PRODUCER_WIRED_EVENTS_EMITTED`].
    pub producer_status: String,
    /// Whether a producer path executed at all. `false` means absence of events
    /// carries no scientific information.
    pub producer_wired: bool,
    /// Human-readable reason for `producer_status`.
    pub producer_status_reason: String,
    /// Runtime artifact refs the producer was offered.
    pub artifact_refs_offered: usize,
    /// Distinct present artifact kinds observed.
    pub artifact_kinds_present: Vec<String>,
    /// Required artifact kinds that were absent.
    pub required_artifact_kinds_missing: Vec<String>,
    /// Streams that reached observable-input construction.
    pub candidate_stream_count: usize,
    /// Observable inputs actually built.
    pub observable_input_count: usize,
    /// Salience events actually emitted. Meaningful only when `producer_wired`.
    pub salience_event_count: usize,
    /// Streams whose spectral comparison was a real measurement.
    pub spectral_measured_count: usize,
    /// Streams whose spectral comparison was not measurable, by reason.
    pub spectral_not_measurable_by_reason: BTreeMap<String, usize>,
    /// Runtime origin.
    pub runtime_origin: String,
}

/// No producer path executed: the consumer must not read absence as evidence.
pub const SALIENCE_PRODUCER_NOT_WIRED: &str = "producer_not_wired";
/// Artifacts were offered but none could be resolved into an observable input.
pub const SALIENCE_PRODUCER_WIRED_NO_INPUTS: &str = "producer_wired_no_observable_inputs";
/// The producer ran on real inputs and legitimately emitted nothing.
pub const SALIENCE_PRODUCER_WIRED_ZERO_EVENTS: &str = "producer_wired_zero_events";
/// The producer ran and emitted events.
pub const SALIENCE_PRODUCER_WIRED_EVENTS_EMITTED: &str = "producer_wired_events_emitted";

/// Artifact kinds the salience producer requires to build an observable input.
pub const SALIENCE_REQUIRED_ARTIFACT_KINDS: [&str; 2] = ["forces_final", "kcc_v2full"];

impl SalienceProducerReport {
    fn base(status: &str, reason: &str) -> Self {
        Self {
            schema_kind: "tsh_salience_producer_status".to_string(),
            schema_version: 1,
            producer_status: status.to_string(),
            producer_wired: status != SALIENCE_PRODUCER_NOT_WIRED,
            producer_status_reason: reason.to_string(),
            artifact_refs_offered: 0,
            artifact_kinds_present: Vec::new(),
            required_artifact_kinds_missing: Vec::new(),
            candidate_stream_count: 0,
            observable_input_count: 0,
            salience_event_count: 0,
            spectral_measured_count: 0,
            spectral_not_measurable_by_reason: BTreeMap::new(),
            runtime_origin: RUNTIME_ORIGIN.to_string(),
        }
    }

    fn from_discovery(
        artifact_refs_offered: usize,
        observed_kinds: BTreeSet<String>,
        candidate_stream_count: usize,
        observable_input_count: usize,
        spectral_measured_count: usize,
        spectral_not_measurable_by_reason: BTreeMap<String, usize>,
    ) -> Self {
        let missing: Vec<String> = SALIENCE_REQUIRED_ARTIFACT_KINDS
            .iter()
            .filter(|kind| !observed_kinds.contains(**kind))
            .map(|kind| (*kind).to_string())
            .collect();
        let (status, reason) = if artifact_refs_offered == 0 {
            (
                SALIENCE_PRODUCER_NOT_WIRED,
                "no runtime artifact refs were offered to the salience producer",
            )
        } else if candidate_stream_count == 0 {
            (
                SALIENCE_PRODUCER_NOT_WIRED,
                "runtime artifact refs carried no present, stream-tagged artifact",
            )
        } else if observable_input_count == 0 {
            (
                SALIENCE_PRODUCER_WIRED_NO_INPUTS,
                "streams were present but no stream resolved into an observable input",
            )
        } else {
            (
                SALIENCE_PRODUCER_WIRED_ZERO_EVENTS,
                "observable inputs were built; event count is set after evaluation",
            )
        };
        let mut report = Self::base(status, reason);
        report.artifact_refs_offered = artifact_refs_offered;
        report.artifact_kinds_present = observed_kinds.into_iter().collect();
        report.required_artifact_kinds_missing = missing;
        report.candidate_stream_count = candidate_stream_count;
        report.observable_input_count = observable_input_count;
        report.spectral_measured_count = spectral_measured_count;
        report.spectral_not_measurable_by_reason = spectral_not_measurable_by_reason;
        report
    }

    /// Report for a caller that supplied observable inputs directly, so no
    /// artifact discovery happened in this module.
    #[must_use]
    pub fn from_caller_supplied_inputs(observable_input_count: usize) -> Self {
        let (status, reason) = if observable_input_count == 0 {
            (
                SALIENCE_PRODUCER_NOT_WIRED,
                "caller supplied no observable inputs and no artifact discovery ran here",
            )
        } else {
            (
                SALIENCE_PRODUCER_WIRED_ZERO_EVENTS,
                "caller supplied observable inputs directly; artifact discovery ran elsewhere",
            )
        };
        let mut report = Self::base(status, reason);
        report.observable_input_count = observable_input_count;
        report.candidate_stream_count = observable_input_count;
        report
    }

    /// Stamp the emitted event count once evaluation has run.
    pub fn record_salience_events(&mut self, salience_event_count: usize) {
        self.salience_event_count = salience_event_count;
        if self.producer_wired && salience_event_count > 0 {
            self.producer_status = SALIENCE_PRODUCER_WIRED_EVENTS_EMITTED.to_string();
            self.producer_status_reason =
                "the salience producer ran and emitted events".to_string();
        } else if self.producer_wired && self.producer_status == SALIENCE_PRODUCER_WIRED_ZERO_EVENTS
        {
            self.producer_status_reason =
                "the salience producer ran on real inputs and emitted no events; absence is a result, not a wiring gap"
                    .to_string();
        }
    }
}

/// Runtime artifact reference used to build live TSH inputs from emitted surfaces.
#[derive(Clone, Debug)]
pub struct RuntimeArtifactRef {
    /// Artifact kind from `md_evidence_manifest`.
    pub kind: String,
    /// Stream id.
    pub stream_id: Option<u32>,
    /// Artifact path.
    pub path: PathBuf,
    /// Whether the artifact is present.
    pub present: bool,
}

/// TSH salience event.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ThermodynamicSalienceEvent {
    /// Schema kind.
    pub schema_kind: String,
    /// Schema version.
    pub schema_version: u32,
    /// Transition id.
    pub transition_id: String,
    /// Candidate id.
    pub candidate_id: String,
    /// Motif id.
    pub motif_id: String,
    /// Consensus id.
    pub consensus_id: String,
    /// Persistent transition object id.
    pub persistent_transition_object_id: String,
    /// Evidence ledger id.
    pub evidence_ledger_id: String,
    /// Stable identifier.
    pub stable_identifier: String,
    /// Version.
    pub version: f64,
    /// Lineage reference.
    pub lineage_ref: String,
    /// Supersession reference.
    pub supersedes_ref: String,
    /// Run id.
    pub run_id: String,
    /// Target id.
    pub target_id: String,
    /// Epoch id.
    pub epoch_id: String,
    /// Stream id.
    pub stream_id: String,
    /// Seed id.
    pub seed_id: String,
    /// Chronology event id.
    pub chronology_event_id: String,
    /// Chronology ref.
    pub chronology_ref: String,
    /// Chronology interval start.
    pub chronology_interval_start: u64,
    /// Chronology interval end.
    pub chronology_interval_end: u64,
    /// Step start.
    pub step_start: u64,
    /// Step end.
    pub step_end: u64,
    /// Time start ps.
    pub time_start_ps: f64,
    /// Time end ps.
    pub time_end_ps: f64,
    /// Scientific uncertainty tensor refs used for normalization.
    pub uncertainty_tensor_refs: Vec<String>,
    /// Primary uncertainty tensor ref.
    pub scientific_uncertainty_tensor_ref: String,
    /// Source refs.
    pub source_refs: Vec<String>,
    /// Source sha256.
    pub source_sha256: String,
    /// Topology transition hash.
    pub topology_hash_transition: String,
    /// Residue set hash.
    pub residue_set_hash: String,
    /// Spectral signature hash.
    pub spectral_signature_hash: String,
    /// Force delta.
    pub force_delta: f64,
    /// Force sigma.
    pub force_delta_sigma: f64,
    /// Force SNR.
    pub force_delta_snr: f64,
    /// Pocket volume delta.
    pub pocket_volume_delta: f64,
    /// Pocket volume sigma.
    pub pocket_volume_delta_sigma: f64,
    /// Pocket volume SNR.
    pub pocket_volume_delta_snr: f64,
    /// Contact delta.
    pub contact_delta: f64,
    /// Contact sigma.
    pub contact_delta_sigma: f64,
    /// Contact SNR.
    pub contact_delta_snr: f64,
    /// Contact normalization ontology role.
    pub contact_support_observable_role: String,
    /// Runtime contact-support observable used for contact normalization.
    pub contact_support_observable: String,
    /// Contact-support uncertainty tensor ref.
    pub contact_support_uncertainty_tensor_ref: String,
    /// Contact uncertainty resolution chain.
    pub contact_uncertainty_resolution_chain: Vec<String>,
    /// Preferred contact-probability uncertainty status.
    pub contact_probability_uncertainty_status: String,
    /// Whether contact_probability and contact-support observables are asserted semantically equivalent.
    pub contact_support_semantic_equivalence_asserted: bool,
    /// Lag, in samples, at which the contact-support autocorrelation peaks.
    pub causal_lag: f64,
    /// Normalized autocorrelation at `causal_lag`. Dimensionless.
    pub lag_corr_peak: f64,
    /// Mean absolute per-lag difference between the two normalized
    /// autocorrelation signatures, halved. Dimensionless, affine-invariant.
    pub spectral_signature_delta: f64,
    /// `measured`, or the reason no spectral comparison exists.
    #[serde(default = "spectral_status_unset")]
    pub spectral_signature_status: String,
    /// Number of autocorrelation lags compared.
    #[serde(default)]
    pub spectral_signature_lags: usize,
    /// Name of the implemented metric, so a consumer never has to infer it.
    #[serde(default = "spectral_metric_kind")]
    pub spectral_metric_kind: String,
    /// Units of `spectral_signature_delta` and `spectral_similarity_score`.
    #[serde(default = "spectral_metric_units")]
    pub spectral_metric_units: String,
    /// The invariance the metric guarantees.
    #[serde(default = "spectral_metric_invariance")]
    pub spectral_metric_invariance: String,
    /// How the force artifact was read and which declared schema was honoured.
    #[serde(default)]
    pub force_signal_provenance: BTreeMap<String, String>,
    /// How the contact-support artifact was read and which declared schema was
    /// honoured.
    #[serde(default)]
    pub contact_signal_provenance: BTreeMap<String, String>,
    /// Replicate recurrence.
    pub replicate_recurrence: f64,
    /// Replicate count.
    pub replicate_count: u64,
    /// Consensus seed count.
    pub consensus_seed_count: u64,
    /// Stability score component.
    pub transition_stability_score: f64,
    /// Recurrence score component.
    pub transition_recurrence_score: f64,
    /// Topology persistence score component.
    pub topology_persistence_score: f64,
    /// Chronology persistence score component.
    pub chronology_persistence_score: f64,
    /// Uncertainty-normalized magnitude component.
    pub uncertainty_normalized_magnitude: f64,
    /// Spectral similarity score component: `1 - spectral_signature_delta`.
    /// Dimensionless, [0, 1] when measured, non-finite (JSON `null`) otherwise.
    pub spectral_similarity_score: f64,
    /// Control rejection score component.
    #[serde(deserialize_with = "deserialize_unavailable_score")]
    pub control_rejection_score: f64,
    /// Scientific evidence score component. The mean of the components that
    /// were actually measured; an unmeasurable component is omitted from the
    /// mean rather than entered as a value.
    pub scientific_evidence_score: f64,
    /// Which components entered `scientific_evidence_score`.
    #[serde(default)]
    pub scientific_evidence_components: Vec<String>,
    /// Decomposed aggregate salience score.
    pub aggregate_salience_score: f64,
    /// Negative control status.
    pub negative_control_status: String,
    /// Acceptance status.
    pub acceptance_status: String,
    /// Runtime origin.
    pub runtime_origin: String,
    /// Tokenizer family.
    pub tokenizer_family: String,
}

/// Transition motif candidate or canonical motif row.
pub type TransitionMotifCandidate = ThermodynamicSalienceEvent;
/// Transition motif row.
pub type TransitionMotif = ThermodynamicSalienceEvent;
/// Consensus event row.
pub type TransitionConsensusEvent = ThermodynamicSalienceEvent;
/// Persistent transition object row.
pub type PersistentTransitionObject = ThermodynamicSalienceEvent;

/// Transition evidence ledger row.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransitionEvidenceLedger {
    /// Schema kind.
    pub schema_kind: String,
    /// Schema version.
    pub schema_version: u32,
    /// Ledger id.
    pub evidence_ledger_id: String,
    /// Run id.
    pub run_id: String,
    /// Status.
    pub acceptance_status: String,
    /// Event count.
    pub salience_event_count: usize,
    /// Candidate count.
    pub candidate_count: usize,
    /// Consensus event count.
    pub consensus_event_count: usize,
    /// Persistent object count.
    pub persistent_transition_object_count: usize,
    /// Whether negative controls were provided.
    pub negative_controls_present: bool,
    /// Whether only ScientificUncertaintyTensor evidence was used.
    pub scientific_uncertainty_tensor_only: bool,
    /// Whether legacy NoiseFloor was rejected.
    pub legacy_noise_floor_rejected: bool,
    /// Contact-normalization ontology decision.
    pub contact_normalization_ontology: String,
    /// Contact-normalization uncertainty resolution chain.
    pub contact_normalization_resolution_chain: Vec<String>,
    /// Runtime contact-support observable used for normalization.
    pub contact_support_observable: String,
    /// Whether the ledger asserts semantic equivalence between contact_probability and support channels.
    pub contact_support_semantic_equivalence_asserted: bool,
    /// Runtime origin.
    pub runtime_origin: String,
    /// Tokenizer family.
    pub tokenizer_family: String,
}

/// Resident transition cache.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResidentTransitionCache {
    /// Schema kind.
    pub schema_kind: String,
    /// Schema version.
    pub schema_version: u32,
    /// Cache id.
    pub cache_id: String,
    /// Runtime origin.
    pub runtime_origin: String,
    /// Persistent transition objects only.
    pub transition_objects: Vec<PersistentTransitionObject>,
    /// Non-interference guard.
    pub steering_state_present: bool,
    /// Explicit forbidden field audit.
    pub forbidden_steering_fields_present: Vec<String>,
}

/// Runtime non-interference ledger.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TshRuntimeNonInterferenceLedger {
    /// Schema kind.
    pub schema_kind: String,
    /// Schema version.
    pub schema_version: u32,
    /// Mutates forces.
    pub force_mutation: bool,
    /// Mutates dt.
    pub dt_mutation: bool,
    /// Mutates sampling.
    pub sampling_bias: bool,
    /// Mutates energy landscape.
    pub energy_landscape_steering: bool,
    /// Mutates trajectory.
    pub trajectory_mutation: bool,
    /// Baseline signature.
    pub baseline_physics_signature: String,
    /// Enabled observer signature.
    pub enabled_physics_signature: String,
    /// Signatures equivalent.
    pub physics_signature_equivalent: bool,
}

/// Stage III compatibility manifest.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TshStageIIICompatibilityManifest {
    /// Schema kind.
    pub schema_kind: String,
    /// Schema version.
    pub schema_version: u32,
    /// Compatible without schema migration.
    pub schema_migration_required: bool,
    /// Fields future Stage III can consume.
    pub stage_iii_consumable_fields: Vec<String>,
    /// Fields intentionally absent in Stage II.
    pub forbidden_stage_ii_fields: Vec<String>,
}

/// Complete TSH runtime output bundle.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TshRuntimeArtifacts {
    /// Salience events.
    pub salience_events: Vec<ThermodynamicSalienceEvent>,
    /// Candidate motifs.
    pub motif_candidates: Vec<TransitionMotifCandidate>,
    /// Canonical motifs.
    pub motifs: Vec<TransitionMotif>,
    /// Consensus events.
    pub consensus_events: Vec<TransitionConsensusEvent>,
    /// Persistent transition objects.
    pub persistent_transition_objects: Vec<PersistentTransitionObject>,
    /// Evidence ledgers.
    pub evidence_ledgers: Vec<TransitionEvidenceLedger>,
    /// Resident cache.
    pub resident_cache: ResidentTransitionCache,
    /// Non-interference ledger.
    pub non_interference_ledger: TshRuntimeNonInterferenceLedger,
    /// Stage III manifest.
    pub stage_iii_manifest: TshStageIIICompatibilityManifest,
    /// C6 no-producer marker: whether the salience producer was wired at all.
    #[serde(default = "salience_producer_report_absent")]
    pub salience_producer: SalienceProducerReport,
}

/// Deserialization default for bundles written before the producer marker
/// existed. Explicitly not-wired rather than silently absent.
fn salience_producer_report_absent() -> SalienceProducerReport {
    SalienceProducerReport::base(
        SALIENCE_PRODUCER_NOT_WIRED,
        "record predates the salience producer marker; wiring is unknown and must not be assumed",
    )
}

/// Emitted file receipt.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TshEmissionReceipt {
    /// Schema kind.
    pub schema_kind: String,
    /// Status.
    pub status: String,
    /// Written files.
    pub files: BTreeMap<String, String>,
    /// Salience event count. Ambiguous on its own — read
    /// `salience_producer_status` before interpreting a zero.
    pub salience_event_count: usize,
    /// Persistent object count.
    pub persistent_transition_object_count: usize,
    /// Negative controls present.
    pub negative_controls_present: bool,
    /// C6 no-producer marker, mirrored into the receipt so a consumer that only
    /// reads the manifest still sees it.
    #[serde(default = "spectral_status_unset")]
    pub salience_producer_status: String,
    /// Whether any producer path executed.
    #[serde(default)]
    pub salience_producer_wired: bool,
}

/// Runtime-native observer.
pub struct TshRuntimeObserver {
    config: TshRuntimeConfig,
    origin: &'static str,
}

impl TshRuntimeObserver {
    /// Construct an observer.
    pub fn new(config: TshRuntimeConfig) -> Self {
        Self { config, origin: RUNTIME_ORIGIN }
    }

    /// Observe saved measurements without claiming inline runtime execution.
    pub fn offline(config: TshRuntimeConfig) -> Self {
        Self { config, origin: "OFFLINE_MEASURED_TSH_STAGE_II_OBSERVER" }
    }

    /// Evaluate runtime transition intervals into TSH scientific objects.
    ///
    /// The emitted bundle carries a [`SalienceProducerReport`] derived from the
    /// supplied inputs. Callers that performed artifact discovery themselves
    /// should use [`Self::evaluate_with_producer_report`] so the richer,
    /// artifact-level wiring evidence survives into the receipts.
    pub fn evaluate(
        &self,
        inputs: &[TshObservableInput],
        uncertainties: &[ScientificUncertaintyRef],
        negative_control_motif_hashes: Option<&BTreeSet<String>>,
    ) -> Result<TshRuntimeArtifacts> {
        self.evaluate_with_producer_report(
            inputs,
            uncertainties,
            negative_control_motif_hashes,
            SalienceProducerReport::from_caller_supplied_inputs(inputs.len()),
        )
    }

    /// Evaluate with an externally determined salience-producer report.
    pub fn evaluate_with_producer_report(
        &self,
        inputs: &[TshObservableInput],
        uncertainties: &[ScientificUncertaintyRef],
        negative_control_motif_hashes: Option<&BTreeSet<String>>,
        mut salience_producer: SalienceProducerReport,
    ) -> Result<TshRuntimeArtifacts> {
        if inputs.is_empty() {
            bail!("TSH requires at least one runtime observable input");
        }
        let valid_uncertainties: Vec<ScientificUncertaintyRef> = uncertainties
            .iter()
            .filter(|uncertainty| uncertainty.validate_for_tsh().is_ok())
            .cloned()
            .collect();
        if valid_uncertainties.is_empty() {
            bail!("TSH requires at least one valid measured ScientificUncertaintyTensor ref");
        }
        let mut tensor_ids = BTreeSet::new();
        for u in uncertainties {
            if !tensor_ids.insert(&u.tensor_id) { bail!("duplicate uncertainty tensor ID {}", u.tensor_id); }
        }
        let mut contact_chains = BTreeSet::new();
        let mut contact_observables = BTreeSet::new();
        let negative_controls_present = negative_control_motif_hashes.is_some();
        let physics_signature_equivalent = self.config.baseline_physics_signature
            == self.config.enabled_physics_signature
            && !matches!(
                self.config.baseline_physics_signature.as_str(),
                "" | "not_measured" | "not_provided"
            );

        let mut salience_events = Vec::new();
        let mut motif_candidates = Vec::new();
        let mut grouped: BTreeMap<String, Vec<ThermodynamicSalienceEvent>> = BTreeMap::new();
        for (idx, input) in inputs.iter().enumerate() {
            if input.step_end < input.step_start || !input.time_start_ps.is_finite()
                || !input.time_end_ps.is_finite() || input.time_end_ps < input.time_start_ps
                || [input.force_delta, input.pocket_volume_delta, input.contact_delta].iter().any(|v| !v.is_finite()) {
                bail!("invalid measured TSH interval {}", input.chronology_event_id);
            }
            let selected: Vec<ScientificUncertaintyRef> = if input.uncertainty_tensor_refs.is_empty() {
                valid_uncertainties.clone()
            } else {
                let mut ids = BTreeSet::new();
                input.uncertainty_tensor_refs.iter().map(|id| {
                    if !ids.insert(id) { bail!("duplicate interval tensor binding {id}"); }
                    valid_uncertainties.iter().find(|u| &u.tensor_id == id).cloned()
                        .with_context(|| format!("missing or invalid bound uncertainty {id}"))
                }).collect::<Result<_>>()?
            };
            let force_u = uncertainty_for(&selected, "force")?;
            let pocket_u = uncertainty_for_any(&selected, &["pocket_volume", "pocket surface area"])?;
            for observable in ["contact_probability", "kcc"] {
                if selected.iter().filter(|u| u.observable.eq_ignore_ascii_case(observable)).count() > 1 {
                    bail!("ambiguous per-interval uncertainty for {observable}; bind exact tensor IDs");
                }
            }
            let contact_support = contact_support_uncertainty(uncertainties, &selected)?;
            contact_chains.extend(contact_support.resolution_chain.iter().cloned());
            contact_observables.insert(contact_support.observable.clone());
            let force_snr = snr(input.force_delta, force_u.sigma);
            let pocket_snr = snr(input.pocket_volume_delta, pocket_u.sigma);
            let contact_snr = snr(input.contact_delta, contact_support.uncertainty.sigma);
            let uncertainty_normalized_magnitude =
                ((force_snr * force_snr + pocket_snr * pocket_snr + contact_snr * contact_snr)
                    / 3.0)
                    .sqrt();
            let motif_id = motif_id_for(input, &self.config);
            let negative_status = match negative_control_motif_hashes {
                None => "missing",
                Some(controls) if controls.contains(&motif_id) => "present_in_controls",
                Some(_) => "absent_from_controls",
            }
            .to_string();
            let control_rejection_score = match negative_status.as_str() {
                "absent_from_controls" => 1.0,
                "present_in_controls" => 0.0,
                _ => f64::NAN,
            };
            let uncertainty_score = (uncertainty_normalized_magnitude / 10.0).clamp(0.0, 1.0);
            // Scale-free spectral recurrence. `spectral_signature_delta` is the
            // dimensionless, affine-invariant autocorrelation-signature distance
            // in [0, 1], so the similarity is its complement and spans the whole
            // range. It is non-finite exactly when the comparison was not a
            // measurement; a non-finite value serializes to JSON `null`, which
            // is out of band for a [0, 1] score.
            let spectral_measured = input.spectral_signature_status == SPECTRAL_STATUS_MEASURED
                && input.spectral_signature_delta.is_finite();
            let spectral_similarity_score = if spectral_measured {
                (1.0 - input.spectral_signature_delta).clamp(0.0, 1.0)
            } else {
                f64::NAN
            };
            let transition_stability_score =
                ((force_snr.min(20.0) + pocket_snr.min(20.0) + contact_snr.min(20.0)) / 60.0)
                    .clamp(0.0, 1.0);
            let chronology_persistence_score = if input.step_end >= input.step_start {
                1.0
            } else {
                0.0
            };
            let topology_persistence_score =
                (self.config.topology_hash_before == self.config.topology_hash_after) as u8 as f64;
            // An unmeasurable component is omitted from the mean rather than
            // entered as a number: averaging in a substituted value would let a
            // fabricated quantity pass as a measurement.
            let mut evidence_components: Vec<(&str, f64)> = vec![
                ("transition_stability_score", transition_stability_score),
                ("uncertainty_score", uncertainty_score),
            ];
            if spectral_measured {
                evidence_components.push(("spectral_similarity_score", spectral_similarity_score));
            }
            if negative_controls_present {
                evidence_components.push(("control_rejection_score", control_rejection_score));
            }
            evidence_components
                .push(("chronology_persistence_score", chronology_persistence_score));
            let scientific_evidence_components: Vec<String> = evidence_components
                .iter()
                .map(|(name, _)| (*name).to_string())
                .collect();
            let scientific_evidence_score = mean(
                &evidence_components
                    .iter()
                    .map(|(_, value)| *value)
                    .collect::<Vec<f64>>(),
            );
            let event = ThermodynamicSalienceEvent {
                schema_kind: "thermodynamic_salience_event".to_string(),
                schema_version: 1,
                transition_id: format!("tsh:{}:{idx}", self.config.run_id),
                candidate_id: format!("tsh_candidate:{}:{idx}", self.config.run_id),
                motif_id: motif_id.clone(),
                consensus_id: String::new(),
                persistent_transition_object_id: String::new(),
                evidence_ledger_id: format!("tsh_evidence_ledger:{}", self.config.run_id),
                stable_identifier: String::new(),
                version: 1.0,
                lineage_ref: format!("lineage:{}", self.config.run_id),
                supersedes_ref: String::new(),
                run_id: self.config.run_id.clone(),
                target_id: self.config.target_id.clone(),
                epoch_id: input.epoch_id.clone(),
                stream_id: input.stream_id.clone(),
                seed_id: input.seed_id.clone(),
                chronology_event_id: input.chronology_event_id.clone(),
                chronology_ref: input.chronology_ref.clone(),
                chronology_interval_start: input.step_start,
                chronology_interval_end: input.step_end,
                step_start: input.step_start,
                step_end: input.step_end,
                time_start_ps: input.time_start_ps,
                time_end_ps: input.time_end_ps,
                uncertainty_tensor_refs: vec![
                    force_u.tensor_id.clone(),
                    pocket_u.tensor_id.clone(),
                    contact_support.uncertainty.tensor_id.clone(),
                ],
                scientific_uncertainty_tensor_ref: force_u.tensor_id.clone(),
                source_refs: input.source_refs.clone(),
                source_sha256: input.source_sha256.clone(),
                topology_hash_transition: format!(
                    "{}->{}",
                    self.config.topology_hash_before, self.config.topology_hash_after
                ),
                residue_set_hash: input.residue_set_hash.clone(),
                spectral_signature_hash: input.spectral_signature_hash.clone(),
                force_delta: input.force_delta,
                force_delta_sigma: force_u.sigma,
                force_delta_snr: force_snr,
                pocket_volume_delta: input.pocket_volume_delta,
                pocket_volume_delta_sigma: pocket_u.sigma,
                pocket_volume_delta_snr: pocket_snr,
                contact_delta: input.contact_delta,
                contact_delta_sigma: contact_support.uncertainty.sigma,
                contact_delta_snr: contact_snr,
                contact_support_observable_role: "ContactSupportObservable".to_string(),
                contact_support_observable: contact_support.observable.clone(),
                contact_support_uncertainty_tensor_ref: contact_support
                    .uncertainty
                    .tensor_id
                    .clone(),
                contact_uncertainty_resolution_chain: contact_support.resolution_chain.clone(),
                contact_probability_uncertainty_status: contact_support
                    .contact_probability_uncertainty_status
                    .clone(),
                contact_support_semantic_equivalence_asserted: false,
                causal_lag: input.causal_lag,
                lag_corr_peak: input.lag_corr_peak,
                spectral_signature_delta: input.spectral_signature_delta,
                spectral_signature_status: input.spectral_signature_status.clone(),
                spectral_signature_lags: input.spectral_signature_lags,
                spectral_metric_kind: spectral_metric_kind(),
                spectral_metric_units: spectral_metric_units(),
                spectral_metric_invariance: spectral_metric_invariance(),
                force_signal_provenance: input.force_signal_provenance.clone(),
                contact_signal_provenance: input.contact_signal_provenance.clone(),
                replicate_recurrence: 0.0,
                replicate_count: 1,
                consensus_seed_count: 1,
                transition_stability_score,
                transition_recurrence_score: 0.0,
                topology_persistence_score,
                chronology_persistence_score,
                uncertainty_normalized_magnitude,
                spectral_similarity_score,
                control_rejection_score,
                scientific_evidence_score,
                scientific_evidence_components,
                aggregate_salience_score: scientific_evidence_score,
                negative_control_status: negative_status,
                acceptance_status: "single_seed_candidate".to_string(),
                runtime_origin: self.origin.to_string(),
                tokenizer_family: "ThermodynamicSalienceEvent".to_string(),
            };
            grouped.entry(motif_id).or_default().push(event.clone());
            motif_candidates.push(TransitionMotifCandidate {
                schema_kind: "transition_motif_candidate".to_string(),
                tokenizer_family: "TransitionMotif".to_string(),
                ..event.clone()
            });
            salience_events.push(event);
        }

        let mut motifs = Vec::new();
        let mut consensus_events = Vec::new();
        let mut persistent_transition_objects = Vec::new();
        for (motif_id, events) in grouped {
            let seeds: BTreeSet<String> = events.iter().map(|e| e.seed_id.clone()).collect();
            let representative = events[0].clone();
            let seed_count = seeds.len() as u64;
            let recurrence = if self.config.min_consensus_seeds == 0 {
                1.0
            } else {
                (seeds.len() as f64 / self.config.min_consensus_seeds as f64).min(1.0)
            };
            let negative_status = representative.negative_control_status.clone();
            let accepted = seed_count as usize >= self.config.min_consensus_seeds
                && negative_status == "absent_from_controls";
            let rejected_by_control = negative_status == "present_in_controls";
            let status = if accepted {
                "persistent"
            } else if rejected_by_control {
                "rejected"
            } else if !negative_controls_present {
                "blocked"
            } else {
                "consensus_candidate"
            };
            let mut consensus = representative.clone();
            consensus.schema_kind = "transition_consensus_event".to_string();
            consensus.consensus_seed_count = seed_count;
            consensus.replicate_count = seed_count;
            consensus.replicate_recurrence = recurrence;
            consensus.transition_recurrence_score = recurrence;
            consensus.acceptance_status = status.to_string();
            consensus.tokenizer_family = "TransitionConsensusEvent".to_string();
            consensus.consensus_id = format!("tsh_consensus:{}:{}", self.config.run_id, motif_id);
            consensus_events.push(consensus.clone());

            let mut motif = consensus.clone();
            motif.schema_kind = "transition_motif".to_string();
            motif.tokenizer_family = "TransitionMotif".to_string();
            motifs.push(motif);

            if accepted {
                let mut persistent = consensus;
                persistent.schema_kind = "persistent_transition_object".to_string();
                persistent.persistent_transition_object_id =
                    format!("pto:{}:{}", self.config.run_id, motif_id);
                persistent.stable_identifier =
                    format!("tsh_stage_ii:{}:{}", self.config.target_id, motif_id);
                persistent.version = 1.0;
                persistent.tokenizer_family = "PersistentTransitionObject".to_string();
                persistent_transition_objects.push(persistent);
            }
        }

        let evidence_ledger = TransitionEvidenceLedger {
            schema_kind: "transition_evidence_ledger".to_string(),
            schema_version: 1,
            evidence_ledger_id: format!("tsh_evidence_ledger:{}", self.config.run_id),
            run_id: self.config.run_id.clone(),
            acceptance_status: if !persistent_transition_objects.is_empty()
                && negative_controls_present
                && physics_signature_equivalent
            {
                "IMPLEMENTATION_VERIFIED".to_string()
            } else {
                "IMPLEMENTATION_PARTIAL".to_string()
            },
            salience_event_count: salience_events.len(),
            candidate_count: motif_candidates.len(),
            consensus_event_count: consensus_events.len(),
            persistent_transition_object_count: persistent_transition_objects.len(),
            negative_controls_present,
            scientific_uncertainty_tensor_only: true,
            legacy_noise_floor_rejected: true,
            contact_normalization_ontology:
                "ContactSupportObservable; preferred contact_probability uncertainty, otherwise validated runtime contact-support observable currently implemented by KCC uncertainty under measured uncertainty constraints".to_string(),
            contact_normalization_resolution_chain: contact_chains.into_iter().collect(),
            contact_support_observable: contact_observables.into_iter().collect::<Vec<_>>().join(","),
            contact_support_semantic_equivalence_asserted: false,
            runtime_origin: self.origin.to_string(),
            tokenizer_family: "TransitionEvidenceLedger".to_string(),
        };
        let resident_cache = ResidentTransitionCache {
            schema_kind: "resident_transition_cache".to_string(),
            schema_version: 1,
            cache_id: format!("resident_transition_cache:{}", self.config.run_id),
            runtime_origin: self.origin.to_string(),
            transition_objects: persistent_transition_objects.clone(),
            steering_state_present: false,
            forbidden_steering_fields_present: Vec::new(),
        };
        let non_interference_ledger = TshRuntimeNonInterferenceLedger {
            schema_kind: "tsh_runtime_non_interference_ledger".to_string(),
            schema_version: 1,
            force_mutation: false,
            dt_mutation: false,
            sampling_bias: false,
            energy_landscape_steering: false,
            trajectory_mutation: false,
            baseline_physics_signature: self.config.baseline_physics_signature.clone(),
            enabled_physics_signature: self.config.enabled_physics_signature.clone(),
            physics_signature_equivalent,
        };
        let stage_iii_manifest = TshStageIIICompatibilityManifest {
            schema_kind: "tsh_stage_iii_compatibility_manifest".to_string(),
            schema_version: 1,
            schema_migration_required: false,
            stage_iii_consumable_fields: vec![
                "persistent_transition_object_id".to_string(),
                "stable_identifier".to_string(),
                "uncertainty_tensor_refs".to_string(),
                "replicate_recurrence".to_string(),
                "transition_stability_score".to_string(),
                "chronology_ref".to_string(),
                "spectral_signature_hash".to_string(),
                "negative_control_status".to_string(),
                "source_refs".to_string(),
                "contact_support_observable".to_string(),
                "contact_uncertainty_resolution_chain".to_string(),
                "contact_support_semantic_equivalence_asserted".to_string(),
            ],
            forbidden_stage_ii_fields: vec![
                "force_modifier".to_string(),
                "dt_modifier".to_string(),
                "sampling_bias".to_string(),
                "energy_bias".to_string(),
                "trajectory_modifier".to_string(),
            ],
        };

        salience_producer.record_salience_events(salience_events.len());

        salience_producer.runtime_origin = self.origin.to_string();
        Ok(TshRuntimeArtifacts {
            salience_events,
            motif_candidates,
            motifs,
            consensus_events,
            persistent_transition_objects,
            evidence_ledgers: vec![evidence_ledger],
            resident_cache,
            non_interference_ledger,
            stage_iii_manifest,
            salience_producer,
        })
    }
}

/// Write TSH runtime artifacts to disk.
pub fn write_tsh_artifacts(
    output_dir: &Path,
    artifacts: &TshRuntimeArtifacts,
) -> Result<TshEmissionReceipt> {
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("create TSH output dir {}", output_dir.display()))?;
    let mut files = BTreeMap::new();
    write_jsonl(
        &output_dir.join("thermodynamic_salience_events.jsonl"),
        &artifacts.salience_events,
        &mut files,
        "thermodynamic_salience_events",
    )?;
    write_jsonl(
        &output_dir.join("transition_motif_candidates.jsonl"),
        &artifacts.motif_candidates,
        &mut files,
        "transition_motif_candidates",
    )?;
    write_jsonl(
        &output_dir.join("transition_motifs.jsonl"),
        &artifacts.motifs,
        &mut files,
        "transition_motifs",
    )?;
    write_jsonl(
        &output_dir.join("transition_consensus_events.jsonl"),
        &artifacts.consensus_events,
        &mut files,
        "transition_consensus_events",
    )?;
    write_jsonl(
        &output_dir.join("persistent_transition_objects.jsonl"),
        &artifacts.persistent_transition_objects,
        &mut files,
        "persistent_transition_objects",
    )?;
    write_jsonl(
        &output_dir.join("transition_evidence_ledger.jsonl"),
        &artifacts.evidence_ledgers,
        &mut files,
        "transition_evidence_ledger",
    )?;
    write_json(
        &output_dir.join("resident_transition_cache.json"),
        &artifacts.resident_cache,
        &mut files,
        "resident_transition_cache",
    )?;
    write_json(
        &output_dir.join("tsh_runtime_non_interference_ledger.json"),
        &artifacts.non_interference_ledger,
        &mut files,
        "tsh_runtime_non_interference_ledger",
    )?;
    write_json(
        &output_dir.join("tsh_stage_iii_compatibility_manifest.json"),
        &artifacts.stage_iii_manifest,
        &mut files,
        "tsh_stage_iii_compatibility_manifest",
    )?;
    // C6 no-producer marker as a first-class sidecar. A training consumer that
    // sees a zero salience count reads this file to learn whether the zero is a
    // result or a wiring gap, without inferring anything from the count itself.
    write_json(
        &output_dir.join("tsh_salience_producer_status.json"),
        &artifacts.salience_producer,
        &mut files,
        "tsh_salience_producer_status",
    )?;
    let receipt = TshEmissionReceipt {
        schema_kind: "tsh_runtime_observer_manifest".to_string(),
        status: artifacts
            .evidence_ledgers
            .first()
            .map(|ledger| ledger.acceptance_status.clone())
            .unwrap_or_else(|| "IMPLEMENTATION_PARTIAL".to_string()),
        files,
        salience_event_count: artifacts.salience_events.len(),
        persistent_transition_object_count: artifacts.persistent_transition_objects.len(),
        negative_controls_present: artifacts
            .evidence_ledgers
            .first()
            .map(|ledger| ledger.negative_controls_present)
            .unwrap_or(false),
        salience_producer_status: artifacts.salience_producer.producer_status.clone(),
        salience_producer_wired: artifacts.salience_producer.producer_wired,
    };
    let manifest_path = output_dir.join("tsh_runtime_observer_manifest.json");
    std::fs::write(&manifest_path, serde_json::to_vec_pretty(&receipt)?)?;
    Ok(receipt)
}

/// Load ScientificUncertaintyTensor refs from JSON/JSONL artifacts.
pub fn load_scientific_uncertainty_refs(
    paths: &[PathBuf],
) -> Result<Vec<ScientificUncertaintyRef>> {
    let mut refs = Vec::new();
    for path in paths {
        if !path.exists() {
            continue;
        }
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read uncertainty refs {}", path.display()))?;
        if path.extension().and_then(|ext| ext.to_str()) == Some("jsonl") {
            for line in text.lines().filter(|line| !line.trim().is_empty()) {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(line) {
                    collect_uncertainty_refs(&value, &mut refs);
                }
            }
        } else if let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) {
            collect_uncertainty_refs(&value, &mut refs);
        }
    }
    Ok(refs)
}

/// Build TSH observable inputs from runtime artifact files.
///
/// Retained for callers that do not need producer provenance. Prefer
/// [`build_observable_inputs_with_producer_status`], which additionally reports
/// whether the salience producer was wired at all — a distinction an empty
/// return value cannot carry, because zero events is a legitimate outcome.
pub fn build_observable_inputs_from_artifacts(
    config: &TshRuntimeConfig,
    artifacts: &[RuntimeArtifactRef],
    base_seed: u64,
) -> Vec<TshObservableInput> {
    build_observable_inputs_with_producer_status(config, artifacts, base_seed).0
}

/// Build TSH observable inputs and report the salience producer's wiring state.
///
/// C6 no-producer marker. A downstream training consumer that sees
/// `salience_event_count == 0` cannot distinguish "the run produced no salience
/// events" from "the producer was never wired", because zero is a legitimate
/// count. The returned [`SalienceProducerReport`] carries that distinction out
/// of band, in a typed field that no count value can collide with.
pub fn build_observable_inputs_with_producer_status(
    config: &TshRuntimeConfig,
    artifacts: &[RuntimeArtifactRef],
    base_seed: u64,
) -> (Vec<TshObservableInput>, SalienceProducerReport) {
    let mut by_stream: BTreeMap<u32, Vec<&RuntimeArtifactRef>> = BTreeMap::new();
    for artifact in artifacts.iter().filter(|artifact| artifact.present) {
        if let Some(stream_id) = artifact.stream_id {
            by_stream.entry(stream_id).or_default().push(artifact);
        }
    }
    let mut observed_kinds: BTreeSet<String> = BTreeSet::new();
    for artifact in artifacts.iter().filter(|artifact| artifact.present) {
        observed_kinds.insert(artifact.kind.clone());
    }
    let candidate_stream_count = by_stream.len();
    let mut spectral_measured = 0usize;
    let mut spectral_not_measured: BTreeMap<String, usize> = BTreeMap::new();

    let inputs: Vec<TshObservableInput> = by_stream
        .into_iter()
        .filter_map(|(stream_id, refs)| {
            let force = refs
                .iter()
                .find(|artifact| artifact.kind == "forces_final")
                .and_then(|artifact| numeric_signal_from_file(&artifact.path, &artifact.kind).ok())
                .unwrap_or_default();
            let kcc = refs
                .iter()
                .find(|artifact| artifact.kind == "kcc_v2full")
                .and_then(|artifact| numeric_signal_from_file(&artifact.path, &artifact.kind).ok())
                .unwrap_or_default();
            let spectral = spectral_recurrence(&force, &kcc);
            if spectral.is_measured() {
                spectral_measured += 1;
            } else {
                *spectral_not_measured
                    .entry(spectral.status.clone())
                    .or_insert(0) += 1;
            }
            let transitions = refs
                .iter()
                .find(|artifact| artifact.kind == "spike_transition_events")
                .and_then(|artifact| line_count(&artifact.path).ok())
                .unwrap_or(0);
            let source_refs: Vec<String> = refs
                .iter()
                .map(|artifact| artifact.path.display().to_string())
                .collect();
            if source_refs.is_empty() {
                return None;
            }
            let mut sha = Sha256::new();
            for artifact in refs {
                sha.update(artifact.kind.as_bytes());
                sha.update(artifact.path.display().to_string().as_bytes());
                if let Ok(hash) = sha256_file(&artifact.path) {
                    sha.update(hash.as_bytes());
                }
            }
            let source_sha256 = format!("sha256:{:x}", sha.finalize());
            Some(TshObservableInput {
                uncertainty_tensor_refs: Vec::new(),
                stream_id: format!("stream_{stream_id:02}"),
                seed_id: format!("seed_{}", base_seed.saturating_add(stream_id as u64)),
                epoch_id: format!("epoch_stream_{stream_id:02}"),
                chronology_event_id: format!("chronology_event_stream_{stream_id:02}"),
                chronology_ref: format!("chronology:{}:stream_{stream_id:02}", config.run_id),
                step_start: 0,
                step_end: transitions.max(force.n_samples).max(kcc.n_samples) as u64,
                time_start_ps: 0.0,
                time_end_ps: transitions.max(force.n_samples).max(kcc.n_samples) as f64,
                residue_set_hash: short_hash(&format!(
                    "{}:{}:residue_transition_bucket",
                    config.target_id,
                    if transitions > 0 {
                        "transition_active"
                    } else {
                        "transition_quiet"
                    }
                )),
                spectral_signature_hash: short_hash(&format!(
                    "{}:{}:spectral_transition_bucket",
                    config.target_id,
                    if transitions > 0 {
                        "transition_active"
                    } else {
                        "transition_quiet"
                    }
                )),
                force_delta: force.rms,
                pocket_volume_delta: transitions as f64,
                contact_delta: kcc.rms,
                // Lag in samples at which the contact-support signature peaks,
                // and the dimensionless normalized autocorrelation there. Both
                // are undefined — not zero — when the signature is undefined.
                causal_lag: kcc.peak_lag.map_or(f64::NAN, |lag| lag as f64),
                lag_corr_peak: kcc.peak_autocorrelation.unwrap_or(f64::NAN),
                spectral_signature_delta: spectral.delta,
                spectral_signature_status: spectral.status.clone(),
                spectral_signature_lags: spectral.lags,
                force_signal_provenance: force.provenance(),
                contact_signal_provenance: kcc.provenance(),
                source_refs,
                source_sha256,
            })
        })
        .collect();

    let report = SalienceProducerReport::from_discovery(
        artifacts.len(),
        observed_kinds,
        candidate_stream_count,
        inputs.len(),
        spectral_measured,
        spectral_not_measured,
    );
    (inputs, report)
}

/// How much of a source file the spectral reader actually decoded.
///
/// Recorded on every signal so a downstream reader never has to guess whether
/// a statistic describes a whole artifact or a prefix of one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignalReadMode {
    /// The entire file was decoded.
    WholeFile,
    /// The file exceeded [`SPECTRAL_FULL_READ_BUDGET_BYTES`], so evenly spaced
    /// contiguous blocks spanning the whole file were decoded. Lag products are
    /// accumulated inside a block only, never across a seam, so a sampled
    /// signal never invents adjacency the file does not contain.
    DeterministicStridedBlocks,
    /// Nothing was decoded.
    NotRead,
}

impl SignalReadMode {
    /// Stable machine-readable label for receipts.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::WholeFile => "whole_file",
            Self::DeterministicStridedBlocks => "deterministic_strided_blocks",
            Self::NotRead => "not_read",
        }
    }
}

/// Which declared on-disk layout the spectral reader honoured.
///
/// A signal never claims to have honoured a schema it did not parse: an
/// undeclared layout is recorded as an assumption, and a declared envelope this
/// build cannot decode is refused outright rather than reinterpreted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignalSchemaStatus {
    /// `PRKCC001` envelope: header parsed, per-field f32 sections hydrated at
    /// the declared record count and 4-byte stride, `PRKCCEND` trailer and
    /// FNV-1a payload checksum verified by the codec.
    Prkcc001FieldSections,
    /// Explicit finite f64 time series with caller-verified chronology.
    MeasuredF64TimeSeries,
    /// Declared array-of-structures f32 vec3 (`forces_final.bin`,
    /// `asc_vectors.bin`, `aromatic_centroids_final.bin`). The 12-byte record
    /// stride is honoured and the per-record Euclidean magnitude is the series;
    /// the components are never treated as three independent scalars.
    RawF32Vec3Aos,
    /// Neither an envelope magic nor a stride declared for this artifact kind.
    /// Little-endian f32 at a 4-byte stride is ASSUMED, and this status is how
    /// the receipt says so.
    UndeclaredFlatF32Le,
    /// The file carries a PRISM envelope magic this reader cannot decode, or is
    /// too large to decode within the documented budget. The bytes are NOT
    /// reinterpreted as a flat f32 array.
    DeclaredEnvelopeNotDecodable,
    /// The file could not be opened, stat-ed, or read.
    Unreadable,
}

impl SignalSchemaStatus {
    /// Stable machine-readable label for receipts.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Prkcc001FieldSections => "prkcc001_field_sections",
            Self::MeasuredF64TimeSeries => "measured_f64_time_series",
            Self::RawF32Vec3Aos => "raw_f32_vec3_aos",
            Self::UndeclaredFlatF32Le => "undeclared_flat_f32_le_assumed",
            Self::DeclaredEnvelopeNotDecodable => "declared_envelope_not_decodable",
            Self::Unreadable => "unreadable",
        }
    }
}

/// Decoded statistics for one runtime artifact.
///
/// `mean_abs` and `rms` retain the artifact's own physical units and are kept
/// only as descriptive magnitudes. They are deliberately NOT used to compare
/// two different observables; see the module note on units and invariance.
/// `autocorrelation` is the dimensionless, affine-invariant quantity that is.
#[derive(Clone, Debug)]
pub struct NumericSignal {
    /// Decoded finite sample count.
    pub n_samples: usize,
    /// Mean absolute value, in the artifact's own units.
    pub mean_abs: f64,
    /// Root mean square, in the artifact's own units.
    pub rms: f64,
    /// Normalized autocorrelation `r_1..r_K`, dimensionless, each in [-1, 1].
    /// Empty when undefined.
    pub autocorrelation: Vec<f64>,
    /// Whether `autocorrelation` is a measurement rather than an absence.
    pub autocorrelation_defined: bool,
    /// Which lag, in samples, carries the largest `|r_k|`. `None` when
    /// undefined.
    pub peak_lag: Option<usize>,
    /// `r_k` at `peak_lag`. `None` when undefined.
    pub peak_autocorrelation: Option<f64>,
    /// Whole-file or deterministic strided sample.
    pub read_mode: SignalReadMode,
    /// Which declared layout was honoured.
    pub schema_status: SignalSchemaStatus,
    /// Named field sections decoded from an envelope, in deterministic order.
    pub decoded_fields: Vec<String>,
    /// File size in bytes.
    pub file_bytes: u64,
    /// Bytes actually decoded.
    pub bytes_considered: u64,
    /// `bytes_considered / file_bytes`, 0.0 for an empty or unread file.
    pub coverage_fraction: f64,
    /// Number of contiguous runs the series was split into.
    pub segment_count: usize,
    /// Samples dropped because they were not finite. Each drop splits a
    /// segment so adjacency is never invented across the hole.
    pub nonfinite_samples: usize,
}

impl Default for NumericSignal {
    fn default() -> Self {
        Self {
            n_samples: 0,
            mean_abs: 0.0,
            rms: 0.0,
            autocorrelation: Vec::new(),
            autocorrelation_defined: false,
            peak_lag: None,
            peak_autocorrelation: None,
            read_mode: SignalReadMode::NotRead,
            schema_status: SignalSchemaStatus::Unreadable,
            decoded_fields: Vec::new(),
            file_bytes: 0,
            bytes_considered: 0,
            coverage_fraction: 0.0,
            segment_count: 0,
            nonfinite_samples: 0,
        }
    }
}

impl NumericSignal {
    /// Machine-readable provenance of this signal for receipts.
    #[must_use]
    pub fn provenance(&self) -> BTreeMap<String, String> {
        let mut out = BTreeMap::new();
        out.insert("read_mode".to_string(), self.read_mode.as_str().to_string());
        out.insert(
            "schema_status".to_string(),
            self.schema_status.as_str().to_string(),
        );
        out.insert("file_bytes".to_string(), self.file_bytes.to_string());
        out.insert(
            "bytes_considered".to_string(),
            self.bytes_considered.to_string(),
        );
        out.insert(
            "coverage_fraction".to_string(),
            format!("{:.6}", self.coverage_fraction),
        );
        out.insert("n_samples".to_string(), self.n_samples.to_string());
        out.insert("segment_count".to_string(), self.segment_count.to_string());
        out.insert(
            "nonfinite_samples".to_string(),
            self.nonfinite_samples.to_string(),
        );
        out.insert(
            "autocorrelation_defined".to_string(),
            self.autocorrelation_defined.to_string(),
        );
        out.insert("decoded_fields".to_string(), self.decoded_fields.join(","));
        out
    }
}

/// Outcome of comparing two spectral signatures.
///
/// `delta` and `score` are non-finite exactly when `status != "measured"`.
/// serde emits JSON `null` for a non-finite float, which is out of band for a
/// quantity whose measured range is [0, 1] — a consumer can therefore never
/// mistake an absent comparison for a low one.
#[derive(Clone, Debug)]
pub struct SpectralRecurrence {
    /// Mean absolute per-lag signature difference, halved. Dimensionless,
    /// [0, 1] when measured, non-finite otherwise.
    pub delta: f64,
    /// `1 - delta`. Dimensionless, [0, 1] when measured, non-finite otherwise.
    pub score: f64,
    /// `measured`, or the specific reason no measurement exists.
    pub status: String,
    /// Number of lags compared. Zero when not measured.
    pub lags: usize,
}

impl SpectralRecurrence {
    /// Whether this carries a real measurement.
    #[must_use]
    pub fn is_measured(&self) -> bool {
        self.status == SPECTRAL_STATUS_MEASURED
    }

    fn not_measurable(status: &str) -> Self {
        Self {
            delta: f64::NAN,
            score: f64::NAN,
            status: status.to_string(),
            lags: 0,
        }
    }
}

/// The one status value that means a spectral comparison actually happened.
pub const SPECTRAL_STATUS_MEASURED: &str = "measured";

/// Compare two spectral signatures.
///
/// Scale-free by construction: both operands are normalized autocorrelation
/// vectors, which are invariant under `x -> a*x + b` for any `a != 0`. The
/// result is dimensionless.
#[must_use]
pub fn spectral_recurrence(a: &NumericSignal, b: &NumericSignal) -> SpectralRecurrence {
    for signal in [a, b] {
        match signal.schema_status {
            SignalSchemaStatus::Unreadable => {
                return SpectralRecurrence::not_measurable("not_measurable_missing_artifact");
            }
            SignalSchemaStatus::DeclaredEnvelopeNotDecodable => {
                return SpectralRecurrence::not_measurable("not_measurable_schema_not_decodable");
            }
            _ => {}
        }
        if signal.n_samples < SPECTRAL_MIN_SAMPLES {
            return SpectralRecurrence::not_measurable("not_measurable_insufficient_samples");
        }
        if !signal.autocorrelation_defined {
            return SpectralRecurrence::not_measurable("not_measurable_degenerate_signal");
        }
    }
    let lags = a.autocorrelation.len().min(b.autocorrelation.len());
    if lags == 0 {
        return SpectralRecurrence::not_measurable("not_measurable_degenerate_signal");
    }
    let delta = (a
        .autocorrelation
        .iter()
        .zip(b.autocorrelation.iter())
        .map(|(ra, rb)| (ra - rb).abs())
        .sum::<f64>()
        / lags as f64
        / 2.0)
        .clamp(0.0, 1.0);
    SpectralRecurrence {
        delta,
        score: 1.0 - delta,
        status: SPECTRAL_STATUS_MEASURED.to_string(),
        lags,
    }
}

/// Declared component stride for an artifact kind, in f32 units.
///
/// `None` means this build has no declared layout for the kind and will record
/// its flat-f32 read as an assumption.
fn declared_components_for_kind(kind: &str) -> Option<usize> {
    match kind {
        // `crates/prism-nhs/src/bin/mechanical_load.rs` documents the layout as
        // "raw f32 AoS n_atoms x 3" for these three surfaces.
        "forces_final" | "asc_vectors" | "aromatic_centroids_final" => Some(3),
        _ => None,
    }
}

/// PRISM envelope magics this reader recognizes but does not decode.
///
/// Recognizing them matters: without this check a `PRSPK001` spike stream would
/// be silently reinterpreted as an f32 array, and its header bytes would enter
/// the statistics as if they were measurements.
const UNDECODABLE_ENVELOPE_MAGICS: [&[u8; 8]; 2] = [b"PRSPK001", b"PRSGD001"];

/// Decode one runtime artifact into a spectral signature, honouring whatever
/// layout the file actually declares.
///
/// `declared_kind` is the `md_evidence_manifest` artifact kind, used to resolve
/// a declared record stride when the file carries no envelope of its own.
pub fn numeric_signal_from_file(path: &Path, declared_kind: &str) -> Result<NumericSignal> {
    let metadata = std::fs::metadata(path)
        .with_context(|| format!("stat spectral source {}", path.display()))?;
    let file_bytes = metadata.len();
    let mut file =
        File::open(path).with_context(|| format!("open spectral source {}", path.display()))?;

    let mut magic = [0u8; 8];
    let magic_len = read_fully_up_to(&mut file, &mut magic)
        .with_context(|| format!("read magic {}", path.display()))?;

    if magic_len == 8 && &magic == prism_tokenizer::codec::PRKCC001_MAGIC {
        return prkcc001_signal(path, file_bytes);
    }
    if magic_len == 8
        && UNDECODABLE_ENVELOPE_MAGICS
            .iter()
            .any(|known| *known == &magic)
    {
        return Ok(NumericSignal {
            schema_status: SignalSchemaStatus::DeclaredEnvelopeNotDecodable,
            read_mode: SignalReadMode::NotRead,
            file_bytes,
            ..NumericSignal::default()
        });
    }

    let components = declared_components_for_kind(declared_kind);
    let schema_status = if components.is_some() {
        SignalSchemaStatus::RawF32Vec3Aos
    } else {
        SignalSchemaStatus::UndeclaredFlatF32Le
    };
    raw_f32_signal(&mut file, path, file_bytes, components, schema_status)
}

/// Read into `buf` until it is full or EOF; return how many bytes landed.
fn read_fully_up_to(file: &mut File, buf: &mut [u8]) -> std::io::Result<usize> {
    let mut filled = 0usize;
    while filled < buf.len() {
        match file.read(&mut buf[filled..])? {
            0 => break,
            n => filled += n,
        }
    }
    Ok(filled)
}

/// Decode a `PRKCC001` envelope through the registered codec.
///
/// The envelope's own header supplies the record count and stride, and the
/// codec verifies the trailer and the FNV-1a payload checksum, so no byte of
/// the header is ever mistaken for a sample. Each named f32 field section is
/// one contiguous per-residue series and therefore one segment; adjacency is
/// never asserted across two different fields. `u32` sections are categorical
/// (`active_causal` and peers) and are excluded from a magnitude statistic.
fn prkcc001_signal(path: &Path, file_bytes: u64) -> Result<NumericSignal> {
    if file_bytes > SPECTRAL_FULL_READ_BUDGET_BYTES {
        return Ok(NumericSignal {
            schema_status: SignalSchemaStatus::DeclaredEnvelopeNotDecodable,
            read_mode: SignalReadMode::NotRead,
            file_bytes,
            ..NumericSignal::default()
        });
    }
    let raw = std::fs::read(path)
        .with_context(|| format!("read PRKCC001 envelope {}", path.display()))?;
    let decoded = match prism_tokenizer::codec::decode_prkcc001(&raw) {
        Ok(decoded) => decoded,
        Err(_) => {
            return Ok(NumericSignal {
                schema_status: SignalSchemaStatus::DeclaredEnvelopeNotDecodable,
                read_mode: SignalReadMode::NotRead,
                file_bytes,
                ..NumericSignal::default()
            });
        }
    };
    let mut segments: Vec<Vec<f32>> = Vec::new();
    let mut decoded_fields: Vec<String> = Vec::new();
    for (name, values) in &decoded.f32_fields {
        decoded_fields.push(name.clone());
        segments.push(values.clone());
    }
    let mut signal = finish_signal(&segments);
    signal.read_mode = SignalReadMode::WholeFile;
    signal.schema_status = SignalSchemaStatus::Prkcc001FieldSections;
    signal.decoded_fields = decoded_fields;
    signal.file_bytes = file_bytes;
    signal.bytes_considered = file_bytes;
    signal.coverage_fraction = 1.0;
    Ok(signal)
}

/// Decode a headerless little-endian f32 artifact.
///
/// Files at or under [`SPECTRAL_FULL_READ_BUDGET_BYTES`] are read whole. Larger
/// files are sampled with evenly spaced contiguous blocks that span the entire
/// file — never a prefix — and the read mode and coverage fraction are recorded
/// so the receipt says which happened.
fn raw_f32_signal(
    file: &mut File,
    path: &Path,
    file_bytes: u64,
    components: Option<usize>,
    schema_status: SignalSchemaStatus,
) -> Result<NumericSignal> {
    let record_bytes = 4 * components.unwrap_or(1) as u64;
    if file_bytes < record_bytes {
        return Ok(NumericSignal {
            schema_status,
            read_mode: SignalReadMode::WholeFile,
            file_bytes,
            bytes_considered: 0,
            ..NumericSignal::default()
        });
    }

    let (blocks, read_mode) = if file_bytes <= SPECTRAL_FULL_READ_BUDGET_BYTES {
        (vec![(0u64, file_bytes)], SignalReadMode::WholeFile)
    } else {
        let block_bytes = (SPECTRAL_SAMPLE_BLOCK_BYTES / record_bytes).max(1) * record_bytes;
        let block_count = (SPECTRAL_SAMPLE_BUDGET_BYTES / block_bytes).max(1);
        let stride = file_bytes / block_count;
        let mut blocks = Vec::new();
        for index in 0..block_count {
            // Align every block start down to a record boundary so a strided
            // read can never straddle a record and manufacture a component.
            let start = (index * stride) / record_bytes * record_bytes;
            if start >= file_bytes {
                break;
            }
            let len = block_bytes.min(file_bytes - start) / record_bytes * record_bytes;
            if len >= record_bytes {
                blocks.push((start, len));
            }
        }
        (blocks, SignalReadMode::DeterministicStridedBlocks)
    };

    let mut segments: Vec<Vec<f32>> = Vec::new();
    let mut bytes_considered = 0u64;
    let mut buffer = Vec::new();
    for (start, len) in blocks {
        file.seek(SeekFrom::Start(start))
            .with_context(|| format!("seek {} to {start}", path.display()))?;
        buffer.clear();
        buffer.resize(len as usize, 0u8);
        let filled = read_fully_up_to(file, &mut buffer)
            .with_context(|| format!("read {} at {start}", path.display()))?;
        let usable = filled / record_bytes as usize * record_bytes as usize;
        if usable == 0 {
            continue;
        }
        bytes_considered += usable as u64;
        let mut segment = Vec::with_capacity(usable / record_bytes as usize);
        match components {
            Some(n) => {
                for record in buffer[..usable].chunks_exact(4 * n) {
                    let mut sum_sq = 0.0f64;
                    for lane in record.chunks_exact(4) {
                        let value = f32::from_le_bytes([lane[0], lane[1], lane[2], lane[3]]) as f64;
                        sum_sq += value * value;
                    }
                    segment.push(sum_sq.sqrt() as f32);
                }
            }
            None => {
                for lane in buffer[..usable].chunks_exact(4) {
                    segment.push(f32::from_le_bytes([lane[0], lane[1], lane[2], lane[3]]));
                }
            }
        }
        segments.push(segment);
    }

    let mut signal = finish_signal(&segments);
    signal.read_mode = read_mode;
    signal.schema_status = schema_status;
    signal.file_bytes = file_bytes;
    signal.bytes_considered = bytes_considered;
    signal.coverage_fraction = if file_bytes == 0 {
        0.0
    } else {
        (bytes_considered as f64 / file_bytes as f64).clamp(0.0, 1.0)
    };
    Ok(signal)
}

/// Accumulate magnitude statistics and the normalized autocorrelation
/// signature over a set of contiguous segments.
///
/// A non-finite sample splits its segment: adjacency is never asserted across a
/// value that was dropped, so `r_k` only ever sums products of samples that are
/// genuinely `k` records apart in the file.
fn finish_signal<T: Copy + Into<f64>>(segments: &[Vec<T>]) -> NumericSignal {
    let mut clean: Vec<Vec<f64>> = Vec::new();
    let mut nonfinite = 0usize;
    for segment in segments {
        let mut current: Vec<f64> = Vec::new();
        for &raw in segment {
            let value: f64 = raw.into();
            if value.is_finite() {
                current.push(value);
            } else {
                nonfinite += 1;
                if !current.is_empty() {
                    clean.push(std::mem::take(&mut current));
                }
            }
        }
        if !current.is_empty() {
            clean.push(current);
        }
    }

    let n: usize = clean.iter().map(Vec::len).sum();
    if n == 0 {
        return NumericSignal {
            segment_count: 0,
            nonfinite_samples: nonfinite,
            ..NumericSignal::default()
        };
    }
    let mut sum = 0.0f64;
    let mut sum_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    for segment in &clean {
        for &value in segment {
            sum += value;
            sum_abs += value.abs();
            sum_sq += value * value;
        }
    }
    let mean = sum / n as f64;
    let centred_sum_sq = (sum_sq - n as f64 * mean * mean).max(0.0);

    let degenerate =
        centred_sum_sq <= SPECTRAL_DEGENERATE_VARIANCE_RATIO * sum_sq.max(f64::MIN_POSITIVE);
    let mut autocorrelation = Vec::new();
    let mut defined = false;
    let mut peak_lag = None;
    let mut peak_value = None;
    if n >= SPECTRAL_MIN_SAMPLES && !degenerate {
        let mut acc = vec![0.0f64; SPECTRAL_SIGNATURE_LAGS];
        let mut pairs = vec![0usize; SPECTRAL_SIGNATURE_LAGS];
        for segment in &clean {
            for (index, slot) in acc.iter_mut().enumerate() {
                let lag = index + 1;
                if segment.len() <= lag {
                    continue;
                }
                for i in 0..segment.len() - lag {
                    *slot += (segment[i] - mean) * (segment[i + lag] - mean);
                }
                pairs[index] += segment.len() - lag;
            }
        }
        if pairs.iter().all(|count| *count > 0) {
            autocorrelation = acc
                .iter()
                .map(|value| (value / centred_sum_sq).clamp(-1.0, 1.0))
                .collect();
            defined = true;
            let mut best = 0usize;
            for (index, value) in autocorrelation.iter().enumerate() {
                if value.abs() > autocorrelation[best].abs() {
                    best = index;
                }
            }
            peak_lag = Some(best + 1);
            peak_value = Some(autocorrelation[best]);
        }
    }

    NumericSignal {
        n_samples: n,
        mean_abs: sum_abs / n as f64,
        rms: (sum_sq / n as f64).sqrt(),
        autocorrelation,
        autocorrelation_defined: defined,
        peak_lag,
        peak_autocorrelation: peak_value,
        read_mode: SignalReadMode::NotRead,
        schema_status: SignalSchemaStatus::UndeclaredFlatF32Le,
        decoded_fields: Vec::new(),
        file_bytes: 0,
        bytes_considered: 0,
        coverage_fraction: 0.0,
        segment_count: clean.len(),
        nonfinite_samples: nonfinite,
    }
}

fn line_count(path: &Path) -> Result<usize> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let reader = BufReader::new(file);
    Ok(reader
        .lines()
        .filter(|line| line.as_ref().map(|s| !s.trim().is_empty()).unwrap_or(false))
        .count())
}

fn collect_uncertainty_refs(value: &serde_json::Value, out: &mut Vec<ScientificUncertaintyRef>) {
    if let Some(records) = value.get("records").and_then(|records| records.as_array()) {
        for record in records {
            collect_uncertainty_refs(record, out);
        }
        return;
    }
    if let Some(records) = value.as_array() {
        for record in records {
            collect_uncertainty_refs(record, out);
        }
        return;
    }
    let Some(obj) = value.as_object() else {
        return;
    };
    let schema_kind = str_value(obj, "schema_kind").unwrap_or("scientific_uncertainty_tensor");
    if !schema_kind.to_ascii_lowercase().contains("uncertainty") {
        return;
    }
    let tensor_id = str_value(obj, "tensor_id")
        .or_else(|| str_value(obj, "uncertainty_tensor_id"))
        .or_else(|| str_value(obj, "scientific_uncertainty_tensor_ref"))
        .unwrap_or("unknown_uncertainty_tensor");
    let observable = str_value(obj, "observable").unwrap_or("unknown");
    let observable_family = str_value(obj, "observable_family").unwrap_or(observable);
    let sigma = number_value(obj, "sigma").unwrap_or(0.0);
    let n_samples = number_value(obj, "n_samples").unwrap_or(0.0).max(0.0) as u64;
    let source_sha256 = str_value(obj, "sha256")
        .or_else(|| str_value(obj, "source_sha256"))
        .unwrap_or("");
    let source_refs = ["source_refs", "source_artifact_refs", "source_ref", "provenance_refs"]
        .iter().filter_map(|key| value_refs(obj.get(*key))).flatten()
        .collect::<BTreeSet<_>>().into_iter().collect();
    out.push(ScientificUncertaintyRef {
        tensor_id: tensor_id.to_string(),
        observable: observable.to_string(),
        observable_family: observable_family.to_string(),
        sigma,
        n_samples,
        measurement_status: str_value(obj, "measurement_status")
            .unwrap_or("missing")
            .to_string(),
        source_status: str_value(obj, "source_status")
            .unwrap_or("missing")
            .to_string(),
        source_family: str_value(obj, "source_family")
            .unwrap_or("unknown")
            .to_string(),
        source_sha256: source_sha256.to_string(),
        source_refs,
        schema_kind: schema_kind.to_string(),
    });
}

struct ContactSupportUncertainty<'a> {
    uncertainty: &'a ScientificUncertaintyRef,
    observable: String,
    resolution_chain: Vec<String>,
    contact_probability_uncertainty_status: String,
}

fn contact_support_uncertainty<'a>(
    all_uncertainties: &'a [ScientificUncertaintyRef],
    valid_uncertainties: &'a [ScientificUncertaintyRef],
) -> Result<ContactSupportUncertainty<'a>> {
    if let Some(contact_probability) =
        uncertainty_for_exact(valid_uncertainties, "contact_probability")
    {
        return Ok(ContactSupportUncertainty {
            uncertainty: contact_probability,
            observable: "contact_probability".to_string(),
            resolution_chain: vec![
                "preferred:contact_probability_uncertainty".to_string(),
                "selected:contact_probability_uncertainty".to_string(),
            ],
            contact_probability_uncertainty_status:
                "preferred_contact_probability_uncertainty_valid".to_string(),
        });
    }

    let contact_probability_uncertainty_status = all_uncertainties
        .iter()
        .find(|u| u.observable.eq_ignore_ascii_case("contact_probability"))
        .map(|u| {
            if u.validate_for_tsh().is_ok() {
                "preferred_contact_probability_uncertainty_valid"
            } else {
                "preferred_contact_probability_uncertainty_unusable_under_measured_sigma_gate"
            }
        })
        .unwrap_or("preferred_contact_probability_uncertainty_missing")
        .to_string();

    let kcc = uncertainty_for_exact(valid_uncertainties, "kcc").with_context(|| {
        "missing validated ContactSupportObservable uncertainty currently implemented by KCC"
    })?;
    Ok(ContactSupportUncertainty {
        uncertainty: kcc,
        observable: "kcc".to_string(),
        resolution_chain: vec![
            "preferred:contact_probability_uncertainty".to_string(),
            contact_probability_uncertainty_status.clone(),
            "validated_contact_support_observable:KCC_uncertainty".to_string(),
            "selected:KCC_uncertainty_as_runtime_contact_support_channel".to_string(),
            "semantic_equivalence:contact_probability_vs_KCC:false".to_string(),
        ],
        contact_probability_uncertainty_status,
    })
}

fn uncertainty_for<'a>(
    uncertainties: &'a [ScientificUncertaintyRef],
    observable: &str,
) -> Result<&'a ScientificUncertaintyRef> {
    uncertainty_for_any(uncertainties, &[observable])
}

fn uncertainty_for_exact<'a>(
    uncertainties: &'a [ScientificUncertaintyRef],
    observable: &str,
) -> Option<&'a ScientificUncertaintyRef> {
    uncertainties
        .iter()
        .find(|u| u.observable.eq_ignore_ascii_case(observable))
}

fn uncertainty_for_any<'a>(
    uncertainties: &'a [ScientificUncertaintyRef],
    observables: &[&str],
) -> Result<&'a ScientificUncertaintyRef> {
    let matches: Vec<_> = uncertainties.iter().filter(|u| {
        let observable = u.observable.to_ascii_lowercase();
        let family = u.observable_family.to_ascii_lowercase();
        observables.iter().any(|needle| observable.contains(needle) || family.contains(needle))
    }).collect();
    if matches.len() != 1 { bail!("expected one per-interval uncertainty for {:?}, found {}; bind exact tensor IDs", observables, matches.len()); }
    Ok(matches[0])
}

fn deserialize_unavailable_score<'de, D: serde::Deserializer<'de>>(deserializer: D) -> std::result::Result<f64, D::Error> {
    Ok(Option::<f64>::deserialize(deserializer)?.unwrap_or(f64::NAN))
}

fn snr(delta: f64, sigma: f64) -> f64 {
    delta.abs() / sigma
}

fn mean(values: &[f64]) -> f64 {
    values.iter().copied().sum::<f64>() / values.len().max(1) as f64
}

fn motif_id_for(input: &TshObservableInput, config: &TshRuntimeConfig) -> String {
    short_hash(&format!(
        "{}:{}:{}:{}:{}",
        config.target_id,
        config.topology_hash_before,
        config.topology_hash_after,
        input.residue_set_hash,
        input.spectral_signature_hash
    ))
}

fn short_hash(text: &str) -> String {
    let mut sha = Sha256::new();
    sha.update(text.as_bytes());
    let digest = sha.finalize();
    format!("tsh_{:x}", digest)[..20].to_string()
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut sha = Sha256::new();
    let mut buf = [0u8; 8192];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        sha.update(&buf[..n]);
    }
    Ok(format!("sha256:{:x}", sha.finalize()))
}

fn str_value<'a>(
    obj: &'a serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Option<&'a str> {
    obj.get(key).and_then(|value| value.as_str())
}

fn number_value(obj: &serde_json::Map<String, serde_json::Value>, key: &str) -> Option<f64> {
    obj.get(key).and_then(|value| value.as_f64())
}

fn value_refs(value: Option<&serde_json::Value>) -> Option<Vec<String>> {
    match value? {
        serde_json::Value::String(s) if !s.trim().is_empty() => Some(vec![s.clone()]),
        serde_json::Value::Array(values) => {
            Some(values.iter().filter_map(value_ref_string).collect())
        }
        serde_json::Value::Object(_) => value_ref_string(value?).map(|s| vec![s]),
        _ => None,
    }
}

fn value_ref_string(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(s) if !s.trim().is_empty() => Some(s.clone()),
        serde_json::Value::Object(obj) => {
            let path = obj.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let sha = obj.get("sha256").and_then(|v| v.as_str()).unwrap_or("");
            let record = obj
                .get("record_index")
                .and_then(|v| v.as_u64())
                .map(|v| format!("#{v}"))
                .unwrap_or_default();
            if path.is_empty() && sha.is_empty() {
                None
            } else if path.is_empty() {
                Some(format!("sha256:{sha}{record}"))
            } else if sha.is_empty() {
                Some(format!("{path}{record}"))
            } else {
                Some(format!("{path}{record}:sha256:{sha}"))
            }
        }
        _ => None,
    }
}

fn write_jsonl<T: Serialize>(
    path: &Path,
    records: &[T],
    files: &mut BTreeMap<String, String>,
    key: &str,
) -> Result<()> {
    let mut file = File::create(path).with_context(|| format!("create {}", path.display()))?;
    for record in records {
        serde_json::to_writer(&mut file, record)?;
        file.write_all(b"\n")?;
    }
    files.insert(key.to_string(), path.display().to_string());
    Ok(())
}

fn write_json<T: Serialize>(
    path: &Path,
    value: &T,
    files: &mut BTreeMap<String, String>,
    key: &str,
) -> Result<()> {
    std::fs::write(path, serde_json::to_vec_pretty(value)?)
        .with_context(|| format!("write {}", path.display()))?;
    files.insert(key.to_string(), path.display().to_string());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uncertainty(observable: &str) -> ScientificUncertaintyRef {
        ScientificUncertaintyRef {
            tensor_id: format!("sut:{observable}"),
            observable: observable.to_string(),
            observable_family: observable.to_string(),
            sigma: 2.0,
            n_samples: 16,
            measurement_status: "measured".to_string(),
            source_status: "measured".to_string(),
            source_family: "ScientificUncertaintyTensor".to_string(),
            source_sha256: "sha256:abc".to_string(),
            source_refs: vec!["artifact.bin".to_string()],
            schema_kind: "scientific_uncertainty_tensor".to_string(),
        }
    }

    fn input(seed: &str) -> TshObservableInput {
        TshObservableInput {
            uncertainty_tensor_refs: Vec::new(),
            stream_id: format!("stream_{seed}"),
            seed_id: seed.to_string(),
            epoch_id: "epoch_1".to_string(),
            chronology_event_id: "chron_1".to_string(),
            chronology_ref: "chronology:1".to_string(),
            step_start: 10,
            step_end: 20,
            time_start_ps: 0.01,
            time_end_ps: 0.02,
            residue_set_hash: "residue_set_a".to_string(),
            spectral_signature_hash: "spectral_a".to_string(),
            force_delta: 12.0,
            pocket_volume_delta: 40.0,
            contact_delta: 6.0,
            causal_lag: 2.0,
            lag_corr_peak: 0.8,
            spectral_signature_delta: 0.25,
            spectral_signature_status: SPECTRAL_STATUS_MEASURED.to_string(),
            spectral_signature_lags: SPECTRAL_SIGNATURE_LAGS,
            force_signal_provenance: BTreeMap::new(),
            contact_signal_provenance: BTreeMap::new(),
            source_refs: vec!["forces.bin".to_string()],
            source_sha256: "sha256:source".to_string(),
        }
    }

    /// A deterministic, non-degenerate series with a controllable shape.
    fn write_flat_f32(path: &Path, values: &[f32]) {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(path, bytes).unwrap();
    }

    fn observer() -> TshRuntimeObserver {
        let mut cfg = TshRuntimeConfig::new("run_1", "target");
        cfg.topology_hash_before = "topo_a".to_string();
        cfg.topology_hash_after = "topo_a".to_string();
        cfg.baseline_physics_signature = "physics_hash".to_string();
        cfg.enabled_physics_signature = "physics_hash".to_string();
        TshRuntimeObserver::new(cfg)
    }

    #[test]
    fn rejects_legacy_noise_floor_as_tsh_gate_evidence() {
        let mut u = uncertainty("force");
        u.source_family = "NoiseFloor".to_string();
        assert!(u.validate_for_tsh().is_err());
    }

    #[test]
    fn measured_sut_object_source_refs_are_runtime_valid() {
        let raw = serde_json::json!({
            "schema_kind": "scientific_uncertainty_tensor",
            "tensor_id": "sut_force_1",
            "observable": "force",
            "observable_family": "dynamics",
            "sigma": 0.25,
            "n_samples": 16,
            "measurement_status": "measured",
            "source_status": "measured",
            "source_family": "ScientificStateEpoch",
            "sha256": "abc",
            "source_refs": [
                {
                    "path": "scientific_state_epochs.jsonl",
                    "record_index": 7,
                    "sha256": "def"
                }
            ]
        });
        let mut refs = Vec::new();
        collect_uncertainty_refs(&raw, &mut refs);
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].source_sha256, "abc");
        assert_eq!(
            refs[0].source_refs[0],
            "scientific_state_epochs.jsonl#7:sha256:def"
        );
        refs[0].validate_for_tsh().unwrap();
    }

    #[test]
    fn uses_validated_kcc_contact_support_observable_when_contact_probability_is_unusable() {
        let mut bad_contact = uncertainty("contact_probability");
        bad_contact.sigma = 0.0;
        let mut kcc = uncertainty("kcc");
        kcc.observable_family = "thermal_observable".to_string();
        let uncertainties = vec![
            uncertainty("force"),
            uncertainty("pocket_volume"),
            bad_contact,
            kcc,
        ];
        let out = observer()
            .evaluate(&[input("seed1")], &uncertainties, Some(&BTreeSet::new()))
            .unwrap();
        assert_eq!(out.salience_events.len(), 1);
        assert_eq!(out.salience_events[0].contact_delta_sigma, 2.0);
        assert_eq!(
            out.salience_events[0].contact_support_observable_role,
            "ContactSupportObservable"
        );
        assert_eq!(out.salience_events[0].contact_support_observable, "kcc");
        assert!(!out.salience_events[0].contact_support_semantic_equivalence_asserted);
        assert!(out.salience_events[0]
            .contact_uncertainty_resolution_chain
            .contains(&"validated_contact_support_observable:KCC_uncertainty".to_string()));
        assert!(out.salience_events[0]
            .uncertainty_tensor_refs
            .iter()
            .any(|reference| reference == "sut:kcc"));
        assert_eq!(out.evidence_ledgers[0].contact_support_observable, "kcc");
        assert!(!out.evidence_ledgers[0].contact_support_semantic_equivalence_asserted);
    }

    #[test]
    fn single_seed_remains_candidate_only() {
        let uncertainties = vec![
            uncertainty("force"),
            uncertainty("pocket_volume"),
            uncertainty("contact_probability"),
        ];
        let out = observer()
            .evaluate(&[input("seed1")], &uncertainties, Some(&BTreeSet::new()))
            .unwrap();
        assert_eq!(out.salience_events.len(), 1);
        assert_eq!(out.motif_candidates.len(), 1);
        assert!(out.persistent_transition_objects.is_empty());
        assert_eq!(out.resident_cache.transition_objects.len(), 0);
    }

    #[test]
    fn replicate_consensus_accepts_persistent_transition_object() {
        let uncertainties = vec![
            uncertainty("force"),
            uncertainty("pocket_volume"),
            uncertainty("contact_probability"),
        ];
        let out = observer()
            .evaluate(
                &[input("seed1"), input("seed2")],
                &uncertainties,
                Some(&BTreeSet::new()),
            )
            .unwrap();
        assert_eq!(out.consensus_events[0].consensus_seed_count, 2);
        assert_eq!(out.persistent_transition_objects.len(), 1);
        assert_eq!(out.resident_cache.transition_objects.len(), 1);
    }

    #[test]
    fn negative_control_rejects_consensus_motif() {
        let uncertainties = vec![
            uncertainty("force"),
            uncertainty("pocket_volume"),
            uncertainty("contact_probability"),
        ];
        let motif_id = motif_id_for(&input("seed1"), &observer().config);
        let mut controls = BTreeSet::new();
        controls.insert(motif_id);
        let out = observer()
            .evaluate(
                &[input("seed1"), input("seed2")],
                &uncertainties,
                Some(&controls),
            )
            .unwrap();
        assert!(out.persistent_transition_objects.is_empty());
        assert_eq!(out.consensus_events[0].acceptance_status, "rejected");
    }

    #[test]
    fn non_interference_and_cache_have_no_steering_state() {
        let uncertainties = vec![
            uncertainty("force"),
            uncertainty("pocket_volume"),
            uncertainty("contact_probability"),
        ];
        let out = observer()
            .evaluate(
                &[input("seed1"), input("seed2")],
                &uncertainties,
                Some(&BTreeSet::new()),
            )
            .unwrap();
        assert!(!out.non_interference_ledger.force_mutation);
        assert!(!out.non_interference_ledger.dt_mutation);
        assert!(!out.non_interference_ledger.sampling_bias);
        assert!(!out.non_interference_ledger.energy_landscape_steering);
        assert!(!out.non_interference_ledger.trajectory_mutation);
        assert!(out.non_interference_ledger.physics_signature_equivalent);
        assert!(!out.resident_cache.steering_state_present);
        assert!(out
            .resident_cache
            .forbidden_steering_fields_present
            .is_empty());
    }

    #[test]
    fn stage_iii_manifest_requires_no_schema_migration() {
        let uncertainties = vec![
            uncertainty("force"),
            uncertainty("pocket_volume"),
            uncertainty("contact_probability"),
        ];
        let out = observer()
            .evaluate(
                &[input("seed1"), input("seed2")],
                &uncertainties,
                Some(&BTreeSet::new()),
            )
            .unwrap();
        assert!(!out.stage_iii_manifest.schema_migration_required);
        assert!(out
            .stage_iii_manifest
            .stage_iii_consumable_fields
            .contains(&"uncertainty_tensor_refs".to_string()));
        assert!(out
            .stage_iii_manifest
            .stage_iii_consumable_fields
            .contains(&"contact_support_observable".to_string()));
        assert!(out
            .stage_iii_manifest
            .forbidden_stage_ii_fields
            .contains(&"force_modifier".to_string()));
    }

    // ── Spectral recurrence: reachability, units, invariance ───────────────

    /// A deterministic non-degenerate series: `scale * (sin + slow drift)`.
    fn shaped_series(n: usize, scale: f32, phase: f32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f32;
                scale * ((t * 0.31 + phase).sin() + 0.35 * (t * 0.07).cos())
            })
            .collect()
    }

    /// The defect this repair exists for. Force magnitudes are O(10^3) and KCC
    /// magnitudes are O(1) — the exact scale separation that pinned the old
    /// `1/(1 + |mean_abs_a - mean_abs_b|)` score at ~1e-3 against a 0.5
    /// threshold. With the shapes identical the scale-free metric must score
    /// at the top of its range, so the gate is reachable.
    #[test]
    fn spectral_gate_is_reachable_across_a_thousandfold_scale_separation() {
        let dir = tempfile::tempdir().unwrap();
        let force_path = dir.path().join("forces.bin");
        let kcc_path = dir.path().join("kcc.bin");
        write_flat_f32(&force_path, &shaped_series(4096, 4221.0, 0.0));
        write_flat_f32(&kcc_path, &shaped_series(4096, 1.0, 0.0));

        let force = numeric_signal_from_file(&force_path, "unclassified").unwrap();
        let kcc = numeric_signal_from_file(&kcc_path, "unclassified").unwrap();

        // The magnitude statistics are three orders of magnitude apart: this is
        // the input that made the superseded metric unreachable.
        assert!(force.mean_abs / kcc.mean_abs > 1.0e3);
        let superseded = 1.0 / (1.0 + (force.mean_abs - kcc.mean_abs).abs());
        assert!(
            superseded < 1.0e-3,
            "superseded magnitude metric should be pinned near zero, got {superseded}"
        );

        let spectral = spectral_recurrence(&force, &kcc);
        assert!(spectral.is_measured(), "status={}", spectral.status);
        assert!(
            spectral.score >= DEFAULT_SPECTRAL_RECURRENCE_THRESHOLD,
            "scale-free score {} must clear the {} threshold",
            spectral.score,
            DEFAULT_SPECTRAL_RECURRENCE_THRESHOLD
        );
        assert!((0.0..=1.0).contains(&spectral.delta));
    }

    /// The stated invariance property: `x -> a*x + b` for any `a != 0` leaves
    /// the signature, and therefore the comparison, unchanged.
    #[test]
    fn spectral_metric_is_invariant_under_affine_rescaling() {
        let dir = tempfile::tempdir().unwrap();
        let base_path = dir.path().join("base.bin");
        let affine_path = dir.path().join("affine.bin");
        let base = shaped_series(2048, 1.0, 0.0);
        let affine: Vec<f32> = base.iter().map(|v| -37.5 * v + 912.0).collect();
        write_flat_f32(&base_path, &base);
        write_flat_f32(&affine_path, &affine);

        let a = numeric_signal_from_file(&base_path, "unclassified").unwrap();
        let b = numeric_signal_from_file(&affine_path, "unclassified").unwrap();
        for (ra, rb) in a.autocorrelation.iter().zip(b.autocorrelation.iter()) {
            // `a < 0` flips the sign of the deviations in both factors, so the
            // product is unchanged; the signature is identical, not mirrored.
            assert!(
                (ra - rb).abs() < 1.0e-4,
                "autocorrelation not affine-invariant: {ra} vs {rb}"
            );
        }
        let spectral = spectral_recurrence(&a, &b);
        assert!(spectral.is_measured());
        assert!(spectral.score > 0.999, "score={}", spectral.score);
    }

    /// The gate must be able to fail as well as pass, or it governs nothing.
    ///
    /// "Opposed" is defined by the metric, not by eye: the two signatures must
    /// disagree at most lags, so the mean per-lag disagreement exceeds one unit
    /// of the two-unit maximum and `delta > 0.5`. A smooth ramp recurs in
    /// phase at every lag (`r_k ≈ +1`). An oscillation of period 12 samples
    /// has its half-period at lag 6, so `r_4..r_8` are all in anti-phase
    /// (`r_6 ≈ −1`) and `Σ r_k ≈ −2.36`. Expected `delta ≈ 0.644`,
    /// `score ≈ 0.356`, well below the 0.5 default.
    #[test]
    fn spectral_gate_rejects_structurally_opposed_signatures() {
        let dir = tempfile::tempdir().unwrap();
        let slow_path = dir.path().join("slow.bin");
        let oscillation_path = dir.path().join("oscillation.bin");
        write_flat_f32(
            &slow_path,
            &(0..2048).map(|i| (i as f32) * 0.001).collect::<Vec<f32>>(),
        );
        // Exact period-12 lookup, `cos(2π·j/12)`: the periodicity is exact in
        // f32 and independent of large-argument trig rounding.
        let period_12: [f32; 12] =
            std::array::from_fn(|j| (2.0 * std::f64::consts::PI * j as f64 / 12.0).cos() as f32);
        write_flat_f32(
            &oscillation_path,
            &(0..2048).map(|i| period_12[i % 12]).collect::<Vec<f32>>(),
        );
        let a = numeric_signal_from_file(&slow_path, "unclassified").unwrap();
        let b = numeric_signal_from_file(&oscillation_path, "unclassified").unwrap();
        assert!(a.autocorrelation_defined && b.autocorrelation_defined);
        // The fixture's premise is asserted, not assumed.
        assert!(
            a.autocorrelation.iter().all(|r| *r > 0.95),
            "ramp must recur in phase at every lag: {:?}",
            a.autocorrelation
        );
        assert!(
            b.autocorrelation[3..8].iter().all(|r| *r < -0.45),
            "period-12 oscillation must be in anti-phase at lags 4..8: {:?}",
            b.autocorrelation
        );
        assert!(
            b.autocorrelation[5] < -0.99,
            "r_6 must sit at the half-period: {}",
            b.autocorrelation[5]
        );
        let sum_b: f64 = b.autocorrelation.iter().sum();
        assert!(
            sum_b < -2.0,
            "Σ r_k of the oscillation should be ≈ −2.36, got {sum_b}"
        );

        let spectral = spectral_recurrence(&a, &b);
        assert!(spectral.is_measured(), "status={}", spectral.status);
        assert!(
            spectral.delta > 0.6,
            "mean per-lag disagreement must exceed one unit with margin, delta={}",
            spectral.delta
        );
        assert!(
            spectral.score < DEFAULT_SPECTRAL_RECURRENCE_THRESHOLD,
            "opposed signatures must fail the gate, score={}",
            spectral.score
        );
    }

    /// Regression guard for the fixture that used to stand in for opposition.
    /// A strict ±1 alternation is NOT opposed to a smooth series under this
    /// metric: `r_k = (−1)^k (1 − k/n)`, so it agrees with the ramp at every
    /// even lag and disagrees by ~2 at every odd lag. That is exactly one unit
    /// of the two-unit per-lag maximum — the metric midpoint — and the ramp's
    /// finite-sample slope `r_k ≈ 1 − 3k/n` lands it at `0.5 + 1.5/n`, a hair
    /// on the passing side of the 0.5 default. A test that treats this pair as
    /// "opposed" is testing a knife edge, not the gate.
    #[test]
    fn alternation_against_a_ramp_scores_the_midpoint_not_opposition() {
        let dir = tempfile::tempdir().unwrap();
        let slow_path = dir.path().join("slow.bin");
        let alternating_path = dir.path().join("alternating.bin");
        let n = 2048usize;
        write_flat_f32(
            &slow_path,
            &(0..n).map(|i| (i as f32) * 0.001).collect::<Vec<f32>>(),
        );
        write_flat_f32(
            &alternating_path,
            &(0..n)
                .map(|i| if i % 2 == 0 { 1.0f32 } else { -1.0f32 })
                .collect::<Vec<f32>>(),
        );
        let a = numeric_signal_from_file(&slow_path, "unclassified").unwrap();
        let b = numeric_signal_from_file(&alternating_path, "unclassified").unwrap();
        for (index, (ra, rb)) in a
            .autocorrelation
            .iter()
            .zip(b.autocorrelation.iter())
            .enumerate()
        {
            let lag = index + 1;
            if lag % 2 == 0 {
                assert!(
                    (ra - rb).abs() < 0.01,
                    "even lag {lag} must agree: {ra} vs {rb}"
                );
            } else {
                assert!(
                    (ra - rb).abs() > 1.98,
                    "odd lag {lag} must disagree: {ra} vs {rb}"
                );
            }
        }
        let spectral = spectral_recurrence(&a, &b);
        assert!(spectral.is_measured());
        let expected = 0.5 + 1.5 / n as f64;
        assert!(
            (spectral.score - expected).abs() < 1.0e-4,
            "midpoint expected {expected}, got {}",
            spectral.score
        );
    }

    /// A constant series has zero centred variance: `r_k` is 0/0. That is an
    /// absence of measurement, and it must not be reported as a low score.
    #[test]
    fn constant_series_yields_no_measurement_not_a_low_score() {
        let dir = tempfile::tempdir().unwrap();
        let flat_path = dir.path().join("flat.bin");
        let shaped_path = dir.path().join("shaped.bin");
        write_flat_f32(&flat_path, &vec![7.25f32; 2048]);
        write_flat_f32(&shaped_path, &shaped_series(2048, 1.0, 0.0));
        let flat = numeric_signal_from_file(&flat_path, "unclassified").unwrap();
        let shaped = numeric_signal_from_file(&shaped_path, "unclassified").unwrap();
        assert!(!flat.autocorrelation_defined);
        let spectral = spectral_recurrence(&flat, &shaped);
        assert!(!spectral.is_measured());
        assert_eq!(spectral.status, "not_measurable_degenerate_signal");
        assert!(!spectral.score.is_finite());
        // Non-finite must serialize out of band, never as a number in [0, 1].
        assert_eq!(serde_json::to_string(&spectral.score).unwrap(), "null");
    }

    /// An absent artifact is not a failed comparison.
    #[test]
    fn missing_artifact_yields_missing_status_not_zero() {
        let present = NumericSignal {
            n_samples: 1024,
            autocorrelation: vec![0.5; SPECTRAL_SIGNATURE_LAGS],
            autocorrelation_defined: true,
            schema_status: SignalSchemaStatus::UndeclaredFlatF32Le,
            ..NumericSignal::default()
        };
        let spectral = spectral_recurrence(&NumericSignal::default(), &present);
        assert_eq!(spectral.status, "not_measurable_missing_artifact");
        assert!(!spectral.delta.is_finite());
    }

    // ── Read extent and declared schema ────────────────────────────────────

    /// Statistics must describe the whole artifact, and the receipt must say
    /// which extent was read. The superseded reader took a 1 MiB prefix and
    /// recorded nothing about having done so.
    #[test]
    fn whole_file_read_is_recorded_with_full_coverage() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wide.bin");
        // Larger than the superseded 1 MiB prefix cap.
        let samples = 1024 * 1024 + 4096;
        write_flat_f32(&path, &shaped_series(samples, 3.0, 0.0));
        let signal = numeric_signal_from_file(&path, "unclassified").unwrap();
        assert_eq!(signal.read_mode, SignalReadMode::WholeFile);
        assert_eq!(signal.n_samples, samples);
        assert_eq!(signal.bytes_considered, signal.file_bytes);
        assert!((signal.coverage_fraction - 1.0).abs() < 1.0e-12);
        assert_eq!(
            signal.provenance().get("read_mode").map(String::as_str),
            Some("whole_file")
        );
    }

    /// The declared 12-byte vec3 record stride is honoured: the series is one
    /// magnitude per atom, not three interleaved components.
    #[test]
    fn declared_vec3_stride_is_honoured_for_forces_final() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("forces_final.bin");
        let atoms = 4096;
        let mut components = Vec::with_capacity(atoms * 3);
        for i in 0..atoms {
            let t = i as f32;
            components.push((t * 0.13).sin());
            components.push((t * 0.13).cos());
            components.push(0.5 * (t * 0.05).sin());
        }
        write_flat_f32(&path, &components);
        let signal = numeric_signal_from_file(&path, "forces_final").unwrap();
        assert_eq!(signal.schema_status, SignalSchemaStatus::RawF32Vec3Aos);
        assert_eq!(signal.n_samples, atoms);
        assert!(signal.mean_abs > 0.0);

        let undeclared = numeric_signal_from_file(&path, "unclassified").unwrap();
        assert_eq!(
            undeclared.schema_status,
            SignalSchemaStatus::UndeclaredFlatF32Le
        );
        assert_eq!(undeclared.n_samples, atoms * 3);
    }

    fn fnv1a_64(bytes: &[u8]) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    fn build_prkcc001(records: usize, fields: &[(&str, Vec<f32>)]) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(&(records as u64).to_le_bytes());
        payload.extend_from_slice(&(fields.len() as u64).to_le_bytes());
        for (name, values) in fields {
            payload.extend_from_slice(&(name.len() as u64).to_le_bytes());
            payload.extend_from_slice(name.as_bytes());
            payload.push(1u8); // dtype f32
            payload.extend_from_slice(&((values.len() * 4) as u64).to_le_bytes());
            for value in values {
                payload.extend_from_slice(&value.to_le_bytes());
            }
        }
        let mut raw = Vec::new();
        raw.extend_from_slice(b"PRKCC001");
        raw.extend_from_slice(&1u32.to_le_bytes());
        raw.extend_from_slice(&0x0102_0304u32.to_le_bytes());
        raw.extend_from_slice(&0u32.to_le_bytes());
        raw.extend_from_slice(&(3u64).to_le_bytes());
        raw.extend_from_slice(b"run");
        raw.extend_from_slice(&(4u64).to_le_bytes());
        raw.extend_from_slice(b"stem");
        raw.extend_from_slice(&(records as u64).to_le_bytes());
        raw.extend_from_slice(&4u64.to_le_bytes());
        raw.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        let checksum = fnv1a_64(&payload);
        raw.extend_from_slice(&payload);
        raw.extend_from_slice(&checksum.to_le_bytes());
        raw.extend_from_slice(b"PRKCCEND");
        raw
    }

    /// The declared PRKCC001 schema is parsed, not blindly reinterpreted: the
    /// header bytes never enter the statistics and the field sections are named
    /// in the receipt.
    #[test]
    fn prkcc001_header_is_parsed_and_field_sections_are_named() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("stream00_kcc_v2full.bin");
        let records = 512;
        let raw = build_prkcc001(
            records,
            &[
                ("direction_score", shaped_series(records, 1.0, 0.0)),
                ("temporal_corr", shaped_series(records, 2.0, 0.4)),
            ],
        );
        std::fs::write(&path, &raw).unwrap();
        let signal = numeric_signal_from_file(&path, "kcc_v2full").unwrap();
        assert_eq!(
            signal.schema_status,
            SignalSchemaStatus::Prkcc001FieldSections
        );
        assert_eq!(signal.decoded_fields.len(), 2);
        assert!(signal
            .decoded_fields
            .contains(&"direction_score".to_string()));
        // Two field sections, so two contiguous segments — adjacency is never
        // asserted across the boundary between two different fields.
        assert_eq!(signal.segment_count, 2);
        assert_eq!(signal.n_samples, records * 2);
        // Header and trailer bytes are excluded: a blind f32 reinterpretation
        // would have produced strictly more samples than the payload holds.
        assert!(signal.n_samples < raw.len() / 4);
        assert!(signal.autocorrelation_defined);
    }

    /// A declared envelope this reader cannot decode is refused. It is never
    /// reinterpreted as a flat f32 array, because that would feed header bytes
    /// into a statistic and call the result a measurement.
    #[test]
    fn undecodable_envelope_is_refused_not_reinterpreted() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("stream00_spikes.bin");
        let mut raw = Vec::new();
        raw.extend_from_slice(b"PRSPK001");
        raw.extend(std::iter::repeat(0x41u8).take(8192));
        std::fs::write(&path, &raw).unwrap();
        let signal = numeric_signal_from_file(&path, "spikes").unwrap();
        assert_eq!(
            signal.schema_status,
            SignalSchemaStatus::DeclaredEnvelopeNotDecodable
        );
        assert_eq!(signal.n_samples, 0);
        assert_eq!(signal.read_mode, SignalReadMode::NotRead);
        let spectral = spectral_recurrence(&signal, &signal);
        assert_eq!(spectral.status, "not_measurable_schema_not_decodable");
    }

    // ── C6 no-producer marker ──────────────────────────────────────────────

    /// Zero salience events must be distinguishable from a producer that never
    /// ran. The count cannot carry that distinction, because zero is a
    /// legitimate count, so it travels in a typed status field instead.
    #[test]
    fn no_producer_marker_distinguishes_absence_from_a_legitimate_zero() {
        let config = TshRuntimeConfig::new("run_1", "target");
        let (inputs, report) = build_observable_inputs_with_producer_status(&config, &[], 7);
        assert!(inputs.is_empty());
        assert_eq!(report.producer_status, SALIENCE_PRODUCER_NOT_WIRED);
        assert!(!report.producer_wired);
        assert_eq!(report.salience_event_count, 0);
        assert_eq!(
            report.required_artifact_kinds_missing,
            vec!["forces_final".to_string(), "kcc_v2full".to_string()]
        );

        // Same zero count, opposite meaning.
        let mut wired = SalienceProducerReport::from_caller_supplied_inputs(3);
        wired.record_salience_events(0);
        assert!(wired.producer_wired);
        assert_eq!(wired.producer_status, SALIENCE_PRODUCER_WIRED_ZERO_EVENTS);
        assert_ne!(wired.producer_status, report.producer_status);
    }

    /// A present-but-unresolvable artifact set is wired-with-no-inputs, which
    /// is neither "never ran" nor "ran and found nothing".
    #[test]
    fn present_artifacts_without_stream_ids_report_not_wired() {
        let config = TshRuntimeConfig::new("run_1", "target");
        let refs = vec![RuntimeArtifactRef {
            kind: "forces_final".to_string(),
            stream_id: None,
            path: PathBuf::from("/nonexistent/forces_final.bin"),
            present: true,
        }];
        let (inputs, report) = build_observable_inputs_with_producer_status(&config, &refs, 7);
        assert!(inputs.is_empty());
        assert_eq!(report.producer_status, SALIENCE_PRODUCER_NOT_WIRED);
        assert_eq!(report.artifact_refs_offered, 1);
        assert_eq!(
            report.artifact_kinds_present,
            vec!["forces_final".to_string()]
        );
        assert_eq!(
            report.required_artifact_kinds_missing,
            vec!["kcc_v2full".to_string()]
        );
    }

    /// The producer marker survives into the emitted manifest and its own
    /// sidecar, so a consumer reading only the receipt still sees it.
    #[test]
    fn producer_marker_is_emitted_into_the_receipt_and_a_sidecar() {
        let dir = tempfile::tempdir().unwrap();
        let uncertainties = vec![
            uncertainty("force"),
            uncertainty("pocket_volume"),
            uncertainty("contact_probability"),
        ];
        let artifacts = observer()
            .evaluate(&[input("seed1")], &uncertainties, Some(&BTreeSet::new()))
            .unwrap();
        let receipt = write_tsh_artifacts(dir.path(), &artifacts).unwrap();
        assert!(receipt.salience_producer_wired);
        assert_eq!(
            receipt.salience_producer_status,
            SALIENCE_PRODUCER_WIRED_EVENTS_EMITTED
        );
        let sidecar = dir.path().join("tsh_salience_producer_status.json");
        assert!(sidecar.exists());
        let value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&sidecar).unwrap()).unwrap();
        assert_eq!(
            value.get("schema_kind").and_then(|v| v.as_str()),
            Some("tsh_salience_producer_status")
        );
        assert_eq!(
            value.get("producer_wired").and_then(|v| v.as_bool()),
            Some(true)
        );
    }

    /// An unmeasurable spectral comparison is omitted from the evidence mean
    /// rather than averaged in as a number.
    #[test]
    fn unmeasurable_spectral_component_is_excluded_from_the_evidence_mean() {
        let uncertainties = vec![
            uncertainty("force"),
            uncertainty("pocket_volume"),
            uncertainty("contact_probability"),
        ];
        let mut unmeasured = input("seed1");
        unmeasured.spectral_signature_delta = f64::NAN;
        unmeasured.spectral_signature_status = "not_measurable_missing_artifact".to_string();
        unmeasured.spectral_signature_lags = 0;
        let out = observer()
            .evaluate(&[unmeasured], &uncertainties, Some(&BTreeSet::new()))
            .unwrap();
        let event = &out.salience_events[0];
        assert!(!event.spectral_similarity_score.is_finite());
        assert!(event.scientific_evidence_score.is_finite());
        assert_eq!(event.scientific_evidence_components.len(), 4);
        assert!(!event
            .scientific_evidence_components
            .contains(&"spectral_similarity_score".to_string()));
        assert_eq!(
            serde_json::to_value(event)
                .unwrap()
                .get("spectral_similarity_score"),
            Some(&serde_json::Value::Null)
        );
    }
    #[test]
    fn measured_binding_preserves_interval_sigma_and_rejects_ambiguity() {
        let mut refs=Vec::new();let mut inputs=Vec::new();
        for index in 0..2 {
            let mut row=input("seed42");
            for observable in ["force","pocket_volume","contact_probability"] {
                let mut u=uncertainty(observable);u.tensor_id=format!("{}:{index}",u.tensor_id);u.sigma=2.0+index as f64*2.0;
                row.uncertainty_tensor_refs.push(u.tensor_id.clone());refs.push(u);
            }
            inputs.push(row);
        }
        refs.reverse();
        let o=TshRuntimeObserver::offline(observer().config);
        let out=o.evaluate(&inputs,&refs,None).unwrap();
        assert_eq!(out.salience_events[0].force_delta_snr,6.0);
        assert_eq!(out.salience_events[1].force_delta_snr,3.0);
        assert_eq!(out.salience_events[0].runtime_origin,"OFFLINE_MEASURED_TSH_STAGE_II_OBSERVER");
        assert!(out.salience_events[0].control_rejection_score.is_nan());
        assert!(!out.salience_events[0].scientific_evidence_components.iter().any(|v|v=="control_rejection_score"));
        let encoded=serde_json::to_value(&out.salience_events[0]).unwrap();
        assert!(encoded["control_rejection_score"].is_null());
        let decoded:ThermodynamicSalienceEvent=serde_json::from_value(encoded).unwrap();
        assert!(decoded.control_rejection_score.is_nan());
        assert_eq!(out.salience_producer.runtime_origin,out.resident_cache.runtime_origin);
        assert!(out.persistent_transition_objects.is_empty());
        inputs[0].uncertainty_tensor_refs.clear();assert!(o.evaluate(&inputs,&refs,None).is_err());
        refs.push(refs[0].clone());assert!(o.evaluate(&[input("x")],&refs,None).is_err());
    }
    #[test]
    fn invalid_measurement_is_not_snr_and_true_zero_stays_zero() {
        let refs=vec![uncertainty("force"),uncertainty("pocket_volume"),uncertainty("contact_probability")];
        let mut row=input("seed42");row.force_delta=f64::NAN;
        assert!(observer().evaluate(&[row.clone()],&refs,None).is_err());
        row.force_delta=0.0;assert_eq!(observer().evaluate(&[row.clone()],&refs,None).unwrap().salience_events[0].force_delta_snr,0.0);
        row.step_end=row.step_start-1;assert!(observer().evaluate(&[row],&refs,None).is_err());
        assert_eq!(snr(1.0e-16,2.0e-16),0.5);
    }
    #[test]
    fn measured_precision_retains_units_degeneracy_and_spectral_minimum() {
        let series:Vec<f64>=(0..64).map(|i|(i as f64*0.3).sin()).collect();
        let tiny:Vec<f64>=series.iter().map(|x|x*1e-12).collect();
        let a=measured_temporal_summary(&series).unwrap();let b=measured_temporal_summary(&tiny).unwrap();
        assert!((a.effective_sample_size-b.effective_sample_size).abs()<1e-10);
        assert!((a.standard_error*1e-12-b.standard_error).abs()<1e-25);
        assert_eq!(measured_temporal_summary(&[2.0;32]).unwrap().standard_error,0.0);
        // Actual contact-occupancy difference: summing 46 identical values
        // used to manufacture variance and a ~5.9e15 SNR through roundoff.
        for value in [0.015151515151515166, -0.015151515151515166, 0.0] {
            for n in [46, 48] {
                let constant=measured_temporal_summary(&vec![value;n]).unwrap();
                assert_eq!(constant.mean,value);
                assert_eq!(constant.sample_variance,0.0);
                assert_eq!(constant.standard_error,0.0);
                assert_eq!(constant.integrated_autocorrelation_time_samples,1.0);
                assert_eq!(constant.effective_sample_size,n as f64);
            }
        }
        assert_eq!(measured_temporal_summary(&[-0.0,0.0]).unwrap().standard_error,0.0);
        assert!(measured_temporal_summary(&[2.0]).is_err());
        assert!(!measured_time_signal(&series[..31]).unwrap().autocorrelation_defined);
        assert!(measured_time_signal(&series[..32]).unwrap().autocorrelation_defined);
        assert!(measured_time_signal(&[f64::NAN]).is_err());
    }

}

/// Reuse the native autocorrelation reducer without downcasting measured f64 values.
pub fn measured_time_signal(values: &[f64]) -> Result<NumericSignal> {
    if values.is_empty() || values.iter().any(|v| !v.is_finite()) { bail!("finite nonempty measured series required"); }
    let mut signal = finish_signal(&[values.to_vec()]);
    signal.schema_status = SignalSchemaStatus::MeasuredF64TimeSeries;
    signal.decoded_fields = vec!["chronology_verified_scalar_observable".to_string()];
    Ok(signal)
}

// Offline measured replay reuses the native salience, spectral and ESS paths.
// It never constructs physical observations from event counts or static ranks.
#[derive(Clone, Debug, Deserialize, Serialize)]
struct MeasuredSourcePin { path: PathBuf, sha256: String }
impl MeasuredSourcePin {
    fn verify(&self) -> Result<()> {
        if sha256_file(&self.path)?.trim_start_matches("sha256:") != self.sha256.trim_start_matches("sha256:") {
            bail!("measured source hash mismatch: {}", self.path.display());
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Deserialize)]
struct MeasuredRegionSupport {
    region_id: u32, residue_ids: Vec<u32>, atom_indices: Vec<usize>, pair_indices: Vec<usize>,
}
#[derive(Deserialize)]
struct MeasuredPairManifest {
    schema_kind: String, run_id: String, target_id: String, seed_id: String,
    topology_sha256: String, n_atoms: usize, cadence_steps: u64, dt_ps: Option<f64>,
    #[serde(default)] stream_id: u16,
    #[serde(default)] reference_time_arm: Option<String>,
    #[serde(default)] physical_time_journals: BTreeMap<String, crate::chemical_state_fiber::mechanistic::physical_time::PhysicalTimeJournalAuthority>,
    #[serde(default)] coordinate_audits: BTreeMap<String, MeasuredSourcePin>,
    force_completed_step_offset: u64, frame_count: usize, pair_count: usize,
    source_files: Vec<MeasuredSourcePin>, force_treatment: MeasuredSourcePin,
    force_control: MeasuredSourcePin, force_chronology_receipt: MeasuredSourcePin,
    frames: MeasuredSourcePin, regions: Vec<MeasuredRegionSupport>,
}
#[derive(Deserialize)]
struct MeasuredRegionFrame {
    region_id: u32, volume: [f64; 2], grid_sem: [f64; 2],
    boundary_area: [f64; 2], clipped: [bool; 2], record_sha256: [String; 2],
}
#[derive(Deserialize)]
struct MeasuredPairFrame {
    step: u64, time_ps: Option<f64>, phase: String,
    treatment_ca_distances_a: Vec<f64>, control_ca_distances_a: Vec<f64>,
    regions: Vec<MeasuredRegionFrame>,
}
#[derive(Clone)]
struct MeasuredSample {
    step: u64, time_ps: f64, arm_times_ps: [f64;2], phase: String, delta: [f64; 3], grid_sem: [f64; 2],
}
#[derive(Debug, Serialize)]
struct MeasuredTemporalSummary {
    n_samples: usize, mean: f64, sample_variance: f64, empirical_standard_deviation: f64,
    integrated_autocorrelation_time_samples: f64, effective_sample_size: f64,
    standard_error: f64,
}
fn measured_temporal_summary(values: &[f64]) -> Result<MeasuredTemporalSummary> {
    if values.len() < 2 || values.iter().any(|v| !v.is_finite()) { bail!("at least two finite measured samples required"); }
    let n = values.len();
    // Exact equality, not a tolerance or a significance threshold. Preserve
    // a constant observation exactly rather than manufacturing fluctuations
    // from rounding its sum. Zero sigma remains unavailable for SNR scoring.
    if values.iter().all(|v| *v == values[0]) {
        return Ok(MeasuredTemporalSummary { n_samples:n, mean:values[0],
            sample_variance:0.0, empirical_standard_deviation:0.0,
            integrated_autocorrelation_time_samples:1.0,
            effective_sample_size:n as f64, standard_error:0.0 });
    }
    let mean = values.iter().sum::<f64>() / n as f64;
    let variance = values.iter().map(|v| (v-mean).powi(2)).sum::<f64>() / (n-1) as f64;
    let sd = variance.sqrt();
    // Normalize only for the existing ESS estimator, avoiding its absolute
    // epsilon cutoff changing ESS when physical units are rescaled.
    let standardized: Vec<f64> = values.iter().map(|v| if sd > 0.0 { (v-mean)/sd } else { 0.0 }).collect();
    let (tau, ess) = crate::pathb_chemical_state::integrated_autocorrelation_ess(&standardized)?;
    Ok(MeasuredTemporalSummary { n_samples:n, mean, sample_variance:variance,
        empirical_standard_deviation:sd, integrated_autocorrelation_time_samples:tau,
        effective_sample_size:ess, standard_error:(variance/ess).sqrt() })
}
fn measured_force_series(pin: &MeasuredSourcePin, manifest: &MeasuredPairManifest)
    -> Result<BTreeMap<u64, Vec<f64>>> {
    let mut reader = BufReader::new(File::open(&pin.path)?);
    let mut header = [0u8;8]; reader.read_exact(&mut header)?;
    if u64::from_le_bytes(header) != manifest.n_atoms as u64 { bail!("ASC atom count mismatch"); }
    let record_bytes = 16 + manifest.n_atoms * 12;
    let size = std::fs::metadata(&pin.path)?.len() as usize;
    if size < 8 || (size-8)%record_bytes != 0 { bail!("ASC partial record"); }
    let mut raw = vec![0u8;record_bytes];
    let mut result = BTreeMap::new(); let mut prior = None;
    for _ in 0..(size-8)/record_bytes {
        reader.read_exact(&mut raw)?;
        let chunk = u64::from_le_bytes(raw[..8].try_into()?);
        let stored = u64::from_le_bytes(raw[8..16].try_into()?);
        if chunk.checked_mul(manifest.cadence_steps) != Some(stored) { bail!("ASC chunk/step disagreement"); }
        let step = stored.checked_add(manifest.force_completed_step_offset).context("ASC clock overflow")?;
        if prior.is_some_and(|p| p + manifest.cadence_steps != step) { bail!("ASC chronology gap or duplicate"); }
        prior = Some(step);
        let mut norm2 = Vec::with_capacity(manifest.n_atoms);
        for atom in raw[16..].chunks_exact(12) {
            let mut sum=0.0;
            for bytes in atom.chunks_exact(4) {
                let v=f32::from_le_bytes(bytes.try_into()?) as f64;
                if !v.is_finite() { bail!("nonfinite recorded force"); } sum+=v*v;
            }
            norm2.push(sum);
        }
        let rms = manifest.regions.iter().map(|region|
            (region.atom_indices.iter().map(|i|norm2[*i]).sum::<f64>() / region.atom_indices.len() as f64).sqrt()
        ).collect();
        if result.insert(step,rms).is_some() { bail!("duplicate ASC completed step"); }
    }
    Ok(result)
}

/// Execute the native observer on complete, source-pinned saved paired data.
/// Uses conditional within-segment sampling precision, not between-seed or
/// experimental uncertainty. Every source region/frame and skipped interval
/// remains materialized, including unavailable force and clipped geometry.
pub fn observe_measured_pair(manifest_path: &Path, output_dir: &Path) -> Result<serde_json::Value> {
    use serde_json::json;
    use std::io::BufWriter;
    let manifest: MeasuredPairManifest = serde_json::from_reader(File::open(manifest_path)?)?;
    let m=&manifest;
    if m.schema_kind != "prism_tsh_measured_pair_inputs_v1" || m.n_atoms==0 || m.cadence_steps==0
        || (m.physical_time_journals.is_empty() && !m.dt_ps.is_some_and(|dt|dt.is_finite()&&dt>0.0))
        || m.regions.is_empty() || m.pair_count==0 {
        bail!("invalid measured pair manifest");
    }
    for pin in m.source_files.iter().chain([&m.force_treatment,&m.force_control,&m.force_chronology_receipt,&m.frames]) { pin.verify()?; }
    let manifest_hash=sha256_file(manifest_path)?;
    let mut time_axes=BTreeMap::new();let mut time_audits=BTreeMap::new();
    let reference_time_arm=if m.physical_time_journals.is_empty() {None}else {
        let arm=m.reference_time_arm.as_deref().context("exact TSH clocks require explicit reference_time_arm")?;
        if !matches!(arm,"treatment"|"control") || m.physical_time_journals.len()!=2 || m.coordinate_audits.len()!=2 {
            bail!("exact TSH time requires both physical arms/audits and one declared reference arm");
        }
        for name in ["treatment","control"] {
            let authority=m.physical_time_journals.get(name).context("missing TSH native arm clock")?;
            if authority.source_stream_id!=m.stream_id as u64 {bail!("TSH journal stream differs from declared physical lane");}
            time_axes.insert(name,crate::chemical_state_fiber::mechanistic::physical_time::VerifiedPhysicalTimeAxis::load(authority)?);
            let pin=m.coordinate_audits.get(name).context("missing TSH coordinate clock binding")?;pin.verify()?;
            let audit:serde_json::Value=serde_json::from_reader(File::open(&pin.path)?)?;
            if audit["stream_id"].as_u64()!=Some(m.stream_id as u64)
                || audit["frames"].as_array().map(Vec::len)!=Some(m.frame_count) {bail!("TSH coordinate audit stream/population mismatch");}
            time_audits.insert(name,audit);
        }
        Some(arm)
    };
    let qc: serde_json::Value=serde_json::from_reader(File::open(&m.force_chronology_receipt.path)?)?;
    if qc["status"]!="IMPLEMENTATION_VERIFIED" || m.force_completed_step_offset != m.cadence_steps { bail!("unverified force correspondence"); }
    for (arm,pin) in [("treatment",&m.force_treatment),("uv_off_control",&m.force_control)] {
        let q=&qc["arms"][arm];
        let sources=q["sources"].as_array().context("force chronology arm missing")?;
        if !sources.iter().any(|p| p["sha256"].as_str()==Some(pin.sha256.trim_start_matches("sha256:")))
            || q["atoms"].as_u64()!=Some(m.n_atoms as u64)
            || q["units"]!="AMBER kcal/mol/Angstrom" || q["content_identity"]!="TOTAL_AMBER_FORCE_BUFFER" { bail!("force chronology proof does not bind input {arm}"); }
        let stored=q["stored_asc_step_first_last"].as_array().context("stored force range missing")?;
        let completed=q["actual_completed_step_first_last"].as_array().context("completed force range missing")?;
        for i in 0..2 { if stored[i].as_u64().and_then(|s|s.checked_add(m.force_completed_step_offset))!=completed[i].as_u64() { bail!("force completion offset does not match verified receipt"); } }
    }
    let mut ids=BTreeSet::new();
    for region in &m.regions {
        if !ids.insert(region.region_id) || region.atom_indices.is_empty() || region.pair_indices.is_empty()
            || region.residue_ids.is_empty() || region.atom_indices.iter().any(|i|*i>=m.n_atoms)
            || region.pair_indices.iter().any(|i|*i>=m.pair_count)
            || region.atom_indices.iter().collect::<BTreeSet<_>>().len()!=region.atom_indices.len()
            || region.pair_indices.iter().collect::<BTreeSet<_>>().len()!=region.pair_indices.len() { bail!("invalid fixed region support"); }
    }
    let treatment=measured_force_series(&m.force_treatment,m)?;
    let control=measured_force_series(&m.force_control,m)?;
    if treatment.keys().ne(control.keys()) { bail!("paired force clocks differ"); }
    for (arm,series) in [("treatment",&treatment),("uv_off_control",&control)] {
        let q=&qc["arms"][arm];
        if q["records"].as_u64()!=Some(series.len() as u64)
            || q["actual_completed_step_first_last"][0].as_u64()!=series.keys().next().copied()
            || q["actual_completed_step_first_last"][1].as_u64()!=series.keys().next_back().copied() { bail!("decoded force population differs from verified receipt"); }
    }
    if output_dir.exists() { bail!("refusing to overwrite {}",output_dir.display()); }
    std::fs::create_dir_all(output_dir)?;
    let mut measured=BufWriter::new(File::create(output_dir.join("measured_region_frames.jsonl"))?);
    let mut clock_index=if reference_time_arm.is_some(){Some(BufWriter::new(File::create(output_dir.join("physical_time_endpoint_index.jsonl"))?))}else{None};
    let mut segments: Vec<(usize,Vec<MeasuredSample>)>=Vec::new();
    let mut current:Vec<Vec<MeasuredSample>>=vec![Vec::new();m.regions.len()];
    let mut frame_count=0;let mut prior=None;let mut available=0usize;let mut counts=BTreeMap::<String,usize>::new();
    for line in BufReader::new(File::open(&m.frames.path)?).lines() {
        let frame:MeasuredPairFrame=serde_json::from_str(&line?)?;
        if frame.treatment_ca_distances_a.len()!=m.pair_count || frame.control_ca_distances_a.len()!=m.pair_count
            || frame.regions.len()!=m.regions.len()
            || prior.is_some_and(|p|p+m.cadence_steps!=frame.step) { bail!("measured frame chronology or population mismatch"); }
        let (arm_times_ps,reference_time_ps)=if let Some(reference)=reference_time_arm {
            let mut endpoints=serde_json::Map::new();
            for arm in ["treatment","control"] {
                let endpoint=time_axes[arm].endpoint(frame.step)?;
                let audit=&time_audits[arm]["frames"][frame_count];
                if audit["step"].as_u64()!=Some(frame.step) || audit["exact_cumulative_time_fs"].as_f64()
                    !=Some(endpoint.native_record.cumulative_simulated_time_ps*1000.0) {
                    bail!("TSH coordinate/journal exact endpoint mismatch");
                }
                endpoints.insert(arm.into(),serde_json::to_value(endpoint)?);
            }
            let timeline=json!({"step":frame.step,"time_axis_scope":"exact_native_per_arm_step_sum","arm_physical_time":endpoints});
            let times=[crate::chemical_state_fiber::mechanistic::physical_time::timeline_arm_time(&timeline,"treatment")?,
                crate::chemical_state_fiber::mechanistic::physical_time::timeline_arm_time(&timeline,"control")?];
            serde_json::to_writer(clock_index.as_mut().unwrap(),&json!({"frame_ordinal":frame_count,"native_time_frame":timeline,
                "reference_time_arm":reference,"stream_id":m.stream_id,"coordinate_audits":m.coordinate_audits}))?;
            writeln!(clock_index.as_mut().unwrap())?;
            (times,crate::chemical_state_fiber::mechanistic::physical_time::timeline_arm_time(&timeline,reference)?)
        } else {
            let time=frame.time_ps.context("historical TSH frame time missing")?;
            if !time.is_finite() || (time-frame.step as f64*m.dt_ps.context("historical dt missing")?).abs()>1e-10 {
                bail!("historical measured frame chronology mismatch");
            }
            ([time,time],time)
        };
        prior=Some(frame.step);frame_count+=1;
        let ft=treatment.get(&frame.step);let fc=control.get(&frame.step);
        let region_rows:BTreeMap<_,_>=frame.regions.iter().map(|r|(r.region_id,r)).collect();
        if region_rows.len()!=m.regions.len() { bail!("duplicate measured region row"); }
        for (i,support) in m.regions.iter().enumerate() {
            let r=region_rows.get(&support.region_id).context("missing measured region")?;
            if r.volume.iter().chain(&r.grid_sem).chain(&r.boundary_area).any(|v|!v.is_finite()||*v<0.0)
                || r.record_sha256.iter().any(|h|h.is_empty()) { bail!("invalid measured geometry/provenance"); }
            let contact = [&frame.treatment_ca_distances_a,&frame.control_ca_distances_a].map(|distances| -> Result<f64> {
                let mut n=0;
                for j in &support.pair_indices { let d=distances[*j];if !d.is_finite()||d<0.0 { bail!("invalid measured CA distance"); }
                    n+=usize::from(d < crate::composition::CA_CONTACT_THRESHOLD_ANGSTROM); }
                Ok(n as f64/support.pair_indices.len() as f64)
            });
            let contacts=[*contact[0].as_ref().map_err(|e|anyhow::anyhow!("{e}"))?,*contact[1].as_ref().map_err(|e|anyhow::anyhow!("{e}"))?];
            let force=ft.zip(fc).map(|(t,c)|[t[i],c[i]]);
            let mut reasons=Vec::new();if force.is_none(){reasons.push("force_history_unavailable");}
            if r.clipped.iter().any(|c|*c){reasons.push("local_window_clipped");}
            for reason in &reasons { *counts.entry((*reason).into()).or_default()+=1; }
            serde_json::to_writer(&mut measured,&json!({"region_id":support.region_id,"step":frame.step,"time_ps":reference_time_ps,"phase":frame.phase,
                "reference_time_arm":reference_time_arm,"arm_times_ps":{"treatment":arm_times_ps[0],"control":arm_times_ps[1]},
                "physical_time_frame_ref":reference_time_arm.map(|_|json!({"artifact":"physical_time_endpoint_index.jsonl","line_ordinal":frame_count-1})),
                "stream_id":m.stream_id,"pair_correspondence":"same MD step; no physical-time interpolation",
                "force_rms_kcal_mol_angstrom":force,"volume_a3":r.volume,"grid_discretization_sem_a3":r.grid_sem,
                "boundary_area_a2":r.boundary_area,"ca_contact_occupancy":contacts,"ca_contact_threshold_angstrom":crate::composition::CA_CONTACT_THRESHOLD_ANGSTROM,
                "fixed_support_ref":format!("{}#region:{}",manifest_path.display(),support.region_id),"source_record_sha256":r.record_sha256,
                "interval_eligibility_reasons":reasons}))?;writeln!(measured)?;
            let continues=current[i].last().is_none_or(|last|last.phase==frame.phase && last.step+m.cadence_steps==frame.step);
            if !continues || !reasons.is_empty() { if !current[i].is_empty() { segments.push((i,std::mem::take(&mut current[i]))); } }
            if let Some(f)=force.filter(|_|reasons.is_empty()) {
                available+=1;current[i].push(MeasuredSample{step:frame.step,time_ps:reference_time_ps,arm_times_ps,phase:frame.phase.clone(),
                    delta:[f[0]-f[1],r.volume[0]-r.volume[1],contacts[0]-contacts[1]],grid_sem:r.grid_sem});
            }
        }
    }
    if frame_count!=m.frame_count { bail!("measured frame count mismatch"); }
    measured.flush()?;
    if let Some(writer)=clock_index.as_mut(){writer.flush()?;}
    for (i,rows) in current.into_iter().enumerate() { if !rows.is_empty(){segments.push((i,rows));} }
    let mut interval_file=BufWriter::new(File::create(output_dir.join("measured_intervals.jsonl"))?);
    let mut uncertainty_file=BufWriter::new(File::create(output_dir.join("scientific_uncertainty_tensors.jsonl"))?);
    let mut inputs=Vec::new();let mut uncertainties=Vec::new();let mut scored_regions=BTreeSet::new();
    for (ordinal,(ri,rows)) in segments.iter().enumerate() {
        let region=&m.regions[*ri];let first=&rows[0];let last=rows.last().unwrap();
        let id=format!("region:{}:step:{}:{}",region.region_id,first.step,last.step);
        let mut reasons=Vec::<String>::new();if rows.len()<SPECTRAL_MIN_SAMPLES { reasons.push("insufficient_contiguous_samples".into()); }
        // ESS precision is conditional on approximately stationary segments;
        // ramps/cooling remain measured but are not scored with stationary SE.
        if !["cold_hold","warm_hold","cold_return"].contains(&first.phase.as_str()) { reasons.push("nonstationary_protocol_phase".into()); }
        let series:[Vec<f64>;3]=std::array::from_fn(|k|rows.iter().map(|r|r.delta[k]).collect());
        let stats:Vec<_>=if rows.len()>=2 {series.iter().map(|s|measured_temporal_summary(s)).collect::<Result<_>>()?}else{Vec::new()};
        for (k,s) in stats.iter().enumerate() {if !s.standard_error.is_finite()||s.standard_error<=0.0 {reasons.push(format!("channel_{k}_sampling_sigma_unavailable_zero_or_nonfinite"));}}
        let grid_means:[f64;2]=std::array::from_fn(|a|rows.iter().map(|r|r.grid_sem[a]).sum::<f64>()/rows.len() as f64);
        let mut interval=json!({"interval_id":id,"region_id":region.region_id,"phase":first.phase,"step_start":first.step,"step_end":last.step,
            "time_start_ps":first.time_ps,"time_end_ps":last.time_ps,"reference_time_arm":reference_time_arm,
            "arm_time_bounds_ps":{"treatment":[first.arm_times_ps[0],last.arm_times_ps[0]],"control":[first.arm_times_ps[1],last.arm_times_ps[1]]},
            "physical_time_journals":m.physical_time_journals,"stream_id":m.stream_id,
            "estimator_time_domain":"recorded sample index; no physical-frequency axis or elapsed-time weighting",
            "n_samples":rows.len(),"cadence_steps":m.cadence_steps,
            "paired_difference_statistics":stats,"mean_grid_discretization_sem_a3":grid_means,"unavailable_reasons":reasons,
            "estimand":"treatment_minus_UV_off_control_on_same_fixed_support","uncertainty_scope":"conditional temporal sampling precision of one seed pair; excludes grid discretization and between-seed/model uncertainty",
            "observer_input_emitted":false});
        if reasons.is_empty() {
            let fs=measured_time_signal(&series[0])?;let cs=measured_time_signal(&series[2])?;
            let recurrence=spectral_recurrence(&fs,&cs);
            let mut tensor_refs=Vec::new();
            for (k,(observable,units)) in [("force","kcal/mol/angstrom"),("pocket_volume","angstrom^3"),("contact_probability","fraction_CA_pairs_distance_lt_8_angstrom")].iter().enumerate() {
                let u=ScientificUncertaintyRef{tensor_id:format!("sut:{}:{ordinal}:{observable}",m.run_id),observable:(*observable).into(),observable_family:(*observable).into(),
                    sigma:stats[k].standard_error,n_samples:rows.len() as u64,measurement_status:"derived_from_measured".into(),source_status:"verified_saved_paired_observations".into(),
                    source_family:"paired_temporal_mean_ESS".into(),source_sha256:manifest_hash.clone(),source_refs:vec![format!("{}#{id}",manifest_path.display())],schema_kind:"scientific_uncertainty_tensor".into()};
                let mut value=serde_json::to_value(&u)?;value["units"]=json!(units);value["interval_id"]=json!(id);value["region_id"]=json!(region.region_id);
                value["estimator"]=json!("existing_PathB_monotone_positive_pair_ESS;sample_variance_over_ESS;standardized_for_unit_invariant_ESS");
                value["sampling_statistics"]=serde_json::to_value(&stats[k])?;value["scope"]=interval["uncertainty_scope"].clone();
                serde_json::to_writer(&mut uncertainty_file,&value)?;writeln!(uncertainty_file)?;
                tensor_refs.push(u.tensor_id.clone());uncertainties.push(u);
            }
            let mut provenance=BTreeMap::from([("interval_id".into(),id.clone()),
                ("estimator_scope".into(),"paired temporal differences; one seed; fixed historical support".into()),("grid_discretization_sigma_scope".into(),"separate recorded metric; not folded into temporal SE".into())]);
            if let Some(reference)=reference_time_arm {
                provenance.insert("reference_time_arm".into(),reference.into());
                provenance.insert("arm_time_bounds_ps".into(),serde_json::to_string(&interval["arm_time_bounds_ps"])?);
                provenance.insert("physical_time_journals".into(),serde_json::to_string(&m.physical_time_journals)?);
                provenance.insert("sample_cadence_ps".into(),"nonuniform_or_arm_specific; use exact per-frame journal index".into());
            }else{provenance.insert("sample_cadence_ps".into(),(m.cadence_steps as f64*m.dt_ps.context("historical dt missing")?).to_string());}
            provenance.insert("estimator_time_domain".into(),"sample-index ACF/ESS; not physical-frequency recurrence; step-matched arm differences".into());
            provenance.insert("score_component_semantics".into(),"chronology_persistence_score=interval_order_validity;topology_persistence_score=input_hash_identity;neither_establishes_dynamical_persistence;missing_control_score_omitted".into());
            provenance.insert("force_observable".into(),"total_recorded_force_RMS_over_fixed_atoms;not_driver_specific".into());
            inputs.push(TshObservableInput{uncertainty_tensor_refs:tensor_refs,stream_id:format!("paired_stream_{}",m.stream_id),seed_id:m.seed_id.clone(),epoch_id:first.phase.clone(),chronology_event_id:id.clone(),
                chronology_ref:format!("{}#{id}",manifest_path.display()),step_start:first.step,step_end:last.step,time_start_ps:first.time_ps,time_end_ps:last.time_ps,
                residue_set_hash:short_hash(&serde_json::to_string(&region.residue_ids)?),spectral_signature_hash:short_hash(&serde_json::to_string(&json!({"interval":id,"force_acf":fs.autocorrelation,"contact_acf":cs.autocorrelation}))?),
                force_delta:stats[0].mean,pocket_volume_delta:stats[1].mean,contact_delta:stats[2].mean,causal_lag:cs.peak_lag.map(|x|x as f64).unwrap_or(f64::NAN),
                lag_corr_peak:cs.peak_autocorrelation.unwrap_or(f64::NAN),spectral_signature_delta:recurrence.delta,spectral_signature_status:recurrence.status,spectral_signature_lags:recurrence.lags,
                force_signal_provenance:provenance.clone(),contact_signal_provenance:provenance,source_refs:vec![format!("{}#{id}",manifest_path.display())],source_sha256:manifest_hash.clone()});
            interval["observer_input_emitted"]=json!(true);scored_regions.insert(region.region_id);
        }
        serde_json::to_writer(&mut interval_file,&interval)?;writeln!(interval_file)?;
    }
    interval_file.flush()?;uncertainty_file.flush()?;
    let receipt=if inputs.is_empty(){None}else{
        let mut config=TshRuntimeConfig::new(format!("{}_offline_pair",m.run_id),&m.target_id);
        config.topology_hash_before=m.topology_sha256.clone();config.topology_hash_after=m.topology_sha256.clone();
        // Both signatures refer to the same immutable observed sources before
        // and after read-only observation, not equality of treatment/control physics.
        config.baseline_physics_signature=manifest_hash.clone();config.enabled_physics_signature=manifest_hash.clone();
        let artifacts=TshRuntimeObserver::offline(config).evaluate(&inputs,&uncertainties,None)?;
        Some(write_tsh_artifacts(output_dir,&artifacts)?)
    };
    for pin in m.source_files.iter().chain([&m.force_treatment,&m.force_control,&m.force_chronology_receipt,&m.frames]) {pin.verify()?;}
    for authority in m.physical_time_journals.values(){authority.source.verify()?;}
    for audit in m.coordinate_audits.values(){audit.verify()?;}
    if sha256_file(manifest_path)?!=manifest_hash {bail!("manifest changed during read-only observation");}
    let summary=json!({"schema_kind":"prism_measured_pair_tsh_execution","status":if inputs.is_empty(){"IMPLEMENTATION_PARTIAL"}else{"IMPLEMENTATION_VERIFIED"},"scope":"native offline measured observation; persistent transition and causal influence not established",
        "runtime_origin":"OFFLINE_MEASURED_TSH_STAGE_II_OBSERVER","manifest_sha256":manifest_hash,"coordinate_frames":frame_count,
        "stream_id":m.stream_id,"reference_time_arm":reference_time_arm,"physical_time_journals":m.physical_time_journals,
        "physical_time_endpoint_index":if reference_time_arm.is_some(){Some(MeasuredSourcePin{path:output_dir.join("physical_time_endpoint_index.jsonl"),sha256:sha256_file(&output_dir.join("physical_time_endpoint_index.jsonl"))?})}else{None},
        "sample_index_recurrence_preserved":true,"TSH_filters_native_fiber_population":false,"regions":m.regions.len(),
        "measured_region_frames":frame_count*m.regions.len(),"force_frames_per_arm":treatment.len(),"both_unclipped_force_region_frames":available,"frame_unavailability_counts":counts,
        "contiguous_segments":segments.len(),"observer_input_intervals":inputs.len(),"scored_region_ids":scored_regions,"uncertainty_tensors":uncertainties.len(),"receipt":receipt,
        "independent_seed_pairs":1,"negative_motif_control_ledger_supplied":false,"paired_UV_off_measurements_used":true,"MD_executed":false,
        "contact_definition":"existing composition CA contact map: distance strictly less than8Angstrom; not chemical interaction classification",
        "spectral_definition":"existing eight-lag normalized ACF distance of paired force/contact time series; not native SO3/TIDE or causal direction",
        "limitations":["historical reference-dependent controller","fixed laboratory ROI and fixed historical lining support","conditional temporal SE excludes grid/model/between-seed uncertainty","hold phase alone does not demonstrate stationarity or convergence","one independent seed pair; no control motif absence claim"]});
    std::fs::write(output_dir.join("measured_pair_tsh_execution.json"),serde_json::to_vec_pretty(&summary)?)?;
    Ok(summary)
}

#[cfg(test)]
mod native_time_adapter_tests {
    use super::*;
    #[test]
    #[ignore="requires an actual complete source-pinned native TSH pair manifest and explicit fresh output"]
    fn actual_native_time_pair_observes_complete_saved_population() {
        let input=PathBuf::from(std::env::var("PRISM_NATIVE_TSH_PAIR_MANIFEST").unwrap());
        let output=PathBuf::from(std::env::var("PRISM_NATIVE_TSH_PAIR_OUTPUT").unwrap());
        let manifest:MeasuredPairManifest=serde_json::from_reader(File::open(&input).unwrap()).unwrap();
        assert_eq!(manifest.physical_time_journals.len(),2);
        let receipt=observe_measured_pair(&input,&output).unwrap();
        assert_eq!(receipt["coordinate_frames"].as_u64(),Some(manifest.frame_count as u64));
        assert_eq!(receipt["measured_region_frames"].as_u64(),Some((manifest.frame_count*manifest.regions.len()) as u64));
        assert_eq!(receipt["stream_id"].as_u64(),Some(manifest.stream_id as u64));
        assert_eq!(receipt["TSH_filters_native_fiber_population"],false);
        println!("{}",serde_json::to_string(&receipt).unwrap());
    }
}
