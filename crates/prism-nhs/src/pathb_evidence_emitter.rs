//! Executable evidence orchestration for the native Path-B chemical-state lane.
//!
//! The emitter writes canonical JSON/JSONL bytes, seals those exact bytes into
//! typed QNANL packets, decodes every packet back to byte equality, attaches
//! canonical PrismCode opcode references, and records process-proven runtime
//! origin. It is intentionally callable from `nhs_rt_full` immediately after a
//! verified segmented Ghost load and native lattice completion.

use crate::candidate_advancement::{
    commit_candidate_append_only, load_append_only_registry, CandidateRecord, ObjectiveDefinition,
    ParetoRegistry, EVENT_CANDIDATE_OBSERVED, EVENT_PARETO_CHAMPION_PROMOTED,
    EVENT_PARETO_CHAMPION_SUPERSEDED, PRISMCODE_OPCODE_CANDIDATE_OBSERVED,
    PRISMCODE_OPCODE_PARETO_CHAMPION_PROMOTED, PRISMCODE_OPCODE_PARETO_CHAMPION_SUPERSEDED,
};
use crate::chemical_state_fiber::{
    canonical_sha256, require_sha256, ChemicalStateFiber, ControlLane, EndpointEligibility,
    FragmentStateObservation, GhostByteLineage, GhostFiberObservation,
    LambdaGhostEndpointJoinMaterialization, LambdaGhostEndpointJoinStatus,
    LambdaGhostJoinGhostNode, OpenPocketEstimand, PhysicalMeasurement, ProtocolPhaseBinding,
    SamplingUncertainty, StructuralResponse, ThermodynamicCondition,
    EVENT_CHEMICAL_STATE_FIBER_MATERIALIZED, PRISMCODE_OPCODE_CHEMICAL_STATE_FIBER_MATERIALIZED,
};
use crate::component_correspondence::{
    establish_component_correspondence, ComponentCorrespondenceLedger,
    ComponentCorrespondencePolicy, ComponentCorrespondenceStatus, NativeComponentDescriptor,
};
use crate::pathb_chemical_state::{
    classify_matched_controls, estimate_native_pathb_from_observations_with_timebase,
    materialize_native_pathb_fibers, GhostLatticeConservation, MatchedControlConfig,
    MatchedControlResult, NativePathBConnectivityAuthority, NativePathBEstimatorConfig,
    NativePathBLatticeObservation, NativePathBMaterialization, NativePathBTimebaseContext,
    EVENT_MATCHED_CONTROL_CLASSIFIED, EVENT_PATHB_CHEMICAL_STATE_ASSIGNED,
    PRISMCODE_OPCODE_MATCHED_CONTROL_CLASSIFIED, PRISMCODE_OPCODE_PATHB_CHEMICAL_STATE_ASSIGNED,
};
use anyhow::{bail, Context, Result};
use prismcode::binary_runtime::opcode_id_for_name;
use prismcode::operand_ledger::RuntimeOrigin;
use prismcode::qnanl_artifact::{
    decode_packet_bytes, encode_bytes, encode_file, QnanlArtifactFamily, QnanlEncodeOptions,
    QnanlPcodeRef,
};
use prismcode::runtime_provenance::RuntimeProvenance;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use statrs::distribution::{ContinuousCDF, StudentsT};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::CString;
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Versioned, content-addressed runtime request consumed by `nhs_rt_full`.
pub const NATIVE_PATHB_RUNTIME_REQUEST_SCHEMA: &str = "prism_native_pathb_runtime_request";
/// Current runtime request schema version.
pub const NATIVE_PATHB_RUNTIME_REQUEST_VERSION: &str = "6.0.0";

/// Predeclared discovery scope. Whole-lattice discovery retains every exact
/// connected component; target mode is an optional hypothesis test and must
/// not masquerade as unbiased discovery.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativePathBSelectionMode {
    WholeLatticeDiscovery,
    TargetResidueHypothesis,
}

/// Hashable body of the explicit production Path-B request. Keeping both
/// estimator configurations inside the signed input prevents silent runtime
/// defaults from changing the scientific estimand.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativePathBRuntimeRequestBody {
    pub pathb_config: NativePathBEstimatorConfig,
    /// Exact signed engine/integrator timebase. It is a required request
    /// authority, never inferred from HMR flags or a nominal runtime default.
    pub timebase_context: NativePathBTimebaseContext,
    pub matched_control_config: MatchedControlConfig,
    pub component_correspondence_policy: ComponentCorrespondencePolicy,
    pub lane_policy: NativePathBLanePolicy,
    pub registry_policy: NativePathBRegistryPolicy,
}

/// Chain-aware residue identity signed by the authoritative topology-to-
/// canonical mapping receipt. Engine residue numbers alone are not accepted
/// as stable cross-run correspondence keys.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignedTrackerContactResidue {
    pub chain_id: String,
    pub canonical_residue_id: String,
    pub engine_residue_id: u32,
    pub chain_mapping_receipt_sha256: String,
}

/// Pre-MD geometric and surveillance definition of a confirmation candidate.
/// Open/closed thresholds are separated by an explicit indeterminate band;
/// rows inside that band are typed NOT_EVALUABLE for binary classification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PathBCandidateTrackerDefinition {
    pub schema_kind: String,
    pub schema_version: String,
    pub tracker_sha256: String,
    pub reference_structure_sha256: String,
    pub reference_coordinate_frame_sha256: String,
    pub reference_frame_index: u64,
    pub reference_centroid_xyz_a: [f64; 3],
    /// Complete signed moving-frame authority. A digest-only reference is not
    /// sufficient because runtime must unwrap/align the declared anchors.
    pub body_fixed_tracker_authority: crate::fixed_cadence_coordinate::BodyFixedTrackerAuthority,
    pub signed_contact_residues: Vec<SignedTrackerContactResidue>,
    pub cavity_estimator_config: crate::coordinate_cavity::CoordinateCavityConfig,
    pub observed_closed_max_volume_a3: f64,
    pub observed_open_min_volume_a3: f64,
    pub local_shell_radius_a: f64,
    pub cadence_steps: u64,
    pub start_step_by_stream: BTreeMap<u16, u64>,
    pub end_step_exclusive_by_stream: BTreeMap<u16, u64>,
    pub stationary_phase_ordinal: u16,
    pub matched_physical_state_sha256: String,
}

#[derive(Serialize)]
struct PathBCandidateTrackerBody<'a> {
    reference_structure_sha256: &'a str,
    reference_coordinate_frame_sha256: &'a str,
    reference_frame_index: u64,
    reference_centroid_xyz_a: [f64; 3],
    body_fixed_tracker_authority: &'a crate::fixed_cadence_coordinate::BodyFixedTrackerAuthority,
    signed_contact_residues: &'a [SignedTrackerContactResidue],
    cavity_estimator_config: &'a crate::coordinate_cavity::CoordinateCavityConfig,
    observed_closed_max_volume_a3: f64,
    observed_open_min_volume_a3: f64,
    local_shell_radius_a: f64,
    cadence_steps: u64,
    start_step_by_stream: &'a BTreeMap<u16, u64>,
    end_step_exclusive_by_stream: &'a BTreeMap<u16, u64>,
    stationary_phase_ordinal: u16,
    matched_physical_state_sha256: &'a str,
}

impl PathBCandidateTrackerDefinition {
    fn body(&self) -> PathBCandidateTrackerBody<'_> {
        PathBCandidateTrackerBody {
            reference_structure_sha256: &self.reference_structure_sha256,
            reference_coordinate_frame_sha256: &self.reference_coordinate_frame_sha256,
            reference_frame_index: self.reference_frame_index,
            reference_centroid_xyz_a: self.reference_centroid_xyz_a,
            body_fixed_tracker_authority: &self.body_fixed_tracker_authority,
            signed_contact_residues: &self.signed_contact_residues,
            cavity_estimator_config: &self.cavity_estimator_config,
            observed_closed_max_volume_a3: self.observed_closed_max_volume_a3,
            observed_open_min_volume_a3: self.observed_open_min_volume_a3,
            local_shell_radius_a: self.local_shell_radius_a,
            cadence_steps: self.cadence_steps,
            start_step_by_stream: &self.start_step_by_stream,
            end_step_exclusive_by_stream: &self.end_step_exclusive_by_stream,
            stationary_phase_ordinal: self.stationary_phase_ordinal,
            matched_physical_state_sha256: &self.matched_physical_state_sha256,
        }
    }

    pub fn verify(&self) -> Result<()> {
        if self.schema_kind != "prism_pathb_candidate_tracker"
            || self.schema_version != "2.0.0"
            || self.signed_contact_residues.is_empty()
            || self
                .signed_contact_residues
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || self
                .reference_centroid_xyz_a
                .iter()
                .any(|value| !value.is_finite())
            || !self.observed_closed_max_volume_a3.is_finite()
            || !self.observed_open_min_volume_a3.is_finite()
            || self.observed_closed_max_volume_a3 < 0.0
            || self.observed_open_min_volume_a3 <= self.observed_closed_max_volume_a3
            || !self.local_shell_radius_a.is_finite()
            || self.local_shell_radius_a <= 0.0
            || self.cadence_steps == 0
            || self.start_step_by_stream.is_empty()
            || self.start_step_by_stream.keys().collect::<Vec<_>>()
                != self.end_step_exclusive_by_stream.keys().collect::<Vec<_>>()
        {
            bail!("invalid predeclared Path-B candidate tracker definition");
        }
        for (digest, field) in [
            (&self.tracker_sha256, "candidate tracker_sha256"),
            (
                &self.reference_structure_sha256,
                "tracker reference structure",
            ),
            (
                &self.reference_coordinate_frame_sha256,
                "tracker reference frame",
            ),
            (
                &self.matched_physical_state_sha256,
                "tracker matched physical state",
            ),
        ] {
            require_sha256(digest, field)?;
        }
        self.cavity_estimator_config.validate()?;
        self.body_fixed_tracker_authority.verify()?;
        if self.body_fixed_tracker_authority.reference_structure_sha256
            != self.reference_structure_sha256
            || self
                .body_fixed_tracker_authority
                .reference_coordinate_frame_sha256
                != self.reference_coordinate_frame_sha256
            || self
                .body_fixed_tracker_authority
                .reference_candidate_centroid_angstrom
                != self.reference_centroid_xyz_a
            || self.signed_contact_residues.iter().any(|residue| {
                residue.chain_mapping_receipt_sha256
                    != self
                        .body_fixed_tracker_authority
                        .chain_mapping_receipt_sha256
            })
        {
            bail!("candidate tracker differs from its signed body-fixed authority");
        }
        for residue in &self.signed_contact_residues {
            if residue.chain_id.trim().is_empty() || residue.canonical_residue_id.trim().is_empty()
            {
                bail!("candidate tracker has an empty chain-aware residue identity");
            }
            require_sha256(
                &residue.chain_mapping_receipt_sha256,
                "tracker chain mapping receipt",
            )?;
        }
        for (stream, start) in &self.start_step_by_stream {
            let end = self.end_step_exclusive_by_stream[stream];
            if end <= *start || (end - start) % self.cadence_steps != 0 {
                bail!("candidate tracker surveillance window is empty or off cadence");
            }
        }
        if canonical_sha256(&self.body())? != self.tracker_sha256 {
            bail!("candidate tracker definition content hash mismatch");
        }
        Ok(())
    }

    /// Bind a newly constructed request-side tracker body to its canonical
    /// digest. The caller supplies all scientific fields and leaves
    /// `tracker_sha256` empty; this function never invents defaults.
    pub fn materialize(mut self) -> Result<Self> {
        if !self.tracker_sha256.is_empty() {
            bail!("candidate tracker materialization requires an empty digest field");
        }
        self.tracker_sha256 = canonical_sha256(&self.body())?;
        self.verify()?;
        Ok(self)
    }
}

/// Stable pre-MD selection policy. It contains no record hash, frame, scalar,
/// state assignment, or other post-run scientific observation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativePathBLanePolicy {
    pub site_key: String,
    pub selection_mode: NativePathBSelectionMode,
    /// Hash of the predeclared hypothesis/control fragment panel shared by all
    /// matched lanes, including the no-probe lane's explicit absence arm.
    pub fragment_panel_sha256: String,
    pub target_residue_ids: Vec<u32>,
    /// Minimum number of distinct target residues whose exact RichSpike
    /// cluster membership must overlap a Ghost record.
    pub minimum_target_residue_overlap_count: usize,
    /// Minimum fraction of the declared target-residue set that must overlap
    /// the exact cluster membership. This is a selection rule, not a claim
    /// that the membership rows are cavity-lining residues.
    pub minimum_target_residue_overlap_fraction: f64,
    pub required_stream_ids: Vec<u16>,
    /// Predeclared fixed observation cadence in integration steps. Runtime
    /// may not derive this from the lag grid or from observed events.
    pub fixed_cadence_steps: u64,
    pub stream_lanes: BTreeMap<u16, ControlLane>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub fragment_slot_ids: BTreeMap<u16, u32>,
    pub coordinate_cavity_config: crate::coordinate_cavity::CoordinateCavityConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate_tracker: Option<PathBCandidateTrackerDefinition>,
}

/// Stable cross-run advancement policy carried by each lane request but never
/// executed by a single-run emitter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativePathBRegistryPolicy {
    pub durable_registry_path: PathBuf,
    pub candidate_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revises_candidate_sha256: Option<String>,
    pub objective_definitions: Vec<ObjectiveDefinition>,
}

/// Exact request admitted by the native engine after Ghost persistence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativePathBRuntimeRequest {
    pub schema_kind: String,
    pub schema_version: String,
    pub request_sha256: String,
    pub body: NativePathBRuntimeRequestBody,
}

impl NativePathBRuntimeRequest {
    /// Verify schema and the canonical content address of the complete request
    /// body. Deeper scientific validation occurs against the actual Ghost
    /// population inside the per-run lane materializer or cross-run
    /// aggregation stage.
    pub fn verify(&self) -> Result<()> {
        if self.schema_kind != NATIVE_PATHB_RUNTIME_REQUEST_SCHEMA
            || self.schema_version != NATIVE_PATHB_RUNTIME_REQUEST_VERSION
        {
            bail!("unsupported native Path-B runtime request schema");
        }
        require_sha256(&self.request_sha256, "native Path-B request_sha256")?;
        if canonical_sha256(&self.body)? != self.request_sha256 {
            bail!("native Path-B runtime request content hash mismatch");
        }
        self.body.timebase_context.verify()?;
        self.body.matched_control_config.validate()?;
        self.body.component_correspondence_policy.verify()?;
        let lane = &self.body.lane_policy;
        let selection_rule_valid = match lane.selection_mode {
            NativePathBSelectionMode::WholeLatticeDiscovery => {
                lane.target_residue_ids.is_empty()
                    && lane.minimum_target_residue_overlap_count == 0
                    && lane.minimum_target_residue_overlap_fraction.to_bits() == 0.0f64.to_bits()
                    && lane.candidate_tracker.is_none()
            }
            NativePathBSelectionMode::TargetResidueHypothesis => {
                !lane.target_residue_ids.is_empty()
                    && lane.minimum_target_residue_overlap_count > 0
                    && lane.minimum_target_residue_overlap_count <= lane.target_residue_ids.len()
                    && lane.minimum_target_residue_overlap_fraction.is_finite()
                    && (0.0..=1.0).contains(&lane.minimum_target_residue_overlap_fraction)
                    && lane.minimum_target_residue_overlap_fraction > 0.0
                    && !lane
                        .target_residue_ids
                        .windows(2)
                        .any(|pair| pair[0] >= pair[1])
                    && lane.candidate_tracker.is_some()
            }
        };
        if lane.site_key.trim().is_empty()
            || !selection_rule_valid
            || lane.required_stream_ids.is_empty()
            || lane.fixed_cadence_steps == 0
            || self
                .body
                .pathb_config
                .candidate_lag_steps
                .iter()
                .any(|lag| *lag % lane.fixed_cadence_steps != 0)
            || lane
                .required_stream_ids
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            bail!("native Path-B lane policy is empty, duplicate, or unsorted");
        }
        require_sha256(&lane.fragment_panel_sha256, "fragment_panel_sha256")?;
        if lane.stream_lanes.keys().copied().collect::<Vec<_>>() != lane.required_stream_ids {
            bail!("native Path-B stream lane map does not equal required streams");
        }
        if lane
            .stream_lanes
            .values()
            .copied()
            .collect::<BTreeSet<_>>()
            .len()
            != 1
        {
            bail!("one runtime request must represent exactly one physical control lane");
        }
        for (stream_id, control_lane) in &lane.stream_lanes {
            match (control_lane, lane.fragment_slot_ids.get(stream_id)) {
                (ControlLane::NoProbe, None)
                | (ControlLane::HypothesisFragment, Some(_))
                | (ControlLane::OrthogonalChemicalControl, Some(_)) => {}
                _ => bail!("native Path-B lane and fragment-slot policy disagree"),
            }
        }
        if lane
            .fragment_slot_ids
            .keys()
            .any(|stream| !lane.stream_lanes.contains_key(stream))
        {
            bail!("fragment-slot policy names an undeclared stream");
        }
        lane.coordinate_cavity_config.validate()?;
        if let Some(tracker) = &lane.candidate_tracker {
            tracker.verify()?;
            if tracker.cavity_estimator_config != lane.coordinate_cavity_config
                || tracker.cadence_steps != lane.fixed_cadence_steps
                || tracker
                    .start_step_by_stream
                    .keys()
                    .copied()
                    .collect::<Vec<_>>()
                    != lane.required_stream_ids
                || tracker
                    .signed_contact_residues
                    .iter()
                    .map(|residue| residue.engine_residue_id)
                    .collect::<BTreeSet<_>>()
                    != lane
                        .target_residue_ids
                        .iter()
                        .copied()
                        .collect::<BTreeSet<_>>()
            {
                bail!("candidate tracker differs from the signed lane hypothesis policy");
            }
        }
        let registry = &self.body.registry_policy;
        if registry.candidate_id.trim().is_empty()
            || registry.objective_definitions.is_empty()
            || !registry.durable_registry_path.is_absolute()
        {
            bail!("native Path-B registry policy is incomplete");
        }
        if let Some(revises) = &registry.revises_candidate_sha256 {
            require_sha256(revises, "revises_candidate_sha256")?;
        }
        let mut objectives = BTreeSet::new();
        for objective in &registry.objective_definitions {
            if objective.objective_id.trim().is_empty()
                || !objectives.insert(objective.objective_id.as_str())
            {
                bail!("native Path-B registry objectives are empty or duplicate");
            }
        }
        Ok(())
    }
}

/// Read an operator-supplied runtime request without following a symlink and
/// without accepting relative/parent-traversal paths.
pub fn load_native_pathb_runtime_request(path: &Path) -> Result<NativePathBRuntimeRequest> {
    use std::path::Component;
    if !path.is_absolute()
        || path
            .components()
            .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
    {
        bail!("native Path-B request path must be absolute and lexically normalized");
    }
    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("inspect native Path-B request {}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        bail!("native Path-B request must be a non-symlink regular file");
    }
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(path)
        .with_context(|| format!("open native Path-B request {}", path.display()))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    if bytes.is_empty() {
        bail!("native Path-B request is empty");
    }
    let request: NativePathBRuntimeRequest =
        serde_json::from_slice(&bytes).context("decode native Path-B runtime request")?;
    request.verify()?;
    Ok(request)
}

/// One emitted raw/QNANL/receipt triplet.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TypedEvidenceArtifact {
    /// QNANL family string.
    pub artifact_family: String,
    /// Canonical raw JSON/JSONL path.
    pub raw_path: String,
    /// SHA-256 of exact raw bytes.
    pub raw_sha256: String,
    /// QNANL packet path.
    pub packet_path: String,
    /// SHA-256 of exact packet bytes.
    pub packet_sha256: String,
    /// QNANL receipt path.
    pub receipt_path: String,
    /// SHA-256 of the rewritten receipt carrying opcode and provenance blocks.
    pub receipt_sha256: String,
    /// Typed PrismCode semantic rows referenced by the receipt.
    pub prismcode_events: Vec<String>,
    /// Runtime-origin label captured before emission.
    pub runtime_origin: String,
    /// Byte-exact QNANL decode verdict.
    pub exact_byte_roundtrip_pass: bool,
}

/// Complete emitted evidence set for one native Path-B operationalization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PathBEvidenceEmission {
    /// Content hash of the admitted cross-run aggregation manifest.
    pub aggregation_manifest_sha256: String,
    /// Verifier-approved committed confirmation handoff carried by the v5
    /// manifest and re-hashed before cross-run inference.
    pub source_confirmation_handoff: PathBConfirmationHandoffIdentity,
    /// Native model and conservation receipt.
    pub pathb: NativePathBMaterialization,
    /// Complete recurrent/ambiguous/unmatched cross-run component identity
    /// ledger. Every lower-ranked component remains preserved.
    pub component_correspondence: ComponentCorrespondenceLedger,
    /// Coordinate-defined, uncertainty-separated open-basin binding.
    pub open_basin: NativeOpenBasinBinding,
    /// Complete fiber population.
    pub fiber_count: usize,
    /// Exact accounting for every fragment observation that could or could not
    /// be joined to a native Path-B assignment. Unassigned rows are preserved
    /// verbatim and never silently filtered.
    pub fragment_assignment: FragmentPathBAssignmentLedger,
    /// Scientific claim/promotion disposition derived from the complete
    /// recurrent/ambiguous/unmatched target component population.
    pub claim_evaluation: CandidateClaimEvaluation,
    /// Matched-control result.
    pub matched_control: MatchedControlResult,
    /// Candidate record appended to the registry. This is absent when the
    /// component identity is scientifically NOT_EVALUABLE; absence never
    /// removes its raw population or correspondence evidence.
    pub candidate: Option<CandidateRecord>,
    /// Verified reconstructed registry after an eligible append.
    pub registry: Option<ParetoRegistry>,
    /// Explicit durable append-only registry path.
    pub durable_registry_path: String,
    /// Hash of the verified durable registry after this exact append.
    pub durable_registry_sha256: Option<String>,
    /// Immutable per-run snapshot bound into typed evidence.
    pub registry_snapshot_path: Option<String>,
    /// Hash of the immutable per-run registry snapshot.
    pub registry_snapshot_sha256: Option<String>,
    /// Complete exact-byte evidence artifacts, including population
    /// preservation and an optional registry snapshot after eligible append.
    pub artifacts: Vec<TypedEvidenceArtifact>,
    /// Content hash of the artifact receipt vector.
    pub emission_sha256: String,
}

/// Scientific disposition for target component identity. This status limits
/// claim/promotion only; it never suppresses the underlying population.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CandidateClaimEvaluationStatus {
    Eligible,
    NotEvaluable,
}

/// Content-addressed component-identity claim decision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateClaimEvaluation {
    pub schema_kind: String,
    pub schema_version: String,
    pub evaluation_sha256: String,
    pub status: CandidateClaimEvaluationStatus,
    pub emitted_component_sha256: Vec<String>,
    pub recurrent_correspondence_sha256: Vec<String>,
    pub ambiguous_descriptor_sha256: Vec<String>,
    pub unmatched_descriptor_sha256: Vec<String>,
    pub reason_codes: Vec<String>,
    pub candidate_append_allowed: bool,
}

impl CandidateClaimEvaluation {
    fn verify(&self) -> Result<()> {
        if self.schema_kind != "prism_pathb_candidate_claim_evaluation"
            || self.schema_version != "1.0.0"
        {
            bail!("unsupported candidate claim evaluation schema");
        }
        for values in [
            &self.emitted_component_sha256,
            &self.recurrent_correspondence_sha256,
            &self.ambiguous_descriptor_sha256,
            &self.unmatched_descriptor_sha256,
        ] {
            if values.windows(2).any(|pair| pair[0] >= pair[1])
                || values
                    .iter()
                    .any(|value| require_sha256(value, "claim evaluation digest").is_err())
            {
                bail!("candidate claim evaluation digests are invalid, duplicate, or unsorted");
            }
        }
        if self.reason_codes.windows(2).any(|pair| pair[0] >= pair[1])
            || self
                .reason_codes
                .iter()
                .any(|reason| reason.trim().is_empty())
        {
            bail!("candidate claim evaluation reason codes are invalid");
        }
        let eligible = self.status == CandidateClaimEvaluationStatus::Eligible;
        if eligible != self.candidate_append_allowed
            || eligible
                != (self.recurrent_correspondence_sha256.len() == 1
                    && self.ambiguous_descriptor_sha256.is_empty()
                    && self.unmatched_descriptor_sha256.is_empty()
                    && !self.emitted_component_sha256.is_empty()
                    && self.reason_codes.is_empty())
        {
            bail!("candidate claim eligibility and component evidence disagree");
        }
        if canonical_sha256(&(
            self.status,
            &self.emitted_component_sha256,
            &self.recurrent_correspondence_sha256,
            &self.ambiguous_descriptor_sha256,
            &self.unmatched_descriptor_sha256,
            &self.reason_codes,
            self.candidate_append_allowed,
        ))? != self.evaluation_sha256
        {
            bail!("candidate claim evaluation content hash mismatch");
        }
        Ok(())
    }
}

/// Exact accounting of fragment observations admitted to or excluded from the
/// fiber join. The excluded rows remain full typed observations with byte
/// lineage and chemical state intact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FragmentPathBAssignmentLedger {
    pub schema_kind: String,
    pub schema_version: String,
    pub ledger_sha256: String,
    pub aggregation_manifest_sha256: String,
    pub pathb_model_sha256: String,
    pub source_population_sha256: String,
    pub source_observation_count: usize,
    pub assigned_ghost_lineage_sha256: Vec<String>,
    pub unassigned_observations: Vec<FragmentStateObservation>,
}

impl FragmentPathBAssignmentLedger {
    fn verify(&self) -> Result<()> {
        if self.schema_kind != "prism_fragment_pathb_assignment_ledger"
            || self.schema_version != "1.0.0"
        {
            bail!("unsupported fragment Path-B assignment ledger schema");
        }
        for (digest, label) in [
            (&self.ledger_sha256, "fragment assignment ledger"),
            (&self.aggregation_manifest_sha256, "aggregation manifest"),
            (&self.pathb_model_sha256, "Path-B model"),
            (&self.source_population_sha256, "fragment source population"),
        ] {
            require_sha256(digest, label)?;
        }
        if self
            .assigned_ghost_lineage_sha256
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
            || self
                .assigned_ghost_lineage_sha256
                .iter()
                .any(|digest| require_sha256(digest, "assigned fragment lineage").is_err())
        {
            bail!("assigned fragment lineage set is invalid, duplicate, or unsorted");
        }
        let mut previous = None;
        for observation in &self.unassigned_observations {
            require_sha256(
                &observation.ghost_lineage_sha256,
                "unassigned fragment lineage",
            )?;
            if previous
                .as_ref()
                .is_some_and(|prior: &String| prior >= &observation.ghost_lineage_sha256)
            {
                bail!("unassigned fragment observations are duplicate or unsorted");
            }
            previous = Some(observation.ghost_lineage_sha256.clone());
        }
        if self.source_observation_count
            != self.assigned_ghost_lineage_sha256.len() + self.unassigned_observations.len()
            || canonical_sha256(&(
                &self.aggregation_manifest_sha256,
                &self.pathb_model_sha256,
                &self.source_population_sha256,
                self.source_observation_count,
                &self.assigned_ghost_lineage_sha256,
                &self.unassigned_observations,
            ))? != self.ledger_sha256
        {
            bail!("fragment Path-B assignment population/hash conservation failed");
        }
        Ok(())
    }
}

/// Typed fail-closed receipt emitted when the current run cannot construct
/// exact per-node lane evidence. This is an evidence object, not a placeholder
/// state assignment: all materialized populations remain exactly zero.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NativePathBAvailabilityReceipt {
    pub schema_kind: String,
    pub schema_version: String,
    pub status: String,
    pub receipt_sha256: String,
    pub runtime_request_sha256: String,
    pub lattice_manifest_sha256: String,
    pub stream_lanes: BTreeMap<u16, ControlLane>,
    pub site_key: String,
    pub required_stream_ids: Vec<u16>,
    pub observed_stream_topology_sha256: BTreeMap<u16, String>,
    pub ghost_lineage_count: u64,
    pub exact_coordinate_frame_count: u64,
    pub exact_lambda_state_count: u64,
    pub materialized_pathb_state_count: u64,
    pub materialized_fiber_count: u64,
    pub candidate_append_count: u64,
    pub reason_codes: Vec<String>,
    pub source_contract: String,
    pub typed_artifact: TypedEvidenceArtifact,
}

/// Pre-MD admission receipt for a runtime mode that cannot satisfy the exact
/// one-stationary-physical-lane Path-B contract. Emitting this before engine
/// construction prevents an expensive run from ending in a known late
/// placeholder state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NativePathBUnsupportedRuntimeReceipt {
    pub schema_kind: String,
    pub schema_version: String,
    pub status: String,
    pub receipt_sha256: String,
    pub runtime_request_sha256: String,
    pub execution_mode: String,
    pub reason_code: String,
    pub materialized_pathb_state_count: u64,
    pub materialized_fiber_count: u64,
    pub candidate_append_count: u64,
    pub typed_artifact: TypedEvidenceArtifact,
}

#[derive(Serialize)]
struct NativePathBUnsupportedRuntimeBody<'a> {
    runtime_request_sha256: &'a str,
    execution_mode: &'a str,
    reason_code: &'a str,
    materialized_pathb_state_count: u64,
    materialized_fiber_count: u64,
    candidate_append_count: u64,
}

/// Emit a typed pre-MD unsupported-mode receipt. This is the only admissible
/// strict-coupled behavior until that producer owns the same per-stream
/// coordinate and lambda journals as the independent physical-lane runtime.
pub fn emit_native_pathb_unsupported_runtime_mode(
    output_dir: &Path,
    request: &NativePathBRuntimeRequest,
    execution_mode: &str,
    reason_code: &str,
    runtime_provenance: &RuntimeProvenance,
) -> Result<NativePathBUnsupportedRuntimeReceipt> {
    request.verify()?;
    if execution_mode.trim().is_empty() || reason_code.trim().is_empty() {
        bail!("unsupported Path-B runtime receipt lacks mode or reason");
    }
    std::fs::create_dir_all(output_dir)?;
    let body = NativePathBUnsupportedRuntimeBody {
        runtime_request_sha256: &request.request_sha256,
        execution_mode,
        reason_code,
        materialized_pathb_state_count: 0,
        materialized_fiber_count: 0,
        candidate_append_count: 0,
    };
    let receipt_sha256 = canonical_sha256(&body)?;
    let raw = serde_json::json!({
        "schema_kind": "prism_native_pathb_unsupported_runtime_receipt",
        "schema_version": "1.0.0",
        "status": "EVIDENCE_NOT_FOUND",
        "receipt_sha256": receipt_sha256,
        "runtime_request_sha256": request.request_sha256,
        "execution_mode": execution_mode,
        "reason_code": reason_code,
        "materialized_pathb_state_count": 0,
        "materialized_fiber_count": 0,
        "candidate_append_count": 0,
    });
    let raw_path = output_dir.join("native_pathb_unsupported_runtime_receipt.json");
    let packet_path = PathBuf::from(format!("{}.prismqnanl", raw_path.display()));
    let qnanl_receipt_path = PathBuf::from(format!("{}.receipt.json", packet_path.display()));
    if raw_path.exists() || packet_path.exists() || qnanl_receipt_path.exists() {
        bail!("refusing to overwrite prior unsupported Path-B runtime receipt");
    }
    write_create_new(&raw_path, &canonical_json_line(&raw)?)?;
    let typed_artifact = emit_qnanl_exact(
        &raw_path,
        QnanlArtifactFamily::PathBChemicalState,
        &[],
        runtime_provenance,
        "nhs_rt_full:native_pathb_pre_md_unsupported_mode",
    )?;
    Ok(NativePathBUnsupportedRuntimeReceipt {
        schema_kind: "prism_native_pathb_unsupported_runtime_receipt".to_string(),
        schema_version: "1.0.0".to_string(),
        status: "EVIDENCE_NOT_FOUND".to_string(),
        receipt_sha256,
        runtime_request_sha256: request.request_sha256.clone(),
        execution_mode: execution_mode.to_string(),
        reason_code: reason_code.to_string(),
        materialized_pathb_state_count: 0,
        materialized_fiber_count: 0,
        candidate_append_count: 0,
        typed_artifact,
    })
}

/// Immutable pre-Path-B evidence for one completed physical lane run. It
/// contains exact native lattice, coordinate and lambda/no-probe observations
/// but intentionally contains no metastable-state assignment, fiber,
/// matched-control classification or candidate advancement.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativePathBLaneEvidenceBody {
    pub runtime_request: NativePathBRuntimeRequest,
    pub site_selection: GhostSiteSelectionReceipt,
    pub source_authority: NativePathBLaneSourceAuthority,
    pub lattice_observations: Vec<NativePathBLatticeObservation>,
    pub exact_connectivity: NativePathBConnectivityAuthority,
    /// Complete chain/spatial descriptor population, exactly one per native
    /// component. This preserves unmatched components for later recurrence.
    pub component_descriptors: Vec<NativeComponentDescriptor>,
    /// Exact coordinate-cavity atom, grid, packed-bit and boundary-face bytes
    /// used to construct every component descriptor. This authority is part of
    /// the fail-atomic lane member and may not be discarded after extracting
    /// descriptor vectors.
    pub physical_lining_descriptor_population:
        crate::component_correspondence::PhysicalLiningDescriptorPopulation,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confirmation_surveillance: Option<PathBConfirmationSurveillanceLedger>,
    pub ghost_measurements: Vec<GhostFiberMeasurementObservation>,
    /// Complete immutable post-persistence Ghost/lambda join, including every
    /// retained non-evaluable lambda key and Ghost node.
    pub lambda_ghost_endpoint_join: LambdaGhostEndpointJoinMaterialization,
    /// Exact-byte QNANL emission of the join materialization.
    pub lambda_ghost_endpoint_join_typed_artifact: TypedEvidenceArtifact,
    pub fragment_observations: Vec<FragmentStateObservation>,
    pub materialized_pathb_state_count: u64,
    pub materialized_fiber_count: u64,
    pub matched_control_classification_count: u64,
    pub candidate_append_count: u64,
}

/// Hash authority for the exact coordinate and λ/no-probe populations joined
/// into one lane unit. These are loader-derived population digests, not caller
/// labels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativePathBLaneSourceAuthority {
    pub coordinate_population_sha256: String,
    pub lambda_population_sha256: String,
    pub lambda_ghost_endpoint_join_sha256: String,
    pub fragment_panel_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub no_probe_absence_receipt_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chemical_potential_calibration_population_sha256: Option<String>,
}

impl NativePathBLaneSourceAuthority {
    fn verify_for(&self, lane: ControlLane, expected_panel_sha256: &str) -> Result<()> {
        for (digest, field) in [
            (
                &self.coordinate_population_sha256,
                "coordinate_population_sha256",
            ),
            (&self.lambda_population_sha256, "lambda_population_sha256"),
            (
                &self.lambda_ghost_endpoint_join_sha256,
                "lambda_ghost_endpoint_join_sha256",
            ),
            (&self.fragment_panel_sha256, "fragment_panel_sha256"),
        ] {
            require_sha256(digest, field)?;
        }
        if self.fragment_panel_sha256 != expected_panel_sha256 {
            bail!("lane source authority fragment panel differs from runtime request");
        }
        match lane {
            ControlLane::NoProbe => {
                let absence = self
                    .no_probe_absence_receipt_sha256
                    .as_ref()
                    .context("no-probe lane requires a closed-interval absence receipt")?;
                require_sha256(absence, "no_probe_absence_receipt_sha256")?;
                if self
                    .chemical_potential_calibration_population_sha256
                    .is_some()
                {
                    bail!("no-probe lane cannot carry chemical-potential calibration");
                }
            }
            ControlLane::HypothesisFragment | ControlLane::OrthogonalChemicalControl => {
                let calibration = self
                    .chemical_potential_calibration_population_sha256
                    .as_ref()
                    .context("probe lane requires calibration population identity")?;
                require_sha256(
                    calibration,
                    "chemical_potential_calibration_population_sha256",
                )?;
                if self.no_probe_absence_receipt_sha256.is_some() {
                    bail!("probe lane cannot carry a no-probe absence receipt");
                }
            }
        }
        Ok(())
    }
}

/// Content-addressed lane evidence document.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativePathBLaneEvidence {
    pub schema_kind: String,
    pub schema_version: String,
    pub lane_evidence_sha256: String,
    pub body: NativePathBLaneEvidenceBody,
}

impl NativePathBLaneEvidence {
    /// Materialize deterministic per-run evidence after strict population and
    /// physical-lineage validation.
    pub fn materialize(mut body: NativePathBLaneEvidenceBody) -> Result<Self> {
        body.lattice_observations.sort_by_key(|row| {
            (
                row.stream_id,
                row.step_idx,
                row.site_id,
                row.lattice_node_index,
            )
        });
        body.ghost_measurements.sort_by_key(|row| row.ghost.key);
        body.fragment_observations
            .sort_by(|left, right| left.ghost_lineage_sha256.cmp(&right.ghost_lineage_sha256));
        body.component_descriptors
            .sort_by(|left, right| left.descriptor_sha256.cmp(&right.descriptor_sha256));
        let lane_evidence_sha256 = canonical_sha256(&body)?;
        let evidence = Self {
            schema_kind: "prism_native_pathb_lane_evidence".to_string(),
            schema_version: "5.0.0".to_string(),
            lane_evidence_sha256,
            body,
        };
        evidence.verify()?;
        Ok(evidence)
    }

    /// Verify that every selected record has exactly one native-node,
    /// coordinate and fragment/no-probe row and that a single run performed no
    /// scientific classification or registry mutation.
    pub fn verify(&self) -> Result<()> {
        if self.schema_kind != "prism_native_pathb_lane_evidence"
            || self.schema_version != "5.0.0"
            || self.body.materialized_pathb_state_count != 0
            || self.body.materialized_fiber_count != 0
            || self.body.matched_control_classification_count != 0
            || self.body.candidate_append_count != 0
        {
            bail!("per-run Path-B lane evidence crossed the cross-run decision boundary");
        }
        self.body.runtime_request.verify()?;
        self.body.site_selection.verify()?;
        if self.body.site_selection.site_key != self.body.runtime_request.body.lane_policy.site_key
            || self.body.site_selection.selection_mode
                != self.body.runtime_request.body.lane_policy.selection_mode
            || self.body.site_selection.target_residue_ids
                != self
                    .body
                    .runtime_request
                    .body
                    .lane_policy
                    .target_residue_ids
            || self
                .body
                .site_selection
                .minimum_target_residue_overlap_count
                != self
                    .body
                    .runtime_request
                    .body
                    .lane_policy
                    .minimum_target_residue_overlap_count
            || self
                .body
                .site_selection
                .minimum_target_residue_overlap_fraction
                .to_bits()
                != self
                    .body
                    .runtime_request
                    .body
                    .lane_policy
                    .minimum_target_residue_overlap_fraction
                    .to_bits()
        {
            bail!("lane evidence selection receipt differs from admitted runtime request");
        }
        let count = self.body.site_selection.selected_records as usize;
        if self.body.lattice_observations.len() != count
            || self.body.ghost_measurements.len() != count
            || self.body.fragment_observations.len() != count
        {
            bail!("per-run lane evidence populations are not exactly conserved");
        }
        self.body.lambda_ghost_endpoint_join.verify()?;
        if self.body.lambda_ghost_endpoint_join.ghost_input_count != count as u64
            || self.body.source_authority.lambda_ghost_endpoint_join_sha256
                != self.body.lambda_ghost_endpoint_join.materialization_sha256
            || self.body.source_authority.lambda_population_sha256
                != self
                    .body
                    .lambda_ghost_endpoint_join
                    .lambda_population_receipt_sha256
        {
            bail!("lane source authority differs from immutable lambda/Ghost join population");
        }
        let join_artifact = &self.body.lambda_ghost_endpoint_join_typed_artifact;
        let expected_join_raw = canonical_json_line(&self.body.lambda_ghost_endpoint_join)?;
        if join_artifact.artifact_family != QnanlArtifactFamily::PathBChemicalState.as_str()
            || !join_artifact.exact_byte_roundtrip_pass
            || join_artifact.raw_sha256 != format!("{:x}", Sha256::digest(&expected_join_raw))
        {
            bail!("lambda/Ghost join typed artifact is not its exact materialization bytes");
        }
        for (digest, field) in [
            (&join_artifact.raw_sha256, "lambda/Ghost join raw"),
            (&join_artifact.packet_sha256, "lambda/Ghost join packet"),
            (&join_artifact.receipt_sha256, "lambda/Ghost join receipt"),
        ] {
            require_sha256(digest, field)?;
        }
        let ghosts = self
            .body
            .ghost_measurements
            .iter()
            .map(|row| (row.ghost.key, row))
            .collect::<BTreeMap<_, _>>();
        let ghosts_by_lineage = self
            .body
            .ghost_measurements
            .iter()
            .map(|row| Ok((row.ghost.lineage_sha256()?, row)))
            .collect::<Result<BTreeMap<_, _>>>()?;
        let fragments = self
            .body
            .fragment_observations
            .iter()
            .map(|row| (row.ghost_lineage_sha256.as_str(), row))
            .collect::<BTreeMap<_, _>>();
        if ghosts.len() != count || fragments.len() != count {
            bail!("per-run lane evidence contains duplicate exact Ghost keys");
        }
        let admitted_lineages = self
            .body
            .ghost_measurements
            .iter()
            .map(|row| row.ghost.clone())
            .collect::<Vec<_>>();
        self.body
            .exact_connectivity
            .verify_for(&admitted_lineages)?;
        let exact_component_sha256 = self
            .body
            .exact_connectivity
            .components
            .iter()
            .map(|component| component.component_sha256.clone())
            .collect::<BTreeSet<_>>();
        let descriptor_components = self
            .body
            .component_descriptors
            .iter()
            .map(|descriptor| {
                descriptor.verify()?;
                Ok(descriptor.source_component_sha256.clone())
            })
            .collect::<Result<BTreeSet<_>>>()?;
        self.body.physical_lining_descriptor_population.verify()?;
        if descriptor_components != exact_component_sha256
            || self.body.component_descriptors.len() != exact_component_sha256.len()
            || self.body.component_descriptors
                != self.body.physical_lining_descriptor_population.descriptors
        {
            bail!("component descriptors do not exactly preserve the physical-lining population");
        }
        let physical_occurrences = self
            .body
            .physical_lining_descriptor_population
            .physical_lining_evidence
            .iter()
            .map(|row| (row.ghost_lineage_sha256.as_str(), row))
            .collect::<BTreeMap<_, _>>();
        if physical_occurrences.len() != count
            || physical_occurrences
                .keys()
                .copied()
                .collect::<BTreeSet<_>>()
                != ghosts_by_lineage
                    .keys()
                    .map(String::as_str)
                    .collect::<BTreeSet<_>>()
        {
            bail!("physical-lining occurrences do not exactly cover selected Ghost lineage");
        }
        for (lineage_sha256, occurrence) in physical_occurrences {
            let measurement = ghosts_by_lineage
                .get(lineage_sha256)
                .context("physical-lining occurrence lacks its Ghost measurement")?;
            let sidecar = &measurement
                .structural_measurements
                .coordinate_cavity_sidecar;
            if occurrence.lattice_node_index != measurement.ghost.lattice_node_index
                || occurrence.cavity_sidecar_sha256 != sidecar.sidecar_sha256
                || occurrence.physical_lining != sidecar.body.physical_lining
            {
                bail!("lane physical-lining bytes differ from their exact cavity sidecar");
            }
        }
        match (
            self.body.runtime_request.body.lane_policy.selection_mode,
            &self.body.confirmation_surveillance,
        ) {
            (NativePathBSelectionMode::WholeLatticeDiscovery, None) => {}
            (NativePathBSelectionMode::TargetResidueHypothesis, Some(surveillance)) => {
                surveillance.verify()?;
                let lane_policy = &self.body.runtime_request.body.lane_policy;
                let tracker = lane_policy
                    .candidate_tracker
                    .as_ref()
                    .context("target lane policy lacks its candidate tracker")?;
                let lane = lane_policy
                    .stream_lanes
                    .values()
                    .next()
                    .copied()
                    .context("target lane policy has no physical lane")?;
                if &surveillance.candidate_tracker != tracker
                    || surveillance.control_lane != lane
                    || surveillance
                        .start_step_by_stream
                        .keys()
                        .copied()
                        .collect::<Vec<_>>()
                        != lane_policy.required_stream_ids
                {
                    bail!("confirmation surveillance differs from the admitted tracker/lane")
                }
            }
            (NativePathBSelectionMode::WholeLatticeDiscovery, Some(_)) => {
                bail!("whole-lattice discovery cannot relabel surveillance as Path-B confirmation")
            }
            (NativePathBSelectionMode::TargetResidueHypothesis, None) => {
                bail!("target confirmation requires a complete fixed-cadence surveillance ledger")
            }
        }
        let run_units = self
            .body
            .ghost_measurements
            .iter()
            .map(|row| {
                (
                    row.ghost.run_id.as_str(),
                    row.ghost.replica_id.as_str(),
                    row.ghost.replica_seed,
                )
            })
            .collect::<BTreeSet<_>>();
        let lane_identities = self
            .body
            .ghost_measurements
            .iter()
            .map(|row| row.control_lane)
            .collect::<BTreeSet<_>>();
        let policy_lane = self
            .body
            .runtime_request
            .body
            .lane_policy
            .stream_lanes
            .values()
            .next()
            .copied()
            .context("lane evidence policy has no physical lane")?;
        let lane_identity = if count == 0 {
            let surveillance = self
                .body
                .confirmation_surveillance
                .as_ref()
                .context("zero-event target lane lacks fixed-cadence surveillance")?;
            if run_units.len() != 0
                || lane_identities.len() != 0
                || surveillance.control_lane != policy_lane
            {
                bail!("zero-event lane identity disagrees with its surveillance authority");
            }
            policy_lane
        } else {
            if run_units.len() != 1 || lane_identities.len() != 1 {
                bail!(
                    "one lane evidence document must represent one run/replica seed and one lane"
                );
            }
            let observed_lane = *lane_identities.iter().next().unwrap();
            if observed_lane != policy_lane {
                bail!("observed lane identity differs from admitted lane policy");
            }
            observed_lane
        };
        for descriptor in &self.body.component_descriptors {
            if descriptor.control_lane != lane_identity {
                bail!("component descriptor control lane differs from lane evidence");
            }
            if let Some(surveillance) = &self.body.confirmation_surveillance {
                if descriptor.run_id != surveillance.run_id
                    || descriptor.replica_id != surveillance.replica_id
                    || descriptor.replica_seed != surveillance.replica_seed
                {
                    bail!("component descriptor run unit differs from surveillance authority");
                }
            }
        }
        self.body.source_authority.verify_for(
            lane_identity,
            &self
                .body
                .runtime_request
                .body
                .lane_policy
                .fragment_panel_sha256,
        )?;
        let selected_ghost_hashes = ghosts_by_lineage.keys().cloned().collect::<BTreeSet<_>>();
        verify_claim_fragment_join_derivation(
            lane_identity,
            &self.body.lambda_ghost_endpoint_join,
            &self.body.fragment_observations,
            &selected_ghost_hashes,
        )?;
        let mut lineage_hashes = Vec::with_capacity(count);
        for lattice in &self.body.lattice_observations {
            let ghost = ghosts_by_lineage
                .get(&lattice.ghost_lineage_sha256)
                .context("lattice observation has no exact global Ghost occurrence")?;
            let key = ghost.ghost.key;
            let fragment = fragments
                .get(lattice.ghost_lineage_sha256.as_str())
                .context("lattice observation key has no fragment/no-probe state")?;
            if fragment.ghost_key != key {
                bail!("fragment local key differs from its exact global Ghost occurrence");
            }
            lattice.verify_for(&ghost.ghost)?;
            validate_lane_measurement_and_fragment(
                ghost,
                &fragment.fragment,
                &self.body.runtime_request,
            )?;
            lineage_hashes.push(ghost.ghost.lineage_sha256()?);
        }
        lineage_hashes.sort();
        if lineage_hashes != self.body.site_selection.selected_ghost_lineage_sha256 {
            bail!("lane evidence record population differs from exact site selection");
        }
        require_sha256(&self.lane_evidence_sha256, "lane_evidence_sha256")?;
        if canonical_sha256(&self.body)? != self.lane_evidence_sha256 {
            bail!("per-run Path-B lane evidence content hash mismatch");
        }
        Ok(())
    }
}

fn verify_claim_fragment_join_derivation(
    lane: ControlLane,
    joins: &LambdaGhostEndpointJoinMaterialization,
    fragments: &[FragmentStateObservation],
    selected_ghost_hashes: &BTreeSet<String>,
) -> Result<()> {
    joins.verify()?;
    let join_ghost_hashes = joins
        .rows
        .iter()
        .filter_map(|row| row.ghost_lineage_sha256.clone())
        .collect::<BTreeSet<_>>();
    let evaluated_join_ghost_hashes = joins
        .rows
        .iter()
        .filter(|row| row.status == LambdaGhostEndpointJoinStatus::Evaluated)
        .filter_map(|row| row.ghost_lineage_sha256.clone())
        .collect::<BTreeSet<_>>();
    let fragment_ghost_hashes = fragments
        .iter()
        .map(|row| row.ghost_lineage_sha256.clone())
        .collect::<BTreeSet<_>>();
    if join_ghost_hashes != *selected_ghost_hashes
        || fragment_ghost_hashes != *selected_ghost_hashes
    {
        bail!("lane Ghost/fragment populations bypass or differ from immutable join input");
    }
    match lane {
        ControlLane::NoProbe => {
            if !evaluated_join_ghost_hashes.is_empty() {
                bail!("no-probe lane contains a claim-bearing lambda endpoint join");
            }
        }
        ControlLane::HypothesisFragment | ControlLane::OrthogonalChemicalControl => {
            if evaluated_join_ghost_hashes != *selected_ghost_hashes {
                bail!(
                    "probe lane lacks an evaluated immutable endpoint join for every fiber input"
                );
            }
            for fragment in fragments {
                let exact_joins = joins
                    .rows
                    .iter()
                    .filter(|row| {
                        row.status == LambdaGhostEndpointJoinStatus::Evaluated
                            && row.ghost_lineage_sha256.as_deref()
                                == Some(fragment.ghost_lineage_sha256.as_str())
                    })
                    .collect::<Vec<_>>();
                if exact_joins.len() != 1
                    || exact_joins[0]
                        .ghost_node
                        .as_ref()
                        .map(|node| node.ghost.key)
                        != Some(fragment.ghost_key)
                    || fragment.fragment.source_endpoint_observation.as_ref()
                        != exact_joins[0].endpoint.as_ref()
                    || fragment.fragment.source_force_event.as_ref()
                        != exact_joins[0].supporting_force.as_ref()
                {
                    bail!(
                        "claim-bearing fragment state did not derive from one evaluated immutable join"
                    );
                }
            }
        }
    }
    Ok(())
}

/// Durable emission receipt for one immutable per-run lane document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativePathBLaneEvidenceEmission {
    pub lane_evidence: NativePathBLaneEvidence,
    pub typed_artifact: TypedEvidenceArtifact,
    pub emission_sha256: String,
    /// Atomic publication authority for the exact join/lane byte population.
    pub transaction_commit: NativePathBLaneTransactionCommit,
    /// Absolute path of the verified commit receipt inside the published root.
    pub transaction_commit_path: String,
}

/// Fixed public directory atomically published by the runtime lane closeout.
pub const NATIVE_PATHB_LANE_TRANSACTION_DIRECTORY: &str = "native_pathb_lane_transaction";
/// Commit receipt filename. Its presence is necessary but not sufficient: the
/// loader authenticates the exact directory population and every member byte.
pub const NATIVE_PATHB_LANE_TRANSACTION_COMMIT_FILE: &str = "COMMIT.json";
pub const NATIVE_PATHB_LANE_TRANSACTION_SCHEMA: &str = "prism_native_pathb_lane_transaction_commit";
pub const NATIVE_PATHB_LANE_TRANSACTION_VERSION: &str = "1.0.0";

const LAMBDA_GHOST_JOIN_RAW_FILE: &str = "lambda_ghost_endpoint_join_materialization.json";
const NATIVE_PATHB_LANE_RAW_FILE: &str = "native_pathb_lane_evidence.json";

/// One exact byte member of the fail-atomic lane transaction. Membership is
/// ordered, closed and relative to the committed directory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativePathBLaneTransactionMember {
    pub ordinal: u8,
    pub role: String,
    pub relative_path: String,
    pub byte_length: u64,
    pub sha256: String,
}

/// Hash body of the fail-atomic lane commit. The commit receipt itself is not
/// a member, avoiding a self-referential hash; it seals the exact six-member
/// raw/QNANL/receipt population and both typed-artifact projections.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativePathBLaneTransactionCommitBody {
    pub transaction_id_sha256: String,
    pub committed_directory: String,
    pub lane_evidence_sha256: String,
    pub lambda_ghost_endpoint_join_sha256: String,
    pub ordered_membership_sha256: String,
    pub member_count: u64,
    pub members: Vec<NativePathBLaneTransactionMember>,
    pub lambda_ghost_endpoint_join_typed_artifact: TypedEvidenceArtifact,
    pub native_pathb_lane_typed_artifact: TypedEvidenceArtifact,
}

/// Durable transaction commit receipt written last inside staging and exposed
/// only by the single atomic directory publication.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativePathBLaneTransactionCommit {
    pub schema_kind: String,
    pub schema_version: String,
    pub status: String,
    pub commit_sha256: String,
    pub body: NativePathBLaneTransactionCommitBody,
}

#[derive(Debug)]
struct VerifiedNativePathBLaneTransactionBytes {
    commit: NativePathBLaneTransactionCommit,
    join_raw: Vec<u8>,
    lane_raw: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NativePathBLaneTransactionFailpoint {
    None,
    AfterJoinTriplet,
    AfterCompleteStageBeforeCommit,
    AfterCommitBeforePublish,
}

#[derive(Serialize)]
struct NativePathBAvailabilityBody<'a> {
    runtime_request_sha256: &'a str,
    lattice_manifest_sha256: &'a str,
    stream_lanes: &'a BTreeMap<u16, ControlLane>,
    site_key: &'a str,
    required_stream_ids: &'a [u16],
    observed_stream_topology_sha256: &'a BTreeMap<u16, String>,
    ghost_lineage_count: u64,
    exact_coordinate_frame_count: u64,
    exact_lambda_state_count: u64,
    materialized_pathb_state_count: u64,
    materialized_fiber_count: u64,
    candidate_append_count: u64,
    reason_codes: &'a [String],
    source_contract: &'static str,
}

fn validate_lane_measurement_and_fragment(
    observation: &GhostFiberMeasurementObservation,
    fragment: &crate::chemical_state_fiber::FragmentState,
    request: &NativePathBRuntimeRequest,
) -> Result<()> {
    observation.ghost.validate()?;
    if observation.site_key != request.body.lane_policy.site_key {
        bail!("lane measurement site key differs from admitted request");
    }
    let lane = request
        .body
        .lane_policy
        .stream_lanes
        .get(&observation.ghost.key.stream_id)
        .context("lane measurement stream is absent from admitted request")?;
    if *lane != observation.control_lane {
        bail!("lane measurement control identity differs from its stream policy");
    }
    observation
        .protocol_phase
        .validate_for(observation.ghost.key)?;
    observation.thermodynamic_condition.validate()?;
    if observation.protocol_phase.phase_ordinal != observation.ghost.protocol_phase
        || observation.protocol_phase.thermodynamic_condition_id
            != observation.thermodynamic_condition.condition_id
        || observation.protocol_phase.thermodynamic_condition_id != observation.ghost.condition_id
        || observation.protocol_phase.protocol_name != observation.ghost.protocol_name
        || observation.protocol_phase.phase_schedule_digest_sha256
            != observation.ghost.phase_schedule_digest_sha256
        || observation.protocol_phase.protocol_state_hash != observation.ghost.protocol_state_hash
    {
        bail!("lane measurement phase/condition differs from exact Ghost lineage");
    }
    observation
        .structural_measurements
        .coordinate_cavity_sidecar
        .verify()?;
    observation
        .structural_measurements
        .coordinate_frame
        .validate_for(&observation.ghost)?;
    if observation
        .structural_measurements
        .coordinate_frame
        .source_contract
        != "ghost_coordinate_segment_exact_join_v1"
    {
        bail!("production lane evidence requires the signed Ghost-coordinate segment join");
    }
    let sidecar = &observation
        .structural_measurements
        .coordinate_cavity_sidecar;
    let coordinate = &observation.structural_measurements.coordinate_frame;
    if sidecar.body.ghost_record_sha256 != observation.ghost.record_sha256
        || sidecar.body.stream_id != observation.ghost.key.stream_id
        || sidecar.body.step_idx != observation.ghost.key.step_idx
        || sidecar.body.coordinate_frame_sha256 != coordinate.coordinate_frame_sha256
        || sidecar.body.trajectory_artifact_sha256 != coordinate.trajectory_artifact_sha256
        || sidecar.body.trajectory_audit_sha256 != coordinate.trajectory_audit_sha256
        || sidecar.body.topology_sha256 != coordinate.topology_sha256
        || sidecar.body.atom_residue_mapping_sha256 != coordinate.atom_residue_mapping_sha256
        || sidecar.sidecar_sha256 != coordinate.cavity_sidecar_record_sha256
    {
        bail!("coordinate cavity sidecar differs from exact coordinate/Ghost lineage");
    }
    let pocket_volume = observation
        .structural_measurements
        .pocket_volume_a3
        .as_ref()
        .context("per-run lane evidence requires exact coordinate cavity volume")?;
    if pocket_volume.value.to_bits() != sidecar.body.mean_volume_a3.to_bits()
        || pocket_volume.method_id != "prism_multiresolution_coordinate_ligsite_v1"
        || pocket_volume.estimator_sha256 != sidecar.body.estimator_sha256
        || pocket_volume.source_artifact_sha256 != sidecar.sidecar_sha256
    {
        bail!("lane pocket-volume scalar differs from its complete cavity sidecar");
    }
    observation.uncertainty.validate()?;
    if observation.uncertainty.replica_seed != observation.ghost.replica_seed
        || observation.uncertainty.replica_id != observation.ghost.replica_id
    {
        bail!("lane uncertainty seed/replica differs from Ghost lineage");
    }
    if observation
        .evidence_artifact_sha256
        .get("coordinate_cavity_sidecar")
        != Some(&sidecar.sidecar_sha256)
        || observation
            .evidence_artifact_sha256
            .get("physical_cavity_lining")
            != Some(&sidecar.body.physical_lining.evidence_sha256)
    {
        bail!("lane evidence map does not preserve exact coordinate-cavity lining bytes");
    }
    for (name, digest) in &observation.evidence_artifact_sha256 {
        if name.trim().is_empty() {
            bail!("lane evidence artifact name is empty");
        }
        require_sha256(digest, "lane evidence artifact_sha256")?;
    }
    fragment.validate_for_ghost_context(
        *lane,
        &observation.ghost,
        &observation.protocol_phase,
        &observation.thermodynamic_condition,
    )?;
    match (
        lane,
        request
            .body
            .lane_policy
            .fragment_slot_ids
            .get(&observation.ghost.key.stream_id),
    ) {
        (ControlLane::NoProbe, None) => {}
        (_, Some(expected_slot)) => match &fragment.occupancy {
            crate::chemical_state_fiber::FragmentOccupancy::Present {
                fragment_slot_id, ..
            } if fragment_slot_id == expected_slot => {}
            crate::chemical_state_fiber::FragmentOccupancy::Absent
                if fragment.lambda == 0.0
                    && fragment
                        .source_force_event
                        .as_ref()
                        .map(|force| force.slot_id == *expected_slot)
                        .unwrap_or(false) => {}
            _ => bail!("fragment observation does not match its predeclared physical slot"),
        },
        _ => bail!("lane fragment-slot policy is incomplete"),
    }
    Ok(())
}

/// Coordinate-derived measurements supplied by their authoritative estimators.
/// Missing measurements remain absent; the orchestrator never fills scalar
/// heuristics or labels lattice AABB volume as cavity volume.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuralMeasurementSet {
    /// Exact coordinate frame and topology/mapping authority.
    pub coordinate_frame: ExactCoordinateFrameBinding,
    /// Complete content-addressed cavity record whose derived scalar is used
    /// below. The emitter recomputes its hash and grid aggregates.
    pub coordinate_cavity_sidecar: crate::coordinate_cavity::CoordinateCavitySidecar,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pocket_volume_a3: Option<PhysicalMeasurement>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pocket_depth_a: Option<PhysicalMeasurement>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub solvent_accessible_area_a2: Option<PhysicalMeasurement>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delta_volume_a3: Option<PhysicalMeasurement>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delta_hydration: Option<PhysicalMeasurement>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fragment_contact_fraction: Option<PhysicalMeasurement>,
}

/// Exact, non-interpolated coordinate authority for one Ghost stream/step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExactCoordinateFrameBinding {
    pub stream_id: u16,
    pub step_idx: u64,
    pub trajectory_frame_ordinal: u64,
    pub trajectory_byte_offset: u64,
    pub trajectory_byte_length: u64,
    pub coordinate_frame_sha256: String,
    pub trajectory_artifact_sha256: String,
    pub trajectory_audit_sha256: String,
    pub topology_sha256: String,
    pub atom_residue_mapping_sha256: String,
    pub cavity_sidecar_record_sha256: String,
    pub atom_count: u64,
    pub source_contract: String,
}

impl ExactCoordinateFrameBinding {
    fn validate_exact_source(&self) -> Result<()> {
        if self.trajectory_byte_length == 0 || self.atom_count == 0 {
            bail!("coordinate frame byte/atom population must be non-zero");
        }
        self.trajectory_byte_offset
            .checked_add(self.trajectory_byte_length)
            .context("coordinate frame byte range overflow")?;
        for (value, field) in [
            (&self.coordinate_frame_sha256, "coordinate_frame_sha256"),
            (
                &self.trajectory_artifact_sha256,
                "trajectory_artifact_sha256",
            ),
            (&self.trajectory_audit_sha256, "trajectory_audit_sha256"),
            (&self.topology_sha256, "topology_sha256"),
            (
                &self.atom_residue_mapping_sha256,
                "atom_residue_mapping_sha256",
            ),
            (
                &self.cavity_sidecar_record_sha256,
                "cavity_sidecar_record_sha256",
            ),
        ] {
            require_sha256(value, field)?;
        }
        if !matches!(
            self.source_contract.as_str(),
            "lossless_v2_exact_stream_step_no_interpolation"
                | "ghost_coordinate_segment_exact_join_v1"
                | "fixed_cadence_candidate_confirmation_v1"
        ) {
            bail!("coordinate source is not an accepted exact-frame contract");
        }
        Ok(())
    }

    fn validate_for(&self, ghost: &GhostByteLineage) -> Result<()> {
        if self.stream_id != ghost.key.stream_id || self.step_idx != ghost.key.step_idx
            || self.topology_sha256 != ghost.topology_sha256 {
            bail!("coordinate frame must match the exact Ghost stream, step, and topology");
        }
        self.validate_exact_source()
    }
}

/// Fixed-cadence observation status for one predeclared component tracker.
/// `NotDetected` is a typed surveillance observation, not a physical claim
/// that the coordinate-defined pocket is closed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum PathBGhostEventStatus {
    Emitted {
        ghost_lineage_sha256: String,
        exact_component_sha256: String,
    },
    /// More than one selected native component emitted at the same scheduled
    /// stream/step. All identities are preserved, but the row cannot be used
    /// as a unique detected-state assignment.
    AmbiguousEmissions {
        ghost_lineage_sha256: Vec<String>,
        exact_component_sha256: Vec<String>,
    },
    NotEmitted {
        typed_absence_receipt_sha256: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PathBEvaluableMeasurement {
    pub coordinate_frame: ExactCoordinateFrameBinding,
    pub local_hydration_shell_sha256: String,
    pub local_fragment_shell_sha256: String,
    pub coordinate_cavity_status_sha256: String,
    /// Complete producer sidecar, including exact packed grids, boundary faces,
    /// atom identities and multigrid conservation. The status SHA above is
    /// retained as a compact index but must resolve to these exact bytes.
    pub coordinate_cavity_sidecar: crate::coordinate_cavity::CoordinateCavitySidecar,
    pub cavity_volume_a3: f64,
    /// Multi-grid discretization interval. Binary open/closed labels are
    /// admitted only when this entire interval lies beyond one predeclared
    /// threshold; threshold-straddling rows remain continuous observations.
    pub cavity_discretization_min_a3: f64,
    pub cavity_discretization_max_a3: f64,
    pub cavity_discretization_standard_error_a3: f64,
    pub kinetic_feature_schema_sha256: String,
    pub kinetic_features: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum PathBMeasurementStatus {
    ObservedOpen {
        evidence: PathBEvaluableMeasurement,
    },
    ObservedClosed {
        evidence: PathBEvaluableMeasurement,
    },
    /// Exact coordinates and continuous structural features are valid, but
    /// the multi-grid cavity interval does not support either binary label.
    /// This row remains eligible for structural TICA/MSM and is excluded only
    /// from open/closed label counts.
    ObservedIndeterminate {
        evidence: PathBEvaluableMeasurement,
    },
    NotEvaluable {
        reason_code: String,
        evidence_receipt_sha256: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CavityIntervalClass {
    Open,
    Closed,
    Indeterminate,
}

fn classify_cavity_discretization_interval(
    minimum_a3: f64,
    maximum_a3: f64,
    tracker: &PathBCandidateTrackerDefinition,
) -> Result<CavityIntervalClass> {
    if !minimum_a3.is_finite()
        || !maximum_a3.is_finite()
        || minimum_a3 < 0.0
        || maximum_a3 < minimum_a3
    {
        bail!("invalid cavity discretization interval");
    }
    Ok(if minimum_a3 >= tracker.observed_open_min_volume_a3 {
        CavityIntervalClass::Open
    } else if maximum_a3 <= tracker.observed_closed_max_volume_a3 {
        CavityIntervalClass::Closed
    } else {
        CavityIntervalClass::Indeterminate
    })
}

/// One scheduled time-at-risk observation with exact coordinates and local
/// solvent/fragment shells even when Ghost emitted no site record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PathBSurveillanceRow {
    pub row_sha256: String,
    pub stream_id: u16,
    pub step_idx: u64,
    pub protocol_phase: ProtocolPhaseBinding,
    pub thermodynamic_condition: ThermodynamicCondition,
    pub ghost_event: PathBGhostEventStatus,
    pub measurement: PathBMeasurementStatus,
}

#[derive(Serialize)]
struct PathBSurveillanceRowBody<'a> {
    stream_id: u16,
    step_idx: u64,
    protocol_phase: &'a ProtocolPhaseBinding,
    thermodynamic_condition: &'a ThermodynamicCondition,
    ghost_event: &'a PathBGhostEventStatus,
    measurement: &'a PathBMeasurementStatus,
}

impl PathBSurveillanceRow {
    fn verify(&self, tracker: &PathBCandidateTrackerDefinition) -> Result<()> {
        self.protocol_phase
            .validate_for(crate::chemical_state_fiber::GhostLineageKey {
                stream_id: self.stream_id,
                frame_idx: self.step_idx,
                step_idx: self.step_idx,
                site_id: 0,
                record_index: self.step_idx,
            })?;
        self.thermodynamic_condition.validate()?;
        if self.protocol_phase.thermodynamic_condition_id
            != self.thermodynamic_condition.condition_id
        {
            bail!("surveillance row phase and thermodynamic condition disagree");
        }
        require_sha256(&self.row_sha256, "surveillance row_sha256")?;
        match &self.ghost_event {
            PathBGhostEventStatus::Emitted {
                ghost_lineage_sha256,
                exact_component_sha256,
            } => {
                require_sha256(ghost_lineage_sha256, "surveillance Ghost lineage")?;
                require_sha256(exact_component_sha256, "surveillance exact component")?;
            }
            PathBGhostEventStatus::AmbiguousEmissions {
                ghost_lineage_sha256,
                exact_component_sha256,
            } => {
                if ghost_lineage_sha256.len() < 2
                    || ghost_lineage_sha256.len() != exact_component_sha256.len()
                    || ghost_lineage_sha256
                        .windows(2)
                        .any(|pair| pair[0] >= pair[1])
                {
                    bail!("ambiguous surveillance emission population is not canonical");
                }
                for digest in ghost_lineage_sha256 {
                    require_sha256(digest, "ambiguous surveillance Ghost lineage")?;
                }
                for digest in exact_component_sha256 {
                    require_sha256(digest, "ambiguous surveillance exact component")?;
                }
                if !matches!(
                    self.measurement,
                    PathBMeasurementStatus::NotEvaluable { .. }
                ) {
                    bail!("ambiguous Ghost emissions must break the evaluable transition series");
                }
            }
            PathBGhostEventStatus::NotEmitted {
                typed_absence_receipt_sha256,
            } => require_sha256(typed_absence_receipt_sha256, "typed_absence_receipt_sha256")?,
        }
        match &self.measurement {
            PathBMeasurementStatus::ObservedOpen { evidence }
            | PathBMeasurementStatus::ObservedClosed { evidence }
            | PathBMeasurementStatus::ObservedIndeterminate { evidence } => {
                let PathBEvaluableMeasurement {
                    coordinate_frame,
                    local_hydration_shell_sha256,
                    local_fragment_shell_sha256,
                    coordinate_cavity_status_sha256,
                    coordinate_cavity_sidecar,
                    cavity_volume_a3,
                    cavity_discretization_min_a3,
                    cavity_discretization_max_a3,
                    cavity_discretization_standard_error_a3,
                    kinetic_feature_schema_sha256,
                    kinetic_features,
                } = evidence;
                if coordinate_frame.stream_id != self.stream_id
                    || coordinate_frame.step_idx != self.step_idx
                {
                    bail!("surveillance coordinate identity differs from scheduled stream/step");
                }
                coordinate_frame.validate_exact_source()?;
                coordinate_cavity_sidecar.verify()?;
                if coordinate_frame.source_contract != "fixed_cadence_candidate_confirmation_v1"
                    || kinetic_features.len() < 3
                    || kinetic_features.iter().any(|value| !value.is_finite())
                    || !cavity_volume_a3.is_finite()
                    || *cavity_volume_a3 < 0.0
                    || !cavity_discretization_min_a3.is_finite()
                    || !cavity_discretization_max_a3.is_finite()
                    || !cavity_discretization_standard_error_a3.is_finite()
                    || *cavity_discretization_min_a3 < 0.0
                    || cavity_discretization_min_a3 > cavity_volume_a3
                    || cavity_volume_a3 > cavity_discretization_max_a3
                    || *cavity_discretization_standard_error_a3 < 0.0
                    || coordinate_cavity_status_sha256 != &coordinate_cavity_sidecar.sidecar_sha256
                    || coordinate_frame.cavity_sidecar_record_sha256
                        != coordinate_cavity_sidecar.sidecar_sha256
                    || coordinate_frame.coordinate_frame_sha256
                        != coordinate_cavity_sidecar.body.coordinate_frame_sha256
                    || coordinate_frame.topology_sha256
                        != coordinate_cavity_sidecar.body.topology_sha256
                    || coordinate_frame.atom_residue_mapping_sha256
                        != coordinate_cavity_sidecar.body.atom_residue_mapping_sha256
                    || cavity_volume_a3.to_bits()
                        != coordinate_cavity_sidecar.body.mean_volume_a3.to_bits()
                    || cavity_discretization_min_a3.to_bits()
                        != coordinate_cavity_sidecar
                            .body
                            .discretization_min_a3
                            .to_bits()
                    || cavity_discretization_max_a3.to_bits()
                        != coordinate_cavity_sidecar
                            .body
                            .discretization_max_a3
                            .to_bits()
                    || cavity_discretization_standard_error_a3.to_bits()
                        != coordinate_cavity_sidecar
                            .body
                            .discretization_standard_error_a3
                            .to_bits()
                {
                    bail!("evaluable surveillance measurement is incomplete");
                }
                for (digest, field) in [
                    (local_hydration_shell_sha256, "local hydration shell"),
                    (local_fragment_shell_sha256, "local fragment shell"),
                    (coordinate_cavity_status_sha256, "coordinate cavity status"),
                    (kinetic_feature_schema_sha256, "kinetic feature schema"),
                ] {
                    require_sha256(digest, field)?;
                }
                match &self.measurement {
                    PathBMeasurementStatus::ObservedOpen { .. }
                        if *cavity_discretization_min_a3 < tracker.observed_open_min_volume_a3 =>
                    {
                        bail!("observed-open row discretization interval crosses its threshold")
                    }
                    PathBMeasurementStatus::ObservedClosed { .. }
                        if *cavity_discretization_max_a3
                            > tracker.observed_closed_max_volume_a3 =>
                    {
                        bail!("observed-closed row discretization interval crosses its threshold")
                    }
                    PathBMeasurementStatus::ObservedIndeterminate { .. }
                        if *cavity_discretization_min_a3 >= tracker.observed_open_min_volume_a3
                            || *cavity_discretization_max_a3
                                <= tracker.observed_closed_max_volume_a3 =>
                    {
                        bail!("indeterminate row supports an unambiguous binary label")
                    }
                    _ => {}
                }
            }
            PathBMeasurementStatus::NotEvaluable {
                reason_code,
                evidence_receipt_sha256,
            } => {
                if reason_code.trim().is_empty() {
                    bail!("not-evaluable surveillance row lacks reason code");
                }
                require_sha256(evidence_receipt_sha256, "not-evaluable evidence receipt")?;
            }
        }
        tracker.verify()?;
        let body = PathBSurveillanceRowBody {
            stream_id: self.stream_id,
            step_idx: self.step_idx,
            protocol_phase: &self.protocol_phase,
            thermodynamic_condition: &self.thermodynamic_condition,
            ghost_event: &self.ghost_event,
            measurement: &self.measurement,
        };
        if canonical_sha256(&(tracker.tracker_sha256.as_str(), body))? != self.row_sha256 {
            bail!("surveillance row content hash mismatch");
        }
        Ok(())
    }

    /// Construct one producer row and bind it to the exact predeclared
    /// tracker. Open/closed/NOT_EVALUABLE classification is validated before
    /// the canonical row digest is returned.
    pub fn materialize(
        tracker: &PathBCandidateTrackerDefinition,
        stream_id: u16,
        step_idx: u64,
        protocol_phase: ProtocolPhaseBinding,
        thermodynamic_condition: ThermodynamicCondition,
        ghost_event: PathBGhostEventStatus,
        measurement: PathBMeasurementStatus,
    ) -> Result<Self> {
        let mut row = Self {
            row_sha256: String::new(),
            stream_id,
            step_idx,
            protocol_phase,
            thermodynamic_condition,
            ghost_event,
            measurement,
        };
        let body = PathBSurveillanceRowBody {
            stream_id: row.stream_id,
            step_idx: row.step_idx,
            protocol_phase: &row.protocol_phase,
            thermodynamic_condition: &row.thermodynamic_condition,
            ghost_event: &row.ghost_event,
            measurement: &row.measurement,
        };
        row.row_sha256 = canonical_sha256(&(tracker.tracker_sha256.as_str(), body))?;
        row.verify(tracker)?;
        Ok(row)
    }
}

/// Complete fixed-cadence time-at-risk ledger for one predeclared candidate
/// tracker and one physical lane/run/seed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PathBConfirmationSurveillanceLedger {
    pub schema_kind: String,
    pub schema_version: String,
    pub ledger_sha256: String,
    pub run_id: String,
    pub replica_id: String,
    pub replica_seed: u64,
    pub control_lane: ControlLane,
    pub engine_sha256: String,
    pub topology_sha256: String,
    pub matched_physical_state_sha256: String,
    pub stationary_phase_ordinal: u16,
    pub stationary_hold_certified: bool,
    pub stationarity_evidence_sha256: String,
    pub candidate_tracker_sha256: String,
    pub candidate_tracker: PathBCandidateTrackerDefinition,
    pub source_authority_sha256: String,
    pub cadence_steps: u64,
    pub start_step_by_stream: BTreeMap<u16, u64>,
    pub end_step_exclusive_by_stream: BTreeMap<u16, u64>,
    pub rows: Vec<PathBSurveillanceRow>,
    pub scheduled_row_count: u64,
    pub persisted_row_count: u64,
    pub decoded_row_count: u64,
    pub evaluable_row_count: u64,
    pub not_evaluable_row_count: u64,
    pub observed_open_row_count: u64,
    pub observed_closed_row_count: u64,
    pub observed_indeterminate_row_count: u64,
    pub ghost_emitted_row_count: u64,
    pub ghost_ambiguous_row_count: u64,
    pub ghost_not_emitted_row_count: u64,
    pub complete_time_at_risk: bool,
    pub source_manifest_sha256: String,
}

#[derive(Serialize)]
struct PathBConfirmationSurveillanceLedgerBody<'a> {
    run_id: &'a str,
    replica_id: &'a str,
    replica_seed: u64,
    control_lane: ControlLane,
    engine_sha256: &'a str,
    topology_sha256: &'a str,
    matched_physical_state_sha256: &'a str,
    stationary_phase_ordinal: u16,
    stationary_hold_certified: bool,
    stationarity_evidence_sha256: &'a str,
    candidate_tracker_sha256: &'a str,
    candidate_tracker: &'a PathBCandidateTrackerDefinition,
    source_authority_sha256: &'a str,
    cadence_steps: u64,
    start_step_by_stream: &'a BTreeMap<u16, u64>,
    end_step_exclusive_by_stream: &'a BTreeMap<u16, u64>,
    rows: &'a [PathBSurveillanceRow],
    scheduled_row_count: u64,
    persisted_row_count: u64,
    decoded_row_count: u64,
    evaluable_row_count: u64,
    not_evaluable_row_count: u64,
    observed_open_row_count: u64,
    observed_closed_row_count: u64,
    observed_indeterminate_row_count: u64,
    ghost_emitted_row_count: u64,
    ghost_ambiguous_row_count: u64,
    ghost_not_emitted_row_count: u64,
    complete_time_at_risk: bool,
    source_manifest_sha256: &'a str,
}

impl PathBConfirmationSurveillanceLedger {
    fn body(&self) -> PathBConfirmationSurveillanceLedgerBody<'_> {
        PathBConfirmationSurveillanceLedgerBody {
            run_id: &self.run_id,
            replica_id: &self.replica_id,
            replica_seed: self.replica_seed,
            control_lane: self.control_lane,
            engine_sha256: &self.engine_sha256,
            topology_sha256: &self.topology_sha256,
            matched_physical_state_sha256: &self.matched_physical_state_sha256,
            stationary_phase_ordinal: self.stationary_phase_ordinal,
            stationary_hold_certified: self.stationary_hold_certified,
            stationarity_evidence_sha256: &self.stationarity_evidence_sha256,
            candidate_tracker_sha256: &self.candidate_tracker_sha256,
            candidate_tracker: &self.candidate_tracker,
            source_authority_sha256: &self.source_authority_sha256,
            cadence_steps: self.cadence_steps,
            start_step_by_stream: &self.start_step_by_stream,
            end_step_exclusive_by_stream: &self.end_step_exclusive_by_stream,
            rows: &self.rows,
            scheduled_row_count: self.scheduled_row_count,
            persisted_row_count: self.persisted_row_count,
            decoded_row_count: self.decoded_row_count,
            evaluable_row_count: self.evaluable_row_count,
            not_evaluable_row_count: self.not_evaluable_row_count,
            observed_open_row_count: self.observed_open_row_count,
            observed_closed_row_count: self.observed_closed_row_count,
            observed_indeterminate_row_count: self.observed_indeterminate_row_count,
            ghost_emitted_row_count: self.ghost_emitted_row_count,
            ghost_ambiguous_row_count: self.ghost_ambiguous_row_count,
            ghost_not_emitted_row_count: self.ghost_not_emitted_row_count,
            complete_time_at_risk: self.complete_time_at_risk,
            source_manifest_sha256: &self.source_manifest_sha256,
        }
    }

    /// Seal a producer-completed surveillance ledger after every row has been
    /// durably persisted and decoded. The producer leaves `ledger_sha256`
    /// empty; no count or scientific status is inferred here.
    pub fn materialize(mut self) -> Result<Self> {
        if !self.ledger_sha256.is_empty() {
            bail!("surveillance ledger materialization requires an empty digest field");
        }
        self.ledger_sha256 = canonical_sha256(&self.body())?;
        self.verify()?;
        Ok(self)
    }

    pub fn verify(&self) -> Result<()> {
        if self.schema_kind != "prism_pathb_confirmation_surveillance_ledger"
            || self.schema_version != "4.0.0"
            || self.run_id.trim().is_empty()
            || self.replica_id.trim().is_empty()
            || !self.stationary_hold_certified
            || self.cadence_steps == 0
            || usize::try_from(self.cadence_steps).is_err()
            || !self.complete_time_at_risk
            || self.scheduled_row_count != self.persisted_row_count
            || self.persisted_row_count != self.decoded_row_count
            || self.decoded_row_count != self.rows.len() as u64
            || self.evaluable_row_count + self.not_evaluable_row_count != self.rows.len() as u64
            || self.observed_open_row_count
                + self.observed_closed_row_count
                + self.observed_indeterminate_row_count
                != self.evaluable_row_count
            || self.ghost_emitted_row_count
                + self.ghost_ambiguous_row_count
                + self.ghost_not_emitted_row_count
                != self.scheduled_row_count
            || self.start_step_by_stream.keys().collect::<Vec<_>>()
                != self.end_step_exclusive_by_stream.keys().collect::<Vec<_>>()
            || self.candidate_tracker_sha256 != self.candidate_tracker.tracker_sha256
            || self.cadence_steps != self.candidate_tracker.cadence_steps
            || self.start_step_by_stream != self.candidate_tracker.start_step_by_stream
            || self.end_step_exclusive_by_stream
                != self.candidate_tracker.end_step_exclusive_by_stream
            || self.stationary_phase_ordinal != self.candidate_tracker.stationary_phase_ordinal
            || self.matched_physical_state_sha256
                != self.candidate_tracker.matched_physical_state_sha256
        {
            bail!("incomplete Path-B confirmation surveillance ledger");
        }
        for (digest, field) in [
            (&self.ledger_sha256, "surveillance ledger_sha256"),
            (&self.engine_sha256, "surveillance engine_sha256"),
            (&self.topology_sha256, "surveillance topology_sha256"),
            (
                &self.matched_physical_state_sha256,
                "surveillance matched physical state",
            ),
            (
                &self.stationarity_evidence_sha256,
                "stationarity evidence_sha256",
            ),
            (&self.candidate_tracker_sha256, "candidate_tracker_sha256"),
            (
                &self.source_authority_sha256,
                "surveillance source_authority_sha256",
            ),
            (
                &self.source_manifest_sha256,
                "surveillance source_manifest_sha256",
            ),
        ] {
            require_sha256(digest, field)?;
        }
        self.candidate_tracker.verify()?;
        let mut rows_by_stream: BTreeMap<u16, Vec<&PathBSurveillanceRow>> = BTreeMap::new();
        let mut feature_schema = BTreeSet::new();
        let mut evaluable = 0u64;
        let mut observed_open = 0u64;
        let mut observed_closed = 0u64;
        let mut observed_indeterminate = 0u64;
        let mut ghost_emitted = 0u64;
        let mut ghost_ambiguous = 0u64;
        let mut ghost_not_emitted = 0u64;
        for row in &self.rows {
            row.verify(&self.candidate_tracker)?;
            if row.protocol_phase.phase_ordinal != self.stationary_phase_ordinal
                || row
                    .thermodynamic_condition
                    .matched_physical_state_sha256(&row.protocol_phase)?
                    != self.matched_physical_state_sha256
            {
                bail!("reversible surveillance ledger crosses phase or physical condition");
            }
            match &row.ghost_event {
                PathBGhostEventStatus::Emitted { .. } => ghost_emitted += 1,
                PathBGhostEventStatus::AmbiguousEmissions { .. } => ghost_ambiguous += 1,
                PathBGhostEventStatus::NotEmitted { .. } => ghost_not_emitted += 1,
            }
            match &row.measurement {
                PathBMeasurementStatus::ObservedOpen { evidence } => {
                    evaluable += 1;
                    observed_open += 1;
                    feature_schema.insert(evidence.kinetic_feature_schema_sha256.clone());
                }
                PathBMeasurementStatus::ObservedClosed { evidence } => {
                    evaluable += 1;
                    observed_closed += 1;
                    feature_schema.insert(evidence.kinetic_feature_schema_sha256.clone());
                }
                PathBMeasurementStatus::ObservedIndeterminate { evidence } => {
                    evaluable += 1;
                    observed_indeterminate += 1;
                    feature_schema.insert(evidence.kinetic_feature_schema_sha256.clone());
                }
                PathBMeasurementStatus::NotEvaluable { .. } => {}
            }
            rows_by_stream.entry(row.stream_id).or_default().push(row);
        }
        if evaluable != self.evaluable_row_count
            || observed_open != self.observed_open_row_count
            || observed_closed != self.observed_closed_row_count
            || observed_indeterminate != self.observed_indeterminate_row_count
            || ghost_emitted != self.ghost_emitted_row_count
            || ghost_ambiguous != self.ghost_ambiguous_row_count
            || ghost_not_emitted != self.ghost_not_emitted_row_count
            || feature_schema.len() > 1
        {
            bail!("surveillance evaluability count or feature schema is inconsistent");
        }
        let cadence = usize::try_from(self.cadence_steps)?;
        for (stream, start) in &self.start_step_by_stream {
            let end = self.end_step_exclusive_by_stream[stream];
            if end <= *start {
                bail!("surveillance stream interval is empty");
            }
            let expected = (*start..end).step_by(cadence).collect::<Vec<_>>();
            let mut observed = rows_by_stream
                .get(stream)
                .context("surveillance ledger omits a declared stream")?
                .iter()
                .map(|row| row.step_idx)
                .collect::<Vec<_>>();
            observed.sort_unstable();
            if observed != expected {
                bail!("surveillance ledger has a gap, duplicate or off-cadence row");
            }
        }
        let mut canonical_rows = self.rows.clone();
        canonical_rows.sort_by_key(|row| (row.stream_id, row.step_idx));
        if canonical_rows != self.rows || canonical_sha256(&self.body())? != self.ledger_sha256 {
            bail!("surveillance ledger order or content hash mismatch");
        }
        Ok(())
    }
}

/// Materialize the complete fixed-cadence target-confirmation ledger. Ghost
/// emission is joined independently of physical coordinate evaluability;
/// Indeterminate cavity volumes remain continuous structural rows without a
/// binary label or segment split. Only truly non-evaluable coordinates and
/// ambiguous multi-component emissions break reversible transitions.
pub fn materialize_fixed_cadence_surveillance(
    request: &NativePathBRuntimeRequest,
    selected: &VerifiedGhostSitePathBInputs,
    connectivity: &NativePathBConnectivityAuthority,
    fixed_streams: &BTreeMap<u16, crate::fixed_cadence_coordinate::FixedCadenceCoordinateLoad>,
    fixed_population_sha256: &str,
    lambda_load: &crate::lambda_fragment::LambdaEventJournalLoad,
    topology: &crate::input::PrismPrepTopology,
) -> Result<PathBConfirmationSurveillanceLedger> {
    request.verify()?;
    validate_fixed_cadence_timebase_binding(request, fixed_streams)?;
    selected.selection.verify()?;
    connectivity.verify_for(&selected.lineages)?;
    require_sha256(fixed_population_sha256, "fixed-cadence population")?;
    if request.body.lane_policy.selection_mode != NativePathBSelectionMode::TargetResidueHypothesis
    {
        bail!("fixed-cadence surveillance is only valid for target confirmation");
    }
    let tracker = request
        .body
        .lane_policy
        .candidate_tracker
        .as_ref()
        .context("target confirmation lacks candidate tracker")?;
    tracker.verify()?;
    let lane = request
        .body
        .lane_policy
        .stream_lanes
        .values()
        .next()
        .copied()
        .context("target confirmation lacks physical lane")?;
    let expected_streams = request
        .body
        .lane_policy
        .required_stream_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    if fixed_streams.keys().copied().collect::<BTreeSet<_>>() != expected_streams {
        bail!("fixed-cadence surveillance stream population differs from request");
    }
    let component_by_lineage = connectivity
        .components
        .iter()
        .flat_map(|component| {
            component
                .node_ghost_lineage_sha256
                .iter()
                .map(move |lineage| (lineage.clone(), component.component_sha256.clone()))
        })
        .collect::<BTreeMap<_, _>>();
    if component_by_lineage.len() != selected.lineages.len() {
        bail!("surveillance component lookup does not conserve selected Ghost nodes");
    }
    let mut detections: BTreeMap<(u16, u64), Vec<(String, String, String)>> = BTreeMap::new();
    for lineage in &selected.lineages {
        let lineage_sha256 = lineage.lineage_sha256()?;
        detections
            .entry((lineage.key.stream_id, lineage.key.step_idx))
            .or_default()
            .push((
                lineage_sha256.clone(),
                component_by_lineage
                    .get(&lineage_sha256)
                    .context("selected Ghost lacks exact component")?
                    .clone(),
                lineage.record_sha256.clone(),
            ));
    }
    for rows in detections.values_mut() {
        rows.sort_by(|left, right| left.0.cmp(&right.0));
    }
    let feature_schema_sha256 = canonical_sha256(&serde_json::json!({
        "schema_kind": "prism_pathb_confirmation_kinetic_features",
        "schema_version": 1,
        "features": [
            {"name": "coordinate_cavity_volume", "unit": "angstrom^3"},
            {"name": "local_hydration_atom_count", "unit": "count"},
            {"name": "coordinate_cavity_discretization_min", "unit": "angstrom^3"},
            {"name": "coordinate_cavity_discretization_max", "unit": "angstrom^3"},
            {"name": "coordinate_cavity_discretization_standard_error", "unit": "angstrom^3"}
        ],
        "coordinate_source": "fixed_cadence_candidate_confirmation_v1"
    }))?;
    let mut rows = Vec::new();
    let mut run_units = BTreeSet::new();
    let mut engine_hashes = BTreeSet::new();
    let mut topology_hashes = BTreeSet::new();
    let mut phase_evidence = BTreeSet::new();
    let mut evaluable = 0u64;
    let mut not_evaluable = 0u64;
    for (stream_id, load) in fixed_streams {
        if load.manifest.context.stream_id != *stream_id
            || load.manifest.context.candidate_tracker_sha256.as_deref()
                != Some(tracker.tracker_sha256.as_str())
            || load.manifest.context.cadence_steps != tracker.cadence_steps
            || load.manifest.context.start_step != tracker.start_step_by_stream[stream_id]
            || load.manifest.context.end_step_exclusive
                != tracker.end_step_exclusive_by_stream[stream_id]
        {
            bail!("fixed-cadence manifest differs from predeclared tracker");
        }
        run_units.insert((
            load.manifest.context.run_id.clone(),
            load.manifest.context.replica_id.clone(),
            load.manifest.context.replica_seed,
        ));
        engine_hashes.insert(load.manifest.context.selected_engine_sha256.clone());
        topology_hashes.insert(load.manifest.context.topology_sha256.clone());
        for (step_idx, frame) in &load.frames {
            let phase_context = lambda_load
                .phase_context_at(*stream_id, *step_idx)
                .context("fixed-cadence row lacks verified lambda/no-probe phase context")?;
            if phase_context.protocol_phase.phase_ordinal != tracker.stationary_phase_ordinal
                || phase_context.protocol_phase.protocol_state_hash
                    != load.manifest.context.protocol_state_hash
                || phase_context.protocol_phase.phase_schedule_digest_sha256
                    != load.manifest.context.phase_schedule_digest_sha256
                || phase_context.thermodynamic_condition.condition_id
                    != load.manifest.context.thermodynamic_condition_id
                || phase_context
                    .thermodynamic_condition
                    .matched_physical_state_sha256(&phase_context.protocol_phase)?
                    != tracker.matched_physical_state_sha256
            {
                bail!("fixed-cadence row crosses stationary phase or physical condition");
            }
            phase_evidence.insert(phase_context.phase_protocol_sha256.clone());
            let detected = detections.get(&(*stream_id, *step_idx));
            let (ghost_event, observation_authority_sha256, ambiguous) = match detected {
                None => {
                    let absence = canonical_sha256(&serde_json::json!({
                        "schema_kind": "prism_pathb_selected_ghost_absence",
                        "schema_version": 1,
                        "tracker_sha256": tracker.tracker_sha256,
                        "fixed_population_sha256": fixed_population_sha256,
                        "stream_id": stream_id,
                        "step_idx": step_idx,
                        "coordinate_frame_sha256": frame.entry.analysis_coordinate_sha256,
                        "selected_record_count": 0,
                    }))?;
                    (
                        PathBGhostEventStatus::NotEmitted {
                            typed_absence_receipt_sha256: absence.clone(),
                        },
                        absence,
                        false,
                    )
                }
                Some(events) if events.len() == 1 => (
                    PathBGhostEventStatus::Emitted {
                        ghost_lineage_sha256: events[0].0.clone(),
                        exact_component_sha256: events[0].1.clone(),
                    },
                    events[0].2.clone(),
                    false,
                ),
                Some(events) => {
                    let lineages = events
                        .iter()
                        .map(|event| event.0.clone())
                        .collect::<Vec<_>>();
                    let components = events
                        .iter()
                        .map(|event| event.1.clone())
                        .collect::<Vec<_>>();
                    let ambiguity = canonical_sha256(&serde_json::json!({
                        "schema_kind": "prism_pathb_ambiguous_ghost_emissions",
                        "schema_version": 1,
                        "tracker_sha256": tracker.tracker_sha256,
                        "stream_id": stream_id,
                        "step_idx": step_idx,
                        "ghost_lineage_sha256": lineages,
                        "exact_component_sha256": components,
                    }))?;
                    (
                        PathBGhostEventStatus::AmbiguousEmissions {
                            ghost_lineage_sha256: lineages,
                            exact_component_sha256: components,
                        },
                        ambiguity,
                        true,
                    )
                }
            };
            let measurement = if ambiguous {
                not_evaluable += 1;
                PathBMeasurementStatus::NotEvaluable {
                    reason_code: "AMBIGUOUS_MULTIPLE_NATIVE_COMPONENTS_AT_SCHEDULED_STEP"
                        .to_string(),
                    evidence_receipt_sha256: observation_authority_sha256,
                }
            } else {
                let cavity =
                    crate::coordinate_cavity::compute_coordinate_cavity_from_fixed_cadence(
                        frame,
                        &load.manifest,
                        &observation_authority_sha256,
                        &request.body.lane_policy.target_residue_ids,
                        topology,
                        &tracker.cavity_estimator_config,
                    )?;
                let hydration_count = frame
                    .local_shell
                    .iter()
                    .filter(|atom| atom.role == "WATER")
                    .count() as f64;
                let evidence = PathBEvaluableMeasurement {
                    coordinate_frame: cavity.coordinate_frame,
                    local_hydration_shell_sha256: frame
                        .entry
                        .hydration_shell_sha256
                        .clone()
                        .context("confirmation frame lacks hydration-shell identity")?,
                    local_fragment_shell_sha256: frame
                        .entry
                        .fragment_shell_sha256
                        .clone()
                        .context("confirmation frame lacks fragment-shell identity")?,
                    coordinate_cavity_status_sha256: cavity.sidecar.sidecar_sha256.clone(),
                    coordinate_cavity_sidecar: cavity.sidecar.clone(),
                    cavity_volume_a3: cavity.pocket_volume_a3.value,
                    cavity_discretization_min_a3: cavity.sidecar.body.discretization_min_a3,
                    cavity_discretization_max_a3: cavity.sidecar.body.discretization_max_a3,
                    cavity_discretization_standard_error_a3: cavity
                        .sidecar
                        .body
                        .discretization_standard_error_a3,
                    kinetic_feature_schema_sha256: feature_schema_sha256.clone(),
                    kinetic_features: vec![
                        cavity.pocket_volume_a3.value,
                        hydration_count,
                        cavity.sidecar.body.discretization_min_a3,
                        cavity.sidecar.body.discretization_max_a3,
                        cavity.sidecar.body.discretization_standard_error_a3,
                    ],
                };
                evaluable += 1;
                match classify_cavity_discretization_interval(
                    evidence.cavity_discretization_min_a3,
                    evidence.cavity_discretization_max_a3,
                    tracker,
                )? {
                    CavityIntervalClass::Open => PathBMeasurementStatus::ObservedOpen { evidence },
                    CavityIntervalClass::Closed => {
                        PathBMeasurementStatus::ObservedClosed { evidence }
                    }
                    CavityIntervalClass::Indeterminate => {
                        PathBMeasurementStatus::ObservedIndeterminate { evidence }
                    }
                }
            };
            rows.push(PathBSurveillanceRow::materialize(
                tracker,
                *stream_id,
                *step_idx,
                phase_context.protocol_phase.clone(),
                phase_context.thermodynamic_condition.clone(),
                ghost_event,
                measurement,
            )?);
        }
    }
    if run_units.len() != 1 || engine_hashes.len() != 1 || topology_hashes.len() != 1 {
        bail!("fixed-cadence surveillance spans multiple run units, engines, or topologies");
    }
    rows.sort_by_key(|row| (row.stream_id, row.step_idx));
    let observed_open_row_count = rows
        .iter()
        .filter(|row| {
            matches!(
                &row.measurement,
                PathBMeasurementStatus::ObservedOpen { .. }
            )
        })
        .count() as u64;
    let observed_closed_row_count = rows
        .iter()
        .filter(|row| {
            matches!(
                &row.measurement,
                PathBMeasurementStatus::ObservedClosed { .. }
            )
        })
        .count() as u64;
    let observed_indeterminate_row_count = rows
        .iter()
        .filter(|row| {
            matches!(
                &row.measurement,
                PathBMeasurementStatus::ObservedIndeterminate { .. }
            )
        })
        .count() as u64;
    let ghost_emitted_row_count = rows
        .iter()
        .filter(|row| matches!(&row.ghost_event, PathBGhostEventStatus::Emitted { .. }))
        .count() as u64;
    let ghost_ambiguous_row_count = rows
        .iter()
        .filter(|row| {
            matches!(
                &row.ghost_event,
                PathBGhostEventStatus::AmbiguousEmissions { .. }
            )
        })
        .count() as u64;
    let ghost_not_emitted_row_count = rows
        .iter()
        .filter(|row| matches!(&row.ghost_event, PathBGhostEventStatus::NotEmitted { .. }))
        .count() as u64;
    let (run_id, replica_id, replica_seed) = run_units.into_iter().next().unwrap();
    let source_authority_sha256 = canonical_sha256(&(
        fixed_population_sha256,
        lambda_load.population_receipt_sha256.as_str(),
        tracker.tracker_sha256.as_str(),
    ))?;
    PathBConfirmationSurveillanceLedger {
        schema_kind: "prism_pathb_confirmation_surveillance_ledger".to_string(),
        schema_version: "4.0.0".to_string(),
        ledger_sha256: String::new(),
        run_id,
        replica_id,
        replica_seed,
        control_lane: lane,
        engine_sha256: engine_hashes.into_iter().next().unwrap(),
        topology_sha256: topology_hashes.into_iter().next().unwrap(),
        matched_physical_state_sha256: tracker.matched_physical_state_sha256.clone(),
        stationary_phase_ordinal: tracker.stationary_phase_ordinal,
        stationary_hold_certified: true,
        stationarity_evidence_sha256: canonical_sha256(&phase_evidence)?,
        candidate_tracker_sha256: tracker.tracker_sha256.clone(),
        candidate_tracker: tracker.clone(),
        source_authority_sha256,
        cadence_steps: tracker.cadence_steps,
        start_step_by_stream: tracker.start_step_by_stream.clone(),
        end_step_exclusive_by_stream: tracker.end_step_exclusive_by_stream.clone(),
        scheduled_row_count: rows.len() as u64,
        persisted_row_count: rows.len() as u64,
        decoded_row_count: rows.len() as u64,
        evaluable_row_count: evaluable,
        not_evaluable_row_count: not_evaluable,
        observed_open_row_count,
        observed_closed_row_count,
        observed_indeterminate_row_count,
        ghost_emitted_row_count,
        ghost_ambiguous_row_count,
        ghost_not_emitted_row_count,
        complete_time_at_risk: true,
        source_manifest_sha256: fixed_population_sha256.to_string(),
        rows,
    }
    .materialize()
}

fn validate_fixed_cadence_timebase_binding(
    request: &NativePathBRuntimeRequest,
    fixed_streams: &BTreeMap<u16, crate::fixed_cadence_coordinate::FixedCadenceCoordinateLoad>,
) -> Result<()> {
    request.verify()?;
    if fixed_streams.is_empty() {
        bail!("fixed-cadence population is empty");
    }
    let timebase = &request.body.timebase_context;
    timebase.verify()?;
    for (stream_id, load) in fixed_streams {
        let context = &load.manifest.context;
        context.verify()?;
        if context.stream_id != *stream_id
            || context.native_pathb_runtime_request_sha256 != request.request_sha256
            || context.integration_timebase_context_sha256 != timebase.context_sha256
            || context.integration_timebase_authority_sha256 != timebase.integration_timebase_sha256
            || context.integration_timestep_femtoseconds.to_bits()
                != timebase.integration_timestep_femtoseconds.to_bits()
            || context.candidate_lag_steps != request.body.pathb_config.candidate_lag_steps
            || context.cadence_steps != request.body.lane_policy.fixed_cadence_steps
        {
            bail!(
                "fixed-cadence stream {} differs from admitted request/timebase/lag authority",
                stream_id
            );
        }
    }
    Ok(())
}

/// Independent-seed uncertainty for one neutral metastable state's declared
/// coordinate-derived cavity observable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateCavityObservableSummary {
    pub state_id: String,
    pub observation_count: usize,
    pub independent_seed_count: usize,
    pub mean_a3: f64,
    pub standard_error_a3: f64,
    pub confidence_interval_lower_a3: f64,
    pub confidence_interval_upper_a3: f64,
}

/// Exact coordinate provenance for one physical lane; indices remain local.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeOpenBasinLaneIdentity {
    pub topology_sha256: String,
    pub atom_residue_mapping_sha256: String,
}

/// Hash-bound physical definition mapping neutral MSM states to an open basin.
/// It is separate from the neutral model and never inferred from Ghost AABBs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NativeOpenBasinBinding {
    pub schema_kind: String,
    pub schema_version: String,
    pub binding_sha256: String,
    pub pathb_model_sha256: String,
    pub observable_id: String,
    pub measurement_method_id: String,
    pub estimator_sha256: String,
    pub topology_sha256: Option<String>,
    pub atom_residue_mapping_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub topology_mapping_by_lane: BTreeMap<ControlLane, NativeOpenBasinLaneIdentity>,
    pub frame_population_sha256: String,
    pub familywise_confidence_level: f64,
    pub selection_rule: String,
    pub open_state_ids: Vec<String>,
    pub state_summaries: Vec<StateCavityObservableSummary>,
}

impl NativeOpenBasinBinding {
    /// Recompute the physical open-basin definition and validate its uncertainty
    /// declarations. The neutral MSM itself never defines "open".
    pub fn verify(&self) -> Result<()> {
        if self.schema_kind != "prism_native_pathb_open_basin_binding"
            || !matches!(self.schema_version.as_str(), "1.0.0" | "2.0.0")
            || self.observable_id != "coordinate_cavity_volume_a3"
            || self.selection_rule
                != "highest_seed_balanced_state_mean_with_bonferroni_t_interval_separation"
            || self.open_state_ids.is_empty()
            || self.state_summaries.len() < 2
            || self.familywise_confidence_level.to_bits() != 0.95f64.to_bits()
        {
            bail!("invalid native Path-B open-basin contract");
        }
        for (value, field) in [
            (&self.binding_sha256, "binding_sha256"),
            (&self.pathb_model_sha256, "pathb_model_sha256"),
            (&self.estimator_sha256, "estimator_sha256"),
            (&self.frame_population_sha256, "frame_population_sha256"),
        ] {
            require_sha256(value, field)?;
        }
        if self.schema_version == "1.0.0" {
            if !self.topology_mapping_by_lane.is_empty() {
                bail!("legacy open-basin binding contains v2 lane provenance");
            }
            require_sha256(self.topology_sha256.as_deref().context("legacy basin lacks topology")?, "topology_sha256")?;
            require_sha256(self.atom_residue_mapping_sha256.as_deref().context("legacy basin lacks mapping")?, "atom_residue_mapping_sha256")?;
        } else {
            if self.topology_sha256.is_some() || self.atom_residue_mapping_sha256.is_some()
                || self.topology_mapping_by_lane.keys().copied().collect::<BTreeSet<_>>() != BTreeSet::from([
                    ControlLane::NoProbe, ControlLane::HypothesisFragment, ControlLane::OrthogonalChemicalControl]) {
                bail!("heterogeneous open-basin binding requires three exact lane identities and no global scalar");
            }
            let mut mappings_by_topology = BTreeMap::new();
            for identity in self.topology_mapping_by_lane.values() {
                require_sha256(&identity.topology_sha256, "lane topology")?;
                require_sha256(&identity.atom_residue_mapping_sha256, "lane atom mapping")?;
                if mappings_by_topology.insert(&identity.topology_sha256, &identity.atom_residue_mapping_sha256)
                    .is_some_and(|previous| previous != &identity.atom_residue_mapping_sha256) {
                    bail!("one exact topology has conflicting atom mappings");
                }
            }
        }
        if self.measurement_method_id.trim().is_empty() {
            bail!("open-basin measurement method is empty");
        }
        let mut seen = BTreeSet::new();
        for summary in &self.state_summaries {
            if summary.state_id.trim().is_empty()
                || !seen.insert(summary.state_id.clone())
                || summary.observation_count == 0
                || summary.independent_seed_count < 5
                || [
                    summary.mean_a3,
                    summary.standard_error_a3,
                    summary.confidence_interval_lower_a3,
                    summary.confidence_interval_upper_a3,
                ]
                .iter()
                .any(|value| !value.is_finite())
                || summary.mean_a3 < 0.0
                || summary.standard_error_a3 < 0.0
                || summary.confidence_interval_lower_a3 > summary.confidence_interval_upper_a3
            {
                bail!("invalid open-basin state uncertainty summary");
            }
        }
        if self.state_summaries.windows(2).any(|pair| {
            pair[0]
                .mean_a3
                .total_cmp(&pair[1].mean_a3)
                .then_with(|| pair[0].state_id.cmp(&pair[1].state_id))
                .is_gt()
        }) {
            bail!("open-basin state summaries are not canonically ordered");
        }
        if self
            .open_state_ids
            .iter()
            .any(|state| !seen.contains(state))
            || self
                .open_state_ids
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            bail!("open-state identities are undeclared, duplicate, or unsorted");
        }
        let excluded_upper = self
            .state_summaries
            .iter()
            .filter(|summary| !self.open_state_ids.contains(&summary.state_id))
            .map(|summary| summary.confidence_interval_upper_a3)
            .fold(f64::NEG_INFINITY, f64::max);
        let selected_lower = self
            .state_summaries
            .iter()
            .filter(|summary| self.open_state_ids.contains(&summary.state_id))
            .map(|summary| summary.confidence_interval_lower_a3)
            .fold(f64::INFINITY, f64::min);
        if !excluded_upper.is_finite()
            || !selected_lower.is_finite()
            || selected_lower <= excluded_upper
        {
            bail!("open basin is not uncertainty-separated from excluded states");
        }
        let body = NativeOpenBasinBindingBody {
            pathb_model_sha256: &self.pathb_model_sha256,
            observable_id: "coordinate_cavity_volume_a3",
            measurement_method_id: &self.measurement_method_id,
            estimator_sha256: &self.estimator_sha256,
            topology_sha256: &self.topology_sha256,
            atom_residue_mapping_sha256: &self.atom_residue_mapping_sha256,
            topology_mapping_by_lane: &self.topology_mapping_by_lane,
            frame_population_sha256: &self.frame_population_sha256,
            familywise_confidence_level: self.familywise_confidence_level,
            selection_rule:
                "highest_seed_balanced_state_mean_with_bonferroni_t_interval_separation",
            open_state_ids: &self.open_state_ids,
            state_summaries: &self.state_summaries,
        };
        if canonical_sha256(&body)? != self.binding_sha256 {
            bail!("open-basin binding content hash mismatch");
        }
        Ok(())
    }
}

#[derive(Serialize)]
struct NativeOpenBasinBindingBody<'a> {
    pathb_model_sha256: &'a str,
    observable_id: &'static str,
    measurement_method_id: &'a str,
    estimator_sha256: &'a str,
    topology_sha256: &'a Option<String>,
    atom_residue_mapping_sha256: &'a Option<String>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    topology_mapping_by_lane: &'a BTreeMap<ControlLane, NativeOpenBasinLaneIdentity>,
    frame_population_sha256: &'a str,
    familywise_confidence_level: f64,
    selection_rule: &'static str,
    open_state_ids: &'a [String],
    state_summaries: &'a [StateCavityObservableSummary],
}

/// Ghost-side record before the native model defines its open-state estimand.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GhostFiberMeasurementObservation {
    pub site_key: String,
    pub control_lane: ControlLane,
    pub ghost: GhostByteLineage,
    pub protocol_phase: ProtocolPhaseBinding,
    pub thermodynamic_condition: ThermodynamicCondition,
    pub structural_measurements: StructuralMeasurementSet,
    pub uncertainty: SamplingUncertainty,
    #[serde(default)]
    pub evidence_artifact_sha256: BTreeMap<String, String>,
}

/// Materialize one exact coordinate-derived structural observation per
/// selected Ghost occurrence. Phase and thermodynamic state are taken from
/// the verified lambda/no-probe journal context covering the same stream and
/// step; static topology defaults and lattice AABB volumes are forbidden.
#[cfg(any(feature = "gpu", feature = "cpu-analysis"))]
pub fn materialize_verified_ghost_measurements(
    request: &NativePathBRuntimeRequest,
    selected: &VerifiedGhostSitePathBInputs,
    full_load: &crate::ghost_phase_lattice::GhostSegmentLatticeLoad,
    coordinate_load: &crate::ghost_coordinate_segment::GhostCoordinateSegmentLoad,
    coordinate_join: &crate::ghost_coordinate_segment::GhostCoordinateLineageJoin,
    lambda_load: &crate::lambda_fragment::LambdaEventJournalLoad,
    topology: &crate::input::PrismPrepTopology,
) -> Result<Vec<GhostFiberMeasurementObservation>> {
    request.verify()?;
    selected.selection.verify()?;
    if selected.nodes.len() != selected.lineages.len()
        || full_load.nodes.len() != full_load.record_lineage.len()
        || !coordinate_load.equality_pass
        || !coordinate_join.conservation.equality_pass
    {
        bail!("Ghost measurement source populations are not exactly conserved");
    }
    let mut output = Vec::with_capacity(selected.lineages.len());
    for (node, lineage) in selected.nodes.iter().zip(&selected.lineages) {
        let source = full_load
            .record_lineage
            .get(usize::try_from(lineage.lattice_node_index)?)
            .context("selected Ghost measurement lacks full byte-lineage row")?;
        if source.record_sha256 != lineage.record_sha256
            || source.stream_id != lineage.key.stream_id
            || source.step_idx != lineage.key.step_idx
            || source.site_id != lineage.key.site_id
        {
            bail!("selected Ghost measurement differs from full byte lineage");
        }
        let key = crate::ghost_coordinate_segment::GhostCoordinateKey {
            stream_id: lineage.key.stream_id,
            step_idx: lineage.key.step_idx,
            ghost_site_id: lineage.key.site_id,
            ghost_record_sha256: lineage.record_sha256.clone(),
        };
        let coordinate = coordinate_join.by_key.get(&key).context(
            "EVIDENCE_NOT_FOUND: selected Ghost occurrence lacks exact joined coordinates",
        )?;
        let coordinate_context = coordinate_load
            .contexts
            .get(&lineage.key.stream_id)
            .context(
                "EVIDENCE_NOT_FOUND: selected Ghost occurrence lacks verified coordinate context",
            )?;
        let context = lambda_load
            .phase_context_at(lineage.key.stream_id, lineage.key.step_idx)
            .context("selected Ghost occurrence lacks lambda/no-probe phase context")?;
        let lane = *request
            .body
            .lane_policy
            .stream_lanes
            .get(&lineage.key.stream_id)
            .context("selected Ghost stream is absent from lane policy")?;
        if context.protocol_phase.phase_ordinal != lineage.protocol_phase
            || context.protocol_phase.protocol_name != lineage.protocol_name
            || context.protocol_phase.protocol_state_hash != lineage.protocol_state_hash
            || context.protocol_phase.phase_schedule_digest_sha256
                != lineage.phase_schedule_digest_sha256
            || context.thermodynamic_condition.condition_id != lineage.condition_id
        {
            bail!("lambda/no-probe context differs from exact Ghost phase lineage");
        }
        let contact_residue_ids = source
            .cluster_residue_membership
            .cluster_residue_spike_counts
            .iter()
            .map(|row| row.residue_id)
            .collect::<Vec<_>>();
        let cavity = crate::coordinate_cavity::compute_coordinate_cavity_from_ghost_join(
            coordinate,
            coordinate_context,
            [
                f64::from(node.centroid_xyz[0]),
                f64::from(node.centroid_xyz[1]),
                f64::from(node.centroid_xyz[2]),
            ],
            &contact_residue_ids,
            topology,
            &request.body.lane_policy.coordinate_cavity_config,
        )?;
        let uncertainty = SamplingUncertainty {
            effective_sample_size: 1.0,
            integrated_autocorrelation_time_frames: 0.0,
            block_count: 1,
            standard_error: cavity.sidecar.body.discretization_standard_error_a3,
            confidence_interval_lower: cavity.sidecar.body.discretization_min_a3,
            confidence_interval_upper: cavity.sidecar.body.discretization_max_a3,
            replica_seed: lineage.replica_seed,
            replica_id: lineage.replica_id.clone(),
        };
        uncertainty.validate()?;
        let cavity_sidecar_sha256 = cavity.sidecar.sidecar_sha256.clone();
        let physical_lining_sha256 = cavity.sidecar.body.physical_lining.evidence_sha256.clone();
        output.push(GhostFiberMeasurementObservation {
            site_key: request.body.lane_policy.site_key.clone(),
            control_lane: lane,
            ghost: lineage.clone(),
            protocol_phase: context.protocol_phase.clone(),
            thermodynamic_condition: context.thermodynamic_condition.clone(),
            structural_measurements: StructuralMeasurementSet {
                coordinate_frame: cavity.coordinate_frame,
                coordinate_cavity_sidecar: cavity.sidecar,
                pocket_volume_a3: Some(cavity.pocket_volume_a3),
                pocket_depth_a: None,
                solvent_accessible_area_a2: None,
                delta_volume_a3: None,
                delta_hydration: None,
                fragment_contact_fraction: None,
            },
            uncertainty,
            evidence_artifact_sha256: BTreeMap::from([
                (
                    "ghost_coordinate_join_conservation".to_string(),
                    coordinate_join.conservation.receipt_sha256.clone(),
                ),
                (
                    "lambda_journal_population".to_string(),
                    lambda_load.population_receipt_sha256.clone(),
                ),
                (
                    "coordinate_cavity_sidecar".to_string(),
                    cavity_sidecar_sha256,
                ),
                ("physical_cavity_lining".to_string(), physical_lining_sha256),
            ]),
        });
    }
    output.sort_by_key(|row| row.ghost.key);
    Ok(output)
}

/// Fail-closed adapter output from the verified segmented Ghost loader.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerifiedGhostPathBInputs {
    pub lineages: Vec<GhostByteLineage>,
    pub conservation: GhostLatticeConservation,
}

/// Lossless site-selection receipt. The full source population remains
/// explicit while the admitted Path-B population records the exact
/// target-residue overlap rule applied to RichSpike cluster membership.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GhostSiteSelectionReceipt {
    pub schema_kind: String,
    pub schema_version: String,
    pub selection_sha256: String,
    pub source_lattice_manifest_sha256: String,
    pub site_key: String,
    pub selection_mode: NativePathBSelectionMode,
    pub target_residue_ids: Vec<u32>,
    pub minimum_target_residue_overlap_count: usize,
    pub minimum_target_residue_overlap_fraction: f64,
    pub source_produced: u64,
    pub source_persisted: u64,
    pub source_decoded: u64,
    pub source_lattice_nodes: u64,
    pub selected_records: u64,
    pub selected_by_stream: BTreeMap<u16, u64>,
    pub selected_ghost_lineage_sha256: Vec<String>,
    pub membership_semantics: String,
}

#[derive(Serialize)]
struct GhostSiteSelectionReceiptBody<'a> {
    source_lattice_manifest_sha256: &'a str,
    site_key: &'a str,
    selection_mode: NativePathBSelectionMode,
    target_residue_ids: &'a [u32],
    minimum_target_residue_overlap_count: usize,
    minimum_target_residue_overlap_fraction: f64,
    source_produced: u64,
    source_persisted: u64,
    source_decoded: u64,
    source_lattice_nodes: u64,
    selected_records: u64,
    selected_by_stream: &'a BTreeMap<u16, u64>,
    selected_ghost_lineage_sha256: &'a [String],
    membership_semantics: &'a str,
}

impl GhostSiteSelectionReceipt {
    /// Verify full-source conservation, selected population identity and the
    /// hash of the exact predeclared selection rule.
    pub fn verify(&self) -> Result<()> {
        let selection_rule_valid = match self.selection_mode {
            NativePathBSelectionMode::WholeLatticeDiscovery => {
                self.membership_semantics == "whole_authenticated_lattice_no_residue_prefilter"
                    && self.target_residue_ids.is_empty()
                    && self.minimum_target_residue_overlap_count == 0
                    && self.minimum_target_residue_overlap_fraction.to_bits() == 0.0f64.to_bits()
            }
            NativePathBSelectionMode::TargetResidueHypothesis => {
                self.membership_semantics
                    == "richspike_cluster_residue_membership_not_cavity_lining"
                    && !self.target_residue_ids.is_empty()
                    && self.minimum_target_residue_overlap_count > 0
                    && self.minimum_target_residue_overlap_count <= self.target_residue_ids.len()
                    && self.minimum_target_residue_overlap_fraction.is_finite()
                    && (0.0..=1.0).contains(&self.minimum_target_residue_overlap_fraction)
                    && self.minimum_target_residue_overlap_fraction > 0.0
            }
        };
        if self.schema_kind != "prism_ghost_site_selection_receipt"
            || self.schema_version != "2.0.0"
            || !selection_rule_valid
            || self.source_produced != self.source_persisted
            || self.source_persisted != self.source_decoded
            || self.source_decoded != self.source_lattice_nodes
            || self.selected_records != self.selected_ghost_lineage_sha256.len() as u64
            || self.selected_records != self.selected_by_stream.values().sum::<u64>()
            || self
                .target_residue_ids
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || self
                .selected_ghost_lineage_sha256
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            bail!("invalid exact Ghost site-selection receipt");
        }
        if self.selection_mode == NativePathBSelectionMode::WholeLatticeDiscovery {
            if self.source_produced == 0 || self.selected_records == 0 {
                bail!("whole-lattice discovery requires a non-empty authenticated lattice");
            }
            if self.selected_records != self.source_lattice_nodes {
                bail!("whole-lattice discovery cannot discard source lattice nodes");
            }
        }
        require_sha256(
            &self.source_lattice_manifest_sha256,
            "source_lattice_manifest_sha256",
        )?;
        require_sha256(&self.selection_sha256, "selection_sha256")?;
        for digest in &self.selected_ghost_lineage_sha256 {
            require_sha256(digest, "selected_ghost_lineage_sha256")?;
        }
        let body = GhostSiteSelectionReceiptBody {
            source_lattice_manifest_sha256: &self.source_lattice_manifest_sha256,
            site_key: &self.site_key,
            selection_mode: self.selection_mode,
            target_residue_ids: &self.target_residue_ids,
            minimum_target_residue_overlap_count: self.minimum_target_residue_overlap_count,
            minimum_target_residue_overlap_fraction: self.minimum_target_residue_overlap_fraction,
            source_produced: self.source_produced,
            source_persisted: self.source_persisted,
            source_decoded: self.source_decoded,
            source_lattice_nodes: self.source_lattice_nodes,
            selected_records: self.selected_records,
            selected_by_stream: &self.selected_by_stream,
            selected_ghost_lineage_sha256: &self.selected_ghost_lineage_sha256,
            membership_semantics: &self.membership_semantics,
        };
        if canonical_sha256(&body)? != self.selection_sha256 {
            bail!("Ghost site-selection receipt content hash mismatch");
        }
        Ok(())
    }
}

/// Exact site-selected native lattice population. Original global lattice
/// indices and byte lineages are preserved; no selected row is reconstructed.
#[cfg(any(feature = "gpu", feature = "cpu-analysis"))]
#[derive(Debug, Clone)]
pub struct VerifiedGhostSitePathBInputs {
    pub nodes: Vec<crate::ghost_phase_lattice::GhostPhaseLatticeNode>,
    pub lineages: Vec<GhostByteLineage>,
    pub conservation: GhostLatticeConservation,
    pub selection: GhostSiteSelectionReceipt,
}

/// Shared exact byte-lineage projection for verified live loads and pinned
/// saved native measurement records. Callers supply the authenticated source
/// row and its bound protocol name; no identities are derived from rank or ROI.
#[cfg(any(feature = "gpu", feature = "cpu-analysis"))]
pub(crate) fn adapt_recorded_ghost_byte_lineage(
    source: &crate::ghost_phase_lattice::GhostRecordByteLineage,
    protocol_name: &str,
) -> Result<GhostByteLineage> {
    if source.schema_version != 1 {
        bail!("unsupported authoritative Ghost byte-lineage schema");
    }
    let lineage = GhostByteLineage {
        key: crate::chemical_state_fiber::GhostLineageKey {
            stream_id: source.stream_id,
            frame_idx: source.frame_idx,
            step_idx: source.step_idx,
            site_id: source.site_id,
            record_index: source.stream_record_ordinal,
        },
        run_id: source.run_id.clone(),
        replica_id: source.replica_id.clone(),
        replica_seed: source.replica_seed,
        stream_seed: source.stream_seed,
        topology_sha256: source.topology_sha256.clone(),
        record_abi_sha256: source.record_abi_sha256.clone(),
        record_source_sha256: source.record_source_sha256.clone(),
        segment_context_digest_sha256: source.segment_context_digest_sha256.clone(),
        segment_manifest_sha256: source.segment_manifest_sha256.clone(),
        segment_sequence: source.segment_ordinal,
        segment_sha256: source.segment_file_sha256.clone(),
        segment_file: source.segment_file.clone(),
        byte_offset: source.payload_byte_offset,
        byte_length: source.byte_length,
        record_sha256: source.record_sha256.clone(),
        lattice_node_index: source.global_lattice_index,
        engine_sha256: source.selected_engine_sha256.clone(),
        condition_id: source.condition_id.clone(),
        protocol_name: protocol_name.to_string(),
        protocol_phase: u16::from(source.protocol_phase),
        phase_schedule_digest_sha256: source.phase_schedule_digest_sha256.clone(),
        protocol_state_hash: source.protocol_state_hash.clone(),
    };
    lineage.validate()?;
    Ok(lineage)
}

/// Consume authoritative loader rows directly. No record, seed, stream,
/// phase, byte offset, or manifest identity is reconstructed downstream.
#[cfg(any(feature = "gpu", feature = "cpu-analysis"))]
pub fn adapt_verified_ghost_load_for_pathb(
    load: &crate::ghost_phase_lattice::GhostSegmentLatticeLoad,
    committed_lattice_manifest_sha256: &str,
) -> Result<VerifiedGhostPathBInputs> {
    require_sha256(
        committed_lattice_manifest_sha256,
        "committed_lattice_manifest_sha256",
    )?;
    let count = load.nodes.len() as u64;
    if !load.equality_pass
        || !load.phase_protocols_bound_by_stream
        || load.produced_records != count
        || load.persisted_records != count
        || load.decoded_records != count
        || load.lattice_node_lineage_count != count
        || load.record_lineage.len() != load.nodes.len()
        || load.streams.is_empty()
        || load.streams.iter().any(|stream| !stream.equality_pass)
    {
        bail!("verified Ghost load failed exact population or phase-binding conservation");
    }
    let stream_receipts = load
        .streams
        .iter()
        .map(|stream| (stream.stream_id, stream))
        .collect::<BTreeMap<_, _>>();
    if stream_receipts.len() != load.streams.len() {
        bail!("verified Ghost load contains duplicate stream receipts");
    }
    let mut lineages = Vec::with_capacity(load.record_lineage.len());
    for (index, (source, node)) in load.record_lineage.iter().zip(&load.nodes).enumerate() {
        if source.schema_version != 1
            || source.global_lattice_index != index as u64
            || source.stream_id != node.stream_id
            || source.frame_idx != node.frame_idx
            || source.step_idx != node.step_idx
            || source.site_id != node.site_id
            || u16::from(source.protocol_phase) != u16::from(node.protocol_phase)
            || source.selected_engine_sha256.trim().is_empty()
        {
            bail!("authoritative Ghost lineage row disagrees with its lattice node");
        }
        let stream = stream_receipts
            .get(&source.stream_id)
            .context("Ghost lineage has no stream conservation receipt")?;
        if source.condition_id != stream.condition_id
            || source.phase_schedule_digest_sha256 != stream.phase_schedule_digest_sha256
            || source.protocol_state_hash != stream.protocol_state_hash
            || source.segment_manifest_sha256 != stream.segment_manifest_sha256
            || source.segment_context_digest_sha256 != stream.segment_context_digest_sha256
        {
            bail!("Ghost lineage disagrees with its authenticated stream context");
        }
        let lineage = adapt_recorded_ghost_byte_lineage(source, &stream.protocol_name)?;
        lineages.push(lineage);
    }
    let conservation = GhostLatticeConservation {
        produced: load.produced_records,
        persisted: load.persisted_records,
        decoded: load.decoded_records,
        lattice_nodes: load.lattice_node_lineage_count,
        lattice_manifest_sha256: committed_lattice_manifest_sha256.to_string(),
    };
    conservation.validate()?;
    Ok(VerifiedGhostPathBInputs {
        lineages,
        conservation,
    })
}

/// Select one predeclared site from an already verified Ghost population by
/// exact RichSpike cluster-residue membership. This function deliberately
/// preserves the distinction between cluster membership and coordinate-
/// derived cavity-lining/contact evidence.
#[cfg(any(feature = "gpu", feature = "cpu-analysis"))]
pub fn select_verified_ghost_site_for_pathb(
    load: &crate::ghost_phase_lattice::GhostSegmentLatticeLoad,
    admitted: &VerifiedGhostPathBInputs,
    policy: &NativePathBLanePolicy,
) -> Result<VerifiedGhostSitePathBInputs> {
    admitted.conservation.validate()?;
    if load.nodes.len() != admitted.lineages.len()
        || load.record_lineage.len() != admitted.lineages.len()
        || load.nodes.len() as u64 != admitted.conservation.lattice_nodes
    {
        bail!("site selection source population is not the admitted Ghost population");
    }
    let target = policy
        .target_residue_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    match policy.selection_mode {
        NativePathBSelectionMode::WholeLatticeDiscovery
            if target.is_empty()
                && policy.minimum_target_residue_overlap_count == 0
                && policy.minimum_target_residue_overlap_fraction.to_bits() == 0.0f64.to_bits() => {
        }
        NativePathBSelectionMode::TargetResidueHypothesis
            if target.len() == policy.target_residue_ids.len()
                && !target.is_empty()
                && policy.minimum_target_residue_overlap_count > 0
                && policy.minimum_target_residue_overlap_count <= target.len()
                && policy.minimum_target_residue_overlap_fraction.is_finite()
                && (0.0..=1.0).contains(&policy.minimum_target_residue_overlap_fraction)
                && policy.minimum_target_residue_overlap_fraction > 0.0 => {}
        _ => bail!("invalid predeclared Ghost lattice selection rule"),
    }
    let required_streams = policy
        .required_stream_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut nodes = Vec::new();
    let mut lineages = Vec::new();
    // Materialize zeroes for every requested stream. In target-confirmation
    // mode these are scientifically meaningful no-detection intervals whose
    // fixed-cadence coordinate surveillance remains authoritative. They must
    // not disappear merely because no Ghost event was emitted.
    let mut selected_by_stream = required_streams
        .iter()
        .copied()
        .map(|stream_id| (stream_id, 0_u64))
        .collect::<BTreeMap<_, _>>();
    let mut selected_hashes = Vec::new();
    for ((node, source), lineage) in load
        .nodes
        .iter()
        .zip(&load.record_lineage)
        .zip(&admitted.lineages)
    {
        if node.stream_id != lineage.key.stream_id
            || node.step_idx != lineage.key.step_idx
            || node.site_id != lineage.key.site_id
            || source.record_sha256 != lineage.record_sha256
            || source.cluster_residue_membership.membership_semantics
                != "richspike_cluster_residue_membership_not_cavity_lining"
        {
            bail!("Ghost site-selection rows disagree with authenticated node lineage");
        }
        if !required_streams.contains(&node.stream_id) {
            continue;
        }
        let selected = match policy.selection_mode {
            NativePathBSelectionMode::WholeLatticeDiscovery => true,
            NativePathBSelectionMode::TargetResidueHypothesis => {
                let members = source
                    .cluster_residue_membership
                    .cluster_residue_spike_counts
                    .iter()
                    .map(|row| row.residue_id)
                    .collect::<BTreeSet<_>>();
                let overlap = target.intersection(&members).count();
                let fraction = overlap as f64 / target.len() as f64;
                overlap >= policy.minimum_target_residue_overlap_count
                    && fraction + f64::EPSILON >= policy.minimum_target_residue_overlap_fraction
            }
        };
        if selected {
            nodes.push(*node);
            lineages.push(lineage.clone());
            *selected_by_stream
                .get_mut(&node.stream_id)
                .context("selected Ghost node was not in the predeclared stream population")? += 1;
            selected_hashes.push(lineage.lineage_sha256()?);
        }
    }
    if selected_by_stream.keys().copied().collect::<BTreeSet<_>>() != required_streams {
        bail!("site selection did not produce evidence in every required stream");
    }
    selected_hashes.sort();
    if selected_hashes.windows(2).any(|pair| pair[0] == pair[1]) {
        bail!("site selection contains duplicate Ghost record identities");
    }
    let mut selection = GhostSiteSelectionReceipt {
        schema_kind: "prism_ghost_site_selection_receipt".to_string(),
        schema_version: "2.0.0".to_string(),
        selection_sha256: String::new(),
        source_lattice_manifest_sha256: admitted.conservation.lattice_manifest_sha256.clone(),
        site_key: policy.site_key.clone(),
        selection_mode: policy.selection_mode,
        target_residue_ids: policy.target_residue_ids.clone(),
        minimum_target_residue_overlap_count: policy.minimum_target_residue_overlap_count,
        minimum_target_residue_overlap_fraction: policy.minimum_target_residue_overlap_fraction,
        source_produced: admitted.conservation.produced,
        source_persisted: admitted.conservation.persisted,
        source_decoded: admitted.conservation.decoded,
        source_lattice_nodes: admitted.conservation.lattice_nodes,
        selected_records: selected_hashes.len() as u64,
        selected_by_stream,
        selected_ghost_lineage_sha256: selected_hashes,
        membership_semantics: match policy.selection_mode {
            NativePathBSelectionMode::WholeLatticeDiscovery => {
                "whole_authenticated_lattice_no_residue_prefilter"
            }
            NativePathBSelectionMode::TargetResidueHypothesis => {
                "richspike_cluster_residue_membership_not_cavity_lining"
            }
        }
        .to_string(),
    };
    selection.selection_sha256 = canonical_sha256(&GhostSiteSelectionReceiptBody {
        source_lattice_manifest_sha256: &selection.source_lattice_manifest_sha256,
        site_key: &selection.site_key,
        selection_mode: selection.selection_mode,
        target_residue_ids: &selection.target_residue_ids,
        minimum_target_residue_overlap_count: selection.minimum_target_residue_overlap_count,
        minimum_target_residue_overlap_fraction: selection.minimum_target_residue_overlap_fraction,
        source_produced: selection.source_produced,
        source_persisted: selection.source_persisted,
        source_decoded: selection.source_decoded,
        source_lattice_nodes: selection.source_lattice_nodes,
        selected_records: selection.selected_records,
        selected_by_stream: &selection.selected_by_stream,
        selected_ghost_lineage_sha256: &selection.selected_ghost_lineage_sha256,
        membership_semantics: &selection.membership_semantics,
    })?;
    selection.verify()?;
    let selected_count = nodes.len() as u64;
    let conservation = GhostLatticeConservation {
        produced: selected_count,
        persisted: selected_count,
        decoded: selected_count,
        lattice_nodes: selected_count,
        lattice_manifest_sha256: selection.source_lattice_manifest_sha256.clone(),
    };
    conservation.validate()?;
    Ok(VerifiedGhostSitePathBInputs {
        nodes,
        lineages,
        conservation,
        selection,
    })
}

#[cfg(any(feature = "gpu", feature = "cpu-analysis"))]
#[derive(Clone, Serialize)]
struct NativePathBLaneMaterializationInputs {
    runtime_request: NativePathBRuntimeRequest,
    site_selection: GhostSiteSelectionReceipt,
    source_authority: NativePathBLaneSourceAuthority,
    lattice_observations: Vec<NativePathBLatticeObservation>,
    exact_connectivity: NativePathBConnectivityAuthority,
    component_descriptors: Vec<NativeComponentDescriptor>,
    physical_lining_descriptor_population:
        crate::component_correspondence::PhysicalLiningDescriptorPopulation,
    confirmation_surveillance: Option<PathBConfirmationSurveillanceLedger>,
    ghost_measurements: Vec<GhostFiberMeasurementObservation>,
    lambda_ghost_endpoint_join: LambdaGhostEndpointJoinMaterialization,
    fragment_observations: Vec<FragmentStateObservation>,
}

#[cfg(any(feature = "gpu", feature = "cpu-analysis"))]
impl NativePathBLaneMaterializationInputs {
    fn materialize(
        &self,
        lambda_ghost_endpoint_join_typed_artifact: TypedEvidenceArtifact,
    ) -> Result<NativePathBLaneEvidence> {
        NativePathBLaneEvidence::materialize(NativePathBLaneEvidenceBody {
            runtime_request: self.runtime_request.clone(),
            site_selection: self.site_selection.clone(),
            source_authority: self.source_authority.clone(),
            lattice_observations: self.lattice_observations.clone(),
            exact_connectivity: self.exact_connectivity.clone(),
            component_descriptors: self.component_descriptors.clone(),
            physical_lining_descriptor_population: self
                .physical_lining_descriptor_population
                .clone(),
            confirmation_surveillance: self.confirmation_surveillance.clone(),
            ghost_measurements: self.ghost_measurements.clone(),
            lambda_ghost_endpoint_join: self.lambda_ghost_endpoint_join.clone(),
            lambda_ghost_endpoint_join_typed_artifact,
            fragment_observations: self.fragment_observations.clone(),
            materialized_pathb_state_count: 0,
            materialized_fiber_count: 0,
            matched_control_classification_count: 0,
            candidate_append_count: 0,
        })
    }

    fn transaction_id_sha256(&self, canonical_preview: &NativePathBLaneEvidence) -> Result<String> {
        canonical_sha256(&(
            "native_pathb_lane_fail_atomic_transaction_v1",
            canonical_preview.lane_evidence_sha256.as_str(),
            self.lambda_ghost_endpoint_join
                .materialization_sha256
                .as_str(),
        ))
    }
}

/// Production closeout entry point used by `nhs_rt_full`. It consumes only
/// verified segmented Ghost bytes/nodes/edges, exact coordinate joins,
/// fixed-cadence confirmation frames, and exact lambda/no-probe journals.
/// Legacy binding-site JSON is neither accepted nor inspected.
#[cfg(any(feature = "gpu", feature = "cpu-analysis"))]
pub fn materialize_native_pathb_runtime_lane(
    output_dir: &Path,
    request: NativePathBRuntimeRequest,
    full_load: &crate::ghost_phase_lattice::GhostSegmentLatticeLoad,
    committed_lattice_manifest_sha256: &str,
    edge_artifact: &crate::ghost_phase_lattice::GhostLatticeAcceptedEdgeArtifact,
    coordinate_load: &crate::ghost_coordinate_segment::GhostCoordinateSegmentLoad,
    coordinate_join: &crate::ghost_coordinate_segment::GhostCoordinateLineageJoin,
    fixed_population: Option<
        &crate::fixed_cadence_coordinate::FixedCadenceCoordinatePopulationLoad,
    >,
    lambda_load: &crate::lambda_fragment::LambdaEventJournalLoad,
    topology: &crate::input::PrismPrepTopology,
    runtime_provenance: &RuntimeProvenance,
    source_stage: &str,
) -> Result<NativePathBLaneEvidenceEmission> {
    request.verify()?;
    let admitted =
        adapt_verified_ghost_load_for_pathb(full_load, committed_lattice_manifest_sha256)?;
    let selected =
        select_verified_ghost_site_for_pathb(full_load, &admitted, &request.body.lane_policy)?;
    let exact_connectivity = crate::pathb_chemical_state::build_exact_pathb_connectivity(
        &full_load.nodes,
        &selected.lineages,
        edge_artifact,
    )?;
    let lane = request
        .body
        .lane_policy
        .stream_lanes
        .values()
        .next()
        .copied()
        .context("native Path-B runtime request lacks physical lane")?;
    let ghost_measurements = materialize_verified_ghost_measurements(
        &request,
        &selected,
        full_load,
        coordinate_load,
        coordinate_join,
        lambda_load,
        topology,
    )?;
    let physical_lining_occurrences = ghost_measurements
        .iter()
        .map(|measurement| {
            let sidecar = &measurement
                .structural_measurements
                .coordinate_cavity_sidecar;
            crate::component_correspondence::PhysicalLiningOccurrenceEvidence::materialize(
                &measurement.ghost,
                sidecar.sidecar_sha256.clone(),
                sidecar.body.physical_lining.clone(),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let physical_lining_descriptor_population =
        crate::component_correspondence::build_native_component_descriptors_with_physical_lining(
            topology,
            &full_load.nodes,
            &full_load.record_lineage,
            &selected.lineages,
            &exact_connectivity,
            lane,
            physical_lining_occurrences,
        )?;
    let component_descriptors = physical_lining_descriptor_population.descriptors.clone();
    let join_ghost_nodes = ghost_measurements
        .iter()
        .map(|measurement| LambdaGhostJoinGhostNode {
            site_key: measurement.site_key.clone(),
            control_lane: measurement.control_lane,
            ghost: measurement.ghost.clone(),
            protocol_phase: measurement.protocol_phase.clone(),
            thermodynamic_condition: measurement.thermodynamic_condition.clone(),
        })
        .collect::<Vec<_>>();
    let lambda_ghost_endpoint_join =
        crate::chemical_state_fiber::materialize_lambda_ghost_endpoint_joins_from_nodes(
            &join_ghost_nodes,
            lambda_load,
        )?;
    let lambda_adapter = crate::lambda_fragment::adapt_verified_lambda_journals_for_pathb(
        lambda_load,
        &lambda_ghost_endpoint_join,
        lane,
        &request.body.lane_policy.fragment_panel_sha256,
        match lane {
            ControlLane::NoProbe => None,
            ControlLane::HypothesisFragment | ControlLane::OrthogonalChemicalControl => Some(
                topology
                    .lambda_fragment_system
                    .as_ref()
                    .context("probe Path-B lane topology lacks lambda fragment system")?,
            ),
        },
    )
    .map_err(anyhow::Error::msg)?;
    lambda_adapter.validate().map_err(anyhow::Error::msg)?;
    let confirmation_surveillance = match request.body.lane_policy.selection_mode {
        NativePathBSelectionMode::WholeLatticeDiscovery => {
            let fixed = fixed_population.context(
                "whole-lattice discovery lacks complete whole-protein fixed-cadence coordinates",
            )?;
            validate_fixed_cadence_timebase_binding(&request, &fixed.streams)?;
            if fixed.streams.values().any(|load| {
                load.manifest.context.mode
                    != crate::fixed_cadence_coordinate::FixedCadenceCoordinateMode::WholeProteinDiscovery
            }) {
                bail!("whole-lattice discovery received confirmation-mode cadence evidence");
            }
            None
        }
        NativePathBSelectionMode::TargetResidueHypothesis => {
            let fixed = fixed_population.context(
                "target native Path-B lane lacks verified fixed-cadence coordinate population",
            )?;
            validate_fixed_cadence_timebase_binding(&request, &fixed.streams)?;
            if fixed.streams.values().any(|load| {
                load.manifest.context.mode
                    != crate::fixed_cadence_coordinate::FixedCadenceCoordinateMode::CandidateConfirmation
            }) {
                bail!("target Path-B lane received discovery-mode cadence evidence");
            }
            Some(materialize_fixed_cadence_surveillance(
                &request,
                &selected,
                &exact_connectivity,
                &fixed.streams,
                &fixed.population_sha256,
                lambda_load,
                topology,
            )?)
        }
    };
    let coordinate_population_sha256 = match fixed_population {
        Some(fixed) => canonical_sha256(&(
            coordinate_join.conservation.receipt_sha256.as_str(),
            fixed.population_sha256.as_str(),
            "ghost_event_coordinates_plus_fixed_cadence_time_at_risk",
        ))?,
        None => coordinate_join.conservation.receipt_sha256.clone(),
    };
    let source_authority = NativePathBLaneSourceAuthority {
        coordinate_population_sha256,
        lambda_population_sha256: lambda_adapter.lambda_population_sha256,
        lambda_ghost_endpoint_join_sha256: lambda_adapter.lambda_ghost_join_materialization_sha256,
        fragment_panel_sha256: lambda_adapter.fragment_panel_sha256,
        no_probe_absence_receipt_sha256: lambda_adapter.no_probe_absence_receipt_sha256,
        chemical_potential_calibration_population_sha256: lambda_adapter
            .chemical_potential_calibration_population_sha256,
    };
    let lattice_observations = selected
        .nodes
        .iter()
        .zip(&selected.lineages)
        .map(|(node, lineage)| NativePathBLatticeObservation::from_native_node(node, lineage))
        .collect::<Result<Vec<_>>>()?;
    let inputs = NativePathBLaneMaterializationInputs {
        runtime_request: request,
        site_selection: selected.selection.clone(),
        source_authority,
        lattice_observations,
        exact_connectivity,
        component_descriptors,
        physical_lining_descriptor_population,
        confirmation_surveillance,
        ghost_measurements,
        lambda_ghost_endpoint_join,
        fragment_observations: lambda_adapter.fragment_observations,
    };

    // The preview uses final logical paths and valid placeholder packet hashes
    // to execute every pure scientific/population validation before the first
    // staging directory or evidence byte is created. No preview artifact is
    // published or returned.
    let public_root = native_pathb_lane_transaction_root(output_dir)?;
    let join_public_raw = public_root.join(LAMBDA_GHOST_JOIN_RAW_FILE);
    let preview_join_raw = canonical_json_line(&inputs.lambda_ghost_endpoint_join)?;
    let preview_typed = TypedEvidenceArtifact {
        artifact_family: QnanlArtifactFamily::PathBChemicalState.as_str().to_string(),
        raw_path: join_public_raw.display().to_string(),
        raw_sha256: sha256_bytes(&preview_join_raw),
        packet_path: format!("{}.prismqnanl", join_public_raw.display()),
        packet_sha256: "00".repeat(32),
        receipt_path: format!("{}.prismqnanl.receipt.json", join_public_raw.display()),
        receipt_sha256: "00".repeat(32),
        prismcode_events: Vec::new(),
        runtime_origin: RuntimeOrigin::RuntimeInline.as_str().to_string(),
        exact_byte_roundtrip_pass: true,
    };
    let canonical_preview = inputs.materialize(preview_typed)?;

    let transaction_id_sha256 = inputs.transaction_id_sha256(&canonical_preview)?;
    commit_native_pathb_lane_transaction(
        output_dir,
        &transaction_id_sha256,
        &inputs,
        runtime_provenance,
        source_stage,
        NativePathBLaneTransactionFailpoint::None,
    )
}

fn native_pathb_lane_transaction_root(output_dir: &Path) -> Result<PathBuf> {
    use std::path::Component;
    if !output_dir.is_absolute()
        || output_dir
            .components()
            .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
    {
        bail!("native Path-B transaction output must be absolute and lexically normalized");
    }
    Ok(output_dir.join(NATIVE_PATHB_LANE_TRANSACTION_DIRECTORY))
}

fn native_pathb_lane_transaction_member_names() -> [(&'static str, &'static str); 6] {
    [
        ("lambda_ghost_endpoint_join_raw", LAMBDA_GHOST_JOIN_RAW_FILE),
        (
            "lambda_ghost_endpoint_join_qnanl",
            "lambda_ghost_endpoint_join_materialization.json.prismqnanl",
        ),
        (
            "lambda_ghost_endpoint_join_qnanl_receipt",
            "lambda_ghost_endpoint_join_materialization.json.prismqnanl.receipt.json",
        ),
        ("native_pathb_lane_raw", NATIVE_PATHB_LANE_RAW_FILE),
        (
            "native_pathb_lane_qnanl",
            "native_pathb_lane_evidence.json.prismqnanl",
        ),
        (
            "native_pathb_lane_qnanl_receipt",
            "native_pathb_lane_evidence.json.prismqnanl.receipt.json",
        ),
    ]
}

fn read_regular_nofollow(path: &Path, label: &str) -> Result<Vec<u8>> {
    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("inspect {label} {}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        bail!("{label} must be a non-symlink regular file");
    }
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(path)
        .with_context(|| format!("open {label} {}", path.display()))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    Ok(bytes)
}

fn sync_regular_nofollow(path: &Path) -> Result<()> {
    OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(path)
        .with_context(|| format!("open immutable transaction member {}", path.display()))?
        .sync_all()?;
    Ok(())
}

fn sync_directory(path: &Path) -> Result<()> {
    OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_DIRECTORY | libc::O_NOFOLLOW)
        .open(path)
        .with_context(|| format!("open transaction directory {}", path.display()))?
        .sync_all()?;
    Ok(())
}

fn transaction_member(
    root: &Path,
    ordinal: usize,
    role: &str,
    relative_path: &str,
) -> Result<NativePathBLaneTransactionMember> {
    let bytes = read_regular_nofollow(&root.join(relative_path), "Path-B transaction member")?;
    Ok(NativePathBLaneTransactionMember {
        ordinal: u8::try_from(ordinal)?,
        role: role.to_string(),
        relative_path: relative_path.to_string(),
        byte_length: u64::try_from(bytes.len())?,
        sha256: sha256_bytes(&bytes),
    })
}

pub(crate) fn emit_qnanl_exact_at_public_paths(
    actual_raw_path: &Path,
    public_raw_path: &Path,
    family: QnanlArtifactFamily,
    event_rows: &[(&str, u16, Vec<&str>)],
    runtime_provenance: &RuntimeProvenance,
    source_stage: &str,
) -> Result<TypedEvidenceArtifact> {
    for (name, expected, _) in event_rows {
        if opcode_id_for_name(name) != Some(*expected) {
            bail!("canonical PrismCode opcode registry mismatch for {name}");
        }
    }
    let actual_packet_path = PathBuf::from(format!("{}.prismqnanl", actual_raw_path.display()));
    let actual_receipt_path =
        PathBuf::from(format!("{}.receipt.json", actual_packet_path.display()));
    let public_packet_path = PathBuf::from(format!("{}.prismqnanl", public_raw_path.display()));
    let public_receipt_path =
        PathBuf::from(format!("{}.receipt.json", public_packet_path.display()));
    let artifact_id = public_raw_path
        .file_name()
        .and_then(|name| name.to_str())
        .context("transaction evidence filename is not UTF-8")?;
    let raw_bytes = read_regular_nofollow(actual_raw_path, "staged transaction raw evidence")?;
    let mut receipt = encode_bytes(
        &raw_bytes,
        &actual_packet_path,
        &actual_receipt_path,
        QnanlEncodeOptions::runtime_inline(
            artifact_id,
            family,
            Some(public_raw_path.to_path_buf()),
            source_stage,
        ),
    )
    .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let pcode_refs = event_rows
        .iter()
        .map(|(name, opcode, operands)| QnanlPcodeRef {
            opcode_name: (*name).to_string(),
            opcode_id: Some(*opcode),
            operands: operands.iter().map(|value| (*value).to_string()).collect(),
        })
        .collect();
    let raw_parent_hash = receipt.raw_hash_blake3.clone();
    let mut provenance = runtime_provenance.artifact_record(
        &actual_packet_path,
        source_stage,
        vec![raw_parent_hash],
    );
    // The bytes were hashed from the staged inode. The logical path records
    // where that same inode becomes reachable after the atomic directory
    // rename, so no receipt contains an orphan staging path.
    provenance.emitted_artifact_path = public_packet_path.display().to_string();
    receipt.source_path = public_raw_path.display().to_string();
    receipt.packet_path = public_packet_path.display().to_string();
    receipt.receipt_path = public_receipt_path.display().to_string();
    receipt = receipt
        .with_pcode_refs(pcode_refs)
        .with_runtime_provenance(provenance);
    let receipt_bytes = receipt.to_json_pretty();
    let mut receipt_file = OpenOptions::new()
        .write(true)
        .truncate(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(&actual_receipt_path)
        .with_context(|| {
            format!(
                "open staged QNANL receipt {}",
                actual_receipt_path.display()
            )
        })?;
    receipt_file.write_all(receipt_bytes.as_bytes())?;
    receipt_file.sync_all()?;
    sync_regular_nofollow(&actual_packet_path)?;
    let packet_bytes = read_regular_nofollow(&actual_packet_path, "staged QNANL packet")?;
    let decoded =
        decode_packet_bytes(&packet_bytes).map_err(|error| anyhow::anyhow!(error.to_string()))?;
    if receipt.status != "PASS"
        || receipt.runtime_origin != RuntimeOrigin::RuntimeInline
        || receipt.runtime_provenance.is_none()
        || !receipt.roundtrip_verified
        || !receipt.codec_roundtrip_verified
        || !receipt.qnanl_seal_constraints_pass
        || decoded != raw_bytes
    {
        bail!("staged typed QNANL evidence failed runtime-origin or exact-byte gates");
    }
    Ok(TypedEvidenceArtifact {
        artifact_family: family.as_str().to_string(),
        raw_path: public_raw_path.display().to_string(),
        raw_sha256: sha256_bytes(&raw_bytes),
        packet_path: public_packet_path.display().to_string(),
        packet_sha256: sha256_bytes(&packet_bytes),
        receipt_path: public_receipt_path.display().to_string(),
        receipt_sha256: sha256_bytes(receipt_bytes.as_bytes()),
        prismcode_events: event_rows
            .iter()
            .map(|(name, _, _)| (*name).to_string())
            .collect(),
        runtime_origin: receipt.runtime_origin.as_str().to_string(),
        exact_byte_roundtrip_pass: true,
    })
}

fn verify_typed_transaction_artifact(
    actual_root: &Path,
    public_root: &Path,
    raw_relative_path: &str,
    artifact: &TypedEvidenceArtifact,
) -> Result<()> {
    let public_raw = public_root.join(raw_relative_path);
    let public_packet = PathBuf::from(format!("{}.prismqnanl", public_raw.display()));
    let public_receipt = PathBuf::from(format!("{}.receipt.json", public_packet.display()));
    if artifact.artifact_family != QnanlArtifactFamily::PathBChemicalState.as_str()
        || artifact.raw_path != public_raw.display().to_string()
        || artifact.packet_path != public_packet.display().to_string()
        || artifact.receipt_path != public_receipt.display().to_string()
        || artifact.runtime_origin != RuntimeOrigin::RuntimeInline.as_str()
        || !artifact.exact_byte_roundtrip_pass
    {
        bail!("Path-B transaction typed artifact has non-public paths or invalid identity");
    }
    let actual_raw = actual_root.join(raw_relative_path);
    let actual_packet = PathBuf::from(format!("{}.prismqnanl", actual_raw.display()));
    let actual_receipt = PathBuf::from(format!("{}.receipt.json", actual_packet.display()));
    let raw_bytes = read_regular_nofollow(&actual_raw, "transaction raw evidence")?;
    let packet_bytes = read_regular_nofollow(&actual_packet, "transaction QNANL packet")?;
    let receipt_bytes = read_regular_nofollow(&actual_receipt, "transaction QNANL receipt")?;
    if artifact.raw_sha256 != sha256_bytes(&raw_bytes)
        || artifact.packet_sha256 != sha256_bytes(&packet_bytes)
        || artifact.receipt_sha256 != sha256_bytes(&receipt_bytes)
        || decode_packet_bytes(&packet_bytes).map_err(|error| anyhow::anyhow!(error.to_string()))?
            != raw_bytes
    {
        bail!("Path-B transaction typed artifact hash or exact-byte decode mismatch");
    }
    let receipt: serde_json::Value =
        serde_json::from_slice(&receipt_bytes).context("decode transaction QNANL receipt")?;
    if receipt.get("status").and_then(|value| value.as_str()) != Some("PASS")
        || receipt.get("source_path").and_then(|value| value.as_str())
            != Some(artifact.raw_path.as_str())
        || receipt.get("packet_path").and_then(|value| value.as_str())
            != Some(artifact.packet_path.as_str())
        || receipt.get("receipt_path").and_then(|value| value.as_str())
            != Some(artifact.receipt_path.as_str())
        || receipt
            .get("runtime_origin")
            .and_then(|value| value.as_str())
            != Some(RuntimeOrigin::RuntimeInline.as_str())
        || receipt
            .pointer("/runtime_provenance/emitted_artifact_path")
            .and_then(|value| value.as_str())
            != Some(artifact.packet_path.as_str())
    {
        bail!("Path-B transaction QNANL receipt does not bind final public paths");
    }
    Ok(())
}

fn verify_native_pathb_lane_transaction_directory(
    actual_root: &Path,
    expected_public_root: Option<&Path>,
) -> Result<VerifiedNativePathBLaneTransactionBytes> {
    let metadata = std::fs::symlink_metadata(actual_root)
        .with_context(|| format!("inspect Path-B transaction {}", actual_root.display()))?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_dir() {
        bail!("native Path-B transaction root must be a non-symlink directory");
    }
    let commit_path = actual_root.join(NATIVE_PATHB_LANE_TRANSACTION_COMMIT_FILE);
    let commit_bytes = read_regular_nofollow(&commit_path, "native Path-B transaction commit")?;
    if commit_bytes.is_empty() || commit_bytes.last() != Some(&b'\n') {
        bail!("native Path-B transaction commit is empty or not newline-terminated");
    }
    let commit: NativePathBLaneTransactionCommit =
        serde_json::from_slice(&commit_bytes).context("decode native Path-B transaction commit")?;
    if canonical_json_line(&commit)? != commit_bytes
        || commit.schema_kind != NATIVE_PATHB_LANE_TRANSACTION_SCHEMA
        || commit.schema_version != NATIVE_PATHB_LANE_TRANSACTION_VERSION
        || commit.status != "PASS"
        || canonical_sha256(&commit.body)? != commit.commit_sha256
    {
        bail!("native Path-B transaction commit schema, bytes or content hash mismatch");
    }
    for (digest, field) in [
        (&commit.commit_sha256, "transaction commit_sha256"),
        (&commit.body.transaction_id_sha256, "transaction_id_sha256"),
        (&commit.body.lane_evidence_sha256, "lane_evidence_sha256"),
        (
            &commit.body.lambda_ghost_endpoint_join_sha256,
            "lambda_ghost_endpoint_join_sha256",
        ),
        (
            &commit.body.ordered_membership_sha256,
            "ordered_membership_sha256",
        ),
    ] {
        require_sha256(digest, field)?;
    }
    let public_root = expected_public_root.unwrap_or(actual_root);
    if commit.body.committed_directory != public_root.display().to_string()
        || commit.body.member_count != 6
        || commit.body.members.len() != 6
        || canonical_sha256(&commit.body.members)? != commit.body.ordered_membership_sha256
    {
        bail!("native Path-B transaction public root or membership conservation mismatch");
    }
    let expected_members = native_pathb_lane_transaction_member_names();
    for (ordinal, ((expected_role, expected_path), member)) in expected_members
        .iter()
        .zip(&commit.body.members)
        .enumerate()
    {
        require_sha256(&member.sha256, "transaction member sha256")?;
        if member.ordinal != u8::try_from(ordinal)?
            || member.role != *expected_role
            || member.relative_path != *expected_path
        {
            bail!("native Path-B transaction member order, role or path mismatch");
        }
        let bytes = read_regular_nofollow(
            &actual_root.join(&member.relative_path),
            "native Path-B transaction member",
        )?;
        if member.byte_length != u64::try_from(bytes.len())?
            || member.sha256 != sha256_bytes(&bytes)
        {
            bail!("native Path-B transaction member length/hash mismatch");
        }
    }
    let expected_names = expected_members
        .iter()
        .map(|(_, name)| (*name).to_string())
        .chain(std::iter::once(
            NATIVE_PATHB_LANE_TRANSACTION_COMMIT_FILE.to_string(),
        ))
        .collect::<BTreeSet<_>>();
    let mut observed_names = BTreeSet::new();
    for entry in std::fs::read_dir(actual_root)? {
        let entry = entry?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| anyhow::anyhow!("transaction member filename is not UTF-8"))?;
        let entry_metadata = std::fs::symlink_metadata(entry.path())?;
        if entry_metadata.file_type().is_symlink() || !entry_metadata.file_type().is_file() {
            bail!("native Path-B transaction contains a non-regular member");
        }
        observed_names.insert(name);
    }
    if observed_names != expected_names {
        bail!("native Path-B transaction contains missing, duplicate or extra members");
    }
    verify_typed_transaction_artifact(
        actual_root,
        public_root,
        LAMBDA_GHOST_JOIN_RAW_FILE,
        &commit.body.lambda_ghost_endpoint_join_typed_artifact,
    )?;
    verify_typed_transaction_artifact(
        actual_root,
        public_root,
        NATIVE_PATHB_LANE_RAW_FILE,
        &commit.body.native_pathb_lane_typed_artifact,
    )?;
    Ok(VerifiedNativePathBLaneTransactionBytes {
        commit,
        join_raw: read_regular_nofollow(
            &actual_root.join(LAMBDA_GHOST_JOIN_RAW_FILE),
            "lambda/Ghost join transaction member",
        )?,
        lane_raw: read_regular_nofollow(
            &actual_root.join(NATIVE_PATHB_LANE_RAW_FILE),
            "native Path-B lane transaction member",
        )?,
    })
}

fn load_emission_from_verified_transaction(
    verified: VerifiedNativePathBLaneTransactionBytes,
    transaction_root: &Path,
    expected_transaction_id_sha256: Option<&str>,
) -> Result<NativePathBLaneEvidenceEmission> {
    if let Some(expected) = expected_transaction_id_sha256 {
        if verified.commit.body.transaction_id_sha256 != expected {
            bail!("refusing to overwrite a different committed native Path-B transaction");
        }
    }
    let join: LambdaGhostEndpointJoinMaterialization =
        serde_json::from_slice(&verified.join_raw)
            .context("decode committed lambda/Ghost endpoint join")?;
    join.verify()?;
    if canonical_json_line(&join)? != verified.join_raw
        || join.materialization_sha256 != verified.commit.body.lambda_ghost_endpoint_join_sha256
    {
        bail!("committed lambda/Ghost join bytes or semantic hash mismatch");
    }
    let lane_evidence: NativePathBLaneEvidence = serde_json::from_slice(&verified.lane_raw)
        .context("decode committed native Path-B lane evidence")?;
    lane_evidence.verify()?;
    if canonical_json_line(&lane_evidence)? != verified.lane_raw
        || lane_evidence.lane_evidence_sha256 != verified.commit.body.lane_evidence_sha256
        || lane_evidence
            .body
            .lambda_ghost_endpoint_join
            .materialization_sha256
            != join.materialization_sha256
        || lane_evidence.body.lambda_ghost_endpoint_join_typed_artifact
            != verified
                .commit
                .body
                .lambda_ghost_endpoint_join_typed_artifact
    {
        bail!("committed native Path-B lane bytes or joined authority mismatch");
    }
    let typed_artifact = verified
        .commit
        .body
        .native_pathb_lane_typed_artifact
        .clone();
    let emission_sha256 = canonical_sha256(&(
        lane_evidence.lane_evidence_sha256.as_str(),
        &typed_artifact,
        verified.commit.commit_sha256.as_str(),
        "single_run_lane_evidence_fail_atomic_transaction_v1",
    ))?;
    Ok(NativePathBLaneEvidenceEmission {
        lane_evidence,
        typed_artifact,
        emission_sha256,
        transaction_commit_path: transaction_root
            .join(NATIVE_PATHB_LANE_TRANSACTION_COMMIT_FILE)
            .display()
            .to_string(),
        transaction_commit: verified.commit,
    })
}

#[cfg(target_os = "linux")]
fn publish_directory_noreplace(staging_root: &Path, public_root: &Path) -> Result<()> {
    let staging = CString::new(staging_root.as_os_str().as_bytes())?;
    let public = CString::new(public_root.as_os_str().as_bytes())?;
    let status = unsafe {
        libc::syscall(
            libc::SYS_renameat2,
            libc::AT_FDCWD,
            staging.as_ptr(),
            libc::AT_FDCWD,
            public.as_ptr(),
            libc::RENAME_NOREPLACE,
        )
    };
    if status != 0 {
        return Err(std::io::Error::last_os_error()).with_context(|| {
            format!(
                "atomically publish native Path-B transaction {} -> {}",
                staging_root.display(),
                public_root.display()
            )
        });
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn publish_directory_noreplace(_staging_root: &Path, _public_root: &Path) -> Result<()> {
    bail!("native Path-B fail-atomic publication requires Linux renameat2 RENAME_NOREPLACE")
}

#[cfg(any(feature = "gpu", feature = "cpu-analysis"))]
fn commit_native_pathb_lane_transaction(
    output_dir: &Path,
    transaction_id_sha256: &str,
    inputs: &NativePathBLaneMaterializationInputs,
    runtime_provenance: &RuntimeProvenance,
    source_stage: &str,
    failpoint: NativePathBLaneTransactionFailpoint,
) -> Result<NativePathBLaneEvidenceEmission> {
    require_sha256(transaction_id_sha256, "native Path-B transaction id")?;
    let public_root = native_pathb_lane_transaction_root(output_dir)?;
    if public_root.exists() {
        let verified = verify_native_pathb_lane_transaction_directory(&public_root, None)?;
        return load_emission_from_verified_transaction(
            verified,
            &public_root,
            Some(transaction_id_sha256),
        );
    }
    std::fs::create_dir_all(output_dir)?;
    let output_metadata = std::fs::symlink_metadata(output_dir)?;
    if output_metadata.file_type().is_symlink() || !output_metadata.file_type().is_dir() {
        bail!("native Path-B transaction output root is not a non-symlink directory");
    }
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let staging_root = output_dir.join(format!(
        ".{NATIVE_PATHB_LANE_TRANSACTION_DIRECTORY}.staging.{}.{}.{}",
        &transaction_id_sha256[..16],
        std::process::id(),
        nonce
    ));
    std::fs::create_dir(&staging_root).with_context(|| {
        format!(
            "create unique native Path-B transaction staging {}",
            staging_root.display()
        )
    })?;

    let join_raw_path = staging_root.join(LAMBDA_GHOST_JOIN_RAW_FILE);
    write_create_new(
        &join_raw_path,
        &canonical_json_line(&inputs.lambda_ghost_endpoint_join)?,
    )?;
    let join_typed = emit_qnanl_exact_at_public_paths(
        &join_raw_path,
        &public_root.join(LAMBDA_GHOST_JOIN_RAW_FILE),
        QnanlArtifactFamily::PathBChemicalState,
        &[],
        runtime_provenance,
        "nhs_rt_full:lambda_ghost_endpoint_join_transaction_stage",
    )?;
    if failpoint == NativePathBLaneTransactionFailpoint::AfterJoinTriplet {
        bail!("TEST_ONLY_NATIVE_PATHB_TRANSACTION_FAILPOINT_AFTER_JOIN_TRIPLET");
    }

    let lane_evidence = inputs.materialize(join_typed.clone())?;
    let lane_raw_path = staging_root.join(NATIVE_PATHB_LANE_RAW_FILE);
    write_create_new(&lane_raw_path, &canonical_json_line(&lane_evidence)?)?;
    let lane_typed = emit_qnanl_exact_at_public_paths(
        &lane_raw_path,
        &public_root.join(NATIVE_PATHB_LANE_RAW_FILE),
        QnanlArtifactFamily::PathBChemicalState,
        &[],
        runtime_provenance,
        source_stage,
    )?;

    let members = native_pathb_lane_transaction_member_names()
        .iter()
        .enumerate()
        .map(|(ordinal, (role, path))| transaction_member(&staging_root, ordinal, role, path))
        .collect::<Result<Vec<_>>>()?;
    if failpoint == NativePathBLaneTransactionFailpoint::AfterCompleteStageBeforeCommit {
        bail!("TEST_ONLY_NATIVE_PATHB_TRANSACTION_FAILPOINT_AFTER_COMPLETE_STAGE");
    }
    let body = NativePathBLaneTransactionCommitBody {
        transaction_id_sha256: transaction_id_sha256.to_string(),
        committed_directory: public_root.display().to_string(),
        lane_evidence_sha256: lane_evidence.lane_evidence_sha256.clone(),
        lambda_ghost_endpoint_join_sha256: inputs
            .lambda_ghost_endpoint_join
            .materialization_sha256
            .clone(),
        ordered_membership_sha256: canonical_sha256(&members)?,
        member_count: u64::try_from(members.len())?,
        members,
        lambda_ghost_endpoint_join_typed_artifact: join_typed,
        native_pathb_lane_typed_artifact: lane_typed,
    };
    let commit = NativePathBLaneTransactionCommit {
        schema_kind: NATIVE_PATHB_LANE_TRANSACTION_SCHEMA.to_string(),
        schema_version: NATIVE_PATHB_LANE_TRANSACTION_VERSION.to_string(),
        status: "PASS".to_string(),
        commit_sha256: canonical_sha256(&body)?,
        body,
    };
    write_create_new(
        &staging_root.join(NATIVE_PATHB_LANE_TRANSACTION_COMMIT_FILE),
        &canonical_json_line(&commit)?,
    )?;
    sync_directory(&staging_root)?;
    let staged_verified =
        verify_native_pathb_lane_transaction_directory(&staging_root, Some(&public_root))?;
    // Execute the same semantic/canonical loader used after publication while
    // the directory is still private. Therefore no byte can become authority
    // unless both the transaction envelope and the joined scientific evidence
    // already pass completely.
    load_emission_from_verified_transaction(
        staged_verified,
        &public_root,
        Some(transaction_id_sha256),
    )?;
    if failpoint == NativePathBLaneTransactionFailpoint::AfterCommitBeforePublish {
        bail!("TEST_ONLY_NATIVE_PATHB_TRANSACTION_FAILPOINT_AFTER_COMMIT");
    }
    if let Err(publish_error) = publish_directory_noreplace(&staging_root, &public_root) {
        if public_root.exists() {
            let verified = verify_native_pathb_lane_transaction_directory(&public_root, None)?;
            return load_emission_from_verified_transaction(
                verified,
                &public_root,
                Some(transaction_id_sha256),
            );
        }
        return Err(publish_error);
    }
    sync_directory(output_dir)?;
    let verified = verify_native_pathb_lane_transaction_directory(&public_root, None)?;
    load_emission_from_verified_transaction(verified, &public_root, Some(transaction_id_sha256))
}

/// Load and authenticate the fail-atomic transaction commit and its complete
/// six-member byte population. This reusable gate does not accept a staging
/// directory, a legacy flat output, or a commit receipt detached from its
/// fixed committed root.
pub fn load_native_pathb_lane_transaction_commit(
    transaction_root: &Path,
) -> Result<NativePathBLaneTransactionCommit> {
    use std::path::Component;
    if !transaction_root.is_absolute()
        || transaction_root
            .components()
            .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
        || transaction_root.file_name().and_then(|name| name.to_str())
            != Some(NATIVE_PATHB_LANE_TRANSACTION_DIRECTORY)
    {
        bail!("native Path-B transaction root path is not the fixed absolute committed root");
    }
    Ok(verify_native_pathb_lane_transaction_directory(transaction_root, None)?.commit)
}

/// Read and verify one immutable per-run lane document without following a
/// symlink. The path may be either the committed transaction directory or its
/// exact `native_pathb_lane_evidence.json` member. Legacy direct files and
/// uncommitted staging directories are rejected. The cross-run aggregator uses
/// only these transaction-verified documents.
pub fn load_native_pathb_lane_evidence(path: &Path) -> Result<NativePathBLaneEvidence> {
    use std::path::Component;
    if !path.is_absolute()
        || path
            .components()
            .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
    {
        bail!("native Path-B lane evidence path must be absolute and normalized");
    }
    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("inspect native Path-B lane evidence {}", path.display()))?;
    if metadata.file_type().is_symlink() {
        bail!("native Path-B lane evidence path cannot be a symlink");
    }
    let transaction_root = if metadata.file_type().is_dir() {
        if path.file_name().and_then(|name| name.to_str())
            != Some(NATIVE_PATHB_LANE_TRANSACTION_DIRECTORY)
        {
            bail!("native Path-B directory is not the committed transaction root");
        }
        path.to_path_buf()
    } else if metadata.file_type().is_file() {
        if path.file_name().and_then(|name| name.to_str()) != Some(NATIVE_PATHB_LANE_RAW_FILE) {
            bail!("native Path-B lane file is not the committed lane member");
        }
        let parent = path
            .parent()
            .context("native Path-B lane file has no parent")?;
        if parent.file_name().and_then(|name| name.to_str())
            != Some(NATIVE_PATHB_LANE_TRANSACTION_DIRECTORY)
        {
            bail!("legacy or uncommitted native Path-B lane evidence is not admissible");
        }
        parent.to_path_buf()
    } else {
        bail!("native Path-B lane evidence path is neither a regular file nor directory");
    };
    let verified = verify_native_pathb_lane_transaction_directory(&transaction_root, None)?;
    Ok(load_emission_from_verified_transaction(verified, &transaction_root, None)?.lane_evidence)
}

/// Cross-run aggregation manifest schema.
pub const PATHB_CROSS_RUN_MANIFEST_SCHEMA: &str = "prism_pathb_cross_run_aggregation_manifest";
/// Cross-run aggregation manifest schema version.
pub const PATHB_CROSS_RUN_MANIFEST_VERSION: &str = "6.0.0";

/// Exact identity projection emitted by the native committed-handoff verifier.
/// The cross-run aggregator re-hashes every referenced public handoff member;
/// it never accepts a simplified discovery-candidate JSON substitute.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PathBConfirmationHandoffIdentity {
    pub handoff_root: PathBuf,
    pub handoff_root_raw_file_sha256: String,
    pub handoff_root_commit_file_sha256: String,
    pub handoff_sha256: String,
    pub handoff_transaction_commit_sha256: String,
    pub handoff_ordered_membership_sha256: String,
    pub discovery_transaction_commit_sha256: String,
    pub discovery_population_sha256: String,
    pub discovery_manifest_sha256: String,
    pub correspondence_ledger_sha256: String,
    pub selected_candidate_id_sha256: String,
    pub selected_candidate_authority_sha256: String,
    pub explicit_selection_authority_sha256: String,
    pub operator_selection_authority_sha256: String,
    pub tracker_design_authority_sha256: String,
    pub supplied_tracker_sha256: String,
    pub operator_design_authority_sha256: String,
    pub disjoint_confirmation_seed_authority_sha256: String,
    pub discovery_replica_seeds: Vec<u64>,
    pub confirmation_replica_seeds: Vec<u64>,
    pub qnanl_artifact_family: String,
    pub qnanl_raw_sha256: String,
    pub qnanl_packet_sha256: String,
    pub qnanl_receipt_sha256: String,
}

impl PathBConfirmationHandoffIdentity {
    fn verify(&self) -> Result<()> {
        if !self.handoff_root.is_absolute()
            || self.handoff_root.components().any(|part| {
                matches!(
                    part,
                    std::path::Component::CurDir | std::path::Component::ParentDir
                )
            })
            || self.discovery_replica_seeds.len() < 5
            || self.confirmation_replica_seeds.len() < 5
            || self
                .discovery_replica_seeds
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || self
                .confirmation_replica_seeds
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || self
                .discovery_replica_seeds
                .iter()
                .any(|seed| self.confirmation_replica_seeds.contains(seed))
            || self.qnanl_artifact_family != "PRISM.PATHB_CONFIRMATION_HANDOFF.V1"
            || self.qnanl_raw_sha256 != self.handoff_root_raw_file_sha256
        {
            bail!("invalid native confirmation handoff identity projection");
        }
        let metadata = std::fs::symlink_metadata(&self.handoff_root).with_context(|| {
            format!(
                "inspect confirmation handoff root {}",
                self.handoff_root.display()
            )
        })?;
        if metadata.file_type().is_symlink()
            || !metadata.is_dir()
            || self.handoff_root.canonicalize()? != self.handoff_root
        {
            bail!("confirmation handoff root is not an immutable canonical directory");
        }
        for (digest, field) in [
            (&self.handoff_root_raw_file_sha256, "handoff raw file"),
            (&self.handoff_root_commit_file_sha256, "handoff commit file"),
            (&self.handoff_sha256, "handoff"),
            (
                &self.handoff_transaction_commit_sha256,
                "handoff transaction commit",
            ),
            (
                &self.handoff_ordered_membership_sha256,
                "handoff ordered membership",
            ),
            (
                &self.discovery_transaction_commit_sha256,
                "discovery transaction commit",
            ),
            (&self.discovery_population_sha256, "discovery population"),
            (&self.discovery_manifest_sha256, "discovery manifest"),
            (&self.correspondence_ledger_sha256, "correspondence ledger"),
            (&self.selected_candidate_id_sha256, "selected candidate id"),
            (
                &self.selected_candidate_authority_sha256,
                "selected candidate authority",
            ),
            (
                &self.explicit_selection_authority_sha256,
                "explicit selection authority",
            ),
            (
                &self.operator_selection_authority_sha256,
                "operator selection authority",
            ),
            (
                &self.tracker_design_authority_sha256,
                "tracker design authority",
            ),
            (&self.supplied_tracker_sha256, "supplied tracker"),
            (
                &self.operator_design_authority_sha256,
                "operator design authority",
            ),
            (
                &self.disjoint_confirmation_seed_authority_sha256,
                "confirmation seed authority",
            ),
            (&self.qnanl_raw_sha256, "handoff QNANL raw"),
            (&self.qnanl_packet_sha256, "handoff QNANL packet"),
            (&self.qnanl_receipt_sha256, "handoff QNANL receipt"),
        ] {
            require_sha256(digest, field)?;
        }
        let raw_path = self
            .handoff_root
            .join("native_pathb_confirmation_handoff.json");
        let commit_path = self.handoff_root.join("COMMIT.json");
        let packet_path = PathBuf::from(format!("{}.prismqnanl", raw_path.display()));
        let receipt_path = PathBuf::from(format!("{}.receipt.json", packet_path.display()));
        for (path, expected, label) in [
            (&raw_path, &self.handoff_root_raw_file_sha256, "handoff raw"),
            (
                &commit_path,
                &self.handoff_root_commit_file_sha256,
                "handoff commit",
            ),
            (
                &packet_path,
                &self.qnanl_packet_sha256,
                "handoff QNANL packet",
            ),
            (
                &receipt_path,
                &self.qnanl_receipt_sha256,
                "handoff QNANL receipt",
            ),
        ] {
            let bytes = read_regular_nofollow(path, label)?;
            if sha256_bytes(&bytes) != *expected {
                bail!("{label} bytes differ from the verifier-approved handoff identity");
            }
        }
        Ok(())
    }
}

/// Immutable endpoint pair identity for one physical probe lane.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProbeLaneMicrostatePairIdentity {
    pub identity_sha256: String,
    pub endpoint_a: crate::chemical_state_fiber::FragmentChemicalIdentity,
    pub endpoint_b: crate::chemical_state_fiber::FragmentChemicalIdentity,
}

impl ProbeLaneMicrostatePairIdentity {
    pub fn verify(&self) -> Result<()> {
        self.endpoint_a.validate()?;
        self.endpoint_b.validate()?;
        require_sha256(
            &self.identity_sha256,
            "probe microstate pair identity_sha256",
        )?;
        if canonical_sha256(&(&self.endpoint_a, &self.endpoint_b))? != self.identity_sha256 {
            bail!("probe microstate pair identity hash mismatch");
        }
        Ok(())
    }
}

/// One immutable lane document admitted to cross-run inference.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CrossRunLaneEvidenceEntry {
    pub lane_evidence_path: PathBuf,
    pub lane_evidence_sha256: String,
}

/// Topology-local realization of the unchanged confirmation candidate.
/// The manifest hashes this complete tracker and its immutable topology path.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PathBTrackerLaneRealization {
    pub topology_path: PathBuf,
    pub candidate_tracker: PathBCandidateTrackerDefinition,
}

/// Validate local atom/residue indices against the actual signed topology.
pub fn validate_candidate_tracker_topology(
    tracker: &PathBCandidateTrackerDefinition,
    topology: &crate::input::PrismPrepTopology,
    topology_sha256: &str,
) -> Result<()> {
    use crate::fixed_cadence_coordinate::BodyFixedBoundaryMode;

    tracker.verify()?;
    topology.validate().map_err(anyhow::Error::msg)?;
    topology.validate_analysis_residue_axis().map_err(anyhow::Error::msg)?;
    let authority = &tracker.body_fixed_tracker_authority;
    let analysis = topology
        .analysis_residue_contract
        .as_ref()
        .context("body-fixed preflight requires signed analysis residue contract")?;
    let atom_analysis_ids = topology
        .atom_analysis_residue_ids
        .as_ref()
        .context("body-fixed preflight requires atom-analysis mapping")?;
    let periodic = topology
        .periodic_cell
        .as_ref()
        .context("body-fixed preflight requires spatial boundary contract")?;
    let periodic_sha256 = periodic
        .contract_sha256
        .as_deref()
        .context("body-fixed spatial boundary contract is not hash-bound")?;
    if authority.topology_sha256 != topology_sha256
        || authority.chain_mapping_receipt_sha256 != analysis.contract_sha256
        || authority.spatial_boundary_contract_sha256 != periodic_sha256
    {
        bail!("body-fixed tracker differs from launched topology/mapping/boundary authority");
    }
    match authority.boundary_mode {
        BodyFixedBoundaryMode::PeriodicTriclinicMinimumImage
            if periodic.periodic && periodic.vectors_angstrom.is_some() => {}
        BodyFixedBoundaryMode::DeclaredNonperiodicEuclidean
            if !periodic.periodic && periodic.vectors_angstrom.is_none() => {}
        _ => bail!("body-fixed boundary mode differs from launched topology"),
    }
    let analysis_rows = analysis
        .analysis_residues
        .iter()
        .map(|row| (row.analysis_residue_id as u32, row))
        .collect::<std::collections::BTreeMap<_, _>>();
    for anchor in &authority.anchor_atoms {
        let atom = anchor.atom_index;
        let row = analysis_rows
            .get(&anchor.analysis_residue_id)
            .context("body-fixed anchor names an unknown analysis residue")?;
        if atom >= topology.n_atoms
            || atom_analysis_ids.get(atom).copied() != Some(anchor.analysis_residue_id as i32)
            || topology.atom_names.get(atom) != Some(&anchor.atom_name)
            || topology.chain_ids.get(atom) != Some(&anchor.chain_id)
            || topology.residue_names.get(atom) != Some(&anchor.residue_name)
            || row.chain_id != anchor.chain_id
            || row.pdb_residue_id != anchor.pdb_residue_id
            || row.residue_name != anchor.residue_name
        {
            bail!("body-fixed anchor identity differs from launched topology");
        }
    }
    let mut contact_keys = BTreeSet::new();
    for contact in &tracker.signed_contact_residues {
        let row = analysis_rows.get(&contact.engine_residue_id)
            .context("candidate contact names an unknown analysis residue")?;
        if contact.chain_mapping_receipt_sha256 != analysis.contract_sha256
            || contact.chain_id != row.chain_id
            || contact.canonical_residue_id != format!("{}:{}", row.pdb_residue_id, row.residue_name)
            || !contact_keys.insert((&contact.chain_id, &contact.canonical_residue_id))
        {
            bail!("candidate contact identity differs from launched topology");
        }
    }
    Ok(())
}


impl PathBTrackerLaneRealization {
    fn verify(&self, expected_topology_sha256: &str, reference: &PathBCandidateTrackerDefinition) -> Result<()> {
        validate_absolute_normalized_regular_path_lexical(&self.topology_path)?;
        let mut file = open_regular_nofollow(&self.topology_path)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        if format!("{:x}", Sha256::digest(&bytes)) != expected_topology_sha256 {
            bail!("lane tracker topology file differs from declared lane topology");
        }
        let topology: crate::input::PrismPrepTopology = serde_json::from_slice(&bytes)?;
        validate_candidate_tracker_topology(&self.candidate_tracker, &topology, expected_topology_sha256)?;
        verify_tracker_realization_design(reference, &self.candidate_tracker)
    }
}

/// Compare the candidate definition after projecting only topology-local indices
/// and authority digests onto the reference. Original trackers remain unchanged.
fn verify_tracker_realization_design(reference: &PathBCandidateTrackerDefinition, local: &PathBCandidateTrackerDefinition) -> Result<()> {
    reference.verify()?;
    local.verify()?;
    let mut projected = local.clone();
    projected.tracker_sha256 = reference.tracker_sha256.clone();
    let a = &mut projected.body_fixed_tracker_authority;
    let r = &reference.body_fixed_tracker_authority;
    a.authority_sha256 = r.authority_sha256.clone();
    a.topology_sha256 = r.topology_sha256.clone();
    a.chain_mapping_receipt_sha256 = r.chain_mapping_receipt_sha256.clone();
    a.spatial_boundary_contract_sha256 = r.spatial_boundary_contract_sha256.clone();
    if a.anchor_atoms.len() != r.anchor_atoms.len()
        || projected.signed_contact_residues.len() != reference.signed_contact_residues.len() {
        bail!("lane tracker changes candidate anchor/contact population");
    }
    for (anchor, original) in a.anchor_atoms.iter_mut().zip(&r.anchor_atoms) {
        anchor.atom_index = original.atom_index;
        anchor.analysis_residue_id = original.analysis_residue_id;
    }
    for (contact, original) in projected.signed_contact_residues.iter_mut().zip(&reference.signed_contact_residues) {
        contact.engine_residue_id = original.engine_residue_id;
        contact.chain_mapping_receipt_sha256 = original.chain_mapping_receipt_sha256.clone();
    }
    if projected != *reference {
        bail!("lane tracker changes the predeclared shared candidate design");
    }
    Ok(())
}

/// Hashable cross-run design and source declaration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PathBCrossRunAggregationManifestBody {
    pub site_key: String,
    pub candidate_tracker_sha256: String,
    pub source_confirmation_handoff: PathBConfirmationHandoffIdentity,
    pub fragment_panel_sha256: String,
    /// Exact hypothesis/control endpoint pairs. A panel digest alone cannot
    /// prove which two chemistry hypotheses were compared.
    pub probe_microstate_pair_by_lane: BTreeMap<ControlLane, ProbeLaneMicrostatePairIdentity>,
    /// Common T/P/ionic-strength/solvent/PBC/PME and phase-interval digest.
    pub matched_physical_state_sha256: String,
    pub engine_sha256: String,
    pub topology_sha256_by_lane: BTreeMap<ControlLane, String>,
    /// v6 only: topology-local realizations of the immutable handoff tracker.
    /// Absent in v5; omission preserves its original canonical body hash.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub tracker_realizations_by_lane: BTreeMap<ControlLane, PathBTrackerLaneRealization>,
    pub calibration_population_sha256_by_probe_lane: BTreeMap<ControlLane, String>,
    pub pathb_config_sha256: String,
    pub timebase_context_sha256: String,
    pub matched_control_config_sha256: String,
    pub registry_policy_sha256: String,
    pub common_replica_seeds: Vec<u64>,
    pub lane_evidence: Vec<CrossRunLaneEvidenceEntry>,
}

/// Content-addressed manifest; callers may not substitute raw post-run arrays.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PathBCrossRunAggregationManifest {
    pub schema_kind: String,
    pub schema_version: String,
    pub aggregation_manifest_sha256: String,
    pub body: PathBCrossRunAggregationManifestBody,
}

impl PathBCrossRunAggregationManifest {
    /// Seal a predeclared design before collecting its lane evidence.
    pub fn materialize(body: PathBCrossRunAggregationManifestBody) -> Result<Self> {
        let manifest = Self {
            schema_kind: PATHB_CROSS_RUN_MANIFEST_SCHEMA.into(),
            schema_version: if body.tracker_realizations_by_lane.is_empty() { "5.0.0" } else { PATHB_CROSS_RUN_MANIFEST_VERSION }.into(),
            aggregation_manifest_sha256: canonical_sha256(&body)?,
            body,
        };
        manifest.verify()?;
        Ok(manifest)
    }
    /// Verify canonical design identity. Evidence-dependent fields are
    /// independently reconstructed by the strict loader below.
    pub fn verify(&self) -> Result<()> {
        if self.schema_kind != PATHB_CROSS_RUN_MANIFEST_SCHEMA
            || !matches!(self.schema_version.as_str(), "5.0.0" | "6.0.0")
            || self.body.site_key.trim().is_empty()
            || self.body.common_replica_seeds.len() < 5
            || self
                .body
                .common_replica_seeds
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || self
                .body
                .source_confirmation_handoff
                .confirmation_replica_seeds
                != self.body.common_replica_seeds
            || self
                .body
                .source_confirmation_handoff
                .supplied_tracker_sha256
                != self.body.candidate_tracker_sha256
            || self.body.lane_evidence.len() != self.body.common_replica_seeds.len() * 3
        {
            bail!("invalid cross-run Path-B aggregation manifest design");
        }
        self.body.source_confirmation_handoff.verify()?;
        for (digest, field) in [
            (
                &self.aggregation_manifest_sha256,
                "aggregation_manifest_sha256",
            ),
            (&self.body.fragment_panel_sha256, "fragment_panel_sha256"),
            (
                &self.body.candidate_tracker_sha256,
                "candidate_tracker_sha256",
            ),
            (
                &self.body.matched_physical_state_sha256,
                "matched_physical_state_sha256",
            ),
            (&self.body.engine_sha256, "engine_sha256"),
            (&self.body.pathb_config_sha256, "pathb_config_sha256"),
            (
                &self.body.timebase_context_sha256,
                "timebase_context_sha256",
            ),
            (
                &self.body.matched_control_config_sha256,
                "matched_control_config_sha256",
            ),
            (&self.body.registry_policy_sha256, "registry_policy_sha256"),
        ] {
            require_sha256(digest, field)?;
        }
        let required_lanes = BTreeSet::from([
            ControlLane::NoProbe,
            ControlLane::HypothesisFragment,
            ControlLane::OrthogonalChemicalControl,
        ]);
        if self
            .body
            .topology_sha256_by_lane
            .keys()
            .copied()
            .collect::<BTreeSet<_>>()
            != required_lanes
            || self
                .body
                .calibration_population_sha256_by_probe_lane
                .keys()
                .copied()
                .collect::<BTreeSet<_>>()
                != BTreeSet::from([
                    ControlLane::HypothesisFragment,
                    ControlLane::OrthogonalChemicalControl,
                ])
            || self
                .body
                .probe_microstate_pair_by_lane
                .keys()
                .copied()
                .collect::<BTreeSet<_>>()
                != BTreeSet::from([
                    ControlLane::HypothesisFragment,
                    ControlLane::OrthogonalChemicalControl,
                ])
        {
            bail!("cross-run manifest does not declare exactly three physical lanes");
        }
        if self.schema_version == "5.0.0" {
            if !self.body.tracker_realizations_by_lane.is_empty() {
                bail!("legacy v5 manifest cannot declare lane tracker realizations");
            }
        } else {
            if self.body.tracker_realizations_by_lane.keys().copied().collect::<BTreeSet<_>>() != required_lanes {
                bail!("v6 manifest requires exactly three topology-bound tracker realizations");
            }
            let raw_path = self.body.source_confirmation_handoff.handoff_root.join("native_pathb_confirmation_handoff.json");
            let mut file = open_regular_nofollow(&raw_path)?;
            let mut bytes = Vec::new();
            file.read_to_end(&mut bytes)?;
            if format!("{:x}", Sha256::digest(&bytes)) != self.body.source_confirmation_handoff.handoff_root_raw_file_sha256 {
                bail!("confirmation handoff changed during lane tracker verification");
            }
            let handoff: crate::pathb_discovery::NativePathBConfirmationHandoff = serde_json::from_slice(&bytes)?;
            handoff.verify()?;
            if handoff.handoff_sha256 != self.body.source_confirmation_handoff.handoff_sha256
                || handoff.body.supplied_tracker_sha256 != self.body.candidate_tracker_sha256 {
                bail!("lane tracker reference differs from immutable confirmation handoff");
            }
            for (lane, realization) in &self.body.tracker_realizations_by_lane {
                realization.verify(&self.body.topology_sha256_by_lane[lane], &handoff.body.supplied_tracker)?;
            }
        }
        for identity in self.body.probe_microstate_pair_by_lane.values() {
            identity.verify()?;
        }
        for digest in self.body.topology_sha256_by_lane.values().chain(
            self.body
                .calibration_population_sha256_by_probe_lane
                .values(),
        ) {
            require_sha256(digest, "cross-run lane authority SHA256")?;
        }
        let mut previous_path: Option<&Path> = None;
        let mut paths = BTreeSet::new();
        for entry in &self.body.lane_evidence {
            require_sha256(&entry.lane_evidence_sha256, "lane_evidence_sha256")?;
            validate_absolute_normalized_regular_path_lexical(&entry.lane_evidence_path)?;
            if previous_path.is_some_and(|previous| previous >= entry.lane_evidence_path.as_path())
                || !paths.insert(entry.lane_evidence_path.as_path())
            {
                bail!("cross-run lane evidence paths are duplicate or not canonically sorted");
            }
            previous_path = Some(&entry.lane_evidence_path);
        }
        if canonical_sha256(&self.body)? != self.aggregation_manifest_sha256 {
            bail!("cross-run aggregation manifest content hash mismatch");
        }
        Ok(())
    }
}

/// Strictly reconstructed cross-run source population.
#[derive(Debug, Clone)]
pub struct LoadedPathBCrossRunAggregation {
    pub manifest: PathBCrossRunAggregationManifest,
    pub lane_evidence: Vec<NativePathBLaneEvidence>,
}

/// Load a canonical cross-run manifest and only its explicitly hash-bound,
/// non-symlink lane documents. This is the sole admissible input authority for
/// matched Path-B inference.
pub fn load_pathb_cross_run_aggregation_manifest(
    manifest_path: &Path,
) -> Result<LoadedPathBCrossRunAggregation> {
    validate_absolute_normalized_regular_path_lexical(manifest_path)?;
    let mut file = open_regular_nofollow(manifest_path)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    if bytes.is_empty() || bytes.last() != Some(&b'\n') {
        bail!("cross-run aggregation manifest is not a complete durable JSON line");
    }
    let manifest: PathBCrossRunAggregationManifest =
        serde_json::from_slice(&bytes).context("decode cross-run aggregation manifest")?;
    manifest.verify()?;
    if canonical_json_line(&manifest)? != bytes {
        bail!("cross-run aggregation manifest bytes are not canonical");
    }
    let mut evidence = Vec::with_capacity(manifest.body.lane_evidence.len());
    for entry in &manifest.body.lane_evidence {
        let lane = load_native_pathb_lane_evidence(&entry.lane_evidence_path)?;
        if lane.lane_evidence_sha256 != entry.lane_evidence_sha256 {
            bail!("lane evidence content hash differs from cross-run manifest");
        }
        evidence.push(lane);
    }
    verify_cross_run_lane_population(&manifest, &evidence)?;
    Ok(LoadedPathBCrossRunAggregation {
        manifest,
        lane_evidence: evidence,
    })
}

fn verify_cross_run_lane_population(
    manifest: &PathBCrossRunAggregationManifest,
    evidence: &[NativePathBLaneEvidence],
) -> Result<()> {
    if evidence.len() != manifest.body.lane_evidence.len() {
        bail!("loaded lane evidence population differs from manifest");
    }
    let mut lane_seeds: BTreeMap<ControlLane, BTreeSet<u64>> = BTreeMap::new();
    let mut run_units = BTreeSet::new();
    let mut observed_engine = BTreeSet::new();
    let mut topology_by_lane: BTreeMap<ControlLane, BTreeSet<String>> = BTreeMap::new();
    let mut calibration_by_lane: BTreeMap<ControlLane, BTreeSet<String>> = BTreeMap::new();
    let mut microstate_pair_by_lane: BTreeMap<ControlLane, BTreeSet<String>> = BTreeMap::new();
    let mut matched_physical_states = BTreeSet::new();
    let first = evidence
        .first()
        .context("cross-run evidence population is empty")?;
    let baseline = &first.body.runtime_request.body;
    for lane_evidence in evidence {
        lane_evidence.verify()?;
        let request = &lane_evidence.body.runtime_request.body;
        if request.pathb_config != baseline.pathb_config
            || request.timebase_context != baseline.timebase_context
            || request.matched_control_config != baseline.matched_control_config
            || request.component_correspondence_policy != baseline.component_correspondence_policy
            || request.registry_policy != baseline.registry_policy
            || request.lane_policy.site_key != baseline.lane_policy.site_key
            || request.lane_policy.selection_mode != baseline.lane_policy.selection_mode
            || (manifest.schema_version == "5.0.0" && request.lane_policy.target_residue_ids != baseline.lane_policy.target_residue_ids)
            || request.lane_policy.minimum_target_residue_overlap_count
                != baseline.lane_policy.minimum_target_residue_overlap_count
            || request
                .lane_policy
                .minimum_target_residue_overlap_fraction
                .to_bits()
                != baseline
                    .lane_policy
                    .minimum_target_residue_overlap_fraction
                    .to_bits()
            || request.lane_policy.coordinate_cavity_config
                != baseline.lane_policy.coordinate_cavity_config
            || (manifest.schema_version == "5.0.0" && request.lane_policy.candidate_tracker != baseline.lane_policy.candidate_tracker)
            || request.lane_policy.fragment_panel_sha256 != manifest.body.fragment_panel_sha256
        {
            bail!("cross-run lane documents do not share one predeclared estimand/design");
        }
        let surveillance = lane_evidence
            .body
            .confirmation_surveillance
            .as_ref()
            .context("cross-run Path-B confirmation lacks fixed-cadence time-at-risk")?;
        surveillance.verify()?;
        let lane = surveillance.control_lane;
        let expected_tracker = if manifest.schema_version == "6.0.0" {
            &manifest.body.tracker_realizations_by_lane.get(&lane)
                .context("surveillance lane has no predeclared tracker realization")?.candidate_tracker
        } else {
            request.lane_policy.candidate_tracker.as_ref()
                .context("cross-run request has no candidate tracker")?
        };
        if surveillance.candidate_tracker_sha256 != expected_tracker.tracker_sha256
            || surveillance.candidate_tracker != *expected_tracker
            || request.lane_policy.candidate_tracker.as_ref() != Some(expected_tracker)
            || (manifest.schema_version == "5.0.0" && expected_tracker.tracker_sha256 != manifest.body.candidate_tracker_sha256)
        {
            bail!("cross-run surveillance candidate tracker differs from its manifest lane authority");
        }
        let seed = surveillance.replica_seed;
        if !run_units.insert((surveillance.run_id.clone(), surveillance.replica_id.clone()))
            || !lane_seeds.entry(lane).or_default().insert(seed)
        {
            bail!("cross-run evidence contains repeated run units or lane/seed pseudoreplication");
        }
        observed_engine.insert(surveillance.engine_sha256.clone());
        matched_physical_states.insert(surveillance.matched_physical_state_sha256.clone());
        topology_by_lane
            .entry(lane)
            .or_default()
            .insert(surveillance.topology_sha256.clone());
        for observation in &lane_evidence.body.ghost_measurements {
            if observation.ghost.engine_sha256 != surveillance.engine_sha256
                || observation.ghost.topology_sha256 != surveillance.topology_sha256
                || observation.ghost.run_id != surveillance.run_id
                || observation.ghost.replica_id != surveillance.replica_id
                || observation.ghost.replica_seed != surveillance.replica_seed
                || observation.control_lane != lane
            {
                bail!("Ghost event identity differs from fixed-cadence lane authority");
            }
        }
        match lane {
            ControlLane::NoProbe => {
                if lane_evidence
                    .body
                    .fragment_observations
                    .iter()
                    .any(|observation| {
                        observation.fragment.endpoint_a_identity.is_some()
                            || observation.fragment.endpoint_b_identity.is_some()
                            || observation.fragment.identity.is_some()
                    })
                {
                    bail!("no-probe cross-run lane contains a chemical microstate identity");
                }
            }
            ControlLane::HypothesisFragment | ControlLane::OrthogonalChemicalControl => {
                for observation in &lane_evidence.body.fragment_observations {
                    let endpoint_a = observation
                        .fragment
                        .endpoint_a_identity
                        .clone()
                        .context("probe observation lacks endpoint A microstate identity")?;
                    let endpoint_b = observation
                        .fragment
                        .endpoint_b_identity
                        .clone()
                        .context("probe observation lacks endpoint B microstate identity")?;
                    let pair = ProbeLaneMicrostatePairIdentity {
                        identity_sha256: canonical_sha256(&(&endpoint_a, &endpoint_b))?,
                        endpoint_a,
                        endpoint_b,
                    };
                    pair.verify()?;
                    microstate_pair_by_lane
                        .entry(lane)
                        .or_default()
                        .insert(pair.identity_sha256);
                }
            }
        }
        if let Some(calibration) = &lane_evidence
            .body
            .source_authority
            .chemical_potential_calibration_population_sha256
        {
            calibration_by_lane
                .entry(lane)
                .or_default()
                .insert(calibration.clone());
        }
    }
    let required_lanes = [
        ControlLane::NoProbe,
        ControlLane::HypothesisFragment,
        ControlLane::OrthogonalChemicalControl,
    ];
    let expected_seeds = manifest
        .body
        .common_replica_seeds
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    for lane in required_lanes {
        if lane_seeds.get(&lane) != Some(&expected_seeds) {
            bail!("cross-run lanes do not have the exact same independent seed identities");
        }
        let topologies = topology_by_lane
            .get(&lane)
            .context("cross-run lane has no topology authority")?;
        if topologies.len() != 1
            || topologies.iter().next() != manifest.body.topology_sha256_by_lane.get(&lane)
        {
            bail!("cross-run topology identity differs from manifest or varies within a lane");
        }
    }
    if observed_engine != BTreeSet::from([manifest.body.engine_sha256.clone()]) {
        bail!("cross-run lane engine identity differs from aggregation manifest");
    }
    if matched_physical_states
        != BTreeSet::from([manifest.body.matched_physical_state_sha256.clone()])
    {
        bail!("cross-run lanes do not share the manifest-bound physical thermodynamic/phase state");
    }
    for lane in [
        ControlLane::HypothesisFragment,
        ControlLane::OrthogonalChemicalControl,
    ] {
        let calibrations = calibration_by_lane
            .get(&lane)
            .context("probe lane lacks chemical-potential calibration identity")?;
        if calibrations.len() != 1
            || calibrations.iter().next()
                != manifest
                    .body
                    .calibration_population_sha256_by_probe_lane
                    .get(&lane)
        {
            bail!("probe-lane calibration identity differs from manifest or across seeds");
        }
        let identities = microstate_pair_by_lane
            .get(&lane)
            .context("probe lane lacks explicit endpoint-pair identity")?;
        let declared = manifest
            .body
            .probe_microstate_pair_by_lane
            .get(&lane)
            .context("manifest lacks probe endpoint-pair identity")?;
        if identities != &BTreeSet::from([declared.identity_sha256.clone()]) {
            bail!("probe-lane endpoint-pair identity differs from manifest or across seeds");
        }
    }
    if canonical_sha256(&baseline.pathb_config)? != manifest.body.pathb_config_sha256
        || baseline.timebase_context.context_sha256 != manifest.body.timebase_context_sha256
        || canonical_sha256(&baseline.matched_control_config)?
            != manifest.body.matched_control_config_sha256
        || canonical_sha256(&baseline.registry_policy)? != manifest.body.registry_policy_sha256
        || baseline.lane_policy.site_key != manifest.body.site_key
        || baseline.lane_policy.selection_mode != NativePathBSelectionMode::TargetResidueHypothesis
        || expected_seeds.len() < baseline.matched_control_config.minimum_independent_seeds
    {
        bail!("cross-run manifest configuration or seed gate differs from lane evidence");
    }
    Ok(())
}

fn validate_absolute_normalized_regular_path_lexical(path: &Path) -> Result<()> {
    use std::path::Component;
    if !path.is_absolute()
        || path
            .components()
            .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
    {
        bail!("evidence path must be absolute and lexically normalized");
    }
    Ok(())
}

fn open_regular_nofollow(path: &Path) -> Result<std::fs::File> {
    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("inspect immutable evidence {}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        bail!("immutable evidence path must be a non-symlink regular file");
    }
    OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(path)
        .with_context(|| format!("open immutable evidence {}", path.display()))
}

fn evaluate_target_component_correspondence(
    emitted_component_sha256: &BTreeSet<String>,
    component_correspondence: &ComponentCorrespondenceLedger,
) -> Result<CandidateClaimEvaluation> {
    let descriptor_by_component = component_correspondence
        .descriptors
        .iter()
        .map(|descriptor| {
            (
                descriptor.source_component_sha256.clone(),
                descriptor.descriptor_sha256.clone(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let status_by_descriptor = component_correspondence
        .assignments
        .iter()
        .map(|assignment| {
            (
                assignment.descriptor_sha256.clone(),
                assignment.status.clone(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut recurrent = BTreeSet::new();
    let mut ambiguous = BTreeSet::new();
    let mut unmatched = BTreeSet::new();
    for component_sha256 in emitted_component_sha256 {
        let descriptor_sha256 = descriptor_by_component
            .get(component_sha256)
            .with_context(|| {
                format!(
                    "emitted tracker component {component_sha256} lacks its chain/spatial descriptor"
                )
            })?;
        match status_by_descriptor
            .get(descriptor_sha256)
            .context("component descriptor lacks a correspondence assignment")?
        {
            ComponentCorrespondenceStatus::Recurrent {
                correspondence_sha256,
                ..
            } => {
                recurrent.insert(correspondence_sha256.clone());
            }
            ComponentCorrespondenceStatus::Ambiguous { .. } => {
                ambiguous.insert(descriptor_sha256.clone());
            }
            ComponentCorrespondenceStatus::Unmatched { .. } => {
                unmatched.insert(descriptor_sha256.clone());
            }
        }
    }
    let mut reason_codes = BTreeSet::new();
    if emitted_component_sha256.is_empty() {
        reason_codes.insert("ZERO_TARGET_DETECTIONS".to_string());
    }
    if !ambiguous.is_empty() {
        reason_codes.insert("AMBIGUOUS_TARGET_COMPONENT_CORRESPONDENCE".to_string());
    }
    if !unmatched.is_empty() {
        reason_codes.insert("UNMATCHED_TARGET_COMPONENT_CORRESPONDENCE".to_string());
    }
    if recurrent.len() != 1 {
        reason_codes.insert("TARGET_DOES_NOT_RESOLVE_TO_ONE_RECURRENT_COMPONENT".to_string());
    }
    let emitted_component_sha256 = emitted_component_sha256.iter().cloned().collect::<Vec<_>>();
    let recurrent_correspondence_sha256 = recurrent.into_iter().collect::<Vec<_>>();
    let ambiguous_descriptor_sha256 = ambiguous.into_iter().collect::<Vec<_>>();
    let unmatched_descriptor_sha256 = unmatched.into_iter().collect::<Vec<_>>();
    let reason_codes = reason_codes.into_iter().collect::<Vec<_>>();
    let status = if reason_codes.is_empty() {
        CandidateClaimEvaluationStatus::Eligible
    } else {
        CandidateClaimEvaluationStatus::NotEvaluable
    };
    let candidate_append_allowed = status == CandidateClaimEvaluationStatus::Eligible;
    let evaluation_sha256 = canonical_sha256(&(
        status,
        &emitted_component_sha256,
        &recurrent_correspondence_sha256,
        &ambiguous_descriptor_sha256,
        &unmatched_descriptor_sha256,
        &reason_codes,
        candidate_append_allowed,
    ))?;
    let evaluation = CandidateClaimEvaluation {
        schema_kind: "prism_pathb_candidate_claim_evaluation".to_string(),
        schema_version: "1.0.0".to_string(),
        evaluation_sha256,
        status,
        emitted_component_sha256,
        recurrent_correspondence_sha256,
        ambiguous_descriptor_sha256,
        unmatched_descriptor_sha256,
        reason_codes,
        candidate_append_allowed,
    };
    evaluation.verify()?;
    Ok(evaluation)
}

fn partition_fragment_observations(
    aggregation_manifest_sha256: &str,
    pathb_model_sha256: &str,
    observations: Vec<FragmentStateObservation>,
    assigned_lineages: &BTreeSet<String>,
) -> Result<(Vec<FragmentStateObservation>, FragmentPathBAssignmentLedger)> {
    require_sha256(aggregation_manifest_sha256, "aggregation manifest")?;
    require_sha256(pathb_model_sha256, "Path-B model")?;
    let mut observations = observations;
    observations.sort_by(|left, right| left.ghost_lineage_sha256.cmp(&right.ghost_lineage_sha256));
    if observations
        .windows(2)
        .any(|pair| pair[0].ghost_lineage_sha256 == pair[1].ghost_lineage_sha256)
    {
        bail!("fragment observation population contains duplicate global Ghost lineage");
    }
    for observation in &observations {
        require_sha256(
            &observation.ghost_lineage_sha256,
            "fragment observation Ghost lineage",
        )?;
    }
    let source_population_sha256 = canonical_sha256(&observations)?;
    let source_observation_count = observations.len();
    let (assigned, unassigned_observations): (Vec<_>, Vec<_>) = observations
        .into_iter()
        .partition(|observation| assigned_lineages.contains(&observation.ghost_lineage_sha256));
    let assigned_ghost_lineage_sha256 = assigned
        .iter()
        .map(|observation| observation.ghost_lineage_sha256.clone())
        .collect::<Vec<_>>();
    let ledger_sha256 = canonical_sha256(&(
        aggregation_manifest_sha256,
        pathb_model_sha256,
        &source_population_sha256,
        source_observation_count,
        &assigned_ghost_lineage_sha256,
        &unassigned_observations,
    ))?;
    let ledger = FragmentPathBAssignmentLedger {
        schema_kind: "prism_fragment_pathb_assignment_ledger".to_string(),
        schema_version: "1.0.0".to_string(),
        ledger_sha256,
        aggregation_manifest_sha256: aggregation_manifest_sha256.to_string(),
        pathb_model_sha256: pathb_model_sha256.to_string(),
        source_population_sha256,
        source_observation_count,
        assigned_ghost_lineage_sha256,
        unassigned_observations,
    };
    ledger.verify()?;
    Ok((assigned, ledger))
}

/// Pool independently materialized lane populations, fit one common native
/// state model, materialize fibers, classify the matched three-lane experiment,
/// and append a candidate revision. This is a cross-run stage: callers must
/// provide no-probe, hypothesis, and orthogonal endpoint evidence spanning the
/// common seed population (at least 15 lane-seed runs for the default gate).
#[cfg(any(feature = "gpu", feature = "cpu-analysis"))]
pub fn aggregate_cross_run_native_pathb_evidence(
    output_dir: &Path,
    loaded: LoadedPathBCrossRunAggregation,
    runtime_provenance: &RuntimeProvenance,
) -> Result<PathBEvidenceEmission> {
    loaded.manifest.verify()?;
    verify_cross_run_lane_population(&loaded.manifest, &loaded.lane_evidence)?;
    let aggregation_manifest_sha256 = loaded.manifest.aggregation_manifest_sha256.clone();
    let source_confirmation_handoff = loaded.manifest.body.source_confirmation_handoff.clone();
    let baseline = &loaded
        .lane_evidence
        .first()
        .context("cross-run aggregation has no lane documents")?
        .body
        .runtime_request
        .body;
    let pathb_config = baseline.pathb_config.clone();
    let control_config = baseline.matched_control_config.clone();
    let registry_policy = baseline.registry_policy.clone();
    require_output_targets_absent(output_dir)?;
    validate_durable_registry_path(output_dir, &registry_policy.durable_registry_path)?;
    std::fs::create_dir_all(output_dir)?;
    let mut lattice_observations = Vec::new();
    let mut lineages = Vec::new();
    let mut exact_connectivities = Vec::new();
    let mut surveillance_ledgers = Vec::new();
    let mut ghost_measurements = Vec::new();
    let mut fragment_observations = Vec::new();
    let mut component_descriptors = Vec::new();
    for lane in &loaded.lane_evidence {
        let measurement_by_lineage = lane
            .body
            .ghost_measurements
            .iter()
            .map(|row| Ok((row.ghost.lineage_sha256()?, row)))
            .collect::<Result<BTreeMap<_, _>>>()?;
        for observation in &lane.body.lattice_observations {
            let measurement = measurement_by_lineage
                .get(&observation.ghost_lineage_sha256)
                .context("cross-run lattice observation lacks exact Ghost occurrence")?;
            observation.verify_for(&measurement.ghost)?;
            lattice_observations.push(observation.clone());
            lineages.push(measurement.ghost.clone());
        }
        exact_connectivities.push(lane.body.exact_connectivity.clone());
        component_descriptors.extend(lane.body.component_descriptors.clone());
        surveillance_ledgers.push(
            lane.body
                .confirmation_surveillance
                .clone()
                .context("cross-run Path-B fitting requires nonempty complete surveillance")?,
        );
        ghost_measurements.extend(lane.body.ghost_measurements.clone());
        fragment_observations.extend(lane.body.fragment_observations.clone());
    }
    let mut contact_mappings_by_lane = BTreeMap::new();
    for (lane, realization) in &loaded.manifest.body.tracker_realizations_by_lane {
        contact_mappings_by_lane.insert(*lane,
            crate::component_correspondence::LaneContactMappingAuthority::from_topology_path(
                &realization.topology_path, &loaded.manifest.body.topology_sha256_by_lane[lane])?);
    }
    for measurement in &ghost_measurements {
        if !contact_mappings_by_lane.is_empty() {
            let mapping = contact_mappings_by_lane.get(&measurement.control_lane)
                .context("coordinate measurement has no declared lane topology")?;
            let frame = &measurement.structural_measurements.coordinate_frame;
            frame.validate_for(&measurement.ghost)?;
            if frame.topology_sha256 != mapping.topology_sha256
                || frame.atom_residue_mapping_sha256 != mapping.atom_analysis_residue_ids_sha256 {
                bail!("coordinate measurement differs from actual lane topology/mapping");
            }
        }
    }
    let component_correspondence = crate::component_correspondence::establish_component_correspondence_with_lane_mappings(
        component_descriptors,
        baseline.component_correspondence_policy.clone(),
        contact_mappings_by_lane,
    )?;
    component_correspondence.verify()?;
    let emitted_component_sha256 = surveillance_ledgers
        .iter()
        .flat_map(|ledger| ledger.rows.iter())
        .flat_map(|row| match &row.ghost_event {
            PathBGhostEventStatus::Emitted {
                exact_component_sha256,
                ..
            } => vec![exact_component_sha256.clone()],
            PathBGhostEventStatus::AmbiguousEmissions {
                exact_component_sha256,
                ..
            } => exact_component_sha256.clone(),
            PathBGhostEventStatus::NotEmitted { .. } => Vec::new(),
        })
        .collect::<BTreeSet<_>>();
    let claim_evaluation = evaluate_target_component_correspondence(
        &emitted_component_sha256,
        &component_correspondence,
    )?;
    let population = lineages.len() as u64;
    let conservation = GhostLatticeConservation {
        produced: population,
        persisted: population,
        decoded: population,
        lattice_nodes: population,
        lattice_manifest_sha256: aggregation_manifest_sha256.clone(),
    };
    conservation.validate()?;
    let observed_atom_counts = crate::pathb_chemical_state::collect_lane_topology_atom_counts(
        ghost_measurements.iter().map(|measurement| {
            measurement.structural_measurements.coordinate_frame.validate_for(&measurement.ghost)?;
            Ok((measurement.control_lane, measurement.ghost.topology_sha256.clone(),
                measurement.structural_measurements.coordinate_frame.atom_count))
        }).collect::<Result<Vec<_>>>()?,
    )?;
    let mut pathb = estimate_native_pathb_from_observations_with_timebase(
        &lattice_observations,
        &lineages,
        &exact_connectivities,
        &surveillance_ledgers,
        &conservation,
        &pathb_config,
        &baseline.timebase_context,
    )?;
    pathb.bind_observed_atom_counts_by_lane_topology(observed_atom_counts)?;
    let (open_basin, ghost_observations) = attach_coordinate_open_basin(
        &pathb,
        pathb_config.open_state_count,
        pathb_config.minimum_independent_seeds,
        ghost_measurements,
    )?;
    let assigned_lineages = ghost_observations
        .iter()
        .map(|observation| observation.ghost.lineage_sha256())
        .collect::<Result<BTreeSet<_>>>()?;
    let (assigned_fragment_observations, fragment_assignment) = partition_fragment_observations(
        &aggregation_manifest_sha256,
        &pathb.model.model_sha256,
        fragment_observations,
        &assigned_lineages,
    )?;
    let bundle = materialize_native_pathb_fibers(
        pathb.clone(),
        ghost_observations,
        assigned_fragment_observations,
    )?;
    bundle.verify()?;
    let endpoint_fibers = bundle
        .fibers
        .iter()
        .filter(|fiber| {
            fiber.endpoint_eligibility == EndpointEligibility::PhysicalEndpoint
                && (fiber.body.control_lane == ControlLane::NoProbe
                    || fiber.body.fragment.lambda == 1.0)
        })
        .cloned()
        .collect::<Vec<_>>();
    let matched_control = classify_matched_controls(&endpoint_fibers, &control_config)?;
    let lane_evidence_population_sha256 = canonical_sha256(
        &loaded
            .lane_evidence
            .iter()
            .map(|lane| lane.lane_evidence_sha256.as_str())
            .collect::<Vec<_>>(),
    )?;
    let candidate = if claim_evaluation.candidate_append_allowed {
        let candidate_objectives = candidate_objectives_from_matched_control(
            &matched_control,
            &registry_policy.objective_definitions,
        )?;
        let target_correspondence_sha256 = claim_evaluation
            .recurrent_correspondence_sha256
            .first()
            .context("eligible claim lacks one recurrent component identity")?
            .clone();
        Some(CandidateRecord::from_matched_control_revision(
            registry_policy.candidate_id.clone(),
            target_correspondence_sha256,
            registry_policy.revises_candidate_sha256.clone(),
            &matched_control,
            candidate_objectives,
            BTreeMap::from([
                (
                    "aggregation_manifest".to_string(),
                    aggregation_manifest_sha256.clone(),
                ),
                (
                    "lane_evidence_population".to_string(),
                    lane_evidence_population_sha256,
                ),
                (
                    "native_pathb_model".to_string(),
                    pathb.model.model_sha256.clone(),
                ),
                (
                    "component_correspondence_ledger".to_string(),
                    component_correspondence.ledger_sha256.clone(),
                ),
                (
                    "fragment_pathb_assignment_ledger".to_string(),
                    fragment_assignment.ledger_sha256.clone(),
                ),
                (
                    "candidate_claim_evaluation".to_string(),
                    claim_evaluation.evaluation_sha256.clone(),
                ),
                (
                    "open_basin_binding".to_string(),
                    open_basin.binding_sha256.clone(),
                ),
                (
                    "matched_control_result".to_string(),
                    matched_control.result_sha256.clone(),
                ),
            ]),
        )?)
    } else {
        None
    };

    let pathb_path = output_dir.join("native_pathb_chemical_state.json");
    write_create_new(
        &pathb_path,
        &canonical_json_line(&NativePathBEvidenceDocument {
            pathb: &pathb,
            open_basin: &open_basin,
            component_correspondence: &component_correspondence,
        })?,
    )?;
    let fiber_path = output_dir.join("chemical_state_fibers.jsonl");
    write_create_new(&fiber_path, &canonical_fiber_jsonl(&bundle.fibers)?)?;
    // Preserve the verified population, model and lane topology receipt as the
    // exact typed object consumed by DSTW's existing_bundle input. The JSONL
    // remains available; consumers need not reconstruct a bundle from pieces.
    let fiber_bundle_path = output_dir.join("native_pathb_fiber_bundle.json");
    write_create_new(&fiber_bundle_path, &canonical_json_line(&bundle)?)?;
    let control_path = output_dir.join("matched_control_discriminator.json");
    write_create_new(&control_path, &canonical_json_line(&matched_control)?)?;
    let preservation_path = output_dir.join("native_pathb_population_preservation.json");
    write_create_new(
        &preservation_path,
        &canonical_json_line(&PopulationPreservationDocument {
            claim_evaluation: &claim_evaluation,
            fragment_assignment: &fragment_assignment,
            component_correspondence: &component_correspondence,
        })?,
    )?;

    // Durable scientific evidence is committed before the cross-run champion
    // registry is mutated. A failed raw/QNANL write can therefore never append
    // a candidate whose supporting run evidence was not durably materialized.
    let mut artifacts = vec![
        emit_qnanl_exact(
            &fiber_bundle_path,
            QnanlArtifactFamily::ChemicalStateFiber,
            &[(
                EVENT_CHEMICAL_STATE_FIBER_MATERIALIZED,
                PRISMCODE_OPCODE_CHEMICAL_STATE_FIBER_MATERIALIZED,
                vec![
                    "fiber_population_sha256",
                    "complete_fiber_population",
                    "native_pathb_model_and_conservation_receipt",
                    "lane_topology_atom_populations",
                ],
            )],
            runtime_provenance,
            "prism_pathb_cross_run_aggregator:native_pathb_fiber_bundle",
        )?,
        emit_qnanl_exact(
            &pathb_path,
            QnanlArtifactFamily::PathBChemicalState,
            &[(
                EVENT_PATHB_CHEMICAL_STATE_ASSIGNED,
                PRISMCODE_OPCODE_PATHB_CHEMICAL_STATE_ASSIGNED,
                vec![
                    "ghost_record_sha256",
                    "control_lane",
                    "lane_specific_reversible_operator",
                    "metastable_state_id",
                    "transition_probability",
                    "forward_committor",
                    "component_correspondence",
                    "ambiguous_unmatched_preserved",
                ],
            )],
            runtime_provenance,
            "prism_pathb_cross_run_aggregator:native_pathb_assignment",
        )?,
        emit_qnanl_exact(
            &fiber_path,
            QnanlArtifactFamily::ChemicalStateFiber,
            &[(
                EVENT_CHEMICAL_STATE_FIBER_MATERIALIZED,
                PRISMCODE_OPCODE_CHEMICAL_STATE_FIBER_MATERIALIZED,
                vec![
                    "ghost_byte_lineage",
                    "fragment_state",
                    "pathb_projection",
                    "uncertainty",
                ],
            )],
            runtime_provenance,
            "prism_pathb_cross_run_aggregator:chemical_state_fiber_materialization",
        )?,
        emit_qnanl_exact(
            &control_path,
            QnanlArtifactFamily::MatchedControlDiscriminator,
            &[(
                EVENT_MATCHED_CONTROL_CLASSIFIED,
                PRISMCODE_OPCODE_MATCHED_CONTROL_CLASSIFIED,
                vec![
                    "no_probe",
                    "hypothesis_fragment",
                    "orthogonal_control",
                    "common_seeds",
                ],
            )],
            runtime_provenance,
            "prism_pathb_cross_run_aggregator:matched_control_classification",
        )?,
        emit_qnanl_exact(
            &preservation_path,
            QnanlArtifactFamily::PathBChemicalState,
            &[(
                EVENT_PATHB_CHEMICAL_STATE_ASSIGNED,
                PRISMCODE_OPCODE_PATHB_CHEMICAL_STATE_ASSIGNED,
                vec![
                    "complete_component_population",
                    "ambiguous_unmatched_preserved",
                    "assigned_fragment_population",
                    "unassigned_fragment_population",
                    "candidate_append_allowed",
                ],
            )],
            runtime_provenance,
            "prism_pathb_cross_run_aggregator:population_preservation",
        )?,
    ];
    let (registry, durable_registry_sha256, registry_snapshot_path, registry_snapshot_sha256) =
        if let Some(candidate) = &candidate {
            commit_candidate_append_only(
                &registry_policy.durable_registry_path,
                registry_policy.objective_definitions.clone(),
                candidate.clone(),
            )?;
            let registry = load_append_only_registry(&registry_policy.durable_registry_path)?;
            registry.verify()?;
            let mut durable_registry_file =
                open_regular_nofollow(&registry_policy.durable_registry_path)?;
            let mut durable_registry_bytes = Vec::new();
            durable_registry_file.read_to_end(&mut durable_registry_bytes)?;
            let durable_registry_sha256 = sha256_bytes(&durable_registry_bytes);
            let registry_snapshot_path =
                output_dir.join("candidate_advancement_registry_snapshot.ndjson");
            write_create_new(&registry_snapshot_path, &durable_registry_bytes)?;
            if load_append_only_registry(&registry_snapshot_path)? != registry {
                bail!("immutable run registry snapshot differs from durable registry");
            }
            let registry_snapshot_sha256 = sha256_bytes(&durable_registry_bytes);
            artifacts.push(emit_qnanl_exact(
                &registry_snapshot_path,
                QnanlArtifactFamily::CandidateAdvancementRegistry,
                &[
                    (
                        EVENT_CANDIDATE_OBSERVED,
                        PRISMCODE_OPCODE_CANDIDATE_OBSERVED,
                        vec!["candidate_sha256", "complete_population"],
                    ),
                    (
                        EVENT_PARETO_CHAMPION_PROMOTED,
                        PRISMCODE_OPCODE_PARETO_CHAMPION_PROMOTED,
                        vec!["candidate_sha256", "objective_vector"],
                    ),
                    (
                        EVENT_PARETO_CHAMPION_SUPERSEDED,
                        PRISMCODE_OPCODE_PARETO_CHAMPION_SUPERSEDED,
                        vec!["superseded_candidate_sha256", "successor_candidate_sha256"],
                    ),
                ],
                runtime_provenance,
                "prism_pathb_cross_run_aggregator:candidate_advancement_registry",
            )?);
            (
                Some(registry),
                Some(durable_registry_sha256),
                Some(registry_snapshot_path),
                Some(registry_snapshot_sha256),
            )
        } else {
            (None, None, None, None)
        };
    let emission_sha256 = canonical_sha256(&PathBEvidenceReceiptBody {
        aggregation_manifest_sha256: &aggregation_manifest_sha256,
        source_confirmation_handoff: &source_confirmation_handoff,
        pathb_model_sha256: &pathb.model.model_sha256,
        component_correspondence_ledger_sha256: &component_correspondence.ledger_sha256,
        open_basin_binding_sha256: &open_basin.binding_sha256,
        matched_control_result_sha256: &matched_control.result_sha256,
        candidate_sha256: candidate
            .as_ref()
            .map(|record| record.candidate_sha256.as_str()),
        candidate_claim_evaluation_sha256: &claim_evaluation.evaluation_sha256,
        fragment_assignment_ledger_sha256: &fragment_assignment.ledger_sha256,
        durable_registry_sha256: durable_registry_sha256.as_deref(),
        registry_snapshot_sha256: registry_snapshot_sha256.as_deref(),
        artifacts: &artifacts,
    })?;
    Ok(PathBEvidenceEmission {
        aggregation_manifest_sha256,
        source_confirmation_handoff,
        pathb,
        component_correspondence,
        open_basin,
        fiber_count: bundle.fibers.len(),
        fragment_assignment,
        claim_evaluation,
        matched_control,
        candidate,
        registry,
        durable_registry_path: registry_policy.durable_registry_path.display().to_string(),
        durable_registry_sha256,
        registry_snapshot_path: registry_snapshot_path
            .as_ref()
            .map(|path| path.display().to_string()),
        registry_snapshot_sha256,
        artifacts,
        emission_sha256,
    })
}

/// Emit a typed EVIDENCE_NOT_FOUND receipt without inventing coordinates,
/// lambda states, Path-B assignments, fibers, or candidate advancement.
#[cfg(any(feature = "gpu", feature = "cpu-analysis"))]
pub fn emit_native_pathb_evidence_not_found(
    output_dir: &Path,
    request: &NativePathBRuntimeRequest,
    lineages: &[GhostByteLineage],
    conservation: &GhostLatticeConservation,
    exact_coordinate_frame_count: u64,
    exact_lambda_state_count: u64,
    mut reason_codes: Vec<String>,
    runtime_provenance: &RuntimeProvenance,
    source_stage: &str,
) -> Result<NativePathBAvailabilityReceipt> {
    request.verify()?;
    conservation.validate()?;
    if lineages.len() as u64 != conservation.lattice_nodes {
        bail!("Path-B availability receipt lineage count is not conserved");
    }
    let mut observed_stream_topology_sha256 = BTreeMap::new();
    for lineage in lineages {
        lineage.validate()?;
        match observed_stream_topology_sha256
            .insert(lineage.key.stream_id, lineage.topology_sha256.clone())
        {
            Some(previous) if previous != lineage.topology_sha256 => {
                bail!("one Ghost stream carries multiple topology identities")
            }
            _ => {}
        }
    }
    reason_codes.sort();
    reason_codes.dedup();
    if reason_codes.is_empty() || reason_codes.iter().any(|reason| reason.trim().is_empty()) {
        bail!("EVIDENCE_NOT_FOUND requires explicit non-empty reason codes");
    }
    std::fs::create_dir_all(output_dir)?;
    let raw_path = output_dir.join("native_pathb_availability.json");
    let packet_path = PathBuf::from(format!("{}.prismqnanl", raw_path.display()));
    let receipt_path = PathBuf::from(format!("{}.receipt.json", packet_path.display()));
    if raw_path.exists() || packet_path.exists() || receipt_path.exists() {
        bail!("refusing to overwrite prior native Path-B availability evidence");
    }
    let lane = &request.body.lane_policy;
    let body = NativePathBAvailabilityBody {
        runtime_request_sha256: &request.request_sha256,
        lattice_manifest_sha256: &conservation.lattice_manifest_sha256,
        stream_lanes: &lane.stream_lanes,
        site_key: &lane.site_key,
        required_stream_ids: &lane.required_stream_ids,
        observed_stream_topology_sha256: &observed_stream_topology_sha256,
        ghost_lineage_count: lineages.len() as u64,
        exact_coordinate_frame_count,
        exact_lambda_state_count,
        materialized_pathb_state_count: 0,
        materialized_fiber_count: 0,
        candidate_append_count: 0,
        reason_codes: &reason_codes,
        source_contract: "verified_ghost_lineage_runtime_inputs_required_no_fallback",
    };
    let receipt_sha256 = canonical_sha256(&body)?;
    let raw_document = serde_json::json!({
        "schema_kind": "prism_native_pathb_availability",
        "schema_version": "1.0.0",
        "status": "EVIDENCE_NOT_FOUND",
        "receipt_sha256": receipt_sha256,
        "body": body,
    });
    write_create_new(&raw_path, &canonical_json_line(&raw_document)?)?;
    let typed_artifact = emit_qnanl_exact(
        &raw_path,
        QnanlArtifactFamily::PathBChemicalState,
        &[],
        runtime_provenance,
        source_stage,
    )?;
    Ok(NativePathBAvailabilityReceipt {
        schema_kind: "prism_native_pathb_availability".to_string(),
        schema_version: "1.0.0".to_string(),
        status: "EVIDENCE_NOT_FOUND".to_string(),
        receipt_sha256,
        runtime_request_sha256: request.request_sha256.clone(),
        lattice_manifest_sha256: conservation.lattice_manifest_sha256.clone(),
        stream_lanes: lane.stream_lanes.clone(),
        site_key: lane.site_key.clone(),
        required_stream_ids: lane.required_stream_ids.clone(),
        observed_stream_topology_sha256,
        ghost_lineage_count: lineages.len() as u64,
        exact_coordinate_frame_count,
        exact_lambda_state_count,
        materialized_pathb_state_count: 0,
        materialized_fiber_count: 0,
        candidate_append_count: 0,
        reason_codes,
        source_contract: "verified_ghost_lineage_runtime_inputs_required_no_fallback".to_string(),
        typed_artifact,
    })
}

#[derive(Serialize)]
struct NativePathBEvidenceDocument<'a> {
    pathb: &'a NativePathBMaterialization,
    open_basin: &'a NativeOpenBasinBinding,
    component_correspondence: &'a ComponentCorrespondenceLedger,
}

#[derive(Serialize)]
struct PopulationPreservationDocument<'a> {
    claim_evaluation: &'a CandidateClaimEvaluation,
    fragment_assignment: &'a FragmentPathBAssignmentLedger,
    component_correspondence: &'a ComponentCorrespondenceLedger,
}

#[derive(Serialize)]
struct PathBEvidenceReceiptBody<'a> {
    aggregation_manifest_sha256: &'a str,
    source_confirmation_handoff: &'a PathBConfirmationHandoffIdentity,
    pathb_model_sha256: &'a str,
    component_correspondence_ledger_sha256: &'a str,
    open_basin_binding_sha256: &'a str,
    matched_control_result_sha256: &'a str,
    candidate_sha256: Option<&'a str>,
    candidate_claim_evaluation_sha256: &'a str,
    fragment_assignment_ledger_sha256: &'a str,
    durable_registry_sha256: Option<&'a str>,
    registry_snapshot_sha256: Option<&'a str>,
    artifacts: &'a [TypedEvidenceArtifact],
}

fn attach_coordinate_open_basin(
    pathb: &NativePathBMaterialization,
    open_state_count: usize,
    minimum_independent_seeds: usize,
    measurements: Vec<GhostFiberMeasurementObservation>,
) -> Result<(NativeOpenBasinBinding, Vec<GhostFiberObservation>)> {
    pathb.verify()?;
    if open_state_count == 0 || open_state_count >= pathb.model.body.states.len() {
        bail!("open-state declaration is incompatible with native Path-B state count");
    }
    let assignments = pathb
        .model
        .body
        .assignments
        .iter()
        .map(|assignment| (assignment.ghost_lineage_sha256.as_str(), assignment))
        .collect::<BTreeMap<_, _>>();
    let measurements = measurements
        .into_iter()
        .filter_map(|measurement| match measurement.ghost.lineage_sha256() {
            Ok(hash) if assignments.contains_key(hash.as_str()) => Some(Ok(measurement)),
            Ok(_) => None,
            Err(error) => Some(Err(error)),
        })
        .collect::<Result<Vec<_>>>()?;
    let mut seen = BTreeSet::new();
    let mut state_seed_values: BTreeMap<String, BTreeMap<u64, Vec<f64>>> = pathb
        .model
        .body
        .states
        .iter()
        .map(|state| (state.state_id.clone(), BTreeMap::new()))
        .collect();
    let mut method_id: Option<String> = None;
    let mut estimator_sha256: Option<String> = None;
    let mut topology_mapping_by_lane: BTreeMap<ControlLane, NativeOpenBasinLaneIdentity> = BTreeMap::new();
    let mut frame_population = Vec::with_capacity(measurements.len());
    for measurement in &measurements {
        measurement.ghost.validate()?;
        measurement
            .structural_measurements
            .coordinate_cavity_sidecar
            .verify()?;
        measurement
            .structural_measurements
            .coordinate_frame
            .validate_for(&measurement.ghost)?;
        let ghost_lineage_sha256 = measurement.ghost.lineage_sha256()?;
        if !seen.insert(ghost_lineage_sha256.clone()) {
            bail!("duplicate exact Ghost occurrence in structural measurement population");
        }
        let assignment = assignments.get(ghost_lineage_sha256.as_str()).context(
            "EVIDENCE_NOT_FOUND: structural measurement has no exact native Path-B assignment",
        )?;
        let volume = measurement
            .structural_measurements
            .pocket_volume_a3
            .as_ref()
            .context(
                "EVIDENCE_NOT_FOUND: exact coordinate cavity volume is required to define an open basin",
            )?;
        let cavity = &measurement
            .structural_measurements
            .coordinate_cavity_sidecar;
        if cavity.body.ghost_record_sha256 != measurement.ghost.record_sha256
            || cavity.body.stream_id != measurement.ghost.key.stream_id
            || cavity.body.step_idx != measurement.ghost.key.step_idx
            || cavity.sidecar_sha256
                != measurement
                    .structural_measurements
                    .coordinate_frame
                    .cavity_sidecar_record_sha256
            || cavity.body.coordinate_frame_sha256
                != measurement
                    .structural_measurements
                    .coordinate_frame
                    .coordinate_frame_sha256
            || cavity.body.trajectory_artifact_sha256
                != measurement
                    .structural_measurements
                    .coordinate_frame
                    .trajectory_artifact_sha256
            || cavity.body.trajectory_audit_sha256
                != measurement
                    .structural_measurements
                    .coordinate_frame
                    .trajectory_audit_sha256
            || cavity.body.topology_sha256
                != measurement
                    .structural_measurements
                    .coordinate_frame
                    .topology_sha256
            || cavity.body.atom_residue_mapping_sha256
                != measurement
                    .structural_measurements
                    .coordinate_frame
                    .atom_residue_mapping_sha256
            || volume.value.to_bits() != cavity.body.mean_volume_a3.to_bits()
            || volume.estimator_sha256 != cavity.body.estimator_sha256
        {
            bail!("coordinate cavity sidecar, frame binding, Ghost lineage, and scalar disagree");
        }
        if !volume.value.is_finite()
            || volume.value < 0.0
            || volume.method_id != "prism_multiresolution_coordinate_ligsite_v1"
        {
            bail!("coordinate cavity volume measurement is invalid");
        }
        require_sha256(&volume.estimator_sha256, "cavity estimator_sha256")?;
        require_sha256(
            &volume.source_artifact_sha256,
            "cavity source_artifact_sha256",
        )?;
        if volume.source_artifact_sha256
            != measurement
                .structural_measurements
                .coordinate_frame
                .cavity_sidecar_record_sha256
        {
            bail!("cavity measurement is not bound to its exact sidecar record");
        }
        require_consistent(&mut method_id, &volume.method_id, "cavity method")?;
        require_consistent(
            &mut estimator_sha256,
            &volume.estimator_sha256,
            "cavity estimator",
        )?;
        let coordinate = &measurement.structural_measurements.coordinate_frame;
        let identity = NativeOpenBasinLaneIdentity {
            topology_sha256: coordinate.topology_sha256.clone(),
            atom_residue_mapping_sha256: coordinate.atom_residue_mapping_sha256.clone(),
        };
        if topology_mapping_by_lane.insert(measurement.control_lane, identity.clone())
            .is_some_and(|previous| previous != identity) {
            bail!("one physical lane changed its exact topology/atom mapping");
        }
        let dominant_state = assignment
            .state_membership
            .iter()
            .max_by(|(id_a, p_a), (id_b, p_b)| p_a.total_cmp(p_b).then_with(|| id_b.cmp(id_a)))
            .map(|(state_id, _)| state_id)
            .context("native state assignment has no memberships")?;
        state_seed_values
            .get_mut(dominant_state)
            .context("dominant native state is undeclared")?
            .entry(measurement.ghost.replica_seed)
            .or_default()
            .push(volume.value);
        frame_population.push((
            ghost_lineage_sha256,
            measurement.structural_measurements.coordinate_frame.clone(),
        ));
    }
    if seen.len() != assignments.len() {
        bail!("structural measurement population does not conserve Path-B assignments");
    }
    frame_population.sort_by(|a, b| a.0.cmp(&b.0));
    let frame_population_sha256 = canonical_sha256(&frame_population)?;
    let state_count = state_seed_values.len();
    let familywise_alpha = 0.05;
    let mut state_summaries = Vec::with_capacity(state_count);
    for (state_id, by_seed) in state_seed_values {
        if by_seed.len() < minimum_independent_seeds.max(5) {
            bail!("state {state_id} lacks independent seeds for cavity uncertainty");
        }
        let observation_count = by_seed.values().map(Vec::len).sum();
        let seed_means = by_seed
            .values()
            .map(|values| values.iter().sum::<f64>() / values.len() as f64)
            .collect::<Vec<_>>();
        let mean = seed_means.iter().sum::<f64>() / seed_means.len() as f64;
        let variance = seed_means
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / (seed_means.len() - 1) as f64;
        let standard_error = (variance / seed_means.len() as f64).sqrt();
        let distribution = StudentsT::new(0.0, 1.0, (seed_means.len() - 1) as f64)?;
        let critical =
            distribution.inverse_cdf(1.0 - familywise_alpha / (2.0 * state_count as f64));
        state_summaries.push(StateCavityObservableSummary {
            state_id,
            observation_count,
            independent_seed_count: seed_means.len(),
            mean_a3: mean,
            standard_error_a3: standard_error,
            confidence_interval_lower_a3: mean - critical * standard_error,
            confidence_interval_upper_a3: mean + critical * standard_error,
        });
    }
    state_summaries.sort_by(|a, b| {
        a.mean_a3
            .total_cmp(&b.mean_a3)
            .then_with(|| a.state_id.cmp(&b.state_id))
    });
    let split = state_summaries.len() - open_state_count;
    let excluded_upper = state_summaries[..split]
        .iter()
        .map(|summary| summary.confidence_interval_upper_a3)
        .fold(f64::NEG_INFINITY, f64::max);
    let selected_lower = state_summaries[split..]
        .iter()
        .map(|summary| summary.confidence_interval_lower_a3)
        .fold(f64::INFINITY, f64::min);
    if selected_lower <= excluded_upper {
        bail!("coordinate cavity observable does not uncertainty-separate an open basin");
    }
    let mut open_state_ids = state_summaries[split..]
        .iter()
        .map(|summary| summary.state_id.clone())
        .collect::<Vec<_>>();
    open_state_ids.sort();
    let method_id = method_id.context("cavity method population is empty")?;
    let estimator_sha256 = estimator_sha256.context("cavity estimator population is empty")?;
    let unique_identities = topology_mapping_by_lane.values().cloned().collect::<BTreeSet<_>>();
    let (topology_sha256, mapping_sha256, schema_version) = if unique_identities.len() == 1 {
        let identity = unique_identities.iter().next().unwrap();
        let pair = (Some(identity.topology_sha256.clone()), Some(identity.atom_residue_mapping_sha256.clone()), "1.0.0");
        topology_mapping_by_lane.clear();
        pair
    } else if unique_identities.is_empty() {
        bail!("coordinate topology/mapping population is empty");
    } else {
        (None, None, "2.0.0")
    };
    let body = NativeOpenBasinBindingBody {
        pathb_model_sha256: &pathb.model.model_sha256,
        observable_id: "coordinate_cavity_volume_a3",
        measurement_method_id: &method_id,
        estimator_sha256: &estimator_sha256,
        topology_sha256: &topology_sha256,
        atom_residue_mapping_sha256: &mapping_sha256,
        topology_mapping_by_lane: &topology_mapping_by_lane,
        frame_population_sha256: &frame_population_sha256,
        familywise_confidence_level: 0.95,
        selection_rule: "highest_seed_balanced_state_mean_with_bonferroni_t_interval_separation",
        open_state_ids: &open_state_ids,
        state_summaries: &state_summaries,
    };
    let definition_sha256 = canonical_sha256(&body)?;
    let binding = NativeOpenBasinBinding {
        schema_kind: "prism_native_pathb_open_basin_binding".to_string(),
        schema_version: schema_version.to_string(),
        binding_sha256: definition_sha256.clone(),
        pathb_model_sha256: pathb.model.model_sha256.clone(),
        observable_id: "coordinate_cavity_volume_a3".to_string(),
        measurement_method_id: method_id,
        estimator_sha256,
        topology_sha256,
        atom_residue_mapping_sha256: mapping_sha256,
        topology_mapping_by_lane,
        frame_population_sha256,
        familywise_confidence_level: 0.95,
        selection_rule: "highest_seed_balanced_state_mean_with_bonferroni_t_interval_separation"
            .to_string(),
        open_state_ids: open_state_ids.clone(),
        state_summaries,
    };
    binding.verify()?;

    let mut observations = Vec::with_capacity(measurements.len());
    for measurement in measurements {
        let measurement_lineage_sha256 = measurement.ghost.lineage_sha256()?;
        let assignment = assignments
            .get(measurement_lineage_sha256.as_str())
            .context("structural measurement has no exact native Path-B assignment")?;
        let probability = open_state_ids.iter().try_fold(0.0, |sum, state_id| {
            assignment
                .state_membership
                .get(state_id)
                .map(|probability| sum + probability)
                .context("open state is absent from native membership distribution")
        })?;
        let structural_response = StructuralResponse {
            open_state: OpenPocketEstimand {
                probability,
                definition_sha256: definition_sha256.clone(),
                pathb_model_sha256: pathb.model.model_sha256.clone(),
                lag_steps: pathb.model.body.lag_steps,
                open_state_ids: open_state_ids.clone(),
            },
            pocket_volume_a3: measurement.structural_measurements.pocket_volume_a3,
            pocket_depth_a: measurement.structural_measurements.pocket_depth_a,
            solvent_accessible_area_a2: measurement
                .structural_measurements
                .solvent_accessible_area_a2,
            delta_volume_a3: measurement.structural_measurements.delta_volume_a3,
            delta_hydration: measurement.structural_measurements.delta_hydration,
            fragment_contact_fraction: measurement
                .structural_measurements
                .fragment_contact_fraction,
        };
        observations.push(GhostFiberObservation {
            site_key: measurement.site_key,
            control_lane: measurement.control_lane,
            ghost: measurement.ghost,
            protocol_phase: measurement.protocol_phase,
            thermodynamic_condition: measurement.thermodynamic_condition,
            structural_response,
            uncertainty: measurement.uncertainty,
            evidence_artifact_sha256: measurement.evidence_artifact_sha256,
        });
    }
    Ok((binding, observations))
}

fn require_consistent(slot: &mut Option<String>, value: &str, field: &str) -> Result<()> {
    match slot {
        Some(expected) if expected != value => bail!("{field} identity varies across frames"),
        Some(_) => Ok(()),
        None => {
            *slot = Some(value.to_string());
            Ok(())
        }
    }
}

fn validate_durable_registry_path(output_dir: &Path, registry_path: &Path) -> Result<()> {
    use std::path::Component;
    for (label, path) in [
        ("run output", output_dir),
        ("durable registry", registry_path),
    ] {
        if !path.is_absolute()
            || path
                .components()
                .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
        {
            bail!("{label} path must be absolute and lexically normalized");
        }
    }
    if registry_path.starts_with(output_dir) {
        bail!("durable candidate registry must be outside the per-run evidence directory");
    }
    let parent = registry_path
        .parent()
        .context("durable registry has no parent directory")?;
    let metadata = std::fs::metadata(parent)
        .with_context(|| format!("inspect durable registry parent {}", parent.display()))?;
    if !metadata.is_dir() {
        bail!("durable registry parent is not a directory");
    }
    Ok(())
}

fn candidate_objectives_from_matched_control(
    result: &MatchedControlResult,
    definitions: &[ObjectiveDefinition],
) -> Result<BTreeMap<String, f64>> {
    result.verify()?;
    let minimum_lane_ess = result
        .no_probe
        .effective_sample_size
        .min(result.hypothesis.effective_sample_size)
        .min(result.orthogonal.effective_sample_size);
    let primary_width =
        result.hypothesis_minus_orthogonal.upper - result.hypothesis_minus_orthogonal.lower;
    definitions
        .iter()
        .map(|definition| {
            let value = match definition.objective_id.as_str() {
                "hypothesis_selective_effect" => result.hypothesis_minus_orthogonal.mean,
                "hypothesis_minus_no_probe_effect" => result.hypothesis_minus_no_probe.mean,
                "hypothesis_open_probability" => result.hypothesis.open_probability.mean,
                "no_probe_open_probability" => result.no_probe.open_probability.mean,
                "orthogonal_open_probability" => result.orthogonal.open_probability.mean,
                "minimum_lane_effective_sample_size" => minimum_lane_ess,
                "primary_effect_confidence_width" => primary_width,
                "independent_seed_recurrence" => result.common_seeds.len() as f64,
                unknown => bail!(
                    "registry objective {unknown} has no deterministic matched-control extractor"
                ),
            };
            if !value.is_finite() {
                bail!("derived registry objective is non-finite");
            }
            let value = if value == 0.0 { 0.0 } else { value };
            Ok((definition.objective_id.clone(), value))
        })
        .collect()
}

fn require_output_targets_absent(output_dir: &Path) -> Result<()> {
    for name in [
        "native_pathb_chemical_state.json",
        "chemical_state_fibers.jsonl",
        "native_pathb_fiber_bundle.json",
        "matched_control_discriminator.json",
        "native_pathb_population_preservation.json",
        "candidate_advancement_registry_snapshot.ndjson",
    ] {
        let raw = output_dir.join(name);
        let packet = PathBuf::from(format!("{}.prismqnanl", raw.display()));
        let receipt = PathBuf::from(format!("{}.receipt.json", packet.display()));
        if raw.exists() || packet.exists() || receipt.exists() {
            bail!(
                "refusing to overwrite prior native Path-B evidence: {}",
                raw.display()
            );
        }
    }
    Ok(())
}

fn canonical_json_line<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn canonical_fiber_jsonl(fibers: &[ChemicalStateFiber]) -> Result<Vec<u8>> {
    let mut ordered = fibers.to_vec();
    ordered.sort_by(|a, b| a.fiber_sha256.cmp(&b.fiber_sha256));
    let mut bytes = Vec::new();
    for fiber in ordered {
        fiber.verify()?;
        serde_json::to_writer(&mut bytes, &fiber)?;
        bytes.push(b'\n');
    }
    Ok(bytes)
}

fn write_create_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .with_context(|| format!("create immutable evidence {}", path.display()))?;
    file.write_all(bytes)?;
    file.sync_all()?;
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    OpenOptions::new().read(true).open(parent)?.sync_all()?;
    Ok(())
}

fn emit_qnanl_exact(
    raw_path: &Path,
    family: QnanlArtifactFamily,
    event_rows: &[(&str, u16, Vec<&str>)],
    runtime_provenance: &RuntimeProvenance,
    source_stage: &str,
) -> Result<TypedEvidenceArtifact> {
    for (name, expected, _) in event_rows {
        if opcode_id_for_name(name) != Some(*expected) {
            bail!("canonical PrismCode opcode registry mismatch for {name}");
        }
    }
    let packet_path = PathBuf::from(format!("{}.prismqnanl", raw_path.display()));
    let receipt_path = PathBuf::from(format!("{}.receipt.json", packet_path.display()));
    let artifact_id = raw_path
        .file_name()
        .and_then(|name| name.to_str())
        .context("evidence filename is not UTF-8")?;
    let receipt = encode_file(
        raw_path,
        &packet_path,
        &receipt_path,
        QnanlEncodeOptions::runtime_inline(
            artifact_id,
            family,
            Some(raw_path.to_path_buf()),
            source_stage,
        ),
    )
    .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let pcode_refs = event_rows
        .iter()
        .map(|(name, opcode, operands)| QnanlPcodeRef {
            opcode_name: (*name).to_string(),
            opcode_id: Some(*opcode),
            operands: operands.iter().map(|value| (*value).to_string()).collect(),
        })
        .collect();
    let raw_parent_hash = receipt.raw_hash_blake3.clone();
    let receipt = receipt.with_pcode_refs(pcode_refs).with_runtime_provenance(
        runtime_provenance.artifact_record(&packet_path, source_stage, vec![raw_parent_hash]),
    );
    let receipt_bytes = receipt.to_json_pretty();
    let mut receipt_file = OpenOptions::new()
        .write(true)
        .truncate(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(&receipt_path)
        .with_context(|| format!("open QNANL receipt {}", receipt_path.display()))?;
    receipt_file.write_all(receipt_bytes.as_bytes())?;
    receipt_file.sync_all()?;
    OpenOptions::new()
        .read(true)
        .open(receipt_path.parent().unwrap_or_else(|| Path::new(".")))?
        .sync_all()?;
    let raw_bytes = std::fs::read(raw_path)?;
    let packet_bytes = std::fs::read(&packet_path)?;
    let decoded =
        decode_packet_bytes(&packet_bytes).map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let exact_byte_roundtrip_pass = decoded == raw_bytes;
    if receipt.status != "PASS"
        || receipt.runtime_origin != RuntimeOrigin::RuntimeInline
        || receipt.runtime_provenance.is_none()
        || !receipt.roundtrip_verified
        || !receipt.codec_roundtrip_verified
        || !receipt.qnanl_seal_constraints_pass
        || !exact_byte_roundtrip_pass
    {
        bail!("typed QNANL evidence failed runtime-origin or exact-byte gates");
    }
    Ok(TypedEvidenceArtifact {
        artifact_family: family.as_str().to_string(),
        raw_path: raw_path.display().to_string(),
        raw_sha256: sha256_bytes(&raw_bytes),
        packet_path: packet_path.display().to_string(),
        packet_sha256: sha256_bytes(&packet_bytes),
        receipt_path: receipt_path.display().to_string(),
        receipt_sha256: sha256_bytes(&std::fs::read(&receipt_path)?),
        prismcode_events: event_rows
            .iter()
            .map(|(name, _, _)| (*name).to_string())
            .collect(),
        runtime_origin: receipt.runtime_origin.as_str().to_string(),
        exact_byte_roundtrip_pass,
    })
}

/// Seal the standalone aggregation execution receipt as exact-byte QNANL
/// evidence. The raw receipt must already be committed create-new and fsynced;
/// this function never rewrites it.
pub fn emit_native_pathb_aggregate_execution_receipt_qnanl(
    raw_receipt_path: &Path,
    runtime_provenance: &RuntimeProvenance,
) -> Result<TypedEvidenceArtifact> {
    emit_qnanl_exact(
        raw_receipt_path,
        QnanlArtifactFamily::PathBChemicalState,
        &[],
        runtime_provenance,
        "prism_native_pathb_aggregate:execution_receipt",
    )
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use prismcode::binary_runtime::opcode;

    fn digest(byte: u8) -> String {
        format!("{byte:02x}").repeat(32)
    }

    #[test]
    fn exact_coordinate_binding_rejects_different_ghost_topology() {
        let ghost = GhostByteLineage {
            key: crate::chemical_state_fiber::GhostLineageKey {
                stream_id: 0, frame_idx: 1, step_idx: 10, site_id: 0, record_index: 0,
            },
            run_id: "coordinate-topology-contract".into(), replica_id: "rep_00".into(),
            replica_seed: 42, stream_seed: 42, topology_sha256: digest(1),
            record_abi_sha256: digest(2), record_source_sha256: digest(3),
            segment_context_digest_sha256: digest(4), segment_manifest_sha256: digest(5),
            segment_sequence: 0, segment_sha256: digest(6), segment_file: "segment.bin".into(),
            byte_offset: 0, byte_length: 192, record_sha256: digest(7), lattice_node_index: 0,
            engine_sha256: digest(8), condition_id: "condition".into(), protocol_name: "protocol".into(),
            protocol_phase: 0, phase_schedule_digest_sha256: digest(9), protocol_state_hash: digest(10),
        };
        let mut coordinate = ExactCoordinateFrameBinding {
            stream_id: 0, step_idx: 10, trajectory_frame_ordinal: 1,
            trajectory_byte_offset: 0, trajectory_byte_length: 120,
            coordinate_frame_sha256: digest(11), trajectory_artifact_sha256: digest(12),
            trajectory_audit_sha256: digest(13), topology_sha256: digest(1),
            atom_residue_mapping_sha256: digest(14), cavity_sidecar_record_sha256: digest(15),
            atom_count: 10, source_contract: "ghost_coordinate_segment_exact_join_v1".into(),
        };
        coordinate.validate_for(&ghost).unwrap();
        coordinate.topology_sha256 = digest(16);
        assert!(coordinate.validate_exact_source().is_ok());
        assert!(coordinate.validate_for(&ghost).is_err());
    }

    #[test]
    fn canonical_request_v6_fixture_passes_native_file_loader() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/native_pathb_runtime_request_v6.json")
            .canonicalize()
            .unwrap();
        let raw_fixture = std::fs::read(&path).unwrap();
        let fixture: NativePathBRuntimeRequest = serde_json::from_slice(&raw_fixture).unwrap();
        let mut canonical_fixture = serde_json::to_vec(&fixture).unwrap();
        canonical_fixture.push(b'\n');
        assert_eq!(raw_fixture, canonical_fixture);
        assert_eq!(
            fixture.request_sha256,
            canonical_sha256(&fixture.body).unwrap(),
            "shared request-v6 fixture body hash must use native serde ordering"
        );
        let request = load_native_pathb_runtime_request(&path).unwrap();
        assert_eq!(request.schema_version, NATIVE_PATHB_RUNTIME_REQUEST_VERSION);
        assert_eq!(
            request
                .body
                .lane_policy
                .coordinate_cavity_config
                .grid_spacings_a,
            vec![0.5, 0.75]
        );
        assert_eq!(
            request
                .body
                .lane_policy
                .coordinate_cavity_config
                .grid_phase_offsets,
            vec![0.0, 0.5]
        );
    }

    #[test]
    fn rust_serde_float_edge_corpus_materializes_exact_bytes_and_hash() {
        #[derive(serde::Deserialize)]
        struct Corpus {
            schema_kind: String,
            schema_version: u32,
            payload: CorpusPayload,
            rust_compact_json: String,
            rust_compact_sha256: String,
        }

        #[derive(serde::Serialize, serde::Deserialize)]
        struct CorpusPayload {
            literal_text: String,
            values: Vec<f64>,
        }

        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/rust_serde_float_edge_corpus_v1.json")
            .canonicalize()
            .unwrap();
        let corpus: Corpus = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(corpus.schema_kind, "prism_rust_serde_float_edge_corpus");
        assert_eq!(corpus.schema_version, 1);
        assert!(corpus.payload.values.iter().all(|value| value.is_finite()));
        let bytes = serde_json::to_vec(&corpus.payload).unwrap();
        assert_eq!(bytes, corpus.rust_compact_json.as_bytes());
        assert_eq!(
            format!("{:x}", Sha256::digest(&bytes)),
            corpus.rust_compact_sha256
        );
    }

    #[test]
    fn cross_run_handoff_identity_rehashes_public_members_and_separates_seeds() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        let raw_path = root.join("native_pathb_confirmation_handoff.json");
        let commit_path = root.join("COMMIT.json");
        let packet_path = PathBuf::from(format!("{}.prismqnanl", raw_path.display()));
        let receipt_path = PathBuf::from(format!("{}.receipt.json", packet_path.display()));
        std::fs::write(&raw_path, b"handoff\n").unwrap();
        std::fs::write(&commit_path, b"commit\n").unwrap();
        std::fs::write(&packet_path, b"packet").unwrap();
        std::fs::write(&receipt_path, b"receipt\n").unwrap();
        let mut identity = PathBConfirmationHandoffIdentity {
            handoff_root: root,
            handoff_root_raw_file_sha256: sha256_bytes(b"handoff\n"),
            handoff_root_commit_file_sha256: sha256_bytes(b"commit\n"),
            handoff_sha256: digest(1),
            handoff_transaction_commit_sha256: digest(2),
            handoff_ordered_membership_sha256: digest(3),
            discovery_transaction_commit_sha256: digest(4),
            discovery_population_sha256: digest(5),
            discovery_manifest_sha256: digest(6),
            correspondence_ledger_sha256: digest(7),
            selected_candidate_id_sha256: digest(8),
            selected_candidate_authority_sha256: digest(9),
            explicit_selection_authority_sha256: digest(10),
            operator_selection_authority_sha256: digest(11),
            tracker_design_authority_sha256: digest(12),
            supplied_tracker_sha256: digest(13),
            operator_design_authority_sha256: digest(14),
            disjoint_confirmation_seed_authority_sha256: digest(15),
            discovery_replica_seeds: vec![1, 2, 3, 4, 5],
            confirmation_replica_seeds: vec![11, 12, 13, 14, 15],
            qnanl_artifact_family: "PRISM.PATHB_CONFIRMATION_HANDOFF.V1".to_string(),
            qnanl_raw_sha256: sha256_bytes(b"handoff\n"),
            qnanl_packet_sha256: sha256_bytes(b"packet"),
            qnanl_receipt_sha256: sha256_bytes(b"receipt\n"),
        };
        identity.verify().unwrap();

        #[derive(Serialize, Deserialize)]
        struct LegacyManifestSerializationContract {
    site_key: String,
    candidate_tracker_sha256: String,
    source_confirmation_handoff: PathBConfirmationHandoffIdentity,
    fragment_panel_sha256: String,
    /// Exact hypothesis/control endpoint pairs. A panel digest alone cannot
    /// prove which two chemistry hypotheses were compared.
    probe_microstate_pair_by_lane: BTreeMap<ControlLane, ProbeLaneMicrostatePairIdentity>,
    /// Common T/P/ionic-strength/solvent/PBC/PME and phase-interval digest.
    matched_physical_state_sha256: String,
    engine_sha256: String,
    topology_sha256_by_lane: BTreeMap<ControlLane, String>,
    calibration_population_sha256_by_probe_lane: BTreeMap<ControlLane, String>,
    pathb_config_sha256: String,
    timebase_context_sha256: String,
    matched_control_config_sha256: String,
    registry_policy_sha256: String,
    common_replica_seeds: Vec<u64>,
    lane_evidence: Vec<CrossRunLaneEvidenceEntry>,
}
        let legacy_body = serde_json::json!({
            "site_key": "serialization-contract-only",
            "candidate_tracker_sha256": digest(13),
            "source_confirmation_handoff": identity,
            "fragment_panel_sha256": digest(14),
            "probe_microstate_pair_by_lane": {},
            "matched_physical_state_sha256": digest(15),
            "engine_sha256": digest(16),
            "topology_sha256_by_lane": {},
            "calibration_population_sha256_by_probe_lane": {},
            "pathb_config_sha256": digest(17),
            "timebase_context_sha256": digest(18),
            "matched_control_config_sha256": digest(19),
            "registry_policy_sha256": digest(20),
            "common_replica_seeds": [11,12,13,14,15],
            "lane_evidence": []
        });
        let decoded: PathBCrossRunAggregationManifestBody = serde_json::from_value(legacy_body.clone()).unwrap();
        assert!(decoded.tracker_realizations_by_lane.is_empty());
        assert_eq!(serde_json::to_value(&decoded).unwrap(), legacy_body);
        let legacy_decoded: LegacyManifestSerializationContract = serde_json::from_value(legacy_body).unwrap();
        assert_eq!(serde_json::to_vec(&decoded).unwrap(), serde_json::to_vec(&legacy_decoded).unwrap());
        assert_eq!(canonical_sha256(&decoded).unwrap(), canonical_sha256(&legacy_decoded).unwrap());
        identity.confirmation_replica_seeds[0] = 1;
        assert!(identity.verify().is_err());
        identity.confirmation_replica_seeds = vec![11, 12, 13, 14, 15];
        std::fs::write(&raw_path, b"substituted\n").unwrap();
        assert!(identity.verify().is_err());
    }

    fn tracker_contract_fixture() -> PathBCandidateTrackerDefinition {
        PathBCandidateTrackerDefinition {
            schema_kind: "prism_pathb_candidate_tracker".to_string(),
            schema_version: "2.0.0".to_string(),
            tracker_sha256: String::new(),
            reference_structure_sha256: digest(101),
            reference_coordinate_frame_sha256: digest(102),
            reference_frame_index: 0,
            reference_centroid_xyz_a: [1.0, 2.0, 3.0],
            body_fixed_tracker_authority:
                crate::fixed_cadence_coordinate::BodyFixedTrackerAuthority {
                    schema_kind: "prism_body_fixed_tracker_authority".to_string(),
                    schema_version: "1.0.0".to_string(),
                    authority_sha256: String::new(),
                    reference_structure_sha256: digest(101),
                    reference_coordinate_frame_sha256: digest(102),
                    chain_mapping_receipt_sha256: digest(103),
                    topology_sha256: digest(105),
                    spatial_boundary_contract_sha256: digest(106),
                    boundary_mode: crate::fixed_cadence_coordinate::BodyFixedBoundaryMode::DeclaredNonperiodicEuclidean,
                    unwrap_semantics: "declared_nonperiodic_no_unwrap_v1".to_string(),
                    frame_construction_semantics:
                        "ordered_three_anchor_right_handed_rigid_frame_v1".to_string(),
                    anchor_atoms: vec![
                        crate::fixed_cadence_coordinate::BodyFixedAnchorAtom {
                            atom_index: 0,
                            analysis_residue_id: 938,
                            chain_id: "MYC".to_string(),
                            pdb_residue_id: 938,
                            residue_name: "LYS".to_string(),
                            atom_name: "N".to_string(),
                            reference_position_angstrom: [0.0, 0.0, 0.0],
                        },
                        crate::fixed_cadence_coordinate::BodyFixedAnchorAtom {
                            atom_index: 1,
                            analysis_residue_id: 938,
                            chain_id: "MYC".to_string(),
                            pdb_residue_id: 938,
                            residue_name: "LYS".to_string(),
                            atom_name: "CA".to_string(),
                            reference_position_angstrom: [1.0, 0.0, 0.0],
                        },
                        crate::fixed_cadence_coordinate::BodyFixedAnchorAtom {
                            atom_index: 2,
                            analysis_residue_id: 938,
                            chain_id: "MYC".to_string(),
                            pdb_residue_id: 938,
                            residue_name: "LYS".to_string(),
                            atom_name: "C".to_string(),
                            reference_position_angstrom: [0.0, 1.0, 0.0],
                        },
                    ],
                    reference_candidate_centroid_angstrom: [1.0, 2.0, 3.0],
                    maximum_anchor_fit_rmsd_angstrom: 0.5,
                    maximum_anchor_fit_residual_angstrom: 1.0,
                }
                .materialize()
                .unwrap(),
            signed_contact_residues: vec![SignedTrackerContactResidue {
                chain_id: "MYC".to_string(),
                canonical_residue_id: "938:LYS".to_string(),
                engine_residue_id: 938,
                chain_mapping_receipt_sha256: digest(103),
            }],
            cavity_estimator_config: crate::coordinate_cavity::CoordinateCavityConfig::default(),
            observed_closed_max_volume_a3: 100.0,
            observed_open_min_volume_a3: 120.0,
            local_shell_radius_a: 8.0,
            cadence_steps: 10,
            start_step_by_stream: BTreeMap::from([(0, 100)]),
            end_step_exclusive_by_stream: BTreeMap::from([(0, 200)]),
            stationary_phase_ordinal: 1,
            matched_physical_state_sha256: digest(104),
        }
        .materialize()
        .unwrap()
    }

    #[test]
    fn cavity_discretization_interval_prevents_mean_only_binary_labels() {
        let tracker = tracker_contract_fixture();
        assert_eq!(
            classify_cavity_discretization_interval(121.0, 130.0, &tracker).unwrap(),
            CavityIntervalClass::Open
        );
        assert_eq!(
            classify_cavity_discretization_interval(80.0, 99.0, &tracker).unwrap(),
            CavityIntervalClass::Closed
        );
        assert_eq!(
            classify_cavity_discretization_interval(99.0, 121.0, &tracker).unwrap(),
            CavityIntervalClass::Indeterminate
        );
        assert_eq!(
            classify_cavity_discretization_interval(105.0, 115.0, &tracker).unwrap(),
            CavityIntervalClass::Indeterminate
        );
    }

    #[test]
    fn zero_event_target_receipt_preserves_negative_time_at_risk() {
        let selected_by_stream = BTreeMap::from([(0_u16, 0_u64), (1_u16, 0_u64)]);
        let target_residue_ids = vec![42_u32];
        let selected_ghost_lineage_sha256 = Vec::new();
        let source_lattice_manifest_sha256 = digest(9);
        let mut receipt = GhostSiteSelectionReceipt {
            schema_kind: "prism_ghost_site_selection_receipt".to_string(),
            schema_version: "2.0.0".to_string(),
            selection_sha256: String::new(),
            source_lattice_manifest_sha256: source_lattice_manifest_sha256.clone(),
            site_key: "site_target_42".to_string(),
            selection_mode: NativePathBSelectionMode::TargetResidueHypothesis,
            target_residue_ids,
            minimum_target_residue_overlap_count: 1,
            minimum_target_residue_overlap_fraction: 1.0,
            source_produced: 0,
            source_persisted: 0,
            source_decoded: 0,
            source_lattice_nodes: 0,
            selected_records: 0,
            selected_by_stream,
            selected_ghost_lineage_sha256,
            membership_semantics: "richspike_cluster_residue_membership_not_cavity_lining"
                .to_string(),
        };
        receipt.selection_sha256 = canonical_sha256(&GhostSiteSelectionReceiptBody {
            source_lattice_manifest_sha256: &receipt.source_lattice_manifest_sha256,
            site_key: &receipt.site_key,
            selection_mode: receipt.selection_mode,
            target_residue_ids: &receipt.target_residue_ids,
            minimum_target_residue_overlap_count: receipt.minimum_target_residue_overlap_count,
            minimum_target_residue_overlap_fraction: receipt
                .minimum_target_residue_overlap_fraction,
            source_produced: receipt.source_produced,
            source_persisted: receipt.source_persisted,
            source_decoded: receipt.source_decoded,
            source_lattice_nodes: receipt.source_lattice_nodes,
            selected_records: receipt.selected_records,
            selected_by_stream: &receipt.selected_by_stream,
            selected_ghost_lineage_sha256: &receipt.selected_ghost_lineage_sha256,
            membership_semantics: &receipt.membership_semantics,
        })
        .unwrap();
        receipt.verify().unwrap();

        receipt.selection_mode = NativePathBSelectionMode::WholeLatticeDiscovery;
        receipt.target_residue_ids.clear();
        receipt.minimum_target_residue_overlap_count = 0;
        receipt.minimum_target_residue_overlap_fraction = 0.0;
        receipt.membership_semantics =
            "whole_authenticated_lattice_no_residue_prefilter".to_string();
        assert!(receipt.verify().is_err());
    }

    #[test]
    fn zero_population_is_exact_conservation_not_a_lineage_failure() {
        GhostLatticeConservation {
            produced: 0,
            persisted: 0,
            decoded: 0,
            lattice_nodes: 0,
            lattice_manifest_sha256: digest(8),
        }
        .validate()
        .unwrap();
    }

    #[test]
    fn canonical_opcode_and_family_bindings_are_not_local_literals() {
        let rows = [
            (
                EVENT_CHEMICAL_STATE_FIBER_MATERIALIZED,
                opcode::PATHB_CHEMICAL_STATE_FIBER_MATERIALIZED,
            ),
            (
                EVENT_PATHB_CHEMICAL_STATE_ASSIGNED,
                opcode::PATHB_CHEMICAL_STATE_ASSIGNED,
            ),
            (
                EVENT_MATCHED_CONTROL_CLASSIFIED,
                opcode::PATHB_MATCHED_CONTROL_CLASSIFIED,
            ),
            (EVENT_CANDIDATE_OBSERVED, opcode::PATHB_CANDIDATE_OBSERVED),
            (
                EVENT_PARETO_CHAMPION_PROMOTED,
                opcode::PATHB_PARETO_CHAMPION_PROMOTED,
            ),
            (
                EVENT_PARETO_CHAMPION_SUPERSEDED,
                opcode::PATHB_PARETO_CHAMPION_SUPERSEDED,
            ),
            (
                crate::pathb_discovery::EVENT_PATHB_DISCOVERY_CANDIDATE_MATERIALIZED,
                opcode::PATHB_DISCOVERY_CANDIDATE_MATERIALIZED,
            ),
        ];
        for (name, opcode) in rows {
            assert_eq!(opcode_id_for_name(name), Some(opcode));
        }
        assert_eq!(
            QnanlArtifactFamily::ChemicalStateFiber.as_str(),
            "PRISM.CHEMICAL_STATE_FIBER.V1"
        );
        assert_eq!(
            QnanlArtifactFamily::PathBChemicalState.as_str(),
            "PRISM.PATHB_CHEMICAL_STATE.V1"
        );
        assert_eq!(
            QnanlArtifactFamily::MatchedControlDiscriminator.as_str(),
            "PRISM.MATCHED_CONTROL_DISCRIMINATOR.V1"
        );
        assert_eq!(
            QnanlArtifactFamily::CandidateAdvancementRegistry.as_str(),
            "PRISM.CANDIDATE_ADVANCEMENT_REGISTRY.V1"
        );
        assert_eq!(
            QnanlArtifactFamily::PathBDiscoveryCandidate.as_str(),
            "PRISM.PATHB_DISCOVERY_CANDIDATE.V1"
        );
    }

    #[test]
    fn claim_fragment_derivation_requires_the_immutable_exact_endpoint_join() {
        let body =
            crate::chemical_state_fiber::tests::body(ControlLane::HypothesisFragment, 1.0, 175, 71);
        let ghost = crate::chemical_state_fiber::tests::ghost_observation(&body);
        let lambda = crate::chemical_state_fiber::tests::lambda_load(&body);
        let joins =
            crate::chemical_state_fiber::materialize_lambda_ghost_endpoint_joins(&[ghost], &lambda)
                .unwrap();
        let lineage = body.ghost.lineage_sha256().unwrap();
        let fragment = FragmentStateObservation {
            ghost_key: body.ghost.key,
            ghost_lineage_sha256: lineage.clone(),
            fragment: body.fragment.clone(),
        };
        verify_claim_fragment_join_derivation(
            ControlLane::HypothesisFragment,
            &joins,
            &[fragment.clone()],
            &BTreeSet::from([lineage]),
        )
        .unwrap();

        // The old direct-record-hash interpretation must never reconstruct a
        // claim-bearing fiber after global occurrence lineage became binding.
        let mut old_direct_hash = fragment.clone();
        old_direct_hash.ghost_lineage_sha256 = body.ghost.record_sha256.clone();
        assert!(verify_claim_fragment_join_derivation(
            ControlLane::HypothesisFragment,
            &joins,
            &[old_direct_hash],
            &BTreeSet::from([body.ghost.lineage_sha256().unwrap()]),
        )
        .is_err());

        // A one-step offset is deliberately not a nearest-frame join. Both
        // source populations remain retained, but neither can enter a fiber.
        let mut shifted = crate::chemical_state_fiber::tests::ghost_observation(&body);
        shifted.ghost.key.step_idx += 1;
        shifted.ghost.key.frame_idx += 1;
        let shifted_lineage = shifted.ghost.lineage_sha256().unwrap();
        let shifted_joins = crate::chemical_state_fiber::materialize_lambda_ghost_endpoint_joins(
            &[shifted.clone()],
            &lambda,
        )
        .unwrap();
        let mut unmatched_fragment = fragment;
        unmatched_fragment.ghost_key = shifted.ghost.key;
        unmatched_fragment.ghost_lineage_sha256 = shifted_lineage.clone();
        assert!(verify_claim_fragment_join_derivation(
            ControlLane::HypothesisFragment,
            &shifted_joins,
            &[unmatched_fragment],
            &BTreeSet::from([shifted_lineage]),
        )
        .is_err());
    }

    #[test]
    fn coordinate_open_basin_hash_and_uncertainty_separation_fail_closed() {
        let summaries = vec![
            StateCavityObservableSummary {
                state_id: "state_000".to_string(),
                observation_count: 50,
                independent_seed_count: 5,
                mean_a3: 120.0,
                standard_error_a3: 2.0,
                confidence_interval_lower_a3: 114.0,
                confidence_interval_upper_a3: 126.0,
            },
            StateCavityObservableSummary {
                state_id: "state_001".to_string(),
                observation_count: 50,
                independent_seed_count: 5,
                mean_a3: 250.0,
                standard_error_a3: 3.0,
                confidence_interval_lower_a3: 241.0,
                confidence_interval_upper_a3: 259.0,
            },
        ];
        let open = vec!["state_001".to_string()];
        let model_sha = digest(1);
        let estimator_sha = digest(2);
        let topology_sha = Some(digest(3));
        let mapping_sha = Some(digest(4));
        let frame_sha = digest(5);
        let body = NativeOpenBasinBindingBody {
            pathb_model_sha256: &model_sha,
            observable_id: "coordinate_cavity_volume_a3",
            measurement_method_id: "prism_multiresolution_coordinate_ligsite_v1",
            estimator_sha256: &estimator_sha,
            topology_sha256: &topology_sha,
            atom_residue_mapping_sha256: &mapping_sha,
            topology_mapping_by_lane: &BTreeMap::new(),
            frame_population_sha256: &frame_sha,
            familywise_confidence_level: 0.95,
            selection_rule:
                "highest_seed_balanced_state_mean_with_bonferroni_t_interval_separation",
            open_state_ids: &open,
            state_summaries: &summaries,
        };
        let mut binding = NativeOpenBasinBinding {
            schema_kind: "prism_native_pathb_open_basin_binding".to_string(),
            schema_version: "1.0.0".to_string(),
            binding_sha256: canonical_sha256(&body).unwrap(),
            pathb_model_sha256: digest(1),
            observable_id: "coordinate_cavity_volume_a3".to_string(),
            measurement_method_id: "prism_multiresolution_coordinate_ligsite_v1".to_string(),
            estimator_sha256: digest(2),
            topology_sha256: Some(digest(3)),
            atom_residue_mapping_sha256: Some(digest(4)),
            topology_mapping_by_lane: BTreeMap::new(),
            frame_population_sha256: digest(5),
            familywise_confidence_level: 0.95,
            selection_rule:
                "highest_seed_balanced_state_mean_with_bonferroni_t_interval_separation".to_string(),
            open_state_ids: open,
            state_summaries: summaries,
        };
        binding.verify().unwrap();
        let legacy = serde_json::to_value(&binding).unwrap();
        assert!(legacy.get("topology_mapping_by_lane").is_none());
        assert!(legacy["topology_sha256"].is_string());
        let decoded: NativeOpenBasinBinding = serde_json::from_value(legacy.clone()).unwrap();
        assert_eq!(serde_json::to_value(&decoded).unwrap(), legacy);
        decoded.verify().unwrap();
        let mut heterogeneous = binding.clone();
        heterogeneous.schema_version = "2.0.0".into();
        heterogeneous.topology_sha256 = None;
        heterogeneous.atom_residue_mapping_sha256 = None;
        heterogeneous.topology_mapping_by_lane = BTreeMap::from([
            (ControlLane::NoProbe, NativeOpenBasinLaneIdentity { topology_sha256: digest(3), atom_residue_mapping_sha256: digest(4) }),
            (ControlLane::HypothesisFragment, NativeOpenBasinLaneIdentity { topology_sha256: digest(6), atom_residue_mapping_sha256: digest(7) }),
            (ControlLane::OrthogonalChemicalControl, NativeOpenBasinLaneIdentity { topology_sha256: digest(8), atom_residue_mapping_sha256: digest(9) }),
        ]);
        let rehash = |value: &mut NativeOpenBasinBinding| {
            value.binding_sha256 = canonical_sha256(&NativeOpenBasinBindingBody {
                pathb_model_sha256: &value.pathb_model_sha256,
                observable_id: "coordinate_cavity_volume_a3",
                measurement_method_id: &value.measurement_method_id,
                estimator_sha256: &value.estimator_sha256,
                topology_sha256: &value.topology_sha256,
                atom_residue_mapping_sha256: &value.atom_residue_mapping_sha256,
                topology_mapping_by_lane: &value.topology_mapping_by_lane,
                frame_population_sha256: &value.frame_population_sha256,
                familywise_confidence_level: value.familywise_confidence_level,
                selection_rule: "highest_seed_balanced_state_mean_with_bonferroni_t_interval_separation",
                open_state_ids: &value.open_state_ids, state_summaries: &value.state_summaries,
            }).unwrap();
        };
        rehash(&mut heterogeneous);
        heterogeneous.verify().unwrap();
        let decoded: NativeOpenBasinBinding = serde_json::from_slice(&serde_json::to_vec(&heterogeneous).unwrap()).unwrap();
        assert_eq!(decoded, heterogeneous);
        let mut swapped = heterogeneous.clone();
        let no_probe = swapped.topology_mapping_by_lane[&ControlLane::NoProbe].clone();
        let hypothesis = swapped.topology_mapping_by_lane[&ControlLane::HypothesisFragment].clone();
        swapped.topology_mapping_by_lane.insert(ControlLane::NoProbe, hypothesis);
        swapped.topology_mapping_by_lane.insert(ControlLane::HypothesisFragment, no_probe);
        assert!(swapped.verify().is_err()); // Lane identity is included in the existing digest.
        let mut inconsistent = heterogeneous;
        inconsistent.topology_mapping_by_lane.get_mut(&ControlLane::HypothesisFragment).unwrap().topology_sha256 = digest(3);
        rehash(&mut inconsistent);
        assert!(inconsistent.verify().is_err()); // Same exact topology cannot map atoms differently.
        binding.state_summaries[1].confidence_interval_lower_a3 = 125.0;
        assert!(binding.verify().is_err());
    }

    fn correspondence_ledger_with_status(
        status: ComponentCorrespondenceStatus,
    ) -> (ComponentCorrespondenceLedger, String, String) {
        use crate::component_correspondence::{
            CanonicalResidueContactProbability, ComponentCorrespondenceAssignment,
            PeriodicAwareSpatialDistribution, SpatialCoordinateMode,
        };
        let source_component_sha256 = digest(40);
        let descriptor = NativeComponentDescriptor {
            descriptor_sha256: String::new(),
            source_component_sha256: source_component_sha256.clone(),
            run_id: "run-ambiguous-preservation".to_string(),
            replica_id: "rep-00".to_string(),
            replica_seed: 42,
            control_lane: ControlLane::NoProbe,
            signed_contact_distribution: vec![CanonicalResidueContactProbability {
                chain_id: "A".to_string(),
                canonical_residue_id: "MYC:938".to_string(),
                chain_mapping_receipt_sha256: digest(41),
                probability: 1.0,
            }],
            spatial_distribution: PeriodicAwareSpatialDistribution {
                coordinate_mode: SpatialCoordinateMode::CartesianAngstrom,
                estimator_semantics: "cartesian_mean_covariance_radial_quantiles_v1".to_string(),
                mean_xyz: [1.0, 2.0, 3.0],
                covariance_upper: [1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
                radial_quantiles_25_50_75: [0.5, 1.0, 1.5],
                periodic_box_vectors_sha256: None,
                periodic_box_metric_tensor_upper: None,
            },
        }
        .materialize()
        .unwrap();
        let descriptor_sha256 = descriptor.descriptor_sha256.clone();
        let policy = ComponentCorrespondencePolicy {
            policy_sha256: String::new(),
            residue_js_weight: 0.5,
            spatial_weight: 0.5,
            maximum_assignment_cost: 1.0,
            ambiguity_margin: 0.1,
        }
        .materialize()
        .unwrap();
        let ledger = ComponentCorrespondenceLedger {
            schema_kind: "prism_component_correspondence_ledger".to_string(),
            schema_version: "1.0.0".to_string(),
            ledger_sha256: digest(42),
            policy,
            descriptor_population_sha256: digest(43),
            contact_mappings_by_lane: BTreeMap::new(),
            descriptors: vec![descriptor],
            assignments: vec![ComponentCorrespondenceAssignment {
                descriptor_sha256: descriptor_sha256.clone(),
                status,
            }],
            recurrent_correspondence_count: 0,
            ambiguous_count: 0,
            unmatched_count: 0,
        };
        (ledger, source_component_sha256, descriptor_sha256)
    }

    #[test]
    fn ambiguous_and_unmatched_components_are_preserved_without_candidate_append() {
        let candidate_a = digest(50);
        let (ambiguous_ledger, source_component, descriptor_sha256) =
            correspondence_ledger_with_status(ComponentCorrespondenceStatus::Ambiguous {
                candidate_correspondence_sha256: vec![candidate_a],
                candidate_cost: vec![0.2],
            });
        let emitted = BTreeSet::from([source_component]);
        let ambiguous =
            evaluate_target_component_correspondence(&emitted, &ambiguous_ledger).unwrap();
        assert_eq!(
            ambiguous.status,
            CandidateClaimEvaluationStatus::NotEvaluable
        );
        assert!(!ambiguous.candidate_append_allowed);
        assert_eq!(
            ambiguous.ambiguous_descriptor_sha256,
            vec![descriptor_sha256]
        );
        assert!(ambiguous
            .reason_codes
            .contains(&"AMBIGUOUS_TARGET_COMPONENT_CORRESPONDENCE".to_string()));

        let (unmatched_ledger, source_component, descriptor_sha256) =
            correspondence_ledger_with_status(ComponentCorrespondenceStatus::Unmatched {
                reason_code: "no_cross_run_match".to_string(),
            });
        let unmatched = evaluate_target_component_correspondence(
            &BTreeSet::from([source_component]),
            &unmatched_ledger,
        )
        .unwrap();
        assert_eq!(
            unmatched.status,
            CandidateClaimEvaluationStatus::NotEvaluable
        );
        assert!(!unmatched.candidate_append_allowed);
        assert_eq!(
            unmatched.unmatched_descriptor_sha256,
            vec![descriptor_sha256]
        );
    }

    #[test]
    fn zero_detection_is_a_preserved_not_evaluable_claim_state() {
        let (ledger, _, _) =
            correspondence_ledger_with_status(ComponentCorrespondenceStatus::Unmatched {
                reason_code: "not_observed".to_string(),
            });
        let evaluation =
            evaluate_target_component_correspondence(&BTreeSet::new(), &ledger).unwrap();
        assert_eq!(
            evaluation.status,
            CandidateClaimEvaluationStatus::NotEvaluable
        );
        assert!(!evaluation.candidate_append_allowed);
        assert!(evaluation.emitted_component_sha256.is_empty());
        assert!(evaluation
            .reason_codes
            .contains(&"ZERO_TARGET_DETECTIONS".to_string()));
    }

    fn no_probe_fragment_observation(lineage_byte: u8, step: u64) -> FragmentStateObservation {
        FragmentStateObservation {
            ghost_key: crate::chemical_state_fiber::GhostLineageKey {
                stream_id: 0,
                frame_idx: step,
                step_idx: step,
                site_id: 1,
                record_index: step,
            },
            ghost_lineage_sha256: digest(lineage_byte),
            fragment: crate::chemical_state_fiber::FragmentState {
                identity: None,
                endpoint_a_identity: None,
                endpoint_b_identity: None,
                occupancy: crate::chemical_state_fiber::FragmentOccupancy::Absent,
                lambda: 0.0,
                du_dlambda_kj_mol: 0.0,
                dudl_unit_conversion: None,
                chemical_potential_kj_mol: 0.0,
                hamiltonian_sha256: digest(60),
                source_force_event: None,
                source_endpoint_observation: None,
                chemical_potential_calibration: None,
            },
        }
    }

    #[test]
    fn fragment_partition_preserves_full_unassigned_rows_and_exact_counts() {
        let assigned = no_probe_fragment_observation(61, 100);
        let unassigned = no_probe_fragment_observation(62, 200);
        let assigned_hash = assigned.ghost_lineage_sha256.clone();
        let (fiber_input, ledger) = partition_fragment_observations(
            &digest(63),
            &digest(64),
            vec![unassigned.clone(), assigned.clone()],
            &BTreeSet::from([assigned_hash.clone()]),
        )
        .unwrap();
        assert_eq!(fiber_input, vec![assigned]);
        assert_eq!(ledger.source_observation_count, 2);
        assert_eq!(ledger.assigned_ghost_lineage_sha256, vec![assigned_hash]);
        assert_eq!(ledger.unassigned_observations, vec![unassigned]);
        ledger.verify().unwrap();
    }

    fn stage_test_lane_transaction(
        output_dir: &Path,
        stage_name: &str,
        transaction_byte: u8,
    ) -> (PathBuf, PathBuf, Vec<u8>, Vec<u8>) {
        std::fs::create_dir_all(output_dir).unwrap();
        let staging_root = output_dir.join(stage_name);
        std::fs::create_dir(&staging_root).unwrap();
        let public_root = output_dir.join(NATIVE_PATHB_LANE_TRANSACTION_DIRECTORY);
        let join_raw = canonical_json_line(&serde_json::json!({
            "fixture": "lambda_ghost_join",
            "transaction_byte": transaction_byte,
        }))
        .unwrap();
        let lane_raw = canonical_json_line(&serde_json::json!({
            "fixture": "native_pathb_lane",
            "transaction_byte": transaction_byte,
        }))
        .unwrap();
        let provenance =
            RuntimeProvenance::capture(output_dir, vec!["pathb-fail-atomic-unit-test".to_string()]);
        let join_actual = staging_root.join(LAMBDA_GHOST_JOIN_RAW_FILE);
        write_create_new(&join_actual, &join_raw).unwrap();
        let join_typed = emit_qnanl_exact_at_public_paths(
            &join_actual,
            &public_root.join(LAMBDA_GHOST_JOIN_RAW_FILE),
            QnanlArtifactFamily::PathBChemicalState,
            &[],
            &provenance,
            "pathb_transaction_unit_test:join",
        )
        .unwrap();
        let lane_actual = staging_root.join(NATIVE_PATHB_LANE_RAW_FILE);
        write_create_new(&lane_actual, &lane_raw).unwrap();
        let lane_typed = emit_qnanl_exact_at_public_paths(
            &lane_actual,
            &public_root.join(NATIVE_PATHB_LANE_RAW_FILE),
            QnanlArtifactFamily::PathBChemicalState,
            &[],
            &provenance,
            "pathb_transaction_unit_test:lane",
        )
        .unwrap();
        let members = native_pathb_lane_transaction_member_names()
            .iter()
            .enumerate()
            .map(|(ordinal, (role, path))| {
                transaction_member(&staging_root, ordinal, role, path).unwrap()
            })
            .collect::<Vec<_>>();
        let body = NativePathBLaneTransactionCommitBody {
            transaction_id_sha256: digest(transaction_byte),
            committed_directory: public_root.display().to_string(),
            lane_evidence_sha256: digest(transaction_byte.wrapping_add(1)),
            lambda_ghost_endpoint_join_sha256: digest(transaction_byte.wrapping_add(2)),
            ordered_membership_sha256: canonical_sha256(&members).unwrap(),
            member_count: members.len() as u64,
            members,
            lambda_ghost_endpoint_join_typed_artifact: join_typed,
            native_pathb_lane_typed_artifact: lane_typed,
        };
        let commit = NativePathBLaneTransactionCommit {
            schema_kind: NATIVE_PATHB_LANE_TRANSACTION_SCHEMA.to_string(),
            schema_version: NATIVE_PATHB_LANE_TRANSACTION_VERSION.to_string(),
            status: "PASS".to_string(),
            commit_sha256: canonical_sha256(&body).unwrap(),
            body,
        };
        write_create_new(
            &staging_root.join(NATIVE_PATHB_LANE_TRANSACTION_COMMIT_FILE),
            &canonical_json_line(&commit).unwrap(),
        )
        .unwrap();
        sync_directory(&staging_root).unwrap();
        verify_native_pathb_lane_transaction_directory(&staging_root, Some(&public_root)).unwrap();
        (staging_root, public_root, join_raw, lane_raw)
    }

    #[test]
    fn pathb_transaction_successful_exact_byte_readback_and_public_commit_gate() {
        let temp = tempfile::tempdir().unwrap();
        let output = temp.path().join("pathb-output");
        let (staging, public, join_raw, lane_raw) =
            stage_test_lane_transaction(&output, ".stage-success", 81);
        publish_directory_noreplace(&staging, &public).unwrap();
        sync_directory(&output).unwrap();

        let verified = verify_native_pathb_lane_transaction_directory(&public, None).unwrap();
        assert_eq!(verified.join_raw, join_raw);
        assert_eq!(verified.lane_raw, lane_raw);
        assert_eq!(verified.commit.status, "PASS");
        assert_eq!(verified.commit.body.member_count, 6);
        assert_eq!(
            load_native_pathb_lane_transaction_commit(&public)
                .unwrap()
                .commit_sha256,
            verified.commit.commit_sha256
        );
    }

    #[test]
    fn pathb_transaction_tamper_and_extra_member_fail_closed() {
        let temp = tempfile::tempdir().unwrap();
        let output = temp.path().join("pathb-output");
        let (staging, public, _, _) = stage_test_lane_transaction(&output, ".stage-tamper", 82);
        publish_directory_noreplace(&staging, &public).unwrap();
        std::fs::write(
            public.join(LAMBDA_GHOST_JOIN_RAW_FILE),
            b"{\"tampered\":true}\n",
        )
        .unwrap();
        assert!(verify_native_pathb_lane_transaction_directory(&public, None).is_err());

        let temp = tempfile::tempdir().unwrap();
        let output = temp.path().join("pathb-output");
        let (staging, public, _, _) = stage_test_lane_transaction(&output, ".stage-extra", 83);
        publish_directory_noreplace(&staging, &public).unwrap();
        std::fs::write(public.join("unexpected_duplicate_member.json"), b"{}\n").unwrap();
        assert!(verify_native_pathb_lane_transaction_directory(&public, None).is_err());
    }

    #[test]
    fn pathb_transaction_duplicate_publication_never_clobbers_first_commit() {
        let temp = tempfile::tempdir().unwrap();
        let output = temp.path().join("pathb-output");
        let (first_stage, public, _, first_lane) =
            stage_test_lane_transaction(&output, ".stage-first", 84);
        publish_directory_noreplace(&first_stage, &public).unwrap();
        let first_commit = load_native_pathb_lane_transaction_commit(&public).unwrap();

        let (duplicate_stage, duplicate_public, _, _) =
            stage_test_lane_transaction(&output, ".stage-duplicate", 85);
        assert_eq!(duplicate_public, public);
        assert!(publish_directory_noreplace(&duplicate_stage, &public).is_err());
        let preserved = verify_native_pathb_lane_transaction_directory(&public, None).unwrap();
        assert_eq!(preserved.lane_raw, first_lane);
        assert_eq!(preserved.commit.commit_sha256, first_commit.commit_sha256);
    }

    #[test]
    fn pathb_transaction_crash_stage_is_ignored_and_late_failure_publishes_nothing() {
        let temp = tempfile::tempdir().unwrap();
        let output = temp.path().join("pathb-output");
        std::fs::create_dir_all(&output).unwrap();
        let orphan = output.join(".native_pathb_lane_transaction.staging.crash");
        std::fs::create_dir(&orphan).unwrap();
        write_create_new(
            &orphan.join(LAMBDA_GHOST_JOIN_RAW_FILE),
            b"{\"late_failure\":true}\n",
        )
        .unwrap();
        let public = output.join(NATIVE_PATHB_LANE_TRANSACTION_DIRECTORY);
        assert!(!public.exists());
        assert!(load_native_pathb_lane_transaction_commit(&orphan).is_err());

        let (retry_stage, retry_public, _, _) =
            stage_test_lane_transaction(&output, ".stage-retry", 86);
        publish_directory_noreplace(&retry_stage, &retry_public).unwrap();
        assert!(orphan.exists());
        assert!(load_native_pathb_lane_transaction_commit(&retry_public).is_ok());
    }

    #[test]
    fn pathb_transaction_commitless_published_root_is_rejected() {
        let temp = tempfile::tempdir().unwrap();
        let public = temp.path().join(NATIVE_PATHB_LANE_TRANSACTION_DIRECTORY);
        std::fs::create_dir(&public).unwrap();
        write_create_new(&public.join(NATIVE_PATHB_LANE_RAW_FILE), b"{}\n").unwrap();
        assert!(load_native_pathb_lane_transaction_commit(&public).is_err());
        assert!(load_native_pathb_lane_evidence(&public).is_err());
    }
    fn reseal_tracker(mut tracker: PathBCandidateTrackerDefinition) -> PathBCandidateTrackerDefinition {
        tracker.body_fixed_tracker_authority.authority_sha256.clear();
        tracker.body_fixed_tracker_authority = tracker.body_fixed_tracker_authority.materialize().unwrap();
        tracker.tracker_sha256.clear();
        tracker.materialize().unwrap()
    }

    #[test]
    fn lane_tracker_projection_preserves_design_and_rejects_contact_anchor_or_threshold_changes() {
        let reference = tracker_contract_fixture();
        let mut local = reference.clone();
        local.body_fixed_tracker_authority.topology_sha256 = digest(201);
        local.body_fixed_tracker_authority.chain_mapping_receipt_sha256 = digest(202);
        local.body_fixed_tracker_authority.spatial_boundary_contract_sha256 = digest(203);
        for anchor in &mut local.body_fixed_tracker_authority.anchor_atoms {
            anchor.atom_index += 50;
            anchor.analysis_residue_id = 7;
        }
        for contact in &mut local.signed_contact_residues {
            contact.engine_residue_id = 7;
            contact.chain_mapping_receipt_sha256 = digest(202);
        }
        let local = reseal_tracker(local);
        assert_ne!(local.tracker_sha256, reference.tracker_sha256);
        verify_tracker_realization_design(&reference, &local).unwrap();
        let mut changed = local.clone();
        changed.signed_contact_residues[0].canonical_residue_id = "939:LYS".into();
        assert!(verify_tracker_realization_design(&reference, &reseal_tracker(changed)).is_err());
        let mut changed = local.clone();
        changed.body_fixed_tracker_authority.anchor_atoms[0].atom_name = "CB".into();
        assert!(verify_tracker_realization_design(&reference, &reseal_tracker(changed)).is_err());
        let mut changed = local.clone();
        changed.observed_open_min_volume_a3 += 1.0;
        assert!(verify_tracker_realization_design(&reference, &reseal_tracker(changed)).is_err());
        let mut changed = local;
        changed.body_fixed_tracker_authority.anchor_atoms[0].reference_position_angstrom[2] = 0.01;
        assert!(verify_tracker_realization_design(&reference, &reseal_tracker(changed)).is_err());
    }

    #[test]
    fn lane_tracker_rejects_topology_file_hash_mismatch_before_decoding() {
        let directory = tempfile::tempdir().unwrap();
        let topology_path = directory.path().join("topology.json");
        std::fs::write(&topology_path, b"{}\n").unwrap();
        let reference = tracker_contract_fixture();
        let realization = PathBTrackerLaneRealization { topology_path, candidate_tracker: reference.clone() };
        assert!(realization.verify(&digest(200), &reference).unwrap_err().to_string().contains("differs from declared lane topology"));
    }

}
