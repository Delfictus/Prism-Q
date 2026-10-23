//! Native Path-B chemical-state validation and matched-control inference.
//!
//! This module consumes native Ghost-derived state assignments. It does not
//! read legacy `binding_sites.json`. Transition rows and metastable memberships
//! are normalized before a [`PathBProjection`](crate::chemical_state_fiber::PathBProjection)
//! can be attached to a chemical-state fiber. Matched-control inference then
//! compares the same endpoint estimand across no-probe, hypothesis-fragment,
//! and orthogonal-control lanes using at least five common independent seeds.

use crate::chemical_state_fiber::{
    assemble_fibers, canonical_sha256, require_sha256, ChemicalStateFiber, ControlLane,
    EndpointEligibility, FragmentStateObservation, GhostByteLineage, GhostFiberObservation,
    PathBProjection, PathBProjectionObservation,
};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, StudentsT};
use std::collections::{BTreeMap, BTreeSet};

/// Native Path-B model schema identifier.
pub const PATHB_CHEMICAL_STATE_SCHEMA: &str = "prism_pathb_chemical_state_model";
/// Native Path-B model schema version.
pub const PATHB_CHEMICAL_STATE_VERSION: &str = "7.0.0";
/// Typed runtime event semantic for a native Path-B assignment.
pub const EVENT_PATHB_CHEMICAL_STATE_ASSIGNED: &str = "PATHB.CHEMICAL_STATE_ASSIGNED";
/// Typed runtime event semantic for matched-control adjudication.
pub const EVENT_MATCHED_CONTROL_CLASSIFIED: &str = "PATHB.MATCHED_CONTROL_CLASSIFIED";
/// Canonical PrismCode opcode for native Path-B chemical-state assignment.
pub use prismcode::binary_runtime::opcode::PATHB_CHEMICAL_STATE_ASSIGNED as PRISMCODE_OPCODE_PATHB_CHEMICAL_STATE_ASSIGNED;
/// Canonical PrismCode opcode for matched-control classification.
pub use prismcode::binary_runtime::opcode::PATHB_MATCHED_CONTROL_CLASSIFIED as PRISMCODE_OPCODE_MATCHED_CONTROL_CLASSIFIED;
/// Canonical QNANL artifact-family code for native Path-B chemical state.
pub const QNANL_ARTIFACT_CODE_PATHB_CHEMICAL_STATE: u16 =
    prismcode::qnanl_artifact::QnanlArtifactFamily::PathBChemicalState as u16;
/// Canonical QNANL artifact-family code for matched-control results.
pub const QNANL_ARTIFACT_CODE_MATCHED_CONTROL: u16 =
    prismcode::qnanl_artifact::QnanlArtifactFamily::MatchedControlDiscriminator as u16;
/// QNANL family for the validated native Path-B model and assignments.
pub const QNANL_FAMILY_PATHB_CHEMICAL_STATE: &str =
    prismcode::qnanl_artifact::QnanlArtifactFamily::PathBChemicalState.as_str();
/// QNANL family for endpoint-only matched-control results.
pub const QNANL_FAMILY_MATCHED_CONTROL: &str =
    prismcode::qnanl_artifact::QnanlArtifactFamily::MatchedControlDiscriminator.as_str();

/// A metastable state in a native Path-B model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetastableState {
    /// Stable state identifier.
    pub state_id: String,
    /// Stationary population probability.
    pub stationary_probability: f64,
}

/// One normalized transition-matrix entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransitionProbability {
    /// Source metastable state.
    pub from_state_id: String,
    /// Destination metastable state.
    pub to_state_id: String,
    /// Row-stochastic transition probability.
    pub probability: f64,
}

/// One reversible transition operator fitted within a single physical lane.
/// State geometry is shared across matched lanes, but stationary populations
/// and kinetics are never pooled across distinct fragment Hamiltonians.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LaneTransitionModel {
    pub control_lane: ControlLane,
    /// Shared global state identifiers admitted to this lane's kinetic
    /// operator. States outside this list retain their global structural
    /// identity but receive neither stationary mass nor transition entries in
    /// this physical Hamiltonian.
    pub active_state_ids: Vec<String>,
    pub stationary_probability_by_state: BTreeMap<String, f64>,
    pub transitions: Vec<TransitionProbability>,
}

/// Evidentiary status of one shared global structural state inside one
/// physical matched-control lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LaneGlobalStateSupportStatus {
    /// The state meets both predeclared row-count and independent-seed gates
    /// and is therefore admitted to this lane's active kinetic subspace.
    Observed,
    /// The state was detected, but does not meet one or both support gates.
    BelowSupport,
    /// No scheduled structurally evaluable row was assigned to this state.
    NotObserved,
}

/// Typed lane-by-global-state evidence. `scheduled_exposure_rows` and
/// `hard_assignment_rows` preserve the exact zero-event denominator needed to
/// interpret an absent control-lane state as a detection-limit observation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LaneGlobalStateSupportEvidence {
    /// Physical matched-control lane.
    pub control_lane: ControlLane,
    /// Shared global structural-state identity.
    pub state_id: String,
    /// Gate-derived support status.
    pub status: LaneGlobalStateSupportStatus,
    /// Exact structurally evaluable scheduled rows in this lane.
    pub scheduled_exposure_rows: u64,
    /// Scheduled rows whose hard structural assignment is this state.
    pub hard_assignment_rows: u64,
    /// Independent replica seeds contributing those assignments.
    pub independent_seed_support: usize,
    /// Predeclared row-count gate copied from estimator authority.
    pub minimum_required_assignment_rows: u64,
    /// Predeclared seed-support gate copied from estimator authority.
    pub minimum_required_independent_seeds: usize,
}

/// Native state assignment for one exact Ghost record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GhostStateAssignment {
    /// Canonical complete Ghost byte-lineage occurrence SHA-256. This is the
    /// globally unique assignment key across independent matched runs.
    pub ghost_lineage_sha256: String,
    /// Exact Ghost record SHA-256.
    pub ghost_record_sha256: String,
    /// Physical matched-control lane whose operator owns this assignment.
    pub control_lane: ControlLane,
    /// Probability distribution over every declared state.
    pub state_membership: BTreeMap<String, f64>,
    /// Dominant state at the preceding lag, when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_state_id: Option<String>,
    /// Forward committor for a hash-bound target basin, unavailable until a
    /// coordinate-derived open-basin definition is attached.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub forward_committor: Option<f64>,
}

/// Binary cavity label supplied by the predeclared tracker. Structural state
/// fitting does not require this label: an indeterminate-band row remains a
/// valid continuous-feature observation with `binary_label = None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CavityBinaryLabel {
    Open,
    Closed,
}

/// Ghost-emission status retained as auxiliary evidence on every structurally
/// evaluable scheduled frame. Absence is typed and must never be interpreted as
/// a closed structural state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum ScheduledGhostEventStatus {
    Emitted {
        ghost_lineage_sha256: String,
        exact_component_sha256: String,
    },
    AmbiguousEmissions {
        ghost_lineage_sha256: Vec<String>,
        exact_component_sha256: Vec<String>,
    },
    NotEmitted {
        typed_absence_receipt_sha256: String,
    },
}

/// Structural-state assignment for one evaluable fixed-cadence surveillance
/// frame. Unlike [`GhostStateAssignment`], this population covers typed
/// no-emission rows and therefore represents structural time-at-risk.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScheduledStructuralStateAssignment {
    pub source_surveillance_ledger_sha256: String,
    pub source_surveillance_row_sha256: String,
    pub run_id: String,
    pub replica_id: String,
    pub replica_seed: u64,
    pub control_lane: ControlLane,
    pub stream_id: u16,
    pub step_idx: u64,
    pub evaluable_segment_id: u64,
    pub cavity_volume_a3: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub binary_label: Option<CavityBinaryLabel>,
    pub ghost_event: ScheduledGhostEventStatus,
    pub state_membership: BTreeMap<String, f64>,
    pub dominant_state_id: String,
}

/// One seed/run endpoint estimand built from the complete scheduled structural
/// time-at-risk population. Stream observations are averaged within each step
/// before step means are averaged within the seed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScheduledSeedOpenOccupancyEstimate {
    pub run_id: String,
    pub replica_id: String,
    pub replica_seed: u64,
    pub control_lane: ControlLane,
    pub scheduled_rows: u64,
    pub structurally_evaluable_rows: u64,
    pub not_evaluable_rows: u64,
    pub averaged_step_count: u64,
    pub stream_row_count: u64,
    pub typed_no_ghost_rows: u64,
    pub mean_open_probability: f64,
}

/// Hash-bound full-time-at-risk occupancy population for a declared set of
/// coordinate-defined open structural states.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScheduledOpenOccupancyPopulation {
    pub schema_kind: String,
    pub schema_version: String,
    pub population_sha256: String,
    pub pathb_model_sha256: String,
    pub open_state_ids: Vec<String>,
    pub seed_estimates: Vec<ScheduledSeedOpenOccupancyEstimate>,
}

/// Markov-model diagnostic status. Inconclusive models remain preserved but
/// cannot enter endpoint fibers or candidate advancement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MarkovDiagnosticStatus {
    /// Every predeclared diagnostic gate passed.
    Pass,
    /// Model is materialized for evidence preservation but is not claim-eligible.
    ComputationallyInconclusive,
}

/// Implied-timescale estimate at one exact lag.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImpliedTimescaleEstimate {
    /// Physical lane owning this kinetic estimate.
    pub control_lane: ControlLane,
    /// Lag in integration steps.
    pub lag_steps: u64,
    /// Second eigenvalue magnitude of the reversible transition operator.
    pub second_eigenvalue_magnitude: f64,
    /// Implied relaxation timescale in integration steps, absent at a singular endpoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub implied_timescale_steps: Option<f64>,
    /// Exact transition pairs supporting this lag.
    pub transition_pairs: u64,
}

/// Bootstrap interval for one transition probability.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransitionBootstrapInterval {
    /// Physical lane owning this transition interval.
    pub control_lane: ControlLane,
    /// Source state.
    pub from_state_id: String,
    /// Destination state.
    pub to_state_id: String,
    /// Bootstrap mean.
    pub mean: f64,
    /// Percentile lower bound.
    pub lower: f64,
    /// Percentile upper bound.
    pub upper: f64,
}

/// Bootstrap interval for one committor coordinate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommittorBootstrapInterval {
    /// State identity.
    pub state_id: String,
    /// Bootstrap mean.
    pub mean: f64,
    /// Percentile lower bound.
    pub lower: f64,
    /// Percentile upper bound.
    pub upper: f64,
}

/// One deterministic leave-one-seed-out candidate in the predeclared
/// kinetic state/lag grid.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KineticCandidateScore {
    pub state_count: usize,
    pub lag_steps: u64,
    pub kinetic_dimensions: usize,
    pub fold_seed: Vec<u64>,
    /// True held-out VAMP-E scores evaluated with the train-fitted singular
    /// functions and singular values. The validation covariance is never
    /// refit into a transition/operator model.
    pub fold_validation_vampe_score: Vec<f64>,
    pub mean_validation_vampe_score: f64,
    pub minimum_fold_validation_vampe_score: f64,
    /// Winner of representation/state selection at this lag. VAMP-E scores
    /// are never compared across lag times.
    pub within_lag_winner: bool,
    /// Full-data kinetic diagnostic used only after within-lag selection.
    pub implied_timescale_relative_spread: f64,
    pub chapman_kolmogorov_max_abs_residual: f64,
    pub minimum_state_seed_support: usize,
    pub lag_diagnostics_pass: bool,
    pub eligible: bool,
    pub ineligible_reasons: Vec<String>,
}

/// Content-addressed deterministic model-selection and sensitivity ledger.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KineticModelSelectionLedger {
    pub ledger_sha256: String,
    pub feature_schema_sha256: String,
    pub surveillance_population_sha256: String,
    pub selection_rule: String,
    pub candidates: Vec<KineticCandidateScore>,
    pub selected_state_count: usize,
    pub selected_kinetic_dimensions: usize,
    pub selected_lag_steps: u64,
    pub selected_mean_validation_vampe_score: f64,
    /// Declared lower bound applied independently to every LOSO fold.
    pub minimum_required_fold_validation_vampe_score: f64,
    pub within_lag_runner_up_score_margin: f64,
    pub loso_seed_count: usize,
    pub sensitivity_pass: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurveillanceRunMissingness {
    pub run_id: String,
    pub replica_id: String,
    pub replica_seed: u64,
    pub control_lane: crate::chemical_state_fiber::ControlLane,
    pub scheduled_rows: u64,
    pub evaluable_rows: u64,
    pub not_evaluable_rows: u64,
}

/// Complete fixed-cadence missingness authority. Scheduled NOT_EVALUABLE rows
/// remain in the model hash even though they are excluded from transitions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurveillanceMissingnessSummary {
    pub surveillance_ledger_population_sha256: String,
    pub scheduled_rows: u64,
    pub evaluable_rows: u64,
    pub not_evaluable_rows: u64,
    pub evaluable_fraction: f64,
    pub observed_open_rows: u64,
    pub observed_closed_rows: u64,
    /// Structurally evaluable rows whose continuous coordinate features are
    /// retained while the tracker open/closed label is withheld.
    pub observed_indeterminate_rows: u64,
    pub ghost_emitted_rows: u64,
    pub ghost_not_emitted_rows: u64,
    pub not_evaluable_reason_counts: BTreeMap<String, u64>,
    pub run_units: Vec<SurveillanceRunMissingness>,
}

/// Fail-closed diagnostics attached to every native Path-B model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MarkovDiagnostics {
    /// Overall diagnostic status.
    pub status: MarkovDiagnosticStatus,
    /// Exact reasons for an inconclusive status.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inconclusive_reasons: Vec<String>,
    /// Maximum normalized directional-flux asymmetry measured from the
    /// unconstrained directed transition counts. This is the evidence-bearing
    /// reversibility diagnostic.
    pub detailed_balance_max_abs_residual: f64,
    /// Numerical residual of the deliberately constrained reversible operator.
    /// This is an implementation check, not evidence for physical reversibility.
    pub constrained_reversible_numerical_residual: f64,
    /// Maximum Chapman-Kolmogorov residual between observed `P(2tau)` and `P(tau)^2`.
    pub chapman_kolmogorov_max_abs_residual: f64,
    /// Maximum lane-specific hard assignment count by shared global state.
    pub state_effective_counts: BTreeMap<String, u64>,
    /// Minimum of the maximum lane-specific counts across global states.
    pub minimum_state_effective_count: u64,
    /// At least two lag-specific implied-timescale estimates.
    pub implied_timescales: Vec<ImpliedTimescaleEstimate>,
    /// Relative spread across finite lag-specific implied timescales.
    pub implied_timescale_relative_spread: f64,
    /// Maximum lane-specific seed support for every global state.
    pub state_seed_support: BTreeMap<String, usize>,
    /// Deterministic LOSO VAMP-E state/lag selection ledger.
    pub kinetic_model_selection: KineticModelSelectionLedger,
    /// Full scheduled/evaluable missingness record, including rows excluded
    /// from exact-lag transition construction.
    pub surveillance_missingness: SurveillanceMissingnessSummary,
    /// Independent replica seeds entering bootstrap resampling.
    pub independent_seed_count: usize,
    /// Number of deterministic seed/block bootstrap replicates.
    pub bootstrap_replicates: usize,
    /// Maximum non-wrapping transition-block length within one
    /// run/stream/evaluable segment of each sampled seed.
    pub bootstrap_block_length: usize,
    /// Transition-probability uncertainty.
    pub transition_uncertainty: Vec<TransitionBootstrapInterval>,
    /// Committor uncertainty.
    pub committor_uncertainty: Vec<CommittorBootstrapInterval>,
    /// Explicit availability of a physically defined committor target basin.
    pub committor_status: String,
    /// Per-Hamiltonian diagnostics. Overall scalar residuals above are
    /// conservative maxima/minima over these rows, never pooled operators.
    pub lane_diagnostics: Vec<LaneMarkovDiagnostics>,
}

/// Markov gates evaluated independently inside one matched physical lane.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LaneMarkovDiagnostics {
    pub control_lane: ControlLane,
    /// Shared global state identifiers present in this lane's fitted kinetic
    /// subspace. Diagnostic matrix dimensions are defined by this list.
    pub active_state_ids: Vec<String>,
    pub detailed_balance_max_abs_residual: f64,
    pub constrained_reversible_numerical_residual: f64,
    pub directed_count_reversibility: DirectedCountReversibilityDiagnostic,
    pub chapman_kolmogorov_max_abs_residual: f64,
    pub state_effective_counts: BTreeMap<String, u64>,
    pub minimum_state_effective_count: u64,
    pub implied_timescales: Vec<ImpliedTimescaleEstimate>,
    pub implied_timescale_relative_spread: f64,
    pub state_seed_support: BTreeMap<String, usize>,
    pub independent_seed_count: usize,
    pub transition_pairs: u64,
    pub second_lag_transition_pairs: u64,
    pub transition_uncertainty: Vec<TransitionBootstrapInterval>,
    pub diagnostic_pass: bool,
    pub ineligible_reasons: Vec<String>,
}

/// Seed/block-bootstrap uncertainty for an unconstrained directed-count
/// reversibility statistic. The statistic is the maximum over state pairs of
/// `|N_ij - N_ji| / total_transition_pairs`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DirectedCountReversibilityDiagnostic {
    pub statistic_id: String,
    pub observed_max_abs_flux_asymmetry: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bootstrap_mean: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bootstrap_lower: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bootstrap_upper: Option<f64>,
    pub bootstrap_replicates: usize,
    pub transition_pairs: u64,
}

/// Hashable native Path-B model body.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NativePathBModelBody {
    /// Integration-step lag used by the transition model.
    pub lag_steps: u64,
    /// Exact signed integration timebase used to interpret step-index lags.
    pub timebase_context: NativePathBTimebaseContext,
    /// Common fixed-cadence interval of every admitted surveillance ledger.
    pub surveillance_cadence_steps: u64,
    /// Floating-point normalization tolerance.
    pub normalization_tolerance: f64,
    /// Declared global metastable states. Their stationary probabilities are
    /// the exact no-probe reference-lane probabilities, with exact zero for a
    /// global state outside the no-probe active subspace; they are never a
    /// pooled estimate.
    pub states: Vec<MetastableState>,
    /// No-probe reference-lane row-stochastic transition matrix over that
    /// lane's adequately supported active subspace.
    pub transitions: Vec<TransitionProbability>,
    /// Reference lane mirrored by `states` and `transitions` for compatible
    /// consumers. Must be `NoProbe`.
    pub reference_lane: ControlLane,
    /// Authoritative lane-specific reversible operators for all three matched
    /// physical Hamiltonians.
    pub lane_transition_models: Vec<LaneTransitionModel>,
    /// Complete Cartesian product of matched lanes and shared global states.
    /// This ledger preserves scheduled exposure and zero-event evidence while
    /// defining each lane operator's adequately supported active subspace.
    pub lane_global_state_support: Vec<LaneGlobalStateSupportEvidence>,
    /// Exact Ghost-to-state assignments.
    pub assignments: Vec<GhostStateAssignment>,
    /// Complete structural-state population for every evaluable scheduled row,
    /// including typed no-Ghost observations.
    pub scheduled_state_assignments: Vec<ScheduledStructuralStateAssignment>,
    /// Hash of the source native Ghost lattice manifest.
    pub source_lattice_manifest_sha256: String,
    /// Hash of the complete pooled exact-connectivity authority population.
    pub source_exact_connectivity_population_sha256: String,
    /// Hash of the canonical accepted-edge rows actually used for kinetics.
    pub source_exact_edge_population_sha256: String,
    /// Markov consistency and bootstrap uncertainty diagnostics.
    pub diagnostics: MarkovDiagnostics,
}

/// Content-addressed native Path-B model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NativePathBModel {
    /// Schema identifier.
    pub schema_kind: String,
    /// Schema version.
    pub schema_version: String,
    /// SHA-256 of the canonical model body.
    pub model_sha256: String,
    /// Complete model body.
    pub body: NativePathBModelBody,
}

impl NativePathBModel {
    /// Materialize a validated model and bind its content identity.
    pub fn materialize(mut body: NativePathBModelBody) -> Result<Self> {
        body.states.sort_by(|a, b| a.state_id.cmp(&b.state_id));
        body.transitions.sort_by(|a, b| {
            (&a.from_state_id, &a.to_state_id).cmp(&(&b.from_state_id, &b.to_state_id))
        });
        for lane_model in &mut body.lane_transition_models {
            lane_model.active_state_ids.sort();
            lane_model.transitions.sort_by(|a, b| {
                (&a.from_state_id, &a.to_state_id).cmp(&(&b.from_state_id, &b.to_state_id))
            });
        }
        body.lane_transition_models
            .sort_by_key(|model| model.control_lane);
        body.lane_global_state_support
            .sort_by(|a, b| (a.control_lane, &a.state_id).cmp(&(b.control_lane, &b.state_id)));
        body.diagnostics
            .lane_diagnostics
            .sort_by_key(|diagnostic| diagnostic.control_lane);
        body.assignments
            .sort_by(|a, b| a.ghost_lineage_sha256.cmp(&b.ghost_lineage_sha256));
        body.scheduled_state_assignments.sort_by(|a, b| {
            (
                &a.source_surveillance_ledger_sha256,
                a.stream_id,
                a.step_idx,
            )
                .cmp(&(
                    &b.source_surveillance_ledger_sha256,
                    b.stream_id,
                    b.step_idx,
                ))
        });
        validate_model_body(&body)?;
        let model_sha256 = canonical_sha256(&body)?;
        Ok(Self {
            schema_kind: PATHB_CHEMICAL_STATE_SCHEMA.to_string(),
            schema_version: PATHB_CHEMICAL_STATE_VERSION.to_string(),
            model_sha256,
            body,
        })
    }

    /// Verify content identity and every normalization invariant.
    pub fn verify(&self) -> Result<()> {
        if self.schema_kind != PATHB_CHEMICAL_STATE_SCHEMA
            || self.schema_version != PATHB_CHEMICAL_STATE_VERSION
        {
            bail!("unsupported native Path-B schema");
        }
        validate_model_body(&self.body)?;
        if canonical_sha256(&self.body)? != self.model_sha256 {
            bail!("native Path-B model content hash mismatch");
        }
        Ok(())
    }

    /// Project one exact Ghost byte-lineage occurrence to its dominant
    /// metastable state.
    pub fn project(&self, lineage: &GhostByteLineage) -> Result<PathBProjection> {
        self.verify()?;
        let ghost_lineage_sha256 = lineage.lineage_sha256()?;
        let assignment = self
            .body
            .assignments
            .binary_search_by(|a| {
                a.ghost_lineage_sha256
                    .as_str()
                    .cmp(ghost_lineage_sha256.as_str())
            })
            .ok()
            .map(|index| &self.body.assignments[index])
            .context("Ghost lineage is absent from native Path-B assignments")?;
        if assignment.ghost_record_sha256 != lineage.record_sha256 {
            bail!("Path-B assignment record hash differs from exact Ghost occurrence");
        }
        let (dominant_state, membership) = assignment
            .state_membership
            .iter()
            .max_by(|(id_a, probability_a), (id_b, probability_b)| {
                probability_a
                    .total_cmp(probability_b)
                    .then_with(|| id_b.cmp(id_a))
            })
            .context("Path-B assignment has no state memberships")?;
        let lane_model = self
            .body
            .lane_transition_models
            .iter()
            .find(|model| model.control_lane == assignment.control_lane)
            .context("assignment physical lane has no reversible transition operator")?;
        let active_state_ids = lane_model
            .active_state_ids
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        let transition = assignment
            .previous_state_id
            .as_deref()
            .filter(|from_state| {
                active_state_ids.contains(from_state)
                    && active_state_ids.contains(dominant_state.as_str())
            })
            .map(|from_state| {
                lane_model
                    .transitions
                    .iter()
                    .find(|entry| {
                        entry.from_state_id == from_state && entry.to_state_id == *dominant_state
                    })
                    .map(|entry| (from_state.to_string(), entry.probability))
                    .context("dominant Path-B transition is missing from active lane subspace")
            })
            .transpose()?;
        let (transition_from_state_id, transition_to_state_id, transition_probability) =
            match transition {
                Some((from, probability)) => {
                    (Some(from), Some(dominant_state.clone()), Some(probability))
                }
                None => (None, None, None),
            };
        Ok(PathBProjection {
            model_sha256: self.model_sha256.clone(),
            control_lane: assignment.control_lane,
            metastable_state_id: dominant_state.clone(),
            state_membership_probability: *membership,
            transition_from_state_id,
            transition_to_state_id,
            transition_probability,
            forward_committor: assignment.forward_committor,
            lag_steps: self.body.lag_steps,
            source_ghost_record_sha256: lineage.record_sha256.clone(),
            source_ghost_lineage_sha256: ghost_lineage_sha256,
        })
    }
}

/// Build endpoint occupancy from structural state assignments rather than
/// Ghost-event fibers. Any truly non-evaluable scheduled frame fails closed so
/// the denominator can equal the complete declared time at risk.
pub fn estimate_scheduled_open_occupancy(
    model: &NativePathBModel,
    mut open_state_ids: Vec<String>,
) -> Result<ScheduledOpenOccupancyPopulation> {
    model.verify()?;
    open_state_ids.sort();
    if open_state_ids.is_empty()
        || open_state_ids.windows(2).any(|pair| pair[0] == pair[1])
        || open_state_ids
            .iter()
            .any(|id| !model.body.states.iter().any(|state| &state.state_id == id))
    {
        bail!("scheduled occupancy open-state declaration is empty, duplicate, or unknown");
    }
    let missingness = &model.body.diagnostics.surveillance_missingness;
    if missingness.not_evaluable_rows != 0
        || missingness.evaluable_rows != missingness.scheduled_rows
    {
        bail!(
            "EVIDENCE_NOT_FOUND: complete scheduled structural time at risk is required for occupancy"
        );
    }

    type SeedKey = (String, String, u64, ControlLane);
    type StepKey = (String, String, u64, ControlLane, u64);
    let mut step_rows = BTreeMap::<StepKey, (f64, u64, u64)>::new();
    for assignment in &model.body.scheduled_state_assignments {
        let open_probability = open_state_ids
            .iter()
            .map(|state| assignment.state_membership[state])
            .sum::<f64>();
        if !open_probability.is_finite() || !(0.0..=1.0 + 1.0e-12).contains(&open_probability) {
            bail!("scheduled open-state membership is not a probability");
        }
        let row = step_rows
            .entry((
                assignment.run_id.clone(),
                assignment.replica_id.clone(),
                assignment.replica_seed,
                assignment.control_lane,
                assignment.step_idx,
            ))
            .or_default();
        row.0 += open_probability.min(1.0);
        row.1 += 1;
        if matches!(
            assignment.ghost_event,
            ScheduledGhostEventStatus::NotEmitted { .. }
        ) {
            row.2 += 1;
        }
    }

    let mut by_seed = BTreeMap::<SeedKey, (f64, u64, u64, u64)>::new();
    for ((run_id, replica_id, seed, lane, _step), (sum, streams, no_ghost)) in step_rows {
        if streams == 0 {
            bail!("scheduled occupancy step has no stream observations");
        }
        let seed_row = by_seed.entry((run_id, replica_id, seed, lane)).or_default();
        seed_row.0 += sum / streams as f64;
        seed_row.1 += 1;
        seed_row.2 += streams;
        seed_row.3 += no_ghost;
    }
    let run_missingness = missingness
        .run_units
        .iter()
        .map(|unit| {
            (
                (
                    unit.run_id.clone(),
                    unit.replica_id.clone(),
                    unit.replica_seed,
                    unit.control_lane,
                ),
                unit,
            )
        })
        .collect::<BTreeMap<_, _>>();
    if run_missingness.len() != missingness.run_units.len()
        || by_seed.keys().collect::<BTreeSet<_>>()
            != run_missingness.keys().collect::<BTreeSet<_>>()
    {
        bail!("scheduled occupancy run population differs from missingness authority");
    }
    let mut seed_estimates = Vec::with_capacity(by_seed.len());
    for (key, (sum_step_means, step_count, stream_rows, no_ghost_rows)) in by_seed {
        let authority = run_missingness[&key];
        if authority.not_evaluable_rows != 0
            || authority.evaluable_rows != stream_rows
            || authority.scheduled_rows != stream_rows
            || step_count == 0
        {
            bail!("scheduled occupancy does not exactly cover one run's time at risk");
        }
        seed_estimates.push(ScheduledSeedOpenOccupancyEstimate {
            run_id: key.0,
            replica_id: key.1,
            replica_seed: key.2,
            control_lane: key.3,
            scheduled_rows: authority.scheduled_rows,
            structurally_evaluable_rows: authority.evaluable_rows,
            not_evaluable_rows: authority.not_evaluable_rows,
            averaged_step_count: step_count,
            stream_row_count: stream_rows,
            typed_no_ghost_rows: no_ghost_rows,
            mean_open_probability: sum_step_means / step_count as f64,
        });
    }
    let population_sha256 = canonical_sha256(&(
        model.model_sha256.as_str(),
        &open_state_ids,
        &seed_estimates,
    ))?;
    Ok(ScheduledOpenOccupancyPopulation {
        schema_kind: "prism_scheduled_open_occupancy_population".to_string(),
        schema_version: "1.0.0".to_string(),
        population_sha256,
        pathb_model_sha256: model.model_sha256.clone(),
        open_state_ids,
        seed_estimates,
    })
}

fn validate_model_body(body: &NativePathBModelBody) -> Result<()> {
    require_sha256(
        &body.source_lattice_manifest_sha256,
        "source_lattice_manifest_sha256",
    )?;
    require_sha256(
        &body.source_exact_connectivity_population_sha256,
        "source_exact_connectivity_population_sha256",
    )?;
    require_sha256(
        &body.source_exact_edge_population_sha256,
        "source_exact_edge_population_sha256",
    )?;
    body.timebase_context.verify()?;
    if body.lag_steps == 0
        || body.surveillance_cadence_steps == 0
        || body.lag_steps % body.surveillance_cadence_steps != 0
    {
        bail!("native Path-B lag must be non-zero and exactly cadence-aligned");
    }
    if !body.normalization_tolerance.is_finite()
        || body.normalization_tolerance <= 0.0
        || body.normalization_tolerance > 0.01
    {
        bail!("normalization_tolerance must be finite and in (0, 0.01]");
    }
    if body.states.len() < 2 {
        bail!("native Path-B requires at least two metastable states");
    }
    let mut state_ids = BTreeSet::new();
    let mut stationary_sum = 0.0;
    for state in &body.states {
        if state.state_id.trim().is_empty() || !state.stationary_probability.is_finite() {
            bail!("invalid metastable state");
        }
        if !(0.0..=1.0).contains(&state.stationary_probability) {
            bail!("stationary probability outside [0, 1]");
        }
        if !state_ids.insert(state.state_id.clone()) {
            bail!("duplicate metastable state id");
        }
        stationary_sum += state.stationary_probability;
    }
    require_normalized(
        stationary_sum,
        body.normalization_tolerance,
        "stationary probabilities",
    )?;

    let mut rows: BTreeMap<&str, BTreeMap<&str, f64>> = BTreeMap::new();
    for transition in &body.transitions {
        if !state_ids.contains(&transition.from_state_id)
            || !state_ids.contains(&transition.to_state_id)
        {
            bail!("transition references an undeclared state");
        }
        if !transition.probability.is_finite() || !(0.0..=1.0).contains(&transition.probability) {
            bail!("transition probability outside [0, 1]");
        }
        if rows
            .entry(&transition.from_state_id)
            .or_default()
            .insert(&transition.to_state_id, transition.probability)
            .is_some()
        {
            bail!("duplicate transition matrix entry");
        }
    }
    let transition_state_ids = body
        .transitions
        .iter()
        .flat_map(|transition| {
            [
                transition.from_state_id.as_str(),
                transition.to_state_id.as_str(),
            ]
        })
        .collect::<BTreeSet<_>>();
    if transition_state_ids.is_empty() {
        bail!("reference transition matrix has no active states");
    }
    for from in &transition_state_ids {
        let row = rows
            .get(*from)
            .context("transition matrix is missing a source row")?;
        if row.len() != transition_state_ids.len()
            || transition_state_ids.iter().any(|to| !row.contains_key(*to))
        {
            bail!("transition matrix row is incomplete");
        }
        require_normalized(
            row.values().sum(),
            body.normalization_tolerance,
            "transition matrix row",
        )?;
    }

    if body.reference_lane != ControlLane::NoProbe {
        bail!("native Path-B reference operator must be the no-probe lane");
    }
    let required_lanes = BTreeSet::from([
        ControlLane::NoProbe,
        ControlLane::HypothesisFragment,
        ControlLane::OrthogonalChemicalControl,
    ]);
    let supplied_lanes = body
        .lane_transition_models
        .iter()
        .map(|model| model.control_lane)
        .collect::<BTreeSet<_>>();
    if supplied_lanes != required_lanes || body.lane_transition_models.len() != 3 {
        bail!("native Path-B requires exactly one reversible operator per matched lane");
    }

    let mut support_by_lane_state = BTreeMap::new();
    let mut globally_observed_states = BTreeSet::new();
    for evidence in &body.lane_global_state_support {
        if !required_lanes.contains(&evidence.control_lane)
            || !state_ids.contains(&evidence.state_id)
            || evidence.scheduled_exposure_rows == 0
            || evidence.hard_assignment_rows > evidence.scheduled_exposure_rows
            || evidence.independent_seed_support > evidence.hard_assignment_rows as usize
            || evidence.minimum_required_assignment_rows == 0
            || evidence.minimum_required_independent_seeds == 0
        {
            bail!("invalid lane-by-global-state support evidence");
        }
        let expected_status = if evidence.hard_assignment_rows == 0 {
            if evidence.independent_seed_support != 0 {
                bail!("zero-event lane-state evidence has nonzero seed support");
            }
            LaneGlobalStateSupportStatus::NotObserved
        } else if evidence.hard_assignment_rows >= evidence.minimum_required_assignment_rows
            && evidence.independent_seed_support >= evidence.minimum_required_independent_seeds
        {
            LaneGlobalStateSupportStatus::Observed
        } else {
            LaneGlobalStateSupportStatus::BelowSupport
        };
        if evidence.status != expected_status {
            bail!("lane-state support status disagrees with exact evidence counts");
        }
        if support_by_lane_state
            .insert((evidence.control_lane, evidence.state_id.clone()), evidence)
            .is_some()
        {
            bail!("duplicate lane-by-global-state support evidence");
        }
        if evidence.status == LaneGlobalStateSupportStatus::Observed {
            globally_observed_states.insert(evidence.state_id.clone());
        }
    }
    if support_by_lane_state.len() != required_lanes.len() * state_ids.len()
        || required_lanes.iter().any(|lane| {
            state_ids
                .iter()
                .any(|state| !support_by_lane_state.contains_key(&(*lane, state.clone())))
        })
        || globally_observed_states != state_ids
        || body
            .lane_global_state_support
            .iter()
            .map(|row| row.minimum_required_assignment_rows)
            .collect::<BTreeSet<_>>()
            .len()
            != 1
        || body
            .lane_global_state_support
            .iter()
            .map(|row| row.minimum_required_independent_seeds)
            .collect::<BTreeSet<_>>()
            .len()
            != 1
    {
        bail!("lane-state support ledger lacks Cartesian coverage, global support, or one gate authority");
    }

    for lane_model in &body.lane_transition_models {
        let active_state_ids = lane_model
            .active_state_ids
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        let evidenced_active_state_ids = state_ids
            .iter()
            .filter(|state| {
                support_by_lane_state[&(lane_model.control_lane, (*state).clone())].status
                    == LaneGlobalStateSupportStatus::Observed
            })
            .cloned()
            .collect::<BTreeSet<_>>();
        if active_state_ids.is_empty()
            || active_state_ids.len() != lane_model.active_state_ids.len()
            || active_state_ids != evidenced_active_state_ids
            || lane_model
                .stationary_probability_by_state
                .keys()
                .cloned()
                .collect::<BTreeSet<_>>()
                != active_state_ids
        {
            bail!("lane stationary distribution does not exactly cover its active subspace");
        }
        for probability in lane_model.stationary_probability_by_state.values() {
            if !probability.is_finite() || !(0.0..=1.0).contains(probability) {
                bail!("lane stationary probability outside [0, 1]");
            }
        }
        require_normalized(
            lane_model.stationary_probability_by_state.values().sum(),
            body.normalization_tolerance,
            "lane stationary probabilities",
        )?;
        let mut lane_rows: BTreeMap<&str, BTreeMap<&str, f64>> = BTreeMap::new();
        for transition in &lane_model.transitions {
            if !active_state_ids.contains(&transition.from_state_id)
                || !active_state_ids.contains(&transition.to_state_id)
                || !transition.probability.is_finite()
                || !(0.0..=1.0).contains(&transition.probability)
                || lane_rows
                    .entry(&transition.from_state_id)
                    .or_default()
                    .insert(&transition.to_state_id, transition.probability)
                    .is_some()
            {
                bail!("invalid lane-specific transition operator");
            }
        }
        for from in &active_state_ids {
            let row = lane_rows
                .get(from.as_str())
                .context("lane transition matrix is missing a source row")?;
            if row.len() != active_state_ids.len()
                || active_state_ids
                    .iter()
                    .any(|to| !row.contains_key(to.as_str()))
            {
                bail!("lane transition matrix row is incomplete");
            }
            require_normalized(
                row.values().sum(),
                body.normalization_tolerance,
                "lane transition matrix row",
            )?;
        }
    }
    let reference = body
        .lane_transition_models
        .iter()
        .find(|model| model.control_lane == body.reference_lane)
        .unwrap();
    let reference_stationary = body
        .states
        .iter()
        .filter_map(|state| {
            reference
                .active_state_ids
                .contains(&state.state_id)
                .then_some((state.state_id.clone(), state.stationary_probability))
        })
        .collect::<BTreeMap<_, _>>();
    if reference.stationary_probability_by_state != reference_stationary
        || reference.transitions != body.transitions
        || body.states.iter().any(|state| {
            !reference.active_state_ids.contains(&state.state_id)
                && state.stationary_probability.to_bits() != 0.0f64.to_bits()
        })
    {
        bail!("compatible reference operator differs from authoritative no-probe operator");
    }

    let mut assignment_lineages = BTreeSet::new();
    for assignment in &body.assignments {
        require_sha256(&assignment.ghost_lineage_sha256, "ghost_lineage_sha256")?;
        require_sha256(&assignment.ghost_record_sha256, "ghost_record_sha256")?;
        if !assignment_lineages.insert(&assignment.ghost_lineage_sha256) {
            bail!("duplicate exact Ghost occurrence assignment");
        }
        if !required_lanes.contains(&assignment.control_lane) {
            bail!("Path-B assignment references an undeclared physical lane");
        }
        if assignment.state_membership.len() != state_ids.len()
            || assignment
                .state_membership
                .keys()
                .any(|id| !state_ids.contains(id))
        {
            bail!("Path-B assignment must cover every declared state exactly once");
        }
        let mut sum = 0.0;
        for probability in assignment.state_membership.values() {
            if !probability.is_finite() || !(0.0..=1.0).contains(probability) {
                bail!("state membership probability outside [0, 1]");
            }
            sum += probability;
        }
        require_normalized(sum, body.normalization_tolerance, "state membership")?;
        if let Some(previous) = &assignment.previous_state_id {
            if !state_ids.contains(previous) {
                bail!("previous Path-B state is undeclared");
            }
        }
        if assignment
            .forward_committor
            .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
        {
            bail!("forward committor outside [0, 1]");
        }
    }
    if body.scheduled_state_assignments.len() as u64
        != body.diagnostics.surveillance_missingness.evaluable_rows
        || body.scheduled_state_assignments.windows(2).any(|pair| {
            (
                pair[0].source_surveillance_ledger_sha256.as_str(),
                pair[0].stream_id,
                pair[0].step_idx,
            ) >= (
                pair[1].source_surveillance_ledger_sha256.as_str(),
                pair[1].stream_id,
                pair[1].step_idx,
            )
        })
    {
        bail!("scheduled structural-state assignments do not exactly cover evaluable rows");
    }
    let mut uniquely_emitted = BTreeMap::new();
    for assignment in &body.scheduled_state_assignments {
        require_sha256(
            &assignment.source_surveillance_ledger_sha256,
            "scheduled assignment surveillance ledger",
        )?;
        require_sha256(
            &assignment.source_surveillance_row_sha256,
            "scheduled assignment surveillance row",
        )?;
        if assignment.run_id.trim().is_empty()
            || assignment.replica_id.trim().is_empty()
            || !required_lanes.contains(&assignment.control_lane)
            || !assignment.cavity_volume_a3.is_finite()
            || assignment.cavity_volume_a3 < 0.0
            || assignment.state_membership.len() != state_ids.len()
            || assignment
                .state_membership
                .keys()
                .any(|id| !state_ids.contains(id))
            || !state_ids.contains(&assignment.dominant_state_id)
        {
            bail!("scheduled structural-state assignment is incomplete");
        }
        let mut membership_sum = 0.0;
        for probability in assignment.state_membership.values() {
            if !probability.is_finite() || !(0.0..=1.0).contains(probability) {
                bail!("scheduled state-membership probability outside [0, 1]");
            }
            membership_sum += probability;
        }
        require_normalized(
            membership_sum,
            body.normalization_tolerance,
            "scheduled state membership",
        )?;
        let dominant = assignment
            .state_membership
            .iter()
            .max_by(|(id_a, probability_a), (id_b, probability_b)| {
                probability_a
                    .total_cmp(probability_b)
                    .then_with(|| id_b.cmp(id_a))
            })
            .map(|(id, _)| id)
            .context("scheduled assignment has no dominant state")?;
        if dominant != &assignment.dominant_state_id {
            bail!("scheduled assignment dominant state disagrees with memberships");
        }
        match &assignment.ghost_event {
            ScheduledGhostEventStatus::Emitted {
                ghost_lineage_sha256,
                exact_component_sha256,
            } => {
                require_sha256(ghost_lineage_sha256, "scheduled emitted Ghost lineage")?;
                require_sha256(exact_component_sha256, "scheduled emitted component")?;
                if uniquely_emitted
                    .insert(ghost_lineage_sha256.as_str(), assignment.control_lane)
                    .is_some()
                {
                    bail!("one Ghost occurrence is emitted by multiple scheduled assignments");
                }
            }
            ScheduledGhostEventStatus::AmbiguousEmissions {
                ghost_lineage_sha256,
                exact_component_sha256,
            } => {
                if ghost_lineage_sha256.len() < 2
                    || ghost_lineage_sha256.len() != exact_component_sha256.len()
                    || ghost_lineage_sha256
                        .windows(2)
                        .any(|pair| pair[0] >= pair[1])
                    || exact_component_sha256
                        .windows(2)
                        .any(|pair| pair[0] >= pair[1])
                {
                    bail!("scheduled ambiguous Ghost population is not canonical");
                }
                for hash in ghost_lineage_sha256 {
                    require_sha256(hash, "scheduled ambiguous Ghost lineage")?;
                }
                for hash in exact_component_sha256 {
                    require_sha256(hash, "scheduled ambiguous component")?;
                }
            }
            ScheduledGhostEventStatus::NotEmitted {
                typed_absence_receipt_sha256,
            } => require_sha256(
                typed_absence_receipt_sha256,
                "scheduled typed absence receipt",
            )?,
        }
    }
    let mut scheduled_exposure_by_lane = BTreeMap::<ControlLane, u64>::new();
    let mut scheduled_hard_counts = BTreeMap::<(ControlLane, String), u64>::new();
    let mut scheduled_seed_support = BTreeMap::<(ControlLane, String), BTreeSet<u64>>::new();
    for assignment in &body.scheduled_state_assignments {
        *scheduled_exposure_by_lane
            .entry(assignment.control_lane)
            .or_default() += 1;
        *scheduled_hard_counts
            .entry((
                assignment.control_lane,
                assignment.dominant_state_id.clone(),
            ))
            .or_default() += 1;
        scheduled_seed_support
            .entry((
                assignment.control_lane,
                assignment.dominant_state_id.clone(),
            ))
            .or_default()
            .insert(assignment.replica_seed);
    }
    for ((lane, state), evidence) in &support_by_lane_state {
        if evidence.scheduled_exposure_rows
            != scheduled_exposure_by_lane.get(lane).copied().unwrap_or(0)
            || evidence.hard_assignment_rows
                != scheduled_hard_counts
                    .get(&(*lane, state.clone()))
                    .copied()
                    .unwrap_or(0)
            || evidence.independent_seed_support
                != scheduled_seed_support
                    .get(&(*lane, state.clone()))
                    .map(BTreeSet::len)
                    .unwrap_or(0)
        {
            bail!("lane-state support ledger differs from scheduled structural exposure");
        }
    }
    if body.assignments.iter().any(|assignment| {
        uniquely_emitted.get(assignment.ghost_lineage_sha256.as_str())
            != Some(&assignment.control_lane)
    }) {
        bail!("Ghost assignment lacks its unique lane-matched scheduled structural observation");
    }
    validate_markov_diagnostics(
        &body.diagnostics,
        &state_ids,
        &body.lane_global_state_support,
    )?;
    Ok(())
}

fn validate_markov_diagnostics(
    diagnostics: &MarkovDiagnostics,
    state_ids: &BTreeSet<String>,
    lane_global_state_support: &[LaneGlobalStateSupportEvidence],
) -> Result<()> {
    for value in [
        diagnostics.detailed_balance_max_abs_residual,
        diagnostics.constrained_reversible_numerical_residual,
        diagnostics.chapman_kolmogorov_max_abs_residual,
        diagnostics.implied_timescale_relative_spread,
    ] {
        if !value.is_finite() || value < 0.0 {
            bail!("Markov diagnostic residual is invalid");
        }
    }
    if diagnostics
        .state_effective_counts
        .keys()
        .collect::<BTreeSet<_>>()
        != state_ids.iter().collect::<BTreeSet<_>>()
        || diagnostics.minimum_state_effective_count
            != diagnostics
                .state_effective_counts
                .values()
                .copied()
                .min()
                .unwrap_or(0)
    {
        bail!("Markov state effective counts do not exactly cover model states");
    }
    if diagnostics
        .state_seed_support
        .keys()
        .collect::<BTreeSet<_>>()
        != state_ids.iter().collect::<BTreeSet<_>>()
        || diagnostics
            .state_seed_support
            .values()
            .any(|support| *support == 0)
    {
        bail!("Markov state seed support does not exactly cover model states");
    }
    validate_kinetic_model_selection(&diagnostics.kinetic_model_selection)?;
    let missingness = &diagnostics.surveillance_missingness;
    require_sha256(
        &missingness.surveillance_ledger_population_sha256,
        "surveillance missingness ledger population",
    )?;
    if missingness.scheduled_rows == 0
        || missingness.scheduled_rows != missingness.evaluable_rows + missingness.not_evaluable_rows
        || missingness.scheduled_rows
            != missingness.observed_open_rows
                + missingness.observed_closed_rows
                + missingness.observed_indeterminate_rows
                + missingness.not_evaluable_rows
        || missingness.scheduled_rows
            != missingness.ghost_emitted_rows + missingness.ghost_not_emitted_rows
        || !missingness.evaluable_fraction.is_finite()
        || !(0.0..=1.0).contains(&missingness.evaluable_fraction)
        || (missingness.evaluable_fraction
            - missingness.evaluable_rows as f64 / missingness.scheduled_rows as f64)
            .abs()
            > 1.0e-12
        || missingness.run_units.is_empty()
        || missingness.run_units.windows(2).any(|pair| {
            (
                &pair[0].run_id,
                &pair[0].replica_id,
                pair[0].replica_seed,
                pair[0].control_lane,
            ) >= (
                &pair[1].run_id,
                &pair[1].replica_id,
                pair[1].replica_seed,
                pair[1].control_lane,
            )
        })
        || missingness
            .run_units
            .iter()
            .any(|unit| unit.scheduled_rows != unit.evaluable_rows + unit.not_evaluable_rows)
    {
        bail!("invalid fixed-cadence surveillance missingness summary");
    }
    let lag_set = diagnostics
        .implied_timescales
        .iter()
        .map(|estimate| estimate.lag_steps)
        .collect::<BTreeSet<_>>();
    if diagnostics.implied_timescales.is_empty()
        || lag_set.len() != diagnostics.implied_timescales.len()
    {
        bail!("Markov diagnostics require unique observed lag estimates");
    }
    for estimate in &diagnostics.implied_timescales {
        if estimate.control_lane != ControlLane::NoProbe
            || estimate.lag_steps == 0
            || estimate.transition_pairs == 0
            || !estimate.second_eigenvalue_magnitude.is_finite()
            || !(0.0..=1.0).contains(&estimate.second_eigenvalue_magnitude)
            || estimate
                .implied_timescale_steps
                .is_some_and(|value| !value.is_finite() || value <= 0.0)
        {
            bail!("invalid implied-timescale diagnostic");
        }
    }
    let required_lanes = BTreeSet::from([
        ControlLane::NoProbe,
        ControlLane::HypothesisFragment,
        ControlLane::OrthogonalChemicalControl,
    ]);
    let diagnostic_lanes = diagnostics
        .lane_diagnostics
        .iter()
        .map(|row| row.control_lane)
        .collect::<BTreeSet<_>>();
    if diagnostic_lanes != required_lanes || diagnostics.lane_diagnostics.len() != 3 {
        bail!("Markov diagnostics require one independent row per physical lane");
    }
    let support_by_lane_state = lane_global_state_support
        .iter()
        .map(|row| ((row.control_lane, row.state_id.as_str()), row))
        .collect::<BTreeMap<_, _>>();
    for lane in &diagnostics.lane_diagnostics {
        let active_state_ids = lane.active_state_ids.iter().collect::<BTreeSet<_>>();
        let evidenced_active_state_ids = state_ids
            .iter()
            .filter(|state| {
                support_by_lane_state
                    .get(&(lane.control_lane, state.as_str()))
                    .is_some_and(|row| row.status == LaneGlobalStateSupportStatus::Observed)
            })
            .collect::<BTreeSet<_>>();
        for value in [
            lane.detailed_balance_max_abs_residual,
            lane.constrained_reversible_numerical_residual,
            lane.chapman_kolmogorov_max_abs_residual,
            lane.implied_timescale_relative_spread,
        ] {
            if !value.is_finite() || value < 0.0 {
                bail!("invalid lane-specific Markov diagnostic residual");
            }
        }
        if active_state_ids.is_empty()
            || active_state_ids.len() != lane.active_state_ids.len()
            || active_state_ids != evidenced_active_state_ids
            || lane.transition_pairs == 0
            || lane.second_lag_transition_pairs == 0
            || lane.implied_timescales.is_empty()
            || lane
                .implied_timescales
                .iter()
                .map(|estimate| estimate.lag_steps)
                .collect::<BTreeSet<_>>()
                .len()
                != lane.implied_timescales.len()
            || lane.state_effective_counts.keys().collect::<BTreeSet<_>>()
                != state_ids.iter().collect::<BTreeSet<_>>()
            || lane.minimum_state_effective_count
                != lane
                    .active_state_ids
                    .iter()
                    .map(|state| lane.state_effective_counts[state])
                    .min()
                    .unwrap_or(0)
            || lane.state_seed_support.keys().collect::<BTreeSet<_>>()
                != state_ids.iter().collect::<BTreeSet<_>>()
            || lane
                .implied_timescales
                .iter()
                .any(|estimate| estimate.control_lane != lane.control_lane)
            || lane
                .transition_uncertainty
                .iter()
                .any(|interval| interval.control_lane != lane.control_lane)
            || (lane.diagnostic_pass != lane.ineligible_reasons.is_empty())
        {
            bail!("invalid lane-specific Markov diagnostic population");
        }
        for state in state_ids {
            let evidence = support_by_lane_state
                .get(&(lane.control_lane, state.as_str()))
                .context("lane diagnostics lack typed state-support evidence")?;
            if lane.state_effective_counts[state] != evidence.hard_assignment_rows
                || lane.state_seed_support[state] != evidence.independent_seed_support
            {
                bail!("lane diagnostics disagree with typed state-support evidence");
            }
        }
        for estimate in &lane.implied_timescales {
            if estimate.control_lane != lane.control_lane
                || estimate.lag_steps == 0
                || estimate.transition_pairs == 0
                || !estimate.second_eigenvalue_magnitude.is_finite()
                || !(0.0..=1.0).contains(&estimate.second_eigenvalue_magnitude)
                || estimate
                    .implied_timescale_steps
                    .is_some_and(|value| !value.is_finite() || value <= 0.0)
            {
                bail!("invalid lane-specific implied timescale");
            }
        }
        for interval in &lane.transition_uncertainty {
            if interval.control_lane != lane.control_lane
                || !active_state_ids.contains(&interval.from_state_id)
                || !active_state_ids.contains(&interval.to_state_id)
                || !valid_probability_interval(interval.mean, interval.lower, interval.upper)
            {
                bail!("invalid lane-specific transition bootstrap interval");
            }
        }
        let directed = &lane.directed_count_reversibility;
        let bootstrap_complete = match (
            directed.bootstrap_mean,
            directed.bootstrap_lower,
            directed.bootstrap_upper,
        ) {
            (Some(mean), Some(lower), Some(upper)) => {
                valid_probability_interval(mean, lower, upper) && directed.bootstrap_replicates > 0
            }
            (None, None, None) => directed.bootstrap_replicates == 0,
            _ => false,
        };
        if directed.statistic_id != "maximum_unconstrained_directed_flux_asymmetry_v1"
            || !directed.observed_max_abs_flux_asymmetry.is_finite()
            || !(0.0..=1.0).contains(&directed.observed_max_abs_flux_asymmetry)
            || directed.observed_max_abs_flux_asymmetry.to_bits()
                != lane.detailed_balance_max_abs_residual.to_bits()
            || directed.transition_pairs != lane.transition_pairs
            || !bootstrap_complete
        {
            bail!("invalid unconstrained directed-count reversibility diagnostic");
        }
        if lane.diagnostic_pass
            && (lane.independent_seed_count < 5
                || lane.transition_uncertainty.len()
                    != active_state_ids.len() * active_state_ids.len()
                || directed.bootstrap_replicates < 100)
        {
            bail!("passing lane diagnostics lack seed/bootstrap support");
        }
    }
    let no_probe_diagnostics = diagnostics
        .lane_diagnostics
        .iter()
        .find(|row| row.control_lane == ControlLane::NoProbe)
        .unwrap();
    if diagnostics.implied_timescales != no_probe_diagnostics.implied_timescales
        || diagnostics.detailed_balance_max_abs_residual.to_bits()
            != diagnostics
                .lane_diagnostics
                .iter()
                .map(|row| row.detailed_balance_max_abs_residual)
                .fold(0.0_f64, f64::max)
                .to_bits()
        || diagnostics
            .constrained_reversible_numerical_residual
            .to_bits()
            != diagnostics
                .lane_diagnostics
                .iter()
                .map(|row| row.constrained_reversible_numerical_residual)
                .fold(0.0_f64, f64::max)
                .to_bits()
        || diagnostics.chapman_kolmogorov_max_abs_residual.to_bits()
            != diagnostics
                .lane_diagnostics
                .iter()
                .map(|row| row.chapman_kolmogorov_max_abs_residual)
                .fold(0.0_f64, f64::max)
                .to_bits()
        || diagnostics.implied_timescale_relative_spread.to_bits()
            != diagnostics
                .lane_diagnostics
                .iter()
                .map(|row| row.implied_timescale_relative_spread)
                .fold(0.0_f64, f64::max)
                .to_bits()
        || diagnostics.independent_seed_count
            != diagnostics
                .lane_diagnostics
                .iter()
                .map(|row| row.independent_seed_count)
                .min()
                .unwrap_or(0)
        || diagnostics.status == MarkovDiagnosticStatus::Pass
            && diagnostics
                .lane_diagnostics
                .iter()
                .any(|row| !row.diagnostic_pass)
    {
        bail!("overall Markov diagnostics are not the conservative lane summary");
    }
    if diagnostics.status == MarkovDiagnosticStatus::Pass {
        if !diagnostics.inconclusive_reasons.is_empty()
            || diagnostics.independent_seed_count < 5
            || diagnostics.implied_timescales.len() < 2
            || diagnostics.bootstrap_replicates < 100
            || diagnostics.bootstrap_block_length == 0
            || diagnostics.transition_uncertainty.len()
                != diagnostics
                    .lane_diagnostics
                    .iter()
                    .map(|lane| lane.active_state_ids.len() * lane.active_state_ids.len())
                    .sum::<usize>()
        {
            bail!("PASS Markov diagnostics are missing seed/bootstrap uncertainty");
        }
    } else if diagnostics.inconclusive_reasons.is_empty() {
        bail!("inconclusive Markov diagnostics must record exact reasons");
    }
    for interval in &diagnostics.transition_uncertainty {
        if !required_lanes.contains(&interval.control_lane)
            || !state_ids.contains(&interval.from_state_id)
            || !state_ids.contains(&interval.to_state_id)
            || !valid_probability_interval(interval.mean, interval.lower, interval.upper)
        {
            bail!("invalid transition bootstrap interval");
        }
    }
    for interval in &diagnostics.committor_uncertainty {
        if !state_ids.contains(&interval.state_id)
            || !valid_probability_interval(interval.mean, interval.lower, interval.upper)
        {
            bail!("invalid committor bootstrap interval");
        }
    }
    if diagnostics.committor_status != "UNAVAILABLE_PENDING_HASH_BOUND_OPEN_BASIN"
        || !diagnostics.committor_uncertainty.is_empty()
    {
        bail!("neutral native Path-B model must not fabricate committor evidence");
    }
    Ok(())
}

fn validate_kinetic_model_selection(ledger: &KineticModelSelectionLedger) -> Result<()> {
    for (digest, field) in [
        (&ledger.ledger_sha256, "kinetic selection ledger_sha256"),
        (
            &ledger.feature_schema_sha256,
            "kinetic feature_schema_sha256",
        ),
        (
            &ledger.surveillance_population_sha256,
            "kinetic surveillance_population_sha256",
        ),
    ] {
        require_sha256(digest, field)?;
    }
    if ledger.selection_rule
        != "within_each_lag_maximize_complete_train_fitted_lane_active_subspace_validation_vampe_with_every_fold_gate_then_select_smallest_lag_passing_per_lane_active_subspace_its_ck_seed_support"
        || ledger.candidates.is_empty()
        || ledger.loso_seed_count < 5
        || ledger.selected_kinetic_dimensions == 0
        || !ledger.selected_mean_validation_vampe_score.is_finite()
        || !ledger
            .minimum_required_fold_validation_vampe_score
            .is_finite()
        || !ledger.within_lag_runner_up_score_margin.is_finite()
        || ledger.within_lag_runner_up_score_margin < 0.0
        || ledger.candidates.windows(2).any(|pair| {
            (
                pair[0].lag_steps,
                pair[0].kinetic_dimensions,
                pair[0].state_count,
            ) >= (
                pair[1].lag_steps,
                pair[1].kinetic_dimensions,
                pair[1].state_count,
            )
        })
    {
        bail!("invalid deterministic kinetic model-selection ledger");
    }
    let mut selected_match = 0usize;
    for candidate in &ledger.candidates {
        if candidate.state_count < 2
            || candidate.lag_steps == 0
            || candidate.kinetic_dimensions == 0
            || candidate.fold_seed.len() != candidate.fold_validation_vampe_score.len()
            || candidate
                .fold_seed
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || candidate
                .fold_validation_vampe_score
                .iter()
                .any(|score| !score.is_finite())
            || !candidate.mean_validation_vampe_score.is_finite()
            || !candidate.minimum_fold_validation_vampe_score.is_finite()
            || !candidate.implied_timescale_relative_spread.is_finite()
            || candidate.implied_timescale_relative_spread < 0.0
            || !candidate.chapman_kolmogorov_max_abs_residual.is_finite()
            || candidate.chapman_kolmogorov_max_abs_residual < 0.0
            || (candidate.eligible
                && (!candidate.ineligible_reasons.is_empty()
                    || candidate.fold_seed.len() != ledger.loso_seed_count
                    || candidate.minimum_fold_validation_vampe_score
                        < ledger.minimum_required_fold_validation_vampe_score))
        {
            bail!("invalid LOSO train-fitted VAMP-E candidate row");
        }
        if candidate.state_count == ledger.selected_state_count
            && candidate.kinetic_dimensions == ledger.selected_kinetic_dimensions
            && candidate.lag_steps == ledger.selected_lag_steps
            && candidate.eligible
            && candidate.within_lag_winner
            && candidate.lag_diagnostics_pass
            && candidate.mean_validation_vampe_score.to_bits()
                == ledger.selected_mean_validation_vampe_score.to_bits()
        {
            selected_match += 1;
        }
    }
    let mut winner_count_by_lag = BTreeMap::<u64, usize>::new();
    let mut passing_winner_lags = BTreeSet::new();
    for candidate in &ledger.candidates {
        if candidate.within_lag_winner {
            *winner_count_by_lag.entry(candidate.lag_steps).or_default() += 1;
            if candidate.eligible && candidate.lag_diagnostics_pass {
                passing_winner_lags.insert(candidate.lag_steps);
            }
        }
    }
    if winner_count_by_lag.values().any(|count| *count != 1)
        || passing_winner_lags.iter().next().copied() != Some(ledger.selected_lag_steps)
        || selected_match != 1
        || kinetic_selection_ledger_hash(ledger)? != ledger.ledger_sha256
    {
        bail!("kinetic model-selection winner or content hash is invalid");
    }
    Ok(())
}

fn valid_probability_interval(mean: f64, lower: f64, upper: f64) -> bool {
    mean.is_finite()
        && lower.is_finite()
        && upper.is_finite()
        && (0.0..=1.0).contains(&mean)
        && (0.0..=1.0).contains(&lower)
        && (0.0..=1.0).contains(&upper)
        && lower <= mean
        && mean <= upper
}

fn require_normalized(sum: f64, tolerance: f64, field: &str) -> Result<()> {
    if (sum - 1.0).abs() > tolerance {
        bail!("{field} sum {sum} violates normalization tolerance {tolerance}");
    }
    Ok(())
}

/// End-to-end Ghost conservation counts supplied by the completed segment decoder.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GhostLatticeConservation {
    /// Records accepted by the engine producer.
    pub produced: u64,
    /// Records durably committed to segment files.
    pub persisted: u64,
    /// Records decoded with verified byte hashes.
    pub decoded: u64,
    /// Nodes admitted to the native phase lattice.
    pub lattice_nodes: u64,
    /// SHA-256 of the completed lattice manifest.
    pub lattice_manifest_sha256: String,
}

impl GhostLatticeConservation {
    /// Enforce exact conservation and manifest identity.
    pub fn validate(&self) -> Result<()> {
        require_sha256(&self.lattice_manifest_sha256, "lattice_manifest_sha256")?;
        // Zero is a conserved population, not an overflow or lineage failure.
        // Whether an empty population is scientifically admissible belongs to
        // the predeclared selection mode: whole-lattice discovery requires at
        // least one node, while fixed-cadence target confirmation must retain
        // a true no-detection interval as negative time-at-risk.
        if self.produced != self.persisted
            || self.persisted != self.decoded
            || self.decoded != self.lattice_nodes
        {
            bail!(
                "Ghost conservation failed: produced={} persisted={} decoded={} lattice_nodes={}",
                self.produced,
                self.persisted,
                self.decoded,
                self.lattice_nodes
            );
        }
        Ok(())
    }
}

/// Exact integration timebase authority supplied separately from request-v6
/// estimator configuration. The SHA names the signed engine/integrator
/// authority artifact; it must not be a digest invented from the scalar alone.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativePathBTimebaseContext {
    pub schema_kind: String,
    pub schema_version: String,
    pub context_sha256: String,
    pub integration_timestep_femtoseconds: f64,
    pub integration_timebase_sha256: String,
}

#[derive(Serialize)]
struct NativePathBTimebaseContextBody<'a> {
    integration_timestep_femtoseconds: f64,
    integration_timebase_sha256: &'a str,
}

impl NativePathBTimebaseContext {
    fn body(&self) -> NativePathBTimebaseContextBody<'_> {
        NativePathBTimebaseContextBody {
            integration_timestep_femtoseconds: self.integration_timestep_femtoseconds,
            integration_timebase_sha256: &self.integration_timebase_sha256,
        }
    }

    pub fn materialize(mut self) -> Result<Self> {
        if !self.context_sha256.is_empty() {
            bail!("timebase context materialization requires an empty digest");
        }
        self.context_sha256 = canonical_sha256(&self.body())?;
        self.verify()?;
        Ok(self)
    }

    pub fn verify(&self) -> Result<()> {
        if self.schema_kind != "prism_native_pathb_timebase_context"
            || self.schema_version != "1.0.0"
            || !self.integration_timestep_femtoseconds.is_finite()
            || self.integration_timestep_femtoseconds <= 0.0
        {
            bail!("invalid exact native Path-B integration timebase context");
        }
        require_sha256(&self.context_sha256, "timebase context_sha256")?;
        require_sha256(
            &self.integration_timebase_sha256,
            "integration_timebase_sha256",
        )?;
        if canonical_sha256(&self.body())? != self.context_sha256 {
            bail!("native Path-B timebase context content hash mismatch");
        }
        Ok(())
    }
}

/// Deterministic native Path-B estimator configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativePathBEstimatorConfig {
    /// Number of coordinate-cavity-defined metastable states selected only
    /// after neutral state estimation and uncertainty separation.
    pub open_state_count: usize,
    /// Predeclared state-count candidates for deterministic LOSO selection.
    pub candidate_state_counts: Vec<usize>,
    /// Predeclared exact lags for deterministic LOSO selection.
    pub candidate_lag_steps: Vec<u64>,
    /// Maximum retained reversible TICA dimensions.
    /// Predeclared reversible TICA representation dimensions. Representation
    /// selection is performed within each lag together with state count.
    pub candidate_kinetic_dimensions: Vec<usize>,
    /// Positive covariance diagonal regularizer.
    pub covariance_regularization: f64,
    /// Minimum train-fitted held-out LOSO VAMP-E score for any eligible
    /// representation candidate.
    pub minimum_loso_validation_vampe_score: f64,
    /// Maximum relative spread allowed for an implied-timescale plateau.
    pub maximum_implied_timescale_relative_spread: f64,
    /// Minimum independent seeds supporting every state.
    pub minimum_state_seed_support: usize,
    /// Minimum winner-to-runner-up LOSO score margin; otherwise sensitivity
    /// remains inconclusive.
    pub minimum_model_selection_margin: f64,
    /// Symmetric Dirichlet pseudocount used before reversible normalization
    /// strictly within each lane's adequately supported active subspace.
    pub reversible_pseudocount: f64,
    /// Maximum deterministic Lloyd iterations.
    pub maximum_iterations: usize,
    /// Temperature for soft state-membership probabilities.
    pub membership_temperature: f64,
    /// Row/stationary normalization tolerance.
    pub normalization_tolerance: f64,
    /// Minimum exact-lag transition pairs required.
    pub minimum_transition_pairs: usize,
    /// Required Ghost SO(3) plane-presence bit mask.
    pub required_so3_plane_mask: u8,
    /// Minimum independent replica seeds required for bootstrap uncertainty.
    pub minimum_independent_seeds: usize,
    /// Deterministic seed/block bootstrap replicate count.
    pub bootstrap_replicates: usize,
    /// Maximum non-wrapping transition block length inside one
    /// run/stream/evaluable segment of each seed.
    pub bootstrap_block_length: usize,
    /// Minimum hard assignment count in every state.
    pub minimum_state_effective_count: u64,
    /// Maximum accepted Chapman-Kolmogorov residual.
    pub maximum_chapman_kolmogorov_residual: f64,
    /// Maximum accepted detailed-balance residual.
    pub maximum_detailed_balance_residual: f64,
}

impl Default for NativePathBEstimatorConfig {
    fn default() -> Self {
        Self {
            open_state_count: 1,
            candidate_state_counts: vec![2, 3, 4],
            candidate_lag_steps: vec![25, 50, 100],
            candidate_kinetic_dimensions: vec![2, 3],
            covariance_regularization: 1.0e-8,
            minimum_loso_validation_vampe_score: 0.0,
            maximum_implied_timescale_relative_spread: 0.35,
            minimum_state_seed_support: 3,
            minimum_model_selection_margin: 1.0e-3,
            reversible_pseudocount: 0.5,
            maximum_iterations: 100,
            membership_temperature: 1.0,
            normalization_tolerance: 1.0e-9,
            minimum_transition_pairs: 10,
            required_so3_plane_mask: 0b0001,
            minimum_independent_seeds: 5,
            bootstrap_replicates: 200,
            bootstrap_block_length: 4,
            minimum_state_effective_count: 10,
            maximum_chapman_kolmogorov_residual: 0.20,
            maximum_detailed_balance_residual: 1.0e-10,
        }
    }
}

impl NativePathBEstimatorConfig {
    fn validate(&self, node_count: usize) -> Result<()> {
        if self.candidate_state_counts.is_empty()
            || self
                .candidate_state_counts
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || self
                .candidate_state_counts
                .iter()
                .any(|count| *count < 2 || *count > node_count)
            || self.candidate_lag_steps.is_empty()
            || self
                .candidate_lag_steps
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || self.candidate_kinetic_dimensions.is_empty()
            || self
                .candidate_kinetic_dimensions
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || self
                .candidate_kinetic_dimensions
                .iter()
                .any(|dimensions| *dimensions == 0)
            || !self.covariance_regularization.is_finite()
            || self.covariance_regularization <= 0.0
            || !self.minimum_loso_validation_vampe_score.is_finite()
            || !self.maximum_implied_timescale_relative_spread.is_finite()
            || self.maximum_implied_timescale_relative_spread < 0.0
            || self.minimum_state_seed_support < 2
            || !self.minimum_model_selection_margin.is_finite()
            || self.minimum_model_selection_margin < 0.0
        {
            bail!("invalid predeclared kinetic state/lag model-selection grid");
        }
        if self.open_state_count == 0 || self.open_state_count >= self.candidate_state_counts[0] {
            bail!("open_state_count must be smaller than every candidate state count");
        }
        if self.maximum_iterations == 0 {
            bail!("Path-B iteration count must be non-zero");
        }
        if !self.reversible_pseudocount.is_finite() || self.reversible_pseudocount <= 0.0 {
            bail!("reversible pseudocount must be finite and positive");
        }
        if !self.membership_temperature.is_finite() || self.membership_temperature <= 0.0 {
            bail!("membership temperature must be finite and positive");
        }
        if !self.normalization_tolerance.is_finite()
            || self.normalization_tolerance <= 0.0
            || self.normalization_tolerance > 0.01
        {
            bail!("normalization tolerance must be finite and in (0, 0.01]");
        }
        if self.minimum_transition_pairs == 0 || self.required_so3_plane_mask == 0 {
            bail!("transition-pair and SO(3)-plane gates must be non-zero");
        }
        if self.minimum_independent_seeds < 5
            || self.bootstrap_replicates < 100
            || self.bootstrap_block_length == 0
            || self.minimum_state_effective_count == 0
        {
            bail!("native Path-B seed/bootstrap/state-count diagnostic gates are too weak");
        }
        if !self.maximum_chapman_kolmogorov_residual.is_finite()
            || !(0.0..=1.0).contains(&self.maximum_chapman_kolmogorov_residual)
            || !self.maximum_detailed_balance_residual.is_finite()
            || self.maximum_detailed_balance_residual < 0.0
        {
            bail!("native Path-B Markov residual thresholds are invalid");
        }
        Ok(())
    }
}

/// Machine-readable receipt for native Path-B estimation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativePathBMaterializationReceipt {
    /// Ghost records produced.
    pub produced: u64,
    /// Ghost records persisted.
    pub persisted: u64,
    /// Ghost records decoded.
    pub decoded: u64,
    /// Native lattice nodes.
    pub lattice_nodes: u64,
    /// Exact byte-lineage records consumed.
    pub byte_lineages: u64,
    /// Native Path-B assignments emitted.
    pub pathb_assignments: u64,
    /// Structural assignments spanning every evaluable scheduled surveillance
    /// frame, including typed no-Ghost rows.
    pub scheduled_state_assignments: u64,
    /// Ghost event rows with complete physical measurements and therefore an
    /// admissible state assignment.
    pub kinetically_evaluable_ghost_events: u64,
    /// Ghost event rows preserved without a state assignment because their
    /// scheduled physical measurement was NOT_EVALUABLE.
    pub not_evaluable_ghost_events: u64,
    /// Exact-lag transition pairs used.
    pub transition_pairs: u64,
    /// Exact accepted edge rows admitted before lag filtering.
    pub accepted_edge_rows: u64,
    /// Pooled exact-connectivity authority digest.
    pub exact_connectivity_population_sha256: String,
    /// Exact accepted-edge population digest.
    pub exact_edge_population_sha256: String,
    /// Metastable states estimated.
    pub metastable_states: u32,
    /// Native model SHA-256.
    pub model_sha256: String,
    /// Lattice manifest SHA-256.
    pub lattice_manifest_sha256: String,
    /// Atom population observed verbatim from the exact coordinate-frame
    /// bindings used for structural inference. `None` is retained only for
    /// pre-coordinate estimation; production evidence emission binds a
    /// non-zero value before serializing the receipt.
    pub observed_atom_count: Option<u64>,
    /// Exact full-system populations by physical lane and source topology.
    /// Empty only for legacy receipts or pre-coordinate estimation. The legacy
    /// scalar is Some only when every recorded topology has the same count.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub observed_atom_counts_by_lane_topology: LaneTopologyAtomCounts,
    /// True only when every population count is exactly equal.
    pub conservation_pass: bool,
    /// Explicit source contract excluding legacy JSON.
    pub source_contract: String,
    /// Markov consistency/uncertainty gate status.
    pub markov_diagnostics_status: MarkovDiagnosticStatus,
}

/// Full coordinate atom populations keyed by the actual physical lane and topology.
/// Different fragment/solvent topologies must not be padded to a common atom count.
pub type LaneTopologyAtomCounts = BTreeMap<ControlLane, BTreeMap<String, u64>>;

pub fn collect_lane_topology_atom_counts(
    sources: impl IntoIterator<Item = (ControlLane, String, u64)>,
) -> Result<LaneTopologyAtomCounts> {
    let mut counts: LaneTopologyAtomCounts = BTreeMap::new();
    let mut topology_counts = BTreeMap::new();
    for (lane, topology_sha256, atom_count) in sources {
        require_sha256(&topology_sha256, "observed coordinate topology_sha256")?;
        if atom_count == 0 {
            bail!("EVIDENCE_NOT_FOUND: exact coordinate atom_count is zero");
        }
        if topology_counts.insert(topology_sha256.clone(), atom_count)
            .is_some_and(|previous| previous != atom_count) {
            bail!("exact coordinate atom_count changed within one topology identity");
        }
        counts.entry(lane).or_default().insert(topology_sha256, atom_count);
    }
    if counts.is_empty() {
        bail!("EVIDENCE_NOT_FOUND: exact coordinate atom_count population is empty");
    }
    Ok(counts)
}

/// Native Path-B model plus its conservation receipt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NativePathBMaterialization {
    /// Estimated model.
    pub model: NativePathBModel,
    /// Conservation and estimator receipt.
    pub receipt: NativePathBMaterializationReceipt,
}

/// Serializable, hash-bound projection of one native Ghost lattice node onto
/// the exact feature/identity fields consumed by Path-B. It allows immutable
/// per-run lane evidence to be pooled later without reading legacy binding-
/// site JSON or inventing any node value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NativePathBLatticeObservation {
    pub schema_kind: String,
    pub schema_version: String,
    pub observation_sha256: String,
    /// Canonical hash of the complete Ghost occurrence. Equal record bytes in
    /// distinct runs or segments must never collapse to one Path-B identity.
    pub ghost_lineage_sha256: String,
    pub ghost_record_sha256: String,
    pub lattice_node_index: u64,
    pub stream_id: u16,
    pub site_id: u32,
    pub frame_idx: u64,
    pub step_idx: u64,
    pub protocol_phase: u8,
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    pub centroid_xyz: [f32; 3],
    pub kl_divergence: f32,
    pub thermo_flux: [f32; 2],
    pub so3_power_spectrum: [[f32; 6]; 4],
    pub so3_plane_status: u8,
}

/// One accepted native lattice edge rewritten onto global Ghost lineage
/// hashes. The device endpoint indices remain as audit fields, but downstream
/// joins never use index identity across independent runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NativePathBExactEdgeObservation {
    pub edge_sha256: String,
    pub source_edge_artifact_sha256: String,
    pub from_ghost_lineage_sha256: String,
    pub to_ghost_lineage_sha256: String,
    pub from_node_index: u64,
    pub to_node_index: u64,
    pub from_step_idx: u64,
    pub to_step_idx: u64,
    pub temporal_delta_steps: u64,
    pub total_score: f32,
    pub so3_score: f32,
    pub causal_score: f32,
    pub thermo_score: f32,
    pub from_protocol_phase: u8,
    pub to_protocol_phase: u8,
    pub common_so3_plane_mask: u8,
    pub predicate_flags: u8,
}

#[derive(Serialize)]
struct NativePathBExactEdgeBody<'a> {
    source_edge_artifact_sha256: &'a str,
    from_ghost_lineage_sha256: &'a str,
    to_ghost_lineage_sha256: &'a str,
    from_node_index: u64,
    to_node_index: u64,
    from_step_idx: u64,
    to_step_idx: u64,
    temporal_delta_steps: u64,
    total_score: f32,
    so3_score: f32,
    causal_score: f32,
    thermo_score: f32,
    from_protocol_phase: u8,
    to_protocol_phase: u8,
    common_so3_plane_mask: u8,
    predicate_flags: u8,
}

impl NativePathBExactEdgeObservation {
    fn body(&self) -> NativePathBExactEdgeBody<'_> {
        NativePathBExactEdgeBody {
            source_edge_artifact_sha256: &self.source_edge_artifact_sha256,
            from_ghost_lineage_sha256: &self.from_ghost_lineage_sha256,
            to_ghost_lineage_sha256: &self.to_ghost_lineage_sha256,
            from_node_index: self.from_node_index,
            to_node_index: self.to_node_index,
            from_step_idx: self.from_step_idx,
            to_step_idx: self.to_step_idx,
            temporal_delta_steps: self.temporal_delta_steps,
            total_score: self.total_score,
            so3_score: self.so3_score,
            causal_score: self.causal_score,
            thermo_score: self.thermo_score,
            from_protocol_phase: self.from_protocol_phase,
            to_protocol_phase: self.to_protocol_phase,
            common_so3_plane_mask: self.common_so3_plane_mask,
            predicate_flags: self.predicate_flags,
        }
    }

    pub fn verify(&self) -> Result<()> {
        for (digest, field) in [
            (&self.edge_sha256, "edge_sha256"),
            (
                &self.source_edge_artifact_sha256,
                "source_edge_artifact_sha256",
            ),
            (&self.from_ghost_lineage_sha256, "from_ghost_lineage_sha256"),
            (&self.to_ghost_lineage_sha256, "to_ghost_lineage_sha256"),
        ] {
            require_sha256(digest, field)?;
        }
        let product = self.so3_score * self.causal_score * self.thermo_score;
        if self.from_ghost_lineage_sha256 == self.to_ghost_lineage_sha256
            || self.from_node_index == self.to_node_index
            || self.to_step_idx < self.from_step_idx
            || self.temporal_delta_steps != self.to_step_idx - self.from_step_idx
            || self.predicate_flags != 0x0f
            || [
                self.total_score,
                self.so3_score,
                self.causal_score,
                self.thermo_score,
            ]
            .iter()
            .any(|value| !value.is_finite())
            || (self.total_score - product).abs() > 1.0e-5 * self.total_score.abs().max(1.0)
            || canonical_sha256(&self.body())? != self.edge_sha256
        {
            bail!("invalid exact native Path-B accepted-edge observation");
        }
        Ok(())
    }
}

/// Exact connected component reconstructed from accepted edge rows. Singleton
/// nodes are retained with an empty edge population.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativePathBExactComponent {
    pub component_sha256: String,
    pub node_ghost_lineage_sha256: Vec<String>,
    pub edge_sha256: Vec<String>,
}

/// Immutable exact-connectivity authority for one completed lane/run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NativePathBConnectivityAuthority {
    pub schema_kind: String,
    pub schema_version: String,
    pub connectivity_sha256: String,
    pub source_edge_artifact_sha256: String,
    pub source_node_lineage_artifact_sha256: String,
    pub source_edge_payload_sha256: String,
    pub source_produced_edges: u64,
    pub selected_node_count: u64,
    pub selected_edge_count: u64,
    pub edges: Vec<NativePathBExactEdgeObservation>,
    pub components: Vec<NativePathBExactComponent>,
}

#[derive(Serialize)]
struct NativePathBConnectivityBody<'a> {
    source_edge_artifact_sha256: &'a str,
    source_node_lineage_artifact_sha256: &'a str,
    source_edge_payload_sha256: &'a str,
    source_produced_edges: u64,
    selected_node_count: u64,
    selected_edge_count: u64,
    edges: &'a [NativePathBExactEdgeObservation],
    components: &'a [NativePathBExactComponent],
}

impl NativePathBConnectivityAuthority {
    fn body(&self) -> NativePathBConnectivityBody<'_> {
        NativePathBConnectivityBody {
            source_edge_artifact_sha256: &self.source_edge_artifact_sha256,
            source_node_lineage_artifact_sha256: &self.source_node_lineage_artifact_sha256,
            source_edge_payload_sha256: &self.source_edge_payload_sha256,
            source_produced_edges: self.source_produced_edges,
            selected_node_count: self.selected_node_count,
            selected_edge_count: self.selected_edge_count,
            edges: &self.edges,
            components: &self.components,
        }
    }

    pub fn verify_for(&self, lineages: &[GhostByteLineage]) -> Result<()> {
        if self.schema_kind != "prism_native_pathb_exact_connectivity"
            || self.schema_version != "1.0.0"
            || self.selected_node_count != lineages.len() as u64
            || self.selected_edge_count != self.edges.len() as u64
        {
            bail!("invalid native Path-B exact-connectivity authority");
        }
        for (digest, field) in [
            (&self.connectivity_sha256, "connectivity_sha256"),
            (
                &self.source_edge_artifact_sha256,
                "source_edge_artifact_sha256",
            ),
            (
                &self.source_node_lineage_artifact_sha256,
                "source_node_lineage_artifact_sha256",
            ),
            (
                &self.source_edge_payload_sha256,
                "source_edge_payload_sha256",
            ),
        ] {
            require_sha256(digest, field)?;
        }
        let nodes = lineages
            .iter()
            .map(GhostByteLineage::lineage_sha256)
            .collect::<Result<BTreeSet<_>>>()?;
        let mut edge_hashes = BTreeSet::new();
        for edge in &self.edges {
            edge.verify()?;
            if edge.source_edge_artifact_sha256 != self.source_edge_artifact_sha256
                || !nodes.contains(&edge.from_ghost_lineage_sha256)
                || !nodes.contains(&edge.to_ghost_lineage_sha256)
                || !edge_hashes.insert(edge.edge_sha256.clone())
            {
                bail!("exact edge population is duplicate or escapes admitted node lineage");
            }
        }
        let mut component_nodes = BTreeSet::new();
        let mut component_edges = BTreeSet::new();
        let mut previous_component = None;
        for component in &self.components {
            require_sha256(&component.component_sha256, "component_sha256")?;
            if component.node_ghost_lineage_sha256.is_empty()
                || component
                    .node_ghost_lineage_sha256
                    .windows(2)
                    .any(|pair| pair[0] >= pair[1])
                || component
                    .edge_sha256
                    .windows(2)
                    .any(|pair| pair[0] >= pair[1])
                || previous_component
                    .as_ref()
                    .is_some_and(|prior| prior >= &component.component_sha256)
                || canonical_sha256(&(
                    self.source_edge_artifact_sha256.as_str(),
                    &component.node_ghost_lineage_sha256,
                    &component.edge_sha256,
                ))? != component.component_sha256
            {
                bail!("invalid canonical exact lattice component");
            }
            previous_component = Some(component.component_sha256.clone());
            for node in &component.node_ghost_lineage_sha256 {
                if !nodes.contains(node) || !component_nodes.insert(node.clone()) {
                    bail!("exact component node population is incomplete or duplicated");
                }
            }
            for edge in &component.edge_sha256 {
                if !edge_hashes.contains(edge) || !component_edges.insert(edge.clone()) {
                    bail!("exact component edge population is incomplete or duplicated");
                }
            }
        }
        if component_nodes != nodes
            || component_edges != edge_hashes
            || canonical_sha256(&self.body())? != self.connectivity_sha256
        {
            bail!("exact connectivity conservation/hash verification failed");
        }
        Ok(())
    }
}

/// Project a verified full-lattice accepted-edge artifact onto an admitted
/// node subset and reconstruct every connected component from those exact
/// rows. No geometric re-clustering is permitted at this boundary.
#[cfg(any(feature = "gpu", feature = "cpu-analysis"))]
pub fn build_exact_pathb_connectivity(
    full_nodes: &[crate::ghost_phase_lattice::GhostPhaseLatticeNode],
    selected_lineages: &[GhostByteLineage],
    artifact: &crate::ghost_phase_lattice::GhostLatticeAcceptedEdgeArtifact,
) -> Result<NativePathBConnectivityAuthority> {
    artifact.verify(full_nodes)?;
    let source_edge_artifact_sha256 = canonical_sha256(artifact)?;
    let selected = selected_lineages
        .iter()
        .map(|lineage| Ok((lineage.lattice_node_index, lineage.lineage_sha256()?)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    if selected.len() != selected_lineages.len()
        || selected
            .keys()
            .any(|index| *index >= full_nodes.len() as u64)
    {
        bail!("selected Ghost lineage indices are duplicate or outside full lattice");
    }
    let mut edges = artifact
        .edges
        .iter()
        .filter_map(|edge| {
            let from = selected.get(&u64::from(edge.from_node_index))?;
            let to = selected.get(&u64::from(edge.to_node_index))?;
            Some((edge, from, to))
        })
        .map(|(edge, from, to)| {
            let mut projected = NativePathBExactEdgeObservation {
                edge_sha256: String::new(),
                source_edge_artifact_sha256: source_edge_artifact_sha256.clone(),
                from_ghost_lineage_sha256: from.clone(),
                to_ghost_lineage_sha256: to.clone(),
                from_node_index: u64::from(edge.from_node_index),
                to_node_index: u64::from(edge.to_node_index),
                from_step_idx: edge.from_step_idx,
                to_step_idx: edge.to_step_idx,
                temporal_delta_steps: edge.temporal_delta_steps,
                total_score: edge.total_score,
                so3_score: edge.so3_score,
                causal_score: edge.causal_score,
                thermo_score: edge.thermo_score,
                from_protocol_phase: edge.from_protocol_phase,
                to_protocol_phase: edge.to_protocol_phase,
                common_so3_plane_mask: edge.common_so3_plane_mask,
                predicate_flags: edge.predicate_flags,
            };
            projected.edge_sha256 = canonical_sha256(&projected.body())?;
            projected.verify()?;
            Ok(projected)
        })
        .collect::<Result<Vec<_>>>()?;
    edges.sort_by(|left, right| {
        left.from_ghost_lineage_sha256
            .cmp(&right.from_ghost_lineage_sha256)
            .then_with(|| {
                left.to_ghost_lineage_sha256
                    .cmp(&right.to_ghost_lineage_sha256)
            })
            .then_with(|| left.edge_sha256.cmp(&right.edge_sha256))
    });

    let mut node_hashes = selected.values().cloned().collect::<Vec<_>>();
    node_hashes.sort();
    let node_position = node_hashes
        .iter()
        .enumerate()
        .map(|(index, hash)| (hash.clone(), index))
        .collect::<BTreeMap<_, _>>();
    let mut parent = (0..node_hashes.len()).collect::<Vec<_>>();
    fn root(parent: &mut [usize], mut index: usize) -> usize {
        while parent[index] != index {
            parent[index] = parent[parent[index]];
            index = parent[index];
        }
        index
    }
    for edge in &edges {
        let left = node_position[&edge.from_ghost_lineage_sha256];
        let right = node_position[&edge.to_ghost_lineage_sha256];
        let left_root = root(&mut parent, left);
        let right_root = root(&mut parent, right);
        if left_root != right_root {
            let (minimum, maximum) = if left_root < right_root {
                (left_root, right_root)
            } else {
                (right_root, left_root)
            };
            parent[maximum] = minimum;
        }
    }
    let mut nodes_by_root: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    for (index, hash) in node_hashes.into_iter().enumerate() {
        let component_root = root(&mut parent, index);
        nodes_by_root.entry(component_root).or_default().push(hash);
    }
    // Partition accepted rows once by the SAME completed union-find roots.
    // Preserve every row (including duplicates, rejected by verify_for below),
    // original hash bytes, singleton nodes and existing canonical sort/hash law.
    // Scanning all edges once per component is prohibitive for full native runs.
    let mut edges_by_root: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    for edge in &edges {
        let component_root = root(&mut parent, node_position[&edge.from_ghost_lineage_sha256]);
        let target_root = root(&mut parent, node_position[&edge.to_ghost_lineage_sha256]);
        if component_root != target_root {
            bail!("accepted edge escaped its completed native union-find component");
        }
        edges_by_root.entry(component_root).or_default().push(edge.edge_sha256.clone());
    }
    let mut components = Vec::with_capacity(nodes_by_root.len());
    for (component_root, mut component_nodes) in nodes_by_root {
        component_nodes.sort();
        let mut component_edges = edges_by_root.remove(&component_root).unwrap_or_default();
        component_edges.sort();
        let component_sha256 = canonical_sha256(&(
            source_edge_artifact_sha256.as_str(),
            &component_nodes,
            &component_edges,
        ))?;
        components.push(NativePathBExactComponent {
            component_sha256,
            node_ghost_lineage_sha256: component_nodes,
            edge_sha256: component_edges,
        });
    }
    if !edges_by_root.is_empty() {
        bail!("accepted edge partition contains no corresponding native node component");
    }
    components.sort_by(|left, right| left.component_sha256.cmp(&right.component_sha256));
    let mut authority = NativePathBConnectivityAuthority {
        schema_kind: "prism_native_pathb_exact_connectivity".to_string(),
        schema_version: "1.0.0".to_string(),
        connectivity_sha256: String::new(),
        source_edge_artifact_sha256,
        source_node_lineage_artifact_sha256: artifact.node_lineage_artifact_sha256.clone(),
        source_edge_payload_sha256: artifact.canonical_payload_sha256.clone(),
        source_produced_edges: artifact.produced_edges,
        selected_node_count: selected.len() as u64,
        selected_edge_count: edges.len() as u64,
        edges,
        components,
    };
    authority.connectivity_sha256 = canonical_sha256(&authority.body())?;
    authority.verify_for(selected_lineages)?;
    Ok(authority)
}

#[derive(Serialize)]
struct NativePathBLatticeObservationBody<'a> {
    ghost_lineage_sha256: &'a str,
    ghost_record_sha256: &'a str,
    lattice_node_index: u64,
    stream_id: u16,
    site_id: u32,
    frame_idx: u64,
    step_idx: u64,
    protocol_phase: u8,
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
    centroid_xyz: [f32; 3],
    kl_divergence: f32,
    thermo_flux: [f32; 2],
    so3_power_spectrum: [[f32; 6]; 4],
    so3_plane_status: u8,
}

impl NativePathBLatticeObservation {
    /// Project one authenticated native node without changing its identity or
    /// numeric feature representation.
    #[cfg(any(feature = "gpu", feature = "cpu-analysis"))]
    pub fn from_native_node(
        node: &crate::ghost_phase_lattice::GhostPhaseLatticeNode,
        lineage: &GhostByteLineage,
    ) -> Result<Self> {
        lineage.validate()?;
        if node.stream_id != lineage.key.stream_id
            || node.site_id != lineage.key.site_id
            || node.frame_idx != lineage.key.frame_idx
            || node.step_idx != lineage.key.step_idx
        {
            bail!("Path-B lattice observation source node disagrees with Ghost lineage");
        }
        let mut observation = Self {
            schema_kind: "prism_native_pathb_lattice_observation".to_string(),
            schema_version: "2.0.0".to_string(),
            observation_sha256: String::new(),
            ghost_lineage_sha256: lineage.lineage_sha256()?,
            ghost_record_sha256: lineage.record_sha256.clone(),
            lattice_node_index: lineage.lattice_node_index,
            stream_id: node.stream_id,
            site_id: node.site_id,
            frame_idx: node.frame_idx,
            step_idx: node.step_idx,
            protocol_phase: node.protocol_phase,
            aabb_min: node.aabb_min,
            aabb_max: node.aabb_max,
            centroid_xyz: node.centroid_xyz,
            kl_divergence: node.kl_divergence,
            thermo_flux: node.thermo_flux,
            so3_power_spectrum: node.so3_power_spectrum,
            so3_plane_status: node.so3_plane_status,
        };
        observation.observation_sha256 = canonical_sha256(&observation.body())?;
        observation.verify_for(lineage)?;
        Ok(observation)
    }

    fn body(&self) -> NativePathBLatticeObservationBody<'_> {
        NativePathBLatticeObservationBody {
            ghost_lineage_sha256: &self.ghost_lineage_sha256,
            ghost_record_sha256: &self.ghost_record_sha256,
            lattice_node_index: self.lattice_node_index,
            stream_id: self.stream_id,
            site_id: self.site_id,
            frame_idx: self.frame_idx,
            step_idx: self.step_idx,
            protocol_phase: self.protocol_phase,
            aabb_min: self.aabb_min,
            aabb_max: self.aabb_max,
            centroid_xyz: self.centroid_xyz,
            kl_divergence: self.kl_divergence,
            thermo_flux: self.thermo_flux,
            so3_power_spectrum: self.so3_power_spectrum,
            so3_plane_status: self.so3_plane_status,
        }
    }

    /// Verify content identity and exact Ghost identity binding.
    pub fn verify_for(&self, lineage: &GhostByteLineage) -> Result<()> {
        if self.schema_kind != "prism_native_pathb_lattice_observation"
            || self.schema_version != "2.0.0"
            || self.ghost_lineage_sha256 != lineage.lineage_sha256()?
            || self.ghost_record_sha256 != lineage.record_sha256
            || self.lattice_node_index != lineage.lattice_node_index
            || self.stream_id != lineage.key.stream_id
            || self.site_id != lineage.key.site_id
            || self.frame_idx != lineage.key.frame_idx
            || self.step_idx != lineage.key.step_idx
            || self.protocol_phase != u8::try_from(lineage.protocol_phase).unwrap_or(u8::MAX)
            || self.protocol_phase > 3
            || u32::try_from(self.lattice_node_index).is_err()
            || self
                .aabb_min
                .iter()
                .chain(&self.aabb_max)
                .chain(&self.centroid_xyz)
                .any(|value| !value.is_finite())
            || self
                .aabb_min
                .iter()
                .zip(&self.aabb_max)
                // Native spatial support uses closed AABBs: an axis
                // may have zero extent. Recording the exact native support
                // does not assert positive cavity volume. The existing
                // lattice_feature_vector converter retains its volume > 0 gate.
                .any(|(min, max)| min > max)
            || !self.kl_divergence.is_finite()
            || self.thermo_flux.iter().any(|value| !value.is_finite())
            || self
                .so3_power_spectrum
                .iter()
                .flatten()
                .any(|value| !value.is_finite())
        {
            bail!("invalid hash-bound native Path-B lattice observation");
        }
        require_sha256(&self.observation_sha256, "observation_sha256")?;
        require_sha256(&self.ghost_lineage_sha256, "ghost_lineage_sha256")?;
        if canonical_sha256(&self.body())? != self.observation_sha256 {
            bail!("native Path-B lattice observation content hash mismatch");
        }
        Ok(())
    }

    #[cfg(any(feature = "gpu", feature = "cpu-analysis"))]
    fn to_native_node(&self) -> crate::ghost_phase_lattice::GhostPhaseLatticeNode {
        let mut node = crate::ghost_phase_lattice::GhostPhaseLatticeNode::empty();
        node.tile_index = self.lattice_node_index as u32;
        node.stream_id = self.stream_id;
        node.site_id = self.site_id;
        node.frame_idx = self.frame_idx;
        node.step_idx = self.step_idx;
        node.protocol_phase = self.protocol_phase;
        node.aabb_min = self.aabb_min;
        node.aabb_max = self.aabb_max;
        node.centroid_xyz = self.centroid_xyz;
        node.kl_divergence = self.kl_divergence;
        node.thermo_flux = self.thermo_flux;
        node.so3_power_spectrum = self.so3_power_spectrum;
        node.so3_plane_status = self.so3_plane_status;
        node
    }
}

/// Re-estimate native Path-B from immutable per-run lattice observations.
/// Every observation must remain one-to-one with its original Ghost byte
/// lineage; the conversion restores only the explicit native node fields.
#[cfg(any(feature = "gpu", feature = "cpu-analysis"))]
pub fn estimate_native_pathb_from_observations(
    _observations: &[NativePathBLatticeObservation],
    _lineages: &[GhostByteLineage],
    _connectivities: &[NativePathBConnectivityAuthority],
    _surveillance_ledgers: &[crate::pathb_evidence_emitter::PathBConfirmationSurveillanceLedger],
    _conservation: &GhostLatticeConservation,
    _config: &NativePathBEstimatorConfig,
) -> Result<NativePathBMaterialization> {
    bail!("EVIDENCE_NOT_FOUND: exact native Path-B integration timebase context is required")
}

/// Timebase-bound variant of [`estimate_native_pathb_from_observations`].
#[cfg(any(feature = "gpu", feature = "cpu-analysis"))]
pub fn estimate_native_pathb_from_observations_with_timebase(
    observations: &[NativePathBLatticeObservation],
    lineages: &[GhostByteLineage],
    connectivities: &[NativePathBConnectivityAuthority],
    surveillance_ledgers: &[crate::pathb_evidence_emitter::PathBConfirmationSurveillanceLedger],
    conservation: &GhostLatticeConservation,
    config: &NativePathBEstimatorConfig,
    timebase_context: &NativePathBTimebaseContext,
) -> Result<NativePathBMaterialization> {
    if observations.len() != lineages.len() {
        bail!("Path-B lattice observation and Ghost lineage populations differ");
    }
    let nodes = observations
        .iter()
        .zip(lineages)
        .map(|(observation, lineage)| {
            observation.verify_for(lineage)?;
            Ok(observation.to_native_node())
        })
        .collect::<Result<Vec<_>>>()?;
    let lineages_by_hash = lineages
        .iter()
        .map(|lineage| Ok((lineage.lineage_sha256()?, lineage.clone())))
        .collect::<Result<BTreeMap<_, _>>>()?;
    let mut connectivity_hashes = Vec::with_capacity(connectivities.len());
    let mut admitted_nodes = BTreeSet::new();
    let mut exact_edges = Vec::new();
    for connectivity in connectivities {
        let component_nodes = connectivity
            .components
            .iter()
            .flat_map(|component| component.node_ghost_lineage_sha256.iter())
            .map(|hash| {
                lineages_by_hash.get(hash).cloned().with_context(|| {
                    format!("connectivity node {hash} is absent from pooled lineage")
                })
            })
            .collect::<Result<Vec<_>>>()?;
        connectivity.verify_for(&component_nodes)?;
        for lineage in &component_nodes {
            let hash = lineage.lineage_sha256()?;
            if !admitted_nodes.insert(hash) {
                bail!("pooled exact-connectivity authorities overlap on a Ghost occurrence");
            }
        }
        connectivity_hashes.push(connectivity.connectivity_sha256.clone());
        exact_edges.extend(connectivity.edges.clone());
    }
    if admitted_nodes != lineages_by_hash.keys().cloned().collect::<BTreeSet<_>>() {
        bail!("pooled exact-connectivity authorities do not cover every admitted Ghost node");
    }
    connectivity_hashes.sort();
    exact_edges.sort_by(|left, right| left.edge_sha256.cmp(&right.edge_sha256));
    if exact_edges
        .windows(2)
        .any(|pair| pair[0].edge_sha256 == pair[1].edge_sha256)
    {
        bail!("pooled exact-connectivity authorities contain duplicate accepted edges");
    }
    let connectivity_population_sha256 = canonical_sha256(&connectivity_hashes)?;
    estimate_native_pathb_from_lattice_with_timebase(
        &nodes,
        lineages,
        &exact_edges,
        &connectivity_population_sha256,
        surveillance_ledgers,
        conservation,
        config,
        timebase_context,
    )
}

impl NativePathBMaterialization {
    /// Bind the receipt to the atom population carried by exact coordinate
    /// frames. The value is never inferred from a topology or reconciled
    /// against a competing count.
    pub fn bind_observed_atom_count(&mut self, observed_atom_count: u64) -> Result<()> {
        if observed_atom_count == 0 {
            bail!("EVIDENCE_NOT_FOUND: exact coordinate atom_count is zero");
        }
        if self
            .receipt
            .observed_atom_count
            .is_some_and(|existing| existing != observed_atom_count)
        {
            bail!("exact coordinate atom_count changed after Path-B receipt binding");
        }
        self.receipt.observed_atom_count = Some(observed_atom_count);
        self.verify()
    }

    /// Bind all exact source populations without flattening lane/topology identity.
    pub fn bind_observed_atom_counts_by_lane_topology(
        &mut self,
        counts: LaneTopologyAtomCounts,
    ) -> Result<()> {
        let validated = collect_lane_topology_atom_counts(counts.iter().flat_map(|(lane, topologies)|
            topologies.iter().map(move |(topology, count)| (*lane, topology.clone(), *count))))?;
        if validated != counts {
            bail!("atom-count binding contains an empty lane population");
        }
        let unique = counts.values().flat_map(|v| v.values().copied()).collect::<BTreeSet<_>>();
        let scalar = if unique.len() == 1 { unique.first().copied() } else { None };
        if (!self.receipt.observed_atom_counts_by_lane_topology.is_empty()
            && self.receipt.observed_atom_counts_by_lane_topology != counts)
            || self.receipt.observed_atom_count.is_some_and(|existing| scalar != Some(existing)) {
            bail!("exact coordinate atom population changed after Path-B receipt binding");
        }
        self.receipt.observed_atom_counts_by_lane_topology = counts;
        self.receipt.observed_atom_count = scalar;
        self.verify()
    }

    /// Verify model identity and population conservation.
    pub fn verify(&self) -> Result<()> {
        self.model.verify()?;
        if !self.receipt.observed_atom_counts_by_lane_topology.is_empty() {
            let counts = &self.receipt.observed_atom_counts_by_lane_topology;
            let validated = collect_lane_topology_atom_counts(counts.iter().flat_map(|(lane, topologies)|
                topologies.iter().map(move |(topology, count)| (*lane, topology.clone(), *count))))?;
            let unique = counts.values().flat_map(|v| v.values().copied()).collect::<BTreeSet<_>>();
            let scalar = if unique.len() == 1 { unique.first().copied() } else { None };
            if &validated != counts || self.receipt.observed_atom_count != scalar {
                bail!("Path-B atom-count scalar/map disagree or contain empty lane populations");
            }
        }
        require_sha256(&self.receipt.model_sha256, "model_sha256")?;
        if self.receipt.model_sha256 != self.model.model_sha256
            || self.receipt.lattice_manifest_sha256
                != self.model.body.source_lattice_manifest_sha256
            || self.receipt.observed_atom_count == Some(0)
            || !self.receipt.conservation_pass
            || self.receipt.produced != self.receipt.persisted
            || self.receipt.persisted != self.receipt.decoded
            || self.receipt.decoded != self.receipt.lattice_nodes
            || self.receipt.lattice_nodes != self.receipt.byte_lineages
            || self.receipt.pathb_assignments != self.receipt.kinetically_evaluable_ghost_events
            || self.receipt.scheduled_state_assignments
                != self.model.body.scheduled_state_assignments.len() as u64
            || self.receipt.scheduled_state_assignments
                != self
                    .model
                    .body
                    .diagnostics
                    .surveillance_missingness
                    .evaluable_rows
            || self.receipt.kinetically_evaluable_ghost_events
                + self.receipt.not_evaluable_ghost_events
                != self.receipt.byte_lineages
            || self.receipt.accepted_edge_rows == 0
            || self.receipt.exact_connectivity_population_sha256
                != self.model.body.source_exact_connectivity_population_sha256
            || self.receipt.exact_edge_population_sha256
                != self.model.body.source_exact_edge_population_sha256
            || self.receipt.source_contract
                != "native_ghost_segments_to_exact_accepted_edges_to_phase_lattice_to_pathb"
            || self.receipt.markov_diagnostics_status != self.model.body.diagnostics.status
        {
            bail!("native Path-B materialization receipt failed conservation");
        }
        Ok(())
    }
}

/// Native Path-B and fiber bundle with end-to-end population conservation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NativePathBFiberBundle {
    /// Native Path-B materialization.
    pub pathb: NativePathBMaterialization,
    /// One exact chemical-state fiber per conserved lattice node.
    pub fibers: Vec<ChemicalStateFiber>,
    /// SHA-256 of the canonical sorted fiber-hash population.
    pub fiber_population_sha256: String,
}

impl NativePathBFiberBundle {
    /// Verify native model, fibers, and one-to-one population conservation.
    pub fn verify(&self) -> Result<()> {
        self.pathb.verify()?;
        if self.fibers.len() as u64 != self.pathb.receipt.pathb_assignments {
            bail!("fiber population does not equal Path-B assignment population");
        }
        let mut hashes = Vec::with_capacity(self.fibers.len());
        for fiber in &self.fibers {
            fiber.verify()?;
            if fiber.body.pathb.model_sha256 != self.pathb.model.model_sha256 {
                bail!("fiber references a different native Path-B model");
            }
            hashes.push(fiber.fiber_sha256.clone());
        }
        hashes.sort();
        if canonical_sha256(&hashes)? != self.fiber_population_sha256 {
            bail!("fiber population content hash mismatch");
        }
        Ok(())
    }
}

/// Estimate native metastable states and reversible transitions directly from
/// completed Ghost phase-lattice nodes and exact byte lineages. A committor is
/// deliberately unavailable until a separate hash-bound physical basin is
/// supplied; neutral cluster order is never used as a target definition.
///
/// The function is GPU-feature gated because `GhostPhaseLatticeNode` is the
/// canonical GPU lattice record. The estimator itself is deterministic host
/// code and never reads legacy binding-site JSON.
#[cfg(any(feature = "gpu", feature = "cpu-analysis"))]
pub fn estimate_native_pathb_from_lattice(
    _nodes: &[crate::ghost_phase_lattice::GhostPhaseLatticeNode],
    _lineages: &[GhostByteLineage],
    _exact_edges: &[NativePathBExactEdgeObservation],
    _connectivity_population_sha256: &str,
    _surveillance_ledgers: &[crate::pathb_evidence_emitter::PathBConfirmationSurveillanceLedger],
    _conservation: &GhostLatticeConservation,
    _config: &NativePathBEstimatorConfig,
) -> Result<NativePathBMaterialization> {
    bail!("EVIDENCE_NOT_FOUND: exact native Path-B integration timebase context is required")
}

/// Estimate native metastable states with an explicit signed integration
/// timebase. No default timestep or implicit frame cadence is admissible.
#[cfg(any(feature = "gpu", feature = "cpu-analysis"))]
pub fn estimate_native_pathb_from_lattice_with_timebase(
    nodes: &[crate::ghost_phase_lattice::GhostPhaseLatticeNode],
    lineages: &[GhostByteLineage],
    exact_edges: &[NativePathBExactEdgeObservation],
    connectivity_population_sha256: &str,
    surveillance_ledgers: &[crate::pathb_evidence_emitter::PathBConfirmationSurveillanceLedger],
    conservation: &GhostLatticeConservation,
    config: &NativePathBEstimatorConfig,
    timebase_context: &NativePathBTimebaseContext,
) -> Result<NativePathBMaterialization> {
    conservation.validate()?;
    require_sha256(
        connectivity_population_sha256,
        "connectivity_population_sha256",
    )?;
    if nodes.len() as u64 != conservation.lattice_nodes
        || lineages.len() as u64 != conservation.lattice_nodes
    {
        bail!("node/lineage population does not match conserved lattice count");
    }
    let surveillance_cadence_steps =
        validate_timebase_and_cadence(surveillance_ledgers, config, timebase_context)?;
    let surveillance_samples = validated_surveillance_samples(surveillance_ledgers, lineages)?;
    let surveillance_missingness = summarize_surveillance_missingness(surveillance_ledgers)?;
    config.validate(surveillance_samples.len())?;
    if exact_edges.is_empty() {
        bail!("native Path-B requires exact accepted lattice-edge lineage");
    }
    if surveillance_ledgers.is_empty() {
        bail!("EVIDENCE_NOT_FOUND: complete fixed-cadence Path-B surveillance is required");
    }

    // Pair the admitted vector directly while preserving each row's original
    // global lattice index. A site-selected Path-B population is a lossless
    // subset of the canonical lattice, so its indices are intentionally not
    // rewritten to a dense local range.
    let mut seen_lineages = BTreeSet::new();
    let mut ordered = nodes
        .iter()
        .zip(lineages)
        .enumerate()
        .map(|(local_index, (node, lineage))| {
            lineage.validate()?;
            if !seen_lineages.insert(lineage.lineage_sha256()?) {
                bail!("byte lineage does not map one-to-one onto admitted lattice nodes");
            }
            Ok((local_index, node, lineage))
        })
        .collect::<Result<Vec<_>>>()?;
    ordered.sort_by(|left, right| {
        left.2
            .run_id
            .cmp(&right.2.run_id)
            .then_with(|| left.2.replica_id.cmp(&right.2.replica_id))
            .then_with(|| left.2.key.cmp(&right.2.key))
    });
    for (_, node, lineage) in &ordered {
        if node.stream_id != lineage.key.stream_id
            || node.frame_idx != lineage.key.frame_idx
            || node.step_idx != lineage.key.step_idx
            || node.site_id != lineage.key.site_id
        {
            bail!("lattice node identity disagrees with exact Ghost byte lineage");
        }
    }

    let (kinetic_model_selection, features, mut hard_assignments, mut centroids) =
        select_kinetic_model(&surveillance_samples, config)?;
    let k = kinetic_model_selection.selected_state_count;
    let selected_lag_steps = kinetic_model_selection.selected_lag_steps;

    // State identities are neutral. Canonicalize cluster labels by their
    // standardized feature centroids only; a Ghost AABB is never interpreted
    // as cavity volume or used to call a state open/closed.
    let mut cluster_count = vec![0usize; k];
    for cluster in &hard_assignments {
        cluster_count[*cluster] += 1;
    }
    if cluster_count.contains(&0) {
        bail!("deterministic metastable-state estimator produced an empty state");
    }
    let mut state_order = (0..k).collect::<Vec<_>>();
    state_order.sort_by(|a, b| {
        centroids[*a]
            .iter()
            .zip(&centroids[*b])
            .map(|(left, right)| left.total_cmp(right))
            .find(|ordering| !ordering.is_eq())
            .unwrap_or_else(|| a.cmp(b))
    });
    let mut remap = vec![0usize; k];
    for (new, old) in state_order.iter().copied().enumerate() {
        remap[old] = new;
    }
    hard_assignments
        .iter_mut()
        .for_each(|state| *state = remap[*state]);
    let old_centroids = centroids;
    centroids = state_order
        .iter()
        .map(|old| old_centroids[*old].clone())
        .collect();

    let sample_position_by_lineage = surveillance_samples
        .iter()
        .enumerate()
        .filter_map(|(position, sample)| {
            sample
                .detected_ghost_lineage_sha256
                .as_ref()
                .map(|hash| (hash.clone(), position))
        })
        .collect::<BTreeMap<_, _>>();
    if sample_position_by_lineage.len() > lineages.len() {
        bail!("surveillance detection-to-Ghost assignment exceeds admitted population");
    }
    let exact_edge_population_sha256 = canonical_sha256(&exact_edges)?;
    for edge in exact_edges {
        edge.verify()?;
        if !seen_lineages.contains(&edge.from_ghost_lineage_sha256)
            || !seen_lineages.contains(&edge.to_ghost_lineage_sha256)
        {
            bail!("exact accepted edge escapes admitted surveillance candidate population");
        }
    }
    let lag_pairs = surveillance_transition_pairs_for_lag(
        &surveillance_samples,
        &hard_assignments,
        selected_lag_steps,
    )?;
    let double_lag_pairs = surveillance_transition_pairs_for_lag(
        &surveillance_samples,
        &hard_assignments,
        selected_lag_steps
            .checked_mul(2)
            .context("Path-B second diagnostic lag overflow")?,
    )?;
    let required_lanes = [
        ControlLane::NoProbe,
        ControlLane::HypothesisFragment,
        ControlLane::OrthogonalChemicalControl,
    ];
    let state_ids = (0..k)
        .map(|index| format!("state_{index:03}"))
        .collect::<Vec<_>>();
    let lane_global_state_support = build_lane_global_state_support(
        &hard_assignments,
        &surveillance_samples,
        &state_ids,
        config,
    )?;
    let mut lane_operators = Vec::new();
    for lane in required_lanes {
        let lane_pairs = transition_pairs_for_lane(&lag_pairs, &surveillance_samples, lane)?;
        let lane_double_pairs =
            transition_pairs_for_lane(&double_lag_pairs, &surveillance_samples, lane)?;
        let active_global_states =
            active_global_state_indices(lane, &state_ids, &lane_global_state_support)?;
        lane_operators.push(fit_lane_operator(
            lane,
            &lane_pairs,
            &lane_double_pairs,
            active_global_states,
            config,
        )?);
    }
    let transition_pairs = lane_operators
        .iter()
        .map(|operator| operator.lag_pairs.len())
        .sum::<usize>();
    let reference = lane_operators
        .iter()
        .find(|operator| operator.control_lane == ControlLane::NoProbe)
        .unwrap();
    let states = state_ids
        .iter()
        .enumerate()
        .map(|(index, state_id)| MetastableState {
            state_id: state_id.clone(),
            stationary_probability: reference
                .active_global_states
                .iter()
                .position(|global| *global == index)
                .map(|local| reference.stationary[local])
                .unwrap_or(0.0),
        })
        .collect::<Vec<_>>();
    let transitions = (0..reference.active_global_states.len())
        .flat_map(|from| {
            let state_ids = &state_ids;
            let reference = reference;
            (0..reference.active_global_states.len()).map(move |to| TransitionProbability {
                from_state_id: state_ids[reference.active_global_states[from]].clone(),
                to_state_id: state_ids[reference.active_global_states[to]].clone(),
                probability: reference.matrix[from][to],
            })
        })
        .collect::<Vec<_>>();
    let lane_transition_models = lane_operators
        .iter()
        .map(|operator| LaneTransitionModel {
            control_lane: operator.control_lane,
            active_state_ids: operator
                .active_global_states
                .iter()
                .map(|global| state_ids[*global].clone())
                .collect(),
            stationary_probability_by_state: operator
                .active_global_states
                .iter()
                .enumerate()
                .map(|(local, global)| (state_ids[*global].clone(), operator.stationary[local]))
                .collect(),
            transitions: (0..operator.active_global_states.len())
                .flat_map(|from| {
                    let state_ids = &state_ids;
                    (0..operator.active_global_states.len()).map(move |to| TransitionProbability {
                        from_state_id: state_ids[operator.active_global_states[from]].clone(),
                        to_state_id: state_ids[operator.active_global_states[to]].clone(),
                        probability: operator.matrix[from][to],
                    })
                })
                .collect(),
        })
        .collect::<Vec<_>>();

    let scheduled_state_assignments = surveillance_samples
        .iter()
        .enumerate()
        .map(|(position, sample)| {
            let memberships = soft_memberships(
                &features[position],
                &centroids,
                config.membership_temperature,
            );
            Ok(ScheduledStructuralStateAssignment {
                source_surveillance_ledger_sha256: sample.ledger_sha256.clone(),
                source_surveillance_row_sha256: sample.source_row_sha256.clone(),
                run_id: sample.run_id.clone(),
                replica_id: sample.replica_id.clone(),
                replica_seed: sample.seed,
                control_lane: sample.control_lane,
                stream_id: sample.stream_id,
                step_idx: sample.step_idx,
                evaluable_segment_id: sample.evaluable_segment_id,
                cavity_volume_a3: sample.cavity_volume_a3,
                binary_label: sample.binary_label,
                ghost_event: sample.ghost_event.clone(),
                state_membership: state_ids.iter().cloned().zip(memberships).collect(),
                dominant_state_id: state_ids
                    .get(hard_assignments[position])
                    .context("scheduled structural assignment state exceeds selected model")?
                    .clone(),
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let mut assignments = Vec::with_capacity(sample_position_by_lineage.len());
    for (_, _, lineage) in &ordered {
        let lineage_sha256 = lineage.lineage_sha256()?;
        let Some(position) = sample_position_by_lineage.get(&lineage_sha256).copied() else {
            // The exact Ghost occurrence remains in lattice/connectivity
            // evidence but cannot receive a kinetic state without a
            // physically evaluable scheduled measurement.
            continue;
        };
        let memberships = soft_memberships(
            &features[position],
            &centroids,
            config.membership_temperature,
        );
        let previous_states = lag_pairs
            .iter()
            .filter(|pair| pair.to_position == position)
            .map(|pair| state_ids[pair.from].clone())
            .collect::<BTreeSet<_>>();
        let previous =
            (previous_states.len() == 1).then(|| previous_states.into_iter().next().unwrap());
        assignments.push(GhostStateAssignment {
            ghost_lineage_sha256: lineage_sha256,
            ghost_record_sha256: lineage.record_sha256.clone(),
            control_lane: surveillance_samples[position].control_lane,
            state_membership: state_ids.iter().cloned().zip(memberships).collect(),
            previous_state_id: previous,
            forward_committor: None,
        });
    }
    let diagnostics = build_markov_diagnostics(
        &hard_assignments,
        &surveillance_samples,
        &lane_operators,
        &lane_global_state_support,
        &state_ids,
        config,
        kinetic_model_selection,
        surveillance_missingness,
    )?;

    let model = NativePathBModel::materialize(NativePathBModelBody {
        lag_steps: selected_lag_steps,
        timebase_context: timebase_context.clone(),
        surveillance_cadence_steps,
        normalization_tolerance: config.normalization_tolerance,
        states,
        transitions,
        reference_lane: ControlLane::NoProbe,
        lane_transition_models,
        lane_global_state_support,
        assignments,
        scheduled_state_assignments,
        source_lattice_manifest_sha256: conservation.lattice_manifest_sha256.clone(),
        source_exact_connectivity_population_sha256: connectivity_population_sha256.to_string(),
        source_exact_edge_population_sha256: exact_edge_population_sha256.clone(),
        diagnostics,
    })?;
    let population = nodes.len() as u64;
    let receipt = NativePathBMaterializationReceipt {
        produced: conservation.produced,
        persisted: conservation.persisted,
        decoded: conservation.decoded,
        lattice_nodes: conservation.lattice_nodes,
        byte_lineages: lineages.len() as u64,
        pathb_assignments: model.body.assignments.len() as u64,
        scheduled_state_assignments: model.body.scheduled_state_assignments.len() as u64,
        kinetically_evaluable_ghost_events: model.body.assignments.len() as u64,
        not_evaluable_ghost_events: lineages.len() as u64 - model.body.assignments.len() as u64,
        transition_pairs: transition_pairs as u64,
        accepted_edge_rows: exact_edges.len() as u64,
        exact_connectivity_population_sha256: connectivity_population_sha256.to_string(),
        exact_edge_population_sha256,
        metastable_states: k as u32,
        model_sha256: model.model_sha256.clone(),
        lattice_manifest_sha256: conservation.lattice_manifest_sha256.clone(),
        observed_atom_count: None,
        observed_atom_counts_by_lane_topology: BTreeMap::new(),
        conservation_pass: conservation.produced == population
            && population == lineages.len() as u64
            && model.body.assignments.len() as u64
                + (lineages.len() as u64 - model.body.assignments.len() as u64)
                == population,
        source_contract: "native_ghost_segments_to_exact_accepted_edges_to_phase_lattice_to_pathb"
            .to_string(),
        markov_diagnostics_status: model.body.diagnostics.status,
    };
    let materialization = NativePathBMaterialization { model, receipt };
    materialization.verify()?;
    Ok(materialization)
}

/// Attach a native Path-B materialization to exact Ghost/fragment observations
/// and emit one conserved fiber per Path-B assignment.
pub fn materialize_native_pathb_fibers(
    pathb: NativePathBMaterialization,
    ghost_observations: Vec<GhostFiberObservation>,
    fragment_observations: Vec<FragmentStateObservation>,
) -> Result<NativePathBFiberBundle> {
    pathb.verify()?;
    if pathb.model.body.diagnostics.status != MarkovDiagnosticStatus::Pass {
        bail!("computationally inconclusive Path-B model cannot materialize claim fibers");
    }
    let assignment_lineages = pathb
        .model
        .body
        .assignments
        .iter()
        .map(|assignment| assignment.ghost_lineage_sha256.clone())
        .collect::<BTreeSet<_>>();
    let ghost_lineages = ghost_observations
        .iter()
        .map(|observation| observation.ghost.lineage_sha256())
        .collect::<Result<BTreeSet<_>>>()?;
    if assignment_lineages != ghost_lineages
        || ghost_observations.len() != assignment_lineages.len()
        || fragment_observations.len() != assignment_lineages.len()
    {
        bail!("native Path-B, Ghost, and fragment populations are not one-to-one");
    }
    let ghosts_by_lineage = ghost_observations
        .iter()
        .map(|observation| Ok((observation.ghost.lineage_sha256()?, &observation.ghost)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    let pathb_observations = assignment_lineages
        .iter()
        .map(|lineage_sha256| {
            let lineage = ghosts_by_lineage
                .get(lineage_sha256)
                .context("Path-B assignment lacks exact Ghost occurrence")?;
            Ok(PathBProjectionObservation {
                ghost_lineage_sha256: lineage_sha256.clone(),
                pathb: pathb.model.project(lineage)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let fibers = assemble_fibers(
        ghost_observations,
        fragment_observations,
        pathb_observations,
    )?;
    let mut hashes = fibers
        .iter()
        .map(|fiber| fiber.fiber_sha256.clone())
        .collect::<Vec<_>>();
    hashes.sort();
    let bundle = NativePathBFiberBundle {
        pathb,
        fibers,
        fiber_population_sha256: canonical_sha256(&hashes)?,
    };
    bundle.verify()?;
    Ok(bundle)
}

#[cfg(any(feature = "gpu", feature = "cpu-analysis"))]
fn lattice_feature_vector(
    node: &crate::ghost_phase_lattice::GhostPhaseLatticeNode,
    required_so3_plane_mask: u8,
) -> Result<Vec<f64>> {
    if !node.has_valid_aabb()
        || node.aabb_volume() <= 0.0
        || (node.so3_plane_status & required_so3_plane_mask) != required_so3_plane_mask
    {
        bail!("native Path-B node is missing required spatial or SO(3) evidence");
    }
    let mut features = Vec::with_capacity(33);
    features.extend(node.centroid_xyz.into_iter().map(f64::from));
    features.push(f64::from(node.aabb_volume()).ln_1p());
    if !node.kl_divergence.is_finite() || node.thermo_flux.iter().any(|value| !value.is_finite()) {
        bail!("native Path-B node has non-finite KL or thermodynamic flux");
    }
    features.push(f64::from(node.kl_divergence));
    features.extend(node.thermo_flux.into_iter().map(f64::from));
    features.push(f64::from(node.protocol_phase) / 3.0);
    for plane in node.so3_power_spectrum {
        features.extend(plane.into_iter().map(f64::from));
    }
    if features.iter().any(|value| !value.is_finite()) {
        bail!("native Path-B feature vector is non-finite");
    }
    Ok(features)
}

fn standardize_features(raw: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    let dimensions = raw.first().context("no Path-B feature vectors")?.len();
    if dimensions == 0 || raw.iter().any(|row| row.len() != dimensions) {
        bail!("Path-B feature matrix is ragged or empty");
    }
    let n = raw.len() as f64;
    let mut means = vec![0.0; dimensions];
    for row in raw {
        for (index, value) in row.iter().enumerate() {
            means[index] += value / n;
        }
    }
    let mut scales = vec![0.0; dimensions];
    for row in raw {
        for (index, value) in row.iter().enumerate() {
            scales[index] += (value - means[index]).powi(2) / n;
        }
    }
    scales.iter_mut().for_each(|scale| {
        *scale = scale.sqrt();
        if *scale <= 1.0e-15 {
            *scale = 1.0;
        }
    });
    Ok(raw
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(index, value)| (value - means[index]) / scales[index])
                .collect()
        })
        .collect())
}

fn squared_distance(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| (a - b).powi(2)).sum()
}

fn deterministic_kmeans(
    features: &[Vec<f64>],
    k: usize,
    maximum_iterations: usize,
) -> Result<(Vec<usize>, Vec<Vec<f64>>)> {
    let mut centroid_indices = vec![0usize];
    while centroid_indices.len() < k {
        let next = (0..features.len())
            .filter(|index| !centroid_indices.contains(index))
            .max_by(|a, b| {
                let distance_a = centroid_indices
                    .iter()
                    .map(|centroid| squared_distance(&features[*a], &features[*centroid]))
                    .fold(f64::INFINITY, f64::min);
                let distance_b = centroid_indices
                    .iter()
                    .map(|centroid| squared_distance(&features[*b], &features[*centroid]))
                    .fold(f64::INFINITY, f64::min);
                distance_a.total_cmp(&distance_b).then_with(|| b.cmp(a))
            })
            .context("cannot initialize requested Path-B metastable states")?;
        centroid_indices.push(next);
    }
    let mut centroids = centroid_indices
        .iter()
        .map(|index| features[*index].clone())
        .collect::<Vec<_>>();
    let mut assignments = vec![usize::MAX; features.len()];
    for _ in 0..maximum_iterations {
        let new_assignments = features
            .iter()
            .map(|feature| {
                centroids
                    .iter()
                    .enumerate()
                    .min_by(|(index_a, a), (index_b, b)| {
                        squared_distance(feature, a)
                            .total_cmp(&squared_distance(feature, b))
                            .then_with(|| index_a.cmp(index_b))
                    })
                    .map(|(index, _)| index)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        if new_assignments == assignments {
            return Ok((assignments, centroids));
        }
        assignments = new_assignments;
        let dimensions = features[0].len();
        let mut sums = vec![vec![0.0; dimensions]; k];
        let mut counts = vec![0usize; k];
        for (feature, cluster) in features.iter().zip(&assignments) {
            counts[*cluster] += 1;
            for (dimension, value) in feature.iter().enumerate() {
                sums[*cluster][dimension] += value;
            }
        }
        if counts.contains(&0) {
            bail!("native Path-B k-means produced an empty metastable state");
        }
        for cluster in 0..k {
            for dimension in 0..dimensions {
                sums[cluster][dimension] /= counts[cluster] as f64;
            }
        }
        centroids = sums;
    }
    bail!("native Path-B deterministic estimator did not converge")
}

fn soft_memberships(feature: &[f64], centroids: &[Vec<f64>], temperature: f64) -> Vec<f64> {
    let distances = centroids
        .iter()
        .map(|centroid| squared_distance(feature, centroid))
        .collect::<Vec<_>>();
    let minimum = distances.iter().copied().fold(f64::INFINITY, f64::min);
    let weights = distances
        .iter()
        .map(|distance| (-(distance - minimum) / temperature).exp())
        .collect::<Vec<_>>();
    let total = weights.iter().sum::<f64>();
    weights.into_iter().map(|weight| weight / total).collect()
}

#[derive(Clone, Copy, Debug)]
struct SeedTransitionPair {
    seed: u64,
    from: usize,
    to: usize,
    from_position: usize,
    to_position: usize,
}

#[derive(Debug, Clone)]
struct FittedLaneOperator {
    control_lane: ControlLane,
    /// Global state indices in local operator order.
    active_global_states: Vec<usize>,
    matrix: Vec<Vec<f64>>,
    stationary: Vec<f64>,
    /// Exact-lag pairs remapped into the active local subspace.
    lag_pairs: Vec<SeedTransitionPair>,
    /// Second-lag pairs remapped into the active local subspace.
    double_lag_pairs: Vec<SeedTransitionPair>,
}

#[derive(Debug, Clone)]
struct SurveillanceSample {
    ledger_sha256: String,
    run_id: String,
    replica_id: String,
    control_lane: crate::chemical_state_fiber::ControlLane,
    /// Maximal contiguous evaluable epoch within one ledger/stream. A
    /// NOT_EVALUABLE scheduled row increments this identity and therefore
    /// prevents any longer-lag pair from bridging the missing interval.
    evaluable_segment_id: u64,
    seed: u64,
    stream_id: u16,
    step_idx: u64,
    source_row_sha256: String,
    cavity_volume_a3: f64,
    binary_label: Option<CavityBinaryLabel>,
    ghost_event: ScheduledGhostEventStatus,
    features: Vec<f64>,
    feature_schema_sha256: String,
    detected_ghost_lineage_sha256: Option<String>,
}

fn validate_cadence_values(
    cadence_values: impl IntoIterator<Item = u64>,
    config: &NativePathBEstimatorConfig,
    timebase_context: &NativePathBTimebaseContext,
) -> Result<u64> {
    timebase_context.verify()?;
    let cadences = cadence_values.into_iter().collect::<BTreeSet<_>>();
    if cadences.len() != 1 || cadences.contains(&0) {
        bail!("EVIDENCE_NOT_FOUND: surveillance population lacks one exact nonzero cadence");
    }
    let cadence_steps = *cadences.iter().next().unwrap();
    if config
        .candidate_lag_steps
        .iter()
        .any(|lag| *lag == 0 || *lag % cadence_steps != 0)
    {
        bail!("EVIDENCE_NOT_FOUND: Path-B lag grid is not exactly surveillance-cadence aligned");
    }
    Ok(cadence_steps)
}

fn validate_timebase_and_cadence(
    ledgers: &[crate::pathb_evidence_emitter::PathBConfirmationSurveillanceLedger],
    config: &NativePathBEstimatorConfig,
    timebase_context: &NativePathBTimebaseContext,
) -> Result<u64> {
    validate_cadence_values(
        ledgers.iter().map(|ledger| ledger.cadence_steps),
        config,
        timebase_context,
    )
}

fn validated_surveillance_samples(
    ledgers: &[crate::pathb_evidence_emitter::PathBConfirmationSurveillanceLedger],
    lineages: &[GhostByteLineage],
) -> Result<Vec<SurveillanceSample>> {
    require_one_stationary_surveillance_stratum(ledgers.iter().map(|ledger| {
        (
            ledger.stationary_phase_ordinal,
            ledger.matched_physical_state_sha256.as_str(),
        )
    }))?;
    let admitted = lineages
        .iter()
        .map(|lineage| Ok((lineage.lineage_sha256()?, lineage)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    let mut detected = BTreeSet::new();
    let mut feature_schema = BTreeSet::new();
    let mut feature_dimension = None;
    let mut samples = Vec::new();
    for ledger in ledgers {
        ledger.verify()?;
        let mut evaluable_segment_by_stream = BTreeMap::<u16, u64>::new();
        for row in &ledger.rows {
            let detected_lineage = match &row.ghost_event {
                crate::pathb_evidence_emitter::PathBGhostEventStatus::Emitted {
                    ghost_lineage_sha256,
                    ..
                } => {
                    let lineage = admitted.get(ghost_lineage_sha256).context(
                        "surveillance detection is absent from admitted Ghost population",
                    )?;
                    if lineage.run_id != ledger.run_id
                        || lineage.replica_id != ledger.replica_id
                        || lineage.replica_seed != ledger.replica_seed
                        || lineage.engine_sha256 != ledger.engine_sha256
                        || lineage.topology_sha256 != ledger.topology_sha256
                        || lineage.key.stream_id != row.stream_id
                        || lineage.key.step_idx != row.step_idx
                        || lineage.protocol_phase != row.protocol_phase.phase_ordinal
                        || lineage.condition_id != row.thermodynamic_condition.condition_id
                        || !detected.insert(ghost_lineage_sha256.clone())
                    {
                        bail!("surveillance detection duplicates or disagrees with Ghost lineage");
                    }
                    Some(ghost_lineage_sha256.clone())
                }
                crate::pathb_evidence_emitter::PathBGhostEventStatus::AmbiguousEmissions {
                    ghost_lineage_sha256,
                    ..
                } => {
                    for hash in ghost_lineage_sha256 {
                        let lineage = admitted.get(hash).context(
                            "ambiguous surveillance detection is absent from admitted Ghost population",
                        )?;
                        if lineage.run_id != ledger.run_id
                            || lineage.replica_id != ledger.replica_id
                            || lineage.replica_seed != ledger.replica_seed
                            || lineage.engine_sha256 != ledger.engine_sha256
                            || lineage.topology_sha256 != ledger.topology_sha256
                            || lineage.key.stream_id != row.stream_id
                            || lineage.key.step_idx != row.step_idx
                            || !detected.insert(hash.clone())
                        {
                            bail!("ambiguous surveillance detection disagrees with Ghost lineage");
                        }
                    }
                    None
                }
                crate::pathb_evidence_emitter::PathBGhostEventStatus::NotEmitted { .. } => None,
            };
            let (evidence, binary_label) = match &row.measurement {
                crate::pathb_evidence_emitter::PathBMeasurementStatus::ObservedOpen {
                    evidence,
                } => (evidence, Some(CavityBinaryLabel::Open)),
                crate::pathb_evidence_emitter::PathBMeasurementStatus::ObservedClosed {
                    evidence,
                } => (evidence, Some(CavityBinaryLabel::Closed)),
                crate::pathb_evidence_emitter::PathBMeasurementStatus::ObservedIndeterminate {
                    evidence,
                } => (evidence, None),
                crate::pathb_evidence_emitter::PathBMeasurementStatus::NotEvaluable { .. } => {
                    // A scheduled NOT_EVALUABLE row remains conserved in the
                    // surveillance ledger, and splits the maximal evaluable
                    // segment so no longer-lag transition can bridge it.
                    let segment = evaluable_segment_by_stream
                        .entry(row.stream_id)
                        .or_default();
                    *segment = segment
                        .checked_add(1)
                        .context("surveillance evaluability segment overflow")?;
                    continue;
                }
            };
            let kinetic_feature_schema_sha256 = &evidence.kinetic_feature_schema_sha256;
            let kinetic_features = &evidence.kinetic_features;
            feature_schema.insert(kinetic_feature_schema_sha256.clone());
            match feature_dimension {
                Some(dimension) if dimension != kinetic_features.len() => {
                    bail!("surveillance kinetic feature dimension varies")
                }
                None => feature_dimension = Some(kinetic_features.len()),
                _ => {}
            }
            samples.push(SurveillanceSample {
                ledger_sha256: ledger.ledger_sha256.clone(),
                run_id: ledger.run_id.clone(),
                replica_id: ledger.replica_id.clone(),
                control_lane: ledger.control_lane,
                evaluable_segment_id: *evaluable_segment_by_stream
                    .entry(row.stream_id)
                    .or_default(),
                seed: ledger.replica_seed,
                stream_id: row.stream_id,
                step_idx: row.step_idx,
                source_row_sha256: row.row_sha256.clone(),
                cavity_volume_a3: evidence.cavity_volume_a3,
                binary_label,
                ghost_event: scheduled_ghost_event_status(&row.ghost_event),
                features: kinetic_features.clone(),
                feature_schema_sha256: kinetic_feature_schema_sha256.clone(),
                detected_ghost_lineage_sha256: detected_lineage,
            });
        }
    }
    if detected != admitted.keys().cloned().collect::<BTreeSet<_>>() {
        bail!("fixed-cadence surveillance does not account for every admitted Ghost occurrence");
    }
    if feature_schema.len() != 1 || feature_dimension.unwrap_or(0) < 3 {
        bail!("surveillance population lacks one common kinetic feature schema");
    }
    let mut output = samples;
    output.sort_by(|left, right| {
        left.ledger_sha256
            .cmp(&right.ledger_sha256)
            .then_with(|| left.stream_id.cmp(&right.stream_id))
            .then_with(|| left.step_idx.cmp(&right.step_idx))
    });
    if output.windows(2).any(|pair| {
        (
            pair[0].ledger_sha256.as_str(),
            pair[0].stream_id,
            pair[0].step_idx,
        ) == (
            pair[1].ledger_sha256.as_str(),
            pair[1].stream_id,
            pair[1].step_idx,
        )
    }) {
        bail!("surveillance population duplicates one ledger/stream/step observation");
    }
    Ok(output)
}

fn scheduled_ghost_event_status(
    status: &crate::pathb_evidence_emitter::PathBGhostEventStatus,
) -> ScheduledGhostEventStatus {
    match status {
        crate::pathb_evidence_emitter::PathBGhostEventStatus::Emitted {
            ghost_lineage_sha256,
            exact_component_sha256,
        } => ScheduledGhostEventStatus::Emitted {
            ghost_lineage_sha256: ghost_lineage_sha256.clone(),
            exact_component_sha256: exact_component_sha256.clone(),
        },
        crate::pathb_evidence_emitter::PathBGhostEventStatus::AmbiguousEmissions {
            ghost_lineage_sha256,
            exact_component_sha256,
        } => ScheduledGhostEventStatus::AmbiguousEmissions {
            ghost_lineage_sha256: ghost_lineage_sha256.clone(),
            exact_component_sha256: exact_component_sha256.clone(),
        },
        crate::pathb_evidence_emitter::PathBGhostEventStatus::NotEmitted {
            typed_absence_receipt_sha256,
        } => ScheduledGhostEventStatus::NotEmitted {
            typed_absence_receipt_sha256: typed_absence_receipt_sha256.clone(),
        },
    }
}

fn require_one_stationary_surveillance_stratum<'a>(
    strata: impl IntoIterator<Item = (u16, &'a str)>,
) -> Result<()> {
    let strata = strata.into_iter().collect::<BTreeSet<_>>();
    if strata.len() != 1 {
        bail!("reversible surveillance fitting crosses stationary phase or physical state strata");
    }
    Ok(())
}

fn summarize_surveillance_missingness(
    ledgers: &[crate::pathb_evidence_emitter::PathBConfirmationSurveillanceLedger],
) -> Result<SurveillanceMissingnessSummary> {
    let mut ledger_hashes = Vec::with_capacity(ledgers.len());
    let mut scheduled_rows = 0u64;
    let mut evaluable_rows = 0u64;
    let mut not_evaluable_rows = 0u64;
    let mut observed_open_rows = 0u64;
    let mut observed_closed_rows = 0u64;
    let mut observed_indeterminate_rows = 0u64;
    let mut ghost_emitted_rows = 0u64;
    let mut ghost_not_emitted_rows = 0u64;
    let mut not_evaluable_reason_counts = BTreeMap::new();
    let mut run_units = Vec::with_capacity(ledgers.len());
    for ledger in ledgers {
        ledger.verify()?;
        ledger_hashes.push(ledger.ledger_sha256.clone());
        scheduled_rows += ledger.scheduled_row_count;
        evaluable_rows += ledger.evaluable_row_count;
        not_evaluable_rows += ledger.not_evaluable_row_count;
        for row in &ledger.rows {
            match &row.ghost_event {
                crate::pathb_evidence_emitter::PathBGhostEventStatus::Emitted { .. } => {
                    ghost_emitted_rows += 1
                }
                crate::pathb_evidence_emitter::PathBGhostEventStatus::AmbiguousEmissions {
                    ..
                } => ghost_emitted_rows += 1,
                crate::pathb_evidence_emitter::PathBGhostEventStatus::NotEmitted { .. } => {
                    ghost_not_emitted_rows += 1
                }
            }
            match &row.measurement {
                crate::pathb_evidence_emitter::PathBMeasurementStatus::ObservedOpen { .. } => {
                    observed_open_rows += 1
                }
                crate::pathb_evidence_emitter::PathBMeasurementStatus::ObservedClosed {
                    ..
                } => observed_closed_rows += 1,
                crate::pathb_evidence_emitter::PathBMeasurementStatus::ObservedIndeterminate {
                    ..
                } => observed_indeterminate_rows += 1,
                crate::pathb_evidence_emitter::PathBMeasurementStatus::NotEvaluable {
                    reason_code,
                    ..
                } => {
                    *not_evaluable_reason_counts
                        .entry(reason_code.clone())
                        .or_insert(0) += 1
                }
            }
        }
        run_units.push(SurveillanceRunMissingness {
            run_id: ledger.run_id.clone(),
            replica_id: ledger.replica_id.clone(),
            replica_seed: ledger.replica_seed,
            control_lane: ledger.control_lane,
            scheduled_rows: ledger.scheduled_row_count,
            evaluable_rows: ledger.evaluable_row_count,
            not_evaluable_rows: ledger.not_evaluable_row_count,
        });
    }
    ledger_hashes.sort();
    if ledger_hashes.windows(2).any(|pair| pair[0] == pair[1]) {
        bail!("surveillance missingness population contains a duplicate ledger");
    }
    run_units.sort_by(|left, right| {
        (
            &left.run_id,
            &left.replica_id,
            left.replica_seed,
            left.control_lane,
        )
            .cmp(&(
                &right.run_id,
                &right.replica_id,
                right.replica_seed,
                right.control_lane,
            ))
    });
    if scheduled_rows == 0
        || scheduled_rows != evaluable_rows + not_evaluable_rows
        || scheduled_rows
            != observed_open_rows
                + observed_closed_rows
                + observed_indeterminate_rows
                + not_evaluable_rows
        || scheduled_rows != ghost_emitted_rows + ghost_not_emitted_rows
    {
        bail!("surveillance missingness conservation failed");
    }
    Ok(SurveillanceMissingnessSummary {
        surveillance_ledger_population_sha256: canonical_sha256(&ledger_hashes)?,
        scheduled_rows,
        evaluable_rows,
        not_evaluable_rows,
        evaluable_fraction: evaluable_rows as f64 / scheduled_rows as f64,
        observed_open_rows,
        observed_closed_rows,
        observed_indeterminate_rows,
        ghost_emitted_rows,
        ghost_not_emitted_rows,
        not_evaluable_reason_counts,
        run_units,
    })
}

fn surveillance_transition_pairs_for_lag(
    samples: &[SurveillanceSample],
    assignments: &[usize],
    lag_steps: u64,
) -> Result<Vec<SeedTransitionPair>> {
    if samples.len() != assignments.len() {
        bail!("surveillance sample and assignment populations differ");
    }
    let positions = samples
        .iter()
        .enumerate()
        .map(|(position, sample)| {
            (
                (
                    sample.ledger_sha256.as_str(),
                    sample.stream_id,
                    sample.evaluable_segment_id,
                    sample.step_idx,
                ),
                position,
            )
        })
        .collect::<BTreeMap<_, _>>();
    if positions.len() != samples.len() {
        bail!("surveillance transition grid contains duplicate observations");
    }
    let mut pairs = Vec::new();
    for (from_position, sample) in samples.iter().enumerate() {
        let Some(target_step) = sample.step_idx.checked_add(lag_steps) else {
            continue;
        };
        if let Some(to_position) = positions.get(&(
            sample.ledger_sha256.as_str(),
            sample.stream_id,
            sample.evaluable_segment_id,
            target_step,
        )) {
            pairs.push(SeedTransitionPair {
                seed: sample.seed,
                from: assignments[from_position],
                to: assignments[*to_position],
                from_position,
                to_position: *to_position,
            });
        }
    }
    Ok(pairs)
}

fn surveillance_position_pairs_for_lag(
    samples: &[SurveillanceSample],
    lag_steps: u64,
) -> Result<Vec<(u64, usize, usize)>> {
    let positions = samples
        .iter()
        .enumerate()
        .map(|(position, sample)| {
            (
                (
                    sample.ledger_sha256.as_str(),
                    sample.stream_id,
                    sample.evaluable_segment_id,
                    sample.step_idx,
                ),
                position,
            )
        })
        .collect::<BTreeMap<_, _>>();
    if positions.len() != samples.len() {
        bail!("surveillance position grid contains duplicate rows");
    }
    let mut pairs = Vec::new();
    for (from_position, sample) in samples.iter().enumerate() {
        let Some(target_step) = sample.step_idx.checked_add(lag_steps) else {
            continue;
        };
        if let Some(to_position) = positions.get(&(
            sample.ledger_sha256.as_str(),
            sample.stream_id,
            sample.evaluable_segment_id,
            target_step,
        )) {
            pairs.push((sample.seed, from_position, *to_position));
        }
    }
    Ok(pairs)
}

#[derive(Debug, Clone)]
struct KineticEmbeddingModel {
    mean: Vec<f64>,
    scale: Vec<f64>,
    projection: Vec<Vec<f64>>,
}

/// Singular functions fitted exclusively on the training transition
/// population. Held-out data enter only through covariance evaluation in the
/// VAMP-E functional; no held-out transition matrix is ever estimated.
#[derive(Debug, Clone, Serialize)]
struct TrainFittedVampEModel {
    left_singular_functions: Vec<Vec<f64>>,
    right_singular_functions: Vec<Vec<f64>>,
    singular_values: Vec<f64>,
}

fn indicator_covariances(
    assignments: &[usize],
    pairs: &[(u64, usize, usize)],
    state_count: usize,
) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    if pairs.is_empty() || state_count == 0 {
        bail!("indicator covariance population is empty or underspecified");
    }
    let mut c00 = vec![vec![0.0; state_count]; state_count];
    let mut c01 = vec![vec![0.0; state_count]; state_count];
    let mut c11 = vec![vec![0.0; state_count]; state_count];
    for (_, from_position, to_position) in pairs {
        let from = *assignments
            .get(*from_position)
            .context("VAMP-E origin position is absent")?;
        let to = *assignments
            .get(*to_position)
            .context("VAMP-E destination position is absent")?;
        if from >= state_count || to >= state_count {
            bail!("VAMP-E assignment exceeds declared state count");
        }
        c00[from][from] += 1.0;
        c01[from][to] += 1.0;
        c11[to][to] += 1.0;
    }
    let denominator = pairs.len() as f64;
    for matrix in [&mut c00, &mut c01, &mut c11] {
        for row in matrix.iter_mut() {
            for value in row {
                *value /= denominator;
            }
        }
    }
    Ok((c00, c01, c11))
}

fn fit_train_vampe_model(
    assignments: &[usize],
    train_pairs: &[(u64, usize, usize)],
    state_count: usize,
    regularization: f64,
) -> Result<TrainFittedVampEModel> {
    if !regularization.is_finite() || regularization <= 0.0 {
        bail!("VAMP-E covariance regularization must be positive");
    }
    let (mut c00, c01, mut c11) = indicator_covariances(assignments, train_pairs, state_count)?;
    for state in 0..state_count {
        c00[state][state] += regularization;
        c11[state][state] += regularization;
    }
    let inv00 = (0..state_count)
        .map(|state| c00[state][state].sqrt().recip())
        .collect::<Vec<_>>();
    let inv11 = (0..state_count)
        .map(|state| c11[state][state].sqrt().recip())
        .collect::<Vec<_>>();
    let whitened = (0..state_count)
        .map(|i| {
            (0..state_count)
                .map(|j| inv00[i] * c01[i][j] * inv11[j])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let transpose = (0..state_count)
        .map(|i| (0..state_count).map(|j| whitened[j][i]).collect())
        .collect::<Vec<Vec<f64>>>();
    let gram = matrix_multiply(&transpose, &whitened)?;
    let (eigenvalues, right_vectors) = jacobi_symmetric_eigendecomposition(gram)?;
    let mut order = (0..state_count).collect::<Vec<_>>();
    order.sort_by(|left, right| {
        eigenvalues[*right]
            .max(0.0)
            .total_cmp(&eigenvalues[*left].max(0.0))
            .then_with(|| left.cmp(right))
    });
    let mut singular_values = Vec::with_capacity(state_count);
    let mut left = vec![vec![0.0; state_count]; state_count];
    let mut right = vec![vec![0.0; state_count]; state_count];
    for (output, mode) in order.into_iter().enumerate() {
        let singular = eigenvalues[mode].max(0.0).sqrt();
        singular_values.push(singular);
        for state in 0..state_count {
            let right_whitened = right_vectors[state][mode];
            right[state][output] = inv11[state] * right_whitened;
        }
        if singular > 1.0e-12 {
            for i in 0..state_count {
                let left_whitened = (0..state_count)
                    .map(|j| whitened[i][j] * right_vectors[j][mode])
                    .sum::<f64>()
                    / singular;
                left[i][output] = inv00[i] * left_whitened;
            }
        }
    }
    Ok(TrainFittedVampEModel {
        left_singular_functions: left,
        right_singular_functions: right,
        singular_values,
    })
}

fn bilinear(left: &[f64], matrix: &[Vec<f64>], right: &[f64]) -> Result<f64> {
    if matrix.len() != left.len() || matrix.iter().any(|row| row.len() != right.len()) {
        bail!("VAMP-E bilinear form dimension mismatch");
    }
    Ok(left
        .iter()
        .enumerate()
        .map(|(i, left_value)| {
            *left_value
                * right
                    .iter()
                    .enumerate()
                    .map(|(j, right_value)| matrix[i][j] * right_value)
                    .sum::<f64>()
        })
        .sum())
}

/// Wu-Noe VAMP-E score evaluated on held-out covariances with a fixed training
/// model:
/// `2 tr(S U^T C01 V) - tr(S U^T C00 U S V^T C11 V)`.
fn held_out_vampe_score(
    model: &TrainFittedVampEModel,
    assignments: &[usize],
    validation_pairs: &[(u64, usize, usize)],
    state_count: usize,
) -> Result<f64> {
    let (c00, c01, c11) = indicator_covariances(assignments, validation_pairs, state_count)?;
    if model.singular_values.len() != state_count
        || model.left_singular_functions.len() != state_count
        || model.right_singular_functions.len() != state_count
        || model
            .left_singular_functions
            .iter()
            .chain(&model.right_singular_functions)
            .any(|row| row.len() != state_count)
    {
        bail!("train-fitted VAMP-E model dimension mismatch");
    }
    let column =
        |matrix: &[Vec<f64>], index: usize| matrix.iter().map(|row| row[index]).collect::<Vec<_>>();
    let mut first_trace = 0.0;
    for mode in 0..state_count {
        first_trace += model.singular_values[mode]
            * bilinear(
                &column(&model.left_singular_functions, mode),
                &c01,
                &column(&model.right_singular_functions, mode),
            )?;
    }
    let mut second_trace = 0.0;
    for left_mode in 0..state_count {
        for right_mode in 0..state_count {
            second_trace += model.singular_values[left_mode]
                * model.singular_values[right_mode]
                * bilinear(
                    &column(&model.left_singular_functions, left_mode),
                    &c00,
                    &column(&model.left_singular_functions, right_mode),
                )?
                * bilinear(
                    &column(&model.right_singular_functions, right_mode),
                    &c11,
                    &column(&model.right_singular_functions, left_mode),
                )?;
        }
    }
    let score = 2.0 * first_trace - second_trace;
    if !score.is_finite() {
        bail!("held-out VAMP-E score is nonfinite");
    }
    Ok(score)
}

impl KineticEmbeddingModel {
    fn project(&self, features: &[f64]) -> Result<Vec<f64>> {
        if features.len() != self.mean.len()
            || self.scale.len() != self.mean.len()
            || self.projection.len() != self.mean.len()
        {
            bail!("kinetic embedding feature dimension mismatch");
        }
        let standardized = features
            .iter()
            .zip(&self.mean)
            .zip(&self.scale)
            .map(|((value, mean), scale)| (value - mean) / scale)
            .collect::<Vec<_>>();
        let dimensions = self.projection.first().map(Vec::len).unwrap_or(0);
        Ok((0..dimensions)
            .map(|dimension| {
                standardized
                    .iter()
                    .enumerate()
                    .map(|(feature, value)| value * self.projection[feature][dimension])
                    .sum()
            })
            .collect())
    }
}

fn fit_reversible_tica(
    raw_features: &[Vec<f64>],
    train_positions: &BTreeSet<usize>,
    lag_pairs: &[(u64, usize, usize)],
    dimensions: usize,
    regularization: f64,
) -> Result<KineticEmbeddingModel> {
    let feature_count = raw_features.first().map(Vec::len).unwrap_or(0);
    if feature_count == 0
        || dimensions == 0
        || !regularization.is_finite()
        || regularization <= 0.0
        || train_positions.len() < 3
    {
        bail!("reversible TICA has an invalid training population");
    }
    let mut mean = vec![0.0; feature_count];
    for position in train_positions {
        if raw_features[*position].len() != feature_count {
            bail!("TICA raw feature dimensions vary");
        }
        for (slot, value) in mean.iter_mut().zip(&raw_features[*position]) {
            *slot += value;
        }
    }
    for value in &mut mean {
        *value /= train_positions.len() as f64;
    }
    let mut scale = vec![0.0; feature_count];
    for position in train_positions {
        for feature in 0..feature_count {
            scale[feature] += (raw_features[*position][feature] - mean[feature]).powi(2);
        }
    }
    for value in &mut scale {
        *value = (*value / (train_positions.len() - 1) as f64).sqrt();
        if !value.is_finite() || *value < 1.0e-12 {
            *value = 1.0;
        }
    }
    let standardized = raw_features
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(feature, value)| (value - mean[feature]) / scale[feature])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let train_pairs = lag_pairs
        .iter()
        .filter(|(_, from, to)| train_positions.contains(from) && train_positions.contains(to))
        .collect::<Vec<_>>();
    if train_pairs.len() < feature_count.max(3) {
        bail!("reversible TICA has insufficient exact-lag training pairs");
    }
    let mut c0 = vec![vec![0.0; feature_count]; feature_count];
    let mut ct = vec![vec![0.0; feature_count]; feature_count];
    for (_, from, to) in &train_pairs {
        let left = &standardized[*from];
        let right = &standardized[*to];
        for i in 0..feature_count {
            for j in 0..feature_count {
                c0[i][j] += 0.5 * (left[i] * left[j] + right[i] * right[j]);
                ct[i][j] += 0.5 * (left[i] * right[j] + right[i] * left[j]);
            }
        }
    }
    let denominator = train_pairs.len() as f64;
    for i in 0..feature_count {
        for j in 0..feature_count {
            c0[i][j] /= denominator;
            ct[i][j] /= denominator;
        }
        c0[i][i] += regularization;
    }
    let (c0_eigenvalues, c0_eigenvectors) = jacobi_symmetric_eigendecomposition(c0)?;
    let mut inverse_sqrt = vec![vec![0.0; feature_count]; feature_count];
    for mode in 0..feature_count {
        let inverse = c0_eigenvalues[mode].max(regularization).sqrt().recip();
        for i in 0..feature_count {
            for j in 0..feature_count {
                inverse_sqrt[i][j] += c0_eigenvectors[i][mode] * inverse * c0_eigenvectors[j][mode];
            }
        }
    }
    let whitened = matrix_multiply(&matrix_multiply(&inverse_sqrt, &ct)?, &inverse_sqrt)?;
    let (kinetic_values, kinetic_vectors) = jacobi_symmetric_eigendecomposition(whitened)?;
    let mut order = (0..feature_count).collect::<Vec<_>>();
    order.sort_by(|left, right| {
        kinetic_values[*right]
            .abs()
            .total_cmp(&kinetic_values[*left].abs())
            .then_with(|| left.cmp(right))
    });
    let retained = dimensions.min(feature_count);
    let mut projection = vec![vec![0.0; retained]; feature_count];
    for (output_dimension, mode) in order.into_iter().take(retained).enumerate() {
        for feature in 0..feature_count {
            projection[feature][output_dimension] = (0..feature_count)
                .map(|inner| inverse_sqrt[feature][inner] * kinetic_vectors[inner][mode])
                .sum();
        }
    }
    Ok(KineticEmbeddingModel {
        mean,
        scale,
        projection,
    })
}

fn matrix_multiply(left: &[Vec<f64>], right: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    if left.is_empty()
        || right.is_empty()
        || left[0].len() != right.len()
        || left.iter().any(|row| row.len() != left[0].len())
        || right.iter().any(|row| row.len() != right[0].len())
    {
        bail!("matrix multiplication dimension mismatch");
    }
    Ok((0..left.len())
        .map(|i| {
            (0..right[0].len())
                .map(|j| (0..right.len()).map(|k| left[i][k] * right[k][j]).sum())
                .collect()
        })
        .collect())
}

fn jacobi_symmetric_eigendecomposition(
    mut matrix: Vec<Vec<f64>>,
) -> Result<(Vec<f64>, Vec<Vec<f64>>)> {
    let n = matrix.len();
    if n == 0 || matrix.iter().any(|row| row.len() != n) {
        bail!("Jacobi eigendecomposition requires a square matrix");
    }
    let mut vectors = vec![vec![0.0; n]; n];
    for (index, row) in vectors.iter_mut().enumerate() {
        row[index] = 1.0;
    }
    for _ in 0..(200 * n * n).max(1) {
        let mut p = 0;
        let mut q = 0;
        let mut maximum = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                if matrix[i][j].abs() > maximum {
                    maximum = matrix[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if maximum < 1.0e-12 {
            return Ok(((0..n).map(|i| matrix[i][i]).collect(), vectors));
        }
        let angle = 0.5 * (2.0 * matrix[p][q]).atan2(matrix[q][q] - matrix[p][p]);
        let (sine, cosine) = angle.sin_cos();
        let app = matrix[p][p];
        let aqq = matrix[q][q];
        let apq = matrix[p][q];
        matrix[p][p] = cosine * cosine * app - 2.0 * sine * cosine * apq + sine * sine * aqq;
        matrix[q][q] = sine * sine * app + 2.0 * sine * cosine * apq + cosine * cosine * aqq;
        matrix[p][q] = 0.0;
        matrix[q][p] = 0.0;
        for r in 0..n {
            if r != p && r != q {
                let arp = matrix[r][p];
                let arq = matrix[r][q];
                matrix[r][p] = cosine * arp - sine * arq;
                matrix[p][r] = matrix[r][p];
                matrix[r][q] = sine * arp + cosine * arq;
                matrix[q][r] = matrix[r][q];
            }
            let vrp = vectors[r][p];
            let vrq = vectors[r][q];
            vectors[r][p] = cosine * vrp - sine * vrq;
            vectors[r][q] = sine * vrp + cosine * vrq;
        }
    }
    bail!("Jacobi eigendecomposition did not converge")
}

fn nearest_centroid_assignments(
    features: &[Vec<f64>],
    centroids: &[Vec<f64>],
) -> Result<Vec<usize>> {
    if features.is_empty() || centroids.is_empty() {
        bail!("centroid assignment requires non-empty inputs");
    }
    features
        .iter()
        .map(|feature| {
            centroids
                .iter()
                .enumerate()
                .map(|(index, centroid)| (index, squared_distance(feature, centroid)))
                .min_by(|left, right| {
                    left.1
                        .total_cmp(&right.1)
                        .then_with(|| left.0.cmp(&right.0))
                })
                .map(|(index, _)| index)
                .context("centroid assignment has no centroid")
        })
        .collect()
}

fn kinetic_selection_ledger_hash(ledger: &KineticModelSelectionLedger) -> Result<String> {
    canonical_sha256(&(
        ledger.feature_schema_sha256.as_str(),
        ledger.surveillance_population_sha256.as_str(),
        ledger.selection_rule.as_str(),
        &ledger.candidates,
        ledger.selected_state_count,
        ledger.selected_kinetic_dimensions,
        ledger.selected_lag_steps,
        ledger.selected_mean_validation_vampe_score,
        ledger.minimum_required_fold_validation_vampe_score,
        ledger.within_lag_runner_up_score_margin,
        ledger.loso_seed_count,
        ledger.sensitivity_pass,
    ))
}

fn candidate_lag_diagnostics(
    raw_features: &[Vec<f64>],
    samples: &[SurveillanceSample],
    state_count: usize,
    kinetic_dimensions: usize,
    lag_steps: u64,
    config: &NativePathBEstimatorConfig,
) -> Result<(f64, f64, usize)> {
    let positions = (0..samples.len()).collect::<BTreeSet<_>>();
    let base_positions = surveillance_position_pairs_for_lag(samples, lag_steps)?;
    let embedding = fit_reversible_tica(
        raw_features,
        &positions,
        &base_positions,
        kinetic_dimensions,
        config.covariance_regularization,
    )?;
    let projected = raw_features
        .iter()
        .map(|row| embedding.project(row))
        .collect::<Result<Vec<_>>>()?;
    let (assignments, _) =
        deterministic_kmeans(&projected, state_count, config.maximum_iterations)?;
    let lag2 = lag_steps
        .checked_mul(2)
        .context("candidate diagnostic second-lag overflow")?;
    let lag3 = lag_steps
        .checked_mul(3)
        .context("candidate diagnostic third-lag overflow")?;
    let all_pairs1 = surveillance_transition_pairs_for_lag(samples, &assignments, lag_steps)?;
    let all_pairs2 = surveillance_transition_pairs_for_lag(samples, &assignments, lag2)?;
    let all_pairs3 = surveillance_transition_pairs_for_lag(samples, &assignments, lag3)?;
    let candidate_state_ids = (0..state_count)
        .map(|state| format!("candidate_state_{state:03}"))
        .collect::<Vec<_>>();
    let lane_state_support =
        build_lane_global_state_support(&assignments, samples, &candidate_state_ids, config)?;
    let mut worst_relative_spread = 0.0_f64;
    let mut worst_ck_residual = 0.0_f64;
    let mut minimum_seed_support = usize::MAX;
    for lane in [
        ControlLane::NoProbe,
        ControlLane::HypothesisFragment,
        ControlLane::OrthogonalChemicalControl,
    ] {
        let pairs1 = transition_pairs_for_lane(&all_pairs1, samples, lane)?;
        let pairs2 = transition_pairs_for_lane(&all_pairs2, samples, lane)?;
        let pairs3 = transition_pairs_for_lane(&all_pairs3, samples, lane)?;
        let active_global_states =
            active_global_state_indices(lane, &candidate_state_ids, &lane_state_support)?;
        let pairs1 = project_transition_pairs_to_active_subspace(&pairs1, &active_global_states)?;
        let pairs2 = project_transition_pairs_to_active_subspace(&pairs2, &active_global_states)?;
        let pairs3 = project_transition_pairs_to_active_subspace(&pairs3, &active_global_states)?;
        if [pairs1.len(), pairs2.len(), pairs3.len()]
            .into_iter()
            .any(|count| count < config.minimum_transition_pairs)
        {
            bail!("candidate lacks per-lane transition support at lag, 2lag or 3lag");
        }
        let (p1, pi1) = reversible_model_from_pairs(
            &pairs1,
            active_global_states.len(),
            config.reversible_pseudocount,
        )?;
        let (p2, pi2) = reversible_model_from_pairs(
            &pairs2,
            active_global_states.len(),
            config.reversible_pseudocount,
        )?;
        let (p3, pi3) = reversible_model_from_pairs(
            &pairs3,
            active_global_states.len(),
            config.reversible_pseudocount,
        )?;
        if active_global_states.len() > 1 {
            let estimates = [
                implied_timescale(&p1, &pi1, lag_steps, pairs1.len(), lane)?,
                implied_timescale(&p2, &pi2, lag2, pairs2.len(), lane)?,
                implied_timescale(&p3, &pi3, lag3, pairs3.len(), lane)?,
            ];
            let timescales = estimates
                .iter()
                .map(|estimate| {
                    estimate
                        .implied_timescale_steps
                        .context("candidate implied timescale is unresolved")
                })
                .collect::<Result<Vec<_>>>()?;
            let minimum = timescales.iter().copied().fold(f64::INFINITY, f64::min);
            let maximum = timescales.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mean = timescales.iter().sum::<f64>() / timescales.len() as f64;
            worst_relative_spread =
                worst_relative_spread.max((maximum - minimum) / mean.max(f64::EPSILON));
        }
        let p1_squared = multiply_square_matrix(&p1)?;
        let p1_cubed = matrix_multiply(&p1_squared, &p1)?;
        let mut lane_ck = 0.0_f64;
        for i in 0..active_global_states.len() {
            for j in 0..active_global_states.len() {
                lane_ck = lane_ck
                    .max((p2[i][j] - p1_squared[i][j]).abs())
                    .max((p3[i][j] - p1_cubed[i][j]).abs());
            }
        }
        worst_ck_residual = worst_ck_residual.max(lane_ck);
        let mut support = vec![BTreeSet::new(); state_count];
        for (assignment, sample) in assignments.iter().zip(samples) {
            if sample.control_lane == lane {
                support[*assignment].insert(sample.seed);
            }
        }
        minimum_seed_support = minimum_seed_support.min(
            active_global_states
                .iter()
                .map(|global| support[*global].len())
                .min()
                .unwrap_or(0),
        );
    }
    Ok((
        worst_relative_spread,
        worst_ck_residual,
        minimum_seed_support,
    ))
}

fn select_kinetic_model(
    samples: &[SurveillanceSample],
    config: &NativePathBEstimatorConfig,
) -> Result<(
    KineticModelSelectionLedger,
    Vec<Vec<f64>>,
    Vec<usize>,
    Vec<Vec<f64>>,
)> {
    let raw_features = samples
        .iter()
        .map(|sample| sample.features.clone())
        .collect::<Vec<_>>();
    let seeds = samples
        .iter()
        .map(|sample| sample.seed)
        .collect::<BTreeSet<_>>();
    if seeds.len() < config.minimum_independent_seeds {
        bail!("LOSO kinetic selection has insufficient independent seeds");
    }
    for lane in [
        ControlLane::NoProbe,
        ControlLane::HypothesisFragment,
        ControlLane::OrthogonalChemicalControl,
    ] {
        let lane_seeds = samples
            .iter()
            .filter_map(|sample| (sample.control_lane == lane).then_some(sample.seed))
            .collect::<BTreeSet<_>>();
        if lane_seeds != seeds {
            bail!("LOSO kinetic selection requires the same evaluable seed set in every lane");
        }
    }
    let feature_schemas = samples
        .iter()
        .map(|sample| sample.feature_schema_sha256.clone())
        .collect::<BTreeSet<_>>();
    if feature_schemas.len() != 1 {
        bail!("kinetic selection mixes surveillance feature schemas");
    }
    let feature_schema_sha256 = feature_schemas.into_iter().next().unwrap();
    let surveillance_population_sha256 = canonical_sha256(
        &samples
            .iter()
            .map(|sample| {
                (
                    sample.ledger_sha256.as_str(),
                    sample.run_id.as_str(),
                    sample.replica_id.as_str(),
                    sample.control_lane,
                    sample.evaluable_segment_id,
                    sample.seed,
                    sample.stream_id,
                    sample.step_idx,
                    sample.source_row_sha256.as_str(),
                    sample.cavity_volume_a3,
                    sample.binary_label,
                    &sample.ghost_event,
                    &sample.features,
                    &sample.detected_ghost_lineage_sha256,
                )
            })
            .collect::<Vec<_>>(),
    )?;
    let mut candidates = Vec::new();
    for &lag_steps in &config.candidate_lag_steps {
        let position_pairs = surveillance_position_pairs_for_lag(samples, lag_steps)?;
        for &kinetic_dimensions in &config.candidate_kinetic_dimensions {
            for &state_count in &config.candidate_state_counts {
                let mut fold_seed = Vec::new();
                let mut fold_validation_vampe_score = Vec::new();
                let mut reasons = Vec::new();
                for &held_seed in &seeds {
                    let train_positions = samples
                        .iter()
                        .enumerate()
                        .filter_map(|(position, sample)| {
                            (sample.seed != held_seed).then_some(position)
                        })
                        .collect::<BTreeSet<_>>();
                    let train_pairs = position_pairs
                        .iter()
                        .filter(|(seed, _, _)| *seed != held_seed)
                        .copied()
                        .collect::<Vec<_>>();
                    let validation_pairs = position_pairs
                        .iter()
                        .filter(|(seed, _, _)| *seed == held_seed)
                        .copied()
                        .collect::<Vec<_>>();
                    let fold = (|| -> Result<f64> {
                        if train_positions.len() < state_count
                            || train_pairs.len() < state_count.max(2)
                            || validation_pairs.len() < state_count.max(2)
                        {
                            bail!("insufficient train/validation population");
                        }
                        let embedding = fit_reversible_tica(
                            &raw_features,
                            &train_positions,
                            &position_pairs,
                            kinetic_dimensions,
                            config.covariance_regularization,
                        )?;
                        let projected = raw_features
                            .iter()
                            .map(|row| embedding.project(row))
                            .collect::<Result<Vec<_>>>()?;
                        let train_features = train_positions
                            .iter()
                            .map(|position| projected[*position].clone())
                            .collect::<Vec<_>>();
                        let (_, centroids) = deterministic_kmeans(
                            &train_features,
                            state_count,
                            config.maximum_iterations,
                        )?;
                        let assignments = nearest_centroid_assignments(&projected, &centroids)?;
                        let candidate_state_ids = (0..state_count)
                            .map(|state| format!("candidate_state_{state:03}"))
                            .collect::<Vec<_>>();
                        let train_samples = train_positions
                            .iter()
                            .map(|position| samples[*position].clone())
                            .collect::<Vec<_>>();
                        let train_assignments = train_positions
                            .iter()
                            .map(|position| assignments[*position])
                            .collect::<Vec<_>>();
                        let train_support = build_lane_global_state_support(
                            &train_assignments,
                            &train_samples,
                            &candidate_state_ids,
                            config,
                        )?;
                        let mut lane_scores = Vec::with_capacity(3);
                        for lane in [
                            ControlLane::NoProbe,
                            ControlLane::HypothesisFragment,
                            ControlLane::OrthogonalChemicalControl,
                        ] {
                            let lane_train = train_pairs
                                .iter()
                                .copied()
                                .filter(|(_, from, to)| {
                                    samples[*from].control_lane == lane
                                        && samples[*to].control_lane == lane
                                })
                                .collect::<Vec<_>>();
                            let lane_validation = validation_pairs
                                .iter()
                                .copied()
                                .filter(|(_, from, to)| {
                                    samples[*from].control_lane == lane
                                        && samples[*to].control_lane == lane
                                })
                                .collect::<Vec<_>>();
                            let active_global_states = active_global_state_indices(
                                lane,
                                &candidate_state_ids,
                                &train_support,
                            )?;
                            let local_by_global = active_global_states
                                .iter()
                                .enumerate()
                                .map(|(local, global)| (*global, local))
                                .collect::<BTreeMap<_, _>>();
                            let local_assignments = assignments
                                .iter()
                                .map(|global| {
                                    local_by_global.get(global).copied().unwrap_or(usize::MAX)
                                })
                                .collect::<Vec<_>>();
                            let lane_train = lane_train
                                .into_iter()
                                .filter(|(_, from, to)| {
                                    local_by_global.contains_key(&assignments[*from])
                                        && local_by_global.contains_key(&assignments[*to])
                                })
                                .collect::<Vec<_>>();
                            let lane_validation = lane_validation
                                .into_iter()
                                .filter(|(_, from, to)| {
                                    local_by_global.contains_key(&assignments[*from])
                                        && local_by_global.contains_key(&assignments[*to])
                                })
                                .collect::<Vec<_>>();
                            let active_state_count = active_global_states.len();
                            if lane_train.len() < active_state_count.max(2)
                                || lane_validation.len() < active_state_count.max(2)
                            {
                                bail!(
                                    "LOSO VAMP-E lacks train/validation pairs in a physical lane"
                                );
                            }
                            let train_model = fit_train_vampe_model(
                                &local_assignments,
                                &lane_train,
                                active_state_count,
                                config.covariance_regularization,
                            )?;
                            lane_scores.push(held_out_vampe_score(
                                &train_model,
                                &local_assignments,
                                &lane_validation,
                                active_state_count,
                            )?);
                        }
                        Ok(lane_scores.iter().sum::<f64>() / lane_scores.len() as f64)
                    })();
                    match fold {
                        Ok(score) => {
                            fold_seed.push(held_seed);
                            fold_validation_vampe_score.push(score);
                        }
                        Err(error) => reasons.push(format!("seed_{held_seed}:{error}")),
                    }
                }
                let complete = fold_validation_vampe_score.len() == seeds.len();
                let mean = if fold_validation_vampe_score.is_empty() {
                    0.0
                } else {
                    fold_validation_vampe_score.iter().sum::<f64>()
                        / fold_validation_vampe_score.len() as f64
                };
                let minimum = fold_validation_vampe_score
                    .iter()
                    .copied()
                    .fold(f64::INFINITY, f64::min);
                if mean < config.minimum_loso_validation_vampe_score {
                    reasons.push(format!(
                        "mean_validation_vampe={mean} below {}",
                        config.minimum_loso_validation_vampe_score
                    ));
                }
                if !every_loso_fold_meets_threshold(
                    &fold_validation_vampe_score,
                    config.minimum_loso_validation_vampe_score,
                ) {
                    reasons.push(format!(
                        "minimum_fold_validation_vampe={minimum} below {}",
                        config.minimum_loso_validation_vampe_score
                    ));
                }
                candidates.push(KineticCandidateScore {
                    state_count,
                    lag_steps,
                    kinetic_dimensions,
                    fold_seed,
                    fold_validation_vampe_score,
                    mean_validation_vampe_score: mean,
                    minimum_fold_validation_vampe_score: if minimum.is_finite() {
                        minimum
                    } else {
                        0.0
                    },
                    within_lag_winner: false,
                    implied_timescale_relative_spread: f64::MAX,
                    chapman_kolmogorov_max_abs_residual: f64::MAX,
                    minimum_state_seed_support: 0,
                    lag_diagnostics_pass: false,
                    eligible: complete && reasons.is_empty(),
                    ineligible_reasons: reasons,
                });
            }
        }
    }
    candidates.sort_by(|left, right| {
        left.lag_steps
            .cmp(&right.lag_steps)
            .then_with(|| left.kinetic_dimensions.cmp(&right.kinetic_dimensions))
            .then_with(|| left.state_count.cmp(&right.state_count))
    });

    // Representation/state selection is performed independently within each
    // lag. Only its winner enters lag diagnostics; VAMP-E values are never
    // compared across lag times.
    let mut margin_by_lag = BTreeMap::new();
    for &lag_steps in &config.candidate_lag_steps {
        let mut indices = candidates
            .iter()
            .enumerate()
            .filter_map(|(index, candidate)| {
                (candidate.lag_steps == lag_steps && candidate.eligible).then_some(index)
            })
            .collect::<Vec<_>>();
        indices.sort_by(|left, right| {
            candidates[*right]
                .mean_validation_vampe_score
                .total_cmp(&candidates[*left].mean_validation_vampe_score)
                .then_with(|| {
                    candidates[*left]
                        .kinetic_dimensions
                        .cmp(&candidates[*right].kinetic_dimensions)
                })
                .then_with(|| {
                    candidates[*left]
                        .state_count
                        .cmp(&candidates[*right].state_count)
                })
        });
        let Some(winner_index) = indices.first().copied() else {
            continue;
        };
        candidates[winner_index].within_lag_winner = true;
        let winner_score = candidates[winner_index].mean_validation_vampe_score;
        let margin = indices
            .get(1)
            .map(|index| winner_score - candidates[*index].mean_validation_vampe_score)
            .unwrap_or_else(|| winner_score.abs());
        margin_by_lag.insert(lag_steps, margin);
        match candidate_lag_diagnostics(
            &raw_features,
            samples,
            candidates[winner_index].state_count,
            candidates[winner_index].kinetic_dimensions,
            lag_steps,
            config,
        ) {
            Ok((its_spread, ck_residual, seed_support)) => {
                let diagnostic_pass = its_spread
                    <= config.maximum_implied_timescale_relative_spread
                    && ck_residual <= config.maximum_chapman_kolmogorov_residual
                    && seed_support >= config.minimum_state_seed_support
                    && margin + f64::EPSILON >= config.minimum_model_selection_margin;
                let candidate = &mut candidates[winner_index];
                candidate.implied_timescale_relative_spread = its_spread;
                candidate.chapman_kolmogorov_max_abs_residual = ck_residual;
                candidate.minimum_state_seed_support = seed_support;
                candidate.lag_diagnostics_pass = diagnostic_pass;
                if !diagnostic_pass {
                    candidate.eligible = false;
                    candidate
                        .ineligible_reasons
                        .push("within_lag_winner_failed_its_ck_seed_support_or_sensitivity".into());
                }
            }
            Err(error) => {
                let candidate = &mut candidates[winner_index];
                candidate.eligible = false;
                candidate
                    .ineligible_reasons
                    .push(format!("lag_diagnostics:{error}"));
            }
        }
    }
    let selected_index = candidates
        .iter()
        .enumerate()
        .filter(|(_, candidate)| {
            candidate.within_lag_winner && candidate.eligible && candidate.lag_diagnostics_pass
        })
        .min_by_key(|(_, candidate)| candidate.lag_steps)
        .map(|(index, _)| index)
        .context("no within-lag VAMP-E winner passed ITS/CK/seed-support diagnostics")?;
    let selected_state_count = candidates[selected_index].state_count;
    let selected_kinetic_dimensions = candidates[selected_index].kinetic_dimensions;
    let selected_lag_steps = candidates[selected_index].lag_steps;
    let selected_mean_validation_vampe_score =
        candidates[selected_index].mean_validation_vampe_score;
    let margin = margin_by_lag[&selected_lag_steps];
    let position_pairs = surveillance_position_pairs_for_lag(samples, selected_lag_steps)?;
    let all_positions = (0..samples.len()).collect::<BTreeSet<_>>();
    let embedding = fit_reversible_tica(
        &raw_features,
        &all_positions,
        &position_pairs,
        selected_kinetic_dimensions,
        config.covariance_regularization,
    )?;
    let projected = raw_features
        .iter()
        .map(|row| embedding.project(row))
        .collect::<Result<Vec<_>>>()?;
    let (assignments, centroids) =
        deterministic_kmeans(&projected, selected_state_count, config.maximum_iterations)?;
    let mut ledger = KineticModelSelectionLedger {
        ledger_sha256: String::new(),
        feature_schema_sha256,
        surveillance_population_sha256,
        selection_rule:
            "within_each_lag_maximize_complete_train_fitted_lane_active_subspace_validation_vampe_with_every_fold_gate_then_select_smallest_lag_passing_per_lane_active_subspace_its_ck_seed_support".to_string(),
        candidates,
        selected_state_count,
        selected_kinetic_dimensions,
        selected_lag_steps,
        selected_mean_validation_vampe_score,
        minimum_required_fold_validation_vampe_score: config
            .minimum_loso_validation_vampe_score,
        within_lag_runner_up_score_margin: margin,
        loso_seed_count: seeds.len(),
        sensitivity_pass: true,
    };
    ledger.ledger_sha256 = kinetic_selection_ledger_hash(&ledger)?;
    Ok((ledger, projected, assignments, centroids))
}

fn every_loso_fold_meets_threshold(scores: &[f64], minimum: f64) -> bool {
    !scores.is_empty()
        && minimum.is_finite()
        && scores
            .iter()
            .all(|score| score.is_finite() && *score >= minimum)
}

fn exact_transition_pairs_for_lag(
    edges: &[NativePathBExactEdgeObservation],
    position_by_lineage: &BTreeMap<String, usize>,
    ordered: &[(
        usize,
        &crate::ghost_phase_lattice::GhostPhaseLatticeNode,
        &GhostByteLineage,
    )],
    assignments: &[usize],
    lag_steps: u64,
) -> Result<Vec<SeedTransitionPair>> {
    let mut pairs = Vec::new();
    for edge in edges {
        edge.verify()?;
        let from_position = *position_by_lineage
            .get(&edge.from_ghost_lineage_sha256)
            .context("exact accepted edge from endpoint is absent from Path-B population")?;
        let to_position = *position_by_lineage
            .get(&edge.to_ghost_lineage_sha256)
            .context("exact accepted edge to endpoint is absent from Path-B population")?;
        let from = ordered[from_position].2;
        let to = ordered[to_position].2;
        if from.key.step_idx != edge.from_step_idx
            || to.key.step_idx != edge.to_step_idx
            || from.protocol_phase != u16::from(edge.from_protocol_phase)
            || to.protocol_phase != u16::from(edge.to_protocol_phase)
        {
            bail!("accepted edge fields disagree with exact Ghost endpoint lineage");
        }
        // Cross-stream geometric/causal edges remain connectivity evidence but
        // are not time evolution of a single stochastic trajectory.
        if edge.temporal_delta_steps == lag_steps
            && from.replica_seed == to.replica_seed
            && from.key.stream_id == to.key.stream_id
        {
            pairs.push(SeedTransitionPair {
                seed: from.replica_seed,
                from: assignments[from_position],
                to: assignments[to_position],
                from_position,
                to_position,
            });
        }
    }
    Ok(pairs)
}

fn build_lane_global_state_support(
    assignments: &[usize],
    samples: &[SurveillanceSample],
    state_ids: &[String],
    config: &NativePathBEstimatorConfig,
) -> Result<Vec<LaneGlobalStateSupportEvidence>> {
    if assignments.len() != samples.len() || state_ids.is_empty() {
        bail!("lane-state support evidence population is invalid");
    }
    let mut output = Vec::with_capacity(3 * state_ids.len());
    let mut globally_observed = BTreeSet::new();
    for lane in [
        ControlLane::NoProbe,
        ControlLane::HypothesisFragment,
        ControlLane::OrthogonalChemicalControl,
    ] {
        let exposure = samples
            .iter()
            .filter(|sample| sample.control_lane == lane)
            .count() as u64;
        if exposure == 0 {
            bail!("lane-state support evidence has no scheduled exposure");
        }
        for (state, state_id) in state_ids.iter().enumerate() {
            let matching = assignments
                .iter()
                .zip(samples)
                .filter(|(assignment, sample)| sample.control_lane == lane && **assignment == state)
                .collect::<Vec<_>>();
            let hard_assignment_rows = matching.len() as u64;
            let independent_seed_support = matching
                .iter()
                .map(|(_, sample)| sample.seed)
                .collect::<BTreeSet<_>>()
                .len();
            let status = if hard_assignment_rows == 0 {
                LaneGlobalStateSupportStatus::NotObserved
            } else if hard_assignment_rows >= config.minimum_state_effective_count
                && independent_seed_support >= config.minimum_state_seed_support
            {
                globally_observed.insert(state);
                LaneGlobalStateSupportStatus::Observed
            } else {
                LaneGlobalStateSupportStatus::BelowSupport
            };
            output.push(LaneGlobalStateSupportEvidence {
                control_lane: lane,
                state_id: state_id.clone(),
                status,
                scheduled_exposure_rows: exposure,
                hard_assignment_rows,
                independent_seed_support,
                minimum_required_assignment_rows: config.minimum_state_effective_count,
                minimum_required_independent_seeds: config.minimum_state_seed_support,
            });
        }
    }
    if globally_observed.len() != state_ids.len() {
        let unsupported = state_ids
            .iter()
            .enumerate()
            .filter_map(|(index, state)| {
                (!globally_observed.contains(&index)).then_some(state.as_str())
            })
            .collect::<Vec<_>>();
        bail!(
            "global structural states lack adequate support in every matched lane: {}",
            unsupported.join(",")
        );
    }
    Ok(output)
}

fn active_global_state_indices(
    lane: ControlLane,
    state_ids: &[String],
    support: &[LaneGlobalStateSupportEvidence],
) -> Result<Vec<usize>> {
    let active = state_ids
        .iter()
        .enumerate()
        .filter_map(|(index, state_id)| {
            support
                .iter()
                .find(|row| row.control_lane == lane && row.state_id == *state_id)
                .filter(|row| row.status == LaneGlobalStateSupportStatus::Observed)
                .map(|_| index)
        })
        .collect::<Vec<_>>();
    if active.is_empty() {
        bail!("lane active kinetic subspace has no adequately supported state");
    }
    Ok(active)
}

fn project_transition_pairs_to_active_subspace(
    pairs: &[SeedTransitionPair],
    active_global_states: &[usize],
) -> Result<Vec<SeedTransitionPair>> {
    let local_by_global = active_global_states
        .iter()
        .enumerate()
        .map(|(local, global)| (*global, local))
        .collect::<BTreeMap<_, _>>();
    let mut projected = Vec::new();
    for pair in pairs {
        match (
            local_by_global.get(&pair.from),
            local_by_global.get(&pair.to),
        ) {
            (Some(from), Some(to)) => projected.push(SeedTransitionPair {
                seed: pair.seed,
                from: *from,
                to: *to,
                from_position: pair.from_position,
                to_position: pair.to_position,
            }),
            // A transition touching an absent/below-support global state is
            // evidence for that status, not kinetic evidence. It is excluded
            // rather than pseudocount-filled into the lane operator.
            _ => {}
        }
    }
    Ok(projected)
}

fn fit_lane_operator(
    control_lane: ControlLane,
    global_lag_pairs: &[SeedTransitionPair],
    global_double_lag_pairs: &[SeedTransitionPair],
    active_global_states: Vec<usize>,
    config: &NativePathBEstimatorConfig,
) -> Result<FittedLaneOperator> {
    let lag_pairs =
        project_transition_pairs_to_active_subspace(global_lag_pairs, &active_global_states)?;
    let double_lag_pairs = project_transition_pairs_to_active_subspace(
        global_double_lag_pairs,
        &active_global_states,
    )?;
    if lag_pairs.len() < config.minimum_transition_pairs
        || double_lag_pairs.len() < config.minimum_transition_pairs
    {
        bail!(
            "native Path-B lane {:?} lacks active-subspace exact-lag/second-lag transition support",
            control_lane
        );
    }
    let (matrix, stationary) = reversible_model_from_pairs(
        &lag_pairs,
        active_global_states.len(),
        config.reversible_pseudocount,
    )?;
    Ok(FittedLaneOperator {
        control_lane,
        active_global_states,
        matrix,
        stationary,
        lag_pairs,
        double_lag_pairs,
    })
}

fn reversible_model_from_pairs(
    pairs: &[SeedTransitionPair],
    state_count: usize,
    pseudocount: f64,
) -> Result<(Vec<Vec<f64>>, Vec<f64>)> {
    if pairs.is_empty() || state_count == 0 || !pseudocount.is_finite() || pseudocount <= 0.0 {
        bail!("cannot estimate reversible transition model from empty/invalid inputs");
    }
    let mut directed = vec![vec![0.0f64; state_count]; state_count];
    for pair in pairs {
        if pair.from >= state_count || pair.to >= state_count {
            bail!("transition pair state exceeds declared state count");
        }
        directed[pair.from][pair.to] += 1.0;
    }
    let mut reversible = vec![vec![0.0f64; state_count]; state_count];
    for i in 0..state_count {
        for j in 0..state_count {
            reversible[i][j] = if i == j {
                directed[i][i] + pseudocount
            } else {
                0.5 * (directed[i][j] + directed[j][i]) + pseudocount
            };
        }
    }
    let row_sums = reversible
        .iter()
        .map(|row| row.iter().sum::<f64>())
        .collect::<Vec<_>>();
    let total = row_sums.iter().sum::<f64>();
    if !total.is_finite() || total <= 0.0 {
        bail!("reversible transition counts are invalid");
    }
    let matrix = reversible
        .iter()
        .zip(&row_sums)
        .map(|(row, sum)| row.iter().map(|value| value / sum).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let stationary = row_sums.iter().map(|sum| sum / total).collect();
    Ok((matrix, stationary))
}

fn transition_pairs_for_lane(
    pairs: &[SeedTransitionPair],
    samples: &[SurveillanceSample],
    lane: ControlLane,
) -> Result<Vec<SeedTransitionPair>> {
    pairs
        .iter()
        .filter_map(|pair| {
            let from = samples.get(pair.from_position);
            let to = samples.get(pair.to_position);
            match (from, to) {
                (Some(from), Some(to)) if from.control_lane == to.control_lane => {
                    (from.control_lane == lane).then_some(Ok(*pair))
                }
                (Some(_), Some(_)) => Some(Err(anyhow::anyhow!(
                    "transition pair crosses physical matched-control lanes"
                ))),
                _ => Some(Err(anyhow::anyhow!(
                    "transition pair references an absent surveillance sample"
                ))),
            }
        })
        .collect()
}

fn build_markov_diagnostics(
    assignments: &[usize],
    surveillance_samples: &[SurveillanceSample],
    lane_operators: &[FittedLaneOperator],
    lane_global_state_support: &[LaneGlobalStateSupportEvidence],
    state_ids: &[String],
    config: &NativePathBEstimatorConfig,
    kinetic_model_selection: KineticModelSelectionLedger,
    surveillance_missingness: SurveillanceMissingnessSummary,
) -> Result<MarkovDiagnostics> {
    let mut lane_diagnostics = Vec::with_capacity(lane_operators.len());
    for operator in lane_operators {
        lane_diagnostics.push(build_lane_markov_diagnostics(
            operator.control_lane,
            kinetic_model_selection.selected_lag_steps,
            assignments,
            surveillance_samples,
            &operator.lag_pairs,
            &operator.double_lag_pairs,
            &operator.matrix,
            &operator.stationary,
            &operator.active_global_states,
            state_ids,
            config,
        )?);
    }
    lane_diagnostics.sort_by_key(|row| row.control_lane);
    for evidence in lane_global_state_support {
        let diagnostic = lane_diagnostics
            .iter()
            .find(|row| row.control_lane == evidence.control_lane)
            .context("lane-state support references a missing diagnostic lane")?;
        if diagnostic.state_effective_counts[&evidence.state_id] != evidence.hard_assignment_rows
            || diagnostic.state_seed_support[&evidence.state_id]
                != evidence.independent_seed_support
        {
            bail!("lane-state support evidence differs from fitted diagnostics");
        }
    }
    let no_probe = lane_diagnostics
        .iter()
        .find(|row| row.control_lane == ControlLane::NoProbe)
        .context("no-probe lane diagnostics are absent")?;
    let state_effective_counts = state_ids
        .iter()
        .map(|state| {
            (
                state.clone(),
                lane_diagnostics
                    .iter()
                    .map(|row| row.state_effective_counts[state])
                    .max()
                    .unwrap_or(0),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let state_seed_support = state_ids
        .iter()
        .map(|state| {
            (
                state.clone(),
                lane_diagnostics
                    .iter()
                    .map(|row| row.state_seed_support[state])
                    .max()
                    .unwrap_or(0),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let transition_uncertainty = lane_diagnostics
        .iter()
        .flat_map(|row| row.transition_uncertainty.clone())
        .collect::<Vec<_>>();
    let mut inconclusive_reasons = lane_diagnostics
        .iter()
        .flat_map(|row| {
            row.ineligible_reasons
                .iter()
                .map(move |reason| format!("lane_{:?}:{reason}", row.control_lane))
        })
        .collect::<Vec<_>>();
    if !kinetic_model_selection.sensitivity_pass {
        inconclusive_reasons.push("kinetic_model_selection_sensitivity_not_separated".to_string());
    }
    if surveillance_missingness.not_evaluable_rows > 0 {
        inconclusive_reasons
            .push("scheduled_structural_state_assignment_missingness_present".to_string());
    }
    let status = if inconclusive_reasons.is_empty() {
        MarkovDiagnosticStatus::Pass
    } else {
        MarkovDiagnosticStatus::ComputationallyInconclusive
    };
    Ok(MarkovDiagnostics {
        status,
        inconclusive_reasons,
        detailed_balance_max_abs_residual: lane_diagnostics
            .iter()
            .map(|row| row.detailed_balance_max_abs_residual)
            .fold(0.0_f64, f64::max),
        constrained_reversible_numerical_residual: lane_diagnostics
            .iter()
            .map(|row| row.constrained_reversible_numerical_residual)
            .fold(0.0_f64, f64::max),
        chapman_kolmogorov_max_abs_residual: lane_diagnostics
            .iter()
            .map(|row| row.chapman_kolmogorov_max_abs_residual)
            .fold(0.0_f64, f64::max),
        minimum_state_effective_count: state_effective_counts.values().copied().min().unwrap_or(0),
        state_effective_counts,
        implied_timescales: no_probe.implied_timescales.clone(),
        implied_timescale_relative_spread: lane_diagnostics
            .iter()
            .map(|row| row.implied_timescale_relative_spread)
            .fold(0.0_f64, f64::max),
        state_seed_support,
        kinetic_model_selection,
        surveillance_missingness,
        independent_seed_count: lane_diagnostics
            .iter()
            .map(|row| row.independent_seed_count)
            .min()
            .unwrap_or(0),
        bootstrap_replicates: if transition_uncertainty.is_empty() {
            0
        } else {
            config.bootstrap_replicates
        },
        bootstrap_block_length: config.bootstrap_block_length,
        transition_uncertainty,
        committor_uncertainty: Vec::new(),
        committor_status: "UNAVAILABLE_PENDING_HASH_BOUND_OPEN_BASIN".to_string(),
        lane_diagnostics,
    })
}

fn build_lane_markov_diagnostics(
    control_lane: ControlLane,
    selected_lag_steps: u64,
    assignments: &[usize],
    surveillance_samples: &[SurveillanceSample],
    lag_pairs: &[SeedTransitionPair],
    double_pairs: &[SeedTransitionPair],
    matrix: &[Vec<f64>],
    stationary: &[f64],
    active_global_states: &[usize],
    state_ids: &[String],
    config: &NativePathBEstimatorConfig,
) -> Result<LaneMarkovDiagnostics> {
    if assignments.len() != surveillance_samples.len() {
        bail!("Markov diagnostics assignments differ from complete surveillance population");
    }
    let k = active_global_states.len();
    let active_state_ids = active_global_states
        .iter()
        .map(|global| {
            state_ids
                .get(*global)
                .cloned()
                .context("active lane state exceeds shared global state population")
        })
        .collect::<Result<Vec<_>>>()?;
    let second_lag = selected_lag_steps
        .checked_mul(2)
        .context("Path-B second diagnostic lag overflow")?;
    let (double_matrix, double_stationary) = if double_pairs.is_empty() {
        (None, None)
    } else {
        let (p, pi) = reversible_model_from_pairs(&double_pairs, k, config.reversible_pseudocount)?;
        (Some(p), Some(pi))
    };

    let constrained_reversible_numerical_residual = (0..k)
        .flat_map(|i| {
            (0..k).map(move |j| (stationary[i] * matrix[i][j] - stationary[j] * matrix[j][i]).abs())
        })
        .fold(0.0f64, f64::max);
    let matrix_squared = multiply_square_matrix(matrix)?;
    let chapman_kolmogorov_max_abs_residual = double_matrix
        .as_ref()
        .map(|observed| {
            (0..k)
                .flat_map(|i| {
                    let matrix_squared = &matrix_squared;
                    (0..k).map(move |j| (observed[i][j] - matrix_squared[i][j]).abs())
                })
                .fold(0.0f64, f64::max)
        })
        .unwrap_or(1.0);

    let mut state_effective_counts = state_ids
        .iter()
        .cloned()
        .map(|id| (id, 0u64))
        .collect::<BTreeMap<_, _>>();
    for (assignment, sample) in assignments.iter().zip(surveillance_samples) {
        if sample.control_lane != control_lane {
            continue;
        }
        state_effective_counts
            .entry(state_ids[*assignment].clone())
            .and_modify(|count| *count += 1);
    }
    let minimum_state_effective_count = active_state_ids
        .iter()
        .map(|state| state_effective_counts[state])
        .min()
        .unwrap_or(0);

    let mut implied_timescales = vec![implied_timescale(
        matrix,
        stationary,
        selected_lag_steps,
        lag_pairs.len(),
        control_lane,
    )?];
    if let (Some(p), Some(pi)) = (&double_matrix, &double_stationary) {
        implied_timescales.push(implied_timescale(
            p,
            pi,
            second_lag,
            double_pairs.len(),
            control_lane,
        )?);
    }
    let finite_timescales = implied_timescales
        .iter()
        .filter_map(|estimate| estimate.implied_timescale_steps)
        .collect::<Vec<_>>();
    let implied_timescale_relative_spread = if k == 1 {
        0.0
    } else if finite_timescales.len() < 2 {
        f64::MAX
    } else {
        let minimum = finite_timescales
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let maximum = finite_timescales
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let mean = finite_timescales.iter().sum::<f64>() / finite_timescales.len() as f64;
        (maximum - minimum) / mean.max(f64::EPSILON)
    };
    let mut seed_support_sets = state_ids
        .iter()
        .cloned()
        .map(|state| (state, BTreeSet::new()))
        .collect::<BTreeMap<_, _>>();
    for (assignment, sample) in assignments.iter().zip(surveillance_samples) {
        if sample.control_lane != control_lane {
            continue;
        }
        seed_support_sets
            .get_mut(&state_ids[*assignment])
            .unwrap()
            .insert(sample.seed);
    }
    let state_seed_support = seed_support_sets
        .into_iter()
        .map(|(state, seeds)| (state, seeds.len()))
        .collect::<BTreeMap<_, _>>();

    let seeds = lag_pairs
        .iter()
        .map(|pair| pair.seed)
        .collect::<BTreeSet<_>>();
    let independent_seed_count = seeds.len();
    let (transition_uncertainty, _committor_uncertainty, directed_count_reversibility) =
        if independent_seed_count >= config.minimum_independent_seeds {
            bootstrap_uncertainty(
                control_lane,
                lag_pairs,
                surveillance_samples,
                &active_state_ids,
                config,
            )?
        } else {
            (
                Vec::new(),
                Vec::new(),
                DirectedCountReversibilityDiagnostic {
                    statistic_id: "maximum_unconstrained_directed_flux_asymmetry_v1".to_string(),
                    observed_max_abs_flux_asymmetry: directed_count_flux_asymmetry(lag_pairs, k)?,
                    bootstrap_mean: None,
                    bootstrap_lower: None,
                    bootstrap_upper: None,
                    bootstrap_replicates: 0,
                    transition_pairs: lag_pairs.len() as u64,
                },
            )
        };
    let detailed_balance_max_abs_residual =
        directed_count_reversibility.observed_max_abs_flux_asymmetry;

    let mut inconclusive_reasons = Vec::new();
    if detailed_balance_max_abs_residual > config.maximum_detailed_balance_residual {
        inconclusive_reasons.push(format!(
            "unconstrained_directed_flux_asymmetry={} exceeds {}",
            detailed_balance_max_abs_residual, config.maximum_detailed_balance_residual
        ));
    }
    if let Some(upper) = directed_count_reversibility.bootstrap_upper {
        if upper > config.maximum_detailed_balance_residual {
            inconclusive_reasons.push(format!(
                "unconstrained_directed_flux_asymmetry_bootstrap_upper={} exceeds {}",
                upper, config.maximum_detailed_balance_residual
            ));
        }
    }
    if double_pairs.len() < config.minimum_transition_pairs {
        inconclusive_reasons.push(format!(
            "second_lag_transition_pairs={} below {}",
            double_pairs.len(),
            config.minimum_transition_pairs
        ));
    }
    if chapman_kolmogorov_max_abs_residual > config.maximum_chapman_kolmogorov_residual {
        inconclusive_reasons.push(format!(
            "chapman_kolmogorov_residual={} exceeds {}",
            chapman_kolmogorov_max_abs_residual, config.maximum_chapman_kolmogorov_residual
        ));
    }
    if minimum_state_effective_count < config.minimum_state_effective_count {
        inconclusive_reasons.push(format!(
            "minimum_state_count={} below {}",
            minimum_state_effective_count, config.minimum_state_effective_count
        ));
    }
    if independent_seed_count < config.minimum_independent_seeds {
        inconclusive_reasons.push(format!(
            "independent_seed_count={} below {}",
            independent_seed_count, config.minimum_independent_seeds
        ));
    }
    if k > 1
        && (implied_timescales.len() < 2
            || implied_timescales
                .iter()
                .any(|estimate| estimate.implied_timescale_steps.is_none()))
    {
        inconclusive_reasons.push("two_lag_implied_timescale_not_resolved".to_string());
    }
    if implied_timescale_relative_spread > config.maximum_implied_timescale_relative_spread {
        inconclusive_reasons.push(format!(
            "implied_timescale_relative_spread={} exceeds {}",
            implied_timescale_relative_spread, config.maximum_implied_timescale_relative_spread
        ));
    }
    if active_state_ids
        .iter()
        .map(|state| state_seed_support[state])
        .any(|support| support < config.minimum_state_seed_support)
    {
        inconclusive_reasons.push(format!(
            "state_seed_support below {}",
            config.minimum_state_seed_support
        ));
    }
    if transition_uncertainty.len() != k * k {
        inconclusive_reasons.push("seed_block_bootstrap_uncertainty_not_materialized".to_string());
    }
    if directed_count_reversibility.bootstrap_replicates != config.bootstrap_replicates {
        inconclusive_reasons
            .push("directed_count_reversibility_bootstrap_not_materialized".to_string());
    }
    Ok(LaneMarkovDiagnostics {
        control_lane,
        active_state_ids,
        detailed_balance_max_abs_residual,
        constrained_reversible_numerical_residual,
        directed_count_reversibility,
        chapman_kolmogorov_max_abs_residual,
        state_effective_counts,
        minimum_state_effective_count,
        implied_timescales,
        implied_timescale_relative_spread,
        state_seed_support,
        independent_seed_count,
        transition_pairs: lag_pairs.len() as u64,
        second_lag_transition_pairs: double_pairs.len() as u64,
        transition_uncertainty,
        diagnostic_pass: inconclusive_reasons.is_empty(),
        ineligible_reasons: inconclusive_reasons,
    })
}

fn multiply_square_matrix(matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    let n = matrix.len();
    if n == 0 || matrix.iter().any(|row| row.len() != n) {
        bail!("matrix multiplication requires a non-empty square matrix");
    }
    let mut output = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            output[i][j] = (0..n).map(|k| matrix[i][k] * matrix[k][j]).sum();
        }
    }
    Ok(output)
}

fn implied_timescale(
    matrix: &[Vec<f64>],
    stationary: &[f64],
    lag_steps: u64,
    transition_pairs: usize,
    control_lane: ControlLane,
) -> Result<ImpliedTimescaleEstimate> {
    let n = matrix.len();
    if stationary.len() != n || stationary.iter().any(|value| *value <= 0.0) {
        bail!("implied-timescale stationary distribution is invalid");
    }
    let mut symmetric = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            symmetric[i][j] = stationary[i].sqrt() * matrix[i][j] / stationary[j].sqrt();
        }
    }
    let mut eigenvalues = jacobi_symmetric_eigenvalues(symmetric)?;
    eigenvalues.sort_by(|a, b| b.abs().total_cmp(&a.abs()));
    let second = eigenvalues
        .into_iter()
        .find(|value| value.abs() < 1.0 - 1.0e-10)
        .unwrap_or(0.0)
        .abs()
        .clamp(0.0, 1.0);
    let implied = (second > 1.0e-12 && second < 1.0).then(|| -(lag_steps as f64) / second.ln());
    Ok(ImpliedTimescaleEstimate {
        control_lane,
        lag_steps,
        second_eigenvalue_magnitude: second,
        implied_timescale_steps: implied,
        transition_pairs: transition_pairs as u64,
    })
}

fn jacobi_symmetric_eigenvalues(mut matrix: Vec<Vec<f64>>) -> Result<Vec<f64>> {
    let n = matrix.len();
    if n == 0 || matrix.iter().any(|row| row.len() != n) {
        bail!("Jacobi eigensolver requires a square matrix");
    }
    for _ in 0..(100 * n * n).max(1) {
        let mut p = 0;
        let mut q = 0;
        let mut maximum = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                if matrix[i][j].abs() > maximum {
                    maximum = matrix[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if maximum < 1.0e-12 {
            return Ok((0..n).map(|index| matrix[index][index]).collect());
        }
        let angle = 0.5 * (2.0 * matrix[p][q]).atan2(matrix[q][q] - matrix[p][p]);
        let (sine, cosine) = angle.sin_cos();
        let app = matrix[p][p];
        let aqq = matrix[q][q];
        let apq = matrix[p][q];
        matrix[p][p] = cosine * cosine * app - 2.0 * sine * cosine * apq + sine * sine * aqq;
        matrix[q][q] = sine * sine * app + 2.0 * sine * cosine * apq + cosine * cosine * aqq;
        matrix[p][q] = 0.0;
        matrix[q][p] = 0.0;
        for r in 0..n {
            if r != p && r != q {
                let arp = matrix[r][p];
                let arq = matrix[r][q];
                matrix[r][p] = cosine * arp - sine * arq;
                matrix[p][r] = matrix[r][p];
                matrix[r][q] = sine * arp + cosine * arq;
                matrix[q][r] = matrix[r][q];
            }
        }
    }
    bail!("Jacobi eigensolver did not converge")
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct BootstrapSegmentKey {
    seed: u64,
    run_id: String,
    replica_id: String,
    ledger_sha256: String,
    stream_id: u16,
    evaluable_segment_id: u64,
}

fn transition_pair_segment_key(
    pair: &SeedTransitionPair,
    samples: &[SurveillanceSample],
) -> Result<BootstrapSegmentKey> {
    let from = samples
        .get(pair.from_position)
        .context("bootstrap transition from-position is absent")?;
    let to = samples
        .get(pair.to_position)
        .context("bootstrap transition to-position is absent")?;
    if pair.seed != from.seed
        || pair.seed != to.seed
        || from.run_id != to.run_id
        || from.replica_id != to.replica_id
        || from.ledger_sha256 != to.ledger_sha256
        || from.stream_id != to.stream_id
        || from.evaluable_segment_id != to.evaluable_segment_id
    {
        bail!("bootstrap transition crosses a run/stream/evaluable-segment boundary");
    }
    Ok(BootstrapSegmentKey {
        seed: pair.seed,
        run_id: from.run_id.clone(),
        replica_id: from.replica_id.clone(),
        ledger_sha256: from.ledger_sha256.clone(),
        stream_id: from.stream_id,
        evaluable_segment_id: from.evaluable_segment_id,
    })
}

/// Resample contiguous transition blocks without circular wrap. Returning
/// explicit blocks makes the boundary invariant directly testable.
fn resample_non_wrapping_blocks(
    source: &[SeedTransitionPair],
    block_length: usize,
    rng: &mut SplitMix64,
) -> Result<Vec<Vec<SeedTransitionPair>>> {
    if source.is_empty() || block_length == 0 {
        bail!("bootstrap segment and block length must be nonzero");
    }
    let mut blocks = Vec::new();
    let mut emitted = 0usize;
    while emitted < source.len() {
        let start = rng.index(source.len());
        let end = start.saturating_add(block_length).min(source.len());
        let take = (end - start).min(source.len() - emitted);
        blocks.push(source[start..start + take].to_vec());
        emitted += take;
    }
    Ok(blocks)
}

fn directed_count_flux_asymmetry(pairs: &[SeedTransitionPair], state_count: usize) -> Result<f64> {
    if pairs.is_empty() || state_count < 2 {
        bail!("directed-count reversibility diagnostic lacks transitions or states");
    }
    let mut directed = vec![vec![0u64; state_count]; state_count];
    for pair in pairs {
        if pair.from >= state_count || pair.to >= state_count {
            bail!("directed-count diagnostic state exceeds declared state count");
        }
        directed[pair.from][pair.to] += 1;
    }
    let denominator = pairs.len() as f64;
    let mut maximum = 0.0_f64;
    for i in 0..state_count {
        for j in (i + 1)..state_count {
            maximum =
                maximum.max((directed[i][j] as f64 - directed[j][i] as f64).abs() / denominator);
        }
    }
    Ok(maximum)
}

fn bootstrap_uncertainty(
    control_lane: ControlLane,
    pairs: &[SeedTransitionPair],
    samples: &[SurveillanceSample],
    state_ids: &[String],
    config: &NativePathBEstimatorConfig,
) -> Result<(
    Vec<TransitionBootstrapInterval>,
    Vec<CommittorBootstrapInterval>,
    DirectedCountReversibilityDiagnostic,
)> {
    let mut by_seed =
        BTreeMap::<u64, BTreeMap<BootstrapSegmentKey, Vec<SeedTransitionPair>>>::new();
    for pair in pairs {
        let key = transition_pair_segment_key(pair, samples)?;
        by_seed
            .entry(pair.seed)
            .or_default()
            .entry(key)
            .or_default()
            .push(*pair);
    }
    for segments in by_seed.values_mut() {
        for segment in segments.values_mut() {
            segment.sort_by_key(|pair| {
                (
                    samples[pair.from_position].step_idx,
                    samples[pair.to_position].step_idx,
                )
            });
        }
    }
    let seed_blocks = by_seed.into_values().collect::<Vec<_>>();
    if seed_blocks.len() < config.minimum_independent_seeds {
        bail!("bootstrap has insufficient independent seeds");
    }
    let k = state_ids.len();
    let mut transition_samples = vec![vec![Vec::new(); k]; k];
    let mut directed_asymmetry_samples = Vec::with_capacity(config.bootstrap_replicates);
    for replicate in 0..config.bootstrap_replicates {
        let mut rng = SplitMix64::new(0x5052_4953_4d50_4154u64 ^ replicate as u64);
        let mut sampled = Vec::new();
        for _ in 0..seed_blocks.len() {
            let source_seed = &seed_blocks[rng.index(seed_blocks.len())];
            for source_segment in source_seed.values() {
                for block in resample_non_wrapping_blocks(
                    source_segment,
                    config.bootstrap_block_length,
                    &mut rng,
                )? {
                    sampled.extend(block);
                }
            }
        }
        directed_asymmetry_samples.push(directed_count_flux_asymmetry(&sampled, k)?);
        let (matrix, _) = reversible_model_from_pairs(&sampled, k, config.reversible_pseudocount)?;
        for i in 0..k {
            for j in 0..k {
                transition_samples[i][j].push(matrix[i][j]);
            }
        }
    }
    let mut transition_intervals = Vec::with_capacity(k * k);
    for i in 0..k {
        for j in 0..k {
            let (mean, lower, upper) = percentile_interval(&mut transition_samples[i][j]);
            transition_intervals.push(TransitionBootstrapInterval {
                control_lane,
                from_state_id: state_ids[i].clone(),
                to_state_id: state_ids[j].clone(),
                mean,
                lower,
                upper,
            });
        }
    }
    let (bootstrap_mean, bootstrap_lower, bootstrap_upper) =
        percentile_interval(&mut directed_asymmetry_samples);
    Ok((
        transition_intervals,
        Vec::new(),
        DirectedCountReversibilityDiagnostic {
            statistic_id: "maximum_unconstrained_directed_flux_asymmetry_v1".to_string(),
            observed_max_abs_flux_asymmetry: directed_count_flux_asymmetry(pairs, k)?,
            bootstrap_mean: Some(bootstrap_mean),
            bootstrap_lower: Some(bootstrap_lower),
            bootstrap_upper: Some(bootstrap_upper),
            bootstrap_replicates: config.bootstrap_replicates,
            transition_pairs: pairs.len() as u64,
        },
    ))
}

fn percentile_interval(values: &mut [f64]) -> (f64, f64, f64) {
    values.sort_by(f64::total_cmp);
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let lower_index = ((values.len() - 1) as f64 * 0.025).floor() as usize;
    let upper_index = ((values.len() - 1) as f64 * 0.975).ceil() as usize;
    (
        mean,
        values[lower_index].min(mean),
        values[upper_index].max(mean),
    )
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn index(&mut self, length: usize) -> usize {
        (self.next() % length as u64) as usize
    }
}

/// Classification produced from three matched endpoint lanes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PocketResponseClass {
    /// The pocket is present without a probe and is not selectively enhanced.
    PreExistingTransient,
    /// The pocket exists without a probe and is selectively stabilized by the hypothesis fragment.
    HypothesisFragmentStabilized,
    /// The pocket is absent without probes and created selectively by the hypothesis fragment.
    HypothesisSelectiveCreated,
    /// Both probe chemistries create the pocket; stimulation is not hypothesis-specific.
    NonspecificProbeCreated,
    /// All endpoint lanes remain below the presence threshold.
    NoEndpointOpening,
    /// Available evidence does not separate the mechanistic classes.
    Inconclusive,
}

/// Configuration for endpoint-only matched-control inference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatchedControlConfig {
    /// Minimum common independent seeds across all lanes. Must be at least five.
    pub minimum_independent_seeds: usize,
    /// Minimum total ESS in every lane.
    pub minimum_effective_sample_size_per_lane: f64,
    /// Family-wise type-I error rate.
    pub familywise_alpha: f64,
    /// Number of predeclared comparisons for Bonferroni correction.
    pub comparison_family_size: usize,
    /// Lower confidence bound required to call an opening present.
    pub presence_probability_threshold: f64,
    /// Upper confidence bound required to call an opening absent.
    pub absence_probability_threshold: f64,
    /// Minimum selective paired probability increase.
    pub minimum_selective_effect: f64,
}

impl Default for MatchedControlConfig {
    fn default() -> Self {
        Self {
            minimum_independent_seeds: 5,
            minimum_effective_sample_size_per_lane: 25.0,
            familywise_alpha: 0.05,
            comparison_family_size: 3,
            presence_probability_threshold: 0.20,
            absence_probability_threshold: 0.10,
            minimum_selective_effect: 0.10,
        }
    }
}

impl MatchedControlConfig {
    /// Validate matched-lane seed, ESS and simultaneous-inference gates.
    pub fn validate(&self) -> Result<()> {
        if self.minimum_independent_seeds < 5 {
            bail!("matched-control inference requires at least five independent seeds");
        }
        if !self.minimum_effective_sample_size_per_lane.is_finite()
            || self.minimum_effective_sample_size_per_lane <= 0.0
        {
            bail!("minimum ESS must be finite and positive");
        }
        if !self.familywise_alpha.is_finite()
            || !(0.0..1.0).contains(&self.familywise_alpha)
            || self.comparison_family_size == 0
        {
            bail!("invalid multiple-comparison configuration");
        }
        for (value, name) in [
            (self.presence_probability_threshold, "presence threshold"),
            (self.absence_probability_threshold, "absence threshold"),
            (self.minimum_selective_effect, "minimum selective effect"),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                bail!("{name} must be in [0, 1]");
            }
        }
        if self.absence_probability_threshold >= self.presence_probability_threshold {
            bail!("absence threshold must be lower than presence threshold");
        }
        Ok(())
    }
}

/// Confidence interval for a lane or paired effect.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EstimateInterval {
    /// Arithmetic mean over seed-level estimands.
    pub mean: f64,
    /// Seed-level standard error.
    pub standard_error: f64,
    /// Simultaneous confidence lower bound.
    pub lower: f64,
    /// Simultaneous confidence upper bound.
    pub upper: f64,
}

/// Endpoint summary for one experimental lane.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LaneEndpointSummary {
    /// Lane identity.
    pub lane: ControlLane,
    /// Number of independent seed-level estimands.
    pub independent_seed_count: usize,
    /// Conservative minimum run-level autocorrelation ESS across seeds. This
    /// is never a sum over node/fiber rows.
    pub effective_sample_size: f64,
    /// Endpoint open-pocket probability interval.
    pub open_probability: EstimateInterval,
}

/// Exactly one autocorrelation-aware estimand for one matched lane/run/seed.
/// Node-level fibers remain auditable detail, but statistical inference uses
/// this run-level object once and only once.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatchedSeedEstimand {
    pub schema_kind: String,
    pub schema_version: String,
    pub estimand_record_sha256: String,
    pub site_key: String,
    pub lane: ControlLane,
    pub replica_seed: u64,
    pub run_id: String,
    pub replica_id: String,
    pub open_state_definition_sha256: String,
    pub pathb_model_sha256: String,
    /// SHA-256 of sorted `(step, mean, source lineage hashes)` rows.
    pub source_timeseries_sha256: String,
    pub unique_step_count: u64,
    pub open_probability: f64,
    pub integrated_autocorrelation_time_steps: f64,
    pub effective_sample_size: f64,
    pub standard_error: f64,
}

#[derive(Serialize)]
struct MatchedSeedEstimandBody<'a> {
    site_key: &'a str,
    lane: ControlLane,
    replica_seed: u64,
    run_id: &'a str,
    replica_id: &'a str,
    open_state_definition_sha256: &'a str,
    pathb_model_sha256: &'a str,
    source_timeseries_sha256: &'a str,
    unique_step_count: u64,
    open_probability: f64,
    integrated_autocorrelation_time_steps: f64,
    effective_sample_size: f64,
    standard_error: f64,
}

impl MatchedSeedEstimand {
    fn body(&self) -> MatchedSeedEstimandBody<'_> {
        MatchedSeedEstimandBody {
            site_key: &self.site_key,
            lane: self.lane,
            replica_seed: self.replica_seed,
            run_id: &self.run_id,
            replica_id: &self.replica_id,
            open_state_definition_sha256: &self.open_state_definition_sha256,
            pathb_model_sha256: &self.pathb_model_sha256,
            source_timeseries_sha256: &self.source_timeseries_sha256,
            unique_step_count: self.unique_step_count,
            open_probability: self.open_probability,
            integrated_autocorrelation_time_steps: self.integrated_autocorrelation_time_steps,
            effective_sample_size: self.effective_sample_size,
            standard_error: self.standard_error,
        }
    }

    pub fn verify(&self) -> Result<()> {
        if self.schema_kind != "prism_pathb_matched_seed_estimand"
            || self.schema_version != "1.0.0"
            || self.unique_step_count == 0
            || !self.open_probability.is_finite()
            || !(0.0..=1.0).contains(&self.open_probability)
            || !self.integrated_autocorrelation_time_steps.is_finite()
            || self.integrated_autocorrelation_time_steps < 1.0
            || !self.effective_sample_size.is_finite()
            || self.effective_sample_size < 1.0
            || self.effective_sample_size > self.unique_step_count as f64 + 1.0e-9
            || !self.standard_error.is_finite()
            || self.standard_error < 0.0
            || self.run_id.trim().is_empty()
            || self.replica_id.trim().is_empty()
        {
            bail!("invalid matched seed/run estimand");
        }
        for (digest, field) in [
            (&self.estimand_record_sha256, "estimand_record_sha256"),
            (
                &self.open_state_definition_sha256,
                "open_state_definition_sha256",
            ),
            (&self.pathb_model_sha256, "pathb_model_sha256"),
            (&self.source_timeseries_sha256, "source_timeseries_sha256"),
        ] {
            require_sha256(digest, field)?;
        }
        if canonical_sha256(&self.body())? != self.estimand_record_sha256 {
            bail!("matched seed/run estimand hash mismatch");
        }
        Ok(())
    }
}

/// Distribution overlap diagnostics between matched lanes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LaneOverlapDiagnostics {
    /// Bhattacharyya coefficient for no-probe versus hypothesis seed estimands.
    pub no_probe_hypothesis: f64,
    /// Bhattacharyya coefficient for no-probe versus orthogonal seed estimands.
    pub no_probe_orthogonal: f64,
    /// Bhattacharyya coefficient for hypothesis versus orthogonal seed estimands.
    pub hypothesis_orthogonal: f64,
}

/// Content-addressed matched-control result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatchedControlResult {
    /// Schema identifier.
    pub schema_kind: String,
    /// Schema version.
    pub schema_version: String,
    /// SHA-256 over the result body excluding this field.
    pub result_sha256: String,
    /// Site identity.
    pub site_key: String,
    /// Shared common-estimand digest.
    pub estimand_sha256: String,
    /// Engine hash shared by all lanes.
    pub engine_sha256: String,
    /// Native Path-B model hash shared by all lanes.
    pub pathb_model_sha256: String,
    /// Mechanistic endpoint classification.
    pub classification: PocketResponseClass,
    /// No-probe endpoint summary.
    pub no_probe: LaneEndpointSummary,
    /// Hypothesis-fragment endpoint summary.
    pub hypothesis: LaneEndpointSummary,
    /// Orthogonal-control endpoint summary.
    pub orthogonal: LaneEndpointSummary,
    /// Paired hypothesis minus no-probe effect.
    pub hypothesis_minus_no_probe: EstimateInterval,
    /// Paired orthogonal minus no-probe effect.
    pub orthogonal_minus_no_probe: EstimateInterval,
    /// Paired hypothesis minus orthogonal effect.
    pub hypothesis_minus_orthogonal: EstimateInterval,
    /// Common independent seeds used in every comparison.
    pub common_seeds: Vec<u64>,
    /// Distribution-overlap diagnostics.
    pub overlap: LaneOverlapDiagnostics,
    /// Family-wise alpha.
    pub familywise_alpha: f64,
    /// Bonferroni per-comparison alpha.
    pub corrected_alpha: f64,
    /// Number of predeclared comparisons.
    pub comparison_family_size: usize,
    /// One immutable estimand per lane/seed; no node row can contribute ESS
    /// more than once.
    pub seed_estimands: Vec<MatchedSeedEstimand>,
    /// Content hashes of every source fiber.
    pub source_fiber_sha256: Vec<String>,
}

#[derive(Serialize)]
struct MatchedControlHashBody<'a> {
    site_key: &'a str,
    estimand_sha256: &'a str,
    engine_sha256: &'a str,
    pathb_model_sha256: &'a str,
    classification: PocketResponseClass,
    no_probe: &'a LaneEndpointSummary,
    hypothesis: &'a LaneEndpointSummary,
    orthogonal: &'a LaneEndpointSummary,
    hypothesis_minus_no_probe: &'a EstimateInterval,
    orthogonal_minus_no_probe: &'a EstimateInterval,
    hypothesis_minus_orthogonal: &'a EstimateInterval,
    common_seeds: &'a [u64],
    overlap: &'a LaneOverlapDiagnostics,
    familywise_alpha: f64,
    corrected_alpha: f64,
    comparison_family_size: usize,
    seed_estimands: &'a [MatchedSeedEstimand],
    source_fiber_sha256: &'a [String],
}

/// Classify endpoint fibers using common seeds and simultaneous intervals.
pub fn classify_matched_controls(
    fibers: &[ChemicalStateFiber],
    config: &MatchedControlConfig,
) -> Result<MatchedControlResult> {
    config.validate()?;
    if fibers.is_empty() {
        bail!("matched-control classification requires fibers");
    }
    for fiber in fibers {
        fiber.verify()?;
        if fiber.endpoint_eligibility != EndpointEligibility::PhysicalEndpoint {
            bail!("partial-lambda fibers are sampling-only and cannot enter endpoint inference");
        }
        if fiber.body.control_lane != ControlLane::NoProbe && fiber.body.fragment.lambda != 1.0 {
            bail!("probe lane endpoint inference requires the physical lambda=1 endpoint");
        }
    }

    let first = &fibers[0].body;
    let site_key = first.site_key.clone();
    let estimand_sha256 = first
        .structural_response
        .open_state
        .definition_sha256
        .clone();
    let engine_sha256 = first.ghost.engine_sha256.clone();
    let pathb_model_sha256 = first.pathb.model_sha256.clone();
    let physical_state_sha256 = first
        .thermodynamic_condition
        .matched_physical_state_sha256(&first.protocol_phase)?;
    for fiber in fibers {
        let body = &fiber.body;
        if body.site_key != site_key
            || body.structural_response.open_state.definition_sha256 != estimand_sha256
            || body.ghost.engine_sha256 != engine_sha256
            || body.pathb.model_sha256 != pathb_model_sha256
            || body
                .thermodynamic_condition
                .matched_physical_state_sha256(&body.protocol_phase)?
                != physical_state_sha256
        {
            bail!("matched lanes do not share site, estimand, engine, Path-B model, and physical thermodynamic state");
        }
    }

    let hypothesis_species = unique_probe_species(fibers, ControlLane::HypothesisFragment)?;
    let orthogonal_species = unique_probe_species(fibers, ControlLane::OrthogonalChemicalControl)?;
    if hypothesis_species == orthogonal_species {
        bail!("orthogonal chemical control cannot be identical to the hypothesis fragment");
    }

    let seed_estimands = derive_matched_seed_estimands(fibers)?;
    let grouped = seed_estimands
        .iter()
        .map(|estimand| ((estimand.lane, estimand.replica_seed), estimand))
        .collect::<BTreeMap<_, _>>();
    let lane_seed_sets = [
        ControlLane::NoProbe,
        ControlLane::HypothesisFragment,
        ControlLane::OrthogonalChemicalControl,
    ]
    .into_iter()
    .map(|lane| {
        grouped
            .iter()
            .filter_map(|((entry_lane, seed), _)| (*entry_lane == lane).then_some(*seed))
            .collect::<BTreeSet<_>>()
    })
    .collect::<Vec<_>>();
    if lane_seed_sets[0] != lane_seed_sets[1] || lane_seed_sets[0] != lane_seed_sets[2] {
        bail!("matched-control lanes must have exactly the same independent seed set");
    }
    let common_seeds = lane_seed_sets[0].iter().copied().collect::<Vec<_>>();
    if common_seeds.len() < config.minimum_independent_seeds {
        bail!(
            "matched-control common seed count {} is below required {}",
            common_seeds.len(),
            config.minimum_independent_seeds
        );
    }

    let values_for = |lane| {
        common_seeds
            .iter()
            .map(|seed| grouped[&(lane, *seed)].open_probability)
            .collect::<Vec<_>>()
    };
    let no_values = values_for(ControlLane::NoProbe);
    let hypothesis_values = values_for(ControlLane::HypothesisFragment);
    let orthogonal_values = values_for(ControlLane::OrthogonalChemicalControl);
    let corrected_alpha = config.familywise_alpha / config.comparison_family_size as f64;

    let no_probe = lane_summary(
        ControlLane::NoProbe,
        &no_values,
        conservative_lane_ess(&grouped, ControlLane::NoProbe),
        corrected_alpha,
    )?;
    let hypothesis = lane_summary(
        ControlLane::HypothesisFragment,
        &hypothesis_values,
        conservative_lane_ess(&grouped, ControlLane::HypothesisFragment),
        corrected_alpha,
    )?;
    let orthogonal = lane_summary(
        ControlLane::OrthogonalChemicalControl,
        &orthogonal_values,
        conservative_lane_ess(&grouped, ControlLane::OrthogonalChemicalControl),
        corrected_alpha,
    )?;
    for summary in [&no_probe, &hypothesis, &orthogonal] {
        if summary.effective_sample_size < config.minimum_effective_sample_size_per_lane {
            bail!("lane {:?} does not meet the ESS gate", summary.lane);
        }
    }

    let paired = |left: &[f64], right: &[f64]| {
        let values = left
            .iter()
            .zip(right)
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>();
        simultaneous_interval(&values, corrected_alpha)
    };
    let hypothesis_minus_no_probe = paired(&hypothesis_values, &no_values)?;
    let orthogonal_minus_no_probe = paired(&orthogonal_values, &no_values)?;
    let hypothesis_minus_orthogonal = paired(&hypothesis_values, &orthogonal_values)?;

    let classification = if no_probe.open_probability.lower >= config.presence_probability_threshold
    {
        if hypothesis_minus_no_probe.lower >= config.minimum_selective_effect
            && hypothesis_minus_orthogonal.lower >= config.minimum_selective_effect
        {
            PocketResponseClass::HypothesisFragmentStabilized
        } else {
            PocketResponseClass::PreExistingTransient
        }
    } else if no_probe.open_probability.upper <= config.absence_probability_threshold
        && hypothesis.open_probability.lower >= config.presence_probability_threshold
    {
        if orthogonal.open_probability.upper <= config.absence_probability_threshold
            && hypothesis_minus_orthogonal.lower >= config.minimum_selective_effect
        {
            PocketResponseClass::HypothesisSelectiveCreated
        } else if orthogonal.open_probability.lower >= config.presence_probability_threshold {
            PocketResponseClass::NonspecificProbeCreated
        } else {
            PocketResponseClass::Inconclusive
        }
    } else if no_probe.open_probability.upper < config.presence_probability_threshold
        && hypothesis.open_probability.upper < config.presence_probability_threshold
        && orthogonal.open_probability.upper < config.presence_probability_threshold
    {
        PocketResponseClass::NoEndpointOpening
    } else {
        PocketResponseClass::Inconclusive
    };

    let overlap = LaneOverlapDiagnostics {
        no_probe_hypothesis: bhattacharyya_overlap(&no_values, &hypothesis_values),
        no_probe_orthogonal: bhattacharyya_overlap(&no_values, &orthogonal_values),
        hypothesis_orthogonal: bhattacharyya_overlap(&hypothesis_values, &orthogonal_values),
    };
    let mut source_fiber_sha256 = fibers
        .iter()
        .map(|fiber| fiber.fiber_sha256.clone())
        .collect::<Vec<_>>();
    source_fiber_sha256.sort();

    let mut result = MatchedControlResult {
        schema_kind: "prism_matched_control_discriminator".to_string(),
        schema_version: "2.0.0".to_string(),
        result_sha256: String::new(),
        site_key,
        estimand_sha256,
        engine_sha256,
        pathb_model_sha256,
        classification,
        no_probe,
        hypothesis,
        orthogonal,
        hypothesis_minus_no_probe,
        orthogonal_minus_no_probe,
        hypothesis_minus_orthogonal,
        common_seeds,
        overlap,
        familywise_alpha: config.familywise_alpha,
        corrected_alpha,
        comparison_family_size: config.comparison_family_size,
        seed_estimands,
        source_fiber_sha256,
    };
    result.result_sha256 = matched_result_hash(&result)?;
    Ok(result)
}

impl MatchedControlResult {
    /// Verify the content address and basic materialized gate fields.
    pub fn verify(&self) -> Result<()> {
        if self.schema_kind != "prism_matched_control_discriminator"
            || self.schema_version != "2.0.0"
        {
            bail!("unsupported matched-control result schema");
        }
        require_sha256(&self.result_sha256, "result_sha256")?;
        require_sha256(&self.estimand_sha256, "estimand_sha256")?;
        require_sha256(&self.engine_sha256, "engine_sha256")?;
        require_sha256(&self.pathb_model_sha256, "pathb_model_sha256")?;
        if self.common_seeds.len() < 5 {
            bail!("matched-control result has fewer than five independent seeds");
        }
        for estimand in &self.seed_estimands {
            estimand.verify()?;
        }
        let expected_estimands = self
            .common_seeds
            .len()
            .checked_mul(3)
            .context("matched-control estimand count overflow")?;
        if self.seed_estimands.len() != expected_estimands {
            bail!("matched-control result must contain exactly one estimand per lane/seed");
        }
        if self.result_sha256 != matched_result_hash(self)? {
            bail!("matched-control result content hash mismatch");
        }
        Ok(())
    }
}

fn matched_result_hash(result: &MatchedControlResult) -> Result<String> {
    canonical_sha256(&MatchedControlHashBody {
        site_key: &result.site_key,
        estimand_sha256: &result.estimand_sha256,
        engine_sha256: &result.engine_sha256,
        pathb_model_sha256: &result.pathb_model_sha256,
        classification: result.classification,
        no_probe: &result.no_probe,
        hypothesis: &result.hypothesis,
        orthogonal: &result.orthogonal,
        hypothesis_minus_no_probe: &result.hypothesis_minus_no_probe,
        orthogonal_minus_no_probe: &result.orthogonal_minus_no_probe,
        hypothesis_minus_orthogonal: &result.hypothesis_minus_orthogonal,
        common_seeds: &result.common_seeds,
        overlap: &result.overlap,
        familywise_alpha: result.familywise_alpha,
        corrected_alpha: result.corrected_alpha,
        comparison_family_size: result.comparison_family_size,
        seed_estimands: &result.seed_estimands,
        source_fiber_sha256: &result.source_fiber_sha256,
    })
}

fn unique_probe_species(fibers: &[ChemicalStateFiber], lane: ControlLane) -> Result<String> {
    let identities = fibers
        .iter()
        .filter(|fiber| fiber.body.control_lane == lane)
        .map(|fiber| {
            fiber
                .body
                .fragment
                .identity
                .as_ref()
                .map(|identity| {
                    format!(
                        "{}:{}:{}:{}",
                        identity.species_id,
                        identity.parameter_digest_sha256,
                        identity.tautomer_id,
                        identity.protonation_state_id
                    )
                })
                .context("probe lane fiber lacks chemical identity")
        })
        .collect::<Result<BTreeSet<_>>>()?;
    if identities.len() != 1 {
        bail!("each probe lane must use one explicit chemical microstate");
    }
    Ok(identities.into_iter().next().unwrap())
}

/// Reduce all node/fiber rows to one estimand for each physical lane/run/seed.
/// Rows at the same integration step are averaged first, producing one
/// run-level time series and preventing stream/node pseudoreplication.
pub fn derive_matched_seed_estimands(
    fibers: &[ChemicalStateFiber],
) -> Result<Vec<MatchedSeedEstimand>> {
    let mut raw: BTreeMap<(ControlLane, u64), Vec<&ChemicalStateFiber>> = BTreeMap::new();
    for fiber in fibers {
        raw.entry((fiber.body.control_lane, fiber.body.uncertainty.replica_seed))
            .or_default()
            .push(fiber);
    }
    let mut grouped = Vec::with_capacity(raw.len());
    for (key, entries) in raw {
        let identities = entries
            .iter()
            .map(|fiber| {
                (
                    fiber.body.ghost.run_id.clone(),
                    fiber.body.uncertainty.replica_id.clone(),
                    fiber.body.site_key.clone(),
                    fiber
                        .body
                        .structural_response
                        .open_state
                        .definition_sha256
                        .clone(),
                    fiber.body.pathb.model_sha256.clone(),
                )
            })
            .collect::<BTreeSet<_>>();
        if identities.len() != 1 {
            bail!("one lane/seed estimand cannot pool runs, replicas, sites, estimands, or Path-B models");
        }
        let (run_id, replica_id, site_key, definition_sha256, pathb_model_sha256) =
            identities.into_iter().next().unwrap();
        let mut by_step: BTreeMap<u64, Vec<&ChemicalStateFiber>> = BTreeMap::new();
        for fiber in entries {
            by_step
                .entry(fiber.body.ghost.key.step_idx)
                .or_default()
                .push(fiber);
        }
        let mut series = Vec::with_capacity(by_step.len());
        let mut source_rows = Vec::with_capacity(by_step.len());
        for (step, step_fibers) in by_step {
            let mean = step_fibers
                .iter()
                .map(|fiber| fiber.body.structural_response.open_state.probability)
                .sum::<f64>()
                / step_fibers.len() as f64;
            if !mean.is_finite() || !(0.0..=1.0).contains(&mean) {
                bail!("run-level open-state time series contains an invalid probability");
            }
            let mut lineages = step_fibers
                .iter()
                .map(|fiber| fiber.body.ghost.lineage_sha256())
                .collect::<Result<Vec<_>>>()?;
            lineages.sort();
            source_rows.push((step, mean, lineages));
            series.push(mean);
        }
        let source_timeseries_sha256 = canonical_sha256(&source_rows)?;
        let mean = series.iter().sum::<f64>() / series.len() as f64;
        let (tau, ess) = integrated_autocorrelation_ess(&series)?;
        let variance = if series.len() > 1 {
            series
                .iter()
                .map(|value| (value - mean).powi(2))
                .sum::<f64>()
                / (series.len() - 1) as f64
        } else {
            0.0
        };
        let standard_error = (variance / ess).sqrt();
        let mut estimand = MatchedSeedEstimand {
            schema_kind: "prism_pathb_matched_seed_estimand".to_string(),
            schema_version: "1.0.0".to_string(),
            estimand_record_sha256: String::new(),
            site_key,
            lane: key.0,
            replica_seed: key.1,
            run_id,
            replica_id,
            open_state_definition_sha256: definition_sha256,
            pathb_model_sha256,
            source_timeseries_sha256,
            unique_step_count: series.len() as u64,
            open_probability: mean,
            integrated_autocorrelation_time_steps: tau,
            effective_sample_size: ess,
            standard_error,
        };
        estimand.estimand_record_sha256 = canonical_sha256(&estimand.body())?;
        estimand.verify()?;
        grouped.push(estimand);
    }
    grouped.sort_by_key(|estimand| (estimand.lane, estimand.replica_seed));
    Ok(grouped)
}

fn conservative_lane_ess(
    grouped: &BTreeMap<(ControlLane, u64), &MatchedSeedEstimand>,
    lane: ControlLane,
) -> f64 {
    grouped
        .iter()
        .filter_map(|((entry_lane, _), value)| {
            (*entry_lane == lane).then_some(value.effective_sample_size)
        })
        .fold(f64::INFINITY, f64::min)
}

/// Geyer initial-positive-sequence estimate over the run-level series. The
/// returned ESS is bounded by the number of unique integration steps.
pub(crate) fn integrated_autocorrelation_ess(values: &[f64]) -> Result<(f64, f64)> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        bail!("autocorrelation ESS requires a non-empty finite time series");
    }
    if values.len() == 1 {
        return Ok((1.0, 1.0));
    }
    let n = values.len();
    let mean = values.iter().sum::<f64>() / n as f64;
    let gamma0 = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / n as f64;
    if gamma0 <= f64::EPSILON {
        return Ok((1.0, n as f64));
    }
    let autocovariance = |lag: usize| {
        values[..n - lag]
            .iter()
            .zip(&values[lag..])
            .map(|(left, right)| (left - mean) * (right - mean))
            .sum::<f64>()
            / (n - lag) as f64
    };
    let mut paired_sum_total = 0.0;
    let mut previous_pair = f64::INFINITY;
    let mut lag = 1usize;
    while lag + 1 < n {
        let pair = (autocovariance(lag) + autocovariance(lag + 1)) / gamma0;
        if !pair.is_finite() || pair <= 0.0 {
            break;
        }
        let monotone_pair = pair.min(previous_pair);
        paired_sum_total += monotone_pair;
        previous_pair = monotone_pair;
        lag += 2;
    }
    let tau = (1.0 + 2.0 * paired_sum_total).clamp(1.0, n as f64);
    Ok((tau, (n as f64 / tau).clamp(1.0, n as f64)))
}

fn lane_summary(
    lane: ControlLane,
    values: &[f64],
    effective_sample_size: f64,
    alpha: f64,
) -> Result<LaneEndpointSummary> {
    Ok(LaneEndpointSummary {
        lane,
        independent_seed_count: values.len(),
        effective_sample_size,
        open_probability: simultaneous_interval(values, alpha)?,
    })
}

fn simultaneous_interval(values: &[f64], alpha: f64) -> Result<EstimateInterval> {
    if values.len() < 2 || values.iter().any(|value| !value.is_finite()) {
        bail!("confidence interval requires at least two finite seed estimands");
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (n - 1.0);
    let standard_error = (variance / n).sqrt();
    let t = StudentsT::new(0.0, 1.0, n - 1.0).context("construct Student-t distribution")?;
    let critical = t.inverse_cdf(1.0 - alpha / 2.0);
    let half_width = critical * standard_error;
    Ok(EstimateInterval {
        mean,
        standard_error,
        lower: mean - half_width,
        upper: mean + half_width,
    })
}

fn bhattacharyya_overlap(left: &[f64], right: &[f64]) -> f64 {
    fn moments(values: &[f64]) -> (f64, f64) {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = if values.len() > 1 {
            values
                .iter()
                .map(|value| (value - mean).powi(2))
                .sum::<f64>()
                / (values.len() - 1) as f64
        } else {
            0.0
        };
        (mean, variance.max(1.0e-15))
    }
    let (mean_a, variance_a) = moments(left);
    let (mean_b, variance_b) = moments(right);
    let denominator = variance_a + variance_b;
    let coefficient = (2.0 * (variance_a * variance_b).sqrt() / denominator).sqrt()
        * (-(mean_a - mean_b).powi(2) / (4.0 * denominator)).exp();
    coefficient.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chemical_state_fiber::*;

    fn digest(byte: u8) -> String {
        format!("{byte:02x}").repeat(32)
    }

    fn model_lineage() -> GhostByteLineage {
        crate::chemical_state_fiber::tests::body(ControlLane::NoProbe, 0.0, 100, 1).ghost
    }


    /// Full original population regression for the recording contract. This
    /// consumes immutable source artifacts; it does not execute physical MD.
    #[cfg(any(feature = "gpu", feature = "cpu-analysis"))]
    #[test]
    #[ignore = "requires the pinned original 7F5N treatment Ghost segment population"]
    fn original_7f5n_native_observation_closed_aabb_population() -> Result<()> {
        use crate::ghost_phase_lattice::load_verified_ghost_segments_from_dir;
        use crate::pathb_evidence_emitter::adapt_verified_ghost_load_for_pathb;
        let dir_buf=std::env::var_os("PRISM_MD_RUNS_ROOT").map(std::path::PathBuf::from).ok_or_else(|| anyhow::anyhow!("PRISM_MD_RUNS_ROOT unset: set it to the root that holds 7F5N_directed_pair_seed42_kcc_20260923_treatment"))?.join("7F5N_directed_pair_seed42_kcc_20260923_treatment/output");
        let dir=dir_buf.as_path();
        let load=load_verified_ghost_segments_from_dir(dir,"7F5N_A.v3",500)?;
        let inputs=adapt_verified_ghost_load_for_pathb(&load,
            "bfbb6c5df3be7a5b16d11e51bef43512e2fe8c84328ecde78acff5b89b35819a")?;
        assert_eq!(load.nodes.len(),233700);
        assert_eq!(inputs.lineages.len(),load.nodes.len());
        assert!(inputs.lineages.iter().all(|lineage|
            lineage.segment_manifest_sha256 == "f1d5d3db4af7eac9e520acc7be9eebc4fd31e24d591c3e216facc10ec3aff0a6"
                && lineage.run_id == "7F5N_A.v3.topology_20260923_210537"),
            "fixture must remain bound to the original treatment segment population");
        let mut zero_extent=0usize;
        for (node,lineage) in load.nodes.iter().zip(&inputs.lineages) {
            let observation=NativePathBLatticeObservation::from_native_node(node,lineage)?;
            observation.verify_for(lineage)?;
            assert_eq!(observation.to_native_node().aabb_min,node.aabb_min);
            assert_eq!(observation.to_native_node().aabb_max,node.aabb_max);
            if node.aabb_volume()==0.0 {
                zero_extent+=1;
                assert!(lattice_feature_vector(node,0).is_err(),
                    "recording closed support must not alter existing feature conversion volume requirement");
            }
        }
        assert!(zero_extent>0);
        let node=&load.nodes[1];let lineage=&inputs.lineages[1];
        assert_eq!(node.site_id,33);
        assert_eq!(node.aabb_min[0],node.aabb_max[0]);
        assert_eq!(node.aabb_min[1],node.aabb_max[1]);
        assert!(node.aabb_min[2]<node.aabb_max[2]);
        let original=NativePathBLatticeObservation::from_native_node(node,lineage)?;
        let mut inverted=original.clone();inverted.aabb_min[2]=node.aabb_max[2]+1.0;
        inverted.observation_sha256=canonical_sha256(&inverted.body())?;
        assert!(inverted.verify_for(lineage).is_err());
        let mut changed_hash=original;changed_hash.centroid_xyz[0]+=1.0;
        assert!(changed_hash.verify_for(lineage).is_err());
        println!("closed_native_aabb_population nodes={} zero_extent={} feature_converter_positive_volume_guard_preserved=true",load.nodes.len(),zero_extent);
        Ok(())
    }

    fn model_body(row_sum_error: f64) -> NativePathBModelBody {
        let lineage = model_lineage();
        let mut kinetic_model_selection = KineticModelSelectionLedger {
            ledger_sha256: String::new(),
            feature_schema_sha256: digest(4),
            surveillance_population_sha256: digest(5),
            selection_rule:
                "within_each_lag_maximize_complete_train_fitted_lane_active_subspace_validation_vampe_with_every_fold_gate_then_select_smallest_lag_passing_per_lane_active_subspace_its_ck_seed_support"
                    .to_string(),
            candidates: vec![KineticCandidateScore {
                state_count: 2,
                lag_steps: 50,
                kinetic_dimensions: 2,
                fold_seed: vec![1, 2, 3, 4, 5],
                fold_validation_vampe_score: vec![0.2; 5],
                mean_validation_vampe_score: 0.2,
                minimum_fold_validation_vampe_score: 0.2,
                within_lag_winner: true,
                implied_timescale_relative_spread: 0.1,
                chapman_kolmogorov_max_abs_residual: 0.1,
                minimum_state_seed_support: 5,
                lag_diagnostics_pass: true,
                eligible: true,
                ineligible_reasons: Vec::new(),
            }],
            selected_state_count: 2,
            selected_kinetic_dimensions: 2,
            selected_lag_steps: 50,
            selected_mean_validation_vampe_score: 0.2,
            minimum_required_fold_validation_vampe_score: 0.1,
            within_lag_runner_up_score_margin: 0.2,
            loso_seed_count: 5,
            sensitivity_pass: true,
        };
        kinetic_model_selection.ledger_sha256 =
            kinetic_selection_ledger_hash(&kinetic_model_selection).unwrap();
        let transitions = vec![
            TransitionProbability {
                from_state_id: "closed".to_string(),
                to_state_id: "closed".to_string(),
                probability: 0.8,
            },
            TransitionProbability {
                from_state_id: "closed".to_string(),
                to_state_id: "open".to_string(),
                probability: 0.2 + row_sum_error,
            },
            TransitionProbability {
                from_state_id: "open".to_string(),
                to_state_id: "closed".to_string(),
                probability: 0.1,
            },
            TransitionProbability {
                from_state_id: "open".to_string(),
                to_state_id: "open".to_string(),
                probability: 0.9,
            },
        ];
        let stationary_probability_by_state =
            BTreeMap::from([("closed".to_string(), 0.6), ("open".to_string(), 0.4)]);
        let lane_transition_models = [
            ControlLane::NoProbe,
            ControlLane::HypothesisFragment,
            ControlLane::OrthogonalChemicalControl,
        ]
        .into_iter()
        .map(|control_lane| LaneTransitionModel {
            control_lane,
            active_state_ids: vec!["closed".to_string(), "open".to_string()],
            stationary_probability_by_state: stationary_probability_by_state.clone(),
            transitions: transitions.clone(),
        })
        .collect::<Vec<_>>();
        let lane_diagnostics = [
            ControlLane::NoProbe,
            ControlLane::HypothesisFragment,
            ControlLane::OrthogonalChemicalControl,
        ]
        .into_iter()
        .map(|control_lane| LaneMarkovDiagnostics {
            control_lane,
            active_state_ids: vec!["closed".to_string(), "open".to_string()],
            detailed_balance_max_abs_residual: 0.0,
            constrained_reversible_numerical_residual: 0.0,
            directed_count_reversibility: DirectedCountReversibilityDiagnostic {
                statistic_id: "maximum_unconstrained_directed_flux_asymmetry_v1".to_string(),
                observed_max_abs_flux_asymmetry: 0.0,
                bootstrap_mean: None,
                bootstrap_lower: None,
                bootstrap_upper: None,
                bootstrap_replicates: 0,
                transition_pairs: 1,
            },
            chapman_kolmogorov_max_abs_residual: 0.1,
            state_effective_counts: BTreeMap::from([
                ("closed".to_string(), 1),
                ("open".to_string(), 1),
            ]),
            minimum_state_effective_count: 1,
            implied_timescales: vec![ImpliedTimescaleEstimate {
                control_lane,
                lag_steps: 50,
                second_eigenvalue_magnitude: 0.7,
                implied_timescale_steps: Some(140.183),
                transition_pairs: 1,
            }],
            implied_timescale_relative_spread: 0.0,
            state_seed_support: BTreeMap::from([
                ("closed".to_string(), 1),
                ("open".to_string(), 1),
            ]),
            independent_seed_count: 1,
            transition_pairs: 1,
            second_lag_transition_pairs: 1,
            transition_uncertainty: Vec::new(),
            diagnostic_pass: false,
            ineligible_reasons: vec!["synthetic schema fixture".to_string()],
        })
        .collect::<Vec<_>>();
        let fixture_lanes = [
            ControlLane::NoProbe,
            ControlLane::HypothesisFragment,
            ControlLane::OrthogonalChemicalControl,
        ];
        let scheduled_state_assignments = fixture_lanes
            .into_iter()
            .enumerate()
            .flat_map(|(lane_index, control_lane)| {
                let lineage = lineage.clone();
                ["closed", "open"].into_iter().enumerate().map(
                    move |(state_index, dominant_state_id)| {
                        let open = dominant_state_id == "open";
                        let emitted = control_lane == ControlLane::NoProbe && open;
                        ScheduledStructuralStateAssignment {
                            source_surveillance_ledger_sha256: digest(60 + lane_index as u8),
                            source_surveillance_row_sha256: digest(
                                70 + (lane_index * 2 + state_index) as u8,
                            ),
                            run_id: format!("run_{lane_index}"),
                            replica_id: "rep_00".into(),
                            replica_seed: 1,
                            control_lane,
                            stream_id: (lane_index * 2 + state_index) as u16,
                            step_idx: 100,
                            evaluable_segment_id: 0,
                            cavity_volume_a3: if open { 200.0 } else { 100.0 },
                            binary_label: Some(if open {
                                CavityBinaryLabel::Open
                            } else {
                                CavityBinaryLabel::Closed
                            }),
                            ghost_event: if emitted {
                                ScheduledGhostEventStatus::Emitted {
                                    ghost_lineage_sha256: lineage.lineage_sha256().unwrap(),
                                    exact_component_sha256: digest(9),
                                }
                            } else {
                                ScheduledGhostEventStatus::NotEmitted {
                                    typed_absence_receipt_sha256: digest(
                                        80 + (lane_index * 2 + state_index) as u8,
                                    ),
                                }
                            },
                            state_membership: BTreeMap::from([
                                ("closed".to_string(), if open { 0.2 } else { 0.8 }),
                                ("open".to_string(), if open { 0.8 } else { 0.2 }),
                            ]),
                            dominant_state_id: dominant_state_id.to_string(),
                        }
                    },
                )
            })
            .collect::<Vec<_>>();
        NativePathBModelBody {
            lag_steps: 50,
            timebase_context: NativePathBTimebaseContext {
                schema_kind: "prism_native_pathb_timebase_context".to_string(),
                schema_version: "1.0.0".to_string(),
                context_sha256: String::new(),
                integration_timestep_femtoseconds: 2.0,
                integration_timebase_sha256: digest(7),
            }
            .materialize()
            .unwrap(),
            surveillance_cadence_steps: 25,
            normalization_tolerance: 1.0e-9,
            states: vec![
                MetastableState {
                    state_id: "closed".to_string(),
                    stationary_probability: 0.6,
                },
                MetastableState {
                    state_id: "open".to_string(),
                    stationary_probability: 0.4,
                },
            ],
            transitions,
            reference_lane: ControlLane::NoProbe,
            lane_transition_models,
            lane_global_state_support: [
                ControlLane::NoProbe,
                ControlLane::HypothesisFragment,
                ControlLane::OrthogonalChemicalControl,
            ]
            .into_iter()
            .flat_map(|control_lane| {
                ["closed", "open"]
                    .into_iter()
                    .map(move |state_id| LaneGlobalStateSupportEvidence {
                        control_lane,
                        state_id: state_id.to_string(),
                        status: LaneGlobalStateSupportStatus::Observed,
                        scheduled_exposure_rows: 2,
                        hard_assignment_rows: 1,
                        independent_seed_support: 1,
                        minimum_required_assignment_rows: 1,
                        minimum_required_independent_seeds: 1,
                    })
            })
            .collect(),
            assignments: vec![GhostStateAssignment {
                ghost_lineage_sha256: lineage.lineage_sha256().unwrap(),
                ghost_record_sha256: lineage.record_sha256.clone(),
                control_lane: ControlLane::NoProbe,
                state_membership: BTreeMap::from([
                    ("closed".to_string(), 0.2),
                    ("open".to_string(), 0.8),
                ]),
                previous_state_id: Some("closed".to_string()),
                forward_committor: None,
            }],
            scheduled_state_assignments,
            source_lattice_manifest_sha256: digest(1),
            source_exact_connectivity_population_sha256: digest(2),
            source_exact_edge_population_sha256: digest(3),
            diagnostics: MarkovDiagnostics {
                status: MarkovDiagnosticStatus::ComputationallyInconclusive,
                inconclusive_reasons: vec!["synthetic schema fixture".to_string()],
                detailed_balance_max_abs_residual: 0.0,
                constrained_reversible_numerical_residual: 0.0,
                chapman_kolmogorov_max_abs_residual: 0.1,
                state_effective_counts: BTreeMap::from([
                    ("closed".to_string(), 1),
                    ("open".to_string(), 1),
                ]),
                minimum_state_effective_count: 1,
                implied_timescales: vec![ImpliedTimescaleEstimate {
                    control_lane: ControlLane::NoProbe,
                    lag_steps: 50,
                    second_eigenvalue_magnitude: 0.7,
                    implied_timescale_steps: Some(140.183),
                    transition_pairs: 1,
                }],
                implied_timescale_relative_spread: 0.0,
                state_seed_support: BTreeMap::from([
                    ("closed".to_string(), 1),
                    ("open".to_string(), 1),
                ]),
                kinetic_model_selection,
                surveillance_missingness: SurveillanceMissingnessSummary {
                    surveillance_ledger_population_sha256: digest(6),
                    scheduled_rows: 6,
                    evaluable_rows: 6,
                    not_evaluable_rows: 0,
                    evaluable_fraction: 1.0,
                    observed_open_rows: 3,
                    observed_closed_rows: 3,
                    observed_indeterminate_rows: 0,
                    ghost_emitted_rows: 1,
                    ghost_not_emitted_rows: 5,
                    not_evaluable_reason_counts: BTreeMap::new(),
                    run_units: fixture_lanes
                        .into_iter()
                        .enumerate()
                        .map(|(lane_index, control_lane)| SurveillanceRunMissingness {
                            run_id: format!("run_{lane_index}"),
                            replica_id: "rep_00".into(),
                            replica_seed: 1,
                            control_lane,
                            scheduled_rows: 2,
                            evaluable_rows: 2,
                            not_evaluable_rows: 0,
                        })
                        .collect(),
                },
                independent_seed_count: 1,
                bootstrap_replicates: 0,
                bootstrap_block_length: 1,
                transition_uncertainty: Vec::new(),
                committor_uncertainty: Vec::new(),
                committor_status: "UNAVAILABLE_PENDING_HASH_BOUND_OPEN_BASIN".to_string(),
                lane_diagnostics,
            },
        }
    }

    fn surveillance_sample(step_idx: u64, segment: u64) -> SurveillanceSample {
        SurveillanceSample {
            ledger_sha256: digest(21),
            run_id: "run".into(),
            replica_id: "rep_00".into(),
            control_lane: ControlLane::NoProbe,
            evaluable_segment_id: segment,
            seed: 7,
            stream_id: 0,
            step_idx,
            source_row_sha256: digest(23),
            cavity_volume_a3: 150.0,
            binary_label: Some(CavityBinaryLabel::Closed),
            ghost_event: ScheduledGhostEventStatus::NotEmitted {
                typed_absence_receipt_sha256: digest(24),
            },
            features: vec![step_idx as f64, 0.0, 1.0],
            feature_schema_sha256: digest(22),
            detected_ghost_lineage_sha256: None,
        }
    }

    fn transition_pair(
        seed: u64,
        from: usize,
        to: usize,
        from_position: usize,
        to_position: usize,
    ) -> SeedTransitionPair {
        SeedTransitionPair {
            seed,
            from,
            to,
            from_position,
            to_position,
        }
    }

    #[test]
    fn request_v6_config_removes_preselected_state_and_lag_claims() {
        let config = NativePathBEstimatorConfig::default();
        let mut value = serde_json::to_value(&config).unwrap();
        assert!(value.get("metastable_state_count").is_none());
        assert!(value.get("lag_steps").is_none());
        config.validate(100).unwrap();

        let object = value.as_object_mut().unwrap();
        object.insert("metastable_state_count".to_string(), serde_json::json!(3));
        object.insert("lag_steps".to_string(), serde_json::json!(50));
        assert!(serde_json::from_value::<NativePathBEstimatorConfig>(value).is_err());
    }

    #[test]
    fn every_loso_fold_must_meet_threshold_even_when_mean_passes() {
        let scores = [0.9, 0.9, 0.9, 0.9, 0.1];
        assert!(scores.iter().sum::<f64>() / scores.len() as f64 > 0.5);
        assert!(!every_loso_fold_meets_threshold(&scores, 0.5));
        assert!(every_loso_fold_meets_threshold(&[0.5, 0.6, 0.7], 0.5));
    }

    #[test]
    fn timebase_is_hash_bound_and_every_candidate_lag_is_cadence_aligned() {
        let context = NativePathBTimebaseContext {
            schema_kind: "prism_native_pathb_timebase_context".to_string(),
            schema_version: "1.0.0".to_string(),
            context_sha256: String::new(),
            integration_timestep_femtoseconds: 2.0,
            integration_timebase_sha256: digest(40),
        }
        .materialize()
        .unwrap();
        let config = NativePathBEstimatorConfig::default();
        assert_eq!(
            validate_cadence_values([25, 25], &config, &context).unwrap(),
            25
        );
        assert!(validate_cadence_values([30], &config, &context).is_err());
        assert!(validate_cadence_values([], &config, &context).is_err());
        let mut tampered = context;
        tampered.integration_timestep_femtoseconds = 4.0;
        assert!(tampered.verify().is_err());
    }

    #[test]
    fn bootstrap_blocks_never_wrap_or_cross_evaluable_segments() {
        let source = vec![
            transition_pair(7, 0, 0, 0, 1),
            transition_pair(7, 0, 1, 1, 2),
            transition_pair(7, 1, 1, 2, 3),
            transition_pair(7, 1, 0, 3, 4),
        ];
        for replicate in 0..64 {
            let mut rng = SplitMix64::new(replicate);
            let blocks = resample_non_wrapping_blocks(&source, 3, &mut rng).unwrap();
            assert_eq!(blocks.iter().map(Vec::len).sum::<usize>(), source.len());
            for block in blocks {
                assert!(block
                    .windows(2)
                    .all(|pair| { pair[1].from_position == pair[0].from_position + 1 }));
            }
        }

        let samples = vec![surveillance_sample(0, 0), surveillance_sample(25, 1)];
        let crossing = transition_pair(7, 0, 1, 0, 1);
        assert!(transition_pair_segment_key(&crossing, &samples).is_err());
    }

    #[test]
    fn unconstrained_directed_counts_are_not_hidden_by_reversible_symmetrization() {
        let pairs = vec![
            transition_pair(1, 0, 1, 0, 1),
            transition_pair(1, 0, 1, 1, 2),
            transition_pair(2, 0, 1, 2, 3),
            transition_pair(2, 1, 0, 3, 4),
        ];
        let observed = directed_count_flux_asymmetry(&pairs, 2).unwrap();
        assert!((observed - 0.5).abs() < 1.0e-12);
        let (matrix, stationary) = reversible_model_from_pairs(&pairs, 2, 0.5).unwrap();
        let constrained = (0..2)
            .flat_map(|i| {
                let matrix = &matrix;
                let stationary = &stationary;
                (0..2).map(move |j| {
                    (stationary[i] * matrix[i][j] - stationary[j] * matrix[j][i]).abs()
                })
            })
            .fold(0.0_f64, f64::max);
        assert!(constrained < 1.0e-12);
        assert!(observed > constrained);
    }

    #[test]
    fn scheduled_occupancy_averages_streams_within_step_and_retains_no_ghost_rows() {
        let mut body = model_body(0.0);
        let mut no_ghost = body
            .scheduled_state_assignments
            .iter()
            .find(|row| row.control_lane == ControlLane::NoProbe && row.dominant_state_id == "open")
            .unwrap()
            .clone();
        no_ghost.source_surveillance_row_sha256 = digest(41);
        no_ghost.stream_id = 7;
        no_ghost.cavity_volume_a3 = 80.0;
        no_ghost.binary_label = Some(CavityBinaryLabel::Closed);
        no_ghost.ghost_event = ScheduledGhostEventStatus::NotEmitted {
            typed_absence_receipt_sha256: digest(42),
        };
        no_ghost.state_membership =
            BTreeMap::from([("closed".to_string(), 0.8), ("open".to_string(), 0.2)]);
        no_ghost.dominant_state_id = "closed".to_string();
        body.scheduled_state_assignments.push(no_ghost);
        for evidence in &mut body.lane_global_state_support {
            if evidence.control_lane == ControlLane::NoProbe {
                evidence.scheduled_exposure_rows = 3;
                if evidence.state_id == "closed" {
                    evidence.hard_assignment_rows = 2;
                }
            }
        }
        body.diagnostics
            .lane_diagnostics
            .iter_mut()
            .find(|row| row.control_lane == ControlLane::NoProbe)
            .unwrap()
            .state_effective_counts
            .insert("closed".to_string(), 2);
        let missingness = &mut body.diagnostics.surveillance_missingness;
        missingness.scheduled_rows = 7;
        missingness.evaluable_rows = 7;
        missingness.evaluable_fraction = 1.0;
        missingness.observed_closed_rows = 4;
        missingness.ghost_not_emitted_rows = 6;
        missingness.run_units[0].scheduled_rows = 3;
        missingness.run_units[0].evaluable_rows = 3;
        let model = NativePathBModel::materialize(body).unwrap();
        let population =
            estimate_scheduled_open_occupancy(&model, vec!["open".to_string()]).unwrap();
        assert_eq!(population.seed_estimates.len(), 3);
        let estimate = population
            .seed_estimates
            .iter()
            .find(|row| row.control_lane == ControlLane::NoProbe)
            .unwrap();
        assert_eq!(estimate.averaged_step_count, 1);
        assert_eq!(estimate.stream_row_count, 3);
        assert_eq!(estimate.typed_no_ghost_rows, 2);
        assert!((estimate.mean_open_probability - 0.4).abs() < 1.0e-12);
    }

    #[test]
    fn scheduled_occupancy_fails_closed_on_any_missing_structural_frame() {
        let mut body = model_body(0.0);
        let missingness = &mut body.diagnostics.surveillance_missingness;
        missingness.scheduled_rows = 7;
        missingness.not_evaluable_rows = 1;
        missingness.evaluable_fraction = 6.0 / 7.0;
        missingness.ghost_not_emitted_rows = 6;
        missingness
            .not_evaluable_reason_counts
            .insert("MISSING_COORDINATE_FRAME".to_string(), 1);
        missingness.run_units[0].scheduled_rows = 3;
        missingness.run_units[0].not_evaluable_rows = 1;
        let model = NativePathBModel::materialize(body).unwrap();
        assert!(estimate_scheduled_open_occupancy(&model, vec!["open".to_string()]).is_err());
    }

    #[test]
    fn indeterminate_binary_label_does_not_split_continuous_kinetic_segment() {
        let mut first = surveillance_sample(0, 0);
        first.binary_label = None;
        let mut second = surveillance_sample(25, 0);
        second.binary_label = None;
        assert_eq!(
            surveillance_position_pairs_for_lag(&[first, second], 25)
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn held_out_vampe_changes_without_refitting_training_operator() {
        let assignments = vec![0, 0, 1, 1, 0, 1, 0, 0, 1, 1];
        let train_pairs = vec![(0, 0, 1), (0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5)];
        let validation_pairs = vec![(1, 6, 7), (1, 7, 8), (1, 8, 9)];
        let model = fit_train_vampe_model(&assignments, &train_pairs, 2, 1.0e-6).unwrap();
        let model_digest = canonical_sha256(&model).unwrap();
        let baseline = held_out_vampe_score(&model, &assignments, &validation_pairs, 2).unwrap();
        let mut perturbed = assignments.clone();
        perturbed[7] = 1;
        perturbed[8] = 0;
        let refit_from_same_training =
            fit_train_vampe_model(&perturbed, &train_pairs, 2, 1.0e-6).unwrap();
        assert_eq!(
            model_digest,
            canonical_sha256(&refit_from_same_training).unwrap()
        );
        let changed = held_out_vampe_score(&model, &perturbed, &validation_pairs, 2).unwrap();
        assert_ne!(baseline.to_bits(), changed.to_bits());
    }

    #[test]
    fn not_evaluable_interior_row_breaks_longer_lag_transition() {
        // The scheduled t+1 row was NOT_EVALUABLE, so t and t+2 occupy
        // different maximal evaluable segments even though both endpoints
        // have complete measurements.
        let split = vec![surveillance_sample(0, 0), surveillance_sample(2, 1)];
        assert!(surveillance_position_pairs_for_lag(&split, 2)
            .unwrap()
            .is_empty());
        let contiguous = vec![surveillance_sample(0, 0), surveillance_sample(2, 0)];
        assert_eq!(
            surveillance_position_pairs_for_lag(&contiguous, 2)
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn reversible_model_rejects_cross_phase_or_physical_state_pooling() {
        assert!(require_one_stationary_surveillance_stratum([
            (0, digest(30).as_str()),
            (1, digest(30).as_str()),
        ])
        .is_err());
        assert!(require_one_stationary_surveillance_stratum([
            (0, digest(30).as_str()),
            (0, digest(31).as_str()),
        ])
        .is_err());
    }

    #[test]
    fn created_state_absent_in_no_probe_is_zero_event_evidence_not_population() {
        let state_ids = vec![
            "baseline_a".to_string(),
            "baseline_b".to_string(),
            "fragment_created".to_string(),
        ];
        let mut samples = Vec::new();
        let mut assignments = Vec::new();
        for (lane_index, (lane, states)) in [
            (ControlLane::NoProbe, vec![0usize, 0, 1]),
            (ControlLane::HypothesisFragment, vec![0usize, 0, 1, 1, 2, 2]),
            (
                ControlLane::OrthogonalChemicalControl,
                vec![0usize, 0, 1, 1],
            ),
        ]
        .into_iter()
        .enumerate()
        {
            for (row_index, state) in states.into_iter().enumerate() {
                let mut sample = surveillance_sample((row_index as u64) * 25, 0);
                sample.control_lane = lane;
                sample.seed = (row_index % 2 + 1) as u64;
                sample.stream_id = lane_index as u16;
                sample.source_row_sha256 = digest(100 + samples.len() as u8);
                samples.push(sample);
                assignments.push(state);
            }
        }
        let mut config = NativePathBEstimatorConfig::default();
        config.minimum_state_effective_count = 2;
        config.minimum_state_seed_support = 2;
        config.minimum_transition_pairs = 1;
        let support =
            build_lane_global_state_support(&assignments, &samples, &state_ids, &config).unwrap();
        let absent = support
            .iter()
            .find(|row| {
                row.control_lane == ControlLane::NoProbe && row.state_id == "fragment_created"
            })
            .unwrap();
        assert_eq!(absent.status, LaneGlobalStateSupportStatus::NotObserved);
        assert_eq!(absent.scheduled_exposure_rows, 3);
        assert_eq!(absent.hard_assignment_rows, 0);
        assert_eq!(absent.independent_seed_support, 0);
        assert!(support.iter().any(|row| {
            row.control_lane == ControlLane::HypothesisFragment
                && row.state_id == "fragment_created"
                && row.status == LaneGlobalStateSupportStatus::Observed
        }));
        assert!(support.iter().any(|row| {
            row.control_lane == ControlLane::NoProbe
                && row.state_id == "baseline_b"
                && row.status == LaneGlobalStateSupportStatus::BelowSupport
        }));

        let active =
            active_global_state_indices(ControlLane::NoProbe, &state_ids, &support).unwrap();
        assert_eq!(active, vec![0]);
        let no_probe_pairs = vec![transition_pair(1, 0, 0, 0, 1)];
        let operator = fit_lane_operator(
            ControlLane::NoProbe,
            &no_probe_pairs,
            &no_probe_pairs,
            active,
            &config,
        )
        .unwrap();
        assert_eq!(operator.active_global_states, vec![0]);
        assert_eq!(operator.stationary, vec![1.0]);
        assert_eq!(operator.matrix, vec![vec![1.0]]);

        let lane_model = LaneTransitionModel {
            control_lane: ControlLane::NoProbe,
            active_state_ids: operator
                .active_global_states
                .iter()
                .map(|state| state_ids[*state].clone())
                .collect(),
            stationary_probability_by_state: operator
                .active_global_states
                .iter()
                .enumerate()
                .map(|(local, global)| (state_ids[*global].clone(), operator.stationary[local]))
                .collect(),
            transitions: operator
                .matrix
                .iter()
                .enumerate()
                .flat_map(|(from, row)| {
                    let active = &operator.active_global_states;
                    let state_ids = &state_ids;
                    row.iter()
                        .enumerate()
                        .map(move |(to, probability)| TransitionProbability {
                            from_state_id: state_ids[active[from]].clone(),
                            to_state_id: state_ids[active[to]].clone(),
                            probability: *probability,
                        })
                })
                .collect(),
        };
        assert!(!lane_model
            .stationary_probability_by_state
            .contains_key("fragment_created"));
        assert!(!lane_model
            .stationary_probability_by_state
            .contains_key("baseline_b"));
        assert!(lane_model.transitions.iter().all(|transition| {
            transition.from_state_id != "fragment_created"
                && transition.to_state_id != "fragment_created"
        }));
    }

    #[test]
    fn pathb_normalization_and_projection_are_native_and_deterministic() {
        let model = NativePathBModel::materialize(model_body(0.0)).unwrap();
        model.verify().unwrap();
        let projection = model.project(&model_lineage()).unwrap();
        assert_eq!(projection.metastable_state_id, "open");
        assert_eq!(
            projection.transition_from_state_id.as_deref(),
            Some("closed")
        );
        assert!((projection.transition_probability.unwrap() - 0.2).abs() < 1.0e-12);
        assert!(NativePathBModel::materialize(model_body(0.01)).is_err());
    }

    #[test]
    fn projection_uses_its_physical_lane_operator_not_the_reference_matrix() {
        let mut body = model_body(0.0);
        body.assignments[0].control_lane = ControlLane::HypothesisFragment;
        let lineage_sha256 = body.assignments[0].ghost_lineage_sha256.clone();
        for row in &mut body.scheduled_state_assignments {
            if matches!(row.ghost_event, ScheduledGhostEventStatus::Emitted { .. }) {
                row.ghost_event = ScheduledGhostEventStatus::NotEmitted {
                    typed_absence_receipt_sha256: digest(91),
                };
            } else if row.control_lane == ControlLane::HypothesisFragment
                && row.dominant_state_id == "open"
            {
                row.ghost_event = ScheduledGhostEventStatus::Emitted {
                    ghost_lineage_sha256: lineage_sha256.clone(),
                    exact_component_sha256: digest(92),
                };
            }
        }
        let hypothesis = body
            .lane_transition_models
            .iter_mut()
            .find(|model| model.control_lane == ControlLane::HypothesisFragment)
            .unwrap();
        for transition in &mut hypothesis.transitions {
            if transition.from_state_id == "closed" && transition.to_state_id == "closed" {
                transition.probability = 0.27;
            } else if transition.from_state_id == "closed" && transition.to_state_id == "open" {
                transition.probability = 0.73;
            }
        }
        let model = NativePathBModel::materialize(body).unwrap();
        let projection = model.project(&model_lineage()).unwrap();
        assert_eq!(projection.control_lane, ControlLane::HypothesisFragment);
        assert_eq!(projection.transition_probability, Some(0.73));
        assert_ne!(projection.transition_probability, Some(0.2));
    }

    #[test]
    fn model_validation_and_projection_accept_lane_subspace_for_created_state() {
        let mut body = model_body(0.0);
        let created = "fragment_created".to_string();
        body.states.push(MetastableState {
            state_id: created.clone(),
            stationary_probability: 0.0,
        });
        for assignment in &mut body.assignments {
            assignment.control_lane = ControlLane::HypothesisFragment;
            assignment
                .state_membership
                .insert("closed".to_string(), 0.1);
            assignment.state_membership.insert("open".to_string(), 0.1);
            assignment.state_membership.insert(created.clone(), 0.8);
        }
        let lineage_sha256 = body.assignments[0].ghost_lineage_sha256.clone();
        for row in &mut body.scheduled_state_assignments {
            row.state_membership.insert(created.clone(), 0.0);
            if matches!(row.ghost_event, ScheduledGhostEventStatus::Emitted { .. }) {
                row.ghost_event = ScheduledGhostEventStatus::NotEmitted {
                    typed_absence_receipt_sha256: digest(93),
                };
            }
            if row.control_lane == ControlLane::HypothesisFragment
                && row.dominant_state_id == "open"
            {
                row.state_membership.insert("closed".to_string(), 0.1);
                row.state_membership.insert("open".to_string(), 0.1);
                row.state_membership.insert(created.clone(), 0.8);
                row.dominant_state_id = created.clone();
                row.ghost_event = ScheduledGhostEventStatus::Emitted {
                    ghost_lineage_sha256: lineage_sha256.clone(),
                    exact_component_sha256: digest(94),
                };
            }
        }
        for evidence in &mut body.lane_global_state_support {
            if evidence.control_lane == ControlLane::HypothesisFragment
                && evidence.state_id == "open"
            {
                evidence.status = LaneGlobalStateSupportStatus::NotObserved;
                evidence.hard_assignment_rows = 0;
                evidence.independent_seed_support = 0;
            }
        }
        body.lane_global_state_support.extend([
            LaneGlobalStateSupportEvidence {
                control_lane: ControlLane::NoProbe,
                state_id: created.clone(),
                status: LaneGlobalStateSupportStatus::NotObserved,
                scheduled_exposure_rows: 2,
                hard_assignment_rows: 0,
                independent_seed_support: 0,
                minimum_required_assignment_rows: 1,
                minimum_required_independent_seeds: 1,
            },
            LaneGlobalStateSupportEvidence {
                control_lane: ControlLane::HypothesisFragment,
                state_id: created.clone(),
                status: LaneGlobalStateSupportStatus::Observed,
                scheduled_exposure_rows: 2,
                hard_assignment_rows: 1,
                independent_seed_support: 1,
                minimum_required_assignment_rows: 1,
                minimum_required_independent_seeds: 1,
            },
            LaneGlobalStateSupportEvidence {
                control_lane: ControlLane::OrthogonalChemicalControl,
                state_id: created.clone(),
                status: LaneGlobalStateSupportStatus::NotObserved,
                scheduled_exposure_rows: 2,
                hard_assignment_rows: 0,
                independent_seed_support: 0,
                minimum_required_assignment_rows: 1,
                minimum_required_independent_seeds: 1,
            },
        ]);
        let hypothesis = body
            .lane_transition_models
            .iter_mut()
            .find(|model| model.control_lane == ControlLane::HypothesisFragment)
            .unwrap();
        hypothesis.active_state_ids = vec!["closed".to_string(), created.clone()];
        hypothesis.stationary_probability_by_state =
            BTreeMap::from([("closed".to_string(), 0.6), (created.clone(), 0.4)]);
        for transition in &mut hypothesis.transitions {
            if transition.from_state_id == "open" {
                transition.from_state_id = created.clone();
            }
            if transition.to_state_id == "open" {
                transition.to_state_id = created.clone();
            }
        }
        for diagnostics in &mut body.diagnostics.lane_diagnostics {
            diagnostics
                .state_effective_counts
                .insert(created.clone(), 0);
            diagnostics.state_seed_support.insert(created.clone(), 0);
            if diagnostics.control_lane == ControlLane::HypothesisFragment {
                diagnostics.active_state_ids = vec!["closed".to_string(), created.clone()];
                diagnostics
                    .state_effective_counts
                    .insert("open".to_string(), 0);
                diagnostics
                    .state_effective_counts
                    .insert(created.clone(), 1);
                diagnostics.state_seed_support.insert("open".to_string(), 0);
                diagnostics.state_seed_support.insert(created.clone(), 1);
            }
        }
        body.diagnostics
            .state_effective_counts
            .insert(created.clone(), 1);
        body.diagnostics
            .state_seed_support
            .insert(created.clone(), 1);
        body.diagnostics.kinetic_model_selection.candidates[0].state_count = 3;
        body.diagnostics
            .kinetic_model_selection
            .selected_state_count = 3;
        body.diagnostics.kinetic_model_selection.ledger_sha256 =
            kinetic_selection_ledger_hash(&body.diagnostics.kinetic_model_selection).unwrap();

        let model = NativePathBModel::materialize(body).unwrap();
        model.verify().unwrap();
        let no_probe = model
            .body
            .lane_transition_models
            .iter()
            .find(|operator| operator.control_lane == ControlLane::NoProbe)
            .unwrap();
        assert!(!no_probe.active_state_ids.contains(&created));
        assert!(!no_probe
            .stationary_probability_by_state
            .contains_key(&created));
        assert!(no_probe.transitions.iter().all(|transition| {
            transition.from_state_id != created && transition.to_state_id != created
        }));
        assert_eq!(
            model
                .body
                .states
                .iter()
                .find(|state| state.state_id == created)
                .unwrap()
                .stationary_probability,
            0.0
        );
        let projection = model.project(&model_lineage()).unwrap();
        assert_eq!(projection.control_lane, ControlLane::HypothesisFragment);
        assert_eq!(projection.metastable_state_id, created);
        assert_eq!(
            projection.transition_from_state_id.as_deref(),
            Some("closed")
        );
        assert_eq!(projection.transition_probability, Some(0.2));
    }

    #[test]
    fn projection_preserves_assignment_without_fabricating_predecessor_transition() {
        let mut body = model_body(0.0);
        body.assignments[0].previous_state_id = None;
        let model = NativePathBModel::materialize(body).unwrap();
        let projection = model.project(&model_lineage()).unwrap();
        assert_eq!(projection.metastable_state_id, "open");
        assert_eq!(projection.transition_from_state_id, None);
        assert_eq!(projection.transition_to_state_id, None);
        assert_eq!(projection.transition_probability, None);
    }

    fn fiber(lane: ControlLane, seed: u64, step: u64, probability: f64) -> ChemicalStateFiber {
        let lambda = if lane == ControlLane::NoProbe {
            0.0
        } else {
            1.0
        };
        let mut body = crate::chemical_state_fiber::tests::body(lane, lambda, step, seed);
        body.site_key = "MYC_MAX_SITE_2".to_string();
        body.structural_response.open_state.probability = probability;
        body.structural_response.open_state.definition_sha256 = digest(9);
        body.structural_response.open_state.pathb_model_sha256 = digest(10);
        body.pathb.model_sha256 = digest(10);
        body.pathb.state_membership_probability = 0.9;
        body.pathb.transition_probability = Some(0.3);
        body.uncertainty.effective_sample_size = 30.0;
        body.uncertainty.integrated_autocorrelation_time_frames = 2.0;
        body.uncertainty.block_count = 10;
        body.uncertainty.standard_error = 0.01;
        body.uncertainty.confidence_interval_lower = probability - 0.02;
        body.uncertainty.confidence_interval_upper = probability + 0.02;
        ChemicalStateFiber::materialize(body).unwrap()
    }

    fn matched_fibers(no_probe: f64, hypothesis: f64, orthogonal: f64) -> Vec<ChemicalStateFiber> {
        let mut fibers = Vec::new();
        for seed in 1..=5 {
            let jitter = (seed as f64 - 3.0) * 0.002;
            for step in 100..132 {
                fibers.push(fiber(ControlLane::NoProbe, seed, step, no_probe + jitter));
                fibers.push(fiber(
                    ControlLane::HypothesisFragment,
                    seed,
                    step,
                    hypothesis + jitter,
                ));
                fibers.push(fiber(
                    ControlLane::OrthogonalChemicalControl,
                    seed,
                    step,
                    orthogonal + jitter,
                ));
            }
        }
        fibers
    }

    #[test]
    fn classifies_preexisting_stabilized_and_created_controls() {
        let config = MatchedControlConfig::default();
        let preexisting =
            classify_matched_controls(&matched_fibers(0.30, 0.31, 0.30), &config).unwrap();
        assert_eq!(
            preexisting.classification,
            PocketResponseClass::PreExistingTransient
        );
        preexisting.verify().unwrap();

        let stabilized =
            classify_matched_controls(&matched_fibers(0.30, 0.70, 0.32), &config).unwrap();
        assert_eq!(
            stabilized.classification,
            PocketResponseClass::HypothesisFragmentStabilized
        );

        let created =
            classify_matched_controls(&matched_fibers(0.03, 0.70, 0.04), &config).unwrap();
        assert_eq!(
            created.classification,
            PocketResponseClass::HypothesisSelectiveCreated
        );
        assert_eq!(created.common_seeds.len(), 5);
        assert_eq!(created.comparison_family_size, 3);
        assert!(created.overlap.no_probe_hypothesis < 1.0);
    }

    #[test]
    fn partial_lambda_and_missing_seed_fail_closed() {
        let config = MatchedControlConfig::default();
        let mut partial = matched_fibers(0.03, 0.70, 0.04);
        let hypothesis = partial
            .iter_mut()
            .find(|fiber| fiber.body.control_lane == ControlLane::HypothesisFragment)
            .unwrap();
        // Replace the complete native lambda observation rather than editing a
        // scalar in isolation.  A hash-bound fiber must remain internally
        // physical even when this test subsequently proves that an
        // intermediate lambda state is sampling-only and cannot enter the
        // matched endpoint classifier.
        hypothesis.body.fragment = crate::chemical_state_fiber::tests::body(
            ControlLane::HypothesisFragment,
            0.5,
            hypothesis.body.ghost.key.step_idx,
            hypothesis.body.uncertainty.replica_seed,
        )
        .fragment;
        *hypothesis = ChemicalStateFiber::materialize(hypothesis.body.clone()).unwrap();
        assert!(classify_matched_controls(&partial, &config).is_err());

        let mut missing = matched_fibers(0.03, 0.70, 0.04);
        missing.retain(|fiber| {
            !(fiber.body.control_lane == ControlLane::OrthogonalChemicalControl
                && fiber.body.uncertainty.replica_seed == 5)
        });
        assert!(classify_matched_controls(&missing, &config).is_err());
    }
}

#[cfg(test)]
mod lane_topology_atom_count_tests {
    use super::*;

    #[test]
    fn preserves_different_native_lane_topology_populations() {
        let no_probe = "01".repeat(32);
        let hypothesis = "02".repeat(32);
        let control = "03".repeat(32);
        let counts = collect_lane_topology_atom_counts([
            (ControlLane::NoProbe, no_probe.clone(), 4865),
            (ControlLane::HypothesisFragment, hypothesis.clone(), 4882),
            (ControlLane::OrthogonalChemicalControl, control.clone(), 4879),
            (ControlLane::NoProbe, no_probe.clone(), 4865),
        ]).unwrap();
        assert_eq!(counts[&ControlLane::NoProbe][&no_probe], 4865);
        assert_eq!(counts[&ControlLane::HypothesisFragment][&hypothesis], 4882);
        assert_eq!(counts[&ControlLane::OrthogonalChemicalControl][&control], 4879);
        let encoded = serde_json::to_vec(&counts).unwrap();
        let decoded: LaneTopologyAtomCounts = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(counts, decoded);
    }

    #[test]
    fn rejects_population_change_for_identical_topology_in_any_lane() {
        let topology = "01".repeat(32);
        assert!(collect_lane_topology_atom_counts([
            (ControlLane::NoProbe, topology.clone(), 4865),
            (ControlLane::NoProbe, topology.clone(), 4866),
        ]).is_err());
        assert!(collect_lane_topology_atom_counts([
            (ControlLane::NoProbe, topology.clone(), 4865),
            (ControlLane::HypothesisFragment, topology, 4866),
        ]).is_err());
    }

    #[test]
    fn rejects_absent_zero_and_unbound_coordinate_population() {
        assert!(collect_lane_topology_atom_counts([]).is_err());
        assert!(collect_lane_topology_atom_counts([(ControlLane::NoProbe, "01".repeat(32), 0)]).is_err());
        assert!(collect_lane_topology_atom_counts([(ControlLane::NoProbe, "unbound".into(), 4865)]).is_err());
    }
}
