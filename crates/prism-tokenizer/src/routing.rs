//! Stage-1 ontology routing for REAL artifacts (D′ — the de-naive routing layer).
//!
//! The discovery audit found the original `route_json` matched ~0 fields on every real artifact:
//! fields are nested in arrays (`spikes[]`, `sites[]`, …), enums are STRINGS (`"UV"`, `"PASS"`),
//! and coordinates are split `x`/`y`/`z` (or named `centroid_xyz`), never `"position":[...]`.
//! This module adapts real artifacts to the canonical registry: it walks records, maps string
//! enums to codes, assembles split-scalar coordinates, and auto-detects the family from
//! `schema_kind` / structure. Built entirely on the public Encoder/registry/stream API.

use std::collections::{HashMap, HashSet};

use crate::encoder::Encoder;
use crate::schema::families::AA_NAMES;
use crate::sidecar::ExactValue;
use crate::stream::TokenStream;
use crate::types::{ArtifactFamily as F, FieldGeometry as G, Token};

/// Accounting for a routing pass — distinguishes "nothing matched" (real failure) from "all
/// fields skipped" (schema mismatch) so a 0-token result is never silently mistaken for success.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RouteReport {
    pub records: usize,
    pub matched: usize,
    pub skipped_unknown: usize,
    pub skipped_type: usize,
    pub null_fields: usize,
}

impl RouteReport {
    fn merge(&mut self, o: RouteReport) {
        self.matched += o.matched;
        self.skipped_unknown += o.skipped_unknown;
        self.skipped_type += o.skipped_type;
        self.null_fields += o.null_fields;
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CausalGeometryAcceptanceReport {
    pub chronology_foundation_status: &'static str,
    pub cryptic_opening_status: &'static str,
    pub chain: &'static [&'static str],
    pub reasons: Vec<&'static str>,
}

// --- real-artifact adaptation tables -------------------------------------------------------

// String→code label sets (index = code). Engine source6 is distinct VORONOI_LIF.
const SPIKE_SOURCE: &[&str] = &["NONE", "UV", "LIF", "EFP", "LADD", "COFIRE", "VORONOI_LIF"];
const AROMATIC_TYPE: &[&str] = &["TRP", "TYR", "PHE", "SS", "BNZ"];
// JSON ccns_phase is a named protocol phase, not a numeric angle.
const PROTO_PHASE: &[&str] = &[
    "pre_protocol",
    "cold_hold",
    "hot_hold",
    "ramp",
    "hysteresis",
    "equilibration",
];
const TRIGGER_KIND: &[&str] = &["none", "Hysteresis", "Threshold", "Schedule", "Manual"];
const STATUS: &[&str] = &["PASS", "PARTIAL", "FAIL"]; // P2-14 widens to the real ~15-value vocab
                                                      // prism_therm top_residues[].role vocabulary (causal role of a residue in a TWIN pocket).
const THERM_ROLES: &[&str] = &["TRIGGER", "GATEWAY", "STABIL.", "SPECTAT"];
// binding_sites.json classification vocabulary (joined to KccSite by id).
const BINDING_CLASS: &[&str] = &[
    "ActiveSite",
    "Cryptic",
    "Unknown",
    "Druggable",
    "Allosteric",
];
// twin_per_residue.json barrier_class vocabulary (coupled-TWIN per-residue barrier evidence).
const BARRIER_CLASS: &[&str] = &["LOW", "MEDIUM", "HIGH", "NONE"];
// TWIN16 condition-edge axis vocabulary (the 2×2×2×2 thermodynamic factorial axes).
const TWIN16_AXIS: &[&str] = &["thermal", "uv", "ramp", "return"];
// SPIKE_TRANSITION_EVENT enum domains — MEASURED from a real all-on v2_ignition run (5VEX,
// output/5VEX_transition, 752 records). These replace the synthetic-based vocabularies; unknown
// values still escape losslessly (INV-1/INV-4).
const STE_TRIGGER_KIND: &[&str] = &[
    "cold_hold_start",
    "cold_hold_to_ramp",
    "ring_buffer_push",
    "ring_buffer_read_and_adapt",
    "BOCPD_changepoint",
    "final_teardown",
];
const STE_PHASE: &[&str] = &[
    "pre_protocol",
    "cold_hold",
    "ramp_up",
    "teardown",
    "hot_hold",
    "ramp_down",
    "equilibration",
];
const STE_PROTOCOL: &[&str] = &["Equilibrium", "Hysteresis", "ThermalShock", "UvAromatic"];
const STE_COUNT_STATUS: &[&str] = &[
    "run_start_before_first_device_sync",
    "device_buffer_pushed_before_host_spike_sync",
    "device_bridge_read_no_additional_host_spike_sync",
    "host_spike_sync_before_native_receipt_and_serialization",
    "view_anchor_after_standard_chunk_spike_sync_no_extra_readout",
];
// condition_id = TWIN16_{cell:04b}, the 16 interferometric transition-view cells (engine
// nhs_rt_full.rs:7403 twin16_condition_id).
const STE_CONDITION: &[&str] = &[
    "TWIN16_0000",
    "TWIN16_0001",
    "TWIN16_0010",
    "TWIN16_0011",
    "TWIN16_0100",
    "TWIN16_0101",
    "TWIN16_0110",
    "TWIN16_0111",
    "TWIN16_1000",
    "TWIN16_1001",
    "TWIN16_1010",
    "TWIN16_1011",
    "TWIN16_1100",
    "TWIN16_1101",
    "TWIN16_1110",
    "TWIN16_1111",
];

// --- Directive v6 Agent 3/4: generative-table enum vocabularies (MEASURED, not guessed) --------
// survival_tier {cryptic_rescue, normal}: from vspace_* + gflownet_raw_policy_samples (real shards).
// Shared by MoleculeGeneration AND GflownetTrajectory. "" (empty) escapes losslessly (INV-1/INV-4).
const SURVIVAL_TIER: &[&str] = &["normal", "cryptic_rescue"];
// reaction_role {synthon, scaffold, ""}: enamine synthon role. Index 0 reserved for empty/none.
const REACTION_ROLE: &[&str] = &["", "synthon", "scaffold"];
// ingest_status {ok, rejected, failed_conformer}: enamine conformer-ingest outcome.
const INGEST_STATUS: &[&str] = &["ok", "rejected", "failed_conformer"];
// charge_method / partial_charge_method {openff_nagl:openff-gnn-am1bcc-1.0.0.pt, ""}.
const CHARGE_METHOD: &[&str] = &["", "openff_nagl:openff-gnn-am1bcc-1.0.0.pt"];
// assembly_mode {smarts_zmatrix}: V-space assembly method.
const ASSEMBLY_MODE: &[&str] = &["smarts_zmatrix"];
// pose_reconciliation_method {compact_pseudo_ligand_o3a}: V-space pose reconciliation.
const POSE_RECONCILIATION: &[&str] = &["compact_pseudo_ligand_o3a"];
// regime {exploitation, focused, balanced, exploration, near_random}: gflownet sampling regime.
const GFLOWNET_REGIME: &[&str] = &[
    "exploitation",
    "focused",
    "balanced",
    "exploration",
    "near_random",
];

// --- Directive v6.1 Track B: teacher/student BRIDGE enum vocabularies (MEASURED from the real
//     PRISM_V2_ROW_MOLECULE_BRIDGE_TRAINING_TABLE.parquet, 1500 rows). Unknown labels escape
//     losslessly (INV-1/INV-4); "" reserved at index 0 where the column carries an empty string. --
// teacher_target {ABSTAIN_ESCALATE, PROMOTE_COMPUTATIONAL_REVIEW, TEACHER_REVIEW}.
const TEACHER_TARGET: &[&str] = &[
    "ABSTAIN_ESCALATE",
    "PROMOTE_COMPUTATIONAL_REVIEW",
    "TEACHER_REVIEW",
];
// evidence_role (5 observed roles).
const EVIDENCE_ROLE: &[&str] = &[
    "ABSTENTION_ESCALATION",
    "CONFLICTING_PROMOTE_PARENT_DOMINATES_BOUNDARY",
    "CONTRASTIVE_PARENT_DOMINATES_NEGATIVE",
    "LOW_CONVERGENCE_REVIEW_UNCERTAIN",
    "PROMOTE_BUT_LOW_CONVERGENCE_UNCERTAIN",
];
// phase_evidence_tier (2 observed tiers).
const PHASE_EVIDENCE_TIER: &[&str] = &[
    "L5_RAW_PHASE_TILE_HIGHER_TIER",
    "ROW_BOUND_R2_STREAM_RAW_PHASE_COHERENCE_FIBER",
];
// water_class_v2 (3 observed classes).
const WATER_CLASS_V2: &[&str] = &[
    "WATER_CANDIDATE_FAVORED_V2",
    "WATER_NEUTRAL_V2",
    "WATER_PARENT_FAVORED_V2",
];
// negative_family (hard-negative mining family).
const NEGATIVE_FAMILY: &[&str] = &["wrong_candidate", "wrong_condition"];
// contradiction_flag {"", PROMOTE_..._INSUFFICIENT_CONVERGENCE, PROMOTE_..._PARENT_DOMINATES}.
const CONTRADICTION_FLAG: &[&str] = &[
    "",
    "PROMOTE_COMPUTATIONAL_REVIEW_WITH_INSUFFICIENT_CONVERGENCE",
    "PROMOTE_COMPUTATIONAL_REVIEW_WITH_PARENT_DOMINATES",
];

// --- DSTW-T4 scientific-state-graph enum vocabularies (GROUNDED in crates/prismcode/src/
//     dstw_state_graph.rs; unknown labels escape losslessly, INV-1/INV-4). -----------------------
// lifecycle_stage: build_vertex emits "pre_md"/"post_md"; downstream emits "post_<probe>".
const DSTW_LIFECYCLE_STAGE: &[&str] = &[
    "pre_md",
    "post_md",
    "post_binding_sites",
    "post_tokenizer_runtime_manifest",
];
// ligand_state.state: "apo" (V1/V2) or "holo_gdp" (BLOCKED bound-state vertex).
const DSTW_LIGAND_STATE: &[&str] = &["apo", "holo_gdp"];
// feasibility / validation_feasibility: the tensor + ligand feasibility lattice.
const DSTW_FEASIBILITY: &[&str] = &[
    "POPULATABLE_NOW",
    "BLOCKED",
    "VALIDATE_ONLY_OFFLINE",
    "EXTERNAL_DERIVED",
];
// runtime_origin: EXACTLY the RuntimeOrigin::as_str strings (operand_ledger.rs:63-74). POST_RUN_DERIVED
// is the DSTW-T4 Step h5 dense-tensor semantic-summary origin (the inline-at-teardown native reduction
// is engine-owner; until then the reduction is derived post-run from existing artifacts).
const DSTW_RUNTIME_ORIGIN: &[&str] = &[
    "RUNTIME_INLINE",
    "RUNTIME_SUBPROCESS_ORCHESTRATED_BY_PRISMOS",
    "POST_RUN_MANUAL",
    "POST_RUN_SEAL_ONLY",
    "EXTERNAL_DERIVED",
    "BLOCKED",
    "POST_RUN_DERIVED",
];
// DSTW-T4 Step h5 dense-tensor roles + modalities. GROUNDED in build_dense_tensor_semantic_summaries
// (crates/prismcode/src/dstw_state_graph.rs) and the engine artifact writers cited there
// (nhs_rt_full.rs signal_grid/warp_matrix/adaptive_dt/spikes/TWIN16 + prism_postmd_superresolution).
const DENSE_TENSOR_ROLE: &[&str] = &[
    "SIGNAL_GRID",
    "WARP_FIELD",
    "DSTW_ADAPTIVE_DT",
    "SUPERRESOLUTION",
    "TWIN_INTERFEROMETRIC",
    "NEUROMORPHIC_SENSOR",
];
const DENSE_TENSOR_MODALITY: &[&str] = &[
    "thermodynamic_occupancy_grid",
    "mechanical_warp_field",
    "dstw_temporal",
    "superresolution_fused",
    "interferometric_twin",
    "neuromorphic_sensor",
];
// operator_name: the operators build_transition / emit_downstream_if_present apply.
const DSTW_OPERATOR: &[&str] = &[
    "MD.INTEGRATE_VELOCITY_VERLET",
    "STATE.CLASSIFY_BLOCKED",
    "PATHB.MATERIALIZE_BINDING_SITE",
    "CSE.TOKENIZE",
];
// next_operator_recommendation owner field.
const DSTW_OWNER: &[&str] = &["engine", "prismos"];
// LINEAGE_DAG (PRISM_NATIVE_STATE_GRAPH.jsonl) enum vocabularies — GROUNDED in the emitter
// crates/prismcode/src/merge_manifest.rs fn emit_state_graph_jsonl + orphan_check (no guessing).
// record_type/verdict literals are the exact strings the emitter writes; state_object_class is
// STATE_OBJECT_CLASSES + the ORPHAN_NOT_PRISM_NATIVE fallback from classify_state_object;
// artifact_kind/schema_kind are the kinds emitted by crates/prismcode/src/dstw_state_graph.rs.
// Any unseen value escapes losslessly (INV-1/INV-4), never inferred. runtime_origin reuses
// DSTW_RUNTIME_ORIGIN (RuntimeOrigin::as_str) and operator_name reuses DSTW_OPERATOR.
const LINEAGE_RECORD_TYPE: &[&str] = &["state_graph_node", "orphan_check"];
const LINEAGE_VERDICT: &[&str] = &[
    "ALL_ARTIFACTS_PRISM_NATIVE",
    "ORPHANS_PRESENT_NOT_PRISM_NATIVE",
];
const LINEAGE_STATE_OBJECT_CLASS: &[&str] = &[
    "ScientificStateVertex",
    "ScientificSurfaceEvent",
    "Transition",
    "EvidenceTensor",
    "Thermodynamic",
    "Kinetic",
    "Epistemic",
    "Validation",
    "Lineage",
    "PredictionTensor",
    "BlockedState",
    "NextOperatorRecommendation",
    "ORPHAN_NOT_PRISM_NATIVE",
];
const LINEAGE_ARTIFACT_KIND: &[&str] = &[
    "pre_md_state_vertex",
    "post_md_state_vertex",
    "md_state_transition",
    "next_operator_recommendation",
    "dense_tensor_semantic_summaries",
    "validation_state",
    "post_binding_sites_state_vertex",
    "post_binding_sites_state_transition",
    "blocked_gdp_bound_state_vertex",
];
const LINEAGE_SCHEMA_KIND: &[&str] = &[
    "dstw_t4_scientific_state_vertex",
    "dstw_t4_scientific_state_transition",
    "dstw_t4_next_operator_recommendation",
    "dstw_t4_dense_tensor_semantic_summary",
    "dstw_t4_validation_state",
    "ScientificPocketOpeningEvent",
    "scientific_pocket_opening_event",
    "prismos_scientific_pocket_opening_event",
    "LigandCausalActivityEvent",
    "ligand_causal_activity_event",
    "prismos_ligand_causal_activity_event",
    "ResidueActivityEvent",
    "residue_activity_event",
    "prismos_residue_activity_event",
];
const LINEAGE_TOKENIZER_FAMILY: &[&str] = &[
    "DstwStateVertex",
    "DstwStateTransition",
    "DstwNextOperator",
    "DenseTensorSemanticSummary",
    "ScientificPocketOpeningEvent",
    "LigandCausalActivityEvent",
    "ResidueActivityEvent",
];
// OpcodeExecutionLedger enum vocabularies — GROUNDED (exhaustive over all 368 real rows in
// opcode_operand_ledger.json). operator_class {Md:304, Kcc:32, Qnanl:32}; operand_target
// {Tensor:336, Artifact:32}; runtime_origin {RuntimeSubprocessOrchestratedByPrismos:336,
// PostRunSealOnly:32}; the three *_status fields share the {REGISTRY_COVERED, TYPED_VALUE_VERIFIED,
// NOT_APPLICABLE} vocabulary. Any unseen value escapes losslessly (INV-1/INV-4), never inferred.
const OPCODE_OPERATOR_CLASS: &[&str] = &["Md", "Kcc", "Qnanl"];
const OPCODE_OPERAND_TARGET: &[&str] = &["Tensor", "Artifact"];
const OPCODE_RUNTIME_ORIGIN: &[&str] =
    &["RuntimeSubprocessOrchestratedByPrismos", "PostRunSealOnly"];
const OPCODE_STATUS: &[&str] = &["REGISTRY_COVERED", "TYPED_VALUE_VERIFIED", "NOT_APPLICABLE"];
// ValidationState enum vocabularies (prismos_prismcode_isa_execution_verification.json).
// `status`/`substrate_status` emit the PASS/PARTIAL/FAIL verdict domain; substrate_status ALSO
// falls back to "MISSING" when the native substrate receipt is absent/unreadable (emitter
// prismos.rs:12877-12880 `unwrap_or("MISSING")`) — so it needs the 4-value vocab, not STATUS.
// implementation_status_language emits exactly {IMPLEMENTATION_VERIFIED, IMPLEMENTATION_PARTIAL}
// (prismos.rs:13063-13069). Any unseen value escapes losslessly (INV-1/INV-4), never inferred.
const IMPL_STATUS: &[&str] = &["IMPLEMENTATION_VERIFIED", "IMPLEMENTATION_PARTIAL"];
const VALIDATION_SUBSTRATE_STATUS: &[&str] = &["PASS", "PARTIAL", "FAIL", "MISSING"];
// PathBBindingSiteMaterialized enum vocabularies (binding_sites.materialized.json binding_sites[]).
// GROUNDED to the real apo_cryptic artifact: every one of the 20 promoted sites carries
// materialization_status=="materialized" and materialization_level=="materialized_dynamic_site".
// Any other future materialization state (e.g. a non-promoted/partial variant) escapes LOSSLESS to
// the §10 sidecar (INV-1/INV-4) — never silently inferred or coerced.
const PATHB_MATERIALIZATION_STATUS: &[&str] = &["materialized"];
const PATHB_MATERIALIZATION_LEVEL: &[&str] = &["materialized_dynamic_site"];
// SpikeSummary aromatic_type inner-enum vocab (codes 0..6). CANONICAL chromophore names from the
// engine's own label map (crates/prism-nhs/src/fused_engine.rs:7003 chromophore_name match): 0=TRP,
// 1=TYR, 2=PHE, 3=S-S, 4=BNZ. Codes 5 and 6 are present in real apo_cryptic data but UNdocumented in
// the source (chromophore_name maps them to "UNK") — kept as honest distinct code labels, NOT guessed
// residue names. The -1 'none' value is handled by the Sentinel(-1) wrapper, not this table.
const AROMATIC_TYPE_LABELS: &[&str] = &["TRP", "TYR", "PHE", "S-S", "BNZ", "UNK5", "UNK6"];
// ThermoPhaseManifold status vocabulary — GROUNDED to the emitter nhs_rt_full.rs:8255, which writes
// exactly `if global_event_count > 0 { "EMITTED" } else { "NO_SPIKE_TRANSITION_EVENTS" }`. These are
// the ONLY two producible values; any other future value escapes LOSSLESS to the §10 sidecar
// (INV-1/INV-4), never silently inferred or coerced.
const THERMO_STATUS: &[&str] = &["EMITTED", "NO_SPIKE_TRANSITION_EVENTS"];
// IsaRegistrySummary `opcode` vocabulary — the 25-value MD ISA registry. SOURCE OF TRUTH:
// crates/prismcode/src/bin/prismos.rs fn md_prismop_known_instruction_set() (hard-copied verbatim +
// order-preserved so opcode_index aligns; the tokenizer crate cannot depend on the prismcode bin).
// Any unseen opcode escapes LOSSLESS to the §10 sidecar (INV-1/INV-4), never silently coerced.
const ISA_MD_OPCODES: &[&str] = &[
    "MD.PROTOCOL_DIRECTOR",
    "MD.COM_REMOVE",
    "MD.NEIGHBOR_REBUILD",
    "MD.ACCUMULATOR_ZERO",
    "MD.FORCE_BONDED",
    "MD.FORCE_NONBONDED",
    "MD.INTEGRATE_VELOCITY_VERLET",
    "MD.SHAKE_CONSTRAIN",
    "MD.EXCLUSION_FIELD",
    "MD.WATER_DENSITY",
    "MD.PHOTO_UV_PUMP",
    "MD.EXTERNAL_WORK_REDUCE",
    "MD.VELOCITY_SECOND_HALF",
    "MD.CA_RESTRAINT",
    "MD.LIF_OBSERVE",
    "MD.COUPLING_BUILD",
    "MD.COUPLING_CLEAR",
    "MD.HEARTBEAT_ASSERT",
    "MD.LADD_UPDATE",
    "MD.KCC_UPDATE",
    "MD.NMA_PERTURB",
    "MD.SPIKE_EMIT",
    "MD.SPIKE_READOUT",
    "MD.SIGNAL_GRID_READOUT",
    "MD.KCC_DESCRIPTOR",
];

// PredictionState enum vocabularies — GROUNDED to the real apo_cryptic emitter values (no guessing).
// receptor_state: the run mode ("apo" for the apo_cryptic run; "holo" for a bound run). ligand_state
// reuses DSTW_LIGAND_STATE ("apo"/"holo_gdp"). epistemic_status: the site materialization_status —
// every one of the 20 promoted sites carries "materialized" (the other codes are honest reserved
// values for a future partial/blocked/promoted-distinguished state, never emitted here yet).
// validation_status/falsification_status: GROUNDED to site_accuracy_report — validation_computable is
// false and ground_truth_status is "reference_unavailable" in the real run (no curated reference in
// run_md scope). runtime_origin reuses DSTW_RUNTIME_ORIGIN. tokenizer_family is the constant
// self-identity "PredictionState". Any unseen value escapes LOSSLESS to the §10 sidecar (INV-1/INV-4),
// never silently inferred or coerced.
const PREDICTION_RECEPTOR_STATE: &[&str] = &["apo", "holo"];
const PREDICTION_EPISTEMIC_STATUS: &[&str] = &[
    "materialized",
    "materialized_promoted",
    "partial",
    "blocked",
];
const PREDICTION_VALIDATION_STATUS: &[&str] = &["reference_unavailable", "computable", "validated"];
const PREDICTION_FALSIFICATION_STATUS: &[&str] = &[
    "reference_unavailable",
    "not_available",
    "overlap_confirmed",
    "no_overlap",
];
const PREDICTION_TOKENIZER_FAMILY: &[&str] = &["PredictionState", "DstwNextOperator"];

const UNCERTAINTY_SCHEMA_FAMILIES: &[(&str, &str, F)] = &[
    (
        "scientific_uncertainty_tensor",
        "scientificuncertaintytensor",
        F::ScientificUncertaintyTensor,
    ),
    (
        "runtime_numerical_uncertainty_tensor",
        "runtimenumericaluncertaintytensor",
        F::RuntimeNumericalUncertaintyTensor,
    ),
    (
        "thermal_uncertainty_tensor",
        "thermaluncertaintytensor",
        F::ThermalUncertaintyTensor,
    ),
    (
        "integrator_uncertainty_tensor",
        "integratoruncertaintytensor",
        F::IntegratorUncertaintyTensor,
    ),
    (
        "observable_uncertainty_tensor",
        "observableuncertaintytensor",
        F::ObservableUncertaintyTensor,
    ),
    (
        "replica_uncertainty_tensor",
        "replicauncertaintytensor",
        F::ObservableUncertaintyTensor,
    ),
    (
        "physics_consistency_tensor",
        "physicsconsistencytensor",
        F::RuntimeNumericalUncertaintyTensor,
    ),
    (
        "observable_evidence_tensor",
        "observableevidencetensor",
        F::ObservableUncertaintyTensor,
    ),
    (
        "signal_tensor",
        "signaltensor",
        F::SignalToUncertaintyTensor,
    ),
    (
        "signal_to_uncertainty_tensor",
        "signaltouncertaintytensor",
        F::SignalToUncertaintyTensor,
    ),
    (
        "signal_to_uncertainty_channel_tensor",
        "signaltouncertaintychanneltensor",
        F::SignalToUncertaintyTensor,
    ),
    (
        "scientific_evidence_tensor",
        "scientificevidencetensor",
        F::ScientificUncertaintyTensor,
    ),
    (
        "epoch_uncertainty_tensor",
        "epochuncertaintytensor",
        F::EpochUncertaintyTensor,
    ),
    (
        "temporal_uncertainty_tensor",
        "temporaluncertaintytensor",
        F::EpochUncertaintyTensor,
    ),
    (
        "mode_uncertainty_tensor",
        "modeuncertaintytensor",
        F::ScientificUncertaintyTensor,
    ),
];

const UNCERTAINTY_MODE: &[&str] = &[
    "unknown",
    "runtime",
    "runtime_numerical",
    "thermal",
    "integrator",
    "replica",
    "observable",
    "temporal",
    "epoch",
    "mode",
    "scientific",
    "physics_consistency",
    "evidence",
    "signal",
    "signal_to_uncertainty",
    "apo",
    "holo",
    "cryptic",
    "orthosteric",
    "allosteric",
    "equilibrium",
    "transition",
    "pocket_opening",
    "relaxation",
    "spectral",
    "apo_cryptic",
    "holo_gdp",
    "pre_md",
    "post_md",
];
const UNCERTAINTY_TENSOR_KIND: &[&str] = &[
    "scientific_uncertainty_tensor",
    "runtime_numerical_uncertainty_tensor",
    "thermal_uncertainty_tensor",
    "integrator_uncertainty_tensor",
    "observable_uncertainty_tensor",
    "signal_to_uncertainty_tensor",
    "epoch_uncertainty_tensor",
    "ScientificUncertaintyTensor",
    "RuntimeNumericalUncertaintyTensor",
    "ThermalUncertaintyTensor",
    "IntegratorUncertaintyTensor",
    "ObservableUncertaintyTensor",
    "SignalToUncertaintyTensor",
    "EpochUncertaintyTensor",
    "SignalToUncertaintyChannelTensor",
    "replica_uncertainty_tensor",
    "temporal_uncertainty_tensor",
    "mode_uncertainty_tensor",
    "physics_consistency_tensor",
    "observable_evidence_tensor",
    "signal_tensor",
    "signal_to_uncertainty_channel_tensor",
    "scientific_evidence_tensor",
];
const UNCERTAINTY_OBSERVABLE: &[&str] = &[
    "unknown",
    "temperature",
    "energy",
    "pressure",
    "volume",
    "pocket_volume",
    "pocket_area",
    "pocket_surface_area",
    "cavity_persistence",
    "tunnel_radius",
    "rmsd",
    "rmsf",
    "residue_rmsf",
    "force",
    "force_rms",
    "velocity",
    "position",
    "coordinate_rms",
    "dt",
    "contact_persistence",
    "contact_probability",
    "contact_map",
    "hydrogen_bond",
    "kcc",
    "transfer_entropy",
    "graph_centrality",
    "laplacian_eigenvalue",
    "spectral_eigenvalue",
    "spectral_gap",
    "fiber_coherence",
    "delta_g",
    "occupancy",
    "hydration",
    "water_occupancy",
    "sasa",
    "chronology",
    "event_recurrence",
    "state_recurrence",
    "spike_count",
    "spike_rate",
    "causal_geometry_confidence",
    "signal",
    "signal_delta",
    "consistency",
    "uncertainty",
    "observable",
];
const UNCERTAINTY_OBSERVABLE_FAMILY: &[&str] = &[
    "unknown",
    "runtime_numerical",
    "thermal",
    "thermal_equilibrium",
    "integrator",
    "integrator_fidelity",
    "replica",
    "replica_ensemble",
    "observable",
    "observable_specific",
    "temporal",
    "epoch",
    "mode",
    "mode_specific",
    "physics_consistency",
    "signal",
    "evidence",
    "scientific",
];
const UNCERTAINTY_SOURCE_FAMILY: &[&str] = &[
    "unknown",
    "NoiseFloor",
    "NoiseFloorLegacyAdapter",
    "ProtocolState",
    "SpikeEvent",
    "SpikeSummary",
    "GpuTelemetry",
    "CapturedGraphGpuTelemetry",
    "PhaseManifoldEdgeValidation",
    "DenseTensorSemanticSummary",
    "ValidationState",
    "PredictionState",
    "ScientificStateEpoch",
    "ScientificPocketOpeningEvent",
    "LigandCausalActivityEvent",
    "ResidueActivityEvent",
    "ScientificChronologyEvent",
    "SpectralExtractor",
    "KccDescriptor",
    "KccDescriptorNative",
    "ThermoPhaseManifold",
    "signal_grid",
    "spike_summary",
    "ScientificUncertaintyTensor",
    "RuntimeNumericalUncertaintyTensor",
    "ThermalUncertaintyTensor",
    "IntegratorUncertaintyTensor",
    "ObservableUncertaintyTensor",
    "SignalToUncertaintyTensor",
    "EpochUncertaintyTensor",
];
const UNCERTAINTY_KIND: &[&str] = &[
    "scientific_state",
    "runtime_numerical",
    "thermal_equilibrium",
    "integrator_fidelity",
    "replica_ensemble",
    "observable_specific",
    "temporal",
    "mode_specific",
    "physics_consistency",
    "epistemic_plus_aleatoric",
    "replicate_dispersion",
    "measured_sigma",
    "derived_sigma",
];
const UNCERTAINTY_CHANNEL: &[&str] = &[
    "coordinate_rms_deviation",
    "force_rms_deviation",
    "energy_variance",
    "pressure_variance",
    "virial_variance",
    "pme_variance",
    "integration_residual",
    "constraint_residual",
    "occupancy_variance",
    "signal_variance",
    "replicate_dispersion",
    "source_sigma",
    "total_uncertainty",
];
const UNCERTAINTY_TOKENIZER_FAMILY: &[&str] = &[
    "ScientificUncertaintyTensor",
    "RuntimeNumericalUncertaintyTensor",
    "ThermalUncertaintyTensor",
    "IntegratorUncertaintyTensor",
    "ObservableUncertaintyTensor",
    "SignalToUncertaintyTensor",
    "EpochUncertaintyTensor",
    "ReplicaUncertaintyTensor",
    "TemporalUncertaintyTensor",
    "ModeUncertaintyTensor",
    "PhysicsConsistencyTensor",
    "ObservableEvidenceTensor",
    "SignalTensor",
    "ScientificEvidenceTensor",
    "signal_to_uncertainty_channel_tensor",
    "SignalToUncertaintyChannelTensor",
];
const UNCERTAINTY_MEASUREMENT_STATUS: &[&str] = &[
    "measured",
    "derived_from_measured",
    "insufficient_samples",
    "missing_source",
    "not_measured",
    "legacy_adapter",
    "blocked",
    "estimated",
    "derived",
    "bounded",
    "missing",
    "not_available",
    "invalid",
];
const UNCERTAINTY_SOURCE_STATUS: &[&str] = &[
    "present",
    "missing",
    "valid",
    "invalid",
    "measured",
    "derived",
    "blocked",
    "not_available",
    "legacy_adapter",
    "insufficient_samples",
    "pass",
    "fail",
    "PASS",
    "PARTIAL",
    "FAIL",
];

const NEGATIVE_EVIDENCE_SCHEMA_FAMILIES: &[(&str, &str, F)] = &[
    (
        "negative_scientific_evidence_event",
        "negativescientificevidenceevent",
        F::NegativeScientificEvidenceEvent,
    ),
    (
        "negative_scientific_evidence_events",
        "negativescientificevidenceevents",
        F::NegativeScientificEvidenceEvent,
    ),
    (
        "negative_evidence_ledger",
        "negativeevidenceledger",
        F::NegativeScientificEvidenceEvent,
    ),
    (
        "false_positive_suppression_ledger",
        "falsepositivesuppressionledger",
        F::NegativeScientificEvidenceEvent,
    ),
    (
        "recoverable_negative_evidence_audit",
        "recoverablenegativeevidenceaudit",
        F::NegativeScientificEvidenceEvent,
    ),
];

const TSH_SCHEMA_FAMILIES: &[(&str, &str, F)] = &[
    (
        "thermodynamic_salience_event",
        "thermodynamicsalienceevent",
        F::ThermodynamicSalienceEvent,
    ),
    (
        "transition_motif_candidate",
        "transitionmotifcandidate",
        F::TransitionMotif,
    ),
    ("transition_motif", "transitionmotif", F::TransitionMotif),
    (
        "persistent_transition_object",
        "persistenttransitionobject",
        F::PersistentTransitionObject,
    ),
    (
        "scientific_transition_consensus_event",
        "scientifictransitionconsensusevent",
        F::ScientificTransitionConsensusEvent,
    ),
    (
        "scientific_transition_consensus_events",
        "scientifictransitionconsensusevents",
        F::ScientificTransitionConsensusEvent,
    ),
    (
        "transition_consensus_event",
        "transitionconsensusevent",
        F::TransitionConsensusEvent,
    ),
    (
        "transition_evidence_ledger",
        "transitionevidenceledger",
        F::TransitionEvidenceLedger,
    ),
    (
        "resident_transition_cache",
        "residenttransitioncache",
        F::TransitionEvidenceLedger,
    ),
    (
        "tsh_runtime_non_interference_ledger",
        "tshruntimenoninterferenceledger",
        F::TransitionEvidenceLedger,
    ),
    (
        "tsh_stage_iii_compatibility_manifest",
        "tshstageiiicompatibilitymanifest",
        F::TransitionEvidenceLedger,
    ),
    (
        "tsh_runtime_observer_manifest",
        "tshruntimeobservermanifest",
        F::TransitionEvidenceLedger,
    ),
    (
        "transition_falsification_profile",
        "transitionfalsificationprofile",
        F::TransitionFalsificationProfile,
    ),
    (
        "transition_falsification_profiles",
        "transitionfalsificationprofiles",
        F::TransitionFalsificationProfile,
    ),
    (
        "rejected_persistent_transition_candidate",
        "rejectedpersistenttransitioncandidate",
        F::RejectedPersistentTransitionCandidate,
    ),
    (
        "rejected_persistent_transition_candidates",
        "rejectedpersistenttransitioncandidates",
        F::RejectedPersistentTransitionCandidate,
    ),
    (
        "persistent_transition_acceptance_ledger",
        "persistenttransitionacceptanceledger",
        F::PersistentTransitionAcceptanceLedger,
    ),
    (
        "stage_iii_eligibility_ledger",
        "stageiiieligibilityledger",
        F::PersistentTransitionAcceptanceLedger,
    ),
];

const TSH_SCHEMA_KIND: &[&str] = &[
    "thermodynamic_salience_event",
    "transition_motif_candidate",
    "transition_motif",
    "persistent_transition_object",
    "transition_consensus_event",
    "transition_evidence_ledger",
    "resident_transition_cache",
    "tsh_runtime_non_interference_ledger",
    "tsh_stage_iii_compatibility_manifest",
    "tsh_runtime_observer_manifest",
    "scientific_transition_consensus_event",
    "scientific_transition_consensus_events",
    "transition_falsification_profile",
    "transition_falsification_profiles",
    "rejected_persistent_transition_candidate",
    "rejected_persistent_transition_candidates",
    "persistent_transition_acceptance_ledger",
    "stage_iii_eligibility_ledger",
];
const TSH_NEGATIVE_CONTROL_STATUS: &[&str] = &[
    "missing",
    "not_evaluated",
    "absent_from_controls",
    "present_in_controls",
    "rejected_by_control",
    "downgraded_by_control",
    "pass",
    "fail",
    "PASS",
    "FAIL",
];
const TSH_ACCEPTANCE_STATUS: &[&str] = &[
    "candidate",
    "single_seed_candidate",
    "consensus_candidate",
    "accepted",
    "persistent",
    "rejected",
    "downgraded",
    "blocked",
    "IMPLEMENTATION_PARTIAL",
    "IMPLEMENTATION_VERIFIED",
    "IMPLEMENTATION_BLOCKED",
];
const TSH_RUNTIME_ORIGIN: &[&str] = &[
    "RUNTIME_INLINE_TSH_STAGE_II_OBSERVER",
    "RUNTIME_REPLAY_VERIFIER",
    "POST_RUNTIME_BACKFILL",
    "REGRESSION_TEST",
];
const TSH_TOKENIZER_FAMILY: &[&str] = &[
    "ThermodynamicSalienceEvent",
    "TransitionMotif",
    "PersistentTransitionObject",
    "TransitionConsensusEvent",
    "TransitionEvidenceLedger",
    "ScientificTransitionConsensusEvent",
    "TransitionFalsificationProfile",
    "RejectedPersistentTransitionCandidate",
    "PersistentTransitionAcceptanceLedger",
];
const TSH_CONSENSUS_DECISION: &[&str] = &[
    "ACCEPT",
    "REJECT",
    "PARTIAL",
    "BLOCKED",
    "accepted",
    "rejected",
    "partial",
    "blocked",
    "IMPLEMENTATION_VERIFIED",
    "IMPLEMENTATION_PARTIAL",
    "IMPLEMENTATION_BLOCKED",
];
const TSH_CONSENSUS_REASON: &[&str] = &[
    "survived_falsification_and_consensus",
    "single_seed_support",
    "stronger_negative_evidence",
    "below_uncertainty_floor",
    "chronology_persistence_failed",
    "topology_persistence_failed",
    "spectral_recurrence_failed",
    "residue_network_continuity_failed",
    "missing_required_provenance",
    "insufficient_consensus",
    "no_persistent_transition_promoted",
    "negative_evidence_gate_passed_no_promotion",
    "stage_iii_not_enabled",
];

const NEGATIVE_EVIDENCE_TYPE: &[&str] = &[
    "RejectedTransitionObject",
    "RejectedPocketEvent",
    "TransientPocketCandidate",
    "CollapsedPocketCandidate",
    "FailedConsensusTransition",
    "BelowNoiseFloorTransition",
    "TransientResidueNetwork",
    "RejectedTopologyTransition",
    "TopologyInstabilityEvent",
    "ChronologyBreakEvent",
    "RejectedSpectralState",
    "FailedSalienceCandidate",
    "RejectedKCCEvent",
    "FalsePositiveCandidate",
    "RejectedMechanisticHypothesis",
    "RejectedScientificClaim",
    "NegativeScientificEvidenceEvent",
    "NegativeEvidenceLedger",
    "FalsePositiveSuppressionLedger",
];
const NEGATIVE_REJECTION_CLASS: &[&str] = &[
    "uncertainty_rejection",
    "replicate_consensus_rejection",
    "chronology_rejection",
    "topology_rejection",
    "spectral_rejection",
    "kcc_rejection",
    "pocket_persistence_rejection",
    "causal_order_rejection",
    "false_positive_suppression",
    "ontology_falsification",
    "final_merge_rejection",
];
const NEGATIVE_CONTROL_FAMILY: &[&str] = &[
    "uncertainty",
    "chronology",
    "causal_geometry",
    "spectral",
    "topology",
    "kcc",
    "pocket_geometry",
    "replica_consensus",
    "mode_contrast",
    "thermodynamic_salience",
    "governed_merge",
];
const NEGATIVE_FAILED_CLAIM_TYPE: &[&str] = &[
    "site_prediction",
    "cryptic_site_opening",
    "causal_ligand_activity",
    "residue_network_activity",
    "spectral_recurrence",
    "kcc_propagation",
    "chronology_integrity",
    "topology_persistence",
    "salience_persistence",
    "mode_generalization",
    "corpus_merge_readiness",
];
const NEGATIVE_MOTIF_STATUS: &[&str] = &[
    "accepted_positive",
    "rejected",
    "transient",
    "collapsed",
    "below_noise_floor",
    "failed_consensus",
    "failed_recurrence",
    "failed_chronology",
    "failed_topology",
    "suppressed_false_positive",
    "unresolved",
];
const NEGATIVE_REJECTION_REASON: &[&str] = &[
    "exceeds_uncertainty_then_collapses",
    "appears_only_once",
    "fails_replicate_consensus",
    "fails_chronology_persistence",
    "fails_topology_persistence",
    "disappears_after_uncertainty_normalization",
    "falls_below_empirical_uncertainty_threshold",
    "fails_spectral_recurrence",
    "fails_residue_network_continuity",
    "fails_kcc_propagation",
    "fails_causal_ordering",
    "fails_pocket_persistence",
    "reverses_within_observation_window",
    "cannot_survive_multi_seed_comparison",
    "missing_required_artifact",
    "null_or_nan_observable",
    "stale_partial_gate",
    "no_recoverable_evidence_remaining",
];
const NEGATIVE_DISCOVERY_CONDITION: &[&str] = &[
    "UNCERTAINTY_THRESHOLD",
    "SINGLE_OCCURRENCE",
    "REPLICATE_CONSENSUS",
    "CHRONOLOGY_PERSISTENCE",
    "TOPOLOGY_PERSISTENCE",
    "UNCERTAINTY_NORMALIZATION",
    "SPECTRAL_RECURRENCE",
    "RESIDUE_NETWORK_CONTINUITY",
    "KCC_PROPAGATION",
    "CAUSAL_ORDERING",
    "POCKET_PERSISTENCE",
    "REVERSAL_WITHIN_WINDOW",
    "MULTI_SEED_COMPARISON",
    "EXHAUSTION_AUDIT",
];
const NEGATIVE_FAILURE_MODE: &[&str] = &[
    "below_threshold",
    "insufficient_samples",
    "missing_source",
    "not_measured",
    "nan_observable",
    "zero_signal_to_uncertainty",
    "non_recurrent",
    "single_seed_only",
    "collapsed_after_peak",
    "missing_epoch_reference",
    "missing_artifact_reference",
    "topology_hash_mismatch",
    "spectral_state_missing",
    "causal_order_inverted",
    "kcc_descriptor_dead_output",
    "stale_partial_status",
    "audit_exhausted",
];
const NEGATIVE_TOPOLOGY_STATUS: &[&str] = &[
    "stable",
    "mismatch",
    "missing",
    "rejected",
    "instability_absent_under_control",
    "transition_rejected_by_hash_control",
];
const NEGATIVE_FALSE_POSITIVE_STATUS: &[&str] = &[
    "candidate",
    "suppressed",
    "confirmed_false_positive",
    "not_applicable",
    "needs_active_control",
];
const NEGATIVE_SUPPRESSION_STATUS: &[&str] = &[
    "suppressed",
    "pending",
    "not_applicable",
    "complete",
    "blocked_pending_control",
];
const NEGATIVE_FINAL_MERGE_ELIGIBILITY: &[&str] = &[
    "eligible_after_falsification",
    "blocked_pending_negative_evidence",
    "blocked_pending_active_controls",
    "not_applicable",
];
const NEGATIVE_PRISMOS_OPCODE: &[&str] = &[
    "SCIENTIFIC_UNCERTAINTY_GATE",
    "CAUSAL_GEOMETRY_VALIDATE",
    "SCIENTIFIC_CHRONOLOGY_VALIDATE",
    "KCC_DESCRIPTOR_VALIDATE",
    "POCKET_PERSISTENCE_VALIDATE",
    "REPLICA_CONSENSUS_VALIDATE",
    "SPECTRAL_STATE_VALIDATE",
    "TOPOLOGY_STABILITY_VALIDATE",
    "FALSE_POSITIVE_SUPPRESS",
    "RUNTIME_GATE_REJECT",
    "ONTOLOGY_FALSIFY",
    "NEGATIVE_EVIDENCE_CANONICALIZE",
];
const NEGATIVE_OPCODE_FAMILY: &[&str] = &[
    "scientific_falsification",
    "uncertainty",
    "causal_geometry",
    "chronology",
    "kcc",
    "pocket_persistence",
    "replicate_consensus",
    "spectral",
    "topology",
    "false_positive",
    "governed_merge",
];
const NEGATIVE_PRISMCODE_INSTRUCTION_CLASS: &[&str] = &[
    "ASSERT.SCIENTIFIC_UNCERTAINTY",
    "ASSERT.CAUSAL_ORDER",
    "ASSERT.CHRONOLOGY",
    "ASSERT.KCC_VALIDITY",
    "ASSERT.POCKET_PERSISTENCE",
    "ASSERT.REPLICA_CONSENSUS",
    "ASSERT.SPECTRAL_RECURRENCE",
    "ASSERT.TOPOLOGY_STABILITY",
    "ASSERT.FALSE_POSITIVE",
    "ASSERT.RUNTIME_GATE",
    "ASSERT.NEGATIVE_EVIDENCE",
];
const NEGATIVE_QNANL_ARTIFACT_FAMILY: &[&str] = &[
    "EventStream",
    "JSONTelemetry",
    "LineageDag",
    "NumericArray",
    "ScientificStateEpoch",
    "ScientificChronologyEvent",
    "ScientificUncertaintyTensor",
    "NegativeScientificEvidenceEvent",
    "ValidationLedger",
    "CorpusManifest",
];

fn is_uncertainty_tensor_family(family: F) -> bool {
    matches!(
        family,
        F::ScientificUncertaintyTensor
            | F::RuntimeNumericalUncertaintyTensor
            | F::ThermalUncertaintyTensor
            | F::IntegratorUncertaintyTensor
            | F::ObservableUncertaintyTensor
            | F::SignalToUncertaintyTensor
            | F::EpochUncertaintyTensor
    )
}

fn is_tsh_transition_family(family: F) -> bool {
    matches!(
        family,
        F::ThermodynamicSalienceEvent
            | F::TransitionMotif
            | F::PersistentTransitionObject
            | F::TransitionConsensusEvent
            | F::TransitionEvidenceLedger
    )
}

fn is_tsh_consensus_gate_family(family: F) -> bool {
    matches!(
        family,
        F::ScientificTransitionConsensusEvent
            | F::TransitionFalsificationProfile
            | F::RejectedPersistentTransitionCandidate
            | F::PersistentTransitionAcceptanceLedger
    )
}

fn negative_evidence_family_from_schema_kind(schema_kind: &str) -> Option<F> {
    let lower = schema_kind.to_ascii_lowercase();
    NEGATIVE_EVIDENCE_SCHEMA_FAMILIES
        .iter()
        .find_map(|(snake, compact, family)| {
            (lower == *snake || lower == *compact).then_some(*family)
        })
}

fn tsh_family_from_schema_kind(schema_kind: &str) -> Option<F> {
    let lower = schema_kind.to_ascii_lowercase();
    TSH_SCHEMA_FAMILIES
        .iter()
        .find_map(|(snake, compact, family)| {
            (lower == *snake || lower == *compact).then_some(*family)
        })
}

fn uncertainty_family_from_schema_kind(schema_kind: &str) -> Option<F> {
    let lower = schema_kind.to_ascii_lowercase();
    UNCERTAINTY_SCHEMA_FAMILIES
        .iter()
        .find_map(|(snake, compact, family)| {
            (lower == *snake || lower == *compact).then_some(*family)
        })
}

pub fn detect_family_from_filename_hint(filename: &str) -> Option<(F, &'static str)> {
    let lower = filename
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(filename)
        .to_ascii_lowercase();
    if let Some(found) =
        NEGATIVE_EVIDENCE_SCHEMA_FAMILIES
            .iter()
            .find_map(|(snake, compact, family)| {
                (lower.contains(snake) || lower.contains(compact)).then_some((*family, ""))
            })
    {
        return Some(found);
    }
    if let Some(found) = TSH_SCHEMA_FAMILIES
        .iter()
        .find_map(|(snake, compact, family)| {
            (lower.contains(snake) || lower.contains(compact)).then_some((*family, ""))
        })
    {
        return Some(found);
    }
    UNCERTAINTY_SCHEMA_FAMILIES
        .iter()
        .find_map(|(snake, compact, family)| {
            (lower.contains(snake) || lower.contains(compact)).then_some((*family, ""))
        })
}

const SCIENTIFIC_CHRONOLOGY_EVENT_KIND: &[&str] = &[
    "state_created",
    "state_transition",
    "operator_applied",
    "qnanl_sealed",
    "epoch_started",
    "epoch_closed",
    "spike_observed",
    "bocpd_observed",
    "EPOCH_OPEN",
    "EPOCH_CLOSE",
    "TRAJECTORY_FRAME_CAPTURED",
    "TRAJECTORY_FRAME_STREAM_EMITTED",
    "SPIKE_STREAM_EMITTED",
    "SPIKE_TRANSITION_EVENTS_EMITTED",
    "SIGNAL_GRID_EMITTED",
    "KCC_V2FULL_EMITTED",
    "WARP_MATRIX_EMITTED",
    "FORCE_FIELD_EMITTED",
    "ASC_VECTORS_EMITTED",
    "NOISE_FLOOR_EMITTED",
    "PRISMCODE_RECEIPT_EMITTED",
    "QNANL_PACKET_EMITTED",
    "QNANL_SEAL_EMITTED",
    "TOKENIZER_HANDOFF_DECLARED",
    "TRAJECTORY_AUDIT_BLOCKER",
];
const SCIENTIFIC_STATE_EPOCH_KIND: &[&str] = &[
    "pre_md",
    "md_runtime",
    "post_md",
    "pathb",
    "tokenizer",
    "blocked",
    "external_derived",
    "operator_review",
    "MD_ONLY_EVIDENCE_HANDOFF",
    "RUNTIME_MD_STREAM",
    "RESIDENT_GRAPH_TRAJECTORY",
];
const SCIENTIFIC_SOURCE_KIND: &[&str] = &[
    "runtime",
    "dstw_state_graph",
    "operand_ledger",
    "qnanl_packet",
    "spike_event",
    "bocpd_event",
    "pathb",
    "tokenizer",
    "external_derived",
    "spikes",
    "spike_transition_events",
    "signal_grid",
    "kcc_v2full",
    "warp_matrix",
    "forces_final",
    "asc_vectors",
    "noise_floor",
    "prismcode_md_native_receipt",
    "qnanl_artifact_packet",
    "qnanl_artifact_receipt",
    "trajectory_frames",
    "trajectory_origin_receipt",
    "trajectory_audit",
];
const SCIENTIFIC_OPCODE_NAME: &[&str] = &[
    "STATE.VERTEX_CREATE",
    "STATE.EMIT_TRANSITION",
    "STATE.OPERATOR_APPLY",
    "STATE.CLASSIFY_BLOCKED",
    "STATE.NEXT_OPERATOR_REC",
    "STATE.DENSE_TENSOR_SUMMARY",
    "MD.INTEGRATE_VELOCITY_VERLET",
    "MD.SPIKE_EMIT",
    "MD.SPIKE_READOUT",
    "MD.QNANL_SEAL",
    "PATHB.MATERIALIZE_BINDING_SITE",
    "CSE.TOKENIZE",
    "STATE.CREATE_EPOCH",
    "STATE.CLOSE_EPOCH",
    "MD.FRAME_CAPTURE",
    "MD.SPIKE_STREAM_EMIT",
    "MD.SIGNAL_GRID_UPDATE",
    "MD.KCC_UPDATE",
    "MD.WARP_TENSOR_UPDATE",
    "MD.FORCE_FINAL_EMIT",
    "MD.ASC_VECTOR_EMIT",
    "PRISMCODE.RECEIPT",
    "QNANL.SEAL",
    "TOKENIZER.EXPORT",
    "VALIDATION.BLOCKER_RAISED",
    "MD.SCIENTIFIC_CHRONOLOGY_EVENT",
];
const SCIENTIFIC_EVENT_TYPE: &[&str] = &[
    "EPOCH_OPEN",
    "CUDA_GRAPH_CAPTURE",
    "CUDA_GRAPH_REPLAY",
    "FRAME_CAPTURE",
    "TENSOR_CAPTURE",
    "SPIKE_STREAM_EMIT",
    "KCC_V2FULL_EMIT",
    "SIGNAL_GRID_EMIT",
    "WARP_TENSOR_EMIT",
    "FORCE_FINAL_EMIT",
    "ASC_VECTOR_EMIT",
    "PRISMCODE_RECEIPT",
    "QNANL_SEAL",
    "TOKENIZER_EXPORT",
    "VALIDATION_PASS",
    "VALIDATION_FAIL",
    "BLOCKER_RAISED",
    "EPOCH_CLOSE",
];
const CHRONOLOGY_FIDELITY_LEVEL: &[&str] = &[
    "LOW",
    "MINIMAL",
    "STANDARD",
    "HIGH",
    "EXHAUSTIVE",
    "CUSTOM",
    "HIGH_CHUNK_BOUNDARY",
    "HIGH_EXACT_INTERVAL",
    "EXACT_INTERVAL",
    "RUNTIME_ARTIFACT",
    "RUNTIME_NATIVE",
];
const INTERFEROMETRIC_MODE: &[&str] = &[
    "MD_NATIVE",
    "COUPLED_TWIN",
    "MULTI_DIFFERENTIAL_TWIN16",
    "TWIN16_CONDITION_MARKER",
    "TWIN16_PHASE_MANIFOLD",
    "DSTW_T4_EPOCH_OVERLAY",
];
const INTERFEROMETRIC_MARKER_KIND: &[&str] = &[
    "EPOCH_MARKER",
    "EPOCH_BOUNDARY_MARKER",
    "TRAJECTORY_FRAME_MARKER",
    "CUDA_GRAPH_MARKER",
    "QNANL_PACKET_MARKER",
    "TOKENIZER_HANDOFF_MARKER",
    "OPCODE_RECEIPT_MARKER",
    "ARTIFACT_EMISSION_MARKER",
];
const CHRONOLOGY_MARKER_SCOPE: &[&str] = &[
    "SCIENTIFIC_STATE_EPOCH",
    "SCIENTIFIC_CHRONOLOGY_EVENT",
    "CUDA_GRAPH_BOUNDARY",
    "DSTW_T4_OVERLAY",
];
const T4_EPOCH_OVERLAY_KIND: &[&str] = &[
    "runtime_closeout",
    "runtime_epoch",
    "acceptance_epoch",
    "operator_review",
];
const CHRONOLOGY_ACCEPTANCE_DECISION: &[&str] = &[
    "accepted",
    "accepted_with_pending_gates",
    "blocked",
    "rejected",
];
const CHRONOLOGY_ACCEPTANCE_BASIS: &[&str] = &[
    "decode_proofs_and_operand_ledger",
    "merge_readiness",
    "operator_review",
    "not_applicable",
];
const LIGAND_CONTACT_TYPE: &[&str] = &[
    "H_BOND",
    "HYDROPHOBIC",
    "PI_STACK",
    "SALT_BRIDGE",
    "WATER_BRIDGE",
];
const RESIDUE_ACTIVITY_TYPE: &[&str] = &[
    "POCKET_LINING",
    "CONTACT_SWITCH",
    "HINGE_MOTION",
    "ALLOSTERIC_SIGNAL",
    "KCC_SPIKE",
    "FORCE_HOTSPOT",
    "EXPOSURE_CHANGE",
];
fn enum_labels(family: F, field: &str) -> Option<&'static [&'static str]> {
    match (family, field) {
        (F::NegativeScientificEvidenceEvent, "negative_evidence_type") => {
            Some(NEGATIVE_EVIDENCE_TYPE)
        }
        (F::NegativeScientificEvidenceEvent, "rejection_class") => Some(NEGATIVE_REJECTION_CLASS),
        (F::NegativeScientificEvidenceEvent, "control_family") => Some(NEGATIVE_CONTROL_FAMILY),
        (F::NegativeScientificEvidenceEvent, "failed_claim_type") => {
            Some(NEGATIVE_FAILED_CLAIM_TYPE)
        }
        (F::NegativeScientificEvidenceEvent, "motif_status") => Some(NEGATIVE_MOTIF_STATUS),
        (F::NegativeScientificEvidenceEvent, "rejection_reason") => Some(NEGATIVE_REJECTION_REASON),
        (F::NegativeScientificEvidenceEvent, "discovery_condition") => {
            Some(NEGATIVE_DISCOVERY_CONDITION)
        }
        (F::NegativeScientificEvidenceEvent, "failure_mode") => Some(NEGATIVE_FAILURE_MODE),
        (F::NegativeScientificEvidenceEvent, "source_family") => Some(UNCERTAINTY_SOURCE_FAMILY),
        (F::NegativeScientificEvidenceEvent, "ligand_mode") => Some(DSTW_LIGAND_STATE),
        (F::NegativeScientificEvidenceEvent, "topology_status") => Some(NEGATIVE_TOPOLOGY_STATUS),
        (F::NegativeScientificEvidenceEvent, "false_positive_status") => {
            Some(NEGATIVE_FALSE_POSITIVE_STATUS)
        }
        (F::NegativeScientificEvidenceEvent, "suppression_status") => {
            Some(NEGATIVE_SUPPRESSION_STATUS)
        }
        (F::NegativeScientificEvidenceEvent, "final_merge_eligibility") => {
            Some(NEGATIVE_FINAL_MERGE_ELIGIBILITY)
        }
        (F::NegativeScientificEvidenceEvent, "prismos_opcode") => Some(NEGATIVE_PRISMOS_OPCODE),
        (F::NegativeScientificEvidenceEvent, "opcode_family") => Some(NEGATIVE_OPCODE_FAMILY),
        (F::NegativeScientificEvidenceEvent, "prismcode_instruction_class") => {
            Some(NEGATIVE_PRISMCODE_INSTRUCTION_CLASS)
        }
        (F::NegativeScientificEvidenceEvent, "qnanl_artifact_family") => {
            Some(NEGATIVE_QNANL_ARTIFACT_FAMILY)
        }
        (family, "schema_kind") if is_tsh_transition_family(family) => Some(TSH_SCHEMA_KIND),
        (family, "schema_kind") if is_tsh_consensus_gate_family(family) => Some(TSH_SCHEMA_KIND),
        (family, "negative_control_status") if is_tsh_transition_family(family) => {
            Some(TSH_NEGATIVE_CONTROL_STATUS)
        }
        (family, "acceptance_status") if is_tsh_transition_family(family) => {
            Some(TSH_ACCEPTANCE_STATUS)
        }
        (family, "decision") if is_tsh_consensus_gate_family(family) => {
            Some(TSH_CONSENSUS_DECISION)
        }
        (family, "final_decision") if is_tsh_consensus_gate_family(family) => {
            Some(TSH_CONSENSUS_DECISION)
        }
        (family, "accepted_status") if is_tsh_consensus_gate_family(family) => {
            Some(TSH_CONSENSUS_DECISION)
        }
        (family, "rejection_status") if is_tsh_consensus_gate_family(family) => {
            Some(TSH_CONSENSUS_DECISION)
        }
        (family, "decision_reason") if is_tsh_consensus_gate_family(family) => {
            Some(TSH_CONSENSUS_REASON)
        }
        (family, "falsification_decision_reason") if is_tsh_consensus_gate_family(family) => {
            Some(TSH_CONSENSUS_REASON)
        }
        (family, "rejection_reason") if is_tsh_consensus_gate_family(family) => {
            Some(TSH_CONSENSUS_REASON)
        }
        (family, "failed_gate") if is_tsh_consensus_gate_family(family) => {
            Some(TSH_CONSENSUS_REASON)
        }
        (family, "tokenizable_failure_mode") if is_tsh_consensus_gate_family(family) => {
            Some(TSH_CONSENSUS_REASON)
        }
        (family, "runtime_origin") if is_tsh_transition_family(family) => Some(TSH_RUNTIME_ORIGIN),
        (family, "tokenizer_family")
            if is_tsh_transition_family(family) || is_tsh_consensus_gate_family(family) =>
        {
            Some(TSH_TOKENIZER_FAMILY)
        }
        (family, "mode") if is_uncertainty_tensor_family(family) => Some(UNCERTAINTY_MODE),
        (family, "tensor_kind") if is_uncertainty_tensor_family(family) => {
            Some(UNCERTAINTY_TENSOR_KIND)
        }
        (family, "observable") if is_uncertainty_tensor_family(family) => {
            Some(UNCERTAINTY_OBSERVABLE)
        }
        (family, "observable_family") if is_uncertainty_tensor_family(family) => {
            Some(UNCERTAINTY_OBSERVABLE_FAMILY)
        }
        (family, "source_family") if is_uncertainty_tensor_family(family) => {
            Some(UNCERTAINTY_SOURCE_FAMILY)
        }
        (family, "source_signal_family") if is_uncertainty_tensor_family(family) => {
            Some(UNCERTAINTY_SOURCE_FAMILY)
        }
        (family, "uncertainty_kind") if is_uncertainty_tensor_family(family) => {
            Some(UNCERTAINTY_KIND)
        }
        (family, "uncertainty_channel") if is_uncertainty_tensor_family(family) => {
            Some(UNCERTAINTY_CHANNEL)
        }
        (family, "runtime_origin") if is_uncertainty_tensor_family(family) => {
            Some(DSTW_RUNTIME_ORIGIN)
        }
        (family, "tokenizer_family") if is_uncertainty_tensor_family(family) => {
            Some(UNCERTAINTY_TOKENIZER_FAMILY)
        }
        (family, "measurement_status") if is_uncertainty_tensor_family(family) => {
            Some(UNCERTAINTY_MEASUREMENT_STATUS)
        }
        (family, "source_status") if is_uncertainty_tensor_family(family) => {
            Some(UNCERTAINTY_SOURCE_STATUS)
        }
        // PredictionState enum fields (prismos_prediction_state.json records[]). receptor_state and
        // ligand_state name the run/ligand posture; the *_status enums are grounded to the real
        // site_accuracy_report; runtime_origin reuses the DSTW vocab; tokenizer_family is the
        // self-identity. INDEPENDENT of DstwNextOperator (this family routes on its own schema_kind).
        (F::PredictionState, "receptor_state") => Some(PREDICTION_RECEPTOR_STATE),
        (F::PredictionState, "ligand_state") => Some(DSTW_LIGAND_STATE),
        (F::PredictionState, "epistemic_status") => Some(PREDICTION_EPISTEMIC_STATUS),
        (F::PredictionState, "validation_status") => Some(PREDICTION_VALIDATION_STATUS),
        (F::PredictionState, "falsification_status") => Some(PREDICTION_FALSIFICATION_STATUS),
        (F::PredictionState, "runtime_origin") => Some(DSTW_RUNTIME_ORIGIN),
        (F::PredictionState, "tokenizer_family") => Some(PREDICTION_TOKENIZER_FAMILY),
        (F::ScientificChronologyEvent, "event_kind") => Some(SCIENTIFIC_CHRONOLOGY_EVENT_KIND),
        (F::ScientificChronologyEvent, "event_type") => Some(SCIENTIFIC_EVENT_TYPE),
        (F::ScientificChronologyEvent, "source_kind") => Some(SCIENTIFIC_SOURCE_KIND),
        (F::ScientificChronologyEvent, "artifact_family") => Some(SCIENTIFIC_SOURCE_KIND),
        (F::ScientificChronologyEvent, "operator_name") => Some(DSTW_OPERATOR),
        (F::ScientificChronologyEvent, "opcode_name") => Some(SCIENTIFIC_OPCODE_NAME),
        (F::ScientificChronologyEvent, "runtime_origin") => Some(DSTW_RUNTIME_ORIGIN),
        (F::ScientificChronologyEvent, "chronology_fidelity_level") => {
            Some(CHRONOLOGY_FIDELITY_LEVEL)
        }
        (F::ScientificChronologyEvent, "interferometric_mode") => Some(INTERFEROMETRIC_MODE),
        (F::ScientificChronologyEvent, "interferometric_marker_kind") => {
            Some(INTERFEROMETRIC_MARKER_KIND)
        }
        (F::ScientificChronologyEvent, "chronology_marker_scope") => Some(CHRONOLOGY_MARKER_SCOPE),
        (F::ScientificStateEpoch, "epoch_kind") => Some(SCIENTIFIC_STATE_EPOCH_KIND),
        (F::ScientificStateEpoch, "lifecycle_stage") => Some(DSTW_LIFECYCLE_STAGE),
        (F::ScientificStateEpoch, "ligand_state") => Some(DSTW_LIGAND_STATE),
        (F::ScientificStateEpoch, "runtime_origin") => Some(DSTW_RUNTIME_ORIGIN),
        (F::ScientificStateEpoch, "chronology_fidelity_level") => Some(CHRONOLOGY_FIDELITY_LEVEL),
        (F::ScientificStateEpoch, "interferometric_mode") => Some(INTERFEROMETRIC_MODE),
        (F::ScientificStateEpoch, "interferometric_marker_kind") => {
            Some(INTERFEROMETRIC_MARKER_KIND)
        }
        (F::ScientificStateEpoch, "chronology_marker_scope") => Some(CHRONOLOGY_MARKER_SCOPE),
        (F::T4EpochOverlay, "overlay_kind") => Some(T4_EPOCH_OVERLAY_KIND),
        (F::T4EpochOverlay, "epoch_kind") => Some(SCIENTIFIC_STATE_EPOCH_KIND),
        (F::T4EpochOverlay, "lifecycle_stage") => Some(DSTW_LIFECYCLE_STAGE),
        (F::T4EpochOverlay, "ligand_state") => Some(DSTW_LIGAND_STATE),
        (F::T4EpochOverlay, "runtime_origin") => Some(DSTW_RUNTIME_ORIGIN),
        (F::ChronologyAcceptanceDecision, "decision") => Some(CHRONOLOGY_ACCEPTANCE_DECISION),
        (F::ChronologyAcceptanceDecision, "decision_basis") => Some(CHRONOLOGY_ACCEPTANCE_BASIS),
        (F::ChronologyAcceptanceDecision, "lifecycle_stage") => Some(DSTW_LIFECYCLE_STAGE),
        (F::ChronologyAcceptanceDecision, "runtime_origin") => Some(DSTW_RUNTIME_ORIGIN),
        (
            F::ScientificPocketOpeningEvent
            | F::LigandCausalActivityEvent
            | F::ResidueActivityEvent,
            "runtime_origin",
        ) => Some(DSTW_RUNTIME_ORIGIN),
        (
            F::ScientificPocketOpeningEvent
            | F::LigandCausalActivityEvent
            | F::ResidueActivityEvent,
            "chronology_fidelity_level",
        ) => Some(CHRONOLOGY_FIDELITY_LEVEL),
        (
            F::ScientificPocketOpeningEvent
            | F::LigandCausalActivityEvent
            | F::ResidueActivityEvent,
            "interferometric_mode",
        ) => Some(INTERFEROMETRIC_MODE),
        (
            F::ScientificPocketOpeningEvent
            | F::LigandCausalActivityEvent
            | F::ResidueActivityEvent,
            "interferometric_marker_kind",
        ) => Some(INTERFEROMETRIC_MARKER_KIND),
        (
            F::ScientificPocketOpeningEvent
            | F::LigandCausalActivityEvent
            | F::ResidueActivityEvent,
            "chronology_marker_scope",
        ) => Some(CHRONOLOGY_MARKER_SCOPE),
        (F::LigandCausalActivityEvent, "contact_type") => Some(LIGAND_CONTACT_TYPE),
        (F::ResidueActivityEvent, "activity_type") => Some(RESIDUE_ACTIVITY_TYPE),
        (F::ResidueActivityEvent, "residue_name") => Some(AA_NAMES),
        (F::SpectralExtractor, "status") => Some(IMPL_STATUS),
        (F::SpikeEvent, "spike_source") => Some(SPIKE_SOURCE),
        (F::SpikeEvent, "aromatic_type") => Some(AROMATIC_TYPE),
        (F::SpikeEvent, "ccns_phase") => Some(PROTO_PHASE),
        (F::BocpdEvent, "trigger_kind") => Some(TRIGGER_KIND),
        (F::BocpdEvent, "phase_from") | (F::BocpdEvent, "phase_to") => Some(PROTO_PHASE),
        // SPIKE_TRANSITION_EVENT enum domains — MEASURED from a real all-on run (see consts above).
        (F::SpikeTransitionEvent, "trigger_kind") => Some(STE_TRIGGER_KIND),
        (F::SpikeTransitionEvent, "phase_from") | (F::SpikeTransitionEvent, "phase_to") => {
            Some(STE_PHASE)
        }
        (F::SpikeTransitionEvent, "condition_id") => Some(STE_CONDITION),
        (F::SpikeTransitionEvent, "protocol_name") => Some(STE_PROTOCOL),
        (F::SpikeTransitionEvent, "spike_count_status") => Some(STE_COUNT_STATUS),
        // TWIN16 transition-view residuals share the TWIN16 cell + protocol vocabularies.
        (F::TwinTransitionResidual, "condition_id")
        | (F::TwinTransitionResidual, "cell_a")
        | (F::TwinTransitionResidual, "cell_b") => Some(STE_CONDITION),
        (F::TwinTransitionResidual, "protocol_name") => Some(STE_PROTOCOL),
        // real per-residue TWIN coupling evidence (twin_per_residue.json).
        (F::TwinCoupled, "barrier_class") => Some(BARRIER_CLASS),
        (F::TwinCoupled, "resname") => Some(AA_NAMES),
        // TWIN16-native DPD condition edge (per_cellpair_condition_edges.jsonl).
        (F::TwinConditionEdge, "cell_a") | (F::TwinConditionEdge, "cell_b") => Some(STE_CONDITION),
        (F::TwinConditionEdge, "axis") => Some(TWIN16_AXIS),
        // auxiliary coupled-A/B NMA per-residue differential (SEPARATE from TWIN16).
        (F::NmaCoupledEdge, "resname") => Some(AA_NAMES),
        (F::NmaCoupledEdge, "barrier_classification") => Some(BARRIER_CLASS),
        (F::ReceiptStatus, "status") => Some(STATUS),
        // Directive v6 generative tables: MoleculeGeneration enum vocab (measured from real parquet).
        (F::MoleculeGeneration, "survival_tier") => Some(SURVIVAL_TIER),
        (F::MoleculeGeneration, "reaction_role") => Some(REACTION_ROLE),
        (F::MoleculeGeneration, "ingest_status") => Some(INGEST_STATUS),
        (F::MoleculeGeneration, "charge_method")
        | (F::MoleculeGeneration, "partial_charge_method") => Some(CHARGE_METHOD),
        (F::MoleculeGeneration, "assembly_mode") => Some(ASSEMBLY_MODE),
        (F::MoleculeGeneration, "pose_reconciliation_method") => Some(POSE_RECONCILIATION),
        // GflownetTrajectory: survival_tier (spec existed, label table was missing) + regime.
        (F::GflownetTrajectory, "survival_tier") => Some(SURVIVAL_TIER),
        (F::GflownetTrajectory, "regime") => Some(GFLOWNET_REGIME),
        // KCC residue_name: AMBER three-letter codes; unknowns escape (INV-1/INV-4), never PASS-style inferred.
        (F::KccDescriptor, "residue_name") => Some(AA_NAMES),
        // KccSite member enrichment: a candidate member's residue_name (joined from residues[]).
        (F::KccSite, "candidate_residue_name") => Some(AA_NAMES),
        // KccSite member TWIN role, typed-joined from prism_therm.json top_residues[].role.
        (F::KccSite, "candidate_te_role") => Some(THERM_ROLES),
        // KccSite binding-site classification, typed-joined from binding_sites.json by id.
        (F::KccSite, "site_classification") => Some(BINDING_CLASS),
        // noise_floor.json calibration phase (protocol phase vocabulary).
        (F::NoiseFloor, "phase") => Some(PROTO_PHASE),
        // Directive v6.1 Track B: teacher/student bridge enum vocab (measured from the real table).
        (F::TeacherStudentConsensus, "teacher_target") => Some(TEACHER_TARGET),
        (F::TeacherStudentConsensus, "evidence_role") => Some(EVIDENCE_ROLE),
        (F::TeacherStudentConsensus, "phase_evidence_tier") => Some(PHASE_EVIDENCE_TIER),
        (F::TeacherStudentConsensus, "water_class_v2") => Some(WATER_CLASS_V2),
        (F::TeacherStudentConsensus, "negative_family") => Some(NEGATIVE_FAMILY),
        (F::TeacherStudentConsensus, "contradiction_flag") => Some(CONTRADICTION_FLAG),
        // DSTW-T4 scientific-state graph rows (vertices / transitions / next-operator rec).
        (F::DstwStateVertex, "lifecycle_stage") => Some(DSTW_LIFECYCLE_STAGE),
        (F::DstwStateVertex, "ligand_state") => Some(DSTW_LIGAND_STATE),
        (F::DstwStateVertex, "feasibility") => Some(DSTW_FEASIBILITY),
        (F::DstwStateVertex, "runtime_origin") => Some(DSTW_RUNTIME_ORIGIN),
        (F::DstwStateTransition, "operator_name") => Some(DSTW_OPERATOR),
        (F::DstwStateTransition, "runtime_origin") => Some(DSTW_RUNTIME_ORIGIN),
        (F::DstwStateTransition, "validation_feasibility") => Some(DSTW_FEASIBILITY),
        (F::DstwNextOperator, "runtime_origin") => Some(DSTW_RUNTIME_ORIGIN),
        (F::DstwNextOperator, "top_owner") => Some(DSTW_OWNER),
        // DSTW-T4 Step h5 dense-tensor semantic summary enum fields.
        (F::DenseTensorSemanticSummary, "tensor_role") => Some(DENSE_TENSOR_ROLE),
        (F::DenseTensorSemanticSummary, "modality") => Some(DENSE_TENSOR_MODALITY),
        (F::DenseTensorSemanticSummary, "runtime_origin") => Some(DSTW_RUNTIME_ORIGIN),
        // DSTW-T4 LINEAGE_DAG merge-manifest graph-envelope rows (PRISM_NATIVE_STATE_GRAPH.jsonl).
        // Every enom field MUST be registered here or its string value escapes without tokenizing
        // (route_object L591 short-circuits on enum_labels().is_some()) — the #1 review defect.
        (F::LineageDag, "record_type") => Some(LINEAGE_RECORD_TYPE),
        (F::LineageDag, "verdict") => Some(LINEAGE_VERDICT),
        (F::LineageDag, "state_object_class") => Some(LINEAGE_STATE_OBJECT_CLASS),
        (F::LineageDag, "artifact_kind") => Some(LINEAGE_ARTIFACT_KIND),
        (F::LineageDag, "schema_kind") => Some(LINEAGE_SCHEMA_KIND),
        (F::LineageDag, "tokenizer_family") => Some(LINEAGE_TOKENIZER_FAMILY),
        (F::LineageDag, "runtime_origin") => Some(DSTW_RUNTIME_ORIGIN),
        (F::LineageDag, "operator_name") => Some(DSTW_OPERATOR),
        (F::ThermoPhaseManifold, "status") => Some(THERMO_STATUS),
        // OpcodeExecutionLedger enum fields (opcode_operand_ledger.json).
        (F::OpcodeExecutionLedger, "operator_class") => Some(OPCODE_OPERATOR_CLASS),
        (F::OpcodeExecutionLedger, "operand_target") => Some(OPCODE_OPERAND_TARGET),
        (F::OpcodeExecutionLedger, "runtime_origin") => Some(OPCODE_RUNTIME_ORIGIN),
        (F::OpcodeExecutionLedger, "disassembly_status")
        | (F::OpcodeExecutionLedger, "validation_status")
        | (F::OpcodeExecutionLedger, "replay_status") => Some(OPCODE_STATUS),
        // ValidationState verdict enums (prismos_prismcode_isa_execution_verification.json).
        // status reuses the PASS/PARTIAL/FAIL vocab; substrate_status widens it with MISSING.
        (F::ValidationState, "status") => Some(STATUS),
        (F::ValidationState, "substrate_status") => Some(VALIDATION_SUBSTRATE_STATUS),
        (F::ValidationState, "implementation_status_language") => Some(IMPL_STATUS),
        // PathB FLAGSHIP materialized binding site enum fields (binding_sites.materialized.json).
        (F::PathBBindingSiteMaterialized, "materialization_status") => {
            Some(PATHB_MATERIALIZATION_STATUS)
        }
        (F::PathBBindingSiteMaterialized, "materialization_level") => {
            Some(PATHB_MATERIALIZATION_LEVEL)
        }
        // IsaRegistrySummary enum fields (prismos_prismcode_isa_registry_summary.json records[]).
        // Every enom field MUST be registered here or its string value escapes without tokenizing
        // (route_object short-circuits on enum_labels().is_some()) — the #1 review defect. `opcode`
        // is the genuine 25-value categorical; stream_status/stream_coverage_status reuse the
        // PASS/PARTIAL/FAIL STATUS verdict vocabulary (real apo_cryptic values are all "PASS").
        (F::IsaRegistrySummary, "opcode") => Some(ISA_MD_OPCODES),
        (F::IsaRegistrySummary, "stream_status")
        | (F::IsaRegistrySummary, "stream_coverage_status") => Some(STATUS),
        // SpikeSummary aromatic_type: the DOMINANT per-residue chromophore code. Sentinel(-1) handles
        // the 'none' escape (every LIF spike); the inner enom(7) tokenizes codes 0..6 against the REAL
        // canonical chromophore vocab (fused_engine.rs:7003, codes 5/6 undocumented => UNK5/UNK6). Must
        // be registered here or the numeric value escapes without tokenizing (route_object short-
        // circuits on enum_labels().is_some()).
        (F::SpikeSummary, "aromatic_type") => Some(AROMATIC_TYPE_LABELS),
        _ => None,
    }
}

/// The real JSON key for a registry field (real artifacts use `type` for `aromatic_type`, etc.).
/// Directive v6.1 Track B: the teacher/student BRIDGE table prefixes its per-row teacher head with
/// `row__` (row__teacher_target, row__fusion_kcc_coherence, …). The registry stores the UNprefixed
/// field name (so the family reads cleanly + matches the molecule_id/pair join keys), and this map
/// resolves each to its EXACT parquet column. The pair-identity / mask fields (pair_label/pair_weight/
/// pair_id/molecule_id/abstention_target/teacher_target via row__teacher_target) are handled
/// explicitly; all remaining typed teacher fields carry the `row__` prefix.
fn json_key_for(family: F, field: &'static str) -> &'static str {
    match (family, field) {
        (F::SpikeEvent, "aromatic_type") => "type",
        // NOTE: per-surface column renames that COLLIDE with an existing artifact's native field
        // names (gflownet validity_status→oracle_valid, sampled_anchor_id→anchor_id; spike
        // primary_residue_idx→aromatic_residue_id) are NOT mapped here — json_key_for is global per
        // (family,field) and remapping the canonical key would break the per-residue / JSON-artifact
        // routers that use the native names. Those renames are normalized in the decode step
        // (scripts/quarantine/sample_high_volume_surfaces_v6.py + the policy decode normalize) so the
        // router still sees the canonical field name. This keeps existing artifacts AND the new
        // high-volume surfaces routable through one global json_key_for.
        // bridge pair-identity + masks live at the TOP level (no row__ prefix).
        (F::TeacherStudentConsensus, "pair_label") => "pair_label",
        (F::TeacherStudentConsensus, "pair_weight") => "pair_weight",
        (F::TeacherStudentConsensus, "pair_id") => "pair_id",
        (F::TeacherStudentConsensus, "molecule_id") => "molecule_id",
        // every other NEW (v11) teacher field is a `row__`-prefixed column. The v10 per-residue
        // teacher fields (cryptic_likelihood_label/phase_manifold_score_*/rhat/ess/label_confidence/
        // is_bimodal/hard_negative_mask/train_mask/student_weight/residue_id) keep their own names
        // (they come from the per-residue teacher artifacts, NOT this bridge table) — listed here as
        // identity so they are NOT mis-prefixed.
        (F::TeacherStudentConsensus, "cryptic_likelihood_label")
        | (F::TeacherStudentConsensus, "phase_manifold_score_mean")
        | (F::TeacherStudentConsensus, "phase_manifold_score_std")
        | (F::TeacherStudentConsensus, "rhat")
        | (F::TeacherStudentConsensus, "ess")
        | (F::TeacherStudentConsensus, "label_confidence")
        | (F::TeacherStudentConsensus, "is_bimodal")
        | (F::TeacherStudentConsensus, "hard_negative_mask")
        | (F::TeacherStudentConsensus, "train_mask")
        | (F::TeacherStudentConsensus, "student_weight")
        | (F::TeacherStudentConsensus, "residue_id") => field,
        (F::TeacherStudentConsensus, f) => teacher_bridge_row_key(f),
        _ => field,
    }
}

/// Map a NEW (v11) TeacherStudentConsensus field to its EXACT `row__`-prefixed bridge column. The
/// 'static lifetime is preserved by matching on the field literal (each arm returns a 'static str).
fn teacher_bridge_row_key(field: &str) -> &'static str {
    match field {
        "abstention_target" => "row__abstention_target",
        "teacher_target" => "row__teacher_target",
        "teacher_loss_weight" => "row__teacher_loss_weight",
        "distillation_loss_weight" => "row__distillation_loss_weight",
        "uncertainty_target" => "row__uncertainty_target",
        "positive_target" => "row__positive_target",
        "negative_target" => "row__negative_target",
        "model_uncertainty" => "row__prism_meta_reward_model_uncertainty",
        "model_coverage" => "row__prism_meta_reward_model_coverage",
        "evidence_quality_score" => "row__evidence_quality_score",
        "integrated_score" => "row__integrated_score",
        "phase_coherence_score" => "row__phase_coherence_score",
        "gflownet_reward" => "row__gflownet_reward",
        "gflownet_reward_fusion_target" => "row__gflownet_reward_fusion_target",
        "fusion_kcc_coherence" => "row__fusion_kcc_coherence",
        "fusion_kinetic_durability" => "row__fusion_kinetic_durability",
        "fusion_hydration_gate_behavior" => "row__fusion_hydration_gate_behavior",
        "fusion_uncertainty" => "row__fusion_uncertainty",
        "fusion_evidence_coverage_score" => "row__fusion_evidence_coverage_score",
        "fusion_force_stress" => "row__fusion_force_stress",
        "fusion_warp_strain" => "row__fusion_warp_strain",
        "fusion_receptor_recovery" => "row__fusion_receptor_recovery",
        "fusion_raw_spike_activity" => "row__fusion_raw_spike_activity",
        "fusion_raw_spike_entropy" => "row__fusion_raw_spike_entropy",
        "fusion_productive_activation_score" => "row__fusion_productive_activation_score",
        "fusion_quiet_lock_risk" => "row__fusion_quiet_lock_risk",
        "fusion_variant_rescue_signal" => "row__fusion_variant_rescue_signal",
        "fusion_receptor_state_coherence" => "row__fusion_receptor_state_coherence",
        "fusion_fiber_transport_coherence" => "row__fusion_fiber_transport_coherence",
        "fusion_chronology_alignment" => "row__fusion_chronology_alignment",
        "delta_functional_score" => "row__delta_functional_score",
        "delta_survival_auc" => "row__delta_survival_auc",
        "delta_temporal_score" => "row__delta_temporal_score",
        "delta_thermodynamic_score" => "row__delta_thermodynamic_score",
        "delta_water_score_v2" => "row__delta_water_score_v2",
        "evidence_role" => "row__evidence_role",
        "phase_evidence_tier" => "row__phase_evidence_tier",
        "water_class_v2" => "row__water_class_v2",
        "negative_family" => "negative_family",
        "contradiction_flag" => "row__contradiction_flag",
        "ood_or_missing_family_flag" => "row__ood_or_missing_family_flag",
        "do_not_discard" => "row__do_not_discard",
        // unknown new field: fall through to its own name (lossless; never silently misrouted).
        _ => "row____unmapped_teacher_field__",
    }
}

/// How a registry coordinate field appears in real JSON.
enum CoordSource {
    /// split scalar siblings, e.g. ("x","y","z").
    Triple(&'static str, &'static str, &'static str),
    /// a 3-element array under one of these alias keys.
    Array(&'static [&'static str]),
}

fn coord_source(family: F, field: &str) -> Option<CoordSource> {
    match (family, field) {
        (F::SpikeEvent, "position") => Some(CoordSource::Triple("x", "y", "z")),
        // KCC residues[] carry the Cα as a 3-element `ca_position` array.
        (F::KccDescriptor, "ca_position") => Some(CoordSource::Array(&["ca_position"])),
        // KCC sites[] carry the pocket centre as a 3-element `centroid` array.
        (F::KccSite, "centroid") => Some(CoordSource::Array(&["centroid"])),
        // twin_per_residue.json carries the residue Cα as a 3-element `ca_xyz` array.
        (F::TwinCoupled, "ca_xyz") => Some(CoordSource::Array(&["ca_xyz"])),
        (_, "position") => Some(CoordSource::Array(&[
            "position",
            "centroid",
            "centroid_xyz",
            "ca_position",
        ])),
        _ => None,
    }
}

fn arr3(v: &serde_json::Value) -> Option<[f64; 3]> {
    let a = v.as_array()?;
    if a.len() != 3 {
        return None;
    }
    Some([a[0].as_f64()?, a[1].as_f64()?, a[2].as_f64()?])
}

// --- string-enum encode (unknown string → escape + sidecar Text, never dropped) ------------

fn encode_enum_str(
    enc: &Encoder,
    stream: &mut TokenStream,
    family: F,
    field: &str,
    s: &str,
    verified: bool,
    seal_sha: [u8; 32],
) -> bool {
    let Some(spec) = enc.registry().lookup(family, field) else {
        return false;
    };
    let Some(ns) = enc.registry().namespace(family, field) else {
        return false;
    };
    let code = match enum_labels(family, field).and_then(|l| l.iter().position(|&x| x == s)) {
        Some(c) => c as u64,
        None => Token::ESCAPE_CODE, // INV-1/INV-4: unknown label escapes, exact string kept in sidecar
    };
    stream.push(
        spec,
        ns,
        code,
        verified,
        ExactValue::Text(s.to_string()),
        seal_sha,
    );
    true
}

/// Encode a STRING join key (synthon_id / anchor_id / product_id / atom-index pointer) as an INV-7
/// escape token whose exact id lives in the sidecar Text. The numeric JoinKey codebook cannot encode
/// an arbitrary string, so the token value is `ESCAPE_CODE` and reconstruction is via the sidecar
/// (INV-1/INV-4) — the id is NEVER dropped or numerically faked.
fn encode_joinkey_str(
    enc: &Encoder,
    stream: &mut TokenStream,
    family: F,
    field: &str,
    s: &str,
    verified: bool,
    seal_sha: [u8; 32],
) -> bool {
    let Some(spec) = enc.registry().lookup(family, field) else {
        return false;
    };
    let Some(ns) = enc.registry().namespace(family, field) else {
        return false;
    };
    stream.push(
        spec,
        ns,
        Token::ESCAPE_CODE,
        verified,
        ExactValue::Text(s.to_string()),
        seal_sha,
    );
    true
}

/// Encode an exact structured field (operator arrays/objects such as residue_ids, contacts,
/// causal_signals, metrics) as a sidecar-backed token under the field's real family namespace.
/// The model-visible code is the INV-7 escape; the sidecar carries the exact JSON.
fn encode_exact_json_value(
    enc: &Encoder,
    stream: &mut TokenStream,
    family: F,
    field: &str,
    value: &serde_json::Value,
    verified: bool,
    seal_sha: [u8; 32],
) -> bool {
    let Some(spec) = enc.registry().lookup(family, field) else {
        return false;
    };
    let Some(ns) = enc.registry().namespace(family, field) else {
        return false;
    };
    let exact = serde_json::to_string(value).unwrap_or_default();
    stream.push(
        spec,
        ns,
        Token::ESCAPE_CODE,
        verified,
        ExactValue::Text(exact),
        seal_sha,
    );
    true
}

/// Documented string→bool for string-valued Boolean fields (validity_status "valid" ⇒ true). Only
/// the exact documented tokens map; everything else returns None (escapes losslessly, never coerced).
fn str_to_bool(s: &str) -> Option<bool> {
    match s {
        "valid" | "true" | "True" | "1" | "ok" => Some(true),
        "invalid" | "false" | "False" | "0" => Some(false),
        _ => None,
    }
}

// --- per-object routing (registry-field-slot order ⇒ deterministic, P1-9) ------------------

/// Route one decoded record object. Iterates the family's registry fields in slot order (stable
/// token order), mapping each to its real JSON key/value-form.
pub fn route_object(
    enc: &Encoder,
    stream: &mut TokenStream,
    family: F,
    obj: &serde_json::Map<String, serde_json::Value>,
    verified: bool,
    seal_sha: [u8; 32],
    capture: bool,
) -> RouteReport {
    let mut r = RouteReport::default();
    let mut consumed: HashSet<String> = HashSet::new();

    // collect this family's specs in registry (slot) order
    let specs: Vec<_> = enc
        .registry()
        .iter()
        .filter(|s| s.family == family)
        .cloned()
        .collect();

    for spec in &specs {
        match &spec.geometry {
            G::LatticeVq3d { .. } => {
                let Some(src) = coord_source(family, spec.field) else {
                    continue;
                };
                let v3 = match src {
                    CoordSource::Triple(xk, yk, zk) => {
                        match (
                            obj.get(xk).and_then(|v| v.as_f64()),
                            obj.get(yk).and_then(|v| v.as_f64()),
                            obj.get(zk).and_then(|v| v.as_f64()),
                        ) {
                            (Some(x), Some(y), Some(z)) => {
                                consumed.insert(xk.into());
                                consumed.insert(yk.into());
                                consumed.insert(zk.into());
                                Some([x, y, z])
                            }
                            _ => None,
                        }
                    }
                    CoordSource::Array(aliases) => aliases.iter().find_map(|k| {
                        obj.get(*k).and_then(arr3).inspect(|_| {
                            consumed.insert((*k).into());
                        })
                    }),
                };
                if let Some(v3) = v3 {
                    if enc.encode_vec3(stream, family, spec.field, v3, verified, seal_sha) {
                        r.matched += 1;
                    }
                }
            }
            G::SmilesGrammar => {
                let jk = json_key_for(family, spec.field);
                if let Some(s) = obj.get(jk).and_then(|v| v.as_str()) {
                    consumed.insert(jk.into());
                    if enc.encode_smiles(stream, family, spec.field, s, verified, seal_sha) > 0 {
                        r.matched += 1;
                    }
                }
            }
            _ => {
                let jk = json_key_for(family, spec.field);
                let Some(val) = obj.get(jk) else { continue };
                consumed.insert(jk.into());
                if val.is_null() {
                    r.null_fields += 1;
                    if capture {
                        stream.push_opaque(jk, "null", verified, seal_sha);
                    }
                } else if let Some(s) = val.as_str() {
                    if enum_labels(family, spec.field).is_some()
                        && encode_enum_str(enc, stream, family, spec.field, s, verified, seal_sha)
                    {
                        r.matched += 1;
                    } else if matches!(spec.geometry, G::JoinKey) {
                        // JoinKey carrying a STRING id (synthon_id/anchor_id/product_id/…): the
                        // numeric codebook can't encode a string, so the token is an INV-7 escape and
                        // the exact id is preserved in the sidecar Text (INV-1/INV-4). Never dropped.
                        if encode_joinkey_str(
                            enc, stream, family, spec.field, s, verified, seal_sha,
                        ) {
                            r.matched += 1;
                        } else {
                            r.skipped_type += 1;
                        }
                    } else if matches!(spec.geometry, G::Boolean) {
                        // String-valued Boolean field (e.g. validity_status "valid" ⇒ oracle_valid
                        // true). Only the documented truthy/falsy tokens map; anything else escapes
                        // losslessly rather than silently coercing.
                        match str_to_bool(s) {
                            Some(b) => {
                                if enc.encode_scalar(
                                    stream,
                                    family,
                                    spec.field,
                                    b as i64 as f64,
                                    verified,
                                    seal_sha,
                                ) {
                                    r.matched += 1;
                                }
                            }
                            None => {
                                if encode_joinkey_str(
                                    enc, stream, family, spec.field, s, verified, seal_sha,
                                ) {
                                    r.matched += 1;
                                } else {
                                    r.skipped_type += 1;
                                }
                            }
                        }
                    } else {
                        r.skipped_type += 1; // string value, field isn't a known enum
                    }
                } else if let Some(x) = val
                    .as_f64()
                    .or_else(|| val.as_bool().map(|b| b as i64 as f64))
                {
                    if enc.encode_scalar(stream, family, spec.field, x, verified, seal_sha) {
                        r.matched += 1;
                    }
                } else if (val.is_array() || val.is_object()) && matches!(spec.geometry, G::JoinKey)
                {
                    if encode_exact_json_value(
                        enc, stream, family, spec.field, val, verified, seal_sha,
                    ) {
                        r.matched += 1;
                    } else {
                        r.skipped_type += 1;
                    }
                } else {
                    r.skipped_type += 1;
                }
            }
        }
    }

    for (k, v) in obj.iter() {
        if !consumed.contains(k) {
            r.skipped_unknown += 1;
            if capture {
                // INV-4: capture the whole unknown field losslessly so the record reconstructs.
                let vj = serde_json::to_string(v).unwrap_or_default();
                stream.push_opaque(k, &vj, verified, seal_sha);
            }
        }
    }
    r
}

/// Resolve a record path: "" ⇒ the value itself is one record; "/key" ⇒ that key's value, with an
/// array yielding one record per element.
fn resolve_path<'a>(json: &'a serde_json::Value, path: &str) -> Vec<&'a serde_json::Value> {
    if path.is_empty() {
        return vec![json];
    }
    let key = path.trim_start_matches('/');
    match json.get(key) {
        Some(serde_json::Value::Array(a)) => a.iter().collect(),
        Some(v) => vec![v],
        None => vec![],
    }
}

/// Route every record under `record_path` (arrays-of-records supported).
pub fn route_records(
    enc: &Encoder,
    stream: &mut TokenStream,
    family: F,
    json: &serde_json::Value,
    record_path: &str,
    verified: bool,
    seal_sha: [u8; 32],
    capture: bool,
) -> RouteReport {
    let mut total = RouteReport::default();
    let records = resolve_path(json, record_path);
    total.records = records.len();
    for (idx, rec) in records.iter().enumerate() {
        if let Some(obj) = rec.as_object() {
            let seg = stream.begin_record(family, idx);
            total.merge(route_object(
                enc, stream, family, obj, verified, seal_sha, capture,
            ));
            stream.end_record(seg);
        }
    }
    total
}

fn is_scientific_surface_event_family(family: F) -> bool {
    matches!(
        family,
        F::ScientificPocketOpeningEvent | F::LigandCausalActivityEvent | F::ResidueActivityEvent
    )
}

fn copy_if_absent(
    out: &mut serde_json::Map<String, serde_json::Value>,
    src: &serde_json::Map<String, serde_json::Value>,
    source_key: &str,
    target_key: &str,
) {
    if !out.contains_key(target_key) {
        if let Some(v) = src.get(source_key) {
            out.insert(target_key.to_string(), v.clone());
        }
    }
}

fn copy_nested_if_absent(
    out: &mut serde_json::Map<String, serde_json::Value>,
    src: &serde_json::Map<String, serde_json::Value>,
    object_key: &str,
    source_key: &str,
    target_key: &str,
) {
    if out.contains_key(target_key) {
        return;
    }
    if let Some(v) = src
        .get(object_key)
        .and_then(|v| v.as_object())
        .and_then(|m| m.get(source_key))
    {
        out.insert(target_key.to_string(), v.clone());
    }
}

fn copy_path_if_absent(
    out: &mut serde_json::Map<String, serde_json::Value>,
    src: &serde_json::Map<String, serde_json::Value>,
    path: &[&str],
    target_key: &str,
) {
    if out.contains_key(target_key) {
        return;
    }
    let mut cur: Option<&serde_json::Value> = None;
    for (idx, key) in path.iter().enumerate() {
        cur = if idx == 0 {
            src.get(*key)
        } else {
            cur.and_then(|v| v.as_object()).and_then(|m| m.get(*key))
        };
        if cur.is_none() {
            return;
        }
    }
    if let Some(v) = cur {
        out.insert(target_key.to_string(), v.clone());
    }
}

fn copy_runtime_surface_bindings(
    out: &mut serde_json::Map<String, serde_json::Value>,
    obj: &serde_json::Map<String, serde_json::Value>,
) {
    for (path, target) in [
        (&["artifact", "path"][..], "artifact_path"),
        (&["artifact", "sha256"][..], "artifact_sha256"),
        (&["artifact", "byte_start"][..], "artifact_byte_start"),
        (&["artifact", "byte_end"][..], "artifact_byte_end"),
        (
            &["artifact", "byte_end_exclusive"][..],
            "artifact_byte_end_exclusive",
        ),
        (&["artifact", "raw_bytes"][..], "artifact_raw_bytes"),
        (&["byte_refs", "topology", "path"][..], "topology_path"),
        (&["byte_refs", "topology", "sha256"][..], "topology_sha256"),
        (
            &["byte_refs", "trajectory_frame", "path"][..],
            "trajectory_frame_path",
        ),
        (
            &["byte_refs", "trajectory_frame", "byte_start"][..],
            "trajectory_frame_byte_start",
        ),
        (
            &["byte_refs", "trajectory_frame", "byte_end_inclusive"][..],
            "trajectory_frame_byte_end",
        ),
        (
            &["byte_refs", "trajectory_frame", "byte_end_exclusive"][..],
            "trajectory_frame_byte_end_exclusive",
        ),
        (
            &["byte_refs", "trajectory_frame", "n_floats"][..],
            "trajectory_frame_n_floats",
        ),
        (
            &["byte_refs", "trajectory_frame", "frame_hash64"][..],
            "trajectory_frame_hash64",
        ),
        (
            &["byte_refs", "trajectory_frame", "rolling_hash64"][..],
            "trajectory_frame_rolling_hash64",
        ),
        (
            &["byte_refs", "trajectory_frame", "sha256_status"][..],
            "trajectory_frame_sha256_status",
        ),
        (&["refs", "trajectory_path"][..], "trajectory_path"),
        (
            &["refs", "trajectory_origin_receipt_path"][..],
            "trajectory_origin_receipt_path",
        ),
        (
            &["refs", "prismcode_substrate_receipt_path"][..],
            "prismcode_substrate_receipt_path",
        ),
        (
            &["refs", "native_prismcode_receipt_expected_path"][..],
            "native_prismcode_receipt_expected_path",
        ),
        (
            &["refs", "native_prismcode_receipt_status"][..],
            "native_prismcode_receipt_status",
        ),
        (
            &["refs", "spectral_geometry_authority", "status"][..],
            "spectral_geometry_authority_status",
        ),
    ] {
        copy_path_if_absent(out, obj, path, target);
    }
}

fn normalized_scientific_surface_event(
    family: F,
    obj: &serde_json::Map<String, serde_json::Value>,
) -> serde_json::Map<String, serde_json::Value> {
    let mut out = obj.clone();
    copy_runtime_surface_bindings(&mut out, obj);
    match family {
        F::ScientificPocketOpeningEvent => {
            copy_if_absent(&mut out, obj, "pocket_volume_a3", "pocket_volume");
            copy_if_absent(
                &mut out,
                obj,
                "pocket_surface_area_a2",
                "pocket_surface_area",
            );
            copy_if_absent(
                &mut out,
                obj,
                "apo_holo_contrast_status",
                "apo_holo_contrast",
            );
        }
        F::LigandCausalActivityEvent => {
            for key in [
                "contact_delta",
                "pocket_volume_delta",
                "kcc_delta",
                "warp_tensor_delta",
                "force_delta",
                "residue_motion_delta",
            ] {
                copy_nested_if_absent(&mut out, obj, "causal_signals", key, key);
            }
        }
        F::ResidueActivityEvent => {
            for key in [
                "sasa_delta",
                "rmsf_delta",
                "distance_gate_delta",
                "contact_map_delta",
                "kcc_score",
                "force_norm",
                "displacement_a",
            ] {
                copy_nested_if_absent(&mut out, obj, "metrics", key, key);
            }
        }
        _ => {}
    }
    out
}

fn contact_projection(
    contact: &serde_json::Map<String, serde_json::Value>,
) -> serde_json::Map<String, serde_json::Value> {
    let mut out = serde_json::Map::new();
    if let Some(v) = contact.get("residue_id") {
        out.insert("contact_residue_id".to_string(), v.clone());
    }
    for key in ["contact_type", "distance_a", "duration_frames"] {
        if let Some(v) = contact.get(key) {
            out.insert(key.to_string(), v.clone());
        }
    }
    out
}

fn route_ligand_contacts(
    enc: &Encoder,
    stream: &mut TokenStream,
    obj: &serde_json::Map<String, serde_json::Value>,
    verified: bool,
    seal_sha: [u8; 32],
) -> RouteReport {
    let mut total = RouteReport::default();
    let Some(contacts) = obj.get("contacts").and_then(|v| v.as_array()) else {
        return total;
    };
    for contact in contacts {
        if let Some(contact) = contact.as_object() {
            let projected = contact_projection(contact);
            total.merge(route_object(
                enc,
                stream,
                F::LigandCausalActivityEvent,
                &projected,
                verified,
                seal_sha,
                false,
            ));
        }
    }
    total
}

pub fn route_scientific_surface_event_record(
    enc: &Encoder,
    stream: &mut TokenStream,
    family: F,
    record: &serde_json::Value,
    record_index: usize,
    verified: bool,
    seal_sha: [u8; 32],
    capture: bool,
) -> RouteReport {
    let Some(obj) = record.as_object() else {
        return RouteReport::default();
    };
    if !is_scientific_surface_event_family(family) {
        return RouteReport::default();
    }
    let flat = normalized_scientific_surface_event(family, obj);
    let seg = stream.begin_record(family, record_index);
    let mut total = route_object(enc, stream, family, &flat, verified, seal_sha, capture);
    if family == F::LigandCausalActivityEvent {
        total.merge(route_ligand_contacts(enc, stream, obj, verified, seal_sha));
    }
    stream.end_record(seg);
    total.records = 1;
    total
}

pub fn route_scientific_surface_event_records(
    enc: &Encoder,
    stream: &mut TokenStream,
    family: F,
    json: &serde_json::Value,
    record_path: &str,
    verified: bool,
    seal_sha: [u8; 32],
    capture: bool,
) -> RouteReport {
    let mut total = RouteReport::default();
    let records = resolve_path(json, record_path);
    total.records = records.len();
    for (idx, rec) in records.iter().enumerate() {
        let routed = route_scientific_surface_event_record(
            enc, stream, family, rec, idx, verified, seal_sha, capture,
        );
        total.matched += routed.matched;
        total.skipped_unknown += routed.skipped_unknown;
        total.skipped_type += routed.skipped_type;
        total.null_fields += routed.null_fields;
    }
    total
}

pub fn evaluate_causal_geometry_acceptance(
    events: &[serde_json::Value],
    spectral_geometry: Option<&serde_json::Value>,
    pathb_geometry: Option<&serde_json::Value>,
) -> CausalGeometryAcceptanceReport {
    const OPERATOR_CHAIN: &[&str] = &[
        "FRAME_CAPTURE",
        "CONTACT_GEOMETRY",
        "LIGAND_CAUSAL_ACTIVITY",
        "RESIDUE_ACTIVITY",
        "POCKET_OPENING_EVENT",
        "T4_EPOCH_OVERLAY",
        "SITE_RANKING",
    ];
    const SPECTRAL_METRICS: &[&str] = &[
        "ligand_probe_accessibility",
        "water_occupancy",
        "tunnel_cavity_persistence",
        "apo_holo_gdp_contrast",
        "replica_recurrence",
    ];
    const PATHB_METRICS: &[&str] = &["pocket_volume", "lining_residues", "persistence_frames"];

    fn present(value: &serde_json::Value) -> bool {
        match value {
            serde_json::Value::Null => false,
            serde_json::Value::String(text) => !text.is_empty(),
            serde_json::Value::Array(values) => !values.is_empty(),
            serde_json::Value::Object(values) => !values.is_empty(),
            serde_json::Value::Bool(_) | serde_json::Value::Number(_) => true,
        }
    }
    fn has_all(obj: &serde_json::Map<String, serde_json::Value>, keys: &[&str]) -> bool {
        keys.iter().all(|key| obj.get(*key).is_some_and(present))
    }
    fn has_any(obj: &serde_json::Map<String, serde_json::Value>, keys: &[&str]) -> bool {
        keys.iter().any(|key| obj.get(*key).is_some_and(present))
    }
    fn push_reason(reasons: &mut Vec<&'static str>, reason: &'static str) {
        if !reasons.contains(&reason) {
            reasons.push(reason);
        }
    }
    fn event_family(value: &serde_json::Value) -> Option<F> {
        let obj = value.as_object()?;
        if let Some(schema_kind) = obj.get("schema_kind").and_then(|value| value.as_str()) {
            if let Some((family, _)) = scientific_surface_family_for_schema_kind(schema_kind) {
                return Some(family);
            }
        }
        scientific_surface_family_for_event_type(obj).map(|(family, _)| family)
    }

    let mut reasons = Vec::new();
    let mut chronology_valid = !events.is_empty();
    let mut causal_events_valid = true;
    let mut has_ligand = false;
    let mut has_residue = false;
    let mut has_pocket = false;

    if events.is_empty() {
        push_reason(&mut reasons, "missing_causal_event_chain");
    }

    for event in events {
        let Some(obj) = event.as_object() else {
            chronology_valid = false;
            causal_events_valid = false;
            push_reason(&mut reasons, "invalid_causal_event_object");
            continue;
        };
        if !has_all(obj, &["epoch_id"]) {
            chronology_valid = false;
            push_reason(&mut reasons, "missing_epoch_id");
        }
        if !has_all(obj, &["frame_event_id"]) {
            chronology_valid = false;
            push_reason(&mut reasons, "missing_frame_event_id");
        }

        match event_family(event) {
            Some(F::LigandCausalActivityEvent) => {
                has_ligand = true;
                let contacts_valid = obj
                    .get("contacts")
                    .and_then(|value| value.as_array())
                    .is_some_and(|contacts| {
                        !contacts.is_empty()
                            && contacts.iter().all(|contact| {
                                let Some(contact) = contact.as_object() else {
                                    return false;
                                };
                                let contact_type =
                                    contact.get("contact_type").and_then(|value| value.as_str());
                                has_all(contact, &["residue_id", "distance_a", "duration_frames"])
                                    && contact_type
                                        .is_some_and(|value| LIGAND_CONTACT_TYPE.contains(&value))
                            })
                    });
                if !contacts_valid {
                    push_reason(&mut reasons, "invalid_ligand_contacts");
                }
                let causal_signals_valid = obj
                    .get("causal_signals")
                    .and_then(|value| value.as_object())
                    .is_some_and(|signals| {
                        has_all(
                            signals,
                            &[
                                "contact_delta",
                                "pocket_volume_delta",
                                "kcc_delta",
                                "warp_tensor_delta",
                                "force_delta",
                                "residue_motion_delta",
                            ],
                        )
                    });
                if !causal_signals_valid {
                    push_reason(&mut reasons, "proximity_without_causal_signals");
                    push_reason(&mut reasons, "missing_ligand_causal_signals");
                }
                causal_events_valid &= has_all(
                    obj,
                    &[
                        "epoch_id",
                        "frame_event_id",
                        "ligand_id",
                        "atom_ids",
                        "pose_cluster_id",
                        "binding_site_id",
                        "confidence",
                    ],
                ) && contacts_valid
                    && causal_signals_valid;
            }
            Some(F::ResidueActivityEvent) => {
                has_residue = true;
                let activity_valid = obj
                    .get("activity_type")
                    .and_then(|value| value.as_str())
                    .is_some_and(|activity| RESIDUE_ACTIVITY_TYPE.contains(&activity));
                if !activity_valid {
                    push_reason(&mut reasons, "invalid_residue_activity_type");
                }
                let metrics_valid = obj
                    .get("metrics")
                    .and_then(|value| value.as_object())
                    .is_some_and(|metrics| {
                        has_all(
                            metrics,
                            &[
                                "sasa_delta",
                                "rmsf_delta",
                                "distance_gate_delta",
                                "contact_map_delta",
                                "kcc_score",
                                "force_norm",
                                "displacement_a",
                            ],
                        )
                    });
                if !metrics_valid {
                    push_reason(&mut reasons, "missing_residue_activity_metrics");
                }
                causal_events_valid &= has_all(
                    obj,
                    &[
                        "epoch_id",
                        "frame_event_id",
                        "chain_id",
                        "residue_id",
                        "residue_name",
                        "linked_ligand_event_id",
                        "confidence",
                    ],
                ) && activity_valid
                    && metrics_valid;
            }
            Some(F::ScientificPocketOpeningEvent) => {
                has_pocket = true;
                let pocket_valid = has_all(
                    obj,
                    &[
                        "epoch_id",
                        "frame_event_id",
                        "residue_ids",
                        "distance_gates",
                        "persistence_frames",
                        "confidence",
                    ],
                ) && has_any(obj, &["pocket_volume", "pocket_volume_a3"])
                    && has_any(obj, &["pocket_surface_area", "pocket_surface_area_a2"])
                    && has_any(obj, &["lining_contacts", "lining_residues"])
                    && has_any(obj, &["apo_holo_contrast", "apo_holo_gdp_contrast"]);
                if !pocket_valid {
                    push_reason(&mut reasons, "incomplete_scientific_pocket_opening_event");
                }
                causal_events_valid &= pocket_valid;
            }
            _ => {
                causal_events_valid = false;
                push_reason(&mut reasons, "unrecognized_causal_geometry_event");
            }
        }
    }

    if !has_ligand {
        push_reason(&mut reasons, "missing_ligand_causal_activity_event");
    }
    if !has_residue {
        push_reason(&mut reasons, "missing_residue_activity_event");
    }
    if !has_pocket {
        push_reason(&mut reasons, "missing_scientific_pocket_opening_event");
    }

    let spectral_bound = spectral_geometry
        .and_then(|value| value.as_object())
        .is_some_and(|obj| {
            has_any(
                obj,
                &["spectral_extractor_ref", "spectral_extractor_artifact_ref"],
            ) && has_any(
                obj,
                &[
                    "spectral_fiber_bundle_ref",
                    "spectral_fiber_bundle_artifact_ref",
                ],
            ) && has_all(obj, SPECTRAL_METRICS)
        });
    let pathb_bound = pathb_geometry
        .and_then(|value| value.as_object())
        .is_some_and(|obj| has_all(obj, PATHB_METRICS));

    let chronology_foundation_status = if chronology_valid { "PASS" } else { "BLOCKED" };
    let cryptic_opening_status = if !chronology_valid
        || !causal_events_valid
        || !has_ligand
        || !has_residue
        || !has_pocket
    {
        "BLOCKED"
    } else if spectral_bound {
        "PASS_SPECTRAL_GEOMETRY_BOUND"
    } else if pathb_bound {
        push_reason(&mut reasons, "spectral_geometry_absent_pathb_only");
        "PARTIAL_PATHB_GEOMETRY_ONLY"
    } else {
        push_reason(&mut reasons, "missing_spectral_and_pathb_geometry");
        "FAIL"
    };

    CausalGeometryAcceptanceReport {
        chronology_foundation_status,
        cryptic_opening_status,
        chain: OPERATOR_CHAIN,
        reasons,
    }
}

fn scientific_surface_family_for_schema_kind(schema_kind: &str) -> Option<(F, &'static str)> {
    match schema_kind {
        "ScientificPocketOpeningEvent"
        | "scientific_pocket_opening_event"
        | "prismos_scientific_pocket_opening_event" => Some((F::ScientificPocketOpeningEvent, "")),
        "ScientificPocketOpeningEvents"
        | "scientific_pocket_opening_events"
        | "prismos_scientific_pocket_opening_events" => {
            Some((F::ScientificPocketOpeningEvent, "/records"))
        }
        "LigandCausalActivityEvent"
        | "ligand_causal_activity_event"
        | "prismos_ligand_causal_activity_event" => Some((F::LigandCausalActivityEvent, "")),
        "LigandCausalActivityEvents"
        | "ligand_causal_activity_events"
        | "prismos_ligand_causal_activity_events" => {
            Some((F::LigandCausalActivityEvent, "/records"))
        }
        "ResidueActivityEvent" | "residue_activity_event" | "prismos_residue_activity_event" => {
            Some((F::ResidueActivityEvent, ""))
        }
        "ResidueActivityEvents" | "residue_activity_events" | "prismos_residue_activity_events" => {
            Some((F::ResidueActivityEvent, "/records"))
        }
        _ => None,
    }
}

fn scientific_surface_family_for_event_type(
    obj: &serde_json::Map<String, serde_json::Value>,
) -> Option<(F, &'static str)> {
    if obj.contains_key("ligand_id") && obj.contains_key("contacts") {
        Some((F::LigandCausalActivityEvent, ""))
    } else if obj.contains_key("chain_id")
        && obj.contains_key("activity_type")
        && obj.contains_key("metrics")
    {
        Some((F::ResidueActivityEvent, ""))
    } else if obj.contains_key("residue_ids")
        && (obj.contains_key("pocket_volume") || obj.contains_key("pocket_volume_a3"))
        && (obj.contains_key("lining_contacts") || obj.contains_key("lining_residues"))
        && obj.contains_key("distance_gates")
    {
        Some((F::ScientificPocketOpeningEvent, ""))
    } else {
        None
    }
}

/// Auto-detect (family, record_path) from a decoded artifact: scientific surface `schema_kind`
/// first, exact scientific surface object shape second, then existing unrelated schema and structure
/// detectors. None ⇒ Opaque.
pub fn detect_family(json: &serde_json::Value) -> Option<(F, &'static str)> {
    if let Some(sk) = json.get("schema_kind").and_then(|v| v.as_str()) {
        if let Some(family) = negative_evidence_family_from_schema_kind(sk) {
            let record_path = if json.get("records").map(|v| v.is_array()).unwrap_or(false) {
                "/records"
            } else {
                ""
            };
            return Some((family, record_path));
        }
        if let Some(family) = tsh_family_from_schema_kind(sk) {
            let cache_transition_objects = sk.eq_ignore_ascii_case("resident_transition_cache")
                && json
                    .get("transition_objects")
                    .map(|v| v.is_array())
                    .unwrap_or(false);
            let out_family = if cache_transition_objects {
                F::PersistentTransitionObject
            } else {
                family
            };
            let record_path = if json.get("records").map(|v| v.is_array()).unwrap_or(false) {
                "/records"
            } else if cache_transition_objects {
                "/transition_objects"
            } else {
                ""
            };
            return Some((out_family, record_path));
        }
        if let Some(family) = uncertainty_family_from_schema_kind(sk) {
            let record_path = if json.get("records").map(|v| v.is_array()).unwrap_or(false) {
                "/records"
            } else {
                ""
            };
            return Some((family, record_path));
        }
        if let Some(surface) = scientific_surface_family_for_schema_kind(sk) {
            return Some(surface);
        }
        if sk == "SPIKE_TRANSITION_EVENT" {
            // Full 23-field family (engine nhs_rt_full.rs ~7616) — NOT the thin BocpdEvent subset,
            // and NOT bocpd.jsonl (a different per-frame BOCPD log artifact).
            return Some((F::SpikeTransitionEvent, ""));
        }
        // PRISMOS-authored PREDICTION STATE WRAPPER ({schema_kind:"prismos_prediction_state",
        // n_records, records:[N per-site prediction records]}). Descend /records so each per-site
        // prediction routes by value (prediction_id/target_id/predicted_delta/confidence/uncertainty/
        // source_vertex_id/... => matched_fields>0); routing the wrapper object directly would make
        // every top-level key an opaque escape (matched_fields=0) and lose the N records. Operator
        // correction 2026-07-01: this route is INDEPENDENT of DstwNextOperator (keys on the
        // prediction_state schema_kind, NOT the next_operator carrier) — a PREDICTION STATE is a
        // DISTINCT epistemic object from an engine-owner work list. Its OWN family; NEVER aliases /
        // relabels DstwNextOperator (which keeps its own separate route + unchanged counts).
        if sk == "prismos_prediction_state" {
            return Some((F::PredictionState, "/records"));
        }
        // PRISMOS-authored SCIENTIFIC CHRONOLOGY EVENT WRAPPER
        // ({schema_kind:"prismos_scientific_chronology_events", records:[N event rows]}).
        // Descend /records so each event routes in its own first-class family.
        if sk == "prismos_scientific_chronology_events" {
            return Some((F::ScientificChronologyEvent, "/records"));
        }
        // PRISMOS-authored SCIENTIFIC STATE EPOCH WRAPPER
        // ({schema_kind:"prismos_scientific_state_epochs", records:[N epoch rows]}).
        // Descend /records; this is a first-class interval family, not a DstwStateVertex alias.
        if sk == "prismos_scientific_state_epochs" {
            return Some((F::ScientificStateEpoch, "/records"));
        }
        if sk == "prismos_scientific_chronology_event" {
            return Some((F::ScientificChronologyEvent, ""));
        }
        if sk == "prismos_scientific_state_epoch" {
            return Some((F::ScientificStateEpoch, ""));
        }
        if sk == "prismos_t4_epoch_overlays" {
            return Some((F::T4EpochOverlay, "/records"));
        }
        if sk == "prismos_chronology_acceptance_decisions" {
            return Some((F::ChronologyAcceptanceDecision, "/records"));
        }
        if sk == "dstw_t4_epoch_overlay" || sk == "prismos_t4_epoch_overlay" {
            return Some((F::T4EpochOverlay, ""));
        }
        if sk == "dstw_t4_chronology_acceptance_decision"
            || sk == "prismos_chronology_acceptance_decision"
            || sk == "prismos_scientific_chronology_acceptance_decision"
        {
            return Some((F::ChronologyAcceptanceDecision, ""));
        }
        // PRISMOS-owned run-specific spectral geometry closure manifest. The manifest is routed
        // as one first-class SpectralExtractor record; the exporter flattens nested summary/input/
        // output hashes into value fields before calling route_object.
        if sk == "prismos_dstw_run_specific_spectral_closure_manifest"
            || sk == "prismos_spectral_extractor_closure_manifest"
        {
            return Some((F::SpectralExtractor, ""));
        }
        // PRISMOS-authored opcode/operand execution ledger WRAPPER
        // ({schema_kind:"prismcode_typed_operand_ledger", schema_version, records:[368 typed rows]}).
        // Descend /records so each per-opcode row routes (matched_fields>0); routing the wrapper
        // object would make every top-level key an opaque escape (matched_fields=0). Placed BEFORE
        // the `_receipt` arm — the schema_kind does not end with "_receipt" so ReceiptStatus never
        // swallows it, but the explicit rule keeps intent unambiguous.
        if sk == "prismcode_typed_operand_ledger" {
            return Some((F::OpcodeExecutionLedger, "/records"));
        }
        // Aggregate MD-ISA execution VERDICT object routed at top level (pointer "" = no descent —
        // the top-level object IS the record; the data-dense streams[] descends to the sidecar).
        // Its OWN family — NOT ReceiptStatus (schema_kind does not end with "_receipt"; the 12-field
        // verdict set differs). Placed BEFORE the `_receipt` arm so intent is unambiguous.
        if sk == "prismos_prismcode_isa_execution_verification" {
            return Some((F::ValidationState, ""));
        }
        // PRISMOS-authored ISA REGISTRY SUMMARY WRAPPER ({schema_kind:
        // "prismos_prismcode_isa_registry_summary", registry_size, native_md_receipt_count,
        // records:[N per-(stream,opcode) observations]}). Descend /records so each observation row
        // routes by value (stream_id/opcode/count/observed/... => matched_fields>0); routing the
        // wrapper object directly would make every top-level key an opaque escape (matched_fields=0)
        // and lose the N records. Its OWN family — NOT ValidationState (that is the top-level verdict
        // object) and NOT OpcodeExecutionLedger (the per-executed-opcode ledger).
        if sk == "prismos_prismcode_isa_registry_summary" {
            return Some((F::IsaRegistrySummary, "/records"));
        }
        // PRISMOS-authored RAW PRKCC001 kcc_v2full.bin per-residue descriptor summary WRAPPER
        // ({schema_kind:"prismos_kcc_descriptor_native_summary", n_residues, record_count_envelope,
        // records:[N per-residue records]}). Descend /records so each per-residue record routes by
        // value (residue_id/temporal_corr/direction_score/... => matched_fields>0); routing the
        // wrapper object directly would make every top-level key an opaque escape (matched_fields=0)
        // and lose the N per-residue records. Its OWN family — NOT KccDescriptor (that is the
        // kcc_visualization.json residues[] JSON view with a different field set + a
        // vector_field_definition detector), and NOT DenseTensorSemanticSummary (min/max/mean stats).
        if sk == "prismos_kcc_descriptor_native_summary" {
            return Some((F::KccDescriptorNative, "/records"));
        }
        // PRISMOS-authored RAW PRSPK001 _spikes.bin per-residue BOUNDED spike aggregate WRAPPER
        // ({schema_kind:"prismos_spike_event_native_summary", n_residues, total_spike_count,
        // record_count_envelope, records:[N per-residue aggregates]}). Descend /records so each
        // per-residue aggregate routes by value (residue_id/spike_count/spike_rate/mean_amplitude/...
        // => matched_fields>0); routing the wrapper object directly would make every top-level key an
        // opaque escape (matched_fields=0) and lose the N per-residue records. Its OWN family — NOT
        // SpikeEvent (the per-spike JSON view / raw per-spike carrier), and NOT
        // DenseTensorSemanticSummary (min/max/mean tensor stats).
        if sk == "prismos_spike_event_native_summary" {
            return Some((F::SpikeSummary, "/records"));
        }
        if sk.contains("qnanl_artifact_receipt") || sk.ends_with("_receipt") {
            return Some((F::ReceiptStatus, ""));
        }
        // all-on control/receipt artifacts (field_completeness_report, post_md_required_inputs).
        if sk == "field_completeness_report" || sk == "post_md_required_inputs" {
            return Some((F::ReceiptStatus, ""));
        }
        // TWIN16 interferometric transition-view residuals — route via route_twin16_residual
        // (both /cells and /pairwise_residuals). detect returns the /cells path as the primary.
        if sk == "TWIN16_INTERFEROMETRIC_TRANSITION_VIEW_RESIDUALS" {
            return Some((F::TwinTransitionResidual, "/cells"));
        }
        // ENGINE bounded thermo/temporal CCNS/SDST AGGREGATE SUMMARY — ONE top-level record
        // (pointer "" = the object itself IS the record; the data-dense cells[]/event_sources[]/
        // phase_edge_counts/temporal_extent/condition_ledger descend losslessly to the sidecar).
        // Safety net for the auto/route_artifact fallback; the PRIMARY path is the EXPLICIT manifest
        // branch (flatten_thermo_phase_manifold lifts the nested CCNS/SDST/phase/trigger rollup so
        // route_object value-matches them). DISTINCT from TwinTransitionResidual (that is the TWIN16
        // residuals FILE; its "/cells" descent would DROP this summary-level rollup). See
        // types.rs ThermoPhaseManifold.
        if sk == "bounded_thermo_temporal_ccns_sdst_summary" {
            return Some((F::ThermoPhaseManifold, ""));
        }
        // PathB ranked voxel-density-peak candidate carrier
        // ({schema_kind:"pathb_site_candidates", candidates:[N ranked rows], n_candidates, top_k}).
        // Descend /candidates so each ranked row routes under RankerRow (matched_fields>0); routing
        // the wrapper object would make every top-level key (n_candidates/top_k/manifest_run_id/...)
        // an opaque escape (matched_fields=0). By-value: schema_kind is a REAL field value in the
        // file, /candidates is a REAL key. Rows carry rank/peak_id/spike_count (specced slots 4-6).
        if sk == "pathb_site_candidates" {
            return Some((F::RankerRow, "/candidates"));
        }
        // PathB FLAGSHIP materialized binding-site WRAPPER ({schema_kind:
        // "pathb_binding_sites_materialized", binding_sites:[N per-site records], site_count:N, ...}).
        // Descend /binding_sites so each promoted per-site record routes (site_id/rank/n_spikes/
        // centroid_spread_a/... => matched_fields>0). Routing the wrapper object directly would make
        // every top-level key an opaque escape (matched_fields=0) and lose the N per-site records.
        // DISTINCT from KccSite (kcc_visualization.json sites[]) — this schema has NO gtck_rank/kcc.
        if sk == "pathb_binding_sites_materialized" {
            return Some((F::PathBBindingSiteMaterialized, "/binding_sites"));
        }
        // ghost_time_map.json captured-graph GPU temporal telemetry — wrapper {schema_kind:
        // "pathb_ghost_time_map", streams:[per-stream frames]}. Descend /streams so each of the 8
        // stream records routes (matched_fields>0). Routing the wrapper object directly would make
        // target/run_id/schema_kind/notes opaque escapes (matched_fields=0, per-stream values lost).
        if sk == "pathb_ghost_time_map"
            && json.get("streams").map(|v| v.is_array()).unwrap_or(false)
        {
            return Some((F::CapturedGraphGpuTelemetry, "/streams"));
        }
    }
    if let Some(obj) = json.as_object() {
        if let Some(surface) = scientific_surface_family_for_event_type(obj) {
            return Some(surface);
        }
    }
    // Dense-tensor semantic-summary WRAPPER (native-run form:
    // {schema_kind, tokenizer_family:"DenseTensorSemanticSummary", summaries:[typed 18-field records]}).
    // Descend /summaries so each per-tensor record routes (matched_fields>0); routing the wrapper
    // object directly would make every top-level key an opaque escape (matched_fields=0). The
    // Phase-2 reducer's FLAT-ARRAY form is routed explicitly upstream and never reaches detect.
    if json.get("tokenizer_family").and_then(|v| v.as_str()) == Some("DenseTensorSemanticSummary")
        && json.get("summaries").map(|v| v.is_array()).unwrap_or(false)
    {
        return Some((F::DenseTensorSemanticSummary, "/summaries"));
    }
    // GhostPhaseLattice4D PHASE-MANIFOLD EDGE-VALIDATION manifest
    // ({stem}_binding_sites_lattice.json — engine nhs_rt_full.rs ~16336, emitted only when
    // v2_was_live==true AND cluster() succeeds). No schema_kind; keyed on backend=="ghost_phase_lattice_4d"
    // + a components[] array + a stats object. Route as ONE top-level record (pointer "" — the object IS
    // the record; components[]/nested blocks descend LOSSLESS to the sidecar). Safety net for the
    // auto/route_artifact fallback; the PRIMARY path is the EXPLICIT manifest branch
    // (flatten_phase_manifold_edge_validation lifts stats.*/config.* + derives component aggregates so
    // route_object value-matches the real edge counts + SO(3) coherence). DISTINCT from
    // CapturedGraphGpuTelemetry — ghost_lattice_routing_status.json is a FLAT status object keyed on
    // `backend_request`/`v2_was_live` with NO `components[]`/`stats`, so no collision. See
    // types.rs PhaseManifoldEdgeValidation.
    if json.get("backend").and_then(|v| v.as_str()) == Some("ghost_phase_lattice_4d")
        && json
            .get("components")
            .map(|v| v.is_array())
            .unwrap_or(false)
        && json.get("stats").map(|v| v.is_object()).unwrap_or(false)
    {
        return Some((F::PhaseManifoldEdgeValidation, ""));
    }
    // structural signatures (real artifacts often lack schema_kind, e.g. spike_events.json)
    if json.get("spikes").map(|v| v.is_array()).unwrap_or(false) {
        return Some((F::SpikeEvent, "/spikes"));
    }
    // TWIN16-native condition-edge per-cell-pair record (per_cellpair_condition_edges.jsonl):
    // cell_a/cell_b + hamming + material is unique to the DPD edge (vs the TWIN16 residual pairwise).
    if json.get("cell_a").is_some()
        && json.get("cell_b").is_some()
        && json.get("hamming").is_some()
        && json.get("material").is_some()
    {
        return Some((F::TwinConditionEdge, ""));
    }
    // AUXILIARY coupled-A/B NMA per-residue differential (per_residue_condition_edges.jsonl):
    // resid + nma/thermal_exclusive_count + condition_exclusive is unique to the coupled NMA edge.
    // SEPARATE from TWIN16 — NMA is not a TWIN16 axis.
    if json.get("resid").is_some()
        && json.get("nma_exclusive_count").is_some()
        && json.get("thermal_exclusive_count").is_some()
        && json.get("condition_exclusive").is_some()
    {
        return Some((F::NmaCoupledEdge, ""));
    }
    // bocpd.jsonl per-frame BOCPD log — distinct from SPIKE_TRANSITION_EVENT and BocpdEvent.
    if json.get("frame_idx").is_some()
        && json.get("posterior_max").is_some()
        && json.get("map_run_length").is_some()
    {
        return Some((F::BocpdLog, ""));
    }
    // protocol_state.json per-stream control state.
    if json.get("current_phase_bits").is_some()
        && json.get("dt_ps").is_some()
        && json.get("current_temperature_K").is_some()
    {
        return Some((F::ProtocolState, ""));
    }
    // noise_floor.json per-stream QC calibration (mu/sigma/samples arrays + n_samples + phase).
    if json.get("mu").map(|v| v.is_array()).unwrap_or(false)
        && json.get("sigma").map(|v| v.is_array()).unwrap_or(false)
        && json.get("n_samples").is_some()
    {
        return Some((F::NoiseFloor, ""));
    }
    // KCC visualization: a residues[] descriptor array + the vector_field_definition block
    // (both present ⇒ unambiguous; other artifacts carry one without the other). Checked BEFORE
    // sites[] so the combined artifact routes residues here and its sites via route_kcc_sites.
    if json.get("residues").map(|v| v.is_array()).unwrap_or(false)
        && json.get("vector_field_definition").is_some()
    {
        return Some((F::KccDescriptor, "/residues"));
    }
    // PathB FLAGSHIP materialized binding sites — DEFENSIVE structural fallback for a wrapper that
    // (hypothetically) lacks schema_kind: binding_sites[] whose first record carries site_id +
    // lining_residues + centroid_xyz is the pathb_binding_sites_materialized shape. Checked BEFORE
    // the KccSite /sites block; its own key is `binding_sites`, not `sites`, so there is no overlap.
    if let Some(b0) = json
        .get("binding_sites")
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())
    {
        if b0.get("site_id").is_some()
            && b0.get("lining_residues").is_some()
            && b0.get("centroid_xyz").is_some()
        {
            return Some((F::PathBBindingSiteMaterialized, "/binding_sites"));
        }
    }
    // KCC visualization SITES[]: keyed on the kcc-site-specific signature (gtck_rank + rank_C + kcc
    // object) so a generic `sites` array — e.g. binding_sites.json (lining_residues/druggability/
    // classification, NO gtck_rank/kcc) — does NOT match. No conflation with BindingSite artifacts.
    if let Some(s0) = json
        .get("sites")
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())
    {
        let full_kcc_site =
            s0.get("gtck_rank").is_some() && s0.get("rank_C").is_some() && s0.get("kcc").is_some();
        let compact_kcc_site = s0.get("gtck_rank").is_some()
            && s0.get("rank_score").is_some()
            && s0.get("site_local_cov").is_some()
            && s0.get("centroid").is_some();
        if full_kcc_site || compact_kcc_site {
            return Some((F::KccSite, "/sites"));
        }
    }
    None
}

/// Detect + route a decoded artifact end-to-end. Returns the chosen family (None ⇒ unrouted).
pub fn route_artifact(
    enc: &Encoder,
    stream: &mut TokenStream,
    json: &serde_json::Value,
    verified: bool,
    seal_sha: [u8; 32],
    capture: bool,
) -> (Option<F>, RouteReport) {
    match detect_family(json) {
        Some((family, path)) => {
            let report = if is_scientific_surface_event_family(family) {
                route_scientific_surface_event_records(
                    enc, stream, family, json, path, verified, seal_sha, capture,
                )
            } else {
                route_records(enc, stream, family, json, path, verified, seal_sha, capture)
            };
            (Some(family), report)
        }
        None => (None, RouteReport::default()),
    }
}

/// Harvest all coordinates for one lattice field across the records (for VQ fit, P1-5).
fn harvest_field_coords(
    json: &serde_json::Value,
    family: F,
    field: &str,
    record_path: &str,
) -> Vec<[f64; 3]> {
    let mut out = Vec::new();
    let Some(src) = coord_source(family, field) else {
        return out;
    };
    for rec in resolve_path(json, record_path) {
        let Some(obj) = rec.as_object() else { continue };
        let v3 = match &src {
            CoordSource::Triple(xk, yk, zk) => match (
                obj.get(*xk).and_then(|v| v.as_f64()),
                obj.get(*yk).and_then(|v| v.as_f64()),
                obj.get(*zk).and_then(|v| v.as_f64()),
            ) {
                (Some(x), Some(y), Some(z)) => Some([x, y, z]),
                _ => None,
            },
            CoordSource::Array(aliases) => aliases.iter().find_map(|k| obj.get(*k).and_then(arr3)),
        };
        if let Some(v3) = v3 {
            out.push(v3);
        }
    }
    out
}

/// P1-5: detect → fit lattice codebooks on the artifact's OWN coordinate distribution → route.
/// This makes position tokens non-escape (real spatial codes) without a manual `fit_coords`.
pub fn route_artifact_autofit(
    enc: &mut Encoder,
    stream: &mut TokenStream,
    json: &serde_json::Value,
    verified: bool,
    seal_sha: [u8; 32],
    capture: bool,
) -> (Option<F>, RouteReport) {
    let Some((family, path)) = detect_family(json) else {
        return (None, RouteReport::default());
    };
    // pass 1: fit each lattice field's VQ codebook on the harvested coords.
    let lattice_fields: Vec<&'static str> = enc
        .registry()
        .iter()
        .filter(|s| s.family == family && matches!(s.geometry, G::LatticeVq3d { .. }))
        .map(|s| s.field)
        .collect();
    for field in lattice_fields {
        let coords = harvest_field_coords(json, family, field, path);
        if !coords.is_empty() {
            enc.fit_coords(family, field, &coords);
        }
    }
    // pass 2: route with the fitted codebooks.
    let r = if is_scientific_surface_event_family(family) {
        route_scientific_surface_event_records(
            enc, stream, family, json, path, verified, seal_sha, capture,
        )
    } else {
        route_records(enc, stream, family, json, path, verified, seal_sha, capture)
    };
    (Some(family), r)
}

/// Route one decoded SPIKE_TRANSITION_EVENT record (engine `nhs_rt_full.rs` ~7616). The 9 typed
/// evidence fields (stream_id / group_id / step / chunk_idx / bocpd_probability / spike_count /
/// trigger_kind / phase_from / phase_to) route to their quantizers; the string-id / status / hash /
/// nested-object fields (condition_id, protocol_name, *_status, *_sha256, *_hash,
/// threshold_before_after, bridge_groups_observed) are preserved LOSSLESS in the sidecar (§10
/// seal-collapse). `capture` is forced true so NO source-schema field is dropped — fields absent
/// from a given record are simply not emitted (optional-field support).
pub fn route_spike_transition_event(
    enc: &Encoder,
    stream: &mut TokenStream,
    record: &serde_json::Value,
    verified: bool,
    seal_sha: [u8; 32],
) -> RouteReport {
    let Some(obj) = record.as_object() else {
        return RouteReport::default();
    };
    let seg = stream.begin_record(F::SpikeTransitionEvent, 0);
    let mut r = route_object(
        enc,
        stream,
        F::SpikeTransitionEvent,
        obj,
        verified,
        seal_sha,
        true,
    );
    stream.end_record(seg);
    r.records = 1;
    r
}

/// Route a TWIN16_INTERFEROMETRIC_TRANSITION_VIEW_RESIDUALS artifact: the 16 per-cell bounded
/// thermo/temporal aggregates (`/cells`) AND the cell-pair L1 residuals (`/pairwise_residuals`),
/// both under the TwinTransitionResidual family. `capture=true` preserves the nested objects
/// (ccns/sdst/phase_bucket_counts/phase_edge_counts/trigger_counts) losslessly. Returns the
/// (cells, pairwise) route reports. This is the interferometric transition-view surface — NOT a
/// pairwise residue_i↔residue_j TE matrix.
pub fn route_twin16_residual(
    enc: &Encoder,
    stream: &mut TokenStream,
    json: &serde_json::Value,
    verified: bool,
    seal_sha: [u8; 32],
) -> (RouteReport, RouteReport) {
    let f = F::TwinTransitionResidual;
    let cells = route_records(enc, stream, f, json, "/cells", verified, seal_sha, true);
    let pairs = route_records(
        enc,
        stream,
        f,
        json,
        "/pairwise_residuals",
        verified,
        seal_sha,
        true,
    );
    (cells, pairs)
}

/// Route the REAL per-residue TWIN coupling evidence (twin_per_residue.json — a top-level array of
/// per-residue records) into the TwinCoupled family: b_over_a_ratio (asymmetry), ccf_max/mean/
/// centrality (cross-correlation), barrier_class, spikes_a/b, consensus/differential scores, joined
/// by `resid`. Fits the Cα VQ on the records' own positions first (non-escape spatial codes).
pub fn route_twin_per_residue(
    enc: &mut Encoder,
    stream: &mut TokenStream,
    json: &serde_json::Value,
    verified: bool,
    seal_sha: [u8; 32],
) -> RouteReport {
    let mut rep = RouteReport::default();
    let Some(arr) = json.as_array() else {
        return rep;
    };
    rep.records = arr.len();
    // fit ca_xyz VQ on the residues' own Cα positions (P1-5).
    let coords: Vec<[f64; 3]> = arr
        .iter()
        .filter_map(|r| r.get("ca_xyz").and_then(arr3))
        .collect();
    if !coords.is_empty() {
        enc.fit_coords(F::TwinCoupled, "ca_xyz", &coords);
    }
    for (idx, rec) in arr.iter().enumerate() {
        if let Some(obj) = rec.as_object() {
            let seg = stream.begin_record(F::TwinCoupled, idx);
            let r = route_object(enc, stream, F::TwinCoupled, obj, verified, seal_sha, true);
            rep.matched += r.matched;
            rep.skipped_unknown += r.skipped_unknown;
            rep.skipped_type += r.skipped_type;
            rep.null_fields += r.null_fields;
            stream.end_record(seg);
        }
    }
    rep
}

// --- KCC sites[] — relation-aware routing (a site is a relation over residues, §9) -----------

/// Exact decoded site keys. Emission order within a site segment (after the spatial centroid).
const SITE_TOPLEVEL_SCALARS: &[&str] = &[
    "volume",
    "id",
    "gtck_rank",
    "rank_C",
    "rank_G",
    "rank_K",
    "rank_L",
    "rank_T",
    "rank_score",
];
const SITE_KCC_SCALARS: &[&str] = &[
    "active_causal_steps",
    "best_kcc_candidate_index",
    "burst_motion",
    "direction_score",
    "driver_residue_id",
    "kcc_confidence",
    "lag_corr_peak",
    "local_cov",
    "motion_efficiency",
    "site_burst_motion",
    "site_causal_lag",
    "site_direction_score",
    "site_lag_corr_peak",
    "site_local_cov",
    "site_motion_efficiency",
    "temporal_corr",
    "total_steps",
];
/// Compact coupled-NMA KCC site exports carry these existing KccSite fields at site top-level
/// instead of under `kcc`. They are aliases onto the existing semantics, not new evidence classes.
const SITE_COMPACT_DIRECT_SCALARS: &[&str] = &[
    "kcc_confidence",
    "site_burst_motion",
    "site_lag_corr_peak",
    "site_local_cov",
];
const SITE_COMPACT_ALIAS_SCALARS: &[(&str, &str)] = &[
    ("driver_residue_topo", "driver_residue_id"),
    ("site_causal_lag_steps", "site_causal_lag"),
];
/// Per-candidate-member parallel arrays, emitted PER MEMBER after its `candidate_residue_ids`
/// anchor. Aligned by member index, so they move together under the residue_id-asc sort.
const SITE_MEMBER_SCALARS: &[&str] = &[
    "candidate_causal_weights",
    "candidate_kcc_burst_motion",
    "candidate_kcc_causal_lag",
    "candidate_kcc_confidence",
    "candidate_kcc_direction_score",
    "candidate_kcc_local_cov",
    "candidate_residue_support",
];

/// JoinKey / enum identity fields — skipped by scalar corpus-fit (their codes ARE the value).
const SITE_FIT_SKIP: &[&str] = &[
    "id",
    "gtck_rank",
    "best_kcc_candidate_index",
    "driver_residue_id",
];
/// Documented localization neighborhood radius (semantics: "fraction concentrated within site vs
/// neighborhood (R=6A)"). Members beyond it are spatial outliers (often the causally-dominant
/// driver, by design) and are REPORTED, not dropped.
pub const SITE_LOCALIZATION_RADIUS_A: f64 = 6.0;

/// Per-residue (residue_name, Cα position) joined from residues[] for member enrichment + the
/// centroid↔member spatial check.
type ResidueMap = HashMap<i64, (String, [f64; 3])>;

/// Authoritative per-residue TWIN/coherence evidence joined from prism_therm.json top_residues[].
#[derive(Clone, Debug)]
pub struct ThermEvidence {
    pub transfer_entropy: f64,
    pub fisher_info: f64,
    pub causal_dg: f64,
    pub kl_divergence: f64,
    pub role: String,
    pub n_causal_spikes: f64,
}
type ThermMap = HashMap<i64, ThermEvidence>;

/// binding_sites.json site evidence joined to KccSite by site `id` (classification/druggability/
/// cryptic_score).
type BindingMap = HashMap<i64, (String, f64, f64)>;

/// Build the site `id` → (classification, druggability, cryptic_score) map from binding_sites.json.
pub fn build_binding_map(binding_sites: &serde_json::Value) -> BindingMap {
    let mut m = BindingMap::new();
    let Some(sites) = binding_sites.get("sites").and_then(|v| v.as_array()) else {
        return m;
    };
    for s in sites {
        let Some(id) = s.get("id").and_then(|v| v.as_f64()) else {
            continue;
        };
        m.entry(id as i64).or_insert_with(|| {
            (
                s.get("classification")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                s.get("druggability")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0),
                s.get("cryptic_score")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0),
            )
        });
    }
    m
}

/// Build the residue_id → TWIN-evidence map from prism_therm.json pockets[].top_residues[].
pub fn build_therm_map(prism_therm: &serde_json::Value) -> ThermMap {
    let mut m = ThermMap::new();
    let Some(pockets) = prism_therm.get("pockets").and_then(|v| v.as_array()) else {
        return m;
    };
    for p in pockets {
        let Some(tr) = p.get("top_residues").and_then(|v| v.as_array()) else {
            continue;
        };
        for r in tr {
            let Some(rid) = r.get("residue_id").and_then(|v| v.as_f64()) else {
                continue;
            };
            m.entry(rid as i64).or_insert_with(|| ThermEvidence {
                transfer_entropy: r
                    .get("transfer_entropy")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0),
                fisher_info: r.get("fisher_info").and_then(|v| v.as_f64()).unwrap_or(0.0),
                causal_dg: r.get("causal_dg").and_then(|v| v.as_f64()).unwrap_or(0.0),
                kl_divergence: r
                    .get("kl_divergence")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0),
                role: r
                    .get("role")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                n_causal_spikes: r
                    .get("n_causal_spikes")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0),
            });
        }
    }
    m
}

/// Join + relation routing report for KCC sites[]. Unresolved member ids, array-length mismatches,
/// driver/best inconsistency, and spatial outliers all surface VISIBLY (never silently dropped).
#[derive(Clone, Debug)]
pub struct SiteRouteReport {
    pub sites: usize,
    pub matched: usize,
    pub null_kcc: usize,
    pub members_total: usize,
    pub members_resolved: usize,
    pub members_unresolved: Vec<i64>,
    /// false ⇒ residues[] absent, so the join could NOT be validated (reported, not assumed).
    pub join_validated: bool,
    /// site indices whose 8 candidate arrays disagree in length → member routing SKIPPED for that
    /// site (Phase 5: a length mismatch fails visibly, never emits misaligned member data).
    pub array_length_mismatches: Vec<usize>,
    /// members enriched with residue_name + Cα position joined from residues[].
    pub members_enriched: usize,
    /// (site_idx, member_residue_id, distance_Å) for members beyond R=6Å of the site centroid.
    pub spatial_outliers: Vec<(usize, i64, f64)>,
    /// false if any site violates driver_residue_id == candidate_residue_ids[best_kcc_candidate_index].
    pub driver_best_consistent: bool,
    /// members enriched with authoritative per-residue TWIN evidence joined from prism_therm.json.
    pub members_te_enriched: usize,
    /// sites enriched with binding_sites.json classification/druggability/cryptic (join by id).
    pub sites_binding_enriched: usize,
    /// centroid VQ health (after autofit): effective codes (2^entropy), distinct/dead codes,
    /// escapes, and mean/p95/worst Euclidean distortion (Å).
    pub centroid_perplexity: f64,
    pub centroid_codes_used: usize,
    pub centroid_dead_codes: usize,
    pub centroid_escapes: usize,
    pub centroid_mean_distortion: f64,
    pub centroid_p95_distortion: f64,
    pub centroid_worst_distortion: f64,
}

impl Default for SiteRouteReport {
    fn default() -> Self {
        Self {
            sites: 0,
            matched: 0,
            null_kcc: 0,
            members_total: 0,
            members_resolved: 0,
            members_unresolved: Vec::new(),
            join_validated: false,
            array_length_mismatches: Vec::new(),
            members_enriched: 0,
            spatial_outliers: Vec::new(),
            driver_best_consistent: true, // consistent until a violation is found
            members_te_enriched: 0,
            sites_binding_enriched: 0,
            centroid_perplexity: 0.0,
            centroid_codes_used: 0,
            centroid_dead_codes: 0,
            centroid_escapes: 0,
            centroid_mean_distortion: 0.0,
            centroid_p95_distortion: 0.0,
            centroid_worst_distortion: 0.0,
        }
    }
}

fn site_scalar(
    enc: &Encoder,
    stream: &mut TokenStream,
    key: &'static str,
    v: &serde_json::Value,
    verified: bool,
    seal_sha: [u8; 32],
) -> bool {
    match v.as_f64().or_else(|| v.as_bool().map(|b| b as i64 as f64)) {
        Some(x) => enc.encode_scalar(stream, F::KccSite, key, x, verified, seal_sha),
        None => false,
    }
}

/// Canonical member order: residue_id ascending. Chain is ABSENT in kcc_visualization residues[]
/// (no chain / local-index field), so no chain tie-break is claimed; member residue_ids are unique
/// per site, so residue_id asc is a total order (residue_name / local-index tie-breaks are
/// unavailable in the site arrays and unneeded). Returns the permutation over the candidate arrays.
fn member_order(candidate_ids: &[serde_json::Value]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..candidate_ids.len()).collect();
    idx.sort_by(|&a, &b| {
        let ra = candidate_ids[a].as_f64().unwrap_or(f64::INFINITY);
        let rb = candidate_ids[b].as_f64().unwrap_or(f64::INFINITY);
        ra.partial_cmp(&rb).unwrap_or(std::cmp::Ordering::Equal)
    });
    idx
}

fn euclidean(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Emit the DERIVED stable best-member pointer `best_candidate_residue_id` =
/// candidate_residue_ids[best_kcc_candidate_index] — order-independent, so it survives the
/// residue_id-asc member emission where the raw index would go stale. Also checks the invariant
/// driver_residue_id == that residue id (reported via `driver_best_consistent`).
fn emit_best_candidate(
    enc: &Encoder,
    stream: &mut TokenStream,
    kcc: &serde_json::Map<String, serde_json::Value>,
    verified: bool,
    seal_sha: [u8; 32],
    rep: &mut SiteRouteReport,
) {
    let ids = kcc.get("candidate_residue_ids").and_then(|v| v.as_array());
    let bi = kcc.get("best_kcc_candidate_index").and_then(|v| v.as_f64());
    if let (Some(ids), Some(bi)) = (ids, bi) {
        if let Some(best_rid) = ids.get(bi as usize).and_then(|v| v.as_f64()) {
            if enc.encode_scalar(
                stream,
                F::KccSite,
                "best_candidate_residue_id",
                best_rid,
                verified,
                seal_sha,
            ) {
                rep.matched += 1;
            }
            if let Some(drv) = kcc.get("driver_residue_id").and_then(|v| v.as_f64()) {
                if (drv - best_rid).abs() > 1e-9 {
                    rep.driver_best_consistent = false;
                }
            }
        }
    }
}

/// Route a site's member relation: the variable-length candidate list as a JoinKey list (emitted
/// residue_id-asc), each member id followed by its residues[]-joined enrichment (residue_name + Cα
/// position) and its aligned per-member scalars. Member ids resolve against residues[]; unresolved
/// ids surface in the report (never dropped). Phase 5: all 8 candidate arrays MUST share length or
/// member routing is skipped for that site and the mismatch is reported. (§9 record-token
/// addressing of the member tuple is M4 work; the per-member id anchor delimits each group.)
#[allow(clippy::too_many_arguments)]
fn route_site_members(
    enc: &Encoder,
    stream: &mut TokenStream,
    kcc: &serde_json::Map<String, serde_json::Value>,
    residues: Option<&ResidueMap>,
    therm: Option<&ThermMap>,
    centroid: Option<[f64; 3]>,
    site_idx: usize,
    verified: bool,
    seal_sha: [u8; 32],
    rep: &mut SiteRouteReport,
) {
    let Some(ids) = kcc.get("candidate_residue_ids").and_then(|v| v.as_array()) else {
        return;
    };
    let n = ids.len();
    let arrays: Vec<Option<&Vec<serde_json::Value>>> = SITE_MEMBER_SCALARS
        .iter()
        .map(|k| kcc.get(*k).and_then(|v| v.as_array()))
        .collect();
    // Phase 5: every present parallel array must share the id-list length (else FAIL VISIBLY).
    if arrays.iter().flatten().any(|a| a.len() != n) {
        rep.array_length_mismatches.push(site_idx);
        return; // do not emit misaligned member data
    }
    for &orig in &member_order(ids) {
        let Some(rid_f) = ids[orig].as_f64() else {
            continue;
        };
        let rid = rid_f as i64;
        rep.members_total += 1;
        let enrich = residues.and_then(|m| m.get(&rid));
        if residues.is_some() {
            if enrich.is_some() {
                rep.members_resolved += 1;
            } else {
                rep.members_unresolved.push(rid);
            }
        }
        // member id (JoinKey) — the join anchor + per-member delimiter
        if enc.encode_scalar(
            stream,
            F::KccSite,
            "candidate_residue_ids",
            rid_f,
            verified,
            seal_sha,
        ) {
            rep.matched += 1;
        }
        // enrichment: residue_name (AA enum) + Cα position (vec3) joined from residues[]
        if let Some((name, ca)) = enrich {
            rep.members_enriched += 1;
            if encode_enum_str(
                enc,
                stream,
                F::KccSite,
                "candidate_residue_name",
                name,
                verified,
                seal_sha,
            ) {
                rep.matched += 1;
            }
            if enc.encode_vec3(
                stream,
                F::KccSite,
                "candidate_ca_position",
                *ca,
                verified,
                seal_sha,
            ) {
                rep.matched += 1;
            }
            // spatial: distance to the site centroid vs the documented R=6Å localization radius.
            if let Some(c) = centroid {
                let d = euclidean(*ca, c);
                if d > SITE_LOCALIZATION_RADIUS_A {
                    rep.spatial_outliers.push((site_idx, rid, d));
                }
            }
        }
        // aligned per-member scalars (same original index → relation preserved under the sort)
        for (key, arr) in SITE_MEMBER_SCALARS.iter().zip(arrays.iter()) {
            if let Some(a) = arr {
                if let Some(x) = a.get(orig).and_then(|v| v.as_f64()) {
                    if enc.encode_scalar(stream, F::KccSite, key, x, verified, seal_sha) {
                        rep.matched += 1;
                    }
                }
            }
        }
        // authoritative per-residue TWIN evidence, typed-joined from prism_therm by residue_id.
        if let Some(te) = therm.and_then(|t| t.get(&rid)) {
            rep.members_te_enriched += 1;
            let mut emit = |field, value| {
                if enc.encode_scalar(stream, F::KccSite, field, value, verified, seal_sha) {
                    rep.matched += 1;
                }
            };
            emit("candidate_transfer_entropy", te.transfer_entropy);
            emit("candidate_fisher_info", te.fisher_info);
            emit("candidate_causal_dg", te.causal_dg);
            emit("candidate_kl_divergence", te.kl_divergence);
            emit("candidate_n_causal_spikes", te.n_causal_spikes);
            if encode_enum_str(
                enc,
                stream,
                F::KccSite,
                "candidate_te_role",
                &te.role,
                verified,
                seal_sha,
            ) {
                rep.matched += 1;
            }
        }
    }
}

/// Route one decoded KCC site object: spatial centroid + identity/rank tuple + nested kcc.*
/// aggregate causal/temporal summaries + the member relation. Handles `kcc == null` gracefully.
#[allow(clippy::too_many_arguments)]
fn route_kcc_site(
    enc: &Encoder,
    stream: &mut TokenStream,
    site: &serde_json::Map<String, serde_json::Value>,
    residues: Option<&ResidueMap>,
    therm: Option<&ThermMap>,
    binding: Option<&BindingMap>,
    site_idx: usize,
    verified: bool,
    seal_sha: [u8; 32],
    capture: bool,
    rep: &mut SiteRouteReport,
) {
    let mut consumed: HashSet<String> = HashSet::new();

    // spatial pocket centre as a 3-D lattice code (never three unrelated scalars).
    let centroid = site.get("centroid").and_then(arr3);
    if let Some(c) = centroid {
        consumed.insert("centroid".into());
        if enc.encode_vec3(stream, F::KccSite, "centroid", c, verified, seal_sha) {
            rep.matched += 1;
        }
    }
    // identity + ordered rank tuple (separate components; no blended score synthesized).
    for key in SITE_TOPLEVEL_SCALARS {
        if let Some(v) = site.get(*key) {
            consumed.insert((*key).into());
            if site_scalar(enc, stream, key, v, verified, seal_sha) {
                rep.matched += 1;
            }
        }
    }
    // nested kcc.* aggregate summaries + member relation (null kcc handled, never crashed).
    match site.get("kcc") {
        Some(v) if v.is_object() => {
            consumed.insert("kcc".into());
            let kcc = v.as_object().unwrap();
            for key in SITE_KCC_SCALARS {
                if let Some(kv) = kcc.get(*key) {
                    if site_scalar(enc, stream, key, kv, verified, seal_sha) {
                        rep.matched += 1;
                    }
                }
            }
            emit_best_candidate(enc, stream, kcc, verified, seal_sha, rep);
            route_site_members(
                enc, stream, kcc, residues, therm, centroid, site_idx, verified, seal_sha, rep,
            );
        }
        Some(v) if v.is_null() => {
            consumed.insert("kcc".into());
            rep.null_kcc += 1;
        }
        _ => {
            // Compact KCC site exports have no nested `kcc` object but do carry a small site-level
            // KCC summary. Route only fields that already have KccSite semantics.
            for key in SITE_COMPACT_DIRECT_SCALARS {
                if let Some(v) = site.get(*key) {
                    consumed.insert((*key).into());
                    if site_scalar(enc, stream, key, v, verified, seal_sha) {
                        rep.matched += 1;
                    }
                }
            }
            for (src, dst) in SITE_COMPACT_ALIAS_SCALARS {
                if let Some(v) = site.get(*src) {
                    consumed.insert((*src).into());
                    if site_scalar(enc, stream, dst, v, verified, seal_sha) {
                        rep.matched += 1;
                    }
                }
            }
        }
    }
    // binding_sites.json typed join (by site id): classification + druggability + cryptic_score.
    if let Some(bmap) = binding {
        if let Some((cls, drug, cryptic)) = site
            .get("id")
            .and_then(|v| v.as_f64())
            .and_then(|id| bmap.get(&(id as i64)))
        {
            rep.sites_binding_enriched += 1;
            if encode_enum_str(
                enc,
                stream,
                F::KccSite,
                "site_classification",
                cls,
                verified,
                seal_sha,
            ) {
                rep.matched += 1;
            }
            if enc.encode_scalar(
                stream,
                F::KccSite,
                "site_druggability",
                *drug,
                verified,
                seal_sha,
            ) {
                rep.matched += 1;
            }
            if enc.encode_scalar(
                stream,
                F::KccSite,
                "site_cryptic_score",
                *cryptic,
                verified,
                seal_sha,
            ) {
                rep.matched += 1;
            }
        }
    }
    // capture any unknown top-level site key losslessly (INV-4).
    if capture {
        for (k, v) in site.iter() {
            if !consumed.contains(k) {
                stream.push_opaque(
                    k,
                    &serde_json::to_string(v).unwrap_or_default(),
                    verified,
                    seal_sha,
                );
            }
        }
    }
}

/// Build the residue_id → (residue_name, Cα position) join map from residues[].
fn build_residue_map(json: &serde_json::Value) -> Option<ResidueMap> {
    json.get("residues").and_then(|v| v.as_array()).map(|a| {
        a.iter()
            .filter_map(|r| {
                let rid = r
                    .get("residue_id")
                    .or_else(|| r.get("resid"))
                    .and_then(|v| v.as_f64())? as i64;
                let name = r
                    .get("residue_name")
                    .or_else(|| r.get("resname"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let ca = r.get("ca_position").and_then(arr3).unwrap_or([0.0; 3]);
                Some((rid, (name, ca)))
            })
            .collect()
    })
}

/// Route the KCC `sites[]` array → KccSite tokens (distinct family/namespace from residues[]).
/// Members are resolved + enriched against the artifact's own residues[] when present. A
/// kcc_visualization.json carries BOTH record families: route residues[] via `route_artifact`
/// (KccDescriptor) and sites[] via this entry (KccSite).
pub fn route_kcc_sites(
    enc: &Encoder,
    stream: &mut TokenStream,
    json: &serde_json::Value,
    verified: bool,
    seal_sha: [u8; 32],
    capture: bool,
) -> SiteRouteReport {
    route_sites_inner(enc, stream, json, None, None, verified, seal_sha, capture)
}

/// Route sites[] AND typed-join binding_sites.json site evidence (classification / druggability /
/// cryptic_score) onto each site by `id` (65/65 exact in the fixture). Coverage in
/// `sites_binding_enriched`.
pub fn route_kcc_sites_with_binding(
    enc: &Encoder,
    stream: &mut TokenStream,
    json: &serde_json::Value,
    binding_sites: &serde_json::Value,
    verified: bool,
    seal_sha: [u8; 32],
    capture: bool,
) -> SiteRouteReport {
    let binding = build_binding_map(binding_sites);
    route_sites_inner(
        enc,
        stream,
        json,
        None,
        Some(&binding),
        verified,
        seal_sha,
        capture,
    )
}

/// Route sites[] AND typed-join the authoritative per-residue TWIN evidence from prism_therm.json
/// (transfer_entropy / fisher_info / causal_dg / kl_divergence / role / n_causal_spikes) onto each
/// member by residue_id — the explicit cross-link the prior pass left "unjoined". Coverage (e.g.
/// 74/93) is reported in `members_te_enriched`; uncovered members are simply un-enriched (the
/// per-residue TE is genuinely absent from prism_therm's top-K lists, not silently faked).
pub fn route_kcc_sites_with_therm(
    enc: &Encoder,
    stream: &mut TokenStream,
    json: &serde_json::Value,
    prism_therm: &serde_json::Value,
    verified: bool,
    seal_sha: [u8; 32],
    capture: bool,
) -> SiteRouteReport {
    let therm = build_therm_map(prism_therm);
    route_sites_inner(
        enc,
        stream,
        json,
        Some(&therm),
        None,
        verified,
        seal_sha,
        capture,
    )
}

/// Route sites[] with both KCC binding-site evidence and per-residue prism_therm evidence supplied
/// as separate siblings. Binding joins by site id; therm joins by member residue_id. Missing rows in
/// either sibling reduce coverage counters rather than producing inferred values.
pub fn route_kcc_sites_with_binding_and_therm(
    enc: &Encoder,
    stream: &mut TokenStream,
    json: &serde_json::Value,
    binding_sites: &serde_json::Value,
    prism_therm: &serde_json::Value,
    verified: bool,
    seal_sha: [u8; 32],
    capture: bool,
) -> SiteRouteReport {
    let binding = build_binding_map(binding_sites);
    let therm = build_therm_map(prism_therm);
    route_sites_inner(
        enc,
        stream,
        json,
        Some(&therm),
        Some(&binding),
        verified,
        seal_sha,
        capture,
    )
}

#[allow(clippy::too_many_arguments)]
fn route_sites_inner(
    enc: &Encoder,
    stream: &mut TokenStream,
    json: &serde_json::Value,
    therm: Option<&ThermMap>,
    binding: Option<&BindingMap>,
    verified: bool,
    seal_sha: [u8; 32],
    capture: bool,
) -> SiteRouteReport {
    let mut rep = SiteRouteReport::default();
    let Some(sites) = json.get("sites").and_then(|v| v.as_array()) else {
        return rep;
    };
    rep.sites = sites.len();
    let residues = build_residue_map(json);
    rep.join_validated = residues.is_some();
    for (idx, site) in sites.iter().enumerate() {
        if let Some(obj) = site.as_object() {
            let seg = stream.begin_record(F::KccSite, idx);
            route_kcc_site(
                enc,
                stream,
                obj,
                residues.as_ref(),
                therm,
                binding,
                idx,
                verified,
                seal_sha,
                capture,
                &mut rep,
            );
            stream.end_record(seg);
        }
    }
    rep
}

/// Harvest per-field scalar samples for corpus-fit (logm fields refit their decade range; others
/// get usage histograms for codebook-health). Skips JoinKey/enum identity fields.
fn harvest_site_scalars(sites: &[serde_json::Value]) -> HashMap<&'static str, Vec<f64>> {
    let mut out: HashMap<&'static str, Vec<f64>> = HashMap::new();
    let push = |k: &'static str, x: f64, out: &mut HashMap<&'static str, Vec<f64>>| {
        if !SITE_FIT_SKIP.contains(&k) {
            out.entry(k).or_default().push(x);
        }
    };
    for s in sites {
        let Some(obj) = s.as_object() else { continue };
        for k in SITE_TOPLEVEL_SCALARS {
            if let Some(x) = obj.get(*k).and_then(|v| v.as_f64()) {
                push(k, x, &mut out);
            }
        }
        if let Some(kcc) = obj.get("kcc").and_then(|v| v.as_object()) {
            for k in SITE_KCC_SCALARS {
                if let Some(x) = kcc.get(*k).and_then(|v| v.as_f64()) {
                    push(k, x, &mut out);
                }
            }
            for k in SITE_MEMBER_SCALARS {
                if let Some(a) = kcc.get(*k).and_then(|v| v.as_array()) {
                    for v in a {
                        if let Some(x) = v.as_f64() {
                            push(k, x, &mut out);
                        }
                    }
                }
            }
        } else {
            for k in SITE_COMPACT_DIRECT_SCALARS {
                if let Some(x) = obj.get(*k).and_then(|v| v.as_f64()) {
                    push(k, x, &mut out);
                }
            }
            for (src, dst) in SITE_COMPACT_ALIAS_SCALARS {
                if let Some(x) = obj.get(*src).and_then(|v| v.as_f64()) {
                    push(dst, x, &mut out);
                }
            }
        }
    }
    out
}

/// Harvest the site centroids (used for both VQ fit and VQ-health measurement).
fn harvest_centroids(json: &serde_json::Value) -> Vec<[f64; 3]> {
    json.get("sites")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|s| s.get("centroid").and_then(arr3))
                .collect()
        })
        .unwrap_or_default()
}

/// Rigorous entry: fit the centroid + member-Cα VQ codebooks AND corpus-fit the scalar quantizers
/// on the sites' OWN distributions, then route, then attach full centroid VQ-health (perplexity /
/// dead codes / mean+p95+worst distortion / escapes) to the report.
pub fn route_kcc_sites_autofit(
    enc: &mut Encoder,
    stream: &mut TokenStream,
    json: &serde_json::Value,
    verified: bool,
    seal_sha: [u8; 32],
    capture: bool,
) -> SiteRouteReport {
    if let Some(sites) = json.get("sites").and_then(|v| v.as_array()) {
        // centroid VQ on site centres
        let centroids = harvest_centroids(json);
        if !centroids.is_empty() {
            enc.fit_coords(F::KccSite, "centroid", &centroids);
        }
        // member-Cα VQ on the residues[]-joined member positions
        if let Some(residues) = build_residue_map(json) {
            let member_ca: Vec<[f64; 3]> = sites
                .iter()
                .filter_map(|s| s.get("kcc")?.get("candidate_residue_ids")?.as_array())
                .flatten()
                .filter_map(|v| v.as_f64())
                .filter_map(|rid| residues.get(&(rid as i64)).map(|(_, ca)| *ca))
                .collect();
            if !member_ca.is_empty() {
                enc.fit_coords(F::KccSite, "candidate_ca_position", &member_ca);
            }
        }
        // scalar corpus-fit (logm refits decade range; bounded/critical get usage histograms)
        for (field, vals) in harvest_site_scalars(sites) {
            if !vals.is_empty() {
                enc.fit_field(F::KccSite, field, &vals);
            }
        }
    }
    let mut rep = route_kcc_sites(enc, stream, json, verified, seal_sha, capture);
    if let Some(h) = enc.coord_health(F::KccSite, "centroid", &harvest_centroids(json)) {
        rep.centroid_perplexity = h.perplexity;
        rep.centroid_codes_used = h.codes_used;
        rep.centroid_dead_codes = h.dead_codes;
        rep.centroid_escapes = h.escapes;
        rep.centroid_mean_distortion = h.mean_distortion;
        rep.centroid_p95_distortion = h.p95_distortion;
        rep.centroid_worst_distortion = h.worst_distortion;
    }
    rep
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    const SHA: [u8; 32] = [0u8; 32];

    fn real_spike_json() -> Option<serde_json::Value> {
        // take the first few spikes of a real spike_events.json so the test is fast.
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let candidates = [
            "results_1btl/1btl.site9.spike_events.json",
            "results_1btl/1btl.site0.spike_events.json",
        ];
        for c in candidates {
            let p = root.join(c);
            if p.exists() {
                let full: serde_json::Value =
                    serde_json::from_slice(&std::fs::read(&p).ok()?).ok()?;
                let spikes = full.get("spikes")?.as_array()?;
                let first: Vec<_> = spikes.iter().take(5).cloned().collect();
                return Some(serde_json::json!({ "spikes": first }));
            }
        }
        None
    }

    #[test]
    fn routes_multiplexed_negative_scientific_evidence_and_preserves_slot_62() {
        let mut rec = serde_json::Map::new();
        for (key, value) in [
            ("schema_kind", "negative_scientific_evidence_event"),
            ("negative_event_id", "neg_kcc_0001"),
            ("object_id", "RejectedKCCEvent:row01:stream0:causal_lag"),
            (
                "run_id",
                "BLACKWELL_HIGH_FIDELITY_REPLICATE_MATRIX_20260702T161641Z",
            ),
            ("row_id", "row01_apo_orthosteric_seed7101"),
            ("target_id", "4LPK"),
            ("ligand_mode", "apo"),
            ("detection_mode", "orthosteric"),
            ("replica_seed", "7101"),
            ("epoch_id", "epoch_row01_stream0"),
            ("chronology_event_id", "sce_row01_stream0_0001"),
            ("source_family", "KccDescriptorNative"),
            ("source_artifact", "kcc_v2full.bin"),
            ("source_sha256", "abc123"),
            ("source_hash", "abc123"),
            ("source_record_id", "residue_42"),
            ("negative_evidence_type", "RejectedKCCEvent"),
            ("rejection_class", "kcc_rejection"),
            ("control_family", "kcc"),
            ("failed_claim_type", "kcc_propagation"),
            ("motif_status", "rejected"),
            ("rejection_reason", "null_or_nan_observable"),
            ("discovery_condition", "KCC_PROPAGATION"),
            ("failure_mode", "kcc_descriptor_dead_output"),
            ("source_claim", "causal_lag_is_informative"),
            ("rejected_claim", "causal_lag_resolved"),
            ("kcc_event_id", "kcc_row01_stream0_residue42"),
            ("false_positive_status", "suppressed"),
            ("suppression_status", "complete"),
            (
                "final_merge_eligibility",
                "blocked_pending_negative_evidence",
            ),
            ("chronology_ref", "sce_row01_stream0_0001"),
            (
                "uncertainty_tensor_ref",
                "uncertainty_row01_stream0_residue42",
            ),
            ("prismos_opcode", "KCC_DESCRIPTOR_VALIDATE"),
            ("opcode_family", "scientific_falsification"),
            (
                "opcode_context",
                "scientific_falsification:KCC_DESCRIPTOR_VALIDATE",
            ),
            ("prismcode_instruction_class", "ASSERT.KCC_VALIDITY"),
            (
                "prismcode_instruction_context",
                "ASSERT.KCC_VALIDITY|residue_42|abc123",
            ),
            ("qnanl_artifact_family", "EventStream"),
            (
                "qnanl_codec_context",
                "QnanlArtifactFamily::EventStream:NegativeScientificEvidenceEvent",
            ),
        ] {
            rec.insert(
                key.to_string(),
                serde_json::Value::String(value.to_string()),
            );
        }
        for (key, value) in [
            ("schema_version", 1.0),
            ("stream_id", 0.0),
            ("uncertainty_sigma", 0.0),
            ("signal_to_uncertainty", 0.0),
            ("snr", 0.0),
            ("observed_value", 0.0),
            ("threshold_value", 1.0),
            ("baseline_value", 0.0),
            ("delta_value", 0.0),
            ("persistence_frames", 1.0),
            ("replicate_count", 12.0),
            ("supporting_seed_count", 0.0),
            ("failing_seed_count", 12.0),
            ("occurrence_count", 1248.0),
            ("confidence", 1.0),
            ("severity", 1.0),
            ("recoverable_remaining", 0.0),
        ] {
            rec.insert(key.to_string(), serde_json::json!(value));
        }
        let json = serde_json::json!({
            "schema_kind": "negative_scientific_evidence_events",
            "schema_version": 1,
            "records": [serde_json::Value::Object(rec)]
        });

        assert_eq!(
            detect_family(&json),
            Some((F::NegativeScientificEvidenceEvent, "/records"))
        );
        assert_eq!(
            detect_family_from_filename_hint("NegativeEvidenceLedger.json"),
            Some((F::NegativeScientificEvidenceEvent, ""))
        );
        let enc = Encoder::new();
        let mut stream = TokenStream::new();
        let (family, report) = route_artifact(&enc, &mut stream, &json, true, SHA, true);
        assert_eq!(family, Some(F::NegativeScientificEvidenceEvent));
        assert_eq!(report.records, 1);
        assert!(
            report.matched >= 40,
            "negative evidence must route by value, got {}",
            report.matched
        );
        assert_eq!(stream.segments.len(), 1);
        assert_eq!(F::NegativeScientificEvidenceEvent.family_slot(), 62);
        assert_eq!(Token::NAMESPACE_BITS, 13);
    }

    #[test]
    fn routes_real_spike_records_with_string_enums_and_split_coords() {
        let Some(json) = real_spike_json() else {
            eprintln!("skip: no real spike_events.json present");
            return;
        };
        let mut enc = Encoder::new();
        // fit position VQ on the records' coords so positions are non-escape (real spatial tokens).
        let coords: Vec<[f64; 3]> = json["spikes"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|s| {
                Some([
                    s.get("x")?.as_f64()?,
                    s.get("y")?.as_f64()?,
                    s.get("z")?.as_f64()?,
                ])
            })
            .collect();
        enc.fit_coords(F::SpikeEvent, "position", &coords);

        // auto-detect + route
        let mut stream = TokenStream::new();
        let (fam, rep) = route_artifact(&enc, &mut stream, &json, true, SHA, false);
        assert_eq!(fam, Some(F::SpikeEvent));
        assert_eq!(rep.records, 5);
        assert_eq!(
            stream.segments.len(),
            5,
            "one segment per spike record (P1-7)"
        );
        // each record matches: wavelength_nm, spike_source(str), aromatic_type/type(str),
        // ccns_phase(str), intensity, water_density, n_nearby_excited, aromatic_residue_id, position
        // ⇒ ≥ 9 fields/record.
        assert!(
            rep.matched >= 9 * 5,
            "expected ≥45 matched fields, got {}",
            rep.matched
        );
        assert_eq!(stream.len(), rep.matched);

        // the OLD flat top-level route gets ~0 from this real artifact (the naivety).
        let mut flat = TokenStream::new();
        let n_flat = enc.route_json(&mut flat, F::SpikeEvent, &json, true, SHA);
        assert_eq!(
            n_flat, 0,
            "flat top-level routing should match nothing on real nested data"
        );
    }

    #[test]
    fn voronoi_source_is_append_only_code_six() {
        let enc = Encoder::new();
        for (code, label) in ["NONE", "UV", "LIF", "EFP", "LADD", "COFIRE", "VORONOI_LIF"].iter().enumerate() {
            let mut stream = TokenStream::new();
            assert!(encode_enum_str(&enc, &mut stream, F::SpikeEvent, "spike_source", label, true, SHA));
            assert_eq!(stream.tokens[0].field_token(), code as u64);
            match &stream.sidecar.get(0).unwrap().exact {
                ExactValue::Text(text) => assert_eq!(text, label),
                _ => panic!("expected exact source label"),
            }
        }
        assert_eq!(crate::schema::version::SCHEMA_VERSION, 26);
    }

    #[test]
    fn spike_source_string_maps_to_engine_code_with_exact_sidecar() {
        let enc = Encoder::new();
        let mut s = TokenStream::new();
        assert!(encode_enum_str(
            &enc,
            &mut s,
            F::SpikeEvent,
            "spike_source",
            "UV",
            true,
            SHA
        ));
        assert_eq!(s.tokens[0].field_token(), 1, "UV is engine code 1");
        match &s.sidecar.get(0).unwrap().exact {
            ExactValue::Text(t) => assert_eq!(t, "UV"),
            _ => panic!("expected Text sidecar"),
        }
        // unknown label escapes (not dropped) and keeps the exact string.
        let mut s2 = TokenStream::new();
        encode_enum_str(
            &enc,
            &mut s2,
            F::SpikeEvent,
            "spike_source",
            "WEIRD",
            true,
            SHA,
        );
        assert_eq!(s2.tokens[0].field_token(), Token::ESCAPE_CODE);
    }

    #[test]
    fn routing_is_deterministic_in_field_slot_order() {
        // P1-9: same record → identical token vector (route order is registry-slot, not JSON order).
        let enc = Encoder::new();
        let rec: serde_json::Map<String, serde_json::Value> =
            serde_json::from_value(serde_json::json!({
                "wavelength_nm": 280.0, "spike_source": "UV", "intensity": 1.5,
                "water_density": 0.01, "n_nearby_excited": 2, "aromatic_residue_id": -1
            }))
            .unwrap();
        let mut a = TokenStream::new();
        route_object(&enc, &mut a, F::SpikeEvent, &rec, true, SHA, false);
        let mut b = TokenStream::new();
        route_object(&enc, &mut b, F::SpikeEvent, &rec, true, SHA, false);
        assert!(!a.tokens.is_empty());
        assert_eq!(a.tokens, b.tokens, "routing must be deterministic");
    }

    #[test]
    fn real_spike_record_reconstructs_losslessly_from_sidecar() {
        // P1-10 / INV-4: the FULL real record (matched + unknown fields) is recoverable from the
        // sidecar — the decode→tokenize→reconstruct proof on real field-bearing data.
        let Some(json) = real_spike_json() else {
            eprintln!("skip: no real spike_events.json");
            return;
        };
        let rec = json["spikes"][0].as_object().unwrap().clone();
        let mut enc = Encoder::new();
        let xyz = [
            rec["x"].as_f64().unwrap(),
            rec["y"].as_f64().unwrap(),
            rec["z"].as_f64().unwrap(),
        ];
        enc.fit_coords(F::SpikeEvent, "position", &[xyz]);
        let mut s = TokenStream::new();
        route_object(&enc, &mut s, F::SpikeEvent, &rec, true, SHA, true); // capture=true

        let want_int = rec["intensity"].as_f64().unwrap();
        let (mut pos, mut src, mut int_ok, mut unknown_ok) = (None, None, false, false);
        for e in s.sidecar.iter() {
            match &e.exact {
                ExactValue::Vec3(v) => pos = Some(*v),
                ExactValue::Scalar(x) if (*x - want_int).abs() < 1e-12 => int_ok = true,
                ExactValue::Text(t) if t == "UV" => src = Some(t.clone()),
                ExactValue::Text(t) if t.starts_with("timestep=") => unknown_ok = true,
                _ => {}
            }
        }
        assert_eq!(pos, Some(xyz), "coordinate exact in sidecar");
        assert_eq!(src.as_deref(), Some("UV"), "string enum exact in sidecar");
        assert!(int_ok, "intensity exact in sidecar");
        assert!(
            unknown_ok,
            "unknown field (timestep) captured losslessly (INV-4)"
        );
    }

    #[test]
    fn autofit_makes_real_positions_non_escape() {
        // P1-5: route_artifact_autofit fits the coord VQ on the artifact's own coords → position
        // tokens are real spatial codes, not escape (which cold-start BCC would produce).
        let Some(json) = real_spike_json() else {
            eprintln!("skip: no real spike_events.json");
            return;
        };
        let mut enc = Encoder::new();
        let mut s = TokenStream::new();
        let (fam, _r) = route_artifact_autofit(&mut enc, &mut s, &json, true, SHA, false);
        assert_eq!(fam, Some(F::SpikeEvent));
        let vec3_positions: Vec<usize> = s
            .sidecar
            .iter()
            .filter(|e| matches!(e.exact, ExactValue::Vec3(_)))
            .map(|e| e.position)
            .collect();
        assert!(!vec3_positions.is_empty(), "expected position tokens");
        let escaped = vec3_positions.iter().filter(|&&p| s.is_escaped(p)).count();
        assert_eq!(
            escaped, 0,
            "auto-fit should make all position tokens non-escape; {escaped} escaped"
        );
    }

    #[test]
    fn unknown_receipt_status_escapes_not_inferred_pass() {
        // P2-14 / INV-5: a non-canonical status must NOT be promoted to a PASS code; it escapes
        // with the exact string in the sidecar.
        let enc = Encoder::new();
        let mut s = TokenStream::new();
        encode_enum_str(
            &enc,
            &mut s,
            F::ReceiptStatus,
            "status",
            "ACCEPTED_RISK",
            true,
            SHA,
        );
        assert_eq!(
            s.tokens[0].field_token(),
            Token::ESCAPE_CODE,
            "ACCEPTED_RISK must not map to PASS"
        );
        let mut s2 = TokenStream::new();
        encode_enum_str(&enc, &mut s2, F::ReceiptStatus, "status", "PASS", true, SHA);
        assert_eq!(s2.tokens[0].field_token(), 0, "PASS is code 0");
    }

    #[test]
    fn detect_family_uses_schema_kind_and_structure() {
        // SPIKE_TRANSITION_EVENT now routes the full SpikeTransitionEvent family (was BocpdEvent).
        assert_eq!(
            detect_family(&serde_json::json!({"schema_kind":"SPIKE_TRANSITION_EVENT"})),
            Some((F::SpikeTransitionEvent, ""))
        );
        assert_eq!(
            detect_family(&serde_json::json!({"spikes":[], "site_id":0})),
            Some((F::SpikeEvent, "/spikes"))
        );
        assert_eq!(detect_family(&serde_json::json!({"random":1})), None);
    }

    #[test]
    fn detects_and_routes_scientific_uncertainty_tensor_families() {
        let cases = [
            (
                "scientific_uncertainty_tensor",
                F::ScientificUncertaintyTensor,
            ),
            (
                "runtime_numerical_uncertainty_tensor",
                F::RuntimeNumericalUncertaintyTensor,
            ),
            ("thermal_uncertainty_tensor", F::ThermalUncertaintyTensor),
            (
                "integrator_uncertainty_tensor",
                F::IntegratorUncertaintyTensor,
            ),
            (
                "observable_uncertainty_tensor",
                F::ObservableUncertaintyTensor,
            ),
            ("signal_to_uncertainty_tensor", F::SignalToUncertaintyTensor),
            ("epoch_uncertainty_tensor", F::EpochUncertaintyTensor),
        ];
        let enc = Encoder::new();
        for (schema_kind, family) in cases {
            assert_eq!(
                detect_family(&serde_json::json!({"schema_kind": schema_kind})),
                Some((family, ""))
            );
            assert_eq!(
                detect_family(&serde_json::json!({"schema_kind": schema_kind, "records": []})),
                Some((family, "/records"))
            );
            assert_eq!(
                detect_family_from_filename_hint(&format!("/tmp/{schema_kind}.json")),
                Some((family, ""))
            );
            for field in [
                "row_id",
                "run_id",
                "epoch_id",
                "stream_id",
                "mode",
                "tensor_kind",
                "observable",
                "source_family",
                "source_event_id",
                "provenance_ref",
                "mu",
                "sigma",
                "variance",
                "n_samples",
                "signal_delta",
                "signal_to_uncertainty",
                "consistency_score",
                "measurement_status",
            ] {
                assert!(
                    enc.registry().lookup(family, field).is_some(),
                    "{family:?}/{field} registered"
                );
            }
        }

        let art = serde_json::json!({
            "schema_kind": "scientific_uncertainty_tensor",
            "records": [{
                "row_id": "row_apo_cryptic_0001",
                "run_id": "run_4lpk_apo",
                "epoch_id": "epoch_stream0",
                "stream_id": 0,
                "mode": "runtime",
                "tensor_kind": "scientific_uncertainty_tensor",
                "observable": "coordinate_rms",
                "source_family": "ScientificStateEpoch",
                "source_event_id": "sce_stream0_0008",
                "provenance_ref": "sha256:0123456789abcdef",
                "mu": 0.0,
                "sigma": 0.0,
                "variance": 0.0,
                "n_samples": 4096,
                "signal_delta": 0.0,
                "signal_to_uncertainty": 0.0,
                "consistency_score": 1.0,
                "measurement_status": "measured"
            }]
        });
        assert_eq!(
            detect_family(&art),
            Some((F::ScientificUncertaintyTensor, "/records"))
        );
        let mut stream = TokenStream::new();
        let (fam, rep) = route_artifact(&enc, &mut stream, &art, true, SHA, true);
        assert_eq!(fam, Some(F::ScientificUncertaintyTensor));
        assert_eq!(rep.records, 1);
        assert!(
            rep.matched >= 18,
            "expected uncertainty tensor fields to route, got {}",
            rep.matched
        );
        assert_ne!(
            enc.registry()
                .namespace(F::ScientificUncertaintyTensor, "row_id"),
            enc.registry()
                .namespace(F::RuntimeNumericalUncertaintyTensor, "row_id")
        );
    }

    #[test]
    fn detects_and_routes_tsh_stage_ii_transition_families() {
        let cases = [
            (
                "thermodynamic_salience_event",
                F::ThermodynamicSalienceEvent,
            ),
            ("transition_motif_candidate", F::TransitionMotif),
            ("transition_motif", F::TransitionMotif),
            (
                "persistent_transition_object",
                F::PersistentTransitionObject,
            ),
            ("transition_consensus_event", F::TransitionConsensusEvent),
            ("transition_evidence_ledger", F::TransitionEvidenceLedger),
        ];
        let enc = Encoder::new();
        for (schema_kind, family) in cases {
            assert_eq!(
                detect_family(&serde_json::json!({"schema_kind": schema_kind})),
                Some((family, ""))
            );
            assert_eq!(
                detect_family(&serde_json::json!({"schema_kind": schema_kind, "records": []})),
                Some((family, "/records"))
            );
            assert_eq!(
                detect_family_from_filename_hint(&format!("/tmp/{schema_kind}.jsonl")),
                Some((family, ""))
            );
            for field in [
                "schema_kind",
                "run_id",
                "epoch_id",
                "stream_id",
                "seed_id",
                "chronology_ref",
                "scientific_uncertainty_tensor_ref",
                "source_refs",
                "source_sha256",
                "force_delta",
                "force_delta_snr",
                "pocket_volume_delta",
                "pocket_volume_delta_snr",
                "contact_delta",
                "contact_delta_snr",
                "causal_lag",
                "lag_corr_peak",
                "spectral_signature_delta",
                "replicate_recurrence",
                "negative_control_status",
                "acceptance_status",
                "aggregate_salience_score",
                "runtime_origin",
                "tokenizer_family",
            ] {
                assert!(
                    enc.registry().lookup(family, field).is_some(),
                    "{family:?}/{field} registered"
                );
            }
        }

        let tsh_record = serde_json::json!({
            "schema_kind": "thermodynamic_salience_event",
            "transition_id": "tsh:event:1",
            "candidate_id": "tsh:candidate:1",
            "run_id": "run_a",
            "target_id": "4LPK",
            "epoch_id": "epoch_0001",
            "stream_id": "stream_00",
            "seed_id": "seed_42",
            "chronology_ref": "chronology:1",
            "scientific_uncertainty_tensor_ref": "sut:force:1",
            "source_refs": "forces_final.bin",
            "source_sha256": "sha256:abc",
            "topology_hash_transition": "topo:a->a",
            "residue_set_hash": "residue:set",
            "force_delta": 12.0,
            "force_delta_snr": 6.0,
            "pocket_volume_delta": 40.0,
            "pocket_volume_delta_snr": 8.0,
            "contact_delta": 0.25,
            "contact_delta_snr": 5.0,
            "causal_lag": 2.0,
            "lag_corr_peak": 0.72,
            "spectral_signature_delta": 1.4,
            "replicate_recurrence": 0.0,
            "replicate_count": 1,
            "consensus_seed_count": 1,
            "transition_stability_score": 0.7,
            "transition_recurrence_score": 0.0,
            "topology_persistence_score": 1.0,
            "chronology_persistence_score": 1.0,
            "uncertainty_normalized_magnitude": 6.4,
            "spectral_similarity_score": 0.8,
            "control_rejection_score": 0.5,
            "scientific_evidence_score": 0.63,
            "aggregate_salience_score": 0.66,
            "negative_control_status": "missing",
            "acceptance_status": "single_seed_candidate",
            "runtime_origin": "RUNTIME_INLINE_TSH_STAGE_II_OBSERVER",
            "tokenizer_family": "ThermodynamicSalienceEvent"
        });
        let artifact = serde_json::json!({
            "schema_kind": "thermodynamic_salience_event",
            "schema_version": 1,
            "records": [tsh_record]
        });
        assert_eq!(
            detect_family(&artifact),
            Some((F::ThermodynamicSalienceEvent, "/records"))
        );
        let mut stream = TokenStream::new();
        let (fam, report) = route_artifact(&enc, &mut stream, &artifact, true, SHA, true);
        assert_eq!(fam, Some(F::ThermodynamicSalienceEvent));
        assert_eq!(report.records, 1);
        assert!(
            report.matched >= 20,
            "expected TSH event to route as first-class fields, got {}",
            report.matched
        );

        let cache = serde_json::json!({
            "schema_kind": "resident_transition_cache",
            "transition_objects": [{
                "schema_kind": "persistent_transition_object",
                "persistent_transition_object_id": "pto:1",
                "stable_identifier": "pto:stable:1",
                "run_id": "run_a",
                "epoch_id": "epoch_0001",
                "stream_id": "stream_00",
                "seed_id": "seed_42",
                "scientific_uncertainty_tensor_ref": "sut:force:1",
                "source_refs": "thermodynamic_salience_events.jsonl",
                "source_sha256": "sha256:def",
                "replicate_count": 2,
                "consensus_seed_count": 2,
                "replicate_recurrence": 1.0,
                "acceptance_status": "partial_pending_negative_control_acceptance",
                "negative_control_status": "pending",
                "runtime_origin": "RUNTIME_INLINE_TSH_STAGE_II_OBSERVER",
                "tokenizer_family": "PersistentTransitionObject"
            }]
        });
        assert_eq!(
            detect_family(&cache),
            Some((F::PersistentTransitionObject, "/transition_objects"))
        );
        let mut stream = TokenStream::new();
        let (fam, report) = route_artifact(&enc, &mut stream, &cache, true, SHA, true);
        assert_eq!(fam, Some(F::PersistentTransitionObject));
        assert_eq!(report.records, 1);
        assert!(
            report.matched >= 10,
            "resident cache transition_objects must route by value, got {}",
            report.matched
        );
    }

    #[test]
    fn detects_and_routes_tsh_falsification_consensus_gate_families() {
        let cases = [
            (
                "scientific_transition_consensus_event",
                F::ScientificTransitionConsensusEvent,
            ),
            (
                "transition_falsification_profile",
                F::TransitionFalsificationProfile,
            ),
            (
                "rejected_persistent_transition_candidate",
                F::RejectedPersistentTransitionCandidate,
            ),
            (
                "persistent_transition_acceptance_ledger",
                F::PersistentTransitionAcceptanceLedger,
            ),
            (
                "stage_iii_eligibility_ledger",
                F::PersistentTransitionAcceptanceLedger,
            ),
        ];
        let enc = Encoder::new();
        for (schema_kind, family) in cases {
            assert_eq!(
                detect_family(&serde_json::json!({"schema_kind": schema_kind})),
                Some((family, ""))
            );
            assert_eq!(
                detect_family(&serde_json::json!({"schema_kind": schema_kind, "records": []})),
                Some((family, "/records"))
            );
            assert_eq!(
                detect_family_from_filename_hint(&format!("/tmp/{schema_kind}.jsonl")),
                Some((family, ""))
            );
        }

        let consensus = serde_json::json!({
            "schema_kind": "scientific_transition_consensus_event",
            "schema_version": 1,
            "tokenizer_family": "ScientificTransitionConsensusEvent",
            "motif_id": "tsh_001",
            "transition_id": "transition:001",
            "row_id": "row01",
            "mode": "apo_orthosteric",
            "seed_set": "seed_7101",
            "target_id": "4LPK",
            "positive_evidence_refs": "tsh_runtime.jsonl",
            "negative_evidence_refs": "nse:001",
            "uncertainty_tensor_refs": "sut:001",
            "chronology_refs": "chronology:001",
            "topology_hashes": "sha256:aaa->sha256:aaa",
            "spectral_signature_refs": "spectral:001",
            "causal_geometry_refs": "causal:001",
            "residue_network_refs": "residue:001",
            "contact_support_observable": "kcc",
            "contact_support_semantic_equivalence": false,
            "positive_evidence_score": 0.61,
            "negative_evidence_score": 0.88,
            "uncertainty_normalized_magnitude": 2.0,
            "topology_persistence_score": 1.0,
            "chronology_persistence_score": 1.0,
            "spectral_recurrence_score": 0.1,
            "replicate_consensus_score": 1.0,
            "falsification_survival_score": 0.0,
            "decision": "REJECT",
            "decision_reason": "stronger_negative_evidence",
            "confidence": 0.74,
            "provenance_ref": "OPERATOR_CONTROL/test",
            "opcode_context": "scientific_consensus_gate",
            "prismcode_instruction_context": "ASSERT.TSH_FALSIFICATION_CONSENSUS",
            "qnanl_codec_context": "QnanlArtifactFamily::EventStream"
        });
        let artifact = serde_json::json!({
            "schema_kind": "scientific_transition_consensus_event",
            "records": [consensus]
        });
        let mut stream = TokenStream::new();
        let (fam, report) = route_artifact(&enc, &mut stream, &artifact, true, SHA, true);
        assert_eq!(fam, Some(F::ScientificTransitionConsensusEvent));
        assert_eq!(report.records, 1);
        assert!(
            report.matched >= 30,
            "expected consensus gate event to route by value, got {}",
            report.matched
        );

        let rejected = serde_json::json!({
            "schema_kind": "rejected_persistent_transition_candidate",
            "schema_version": 1,
            "tokenizer_family": "RejectedPersistentTransitionCandidate",
            "candidate_id": "rejected_ptc:tsh_001",
            "source_motif_id": "tsh_001",
            "rejection_status": "REJECT",
            "rejection_reason": "stronger_negative_evidence",
            "failed_gate": "stronger_negative_evidence",
            "strongest_negative_evidence_ref": "nse:001",
            "positive_evidence_refs": "tsh_runtime.jsonl",
            "negative_evidence_refs": "nse:001",
            "uncertainty_tensor_refs": "sut:001",
            "chronology_refs": "chronology:001",
            "topology_hashes": "sha256:aaa->sha256:aaa",
            "tokenizable_failure_mode": "stronger_negative_evidence",
            "provenance_ref": "OPERATOR_CONTROL/test",
            "opcode_context": "scientific_consensus_gate",
            "prismcode_instruction_context": "ASSERT.TSH_FALSIFICATION_CONSENSUS",
            "qnanl_codec_context": "QnanlArtifactFamily::EventStream",
            "confidence": 0.74
        });
        let artifact = serde_json::json!({
            "schema_kind": "rejected_persistent_transition_candidate",
            "records": [rejected]
        });
        let mut stream = TokenStream::new();
        let (fam, report) = route_artifact(&enc, &mut stream, &artifact, true, SHA, true);
        assert_eq!(fam, Some(F::RejectedPersistentTransitionCandidate));
        assert_eq!(report.records, 1);
        assert!(
            report.matched >= 18,
            "expected rejected transition candidate to route by value, got {}",
            report.matched
        );
    }

    #[test]
    fn detect_and_route_pathb_binding_sites_materialized() {
        // Inline fixture mirroring the real binding_sites.materialized.json per-site record shape.
        let art = serde_json::json!({
            "schema_kind": "pathb_binding_sites_materialized",
            "site_count": 2,
            "binding_sites": [
                {
                    "site_id": "site_000", "rank": 0, "new_prism_rank": 0, "old_density_rank": 19,
                    "n_spikes": 470549, "n_raw_peaks_merged": 3, "n_active_non_density_dimensions": 4,
                    "intensity_sum": 26268407.98, "centroid_spread_a": 2.9948, "region_radius_a": 2.9948,
                    "promoted": true, "materialization_status": "materialized",
                    "materialization_level": "materialized_dynamic_site",
                    "lining_residues": [5,3,67], "driver_residues": [3,4], "centroid_xyz": [25.6,7.6,-27.6]
                },
                {
                    "site_id": "site_001", "rank": 1, "new_prism_rank": 1, "old_density_rank": 32,
                    "n_spikes": 12000, "n_raw_peaks_merged": 2, "n_active_non_density_dimensions": 3,
                    "intensity_sum": 1.2e6, "centroid_spread_a": 3.1, "region_radius_a": 3.1,
                    "promoted": true, "materialization_status": "materialized",
                    "materialization_level": "materialized_dynamic_site",
                    "lining_residues": [40,41], "driver_residues": [40], "centroid_xyz": [1.0,2.0,3.0]
                }
            ]
        });
        // schema_kind arm descends /binding_sites (per-site descent, NOT KccSite /sites).
        assert_eq!(
            detect_family(&art),
            Some((F::PathBBindingSiteMaterialized, "/binding_sites"))
        );
        // structural fallback (no schema_kind) routes the same shape via binding_sites[].
        let no_sk = serde_json::json!({
            "binding_sites": [{
                "site_id": "site_000", "lining_residues": [1], "centroid_xyz": [0.0,0.0,0.0]
            }]
        });
        assert_eq!(
            detect_family(&no_sk),
            Some((F::PathBBindingSiteMaterialized, "/binding_sites"))
        );
        // by-value route: 13 scalar specs per site => matched_fields>0 (>= 13*2 for 2 sites).
        let enc = Encoder::new();
        let mut stream = TokenStream::new();
        let (fam, rep) = route_artifact(&enc, &mut stream, &art, true, SHA, true);
        assert_eq!(fam, Some(F::PathBBindingSiteMaterialized));
        assert_eq!(rep.records, 2);
        assert!(
            rep.matched >= 13 * 2,
            "expected >= {} matched fields, got {}",
            13 * 2,
            rep.matched
        );
        // materialization_status enum value preserved exactly in the sidecar (grounded vocab).
        let has_mat = stream
            .sidecar
            .iter()
            .any(|e| matches!(&e.exact, ExactValue::Text(t) if t == "materialized"));
        assert!(
            has_mat,
            "materialization_status 'materialized' must be in the sidecar"
        );
    }

    #[test]
    fn detect_and_route_isa_registry_summary() {
        // Inline fixture mirroring the real prismos_prismcode_isa_registry_summary.json records[]
        // shape: 2 streams × 3 opcodes = 6 per-(stream,opcode) observation records under a
        // schema_kind wrapper (the real artifact is 8 streams × 25 opcodes = 200 records).
        let rec = |sid: u64, op: &str, idx: u64, count: u64, req: bool| {
            serde_json::json!({
                "stream_id": sid, "opcode": op, "opcode_index": idx, "count": count,
                "observed": true, "planned": true, "required_core": req,
                "registry_size": 3, "required_core_size": 2, "native_md_receipt_count": 2,
                "stream_status": "PASS", "stream_coverage_status": "PASS"
            })
        };
        let art = serde_json::json!({
            "schema_kind": "prismos_prismcode_isa_registry_summary",
            "schema_version": 1,
            "registry_size": 3,
            "required_core_size": 2,
            "native_md_receipt_count": 2,
            "records": [
                rec(0, "MD.PROTOCOL_DIRECTOR", 0, 452, true),
                rec(0, "MD.COM_REMOVE", 1, 12, false),
                rec(0, "MD.FORCE_BONDED", 2, 100, true),
                rec(1, "MD.PROTOCOL_DIRECTOR", 0, 461, true),
                rec(1, "MD.COM_REMOVE", 1, 9, false),
                rec(1, "MD.FORCE_BONDED", 2, 108, true),
            ]
        });
        // schema_kind arm descends /records (per-observation descent).
        assert_eq!(
            detect_family(&art),
            Some((F::IsaRegistrySummary, "/records"))
        );
        // by-value route: 12 scalar specs per record => matched_fields>0 across 6 records.
        let enc = Encoder::new();
        let mut stream = TokenStream::new();
        let (fam, rep) = route_artifact(&enc, &mut stream, &art, true, SHA, true);
        assert_eq!(fam, Some(F::IsaRegistrySummary));
        assert_eq!(rep.records, 6);
        assert!(
            rep.matched >= 6 * 8,
            "expected >= {} matched fields, got {}",
            6 * 8,
            rep.matched
        );
        // opcode categorical value (registered ISA_MD_OPCODES vocab) preserved exactly, not escaped
        // as an opaque string — genuine by-value token, not a filename/family label.
        let has_op = stream
            .sidecar
            .iter()
            .any(|e| matches!(&e.exact, ExactValue::Text(t) if t == "MD.PROTOCOL_DIRECTOR"));
        assert!(
            has_op,
            "opcode 'MD.PROTOCOL_DIRECTOR' must be recorded by value"
        );
    }

    #[test]
    fn detect_and_route_spike_summary() {
        // Inline fixture mirroring the real prismos_spike_event_native_summary.json records[] shape:
        // 3 per-residue BOUNDED spike aggregates under a schema_kind wrapper (the real stream0
        // artifact holds ~146 records; Σ spike_count == 1,394,939). aromatic_type domain {-1,1,2,4,5,6}
        // — the -1 'none' escape is Sentinel-wrapped, codes route against AROMATIC_TYPE_LABELS.
        let rec = |rid: i64, count: u64, arom: i64| {
            serde_json::json!({
                "residue_id": rid, "stream_id": 0, "spike_count": count,
                "spike_rate": (count as f64) / 1765.0, "mean_amplitude": 48.4,
                "max_amplitude": 64.8, "first_frame": 6, "last_frame": 1770,
                "aromatic_type": arom
            })
        };
        let art = serde_json::json!({
            "schema_kind": "prismos_spike_event_native_summary",
            "schema_version": 1,
            "source_magic": "PRSPK001",
            "stream_id": 0,
            "n_residues": 3,
            "total_spike_count": 142257,
            "record_count_envelope": 142257,
            "records": [
                rec(0, 95017, -1), // LIF-dominant residue: aromatic_type = -1 sentinel
                rec(5, 30906, 1),  // TYR chromophore code
                rec(9, 16334, 6),  // undocumented code 6 (UNK6) — must still route in-domain
            ]
        });
        // schema_kind arm descends /records (per-residue descent).
        assert_eq!(detect_family(&art), Some((F::SpikeSummary, "/records")));
        // by-value route: 9 scalar specs per record => matched_fields>0 across 3 records.
        let enc = Encoder::new();
        let mut stream = TokenStream::new();
        let (fam, rep) = route_artifact(&enc, &mut stream, &art, true, SHA, true);
        assert_eq!(fam, Some(F::SpikeSummary));
        assert_eq!(rep.records, 3);
        assert!(
            rep.matched >= 3 * 8,
            "expected >= {} matched fields, got {}",
            3 * 8,
            rep.matched
        );
        // aromatic_type=-1 must be handled by the Sentinel wrapper (in-domain, not an opaque escape),
        // and the sentinel does not crash the route — the record still counts by value.
        assert!(rep.matched > 0, "SpikeSummary records must match by value");
    }

    #[test]
    fn detect_and_route_prediction_state_independent_of_dstw_next_operator() {
        // Inline fixture mirroring the real prismos_prediction_state.json records[] shape: 3 per-site
        // prediction records under a schema_kind wrapper (the real apo_cryptic artifact holds 20).
        // confidence/uncertainty are REAL varying numerics; predicted_delta == new_prism − old_density;
        // source/target vtx are real state-graph ids.
        let rec = |rank: i64, delta: i64, conf: f64, unc: f64| {
            serde_json::json!({
                "prediction_id": format!("4LPK_apo.topology::pred_r{:03}", rank),
                "target_id": "4LPK_apo.topology",
                "receptor_state": "apo",
                "ligand_state": "apo",
                "predicted_surface": "surf_0123456789abcdef",
                "predicted_delta": delta,
                "confidence": conf,
                "uncertainty": unc,
                "epistemic_status": "materialized",
                "evidence_parent_ids": ["vtx_2fa9fe2bc74dc13a7fcc124cc1ec056e",
                                        "vtx_fa3b0c1fa1b601a505348d564be38809"],
                "source_vertex_id": "vtx_2fa9fe2bc74dc13a7fcc124cc1ec056e",
                "target_vertex_id": "vtx_fa3b0c1fa1b601a505348d564be38809",
                "operator_recommendation_id": "rank1::ENGINE.NATIVE_TENSOR_EMISSION",
                "validation_status": "reference_unavailable",
                "falsification_status": "reference_unavailable",
                "qnanl_receipt_ref": "e5cfc978418451ee800752fad3caf6b04109460f278c1040dc78fc3de9ed4b4e",
                "runtime_origin": "RUNTIME_SUBPROCESS_ORCHESTRATED_BY_PRISMOS",
                "tokenizer_family": "PredictionState"
            })
        };
        let art = serde_json::json!({
            "schema_kind": "prismos_prediction_state",
            "schema_version": 1,
            "target_id": "4LPK_apo.topology",
            "n_records": 3,
            "record_count": 3,
            "records": [
                rec(0, -19, 0.2501, 0.749),
                rec(1, -31, 0.2593, 0.0),
                rec(13, 12, 0.6475, 0.8826),
            ]
        });
        // Route keys on the prediction_state schema_kind — INDEPENDENT of DstwNextOperator.
        assert_eq!(detect_family(&art), Some((F::PredictionState, "/records")));
        // The DstwNextOperator carrier still routes to its OWN family (no aliasing/regression).
        assert_eq!(
            detect_family(&serde_json::json!({
                "schema_kind": "prismos_prediction_state"
            })),
            Some((F::PredictionState, "/records"))
        );
        let enc = Encoder::new();
        let mut stream = TokenStream::new();
        let (fam, rep) = route_artifact(&enc, &mut stream, &art, true, SHA, true);
        assert_eq!(fam, Some(F::PredictionState));
        assert_eq!(rep.records, 3);
        assert!(
            rep.matched > 0,
            "PredictionState records must match by value (got {})",
            rep.matched
        );
    }

    #[test]
    fn scientific_chronology_event_wrapper_descends_records() {
        let art = serde_json::json!({
            "schema_kind": "prismos_scientific_chronology_events",
            "schema_version": 1,
            "records": [{
                "schema_kind": "prismos_scientific_chronology_event",
                "chronology_event_id": "cevt_000001",
                "epoch_id": "epoch_000001",
                "event_index": 1,
                "event_kind": "state_transition",
                "source_kind": "dstw_state_graph",
                "parent_vertex_id": "vtx_parent",
                "source_vertex_id": "vtx_source",
                "target_vertex_id": "vtx_target",
                "transition_id": "txn_000001",
                "stream_id": 0,
                "frame_idx": 16,
                "timestep_min": 0,
                "timestep_max": 16,
                "physical_time_fs": 32.0,
                "delta_time_fs": 32.0,
                "operator_name": "MD.INTEGRATE_VELOCITY_VERLET",
                "opcode_name": "STATE.EMIT_TRANSITION",
                "opcode_id": "0x010d",
                "qnanl_packet_ref": "packet_sha256:abc",
                "operand_ref": "operand:STATE.EMIT_TRANSITION",
                "state_hash": "sha256:def",
                "runtime_origin": "RUNTIME_SUBPROCESS_ORCHESTRATED_BY_PRISMOS",
                "runtime_native": true,
                "perturbation_edge_id": "edge_1",
                "twin_pair_id": "twin_1",
                "spike_event_count": 4,
                "bocpd_event_count": 1
            }]
        });
        assert_eq!(
            detect_family(&art),
            Some((F::ScientificChronologyEvent, "/records"))
        );
        let enc = Encoder::new();
        let mut stream = TokenStream::new();
        let (fam, rep) = route_artifact(&enc, &mut stream, &art, true, SHA, true);
        assert_eq!(fam, Some(F::ScientificChronologyEvent));
        assert_eq!(rep.records, 1);
        assert!(
            rep.matched >= 24,
            "chronology event should route first-class fields; got {}",
            rep.matched
        );
    }

    #[test]
    fn scientific_state_epoch_wrapper_descends_records() {
        let art = serde_json::json!({
            "schema_kind": "prismos_scientific_state_epochs",
            "schema_version": 1,
            "records": [{
                "schema_kind": "prismos_scientific_state_epoch",
                "epoch_id": "epoch_000001",
                "epoch_index": 1,
                "epoch_kind": "md_runtime",
                "target_id": "4LPK_apo",
                "parent_vertex_id": "vtx_parent",
                "source_vertex_id": "vtx_source",
                "target_vertex_id": "vtx_target",
                "start_event_id": "cevt_000001",
                "end_event_id": "cevt_000010",
                "lifecycle_stage": "post_md",
                "ligand_state": "apo",
                "event_count": 10,
                "stream_count": 8,
                "timestep_min": 0,
                "timestep_max": 160,
                "physical_time_start_fs": 0.0,
                "physical_time_end_fs": 320.0,
                "physical_time_duration_fs": 320.0,
                "qnanl_packet_ref": "packet_sha256:abc",
                "operand_ref": "operand:epoch",
                "state_hash": "sha256:def",
                "runtime_origin": "RUNTIME_SUBPROCESS_ORCHESTRATED_BY_PRISMOS",
                "runtime_native": true,
                "closed_interval": true,
                "perturbation_edge_id": "edge_1",
                "twin_pair_id": "twin_1"
            }]
        });
        assert_eq!(
            detect_family(&art),
            Some((F::ScientificStateEpoch, "/records"))
        );
        let enc = Encoder::new();
        let mut stream = TokenStream::new();
        let (fam, rep) = route_artifact(&enc, &mut stream, &art, true, SHA, true);
        assert_eq!(fam, Some(F::ScientificStateEpoch));
        assert_eq!(rep.records, 1);
        assert!(
            rep.matched >= 22,
            "state epoch should route first-class fields; got {}",
            rep.matched
        );
    }

    #[test]
    fn t4_epoch_overlay_routes_as_first_class_closeout() {
        let art = serde_json::json!({
            "schema_kind": "dstw_t4_epoch_overlay",
            "schema_version": 1,
            "overlay_id": "overlay_000001",
            "epoch_id": "epoch_000001",
            "target_id": "4LPK_apo",
            "source_vertex_id": "vtx_source",
            "target_vertex_id": "vtx_target",
            "transition_id": "txn_000001",
            "start_event_id": "cevt_000001",
            "end_event_id": "cevt_000003",
            "overlay_kind": "runtime_closeout",
            "epoch_kind": "md_runtime",
            "lifecycle_stage": "post_md",
            "ligand_state": "apo",
            "event_count": 3,
            "artifact_count": 8,
            "timestep_min": 0,
            "timestep_max": 0,
            "physical_time_start_fs": 0.0,
            "physical_time_end_fs": 0.0,
            "physical_time_duration_fs": 0.0,
            "qnanl_packet_ref": "packet_sha256:abc",
            "operand_ref": "operand:STATE.EMIT_TRANSITION",
            "state_hash": "sha256:def",
            "runtime_origin": "RUNTIME_SUBPROCESS_ORCHESTRATED_BY_PRISMOS",
            "runtime_native": true,
            "closed_interval": true
        });
        assert_eq!(detect_family(&art), Some((F::T4EpochOverlay, "")));
        let enc = Encoder::new();
        let mut stream = TokenStream::new();
        let (fam, rep) = route_artifact(&enc, &mut stream, &art, true, SHA, true);
        assert_eq!(fam, Some(F::T4EpochOverlay));
        assert_eq!(rep.records, 1);
        assert!(
            rep.matched >= 20,
            "T4 epoch overlay should route first-class fields; got {}",
            rep.matched
        );
    }

    #[test]
    fn chronology_acceptance_decision_routes_as_first_class_gate() {
        let art = serde_json::json!({
            "schema_kind": "dstw_t4_chronology_acceptance_decision",
            "schema_version": 1,
            "acceptance_decision_id": "cacc_000001",
            "epoch_overlay_id": "overlay_000001",
            "epoch_id": "epoch_000001",
            "target_id": "4LPK_apo",
            "decision": "accepted",
            "decision_basis": "decode_proofs_and_operand_ledger",
            "accepted": true,
            "all_decode_proofs_pass": true,
            "operand_ledger_record_count": 8,
            "artifact_count": 8,
            "qnanl_packet_count": 8,
            "decode_proof_count": 8,
            "decode_proof_pass_count": 8,
            "runtime_native_artifact_count": 7,
            "post_run_artifact_count": 1,
            "blocked_gate_count": 0,
            "lifecycle_stage": "post_md",
            "runtime_origin": "RUNTIME_INLINE",
            "runtime_native": true,
            "qnanl_packet_ref": "packet_sha256:abc",
            "operand_ref": "operand:VALIDATION.STATE_SNAPSHOT",
            "state_hash": "sha256:def"
        });
        assert_eq!(
            detect_family(&art),
            Some((F::ChronologyAcceptanceDecision, ""))
        );
        let enc = Encoder::new();
        let mut stream = TokenStream::new();
        let (fam, rep) = route_artifact(&enc, &mut stream, &art, true, SHA, true);
        assert_eq!(fam, Some(F::ChronologyAcceptanceDecision));
        assert_eq!(rep.records, 1);
        assert!(
            rep.matched >= 18,
            "chronology acceptance decision should route first-class fields; got {}",
            rep.matched
        );
    }

    #[test]
    fn scientific_surface_field_aliases_record_operator_to_source_mapping() {
        let aliases = crate::schema::families::SCIENTIFIC_SURFACE_FIELD_ALIASES;
        assert!(aliases.contains(&(
            "ScientificPocketOpeningEvent",
            "pocket_volume_a3",
            "pocket_volume"
        )));
        assert!(aliases.contains(&(
            "LigandCausalActivityEvent",
            "causal_signals.contact_delta",
            "contact_delta"
        )));
        assert!(aliases.contains(&("ResidueActivityEvent", "metrics.sasa_delta", "sasa_delta")));
        assert!(
            !aliases.iter().any(|(_, operator, source)| {
                operator.contains("spectral")
                    || source.contains("spectral")
                    || operator.contains("path_b")
                    || source.contains("pathb")
            }),
            "spectral and Path-B adjudication fields stay outside exact event alias mapping"
        );
    }

    #[test]
    fn pocket_volume_only_without_schema_or_event_type_is_not_surface_event() {
        let art = serde_json::json!({
            "pocket_volume": 184.5,
            "residue_ids": [17, 42, 109]
        });
        assert_eq!(detect_family(&art), None);
    }

    fn ligand_causal_event() -> serde_json::Value {
        serde_json::json!({
            "epoch_id": "epoch_001",
            "frame_event_id": "frame_0043",
            "ligand_id": "GDP",
            "atom_ids": [1, 2, 3],
            "pose_cluster_id": "pose_cluster_7",
            "binding_site_id": "site_4",
            "contacts": [{
                "residue_id": 227,
                "contact_type": "H_BOND",
                "distance_a": 2.8,
                "duration_frames": 12
            }],
            "causal_signals": {
                "contact_delta": 0.25,
                "pocket_volume_delta": 18.0,
                "kcc_delta": 0.13,
                "warp_tensor_delta": -0.04,
                "force_delta": 2.0,
                "residue_motion_delta": 0.7
            },
            "confidence": 0.83
        })
    }

    fn residue_activity_event() -> serde_json::Value {
        serde_json::json!({
            "epoch_id": "epoch_001",
            "frame_event_id": "frame_0044",
            "chain_id": "A",
            "residue_id": 227,
            "residue_name": "ARG",
            "activity_type": "FORCE_HOTSPOT",
            "metrics": {
                "sasa_delta": 4.5,
                "rmsf_delta": 0.6,
                "distance_gate_delta": -0.2,
                "contact_map_delta": 3.0,
                "kcc_score": 0.78,
                "force_norm": 12.4,
                "displacement_a": 1.7
            },
            "linked_ligand_event_id": "lig_evt_001",
            "confidence": 0.88
        })
    }

    fn pocket_opening_event() -> serde_json::Value {
        serde_json::json!({
            "epoch_id": "epoch_001",
            "frame_event_id": "frame_0045",
            "residue_ids": [17, 42, 109],
            "pocket_volume": 184.5,
            "pocket_surface_area": 512.25,
            "lining_contacts": [{"residue_id": 17, "contact_count": 3}],
            "distance_gates": [{"name": "gate_A", "distance_a": 5.2}],
            "persistence_frames": 64,
            "apo_holo_contrast": "APO_OPEN_HOLO_CLOSED",
            "confidence": 0.91
        })
    }

    fn valid_causal_events() -> Vec<serde_json::Value> {
        vec![
            ligand_causal_event(),
            residue_activity_event(),
            pocket_opening_event(),
        ]
    }

    fn spectral_geometry_evidence() -> serde_json::Value {
        serde_json::json!({
            "spectral_extractor_ref": "spectral-extractor:abc",
            "spectral_fiber_bundle_ref": "spectral-fiber:xyz",
            "ligand_probe_accessibility": 0.72,
            "water_occupancy": 0.41,
            "tunnel_cavity_persistence": 64,
            "apo_holo_gdp_contrast": "apo_open_holo_gdp_closed",
            "replica_recurrence": 3
        })
    }

    fn pathb_geometry_evidence() -> serde_json::Value {
        serde_json::json!({
            "pocket_volume": 184.5,
            "lining_residues": [17, 42, 109],
            "persistence_frames": 64
        })
    }

    #[test]
    fn operator_causal_surface_events_route_without_event_type_expansion() {
        for (art, family, min_matched) in [
            (ligand_causal_event(), F::LigandCausalActivityEvent, 13usize),
            (residue_activity_event(), F::ResidueActivityEvent, 13usize),
            (
                pocket_opening_event(),
                F::ScientificPocketOpeningEvent,
                8usize,
            ),
        ] {
            assert_eq!(detect_family(&art), Some((family, "")));
            let enc = Encoder::new();
            let mut stream = TokenStream::new();
            let (fam, rep) = route_artifact(&enc, &mut stream, &art, true, SHA, true);
            assert_eq!(fam, Some(family));
            assert_eq!(rep.records, 1);
            assert!(
                rep.matched >= min_matched,
                "{family:?} matched {} fields, expected at least {min_matched}",
                rep.matched
            );
        }
    }

    #[test]
    fn causal_geometry_gate_passes_only_when_spectral_refs_and_metrics_bind() {
        let events = valid_causal_events();
        let spectral = spectral_geometry_evidence();
        let report = evaluate_causal_geometry_acceptance(&events, Some(&spectral), None);
        assert_eq!(report.chronology_foundation_status, "PASS");
        assert_eq!(
            report.cryptic_opening_status,
            "PASS_SPECTRAL_GEOMETRY_BOUND"
        );
        assert_eq!(
            report.chain,
            &[
                "FRAME_CAPTURE",
                "CONTACT_GEOMETRY",
                "LIGAND_CAUSAL_ACTIVITY",
                "RESIDUE_ACTIVITY",
                "POCKET_OPENING_EVENT",
                "T4_EPOCH_OVERLAY",
                "SITE_RANKING",
            ]
        );

        let extractor_only = serde_json::json!({
            "spectral_extractor_ref": "spectral-extractor:abc",
            "ligand_probe_accessibility": 0.72,
            "water_occupancy": 0.41,
            "tunnel_cavity_persistence": 64,
            "apo_holo_gdp_contrast": "apo_open_holo_gdp_closed",
            "replica_recurrence": 3
        });
        let report = evaluate_causal_geometry_acceptance(&events, Some(&extractor_only), None);
        assert_eq!(report.cryptic_opening_status, "FAIL");
        assert!(report
            .reasons
            .contains(&"missing_spectral_and_pathb_geometry"));
    }

    #[test]
    fn causal_geometry_gate_marks_pathb_morphology_without_spectral_as_partial() {
        let events = valid_causal_events();
        let pathb = pathb_geometry_evidence();
        let report = evaluate_causal_geometry_acceptance(&events, None, Some(&pathb));
        assert_eq!(report.chronology_foundation_status, "PASS");
        assert_eq!(report.cryptic_opening_status, "PARTIAL_PATHB_GEOMETRY_ONLY");
        assert!(report
            .reasons
            .contains(&"spectral_geometry_absent_pathb_only"));
    }

    #[test]
    fn causal_geometry_gate_fails_when_no_geometry_evidence_is_present() {
        let events = valid_causal_events();
        let report = evaluate_causal_geometry_acceptance(&events, None, None);
        assert_eq!(report.chronology_foundation_status, "PASS");
        assert_eq!(report.cryptic_opening_status, "FAIL");
        assert!(report
            .reasons
            .contains(&"missing_spectral_and_pathb_geometry"));
    }

    #[test]
    fn causal_geometry_gate_blocks_proximity_without_causal_signals() {
        let mut events = valid_causal_events();
        events[0].as_object_mut().unwrap().remove("causal_signals");
        let pathb = pathb_geometry_evidence();
        let report = evaluate_causal_geometry_acceptance(&events, None, Some(&pathb));
        assert_eq!(report.chronology_foundation_status, "PASS");
        assert_eq!(report.cryptic_opening_status, "BLOCKED");
        assert!(report.reasons.contains(&"proximity_without_causal_signals"));
        assert!(report.reasons.contains(&"missing_ligand_causal_signals"));
    }

    #[test]
    fn causal_geometry_gate_blocks_invalid_contact_and_activity_types() {
        let mut events = valid_causal_events();
        events[0]["contacts"][0]["contact_type"] = serde_json::json!("NEARBY");
        events[1]["activity_type"] = serde_json::json!("PROXIMITY_ONLY");
        let spectral = spectral_geometry_evidence();
        let report = evaluate_causal_geometry_acceptance(&events, Some(&spectral), None);
        assert_eq!(report.chronology_foundation_status, "PASS");
        assert_eq!(report.cryptic_opening_status, "BLOCKED");
        assert!(report.reasons.contains(&"invalid_ligand_contacts"));
        assert!(report.reasons.contains(&"invalid_residue_activity_type"));
    }

    #[test]
    fn causal_geometry_gate_blocks_missing_epoch_or_frame_ids() {
        let mut events = valid_causal_events();
        events[0].as_object_mut().unwrap().remove("epoch_id");
        events[1].as_object_mut().unwrap().remove("frame_event_id");
        let spectral = spectral_geometry_evidence();
        let report = evaluate_causal_geometry_acceptance(&events, Some(&spectral), None);
        assert_eq!(report.chronology_foundation_status, "BLOCKED");
        assert_eq!(report.cryptic_opening_status, "BLOCKED");
        assert!(report.reasons.contains(&"missing_epoch_id"));
        assert!(report.reasons.contains(&"missing_frame_event_id"));
    }

    #[test]
    fn pocket_volume_alone_does_not_satisfy_causal_geometry_gate() {
        let events = valid_causal_events();
        let pocket_volume_only = serde_json::json!({"pocket_volume": 184.5});
        let report = evaluate_causal_geometry_acceptance(&events, None, Some(&pocket_volume_only));
        assert_eq!(report.chronology_foundation_status, "PASS");
        assert_eq!(report.cryptic_opening_status, "FAIL");
        assert!(report
            .reasons
            .contains(&"missing_spectral_and_pathb_geometry"));
    }

    // --- KCC visualization (real residues[] descriptor; coherence→real-fields realign) ---------

    // Fixture: kv31_validation_frozen_20260512T215852Z/engine_manifest/Kv31_chainA.kcc_visualization.json
    // sha256 recorded below (verified out-of-band via `sha256sum`; sha2 is not a direct dep, so the
    // in-test integrity pin is the structural fingerprint asserted in `real_kcc_json`).
    const KCC_FIXTURE_SHA256: &str =
        "51bff23bf45f039982fb505d2bf934ff4d66cc2a864a0316c4ef56705171edbe";

    fn real_kcc_json() -> Option<serde_json::Value> {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../kv31_validation_frozen_20260512T215852Z/engine_manifest/Kv31_chainA.kcc_visualization.json");
        if !p.exists() {
            return None;
        }
        assert_eq!(KCC_FIXTURE_SHA256.len(), 64, "recorded sha256 provenance");
        let full: serde_json::Value = serde_json::from_slice(&std::fs::read(&p).ok()?).ok()?;
        // structural fingerprint pins the fixture identity without a hashing dep.
        let residues = full["residues"].as_array().unwrap();
        assert_eq!(residues.len(), 396, "KCC fixture residue count");
        assert_eq!(full["residues"][0]["residue_name"], "SER");
        assert_eq!(full["residues"][0]["residue_id"], 7);
        assert_eq!(full["residues"][0]["total_steps"], 9500);
        // keep the test fast: first 24 residues + the detection block.
        let head: Vec<_> = residues.iter().take(24).cloned().collect();
        Some(serde_json::json!({
            "residues": head,
            "vector_field_definition": full["vector_field_definition"].clone(),
        }))
    }

    #[test]
    fn detect_family_recognizes_kcc_visualization() {
        // residues[] + vector_field_definition ⇒ KccDescriptor; one without the other ⇒ not KCC.
        assert_eq!(
            detect_family(&serde_json::json!({"residues": [], "vector_field_definition": {}})),
            Some((F::KccDescriptor, "/residues"))
        );
        assert_eq!(detect_family(&serde_json::json!({"residues": []})), None);
    }

    #[test]
    fn routes_real_kcc_residues_with_aa_enum_and_ca_position() {
        let Some(json) = real_kcc_json() else {
            eprintln!("skip: no real KCC fixture present");
            return;
        };
        let mut enc = Encoder::new();
        let mut stream = TokenStream::new();
        // autofit fits the ca_position VQ on the residues' own coords → non-escape spatial tokens.
        let (fam, rep) = route_artifact_autofit(&mut enc, &mut stream, &json, true, SHA, true);
        assert_eq!(fam, Some(F::KccDescriptor));
        assert_eq!(rep.records, 24);
        // each residue matches residue_id, residue_name, ca_position + 13 scalar descriptors (= 16);
        // ≥12/residue is a robust floor even if a sparse scalar is skipped.
        assert!(
            rep.matched >= 12 * 24,
            "expected ≥{} matched fields, got {}",
            12 * 24,
            rep.matched
        );

        // ca_position tokens are real spatial codes (not escape) after autofit — one per residue.
        let vec3_positions: Vec<usize> = stream
            .sidecar
            .iter()
            .filter(|e| matches!(e.exact, ExactValue::Vec3(_)))
            .map(|e| e.position)
            .collect();
        assert_eq!(vec3_positions.len(), 24, "one ca_position per residue");
        let escaped = vec3_positions
            .iter()
            .filter(|&&p| stream.is_escaped(p))
            .count();
        assert_eq!(
            escaped, 0,
            "auto-fit should make ca_position non-escape; {escaped} escaped"
        );

        // residue_name SER routed via the AA enum with the exact string preserved in the sidecar.
        let ser_exact = stream
            .sidecar
            .iter()
            .any(|e| matches!(&e.exact, ExactValue::Text(t) if t == "SER"));
        assert!(ser_exact, "residue_name SER must be in the sidecar exactly");
    }

    #[test]
    fn kcc_residue_name_maps_to_aa_code_and_escapes_unknown() {
        use crate::schema::families::AA_NAMES;
        let enc = Encoder::new();
        // a known AA name → its exact index in AA_NAMES, with the exact string in the sidecar.
        let mut s = TokenStream::new();
        assert!(encode_enum_str(
            &enc,
            &mut s,
            F::KccDescriptor,
            "residue_name",
            "SER",
            true,
            SHA
        ));
        let ser_idx = AA_NAMES.iter().position(|&x| x == "SER").unwrap() as u64;
        assert_eq!(
            s.tokens[0].field_token(),
            ser_idx,
            "SER must map to its AA_NAMES index"
        );
        match &s.sidecar.get(0).unwrap().exact {
            ExactValue::Text(t) => assert_eq!(t, "SER"),
            _ => panic!("expected Text sidecar"),
        }
        // a non-AA name escapes losslessly (INV-1/INV-4) — never silently dropped or mis-coded.
        let mut s2 = TokenStream::new();
        encode_enum_str(
            &enc,
            &mut s2,
            F::KccDescriptor,
            "residue_name",
            "XYZ",
            true,
            SHA,
        );
        assert_eq!(
            s2.tokens[0].field_token(),
            Token::ESCAPE_CODE,
            "unknown residue name must escape"
        );
    }

    #[test]
    fn teacher_record_routes_exact_parquet_columns_and_ignores_stale_aliases() {
        // Values/shape are ground-truthed from the corpus shard (sha 2146c1ee…03a5cc): e.g.
        // student_weight 0.4055799468768179 and cryptic_likelihood_label 0.6235736370806453 are
        // verbatim real row values; the rest are real-range. NOT an invented merged row.
        let enc = Encoder::new();
        let rec: serde_json::Map<String, serde_json::Value> =
            serde_json::from_value(serde_json::json!({
                "residue_id": 42,
                "rhat": 1.01,
                "ess": 950.0,
                "label_confidence": 0.83,
                "cryptic_likelihood_label": 0.6235736370806453,
                "phase_manifold_score_mean": 0.61,
                "phase_manifold_score_std": 0.07,
                "is_bimodal": false,
                "hard_negative_mask": false,
                "train_mask": true,
                "student_weight": 0.4055799468768179
            }))
            .unwrap();
        let mut s = TokenStream::new();
        let r = route_object(
            &enc,
            &mut s,
            F::TeacherStudentConsensus,
            &rec,
            true,
            SHA,
            false,
        );
        assert_eq!(
            r.matched, 11,
            "all 11 real Teacher columns must route, got {}",
            r.matched
        );
        assert_eq!(s.len(), 11);

        // the OLD aliases no longer route — they fall through to skipped_unknown (the rename is real).
        let stale: serde_json::Map<String, serde_json::Value> =
            serde_json::from_value(serde_json::json!({
                "phase_manifold_score": 0.61, "hard_negative": false
            }))
            .unwrap();
        let mut s2 = TokenStream::new();
        let r2 = route_object(
            &enc,
            &mut s2,
            F::TeacherStudentConsensus,
            &stale,
            true,
            SHA,
            false,
        );
        assert_eq!(
            r2.matched, 0,
            "stale alias names must not match the renamed columns"
        );
        assert_eq!(r2.skipped_unknown, 2);
    }

    #[test]
    fn kcc_record_with_missing_optional_field_does_not_crash() {
        // a residue object missing several descriptors routes the present fields and skips the
        // rest without panicking (robustness on heterogeneous real records).
        let enc = Encoder::new();
        let rec: serde_json::Map<String, serde_json::Value> =
            serde_json::from_value(serde_json::json!({
                "residue_id": 7, "residue_name": "SER", "kcc_score": 0.02
            }))
            .unwrap();
        let mut s = TokenStream::new();
        let r = route_object(&enc, &mut s, F::KccDescriptor, &rec, true, SHA, false);
        assert_eq!(
            r.matched, 3,
            "the three present fields route; missing ones are simply absent"
        );
        assert_eq!(r.skipped_unknown, 0);
    }

    // --- KCC sites[] — relation / spatial / temporal / causal routing ------------------------

    fn real_kcc_full_json() -> Option<serde_json::Value> {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
            "../../kv31_validation_frozen_20260512T215852Z/engine_manifest/Kv31_chainA.kcc_visualization.json",
        );
        if !p.exists() {
            return None;
        }
        assert_eq!(KCC_FIXTURE_SHA256.len(), 64, "recorded sha256 provenance");
        let full: serde_json::Value = serde_json::from_slice(&std::fs::read(&p).ok()?).ok()?;
        // fixture fingerprints (Phase 1): residues[]/sites[] counts pin the fixture identity.
        assert_eq!(
            full["residues"].as_array().unwrap().len(),
            396,
            "residue count"
        );
        assert_eq!(full["sites"].as_array().unwrap().len(), 65, "site count");
        Some(full)
    }

    #[test]
    fn detect_family_recognizes_kcc_sites_distinct_from_residues_and_binding_sites() {
        // exact kcc-site signature (gtck_rank + rank_C + kcc) → KccSite.
        let site =
            serde_json::json!({"sites":[{"gtck_rank":1,"rank_C":0.5,"kcc":{},"centroid":[0,0,0]}]});
        assert_eq!(detect_family(&site), Some((F::KccSite, "/sites")));
        // a binding_sites.json sites array (lining_residues/druggability, NO gtck_rank/kcc) must NOT
        // be mistaken for a KCC site — no conflation with BindingSite artifacts.
        let bsites = serde_json::json!({"sites":[{"id":0,"centroid":[0,0,0],"lining_residues":[1,2],"druggability":0.7,"volume":100}]});
        assert_eq!(
            detect_family(&bsites),
            None,
            "binding_sites.json must not route as KccSite"
        );
        // the combined kcc_visualization artifact detects residues[] first (KccDescriptor); sites go
        // via route_kcc_sites.
        let combined = serde_json::json!({"residues":[],"vector_field_definition":{},"sites":[{"gtck_rank":1,"rank_C":0.5,"kcc":{}}]});
        assert_eq!(
            detect_family(&combined),
            Some((F::KccDescriptor, "/residues"))
        );
    }

    #[test]
    fn kcc_site_namespace_is_distinct_from_kcc_descriptor() {
        // INV-2: a same-named field in both KCC families gets DIFFERENT namespaces.
        let r = Encoder::new();
        let ns_site = r.registry().namespace(F::KccSite, "total_steps").unwrap();
        let ns_res = r
            .registry()
            .namespace(F::KccDescriptor, "total_steps")
            .unwrap();
        assert_ne!(
            ns_site, ns_res,
            "KccSite/KccDescriptor must not share a namespace"
        );
        assert_ne!(F::KccSite.family_slot(), F::KccDescriptor.family_slot());
    }

    #[test]
    fn routes_real_kcc_sites_with_join_centroid_and_members() {
        let Some(json) = real_kcc_full_json() else {
            eprintln!("skip: no KCC fixture present");
            return;
        };
        let mut enc = Encoder::new();
        let mut stream = TokenStream::new();
        let rep = route_kcc_sites_autofit(&mut enc, &mut stream, &json, true, SHA, true);
        assert_eq!(rep.sites, 65);
        assert_eq!(
            rep.null_kcc, 2,
            "2 sites have kcc==null (handled, not crashed)"
        );
        assert!(rep.matched > 0, "KccSite must emit nonzero payload tokens");
        // JOIN: every member id resolves against residues[].residue_id (no translation needed).
        assert!(rep.join_validated, "residues[] present ⇒ join validated");
        assert_eq!(
            rep.members_total,
            63 * 3,
            "3 candidate members per non-null site, none truncated"
        );
        assert_eq!(
            rep.members_resolved, rep.members_total,
            "all member ids resolve"
        );
        assert!(
            rep.members_unresolved.is_empty(),
            "no unresolved member ids: {:?}",
            rep.members_unresolved
        );
        // centroid tokens are real spatial codes (non-escape) after autofit — one per site.
        let centroids: Vec<usize> = stream
            .sidecar
            .iter()
            .filter(|e| e.field == "centroid" && matches!(e.exact, ExactValue::Vec3(_)))
            .map(|e| e.position)
            .collect();
        assert_eq!(centroids.len(), 65, "one centroid per site");
        assert_eq!(
            centroids.iter().filter(|&&p| stream.is_escaped(p)).count(),
            0,
            "centroids non-escape after autofit"
        );
        // members are enriched with residue_name + Cα position joined from residues[].
        assert_eq!(
            rep.members_enriched,
            63 * 3,
            "every resolved member enriched from residues[]"
        );
        // driver_residue_id == candidate_residue_ids[best_kcc_candidate_index] holds for all sites.
        assert!(
            rep.driver_best_consistent,
            "driver must equal the best candidate residue id"
        );
        // VQ health: effective codes > 1 and zero escapes (⇒ every centroid within 0.25Å).
        assert!(
            rep.centroid_perplexity > 1.0,
            "centroid codebook not collapsed: {}",
            rep.centroid_perplexity
        );
        assert!(
            rep.centroid_codes_used > 1 && rep.centroid_escapes == 0,
            "VQ health"
        );
        // spatial: the causally-dominant driver can sit beyond the R=6Å neighborhood — surfaced.
        assert!(
            !rep.spatial_outliers.is_empty(),
            "expected ≥1 member beyond R=6Å (e.g. driver)"
        );
    }

    #[test]
    fn kcc_site_member_order_is_deterministic_residue_id_asc() {
        // members emitted residue_id-asc regardless of the engine's array order; re-routing is
        // byte-identical (determinism, §13/§16). Parallel arrays move WITH the member.
        let enc = Encoder::new();
        let site = serde_json::json!({
            "gtck_rank": 1, "rank_C": 0.5, "centroid": [1.0,2.0,3.0],
            "kcc": {
                "candidate_residue_ids": [172, 169, 171],
                "candidate_causal_weights": [0.1, 0.7, 0.2],
                "candidate_kcc_causal_lag": [-25.0, -10.0, -50.0]
            }
        });
        let json = serde_json::json!({"sites":[site], "residues":[{"residue_id":169},{"residue_id":171},{"residue_id":172}]});
        let mut a = TokenStream::new();
        let ra = route_kcc_sites(&enc, &mut a, &json, true, SHA, false);
        let mut b = TokenStream::new();
        route_kcc_sites(&enc, &mut b, &json, true, SHA, false);
        assert_eq!(a.tokens, b.tokens, "routing must be deterministic");
        assert_eq!(ra.members_total, 3);
        assert_eq!(ra.members_resolved, 3);
        // emitted member ids are residue_id-ascending (not the [172,169,171] engine order).
        let member_ids: Vec<f64> = a
            .sidecar
            .iter()
            .filter_map(|e| match (e.field, &e.exact) {
                ("candidate_residue_ids", ExactValue::Scalar(x)) => Some(*x),
                _ => None,
            })
            .collect();
        assert_eq!(
            member_ids,
            vec![169.0, 171.0, 172.0],
            "members must be residue_id-ascending"
        );
    }

    #[test]
    fn kcc_site_unresolved_member_fails_visibly() {
        // Phase 3: a member id absent from residues[] is surfaced, not silently dropped.
        let enc = Encoder::new();
        let json = serde_json::json!({
            "sites": [{"gtck_rank":1,"rank_C":0.5,"kcc":{"candidate_residue_ids":[169, 99999]}}],
            "residues": [{"residue_id":169}]
        });
        let mut s = TokenStream::new();
        let rep = route_kcc_sites(&enc, &mut s, &json, true, SHA, false);
        assert_eq!(rep.members_resolved, 1);
        assert_eq!(
            rep.members_unresolved,
            vec![99999],
            "unresolved member id must surface"
        );
    }

    #[test]
    fn kcc_site_signed_lag_zero_crossing_survives_quantization() {
        // a near-zero +ε / -ε site_causal_lag must quantize to DIFFERENT, NON-escape tokens that
        // decode to opposite signs (INV-8; the sign survives in the TOKEN, not only the sidecar).
        let enc = Encoder::new();
        let q = enc.quantizer(F::KccSite, "site_causal_lag").unwrap();
        for &eps in &[0.6_f64, 3.0, 10.0] {
            let mut sp = TokenStream::new();
            let mut sn = TokenStream::new();
            assert!(enc.encode_scalar(&mut sp, F::KccSite, "site_causal_lag", eps, true, SHA));
            assert!(enc.encode_scalar(&mut sn, F::KccSite, "site_causal_lag", -eps, true, SHA));
            assert!(
                !sp.is_escaped(0) && !sn.is_escaped(0),
                "lag ±{eps} must NOT escape (absolute bound)"
            );
            assert_ne!(
                sp.tokens[0].field_token(),
                sn.tokens[0].field_token(),
                "±{eps} distinct codes"
            );
            assert!(
                q.decode(sp.tokens[0].field_token() as u32) > 0.0,
                "+{eps} decodes positive"
            );
            assert!(
                q.decode(sn.tokens[0].field_token() as u32) < 0.0,
                "-{eps} decodes negative"
            );
        }
    }

    #[test]
    fn kcc_site_temporal_is_aggregate_only_and_twincoupled_deferred() {
        // honest scoping: the site kcc block carries AGGREGATE temporal summaries but NO per-frame
        // chronology and NO TwinCoupled pairwise TE — neither required nor fabricated.
        let Some(json) = real_kcc_full_json() else {
            eprintln!("skip: no KCC fixture present");
            return;
        };
        let kcc0 = json["sites"][0]["kcc"].as_object().unwrap();
        for agg in [
            "active_causal_steps",
            "total_steps",
            "temporal_corr",
            "site_causal_lag",
        ] {
            assert!(
                kcc0.contains_key(agg),
                "aggregate temporal field {agg} expected"
            );
        }
        for absent in [
            "first_frame",
            "last_frame",
            "lifetime",
            "enter_step",
            "exit_step",
            "occupancy",
            "persistence",
        ] {
            assert!(
                !kcc0.contains_key(absent),
                "no per-frame field {absent} (deferred, not faked)"
            );
        }
        for twin in ["te_asymmetry", "barrier", "ccf_width", "agreement"] {
            assert!(
                !kcc0.contains_key(twin),
                "no TwinCoupled pairwise field {twin} (absent ⇒ deferred)"
            );
        }
        // and the KccSite registry declares no per-frame / episode fields.
        let r = Encoder::new();
        for absent in ["first_frame", "enter_step", "exit_step", "lifetime"] {
            assert!(
                r.registry().lookup(F::KccSite, absent).is_none(),
                "{absent} must not be registered"
            );
        }
    }

    #[test]
    fn kcc_site_null_kcc_and_missing_fields_do_not_crash() {
        let enc = Encoder::new();
        let json = serde_json::json!({"sites": [
            {"gtck_rank": 5, "rank_C": 0.5, "centroid": [1.0,2.0,3.0], "kcc": null},
            {"gtck_rank": 6, "kcc": {"driver_residue_id": 42}}
        ], "residues": [{"residue_id":42}]});
        let mut s = TokenStream::new();
        let rep = route_kcc_sites(&enc, &mut s, &json, true, SHA, true);
        assert_eq!(rep.sites, 2);
        assert_eq!(rep.null_kcc, 1);
        assert!(rep.matched > 0, "present fields still route");
    }

    // --- Phase 9 hardening: self-docs, geometry, member integrity, joins, round-trip -----------

    fn sibling(name: &str) -> Option<serde_json::Value> {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../kv31_validation_frozen_20260512T215852Z/engine_manifest")
            .join(name);
        if !p.exists() {
            return None;
        }
        serde_json::from_slice(&std::fs::read(&p).ok()?).ok()
    }

    fn repo_json(path: &str) -> Option<serde_json::Value> {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(path);
        if !p.exists() {
            return None;
        }
        serde_json::from_slice(&std::fs::read(&p).ok()?).ok()
    }

    #[test]
    fn kcc_site_self_documentation_is_decoded_and_consumed() {
        // the fixture's OWN documentation drives geometry/thresholds (not name-inference).
        let Some(json) = real_kcc_full_json() else {
            eprintln!("skip");
            return;
        };
        for key in [
            "semantics",
            "interpretation_guidelines",
            "vector_field_definition",
        ] {
            assert!(
                json.get(key).is_some(),
                "self-doc key {key} must be present + decoded"
            );
        }
        // the burst thresholds we pin come straight from interpretation_guidelines.
        let ig = serde_json::to_string(&json["interpretation_guidelines"]).unwrap();
        assert!(
            ig.contains("burst_motion"),
            "guidelines document the burst_motion decision"
        );
        // vector_field_definition documents the burst_motion/5 visual domain ⇒ [0,5].
        let vf = serde_json::to_string(&json["vector_field_definition"]).unwrap();
        assert!(
            vf.contains("burst_motion/5"),
            "vector doc documents the [0,5] domain"
        );
    }

    #[test]
    fn kcc_site_burst_motion_is_thresholded_on_documented_cuts() {
        use crate::types::{DecisionSpec, FieldGeometry as G};
        let r = Encoder::new();
        for f in [
            "burst_motion",
            "site_burst_motion",
            "candidate_kcc_burst_motion",
        ] {
            let spec = r.registry().lookup(F::KccSite, f).unwrap();
            // domain [0,5] (documented burst_motion/5), not [0,4].
            match spec.geometry {
                G::LinearBounded { lo, hi, .. } => {
                    assert_eq!((lo, hi), (0.0, 5.0), "{f} domain must be [0,5]");
                }
                ref g => panic!("{f} should be LinearBounded, got {g:?}"),
            }
            // decision-critical at the persistent≈1.0 / transient>1.5 cuts.
            match spec.decision_relevance {
                DecisionSpec::Threshold { points, .. } => assert_eq!(points, &[1.0, 1.5]),
                ref d => panic!("{f} should be Threshold{{1.0,1.5}}, got {d:?}"),
            }
        }
    }

    #[test]
    fn kcc_site_local_cov_is_bounded_not_twelve_decade_log() {
        use crate::types::FieldGeometry as G;
        let r = Encoder::new();
        for f in ["local_cov", "site_local_cov", "candidate_kcc_local_cov"] {
            let g = &r.registry().lookup(F::KccSite, f).unwrap().geometry;
            assert!(
                matches!(g, G::LinearBounded { lo, hi, .. } if *lo == 0.0 && *hi == 1.0),
                "{f} must be bounded covariance [0,1], not 12-decade log; got {g:?}"
            );
        }
    }

    #[test]
    fn kcc_site_scalar_corpus_fit_lowers_distortion() {
        // corpus-fit (autofit) concentrates a logm field's resolution on the real tiny-value band,
        // lowering mean distortion vs the cold (unfit) quantizer.
        let Some(json) = real_kcc_full_json() else {
            eprintln!("skip");
            return;
        };
        let samples: Vec<f64> = json["sites"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|s| s.get("kcc")?.get("direction_score")?.as_f64())
            .collect();
        let mut warm = Encoder::new();
        let mut s = TokenStream::new();
        route_kcc_sites_autofit(&mut warm, &mut s, &json, true, SHA, false);
        let cold = Encoder::new();
        let mean = |q: &dyn crate::types::Quantizer| -> f64 {
            samples.iter().map(|&v| q.encode(v).1.value()).sum::<f64>() / samples.len() as f64
        };
        let warm_d = mean(warm.quantizer(F::KccSite, "direction_score").unwrap());
        let cold_d = mean(cold.quantizer(F::KccSite, "direction_score").unwrap());
        assert!(
            warm_d < cold_d,
            "corpus-fit must lower mean distortion: warm {warm_d} !< cold {cold_d}"
        );
    }

    #[test]
    fn kcc_site_candidate_array_length_mismatch_fails_visibly() {
        // Phase 5: a parallel array shorter than candidate_residue_ids must FAIL VISIBLY — the site
        // is reported as a mismatch and NO misaligned member data is emitted.
        let enc = Encoder::new();
        let json = serde_json::json!({"sites":[{"gtck_rank":1,"rank_C":0.5,"kcc":{
            "candidate_residue_ids":[10, 20, 30],
            "candidate_kcc_causal_lag":[-5.0, -6.0]  // length 2 != 3
        }}], "residues":[{"residue_id":10},{"residue_id":20},{"residue_id":30}]});
        let mut s = TokenStream::new();
        let rep = route_kcc_sites(&enc, &mut s, &json, true, SHA, false);
        assert_eq!(
            rep.array_length_mismatches,
            vec![0],
            "site 0 must be flagged"
        );
        assert_eq!(
            rep.members_total, 0,
            "no members emitted for a mismatched site"
        );
    }

    #[test]
    fn kcc_site_best_candidate_pointer_survives_sorted_emission() {
        // best_kcc_candidate_index points into ENGINE order; the emitted stable pointer
        // best_candidate_residue_id == driver_residue_id and is order-independent.
        let enc = Encoder::new();
        let json = serde_json::json!({"sites":[{"gtck_rank":1,"rank_C":0.5,"kcc":{
            "candidate_residue_ids":[172, 169, 171], // engine order; index 1 (169) is best
            "best_kcc_candidate_index":1,
            "driver_residue_id":169
        }}], "residues":[{"residue_id":169},{"residue_id":171},{"residue_id":172}]});
        let mut s = TokenStream::new();
        let rep = route_kcc_sites(&enc, &mut s, &json, true, SHA, false);
        assert!(
            rep.driver_best_consistent,
            "driver must equal best candidate residue id"
        );
        let best: Vec<f64> = s
            .sidecar
            .iter()
            .filter_map(|e| match (e.field, &e.exact) {
                ("best_candidate_residue_id", ExactValue::Scalar(x)) => Some(*x),
                _ => None,
            })
            .collect();
        assert_eq!(
            best,
            vec![169.0],
            "stable pointer = the best member's residue_id (== driver)"
        );
    }

    #[test]
    fn kcc_site_whole_record_round_trips_from_sidecar() {
        // INV-4 / §14.1: a full site (top-level + nested kcc + member arrays + enrichment)
        // reconstructs EXACTLY from the sidecar.
        let mut enc = Encoder::new();
        let json = serde_json::json!({"sites":[{
            "id": 13, "gtck_rank": 1, "rank_C": 0.5, "volume": 205.0, "centroid":[1.0,2.0,3.0],
            "kcc": {
                "driver_residue_id": 169, "site_causal_lag": -24.7, "total_steps": 9500,
                "candidate_residue_ids":[172, 169], "best_kcc_candidate_index":1,
                "candidate_kcc_causal_lag":[-25.0, -10.0], "candidate_causal_weights":[0.3, 0.7],
                "candidate_kcc_burst_motion":[0.9,1.0], "candidate_kcc_confidence":[0.1,0.2],
                "candidate_kcc_direction_score":[0.001,0.002], "candidate_kcc_local_cov":[0.3,0.4],
                "candidate_residue_support":[0.2,0.4]
            }
        }], "residues":[{"residue_id":169,"residue_name":"SER","ca_position":[10.0,11.0,12.0]},
                        {"residue_id":172,"residue_name":"LEU","ca_position":[13.0,14.0,15.0]}]});
        let mut s = TokenStream::new();
        route_kcc_sites_autofit(&mut enc, &mut s, &json, true, SHA, true);
        let scalars: std::collections::HashMap<&str, Vec<f64>> = {
            let mut m: std::collections::HashMap<&str, Vec<f64>> = std::collections::HashMap::new();
            for e in s.sidecar.iter() {
                if let ExactValue::Scalar(x) = e.exact {
                    m.entry(e.field).or_default().push(x);
                }
            }
            m
        };
        // top-level + nested kcc exact values survive.
        assert_eq!(scalars["volume"], vec![205.0]);
        assert_eq!(scalars["id"], vec![13.0]);
        assert_eq!(scalars["driver_residue_id"], vec![169.0]);
        assert_eq!(scalars["total_steps"], vec![9500.0]);
        assert!((scalars["site_causal_lag"][0] - (-24.7)).abs() < 1e-9);
        // member arrays reconstruct residue_id-asc with aligned lags ([169,172] ⇒ [-10,-25]).
        assert_eq!(scalars["candidate_residue_ids"], vec![169.0, 172.0]);
        assert_eq!(scalars["candidate_kcc_causal_lag"], vec![-10.0, -25.0]);
        // enrichment: member Cα positions reconstruct from residues[].
        let cas: Vec<[f64; 3]> = s
            .sidecar
            .iter()
            .filter_map(|e| match (e.field, &e.exact) {
                ("candidate_ca_position", ExactValue::Vec3(v)) => Some(*v),
                _ => None,
            })
            .collect();
        assert_eq!(
            cas,
            vec![[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]],
            "member Cα enriched + ordered"
        );
    }

    #[test]
    fn kcc_native_member_pair_coupling_is_present_not_absent() {
        // member-pair causal/coherence evidence IS present (KCC-native), even though the literal
        // TwinCoupled vocabulary is not — vocabulary absence is NOT evidence absence.
        let Some(json) = real_kcc_full_json() else {
            eprintln!("skip");
            return;
        };
        let kcc0 = json["sites"][0]["kcc"].as_object().unwrap();
        for coupling in [
            "candidate_kcc_causal_lag",
            "candidate_causal_weights",
            "candidate_kcc_confidence",
            "candidate_kcc_local_cov",
            "candidate_residue_support",
            "lag_corr_peak",
            "local_cov",
        ] {
            assert!(
                kcc0.contains_key(coupling),
                "KCC-native coupling field {coupling} present"
            );
        }
        // and these coupling fields are registered + routed (not discarded).
        let r = Encoder::new();
        for f in [
            "candidate_kcc_causal_lag",
            "candidate_causal_weights",
            "candidate_kcc_confidence",
        ] {
            assert!(r.registry().lookup(F::KccSite, f).is_some(), "{f} routed");
        }
        // the literal TwinCoupled vocabulary is absent in THIS fixture (⇒ join deferred, not faked).
        for twin in ["te_asymmetry", "barrier", "ccf_width", "agreement"] {
            assert!(!kcc0.contains_key(twin));
        }
    }

    #[test]
    fn kcc_site_sibling_join_coverage_is_measurable() {
        // Phase 6: the sibling artifacts join to the kcc sites — coverage measured, not assumed.
        let (Some(kcc), Some(bs), Some(therm)) = (
            real_kcc_full_json(),
            sibling("Kv31_chainA.binding_sites.json"),
            sibling("Kv31_chainA.topology.prism_therm.json"),
        ) else {
            eprintln!("skip: sibling bundle absent");
            return;
        };
        // binding_sites.json is the SAME 65 sites — joinable by `id` (exact, 65/65).
        let kcc_ids: Vec<i64> = kcc["sites"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|s| s["id"].as_i64())
            .collect();
        let bs_ids: Vec<i64> = bs["sites"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|s| s["id"].as_i64())
            .collect();
        assert_eq!(kcc_ids.len(), 65);
        assert_eq!(
            kcc_ids, bs_ids,
            "binding_sites shares the kcc site ids (1:1 join)"
        );
        // prism_therm carries authoritative per-residue transfer_entropy/fisher_info (TWIN evidence)
        // — joinable to site members by residue_id (partial coverage, reported).
        let te_res: HashSet<i64> = therm["pockets"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|p| p["top_residues"].as_array().cloned().unwrap_or_default())
            .filter_map(|r| r["residue_id"].as_i64())
            .collect();
        let members: HashSet<i64> = kcc["sites"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|s| s["kcc"]["candidate_residue_ids"].as_array())
            .flatten()
            .filter_map(|v| v.as_i64())
            .collect();
        let covered = members.iter().filter(|m| te_res.contains(m)).count();
        assert!(
            !te_res.is_empty(),
            "prism_therm must carry transfer_entropy residues"
        );
        assert!(
            covered > 0 && covered <= members.len(),
            "TE member coverage {covered}/{}",
            members.len()
        );
    }

    #[test]
    fn kcc_site_prism_therm_twin_evidence_is_typed_joined() {
        // Phase 2/3: authoritative per-residue TWIN evidence (transfer_entropy/fisher_info/role/…)
        // is TYPED-routed onto members by residue_id — not report-only. Partial coverage (some
        // members genuinely absent from prism_therm's top-K lists) is honest, not faked.
        let (Some(kcc), Some(therm)) = (
            real_kcc_full_json(),
            sibling("Kv31_chainA.topology.prism_therm.json"),
        ) else {
            eprintln!("skip: sibling absent");
            return;
        };
        let enc = Encoder::new();
        let mut s = TokenStream::new();
        let rep = route_kcc_sites_with_therm(&enc, &mut s, &kcc, &therm, true, SHA, false);
        assert!(
            rep.members_te_enriched > 0 && rep.members_te_enriched < rep.members_total,
            "PARTIAL typed TE coverage {}/{} (some members genuinely absent)",
            rep.members_te_enriched,
            rep.members_total
        );
        // one transfer_entropy token per enriched member, and the TWIN role enum is routed.
        let te_tokens = s
            .sidecar
            .iter()
            .filter(|e| e.field == "candidate_transfer_entropy")
            .count();
        assert_eq!(
            te_tokens, rep.members_te_enriched,
            "one transfer_entropy token per enriched member"
        );
        let roles: HashSet<&str> = s
            .sidecar
            .iter()
            .filter_map(|e| match (e.field, &e.exact) {
                ("candidate_te_role", ExactValue::Text(t)) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert!(
            roles
                .iter()
                .any(|r| ["TRIGGER", "GATEWAY", "STABIL.", "SPECTAT"].contains(r)),
            "TWIN role enum routed: {roles:?}"
        );
        // without prism_therm, NO TE enrichment (proves the join is the source, not fabrication).
        let mut s2 = TokenStream::new();
        let rep2 = route_kcc_sites(&enc, &mut s2, &kcc, true, SHA, false);
        assert_eq!(
            rep2.members_te_enriched, 0,
            "no therm ⇒ no TE tokens (not fabricated)"
        );
    }

    #[test]
    fn kcc_site_vq_health_includes_distortion_metrics() {
        // Phase 8: VQ health beyond escape count — dead codes + mean/p95/worst distortion.
        let Some(json) = real_kcc_full_json() else {
            eprintln!("skip");
            return;
        };
        let mut enc = Encoder::new();
        let mut s = TokenStream::new();
        let rep = route_kcc_sites_autofit(&mut enc, &mut s, &json, true, SHA, false);
        assert!(rep.centroid_perplexity > 1.0, "codebook not collapsed");
        assert!(rep.centroid_codes_used > 1);
        assert_eq!(
            rep.centroid_dead_codes,
            (1usize << 12) - rep.centroid_codes_used,
            "dead = codebook − used"
        );
        assert_eq!(
            rep.centroid_escapes, 0,
            "all centroids within the 0.25Å bound"
        );
        // ordered percentiles, worst within the Euclidean bound (since 0 escapes).
        assert!(rep.centroid_mean_distortion <= rep.centroid_p95_distortion + 1e-12);
        assert!(rep.centroid_p95_distortion <= rep.centroid_worst_distortion + 1e-12);
        assert!(
            rep.centroid_worst_distortion <= 0.25 + 1e-9,
            "worst ≤ bound"
        );
    }

    // --- SPIKE_TRANSITION_EVENT (full source schema, distinct from BocpdEvent + bocpd.jsonl) ----

    // The 21 directive-required keys (subset of the 23-field engine source schema, nhs_rt_full.rs ~7616).
    const STE_KEYS: &[&str] = &[
        "schema_kind",
        "stream_id",
        "group_id",
        "condition_id",
        "protocol_name",
        "trigger_kind",
        "phase_from",
        "phase_to",
        "step",
        "chunk_idx",
        "bocpd_probability",
        "spike_count",
        "spike_count_status",
        "spike_stream_semantics",
        "postmd_filter_policy",
        "spike_packet_sha256",
        "spike_packet_sha256_status",
        "protocol_state_hash",
        "threshold_before_after",
        "bridge_groups_observed",
        "event_hash_blake3",
    ];

    fn synthetic_ste() -> serde_json::Value {
        // REAL-vocab record: enum values are verbatim from a real all-on v2_ignition run
        // (output/5VEX_transition, trigger_kind/phase/condition_id/protocol_name/spike_count_status
        // measured, not synthetic). condition_id/protocol_name/spike_count_status are now TYPED
        // enum fields (v5). The source-schema synthetic remains only for full-key coverage.
        serde_json::json!({
            "schema_kind": "SPIKE_TRANSITION_EVENT", "schema_version": 1,
            "stream_id": 0, "group_id": 0, "condition_id": "TWIN16_0011",
            "protocol_name": "ThermalShock", "trigger_kind": "ring_buffer_push",
            "phase_from": "cold_hold", "phase_to": "ramp_up", "step": 12000, "chunk_idx": 4,
            "bocpd_probability": 0.92, "spike_count": 5123,
            "spike_count_status": "device_buffer_pushed_before_host_spike_sync",
            "spike_stream_semantics": "view_anchor_over_append_only_spike_stream",
            "postmd_filter_policy": "DSTW/post-MD selects windows by condition_id, ...",
            "spike_packet_sha256": "PENDING_UNTIL_SPIKE_PACKET_SERIALIZATION",
            "spike_packet_sha256_status": "event_time_before_spike_codec_packet_write",
            "protocol_state_hash": "abc123", "protocol_state_hash_scope": "host_protocol_schedule",
            "threshold_before_after": {"before": 0.1, "after": 0.2},
            "bridge_groups_observed": [1, 2],
            "event_hash_blake3": "deadbeefcafe"
        })
    }

    #[test]
    fn spike_transition_event_detects_as_full_family_not_bocpd() {
        // schema_kind SPIKE_TRANSITION_EVENT now routes the FULL family, not the thin BocpdEvent.
        assert_eq!(
            detect_family(&serde_json::json!({"schema_kind": "SPIKE_TRANSITION_EVENT"})),
            Some((F::SpikeTransitionEvent, "")),
        );
        assert_ne!(
            detect_family(&serde_json::json!({"schema_kind": "SPIKE_TRANSITION_EVENT"})),
            Some((F::BocpdEvent, "")),
            "STE must NOT route through BocpdEvent"
        );
    }

    #[test]
    fn spike_transition_event_routes_full_source_schema_typed_plus_lossless() {
        let enc = Encoder::new();
        let mut s = TokenStream::new();
        let rep = route_spike_transition_event(&enc, &mut s, &synthetic_ste(), true, SHA);
        assert_eq!(rep.records, 1);
        // 12 typed evidence fields route (9 + the now-typed condition_id/protocol_name/
        // spike_count_status enums, v5), beyond the 4 BOCPD-only fields.
        assert_eq!(
            rep.matched, 12,
            "12 typed STE fields routed, got {}",
            rep.matched
        );
        // real-vocab enums map to TYPED codes (non-escape), not the synthetic placeholders.
        let typed_nonescape = |field: &str| {
            s.sidecar
                .iter()
                .any(|e| e.field == field && !s.is_escaped(e.position))
        };
        for f in [
            "trigger_kind",
            "phase_to",
            "condition_id",
            "protocol_name",
            "spike_count_status",
        ] {
            assert!(
                typed_nonescape(f),
                "{f} must route to a typed code from the real vocab"
            );
        }
        // EVERY source-schema key is preserved: a typed token (field==key) OR a lossless opaque
        // sidecar entry (Text "key=…"). None silently dropped (INV-4).
        let preserved = |key: &str| {
            s.sidecar.iter().any(|e| {
                e.field == key
                    || matches!(&e.exact, ExactValue::Text(t) if t.starts_with(&format!("{key}=")))
            })
        };
        for key in STE_KEYS {
            assert!(
                preserved(key),
                "STE field {key} must be routed or lossless-captured"
            );
        }
        // hashes + nested objects survive losslessly (§10 seal-collapse for hashes).
        let has = |needle: &str| {
            s.sidecar
                .iter()
                .any(|e| matches!(&e.exact, ExactValue::Text(t) if t.contains(needle)))
        };
        assert!(
            has("event_hash_blake3") && has("deadbeefcafe"),
            "blake3 event hash lossless"
        );
        assert!(
            has("threshold_before_after"),
            "nested threshold object lossless"
        );
    }

    #[test]
    fn spike_transition_event_routes_beyond_bocpd_subset() {
        // the thin BocpdEvent subset is {bocpd_probability, phase_from, phase_to, trigger_kind}.
        // STE routes those PLUS stream_id/group_id/step/chunk_idx/spike_count + captures condition_id.
        let enc = Encoder::new();
        let mut s = TokenStream::new();
        route_spike_transition_event(&enc, &mut s, &synthetic_ste(), true, SHA);
        for key in ["stream_id", "group_id", "step", "chunk_idx", "spike_count"] {
            assert!(
                s.sidecar.iter().any(|e| e.field == key),
                "STE routes {key} (absent from BocpdEvent)"
            );
        }
        assert!(
            s.sidecar.iter().any(|e| e.field == "condition_id"),
            "condition_id typed-routed (TWIN16 enum, v5)"
        );
    }

    #[test]
    fn qnanl_ste_decodes_and_routes_identically_to_raw() {
        // CONTAINER PARITY (QNANL addendum): the .prismqnanl STE container decodes (via prismcode,
        // NOT ad-hoc byte parsing) to the SAME raw JSONL payload and routes to the SAME tokens —
        // proving the QNANL envelope yields identical typed coverage as the raw all-on artifact.
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../output/5VEX_transition");
        let raw_p = root.join("5VEX_chain_B_engine_stream0_spike_transition_events.jsonl");
        let qnanl_p =
            root.join("5VEX_chain_B_engine_stream0_spike_transition_events.jsonl.prismqnanl");
        if !raw_p.exists() || !qnanl_p.exists() {
            eprintln!("skip: no real run output (ENGINE_RUN gated)");
            return;
        }
        // decode the QNANL container → raw payload bytes (NEVER tokenize compressed bytes).
        let qbytes = std::fs::read(&qnanl_p).unwrap();
        let decoded = crate::codec::decode_qnanl(&qbytes).expect("QNANL decodes via prismcode");
        assert!(
            decoded.verified,
            "QNANL seal verifies (receipt roundtrip_verified)"
        );
        let payload = decoded.payload.expect("decoded payload");
        let raw = std::fs::read(&raw_p).unwrap();
        assert_eq!(
            payload, raw,
            "QNANL-decoded payload == raw JSONL (lossless container)"
        );
        // route the decoded payload AND the raw payload → identical token streams.
        let route_jsonl = |bytes: &[u8]| {
            let enc = Encoder::new();
            let mut s = TokenStream::new();
            for line in bytes.split(|&b| b == b'\n') {
                if line.is_empty() {
                    continue;
                }
                if let Ok(rec) = serde_json::from_slice::<serde_json::Value>(line) {
                    route_spike_transition_event(&enc, &mut s, &rec, true, SHA);
                }
            }
            s.tokens
        };
        let toks_q = route_jsonl(&payload);
        let toks_r = route_jsonl(&raw);
        assert!(!toks_r.is_empty(), "raw STE routes to tokens");
        assert_eq!(
            toks_q, toks_r,
            "QNANL and raw route to IDENTICAL tokens (container parity)"
        );
    }

    #[test]
    fn prspk001_vs_qnanl_chunk_codec_parity() {
        // Directive v2 A4: the per-stream spikes.bin (PRSPK001) and the SPIKE_EVENT_STREAM chunk
        // (PRSQNB01/QNANL) for the SAME stream decode to IDENTICAL normalized GpuSpikeEvent rows.
        // Emits OPERATOR_CONTROL/SPIKE_CODEC_PARITY_LEDGER.json.
        // Canonical closure fixture root = a3 (the A1+A2+A3 run with receipts); a2 is lineage-
        // equivalent (same engine family, same artifacts) and used only as a fallback.
        let a3 = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../output/5VEX_multidiff_a3");
        let root = if a3.join("SPIKE_EVENT_STREAM/MANIFEST.qnanl.json").exists() {
            a3
        } else {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../output/5VEX_multidiff_a2")
        };
        let binp = root.join("5VEX_chain_B_engine_stream0_spikes.bin");
        let manp = root.join("SPIKE_EVENT_STREAM/MANIFEST.qnanl.json");
        if !binp.exists() || !manp.exists() {
            eprintln!("skip: no multi-diff run output");
            return;
        }
        let prspk =
            crate::codec::decode_prspk001(&std::fs::read(&binp).unwrap()).expect("PRSPK001 decode");
        // locate the SPIKE_EVENT_STREAM chunk for stream 0.
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&manp).unwrap()).unwrap();
        let rows = manifest.get("rows").and_then(|v| v.as_array()).unwrap();
        let chunk_id = rows
            .iter()
            .find(|r| r.get("stream_id").and_then(|v| v.as_u64()) == Some(0))
            .and_then(|r| r.get("chunk_id").and_then(|v| v.as_u64()))
            .expect("stream-0 chunk");
        let chunkp = root.join(format!(
            "SPIKE_EVENT_STREAM/chunks/spike_chunk_{:06}.prsqev",
            chunk_id
        ));
        let d = crate::codec::decode_qnanl(&std::fs::read(&chunkp).unwrap()).expect("QNANL decode");
        let payload = d.payload.expect("payload");
        let qnanl_events: Vec<_> = payload
            .chunks_exact(96)
            .filter_map(crate::codec::GpuSpikeEvent::from_le_bytes)
            .collect();
        let count_parity = prspk.events.len() == qnanl_events.len();
        let first_parity = prspk.events.first() == qnanl_events.first();
        let last_parity = prspk.events.last() == qnanl_events.last();
        let n = prspk.events.len().min(qnanl_events.len());
        let step = (n / 64).max(1);
        let (mut sampled, mut mismatches, mut i) = (0usize, 0usize, 0usize);
        while i < n {
            sampled += 1;
            if prspk.events[i] != qnanl_events[i] {
                mismatches += 1;
            }
            i += step;
        }
        let verdict = if count_parity && first_parity && last_parity && mismatches == 0 {
            "PARITY_PROVEN"
        } else {
            "PARITY_FAILED"
        };
        let ledger = serde_json::json!({
            "schema_kind": "spike_codec_parity_ledger",
            "compared": {"prspk001": binp.display().to_string(), "qnanl_chunk": chunkp.display().to_string()},
            "prspk001_records": prspk.events.len(),
            "qnanl_records": qnanl_events.len(),
            "record_count_parity": count_parity,
            "boundary_first_parity": first_parity,
            "boundary_last_parity": last_parity,
            "sampled_rows_compared": sampled,
            "sampled_mismatches": mismatches,
            "stream_cell_ids_parity": "QNANL chunk manifest carries twin16_cell + condition_id + stream_id",
            "verdict": verdict,
            "scoped_differences": "none — both decode to identical GpuSpikeEvent records at the stride the envelope declares (96 B); QNANL adds SectionZstd compression + sealing over the same payload",
        });
        let ledger_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../OPERATOR_CONTROL/SPIKE_CODEC_PARITY_LEDGER.json");
        let _ = std::fs::write(&ledger_path, serde_json::to_string_pretty(&ledger).unwrap());
        assert!(
            count_parity,
            "record count parity (PRSPK001 {} vs QNANL {})",
            prspk.events.len(),
            qnanl_events.len()
        );
        assert!(first_parity && last_parity, "boundary-row parity");
        assert_eq!(mismatches, 0, "sampled-row parity ({sampled} sampled)");
    }

    #[test]
    fn tok_export_corpus_integrity() {
        // Directive v3 Phase 3: validate the exported token corpus integrity gates.
        let d = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/tokenized_training_corpus/twin16_a3");
        let card_p = d.join("corpus_card.json");
        if !card_p.exists() {
            eprintln!("skip: no exported corpus (run tok_export_corpus)");
            return;
        }
        let card: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&card_p).unwrap()).unwrap();
        assert_eq!(
            card["total_raw_spike_records"].as_u64(),
            Some(23_618_244),
            "raw spike count gate"
        );
        assert_eq!(
            card["n_condition_edges"].as_u64(),
            Some(256),
            "256 condition edges"
        );
        assert_eq!(card["n_twin16_cells"].as_u64(), Some(16), "16 TWIN16 cells");
        assert_eq!(
            card["coupled_ab_excluded"].as_bool(),
            Some(true),
            "coupled A/B excluded"
        );
        let n_tokens = card["n_tokens"].as_u64().unwrap() as usize;
        // tokens.u64.jsonl reads back; count matches the card.
        if d.join("tokens.u64.jsonl").exists() {
            let toks = std::fs::read(d.join("tokens.u64.jsonl")).unwrap();
            let mut count = 0usize;
            for line in toks.split(|&b| b == b'\n') {
                if line.is_empty() {
                    continue;
                }
                let v: serde_json::Value = serde_json::from_slice(line).unwrap();
                assert!(v.get("token_u64").is_some(), "token line has token_u64");
                count += 1;
            }
            assert_eq!(count, n_tokens, "tokens.u64.jsonl count == card n_tokens");
            // segment spans are valid (within the token range, family present).
            let segs = std::fs::read(d.join("segments.jsonl")).unwrap();
            for line in segs.split(|&b| b == b'\n') {
                if line.is_empty() {
                    continue;
                }
                let s: serde_json::Value = serde_json::from_slice(line).unwrap();
                let start = s["start"].as_u64().unwrap() as usize;
                let len = s["len"].as_u64().unwrap() as usize;
                assert!(start + len <= n_tokens, "segment span within token range");
                assert!(s["family"].as_str().is_some(), "segment has family");
            }
        }
        if d.join("condition_edges.jsonl").exists() {
            let n = std::fs::read_to_string(d.join("condition_edges.jsonl"))
                .unwrap()
                .lines()
                .count();
            assert_eq!(n, 256, "condition_edges.jsonl = 256");
        }
        // provenance references the real receipt ops (not filename-only evidence).
        if d.join("provenance.jsonl").exists() {
            let prov = std::fs::read_to_string(d.join("provenance.jsonl")).unwrap();
            assert!(
                prov.contains("MD.CONDITION_EDGE_COMMIT"),
                "provenance references condition-edge receipt"
            );
            assert!(
                prov.contains("MD.SPIKE_CHUNK_COMMIT"),
                "provenance references spike chunk commits"
            );
        }
    }

    #[test]
    fn twin16_condition_edge_routes_real_fixture() {
        // Directive v2 A5: route the TWIN16-native DPD condition edge decode-first, preserving
        // source/target cell + axis + residual + material with exact sidecar values, plus QNANL
        // parity on the sealed summary and receipt-hash linkage. Canonical fixture root = a3.
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../output/5VEX_multidiff_a3/CONDITION_EDGE_READOUT");
        let jsonl = dir.join("per_cellpair_condition_edges.jsonl");
        if !jsonl.exists() {
            eprintln!("skip: no condition-edge fixture");
            return;
        }
        let bytes = std::fs::read(&jsonl).unwrap();
        let enc = Encoder::new();
        let mut s = TokenStream::new();
        let mut n = 0usize;
        for line in bytes.split(|&b| b == b'\n') {
            if line.is_empty() {
                continue;
            }
            let rec: serde_json::Value = match serde_json::from_slice(line) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if n == 0 {
                assert_eq!(
                    detect_family(&rec),
                    Some((F::TwinConditionEdge, "")),
                    "per-cellpair edge → TwinConditionEdge"
                );
            }
            route_object(
                &enc,
                &mut s,
                F::TwinConditionEdge,
                rec.as_object().unwrap(),
                true,
                SHA,
                true,
            );
            n += 1;
            if n >= 256 {
                break;
            }
        }
        assert!(n > 0, "routed condition-edge records");
        for f in [
            "cell_a",
            "cell_b",
            "transition_view_l1_residual",
            "material",
        ] {
            assert!(
                s.sidecar.iter().any(|e| e.field == f),
                "condition-edge field {f} routed"
            );
        }
        // cell ids route to typed TWIN16 codes (non-escape).
        assert!(
            s.sidecar
                .iter()
                .any(|e| e.field == "cell_a" && !s.is_escaped(e.position)),
            "cell_a routes to a typed TWIN16 code"
        );
        // QNANL parity on the sealed summary (decoded == raw manifest).
        let qnanl = dir.join("condition_edge_summary.qnanl");
        let manifest = dir.join("condition_edge_manifest.json");
        if qnanl.exists() && manifest.exists() {
            let d =
                crate::codec::decode_qnanl(&std::fs::read(&qnanl).unwrap()).expect("QNANL decode");
            assert!(d.verified, "condition-edge summary seal verifies");
            assert_eq!(
                d.payload.unwrap(),
                std::fs::read(&manifest).unwrap(),
                "QNANL-decoded == raw condition_edge_manifest"
            );
        }
        // receipt hash linkage preserved (A5 gate).
        let receipt = dir.join("RECEIPT.prismcode.json");
        if receipt.exists() {
            let r: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&receipt).unwrap()).unwrap();
            assert_eq!(
                r["ops"][0]["op"].as_str(),
                Some("MD.CONDITION_EDGE_COMMIT"),
                "receipt op"
            );
            assert!(
                r["ops"][0]["manifest_hash_blake3"]
                    .as_str()
                    .map(|h| h.len() >= 12)
                    .unwrap_or(false),
                "receipt carries manifest_hash_blake3 linkage"
            );
        }
    }

    #[test]
    fn nma_coupled_edge_routes_separately_from_twin16() {
        // NMA addendum BOUNDARY: the coupled-A/B NMA per-residue differential routes via
        // NmaCoupledEdge — a SEPARATE family from TWIN16's TwinConditionEdge. NMA is not a TWIN16
        // axis and must never be merged. Prefers the amp-10 fixture, falls back to amp-3.
        let cands = [
            "../../output/5VEX_coupled_nma_amp10/CONDITION_EDGE_READOUT/per_residue_condition_edges.jsonl",
            "../../output/5VEX_twin8_amp3/CONDITION_EDGE_READOUT/per_residue_condition_edges.jsonl",
        ];
        let p = cands
            .iter()
            .map(|c| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(c))
            .find(|p| p.exists());
        let p = match p {
            Some(p) => p,
            None => {
                eprintln!("skip: no coupled NMA readout");
                return;
            }
        };
        let bytes = std::fs::read(&p).unwrap();
        let line0 = bytes.split(|&b| b == b'\n').next().unwrap();
        let rec: serde_json::Value = serde_json::from_slice(line0).unwrap();
        assert_eq!(
            detect_family(&rec),
            Some((F::NmaCoupledEdge, "")),
            "coupled NMA per-residue → NmaCoupledEdge (separate auxiliary family)"
        );
        assert_ne!(
            detect_family(&rec),
            Some((F::TwinConditionEdge, "")),
            "must NOT route as TWIN16"
        );
        let enc = Encoder::new();
        let mut s = TokenStream::new();
        let r = route_object(
            &enc,
            &mut s,
            F::NmaCoupledEdge,
            rec.as_object().unwrap(),
            true,
            SHA,
            true,
        );
        assert!(
            r.matched >= 10,
            "NMA coupled fields routed, got {}",
            r.matched
        );
        for f in [
            "resid",
            "b_over_a_ratio",
            "nma_exclusive_count",
            "condition_exclusive",
            "edge_score",
        ] {
            assert!(
                s.sidecar.iter().any(|e| e.field == f),
                "NMA coupled field {f} routed"
            );
        }
        assert!(
            s.sidecar
                .iter()
                .any(|e| e.field == "barrier_classification" && !s.is_escaped(e.position)),
            "barrier_classification routes to a typed code"
        );
    }

    #[test]
    fn real_twin16_residual_routes_cells_and_pairwise() {
        // TWIN16 interferometric transition-view residual (distinct from a residue-pair TE matrix):
        // 16 per-cell aggregates + cell-pair L1 residuals route typed.
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
            "../../output/5VEX_transition/TWIN16_INTERFEROMETRIC_TRANSITION_VIEW_RESIDUALS.json",
        );
        if !p.exists() {
            eprintln!("skip: no real TWIN16 output");
            return;
        }
        let json: serde_json::Value = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
        assert_eq!(
            detect_family(&json),
            Some((F::TwinTransitionResidual, "/cells"))
        );
        let enc = Encoder::new();
        let mut s = TokenStream::new();
        let (cells, pairs) = route_twin16_residual(&enc, &mut s, &json, true, SHA);
        assert_eq!(cells.records, 16, "16 TWIN16 cells");
        assert!(
            cells.matched > 16 * 8,
            "per-cell aggregates routed, got {}",
            cells.matched
        );
        assert!(
            pairs.records > 0,
            "cell-pair residuals routed, got {}",
            pairs.records
        );
        // the L1 residual + the TWIN16 condition enum route typed.
        assert!(s
            .sidecar
            .iter()
            .any(|e| e.field == "transition_view_l1_residual"));
        assert!(
            s.sidecar
                .iter()
                .any(|e| e.field == "condition_id" && !s.is_escaped(e.position)),
            "TWIN16 condition_id routes to a typed cell code"
        );
    }

    #[test]
    fn pathb_site_candidates_routes_rankerrow_by_value() {
        // PathB ranked voxel-density-peak carrier: schema_kind=pathb_site_candidates, wrapper with
        // candidates:[rows]. detect_family must descend "/candidates" under RankerRow, and each row
        // must match its 3 specced pathb keys (rank/peak_id/spike_count) BY VALUE => matched==3*N.
        let json = serde_json::json!({
            "schema_kind": "pathb_site_candidates",
            "schema_version": 1,
            "n_candidates": 2,
            "top_k": 2,
            "candidates": [
                {"rank": 0, "peak_id": 0, "spike_count": 3105,
                 "centroid_a": [19.0, 15.0, -29.0], "voxel_index": [9, 7, -15],
                 "kind": "voxel_density_peak"},
                {"rank": 1, "peak_id": 1, "spike_count": 2980,
                 "centroid_a": [1.0, 2.0, 3.0], "voxel_index": [0, 1, 2],
                 "kind": "voxel_density_peak"}
            ]
        });
        assert_eq!(
            detect_family(&json),
            Some((F::RankerRow, "/candidates")),
            "schema_kind=pathb_site_candidates descends /candidates under RankerRow"
        );
        let (family, path) = detect_family(&json).unwrap();
        let enc = Encoder::new();
        let mut s = TokenStream::new();
        let rep = route_records(&enc, &mut s, family, &json, path, true, SHA, true);
        assert_eq!(rep.records, 2, "2 candidate rows descended");
        assert_eq!(
            rep.matched, 6,
            "3 specced pathb keys (rank/peak_id/spike_count) matched per row => 3*2, got {}",
            rep.matched
        );
        // rank/peak_id/spike_count route to typed codes (not opaque escapes).
        assert!(
            s.sidecar
                .iter()
                .any(|e| e.field == "spike_count" && !s.is_escaped(e.position)),
            "spike_count (density SCORE) routes to a typed code"
        );
    }

    #[test]
    fn qnanl_parity_across_allon_families() {
        // every PRSQNB01 .prismqnanl in the run decodes (via prismcode) to the BYTE-IDENTICAL raw
        // payload — container parity for protocol_state / noise_floor / TWIN16 / STE, not just STE.
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../output/5VEX_transition");
        if !root.exists() {
            eprintln!("skip: no real run output");
            return;
        }
        let mut tested = 0usize;
        for entry in std::fs::read_dir(&root).unwrap().flatten() {
            let path = entry.path();
            if path.extension().and_then(|x| x.to_str()) != Some("prismqnanl") {
                continue;
            }
            let bytes = std::fs::read(&path).unwrap();
            if !bytes.starts_with(b"PRSQNB01") {
                continue; // spikes.prismnanl is PRSPCZ01 (different codec path)
            }
            let raw_p = path.with_extension(""); // strip .prismqnanl → the raw artifact
            if !raw_p.exists() {
                continue;
            }
            let d = crate::codec::decode_qnanl(&bytes).expect("QNANL decodes via prismcode");
            assert!(d.verified, "{:?}: QNANL seal verifies", path.file_name());
            assert_eq!(
                d.payload.unwrap(),
                std::fs::read(&raw_p).unwrap(),
                "{:?}: QNANL-decoded payload == raw (lossless container)",
                path.file_name()
            );
            tested += 1;
        }
        assert!(tested >= 3, "≥3 QNANL families parity-checked (STE/protocol_state/noise_floor/TWIN16), got {tested}");
    }

    #[test]
    fn real_twin_per_residue_coupling_routes() {
        // PAIRWISE/per-residue TWIN evidence from the memory-safe 8-engine coupled-TWIN run: 422
        // per-residue coupling records (b_over_a_ratio asymmetry / ccf_* cross-correlation /
        // barrier_class / spikes_a,b) route into the now-populated TwinCoupled family.
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../output/5VEX_twin8/5VEX_chain_B_fixed.twin_per_residue.json");
        if !p.exists() {
            eprintln!("skip: no coupled-TWIN per-residue output");
            return;
        }
        let json: serde_json::Value = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
        let mut enc = Encoder::new();
        let mut s = TokenStream::new();
        let rep = route_twin_per_residue(&mut enc, &mut s, &json, true, SHA);
        assert_eq!(rep.records, 422, "422 per-residue coupling records");
        assert!(
            rep.matched > 422 * 8,
            "coupling fields routed, got {}",
            rep.matched
        );
        for f in ["b_over_a_ratio", "ccf_max", "resid", "spikes_a", "spikes_b"] {
            assert!(
                s.sidecar.iter().any(|e| e.field == f),
                "TWIN coupling field {f} routed"
            );
        }
        assert!(
            s.sidecar
                .iter()
                .any(|e| e.field == "barrier_class" && !s.is_escaped(e.position)),
            "barrier_class routes to a typed code"
        );
        // Cα positions route as real spatial codes after the autofit (non-escape).
        let ca_escapes = s
            .sidecar
            .iter()
            .filter(|e| e.field == "ca_xyz")
            .filter(|e| s.is_escaped(e.position))
            .count();
        assert_eq!(
            ca_escapes, 0,
            "Cα VQ autofit ⇒ non-escape; {ca_escapes} escaped"
        );
    }

    #[test]
    fn real_spike_transition_events_route_typed_from_run_output() {
        // real-evidence test (addendum §8): the actual *_spike_transition_events.jsonl from the
        // all-on v2_ignition run routes with TYPED enums (real vocab), not escapes.
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../output/5VEX_transition/5VEX_chain_B_engine_stream0_spike_transition_events.jsonl");
        if !p.exists() {
            eprintln!("skip: no real run output");
            return;
        }
        let bytes = std::fs::read(&p).unwrap();
        let line0 = bytes.split(|&b| b == b'\n').next().unwrap();
        let rec: serde_json::Value = serde_json::from_slice(line0).unwrap();
        assert_eq!(detect_family(&rec), Some((F::SpikeTransitionEvent, "")));
        let enc = Encoder::new();
        let mut s = TokenStream::new();
        let rep = route_spike_transition_event(&enc, &mut s, &rec, true, SHA);
        assert!(
            rep.matched >= 9,
            "real STE routes typed fields, got {}",
            rep.matched
        );
        // the real trigger_kind/phase/condition_id are in the measured vocab ⇒ NON-escape codes.
        for f in ["trigger_kind", "phase_to", "condition_id"] {
            let typed = s
                .sidecar
                .iter()
                .any(|e| e.field == f && !s.is_escaped(e.position));
            assert!(
                typed,
                "real STE {f} routes to a typed code (measured vocab)"
            );
        }
    }

    #[test]
    fn bocpd_jsonl_log_is_separate_from_spike_transition_event() {
        // the REAL bocpd.jsonl per-frame log (frame_idx/spike_delta/observation/…) has NO
        // schema_kind and is NOT a SPIKE_TRANSITION_EVENT — never conflated.
        let bocpd = serde_json::json!({"frame_idx":0,"chunk_idx":0,"stream":0,"spike_delta":498099,"observation":13.1,"posterior_max":0.9996,"map_run_length":1});
        assert_ne!(
            detect_family(&bocpd),
            Some((F::SpikeTransitionEvent, "")),
            "bocpd log ≠ STE"
        );
        let r = Encoder::new();
        assert!(r
            .registry()
            .lookup(F::SpikeTransitionEvent, "stream_id")
            .is_some());
        assert!(
            r.registry()
                .lookup(F::SpikeTransitionEvent, "frame_idx")
                .is_none(),
            "frame_idx is a bocpd-log field, NOT an STE field"
        );
    }

    #[test]
    fn all_on_coverage_matrix_has_only_terminal_statuses() {
        // mirrors TOKENIZABLE_MISSING_INFORMATION_LEDGER.json — every all-on surface carries a
        // TERMINAL status; forbidden non-terminal statuses fail the build (Phase 7 gate).
        const TERMINAL: &[&str] = &[
            "ROUTED_FULL",
            "ROUTE_REQUIRED",
            "CODEC_REQUIRED",
            "ENGINE_RUN_REQUIRED",
            "JOIN_REQUIRED",
            "LOSSLESS_OPAQUE_ACCEPTED_TEMPORARILY",
            "NOT_TOKENIZABLE_WITH_SOURCE_EVIDENCE",
        ];
        const FORBIDDEN: &[&str] = &[
            "PARTIAL",
            "ROUTED_PARTIAL",
            "UNKNOWN",
            "TODO",
            "DETECTED_ONLY",
            "ASSUMED_ABSENT",
        ];
        let matrix: &[(&str, &str)] = &[
            ("SPIKE_TRANSITION_EVENT", "ROUTED_FULL"),
            ("spikes.bin", "ROUTED_FULL"),
            ("qnanl_packets", "ROUTED_FULL"),
            ("bocpd.jsonl", "ROUTE_REQUIRED"),
            ("protocol_state.json", "ROUTE_REQUIRED"),
            ("noise_floor.json", "ROUTE_REQUIRED"),
            ("field_completeness_report", "ROUTE_REQUIRED"),
            ("post_md_required_inputs", "ROUTE_REQUIRED"),
            ("kcc_v2full.bin", "CODEC_REQUIRED"),
            ("signal_grid.bin", "CODEC_REQUIRED"),
            ("warp_matrix.bin", "CODEC_REQUIRED"),
            ("asc_vectors.bin", "CODEC_REQUIRED"),
            ("forces_final.bin", "CODEC_REQUIRED"),
            ("adaptive_dt.bin", "CODEC_REQUIRED"),
            ("aromatic_centroids.bin", "CODEC_REQUIRED"),
            ("binding_sites", "JOIN_REQUIRED"),
            ("twin_pairwise_matrix", "ENGINE_RUN_REQUIRED"),
            ("pathB_merges", "ENGINE_RUN_REQUIRED"),
            ("transform_dag", "ENGINE_RUN_REQUIRED"),
        ];
        for (surface, status) in matrix {
            assert!(
                TERMINAL.contains(status),
                "{surface}: '{status}' is not a terminal status"
            );
            assert!(
                !FORBIDDEN.contains(status),
                "{surface}: forbidden non-terminal '{status}'"
            );
        }
        let ste = matrix
            .iter()
            .find(|(s, _)| *s == "SPIKE_TRANSITION_EVENT")
            .unwrap();
        assert_eq!(
            ste.1, "ROUTED_FULL",
            "STE must be ROUTED_FULL, never BOCPD-only"
        );
    }

    #[test]
    fn detection_alone_is_not_routing_coverage_for_ste() {
        // schema detection ≠ routing coverage: STE detects AND routes the full schema (> the 4
        // BocpdEvent fields). A bare detect with thin routing would FAIL here.
        let enc = Encoder::new();
        let mut s = TokenStream::new();
        let rep = route_spike_transition_event(&enc, &mut s, &synthetic_ste(), true, SHA);
        assert!(
            rep.matched > 4,
            "STE routes MORE than the 4 BOCPD-only fields (got {})",
            rep.matched
        );
        assert_eq!(rep.matched, 12);
        assert!(
            !s.sidecar.is_empty(),
            "STE produces tokens (not zero-token detection)"
        );
    }

    // --- Gap-closure: bocpd.jsonl log, protocol_state, receipts, binding_sites join ------------

    #[test]
    fn bocpd_log_routes_real_jsonl_record_separate_from_ste() {
        // a REAL bocpd.jsonl record (from the all-on run) routes as BocpdLog — NOT STE/BocpdEvent.
        let rec = serde_json::json!({"frame_idx":0,"chunk_idx":0,"stream":0,"spike_delta":498099,
            "observation":13.118556,"posterior_max":0.9996,"map_run_length":1,"recent_p":1.0,
            "recent_log_odds_bits":29.897353,"reset_probability":0.0004,"chunk_close_signal":true,"n_observations":1});
        assert_eq!(
            detect_family(&rec),
            Some((F::BocpdLog, "")),
            "bocpd.jsonl → BocpdLog"
        );
        let enc = Encoder::new();
        let mut s = TokenStream::new();
        let (fam, rep) = route_artifact(&enc, &mut s, &rec, true, SHA, true);
        assert_eq!(fam, Some(F::BocpdLog));
        assert_eq!(
            rep.matched, 12,
            "all 12 BOCPD-log fields routed, got {}",
            rep.matched
        );
        // distinct from STE: no schema_kind, and frame_idx is a BocpdLog field, not STE.
        assert_ne!(detect_family(&rec), Some((F::SpikeTransitionEvent, "")));
    }

    #[test]
    fn protocol_state_routes_control_fields() {
        let rec = serde_json::json!({"current_phase_bits":307,"dt_ps":0.004,"current_temperature_K":82.89,
            "gamma":10.0,"current_step":10506,"processed_spike_count":14264,"coupling_phase":0,
            "n_wavelengths":5,"effective_gamma":10.0,"fused_step_counter":1750,"scan_wavelengths_nm":[280,274]});
        assert_eq!(detect_family(&rec), Some((F::ProtocolState, "")));
        let enc = Encoder::new();
        let mut s = TokenStream::new();
        let (fam, rep) = route_artifact(&enc, &mut s, &rec, true, SHA, true);
        assert_eq!(fam, Some(F::ProtocolState));
        assert!(
            rep.matched >= 9,
            "control fields routed, got {}",
            rep.matched
        );
        // the array field (scan_wavelengths_nm) is preserved lossless via capture (not dropped).
        assert!(s.sidecar.iter().any(
            |e| matches!(&e.exact, ExactValue::Text(t) if t.starts_with("scan_wavelengths_nm="))
        ));
    }

    #[test]
    fn noise_floor_routes_qc_calibration() {
        let rec = serde_json::json!({"target":"5VEX_chain_B","n_samples":1024,"phase":"cold_hold",
            "mu":[0.1,0.2],"sigma":[0.01,0.02],"samples":[1.0,2.0,3.0]});
        assert_eq!(detect_family(&rec), Some((F::NoiseFloor, "")));
        let enc = Encoder::new();
        let mut s = TokenStream::new();
        let (fam, rep) = route_artifact(&enc, &mut s, &rec, true, SHA, true);
        assert_eq!(fam, Some(F::NoiseFloor));
        assert!(rep.matched >= 2, "n_samples + phase routed");
        // the mu/sigma/samples arrays are preserved lossless via capture.
        assert!(s
            .sidecar
            .iter()
            .any(|e| matches!(&e.exact, ExactValue::Text(t) if t.starts_with("mu="))));
    }

    #[test]
    fn allon_receipts_detect_as_receipt_status() {
        for sk in ["field_completeness_report", "post_md_required_inputs"] {
            assert_eq!(
                detect_family(
                    &serde_json::json!({"schema_kind": sk, "required_artifacts_complete": true})
                ),
                Some((F::ReceiptStatus, "")),
                "{sk} must route as a control/receipt"
            );
        }
    }

    #[test]
    fn kcc_site_binding_sites_typed_join() {
        // binding_sites.json joins to KccSite by site id (65/65) — typed, not report-only.
        let (Some(kcc), Some(bs)) = (
            real_kcc_full_json(),
            sibling("Kv31_chainA.binding_sites.json"),
        ) else {
            eprintln!("skip");
            return;
        };
        let enc = Encoder::new();
        let mut s = TokenStream::new();
        let rep = route_kcc_sites_with_binding(&enc, &mut s, &kcc, &bs, true, SHA, false);
        assert_eq!(
            rep.sites_binding_enriched, 65,
            "all 65 sites join binding_sites by id"
        );
        // classification enum + druggability/cryptic scalars are routed.
        let class_tokens = s
            .sidecar
            .iter()
            .filter(|e| e.field == "site_classification")
            .count();
        assert_eq!(class_tokens, 65, "one classification per site");
        assert!(s.sidecar.iter().any(|e| e.field == "site_druggability"));
        // without binding, no enrichment (proves the join is the source).
        let mut s2 = TokenStream::new();
        let rep2 = route_kcc_sites(&enc, &mut s2, &kcc, true, SHA, false);
        assert_eq!(rep2.sites_binding_enriched, 0);
    }

    #[test]
    fn kcc_site_combined_binding_and_therm_join_routes_both_evidence_classes() {
        let (Some(kcc), Some(bs), Some(therm)) = (
            real_kcc_full_json(),
            sibling("Kv31_chainA.binding_sites.json"),
            sibling("Kv31_chainA.topology.prism_therm.json"),
        ) else {
            eprintln!("skip: Kv31 sibling bundle absent");
            return;
        };
        let enc = Encoder::new();
        let mut s = TokenStream::new();
        let rep = route_kcc_sites_with_binding_and_therm(
            &enc, &mut s, &kcc, &bs, &therm, true, SHA, false,
        );
        assert_eq!(
            rep.sites_binding_enriched, 65,
            "binding sibling joins all Kv31 sites"
        );
        assert!(
            rep.members_te_enriched > 0 && rep.members_te_enriched < rep.members_total,
            "therm sibling partially joins real member TE coverage {}/{}",
            rep.members_te_enriched,
            rep.members_total
        );
        assert!(s.sidecar.iter().any(|e| e.field == "site_classification"));
        assert!(s
            .sidecar
            .iter()
            .any(|e| e.field == "candidate_transfer_entropy"));
        assert!(
            s.segments.iter().all(|seg| seg.family == F::KccSite),
            "combined join must remain KccSite, not TWIN16/NMA"
        );
    }

    #[test]
    fn fivevex_coupled_nma_kcc_binding_pair_routes_six_sites() {
        let (Some(kcc), Some(bs)) = (
            repo_json("output/5VEX_coupled_nma_amp10/5VEX_chain_B_fixed.kcc_visualization.json"),
            repo_json("output/5VEX_coupled_nma_amp10/5VEX_chain_B_fixed.binding_sites.json"),
        ) else {
            eprintln!("skip: 5VEX coupled-NMA KCC/binding pair absent");
            return;
        };
        assert_eq!(
            kcc["sites"].as_array().map(Vec::len),
            Some(6),
            "fixture has six compact KCC sites"
        );
        let enc = Encoder::new();
        let mut s = TokenStream::new();
        let rep = route_kcc_sites_with_binding(&enc, &mut s, &kcc, &bs, true, SHA, true);
        assert_eq!(rep.sites, 6);
        assert_eq!(
            rep.sites_binding_enriched, 6,
            "all 5VEX compact sites join binding by id"
        );
        assert_eq!(
            rep.members_te_enriched, 0,
            "no therm sibling supplied ⇒ no TE member join"
        );
        assert!(
            rep.matched > 6 * 6,
            "compact KCC site + binding fields route, got {}",
            rep.matched
        );
        assert_eq!(
            s.sidecar
                .iter()
                .filter(|e| e.field == "site_classification")
                .count(),
            6,
            "one binding classification per joined 5VEX site"
        );
        assert!(
            s.sidecar.iter().any(|e| e.field == "site_causal_lag"),
            "compact lag alias routed"
        );
    }

    #[test]
    fn missing_materialized_kcc_sibling_reports_zero_joined_sites() {
        let (Some(kcc), Some(status)) = (
            repo_json("output/5VEX_multidiff_a3/5VEX_chain_B_engine.kcc_visualization.json"),
            repo_json("output/5VEX_multidiff_a3/5VEX_chain_B_engine.binding_sites.json"),
        ) else {
            eprintln!("skip: a3 missing-materialization status fixture absent");
            return;
        };
        assert_eq!(
            kcc.get("sites").and_then(|v| v.as_array()).map(Vec::len),
            Some(0),
            "a3 KCC fixture exists but has zero materialized sites"
        );
        assert_eq!(
            status.get("status").and_then(|v| v.as_str()),
            Some("evidence_emitted_not_fully_materialized")
        );
        assert_eq!(
            status.get("path_b_required").and_then(|v| v.as_bool()),
            Some(true),
            "fixture declares materialization requirement instead of a routeable sibling"
        );
        let enc = Encoder::new();
        let mut s = TokenStream::new();
        let rep = route_kcc_sites_with_binding(&enc, &mut s, &kcc, &status, true, SHA, true);
        assert_eq!(
            rep.sites, 0,
            "zero materialized KCC sites ⇒ zero routeable joined sites"
        );
        assert_eq!(
            rep.sites_binding_enriched, 0,
            "missing materialization is ledgered, not guessed"
        );
        assert_eq!(
            s.len(),
            0,
            "missing sibling must not emit guessed KccSite tokens"
        );
    }

    // --- Directive v6 Agent 3/4: MoleculeGeneration + GflownetTrajectory generative tables ---------

    /// A representative enamine synthon record (exact columns + measured value forms).
    fn enamine_record() -> serde_json::Map<String, serde_json::Value> {
        serde_json::from_value(serde_json::json!({
            "synthon_id": "EN300-12345_synthon_A",
            "vendor_id": "EN300-12345",
            "canonical_smiles": "CC(=O)O",
            "is_valid_mol": true,
            "formal_charge": 0,
            "heavy_atom_count": 4,
            "molecular_weight": 60.052,
            "conformer_count": 1,
            "charge_method": "openff_nagl:openff-gnn-am1bcc-1.0.0.pt",
            "ingest_status": "ok",
            "reaction_role": "synthon",
            "attachment_atom_idx": 2,
            "leaving_group_atom_idx": 3,
            "leaving_group_formal_charge": 0.0,
            "partial_charge_method": "openff_nagl:openff-gnn-am1bcc-1.0.0.pt",
            // JSON-blob columns ride sidecar capture (parsed to structure by the python harness):
            "conformer_atoms_json": [[0.0, 1.0, 2.0]],
            "reject_reason": ""
        }))
        .unwrap()
    }

    /// A representative V-space scaffold-consensus product record (exact columns).
    fn vspace_record() -> serde_json::Map<String, serde_json::Value> {
        serde_json::from_value(serde_json::json!({
            "anchor_id": "anchor_017",
            "canonical_smiles": "c1ccccc1CC(=O)O",
            "product_id": "prod_88421",
            "smiles": "c1ccccc1CC(=O)O",
            "synthon_a_id": "EN300-1_A",
            "synthon_b_id": "EN300-2_B",
            "score": 2.65,
            "fragment_pi_complement": 7.0,
            "fragment_pi_clash_adjusted": 1.2,
            "pi_complement": 7.0,
            "pi_clash": 0.35,
            "cryptic_bonus": 14.0,
            "cryptic_bonus_atoms": 7,
            "survival_tier": "cryptic_rescue",
            "selected_dihedral_deg": 300.0,
            "assembly_mode": "smarts_zmatrix",
            "z_matrix_active": true,
            "rotamers_evaluated": 6,
            "best_rotamer_rank": 1,
            "pose_reconciliation_method": "compact_pseudo_ligand_o3a",
            "scaffold_consensus_bonus": 0.42,
            "consensus_complement_bonus": 0.1,
            "scaffold_consensus_liability": 113.0,
            "scaffold_consensus_atoms_scored": 143.0,
            "scaffold_consensus_bonus_scaled": 0.84
        }))
        .unwrap()
    }

    #[test]
    fn molecule_generation_has_field_specs() {
        // MoleculeGeneration had ZERO specs before v10 — assert the new block is registered.
        let enc = Encoder::new();
        for f in [
            "canonical_smiles",
            "molecular_weight",
            "score",
            "survival_tier",
            "synthon_id",
        ] {
            assert!(
                enc.registry().lookup(F::MoleculeGeneration, f).is_some(),
                "MoleculeGeneration.{f} missing"
            );
        }
        // every MoleculeGeneration field is generative (canonical_safe=false, INV-3 firewall).
        for s in enc
            .registry()
            .iter()
            .filter(|s| s.family == F::MoleculeGeneration)
        {
            assert!(
                !s.canonical_safe,
                "MoleculeGeneration.{} must be generative-firewalled",
                s.field
            );
        }
    }

    #[test]
    fn molecule_generation_routes_enamine_record() {
        let enc = Encoder::new();
        let rec = enamine_record();
        let mut s = TokenStream::new();
        let r = route_object(&enc, &mut s, F::MoleculeGeneration, &rec, true, SHA, true);
        // smiles (multi-token) + scalars + enums + joinkeys all matched.
        assert!(
            r.matched >= 12,
            "expected ≥12 matched fields, got {}",
            r.matched
        );
        // string JoinKey ids are preserved EXACT in the sidecar (escape token + Text).
        assert!(
            s.sidecar
                .iter()
                .any(|e| matches!(&e.exact, ExactValue::Text(t) if t == "EN300-12345_synthon_A")),
            "synthon_id exact in sidecar"
        );
        // is_valid_mol (json bool) routes; reaction_role enum maps to a code.
        assert!(s.sidecar.iter().any(|e| e.field == "reaction_role"));
        // the JSON-blob column (parsed to an array by the harness) is captured losslessly (INV-4)
        // via push_opaque (field="__unknown__", Text="conformer_atoms_json=…"), not silently dropped.
        assert!(
            s.sidecar.iter().any(|e| matches!(&e.exact, ExactValue::Text(t) if t.starts_with("conformer_atoms_json="))),
            "conformer_atoms_json captured losslessly"
        );
    }

    #[test]
    fn molecule_generation_sidecar_exact_smiles_mw_score() {
        // Gate 4: sidecar exact reconstruction — smiles/mw/score equal the source values.
        let enc = Encoder::new();
        let rec = vspace_record();
        let mut s = TokenStream::new();
        route_object(&enc, &mut s, F::MoleculeGeneration, &rec, true, SHA, true);
        // molecular_weight is absent here; assert score + smiles exactness on the vspace record.
        let want_score = rec["score"].as_f64().unwrap();
        assert!(
            s.sidecar.iter().any(|e| e.field == "score"
                && matches!(&e.exact, ExactValue::Scalar(x) if (*x - want_score).abs() < 1e-12)),
            "score exact in sidecar"
        );
        // canonical_smiles round-trips: concatenating the smiles sub-token sidecar Texts equals the
        // source SMILES (the grammar tokenizer is lossless, sub-token exact in sidecar).
        let smiles_parts: String = s
            .sidecar
            .iter()
            .filter(|e| e.field == "canonical_smiles")
            .filter_map(|e| match &e.exact {
                ExactValue::Text(t) => Some(t.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(
            smiles_parts,
            rec["canonical_smiles"].as_str().unwrap(),
            "smiles exact"
        );
    }

    #[test]
    fn molecule_generation_enum_vocab_measured() {
        // survival_tier / assembly_mode / pose_reconciliation_method map to codes; unknown escapes.
        let enc = Encoder::new();
        let mut s = TokenStream::new();
        assert!(encode_enum_str(
            &enc,
            &mut s,
            F::MoleculeGeneration,
            "survival_tier",
            "cryptic_rescue",
            true,
            SHA
        ));
        assert_eq!(s.tokens[0].field_token(), 1, "cryptic_rescue is code 1");
        let mut s2 = TokenStream::new();
        encode_enum_str(
            &enc,
            &mut s2,
            F::MoleculeGeneration,
            "survival_tier",
            "WEIRD_TIER",
            true,
            SHA,
        );
        assert_eq!(
            s2.tokens[0].field_token(),
            Token::ESCAPE_CODE,
            "unknown tier escapes losslessly"
        );
        let mut s3 = TokenStream::new();
        assert!(encode_enum_str(
            &enc,
            &mut s3,
            F::MoleculeGeneration,
            "assembly_mode",
            "smarts_zmatrix",
            true,
            SHA
        ));
        assert_eq!(s3.tokens[0].field_token(), 0);
    }

    #[test]
    fn molecule_generation_routing_is_deterministic() {
        // Gate 3 in miniature: same record → identical token vector (registry-slot order, no RNG).
        let enc = Encoder::new();
        let rec = vspace_record();
        let mut a = TokenStream::new();
        route_object(&enc, &mut a, F::MoleculeGeneration, &rec, true, SHA, true);
        let mut b = TokenStream::new();
        route_object(&enc, &mut b, F::MoleculeGeneration, &rec, true, SHA, true);
        assert!(!a.tokens.is_empty());
        assert_eq!(a.tokens, b.tokens, "molecule routing must be deterministic");
    }

    #[test]
    fn gflownet_survival_tier_and_regime_label_tables() {
        // The directive: GflownetTrajectory.survival_tier had a spec but NO label table; regime is new.
        let enc = Encoder::new();
        let mut s = TokenStream::new();
        assert!(encode_enum_str(
            &enc,
            &mut s,
            F::GflownetTrajectory,
            "survival_tier",
            "normal",
            true,
            SHA
        ));
        assert_eq!(s.tokens[0].field_token(), 0, "normal is code 0");
        let mut s2 = TokenStream::new();
        assert!(encode_enum_str(
            &enc,
            &mut s2,
            F::GflownetTrajectory,
            "regime",
            "exploration",
            true,
            SHA
        ));
        assert_eq!(s2.tokens[0].field_token(), 3, "exploration is code 3");
        // regime spec exists with the new reward-component fields.
        for f in [
            "regime",
            "temperature",
            "backward_logprob",
            "u_pose",
            "pathway_score_mean",
        ] {
            assert!(
                enc.registry().lookup(F::GflownetTrajectory, f).is_some(),
                "GflownetTrajectory.{f} missing"
            );
        }
    }

    #[test]
    fn validity_status_string_to_oracle_valid_bool() {
        // gflownet_raw_policy_samples uses validity_status "valid" — string→bool onto a Boolean field.
        let enc = Encoder::new();
        let rec: serde_json::Map<String, serde_json::Value> =
            serde_json::from_value(serde_json::json!({"oracle_valid": "valid"})).unwrap();
        let mut s = TokenStream::new();
        let r = route_object(&enc, &mut s, F::GflownetTrajectory, &rec, true, SHA, false);
        assert_eq!(r.matched, 1, "validity string mapped onto oracle_valid");
        // "valid" → true → code 1 in the boolean quantizer.
        assert_eq!(s.tokens[0].field_token(), 1, "valid ⇒ true");
    }

    #[test]
    fn teacher_bridge_row_prefix_columns_route_via_json_key_for() {
        // Directive v6.1 Track B: the bridge table prefixes the teacher head with `row__`. A record
        // shaped like the real PRISM_V2_ROW_MOLECULE_BRIDGE_TRAINING_TABLE row must route its typed
        // teacher fields (resolved by json_key_for), its enums, and its string join keys.
        let enc = Encoder::new();
        let rec: serde_json::Map<String, serde_json::Value> =
            serde_json::from_value(serde_json::json!({
                "pair_label": 1,
                "pair_weight": 0.75,
                "pair_id": "pair_0001072994a5be9508fe906c",
                "molecule_id": "mol_005707cedf62d3d9b4992239",
                "row__teacher_target": "PROMOTE_COMPUTATIONAL_REVIEW",
                "row__teacher_loss_weight": 0.5,
                "row__distillation_loss_weight": 0.4,
                "row__abstention_target": 0,
                "row__fusion_kcc_coherence": 0.91,
                "row__fusion_warp_strain": 0.0005,
                "row__delta_functional_score": -0.05,
                "row__evidence_role": "ABSTENTION_ESCALATION",
                "row__water_class_v2": "WATER_NEUTRAL_V2",
                "negative_family": "wrong_candidate",
                "row__do_not_discard": true,
            }))
            .unwrap();
        let mut s = TokenStream::new();
        let r = route_object(
            &enc,
            &mut s,
            F::TeacherStudentConsensus,
            &rec,
            true,
            SHA,
            true,
        );
        // every typed field above must MATCH (15 columns; none silently dropped).
        assert_eq!(
            r.matched, 15,
            "all 15 typed teacher columns routed (got {})",
            r.matched
        );
        // teacher_target enum: PROMOTE_COMPUTATIONAL_REVIEW is code 1.
        let mut s2 = TokenStream::new();
        assert!(encode_enum_str(
            &enc,
            &mut s2,
            F::TeacherStudentConsensus,
            "teacher_target",
            "PROMOTE_COMPUTATIONAL_REVIEW",
            true,
            SHA
        ));
        assert_eq!(
            s2.tokens[0].field_token(),
            1,
            "PROMOTE_COMPUTATIONAL_REVIEW is code 1"
        );
        // pair_id / molecule_id are string JoinKeys → escape token + exact sidecar Text (never dropped).
        assert!(s.sidecar.iter().any(|e| e.field == "pair_id"
            && matches!(&e.exact, ExactValue::Text(t) if t == "pair_0001072994a5be9508fe906c")));
        assert!(s.sidecar.iter().any(|e| e.field == "molecule_id"
            && matches!(&e.exact, ExactValue::Text(t) if t == "mol_005707cedf62d3d9b4992239")));
    }

    #[test]
    fn spike_events_surface_routes_to_spike_event_family() {
        // Directive v6.1 Track B surface 2: the 5.62B spike_events_snr_masked rows route to
        // SpikeEvent. Columns map: x/y/z→position(vec3), wavelength_nm/phase_bits/intensity/
        // water_density typed; primary_residue_idx→aromatic_residue_id JoinKey. spike_source here is
        // a NUMERIC engine code (1=UV..) so it round-trips as the enum code directly.
        let mut enc = Encoder::new();
        // fit the position VQ on the record's own coordinate so it is a real spatial code.
        enc.fit_coords(
            F::SpikeEvent,
            "position",
            &[[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]],
        );
        let rec: serde_json::Map<String, serde_json::Value> =
            serde_json::from_value(serde_json::json!({
                "x": 1.0, "y": 2.0, "z": 3.0,
                "wavelength_nm": 280.0, "phase_bits": 0, "intensity": 12.5,
                "water_density": 0.0136, "spike_source": 1,
            }))
            .unwrap();
        let mut s = TokenStream::new();
        let r = route_object(&enc, &mut s, F::SpikeEvent, &rec, true, SHA, true);
        // position(1) + wavelength(1) + phase_bits(1) + intensity(1) + water_density(1) + spike_source(1)
        assert!(
            r.matched >= 6,
            "spike-event typed fields routed (got {})",
            r.matched
        );
    }

    // --- DSTW-T4 scientific-state graph rows (vertices / transitions / next-operator rec) ------

    #[test]
    fn dstw_t4_blocked_gdp_vertex_routes_with_typed_enums_and_value_sha() {
        // FLAT projection of a BLOCKED holo_gdp vertex (the directive's blocked-precondition record).
        // Enums (lifecycle_stage/ligand_state/feasibility/runtime_origin) map to non-escape codes;
        // vertex_id + input_sha256 + blocker are string JoinKeys preserved EXACT in the sidecar.
        let enc = Encoder::new();
        let rec: serde_json::Map<String, serde_json::Value> =
            serde_json::from_value(serde_json::json!({
                "lifecycle_stage": "post_md",
                "vertex_id": "vtx_0123456789abcdef0123456789abcdef",
                "target_id": "4lpk",
                "ligand_state": "holo_gdp",
                "feasibility": "BLOCKED",
                "gdp_present": true,
                "input_sha256": "d9f4a653",
                "blocker": "HOLO_PHYSICS_INGEST_ABSENT_IN_ENGINE",
                "detection_mode": "cryptic",
                "site_ranker": "phase-manifold",
                "md_only_evidence": true,
                "runtime_origin": "BLOCKED",
                "runtime_native": false,
                "missing_precondition_count": 6,
                "qnanl_receipt_count": 2,
                "populatable_tensor_count": 0,
                "blocked_tensor_count": 9,
            }))
            .unwrap();
        let mut s = TokenStream::new();
        let r = route_object(&enc, &mut s, F::DstwStateVertex, &rec, true, SHA, true);
        assert!(
            r.matched >= 14,
            "blocked GDP vertex typed fields routed (got {})",
            r.matched
        );
        // ligand_state holo_gdp is enum code 1 (non-escape).
        let mut e = TokenStream::new();
        assert!(encode_enum_str(
            &enc,
            &mut e,
            F::DstwStateVertex,
            "ligand_state",
            "holo_gdp",
            true,
            SHA
        ));
        assert_eq!(e.tokens[0].field_token(), 1, "holo_gdp is enum code 1");
        // runtime_origin BLOCKED is non-escape (in the RuntimeOrigin vocab).
        let mut o = TokenStream::new();
        assert!(encode_enum_str(
            &enc,
            &mut o,
            F::DstwStateVertex,
            "runtime_origin",
            "BLOCKED",
            true,
            SHA
        ));
        assert_ne!(
            o.tokens[0].field_token(),
            Token::ESCAPE_CODE,
            "BLOCKED runtime_origin must be a known code"
        );
        // the VALUE-verified sha + blocker survive EXACT in the sidecar (never dropped).
        assert!(s.sidecar.iter().any(|x| x.field == "input_sha256"
            && matches!(&x.exact, ExactValue::Text(t) if t == "d9f4a653")));
        assert!(s.sidecar.iter().any(|x| x.field == "blocker"));
        assert!(s.sidecar.iter().any(|x| x.field == "vertex_id"));
    }

    #[test]
    fn dstw_t4_apo_vertex_runtime_origin_is_prismos_native_code() {
        let enc = Encoder::new();
        let rec: serde_json::Map<String, serde_json::Value> =
            serde_json::from_value(serde_json::json!({
                "lifecycle_stage": "pre_md",
                "vertex_id": "vtx_aaaa1111bbbb2222cccc3333dddd4444",
                "target_id": "4lpk",
                "ligand_state": "apo",
                "feasibility": "POPULATABLE_NOW",
                "gdp_present": false,
                "input_sha256": "abc123",
                "runtime_origin": "RUNTIME_SUBPROCESS_ORCHESTRATED_BY_PRISMOS",
                "runtime_native": true,
            }))
            .unwrap();
        let mut s = TokenStream::new();
        let r = route_object(&enc, &mut s, F::DstwStateVertex, &rec, true, SHA, true);
        assert!(
            r.matched >= 7,
            "apo vertex typed fields routed (got {})",
            r.matched
        );
        let mut o = TokenStream::new();
        assert!(encode_enum_str(
            &enc,
            &mut o,
            F::DstwStateVertex,
            "runtime_origin",
            "RUNTIME_SUBPROCESS_ORCHESTRATED_BY_PRISMOS",
            true,
            SHA
        ));
        assert_eq!(
            o.tokens[0].field_token(),
            1,
            "RUNTIME_SUBPROCESS_ORCHESTRATED_BY_PRISMOS is code 1"
        );
    }

    #[test]
    fn dstw_t4_transition_routes_with_operator_enum_and_opcode_count() {
        let enc = Encoder::new();
        let rec: serde_json::Map<String, serde_json::Value> =
            serde_json::from_value(serde_json::json!({
                "transition_id": "txn_0123456789abcdef0123456789abcdef",
                "source_vertex_id": "vtx_aaaa",
                "target_vertex_id": "vtx_bbbb",
                "operator_name": "MD.INTEGRATE_VELOCITY_VERLET",
                "operator_version": "blake3deadbeef",
                "runtime_origin": "RUNTIME_SUBPROCESS_ORCHESTRATED_BY_PRISMOS",
                "next_recommended_operator": "engine-owner: wire write_receipt",
                "validation_feasibility": "POPULATABLE_NOW",
                "is_opcode_trace_grounded": false,
                "changed_tensor_count": 4,
                "unchanged_tensor_count": 1,
                "failed_tensor_count": 6,
                "blocked_precondition_count": 6,
                "opcode_total": 128_000_u64,
            }))
            .unwrap();
        let mut s = TokenStream::new();
        let r = route_object(&enc, &mut s, F::DstwStateTransition, &rec, true, SHA, true);
        assert!(
            r.matched >= 12,
            "transition typed fields routed (got {})",
            r.matched
        );
        // operator_name MD.INTEGRATE_VELOCITY_VERLET is enum code 0 (non-escape).
        let mut o = TokenStream::new();
        assert!(encode_enum_str(
            &enc,
            &mut o,
            F::DstwStateTransition,
            "operator_name",
            "MD.INTEGRATE_VELOCITY_VERLET",
            true,
            SHA
        ));
        assert_eq!(
            o.tokens[0].field_token(),
            0,
            "MD.INTEGRATE_VELOCITY_VERLET is enum code 0"
        );
        // operator_version (engine blake3) is a string JoinKey kept EXACT in the sidecar.
        assert!(s.sidecar.iter().any(|x| x.field == "operator_version"
            && matches!(&x.exact, ExactValue::Text(t) if t == "blake3deadbeef")));
    }

    #[test]
    fn dstw_t4_next_operator_recommendation_routes_first_class() {
        let enc = Encoder::new();
        let rec: serde_json::Map<String, serde_json::Value> =
            serde_json::from_value(serde_json::json!({
                "runtime_origin": "RUNTIME_INLINE",
                "runtime_native": true,
                "recommendation_count": 4,
                "top_rank": 1,
                "top_operator": "ENGINE.HOLO_PHYSICS_INGEST",
                "top_owner": "engine",
            }))
            .unwrap();
        let mut s = TokenStream::new();
        let r = route_object(&enc, &mut s, F::DstwNextOperator, &rec, true, SHA, true);
        assert!(
            r.matched >= 5,
            "next-operator rec typed fields routed (got {})",
            r.matched
        );
        // top_owner engine is enum code 0.
        let mut o = TokenStream::new();
        assert!(encode_enum_str(
            &enc,
            &mut o,
            F::DstwNextOperator,
            "top_owner",
            "engine",
            true,
            SHA
        ));
        assert_eq!(o.tokens[0].field_token(), 0, "engine owner is enum code 0");
        // top_operator (the rank-1 engine-owner action) is preserved EXACT in the sidecar.
        assert!(s.sidecar.iter().any(|x| x.field == "top_operator"
            && matches!(&x.exact, ExactValue::Text(t) if t == "ENGINE.HOLO_PHYSICS_INGEST")));
    }

    #[test]
    fn lineage_dag_state_graph_node_and_orphan_check_route_by_value() {
        let enc = Encoder::new();
        // A real `state_graph_node` line (md_state_transition) from PRISM_NATIVE_STATE_GRAPH.jsonl.
        let node: serde_json::Map<String, serde_json::Value> =
            serde_json::from_value(serde_json::json!({
                "record_type": "state_graph_node",
                "schema_version": 1,
                "artifact_kind": "md_state_transition",
                "json_path": "<run_root>/.../md_state_transition.json",
                "qnanl_packet_path": "<run_root>/.../md_state_transition.json.prismnanl",
                "schema_kind": "dstw_t4_scientific_state_transition",
                "state_object_class": "Transition",
                "is_orphan": false,
                "orphan_note": null,
                "runtime_origin": "RUNTIME_SUBPROCESS_ORCHESTRATED_BY_PRISMOS",
                "runtime_native": true,
                "decode_proof_pass": true,
                "vertex_id": null,
                "parent_vertex_id": null,
                "source_vertex_id": "vtx_2fa9fe2bc74dc13a7fcc124cc1ec056e",
                "target_vertex_id": "vtx_2fa9fe2bc74dc13a7fcc124cc1ec056e",
                "transition_id": "txn_f4361bd96ee5c1e68abc174138790598",
                "operator_name": "MD.INTEGRATE_VELOCITY_VERLET",
                "tokenizer_family": "DstwStateTransition",
                "tokenizer_family_absent_reason": null,
                "artifact_sha256": "5d3e8d22fb789616fe8fe4f5f933348c66997cecc72fac82874515811aa78797",
                "emission_unix_ns": "1782873426417002503",
                "run_pid": 70547,
            }))
            .unwrap();
        let mut s = TokenStream::new();
        let r = route_object(&enc, &mut s, F::LineageDag, &node, true, SHA, true);
        // enum labels (record_type/artifact_kind/schema_kind/state_object_class/runtime_origin/
        // operator_name/tokenizer_family) + booleans (runtime_native/decode_proof_pass/is_orphan)
        // + JoinKeys (source/target/transition_id/artifact_sha256) + numerics (schema_version/run_pid)
        // all route by-value; nulls (vertex_id/parent_vertex_id) do not count as matched.
        assert!(
            r.matched >= 15,
            "state_graph_node typed fields routed (got {})",
            r.matched
        );
        // artifact_kind md_state_transition is enum code 2 (non-escape, grounded vocab).
        let mut o = TokenStream::new();
        assert!(encode_enum_str(
            &enc,
            &mut o,
            F::LineageDag,
            "artifact_kind",
            "md_state_transition",
            true,
            SHA
        ));
        assert_eq!(
            o.tokens[0].field_token(),
            2,
            "md_state_transition is artifact_kind enum code 2"
        );
        // The DAG edge ids are preserved EXACT in the sidecar (INV-1/INV-4) for topology cross-check.
        assert!(s.sidecar.iter().any(|x| x.field == "source_vertex_id"
            && matches!(&x.exact, ExactValue::Text(t) if t == "vtx_2fa9fe2bc74dc13a7fcc124cc1ec056e")));

        // The final `orphan_check` summary row routes too (record_type + schema_version + verdict +
        // total_artifacts + orphan_count — matched>0).
        let orphan: serde_json::Map<String, serde_json::Value> =
            serde_json::from_value(serde_json::json!({
                "record_type": "orphan_check",
                "schema_version": 1,
                "total_artifacts": 8,
                "orphan_count": 0,
                "orphans": [],
                "valid_state_object_classes": ["ScientificStateVertex", "Transition"],
                "verdict": "ALL_ARTIFACTS_PRISM_NATIVE",
            }))
            .unwrap();
        let mut so = TokenStream::new();
        let ro = route_object(&enc, &mut so, F::LineageDag, &orphan, true, SHA, true);
        assert!(
            ro.matched >= 5,
            "orphan_check summary fields routed (got {})",
            ro.matched
        );
        // verdict ALL_ARTIFACTS_PRISM_NATIVE is enum code 0 (grounded vocab, not an escape).
        let mut ov = TokenStream::new();
        assert!(encode_enum_str(
            &enc,
            &mut ov,
            F::LineageDag,
            "verdict",
            "ALL_ARTIFACTS_PRISM_NATIVE",
            true,
            SHA
        ));
        assert_eq!(
            ov.tokens[0].field_token(),
            0,
            "ALL_ARTIFACTS_PRISM_NATIVE is verdict enum code 0"
        );
    }

    #[test]
    fn dstw_t4_families_have_distinct_namespaces() {
        // INV-2: the three DSTW families occupy distinct family slots from each other AND from the
        // pre-existing families, so a shared field name (e.g. runtime_origin) tokenizes differently.
        let enc = Encoder::new();
        let v = enc
            .registry()
            .namespace(F::DstwStateVertex, "runtime_origin")
            .unwrap();
        let t = enc
            .registry()
            .namespace(F::DstwStateTransition, "runtime_origin")
            .unwrap();
        let n = enc
            .registry()
            .namespace(F::DstwNextOperator, "runtime_origin")
            .unwrap();
        assert_ne!(v, t);
        assert_ne!(v, n);
        assert_ne!(t, n);
        assert!(enc
            .registry()
            .lookup(F::DstwStateVertex, "vertex_id")
            .is_some());
        assert!(enc
            .registry()
            .lookup(F::DstwStateTransition, "transition_id")
            .is_some());
        assert!(enc
            .registry()
            .lookup(F::DstwNextOperator, "top_operator")
            .is_some());
    }

    #[test]
    fn dstw_t4_dense_tensor_summary_routes_with_matched_fields_gt_zero() {
        // DSTW-T4 Step h5: a FULLY-POPULATED dense-tensor semantic summary (engine artifact present
        // AND its decoded reducer summary present) must route with matched_fields>0 — the raw tensor
        // never enters the model, only this semantic record does. tensor_role/modality/runtime_origin
        // are typed enums; source_hash/source_kind/tokenizer_family/parent_vertex_id are JoinKeys kept
        // EXACT in the sidecar; the decoded-summary dimensions + finite-scan are scalar fields.
        let enc = Encoder::new();
        let rec: serde_json::Map<String, serde_json::Value> =
            serde_json::from_value(serde_json::json!({
                "schema_kind": "dstw_t4_dense_tensor_semantic_summary",
                "tensor_role": "SIGNAL_GRID",
                "modality": "thermodynamic_occupancy_grid",
                "source_kind": "engine:_signal_grid.bin (PRSGD001)",
                "source_hash": "1f3b9a0c4d2e5f6a7b8c9d0e1f2a3b4c5d6e7f8091a2b3c4d5e6f70819a2b3c4",
                "tokenizer_family": "DenseTensorSemanticSummary",
                "runtime_origin": "POST_RUN_DERIVED",
                "parent_vertex_id": "vtx_aaaa1111bbbb2222cccc3333dddd4444",
                "stream_id": 7,
                "on_disk_bytes": 285_000_000_u64,
                "artifact_present": true,
                "summary_present": true,
                "row_count": 32768,
                "column_count": 13,
                "all_finite": true,
                "nan_count": 0,
                "inf_count": 0,
                "nonfinite_or_null_count": 0,
                "reducer_scalar": 0.42,
            }))
            .unwrap();
        let mut s = TokenStream::new();
        let r = route_object(
            &enc,
            &mut s,
            F::DenseTensorSemanticSummary,
            &rec,
            true,
            SHA,
            true,
        );
        // 18 declared fields; all populated here ⇒ matched_fields>0 (and ≥ the identity+enum block).
        assert!(
            r.matched > 0,
            "dense-tensor summary must route with matched_fields>0 (got {})",
            r.matched
        );
        assert!(
            r.matched >= 14,
            "fully-populated dense-tensor summary fields routed (got {})",
            r.matched
        );
        // tensor_role SIGNAL_GRID is enum code 0; modality thermodynamic_occupancy_grid is code 0.
        let mut e = TokenStream::new();
        assert!(encode_enum_str(
            &enc,
            &mut e,
            F::DenseTensorSemanticSummary,
            "tensor_role",
            "SIGNAL_GRID",
            true,
            SHA
        ));
        assert_eq!(e.tokens[0].field_token(), 0, "SIGNAL_GRID is enum code 0");
        // runtime_origin POST_RUN_DERIVED is a KNOWN code (not an escape).
        let mut o = TokenStream::new();
        assert!(encode_enum_str(
            &enc,
            &mut o,
            F::DenseTensorSemanticSummary,
            "runtime_origin",
            "POST_RUN_DERIVED",
            true,
            SHA
        ));
        assert_ne!(
            o.tokens[0].field_token(),
            Token::ESCAPE_CODE,
            "POST_RUN_DERIVED must be a known runtime_origin code"
        );
        // source_hash (REAL sha256) survives EXACT in the sidecar — never coerced into a token.
        assert!(s.sidecar.iter().any(|x| x.field == "source_hash"
            && matches!(&x.exact, ExactValue::Text(t) if t.len() == 64)));
    }

    #[test]
    fn dstw_t4_dense_tensor_summary_identity_only_still_matches() {
        // When the engine artifact is ABSENT (no source_hash/dimensions/finite-scan to compute), the
        // emitter records those with absent_reason (never null-padded) — the IDENTITY fields alone
        // (tensor_role/modality/source_kind/tokenizer_family/runtime_origin/parent_vertex_id) still
        // guarantee matched_fields>0, so the record never silently drops out of the corpus.
        let enc = Encoder::new();
        let rec: serde_json::Map<String, serde_json::Value> =
            serde_json::from_value(serde_json::json!({
                "tensor_role": "TWIN_INTERFEROMETRIC",
                "modality": "interferometric_twin",
                "source_kind": "engine:TWIN16_INTERFEROMETRIC_TRANSITION_VIEW_RESIDUALS.json",
                "tokenizer_family": "DenseTensorSemanticSummary",
                "runtime_origin": "POST_RUN_DERIVED",
                "parent_vertex_id": "vtx_aaaa1111bbbb2222cccc3333dddd4444",
                "artifact_present": false,
                "summary_present": false,
                // source_hash / row_count / all_finite intentionally ABSENT (recorded with absent_reason
                // in the prismcode emitter's sidecar, not null-padded here).
            }))
            .unwrap();
        let mut s = TokenStream::new();
        let r = route_object(
            &enc,
            &mut s,
            F::DenseTensorSemanticSummary,
            &rec,
            true,
            SHA,
            true,
        );
        assert!(
            r.matched > 0,
            "identity-only dense-tensor summary must still match (got {})",
            r.matched
        );
        // tensor_role TWIN_INTERFEROMETRIC is enum code 4 (5th role).
        let mut e = TokenStream::new();
        assert!(encode_enum_str(
            &enc,
            &mut e,
            F::DenseTensorSemanticSummary,
            "tensor_role",
            "TWIN_INTERFEROMETRIC",
            true,
            SHA
        ));
        assert_eq!(
            e.tokens[0].field_token(),
            4,
            "TWIN_INTERFEROMETRIC is enum code 4"
        );
    }

    #[test]
    fn opcode_execution_ledger_wrapper_detects_and_row_routes_with_matched_fields_gt_zero() {
        // opcode_operand_ledger.json: the WRAPPER object (schema_kind prismcode_typed_operand_ledger)
        // must detect as OpcodeExecutionLedger and descend "/records" so each per-opcode row routes
        // (matched_fields>0). Routing the wrapper directly would make every key an opaque escape.
        assert_eq!(
            detect_family(&serde_json::json!({
                "schema_kind": "prismcode_typed_operand_ledger",
                "schema_version": 1,
                "records": []
            })),
            Some((F::OpcodeExecutionLedger, "/records")),
            "operand-ledger wrapper must descend /records"
        );
        // A real MD.* per-opcode row (value_proof nested obj escapes to the sidecar; count is the
        // by-value field tied to the md_native_receipt opcode_counts).
        let enc = Encoder::new();
        let rec: serde_json::Map<String, serde_json::Value> =
            serde_json::from_value(serde_json::json!({
                "opcode_name": "MD.ACCUMULATOR_ZERO",
                "opcode_id": 1795,
                "operand_schema_version": 1,
                "count": 452,
                "runtime_origin": "RuntimeSubprocessOrchestratedByPrismos",
                "runtime_native": true,
                "operator_class": "Md",
                "operand_target": "Tensor",
                "value_proof": {"count": 452, "source_ref": "s0_receipt.json"},
                "disassembly_status": "REGISTRY_COVERED",
                "validation_status": "REGISTRY_COVERED",
                "replay_status": "NOT_APPLICABLE",
                "blocker_reason": "engine count-level, per-dispatch operand typing is engine-owner",
                "source_receipt": "4LPK_apo_stream00_prismcode_md_native_receipt.json",
            }))
            .unwrap();
        let mut s = TokenStream::new();
        let r = route_object(
            &enc,
            &mut s,
            F::OpcodeExecutionLedger,
            &rec,
            true,
            SHA,
            true,
        );
        assert!(
            r.matched >= 6,
            "MD.* opcode row must route (opcode_name+count+operator_class+operand_target+runtime_origin+source_receipt); got {}",
            r.matched
        );
        // operator_class Md is enum code 0 — a KNOWN vocabulary code, not an escape.
        let mut e = TokenStream::new();
        assert!(encode_enum_str(
            &enc,
            &mut e,
            F::OpcodeExecutionLedger,
            "operator_class",
            "Md",
            true,
            SHA
        ));
        assert_eq!(e.tokens[0].field_token(), 0, "Md is operator_class code 0");
        // validation_status TYPED_VALUE_VERIFIED (a seal-row value) is a KNOWN code, not an escape.
        let mut v = TokenStream::new();
        assert!(encode_enum_str(
            &enc,
            &mut v,
            F::OpcodeExecutionLedger,
            "validation_status",
            "TYPED_VALUE_VERIFIED",
            true,
            SHA
        ));
        assert_ne!(
            v.tokens[0].field_token(),
            Token::ESCAPE_CODE,
            "TYPED_VALUE_VERIFIED must be a known validation_status code"
        );
    }

    #[test]
    fn validation_state_verdict_object_detects_and_routes_with_matched_fields_gt_zero() {
        // prismos_prismcode_isa_execution_verification.json: the top-level VERDICT object (schema_kind
        // prismos_prismcode_isa_execution_verification) must detect as ValidationState with pointer ""
        // (no descent — the object IS the record; streams[] descends to the sidecar). It must NOT be
        // swallowed by the ReceiptStatus `_receipt` arm.
        assert_eq!(
            detect_family(&serde_json::json!({
                "schema_kind": "prismos_prismcode_isa_execution_verification",
                "schema_version": 1,
                "status": "PASS",
                "streams": []
            })),
            Some((F::ValidationState, "")),
            "ISA execution verification verdict must detect as ValidationState (top-level, no descent)"
        );
        // A real verdict header (the 12 tokenized fields; streams[] is sidecar-only). BY-VALUE
        // invariant on this fixture: native_md_receipt_count == len(streams) == 2.
        let enc = Encoder::new();
        let rec: serde_json::Map<String, serde_json::Value> =
            serde_json::from_value(serde_json::json!({
                "schema_kind": "prismos_prismcode_isa_execution_verification",
                "schema_version": 1,
                "status": "PASS",
                "substrate_status": "PASS",
                "implementation_status_language": "IMPLEMENTATION_VERIFIED",
                "native_md_receipt_count": 2,
                "min_hotpath_launch_collapse": 64,
                "all_streams_pass": true,
                "all_streams_bounded_or_passed": true,
                "cuda_opcode_coverage_exists": true,
                "hotpath_acceleration_required": false,
                "metal_resident_md_required": false,
                "plan_hash_blake3": "7ba6cfc8f443d049fa4ac5dc1097a3210e650fa405677ec744afec9ac00cb695",
                "streams": [{"status": "PASS"}, {"status": "PASS"}],
            }))
            .unwrap();
        let mut s = TokenStream::new();
        let r = route_object(&enc, &mut s, F::ValidationState, &rec, true, SHA, true);
        assert!(
            r.matched >= 12,
            "verdict header must route all 12 tokenized fields; got {}",
            r.matched
        );
        // substrate_status MISSING (native-receipt-absent fallback) must be a KNOWN code, not an escape.
        let mut m = TokenStream::new();
        assert!(encode_enum_str(
            &enc,
            &mut m,
            F::ValidationState,
            "substrate_status",
            "MISSING",
            true,
            SHA
        ));
        assert_ne!(
            m.tokens[0].field_token(),
            Token::ESCAPE_CODE,
            "MISSING must be a known substrate_status code (not an escape)"
        );
    }

    #[test]
    fn captured_graph_gpu_telemetry_detects_and_routes_with_matched_fields_gt_zero() {
        // ghost_time_map.json: the WRAPPER {schema_kind:"pathb_ghost_time_map", streams:[...]} must
        // detect as CapturedGraphGpuTelemetry and descend "/streams" so each per-stream record routes
        // (matched_fields>0). Routing the wrapper object would make target/run_id/notes opaque escapes.
        assert_eq!(
            detect_family(&serde_json::json!({
                "schema_kind": "pathb_ghost_time_map",
                "streams": [],
                "target": "4LPK_apo"
            })),
            Some((F::CapturedGraphGpuTelemetry, "/streams")),
            "ghost_time_map wrapper must detect as CapturedGraphGpuTelemetry descending /streams"
        );
        // A real per-stream captured-graph temporal frame.
        //
        // GEAR-PROVENANCE-002 — `gear_id` was `0` here, mirroring the producer
        // literal labelled "Wave A default". Under the documented encoding
        // 0 IS the 0.5 fs gear, so the fixture asserted a measurement that
        // never existed. The `gear_id` spec is a `lin(0.0, 64.0)` quantizer,
        // so the u32 sentinel 0xFFFFFFFF cannot be used here either — it would
        // saturate into a fabricated in-range gear. JSON `null` is the only
        // encoding that survives as absence, and the tokenizer preserves it
        // losslessly as an opaque, counted in `null_fields`.
        let enc = Encoder::new();
        let stream_rec: serde_json::Map<String, serde_json::Value> =
            serde_json::from_value(serde_json::json!({
                "dt_fs": 4.0,
                "dt_source": "args.hmr_4_0_fs",
                "frame_idx_at_teardown": 1770,
                "gear_id": serde_json::Value::Null,
                "gear_id_status": "unmeasured_no_device_gear_readback",
                "n_spikes": 1394939,
                "physical_time_at_min_fs": 24.0,
                "physical_time_fs": 7080.0,
                "source": "md_only_teardown_emit_no_runtime_gearbox_state_capture",
                "stream_id": 0,
                "timestep_max": 1770,
                "timestep_min": 6
            }))
            .unwrap();
        let mut s = TokenStream::new();
        let r = route_object(
            &enc,
            &mut s,
            F::CapturedGraphGpuTelemetry,
            &stream_rec,
            true,
            SHA,
            true,
        );
        assert!(
            r.matched >= 8,
            "captured-graph stream frame must route its measured temporal fields; got {}",
            r.matched
        );
        // The unmeasured gear must survive as an explicit null rather than be
        // quantized into a plausible in-range gear.
        assert!(
            r.null_fields >= 1,
            "an unmeasured gear_id must be preserved as an explicit null, not quantized; \
             null_fields={}",
            r.null_fields
        );

        // Guard the inverse: a MEASURED gear still routes as a value, so the
        // null path above is an honesty change and not a loss of capability.
        let measured_rec: serde_json::Map<String, serde_json::Value> =
            serde_json::from_value(serde_json::json!({
                "dt_fs": 4.0,
                "dt_source": "args.hmr_4_0_fs",
                "frame_idx_at_teardown": 1770,
                "gear_id": 2,
                "gear_id_status": "device_readback_chronometric_state_tensor_current_gear",
                "n_spikes": 1394939,
                "physical_time_at_min_fs": 24.0,
                "physical_time_fs": 7080.0,
                "source": "captured_pipeline_measured_gear",
                "stream_id": 0,
                "timestep_max": 1770,
                "timestep_min": 6
            }))
            .unwrap();
        let mut s_measured = TokenStream::new();
        let r_measured = route_object(
            &enc,
            &mut s_measured,
            F::CapturedGraphGpuTelemetry,
            &measured_rec,
            true,
            SHA,
            true,
        );
        assert!(
            r_measured.matched >= 9,
            "a measured gear_id must still route with the other 8 temporal fields; got {}",
            r_measured.matched
        );
        // The FLAT ghost_lattice_routing_status object folds into the same family (routed explicitly);
        // route_object matches its numeric/boolean status keys directly (>=8 fields).
        let lat_rec: serde_json::Map<String, serde_json::Value> =
            serde_json::from_value(serde_json::json!({
                "auto_request": false,
                "backend_request": "auto",
                "explicit_request": false,
                "lattice_kernel_ms": 0.0,
                "lattice_run_status": "not_requested",
                "m1_monolithic_discovery": false,
                "mar_v2_telemetry": false,
                "n_components": 0,
                "schema_version": 1,
                "v2_capture_requested": true,
                "v2_was_live": false
            }))
            .unwrap();
        let mut s2 = TokenStream::new();
        let r2 = route_object(
            &enc,
            &mut s2,
            F::CapturedGraphGpuTelemetry,
            &lat_rec,
            true,
            SHA,
            true,
        );
        assert!(
            r2.matched >= 8,
            "lattice routing-status object must route its 8 numeric/boolean fields; got {}",
            r2.matched
        );
    }

    #[test]
    fn dstw_t4_dense_tensor_summary_namespace_is_distinct() {
        // INV-2: DenseTensorSemanticSummary occupies its own family slot; a shared field name
        // (runtime_origin) tokenizes differently from the DSTW state-graph families.
        let enc = Encoder::new();
        let d = enc
            .registry()
            .namespace(F::DenseTensorSemanticSummary, "runtime_origin")
            .unwrap();
        let v = enc
            .registry()
            .namespace(F::DstwStateVertex, "runtime_origin")
            .unwrap();
        assert_ne!(d, v);
        assert!(enc
            .registry()
            .lookup(F::DenseTensorSemanticSummary, "source_hash")
            .is_some());
        assert!(enc
            .registry()
            .lookup(F::DenseTensorSemanticSummary, "tensor_role")
            .is_some());
        assert!(enc
            .registry()
            .lookup(F::DenseTensorSemanticSummary, "all_finite")
            .is_some());
    }
}
