//! Canonical, versioned Hyper-Runtime request authority.
//!
//! Client attribution is deliberately outside [`HyperRuntimeRequestIdentityV1`]:
//! equivalent CLI, systemd, GUI, agent, and CoPilot intents must serialize to
//! and hash as one physical/evidentiary request.

use crate::binary_runtime::{
    load_validated_md_engine_policy, MdEnginePolicyV1, MdExecutionProfileIdentityV1,
    MdForceContractV1, MdIntegratorPolicyV1, MdPostMdAggregationPolicyV1,
    MdThermSurfaceFallbackReasonV1, MdThermSurfacePolicyV1, PcError, PcResult,
    ValidatedMdEnginePolicyV1,
};
use crate::operand_ledger::sha256_file_hex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::collections::BTreeMap;
#[cfg(unix)]
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Component, Path, PathBuf};

pub const EXECUTION_PROFILE_SCHEMA: &str = "prismos.hyper_runtime.execution_profile";
pub const TARGET_EXECUTION_INTENT_SCHEMA: &str = "prismos.hyper_runtime.target_execution_intent";
pub const HYPER_RUNTIME_REQUEST_SCHEMA: &str = "prismos.hyper_runtime.request";
pub const CLIENT_INTENT_SCHEMA: &str = "prismos.hyper_runtime.client_intent";
pub const GOVERNED_BRANCH_SELECTION_SCHEMA: &str = "prismos.cognition.governed_branch_selection";
pub const NEXT_BRANCH_EXECUTION_REQUEST_SCHEMA: &str =
    "prismos.cognition.next_branch_execution_request";
pub const CANONICAL_EXECUTION_PROFILE_SHA256: &str =
    "32b3ad26c4f75d9c89ac71a248ca189ab731aff3d51cbb6c6f1e9aff79cd9ce4";
const CANONICAL_ACTIVE_RELEASE_POINTER: &str = "/var/lib/prismos/engine-releases/ACTIVE";

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ArtifactIdentityV1 {
    pub locator: PathBuf,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SizedArtifactIdentityV1 {
    pub locator: PathBuf,
    pub sha256: String,
    pub size_bytes: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum MolecularContrastKindV1 {
    ApoHolo,
    DesignReference,
    GlycanMechanisticProbe,
    IdenticalTopologyNegativeControl,
}

/// Complete second-system identity for a governed coupled-TWIN request.
///
/// This is intentionally one optional object rather than independently
/// optional fields: a target is either an ordinary single-system request or it
/// carries the complete coupled identity before request materialization.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CoupledExecutionIntentV1 {
    pub topology_b: ArtifactIdentityV1,
    pub contrast_kind: MolecularContrastKindV1,
    pub identical_topology_negative_control: bool,
    pub probe_residue_ids: Vec<usize>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ExecutionProfileIdentityV1 {
    pub id: String,
    pub version: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PacketAuthorityV1 {
    pub prismcode_packet: ArtifactIdentityV1,
    pub qnanl_binding: ArtifactIdentityV1,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TopologyAndInputIdentityPolicyV1 {
    pub target_id_required: bool,
    pub target_request_root: PathBuf,
    pub target_request_schema: String,
    pub target_request_schema_version: u32,
    pub topology_locator_policy: String,
    pub input_identity_policy: String,
    pub topology_sha256_required: bool,
    pub input_sha256_required: bool,
    pub unresolved_runtime_input_status: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EngineAndSourceEpochAuthorityV1 {
    pub engine_locator: PathBuf,
    pub active_release_required: bool,
    pub source_epoch_required: bool,
    pub raw_engine_safe_defaults_unchanged: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ForceAndIntegratorPolicyV1 {
    pub force_contract: String,
    pub integrator: String,
    pub independent_raw_engine_flags_forbidden: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct GraphAndMetalResidentPolicyV1 {
    pub cuda_graph_required: bool,
    pub metal_resident_required: bool,
    pub hotpath_acceleration_required: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PathBDstwSuperresolutionPolicyV1 {
    pub path_b_required: bool,
    pub dstw_required: bool,
    pub superresolution_required: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ObserverAndEvidencePolicyV1 {
    pub qnanl_inline_seal_verify_required: bool,
    pub runtime_events_required: bool,
    pub evidence_qualification_required: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub enum PostMdAggregationPolicyV1 {
    #[serde(rename = "preserve_lane_default")]
    PreserveLaneDefault,
    #[serde(rename = "full")]
    Full,
    #[serde(rename = "skip")]
    Skip,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ThermSurfacePolicyV1 {
    PreserveLaneDefault,
    Clusters,
    Lattice,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ThermSurfaceFallbackReasonV1 {
    RequestedLatticeUnavailable,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ThermAndPostMdPolicyV1 {
    pub therm_required: bool,
    pub requested_post_md_aggregation: PostMdAggregationPolicyV1,
    pub effective_post_md_aggregation: PostMdAggregationPolicyV1,
    pub full_post_md_authorized: bool,
    pub requested_therm_surface: ThermSurfacePolicyV1,
    pub effective_therm_surface: ThermSurfacePolicyV1,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fallback_reason: Option<ThermSurfaceFallbackReasonV1>,
    pub therm_surface_fallback_authorized: bool,
    pub effective_surface_provenance_required: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct OutputAndCorpusPolicyV1 {
    pub run_output_root: PathBuf,
    pub governed_corpus_handoff_required: bool,
    pub output_identity_required: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ExecutionProfileV1 {
    pub schema_kind: String,
    pub schema_version: u32,
    pub profile: ExecutionProfileIdentityV1,
    pub packet_authority: PacketAuthorityV1,
    pub topology_and_input_identity: TopologyAndInputIdentityPolicyV1,
    pub engine_and_source_epoch_authority: EngineAndSourceEpochAuthorityV1,
    pub force_and_integrator_policy: ForceAndIntegratorPolicyV1,
    pub graph_and_metal_resident_policy: GraphAndMetalResidentPolicyV1,
    pub path_b_dstw_superresolution_policy: PathBDstwSuperresolutionPolicyV1,
    pub observer_and_evidence_policy: ObserverAndEvidencePolicyV1,
    pub therm_and_post_md_policy: ThermAndPostMdPolicyV1,
    pub output_and_corpus_policy: OutputAndCorpusPolicyV1,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TargetExecutionIntentV1 {
    pub schema_kind: String,
    pub schema_version: u32,
    pub target_id: String,
    pub topology: ArtifactIdentityV1,
    pub input: ArtifactIdentityV1,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coupled: Option<CoupledExecutionIntentV1>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReplicateIntentV1 {
    pub group_id: String,
    pub index: u32,
    pub count: u32,
    pub seed: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ClientExecutionIntentV1 {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub experiment: Option<crate::durable_experiment::ExperimentPolicyV1>,
    pub schema_kind: String,
    pub schema_version: u32,
    pub execution_profile: PathBuf,
    pub target_id: String,
    pub topology: PathBuf,
    pub input: PathBuf,
    pub output: PathBuf,
    pub engine_argv: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replicate: Option<ReplicateIntentV1>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_transport: Option<ClientArtifactTransportV1>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ClientLigandModeV1 { Apo, Holo }

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ClientArtifactTransportV1 {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prism_bundle_root: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qnanl_force_kernel_packet: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qnanl_force_kernel_receipt: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prismcode_cuda_opcode_coverage: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub science_roots: Option<Vec<PathBuf>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ligand_mode: Option<ClientLigandModeV1>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct NativeEngineIntentV1 {
    pub argv: Vec<String>,
    pub normalized_arguments: BTreeMap<String, Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replicate: Option<ReplicateIntentV1>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_transport: Option<ClientArtifactTransportV1>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub transport_input_bindings: Vec<ArtifactIdentityV1>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NativeRunIntentValidationV1 {
    pub schema_kind: String,
    pub schema_version: u32,
    pub validated: bool,
    pub argv: Vec<String>,
    pub normalized_arguments: BTreeMap<String, Vec<String>>,
    pub explicit_arguments: Vec<String>,
    pub argument_metadata: serde_json::Value,
    pub validation_scope: String,
    pub runtime_admission_performed: bool,
    pub device_initialized: bool,
}

pub fn validate_native_engine_intent(engine: &ArtifactIdentityV1, argv: &[String]) -> PcResult<NativeRunIntentValidationV1> {
    verify_artifact(engine, "native intent parser executable")?;
    let output = std::process::Command::new(&engine.locator)
        .arg("validate-run-intent").arg("--").args(argv).output()?;
    if !output.status.success() {
        return Err(invalid(format!("native engine rejected client intent before launch: status={} stderr={}",
            output.status, String::from_utf8_lossy(&output.stderr).trim())));
    }
    let report: NativeRunIntentValidationV1 = serde_json::from_slice(&output.stdout)?;
    if report.schema_kind != "prismos.native_run_intent_validation" || report.schema_version != 1
        || !report.validated || report.argv != argv || report.runtime_admission_performed
        || report.device_initialized || report.validation_scope != "NATIVE_CLI_TYPES_AND_INTENT_AUTHORITY_BOUNDARY"
    {
        return Err(invalid("native intent parser returned an incompatible validation receipt"));
    }
    verify_artifact(engine, "native intent parser executable after validation")?;
    Ok(report)
}

pub fn native_intent_value<'a>(intent: &'a NativeEngineIntentV1, key: &str) -> Option<&'a str> {
    intent.normalized_arguments.get(key).and_then(|values| values.first()).map(String::as_str)
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RuntimeAuthorityIdentityV1 {
    pub source_epoch_id: String,
    pub active_release: String,
    pub active_lock: ArtifactIdentityV1,
    pub engine: ArtifactIdentityV1,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct HyperRuntimeRequestIdentityV1 {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub experiment: Option<crate::durable_experiment::ExperimentPolicyV1>,
    pub execution_profile: ExecutionProfileIdentityV1,
    pub execution_profile_sha256: String,
    pub packet_authority: PacketAuthorityV1,
    /// Policy decoded from the exact hash-verified packet bytes.  This is part
    /// of canonical identity and must equal the separately configured profile
    /// intent before the request can reach the native adapter.
    pub validated_packet_policy: ValidatedMdEnginePolicyV1,
    /// Present only when an explicitly selected, sealed cognition proposal
    /// authorizes this already-compiled next-branch packet. This authority is
    /// inside request identity; attribution remains outside it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub governed_branch: Option<GovernedBranchAuthorityV1>,
    pub target_id: String,
    pub topology: ArtifactIdentityV1,
    pub input: ArtifactIdentityV1,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coupled: Option<CoupledExecutionIntentV1>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub native_engine_intent: Option<NativeEngineIntentV1>,
    pub runtime_authority: RuntimeAuthorityIdentityV1,
    pub force_and_integrator_policy: ForceAndIntegratorPolicyV1,
    pub graph_and_metal_resident_policy: GraphAndMetalResidentPolicyV1,
    pub path_b_dstw_superresolution_policy: PathBDstwSuperresolutionPolicyV1,
    pub observer_and_evidence_policy: ObserverAndEvidencePolicyV1,
    pub therm_and_post_md_policy: ThermAndPostMdPolicyV1,
    pub output_and_corpus_policy: OutputAndCorpusPolicyV1,
    pub output: PathBuf,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct GovernedBranchAuthorityV1 {
    pub branch_id_sha256: String,
    pub governed_branch_selection: SizedArtifactIdentityV1,
    pub explicit_selection: SizedArtifactIdentityV1,
    pub proposal_id_sha256: String,
    pub selected_proposal: SizedArtifactIdentityV1,
    pub normalization_receipt: SizedArtifactIdentityV1,
    pub validation_receipt: SizedArtifactIdentityV1,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct GovernedExecutionProfileIdentityV1 {
    pub id: String,
    pub version: String,
    pub locator: PathBuf,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct GovernedTargetIdentityV1 {
    pub target_id: String,
    pub descriptor: ArtifactIdentityV1,
    pub topology: ArtifactIdentityV1,
    pub input: ArtifactIdentityV1,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct GovernedPacketIdentityV1 {
    pub locator: PathBuf,
    pub sha256: String,
    pub internal_blake3: String,
    pub schema_kind: String,
    pub schema_version: u32,
    pub validation_level: String,
    pub normalization_receipt: SizedArtifactIdentityV1,
    pub validation_receipt: SizedArtifactIdentityV1,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct GovernedPolicySnapshotV1 {
    pub force_and_integrator_policy: ForceAndIntegratorPolicyV1,
    pub graph_and_metal_resident_policy: GraphAndMetalResidentPolicyV1,
    pub path_b_dstw_superresolution_policy: PathBDstwSuperresolutionPolicyV1,
    pub therm_and_post_md_policy: ThermAndPostMdPolicyV1,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ExplicitSelectionAuthorityV1 {
    pub authority_kind: String,
    pub authority_id: String,
    pub directive_id: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct GovernedProposalIdentityV1 {
    pub proposal_id_sha256: String,
    pub row_sha256: String,
    pub proposal_set_receipt: SizedArtifactIdentityV1,
    pub proposal_artifact: SizedArtifactIdentityV1,
    pub selected_proposal: SizedArtifactIdentityV1,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub non_authoritative_rank: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct GovernedBranchSelectionV1 {
    pub schema_kind: String,
    pub schema_version: u32,
    pub selection_mode: String,
    pub selected: bool,
    pub explicit_selection: SizedArtifactIdentityV1,
    pub selection_authority: ExplicitSelectionAuthorityV1,
    pub proposal: GovernedProposalIdentityV1,
    pub source_epoch_id: String,
    pub execution_profile: GovernedExecutionProfileIdentityV1,
    pub target: GovernedTargetIdentityV1,
    pub packet: GovernedPacketIdentityV1,
    pub policy_snapshot: GovernedPolicySnapshotV1,
    pub evidence_snapshot: ObserverAndEvidencePolicyV1,
    pub output_snapshot: OutputAndCorpusPolicyV1,
    pub output: PathBuf,
    pub branch_source: SizedArtifactIdentityV1,
    pub branch_id_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct NextBranchExecutionRequestV1 {
    pub schema_kind: String,
    pub schema_version: u32,
    pub branch_id_sha256: String,
    pub governed_branch_selection: SizedArtifactIdentityV1,
    pub source_epoch_id: String,
    pub execution_profile: GovernedExecutionProfileIdentityV1,
    pub target: GovernedTargetIdentityV1,
    pub packet: GovernedPacketIdentityV1,
    pub policy_snapshot: GovernedPolicySnapshotV1,
    pub evidence_snapshot: ObserverAndEvidencePolicyV1,
    pub output_snapshot: OutputAndCorpusPolicyV1,
    pub client_surface: String,
    pub output: PathBuf,
}

fn profile_md_engine_policy(profile: &ExecutionProfileV1) -> PcResult<MdEnginePolicyV1> {
    let force_contract = match profile.force_and_integrator_policy.force_contract.as_str() {
        "prismcode-authorized" => MdForceContractV1::PrismcodeAuthorized,
        other => return Err(invalid(format!("unsupported force contract {other}"))),
    };
    let integrator = match profile.force_and_integrator_policy.integrator.as_str() {
        "canonical-hyper-runtime" => MdIntegratorPolicyV1::CanonicalHyperRuntime,
        other => return Err(invalid(format!("unsupported integrator policy {other}"))),
    };
    let map_post_md = |policy: &PostMdAggregationPolicyV1| match policy {
        PostMdAggregationPolicyV1::PreserveLaneDefault => {
            MdPostMdAggregationPolicyV1::PreserveLaneDefault
        }
        PostMdAggregationPolicyV1::Full => MdPostMdAggregationPolicyV1::Full,
        PostMdAggregationPolicyV1::Skip => MdPostMdAggregationPolicyV1::Skip,
    };
    let map_surface = |policy: &ThermSurfacePolicyV1| match policy {
        ThermSurfacePolicyV1::PreserveLaneDefault => MdThermSurfacePolicyV1::PreserveLaneDefault,
        ThermSurfacePolicyV1::Clusters => MdThermSurfacePolicyV1::Clusters,
        ThermSurfacePolicyV1::Lattice => MdThermSurfacePolicyV1::Lattice,
    };
    let map_fallback_reason = |reason: &ThermSurfaceFallbackReasonV1| match reason {
        ThermSurfaceFallbackReasonV1::RequestedLatticeUnavailable => {
            MdThermSurfaceFallbackReasonV1::RequestedLatticeUnavailable
        }
    };
    let therm_policy = &profile.therm_and_post_md_policy;
    Ok(MdEnginePolicyV1 {
        schema_version: 2,
        execution_profile: Some(MdExecutionProfileIdentityV1 {
            id: profile.profile.id.clone(),
            version: profile.profile.version.clone(),
        }),
        force_contract,
        integrator,
        raw_engine_safe_defaults_unchanged: profile
            .engine_and_source_epoch_authority
            .raw_engine_safe_defaults_unchanged,
        independent_raw_engine_flags_forbidden: profile
            .force_and_integrator_policy
            .independent_raw_engine_flags_forbidden,
        topology_sha256_required: profile.topology_and_input_identity.topology_sha256_required,
        input_sha256_required: profile.topology_and_input_identity.input_sha256_required,
        source_epoch_required: profile
            .engine_and_source_epoch_authority
            .source_epoch_required,
        cuda_graph_required: profile.graph_and_metal_resident_policy.cuda_graph_required,
        metal_resident_required: profile
            .graph_and_metal_resident_policy
            .metal_resident_required,
        hotpath_acceleration_required: profile
            .graph_and_metal_resident_policy
            .hotpath_acceleration_required,
        path_b_required: profile.path_b_dstw_superresolution_policy.path_b_required,
        dstw_required: profile.path_b_dstw_superresolution_policy.dstw_required,
        superresolution_required: profile
            .path_b_dstw_superresolution_policy
            .superresolution_required,
        qnanl_inline_seal_verify_required: profile
            .observer_and_evidence_policy
            .qnanl_inline_seal_verify_required,
        runtime_events_required: profile.observer_and_evidence_policy.runtime_events_required,
        evidence_qualification_required: profile
            .observer_and_evidence_policy
            .evidence_qualification_required,
        therm_required: therm_policy.therm_required,
        requested_post_md_aggregation: map_post_md(&therm_policy.requested_post_md_aggregation),
        effective_post_md_aggregation: map_post_md(&therm_policy.effective_post_md_aggregation),
        full_post_md_authorized: therm_policy.full_post_md_authorized,
        requested_therm_surface: map_surface(&therm_policy.requested_therm_surface),
        effective_therm_surface: map_surface(&therm_policy.effective_therm_surface),
        therm_surface_fallback_reason: therm_policy
            .fallback_reason
            .as_ref()
            .map(map_fallback_reason),
        therm_surface_fallback_authorized: therm_policy.therm_surface_fallback_authorized,
        effective_surface_provenance_required: profile
            .therm_and_post_md_policy
            .effective_surface_provenance_required,
        governed_corpus_handoff_required: profile
            .output_and_corpus_policy
            .governed_corpus_handoff_required,
        output_identity_required: profile.output_and_corpus_policy.output_identity_required,
    })
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RequestAttributionV1 {
    pub client_surface: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct HyperRuntimeRequestV1 {
    pub schema_kind: String,
    pub schema_version: u32,
    pub request_id_sha256: String,
    pub request: HyperRuntimeRequestIdentityV1,
    pub attribution: RequestAttributionV1,
}

#[derive(Clone, Debug, Deserialize)]
struct ActiveStateLockV1 {
    schema_kind: String,
    schema_version: u32,
    active_release: String,
    engine_sha256: String,
    source_epoch_id: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ExplicitProposalIdentityV1 {
    proposal_id_sha256: String,
    row_sha256: String,
    proposal_set_receipt: SizedArtifactIdentityV1,
    proposal_artifact: SizedArtifactIdentityV1,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    non_authoritative_rank: Option<u64>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ExplicitBranchSelectionV1 {
    schema_kind: String,
    schema_version: u32,
    selection_mode: String,
    selected: bool,
    selection_authority: ExplicitSelectionAuthorityV1,
    proposal: ExplicitProposalIdentityV1,
    source_epoch_id: String,
    execution_profile: GovernedExecutionProfileIdentityV1,
    target: GovernedTargetIdentityV1,
    output: PathBuf,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SelectedProposalV1 {
    schema_kind: String,
    schema_version: u32,
    proposal_id_sha256: String,
    row_sha256: String,
    canonical_row_utf8: String,
    row: serde_json::Value,
    proposal_set_artifact: SizedArtifactIdentityV1,
    proposal_set_receipt: SizedArtifactIdentityV1,
    source_epoch_id: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SealedProposalSetReceiptV1 {
    schema_kind: String,
    schema_version: u32,
    seal_status: String,
    source_epoch_id: String,
    proposal_set: SizedArtifactIdentityV1,
    sample_execution_receipt: SizedArtifactIdentityV1,
    proposal_count: u64,
    proposal_id_algorithm: String,
    proposal_ids: Vec<String>,
    proposal_ids_sha256: String,
    seal_sha256: String,
}

#[derive(Debug, Serialize)]
struct SealedProposalSetDigestV1<'a> {
    schema_kind: &'a str,
    schema_version: u32,
    seal_status: &'a str,
    source_epoch_id: &'a str,
    proposal_set: &'a SizedArtifactIdentityV1,
    sample_execution_receipt: &'a SizedArtifactIdentityV1,
    proposal_count: u64,
    proposal_id_algorithm: &'a str,
    proposal_ids: &'a [String],
    proposal_ids_sha256: &'a str,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PacketNormalizationReceiptV1 {
    schema_kind: String,
    schema_version: u32,
    status: String,
    input_packet: SizedArtifactIdentityV1,
    output_packet: SizedPacketArtifactIdentityV1,
    native_receipt: SizedArtifactIdentityV1,
    source_epoch_id: String,
    branch_id_sha256: String,
    command_argv: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct SizedPacketArtifactIdentityV1 {
    locator: PathBuf,
    sha256: String,
    size_bytes: u64,
    internal_blake3: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PacketValidationWitnessV1 {
    status: String,
    level: String,
    packet_hash: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PacketValidationReceiptV1 {
    schema_kind: String,
    schema_version: u32,
    status: String,
    validation_level: String,
    packet: SizedPacketArtifactIdentityV1,
    native_validation_report: SizedArtifactIdentityV1,
    native_validation: PacketValidationWitnessV1,
    source_epoch_id: String,
    branch_id_sha256: String,
    command_argv: Vec<String>,
}

#[derive(Debug, Serialize)]
struct BranchIdentityV1<'a> {
    schema_kind: &'static str,
    schema_version: u32,
    explicit_selection_sha256: &'a str,
    proposal_id_sha256: &'a str,
    source_epoch_id: &'a str,
    execution_profile: &'a GovernedExecutionProfileIdentityV1,
    target: &'a GovernedTargetIdentityV1,
    output: &'a Path,
}

fn invalid(message: impl Into<String>) -> PcError {
    PcError::Security(format!("HYPER_RUNTIME_REQUEST_INVALID:{}", message.into()))
}

fn require_sha256(value: &str, field: &str) -> PcResult<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(invalid(format!("{field} must be lowercase SHA-256")));
    }
    Ok(())
}

fn canonical_json_sha256<T: Serialize>(value: &T) -> PcResult<String> {
    // serde_json's default map is key-sorted (preserve_order is not enabled),
    // matching the cognition producer's sort_keys/separators canonical form.
    let value = serde_json::to_value(value)?;
    let canonical = serde_json::to_vec(&value)?;
    Ok(format!("{:x}", Sha256::digest(&canonical)))
}

fn sized_artifact(identity: &SizedArtifactIdentityV1) -> ArtifactIdentityV1 {
    ArtifactIdentityV1 {
        locator: identity.locator.clone(),
        sha256: identity.sha256.clone(),
    }
}

fn verify_sized_artifact(identity: &SizedArtifactIdentityV1, field: &str) -> PcResult<()> {
    require_absolute(&identity.locator, &format!("{field}.locator"))?;
    verify_artifact(&sized_artifact(identity), field)?;
    let observed = fs::metadata(&identity.locator)?.len();
    if identity.size_bytes == 0 || observed != identity.size_bytes {
        return Err(invalid(format!(
            "{field} size mismatch: declared={} observed={observed}",
            identity.size_bytes
        )));
    }
    Ok(())
}

fn packet_sized_artifact(identity: &SizedPacketArtifactIdentityV1) -> SizedArtifactIdentityV1 {
    SizedArtifactIdentityV1 {
        locator: identity.locator.clone(),
        sha256: identity.sha256.clone(),
        size_bytes: identity.size_bytes,
    }
}

fn strict_regular(path: &Path, field: &str) -> PcResult<()> {
    reject_symlink_components(path, field)?;
    let metadata = fs::symlink_metadata(path).map_err(|error| {
        invalid(format!(
            "{field} unavailable at {}: {error}",
            path.display()
        ))
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() == 0 {
        return Err(invalid(format!(
            "{field} must be a nonempty regular non-symlink file: {}",
            path.display()
        )));
    }
    Ok(())
}

// Production authority always starts at filesystem root and requires UID 0.
// The explicit test-only domain permits real filesystem fixtures owned by the
// test runner; it cannot be selected by a runtime flag or environment variable.
enum AuthorityProtection {
    Production,
    #[cfg(test)]
    CpuFixture { root: PathBuf, uid: u32 },
}

#[cfg(unix)]
fn require_protected_ancestry(path: &Path, field: &str, protection: &AuthorityProtection) -> PcResult<()> {
    let (root, expected_uid) = match protection {
        AuthorityProtection::Production => (Path::new("/"), 0),
        #[cfg(test)]
        AuthorityProtection::CpuFixture { root, uid } => (root.as_path(), *uid),
    };
    let relative = path.strip_prefix(root).map_err(|_| invalid(format!("{field} escapes protection root")))?;
    let mut current = root.to_path_buf();
    let mut paths = vec![current.clone()];
    for component in relative.components() {
        if !matches!(component, Component::Normal(_)) {
            return Err(invalid(format!("{field} has a non-normal authority component")));
        }
        current.push(component.as_os_str());
        paths.push(current.clone());
    }
    for path in paths {
        let metadata = fs::symlink_metadata(&path)?;
        if metadata.file_type().is_symlink() || metadata.uid() != expected_uid
            || metadata.permissions().mode() & 0o022 != 0
            || (path != current && !metadata.is_dir())
        {
            return Err(invalid(format!(
                "{field} requires protected ancestry: {} uid={} mode={:04o}",
                path.display(), metadata.uid(), metadata.permissions().mode() & 0o7777
            )));
        }
    }
    Ok(())
}

#[cfg(not(unix))]
fn require_protected_ancestry(_path: &Path, _field: &str, _protection: &AuthorityProtection) -> PcResult<()> {
    Err(invalid("protected ACTIVE authority requires Unix ownership and permissions"))
}

#[cfg(unix)]
fn require_immutable_mode(path: &Path, field: &str) -> PcResult<()> {
    let mode = fs::symlink_metadata(path)?.permissions().mode();
    if mode & 0o222 != 0 {
        return Err(invalid(format!(
            "{field} must not be user/group/world writable: {} mode={:04o}",
            path.display(),
            mode & 0o7777
        )));
    }
    Ok(())
}

#[cfg(not(unix))]
fn require_immutable_mode(_path: &Path, _field: &str) -> PcResult<()> {
    Err(invalid(
        "immutable ACTIVE release mode verification requires Unix permissions",
    ))
}

fn reject_symlink_components(path: &Path, field: &str) -> PcResult<()> {
    let mut current = PathBuf::new();
    for component in path.components() {
        current.push(component.as_os_str());
        if matches!(component, Component::RootDir | Component::Prefix(_)) {
            continue;
        }
        match fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(invalid(format!(
                    "{field} contains a symlink component: {}",
                    current.display()
                )));
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => break,
            Err(error) => {
                return Err(invalid(format!(
                    "{field} component unavailable at {}: {error}",
                    current.display()
                )));
            }
        }
    }
    Ok(())
}

fn reject_lexical_dot_components(path: &Path, field: &str) -> PcResult<()> {
    // `Path::components()` intentionally normalizes interior `.` segments, so
    // inspect the lexical representation before any filesystem resolution.
    let lexical = path
        .to_str()
        .ok_or_else(|| invalid(format!("{field} must be valid UTF-8")))?;
    if lexical
        .split(std::path::MAIN_SEPARATOR)
        .any(|component| matches!(component, "." | ".."))
    {
        return Err(invalid(format!(
            "{field} may not contain lexical dot components"
        )));
    }
    Ok(())
}

fn read_strict_json<T: for<'de> Deserialize<'de>>(path: &Path, field: &str) -> PcResult<T> {
    strict_regular(path, field)?;
    let raw = fs::read(path)?;
    serde_json::from_slice(&raw)
        .map_err(|error| invalid(format!("{field} malformed at {}: {error}", path.display())))
}

fn verify_artifact(identity: &ArtifactIdentityV1, field: &str) -> PcResult<()> {
    require_sha256(&identity.sha256, &format!("{field}.sha256"))?;
    strict_regular(&identity.locator, field)?;
    let observed = sha256_file_hex(&identity.locator)?;
    if observed != identity.sha256 {
        return Err(invalid(format!(
            "{field} hash mismatch at {}: declared={} observed={observed}",
            identity.locator.display(),
            identity.sha256
        )));
    }
    Ok(())
}

fn require_absolute(path: &Path, field: &str) -> PcResult<()> {
    if !path.is_absolute() {
        return Err(invalid(format!("{field} must be absolute")));
    }
    Ok(())
}

fn require_nonempty(value: &str, field: &str) -> PcResult<()> {
    if value.trim().is_empty() {
        return Err(invalid(format!("{field} must be nonempty")));
    }
    Ok(())
}

pub fn validate_execution_profile(profile: &ExecutionProfileV1) -> PcResult<()> {
    if profile.schema_kind != EXECUTION_PROFILE_SCHEMA || profile.schema_version != 2 {
        return Err(invalid("unsupported execution-profile schema"));
    }
    if !matches!(
        profile.profile.id.as_str(),
        "prismos.hyper-runtime-md" | "prismos.therm-postmd"
    ) || profile.profile.version != "1.0.0"
    {
        return Err(invalid("unsupported execution-profile identity/version"));
    }
    validate_therm_and_post_md_policy(&profile.profile, &profile.therm_and_post_md_policy)?;
    let target = &profile.topology_and_input_identity;
    if !target.target_id_required
        || target.target_request_schema != TARGET_EXECUTION_INTENT_SCHEMA
        || target.target_request_schema_version != 1
        || target.topology_locator_policy != "required_runtime_input"
        || target.input_identity_policy != "required_runtime_input"
        || !target.topology_sha256_required
        || !target.input_sha256_required
        || target.unresolved_runtime_input_status != "EVIDENCE_UNAVAILABLE"
    {
        return Err(invalid(
            "unsupported or non-fail-closed target identity policy",
        ));
    }
    require_absolute(&target.target_request_root, "target_request_root")?;
    require_absolute(
        &profile.packet_authority.prismcode_packet.locator,
        "prismcode_packet.locator",
    )?;
    require_absolute(
        &profile.packet_authority.qnanl_binding.locator,
        "qnanl_binding.locator",
    )?;
    require_absolute(
        &profile.engine_and_source_epoch_authority.engine_locator,
        "engine_locator",
    )?;
    require_absolute(
        &profile.output_and_corpus_policy.run_output_root,
        "run_output_root",
    )?;
    require_sha256(
        &profile.packet_authority.prismcode_packet.sha256,
        "prismcode_packet.sha256",
    )?;
    require_sha256(
        &profile.packet_authority.qnanl_binding.sha256,
        "qnanl_binding.sha256",
    )?;
    if !profile
        .engine_and_source_epoch_authority
        .active_release_required
        || !profile
            .engine_and_source_epoch_authority
            .source_epoch_required
        || !profile
            .engine_and_source_epoch_authority
            .raw_engine_safe_defaults_unchanged
        || profile.force_and_integrator_policy.force_contract != "prismcode-authorized"
        || profile.force_and_integrator_policy.integrator != "canonical-hyper-runtime"
        || !profile
            .force_and_integrator_policy
            .independent_raw_engine_flags_forbidden
        || !profile.graph_and_metal_resident_policy.cuda_graph_required
        || !profile
            .graph_and_metal_resident_policy
            .metal_resident_required
        || !profile
            .graph_and_metal_resident_policy
            .hotpath_acceleration_required
        || !profile.path_b_dstw_superresolution_policy.path_b_required
        || !profile.path_b_dstw_superresolution_policy.dstw_required
        || !profile
            .path_b_dstw_superresolution_policy
            .superresolution_required
        || !profile
            .observer_and_evidence_policy
            .qnanl_inline_seal_verify_required
        || !profile.observer_and_evidence_policy.runtime_events_required
        || !profile
            .observer_and_evidence_policy
            .evidence_qualification_required
        || !profile.therm_and_post_md_policy.therm_required
        || !profile
            .therm_and_post_md_policy
            .effective_surface_provenance_required
        || !profile
            .output_and_corpus_policy
            .governed_corpus_handoff_required
        || !profile.output_and_corpus_policy.output_identity_required
    {
        return Err(invalid("unsupported execution-profile semantics"));
    }
    Ok(())
}

/// Validate the exact requested/effective product-policy tuple before packet
/// comparison or native delegation. The current THERM_POSTMD and strict
/// HYPER_RUNTIME profiles are both fail-closed lattice/Full profiles.
pub fn validate_therm_and_post_md_policy(
    profile: &ExecutionProfileIdentityV1,
    policy: &ThermAndPostMdPolicyV1,
) -> PcResult<()> {
    if policy.requested_post_md_aggregation != policy.effective_post_md_aggregation {
        return Err(invalid("requested/effective post-MD aggregation mismatch"));
    }
    if policy.therm_required
        && (policy.requested_post_md_aggregation != PostMdAggregationPolicyV1::Full
            || !policy.full_post_md_authorized)
    {
        return Err(invalid(
            "required therm profile requires explicitly authorized Full post-MD aggregation",
        ));
    }
    let surface_differs = policy.requested_therm_surface != policy.effective_therm_surface;
    if surface_differs
        && (!policy.therm_surface_fallback_authorized || policy.fallback_reason.is_none())
    {
        return Err(invalid(
            "requested/effective therm surface mismatch without explicit fallback authority and reason",
        ));
    }
    if !surface_differs && policy.fallback_reason.is_some() {
        return Err(invalid(
            "fallback_reason is invalid when requested and effective therm surfaces match",
        ));
    }
    if matches!(
        profile.id.as_str(),
        "prismos.hyper-runtime-md" | "prismos.therm-postmd"
    ) && (surface_differs
        || policy.therm_surface_fallback_authorized
        || policy.requested_therm_surface != ThermSurfacePolicyV1::Lattice)
    {
        return Err(invalid(
            "strict product profile forbids therm surface fallback and requires lattice",
        ));
    }
    Ok(())
}

/// Recheck the request-to-packet policy seam immediately before native
/// delegation. This deliberately compares requested and effective values; a
/// caller cannot satisfy admission by presenting only the desired request.
pub fn validate_therm_and_post_md_packet_admission(
    profile: &ExecutionProfileIdentityV1,
    policy: &ThermAndPostMdPolicyV1,
    packet_policy: &MdEnginePolicyV1,
) -> PcResult<()> {
    validate_therm_and_post_md_policy(profile, policy)?;
    let mapped = ExecutionProfileV1 {
        schema_kind: EXECUTION_PROFILE_SCHEMA.to_string(),
        schema_version: 2,
        profile: profile.clone(),
        packet_authority: PacketAuthorityV1 {
            prismcode_packet: ArtifactIdentityV1 {
                locator: PathBuf::from("/policy-admission/packet"),
                sha256: "0".repeat(64),
            },
            qnanl_binding: ArtifactIdentityV1 {
                locator: PathBuf::from("/policy-admission/binding"),
                sha256: "0".repeat(64),
            },
        },
        topology_and_input_identity: TopologyAndInputIdentityPolicyV1 {
            target_id_required: true,
            target_request_root: PathBuf::from("/policy-admission/targets"),
            target_request_schema: TARGET_EXECUTION_INTENT_SCHEMA.to_string(),
            target_request_schema_version: 1,
            topology_locator_policy: "required_runtime_input".to_string(),
            input_identity_policy: "required_runtime_input".to_string(),
            topology_sha256_required: packet_policy.topology_sha256_required,
            input_sha256_required: packet_policy.input_sha256_required,
            unresolved_runtime_input_status: "EVIDENCE_UNAVAILABLE".to_string(),
        },
        engine_and_source_epoch_authority: EngineAndSourceEpochAuthorityV1 {
            engine_locator: PathBuf::from("/policy-admission/engine"),
            active_release_required: true,
            source_epoch_required: packet_policy.source_epoch_required,
            raw_engine_safe_defaults_unchanged: packet_policy.raw_engine_safe_defaults_unchanged,
        },
        force_and_integrator_policy: ForceAndIntegratorPolicyV1 {
            force_contract: "prismcode-authorized".to_string(),
            integrator: "canonical-hyper-runtime".to_string(),
            independent_raw_engine_flags_forbidden: packet_policy
                .independent_raw_engine_flags_forbidden,
        },
        graph_and_metal_resident_policy: GraphAndMetalResidentPolicyV1 {
            cuda_graph_required: packet_policy.cuda_graph_required,
            metal_resident_required: packet_policy.metal_resident_required,
            hotpath_acceleration_required: packet_policy.hotpath_acceleration_required,
        },
        path_b_dstw_superresolution_policy: PathBDstwSuperresolutionPolicyV1 {
            path_b_required: packet_policy.path_b_required,
            dstw_required: packet_policy.dstw_required,
            superresolution_required: packet_policy.superresolution_required,
        },
        observer_and_evidence_policy: ObserverAndEvidencePolicyV1 {
            qnanl_inline_seal_verify_required: packet_policy.qnanl_inline_seal_verify_required,
            runtime_events_required: packet_policy.runtime_events_required,
            evidence_qualification_required: packet_policy.evidence_qualification_required,
        },
        therm_and_post_md_policy: policy.clone(),
        output_and_corpus_policy: OutputAndCorpusPolicyV1 {
            run_output_root: PathBuf::from("/policy-admission/output"),
            governed_corpus_handoff_required: packet_policy.governed_corpus_handoff_required,
            output_identity_required: packet_policy.output_identity_required,
        },
    };
    let expected = profile_md_engine_policy(&mapped)?;
    if &expected != packet_policy {
        return Err(invalid(format!(
            "requested/effective therm/post-MD policy differs from packet authority: request={expected:?} packet={packet_policy:?}"
        )));
    }
    Ok(())
}

pub fn load_execution_profile(path: &Path) -> PcResult<ExecutionProfileV1> {
    let profile = read_strict_json(path, "execution profile")?;
    validate_execution_profile(&profile)?;
    Ok(profile)
}

pub fn validate_target_id(target_id: &str) -> PcResult<()> {
    if target_id.is_empty()
        || target_id.len() > 128
        || !target_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
        || target_id == "."
        || target_id == ".."
        || target_id.contains("..")
    {
        return Err(invalid("target_id contains unsafe characters"));
    }
    Ok(())
}

fn validate_coupled_execution_intent(
    topology_a: &ArtifactIdentityV1,
    coupled: &CoupledExecutionIntentV1,
) -> PcResult<()> {
    require_absolute(&coupled.topology_b.locator, "coupled.topology_b.locator")?;
    verify_artifact(&coupled.topology_b, "coupled topology B")?;

    if coupled
        .probe_residue_ids
        .windows(2)
        .any(|pair| pair[0] >= pair[1])
    {
        return Err(invalid(
            "coupled.probe_residue_ids must be unique and strictly increasing",
        ));
    }
    match coupled.contrast_kind {
        MolecularContrastKindV1::GlycanMechanisticProbe if coupled.probe_residue_ids.is_empty() => {
            return Err(invalid(
                "glycan-mechanistic-probe requires at least one coupled probe residue ID",
            ));
        }
        MolecularContrastKindV1::GlycanMechanisticProbe => {}
        _ if !coupled.probe_residue_ids.is_empty() => {
            return Err(invalid(
                "coupled probe residue IDs are valid only for glycan-mechanistic-probe",
            ));
        }
        _ => {}
    }

    let identical_kind = matches!(
        coupled.contrast_kind,
        MolecularContrastKindV1::IdenticalTopologyNegativeControl
    );
    if identical_kind != coupled.identical_topology_negative_control {
        return Err(invalid(
            "identical-topology control kind and control identity must be selected together",
        ));
    }
    let identical_hashes = topology_a.sha256 == coupled.topology_b.sha256;
    if identical_kind != identical_hashes {
        return Err(invalid(if identical_hashes {
            "same-hash coupled topologies require the explicit identical-topology negative control"
        } else {
            "identical-topology negative control requires equal topology hashes"
        }));
    }
    Ok(())
}

pub fn resolve_target_intent(
    profile: &ExecutionProfileV1,
    target_id: &str,
) -> PcResult<TargetExecutionIntentV1> {
    validate_target_id(target_id)?;
    let descriptor_path = profile
        .topology_and_input_identity
        .target_request_root
        .join(format!("{target_id}.json"));
    let descriptor: TargetExecutionIntentV1 =
        read_strict_json(&descriptor_path, "target execution intent")?;
    validate_target_intent(profile, target_id, &descriptor)?;
    Ok(descriptor)
}

fn validate_target_intent(profile: &ExecutionProfileV1, target_id: &str, descriptor: &TargetExecutionIntentV1) -> PcResult<()> {
    validate_target_id(target_id)?;
    if descriptor.schema_kind != profile.topology_and_input_identity.target_request_schema
        || descriptor.schema_version
            != profile
                .topology_and_input_identity
                .target_request_schema_version
        || descriptor.target_id != target_id
    {
        return Err(invalid("target execution intent identity mismatch"));
    }
    require_absolute(&descriptor.topology.locator, "topology.locator")?;
    require_absolute(&descriptor.input.locator, "input.locator")?;
    verify_artifact(&descriptor.topology, "topology")?;
    verify_artifact(&descriptor.input, "input")?;
    if let Some(coupled) = descriptor.coupled.as_ref() {
        validate_coupled_execution_intent(&descriptor.topology, coupled)?;
    }
    Ok(())
}

fn load_runtime_authority_from_active_pointer(
    profile: &ExecutionProfileV1,
    active_pointer: &Path,
    protection: &AuthorityProtection,
) -> PcResult<RuntimeAuthorityIdentityV1> {
    let installed_engine = &profile.engine_and_source_epoch_authority.engine_locator;
    strict_regular(installed_engine, "installed profile engine")?;
    require_protected_ancestry(installed_engine, "installed profile engine", protection)?;
    require_absolute(active_pointer, "ACTIVE release pointer")?;
    let pointer_metadata = fs::symlink_metadata(active_pointer).map_err(|error| {
        invalid(format!(
            "canonical ACTIVE release pointer unavailable at {}: {error}",
            active_pointer.display()
        ))
    })?;
    if !pointer_metadata.file_type().is_symlink() {
        return Err(invalid(
            "canonical ACTIVE release pointer must be a symlink",
        ));
    }
    #[cfg(unix)]
    {
        let expected_uid = match protection {
            AuthorityProtection::Production => 0,
            #[cfg(test)]
            AuthorityProtection::CpuFixture { uid, .. } => *uid,
        };
        if pointer_metadata.uid() != expected_uid {
            return Err(invalid("canonical ACTIVE pointer has an untrusted owner"));
        }
    }
    let pointer_target = fs::read_link(active_pointer)?;
    let mut components = pointer_target.components();
    let release_name = match (components.next(), components.next()) {
        (Some(Component::Normal(name)), None) => name
            .to_str()
            .ok_or_else(|| invalid("ACTIVE release name must be valid UTF-8"))?
            .to_string(),
        _ => {
            return Err(invalid(
                "ACTIVE release pointer must name one relative immutable release member",
            ));
        }
    };
    require_nonempty(&release_name, "ACTIVE release name")?;
    let releases_root = active_pointer
        .parent()
        .ok_or_else(|| invalid("ACTIVE release pointer has no releases root"))?;
    require_protected_ancestry(releases_root, "engine releases root", protection)?;
    reject_symlink_components(releases_root, "engine releases root")?;
    let releases_root = fs::canonicalize(releases_root)?;
    let active_release_root = fs::canonicalize(active_pointer).map_err(|error| {
        invalid(format!(
            "ACTIVE release target unavailable at {}: {error}",
            active_pointer.display()
        ))
    })?;
    if active_release_root.parent() != Some(releases_root.as_path())
        || active_release_root
            .file_name()
            .and_then(|name| name.to_str())
            != Some(release_name.as_str())
        || !active_release_root.is_dir()
    {
        return Err(invalid(
            "ACTIVE release pointer escapes the canonical immutable releases root",
        ));
    }
    require_immutable_mode(&active_release_root, "ACTIVE release root")?;
    let engine = active_release_root.join("bin/nhs_rt_full");
    let active_lock_path = active_release_root.join("ledger/PRISMOS_ACTIVE_STATE_LOCK.json");
    strict_regular(&engine, "ACTIVE release engine")?;
    strict_regular(&active_lock_path, "ACTIVE release lock")?;
    require_immutable_mode(&engine, "ACTIVE release engine")?;
    require_immutable_mode(&active_lock_path, "ACTIVE release lock")?;
    for (member, label) in [
        (&engine, "ACTIVE release engine"),
        (&active_lock_path, "ACTIVE release lock"),
    ] {
        require_protected_ancestry(member, label, protection)?;
        let canonical = fs::canonicalize(member)?;
        if !canonical.starts_with(&active_release_root) {
            return Err(invalid(format!(
                "{label} is not an immutable ACTIVE release member"
            )));
        }
    }
    let active_lock_bytes = fs::read(&active_lock_path)?;
    let lock: ActiveStateLockV1 = serde_json::from_slice(&active_lock_bytes)?;
    let active_lock_sha256 = format!("{:x}", Sha256::digest(&active_lock_bytes));
    if lock.schema_kind != "prismos_active_state_lock" || lock.schema_version != 3 {
        return Err(invalid("unsupported active-state-lock schema"));
    }
    require_nonempty(&lock.active_release, "active_release")?;
    if lock.active_release != release_name {
        return Err(invalid(format!(
            "ACTIVE release lock names the wrong release: pointer={release_name} lock={}",
            lock.active_release
        )));
    }
    require_sha256(&lock.source_epoch_id, "source_epoch_id")?;
    require_sha256(&lock.engine_sha256, "engine_sha256")?;
    let observed_engine = sha256_file_hex(&engine)?;
    let observed_installed_engine = sha256_file_hex(installed_engine)?;
    if observed_engine != lock.engine_sha256 || observed_installed_engine != observed_engine {
        return Err(invalid(format!(
            "profile/ACTIVE engine does not match active authority: declared={} active={observed_engine} installed={observed_installed_engine}",
            lock.engine_sha256,
        )));
    }
    if fs::read_link(active_pointer)? != pointer_target
        || fs::read(&active_lock_path)? != active_lock_bytes
    {
        return Err(invalid("ACTIVE authority changed while materializing the request"));
    }
    Ok(RuntimeAuthorityIdentityV1 {
        source_epoch_id: lock.source_epoch_id,
        active_release: release_name,
        active_lock: ArtifactIdentityV1 {
            locator: active_lock_path,
            sha256: active_lock_sha256,
        },
        engine: ArtifactIdentityV1 {
            locator: engine,
            sha256: observed_engine,
        },
    })
}

pub fn canonical_request_identity_json(
    request: &HyperRuntimeRequestIdentityV1,
) -> PcResult<String> {
    if request.native_engine_intent.is_none() {
        return serde_json::to_string(request).map_err(PcError::from);
    }
    // Raw token order is transport. The actual native parser's normalized
    // values are authority; token-to-value equivalence is rechecked at launch.
    let mut value = serde_json::to_value(request)?;
    if let Some(intent) = value.get_mut("native_engine_intent").and_then(serde_json::Value::as_object_mut) {
        intent.remove("argv");
    }
    serde_json::to_string(&value).map_err(PcError::from)
}

pub fn canonical_request_identity_sha256(
    request: &HyperRuntimeRequestIdentityV1,
) -> PcResult<String> {
    let canonical = canonical_request_identity_json(request)?;
    Ok(format!("{:x}", Sha256::digest(canonical.as_bytes())))
}

fn materialize_hyper_runtime_request_with_profile_identity(
    profile_path: &Path,
    target_id: &str,
    output: &Path,
    client_surface: &str,
    expected_profile_sha256: &str,
    active_pointer: &Path,
    protection: &AuthorityProtection,
) -> PcResult<HyperRuntimeRequestV1> {
    materialize_hyper_runtime_request_with_target(profile_path, target_id, output, client_surface,
        expected_profile_sha256, active_pointer, protection, None)
}

fn materialize_hyper_runtime_request_with_target(
    profile_path: &Path,
    target_id: &str,
    output: &Path,
    client_surface: &str,
    expected_profile_sha256: &str,
    active_pointer: &Path,
    protection: &AuthorityProtection,
    inline_target: Option<TargetExecutionIntentV1>,
) -> PcResult<HyperRuntimeRequestV1> {
    require_nonempty(client_surface, "client_surface")?;
    let profile = load_execution_profile(profile_path)?;
    let profile_sha256 = sha256_file_hex(profile_path)?;
    require_sha256(expected_profile_sha256, "expected_execution_profile_sha256")?;
    if profile_sha256 != expected_profile_sha256 {
        return Err(invalid(format!(
            "execution profile does not match compiled canonical authority: expected={expected_profile_sha256} observed={profile_sha256}"
        )));
    }
    let target = match inline_target {
        Some(target) => { validate_target_intent(&profile, target_id, &target)?; target }
        None => resolve_target_intent(&profile, target_id)?,
    };
    verify_artifact(
        &profile.packet_authority.prismcode_packet,
        "PrismCode packet",
    )?;
    let validated_packet_policy =
        load_validated_md_engine_policy(&profile.packet_authority.prismcode_packet.locator)?;
    if validated_packet_policy.packet_sha256 != profile.packet_authority.prismcode_packet.sha256 {
        return Err(invalid(
            "decoded packet policy identity differs from profile packet authority",
        ));
    }
    let configured_policy = profile_md_engine_policy(&profile)?;
    if validated_packet_policy.policy != configured_policy {
        return Err(invalid(format!(
            "decoded PrismCode packet policy differs from execution-profile intent: packet={:?} profile={:?}",
            validated_packet_policy.policy, configured_policy
        )));
    }
    verify_artifact(&profile.packet_authority.qnanl_binding, "QNANL binding")?;
    require_absolute(output, "output")?;
    reject_lexical_dot_components(
        &profile.output_and_corpus_policy.run_output_root,
        "run_output_root",
    )?;
    reject_lexical_dot_components(output, "output")?;
    reject_symlink_components(output, "output")?;
    let expected_output = profile
        .output_and_corpus_policy
        .run_output_root
        .join(target_id);
    if output != expected_output {
        return Err(invalid(format!(
            "output must equal the governed target run path: expected={} actual={}",
            expected_output.display(),
            output.display()
        )));
    }
    let runtime_authority = load_runtime_authority_from_active_pointer(&profile, active_pointer, protection)?;
    let request = HyperRuntimeRequestIdentityV1 {
        experiment: None,
        execution_profile: profile.profile,
        execution_profile_sha256: profile_sha256,
        packet_authority: profile.packet_authority,
        validated_packet_policy,
        governed_branch: None,
        target_id: target.target_id,
        topology: target.topology,
        input: target.input,
        coupled: target.coupled,
        native_engine_intent: None,
        runtime_authority,
        force_and_integrator_policy: profile.force_and_integrator_policy,
        graph_and_metal_resident_policy: profile.graph_and_metal_resident_policy,
        path_b_dstw_superresolution_policy: profile.path_b_dstw_superresolution_policy,
        observer_and_evidence_policy: profile.observer_and_evidence_policy,
        therm_and_post_md_policy: profile.therm_and_post_md_policy,
        output_and_corpus_policy: profile.output_and_corpus_policy,
        output: output.to_path_buf(),
    };
    let request_id_sha256 = canonical_request_identity_sha256(&request)?;
    Ok(HyperRuntimeRequestV1 {
        schema_kind: HYPER_RUNTIME_REQUEST_SCHEMA.to_string(),
        schema_version: 2,
        request_id_sha256,
        request,
        attribution: RequestAttributionV1 {
            client_surface: client_surface.to_string(),
        },
    })
}

pub fn materialize_hyper_runtime_request(
    profile_path: &Path,
    target_id: &str,
    output: &Path,
    client_surface: &str,
) -> PcResult<HyperRuntimeRequestV1> {
    materialize_hyper_runtime_request_with_profile_identity(
        profile_path,
        target_id,
        output,
        client_surface,
        CANONICAL_EXECUTION_PROFILE_SHA256,
        Path::new(CANONICAL_ACTIVE_RELEASE_POINTER),
        &AuthorityProtection::Production,
    )
}

pub fn canonical_client_runtime_authority(profile_path: &Path) -> PcResult<RuntimeAuthorityIdentityV1> {
    let profile = load_execution_profile(profile_path)?;
    if sha256_file_hex(profile_path)? != CANONICAL_EXECUTION_PROFILE_SHA256 {
        return Err(invalid("client intent profile differs from compiled canonical authority"));
    }
    load_runtime_authority_from_active_pointer(&profile, Path::new(CANONICAL_ACTIVE_RELEASE_POINTER), &AuthorityProtection::Production)
}

pub fn materialize_client_hyper_runtime_request(
    intent: &ClientExecutionIntentV1,
    target: TargetExecutionIntentV1,
    native: NativeEngineIntentV1,
    client_surface: &str,
) -> PcResult<HyperRuntimeRequestV1> {
    if intent.schema_kind != CLIENT_INTENT_SCHEMA || intent.schema_version != 1 {
        return Err(invalid("unsupported client execution intent schema"));
    }
    if let Some(replicate) = intent.replicate.as_ref() {
        validate_target_id(&replicate.group_id)?;
        if replicate.count == 0 || replicate.index >= replicate.count {
            return Err(invalid("replicate index must be zero-based and less than its positive count"));
        }
        if native.normalized_arguments.get("replica_seed").and_then(|values| values.first())
            != Some(&replicate.seed.to_string())
        {
            return Err(invalid("replicate seed differs from native parsed replica_seed"));
        }
    }
    let mut request = materialize_hyper_runtime_request_with_target(&intent.execution_profile,
        &intent.target_id, &intent.output, client_surface, CANONICAL_EXECUTION_PROFILE_SHA256,
        Path::new(CANONICAL_ACTIVE_RELEASE_POINTER), &AuthorityProtection::Production, Some(target))?;
    request.request.experiment = intent.experiment.clone();
    request.request.native_engine_intent = Some(native);
    request.schema_version = if intent.experiment.is_some() { 4 } else { 3 };
    request.request_id_sha256 = canonical_request_identity_sha256(&request.request)?;
    if let Some(policy) = &intent.experiment {
        crate::durable_experiment::validate_request_policy(&request, policy)?;
    }
    Ok(request)
}

fn observed_artifact(path: &Path, label: &str) -> PcResult<ArtifactIdentityV1> {
    require_absolute(path, label)?;
    strict_regular(path, label)?;
    reject_lexical_dot_components(path, label)?;
    reject_symlink_components(path, label)?;
    Ok(ArtifactIdentityV1 { locator: path.to_path_buf(), sha256: sha256_file_hex(path)? })
}

fn bind_client_artifact_transport(transport: Option<&ClientArtifactTransportV1>) -> PcResult<Vec<ArtifactIdentityV1>> {
    let Some(transport) = transport else { return Ok(Vec::new()); };
    let mut bindings = Vec::new();
    for (path, label) in transport.prism_bundle_root.iter().map(|path| (path, "client bundle root"))
        .chain(transport.science_roots.iter().flatten().map(|path| (path, "client science root"))) {
        require_absolute(path, label)?;
        reject_lexical_dot_components(path, label)?;
        reject_symlink_components(path, label)?;
        if !fs::metadata(path)?.is_dir() {
            return Err(invalid(format!("{label} must be a directory")));
        }
    }
    if let Some(path) = &transport.qnanl_force_kernel_packet {
        bindings.push(observed_artifact(path, "client QNANL force-kernel packet")?);
    }
    if let Some(root) = &transport.prism_bundle_root {
        for name in ["release/prismcode_cuda_manifest.json", "release/cuda_manifest.sig"] {
            let path = root.join(name);
            if path.exists() {
                bindings.push(observed_artifact(&path, "client CUDA bundle trust artifact")?);
            }
        }
    }
    for path in transport.qnanl_force_kernel_receipt.iter().chain(transport.prismcode_cuda_opcode_coverage.iter()) {
        require_absolute(path, "client receipt output")?;
        reject_lexical_dot_components(path, "client receipt output")?;
        reject_symlink_components(path, "client receipt output")?;
    }
    Ok(bindings)
}

pub fn verify_native_artifact_transport(native: &NativeEngineIntentV1) -> PcResult<()> {
    if bind_client_artifact_transport(native.artifact_transport.as_ref())? != native.transport_input_bindings {
        return Err(invalid("native client transport input bytes changed after request identity was bound"));
    }
    Ok(())
}

fn native_coupled_intent(native: &NativeEngineIntentV1) -> PcResult<Option<CoupledExecutionIntentV1>> {
    match native_intent_value(native, "engine_b_topology") {
        Some(path) => Ok(Some(CoupledExecutionIntentV1 {
            topology_b: observed_artifact(Path::new(path), "client coupled topology")?,
            contrast_kind: serde_json::from_value(serde_json::Value::String(native_intent_value(native, "contrast_kind")
                .ok_or_else(|| invalid("coupled client intent requires contrast_kind"))?.to_string()))?,
            identical_topology_negative_control: native_intent_value(native, "identical_topology_negative_control") == Some("true"),
            probe_residue_ids: native.normalized_arguments.get("coupled_probe_residue_ids").into_iter().flatten()
                .map(|value| value.parse::<usize>().map_err(|_| invalid("invalid native probe residue ID"))).collect::<PcResult<Vec<_>>>()?,
        })),
        None => Ok(None),
    }
}

fn profile_surface_defaults(parsed: &NativeRunIntentValidationV1) -> Vec<String> {
    let value = |key: &str| parsed.normalized_arguments.get(key).and_then(|values| values.first()).map(String::as_str);
    if value("engine_b_topology").is_some() {
        return if value("coupled_twin") == Some("true") { Vec::new() } else { vec!["--coupled-twin".into()] };
    }
    if value("coupled_twin") == Some("true") {
        return Vec::new();
    }
    let explicit = |key: &str| parsed.explicit_arguments.iter().any(|argument| argument == key);
    let mut defaults = Vec::new();
    if !explicit("engine_surface") && !explicit("clustering_backend")
        && !explicit("mar_v2_telemetry") && !explicit("m1_monolithic_discovery") {
        defaults.extend(["--engine-surface".into(), "ghost-tiles".into()]);
    }
    // Evidence selection cannot choose a different physical protocol or stream
    // population. Explicit differential/named-surface expansion belongs to NHS.
    defaults
}

fn validate_client_lattice_producer(native: &NativeEngineIntentV1) -> PcResult<()> {
    if native_intent_value(native, "engine_b_topology").is_some() {
        return Ok(());
    }
    let layers: Vec<_> = native_intent_value(native, "engine_surface").unwrap_or("none").split('+').collect();
    let ghost = layers.contains(&"ghost-tiles") || (
        matches!(native_intent_value(native, "clustering_backend"), Some("ghost-phase-lattice-4d" | "ghost-lattice" | "lattice-4d"))
        && native_intent_value(native, "mar_v2_telemetry") == Some("true")
        && (native_intent_value(native, "m1_monolithic_discovery") == Some("true")
            || native_intent_value(native, "path_a_v2_trigger_steps").is_some()
            || native_intent_value(native, "save_trajectory_interval").and_then(|value| value.parse::<u32>().ok()).is_some_and(|interval| interval > 0)));
    if !ghost || native_intent_value(native, "coupled_twin") == Some("true") {
        return Err(invalid("selected lattice profile requires a native Ghost/V2-producing Standard or differential lane, or explicit topology-B coupled intent; explicit scientific protocol controls were preserved and are incompatible"));
    }
    Ok(())
}

pub fn prepare_client_hyper_runtime_request(path: &Path, client_surface: &str) -> PcResult<HyperRuntimeRequestV1> {
    let intent: ClientExecutionIntentV1 = read_strict_json(path, "client execution intent")?;
    if intent.schema_kind != CLIENT_INTENT_SCHEMA || intent.schema_version != 1 {
        return Err(invalid("unsupported client execution intent schema"));
    }
    validate_target_id(&intent.target_id)?;
    let profile = load_execution_profile(&intent.execution_profile)?;
    let authority = canonical_client_runtime_authority(&intent.execution_profile)?;
    let topology = observed_artifact(&intent.topology, "client topology")?;
    let input = observed_artifact(&intent.input, "client input")?;
    let mut argv = intent.engine_argv.clone();
    for token in &argv {
        let head = token.split('=').next().unwrap_or(token);
        if matches!(head, "-t" | "--topology" | "-o" | "--output" | "--canonical-input" | "--help" | "-h" | "--version" | "-V")
            || head.starts_with("--expected-") || head.starts_with("--successor-")
        {
            return Err(invalid(format!("client engine token is owned by native request authority: {token}")));
        }
    }
    argv.extend(["--topology".into(), intent.topology.display().to_string(), "--output".into(), intent.output.display().to_string()]);
    let mut ensure = |flag: &str, value: Option<String>| {
        if !argv.iter().any(|token| token.split('=').next() == Some(flag)) {
            argv.push(flag.to_string());
            if let Some(value) = value { argv.push(value); }
        }
    };
    ensure("--prismcode-packet", Some(profile.packet_authority.prismcode_packet.locator.display().to_string()));
    ensure("--prismcode-backend", Some("cuda".into()));
    ensure("--prismcode-validation-level", Some("L3".into()));
    ensure("--post-md-aggregation", Some("full".into()));
    ensure("--prism-therm-site-surface", Some("lattice".into()));
    ensure("--min-hotpath-launch-collapse", Some("64".into()));
    for flag in ["--require-prismcode-substrate", "--require-prismcode-md-native", "--require-cuda-graph-native",
        "--metal-resident-md", "--require-metal-resident-md", "--require-hotpath-acceleration", "--qnanl-primary-artifacts", "--prism-therm"] {
        ensure(flag, None);
    }
    if let Some(replicate) = intent.replicate.as_ref() {
        ensure("--replica-seed", Some(replicate.seed.to_string()));
    }
    let mut parsed = validate_native_engine_intent(&authority.engine, &argv)?;
    let defaults = profile_surface_defaults(&parsed);
    if !defaults.is_empty() {
        argv.extend(defaults);
        parsed = validate_native_engine_intent(&authority.engine, &argv)?;
    }
    let native = NativeEngineIntentV1 { argv, normalized_arguments: parsed.normalized_arguments, replicate: intent.replicate.clone(),
        artifact_transport: intent.artifact_transport.clone(),
        transport_input_bindings: bind_client_artifact_transport(intent.artifact_transport.as_ref())? };
    for (field, expected) in [("prismcode_backend", "cuda"), ("post_md_aggregation", "full"),
        ("prism_therm_site_surface", "lattice"), ("require_prismcode_substrate", "true"),
        ("require_prismcode_md_native", "true"), ("require_cuda_graph_native", "true"),
        ("metal_resident_md", "true"), ("require_metal_resident_md", "true"),
        ("require_hotpath_acceleration", "true"), ("qnanl_primary_artifacts", "true"), ("prism_therm", "true")] {
        if native_intent_value(&native, field) != Some(expected) {
            return Err(invalid(format!("native client intent conflicts with the selected profile: {field} requires {expected}")));
        }
    }
    if native_intent_value(&native, "prismcode_packet") != profile.packet_authority.prismcode_packet.locator.to_str()
        || native_intent_value(&native, "non_production_legacy_nonbonded") == Some("true")
    {
        return Err(invalid("native client packet or force contract differs from selected profile authority"));
    }
    if native_intent_value(&native, "manifest").is_some() {
        return Err(invalid("single-target native request cannot bind an independent batch manifest; each topology requires its own exact request identity"));
    }
    validate_client_lattice_producer(&native)?;
    let coupled = native_coupled_intent(&native)?;
    let target = TargetExecutionIntentV1 { schema_kind: TARGET_EXECUTION_INTENT_SCHEMA.into(), schema_version: 1,
        target_id: intent.target_id.clone(), topology, input, coupled };
    let request = materialize_client_hyper_runtime_request(&intent, target, native, client_surface)?;
    if request.request.runtime_authority != authority {
        return Err(invalid("ACTIVE authority changed during native client-intent validation"));
    }
    Ok(request)
}

fn observed_sized_artifact(path: &Path, field: &str) -> PcResult<SizedArtifactIdentityV1> {
    strict_regular(path, field)?;
    Ok(SizedArtifactIdentityV1 {
        locator: path.to_path_buf(),
        sha256: sha256_file_hex(path)?,
        size_bytes: fs::metadata(path)?.len(),
    })
}

fn require_exact_argv(observed: &[String], expected: Vec<String>, field: &str) -> PcResult<()> {
    if observed != expected {
        return Err(invalid(format!("{field} command_argv mismatch")));
    }
    Ok(())
}

fn validate_governed_proposal(
    selection: &GovernedBranchSelectionV1,
    explicit: &ExplicitBranchSelectionV1,
) -> PcResult<()> {
    let proposal = &selection.proposal;
    require_sha256(&proposal.proposal_id_sha256, "proposal_id_sha256")?;
    if proposal.proposal_id_sha256 != proposal.row_sha256
        || explicit.proposal.proposal_id_sha256 != proposal.proposal_id_sha256
        || explicit.proposal.row_sha256 != proposal.row_sha256
        || explicit.proposal.proposal_set_receipt != proposal.proposal_set_receipt
        || explicit.proposal.proposal_artifact != proposal.proposal_artifact
    {
        return Err(invalid("explicit selection/proposal identity mismatch"));
    }
    for (identity, field) in [
        (&proposal.proposal_set_receipt, "proposal_set_receipt"),
        (&proposal.proposal_artifact, "proposal_artifact"),
        (&proposal.selected_proposal, "selected_proposal"),
    ] {
        verify_sized_artifact(identity, field)?;
    }

    let selected: SelectedProposalV1 =
        read_strict_json(&proposal.selected_proposal.locator, "selected proposal")?;
    if selected.schema_kind != "prismos.cognition.selected_gflownet_proposal"
        || selected.schema_version != 1
        || selected.proposal_id_sha256 != proposal.proposal_id_sha256
        || selected.row_sha256 != proposal.row_sha256
        || selected.proposal_set_artifact != proposal.proposal_artifact
        || selected.proposal_set_receipt != proposal.proposal_set_receipt
        || selected.source_epoch_id != selection.source_epoch_id
    {
        return Err(invalid("selected proposal artifact identity mismatch"));
    }
    let canonical_row: serde_json::Value = serde_json::from_str(&selected.canonical_row_utf8)
        .map_err(|error| {
            invalid(format!(
                "selected proposal canonical row malformed: {error}"
            ))
        })?;
    if canonical_row != selected.row
        || format!(
            "{:x}",
            Sha256::digest(selected.canonical_row_utf8.as_bytes())
        ) != proposal.proposal_id_sha256
    {
        return Err(invalid("selected proposal row canonical identity mismatch"));
    }
    if proposal.non_authoritative_rank != explicit.proposal.non_authoritative_rank
        || proposal.non_authoritative_rank.is_some_and(|rank| {
            selected.row.get("rank").and_then(serde_json::Value::as_u64) != Some(rank)
        })
    {
        return Err(invalid(
            "non-authoritative rank differs from the hash-selected proposal row",
        ));
    }

    let seal: SealedProposalSetReceiptV1 =
        read_strict_json(&proposal.proposal_set_receipt.locator, "proposal-set seal")?;
    if seal.schema_kind != "prismos.cognition.sealed_gflownet_proposal_set"
        || seal.schema_version != 1
        || seal.seal_status != "SEALED"
        || seal.source_epoch_id != selection.source_epoch_id
        || seal.proposal_set != proposal.proposal_artifact
        || seal.proposal_id_algorithm != "sha256-canonical-json-v1"
        || seal.proposal_count == 0
        || seal.proposal_count as usize != seal.proposal_ids.len()
    {
        return Err(invalid(
            "proposal-set seal is missing or does not bind selection",
        ));
    }
    verify_sized_artifact(&seal.sample_execution_receipt, "sample execution receipt")?;
    let mut sorted = seal.proposal_ids.clone();
    for proposal_id in &sorted {
        require_sha256(proposal_id, "sealed proposal ID")?;
    }
    sorted.sort();
    sorted.dedup();
    if sorted != seal.proposal_ids || !sorted.contains(&proposal.proposal_id_sha256) {
        return Err(invalid(
            "proposal-set seal IDs must be sorted, unique, and contain the explicit proposal",
        ));
    }
    if canonical_json_sha256(&seal.proposal_ids)? != seal.proposal_ids_sha256 {
        return Err(invalid("proposal-set membership digest mismatch"));
    }
    let digest = SealedProposalSetDigestV1 {
        schema_kind: &seal.schema_kind,
        schema_version: seal.schema_version,
        seal_status: &seal.seal_status,
        source_epoch_id: &seal.source_epoch_id,
        proposal_set: &seal.proposal_set,
        sample_execution_receipt: &seal.sample_execution_receipt,
        proposal_count: seal.proposal_count,
        proposal_id_algorithm: &seal.proposal_id_algorithm,
        proposal_ids: &seal.proposal_ids,
        proposal_ids_sha256: &seal.proposal_ids_sha256,
    };
    if canonical_json_sha256(&digest)? != seal.seal_sha256 {
        return Err(invalid("proposal-set seal digest mismatch"));
    }
    Ok(())
}

fn validate_branch_packet_receipts(
    selection: &GovernedBranchSelectionV1,
    validated_packet: &ValidatedMdEnginePolicyV1,
) -> PcResult<()> {
    let packet = &selection.packet;
    let normalization: PacketNormalizationReceiptV1 = read_strict_json(
        &packet.normalization_receipt.locator,
        "packet normalization receipt",
    )?;
    verify_sized_artifact(
        &packet.normalization_receipt,
        "packet normalization receipt",
    )?;
    if normalization.schema_kind != "prismos.cognition.prismcode_policy_normalization_receipt"
        || normalization.schema_version != 1
        || normalization.status != "PASS"
        || normalization.source_epoch_id != selection.source_epoch_id
        || normalization.branch_id_sha256 != selection.branch_id_sha256
        || packet_sized_artifact(&normalization.output_packet)
            != (SizedArtifactIdentityV1 {
                locator: packet.locator.clone(),
                sha256: packet.sha256.clone(),
                size_bytes: validated_packet.packet_bytes,
            })
        || normalization.output_packet.internal_blake3 != packet.internal_blake3
    {
        return Err(invalid(
            "packet normalization receipt does not bind branch packet",
        ));
    }
    verify_sized_artifact(&normalization.input_packet, "normalization input packet")?;
    verify_sized_artifact(
        &normalization.native_receipt,
        "native normalization receipt",
    )?;
    require_exact_argv(
        &normalization.command_argv,
        vec![
            "prismcode-packet-policy".into(),
            "--input".into(),
            normalization.input_packet.locator.display().to_string(),
            "--output".into(),
            packet.locator.display().to_string(),
            "--report".into(),
            normalization.native_receipt.locator.display().to_string(),
        ],
        "normalization receipt",
    )?;

    let validation: PacketValidationReceiptV1 = read_strict_json(
        &packet.validation_receipt.locator,
        "packet validation receipt",
    )?;
    verify_sized_artifact(&packet.validation_receipt, "packet validation receipt")?;
    if validation.schema_kind != "prismos.cognition.prismcode_packet_validation_receipt"
        || validation.schema_version != 1
        || validation.status != "PASS"
        || validation.validation_level != "L3"
        || validation.source_epoch_id != selection.source_epoch_id
        || validation.branch_id_sha256 != selection.branch_id_sha256
        || packet_sized_artifact(&validation.packet)
            != (SizedArtifactIdentityV1 {
                locator: packet.locator.clone(),
                sha256: packet.sha256.clone(),
                size_bytes: validated_packet.packet_bytes,
            })
        || validation.packet.internal_blake3 != packet.internal_blake3
        || validation.native_validation.status != "PASS"
        || validation.native_validation.level != "L3"
        || validation.native_validation.packet_hash != packet.internal_blake3
    {
        return Err(invalid(
            "packet validation receipt does not bind L3 branch packet",
        ));
    }
    verify_sized_artifact(
        &validation.native_validation_report,
        "native packet validation report",
    )?;
    let native_report: serde_json::Value = read_strict_json(
        &validation.native_validation_report.locator,
        "native packet validation report",
    )?;
    if native_report
        .get("status")
        .and_then(serde_json::Value::as_str)
        != Some("PASS")
        || native_report
            .get("level")
            .and_then(serde_json::Value::as_str)
            != Some("L3")
        || native_report
            .get("packet_hash")
            .and_then(serde_json::Value::as_str)
            != Some(packet.internal_blake3.as_str())
    {
        return Err(invalid("native validation report witness mismatch"));
    }
    require_exact_argv(
        &validation.command_argv,
        vec![
            "prismval".into(),
            "validate".into(),
            packet.locator.display().to_string(),
            "--level".into(),
            "L3".into(),
            "--out".into(),
            validation
                .native_validation_report
                .locator
                .display()
                .to_string(),
        ],
        "validation receipt",
    )?;
    Ok(())
}

/// Materialize a governed next-branch request from explicit cognition
/// selection authority. Validation and authorization complete here; engine
/// execution and evidence qualification remain separate downstream phases.
fn materialize_governed_branch_hyper_runtime_request_with_profile_identity(
    selection_path: &Path,
    execution_request_path: &Path,
    expected_profile_sha256: &str,
    active_pointer: &Path,
    protection: &AuthorityProtection,
) -> PcResult<HyperRuntimeRequestV1> {
    let selection_artifact = observed_sized_artifact(selection_path, "governed branch selection")?;
    let selection: GovernedBranchSelectionV1 =
        read_strict_json(selection_path, "governed branch selection")?;
    let execution: NextBranchExecutionRequestV1 =
        read_strict_json(execution_request_path, "next-branch execution request")?;
    if selection.schema_kind != GOVERNED_BRANCH_SELECTION_SCHEMA
        || selection.schema_version != 1
        || execution.schema_kind != NEXT_BRANCH_EXECUTION_REQUEST_SCHEMA
        || execution.schema_version != 1
        || selection.selection_mode != "explicit-governed-selection"
        || !selection.selected
        || execution.client_surface != "agent"
        || execution.governed_branch_selection != selection_artifact
    {
        return Err(invalid(
            "unsupported or non-explicit governed branch request",
        ));
    }
    if execution.branch_id_sha256 != selection.branch_id_sha256
        || execution.source_epoch_id != selection.source_epoch_id
        || execution.execution_profile != selection.execution_profile
        || execution.target != selection.target
        || execution.packet != selection.packet
        || execution.policy_snapshot != selection.policy_snapshot
        || execution.evidence_snapshot != selection.evidence_snapshot
        || execution.output_snapshot != selection.output_snapshot
        || execution.output != selection.output
    {
        return Err(invalid(
            "execution request differs from governed branch selection",
        ));
    }
    require_sha256(&selection.branch_id_sha256, "branch_id_sha256")?;
    require_sha256(&selection.source_epoch_id, "source_epoch_id")?;
    if selection.selection_authority.authority_kind != "operator-explicit-artifact" {
        return Err(invalid(
            "rank-only or inferred branch selection is forbidden",
        ));
    }
    require_nonempty(
        &selection.selection_authority.authority_id,
        "selection authority ID",
    )?;
    require_nonempty(
        &selection.selection_authority.directive_id,
        "selection directive ID",
    )?;
    verify_sized_artifact(&selection.explicit_selection, "explicit selection")?;
    let explicit: ExplicitBranchSelectionV1 =
        read_strict_json(&selection.explicit_selection.locator, "explicit selection")?;
    if explicit.schema_kind != "prismos.cognition.explicit_branch_selection"
        || explicit.schema_version != 1
        || explicit.selection_mode != selection.selection_mode
        || !explicit.selected
        || explicit.selection_authority != selection.selection_authority
        || explicit.source_epoch_id != selection.source_epoch_id
        || explicit.execution_profile != selection.execution_profile
        || explicit.target != selection.target
        || explicit.output != selection.output
    {
        return Err(invalid(
            "explicit selection artifact does not authorize branch",
        ));
    }
    let expected_branch_id = canonical_json_sha256(&BranchIdentityV1 {
        schema_kind: "prismos.cognition.next_branch_identity",
        schema_version: 1,
        explicit_selection_sha256: &selection.explicit_selection.sha256,
        proposal_id_sha256: &selection.proposal.proposal_id_sha256,
        source_epoch_id: &selection.source_epoch_id,
        execution_profile: &selection.execution_profile,
        target: &selection.target,
        output: &execution.output,
    })?;
    if expected_branch_id != selection.branch_id_sha256 {
        return Err(invalid("branch identity digest mismatch"));
    }
    validate_governed_proposal(&selection, &explicit)?;

    if selection.execution_profile.id != "prismos.hyper-runtime-md"
        || selection.execution_profile.version != "1.0.0"
        || selection.execution_profile.sha256 != expected_profile_sha256
    {
        return Err(invalid(
            "unsupported governed execution-profile identity/version",
        ));
    }
    verify_artifact(
        &ArtifactIdentityV1 {
            locator: selection.execution_profile.locator.clone(),
            sha256: selection.execution_profile.sha256.clone(),
        },
        "governed execution profile",
    )?;
    let profile = load_execution_profile(&selection.execution_profile.locator)?;
    if profile.profile.id != selection.execution_profile.id
        || profile.profile.version != selection.execution_profile.version
    {
        return Err(invalid("governed execution-profile content mismatch"));
    }
    let target = resolve_target_intent(&profile, &selection.target.target_id)?;
    let expected_descriptor = profile
        .topology_and_input_identity
        .target_request_root
        .join(format!("{}.json", selection.target.target_id));
    verify_artifact(&selection.target.descriptor, "target descriptor")?;
    if selection.target.descriptor.locator != expected_descriptor
        || selection.target.topology != target.topology
        || selection.target.input != target.input
        || target.coupled.is_some()
    {
        return Err(invalid("governed target identity mismatch"));
    }
    let expected_policy = GovernedPolicySnapshotV1 {
        force_and_integrator_policy: profile.force_and_integrator_policy.clone(),
        graph_and_metal_resident_policy: profile.graph_and_metal_resident_policy.clone(),
        path_b_dstw_superresolution_policy: profile.path_b_dstw_superresolution_policy.clone(),
        therm_and_post_md_policy: profile.therm_and_post_md_policy.clone(),
    };
    if selection.policy_snapshot != expected_policy
        || selection.evidence_snapshot != profile.observer_and_evidence_policy
        || selection.output_snapshot != profile.output_and_corpus_policy
    {
        return Err(invalid(
            "branch policy/evidence/output snapshot differs from profile",
        ));
    }
    require_absolute(&execution.output, "branch output")?;
    reject_lexical_dot_components(&execution.output, "branch output")?;
    reject_symlink_components(&execution.output, "branch output")?;
    let expected_output = profile
        .output_and_corpus_policy
        .run_output_root
        .join(&selection.target.target_id);
    if execution.output != expected_output {
        return Err(invalid(
            "branch output differs from governed target run path",
        ));
    }

    if selection.packet.schema_kind != "prismos.cognition.next_branch_prismcode_qnanl_packet"
        || selection.packet.schema_version != 1
        || selection.packet.validation_level != "L3"
        || selection.packet.sha256 == profile.packet_authority.prismcode_packet.sha256
    {
        return Err(invalid(
            "next-branch packet must be a distinct compiled L3 PrismCode/QNANL packet",
        ));
    }
    verify_artifact(
        &ArtifactIdentityV1 {
            locator: selection.packet.locator.clone(),
            sha256: selection.packet.sha256.clone(),
        },
        "next-branch packet",
    )?;
    let validated_packet = load_validated_md_engine_policy(&selection.packet.locator)?;
    require_sha256(&selection.packet.internal_blake3, "packet.internal_blake3")?;
    if validated_packet.packet_sha256 != selection.packet.sha256
        || validated_packet.packet_hash_blake3 != selection.packet.internal_blake3
        || validated_packet.policy != profile_md_engine_policy(&profile)?
    {
        return Err(invalid("next-branch packet identity/policy mismatch"));
    }
    validate_branch_packet_receipts(&selection, &validated_packet)?;
    verify_sized_artifact(&selection.branch_source, "branch source")?;

    let runtime_authority = load_runtime_authority_from_active_pointer(&profile, active_pointer, protection)?;
    if runtime_authority.source_epoch_id != selection.source_epoch_id {
        return Err(invalid(
            "branch SOURCE_EPOCH_ID differs from active runtime authority",
        ));
    }
    let request = HyperRuntimeRequestIdentityV1 {
        experiment: None,
        execution_profile: profile.profile,
        execution_profile_sha256: selection.execution_profile.sha256,
        packet_authority: PacketAuthorityV1 {
            prismcode_packet: ArtifactIdentityV1 {
                locator: selection.packet.locator,
                sha256: selection.packet.sha256,
            },
            // The branch validation receipt replaces the fixed packet's build
            // binding and is retained in governed_branch with full size identity.
            qnanl_binding: sized_artifact(&selection.packet.validation_receipt),
        },
        validated_packet_policy: validated_packet,
        governed_branch: Some(GovernedBranchAuthorityV1 {
            branch_id_sha256: selection.branch_id_sha256,
            governed_branch_selection: selection_artifact,
            explicit_selection: selection.explicit_selection,
            proposal_id_sha256: selection.proposal.proposal_id_sha256,
            selected_proposal: selection.proposal.selected_proposal,
            normalization_receipt: selection.packet.normalization_receipt,
            validation_receipt: selection.packet.validation_receipt,
        }),
        target_id: target.target_id,
        topology: target.topology,
        input: target.input,
        coupled: None,
        native_engine_intent: None,
        runtime_authority,
        force_and_integrator_policy: profile.force_and_integrator_policy,
        graph_and_metal_resident_policy: profile.graph_and_metal_resident_policy,
        path_b_dstw_superresolution_policy: profile.path_b_dstw_superresolution_policy,
        observer_and_evidence_policy: profile.observer_and_evidence_policy,
        therm_and_post_md_policy: profile.therm_and_post_md_policy,
        output_and_corpus_policy: profile.output_and_corpus_policy,
        output: execution.output,
    };
    Ok(HyperRuntimeRequestV1 {
        schema_kind: HYPER_RUNTIME_REQUEST_SCHEMA.into(),
        schema_version: 2,
        request_id_sha256: canonical_request_identity_sha256(&request)?,
        request,
        attribution: RequestAttributionV1 {
            client_surface: execution.client_surface,
        },
    })
}

pub fn materialize_governed_branch_hyper_runtime_request(
    selection_path: &Path,
    execution_request_path: &Path,
) -> PcResult<HyperRuntimeRequestV1> {
    materialize_governed_branch_hyper_runtime_request_with_profile_identity(
        selection_path,
        execution_request_path,
        CANONICAL_EXECUTION_PROFILE_SHA256,
        Path::new(CANONICAL_ACTIVE_RELEASE_POINTER),
        &AuthorityProtection::Production,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn cpu_fixture_protection(root: &Path) -> AuthorityProtection {
        AuthorityProtection::CpuFixture { root: root.to_path_buf(), uid: fs::metadata(root).unwrap().uid() }
    }

    fn write(path: &Path, bytes: &[u8]) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, bytes).unwrap();
    }

    fn identity(path: &Path) -> ArtifactIdentityV1 {
        ArtifactIdentityV1 {
            locator: path.to_path_buf(),
            sha256: sha256_file_hex(path).unwrap(),
        }
    }

    fn fixture() -> (TempDir, PathBuf, PathBuf) {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path();
        let packet = root.join("packets/run.pcode");
        let binding = root.join("packets/run.binding.json");
        let topology = root.join("inputs/target.topology.json");
        let input = root.join("inputs/target.prmtop");
        let engine = root.join("installed/bin/nhs_rt_full");
        let active_release = root.join("engine-releases/release-1");
        let active_engine = active_release.join("bin/nhs_rt_full");
        let base_packet = include_bytes!(
            "../../../../appliance_payload_reference/packets/prismos_hyper_runtime_reference.pcode"
        );
        let (packet_bytes, _) =
            crate::binary_runtime::normalize_md_engine_policy_packet_bytes(base_packet)
                .expect("normalize fixture packet policy");
        write(&packet, &packet_bytes);
        write(&binding, b"binding");
        write(&topology, b"topology");
        write(&input, b"input");
        write(&engine, b"engine");
        write(&active_engine, b"engine");
        let engine_sha256 = sha256_file_hex(&engine).unwrap();
        write(
            &active_release.join("ledger/PRISMOS_ACTIVE_STATE_LOCK.json"),
            serde_json::to_string(&serde_json::json!({
                "schema_kind": "prismos_active_state_lock",
                "schema_version": 3,
                "active_release": "release-1",
                "engine_sha256": engine_sha256,
                "source_epoch_id": "7".repeat(64),
                "additional_governed_field": true
            }))
            .unwrap()
            .as_bytes(),
        );
        #[cfg(unix)]
        std::os::unix::fs::symlink("release-1", root.join("engine-releases/ACTIVE")).unwrap();
        #[cfg(unix)]
        {
            fs::set_permissions(root, fs::Permissions::from_mode(0o700)).unwrap();
            fs::set_permissions(&engine, fs::Permissions::from_mode(0o644)).unwrap();
            for directory in [root.join("installed"), root.join("installed/bin"),
                root.join("engine-releases"), active_release.join("bin"), active_release.join("ledger")] {
                fs::set_permissions(directory, fs::Permissions::from_mode(0o755)).unwrap();
            }
            fs::set_permissions(&active_engine, fs::Permissions::from_mode(0o555)).unwrap();
            fs::set_permissions(
                active_release.join("ledger/PRISMOS_ACTIVE_STATE_LOCK.json"),
                fs::Permissions::from_mode(0o444),
            )
            .unwrap();
            fs::set_permissions(&active_release, fs::Permissions::from_mode(0o555)).unwrap();
        }
        let target_root = root.join("requests");
        let descriptor = TargetExecutionIntentV1 {
            schema_kind: TARGET_EXECUTION_INTENT_SCHEMA.into(),
            schema_version: 1,
            target_id: "target-1".into(),
            topology: identity(&topology),
            input: identity(&input),
            coupled: None,
        };
        write(
            &target_root.join("target-1.json"),
            serde_json::to_string(&descriptor).unwrap().as_bytes(),
        );
        let output_root = root.join("runs");
        let profile = ExecutionProfileV1 {
            schema_kind: EXECUTION_PROFILE_SCHEMA.into(),
            schema_version: 2,
            profile: ExecutionProfileIdentityV1 {
                id: "prismos.hyper-runtime-md".into(),
                version: "1.0.0".into(),
            },
            packet_authority: PacketAuthorityV1 {
                prismcode_packet: identity(&packet),
                qnanl_binding: identity(&binding),
            },
            topology_and_input_identity: TopologyAndInputIdentityPolicyV1 {
                target_id_required: true,
                target_request_root: target_root,
                target_request_schema: TARGET_EXECUTION_INTENT_SCHEMA.into(),
                target_request_schema_version: 1,
                topology_locator_policy: "required_runtime_input".into(),
                input_identity_policy: "required_runtime_input".into(),
                topology_sha256_required: true,
                input_sha256_required: true,
                unresolved_runtime_input_status: "EVIDENCE_UNAVAILABLE".into(),
            },
            engine_and_source_epoch_authority: EngineAndSourceEpochAuthorityV1 {
                engine_locator: engine,
                active_release_required: true,
                source_epoch_required: true,
                raw_engine_safe_defaults_unchanged: true,
            },
            force_and_integrator_policy: ForceAndIntegratorPolicyV1 {
                force_contract: "prismcode-authorized".into(),
                integrator: "canonical-hyper-runtime".into(),
                independent_raw_engine_flags_forbidden: true,
            },
            graph_and_metal_resident_policy: GraphAndMetalResidentPolicyV1 {
                cuda_graph_required: true,
                metal_resident_required: true,
                hotpath_acceleration_required: true,
            },
            path_b_dstw_superresolution_policy: PathBDstwSuperresolutionPolicyV1 {
                path_b_required: true,
                dstw_required: true,
                superresolution_required: true,
            },
            observer_and_evidence_policy: ObserverAndEvidencePolicyV1 {
                qnanl_inline_seal_verify_required: true,
                runtime_events_required: true,
                evidence_qualification_required: true,
            },
            therm_and_post_md_policy: ThermAndPostMdPolicyV1 {
                therm_required: true,
                requested_post_md_aggregation: PostMdAggregationPolicyV1::Full,
                effective_post_md_aggregation: PostMdAggregationPolicyV1::Full,
                full_post_md_authorized: true,
                requested_therm_surface: ThermSurfacePolicyV1::Lattice,
                effective_therm_surface: ThermSurfacePolicyV1::Lattice,
                fallback_reason: None,
                therm_surface_fallback_authorized: false,
                effective_surface_provenance_required: true,
            },
            output_and_corpus_policy: OutputAndCorpusPolicyV1 {
                run_output_root: output_root.clone(),
                governed_corpus_handoff_required: true,
                output_identity_required: true,
            },
        };
        let profile_path = root.join("profile.json");
        write(
            &profile_path,
            serde_json::to_string(&profile).unwrap().as_bytes(),
        );
        (temporary, profile_path, output_root.join("target-1"))
    }

    fn materialize_fixture_request(
        profile: &Path,
        output: &Path,
        client_surface: &str,
    ) -> PcResult<HyperRuntimeRequestV1> {
        let expected_profile_sha256 = sha256_file_hex(profile)?;
        materialize_hyper_runtime_request_with_profile_identity(
            profile,
            "target-1",
            output,
            client_surface,
            &expected_profile_sha256,
            &profile.parent().unwrap().join("engine-releases/ACTIVE"),
            &cpu_fixture_protection(profile.parent().unwrap()),
        )
    }

    #[test]
    fn authority_rejects_writable_ancestry_and_wrong_owner_with_explicit_cpu_scope() {
        let (temporary, profile, output) = fixture();
        materialize_fixture_request(&profile, &output, "cpu-fixture").unwrap();
        for path in [temporary.path().join("engine-releases"),
            temporary.path().join("engine-releases/release-1/bin"),
            temporary.path().join("installed")] {
            fs::set_permissions(&path, fs::Permissions::from_mode(0o775)).unwrap();
            let result = materialize_fixture_request(&profile, &output, "cpu-fixture");
            fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
            assert!(result.unwrap_err().to_string().contains("protected ancestry"));
        }
        let current_uid=fs::metadata(temporary.path()).unwrap().uid();
        let wrong_owner=AuthorityProtection::CpuFixture { root: temporary.path().to_path_buf(), uid: current_uid.wrapping_add(1) };
        assert!(require_protected_ancestry(&temporary.path().join("installed/bin/nhs_rt_full"), "owner", &wrong_owner).is_err());
        // The test-only acceptance above does not assert root ownership or
        // production filesystem admission. Production rejects this temp tree.
        assert!(require_protected_ancestry(&temporary.path().join("installed/bin/nhs_rt_full"), "production", &AuthorityProtection::Production).is_err());
        assert_eq!(CANONICAL_ACTIVE_RELEASE_POINTER,"/var/lib/prismos/engine-releases/ACTIVE");
    }

    #[test]
    fn hyper_runtime_and_therm_postmd_profiles_explicitly_require_full_lattice_policy() {
        for id in ["prismos.hyper-runtime-md", "prismos.therm-postmd"] {
            let profile = ExecutionProfileIdentityV1 {
                id: id.to_string(),
                version: "1.0.0".to_string(),
            };
            let policy = ThermAndPostMdPolicyV1 {
                therm_required: true,
                requested_post_md_aggregation: PostMdAggregationPolicyV1::Full,
                effective_post_md_aggregation: PostMdAggregationPolicyV1::Full,
                full_post_md_authorized: true,
                requested_therm_surface: ThermSurfacePolicyV1::Lattice,
                effective_therm_surface: ThermSurfacePolicyV1::Lattice,
                fallback_reason: None,
                therm_surface_fallback_authorized: false,
                effective_surface_provenance_required: true,
            };
            validate_therm_and_post_md_policy(&profile, &policy).unwrap();
            assert_eq!(
                policy.requested_post_md_aggregation,
                PostMdAggregationPolicyV1::Full
            );
            assert_eq!(
                policy.effective_post_md_aggregation,
                PostMdAggregationPolicyV1::Full
            );
            assert_eq!(
                policy.effective_therm_surface,
                ThermSurfacePolicyV1::Lattice
            );
        }
    }

    #[test]
    fn requested_effective_or_authorization_mismatch_fails_before_materialization() {
        for mutation in ["effective-skip", "full-not-authorized", "surface-fallback"] {
            let (_temporary, profile_path, output) = fixture();
            let mut profile: ExecutionProfileV1 =
                serde_json::from_slice(&fs::read(&profile_path).unwrap()).unwrap();
            match mutation {
                "effective-skip" => {
                    profile
                        .therm_and_post_md_policy
                        .effective_post_md_aggregation = PostMdAggregationPolicyV1::Skip;
                }
                "full-not-authorized" => {
                    profile.therm_and_post_md_policy.full_post_md_authorized = false;
                }
                "surface-fallback" => {
                    profile.therm_and_post_md_policy.effective_therm_surface =
                        ThermSurfacePolicyV1::Clusters;
                }
                _ => unreachable!(),
            }
            write(&profile_path, &serde_json::to_vec(&profile).unwrap());
            let error = materialize_fixture_request(&profile_path, &output, "systemd")
                .expect_err("policy mismatch must fail before request materialization");
            assert!(
                error.to_string().contains("HYPER_RUNTIME_REQUEST_INVALID"),
                "{mutation} produced unexpected error: {error}"
            );
        }
    }

    fn sized_identity(path: &Path) -> SizedArtifactIdentityV1 {
        SizedArtifactIdentityV1 {
            locator: path.to_path_buf(),
            sha256: sha256_file_hex(path).unwrap(),
            size_bytes: fs::metadata(path).unwrap().len(),
        }
    }

    struct GovernedFixture {
        _temporary: TempDir,
        selection: PathBuf,
        request: PathBuf,
        profile_sha256: String,
    }

    fn governed_fixture() -> GovernedFixture {
        let (temporary, profile_path, output) = fixture();
        let root = temporary.path();
        // The fixed profile packet is intentionally a different declared
        // identity; branch execution must not be locked to it.
        let mut profile: serde_json::Value =
            serde_json::from_slice(&fs::read(&profile_path).unwrap()).unwrap();
        profile["packet_authority"]["prismcode_packet"]["sha256"] =
            serde_json::json!("f".repeat(64));
        write(&profile_path, &serde_json::to_vec(&profile).unwrap());
        let profile_sha256 = sha256_file_hex(&profile_path).unwrap();
        let profile_identity = GovernedExecutionProfileIdentityV1 {
            id: "prismos.hyper-runtime-md".into(),
            version: "1.0.0".into(),
            locator: profile_path.clone(),
            sha256: profile_sha256.clone(),
        };
        let descriptor_path = root.join("requests/target-1.json");
        let descriptor: TargetExecutionIntentV1 =
            serde_json::from_slice(&fs::read(&descriptor_path).unwrap()).unwrap();
        let target = GovernedTargetIdentityV1 {
            target_id: "target-1".into(),
            descriptor: identity(&descriptor_path),
            topology: descriptor.topology,
            input: descriptor.input,
        };
        let source_epoch_id = "7".repeat(64);

        let proposal_row = serde_json::json!({"candidate_id":"candidate-1","rank":1,"score":7});
        let proposal_id = canonical_json_sha256(&proposal_row).unwrap();
        let proposal_set = root.join("cognition/proposals.parquet");
        let sample_receipt = root.join("cognition/sample-execution.json");
        write(&proposal_set, b"parquet proposal-set bytes");
        write(&sample_receipt, b"sample execution receipt");
        let proposal_ids = vec![proposal_id.clone()];
        let proposal_ids_sha256 = canonical_json_sha256(&proposal_ids).unwrap();
        let seal_path = root.join("cognition/proposal-set.seal.json");
        let mut seal = SealedProposalSetReceiptV1 {
            schema_kind: "prismos.cognition.sealed_gflownet_proposal_set".into(),
            schema_version: 1,
            seal_status: "SEALED".into(),
            source_epoch_id: source_epoch_id.clone(),
            proposal_set: sized_identity(&proposal_set),
            sample_execution_receipt: sized_identity(&sample_receipt),
            proposal_count: 1,
            proposal_id_algorithm: "sha256-canonical-json-v1".into(),
            proposal_ids,
            proposal_ids_sha256,
            seal_sha256: String::new(),
        };
        seal.seal_sha256 = canonical_json_sha256(&SealedProposalSetDigestV1 {
            schema_kind: &seal.schema_kind,
            schema_version: seal.schema_version,
            seal_status: &seal.seal_status,
            source_epoch_id: &seal.source_epoch_id,
            proposal_set: &seal.proposal_set,
            sample_execution_receipt: &seal.sample_execution_receipt,
            proposal_count: seal.proposal_count,
            proposal_id_algorithm: &seal.proposal_id_algorithm,
            proposal_ids: &seal.proposal_ids,
            proposal_ids_sha256: &seal.proposal_ids_sha256,
        })
        .unwrap();
        write(&seal_path, &serde_json::to_vec(&seal).unwrap());

        let selection_authority = ExplicitSelectionAuthorityV1 {
            authority_kind: "operator-explicit-artifact".into(),
            authority_id: "operator-1".into(),
            directive_id: "directive-1".into(),
        };
        let explicit_path = root.join("cognition/explicit-selection.json");
        let explicit = ExplicitBranchSelectionV1 {
            schema_kind: "prismos.cognition.explicit_branch_selection".into(),
            schema_version: 1,
            selection_mode: "explicit-governed-selection".into(),
            selected: true,
            selection_authority: selection_authority.clone(),
            proposal: ExplicitProposalIdentityV1 {
                proposal_id_sha256: proposal_id.clone(),
                row_sha256: proposal_id.clone(),
                proposal_set_receipt: sized_identity(&seal_path),
                proposal_artifact: sized_identity(&proposal_set),
                non_authoritative_rank: Some(1),
            },
            source_epoch_id: source_epoch_id.clone(),
            execution_profile: profile_identity.clone(),
            target: target.clone(),
            output: output.clone(),
        };
        write(&explicit_path, &serde_json::to_vec(&explicit).unwrap());
        let branch_id = canonical_json_sha256(&BranchIdentityV1 {
            schema_kind: "prismos.cognition.next_branch_identity",
            schema_version: 1,
            explicit_selection_sha256: &sha256_file_hex(&explicit_path).unwrap(),
            proposal_id_sha256: &proposal_id,
            source_epoch_id: &source_epoch_id,
            execution_profile: &profile_identity,
            target: &target,
            output: &output,
        })
        .unwrap();

        let selected_path = root.join("cognition/SELECTED_PROPOSAL.json");
        write(
            &selected_path,
            &serde_json::to_vec(&SelectedProposalV1 {
                schema_kind: "prismos.cognition.selected_gflownet_proposal".into(),
                schema_version: 1,
                proposal_id_sha256: proposal_id.clone(),
                row_sha256: proposal_id.clone(),
                canonical_row_utf8: serde_json::to_string(&proposal_row).unwrap(),
                row: proposal_row,
                proposal_set_artifact: sized_identity(&proposal_set),
                proposal_set_receipt: sized_identity(&seal_path),
                source_epoch_id: source_epoch_id.clone(),
            })
            .unwrap(),
        );
        let packet_path = root.join("cognition/branch.pcode");
        fs::copy(root.join("packets/run.pcode"), &packet_path).unwrap();
        let validated = load_validated_md_engine_policy(&packet_path).unwrap();
        let input_packet = root.join("cognition/input.pcode");
        write(&input_packet, b"pre-normalized source packet");
        let native_normalization = root.join("cognition/native-normalization.json");
        write(&native_normalization, b"native normalization receipt");
        let normalization_path = root.join("cognition/normalization.json");
        write(
            &normalization_path,
            &serde_json::to_vec(&serde_json::json!({
                "schema_kind":"prismos.cognition.prismcode_policy_normalization_receipt",
                "schema_version":1,
                "status":"PASS",
                "input_packet":sized_identity(&input_packet),
                "output_packet":{
                    "locator":packet_path,
                    "sha256":validated.packet_sha256,
                    "size_bytes":validated.packet_bytes,
                    "internal_blake3":validated.packet_hash_blake3
                },
                "native_receipt":sized_identity(&native_normalization),
                "source_epoch_id":source_epoch_id,
                "branch_id_sha256":branch_id,
                "command_argv":["prismcode-packet-policy","--input",input_packet,"--output",packet_path,"--report",native_normalization]
            }))
            .unwrap(),
        );
        let native_validation = root.join("cognition/native-validation.json");
        write(
            &native_validation,
            &serde_json::to_vec(&serde_json::json!({
                "status":"PASS","level":"L3","packet_hash":validated.packet_hash_blake3
            }))
            .unwrap(),
        );
        let validation_path = root.join("cognition/validation.json");
        write(
            &validation_path,
            &serde_json::to_vec(&serde_json::json!({
                "schema_kind":"prismos.cognition.prismcode_packet_validation_receipt",
                "schema_version":1,
                "status":"PASS",
                "validation_level":"L3",
                "packet":{"locator":packet_path,"sha256":validated.packet_sha256,"size_bytes":validated.packet_bytes,"internal_blake3":validated.packet_hash_blake3},
                "native_validation_report":sized_identity(&native_validation),
                "native_validation":{"status":"PASS","level":"L3","packet_hash":validated.packet_hash_blake3},
                "source_epoch_id":source_epoch_id,
                "branch_id_sha256":branch_id,
                "command_argv":["prismval","validate",packet_path,"--level","L3","--out",native_validation]
            }))
            .unwrap(),
        );
        let profile: ExecutionProfileV1 =
            serde_json::from_slice(&fs::read(&profile_path).unwrap()).unwrap();
        let policy_snapshot = GovernedPolicySnapshotV1 {
            force_and_integrator_policy: profile.force_and_integrator_policy.clone(),
            graph_and_metal_resident_policy: profile.graph_and_metal_resident_policy.clone(),
            path_b_dstw_superresolution_policy: profile.path_b_dstw_superresolution_policy.clone(),
            therm_and_post_md_policy: profile.therm_and_post_md_policy.clone(),
        };
        let branch_source = root.join("cognition/branch-source.json");
        write(&branch_source, b"branch compiler source");
        let packet = GovernedPacketIdentityV1 {
            locator: packet_path,
            sha256: validated.packet_sha256,
            internal_blake3: validated.packet_hash_blake3,
            schema_kind: "prismos.cognition.next_branch_prismcode_qnanl_packet".into(),
            schema_version: 1,
            validation_level: "L3".into(),
            normalization_receipt: sized_identity(&normalization_path),
            validation_receipt: sized_identity(&validation_path),
        };
        let selection_path = root.join("cognition/governed-selection.json");
        let selection = GovernedBranchSelectionV1 {
            schema_kind: GOVERNED_BRANCH_SELECTION_SCHEMA.into(),
            schema_version: 1,
            selection_mode: "explicit-governed-selection".into(),
            selected: true,
            explicit_selection: sized_identity(&explicit_path),
            selection_authority,
            proposal: GovernedProposalIdentityV1 {
                proposal_id_sha256: proposal_id.clone(),
                row_sha256: proposal_id,
                proposal_set_receipt: sized_identity(&seal_path),
                proposal_artifact: sized_identity(&proposal_set),
                selected_proposal: sized_identity(&selected_path),
                non_authoritative_rank: Some(1),
            },
            source_epoch_id: source_epoch_id.clone(),
            execution_profile: profile_identity.clone(),
            target: target.clone(),
            packet: packet.clone(),
            policy_snapshot: policy_snapshot.clone(),
            evidence_snapshot: profile.observer_and_evidence_policy.clone(),
            output_snapshot: profile.output_and_corpus_policy.clone(),
            output: output.clone(),
            branch_source: sized_identity(&branch_source),
            branch_id_sha256: branch_id.clone(),
        };
        write(&selection_path, &serde_json::to_vec(&selection).unwrap());
        let request_path = root.join("cognition/request.json");
        write(
            &request_path,
            &serde_json::to_vec(&NextBranchExecutionRequestV1 {
                schema_kind: NEXT_BRANCH_EXECUTION_REQUEST_SCHEMA.into(),
                schema_version: 1,
                branch_id_sha256: branch_id,
                governed_branch_selection: sized_identity(&selection_path),
                source_epoch_id,
                execution_profile: profile_identity,
                target,
                packet,
                policy_snapshot,
                evidence_snapshot: profile.observer_and_evidence_policy,
                output_snapshot: profile.output_and_corpus_policy,
                client_surface: "agent".into(),
                output,
            })
            .unwrap(),
        );
        GovernedFixture {
            _temporary: temporary,
            selection: selection_path,
            request: request_path,
            profile_sha256,
        }
    }

    #[test]
    fn unsafe_target_ids_are_rejected() {
        for value in ["", "..", "../x", "x/y", "x\\y", "x y"] {
            assert!(validate_target_id(value).is_err(), "accepted {value:?}");
        }
        assert!(validate_target_id("glp1r-6XOX_apo.v1").is_ok());
    }

    #[test]
    fn attribution_is_outside_canonical_request_identity() {
        let identity = HyperRuntimeRequestIdentityV1 {
        experiment: None,
            execution_profile: ExecutionProfileIdentityV1 {
                id: "profile".into(),
                version: "1".into(),
            },
            execution_profile_sha256: "0".repeat(64),
            packet_authority: PacketAuthorityV1 {
                prismcode_packet: ArtifactIdentityV1 {
                    locator: "/packet".into(),
                    sha256: "1".repeat(64),
                },
                qnanl_binding: ArtifactIdentityV1 {
                    locator: "/binding".into(),
                    sha256: "2".repeat(64),
                },
            },
            validated_packet_policy: ValidatedMdEnginePolicyV1 {
                policy: MdEnginePolicyV1::canonical_hyper_runtime(),
                packet_hash_blake3: "6".repeat(64),
                packet_sha256: "1".repeat(64),
                packet_bytes: 4096,
                instruction_count: 24,
            },
            governed_branch: None,
            target_id: "target".into(),
            topology: ArtifactIdentityV1 {
                locator: "/topology".into(),
                sha256: "3".repeat(64),
            },
            input: ArtifactIdentityV1 {
                locator: "/input".into(),
                sha256: "4".repeat(64),
            },
            coupled: None,
            native_engine_intent: None,
            runtime_authority: RuntimeAuthorityIdentityV1 {
                source_epoch_id: "epoch".into(),
                active_release: "release".into(),
                active_lock: ArtifactIdentityV1 {
                    locator: "/release/ledger/PRISMOS_ACTIVE_STATE_LOCK.json".into(),
                    sha256: "6".repeat(64),
                },
                engine: ArtifactIdentityV1 {
                    locator: "/engine".into(),
                    sha256: "5".repeat(64),
                },
            },
            force_and_integrator_policy: ForceAndIntegratorPolicyV1 {
                force_contract: "prismcode-authorized".into(),
                integrator: "canonical-hyper-runtime".into(),
                independent_raw_engine_flags_forbidden: true,
            },
            graph_and_metal_resident_policy: GraphAndMetalResidentPolicyV1 {
                cuda_graph_required: true,
                metal_resident_required: true,
                hotpath_acceleration_required: true,
            },
            path_b_dstw_superresolution_policy: PathBDstwSuperresolutionPolicyV1 {
                path_b_required: true,
                dstw_required: true,
                superresolution_required: true,
            },
            observer_and_evidence_policy: ObserverAndEvidencePolicyV1 {
                qnanl_inline_seal_verify_required: true,
                runtime_events_required: true,
                evidence_qualification_required: true,
            },
            therm_and_post_md_policy: ThermAndPostMdPolicyV1 {
                therm_required: true,
                requested_post_md_aggregation: PostMdAggregationPolicyV1::Full,
                effective_post_md_aggregation: PostMdAggregationPolicyV1::Full,
                full_post_md_authorized: true,
                requested_therm_surface: ThermSurfacePolicyV1::Lattice,
                effective_therm_surface: ThermSurfacePolicyV1::Lattice,
                fallback_reason: None,
                therm_surface_fallback_authorized: false,
                effective_surface_provenance_required: true,
            },
            output_and_corpus_policy: OutputAndCorpusPolicyV1 {
                run_output_root: "/output".into(),
                governed_corpus_handoff_required: true,
                output_identity_required: true,
            },
            output: "/output/run".into(),
        };
        let cli = HyperRuntimeRequestV1 {
            schema_kind: HYPER_RUNTIME_REQUEST_SCHEMA.into(),
            schema_version: 2,
            request_id_sha256: canonical_request_identity_sha256(&identity).unwrap(),
            request: identity.clone(),
            attribution: RequestAttributionV1 {
                client_surface: "cli".into(),
            },
        };
        let service = HyperRuntimeRequestV1 {
            attribution: RequestAttributionV1 {
                client_surface: "systemd".into(),
            },
            ..cli.clone()
        };
        assert_eq!(cli.request_id_sha256, service.request_id_sha256);
        assert_eq!(cli.request, service.request);
        assert_ne!(cli.attribution, service.attribution);
    }

    #[test]
    fn equivalent_clients_materialize_one_canonical_identity() {
        let (_temporary, profile, output) = fixture();
        let cli = materialize_fixture_request(&profile, &output, "cli").unwrap();
        let service = materialize_fixture_request(&profile, &output, "systemd").unwrap();
        assert_eq!(cli.request, service.request);
        assert_eq!(cli.request_id_sha256, service.request_id_sha256);
        assert_ne!(cli.attribution, service.attribution);
        assert_eq!(
            cli.request.runtime_authority.source_epoch_id,
            "7".repeat(64)
        );
        assert!(
            cli.request
                .path_b_dstw_superresolution_policy
                .path_b_required
        );
        let serialized = serde_json::to_vec(&cli).unwrap();
        let roundtrip: HyperRuntimeRequestV1 = serde_json::from_slice(&serialized).unwrap();
        assert_eq!(roundtrip, cli);
        validate_therm_and_post_md_packet_admission(
            &roundtrip.request.execution_profile,
            &roundtrip.request.therm_and_post_md_policy,
            &roundtrip.request.validated_packet_policy.policy,
        )
        .unwrap();
    }

    #[test]
    fn exact_active_lock_bytes_are_part_of_request_identity() {
        let (temporary, profile, output) = fixture();
        let first = materialize_fixture_request(&profile, &output, "cli").unwrap();
        let lock = &first.request.runtime_authority.active_lock;
        assert_eq!(lock.sha256, sha256_file_hex(&lock.locator).unwrap());
        let mut bytes = fs::read(&lock.locator).unwrap();
        bytes.push(b'\n');
        fs::set_permissions(&lock.locator, fs::Permissions::from_mode(0o644)).unwrap();
        fs::write(&lock.locator, bytes).unwrap();
        fs::set_permissions(&lock.locator, fs::Permissions::from_mode(0o444)).unwrap();
        let second = materialize_fixture_request(&profile, &output, "gui").unwrap();
        assert_eq!(first.request.runtime_authority.source_epoch_id, second.request.runtime_authority.source_epoch_id);
        assert_ne!(first.request.runtime_authority.active_lock.sha256, second.request.runtime_authority.active_lock.sha256);
        assert_ne!(first.request_id_sha256, second.request_id_sha256);
        drop(temporary);
    }

    #[test]
    fn native_parsed_intent_and_replicate_identity_survive_transport_order() {
        let (_temporary, profile, output) = fixture();
        let mut request = materialize_fixture_request(&profile, &output, "cli").unwrap();
        let normalized = BTreeMap::from([
            ("temperature".into(), vec!["310".into()]),
            ("replica_seed".into(), vec!["42".into()]),
        ]);
        request.request.native_engine_intent = Some(NativeEngineIntentV1 {
            argv: vec!["--temperature".into(), "310".into(), "--replica-seed".into(), "42".into()],
            normalized_arguments: normalized,
            replicate: Some(ReplicateIntentV1 { group_id: "campaign".into(), index: 0, count: 3, seed: 42 }),
            artifact_transport: None,
            transport_input_bindings: Vec::new(),
        });
        let first = canonical_request_identity_sha256(&request.request).unwrap();
        let native = request.request.native_engine_intent.as_mut().unwrap();
        native.argv.rotate_left(2);
        assert_eq!(first, canonical_request_identity_sha256(&request.request).unwrap());
        request.request.native_engine_intent.as_mut().unwrap().normalized_arguments.insert("temperature".into(), vec!["320".into()]);
        assert_ne!(first, canonical_request_identity_sha256(&request.request).unwrap());
    }

    fn native_validation_fixture() -> NativeRunIntentValidationV1 {
        NativeRunIntentValidationV1 {
            schema_kind: "prismos.native_run_intent_validation".into(), schema_version: 1,
            validated: true, argv: Vec::new(), normalized_arguments: BTreeMap::new(),
            explicit_arguments: Vec::new(), argument_metadata: serde_json::json!({}),
            validation_scope: "NATIVE_CLI_TYPES_AND_INTENT_AUTHORITY_BOUNDARY".into(),
            runtime_admission_performed: false, device_initialized: false,
        }
    }

    #[test]
    fn native_client_profile_defaults_preserve_explicit_scientific_lanes() {
        let mut parsed = native_validation_fixture();
        assert_eq!(profile_surface_defaults(&parsed), vec!["--engine-surface", "ghost-tiles"]);
        parsed.explicit_arguments = vec!["engine_surface".into(), "multi_stream".into(), "multi_temp".into()];
        parsed.normalized_arguments.insert("engine_surface".into(), vec!["twin16+ghost-tiles".into()]);
        assert!(profile_surface_defaults(&parsed).is_empty());
        let mut native = NativeEngineIntentV1 { argv: Vec::new(), normalized_arguments: parsed.normalized_arguments.clone(),
            replicate: None, artifact_transport: None, transport_input_bindings: Vec::new() };
        validate_client_lattice_producer(&native).unwrap();
        native.normalized_arguments.insert("engine_surface".into(), vec!["ghost-tiles".into()]);
        native.normalized_arguments.insert("multi_stream".into(), vec!["8".into()]);
        validate_client_lattice_producer(&native).unwrap();
        native.normalized_arguments.insert("multi_differential".into(), vec!["true".into()]);
        validate_client_lattice_producer(&native).unwrap();
        native.normalized_arguments.insert("engine_surface".into(), vec!["none".into()]);
        assert!(validate_client_lattice_producer(&native).is_err());
        parsed.normalized_arguments.insert("engine_surface".into(), vec!["ghost-tiles".into()]);
        parsed.normalized_arguments.insert("multi_temp".into(), vec!["true".into()]);
        assert!(profile_surface_defaults(&parsed).is_empty());
        parsed.normalized_arguments.remove("multi_temp");
        parsed.normalized_arguments.insert("rest2".into(), vec!["true".into()]);
        assert!(profile_surface_defaults(&parsed).is_empty());
        parsed.normalized_arguments.insert("engine_b_topology".into(), vec!["/topology-b.json".into()]);
        parsed.explicit_arguments.clear();
        assert_eq!(profile_surface_defaults(&parsed), vec!["--coupled-twin"]);
        parsed.normalized_arguments.insert("coupled_twin".into(), vec!["true".into()]);
        assert!(profile_surface_defaults(&parsed).is_empty());
    }

    #[test]
    fn native_client_omitted_lane_does_not_choose_scientific_protocol_or_streams() {
        let mut parsed = native_validation_fixture();
        parsed.normalized_arguments = BTreeMap::from([
            ("multi_differential".into(), vec!["false".into()]),
            ("multi_stream".into(), vec!["0".into()]),
            ("engine_surface".into(), vec!["none".into()]),
        ]);
        let original = parsed.normalized_arguments.clone();
        assert_eq!(profile_surface_defaults(&parsed), vec!["--engine-surface", "ghost-tiles"]);
        assert_eq!(parsed.normalized_arguments, original);
        let native = NativeEngineIntentV1 { argv: Vec::new(), normalized_arguments: original,
            replicate: None, artifact_transport: None, transport_input_bindings: Vec::new() };
        assert!(validate_client_lattice_producer(&native).unwrap_err().to_string()
            .contains("requires a native Ghost/V2-producing"));
    }

    #[test]
    fn native_client_explicit_differential_keeps_native_stream_expansion_authority() {
        let mut parsed = native_validation_fixture();
        parsed.argv = vec!["--multi-differential".into()];
        parsed.explicit_arguments = vec!["multi_differential".into()];
        parsed.normalized_arguments = BTreeMap::from([
            ("multi_differential".into(), vec!["true".into()]),
            ("multi_stream".into(), vec!["0".into()]),
            ("engine_surface".into(), vec!["none".into()]),
        ]);
        assert_eq!(profile_surface_defaults(&parsed), vec!["--engine-surface", "ghost-tiles"]);
        assert_eq!(parsed.argv, vec!["--multi-differential"]);
        assert_eq!(parsed.normalized_arguments["multi_stream"], vec!["0"]);
        parsed.normalized_arguments.insert("engine_surface".into(), vec!["ghost-tiles".into()]);
        let native = NativeEngineIntentV1 { argv: parsed.argv, normalized_arguments: parsed.normalized_arguments,
            replicate: None, artifact_transport: None, transport_input_bindings: Vec::new() };
        validate_client_lattice_producer(&native).unwrap();
    }

    #[test]
    fn native_client_named_differential_surface_remains_explicit_native_authority() {
        for surface in ["captured-graph-twin+ghost-tiles", "interferometric-twin+ghost-tiles", "twin16+ghost-tiles"] {
            let mut parsed = native_validation_fixture();
            parsed.argv = vec!["--engine-surface".into(), surface.into()];
            parsed.explicit_arguments = vec!["engine_surface".into()];
            parsed.normalized_arguments = BTreeMap::from([
                ("multi_differential".into(), vec!["false".into()]),
                ("multi_stream".into(), vec!["0".into()]),
                ("engine_surface".into(), vec![surface.into()]),
            ]);
            assert!(profile_surface_defaults(&parsed).is_empty());
            let native = NativeEngineIntentV1 { argv: parsed.argv, normalized_arguments: parsed.normalized_arguments,
                replicate: None, artifact_transport: None, transport_input_bindings: Vec::new() };
            validate_client_lattice_producer(&native).unwrap();
        }
    }

    #[test]
    fn native_client_genuine_gui_standard_preserves_physics_with_native_ghost_producer() {
        // Exact stdout of the native pure parser on the production UI library's
        // Standard command, not a reconstructed argument map. Provenance is
        // retained in RESUME_ISA/GENUINE_STANDARD_PARSER_E_02.
        let bytes = include_bytes!("../tests/fixtures/gui_standard_native_intent_v1.json");
        assert_eq!(format!("{:x}", Sha256::digest(bytes)),
            "aaf3c9654e706f059368ae5b483c80452818c231cbc36372f4ed8fa9ad54cd59");
        let parsed: NativeRunIntentValidationV1 = serde_json::from_slice(bytes).unwrap();
        assert!(parsed.validated);
        assert!(!parsed.runtime_admission_performed && !parsed.device_initialized);
        for (field, expected) in [("temperature", "297.25"), ("cryo_temp", "73.5"),
            ("steps", "1234567"), ("multi_stream", "8"), ("fast", "false"),
            ("multi_differential", "false")] {
            assert_eq!(parsed.normalized_arguments[field], vec![expected]);
        }
        for field in ["temperature", "cryo_temp", "steps", "multi_stream"] {
            assert!(parsed.explicit_arguments.iter().any(|value| value == field));
        }
        assert!(!parsed.argv.iter().any(|value| matches!(value.as_str(), "--fast" | "--multi-differential")));
        let original_argv = parsed.argv.clone();
        let original_normalized = parsed.normalized_arguments.clone();
        assert_eq!(profile_surface_defaults(&parsed), vec!["--engine-surface", "ghost-tiles"]);
        let mut native = NativeEngineIntentV1 { argv: parsed.argv, normalized_arguments: parsed.normalized_arguments,
            replicate: None, artifact_transport: None, transport_input_bindings: Vec::new() };
        assert!(validate_client_lattice_producer(&native).is_err());
        assert_eq!(native.argv, original_argv);
        assert_eq!(native.normalized_arguments, original_normalized);
        native.normalized_arguments.insert("engine_surface".into(), vec!["ghost-tiles".into()]);
        validate_client_lattice_producer(&native).unwrap();
        for field in ["temperature", "cryo_temp", "steps", "multi_stream", "multi_differential"] {
            assert_eq!(native.normalized_arguments[field], original_normalized[field]);
        }
    }

    #[test]
    fn native_transport_and_coupled_parser_ids_are_bound_without_dropped_controls() {
        let temporary = TempDir::new().unwrap();
        let packet = temporary.path().join("kernel.prismqnanl");
        let topology_b = temporary.path().join("topology-b.json");
        write(&packet, b"exact kernel packet");
        write(&topology_b, b"exact coupled topology");
        let transport = ClientArtifactTransportV1 {
            qnanl_force_kernel_packet: Some(packet.clone()), science_roots: Some(vec![temporary.path().to_path_buf()]),
            qnanl_force_kernel_receipt: Some(temporary.path().join("attempt/receipt.json")),
            prismcode_cuda_opcode_coverage: Some(temporary.path().join("attempt/coverage.json")),
            ligand_mode: Some(ClientLigandModeV1::Holo), ..Default::default()
        };
        let mut native = NativeEngineIntentV1 { argv: Vec::new(), normalized_arguments: BTreeMap::from([
            ("engine_b_topology".into(), vec![topology_b.display().to_string()]),
            ("contrast_kind".into(), vec!["glycan-mechanistic-probe".into()]),
            ("coupled_probe_residue_ids".into(), vec!["380".into(), "381".into()]),
        ]), replicate: None, transport_input_bindings: bind_client_artifact_transport(Some(&transport)).unwrap(),
            artifact_transport: Some(transport) };
        verify_native_artifact_transport(&native).unwrap();
        let coupled = native_coupled_intent(&native).unwrap().unwrap();
        assert_eq!(coupled.topology_b.locator, topology_b);
        assert_eq!(coupled.probe_residue_ids, vec![380,381]);
        assert_eq!(coupled.contrast_kind, MolecularContrastKindV1::GlycanMechanisticProbe);
        let roundtrip: NativeEngineIntentV1 = serde_json::from_slice(&serde_json::to_vec(&native).unwrap()).unwrap();
        assert_eq!(native, roundtrip);
        native.normalized_arguments.remove("contrast_kind");
        assert!(native_coupled_intent(&native).is_err());
        write(&packet, b"changed kernel packet");
        assert!(verify_native_artifact_transport(&native).is_err());
    }

    #[test]
    fn complete_coupled_intent_is_verified_and_bound_into_request_identity() {
        let (temporary, profile, output) = fixture();
        let single = materialize_fixture_request(&profile, &output, "cli").unwrap();
        let topology_b = temporary.path().join("inputs/target-b.topology.json");
        write(&topology_b, b"distinct topology B");
        let descriptor_path = temporary.path().join("requests/target-1.json");
        let mut descriptor: TargetExecutionIntentV1 =
            serde_json::from_slice(&fs::read(&descriptor_path).unwrap()).unwrap();
        descriptor.coupled = Some(CoupledExecutionIntentV1 {
            topology_b: identity(&topology_b),
            contrast_kind: MolecularContrastKindV1::DesignReference,
            identical_topology_negative_control: false,
            probe_residue_ids: Vec::new(),
        });
        write(
            &descriptor_path,
            serde_json::to_string(&descriptor).unwrap().as_bytes(),
        );

        let coupled = materialize_fixture_request(&profile, &output, "cli").unwrap();
        assert_eq!(coupled.request.coupled, descriptor.coupled);
        assert_ne!(coupled.request_id_sha256, single.request_id_sha256);
    }

    #[test]
    fn partial_coupled_tuple_and_bad_topology_hash_fail_before_materialization() {
        let (temporary, profile, output) = fixture();
        let topology_b = temporary.path().join("inputs/target-b.topology.json");
        write(&topology_b, b"distinct topology B");
        let descriptor_path = temporary.path().join("requests/target-1.json");
        let base: serde_json::Value =
            serde_json::from_slice(&fs::read(&descriptor_path).unwrap()).unwrap();
        let complete = serde_json::json!({
            "topology_b": {
                "locator": topology_b,
                "sha256": sha256_file_hex(&topology_b).unwrap()
            },
            "contrast_kind": "design-reference",
            "identical_topology_negative_control": false,
            "probe_residue_ids": []
        });

        for missing in [
            "topology_b",
            "contrast_kind",
            "identical_topology_negative_control",
            "probe_residue_ids",
        ] {
            let mut value = base.clone();
            value["coupled"] = complete.clone();
            value["coupled"].as_object_mut().unwrap().remove(missing);
            write(
                &descriptor_path,
                serde_json::to_vec(&value).unwrap().as_slice(),
            );
            let error = materialize_fixture_request(&profile, &output, "cli").unwrap_err();
            assert!(
                error.to_string().contains("missing field"),
                "missing {missing} produced unexpected error: {error}"
            );
        }

        let mut bad_hash = base;
        bad_hash["coupled"] = complete;
        bad_hash["coupled"]["topology_b"]["sha256"] = serde_json::json!("0".repeat(64));
        write(
            &descriptor_path,
            serde_json::to_vec(&bad_hash).unwrap().as_slice(),
        );
        let error = materialize_fixture_request(&profile, &output, "cli").unwrap_err();
        assert!(error
            .to_string()
            .contains("coupled topology B hash mismatch"));
    }

    #[test]
    fn coupled_control_and_probe_conflicts_fail_closed() {
        let (temporary, profile, output) = fixture();
        let topology_b = temporary.path().join("inputs/target-b.topology.json");
        write(&topology_b, b"distinct topology B");
        let descriptor_path = temporary.path().join("requests/target-1.json");
        let mut descriptor: TargetExecutionIntentV1 =
            serde_json::from_slice(&fs::read(&descriptor_path).unwrap()).unwrap();
        descriptor.coupled = Some(CoupledExecutionIntentV1 {
            topology_b: identity(&topology_b),
            contrast_kind: MolecularContrastKindV1::IdenticalTopologyNegativeControl,
            identical_topology_negative_control: true,
            probe_residue_ids: Vec::new(),
        });
        write(
            &descriptor_path,
            serde_json::to_vec(&descriptor).unwrap().as_slice(),
        );
        let error = materialize_fixture_request(&profile, &output, "cli").unwrap_err();
        assert!(error
            .to_string()
            .contains("identical-topology negative control requires equal topology hashes"));

        descriptor.coupled = Some(CoupledExecutionIntentV1 {
            topology_b: identity(&topology_b),
            contrast_kind: MolecularContrastKindV1::ApoHolo,
            identical_topology_negative_control: false,
            probe_residue_ids: vec![4],
        });
        write(
            &descriptor_path,
            serde_json::to_vec(&descriptor).unwrap().as_slice(),
        );
        let error = materialize_fixture_request(&profile, &output, "cli").unwrap_err();
        assert!(error
            .to_string()
            .contains("probe residue IDs are valid only"));
    }

    #[test]
    fn packet_hash_mismatch_fails_before_request_materialization() {
        let (_temporary, profile, output) = fixture();
        let mut value: serde_json::Value =
            serde_json::from_slice(&fs::read(&profile).unwrap()).unwrap();
        value["packet_authority"]["prismcode_packet"]["sha256"] =
            serde_json::Value::String("0".repeat(64));
        fs::write(&profile, serde_json::to_vec(&value).unwrap()).unwrap();
        let error = materialize_fixture_request(&profile, &output, "systemd").unwrap_err();
        assert!(error.to_string().contains("PrismCode packet hash mismatch"));
    }

    #[test]
    fn noncanonical_profile_policy_fails_before_request_materialization() {
        let (_temporary, profile, output) = fixture();
        let mut value: serde_json::Value =
            serde_json::from_slice(&fs::read(&profile).unwrap()).unwrap();
        value["graph_and_metal_resident_policy"]["hotpath_acceleration_required"] =
            serde_json::Value::Bool(false);
        fs::write(&profile, serde_json::to_vec(&value).unwrap()).unwrap();
        let error = materialize_fixture_request(&profile, &output, "systemd").unwrap_err();
        assert!(error
            .to_string()
            .contains("unsupported execution-profile semantics"));
    }

    #[test]
    fn unknown_profile_fields_fail_closed() {
        let (_temporary, profile, output) = fixture();
        let mut value: serde_json::Value =
            serde_json::from_slice(&fs::read(&profile).unwrap()).unwrap();
        value["private_raw_engine_flags"] = serde_json::json!(["--fast"]);
        fs::write(&profile, serde_json::to_vec(&value).unwrap()).unwrap();
        let error = materialize_fixture_request(&profile, &output, "service").unwrap_err();
        assert!(error.to_string().contains("unknown field"));
    }

    #[test]
    fn missing_source_epoch_authority_fails_closed() {
        let (temporary, profile, output) = fixture();
        let lock = temporary
            .path()
            .join("engine-releases/release-1/ledger/PRISMOS_ACTIVE_STATE_LOCK.json");
        #[cfg(unix)]
        fs::set_permissions(&lock, fs::Permissions::from_mode(0o644)).unwrap();
        let mut value: serde_json::Value =
            serde_json::from_slice(&fs::read(&lock).unwrap()).unwrap();
        value.as_object_mut().unwrap().remove("source_epoch_id");
        fs::write(&lock, serde_json::to_vec(&value).unwrap()).unwrap();
        #[cfg(unix)]
        fs::set_permissions(&lock, fs::Permissions::from_mode(0o444)).unwrap();
        let error = materialize_fixture_request(&profile, &output, "systemd").unwrap_err();
        assert!(error.to_string().contains("source_epoch_id"));
    }

    #[cfg(unix)]
    #[test]
    fn active_pointer_and_immutable_release_membership_fail_closed() {
        use std::os::unix::fs::symlink;

        // A copied/stale adjacent installed lock is never authority; the
        // canonical ACTIVE member remains the sole positive source.
        let (temporary, profile, output) = fixture();
        write(
            &temporary
                .path()
                .join("installed/ledger/PRISMOS_ACTIVE_STATE_LOCK.json"),
            br#"{"schema_kind":"prismos_active_state_lock","schema_version":3,"active_release":"stale"}"#,
        );
        let admitted = materialize_fixture_request(&profile, &output, "systemd").unwrap();
        assert_eq!(
            admitted.request.runtime_authority.active_release,
            "release-1"
        );
        assert!(admitted
            .request
            .runtime_authority
            .engine
            .locator
            .starts_with(temporary.path().join("engine-releases/release-1")));

        // A copied lock reached through a symlink is not an immutable member.
        let (temporary, profile, output) = fixture();
        let release = temporary.path().join("engine-releases/release-1");
        let lock = release.join("ledger/PRISMOS_ACTIVE_STATE_LOCK.json");
        let copied = temporary.path().join("copied-active-lock.json");
        fs::copy(&lock, &copied).unwrap();
        fs::set_permissions(&release, fs::Permissions::from_mode(0o755)).unwrap();
        fs::remove_file(&lock).unwrap();
        symlink(&copied, &lock).unwrap();
        fs::set_permissions(&release, fs::Permissions::from_mode(0o555)).unwrap();
        assert!(materialize_fixture_request(&profile, &output, "systemd")
            .unwrap_err()
            .to_string()
            .contains("symlink"));

        // Lock/pointer disagreement rejects the selected release.
        let (temporary, profile, output) = fixture();
        let lock = temporary
            .path()
            .join("engine-releases/release-1/ledger/PRISMOS_ACTIVE_STATE_LOCK.json");
        fs::set_permissions(&lock, fs::Permissions::from_mode(0o644)).unwrap();
        let mut value: serde_json::Value =
            serde_json::from_slice(&fs::read(&lock).unwrap()).unwrap();
        value["active_release"] = serde_json::json!("release-2");
        fs::write(&lock, serde_json::to_vec(&value).unwrap()).unwrap();
        fs::set_permissions(&lock, fs::Permissions::from_mode(0o444)).unwrap();
        assert!(materialize_fixture_request(&profile, &output, "systemd")
            .unwrap_err()
            .to_string()
            .contains("wrong release"));

        // Engine membership cannot escape through a symlink.
        let (temporary, profile, output) = fixture();
        let release = temporary.path().join("engine-releases/release-1");
        let engine = release.join("bin/nhs_rt_full");
        fs::set_permissions(&release, fs::Permissions::from_mode(0o755)).unwrap();
        fs::set_permissions(&engine, fs::Permissions::from_mode(0o755)).unwrap();
        fs::remove_file(&engine).unwrap();
        symlink(temporary.path().join("installed/bin/nhs_rt_full"), &engine).unwrap();
        fs::set_permissions(&release, fs::Permissions::from_mode(0o555)).unwrap();
        assert!(materialize_fixture_request(&profile, &output, "systemd")
            .unwrap_err()
            .to_string()
            .contains("symlink"));

        // ACTIVE must be a one-component relative member, never an escape.
        let (temporary, profile, output) = fixture();
        let pointer = temporary.path().join("engine-releases/ACTIVE");
        fs::remove_file(&pointer).unwrap();
        symlink("../escape-release", &pointer).unwrap();
        assert!(materialize_fixture_request(&profile, &output, "systemd")
            .unwrap_err()
            .to_string()
            .contains("one relative immutable release member"));

        // Writable release roots are not admitted as immutable authority.
        let (temporary, profile, output) = fixture();
        let release = temporary.path().join("engine-releases/release-1");
        fs::set_permissions(&release, fs::Permissions::from_mode(0o755)).unwrap();
        assert!(materialize_fixture_request(&profile, &output, "systemd")
            .unwrap_err()
            .to_string()
            .contains("must not be user/group/world writable"));
    }

    #[test]
    fn output_must_be_exact_target_path_without_dot_components() {
        let (temporary, profile, output) = fixture();
        let output_root = output.parent().unwrap();
        for rejected in [
            output_root.join("target-2"),
            output_root.parent().unwrap().join("runs-prefix/target-1"),
            output_root.join("target-1/../target-1"),
            output_root.join("./target-1"),
        ] {
            let error = materialize_fixture_request(&profile, &rejected, "systemd").unwrap_err();
            assert!(
                error.to_string().contains("output"),
                "unexpected error for {}: {error}",
                rejected.display()
            );
        }
        assert!(materialize_fixture_request(&profile, &output, "systemd").is_ok());
        drop(temporary);
    }

    #[cfg(unix)]
    #[test]
    fn artifact_parent_symlink_is_rejected() {
        use std::os::unix::fs::symlink;

        let (temporary, profile, output) = fixture();
        let root = temporary.path();
        symlink(root.join("inputs"), root.join("input-alias")).unwrap();
        let descriptor_path = root.join("requests/target-1.json");
        let mut descriptor: TargetExecutionIntentV1 =
            serde_json::from_slice(&fs::read(&descriptor_path).unwrap()).unwrap();
        descriptor.topology.locator = root.join("input-alias/target.topology.json");
        fs::write(&descriptor_path, serde_json::to_vec(&descriptor).unwrap()).unwrap();
        let error = materialize_fixture_request(&profile, &output, "systemd").unwrap_err();
        assert!(error.to_string().contains("symlink component"));
    }

    #[cfg(unix)]
    #[test]
    fn existing_final_output_symlink_is_rejected() {
        use std::os::unix::fs::symlink;

        let (temporary, profile, output) = fixture();
        fs::create_dir_all(output.parent().unwrap()).unwrap();
        let redirect = temporary.path().join("redirect-output");
        fs::create_dir(&redirect).unwrap();
        symlink(&redirect, &output).unwrap();
        let error = materialize_fixture_request(&profile, &output, "systemd").unwrap_err();
        assert!(error
            .to_string()
            .contains("output contains a symlink component"));
    }

    #[test]
    fn production_materializer_rejects_unpinned_profile_bytes() {
        let (_temporary, profile, output) = fixture();
        let error = materialize_hyper_runtime_request(&profile, "target-1", &output, "systemd")
            .unwrap_err();
        assert!(error.to_string().contains("compiled canonical authority"));
    }

    #[test]
    fn canonical_profile_source_matches_compiled_authority_identity() {
        let profile = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../appliance_payload_reference/profiles/hyper-runtime-md-v1.json");
        assert_eq!(
            sha256_file_hex(&profile).unwrap(),
            CANONICAL_EXECUTION_PROFILE_SHA256
        );
    }

    #[test]
    fn explicitly_selected_sealed_proposal_materializes_distinct_branch_packet() {
        let fixture = governed_fixture();
        let request = materialize_governed_branch_hyper_runtime_request_with_profile_identity(
            &fixture.selection,
            &fixture.request,
            &fixture.profile_sha256,
            &fixture
                .selection
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .join("engine-releases/ACTIVE"),
            &cpu_fixture_protection(fixture._temporary.path()),
        )
        .unwrap();
        let branch = request.request.governed_branch.as_ref().unwrap();
        assert_eq!(request.attribution.client_surface, "agent");
        assert_eq!(
            branch.proposal_id_sha256,
            branch.proposal_id_sha256.to_ascii_lowercase()
        );
        assert_eq!(
            request.request.packet_authority.prismcode_packet.sha256,
            request.request.validated_packet_policy.packet_sha256
        );
        assert_ne!(
            request.request.packet_authority.prismcode_packet.sha256,
            "f".repeat(64)
        );
        assert_eq!(
            request.request.validated_packet_policy.policy,
            MdEnginePolicyV1::canonical_hyper_runtime()
        );
    }

    #[test]
    fn governed_branch_rejects_rank_only_missing_seal_epoch_and_request_smuggling() {
        for mutation in [
            "rank-only",
            "missing-seal",
            "wrong-epoch",
            "smuggled-packet",
        ] {
            let fixture = governed_fixture();
            match mutation {
                "rank-only" => {
                    let mut selection: serde_json::Value =
                        serde_json::from_slice(&fs::read(&fixture.selection).unwrap()).unwrap();
                    selection["selection_authority"]["authority_kind"] =
                        serde_json::json!("rank-inferred");
                    write(&fixture.selection, &serde_json::to_vec(&selection).unwrap());
                    let mut request: serde_json::Value =
                        serde_json::from_slice(&fs::read(&fixture.request).unwrap()).unwrap();
                    request["governed_branch_selection"] =
                        serde_json::to_value(sized_identity(&fixture.selection)).unwrap();
                    write(&fixture.request, &serde_json::to_vec(&request).unwrap());
                }
                "missing-seal" => {
                    let selection: GovernedBranchSelectionV1 =
                        serde_json::from_slice(&fs::read(&fixture.selection).unwrap()).unwrap();
                    fs::remove_file(&selection.proposal.proposal_set_receipt.locator).unwrap();
                }
                "wrong-epoch" => {
                    let mut selection: serde_json::Value =
                        serde_json::from_slice(&fs::read(&fixture.selection).unwrap()).unwrap();
                    selection["source_epoch_id"] = serde_json::json!("8".repeat(64));
                    write(&fixture.selection, &serde_json::to_vec(&selection).unwrap());
                    let mut request: serde_json::Value =
                        serde_json::from_slice(&fs::read(&fixture.request).unwrap()).unwrap();
                    request["source_epoch_id"] = serde_json::json!("8".repeat(64));
                    request["governed_branch_selection"] =
                        serde_json::to_value(sized_identity(&fixture.selection)).unwrap();
                    write(&fixture.request, &serde_json::to_vec(&request).unwrap());
                }
                "smuggled-packet" => {
                    let mut request: serde_json::Value =
                        serde_json::from_slice(&fs::read(&fixture.request).unwrap()).unwrap();
                    request["packet"]["sha256"] = serde_json::json!("0".repeat(64));
                    write(&fixture.request, &serde_json::to_vec(&request).unwrap());
                }
                _ => unreachable!(),
            }
            let error = materialize_governed_branch_hyper_runtime_request_with_profile_identity(
                &fixture.selection,
                &fixture.request,
                &fixture.profile_sha256,
                &fixture
                    .selection
                    .parent()
                    .unwrap()
                    .parent()
                    .unwrap()
                    .join("engine-releases/ACTIVE"),
                &cpu_fixture_protection(fixture._temporary.path()),
            )
            .unwrap_err();
            let text = error.to_string();
            assert!(
                text.contains("branch")
                    || text.contains("selection")
                    || text.contains("seal")
                    || text.contains("unavailable"),
                "{mutation} produced unexpected error: {text}"
            );
        }
    }
}
