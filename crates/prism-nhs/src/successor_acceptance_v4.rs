//! Engine-native MYC:MAX successor acceptance-v4 evidence production.
//!
//! The only production ingestion API is [`record_runtime_measurement`]. It is
//! called by `nhs_rt_full` measurement sites while the governed run is still
//! live. There is deliberately no production API that imports an observations
//! summary or converts a completed run after the fact. At successful runtime
//! closeout, [`publish_runtime_acceptance`] writes the canonical raw-record
//! journal, typed observation authority, and gate producer receipt as one
//! no-clobber transaction. Missing or duplicate members fail before any public
//! artifact name is created.

use anyhow::{anyhow, bail, Context, Result};
use prismcode::binary_runtime::opcode_id_for_name;
use prismcode::operand_ledger::RuntimeOrigin;
use prismcode::qnanl_artifact::{
    decode_packet_bytes, encode_bytes, QnanlArtifactFamily, QnanlEncodeOptions, QnanlPcodeRef,
};
use prismcode::runtime_provenance::RuntimeProvenance;
use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Number, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

pub const CONTRACT_SCHEMA: &str = "prismos_expanded_acceptance_gates";
pub const CONTRACT_VERSION: &str = "4.0.0";
pub const CONTRACT_DIRECTIVE_ID: &str = "MYC_MAX_LAMBDA_GHOST_PATHB_SUCCESSOR";
pub const CONTRACT_STATUS: &str = "ACTIVE";
pub const CONTRACT_SUBSTITUTION_POLICY: &str = "NO_CANARY_NO_PROXY_NO_WRAPPER_ONLY";
pub const CONTRACT_RELEASE_RULE: &str =
    "Rebless and seal are forbidden until every gate is PASS on one successor identity.";
pub const ACCEPTANCE_VALIDATOR_SHA256: &str =
    "603161b939a926c24c0f73465eabbfc15c3e11cc95bc6f519ae428481001ca5f";
pub const ABI_DECLARATION_CONTRACT_SHA256: &str =
    "92ce4b0c2a29671d60ee645e2935342b74bbb816f688324206e15526c491a452";
pub const TERMINAL_SCHEMA: &str = "prismos_successor_terminal_gate_receipt";
pub const TERMINAL_VERSION: u64 = 4;
pub const PRODUCER_SCHEMA_VERSION: u64 = 3;
pub const SOURCE_SCHEMA_VERSION: u64 = 3;
pub const MEASUREMENT_MEMBER_SCHEMA_VERSION: u64 = 2;
pub const MEASUREMENT_IDENTITY_ALGORITHM: &str = "RAW_RUNTIME_RECORD_LOCATOR_V1";
pub const NATIVE_PRODUCER_POLICY: &str = "CONTRACT_PINNED_NHS_RT_FULL_DIRECT_EXECUTION_V1";
pub const SOURCE_POPULATION_HASH_ALGORITHM: &str = "SHA256_CANONICAL_MEMBER_INDEX_V1";
pub const CANDIDATE_BUILD_AUTHORITY_SCHEMA: &str = "prismos_successor_candidate_build_authority";
pub const MEASUREMENT_IMPLEMENTATION_AUTHORITY_SCHEMA: &str =
    "prismos_successor_measurement_implementation_authority";
pub const CANDIDATE_BUILD_MANIFEST_SCHEMA: &str = "prismos_successor_candidate_build_manifest";
pub const MEASUREMENT_IMPLEMENTATION_MANIFEST_SCHEMA: &str =
    "prismos_successor_measurement_implementation_manifest";
pub const FRONTIER_PROBE_PANEL_AUTHORITY_SCHEMA: &str =
    "prismos_successor_frontier_probe_panel_authority";
pub const FRONTIER_PROBE_PANEL_ID: &str = "PRISM_MYC_MAX_SHALLOW_PPI_INTERACTION_BASIS_V1";
pub const FRONTIER_PROBE_PANEL_RAW_SHA256: &str =
    "ee3b4e728e09d6e5d574def1704d126ad0e75c45f3db4f3ef9cf9b3b67d48767";
pub const FRONTIER_PROBE_PANEL_SCIENTIFIC_SHA256: &str =
    "4259e8829b9392d01b476e1de06a28f35330690b8ce415fd9cf946b5165e386a";
pub const FRONTIER_PROBE_PANEL_RAW_BYTES: u64 = 1_952_937;
pub const FRONTIER_PROBE_PANEL_SPECIES_COUNT: u64 = 12;
pub const FRONTIER_PROBE_PANEL_CHEMICAL_STATE_COUNT: u64 = 15;
pub const FRONTIER_PROBE_PANEL_MATCHED_PAIR_COUNT: u64 = 6;
pub const PANEL_SCI_DIGEST_ALGORITHM: &str = "PRISM_PANEL_SCI_DIGEST_V1";
pub const AUTHORITY_MANIFEST_SCHEMA_VERSION: u64 = 1;
pub const RAW_RUNTIME_MEASUREMENT_ABI_SCHEMA: &str =
    "prismos_successor_raw_runtime_measurement_abi";
pub const RAW_RUNTIME_RECORD_SCHEMA: &str = "prismos_nhs_rt_full_runtime_measurement_record";
pub const RAW_RUNTIME_RECORD_VERSION: u64 = 1;
pub const RAW_RUNTIME_RECORD_FORMAT: &str = "UTF8_CANONICAL_JSON";
pub const RAW_RECONSTRUCTION_ALGORITHM: &str = "JSON_POINTER_EXACT_VALUE_V1";
pub const RAW_RUNTIME_MEASUREMENT_ABI_SHA256: &str =
    "d11e768aaf9bebae0aa46a39f9c60945666ee9e399613eb72b05ee7e80eaa80c";
pub const RECONSTRUCTION_METHOD_POPULATION_SHA256: &str =
    "e7f7a86d2a1fd94931d99875a1227f2a688a9bfb26b3c4c77d44cfcc9e40b26b";
pub const RECONSTRUCTION_METHOD_COUNT: usize = 654;
/// Existing governed `nhs_rt_full` minimum host-launch collapse contract.
pub const GOVERNED_HOTPATH_MIN_LAUNCH_COLLAPSE: f64 = 64.0;

const CANDIDATE_AUTHORITY_MISSING_ABI: [&str; 4] = [
    "producer binding {path,sha256,size_bytes} with basename nhs_rt_full",
    "build_manifest binding {path,sha256,size_bytes}",
    "source_snapshot binding {path,sha256,size_bytes}",
    "native build manifest field raw_runtime_measurement_abi_sha256",
];
const MEASUREMENT_AUTHORITY_MISSING_ABI: [&str; 3] = [
    "implementation binding {path,sha256,size_bytes}",
    "implementation_manifest binding {path,sha256,size_bytes}",
    "gate-field reconstruction population matching reconstruction_schema_kind(gate_id,field)",
];

pub const GATE_ORDER: [&str; 26] = [
    "G_GRAPH_RESIDENT",
    "G_GHOST_SEGMENT",
    "G_GHOST_CONSERVATION",
    "G_PHASE_BINDING",
    "G_TOPOLOGY_MOLECULES",
    "G_PREP_REPRODUCIBILITY",
    "G_HAMILTONIAN",
    "G_GRADIENT",
    "G_ENSEMBLE_BALANCE",
    "G_SLOT_ENDPOINT_REACHABILITY",
    "G_SOLVENT",
    "G_NPT_EXECUTION_PURPOSE",
    "G_TARGET_CAPACITY",
    "G_PRISMCODE_QNANL",
    "G_NATIVE_LATTICE_PATHB",
    "G_RUNTIME_OBSERVATION_ORIGIN",
    "G_EXACT_COORDINATE_COVERAGE",
    "G_FIBER",
    "G_MATCHED_CONTROLS",
    "G_CROSS_RUN_MATCHED_AGGREGATION",
    "G_SEEDS_UNCERTAINTY",
    "G_CANDIDATE_PRESERVATION",
    "G_RANKING_IDENTITY",
    "G_FULL_BENCHMARK",
    "G_CANONICAL_ENGINE",
    "G_SEAL",
];

fn require_implemented_gate(gate_id: &str) -> Result<()> {
    if !matches!(gate_id, "G_GRAPH_RESIDENT" | "G_GHOST_CONSERVATION") {
        bail!(
            "EVIDENCE_NOT_FOUND: {gate_id} has no complete engine-native Layer 05 field authority; only G_GRAPH_RESIDENT and G_GHOST_CONSERVATION are implemented"
        );
    }
    Ok(())
}

fn validate_gate_runtime_admission(
    gate_id: &str,
    require_hotpath_acceleration: bool,
    min_hotpath_launch_collapse: f64,
    ghost_lattice_intent_selected: bool,
) -> Result<()> {
    match gate_id {
        "G_GRAPH_RESIDENT" => validate_graph_hotpath_admission(
            require_hotpath_acceleration,
            min_hotpath_launch_collapse,
        ),
        "G_GHOST_CONSERVATION" if ghost_lattice_intent_selected => Ok(()),
        "G_GHOST_CONSERVATION" => bail!(
            "EVIDENCE_NOT_FOUND: G_GHOST_CONSERVATION requires an explicit or auto-selected production Ghost Phase-Lattice path"
        ),
        _ => require_implemented_gate(gate_id),
    }
}

fn validate_graph_hotpath_admission(
    require_hotpath_acceleration: bool,
    min_hotpath_launch_collapse: f64,
) -> Result<()> {
    if !require_hotpath_acceleration {
        bail!("EVIDENCE_NOT_FOUND: G_GRAPH_RESIDENT requires --require-hotpath-acceleration");
    }
    if !min_hotpath_launch_collapse.is_finite()
        || min_hotpath_launch_collapse != GOVERNED_HOTPATH_MIN_LAUNCH_COLLAPSE
    {
        bail!(
            "EVIDENCE_NOT_FOUND: G_GRAPH_RESIDENT requires the exact governed --min-hotpath-launch-collapse={GOVERNED_HOTPATH_MIN_LAUNCH_COLLAPSE}; observed={min_hotpath_launch_collapse}"
        );
    }
    Ok(())
}

/// Evaluate the governed per-stream collapse predicate without weakening it to
/// the generic PrismCode receipt's internal `> 1` hotpath predicate.
pub fn governed_graph_collapse_valid(
    require_hotpath_acceleration: bool,
    configured_min_hotpath_launch_collapse: f64,
    receipt_hotpath_status: &str,
    observed_collapse_ratio: f64,
) -> bool {
    require_hotpath_acceleration
        && configured_min_hotpath_launch_collapse == GOVERNED_HOTPATH_MIN_LAUNCH_COLLAPSE
        && receipt_hotpath_status == "PASS"
        && observed_collapse_ratio.is_finite()
        && observed_collapse_ratio >= GOVERNED_HOTPATH_MIN_LAUNCH_COLLAPSE
}

fn unresolved_gate_reason(gate_id: &str) -> (&'static str, &'static str) {
    match gate_id {
        "G_GRAPH_RESIDENT" | "G_GHOST_CONSERVATION" => (
            "SEPARATE_GOVERNED_EXECUTION_REQUIRED",
            "implemented gate requires its own governed single-gate execution and cannot inherit another gate's measurements",
        ),
        "G_CROSS_RUN_MATCHED_AGGREGATION" => (
            "CROSS_RUN_AUTHORITY_REQUIRED",
            "requires a cross-execution authority and cannot be emitted by one governed process closeout",
        ),
        "G_FULL_BENCHMARK" => (
            "FULL_BENCHMARK_AUTHORITY_REQUIRED",
            "requires the complete non-canary benchmark authority rather than one runtime invocation",
        ),
        "G_CANONICAL_ENGINE" => (
            "POST_REBLESS_CANONICAL_AUTHORITY_REQUIRED",
            "requires explicit post-acceptance rebless ordering",
        ),
        "G_SEAL" => (
            "POST_REBLESS_SEAL_READBACK_REQUIRED",
            "requires post-rebless portable seal readback ordering",
        ),
        _ => (
            "UNMAPPED_SINGLE_RUN_GATE",
            "Layer 03 has no complete direct-runtime field mapping for this single-run gate",
        ),
    }
}

fn unresolved_gate_ledger(active_gate_id: &str) -> Vec<Value> {
    GATE_ORDER
        .iter()
        .copied()
        .filter(|gate_id| *gate_id != active_gate_id)
        .map(|gate_id| {
            let (reason_class, reason) = unresolved_gate_reason(gate_id);
            serde_json::json!({
                "gate_id": gate_id,
                "status": "EVIDENCE_NOT_FOUND",
                "reason_class": reason_class,
                "reason": reason,
            })
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileBinding {
    pub path: String,
    pub sha256: String,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthorityBindings {
    pub execution_authority_sha256: String,
    pub campaign_sha256: String,
    pub topology_sha256: String,
    pub production_contract_sha256: String,
    pub panel_sha256: String,
    pub panel_scientific_sha256: String,
    pub discovery_authority_sha256: String,
    pub confirmation_authority_sha256: String,
}

impl AuthorityBindings {
    fn validate(&self) -> Result<()> {
        for (name, value) in [
            (
                "execution_authority_sha256",
                &self.execution_authority_sha256,
            ),
            ("campaign_sha256", &self.campaign_sha256),
            ("topology_sha256", &self.topology_sha256),
            (
                "production_contract_sha256",
                &self.production_contract_sha256,
            ),
            ("panel_sha256", &self.panel_sha256),
            ("panel_scientific_sha256", &self.panel_scientific_sha256),
            (
                "discovery_authority_sha256",
                &self.discovery_authority_sha256,
            ),
            (
                "confirmation_authority_sha256",
                &self.confirmation_authority_sha256,
            ),
        ] {
            require_sha256(name, value)?;
        }
        if self.panel_sha256 == self.panel_scientific_sha256 {
            bail!("panel raw and scientific SHA256 identities must be distinct");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default, clap::Args)]
pub struct SuccessorAcceptanceCloseoutArgs {
    /// Resolved acceptance-v4 contract used by this governed execution.
    #[arg(long)]
    pub successor_acceptance_gate_contract: Option<PathBuf>,
    /// Exact single gate produced by this full governed execution.
    #[arg(long)]
    pub successor_acceptance_gate_id: Option<String>,
    /// Cross-gate execution identity.
    #[arg(long)]
    pub successor_acceptance_execution_id: Option<String>,
    /// No-clobber raw runtime record journal output.
    #[arg(long)]
    pub successor_acceptance_raw_runtime_artifact: Option<PathBuf>,
    /// No-clobber typed observation authority output.
    #[arg(long)]
    pub successor_acceptance_observation_source: Option<PathBuf>,
    /// No-clobber gate producer receipt output.
    #[arg(long)]
    pub successor_acceptance_gate_evidence: Option<PathBuf>,
    /// No-clobber acceptance transaction closeout receipt output.
    #[arg(long)]
    pub successor_acceptance_transaction_closeout_receipt: Option<PathBuf>,
    #[arg(long)]
    pub successor_acceptance_execution_authority_sha256: Option<String>,
    #[arg(long)]
    pub successor_acceptance_campaign_sha256: Option<String>,
    #[arg(long)]
    pub successor_acceptance_topology_sha256: Option<String>,
    #[arg(long)]
    pub successor_acceptance_production_contract_sha256: Option<String>,
    #[arg(long)]
    pub successor_acceptance_panel_sha256: Option<String>,
    #[arg(long)]
    pub successor_acceptance_panel_scientific_sha256: Option<String>,
    #[arg(long)]
    pub successor_acceptance_discovery_authority_sha256: Option<String>,
    #[arg(long)]
    pub successor_acceptance_confirmation_authority_sha256: Option<String>,
}

#[derive(Debug, Clone, clap::Args)]
pub struct SuccessorAuthorityManifestArgs {
    /// Immutable source snapshot manifest used to build this exact ELF.
    #[arg(long)]
    pub source_snapshot: PathBuf,
    /// Exact native measurement implementation source/binary identity.
    #[arg(long)]
    pub measurement_implementation: PathBuf,
    /// Frozen frontier-probe panel bound by raw and normalized scientific identity.
    #[arg(long)]
    pub frontier_probe_panel: PathBuf,
    /// New candidate-build manifest path.
    #[arg(long)]
    pub candidate_build_manifest: PathBuf,
    /// New measurement-implementation manifest path.
    #[arg(long)]
    pub measurement_implementation_manifest: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ContractGate {
    id: String,
    pass: String,
    machine_rule_sha256: String,
    producer_schema_kind: String,
    source_schema_kind: String,
    measurement_field_population_sha256: String,
    evidence_class: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CandidateBuildAuthority {
    status: String,
    schema_kind: String,
    schema_version: u64,
    producer: Option<FileBinding>,
    build_manifest: Option<FileBinding>,
    source_snapshot: Option<FileBinding>,
    missing_abi: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MeasurementImplementationAuthority {
    status: String,
    schema_kind: String,
    schema_version: u64,
    implementation: Option<FileBinding>,
    implementation_manifest: Option<FileBinding>,
    missing_abi: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct AcceptanceContract {
    schema_kind: String,
    schema_version: String,
    directive_id: String,
    status: String,
    substitution_policy: String,
    terminal_receipt_schema: String,
    terminal_receipt_schema_version: u64,
    gate_producer_schema_version: u64,
    typed_source_schema_version: u64,
    measurement_member_schema_version: u64,
    measurement_identity_algorithm: String,
    native_producer_policy: String,
    source_population_hash_algorithm: String,
    machine_rule_hash_algorithm: String,
    acceptance_validator_sha256: String,
    candidate_build_manifest_schema_kind: String,
    candidate_build_manifest_schema_version: u64,
    measurement_implementation_manifest_schema_kind: String,
    measurement_implementation_manifest_schema_version: u64,
    raw_runtime_measurement_abi_sha256: String,
    candidate_build_authority: CandidateBuildAuthority,
    measurement_implementation_authority: MeasurementImplementationAuthority,
    raw_runtime_measurement_abi: Value,
    frontier_probe_panel_authority: FrontierProbePanelAuthority,
    gates: Vec<ContractGate>,
    release_rule: String,
}

#[derive(Debug, Clone)]
struct ContractAuthorities {
    producer: FileBinding,
    candidate_build_manifest: FileBinding,
    source_snapshot: FileBinding,
    measurement_implementation: FileBinding,
    measurement_implementation_manifest: FileBinding,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReconstructionMethod {
    pub gate_id: String,
    pub field: String,
    pub schema_kind: String,
    pub schema_version: u64,
    pub algorithm: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrontierProbePanelAuthority {
    pub schema_kind: String,
    pub schema_version: u64,
    pub panel_id: String,
    pub panel_raw_sha256: String,
    pub panel_scientific_sha256: String,
    pub raw_byte_size: u64,
    pub species_count: u64,
    pub chemical_state_count: u64,
    pub matched_pair_count: u64,
}

impl FrontierProbePanelAuthority {
    fn validate(&self) -> Result<()> {
        require_sha256("panel_raw_sha256", &self.panel_raw_sha256)?;
        require_sha256("panel_scientific_sha256", &self.panel_scientific_sha256)?;
        if self.schema_kind != FRONTIER_PROBE_PANEL_AUTHORITY_SCHEMA
            || self.schema_version != 1
            || self.panel_id != FRONTIER_PROBE_PANEL_ID
            || self.panel_raw_sha256 != FRONTIER_PROBE_PANEL_RAW_SHA256
            || self.panel_scientific_sha256 != FRONTIER_PROBE_PANEL_SCIENTIFIC_SHA256
            || self.panel_raw_sha256 == self.panel_scientific_sha256
            || self.panel_raw_sha256 == FRONTIER_PROBE_PANEL_SCIENTIFIC_SHA256
            || self.panel_scientific_sha256 == FRONTIER_PROBE_PANEL_RAW_SHA256
            || self.raw_byte_size != FRONTIER_PROBE_PANEL_RAW_BYTES
            || self.species_count != FRONTIER_PROBE_PANEL_SPECIES_COUNT
            || self.chemical_state_count != FRONTIER_PROBE_PANEL_CHEMICAL_STATE_COUNT
            || self.matched_pair_count != FRONTIER_PROBE_PANEL_MATCHED_PAIR_COUNT
        {
            bail!("frontier probe panel authority differs from the frozen release invariant");
        }
        Ok(())
    }
}

fn expected_frontier_probe_panel_authority() -> FrontierProbePanelAuthority {
    FrontierProbePanelAuthority {
        schema_kind: FRONTIER_PROBE_PANEL_AUTHORITY_SCHEMA.to_string(),
        schema_version: 1,
        panel_id: FRONTIER_PROBE_PANEL_ID.to_string(),
        panel_raw_sha256: FRONTIER_PROBE_PANEL_RAW_SHA256.to_string(),
        panel_scientific_sha256: FRONTIER_PROBE_PANEL_SCIENTIFIC_SHA256.to_string(),
        raw_byte_size: FRONTIER_PROBE_PANEL_RAW_BYTES,
        species_count: FRONTIER_PROBE_PANEL_SPECIES_COUNT,
        chemical_state_count: FRONTIER_PROBE_PANEL_CHEMICAL_STATE_COUNT,
        matched_pair_count: FRONTIER_PROBE_PANEL_MATCHED_PAIR_COUNT,
    }
}

fn sha256_bytes(raw: &[u8]) -> String {
    format!("{:x}", Sha256::digest(raw))
}

fn require_sha256(label: &str, value: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("{label} is not a lowercase SHA256");
    }
    Ok(())
}

fn canonical_float(value: f64) -> Result<String> {
    if !value.is_finite() {
        bail!("canonical JSON refuses a nonfinite float");
    }
    let mut buffer = ryu::Buffer::new();
    let rendered = buffer.format_finite(value);
    let (sign, magnitude) = if let Some(rest) = rendered.strip_prefix('-') {
        ("-", rest)
    } else {
        ("", rendered)
    };
    let (mantissa, exponent) = match magnitude.split_once(['e', 'E']) {
        Some((mantissa, exponent)) => (
            mantissa,
            exponent
                .parse::<i32>()
                .context("parse finite float exponent")?,
        ),
        None => (magnitude, 0),
    };
    let (integer, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let mut digits = format!("{integer}{fraction}");
    let mut decimal_position = integer.len() as i32 + exponent;
    while digits.len() > 1 && digits.starts_with('0') {
        digits.remove(0);
        decimal_position -= 1;
    }
    while digits.len() > 1 && digits.ends_with('0') {
        digits.pop();
    }
    let scientific_exponent = decimal_position - 1;
    let magnitude = if (-4..16).contains(&scientific_exponent) {
        if decimal_position <= 0 {
            format!("0.{}{}", "0".repeat((-decimal_position) as usize), digits)
        } else if decimal_position as usize >= digits.len() {
            format!(
                "{}{}.0",
                digits,
                "0".repeat(decimal_position as usize - digits.len())
            )
        } else {
            let split = decimal_position as usize;
            format!("{}.{}", &digits[..split], &digits[split..])
        }
    } else {
        let mantissa = if digits.len() == 1 {
            digits
        } else {
            format!("{}.{}", &digits[..1], &digits[1..])
        };
        format!("{mantissa}e{scientific_exponent:+03}")
    };
    Ok(format!("{sign}{magnitude}"))
}

fn append_canonical_json(value: &Value, output: &mut Vec<u8>) -> Result<()> {
    match value {
        Value::Null => output.extend_from_slice(b"null"),
        Value::Bool(true) => output.extend_from_slice(b"true"),
        Value::Bool(false) => output.extend_from_slice(b"false"),
        Value::Number(number) => {
            if number.is_f64() {
                output.extend_from_slice(
                    canonical_float(number.as_f64().context("decode JSON float")?)?.as_bytes(),
                );
            } else {
                output.extend_from_slice(number.to_string().as_bytes());
            }
        }
        Value::String(text) => output.extend_from_slice(serde_json::to_string(text)?.as_bytes()),
        Value::Array(values) => {
            output.push(b'[');
            for (index, member) in values.iter().enumerate() {
                if index != 0 {
                    output.push(b',');
                }
                append_canonical_json(member, output)?;
            }
            output.push(b']');
        }
        Value::Object(values) => {
            output.push(b'{');
            let mut keys = values.keys().collect::<Vec<_>>();
            keys.sort();
            for (index, key) in keys.into_iter().enumerate() {
                if index != 0 {
                    output.push(b',');
                }
                output.extend_from_slice(serde_json::to_string(key)?.as_bytes());
                output.push(b':');
                append_canonical_json(&values[key], output)?;
            }
            output.push(b'}');
        }
    }
    Ok(())
}

pub fn canonical_json<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    let value = serde_json::to_value(value)?;
    let mut output = Vec::new();
    append_canonical_json(&value, &mut output)?;
    Ok(output)
}

fn validate_frontier_probe_panel_bytes(
    raw: &[u8],
    label: &str,
) -> Result<FrontierProbePanelAuthority> {
    let raw_sha256 = sha256_bytes(raw);
    if raw_sha256 != FRONTIER_PROBE_PANEL_RAW_SHA256
        || raw.len() as u64 != FRONTIER_PROBE_PANEL_RAW_BYTES
    {
        bail!("{label} raw-byte identity differs from the frozen panel");
    }
    let panel = strict_json(raw, label)?;
    let panel_object = panel
        .as_object()
        .with_context(|| format!("{label} is not a JSON object"))?;
    if panel_object.get("panel_id").and_then(Value::as_str) != Some(FRONTIER_PROBE_PANEL_ID) {
        bail!("{label} panel_id differs from the frozen panel");
    }
    let embedded_scientific_sha256 = panel_object
        .get("panel_scientific_sha256")
        .and_then(Value::as_str)
        .context("frontier probe panel embedded scientific SHA256 absent")?;
    require_sha256(
        "frontier probe panel embedded scientific SHA256",
        embedded_scientific_sha256,
    )?;
    let mut normalized = panel_object.clone();
    normalized.remove("panel_scientific_sha256");
    let scientific_sha256 = sha256_bytes(&canonical_json(&Value::Object(normalized))?);
    if scientific_sha256 != FRONTIER_PROBE_PANEL_SCIENTIFIC_SHA256
        || embedded_scientific_sha256 != scientific_sha256
        || raw_sha256 == scientific_sha256
    {
        bail!(
            "{label} {} normalized scientific identity differs from its distinct raw identity",
            PANEL_SCI_DIGEST_ALGORITHM
        );
    }

    let species = panel_object
        .get("species")
        .and_then(Value::as_array)
        .context("frontier probe panel species array absent")?;
    let chemical_state_count = species.iter().try_fold(0_u64, |count, member| {
        let states = member
            .get("chemical_states")
            .and_then(Value::as_array)
            .context("frontier probe panel species chemical_states array absent")?;
        Ok::<u64, anyhow::Error>(count + states.len() as u64)
    })?;
    let matched_pair_count = panel_object
        .get("matched_pairs")
        .and_then(Value::as_array)
        .context("frontier probe panel matched_pairs array absent")?
        .len() as u64;
    let authority = FrontierProbePanelAuthority {
        schema_kind: FRONTIER_PROBE_PANEL_AUTHORITY_SCHEMA.to_string(),
        schema_version: 1,
        panel_id: FRONTIER_PROBE_PANEL_ID.to_string(),
        panel_raw_sha256: raw_sha256,
        panel_scientific_sha256: scientific_sha256,
        raw_byte_size: raw.len() as u64,
        species_count: species.len() as u64,
        chemical_state_count,
        matched_pair_count,
    };
    authority.validate()?;
    Ok(authority)
}

fn verify_frontier_probe_panel_binding(
    binding: &FileBinding,
    authority: &FrontierProbePanelAuthority,
    label: &str,
) -> Result<FileBinding> {
    authority.validate()?;
    let observed_binding = verify_file_binding(binding, label)?;
    let raw = read_regular_nofollow(Path::new(&binding.path), label)?;
    let observed_authority = validate_frontier_probe_panel_bytes(&raw, label)?;
    if &observed_authority != authority
        || observed_binding.sha256 != authority.panel_raw_sha256
        || observed_binding.size_bytes != authority.raw_byte_size
        || observed_binding.sha256 == authority.panel_scientific_sha256
    {
        bail!("{label} raw/scientific panel binding differs from frozen authority");
    }
    Ok(observed_binding)
}

#[derive(Debug)]
struct StrictJson(Value);

struct StrictJsonVisitor;

impl<'de> Visitor<'de> for StrictJsonVisitor {
    type Value = StrictJson;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("strict JSON without duplicate object keys")
    }

    fn visit_bool<E>(self, value: bool) -> std::result::Result<Self::Value, E> {
        Ok(StrictJson(Value::Bool(value)))
    }

    fn visit_i64<E>(self, value: i64) -> std::result::Result<Self::Value, E> {
        Ok(StrictJson(Value::Number(Number::from(value))))
    }

    fn visit_u64<E>(self, value: u64) -> std::result::Result<Self::Value, E> {
        Ok(StrictJson(Value::Number(Number::from(value))))
    }

    fn visit_f64<E: de::Error>(self, value: f64) -> std::result::Result<Self::Value, E> {
        Number::from_f64(value)
            .map(|number| StrictJson(Value::Number(number)))
            .ok_or_else(|| E::custom("nonfinite JSON number"))
    }

    fn visit_str<E: de::Error>(self, value: &str) -> std::result::Result<Self::Value, E> {
        Ok(StrictJson(Value::String(value.to_string())))
    }

    fn visit_string<E>(self, value: String) -> std::result::Result<Self::Value, E> {
        Ok(StrictJson(Value::String(value)))
    }

    fn visit_none<E>(self) -> std::result::Result<Self::Value, E> {
        Ok(StrictJson(Value::Null))
    }

    fn visit_unit<E>(self) -> std::result::Result<Self::Value, E> {
        Ok(StrictJson(Value::Null))
    }

    fn visit_seq<A: SeqAccess<'de>>(
        self,
        mut sequence: A,
    ) -> std::result::Result<Self::Value, A::Error> {
        let mut values = Vec::new();
        while let Some(StrictJson(value)) = sequence.next_element::<StrictJson>()? {
            values.push(value);
        }
        Ok(StrictJson(Value::Array(values)))
    }

    fn visit_map<A: MapAccess<'de>>(
        self,
        mut mapping: A,
    ) -> std::result::Result<Self::Value, A::Error> {
        let mut values = Map::new();
        while let Some(key) = mapping.next_key::<String>()? {
            if values.contains_key(&key) {
                return Err(de::Error::custom(format!("duplicate JSON key: {key}")));
            }
            let StrictJson(value) = mapping.next_value::<StrictJson>()?;
            values.insert(key, value);
        }
        Ok(StrictJson(Value::Object(values)))
    }
}

impl<'de> Deserialize<'de> for StrictJson {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        deserializer.deserialize_any(StrictJsonVisitor)
    }
}

fn strict_json(raw: &[u8], label: &str) -> Result<Value> {
    let mut deserializer = serde_json::Deserializer::from_slice(raw);
    let StrictJson(value) = StrictJson::deserialize(&mut deserializer)
        .with_context(|| format!("{label} is not strict JSON"))?;
    deserializer
        .end()
        .with_context(|| format!("{label} has trailing JSON data"))?;
    Ok(value)
}

fn raw_runtime_measurement_abi() -> Value {
    serde_json::json!({
        "schema_kind": RAW_RUNTIME_MEASUREMENT_ABI_SCHEMA,
        "schema_version": 1,
        "record_schema_kind": RAW_RUNTIME_RECORD_SCHEMA,
        "record_schema_version": RAW_RUNTIME_RECORD_VERSION,
        "record_format": RAW_RUNTIME_RECORD_FORMAT,
        "reconstruction_algorithm": RAW_RECONSTRUCTION_ALGORITHM,
        "locator_fields": ["byte_offset", "byte_length", "record_sha256", "record_format"],
        "record_identity_fields": [
            "schema_kind", "schema_version", "gate_id", "field", "value", "execution_id",
            "gate_contract_sha256", "successor_engine_sha256", "measurement_implementation_sha256"
        ]
    })
}

fn gate_rules() -> &'static BTreeMap<String, Value> {
    static RULES: OnceLock<BTreeMap<String, Value>> = OnceLock::new();
    RULES.get_or_init(|| {
        serde_json::from_str(include_str!("successor_acceptance_v4_rules.json"))
            .expect("embedded acceptance-v4 rule JSON must parse")
    })
}

fn producer_schema_kind(gate_id: &str) -> String {
    format!(
        "prismos_successor_{}_producer_receipt",
        gate_id.to_ascii_lowercase()
    )
}

fn source_schema_kind(gate_id: &str) -> String {
    format!(
        "prismos_successor_{}_observation_authority",
        gate_id.to_ascii_lowercase()
    )
}

fn measurement_schema_kind(gate_id: &str, field: &str) -> String {
    format!(
        "prismos_successor_{}_{}_measurement",
        gate_id.to_ascii_lowercase(),
        field.replace('.', "__")
    )
}

fn reconstruction_schema_kind(gate_id: &str, field: &str) -> String {
    format!(
        "prismos_successor_{}_{}_raw_record_reconstruction",
        gate_id.to_ascii_lowercase(),
        field.replace('.', "__")
    )
}

fn expected_evidence_class(gate_id: &str) -> &'static str {
    match gate_id {
        "G_FULL_BENCHMARK" => "FULL_NON_CANARY_RUNTIME",
        "G_CANONICAL_ENGINE" => "CANONICAL_RELEASE_GATE",
        "G_SEAL" => "PORTABLE_SEAL_VERIFICATION",
        _ => "FULL_PRODUCTION_ACCEPTANCE",
    }
}

fn required_observation_fields(gate_id: &str) -> Result<BTreeSet<String>> {
    let rules = gate_rules()
        .get(gate_id)
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow!("{gate_id} has no production machine rule"))?;
    let mut fields = BTreeSet::new();
    for key in [
        "truthy",
        "zero",
        "positive",
        "finite",
        "nonnegative",
        "sha256",
    ] {
        if let Some(values) = rules.get(key).and_then(Value::as_array) {
            for value in values {
                fields.insert(
                    value
                        .as_str()
                        .ok_or_else(|| anyhow!("{gate_id}.{key} contains a non-string field"))?
                        .to_string(),
                );
            }
        }
    }
    for key in ["minimum", "values"] {
        if let Some(values) = rules.get(key).and_then(Value::as_object) {
            fields.extend(values.keys().cloned());
        }
    }
    for key in ["equal", "sha256_equal"] {
        if let Some(groups) = rules.get(key).and_then(Value::as_array) {
            for group in groups {
                for value in group
                    .as_array()
                    .ok_or_else(|| anyhow!("{gate_id}.{key} group is not an array"))?
                {
                    fields.insert(
                        value
                            .as_str()
                            .ok_or_else(|| anyhow!("{gate_id}.{key} field is not a string"))?
                            .to_string(),
                    );
                }
            }
        }
    }
    for key in ["le_pairs", "positive_integer_list_divisible_by"] {
        if let Some(pairs) = rules.get(key).and_then(Value::as_array) {
            for pair in pairs {
                for value in pair
                    .as_array()
                    .ok_or_else(|| anyhow!("{gate_id}.{key} pair is not an array"))?
                {
                    fields.insert(
                        value
                            .as_str()
                            .ok_or_else(|| anyhow!("{gate_id}.{key} field is not a string"))?
                            .to_string(),
                    );
                }
            }
        }
    }
    if let Some(groups) = rules.get("sum_equal").and_then(Value::as_array) {
        for group in groups {
            let group = group
                .as_array()
                .ok_or_else(|| anyhow!("{gate_id}.sum_equal group is not an array"))?;
            fields.insert(
                group[0]
                    .as_str()
                    .ok_or_else(|| anyhow!("{gate_id}.sum_equal total is not a string"))?
                    .to_string(),
            );
            for value in group[1]
                .as_array()
                .ok_or_else(|| anyhow!("{gate_id}.sum_equal parts are not an array"))?
            {
                fields.insert(
                    value
                        .as_str()
                        .ok_or_else(|| anyhow!("{gate_id}.sum_equal part is not a string"))?
                        .to_string(),
                );
            }
        }
    }
    Ok(fields)
}

pub fn reconstruction_method_population() -> Result<Vec<ReconstructionMethod>> {
    let mut methods = Vec::new();
    for gate_id in GATE_ORDER {
        for field in required_observation_fields(gate_id)? {
            methods.push(ReconstructionMethod {
                gate_id: gate_id.to_string(),
                schema_kind: reconstruction_schema_kind(gate_id, &field),
                field,
                schema_version: 1,
                algorithm: RAW_RECONSTRUCTION_ALGORITHM.to_string(),
            });
        }
    }
    if methods.len() != RECONSTRUCTION_METHOD_COUNT {
        bail!(
            "embedded reconstruction population count differs: expected={} observed={}",
            RECONSTRUCTION_METHOD_COUNT,
            methods.len()
        );
    }
    let hash = sha256_bytes(&canonical_json(&methods)?);
    if hash != RECONSTRUCTION_METHOD_POPULATION_SHA256 {
        bail!(
            "embedded reconstruction population hash differs: expected={} observed={}",
            RECONSTRUCTION_METHOD_POPULATION_SHA256,
            hash
        );
    }
    Ok(methods)
}

fn measurement_field_population_sha256(gate_id: &str) -> Result<String> {
    let population = required_observation_fields(gate_id)?
        .into_iter()
        .map(|field| {
            serde_json::json!({
                "field": field,
                "measurement_schema_kind": measurement_schema_kind(gate_id, &field),
            })
        })
        .collect::<Vec<_>>();
    Ok(sha256_bytes(&canonical_json(&population)?))
}

fn machine_rule_sha256(gate_id: &str) -> Result<String> {
    let rule = gate_rules()
        .get(gate_id)
        .ok_or_else(|| anyhow!("{gate_id} has no production machine rule"))?;
    Ok(sha256_bytes(&canonical_json(rule)?))
}

fn read_regular_nofollow(path: &Path, label: &str) -> Result<Vec<u8>> {
    if !path.is_absolute() {
        bail!("{label} path must be absolute: {}", path.display());
    }
    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("inspect {label} {}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        bail!(
            "{label} must be a non-symlink regular file: {}",
            path.display()
        );
    }
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(path)
        .with_context(|| format!("open {label} without following links {}", path.display()))?;
    let mut raw = Vec::new();
    file.read_to_end(&mut raw)?;
    Ok(raw)
}

pub fn bind_file(path: &Path, label: &str) -> Result<FileBinding> {
    let raw = read_regular_nofollow(path, label)?;
    Ok(FileBinding {
        path: path.display().to_string(),
        sha256: sha256_bytes(&raw),
        size_bytes: raw.len() as u64,
    })
}

fn verify_file_binding(binding: &FileBinding, label: &str) -> Result<FileBinding> {
    require_sha256(&format!("{label}.sha256"), &binding.sha256)?;
    let observed = bind_file(Path::new(&binding.path), label)?;
    if &observed != binding {
        bail!("{label} live identity mismatch: expected={binding:?} observed={observed:?}");
    }
    Ok(observed)
}

fn bind_current_executable() -> Result<FileBinding> {
    let path = std::env::current_exe()?.canonicalize()?;
    let binding = bind_file(&path, "directly executing nhs_rt_full ELF")?;
    let metadata = std::fs::symlink_metadata(&path)?;
    if metadata.permissions().mode() & 0o111 == 0 {
        bail!("directly executing nhs_rt_full ELF lacks an executable mode bit");
    }
    let process_metadata = std::fs::metadata("/proc/self/exe")?;
    if metadata.dev() != process_metadata.dev() || metadata.ino() != process_metadata.ino() {
        bail!("current executable path does not identify the directly executing ELF inode");
    }
    let raw = read_regular_nofollow(&path, "directly executing nhs_rt_full ELF")?;
    if raw.len() < 4 || &raw[..4] != b"\x7fELF" {
        bail!("directly executing producer is not an ELF artifact");
    }
    if path.file_name().and_then(|value| value.to_str()) != Some("nhs_rt_full") {
        bail!("directly executing producer basename is not nhs_rt_full");
    }
    Ok(binding)
}

fn normalized_output_path(path: &Path, label: &str) -> Result<PathBuf> {
    if !path.is_absolute() {
        bail!("{label} path must be absolute: {}", path.display());
    }
    if path
        .components()
        .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
    {
        bail!(
            "{label} path may not contain dot components: {}",
            path.display()
        );
    }
    let name = path
        .file_name()
        .ok_or_else(|| anyhow!("{label} path has no file name"))?;
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("{label} path has no parent"))?;
    let parent_name = parent
        .file_name()
        .ok_or_else(|| anyhow!("{label} transaction directory has no name"))?;
    let public_parent = parent
        .parent()
        .ok_or_else(|| anyhow!("{label} transaction directory has no public parent"))?
        .canonicalize()
        .with_context(|| format!("resolve existing {label} public parent"))?;
    let metadata = std::fs::symlink_metadata(&public_parent)?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        bail!(
            "{label} public parent is not a real directory: {}",
            public_parent.display()
        );
    }
    Ok(public_parent.join(parent_name).join(name))
}

fn file_binding_for_bytes(path: &Path, raw: &[u8]) -> FileBinding {
    FileBinding {
        path: path.display().to_string(),
        sha256: sha256_bytes(raw),
        size_bytes: raw.len() as u64,
    }
}

fn manifest_document(schema_kind: &str, body: Value) -> Result<Value> {
    Ok(serde_json::json!({
        "schema_kind": schema_kind,
        "schema_version": AUTHORITY_MANIFEST_SCHEMA_VERSION,
        "manifest_sha256": sha256_bytes(&canonical_json(&body)?),
        "body": body,
    }))
}

fn exact_object<'a>(
    value: &'a Value,
    expected: &[&str],
    label: &str,
) -> Result<&'a Map<String, Value>> {
    let object = value
        .as_object()
        .ok_or_else(|| anyhow!("{label} is not an object"))?;
    let observed = object.keys().map(String::as_str).collect::<BTreeSet<_>>();
    let expected = expected.iter().copied().collect::<BTreeSet<_>>();
    if observed != expected {
        bail!(
            "{label} field population differs: missing={:?} extra={:?}",
            expected.difference(&observed).collect::<Vec<_>>(),
            observed.difference(&expected).collect::<Vec<_>>()
        );
    }
    Ok(object)
}

fn binding_from_value(value: &Value, label: &str) -> Result<FileBinding> {
    exact_object(value, &["path", "sha256", "size_bytes"], label)?;
    serde_json::from_value(value.clone()).with_context(|| format!("decode {label}"))
}

fn verify_manifest_document(
    binding: &FileBinding,
    schema_kind: &str,
    label: &str,
) -> Result<Value> {
    verify_file_binding(binding, label)?;
    let raw = read_regular_nofollow(Path::new(&binding.path), label)?;
    let document = strict_json(&raw, label)?;
    let object = exact_object(
        &document,
        &["schema_kind", "schema_version", "manifest_sha256", "body"],
        label,
    )?;
    if object.get("schema_kind").and_then(Value::as_str) != Some(schema_kind)
        || object.get("schema_version").and_then(Value::as_u64)
            != Some(AUTHORITY_MANIFEST_SCHEMA_VERSION)
    {
        bail!("{label} schema identity mismatch");
    }
    let body = object.get("body").context("manifest body absent")?.clone();
    let expected = sha256_bytes(&canonical_json(&body)?);
    if object.get("manifest_sha256").and_then(Value::as_str) != Some(expected.as_str()) {
        bail!("{label} canonical body hash mismatch");
    }
    Ok(body)
}

fn validate_candidate_build_manifest(
    binding: &FileBinding,
    producer: &FileBinding,
    source_snapshot: &FileBinding,
) -> Result<()> {
    let body = verify_manifest_document(
        binding,
        CANDIDATE_BUILD_MANIFEST_SCHEMA,
        "candidate build manifest",
    )?;
    let object = exact_object(
        &body,
        &[
            "candidate_engine",
            "source_snapshot",
            "raw_runtime_measurement_abi_sha256",
            "frontier_probe_panel_authority",
            "panel_binding",
        ],
        "candidate build manifest body",
    )?;
    let candidate = binding_from_value(
        object
            .get("candidate_engine")
            .context("candidate_engine absent")?,
        "candidate build manifest candidate_engine",
    )?;
    let snapshot = binding_from_value(
        object
            .get("source_snapshot")
            .context("source_snapshot absent")?,
        "candidate build manifest source_snapshot",
    )?;
    if &verify_file_binding(&candidate, "candidate build manifest candidate_engine")? != producer {
        bail!("candidate build manifest binds a different candidate engine");
    }
    if &verify_file_binding(&snapshot, "candidate build manifest source_snapshot")?
        != source_snapshot
    {
        bail!("candidate build manifest binds a different source snapshot");
    }
    if object
        .get("raw_runtime_measurement_abi_sha256")
        .and_then(Value::as_str)
        != Some(RAW_RUNTIME_MEASUREMENT_ABI_SHA256)
    {
        bail!("candidate build manifest exports a different raw runtime measurement ABI");
    }
    let panel_authority: FrontierProbePanelAuthority = serde_json::from_value(
        object
            .get("frontier_probe_panel_authority")
            .context("frontier_probe_panel_authority absent")?
            .clone(),
    )?;
    panel_authority.validate()?;
    if panel_authority != expected_frontier_probe_panel_authority() {
        bail!("candidate build manifest panel authority differs from frozen invariant");
    }
    let panel_binding = binding_from_value(
        object
            .get("panel_binding")
            .context("panel_binding absent")?,
        "candidate build manifest panel_binding",
    )?;
    verify_frontier_probe_panel_binding(
        &panel_binding,
        &panel_authority,
        "candidate build manifest frontier probe panel",
    )?;
    Ok(())
}

fn validate_measurement_implementation_manifest(
    binding: &FileBinding,
    implementation: &FileBinding,
    producer: &FileBinding,
) -> Result<()> {
    let body = verify_manifest_document(
        binding,
        MEASUREMENT_IMPLEMENTATION_MANIFEST_SCHEMA,
        "measurement implementation manifest",
    )?;
    let object = exact_object(
        &body,
        &[
            "measurement_implementation",
            "candidate_engine_sha256",
            "raw_runtime_measurement_abi_sha256",
            "reconstruction_method_population_sha256",
            "reconstruction_methods",
        ],
        "measurement implementation manifest body",
    )?;
    let bound_implementation = binding_from_value(
        object
            .get("measurement_implementation")
            .context("measurement_implementation absent")?,
        "measurement implementation manifest implementation",
    )?;
    if &verify_file_binding(
        &bound_implementation,
        "measurement implementation manifest implementation",
    )? != implementation
    {
        bail!("measurement implementation manifest binds a different implementation");
    }
    if object
        .get("candidate_engine_sha256")
        .and_then(Value::as_str)
        != Some(producer.sha256.as_str())
        || object
            .get("raw_runtime_measurement_abi_sha256")
            .and_then(Value::as_str)
            != Some(RAW_RUNTIME_MEASUREMENT_ABI_SHA256)
        || object
            .get("reconstruction_method_population_sha256")
            .and_then(Value::as_str)
            != Some(RECONSTRUCTION_METHOD_POPULATION_SHA256)
    {
        bail!("measurement implementation manifest semantic identity mismatch");
    }
    let methods: Vec<ReconstructionMethod> = serde_json::from_value(
        object
            .get("reconstruction_methods")
            .context("reconstruction_methods absent")?
            .clone(),
    )?;
    if methods != reconstruction_method_population()? {
        bail!("measurement implementation reconstruction population differs");
    }
    Ok(())
}

fn validate_contract(path: &Path) -> Result<(AcceptanceContract, Vec<u8>, ContractAuthorities)> {
    let raw = read_regular_nofollow(path, "expanded acceptance contract")?;
    let contract: AcceptanceContract =
        serde_yaml::from_slice(&raw).context("expanded acceptance contract is not strict YAML")?;
    if contract.schema_kind != CONTRACT_SCHEMA
        || contract.schema_version != CONTRACT_VERSION
        || contract.directive_id != CONTRACT_DIRECTIVE_ID
        || contract.status != CONTRACT_STATUS
        || contract.substitution_policy != CONTRACT_SUBSTITUTION_POLICY
        || contract.release_rule != CONTRACT_RELEASE_RULE
    {
        bail!("expanded acceptance contract identity or active policy differs");
    }
    let exact_metadata = contract.terminal_receipt_schema == TERMINAL_SCHEMA
        && contract.terminal_receipt_schema_version == TERMINAL_VERSION
        && contract.gate_producer_schema_version == PRODUCER_SCHEMA_VERSION
        && contract.typed_source_schema_version == SOURCE_SCHEMA_VERSION
        && contract.measurement_member_schema_version == MEASUREMENT_MEMBER_SCHEMA_VERSION
        && contract.measurement_identity_algorithm == MEASUREMENT_IDENTITY_ALGORITHM
        && contract.native_producer_policy == NATIVE_PRODUCER_POLICY
        && contract.source_population_hash_algorithm == SOURCE_POPULATION_HASH_ALGORITHM
        && contract.machine_rule_hash_algorithm == "SHA256_CANONICAL_JSON_V1"
        && contract.acceptance_validator_sha256 == ACCEPTANCE_VALIDATOR_SHA256
        && contract.candidate_build_manifest_schema_kind == CANDIDATE_BUILD_MANIFEST_SCHEMA
        && contract.candidate_build_manifest_schema_version == AUTHORITY_MANIFEST_SCHEMA_VERSION
        && contract.measurement_implementation_manifest_schema_kind
            == MEASUREMENT_IMPLEMENTATION_MANIFEST_SCHEMA
        && contract.measurement_implementation_manifest_schema_version
            == AUTHORITY_MANIFEST_SCHEMA_VERSION
        && contract.raw_runtime_measurement_abi_sha256 == RAW_RUNTIME_MEASUREMENT_ABI_SHA256
        && contract.raw_runtime_measurement_abi == raw_runtime_measurement_abi()
        && contract.frontier_probe_panel_authority.validate().is_ok()
        && contract.frontier_probe_panel_authority == expected_frontier_probe_panel_authority();
    if !exact_metadata {
        bail!("expanded acceptance metadata differs from engine-native v4 authority");
    }
    if contract.gates.len() != GATE_ORDER.len() {
        bail!("expanded acceptance gate population differs");
    }
    for (index, gate) in contract.gates.iter().enumerate() {
        let gate_id = GATE_ORDER[index];
        if gate.id != gate_id
            || gate.pass.trim().is_empty()
            || gate.machine_rule_sha256 != machine_rule_sha256(gate_id)?
            || gate.producer_schema_kind != producer_schema_kind(gate_id)
            || gate.source_schema_kind != source_schema_kind(gate_id)
            || gate.measurement_field_population_sha256
                != measurement_field_population_sha256(gate_id)?
            || gate.evidence_class != expected_evidence_class(gate_id)
        {
            bail!("{gate_id} contract row differs from embedded machine authority");
        }
    }
    let candidate = &contract.candidate_build_authority;
    let measurement = &contract.measurement_implementation_authority;
    if candidate.schema_kind != CANDIDATE_BUILD_AUTHORITY_SCHEMA
        || candidate.schema_version != 1
        || candidate.missing_abi != CANDIDATE_AUTHORITY_MISSING_ABI
        || measurement.schema_kind != MEASUREMENT_IMPLEMENTATION_AUTHORITY_SCHEMA
        || measurement.schema_version != 1
        || measurement.missing_abi != MEASUREMENT_AUTHORITY_MISSING_ABI
    {
        bail!("expanded acceptance authority declaration differs");
    }
    if candidate.status != "RESOLVED" || measurement.status != "RESOLVED" {
        bail!(
            "EVIDENCE_NOT_FOUND: unresolved native successor measurement ABI: {}",
            CANDIDATE_AUTHORITY_MISSING_ABI
                .iter()
                .chain(MEASUREMENT_AUTHORITY_MISSING_ABI.iter())
                .copied()
                .collect::<Vec<_>>()
                .join("; ")
        );
    }
    let authorities = ContractAuthorities {
        producer: verify_file_binding(
            candidate
                .producer
                .as_ref()
                .context("resolved producer absent")?,
            "candidate_build_authority.producer",
        )?,
        candidate_build_manifest: verify_file_binding(
            candidate
                .build_manifest
                .as_ref()
                .context("resolved build manifest absent")?,
            "candidate_build_authority.build_manifest",
        )?,
        source_snapshot: verify_file_binding(
            candidate
                .source_snapshot
                .as_ref()
                .context("resolved source snapshot absent")?,
            "candidate_build_authority.source_snapshot",
        )?,
        measurement_implementation: verify_file_binding(
            measurement
                .implementation
                .as_ref()
                .context("resolved measurement implementation absent")?,
            "measurement_implementation_authority.implementation",
        )?,
        measurement_implementation_manifest: verify_file_binding(
            measurement
                .implementation_manifest
                .as_ref()
                .context("resolved measurement implementation manifest absent")?,
            "measurement_implementation_authority.implementation_manifest",
        )?,
    };
    if Path::new(&authorities.producer.path)
        .file_name()
        .and_then(|name| name.to_str())
        != Some("nhs_rt_full")
    {
        bail!("candidate build authority producer basename is not nhs_rt_full");
    }
    validate_candidate_build_manifest(
        &authorities.candidate_build_manifest,
        &authorities.producer,
        &authorities.source_snapshot,
    )?;
    validate_measurement_implementation_manifest(
        &authorities.measurement_implementation_manifest,
        &authorities.measurement_implementation,
        &authorities.producer,
    )?;
    Ok((contract, raw, authorities))
}

fn numeric(value: &Value, label: &str) -> Result<f64> {
    if value.is_boolean() {
        bail!("{label} is boolean, not numeric");
    }
    let number = value
        .as_f64()
        .ok_or_else(|| anyhow!("{label} is not numeric"))?;
    if !number.is_finite() {
        bail!("{label} is not finite");
    }
    Ok(number)
}

fn integer(value: &Value, label: &str) -> Result<i64> {
    value
        .as_i64()
        .ok_or_else(|| anyhow!("{label} is not an integer"))
}

fn observation<'a>(
    values: &'a BTreeMap<String, Value>,
    field: &str,
    gate_id: &str,
) -> Result<&'a Value> {
    values
        .get(field)
        .ok_or_else(|| anyhow!("{gate_id} observation is absent: {field}"))
}

fn count_observation_name(name: &str) -> bool {
    name.ends_with("_count")
        || name.ends_with("_atoms")
        || name.ends_with("_steps")
        || name.ends_with("_replicates")
        || name.ends_with("_molecules")
        || name.ends_with("_nodes")
}

fn validate_gate_observations(gate_id: &str, values: &BTreeMap<String, Value>) -> Result<()> {
    let required = required_observation_fields(gate_id)?;
    let observed = values.keys().cloned().collect::<BTreeSet<_>>();
    if observed != required {
        bail!(
            "EVIDENCE_NOT_FOUND: {gate_id} exact observation population differs: missing={:?} extra={:?}",
            required.difference(&observed).collect::<Vec<_>>(),
            observed.difference(&required).collect::<Vec<_>>()
        );
    }
    for (name, value) in values {
        if count_observation_name(name) && integer(value, name)? < 0 {
            bail!("{gate_id} count observation is negative: {name}");
        }
    }
    let rules = gate_rules()[gate_id]
        .as_object()
        .context("machine rule is not an object")?;
    for name in rules
        .get("truthy")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let name = name.as_str().context("truthy field is not a string")?;
        if observation(values, name, gate_id)? != &Value::Bool(true) {
            bail!("{gate_id} required observation is not true: {name}");
        }
    }
    for (rule, predicate, message) in [
        (
            "zero",
            fn_zero as fn(f64) -> bool,
            "required zero observation is nonzero",
        ),
        (
            "positive",
            fn_positive,
            "required positive observation is not positive",
        ),
        (
            "nonnegative",
            fn_nonnegative,
            "required nonnegative observation is negative",
        ),
        (
            "finite",
            fn_finite,
            "required finite observation is nonfinite",
        ),
    ] {
        for name in rules
            .get(rule)
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let name = name.as_str().context("numeric field is not a string")?;
            if !predicate(numeric(observation(values, name, gate_id)?, name)?) {
                bail!("{gate_id} {message}: {name}");
            }
        }
    }
    if let Some(minima) = rules.get("minimum").and_then(Value::as_object) {
        for (name, minimum) in minima {
            if numeric(observation(values, name, gate_id)?, name)? < numeric(minimum, "minimum")? {
                bail!("{gate_id} observation is below minimum: {name}");
            }
        }
    }
    if let Some(expected_values) = rules.get("values").and_then(Value::as_object) {
        for (name, expected) in expected_values {
            if observation(values, name, gate_id)? != expected {
                bail!("{gate_id} observation differs from required value: {name}");
            }
        }
    }
    for group in rules
        .get("equal")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let names = group.as_array().context("equal group is not an array")?;
        let first = observation(values, names[0].as_str().context("equal field")?, gate_id)?;
        if names.iter().skip(1).any(|name| {
            observation(values, name.as_str().unwrap_or(""), gate_id).ok() != Some(first)
        }) {
            bail!("{gate_id} conservation equality failed");
        }
    }
    for group in rules
        .get("sum_equal")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let group = group
            .as_array()
            .context("sum_equal group is not an array")?;
        let total_name = group[0].as_str().context("sum_equal total field")?;
        let total = integer(observation(values, total_name, gate_id)?, total_name)?;
        let mut sum = 0_i64;
        for part in group[1].as_array().context("sum_equal parts")? {
            let name = part.as_str().context("sum_equal part field")?;
            let value = integer(observation(values, name, gate_id)?, name)?;
            if value < 0 {
                bail!("{gate_id} population member is negative: {name}");
            }
            sum = sum.checked_add(value).context("sum_equal overflow")?;
        }
        if total < 0 || sum != total {
            bail!("{gate_id} population conservation failed: {total_name}");
        }
    }
    for pair in rules
        .get("le_pairs")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let pair = pair.as_array().context("le pair is not an array")?;
        let measured_name = pair[0].as_str().context("le measured field")?;
        let tolerance_name = pair[1].as_str().context("le tolerance field")?;
        let measured = numeric(observation(values, measured_name, gate_id)?, measured_name)?;
        let tolerance = numeric(
            observation(values, tolerance_name, gate_id)?,
            tolerance_name,
        )?;
        if measured < 0.0 || tolerance < 0.0 || measured > tolerance {
            bail!("{gate_id} error exceeds declared tolerance: {measured_name}>{tolerance_name}");
        }
    }
    for name in rules
        .get("sha256")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let name = name.as_str().context("sha256 field")?;
        require_sha256(
            name,
            observation(values, name, gate_id)?.as_str().unwrap_or(""),
        )?;
    }
    for group in rules
        .get("sha256_equal")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let names = group.as_array().context("sha256_equal group")?;
        let mut hashes = Vec::new();
        for name in names {
            let name = name.as_str().context("sha256_equal field")?;
            let hash = observation(values, name, gate_id)?.as_str().unwrap_or("");
            require_sha256(name, hash)?;
            hashes.push(hash);
        }
        if hashes.iter().skip(1).any(|hash| hash != &hashes[0]) {
            bail!("{gate_id} SHA256 join failed");
        }
    }
    for pair in rules
        .get("positive_integer_list_divisible_by")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let pair = pair.as_array().context("divisibility pair")?;
        let list_name = pair[0].as_str().context("integer list field")?;
        let divisor_name = pair[1].as_str().context("divisor field")?;
        let divisor = integer(observation(values, divisor_name, gate_id)?, divisor_name)?;
        let list = observation(values, list_name, gate_id)?
            .as_array()
            .context("positive integer list")?;
        if divisor <= 0
            || list.is_empty()
            || list.iter().any(|member| {
                member
                    .as_i64()
                    .map(|value| value <= 0 || value % divisor != 0)
                    .unwrap_or(true)
            })
        {
            bail!("{gate_id} positive integer lag grid is not divisible by {divisor_name}");
        }
    }
    Ok(())
}

fn fn_zero(value: f64) -> bool {
    value == 0.0
}
fn fn_positive(value: f64) -> bool {
    value > 0.0
}
fn fn_nonnegative(value: f64) -> bool {
    value >= 0.0
}
fn fn_finite(value: f64) -> bool {
    value.is_finite()
}

fn measurement_value_type(value: &Value) -> Result<&'static str> {
    match value {
        Value::Bool(_) => Ok("BOOLEAN"),
        Value::Number(number) if number.is_i64() || number.is_u64() => Ok("INTEGER"),
        Value::Number(number) if number.as_f64().is_some_and(f64::is_finite) => Ok("FLOAT"),
        Value::String(_) => Ok("STRING"),
        Value::Array(members)
            if members
                .iter()
                .all(|member| member.as_i64().is_some() || member.as_u64().is_some()) =>
        {
            Ok("INTEGER_LIST")
        }
        _ => bail!("measurement value has no admitted JSON scalar/list type"),
    }
}

#[derive(Debug, Clone)]
struct InvocationIdentity {
    executable: FileBinding,
    command_identity_sha256: String,
}

fn invocation_identity(executable: FileBinding) -> Result<InvocationIdentity> {
    let mut argv = std::env::args_os()
        .map(|value| {
            value
                .into_string()
                .map_err(|_| anyhow!("nhs_rt_full argv contains non-UTF8 bytes"))
        })
        .collect::<Result<Vec<_>>>()?;
    if argv.is_empty() {
        bail!("nhs_rt_full argv is empty");
    }
    argv[0] = executable.path.clone();
    let cwd = std::env::current_dir()?.canonicalize()?;
    if !cwd.is_dir() {
        bail!("nhs_rt_full current working directory is not a live directory");
    }
    let cwd = cwd.display().to_string();
    let identity = serde_json::json!({
        "argv": argv,
        "cwd": cwd,
        "executable": executable,
    });
    let command_identity_sha256 = sha256_bytes(&canonical_json(&identity)?);
    let executable = serde_json::from_value(identity["executable"].clone())?;
    Ok(InvocationIdentity {
        executable,
        command_identity_sha256,
    })
}

#[derive(Debug)]
struct RuntimeSession {
    gate_id: String,
    execution_id: String,
    gate_contract_sha256: String,
    contract_path: PathBuf,
    contract_raw: Vec<u8>,
    authorities: ContractAuthorities,
    authority_bindings: AuthorityBindings,
    invocation: InvocationIdentity,
    raw_runtime_path: PathBuf,
    observation_source_path: PathBuf,
    gate_evidence_path: PathBuf,
    transaction_closeout_receipt_path: PathBuf,
    values: BTreeMap<String, Value>,
    runtime_provenance: RuntimeProvenance,
    graph_opcode_counts_by_stream: BTreeMap<u32, BTreeMap<String, u64>>,
    ghost_conservation_records_by_stream: Option<BTreeMap<u16, u64>>,
}

#[derive(Debug)]
enum RecorderState {
    Unconfigured,
    Active(RuntimeSession),
    PublishedSuccess,
    PublishedFailure,
}

static RECORDER: OnceLock<Mutex<RecorderState>> = OnceLock::new();

const TERMINAL_UNCLAIMED: u8 = 0;
const TERMINAL_SUCCESS: u8 = 1;
const TERMINAL_FAILURE: u8 = 2;

#[derive(Debug)]
struct TerminalCoordinator {
    owner: AtomicU8,
}

impl TerminalCoordinator {
    const fn new() -> Self {
        Self {
            owner: AtomicU8::new(TERMINAL_UNCLAIMED),
        }
    }

    fn try_claim(&self, owner: u8) -> bool {
        self.owner
            .compare_exchange(
                TERMINAL_UNCLAIMED,
                owner,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    fn release_failed_publication(&self, owner: u8) {
        let _ = self.owner.compare_exchange(
            owner,
            TERMINAL_UNCLAIMED,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }
}

static TERMINAL_COORDINATOR: TerminalCoordinator = TerminalCoordinator::new();

fn recorder() -> &'static Mutex<RecorderState> {
    RECORDER.get_or_init(|| Mutex::new(RecorderState::Unconfigured))
}

fn configured_option_count(args: &SuccessorAcceptanceCloseoutArgs) -> usize {
    [
        args.successor_acceptance_gate_contract.is_some(),
        args.successor_acceptance_gate_id.is_some(),
        args.successor_acceptance_execution_id.is_some(),
        args.successor_acceptance_raw_runtime_artifact.is_some(),
        args.successor_acceptance_observation_source.is_some(),
        args.successor_acceptance_gate_evidence.is_some(),
        args.successor_acceptance_transaction_closeout_receipt
            .is_some(),
        args.successor_acceptance_execution_authority_sha256
            .is_some(),
        args.successor_acceptance_campaign_sha256.is_some(),
        args.successor_acceptance_topology_sha256.is_some(),
        args.successor_acceptance_production_contract_sha256
            .is_some(),
        args.successor_acceptance_panel_sha256.is_some(),
        args.successor_acceptance_panel_scientific_sha256.is_some(),
        args.successor_acceptance_discovery_authority_sha256
            .is_some(),
        args.successor_acceptance_confirmation_authority_sha256
            .is_some(),
    ]
    .into_iter()
    .filter(|configured| *configured)
    .count()
}

/// Configure a single engine-native gate closeout before any GPU work begins.
///
/// The option set is all-or-none. An ordinary `nhs_rt_full` invocation with
/// none of these options remains byte-for-byte outside the acceptance path.
pub fn initialize_runtime_acceptance(
    args: &SuccessorAcceptanceCloseoutArgs,
    require_hotpath_acceleration: bool,
    min_hotpath_launch_collapse: f64,
    ghost_lattice_intent_selected: bool,
    runtime_provenance: RuntimeProvenance,
) -> Result<bool> {
    let configured = configured_option_count(args);
    if configured == 0 {
        return Ok(false);
    }
    if configured != 15 {
        bail!(
            "successor acceptance closeout options are all-or-none: configured={configured} required=15"
        );
    }
    let gate_id = args
        .successor_acceptance_gate_id
        .as_ref()
        .context("acceptance gate id absent")?
        .clone();
    if !GATE_ORDER.contains(&gate_id.as_str()) {
        bail!("unknown acceptance-v4 gate: {gate_id}");
    }
    require_implemented_gate(&gate_id)?;
    validate_gate_runtime_admission(
        &gate_id,
        require_hotpath_acceleration,
        min_hotpath_launch_collapse,
        ghost_lattice_intent_selected,
    )?;
    let contract_path = args
        .successor_acceptance_gate_contract
        .as_ref()
        .context("acceptance gate contract absent")?
        .canonicalize()?;
    let execution_id = args
        .successor_acceptance_execution_id
        .as_ref()
        .context("acceptance execution id absent")?
        .trim()
        .to_string();
    if execution_id.is_empty() {
        bail!("acceptance execution id is empty");
    }
    let (_contract, contract_raw, authorities) = validate_contract(&contract_path)?;
    let current_executable = bind_current_executable()?;
    if current_executable != authorities.producer {
        bail!("directly executing nhs_rt_full differs from contract-pinned producer");
    }
    let authority_bindings = AuthorityBindings {
        execution_authority_sha256: args
            .successor_acceptance_execution_authority_sha256
            .clone()
            .unwrap(),
        campaign_sha256: args.successor_acceptance_campaign_sha256.clone().unwrap(),
        topology_sha256: args.successor_acceptance_topology_sha256.clone().unwrap(),
        production_contract_sha256: args
            .successor_acceptance_production_contract_sha256
            .clone()
            .unwrap(),
        panel_sha256: args.successor_acceptance_panel_sha256.clone().unwrap(),
        panel_scientific_sha256: args
            .successor_acceptance_panel_scientific_sha256
            .clone()
            .unwrap(),
        discovery_authority_sha256: args
            .successor_acceptance_discovery_authority_sha256
            .clone()
            .unwrap(),
        confirmation_authority_sha256: args
            .successor_acceptance_confirmation_authority_sha256
            .clone()
            .unwrap(),
    };
    authority_bindings.validate()?;
    let raw_runtime_path = normalized_output_path(
        args.successor_acceptance_raw_runtime_artifact
            .as_ref()
            .unwrap(),
        "raw runtime artifact",
    )?;
    let observation_source_path = normalized_output_path(
        args.successor_acceptance_observation_source
            .as_ref()
            .unwrap(),
        "typed observation authority",
    )?;
    let gate_evidence_path = normalized_output_path(
        args.successor_acceptance_gate_evidence.as_ref().unwrap(),
        "gate producer receipt",
    )?;
    let transaction_closeout_receipt_path = normalized_output_path(
        args.successor_acceptance_transaction_closeout_receipt
            .as_ref()
            .unwrap(),
        "acceptance transaction closeout receipt",
    )?;
    let paths = [
        &raw_runtime_path,
        &observation_source_path,
        &gate_evidence_path,
        &transaction_closeout_receipt_path,
    ];
    if paths
        .iter()
        .map(|path| path.as_path())
        .collect::<BTreeSet<_>>()
        .len()
        != paths.len()
    {
        bail!("acceptance closeout output paths are not distinct");
    }
    let parents = paths
        .iter()
        .map(|path| path.parent().unwrap())
        .collect::<BTreeSet<_>>();
    if parents.len() != 1 {
        bail!("acceptance closeout outputs must be siblings in one transaction directory");
    }
    let transaction_root = raw_runtime_path.parent().unwrap();
    if transaction_root.exists() || std::fs::symlink_metadata(transaction_root).is_ok() {
        bail!(
            "acceptance closeout transaction directory must be new: {}",
            transaction_root.display()
        );
    }
    let invocation = invocation_identity(current_executable)?;
    let session = RuntimeSession {
        gate_id,
        execution_id,
        gate_contract_sha256: sha256_bytes(&contract_raw),
        contract_path,
        contract_raw,
        authorities,
        authority_bindings,
        invocation,
        raw_runtime_path,
        observation_source_path,
        gate_evidence_path,
        transaction_closeout_receipt_path,
        values: BTreeMap::new(),
        runtime_provenance,
        graph_opcode_counts_by_stream: BTreeMap::new(),
        ghost_conservation_records_by_stream: None,
    };
    let mut state = recorder()
        .lock()
        .map_err(|_| anyhow!("acceptance recorder poisoned"))?;
    if !matches!(*state, RecorderState::Unconfigured) {
        bail!("successor acceptance recorder was already initialized");
    }
    *state = RecorderState::Active(session);
    Ok(true)
}

/// Record one exact field from the governed runtime's in-memory measurement.
///
/// Returns `false` when acceptance mode was not configured. Unknown fields,
/// duplicate fields, non-admitted values, and gate mismatches fail closed.
pub fn record_runtime_measurement<T: Serialize>(
    gate_id: &str,
    field: &str,
    value: T,
) -> Result<bool> {
    let value = serde_json::to_value(value).context("serialize direct runtime measurement")?;
    measurement_value_type(&value)?;
    let mut state = recorder()
        .lock()
        .map_err(|_| anyhow!("acceptance recorder poisoned"))?;
    match &mut *state {
        RecorderState::Unconfigured => Ok(false),
        RecorderState::PublishedSuccess | RecorderState::PublishedFailure => {
            bail!("acceptance measurement recorded after publication")
        }
        RecorderState::Active(session) => {
            if session.gate_id != gate_id {
                bail!(
                    "acceptance measurement gate mismatch: active={} observed={gate_id}",
                    session.gate_id
                );
            }
            if !required_observation_fields(gate_id)?.contains(field) {
                bail!("acceptance measurement field is not in the contract population: {gate_id}.{field}");
            }
            if session.values.insert(field.to_string(), value).is_some() {
                bail!("duplicate direct runtime measurement: {gate_id}.{field}");
            }
            Ok(true)
        }
    }
}

/// Return the configured single-run gate without exposing recorder contents.
pub fn active_runtime_acceptance_gate() -> Result<Option<String>> {
    let state = recorder()
        .lock()
        .map_err(|_| anyhow!("acceptance recorder poisoned"))?;
    Ok(match &*state {
        RecorderState::Active(session) => Some(session.gate_id.clone()),
        RecorderState::Unconfigured
        | RecorderState::PublishedSuccess
        | RecorderState::PublishedFailure => None,
    })
}

/// Preserve the complete, actual per-stream opcode population captured from
/// the in-memory native MD receipt. Unknown registry names are retained with a
/// null opcode id in QNANL metadata; they are never filtered or promoted into a
/// behavioral claim.
pub fn record_runtime_graph_opcode_population(
    stream_id: u32,
    opcode_counts: &BTreeMap<String, u64>,
) -> Result<bool> {
    if opcode_counts.is_empty() || opcode_counts.values().any(|count| *count == 0) {
        bail!(
            "EVIDENCE_NOT_FOUND: runtime graph opcode population is empty or contains zero counts"
        );
    }
    let mut state = recorder()
        .lock()
        .map_err(|_| anyhow!("acceptance recorder poisoned"))?;
    match &mut *state {
        RecorderState::Unconfigured => Ok(false),
        RecorderState::PublishedSuccess | RecorderState::PublishedFailure => {
            bail!("runtime graph opcode population recorded after publication")
        }
        RecorderState::Active(session) => {
            if session.gate_id != "G_GRAPH_RESIDENT" {
                bail!("runtime graph opcode population is valid only for G_GRAPH_RESIDENT");
            }
            if session
                .graph_opcode_counts_by_stream
                .insert(stream_id, opcode_counts.clone())
                .is_some()
            {
                bail!("duplicate runtime graph opcode population for stream {stream_id}");
            }
            Ok(true)
        }
    }
}

/// Bind the gate-specific `MD.GHOST_CONSERVATION` opcode population to the
/// exact per-stream record totals reconstructed by `nhs_rt_full` from one
/// nonempty [`crate::ghost_phase_lattice::GhostSegmentLatticeLoad`]. This is
/// deliberately distinct from the older RuntimeSubprocess Ghost event packets:
/// only this same-process population can enter the acceptance RuntimeInline
/// transaction.
pub fn record_runtime_ghost_conservation_opcode_population(
    records_by_stream: &BTreeMap<u16, u64>,
) -> Result<bool> {
    if records_by_stream.is_empty() {
        bail!("EVIDENCE_NOT_FOUND: Ghost conservation opcode population has no streams");
    }
    let total = records_by_stream.values().try_fold(0_u64, |sum, count| {
        sum.checked_add(*count)
            .context("Ghost conservation opcode population overflow")
    })?;
    if total == 0 {
        bail!("EVIDENCE_NOT_FOUND: Ghost conservation opcode population is empty");
    }
    let mut state = recorder()
        .lock()
        .map_err(|_| anyhow!("acceptance recorder poisoned"))?;
    match &mut *state {
        RecorderState::Unconfigured => Ok(false),
        RecorderState::PublishedSuccess | RecorderState::PublishedFailure => {
            bail!("runtime Ghost conservation opcode population recorded after publication")
        }
        RecorderState::Active(session) => {
            if session.gate_id != "G_GHOST_CONSERVATION" {
                bail!(
                    "runtime Ghost conservation opcode population is valid only for G_GHOST_CONSERVATION"
                );
            }
            validate_gate_observations(&session.gate_id, &session.values)?;
            let produced = session
                .values
                .get("produced")
                .and_then(Value::as_u64)
                .context("G_GHOST_CONSERVATION produced count absent before opcode binding")?;
            if total != produced {
                bail!(
                    "EVIDENCE_NOT_FOUND: Ghost conservation opcode population total differs: per_stream={total} produced={produced}"
                );
            }
            if session
                .ghost_conservation_records_by_stream
                .replace(records_by_stream.clone())
                .is_some()
            {
                bail!("duplicate runtime Ghost conservation opcode population");
            }
            Ok(true)
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct EmissionSummary {
    pub gate_id: String,
    pub measurement_member_count: usize,
    pub source_population_sha256: String,
    pub raw_runtime_artifact: FileBinding,
    pub observation_source: FileBinding,
    pub gate_evidence: FileBinding,
    pub transaction_closeout_receipt: FileBinding,
    pub runtimeinline_qnanl_packet: FileBinding,
    pub runtimeinline_qnanl_receipt: FileBinding,
}

fn source_population_sha256(members: &[Value]) -> Result<String> {
    let population = members
        .iter()
        .map(|member| {
            serde_json::json!({
                "field": member["body"]["field"],
                "measurement_schema_kind": member["schema_kind"],
                "member_sha256": member["member_sha256"],
            })
        })
        .collect::<Vec<_>>();
    Ok(sha256_bytes(&canonical_json(&population)?))
}

#[derive(Debug)]
struct RuntimeInlineQnanlArtifact {
    packet_path: PathBuf,
    packet_raw: Vec<u8>,
    receipt_path: PathBuf,
    receipt_raw: Vec<u8>,
    opcode_population: Vec<Value>,
}

fn assemble_runtimeinline_qnanl(
    session: &RuntimeSession,
    raw: &[u8],
) -> Result<RuntimeInlineQnanlArtifact> {
    let (artifact_id, family, mut pcode_refs) = match session.gate_id.as_str() {
        "G_GRAPH_RESIDENT" => {
            let governed_stream_count = session
                .values
                .get("governed_stream_count")
                .and_then(Value::as_u64)
                .context("G_GRAPH_RESIDENT governed_stream_count absent before QNANL encode")?;
            if session.graph_opcode_counts_by_stream.len() as u64 != governed_stream_count {
                bail!(
                    "EVIDENCE_NOT_FOUND: QNANL opcode stream population differs: observed={} governed={governed_stream_count}",
                    session.graph_opcode_counts_by_stream.len()
                );
            }
            let mut by_opcode = BTreeMap::<String, BTreeMap<u32, u64>>::new();
            for (stream_id, counts) in &session.graph_opcode_counts_by_stream {
                for (name, count) in counts {
                    by_opcode
                        .entry(name.clone())
                        .or_default()
                        .insert(*stream_id, *count);
                }
            }
            if by_opcode.is_empty() {
                bail!(
                    "EVIDENCE_NOT_FOUND: RuntimeInline QNANL has no actual graph opcode population"
                );
            }
            (
                "successor-g-graph-resident-runtime-records",
                QnanlArtifactFamily::PrismcodeExecutionArtifact,
                by_opcode
                    .iter()
                    .map(|(name, counts)| QnanlPcodeRef {
                        opcode_name: name.clone(),
                        opcode_id: opcode_id_for_name(name),
                        operands: counts
                            .iter()
                            .map(|(stream_id, count)| format!("stream_{stream_id}_count={count}"))
                            .collect(),
                    })
                    .collect::<Vec<_>>(),
            )
        }
        "G_GHOST_CONSERVATION" => {
            let records_by_stream = session
                .ghost_conservation_records_by_stream
                .as_ref()
                .context("G_GHOST_CONSERVATION RuntimeInline opcode population absent")?;
            let produced = session
                .values
                .get("produced")
                .and_then(Value::as_u64)
                .context("G_GHOST_CONSERVATION produced count absent before QNANL encode")?;
            let total = records_by_stream.values().try_fold(0_u64, |sum, count| {
                sum.checked_add(*count)
                    .context("Ghost conservation RuntimeInline opcode count overflow")
            })?;
            if records_by_stream.is_empty() || total == 0 || total != produced {
                bail!(
                    "EVIDENCE_NOT_FOUND: Ghost conservation RuntimeInline opcode population differs: streams={} total={} produced={}",
                    records_by_stream.len(),
                    total,
                    produced,
                );
            }
            let opcode_name = "MD.GHOST_CONSERVATION";
            let opcode_id = opcode_id_for_name(opcode_name)
                .context("canonical MD.GHOST_CONSERVATION opcode absent")?;
            let mut operands = records_by_stream
                .iter()
                .map(|(stream_id, count)| format!("stream_{stream_id}_record_count={count}"))
                .collect::<Vec<_>>();
            operands.push(format!("run_total_record_count={total}"));
            operands.push(
                "population_source=verified_per_stream_and_total_GhostSegmentLatticeLoad"
                    .to_string(),
            );
            (
                "successor-g-ghost-conservation-runtime-records",
                QnanlArtifactFamily::GhostConservation,
                vec![QnanlPcodeRef {
                    opcode_name: opcode_name.to_string(),
                    opcode_id: Some(opcode_id),
                    operands,
                }],
            )
        }
        gate_id => bail!("RuntimeInline QNANL unsupported for active gate {gate_id}"),
    };
    let opcode_population = pcode_refs
        .iter()
        .map(|reference| {
            serde_json::json!({
                "opcode_name": reference.opcode_name,
                "opcode_id": reference.opcode_id.map(|id| format!("0x{id:04X}")),
                "operands": reference.operands,
                "registry_status": if reference.opcode_id.is_some() { "RESOLVED_METADATA" } else { "UNRESOLVED_METADATA_PRESERVED" },
                "behavioral_claim": false,
            })
        })
        .collect::<Vec<_>>();
    for (name, operands) in [
        (
            "MD.QNANL_SEAL",
            vec!["raw_journal_sha256", "packet_blake3", "runtime_inline"],
        ),
        (
            "MD.QNANL_VERIFY",
            vec!["exact_decode", "codec_roundtrip", "qnanl_constraints"],
        ),
    ] {
        let opcode_id = opcode_id_for_name(name)
            .with_context(|| format!("canonical QNANL metadata opcode absent: {name}"))?;
        pcode_refs.push(QnanlPcodeRef {
            opcode_name: name.to_string(),
            opcode_id: Some(opcode_id),
            operands: operands.into_iter().map(str::to_string).collect(),
        });
    }

    let packet_path = PathBuf::from(format!("{}.prismqnanl", session.raw_runtime_path.display()));
    let receipt_path = PathBuf::from(format!("{}.receipt.json", packet_path.display()));
    let public_parent = session
        .raw_runtime_path
        .parent()
        .and_then(Path::parent)
        .context("acceptance public parent absent")?;
    use std::os::unix::fs::DirBuilderExt;
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let private_root =
        public_parent.join(format!(".successor-qnanl-{}-{nonce}", std::process::id()));
    let mut builder = std::fs::DirBuilder::new();
    builder.mode(0o700);
    builder.create(&private_root).with_context(|| {
        format!(
            "create private QNANL encode root {}",
            private_root.display()
        )
    })?;
    let staged_packet = private_root.join("raw.prismqnanl");
    let staged_receipt = private_root.join("raw.prismqnanl.receipt.json");
    let encode_result = (|| -> Result<RuntimeInlineQnanlArtifact> {
        // First encode yields the exact packet bytes whose real inode is hashed
        // into ArtifactProvenance. The second encode attaches that provenance in
        // QnanlEncodeOptions before its receipt is written; packet bytes are
        // required to remain identical.
        let provisional = encode_bytes(
            raw,
            &staged_packet,
            &staged_receipt,
            QnanlEncodeOptions::runtime_inline(
                artifact_id,
                family,
                Some(session.raw_runtime_path.clone()),
                "nhs_rt_full::successor_acceptance_v4::runtimeinline_qnanl",
            ),
        )
        .map_err(|error| anyhow!(error.to_string()))?;
        let provisional_packet = read_regular_nofollow(&staged_packet, "provisional QNANL packet")?;
        let mut provenance = session.runtime_provenance.artifact_record(
            &staged_packet,
            "nhs_rt_full::successor_acceptance_v4::runtimeinline_qnanl",
            vec![sha256_bytes(raw)],
        );
        provenance.emitted_artifact_path = packet_path.display().to_string();
        let mut receipt = encode_bytes(
            raw,
            &staged_packet,
            &staged_receipt,
            QnanlEncodeOptions::runtime_inline(
                artifact_id,
                family,
                Some(session.raw_runtime_path.clone()),
                "nhs_rt_full::successor_acceptance_v4::runtimeinline_qnanl",
            )
            .with_runtime_provenance(provenance),
        )
        .map_err(|error| anyhow!(error.to_string()))?;
        let packet_raw = read_regular_nofollow(&staged_packet, "RuntimeInline QNANL packet")?;
        if packet_raw != provisional_packet
            || receipt.packet_hash_blake3 != provisional.packet_hash_blake3
        {
            bail!("RuntimeInline QNANL packet changed while attaching provenance");
        }
        receipt.source_path = session.raw_runtime_path.display().to_string();
        receipt.packet_path = packet_path.display().to_string();
        receipt.receipt_path = receipt_path.display().to_string();
        receipt = receipt.with_pcode_refs(pcode_refs);
        if receipt.runtime_origin != RuntimeOrigin::RuntimeInline
            || receipt.runtime_provenance.is_none()
            || receipt.pcode_refs.is_empty()
            || !receipt.roundtrip_verified
            || !receipt.codec_roundtrip_verified
            || !receipt.qnanl_seal_constraints_pass
            || session.runtime_provenance.capture_unix_ns > receipt.runtime_origin_assigned_unix_ns
            || decode_packet_bytes(&packet_raw).map_err(|error| anyhow!(error.to_string()))? != raw
        {
            bail!("RuntimeInline QNANL provenance, pcode, or exact-decode invariant failed");
        }
        let provenance = receipt
            .runtime_provenance
            .as_ref()
            .context("RuntimeInline QNANL artifact provenance absent")?;
        let packet_sha256 = sha256_bytes(&packet_raw);
        if provenance.emitted_artifact_path != packet_path.display().to_string()
            || provenance.emitted_artifact_sha256.as_deref() != Some(packet_sha256.as_str())
            || provenance.run_packaged_binary_sha256.as_deref()
                != Some(session.invocation.executable.sha256.as_str())
        {
            bail!("RuntimeInline QNANL public path or executable provenance binding failed");
        }
        let receipt_raw = receipt.to_json_pretty().into_bytes();
        let parsed = strict_json(&receipt_raw, "RuntimeInline QNANL receipt")?;
        if parsed["runtime_origin"] != "RUNTIME_INLINE"
            || parsed["pcode_refs"].as_array().is_none_or(Vec::is_empty)
            || parsed["packet_path"] != packet_path.display().to_string()
            || parsed["receipt_path"] != receipt_path.display().to_string()
        {
            bail!("RuntimeInline QNANL serialized receipt identity failed");
        }
        Ok(RuntimeInlineQnanlArtifact {
            packet_path,
            packet_raw,
            receipt_path,
            receipt_raw,
            opcode_population,
        })
    })();
    let _ = std::fs::remove_file(&staged_receipt);
    let _ = std::fs::remove_file(&staged_packet);
    let _ = std::fs::remove_dir(&private_root);
    encode_result
}

fn build_closeout_artifacts(
    session: &RuntimeSession,
) -> Result<(Vec<(PathBuf, Vec<u8>)>, EmissionSummary)> {
    let current_contract_raw = read_regular_nofollow(
        &session.contract_path,
        "expanded acceptance contract at closeout",
    )?;
    if current_contract_raw != session.contract_raw {
        bail!("expanded acceptance contract changed during governed execution");
    }
    if bind_current_executable()? != session.invocation.executable {
        bail!("directly executing nhs_rt_full identity changed during governed execution");
    }
    for (binding, label) in [
        (
            &session.authorities.candidate_build_manifest,
            "candidate build manifest",
        ),
        (&session.authorities.source_snapshot, "source snapshot"),
        (
            &session.authorities.measurement_implementation,
            "measurement implementation",
        ),
        (
            &session.authorities.measurement_implementation_manifest,
            "measurement implementation manifest",
        ),
    ] {
        verify_file_binding(binding, label)?;
    }

    assemble_closeout_artifacts(session)
}

fn assemble_closeout_artifacts(
    session: &RuntimeSession,
) -> Result<(Vec<(PathBuf, Vec<u8>)>, EmissionSummary)> {
    validate_gate_observations(&session.gate_id, &session.values)?;
    let mut raw = Vec::new();
    let mut records = Vec::new();
    for (field, value) in &session.values {
        let record = serde_json::json!({
            "schema_kind": RAW_RUNTIME_RECORD_SCHEMA,
            "schema_version": RAW_RUNTIME_RECORD_VERSION,
            "gate_id": session.gate_id,
            "field": field,
            "value": value,
            "execution_id": session.execution_id,
            "gate_contract_sha256": session.gate_contract_sha256,
            "successor_engine_sha256": session.invocation.executable.sha256,
            "measurement_implementation_sha256": session.authorities.measurement_implementation.sha256,
        });
        let record_raw = canonical_json(&record)?;
        let offset = raw.len() as u64;
        let length = record_raw.len() as u64;
        let hash = sha256_bytes(&record_raw);
        raw.extend_from_slice(&record_raw);
        raw.push(b'\n');
        records.push((field.clone(), value.clone(), offset, length, hash));
    }
    let raw_binding = file_binding_for_bytes(&session.raw_runtime_path, &raw);
    let mut members = Vec::new();
    for (field, value, offset, length, record_hash) in records {
        let body = serde_json::json!({
            "gate_id": session.gate_id,
            "field": field,
            "value_type": measurement_value_type(&value)?,
            "value": value,
            "execution_id": session.execution_id,
            "gate_contract_sha256": session.gate_contract_sha256,
            "command_identity_sha256": session.invocation.command_identity_sha256,
            "successor_engine_sha256": session.invocation.executable.sha256,
            "native_producer_sha256": session.invocation.executable.sha256,
            "raw_runtime_artifact": raw_binding,
            "raw_record_locator": {
                "byte_offset": offset,
                "byte_length": length,
                "record_sha256": record_hash,
                "record_format": RAW_RUNTIME_RECORD_FORMAT,
            },
            "reconstruction_method": {
                "schema_kind": reconstruction_schema_kind(&session.gate_id, &field),
                "schema_version": 1,
                "algorithm": RAW_RECONSTRUCTION_ALGORITHM,
                "json_pointer": "/value",
                "measurement_implementation_sha256": session.authorities.measurement_implementation.sha256,
                "reconstructed_value_sha256": sha256_bytes(&canonical_json(&value)?),
            },
        });
        members.push(serde_json::json!({
            "schema_kind": measurement_schema_kind(&session.gate_id, &field),
            "schema_version": MEASUREMENT_MEMBER_SCHEMA_VERSION,
            "member_sha256": sha256_bytes(&canonical_json(&body)?),
            "body": body,
        }));
    }
    let population_hash = source_population_sha256(&members)?;
    let source_body = serde_json::json!({
        "gate_id": session.gate_id,
        "successor_engine_sha256": session.invocation.executable.sha256,
        "authority_bindings": session.authority_bindings,
        "execution_id": session.execution_id,
        "gate_contract_sha256": session.gate_contract_sha256,
        "command_identity_sha256": session.invocation.command_identity_sha256,
        "native_producer_sha256": session.invocation.executable.sha256,
        "candidate_build_manifest_sha256": session.authorities.candidate_build_manifest.sha256,
        "measurement_implementation_sha256": session.authorities.measurement_implementation.sha256,
        "source_population_sha256": population_hash,
        "members": members,
    });
    let source = serde_json::json!({
        "schema_kind": source_schema_kind(&session.gate_id),
        "schema_version": SOURCE_SCHEMA_VERSION,
        "artifact_sha256": sha256_bytes(&canonical_json(&source_body)?),
        "body": source_body,
    });
    let mut source_raw = canonical_json(&source)?;
    source_raw.push(b'\n');
    let source_binding = file_binding_for_bytes(&session.observation_source_path, &source_raw);
    let source_reference = serde_json::json!({
        "role": "primary_observation_authority",
        "schema_kind": source_schema_kind(&session.gate_id),
        "schema_version": SOURCE_SCHEMA_VERSION,
        "path": source_binding.path,
        "sha256": source_binding.sha256,
        "size_bytes": source_binding.size_bytes,
        "source_population_sha256": population_hash,
    });
    let producer_body = serde_json::json!({
        "gate_id": session.gate_id,
        "successor_engine_sha256": session.invocation.executable.sha256,
        "authority_bindings": session.authority_bindings,
        "execution_id": session.execution_id,
        "gate_contract_sha256": session.gate_contract_sha256,
        "command_identity_sha256": session.invocation.command_identity_sha256,
        "native_producer": session.invocation.executable,
        "candidate_build_manifest": session.authorities.candidate_build_manifest,
        "source_snapshot": session.authorities.source_snapshot,
        "measurement_implementation": session.authorities.measurement_implementation,
        "measurement_implementation_manifest": session.authorities.measurement_implementation_manifest,
        "observation_source": source_reference,
        "source_population_sha256": population_hash,
    });
    let producer = serde_json::json!({
        "schema_kind": producer_schema_kind(&session.gate_id),
        "schema_version": PRODUCER_SCHEMA_VERSION,
        "producer_receipt_sha256": sha256_bytes(&canonical_json(&producer_body)?),
        "body": producer_body,
    });
    let mut producer_raw = canonical_json(&producer)?;
    producer_raw.push(b'\n');
    let producer_binding = file_binding_for_bytes(&session.gate_evidence_path, &producer_raw);
    let runtimeinline_qnanl = assemble_runtimeinline_qnanl(session, &raw)?;
    let qnanl_packet_binding = file_binding_for_bytes(
        &runtimeinline_qnanl.packet_path,
        &runtimeinline_qnanl.packet_raw,
    );
    let qnanl_receipt_binding = file_binding_for_bytes(
        &runtimeinline_qnanl.receipt_path,
        &runtimeinline_qnanl.receipt_raw,
    );
    let measurement_members = source["body"]["members"]
        .as_array()
        .context("typed observation members absent at closeout")?
        .iter()
        .map(|member| {
            serde_json::json!({
                "field": member["body"]["field"],
                "schema_kind": member["schema_kind"],
                "member_sha256": member["member_sha256"],
            })
        })
        .collect::<Vec<_>>();

    // This is an ordinary canonical transaction closeout record over real
    // measurement members. It is not a QNANL artifact and cannot satisfy the
    // typed RuntimeInline QNANL gate.
    let closeout_body = serde_json::json!({
        "gate_id": session.gate_id,
        "execution_id": session.execution_id,
        "gate_contract_sha256": session.gate_contract_sha256,
        "successor_engine_sha256": session.invocation.executable.sha256,
        "measurement_implementation_sha256": session.authorities.measurement_implementation.sha256,
        "source_population_sha256": population_hash,
        "raw_runtime_artifact_sha256": raw_binding.sha256,
        "typed_observation_authority_sha256": source_binding.sha256,
        "producer_receipt_file_sha256": producer_binding.sha256,
        "measurement_members": measurement_members,
        "unresolved_gate_ledger": unresolved_gate_ledger(&session.gate_id),
        "runtimeinline_typed_qnanl_authority": {
            "status": "SUPPORTING_EVIDENCE",
            "supports_gate": session.gate_id,
            "gate_pass_claim": false,
            "g_prismcode_qnanl_status": "EVIDENCE_NOT_FOUND",
            "reason": "RuntimeInline packet proves exact-byte and gate-specific actual-opcode lineage for the active gate only; RuntimeSubprocess Ghost packets are excluded and this packet does not satisfy G_PRISMCODE_QNANL",
            "packet": qnanl_packet_binding,
            "receipt": qnanl_receipt_binding,
            "opcode_population": runtimeinline_qnanl.opcode_population,
            "pcode_refs_are_behavioral_claims": false,
        },
        "failure_closeout_coverage": {
            "status": "DIAGNOSTIC_ONLY_DISJOINT_TRANSACTION",
            "reason": "failure diagnostics can never satisfy a gate or enter this success transaction",
            "unhandled_signals": ["SIGTERM", "SIGHUP", "SIGQUIT", "SIGKILL"],
        },
    });
    let closeout = serde_json::json!({
        "schema_kind": "prismos_successor_acceptance_transaction_closeout_receipt",
        "schema_version": 1,
        "receipt_sha256": sha256_bytes(&canonical_json(&closeout_body)?),
        "body": closeout_body,
    });
    let mut closeout_raw = canonical_json(&closeout)?;
    closeout_raw.push(b'\n');
    let closeout_binding =
        file_binding_for_bytes(&session.transaction_closeout_receipt_path, &closeout_raw);
    let artifacts = vec![
        (session.raw_runtime_path.clone(), raw),
        (session.observation_source_path.clone(), source_raw),
        (session.gate_evidence_path.clone(), producer_raw),
        (
            runtimeinline_qnanl.packet_path.clone(),
            runtimeinline_qnanl.packet_raw,
        ),
        (
            runtimeinline_qnanl.receipt_path.clone(),
            runtimeinline_qnanl.receipt_raw,
        ),
        (
            session.transaction_closeout_receipt_path.clone(),
            closeout_raw,
        ),
    ];
    Ok((
        artifacts,
        EmissionSummary {
            gate_id: session.gate_id.clone(),
            measurement_member_count: session.values.len(),
            source_population_sha256: population_hash,
            raw_runtime_artifact: raw_binding,
            observation_source: source_binding,
            gate_evidence: producer_binding,
            transaction_closeout_receipt: closeout_binding,
            runtimeinline_qnanl_packet: qnanl_packet_binding,
            runtimeinline_qnanl_receipt: qnanl_receipt_binding,
        },
    ))
}

fn fsync_directory(path: &Path) -> Result<()> {
    let directory = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(path)
        .with_context(|| format!("open transaction directory {}", path.display()))?;
    directory.sync_all()?;
    Ok(())
}

fn rename_noreplace(source: &Path, destination: &Path) -> Result<()> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    let source = CString::new(source.as_os_str().as_bytes())?;
    let destination = CString::new(destination.as_os_str().as_bytes())?;
    let status = unsafe {
        libc::renameat2(
            libc::AT_FDCWD,
            source.as_ptr(),
            libc::AT_FDCWD,
            destination.as_ptr(),
            libc::RENAME_NOREPLACE,
        )
    };
    if status != 0 {
        return Err(std::io::Error::last_os_error()).context("renameat2(RENAME_NOREPLACE)");
    }
    Ok(())
}

const TRANSACTION_COMMIT_MARKER: &str = ".ACCEPTANCE_TRANSACTION_COMMIT.json";
const FAILURE_TRANSACTION_COMMIT_MARKER: &str = ".FAILURE_DIAGNOSTIC_TRANSACTION_COMMIT.json";

fn transaction_commit_bytes(
    artifacts: &[(PathBuf, Vec<u8>)],
    schema_kind: &str,
) -> Result<Vec<u8>> {
    let members = artifacts
        .iter()
        .map(|(path, raw)| {
            Ok(serde_json::json!({
                "name": path.file_name().and_then(|name| name.to_str()).context("transaction member name is not UTF-8")?,
                "sha256": sha256_bytes(raw),
                "size_bytes": raw.len(),
            }))
        })
        .collect::<Result<Vec<_>>>()?;
    let body = serde_json::json!({
        "publication_algorithm": "PRIVATE_DIRECTORY_FSYNC_RENAMEAT2_NOREPLACE_V1",
        "member_count": members.len(),
        "members": members,
    });
    let document = serde_json::json!({
        "schema_kind": schema_kind,
        "schema_version": 1,
        "transaction_sha256": sha256_bytes(&canonical_json(&body)?),
        "body": body,
    });
    let mut raw = canonical_json(&document)?;
    raw.push(b'\n');
    Ok(raw)
}

fn remove_private_stage(stage: &Path, member_names: &[String], marker_name: &str) {
    for name in member_names {
        let _ = std::fs::remove_file(stage.join(name));
    }
    let _ = std::fs::remove_file(stage.join(marker_name));
    let _ = std::fs::remove_dir(stage);
}

fn verify_identical_transaction(
    final_root: &Path,
    artifacts: &[(PathBuf, Vec<u8>)],
    commit_raw: &[u8],
    marker_name: &str,
) -> Result<bool> {
    let metadata = match std::fs::symlink_metadata(final_root) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error.into()),
    };
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        bail!(
            "transaction collision is not a real directory: {}",
            final_root.display()
        );
    }
    let expected_names = artifacts
        .iter()
        .map(|(path, _)| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(str::to_string)
                .context("transaction member name is not UTF-8")
        })
        .chain(std::iter::once(Ok(marker_name.to_string())))
        .collect::<Result<BTreeSet<_>>>()?;
    let observed_names = std::fs::read_dir(final_root)?
        .map(|entry| {
            entry?
                .file_name()
                .into_string()
                .map_err(|_| anyhow!("published transaction contains a non-UTF8 name"))
        })
        .collect::<Result<BTreeSet<_>>>()?;
    if observed_names != expected_names {
        bail!("transaction collision has a different member population");
    }
    for (path, expected) in artifacts {
        if read_regular_nofollow(path, "existing transaction artifact")? != *expected {
            bail!(
                "transaction collision contains different member bytes: {}",
                path.display()
            );
        }
    }
    if read_regular_nofollow(
        &final_root.join(marker_name),
        "existing transaction commit marker",
    )? != commit_raw
    {
        bail!("transaction collision contains a different commit marker");
    }
    Ok(true)
}

/// Stage and fsync the complete directory, write/fsync the commit marker last,
/// then publish the directory with one `renameat2(RENAME_NOREPLACE)`.
///
/// No member is visible under the public final root before that single rename.
/// An already-published directory is accepted only when its exact member
/// population, bytes, and deterministic commit marker all match.
fn durable_sibling_transaction(artifacts: &[(PathBuf, Vec<u8>)]) -> Result<()> {
    durable_sibling_transaction_with_marker(
        artifacts,
        TRANSACTION_COMMIT_MARKER,
        "prismos_successor_acceptance_transaction_commit",
    )
}

fn durable_sibling_transaction_with_marker(
    artifacts: &[(PathBuf, Vec<u8>)],
    marker_name: &str,
    marker_schema_kind: &str,
) -> Result<()> {
    if artifacts.is_empty() {
        bail!("acceptance transaction has no artifacts");
    }
    let final_root = artifacts[0]
        .0
        .parent()
        .context("transaction final root absent")?;
    if artifacts
        .iter()
        .any(|(path, _)| path.parent() != Some(final_root))
    {
        bail!("acceptance transaction artifacts are not siblings");
    }
    let member_names = artifacts
        .iter()
        .map(|(path, _)| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(str::to_string)
                .context("transaction member name is not UTF-8")
        })
        .collect::<Result<Vec<_>>>()?;
    if member_names.iter().collect::<BTreeSet<_>>().len() != member_names.len()
        || member_names.iter().any(|name| name == marker_name)
    {
        bail!("acceptance transaction has duplicate/reserved member names");
    }
    let commit_raw = transaction_commit_bytes(artifacts, marker_schema_kind)?;
    if verify_identical_transaction(final_root, artifacts, &commit_raw, marker_name)? {
        return Ok(());
    }
    let public_parent = final_root
        .parent()
        .context("transaction public parent absent")?;
    let public_parent = public_parent.canonicalize()?;
    if final_root.parent() != Some(public_parent.as_path()) {
        bail!("transaction final root is not normalized beneath its public parent");
    }
    use std::os::unix::fs::DirBuilderExt;
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let final_name = final_root
        .file_name()
        .context("transaction final root name absent")?
        .to_string_lossy();
    let stage = public_parent.join(format!(
        ".{final_name}.{}.{}.acceptance-stage",
        std::process::id(),
        nonce
    ));
    let mut builder = std::fs::DirBuilder::new();
    builder.mode(0o750);
    builder
        .create(&stage)
        .with_context(|| format!("create private transaction directory {}", stage.display()))?;
    let stage_result = (|| -> Result<()> {
        for ((_, raw), file_name) in artifacts.iter().zip(&member_names) {
            let stage_path = stage.join(file_name);
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o640)
                .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
                .open(&stage_path)
                .with_context(|| format!("create transaction stage {}", stage_path.display()))?;
            file.write_all(raw)?;
            file.sync_all()?;
        }
        // The deterministic commit marker is the last file created and fsynced.
        let marker_path = stage.join(marker_name);
        let mut marker = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o640)
            .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
            .open(&marker_path)?;
        marker.write_all(&commit_raw)?;
        marker.sync_all()?;
        fsync_directory(&stage)?;
        Ok(())
    })();
    if let Err(error) = stage_result {
        remove_private_stage(&stage, &member_names, marker_name);
        return Err(error);
    }
    if let Err(error) = rename_noreplace(&stage, final_root) {
        remove_private_stage(&stage, &member_names, marker_name);
        if verify_identical_transaction(final_root, artifacts, &commit_raw, marker_name)? {
            return Ok(());
        }
        return Err(error).with_context(|| {
            format!(
                "publish acceptance transaction directory {}",
                final_root.display()
            )
        });
    }
    fsync_directory(&public_parent)?;
    for (path, expected) in artifacts {
        let observed = read_regular_nofollow(path, "committed transaction artifact")?;
        if &observed != expected {
            bail!(
                "full transaction reload differs after commit: {}",
                path.display()
            );
        }
    }
    let observed_commit = read_regular_nofollow(
        &final_root.join(marker_name),
        "committed transaction marker",
    )?;
    if observed_commit != commit_raw {
        bail!("full transaction commit-marker reload differs after publication");
    }
    Ok(())
}

/// Publish the complete gate evidence transaction at successful runtime closeout.
///
/// Missing members or failed machine rules are evaluated before any public
/// destination is created. A deterministic transaction marker is written and
/// fsynced last in private staging before the complete directory is published.
pub fn publish_runtime_acceptance() -> Result<Option<EmissionSummary>> {
    let mut state = recorder()
        .lock()
        .map_err(|_| anyhow!("acceptance recorder poisoned"))?;
    match &*state {
        RecorderState::Unconfigured => return Ok(None),
        RecorderState::PublishedSuccess => bail!("successor acceptance was already published"),
        RecorderState::PublishedFailure => {
            bail!("successor acceptance failure diagnostic already owns terminal publication")
        }
        RecorderState::Active(_) => {}
    }
    let (artifacts, summary) = match &*state {
        RecorderState::Active(session) => build_closeout_artifacts(session)?,
        _ => unreachable!(),
    };
    if !TERMINAL_COORDINATOR.try_claim(TERMINAL_SUCCESS) {
        bail!("successor terminal coordinator rejected duplicate success publication");
    }
    if let Err(error) = durable_sibling_transaction(&artifacts) {
        TERMINAL_COORDINATOR.release_failed_publication(TERMINAL_SUCCESS);
        return Err(error);
    }
    *state = RecorderState::PublishedSuccess;
    Ok(Some(summary))
}

#[derive(Debug, Clone, Serialize)]
pub struct FailureDiagnosticSummary {
    pub diagnostic_path: FileBinding,
    pub error_class: String,
}

fn failure_diagnostic_artifacts(
    session: &RuntimeSession,
    error_class: &str,
    message: &str,
) -> Result<(Vec<(PathBuf, Vec<u8>)>, FailureDiagnosticSummary)> {
    let success_root = session
        .raw_runtime_path
        .parent()
        .context("success transaction root absent")?;
    let success_name = success_root
        .file_name()
        .and_then(|name| name.to_str())
        .context("success transaction root name is not UTF-8")?;
    let diagnostic_root = success_root
        .parent()
        .context("success transaction public parent absent")?
        .join(format!("{success_name}.failure-diagnostic"));
    let diagnostic_path = diagnostic_root.join("successor_failure_diagnostic.json");
    let unadjudicated_observation_hashes = session
        .values
        .iter()
        .map(|(field, value)| {
            Ok(serde_json::json!({
                "field": field,
                "canonical_value_sha256": sha256_bytes(&canonical_json(value)?),
                "adjudicated": false,
            }))
        })
        .collect::<Result<Vec<_>>>()?;
    let unresolved_gates = GATE_ORDER
        .iter()
        .map(|gate_id| {
            serde_json::json!({
                "gate_id": gate_id,
                "status": "EVIDENCE_NOT_FOUND",
                "reason_class": "FAILURE_DIAGNOSTIC_CANNOT_SATISFY_GATE",
            })
        })
        .collect::<Vec<_>>();
    let body = serde_json::json!({
        "diagnostic_only": true,
        "gate_satisfaction_capability": "NONE",
        "gate_outcome": "NOT_EVALUATED",
        "gate_pass_claim_count": 0,
        "success_transaction_published": false,
        "active_gate_id_at_failure": session.gate_id,
        "execution_id": session.execution_id,
        "successor_engine": session.invocation.executable,
        "command_identity_sha256": session.invocation.command_identity_sha256,
        "error": {
            "class": error_class,
            "message": message,
        },
        "unadjudicated_observation_hashes": unadjudicated_observation_hashes,
        "unresolved_gate_ledger": unresolved_gates,
        "scientific_pass_status": "PROHIBITED",
        "typed_qnanl_gate_status": "EVIDENCE_NOT_FOUND",
        "signal_coverage": {
            "SIGINT": "WATCHDOG_POLLING_PATH_ONLY",
            "SIGTERM": "EVIDENCE_NOT_FOUND",
            "SIGHUP": "EVIDENCE_NOT_FOUND",
            "SIGQUIT": "EVIDENCE_NOT_FOUND",
            "SIGKILL": "EVIDENCE_NOT_FOUND_UNCATCHABLE",
        },
    });
    let document = serde_json::json!({
        "schema_kind": "prismos_successor_failure_diagnostic_transaction",
        "schema_version": 1,
        "status": "DIAGNOSTIC_FAILURE",
        "diagnostic_sha256": sha256_bytes(&canonical_json(&body)?),
        "body": body,
    });
    let mut raw = canonical_json(&document)?;
    raw.push(b'\n');
    let binding = file_binding_for_bytes(&diagnostic_path, &raw);
    Ok((
        vec![(diagnostic_path, raw)],
        FailureDiagnosticSummary {
            diagnostic_path: binding,
            error_class: error_class.to_string(),
        },
    ))
}

/// Publish a disjoint, diagnostic-only failure transaction. This schema has no
/// gate evidence member, contains zero PASS claims, and can never satisfy any
/// acceptance-v4 gate. Failure to publish is intentionally best-effort so it
/// can never replace or weaken the original engine error.
pub fn publish_runtime_failure_diagnostic(
    error_class: &str,
    message: &str,
) -> Result<Option<FailureDiagnosticSummary>> {
    let mut state = recorder()
        .lock()
        .map_err(|_| anyhow!("acceptance recorder poisoned during failure closeout"))?;
    let session = match &*state {
        RecorderState::Unconfigured
        | RecorderState::PublishedSuccess
        | RecorderState::PublishedFailure => return Ok(None),
        RecorderState::Active(session) => session,
    };
    let (artifacts, summary) = failure_diagnostic_artifacts(session, error_class, message)?;
    if !TERMINAL_COORDINATOR.try_claim(TERMINAL_FAILURE) {
        return Ok(None);
    }
    if let Err(error) = durable_sibling_transaction_with_marker(
        &artifacts,
        FAILURE_TRANSACTION_COMMIT_MARKER,
        "prismos_successor_failure_diagnostic_transaction_commit",
    ) {
        TERMINAL_COORDINATOR.release_failed_publication(TERMINAL_FAILURE);
        return Err(error);
    }
    *state = RecorderState::PublishedFailure;
    Ok(Some(summary))
}

/// Emit the two semantic authority manifests for this exact compiled ELF.
///
/// This command performs no scientific computation and does not convert run
/// output. It binds an already-created immutable source snapshot, the exact
/// measurement implementation, the executable bytes, the ABI, and all 654
/// reconstruction methods before the contract authority is resolved.
pub fn emit_authority_manifests(args: &SuccessorAuthorityManifestArgs) -> Result<Vec<FileBinding>> {
    let producer = bind_current_executable()?;
    let source_snapshot_path = args.source_snapshot.canonicalize()?;
    let measurement_implementation_path = args.measurement_implementation.canonicalize()?;
    let panel_input_metadata = std::fs::symlink_metadata(&args.frontier_probe_panel)?;
    if panel_input_metadata.file_type().is_symlink() || !panel_input_metadata.file_type().is_file()
    {
        bail!("frontier probe panel input must be a non-symlink regular file");
    }
    let frontier_probe_panel_path = args.frontier_probe_panel.canonicalize()?;
    let source_snapshot = bind_file(&source_snapshot_path, "source snapshot")?;
    let measurement_implementation = bind_file(
        &measurement_implementation_path,
        "measurement implementation",
    )?;
    let panel_binding = bind_file(&frontier_probe_panel_path, "frontier probe panel")?;
    let panel_raw = read_regular_nofollow(&frontier_probe_panel_path, "frontier probe panel")?;
    let frontier_probe_panel_authority =
        validate_frontier_probe_panel_bytes(&panel_raw, "frontier probe panel")?;
    verify_frontier_probe_panel_binding(
        &panel_binding,
        &frontier_probe_panel_authority,
        "frontier probe panel",
    )?;
    let build_path = normalized_output_path(
        &args.candidate_build_manifest,
        "candidate build manifest output",
    )?;
    let implementation_path = normalized_output_path(
        &args.measurement_implementation_manifest,
        "measurement implementation manifest output",
    )?;
    if build_path == implementation_path || build_path.parent() != implementation_path.parent() {
        bail!("authority manifest outputs must be distinct siblings");
    }
    let build_body = serde_json::json!({
        "candidate_engine": producer,
        "source_snapshot": source_snapshot,
        "raw_runtime_measurement_abi_sha256": RAW_RUNTIME_MEASUREMENT_ABI_SHA256,
        "frontier_probe_panel_authority": frontier_probe_panel_authority,
        "panel_binding": panel_binding,
    });
    let build_document = manifest_document(CANDIDATE_BUILD_MANIFEST_SCHEMA, build_body)?;
    let mut build_raw = canonical_json(&build_document)?;
    build_raw.push(b'\n');
    let implementation_body = serde_json::json!({
        "measurement_implementation": measurement_implementation,
        "candidate_engine_sha256": producer.sha256,
        "raw_runtime_measurement_abi_sha256": RAW_RUNTIME_MEASUREMENT_ABI_SHA256,
        "reconstruction_method_population_sha256": RECONSTRUCTION_METHOD_POPULATION_SHA256,
        "reconstruction_methods": reconstruction_method_population()?,
    });
    let implementation_document = manifest_document(
        MEASUREMENT_IMPLEMENTATION_MANIFEST_SCHEMA,
        implementation_body,
    )?;
    let mut implementation_raw = canonical_json(&implementation_document)?;
    implementation_raw.push(b'\n');
    let artifacts = vec![
        (build_path.clone(), build_raw),
        (implementation_path.clone(), implementation_raw),
    ];
    durable_sibling_transaction(&artifacts)?;
    let bindings = artifacts
        .iter()
        .map(|(path, _)| bind_file(path, "emitted authority manifest"))
        .collect::<Result<Vec<_>>>()?;
    validate_candidate_build_manifest(&bindings[0], &producer, &source_snapshot)?;
    validate_measurement_implementation_manifest(
        &bindings[1],
        &measurement_implementation,
        &producer,
    )?;
    Ok(bindings)
}

#[cfg(test)]
fn import_serialized_population_for_test(
    gate_id: &str,
    serialized: &[u8],
    allowed_direct_members: &BTreeSet<String>,
) -> Result<BTreeMap<String, Value>> {
    let value = strict_json(serialized, "test-only serialized population")?;
    let object = value
        .as_object()
        .context("test-only population is not an object")?;
    let observed = object.keys().cloned().collect::<BTreeSet<_>>();
    let required = required_observation_fields(gate_id)?;
    if observed != required || observed != *allowed_direct_members {
        bail!("serialized summary is not the exact admitted direct-member population");
    }
    Ok(object
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_panel_authority_missing_rejected() {
        let result = serde_json::from_value::<FrontierProbePanelAuthority>(serde_json::json!({
            "schema_kind": FRONTIER_PROBE_PANEL_AUTHORITY_SCHEMA,
            "schema_version": 1,
        }));
        assert!(result.is_err());
    }

    #[test]
    fn test_panel_wrong_id_rejected() {
        let mut authority = expected_frontier_probe_panel_authority();
        authority.panel_id = "ALTERNATE_PANEL".to_string();
        assert!(authority.validate().is_err());
    }

    #[test]
    fn test_panel_wrong_scientific_digest_rejected() {
        let mut authority = expected_frontier_probe_panel_authority();
        authority.panel_scientific_sha256 = "0".repeat(64);
        assert!(authority.validate().is_err());
    }

    #[test]
    fn test_panel_sci_raw_substitution_rejected() {
        let mut substituted = expected_frontier_probe_panel_authority();
        substituted.panel_scientific_sha256 = FRONTIER_PROBE_PANEL_RAW_SHA256.to_string();
        assert!(substituted.validate().is_err());

        let mut swapped = expected_frontier_probe_panel_authority();
        swapped.panel_raw_sha256 = FRONTIER_PROBE_PANEL_SCIENTIFIC_SHA256.to_string();
        swapped.panel_scientific_sha256 = FRONTIER_PROBE_PANEL_RAW_SHA256.to_string();
        assert!(swapped.validate().is_err());
    }

    #[test]
    fn test_panel_changed_counts_rejected() {
        let mut authority = expected_frontier_probe_panel_authority();
        authority.chemical_state_count += 1;
        assert!(authority.validate().is_err());
    }

    #[test]
    fn test_panel_fragment_mutation_with_copied_digest_rejected() {
        let mutated = serde_json::to_vec(&serde_json::json!({
            "panel_id": FRONTIER_PROBE_PANEL_ID,
            "panel_scientific_sha256": FRONTIER_PROBE_PANEL_SCIENTIFIC_SHA256,
            "species": [],
            "matched_pairs": [],
            "mutated_fragment": true,
        }))
        .unwrap();
        assert!(validate_frontier_probe_panel_bytes(&mutated, "mutated panel").is_err());
    }

    #[test]
    fn test_panel_alternate_panel_rejected() {
        let alternate = serde_json::to_vec(&serde_json::json!({
            "panel_id": "ALTERNATE_PANEL",
            "panel_scientific_sha256": "0".repeat(64),
            "species": [],
            "matched_pairs": [],
        }))
        .unwrap();
        assert!(validate_frontier_probe_panel_bytes(&alternate, "alternate panel").is_err());
    }

    #[test]
    fn canonical_json_matches_python_float_spelling() {
        let vectors = [
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (1.0e15, "1000000000000000.0"),
            (1.0e16, "1e+16"),
            (1.0e-7, "1e-07"),
        ];
        for (value, expected) in vectors {
            assert_eq!(
                String::from_utf8(canonical_json(&value).unwrap()).unwrap(),
                expected
            );
        }
    }

    #[test]
    fn raw_abi_and_reconstruction_population_are_contract_exact() {
        assert_eq!(
            sha256_bytes(&canonical_json(&raw_runtime_measurement_abi()).unwrap()),
            RAW_RUNTIME_MEASUREMENT_ABI_SHA256
        );
        let population = reconstruction_method_population().unwrap();
        assert_eq!(population.len(), RECONSTRUCTION_METHOD_COUNT);
        assert_eq!(
            sha256_bytes(&canonical_json(&population).unwrap()),
            RECONSTRUCTION_METHOD_POPULATION_SHA256
        );
    }

    #[test]
    fn admission_rejects_all_twenty_four_unimplemented_gates() {
        let temp = tempfile::tempdir().unwrap();
        let base_args = SuccessorAcceptanceCloseoutArgs {
            successor_acceptance_gate_contract: Some(temp.path().join("missing-contract.json")),
            successor_acceptance_gate_id: None,
            successor_acceptance_execution_id: Some("shadow-rejection-test".to_string()),
            successor_acceptance_raw_runtime_artifact: Some(temp.path().join("raw.jsonl")),
            successor_acceptance_observation_source: Some(temp.path().join("source.json")),
            successor_acceptance_gate_evidence: Some(temp.path().join("evidence.json")),
            successor_acceptance_transaction_closeout_receipt: Some(
                temp.path().join("closeout.json"),
            ),
            successor_acceptance_execution_authority_sha256: Some("0".repeat(64)),
            successor_acceptance_campaign_sha256: Some("1".repeat(64)),
            successor_acceptance_topology_sha256: Some("2".repeat(64)),
            successor_acceptance_production_contract_sha256: Some("3".repeat(64)),
            successor_acceptance_panel_sha256: Some("4".repeat(64)),
            successor_acceptance_panel_scientific_sha256: Some("5".repeat(64)),
            successor_acceptance_discovery_authority_sha256: Some("5".repeat(64)),
            successor_acceptance_confirmation_authority_sha256: Some("6".repeat(64)),
        };
        assert!(require_implemented_gate("G_GRAPH_RESIDENT").is_ok());
        assert!(require_implemented_gate("G_GHOST_CONSERVATION").is_ok());
        let rejected = GATE_ORDER
            .iter()
            .copied()
            .filter(|gate_id| !matches!(*gate_id, "G_GRAPH_RESIDENT" | "G_GHOST_CONSERVATION"))
            .map(|gate_id| {
                let error = require_implemented_gate(gate_id).unwrap_err().to_string();
                assert!(
                    error.starts_with("EVIDENCE_NOT_FOUND:"),
                    "{gate_id}: {error}"
                );
                let mut args = base_args.clone();
                args.successor_acceptance_gate_id = Some(gate_id.to_string());
                let initialize_error = initialize_runtime_acceptance(
                    &args,
                    true,
                    GOVERNED_HOTPATH_MIN_LAUNCH_COLLAPSE,
                    true,
                    RuntimeProvenance::capture(temp.path(), vec!["test".to_string()]),
                )
                .unwrap_err()
                .to_string();
                assert!(
                    initialize_error.starts_with("EVIDENCE_NOT_FOUND:"),
                    "{gate_id}: {initialize_error}"
                );
                gate_id
            })
            .collect::<Vec<_>>();
        assert_eq!(rejected.len(), 24);
    }

    #[test]
    fn unresolved_ledger_has_exact_other_gate_order_and_reason_classes() {
        let ledger = unresolved_gate_ledger("G_GRAPH_RESIDENT");
        assert_eq!(ledger.len(), 25);
        let expected = GATE_ORDER
            .iter()
            .copied()
            .filter(|gate_id| *gate_id != "G_GRAPH_RESIDENT")
            .collect::<Vec<_>>();
        let observed = ledger
            .iter()
            .map(|row| row["gate_id"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(observed, expected);
        assert_eq!(observed.iter().copied().collect::<BTreeSet<_>>().len(), 25);
        let classes = ledger
            .iter()
            .fold(BTreeMap::<&str, usize>::new(), |mut counts, row| {
                *counts
                    .entry(row["reason_class"].as_str().unwrap())
                    .or_default() += 1;
                counts
            });
        assert_eq!(classes.get("UNMAPPED_SINGLE_RUN_GATE"), Some(&20));
        assert_eq!(
            classes.get("SEPARATE_GOVERNED_EXECUTION_REQUIRED"),
            Some(&1)
        );
        assert_eq!(classes.get("CROSS_RUN_AUTHORITY_REQUIRED"), Some(&1));
        assert_eq!(classes.get("FULL_BENCHMARK_AUTHORITY_REQUIRED"), Some(&1));
        assert_eq!(
            classes.get("POST_REBLESS_CANONICAL_AUTHORITY_REQUIRED"),
            Some(&1)
        );
        assert_eq!(classes.get("POST_REBLESS_SEAL_READBACK_REQUIRED"), Some(&1));
    }

    #[test]
    fn graph_admission_requires_bound_governed_hotpath_contract() {
        assert!(
            validate_graph_hotpath_admission(true, GOVERNED_HOTPATH_MIN_LAUNCH_COLLAPSE).is_ok()
        );
        assert!(
            validate_graph_hotpath_admission(false, GOVERNED_HOTPATH_MIN_LAUNCH_COLLAPSE)
                .unwrap_err()
                .to_string()
                .contains("--require-hotpath-acceleration")
        );
        assert!(validate_graph_hotpath_admission(true, 2.0)
            .unwrap_err()
            .to_string()
            .contains("exact governed"));
    }

    #[test]
    fn gate_specific_admission_separates_graph_hotpath_from_ghost_intent() {
        assert!(validate_gate_runtime_admission(
            "G_GRAPH_RESIDENT",
            true,
            GOVERNED_HOTPATH_MIN_LAUNCH_COLLAPSE,
            false,
        )
        .is_ok());
        assert!(validate_gate_runtime_admission(
            "G_GRAPH_RESIDENT",
            false,
            GOVERNED_HOTPATH_MIN_LAUNCH_COLLAPSE,
            true,
        )
        .unwrap_err()
        .to_string()
        .contains("--require-hotpath-acceleration"));
        assert!(validate_gate_runtime_admission("G_GHOST_CONSERVATION", false, 0.0, true,).is_ok());
        assert!(validate_gate_runtime_admission(
            "G_GHOST_CONSERVATION",
            true,
            GOVERNED_HOTPATH_MIN_LAUNCH_COLLAPSE,
            false,
        )
        .unwrap_err()
        .to_string()
        .starts_with("EVIDENCE_NOT_FOUND:"));
    }

    #[test]
    fn generic_receipt_pass_or_lowered_threshold_cannot_pass_governed_collapse() {
        assert!(!governed_graph_collapse_valid(
            false,
            GOVERNED_HOTPATH_MIN_LAUNCH_COLLAPSE,
            "PASS",
            128.0
        ));
        assert!(!governed_graph_collapse_valid(true, 2.0, "PASS", 128.0));
        assert!(!governed_graph_collapse_valid(
            true,
            GOVERNED_HOTPATH_MIN_LAUNCH_COLLAPSE,
            "PASS",
            2.0
        ));
        assert!(!governed_graph_collapse_valid(
            true,
            GOVERNED_HOTPATH_MIN_LAUNCH_COLLAPSE,
            "FAIL",
            128.0
        ));
        assert!(governed_graph_collapse_valid(
            true,
            GOVERNED_HOTPATH_MIN_LAUNCH_COLLAPSE,
            "PASS",
            GOVERNED_HOTPATH_MIN_LAUNCH_COLLAPSE
        ));
    }

    fn passing_graph_resident_population() -> BTreeMap<String, Value> {
        BTreeMap::from([
            ("governed_stream_count".to_string(), serde_json::json!(8)),
            ("counted_window_count".to_string(), serde_json::json!(32)),
            (
                "resident_dispatch_count".to_string(),
                serde_json::json!(128),
            ),
            (
                "all_counted_windows_resident".to_string(),
                serde_json::json!(true),
            ),
            ("exact_tail_valid".to_string(), serde_json::json!(true)),
            (
                "required_collapse_valid".to_string(),
                serde_json::json!(true),
            ),
            (
                "legacy_graph_replay_dispatch_count".to_string(),
                serde_json::json!(0),
            ),
            ("window_gap_count".to_string(), serde_json::json!(0)),
            ("window_overlap_count".to_string(), serde_json::json!(0)),
        ])
    }

    fn passing_ghost_conservation_population() -> BTreeMap<String, Value> {
        BTreeMap::from([
            ("produced".to_string(), serde_json::json!(16)),
            ("persisted".to_string(), serde_json::json!(16)),
            ("decoded".to_string(), serde_json::json!(16)),
            ("lattice_nodes".to_string(), serde_json::json!(16)),
            ("byte_lineage_exact".to_string(), serde_json::json!(true)),
            (
                "per_stream_conservation_pass".to_string(),
                serde_json::json!(true),
            ),
            (
                "versioned_membership_payload_pass".to_string(),
                serde_json::json!(true),
            ),
            (
                "membership_truncation_count".to_string(),
                serde_json::json!(0),
            ),
            ("duplicate_count".to_string(), serde_json::json!(0)),
            ("gap_count".to_string(), serde_json::json!(0)),
            ("overflow_loss_count".to_string(), serde_json::json!(0)),
        ])
    }

    fn test_runtime_session(directory: &Path, published: PathBuf) -> RuntimeSession {
        let hash = "a".repeat(64);
        let runtime_provenance =
            RuntimeProvenance::capture(directory, vec!["nhs_rt_full-test".to_string()]);
        let engine = FileBinding {
            path: "/authority/nhs_rt_full".to_string(),
            sha256: runtime_provenance
                .packaged_binary_sha256
                .clone()
                .expect("test executable hash"),
            size_bytes: 1,
        };
        let binding = |name: &str| FileBinding {
            path: format!("/authority/{name}"),
            sha256: hash.clone(),
            size_bytes: 1,
        };
        RuntimeSession {
            gate_id: "G_GRAPH_RESIDENT".to_string(),
            execution_id: "test-execution".to_string(),
            gate_contract_sha256: "c".repeat(64),
            contract_path: PathBuf::from("/unread/test-contract"),
            contract_raw: Vec::new(),
            authorities: ContractAuthorities {
                producer: engine.clone(),
                candidate_build_manifest: binding("candidate-build.json"),
                source_snapshot: binding("source-snapshot.json"),
                measurement_implementation: binding("measurement.rs"),
                measurement_implementation_manifest: binding("measurement-manifest.json"),
            },
            authority_bindings: AuthorityBindings {
                execution_authority_sha256: hash.clone(),
                campaign_sha256: hash.clone(),
                topology_sha256: hash.clone(),
                production_contract_sha256: hash.clone(),
                panel_sha256: hash.clone(),
                panel_scientific_sha256: "b".repeat(64),
                discovery_authority_sha256: hash.clone(),
                confirmation_authority_sha256: hash,
            },
            invocation: InvocationIdentity {
                executable: engine,
                command_identity_sha256: "d".repeat(64),
            },
            raw_runtime_path: published.join("runtime.jsonl"),
            observation_source_path: published.join("observations.json"),
            gate_evidence_path: published.join("producer.json"),
            transaction_closeout_receipt_path: published.join("transaction-closeout.json"),
            values: passing_graph_resident_population(),
            runtime_provenance,
            graph_opcode_counts_by_stream: (0_u32..8)
                .map(|stream_id| {
                    (
                        stream_id,
                        BTreeMap::from([
                            ("MD.PROTOCOL_DIRECTOR".to_string(), 4),
                            ("MD.VELOCITY_CLAMP".to_string(), 4),
                        ]),
                    )
                })
                .collect(),
            ghost_conservation_records_by_stream: None,
        }
    }

    fn test_ghost_runtime_session(directory: &Path, published: PathBuf) -> RuntimeSession {
        let mut session = test_runtime_session(directory, published);
        session.gate_id = "G_GHOST_CONSERVATION".to_string();
        session.values = passing_ghost_conservation_population();
        session.graph_opcode_counts_by_stream.clear();
        session.ghost_conservation_records_by_stream =
            Some((0_u16..8).map(|stream_id| (stream_id, 2_u64)).collect());
        session
    }

    #[test]
    fn complete_direct_graph_population_passes_machine_rule() {
        validate_gate_observations("G_GRAPH_RESIDENT", &passing_graph_resident_population())
            .unwrap();
    }

    #[test]
    fn complete_direct_ghost_population_passes_machine_rule_and_adversarial_fields_fail() {
        let population = passing_ghost_conservation_population();
        validate_gate_observations("G_GHOST_CONSERVATION", &population).unwrap();
        for (field, corrupt) in [
            ("produced", serde_json::json!(0)),
            ("persisted", serde_json::json!(15)),
            ("decoded", serde_json::json!(15)),
            ("lattice_nodes", serde_json::json!(15)),
            ("byte_lineage_exact", serde_json::json!(false)),
            ("per_stream_conservation_pass", serde_json::json!(false)),
            (
                "versioned_membership_payload_pass",
                serde_json::json!(false),
            ),
            ("membership_truncation_count", serde_json::json!(1)),
            ("duplicate_count", serde_json::json!(1)),
            ("gap_count", serde_json::json!(1)),
            ("overflow_loss_count", serde_json::json!(1)),
        ] {
            let mut adversarial = population.clone();
            adversarial.insert(field.to_string(), corrupt);
            assert!(
                validate_gate_observations("G_GHOST_CONSERVATION", &adversarial).is_err(),
                "corrupted Ghost conservation field unexpectedly passed: {field}"
            );
        }
    }

    #[test]
    fn copied_summary_and_nonmember_population_fail_closed() {
        let population = passing_graph_resident_population();
        let serialized = canonical_json(&population).unwrap();
        let mut admitted = required_observation_fields("G_GRAPH_RESIDENT").unwrap();
        admitted.remove("window_overlap_count");
        assert!(
            import_serialized_population_for_test("G_GRAPH_RESIDENT", &serialized, &admitted)
                .unwrap_err()
                .to_string()
                .contains("not the exact admitted direct-member population")
        );

        let mut nonmember = population;
        nonmember.insert("post_run_scalar_summary".to_string(), serde_json::json!(0));
        assert!(validate_gate_observations("G_GRAPH_RESIDENT", &nonmember)
            .unwrap_err()
            .to_string()
            .contains("exact observation population differs"));
    }

    #[test]
    fn machine_rule_rejects_default_zero_positive_field() {
        let mut population = passing_graph_resident_population();
        population.insert("resident_dispatch_count".to_string(), serde_json::json!(0));
        assert!(validate_gate_observations("G_GRAPH_RESIDENT", &population)
            .unwrap_err()
            .to_string()
            .contains("not positive"));
    }

    #[test]
    fn sibling_transaction_is_no_clobber_and_idempotent_only_for_exact_bytes() {
        let directory = tempfile::tempdir().unwrap();
        let published = directory.path().join("published");
        let first = published.join("raw.jsonl");
        let second = published.join("source.json");
        let artifacts = vec![
            (first.clone(), b"{\"field\":1}\n".to_vec()),
            (second.clone(), b"{\"source\":true}\n".to_vec()),
        ];
        durable_sibling_transaction(&artifacts).unwrap();
        durable_sibling_transaction(&artifacts).unwrap();
        assert_eq!(std::fs::read(&first).unwrap(), artifacts[0].1);
        assert_eq!(std::fs::read(&second).unwrap(), artifacts[1].1);
        assert!(published.join(TRANSACTION_COMMIT_MARKER).is_file());

        let conflicting = vec![
            (first.clone(), b"{\"field\":2}\n".to_vec()),
            (second.clone(), artifacts[1].1.clone()),
        ];
        assert!(durable_sibling_transaction(&conflicting)
            .unwrap_err()
            .to_string()
            .contains("collision"));
        assert_eq!(std::fs::read(&first).unwrap(), artifacts[0].1);
    }

    #[test]
    fn raw_jsonl_locator_excludes_delimiter_newline() {
        let first = canonical_json(&serde_json::json!({"value": 1})).unwrap();
        let second = canonical_json(&serde_json::json!({"value": 2})).unwrap();
        let mut journal = first.clone();
        journal.push(b'\n');
        let second_offset = journal.len();
        journal.extend_from_slice(&second);
        journal.push(b'\n');
        assert_eq!(&journal[..first.len()], first.as_slice());
        assert_eq!(journal[first.len()], b'\n');
        assert_eq!(
            &journal[second_offset..second_offset + second.len()],
            second.as_slice()
        );
        assert_eq!(journal[second_offset + second.len()], b'\n');
    }

    #[test]
    fn complete_graph_population_builds_and_publishes_reconstructable_artifacts() {
        let directory = tempfile::tempdir().unwrap();
        let published = directory.path().join("gate-transaction");
        let hash = "a".repeat(64);
        let runtime_provenance =
            RuntimeProvenance::capture(directory.path(), vec!["nhs_rt_full-test".to_string()]);
        let binding = |name: &str| FileBinding {
            path: format!("/authority/{name}"),
            sha256: hash.clone(),
            size_bytes: 1,
        };
        let engine = FileBinding {
            path: "/authority/nhs_rt_full".to_string(),
            sha256: runtime_provenance
                .packaged_binary_sha256
                .clone()
                .expect("test executable hash"),
            size_bytes: 1,
        };
        let session = RuntimeSession {
            gate_id: "G_GRAPH_RESIDENT".to_string(),
            execution_id: "test-execution".to_string(),
            gate_contract_sha256: "c".repeat(64),
            contract_path: PathBuf::from("/unread/test-contract"),
            contract_raw: Vec::new(),
            authorities: ContractAuthorities {
                producer: engine.clone(),
                candidate_build_manifest: binding("candidate-build.json"),
                source_snapshot: binding("source-snapshot.json"),
                measurement_implementation: binding("measurement.rs"),
                measurement_implementation_manifest: binding("measurement-manifest.json"),
            },
            authority_bindings: AuthorityBindings {
                execution_authority_sha256: hash.clone(),
                campaign_sha256: hash.clone(),
                topology_sha256: hash.clone(),
                production_contract_sha256: hash.clone(),
                panel_sha256: hash.clone(),
                panel_scientific_sha256: "b".repeat(64),
                discovery_authority_sha256: hash.clone(),
                confirmation_authority_sha256: hash,
            },
            invocation: InvocationIdentity {
                executable: engine,
                command_identity_sha256: "d".repeat(64),
            },
            raw_runtime_path: published.join("runtime.jsonl"),
            observation_source_path: published.join("observations.json"),
            gate_evidence_path: published.join("producer.json"),
            transaction_closeout_receipt_path: published.join("transaction-closeout.json"),
            values: passing_graph_resident_population(),
            runtime_provenance,
            graph_opcode_counts_by_stream: (0_u32..8)
                .map(|stream_id| {
                    (
                        stream_id,
                        BTreeMap::from([
                            ("MD.PROTOCOL_DIRECTOR".to_string(), 4),
                            ("MD.VELOCITY_CLAMP".to_string(), 4),
                        ]),
                    )
                })
                .collect(),
            ghost_conservation_records_by_stream: None,
        };
        let (artifacts, summary) = assemble_closeout_artifacts(&session).unwrap();
        assert_eq!(summary.measurement_member_count, 9);
        durable_sibling_transaction(&artifacts).unwrap();

        let source = strict_json(
            &read_regular_nofollow(&session.observation_source_path, "test source").unwrap(),
            "test source",
        )
        .unwrap();
        let members = source["body"]["members"].as_array().unwrap();
        assert_eq!(members.len(), 9);
        let raw = read_regular_nofollow(&session.raw_runtime_path, "test raw").unwrap();
        for member in members {
            let locator = &member["body"]["raw_record_locator"];
            let offset = locator["byte_offset"].as_u64().unwrap() as usize;
            let length = locator["byte_length"].as_u64().unwrap() as usize;
            let record_raw = &raw[offset..offset + length];
            assert_ne!(record_raw.last(), Some(&b'\n'));
            assert_eq!(sha256_bytes(record_raw), locator["record_sha256"]);
            let record = strict_json(record_raw, "test located record").unwrap();
            assert_eq!(record["field"], member["body"]["field"]);
            assert_eq!(record["value"], member["body"]["value"]);
        }
        let closeout = strict_json(
            &read_regular_nofollow(
                &session.transaction_closeout_receipt_path,
                "test transaction closeout",
            )
            .unwrap(),
            "test transaction closeout",
        )
        .unwrap();
        assert_eq!(
            closeout["schema_kind"],
            "prismos_successor_acceptance_transaction_closeout_receipt"
        );
        assert_eq!(
            closeout["body"]["unresolved_gate_ledger"]
                .as_array()
                .unwrap()
                .len(),
            25
        );
        assert_eq!(
            closeout["body"]["runtimeinline_typed_qnanl_authority"]["status"],
            "SUPPORTING_EVIDENCE"
        );
        assert_eq!(
            closeout["body"]["runtimeinline_typed_qnanl_authority"]["g_prismcode_qnanl_status"],
            "EVIDENCE_NOT_FOUND"
        );
        let packet = read_regular_nofollow(
            Path::new(&summary.runtimeinline_qnanl_packet.path),
            "test RuntimeInline QNANL packet",
        )
        .unwrap();
        assert_eq!(decode_packet_bytes(&packet).unwrap(), raw);
        let receipt = strict_json(
            &read_regular_nofollow(
                Path::new(&summary.runtimeinline_qnanl_receipt.path),
                "test RuntimeInline QNANL receipt",
            )
            .unwrap(),
            "test RuntimeInline QNANL receipt",
        )
        .unwrap();
        assert_eq!(receipt["runtime_origin"], "RUNTIME_INLINE");
        assert!(receipt["runtime_provenance"].is_object());
        assert!(receipt["pcode_refs"].as_array().unwrap().len() >= 4);
        assert!(published.join(TRANSACTION_COMMIT_MARKER).is_file());
    }

    #[test]
    fn ghost_conservation_builds_exact_runtimeinline_opcode_population() {
        let directory = tempfile::tempdir().unwrap();
        let session = test_ghost_runtime_session(
            directory.path(),
            directory.path().join("ghost-conservation-transaction"),
        );
        let (artifacts, summary) = assemble_closeout_artifacts(&session).unwrap();
        assert_eq!(summary.measurement_member_count, 11);
        let receipt_raw = artifacts
            .iter()
            .find(|(path, _)| path.to_string_lossy().ends_with(".prismqnanl.receipt.json"))
            .map(|(_, raw)| raw)
            .unwrap();
        let receipt = strict_json(receipt_raw, "Ghost RuntimeInline QNANL receipt").unwrap();
        assert_eq!(receipt["runtime_origin"], "RUNTIME_INLINE");
        assert_eq!(receipt["artifact_family"], "GHOST_CONSERVATION");
        let ghost_ref = receipt["pcode_refs"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["opcode_name"] == "MD.GHOST_CONSERVATION")
            .unwrap();
        assert!(ghost_ref["opcode_id"]
            .as_str()
            .is_some_and(|opcode_id| opcode_id.starts_with("0x")));
        let operands = ghost_ref["operands"].as_array().unwrap();
        assert_eq!(
            operands
                .iter()
                .filter(|operand| operand.as_str().unwrap().starts_with("stream_"))
                .count(),
            8
        );
        assert!(operands
            .iter()
            .any(|operand| operand == "run_total_record_count=16"));

        let closeout_raw = artifacts
            .iter()
            .find(|(path, _)| path.ends_with("transaction-closeout.json"))
            .map(|(_, raw)| raw)
            .unwrap();
        let closeout = strict_json(closeout_raw, "Ghost transaction closeout").unwrap();
        let authority = &closeout["body"]["runtimeinline_typed_qnanl_authority"];
        assert_eq!(authority["opcode_population"].as_array().unwrap().len(), 1);
        assert_eq!(
            authority["opcode_population"][0]["opcode_name"],
            "MD.GHOST_CONSERVATION"
        );
        assert!(authority["reason"]
            .as_str()
            .unwrap()
            .contains("RuntimeSubprocess Ghost packets are excluded"));

        let mut missing_opcode = test_ghost_runtime_session(
            directory.path(),
            directory.path().join("ghost-conservation-missing-opcode"),
        );
        missing_opcode.ghost_conservation_records_by_stream = None;
        assert!(assemble_closeout_artifacts(&missing_opcode)
            .unwrap_err()
            .to_string()
            .contains("RuntimeInline opcode population absent"));
        let mut wrong_total = test_ghost_runtime_session(
            directory.path(),
            directory.path().join("ghost-conservation-wrong-total"),
        );
        wrong_total.ghost_conservation_records_by_stream = Some(BTreeMap::from([(0_u16, 15_u64)]));
        assert!(assemble_closeout_artifacts(&wrong_total)
            .unwrap_err()
            .to_string()
            .contains("RuntimeInline opcode population differs"));
    }

    #[test]
    fn runtimeinline_qnanl_requires_pre_md_provenance_and_preserves_unknown_names() {
        let directory = tempfile::tempdir().unwrap();
        let mut session = test_runtime_session(
            directory.path(),
            directory.path().join("unknown-opcode-transaction"),
        );
        for counts in session.graph_opcode_counts_by_stream.values_mut() {
            counts.insert("MD.FUTURE_RESIDENT_GRAPH_OPCODE".to_string(), 3);
        }
        let (artifacts, _) = assemble_closeout_artifacts(&session).unwrap();
        let receipt_raw = artifacts
            .iter()
            .find(|(path, _)| path.to_string_lossy().ends_with(".prismqnanl.receipt.json"))
            .map(|(_, raw)| raw)
            .unwrap();
        let receipt = strict_json(receipt_raw, "unknown-opcode receipt").unwrap();
        let references = receipt["pcode_refs"].as_array().unwrap();
        let unknown = references
            .iter()
            .find(|row| row["opcode_name"] == "MD.FUTURE_RESIDENT_GRAPH_OPCODE")
            .unwrap();
        assert!(unknown["opcode_id"].is_null());
        assert!(references
            .iter()
            .any(|row| row["opcode_name"] == "MD.QNANL_SEAL"));
        assert!(references
            .iter()
            .any(|row| row["opcode_name"] == "MD.QNANL_VERIFY"));

        session.runtime_provenance.packaged_binary_sha256 = None;
        session
            .runtime_provenance
            .packaged_binary_sha256_absent_reason =
            Some("adversarial missing provenance".to_string());
        assert!(assemble_closeout_artifacts(&session)
            .unwrap_err()
            .to_string()
            .contains("executable provenance binding failed"));
    }

    #[test]
    fn ordinary_closeout_and_failure_diagnostic_cannot_impersonate_typed_gate_evidence() {
        let directory = tempfile::tempdir().unwrap();
        let session = test_runtime_session(
            directory.path(),
            directory.path().join("diagnostic-boundary-transaction"),
        );
        let (success_artifacts, _) = assemble_closeout_artifacts(&session).unwrap();
        let closeout = success_artifacts
            .iter()
            .find(|(path, _)| path.ends_with("transaction-closeout.json"))
            .map(|(_, raw)| strict_json(raw, "ordinary closeout").unwrap())
            .unwrap();
        assert_eq!(
            closeout["body"]["runtimeinline_typed_qnanl_authority"]["g_prismcode_qnanl_status"],
            "EVIDENCE_NOT_FOUND"
        );
        assert_ne!(
            closeout["schema_kind"],
            prismcode::qnanl_artifact::QNANL_ARTIFACT_RECEIPT_KIND
        );

        let (diagnostic_artifacts, _) =
            failure_diagnostic_artifacts(&session, "ADVERSARIAL", "failure").unwrap();
        let diagnostic = strict_json(&diagnostic_artifacts[0].1, "failure diagnostic").unwrap();
        assert_eq!(diagnostic["body"]["gate_satisfaction_capability"], "NONE");
        assert_eq!(diagnostic["body"]["gate_outcome"], "NOT_EVALUATED");
        assert_eq!(diagnostic["body"]["gate_pass_claim_count"], 0);
        assert_eq!(diagnostic["body"]["success_transaction_published"], false);
        assert_eq!(
            diagnostic["body"]["unresolved_gate_ledger"]
                .as_array()
                .unwrap()
                .len(),
            26
        );
        assert!(
            !String::from_utf8_lossy(&diagnostic_artifacts[0].1).contains("\"status\":\"PASS\"")
        );
    }

    #[test]
    fn typed_member_corruption_collision_and_partial_population_fail_closed() {
        let directory = tempfile::tempdir().unwrap();
        let session = test_runtime_session(
            directory.path(),
            directory.path().join("typed-corruption-transaction"),
        );
        let (artifacts, _) = assemble_closeout_artifacts(&session).unwrap();
        durable_sibling_transaction(&artifacts).unwrap();
        let receipt_path = artifacts
            .iter()
            .find(|(path, _)| path.to_string_lossy().ends_with(".prismqnanl.receipt.json"))
            .map(|(path, _)| path)
            .unwrap();
        std::fs::write(receipt_path, b"{\"corrupted\":true}\n").unwrap();
        assert!(durable_sibling_transaction(&artifacts)
            .unwrap_err()
            .to_string()
            .contains("different member bytes"));

        let partial_root = directory.path().join("partial-transaction");
        std::fs::create_dir(&partial_root).unwrap();
        std::fs::write(partial_root.join("only.json"), b"{}\n").unwrap();
        let expected = vec![
            (partial_root.join("only.json"), b"{}\n".to_vec()),
            (partial_root.join("missing.json"), b"{}\n".to_vec()),
        ];
        assert!(durable_sibling_transaction(&expected)
            .unwrap_err()
            .to_string()
            .contains("different member population"));
    }

    #[test]
    fn terminal_race_elects_one_owner_and_success_failure_are_mutually_exclusive() {
        let coordinator = std::sync::Arc::new(TerminalCoordinator::new());
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(3));
        let success = {
            let coordinator = coordinator.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                coordinator.try_claim(TERMINAL_SUCCESS)
            })
        };
        let failure = {
            let coordinator = coordinator.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                coordinator.try_claim(TERMINAL_FAILURE)
            })
        };
        barrier.wait();
        let success_won = success.join().unwrap();
        let failure_won = failure.join().unwrap();
        assert_ne!(success_won, failure_won);
        assert!(!coordinator.try_claim(TERMINAL_SUCCESS));
        assert!(!coordinator.try_claim(TERMINAL_FAILURE));
    }
}
