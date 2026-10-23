use clap::{Parser, Subcommand};
use prismcode::binary_runtime::{execute_packet, Backend, ExecutionReport, PcError, PcResult};
use prismcode::operand_ledger::sha256_file_hex;
use prismcode::qnanl_artifact::{
    decode_packet_bytes, QNANL_ARTIFACT_RECEIPT_KIND, QNANL_ARTIFACT_SCHEMA_VERSION,
};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const CANDIDATE_PAYLOAD_SCHEMA: &str = "prismos_gflownet_candidate_evidence_packet";
const CANDIDATE_BINDING_SCHEMA: &str = "prismos_gflownet_source_artifact_binding_contract";
const CONSUMPTION_RECEIPT_SCHEMA: &str =
    "prismcode_gflownet_candidate_evidence_consumption_receipt";
const ACCEPTED_RUN_ID: &str = "PRISM_RUNPOD_OFFLOAD_20260724T184355Z_GFLOWNET_V5_ACTIVE";
const ACCEPTED_METHOD: &str = "v4_full_stack";
const ACCEPTED_POLICY_SHA256: &str =
    "3a0a731752b4cd8254d6450864955ce475426c1c9bd70143f0a0a685e5a26014";
const ACCEPTED_CHECKPOINT_SHA256: &str =
    "a2e98f99f3ec1abf09cddf1b6433a8e1a5ec1fa4a562076c1aa263c511e1dc48";
const ACCEPTED_TERMINAL_EPOCH: u64 = 459;
const ACCEPTED_CONVERGENCE_REASON: &str = "unique_exhausted";
const ACCEPTED_RANKING_METHOD: &str =
    "compute_effective_reward_tensor_then_apply_meta_reward_guidance";
const REQUIRED_COMPUTATIONAL_BOUNDARY: &str =
    "All claims are computational and PRISM-derived unless explicitly stated otherwise. Wet-lab validation remains required before therapeutic, safety, clinical, regulatory, synthesis, or ADMET claims.";

#[derive(Debug, Parser)]
#[command(name = "prismrun", about = "Execute PRISMCODE packets")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Run {
        input: PathBuf,
        #[arg(long, default_value = "cpu")]
        backend: String,
        #[arg(long)]
        out: PathBuf,

        /// Canonical GFlowNet candidate evidence payload emitted by the packet builder.
        #[arg(long)]
        candidate_evidence_payload: Option<PathBuf>,
        /// QNANL packet whose decoded decision body contains the candidate payload.
        #[arg(long)]
        candidate_qnanl_packet: Option<PathBuf>,
        /// Native prismqnanl receipt for the candidate QNANL packet.
        #[arg(long)]
        candidate_qnanl_receipt: Option<PathBuf>,
        /// Exact SHA256 of --candidate-evidence-payload.
        #[arg(long)]
        candidate_evidence_payload_sha256: Option<String>,
        /// Exact SHA256 of --candidate-qnanl-packet.
        #[arg(long)]
        candidate_qnanl_packet_sha256: Option<String>,
        /// Exact SHA256 of --candidate-qnanl-receipt.
        #[arg(long)]
        candidate_qnanl_receipt_sha256: Option<String>,
        /// Commit-last, no-clobber PrismCode receipt proving candidate evidence consumption.
        #[arg(long)]
        candidate_consumption_receipt: Option<PathBuf>,
    },
}

#[derive(Debug)]
struct CandidateEvidenceOperands {
    payload: PathBuf,
    packet: PathBuf,
    qnanl_receipt: PathBuf,
    payload_sha256: String,
    packet_sha256: String,
    qnanl_receipt_sha256: String,
    consumption_receipt: PathBuf,
}

#[derive(Debug)]
struct VerifiedSourceBinding {
    artifact_class: String,
    sha256: String,
    format: String,
    path: PathBuf,
    bytes: Vec<u8>,
}

impl CandidateEvidenceOperands {
    #[allow(clippy::too_many_arguments)]
    fn from_options(
        payload: Option<PathBuf>,
        packet: Option<PathBuf>,
        qnanl_receipt: Option<PathBuf>,
        payload_sha256: Option<String>,
        packet_sha256: Option<String>,
        qnanl_receipt_sha256: Option<String>,
        consumption_receipt: Option<PathBuf>,
    ) -> PcResult<Option<Self>> {
        let present = [
            payload.is_some(),
            packet.is_some(),
            qnanl_receipt.is_some(),
            payload_sha256.is_some(),
            packet_sha256.is_some(),
            qnanl_receipt_sha256.is_some(),
            consumption_receipt.is_some(),
        ];
        if present.iter().all(|value| !value) {
            return Ok(None);
        }
        if !present.iter().all(|value| *value) {
            return Err(security(
                "candidate evidence is an all-or-none operand group: payload, QNANL packet, "
                    .to_string()
                    + "QNANL receipt, all three exact SHA256 values, and a consumption receipt "
                    + "path are required",
            ));
        }
        Ok(Some(Self {
            payload: payload.expect("presence checked"),
            packet: packet.expect("presence checked"),
            qnanl_receipt: qnanl_receipt.expect("presence checked"),
            payload_sha256: exact_sha256(
                &payload_sha256.expect("presence checked"),
                "candidate evidence payload SHA256",
            )?,
            packet_sha256: exact_sha256(
                &packet_sha256.expect("presence checked"),
                "candidate QNANL packet SHA256",
            )?,
            qnanl_receipt_sha256: exact_sha256(
                &qnanl_receipt_sha256.expect("presence checked"),
                "candidate QNANL receipt SHA256",
            )?,
            consumption_receipt: consumption_receipt.expect("presence checked"),
        }))
    }
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(match err {
            PcError::Gpu(_) => 12,
            _ => 2,
        });
    }
}

fn run() -> PcResult<()> {
    match Cli::parse().command {
        Command::Run {
            input,
            backend,
            out,
            candidate_evidence_payload,
            candidate_qnanl_packet,
            candidate_qnanl_receipt,
            candidate_evidence_payload_sha256,
            candidate_qnanl_packet_sha256,
            candidate_qnanl_receipt_sha256,
            candidate_consumption_receipt,
        } => {
            let candidate_operands = CandidateEvidenceOperands::from_options(
                candidate_evidence_payload,
                candidate_qnanl_packet,
                candidate_qnanl_receipt,
                candidate_evidence_payload_sha256,
                candidate_qnanl_packet_sha256,
                candidate_qnanl_receipt_sha256,
                candidate_consumption_receipt,
            )?;
            let candidate_binding = match candidate_operands.as_ref() {
                Some(operands) => {
                    if out.exists() {
                        return Err(security(format!(
                            "candidate-bound execution refuses to overwrite PCOUT {}",
                            out.display()
                        )));
                    }
                    if operands.consumption_receipt.exists() {
                        return Err(security(format!(
                            "candidate consumption receipt already exists: {}",
                            operands.consumption_receipt.display()
                        )));
                    }
                    Some(verify_candidate_evidence(operands)?)
                }
                None => None,
            };

            let backend = Backend::parse(&backend)?;
            let execution = execute_packet(&input, backend, &out)?;
            if let (Some(operands), Some(binding)) =
                (candidate_operands.as_ref(), candidate_binding)
            {
                let receipt =
                    bind_execution_to_candidate_evidence(binding, &input, &out, &execution)?;
                write_json_no_clobber(&operands.consumption_receipt, &receipt)?;
                let receipt_sha256 = sha256_file_hex(&operands.consumption_receipt)
                    .map_err(|err| PcError::Io(err.to_string()))?;
                println!(
                    "{}",
                    serde_json::to_string_pretty(&json!({
                        "status": "PASS",
                        "execution": execution,
                        "candidate_evidence_consumption": receipt,
                        "candidate_consumption_receipt": {
                            "path": operands.consumption_receipt,
                            "sha256": receipt_sha256,
                        },
                    }))?
                );
            } else {
                println!("{}", serde_json::to_string_pretty(&execution)?);
            }
            Ok(())
        }
    }
}

fn security(message: impl Into<String>) -> PcError {
    PcError::Security(message.into())
}

fn exact_sha256(value: &str, label: &str) -> PcResult<String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(security(format!(
            "{label} must be exactly 64 lowercase hexadecimal characters"
        )));
    }
    Ok(value.to_string())
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn canonical_json_bytes(value: &Value) -> PcResult<Vec<u8>> {
    serde_json::to_vec_pretty(value).map_err(PcError::from)
}

fn require_object<'a>(value: &'a Value, label: &str) -> PcResult<&'a Map<String, Value>> {
    value
        .as_object()
        .ok_or_else(|| security(format!("{label} must be a JSON object")))
}

fn require_array<'a>(value: &'a Value, label: &str) -> PcResult<&'a Vec<Value>> {
    value
        .as_array()
        .ok_or_else(|| security(format!("{label} must be a JSON array")))
}

fn require_str<'a>(value: &'a Value, label: &str) -> PcResult<&'a str> {
    value
        .as_str()
        .filter(|text| !text.is_empty())
        .ok_or_else(|| security(format!("{label} must be a non-empty string")))
}

fn require_u64(value: &Value, label: &str) -> PcResult<u64> {
    value
        .as_u64()
        .ok_or_else(|| security(format!("{label} must be a non-negative integer")))
}

fn required<'a>(object: &'a Map<String, Value>, key: &str, label: &str) -> PcResult<&'a Value> {
    object
        .get(key)
        .ok_or_else(|| security(format!("{label}.{key} is required")))
}

fn canonical_regular_file(path: &Path, label: &str) -> PcResult<PathBuf> {
    if !path.is_absolute() {
        return Err(security(format!(
            "{label} path must be absolute: {}",
            path.display()
        )));
    }
    let metadata = fs::symlink_metadata(path)
        .map_err(|err| PcError::Io(format!("cannot inspect {label} {}: {err}", path.display())))?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        return Err(security(format!(
            "{label} must be a regular non-symlink file: {}",
            path.display()
        )));
    }
    let canonical = fs::canonicalize(path).map_err(|err| {
        PcError::Io(format!(
            "cannot canonicalize {label} {}: {err}",
            path.display()
        ))
    })?;
    if canonical != path {
        return Err(security(format!(
            "{label} path is not canonical: declared={} canonical={}",
            path.display(),
            canonical.display()
        )));
    }
    Ok(canonical)
}

fn verify_file_hash(path: &Path, expected: &str, label: &str) -> PcResult<(PathBuf, u64)> {
    let canonical = canonical_regular_file(path, label)?;
    let observed = sha256_file_hex(&canonical).map_err(|err| {
        PcError::Io(format!(
            "cannot hash {label} {}: {err}",
            canonical.display()
        ))
    })?;
    if observed != expected {
        return Err(PcError::Hash(format!(
            "{label} SHA256 mismatch: expected={expected} observed={observed}"
        )));
    }
    let size = fs::metadata(&canonical)
        .map_err(|err| PcError::Io(err.to_string()))?
        .len();
    Ok((canonical, size))
}

fn verify_exact_policy(payload: &Map<String, Value>) -> PcResult<Value> {
    let policy = require_object(
        required(payload, "policy_identity", "candidate payload")?,
        "policy_identity",
    )?;
    let expected_strings = [
        ("run_id", ACCEPTED_RUN_ID),
        ("method", ACCEPTED_METHOD),
        ("policy_sha256", ACCEPTED_POLICY_SHA256),
        ("checkpoint_sha256", ACCEPTED_CHECKPOINT_SHA256),
        ("convergence_reason", ACCEPTED_CONVERGENCE_REASON),
        ("canonical_ranking_reward_method", ACCEPTED_RANKING_METHOD),
    ];
    for (key, expected) in expected_strings {
        let observed = require_str(
            required(policy, key, "policy_identity")?,
            &format!("policy_identity.{key}"),
        )?;
        if observed != expected {
            return Err(security(format!(
                "policy_identity.{key} does not match accepted v5 epoch-459 identity: expected={expected:?} observed={observed:?}"
            )));
        }
    }
    if require_u64(
        required(policy, "terminal_epoch", "policy_identity")?,
        "policy_identity.terminal_epoch",
    )? != ACCEPTED_TERMINAL_EPOCH
    {
        return Err(security(format!(
            "policy_identity.terminal_epoch must equal {ACCEPTED_TERMINAL_EPOCH}"
        )));
    }
    Ok(Value::Object(policy.clone()))
}

fn top_level_values_for_keys(object: &Map<String, Value>, keys: &BTreeSet<&str>) -> Vec<Value> {
    keys.iter()
        .filter_map(|key| object.get(*key).cloned())
        .collect()
}

fn verify_upstream_receipt_contents(
    bindings: &BTreeMap<String, VerifiedSourceBinding>,
    asserted: &Map<String, Value>,
) -> PcResult<()> {
    let mut documents = Vec::new();
    let accepted_statuses = ["PASS", "VERIFIED", "ACCEPTED", "IMPLEMENTATION_VERIFIED"];
    for (binding_id, binding) in bindings {
        if !matches!(
            binding.artifact_class.as_str(),
            "UPSTREAM_RECEIPT" | "UPSTREAM_ENVELOPE"
        ) {
            continue;
        }
        if binding.format != "JSON" {
            return Err(security(format!(
                "upstream receipt/envelope {binding_id} must be JSON for independent content verification"
            )));
        }
        let document: Value = serde_json::from_slice(&binding.bytes).map_err(|err| {
            security(format!(
                "upstream receipt/envelope {binding_id} is not valid JSON: {err}"
            ))
        })?;
        let object = require_object(&document, &format!("upstream receipt {binding_id}"))?;
        let schema_present = ["schema", "schema_kind", "schema_version"]
            .iter()
            .any(|key| {
                object
                    .get(*key)
                    .and_then(Value::as_str)
                    .is_some_and(|value| !value.is_empty())
            });
        if !schema_present {
            return Err(security(format!(
                "upstream receipt/envelope {binding_id} has no string schema identity"
            )));
        }
        let status_present = ["status", "implementation_status"].iter().any(|key| {
            object
                .get(*key)
                .and_then(Value::as_str)
                .is_some_and(|value| accepted_statuses.contains(&value))
        });
        if !status_present {
            return Err(security(format!(
                "upstream receipt/envelope {binding_id} has no accepted status"
            )));
        }
        let run_values = top_level_values_for_keys(object, &BTreeSet::from(["run_id"]));
        if run_values != [Value::String(ACCEPTED_RUN_ID.to_string())] {
            return Err(security(format!(
                "upstream receipt/envelope {binding_id} does not corroborate accepted top-level run_id {ACCEPTED_RUN_ID}: observed={run_values:?}"
            )));
        }
        documents.push(document);
    }
    if documents.is_empty() {
        return Err(security(
            "no upstream receipt/envelope content was independently parsed",
        ));
    }
    if require_u64(
        required(asserted, "receipt_count", "receipt content verification")?,
        "receipt content verification receipt_count",
    )? != documents.len() as u64
    {
        return Err(security(
            "asserted receipt_count differs from independently parsed receipt/envelope count",
        ));
    }

    let expected: [(&str, BTreeSet<&str>, Value); 7] = [
        ("run_id", BTreeSet::from(["run_id"]), json!(ACCEPTED_RUN_ID)),
        (
            "method",
            BTreeSet::from(["method", "reward_version"]),
            json!(ACCEPTED_METHOD),
        ),
        (
            "policy_sha256",
            BTreeSet::from(["policy_sha256"]),
            json!(ACCEPTED_POLICY_SHA256),
        ),
        (
            "checkpoint_sha256",
            BTreeSet::from(["checkpoint_sha256", "terminal_checkpoint_sha256"]),
            json!(ACCEPTED_CHECKPOINT_SHA256),
        ),
        (
            "terminal_epoch",
            BTreeSet::from([
                "terminal_epoch",
                "terminal_checkpoint_epoch",
                "completed_epochs",
            ]),
            json!(ACCEPTED_TERMINAL_EPOCH),
        ),
        (
            "convergence_reason",
            BTreeSet::from(["convergence_reason", "terminal_reason"]),
            json!(ACCEPTED_CONVERGENCE_REASON),
        ),
        (
            "canonical_ranking_reward_method",
            BTreeSet::from(["canonical_ranking_reward_method", "ranking_method"]),
            json!(ACCEPTED_RANKING_METHOD),
        ),
    ];
    for (identity_key, aliases, expected_value) in expected {
        let mut observed = Vec::new();
        for document in &documents {
            let object = require_object(document, "parsed upstream receipt/envelope")?;
            observed.extend(top_level_values_for_keys(object, &aliases));
        }
        if observed.is_empty() || observed.iter().any(|value| value != &expected_value) {
            return Err(security(format!(
                "independently parsed upstream receipts have missing or conflicting top-level {identity_key}: expected={expected_value} observed={observed:?}"
            )));
        }
    }
    Ok(())
}

fn extract_source_row(binding: &VerifiedSourceBinding, row_index: u64) -> PcResult<Value> {
    match binding.format.as_str() {
        "JSONL" => {
            let text = std::str::from_utf8(&binding.bytes).map_err(|err| {
                security(format!(
                    "candidate source {} is not UTF-8 JSONL: {err}",
                    binding.path.display()
                ))
            })?;
            let line = text
                .lines()
                .filter(|line| !line.trim().is_empty())
                .nth(row_index as usize)
                .ok_or_else(|| {
                    security(format!(
                        "source_row_index {row_index} is out of bounds for {}",
                        binding.path.display()
                    ))
                })?;
            serde_json::from_str(line).map_err(|err| {
                security(format!(
                    "candidate source row {}[{row_index}] is invalid JSON: {err}",
                    binding.path.display()
                ))
            })
        }
        "JSON" => {
            let document: Value = serde_json::from_slice(&binding.bytes).map_err(|err| {
                security(format!(
                    "candidate source {} is invalid JSON: {err}",
                    binding.path.display()
                ))
            })?;
            let rows: Vec<Value> = match document {
                Value::Array(rows) => rows,
                Value::Object(mut object) => {
                    let collection = ["rows", "candidates", "records"]
                        .iter()
                        .find_map(|key| object.remove(*key).and_then(|value| value.as_array().cloned()));
                    collection.unwrap_or_else(|| vec![Value::Object(object)])
                }
                _ => {
                    return Err(security(format!(
                        "candidate source {} has no JSON row collection",
                        binding.path.display()
                    )))
                }
            };
            rows.get(row_index as usize).cloned().ok_or_else(|| {
                security(format!(
                    "source_row_index {row_index} is out of bounds for {} with {} rows",
                    binding.path.display(),
                    rows.len()
                ))
            })
        }
        "CSV" => {
            let mut reader = csv::ReaderBuilder::new()
                .has_headers(true)
                .from_reader(binding.bytes.as_slice());
            let headers = reader
                .headers()
                .map_err(|err| security(format!("candidate CSV headers are invalid: {err}")))?
                .clone();
            let record = reader
                .records()
                .nth(row_index as usize)
                .ok_or_else(|| {
                    security(format!(
                        "source_row_index {row_index} is out of bounds for {}",
                        binding.path.display()
                    ))
                })?
                .map_err(|err| security(format!("candidate CSV row is invalid: {err}")))?;
            let object = headers
                .iter()
                .zip(record.iter())
                .map(|(key, value)| (key.to_string(), Value::String(value.to_string())))
                .collect::<Map<String, Value>>();
            Ok(Value::Object(object))
        }
        "PARQUET" => Err(security(format!(
            "EVIDENCE_NOT_FOUND: independent PrismCode row parser for PARQUET candidate source {} is not implemented; convert the bounded accepted rows to canonical JSONL/JSON/CSV before consumption",
            binding.path.display()
        ))),
        other => Err(security(format!(
            "candidate source {} has unsupported format {other}",
            binding.path.display()
        ))),
    }
}

fn value_as_u64(value: &Value, label: &str) -> PcResult<u64> {
    if let Some(number) = value.as_u64() {
        return Ok(number);
    }
    value
        .as_str()
        .and_then(|text| text.parse::<u64>().ok())
        .ok_or_else(|| security(format!("{label} is not a non-negative integer")))
}

fn verify_source_binding_contract(
    payload: &Map<String, Value>,
    candidates: &[Value],
) -> PcResult<Value> {
    let contract = require_object(
        required(
            payload,
            "source_artifact_binding_contract",
            "candidate payload",
        )?,
        "source_artifact_binding_contract",
    )?;
    if require_str(
        required(contract, "schema_kind", "source binding contract")?,
        "source binding schema",
    )? != CANDIDATE_BINDING_SCHEMA
        || require_u64(
            required(contract, "schema_version", "source binding contract")?,
            "source binding schema_version",
        )? != 1
        || require_str(
            required(contract, "status", "source binding contract")?,
            "source binding status",
        )? != "VERIFIED"
    {
        return Err(security(
            "source artifact binding contract identity/status is not VERIFIED v1",
        ));
    }
    let bindings = require_array(
        required(contract, "bindings", "source binding contract")?,
        "source bindings",
    )?;
    let rows = require_array(
        required(contract, "row_bindings", "source binding contract")?,
        "row bindings",
    )?;
    if require_u64(
        required(contract, "binding_count", "source binding contract")?,
        "binding_count",
    )? != bindings.len() as u64
        || require_u64(
            required(contract, "row_binding_count", "source binding contract")?,
            "row_binding_count",
        )? != rows.len() as u64
        || rows.len() != candidates.len()
    {
        return Err(security(
            "source binding contract counts do not match bound arrays/candidates",
        ));
    }

    let mut bindings_by_id: BTreeMap<String, VerifiedSourceBinding> = BTreeMap::new();
    for (index, binding_value) in bindings.iter().enumerate() {
        let binding = require_object(binding_value, &format!("source binding {index}"))?;
        let id = require_str(
            required(binding, "binding_id", "source binding")?,
            "binding_id",
        )?
        .to_string();
        let class = require_str(
            required(binding, "artifact_class", "source binding")?,
            "artifact_class",
        )?
        .to_string();
        if !matches!(
            class.as_str(),
            "CANDIDATE_SOURCE_ARTIFACT" | "UPSTREAM_RECEIPT" | "UPSTREAM_ENVELOPE"
        ) {
            return Err(security(format!(
                "source binding {id} has unknown artifact_class {class}"
            )));
        }
        if require_str(
            required(binding, "verification_status", "source binding")?,
            "verification_status",
        )? != "SHA256_SIZE_FILE_IDENTITY_VERIFIED"
        {
            return Err(security(format!(
                "source binding {id} lacks verified file identity"
            )));
        }
        let declared_sha = exact_sha256(
            require_str(
                required(binding, "sha256", "source binding")?,
                "source binding sha256",
            )?,
            "source binding SHA256",
        )?;
        let declared_size = require_u64(
            required(binding, "size_bytes", "source binding")?,
            "source binding size_bytes",
        )?;
        let declared_path = PathBuf::from(require_str(
            required(binding, "path", "source binding")?,
            "source binding path",
        )?);
        let format = require_str(
            required(binding, "format", "source binding")?,
            "source binding format",
        )?
        .to_string();
        if !matches!(format.as_str(), "JSON" | "JSONL" | "CSV" | "PARQUET") {
            return Err(security(format!(
                "source binding {id} has unsupported format {format}"
            )));
        }
        let (canonical_path, actual_size) = verify_file_hash(
            &declared_path,
            &declared_sha,
            &format!("source binding {id}"),
        )?;
        if actual_size != declared_size {
            return Err(PcError::Hash(format!(
                "source binding {id} size mismatch: expected={declared_size} observed={actual_size}"
            )));
        }
        let bytes = fs::read(&canonical_path)?;
        if sha256_bytes(&bytes) != declared_sha || bytes.len() as u64 != declared_size {
            return Err(PcError::Hash(format!(
                "source binding {id} changed between file verification and content ingestion"
            )));
        }
        if bindings_by_id
            .insert(
                id.clone(),
                VerifiedSourceBinding {
                    artifact_class: class,
                    sha256: declared_sha,
                    format,
                    path: canonical_path,
                    bytes,
                },
            )
            .is_some()
        {
            return Err(security(format!("duplicate source binding id {id}")));
        }
    }

    let receipt_hashes = require_object(
        required(payload, "receipt_hashes", "candidate payload")?,
        "receipt_hashes",
    )?;
    if receipt_hashes.is_empty() {
        return Err(security("receipt_hashes must not be empty"));
    }
    for (receipt_id, digest_value) in receipt_hashes {
        let digest = exact_sha256(require_str(digest_value, "receipt hash")?, "receipt hash")?;
        let binding = bindings_by_id.get(receipt_id).ok_or_else(|| {
            security(format!(
                "receipt hash {receipt_id} has no bound source artifact"
            ))
        })?;
        if !matches!(
            binding.artifact_class.as_str(),
            "UPSTREAM_RECEIPT" | "UPSTREAM_ENVELOPE"
        ) || binding.sha256 != digest
        {
            return Err(security(format!(
                "receipt hash {receipt_id} does not match its opened receipt/envelope binding"
            )));
        }
    }

    let content_verification = require_object(
        required(
            contract,
            "receipt_content_verification",
            "source binding contract",
        )?,
        "receipt_content_verification",
    )?;
    if require_str(
        required(
            content_verification,
            "status",
            "receipt content verification",
        )?,
        "receipt content status",
    )? != "VERIFIED"
    {
        return Err(security(
            "upstream receipt content verification is not VERIFIED",
        ));
    }
    verify_upstream_receipt_contents(&bindings_by_id, content_verification)?;
    let corroboration = require_object(
        required(
            content_verification,
            "accepted_identity_corroboration",
            "receipt content verification",
        )?,
        "accepted_identity_corroboration",
    )?;
    for key in [
        "run_id",
        "method",
        "policy_sha256",
        "checkpoint_sha256",
        "terminal_epoch",
        "convergence_reason",
        "canonical_ranking_reward_method",
    ] {
        if corroboration.get(key).and_then(Value::as_bool) != Some(true) {
            return Err(security(format!(
                "upstream receipts do not corroborate accepted identity field {key}"
            )));
        }
    }

    let mut row_keys: BTreeMap<(String, u64, String, u64, String), (String, String)> =
        BTreeMap::new();
    for row_value in rows {
        let row = require_object(row_value, "row binding")?;
        if row
            .get("identity_smiles_order_scores_verified")
            .and_then(Value::as_bool)
            != Some(true)
        {
            return Err(security(
                "row binding lacks identity/SMILES/order/score verification",
            ));
        }
        let key = (
            require_str(
                required(row, "candidate_id", "row binding")?,
                "row candidate_id",
            )?
            .to_string(),
            require_u64(
                required(row, "execution_order", "row binding")?,
                "row execution_order",
            )?,
            require_str(
                required(row, "source_artifact_binding_id", "row binding")?,
                "row source binding",
            )?
            .to_string(),
            require_u64(
                required(row, "source_row_index", "row binding")?,
                "row source index",
            )?,
            exact_sha256(
                require_str(
                    required(row, "source_row_sha256", "row binding")?,
                    "row source SHA256",
                )?,
                "row source SHA256",
            )?,
        );
        let identity_field = require_str(
            required(row, "source_identity_field", "row binding")?,
            "row source identity field",
        )?
        .to_string();
        let smiles_field = require_str(
            required(row, "source_smiles_field", "row binding")?,
            "row source SMILES field",
        )?
        .to_string();
        if row_keys
            .insert(key, (identity_field, smiles_field))
            .is_some()
        {
            return Err(security("duplicate row binding identity"));
        }
    }

    for candidate_value in candidates {
        let candidate = require_object(candidate_value, "candidate")?;
        let candidate_id = require_str(
            required(candidate, "candidate_id", "candidate")?,
            "candidate_id",
        )?;
        let order = require_u64(
            required(candidate, "execution_order", "candidate")?,
            "execution_order",
        )?;
        let candidate_smiles = require_str(
            required(candidate, "canonical_smiles", "candidate")?,
            "canonical_smiles",
        )?;
        let candidate_scores = require_object(
            required(candidate, "score_components", "candidate")?,
            "score_components",
        )?;
        let lineage = require_object(
            required(candidate, "source_row_lineage", "candidate")?,
            "source_row_lineage",
        )?;
        let binding_id = require_str(
            required(lineage, "source_artifact_binding_id", "lineage")?,
            "lineage binding",
        )?;
        let source_index = require_u64(
            required(lineage, "source_row_index", "lineage")?,
            "lineage source index",
        )?;
        let source_file_sha = exact_sha256(
            require_str(
                required(lineage, "source_artifact_sha256", "lineage")?,
                "lineage source file SHA256",
            )?,
            "lineage source file SHA256",
        )?;
        let source_row_sha = exact_sha256(
            require_str(
                required(lineage, "source_row_sha256", "lineage")?,
                "lineage source row SHA256",
            )?,
            "lineage source row SHA256",
        )?;
        let source_binding = bindings_by_id.get(binding_id).ok_or_else(|| {
            security(format!(
                "candidate {candidate_id} references unknown source binding {binding_id}"
            ))
        })?;
        if source_binding.artifact_class != "CANDIDATE_SOURCE_ARTIFACT"
            || source_binding.sha256 != source_file_sha
        {
            return Err(security(format!("candidate {candidate_id} source lineage does not match opened candidate source artifact")));
        }
        let row_key = (
            candidate_id.to_string(),
            order,
            binding_id.to_string(),
            source_index,
            source_row_sha.clone(),
        );
        let (asserted_identity_field, asserted_smiles_field) =
            if let Some(fields) = row_keys.get(&row_key) {
                fields
            } else {
                return Err(security(format!(
                    "candidate {candidate_id} has no exact row-binding verification"
                )));
            };

        let source_row = extract_source_row(source_binding, source_index)?;
        let source_row_object =
            require_object(&source_row, "independently parsed candidate source row")?;
        let observed_source_row_sha = sha256_bytes(&canonical_json_bytes(&source_row)?);
        if observed_source_row_sha != source_row_sha {
            return Err(PcError::Hash(format!(
                "candidate {candidate_id} source_row_sha256 mismatch: declared={source_row_sha} observed={observed_source_row_sha}"
            )));
        }
        let actual_identity_field = ["candidate_id", "trajectory_id", "id"]
            .iter()
            .find(|key| {
                source_row_object
                    .get(**key)
                    .is_some_and(|value| !value.is_null())
            })
            .copied()
            .ok_or_else(|| {
                security(format!(
                    "candidate {candidate_id} source row has no identity field"
                ))
            })?;
        let actual_identity = source_row_object
            .get(actual_identity_field)
            .and_then(|value| {
                value
                    .as_str()
                    .map(str::to_string)
                    .or_else(|| Some(value.to_string()))
            })
            .ok_or_else(|| {
                security(format!(
                    "candidate {candidate_id} source identity is unreadable"
                ))
            })?;
        if actual_identity != candidate_id || asserted_identity_field != actual_identity_field {
            return Err(security(format!(
                "candidate {candidate_id} identity does not match independently parsed source row: field={actual_identity_field} value={actual_identity}"
            )));
        }
        let actual_smiles_field = ["canonical_smiles", "smiles", "SMILES"]
            .iter()
            .find(|key| {
                source_row_object
                    .get(**key)
                    .is_some_and(|value| !value.is_null())
            })
            .copied()
            .ok_or_else(|| {
                security(format!(
                    "candidate {candidate_id} source row has no SMILES field"
                ))
            })?;
        let actual_smiles = require_str(
            source_row_object
                .get(actual_smiles_field)
                .expect("field selected from object"),
            "source row SMILES",
        )?
        .trim();
        if actual_smiles != candidate_smiles || asserted_smiles_field != actual_smiles_field {
            return Err(security(format!(
                "candidate {candidate_id} canonical_smiles does not match independently parsed source row"
            )));
        }
        let actual_order = value_as_u64(
            source_row_object.get("execution_order").ok_or_else(|| {
                security(format!(
                    "candidate {candidate_id} source row lacks execution_order"
                ))
            })?,
            "source row execution_order",
        )?;
        if actual_order != order {
            return Err(security(format!(
                "candidate {candidate_id} execution_order does not match independently parsed source row"
            )));
        }
        let source_scores_value = source_row_object.get("score_components").ok_or_else(|| {
            security(format!(
                "candidate {candidate_id} source row lacks score_components"
            ))
        })?;
        let parsed_source_scores = if let Some(text) = source_scores_value.as_str() {
            serde_json::from_str::<Value>(text).map_err(|err| {
                security(format!(
                    "candidate {candidate_id} source score_components JSON is invalid: {err}"
                ))
            })?
        } else {
            source_scores_value.clone()
        };
        let parsed_source_scores = require_object(
            &parsed_source_scores,
            "independently parsed source score_components",
        )?;
        if parsed_source_scores != candidate_scores {
            return Err(security(format!(
                "candidate {candidate_id} score_components do not match independently parsed source row"
            )));
        }

        let declared_row_sha = exact_sha256(
            require_str(
                required(candidate, "canonical_row_sha256", "candidate")?,
                "candidate canonical_row_sha256",
            )?,
            "candidate canonical row SHA256",
        )?;
        let mut canonical_row = Value::Object(candidate.clone());
        canonical_row
            .as_object_mut()
            .expect("constructed object")
            .remove("canonical_row_sha256");
        let observed_row_sha = sha256_bytes(&canonical_json_bytes(&canonical_row)?);
        if declared_row_sha != observed_row_sha {
            return Err(PcError::Hash(format!(
                "candidate {candidate_id} canonical row SHA256 mismatch: expected={declared_row_sha} observed={observed_row_sha}"
            )));
        }
    }
    Ok(Value::Object(contract.clone()))
}

fn verify_payload_semantics(
    payload_value: &Value,
    payload_bytes: &[u8],
) -> PcResult<(Value, Value)> {
    let payload = require_object(payload_value, "candidate evidence payload")?;
    if require_str(
        required(payload, "schema_kind", "candidate payload")?,
        "candidate schema_kind",
    )? != CANDIDATE_PAYLOAD_SCHEMA
        || require_u64(
            required(payload, "schema_version", "candidate payload")?,
            "candidate schema_version",
        )? != 1
    {
        return Err(security(
            "candidate payload schema identity is not accepted v1",
        ));
    }
    if canonical_json_bytes(payload_value)? != payload_bytes {
        return Err(security("candidate payload bytes are not canonical JSON"));
    }
    if require_str(
        required(
            payload,
            "required_computational_boundary",
            "candidate payload",
        )?,
        "required computational boundary",
    )? != REQUIRED_COMPUTATIONAL_BOUNDARY
    {
        return Err(security(
            "candidate payload computational claim boundary is missing or altered",
        ));
    }
    let policy = verify_exact_policy(payload)?;
    let candidates = require_array(
        required(payload, "candidates", "candidate payload")?,
        "candidates",
    )?;
    let candidate_count = require_u64(
        required(payload, "candidate_count", "candidate payload")?,
        "candidate_count",
    )?;
    if candidates.is_empty()
        || candidates.len() > 4096
        || candidate_count != candidates.len() as u64
    {
        return Err(security(
            "candidate_count is empty, exceeds 4096, or differs from candidates length",
        ));
    }
    let mut ids = BTreeSet::new();
    let mut orders = BTreeSet::new();
    for candidate_value in candidates {
        let candidate = require_object(candidate_value, "candidate")?;
        ids.insert(
            require_str(
                required(candidate, "candidate_id", "candidate")?,
                "candidate_id",
            )?
            .to_string(),
        );
        orders.insert(require_u64(
            required(candidate, "execution_order", "candidate")?,
            "execution_order",
        )?);
    }
    if ids.len() != candidates.len() || orders != (0..candidate_count).collect::<BTreeSet<_>>() {
        return Err(security("candidate identities must be unique and execution_order must cover contiguous 0..count-1"));
    }
    let contract = verify_source_binding_contract(payload, candidates)?;

    let mut core = Map::new();
    for key in [
        "policy_identity",
        "candidates",
        "claim_scope",
        "receipt_hashes",
        "source_artifact_binding_contract",
    ] {
        core.insert(
            key.to_string(),
            required(payload, key, "candidate payload")?.clone(),
        );
    }
    if let Some(metadata) = payload.get("batch_metadata") {
        core.insert("batch_metadata".to_string(), metadata.clone());
    }
    let evidence_root = sha256_bytes(&canonical_json_bytes(&Value::Object(core))?);
    let declared_root = exact_sha256(
        require_str(
            required(payload, "evidence_root_sha256", "candidate payload")?,
            "evidence_root_sha256",
        )?,
        "candidate evidence root SHA256",
    )?;
    if declared_root != evidence_root {
        return Err(PcError::Hash(format!(
            "candidate evidence root mismatch: expected={declared_root} observed={evidence_root}"
        )));
    }
    let expected_batch = format!("gfcand_{}", &evidence_root[..32]);
    if require_str(
        required(payload, "batch_id", "candidate payload")?,
        "batch_id",
    )? != expected_batch
    {
        return Err(PcError::Hash(format!(
            "candidate batch_id is not derived from evidence root {evidence_root}"
        )));
    }
    Ok((policy, contract))
}

fn verify_candidate_evidence(operands: &CandidateEvidenceOperands) -> PcResult<Value> {
    let (payload_path, payload_size) = verify_file_hash(
        &operands.payload,
        &operands.payload_sha256,
        "candidate evidence payload",
    )?;
    let (packet_path, packet_size) = verify_file_hash(
        &operands.packet,
        &operands.packet_sha256,
        "candidate QNANL packet",
    )?;
    let (receipt_path, receipt_size) = verify_file_hash(
        &operands.qnanl_receipt,
        &operands.qnanl_receipt_sha256,
        "candidate QNANL receipt",
    )?;

    let payload_bytes = fs::read(&payload_path)?;
    let payload_value: Value = serde_json::from_slice(&payload_bytes)?;
    let (policy, source_contract) = verify_payload_semantics(&payload_value, &payload_bytes)?;

    let packet_bytes = fs::read(&packet_path)?;
    let decoded = decode_packet_bytes(&packet_bytes)
        .map_err(|err| PcError::Hash(format!("candidate QNANL packet decode failed: {err}")))?;
    let decoded_value: Value = serde_json::from_slice(&decoded)?;
    if canonical_json_bytes(&decoded_value)? != decoded {
        return Err(security(
            "decoded candidate QNANL document is not canonical JSON",
        ));
    }
    let decoded_object = require_object(&decoded_value, "decoded QNANL decision document")?;
    if require_str(
        required(decoded_object, "schema_kind", "decoded decision")?,
        "decoded schema_kind",
    )? != "prismos_gflownet_branch_recommendation"
        || require_str(
            required(decoded_object, "adapter_logical_family", "decoded decision")?,
            "decoded logical family",
        )? != "GFLOWNET_BRANCH_RECOMMENDATION"
        || decoded_object.get("accepted").and_then(Value::as_bool) != Some(true)
        || require_str(
            required(decoded_object, "lifecycle_stage", "decoded decision")?,
            "decoded lifecycle_stage",
        )? != "candidate_evidence_packet"
    {
        return Err(security("decoded QNANL decision is not the accepted GFlowNet candidate evidence family/lifecycle"));
    }
    let decision_body = require_object(
        required(decoded_object, "decision_body", "decoded decision")?,
        "decision_body",
    )?;
    if exact_sha256(
        require_str(
            required(decision_body, "candidate_payload_sha256", "decision_body")?,
            "decision body candidate payload SHA256",
        )?,
        "decision body candidate payload SHA256",
    )? != operands.payload_sha256
        || required(decision_body, "candidate_evidence_payload", "decision_body")? != &payload_value
    {
        return Err(PcError::Hash(
            "decoded QNANL decision does not contain the exact candidate payload/hash".to_string(),
        ));
    }
    let state_hash = format!(
        "sha256:{}",
        sha256_bytes(&canonical_json_bytes(&Value::Object(
            decision_body.clone()
        ))?)
    );
    if require_str(
        required(decoded_object, "state_hash", "decoded decision")?,
        "decoded state_hash",
    )? != state_hash
    {
        return Err(PcError::Hash(
            "decoded QNANL decision state_hash does not bind its decision_body".to_string(),
        ));
    }

    let receipt_bytes = fs::read(&receipt_path)?;
    let qnanl_receipt: Value = serde_json::from_slice(&receipt_bytes)?;
    let receipt = require_object(&qnanl_receipt, "candidate QNANL receipt")?;
    if require_str(
        required(receipt, "schema_kind", "QNANL receipt")?,
        "QNANL receipt schema_kind",
    )? != QNANL_ARTIFACT_RECEIPT_KIND
        || require_u64(
            required(receipt, "schema_version", "QNANL receipt")?,
            "QNANL receipt schema_version",
        )? != QNANL_ARTIFACT_SCHEMA_VERSION as u64
        || require_str(
            required(receipt, "status", "QNANL receipt")?,
            "QNANL receipt status",
        )? != "PASS"
        || require_str(
            required(receipt, "artifact_family", "QNANL receipt")?,
            "QNANL artifact family",
        )? != "CHRONOLOGY_ACCEPTANCE_DECISION"
        || require_str(
            required(receipt, "qnanl_seal_status", "QNANL receipt")?,
            "QNANL seal status",
        )? != "PASS"
        || receipt.get("roundtrip_verified").and_then(Value::as_bool) != Some(true)
        || receipt
            .get("codec_roundtrip_verified")
            .and_then(Value::as_bool)
            != Some(true)
        || receipt
            .get("qnanl_seal_constraints_pass")
            .and_then(Value::as_bool)
            != Some(true)
        || receipt
            .get("qnanl_seal_merkle_verified")
            .and_then(Value::as_bool)
            != Some(true)
        || receipt
            .get("qnanl_seal_ecc_verified")
            .and_then(Value::as_bool)
            != Some(true)
        || require_str(
            required(receipt, "runtime_origin", "QNANL receipt")?,
            "QNANL runtime_origin",
        )? != "POST_RUN_SEAL_ONLY"
        || receipt.get("runtime_native").and_then(Value::as_bool) != Some(false)
    {
        return Err(security(
            "candidate QNANL receipt identity, codec, seal, or provenance gate failed",
        ));
    }
    if require_u64(
        required(receipt, "raw_bytes", "QNANL receipt")?,
        "QNANL raw_bytes",
    )? != decoded.len() as u64
        || require_u64(
            required(receipt, "packet_bytes", "QNANL receipt")?,
            "QNANL packet_bytes",
        )? != packet_size
        || require_str(
            required(receipt, "raw_hash_blake3", "QNANL receipt")?,
            "QNANL raw hash",
        )? != blake3::hash(&decoded).to_hex().as_str()
        || require_str(
            required(receipt, "packet_hash_blake3", "QNANL receipt")?,
            "QNANL packet hash",
        )? != blake3::hash(&packet_bytes).to_hex().as_str()
    {
        return Err(PcError::Hash(
            "candidate QNANL receipt content hashes/sizes do not match opened packet".to_string(),
        ));
    }
    let receipt_packet_path = canonical_regular_file(
        Path::new(require_str(
            required(receipt, "packet_path", "QNANL receipt")?,
            "QNANL receipt packet_path",
        )?),
        "QNANL receipt packet_path",
    )?;
    let receipt_self_path = canonical_regular_file(
        Path::new(require_str(
            required(receipt, "receipt_path", "QNANL receipt")?,
            "QNANL receipt receipt_path",
        )?),
        "QNANL receipt receipt_path",
    )?;
    let canonical_source_path = canonical_regular_file(
        Path::new(require_str(
            required(receipt, "source_path", "QNANL receipt")?,
            "QNANL receipt source_path",
        )?),
        "QNANL canonical source",
    )?;
    if receipt_packet_path != packet_path
        || receipt_self_path != receipt_path
        || fs::read(&canonical_source_path)? != decoded
    {
        return Err(security("QNANL receipt paths or canonical source bytes do not match opened operands/decoded packet"));
    }

    let payload = require_object(&payload_value, "candidate payload")?;
    Ok(json!({
        "schema_kind": CONSUMPTION_RECEIPT_SCHEMA,
        "schema_version": 1,
        "status": "PASS",
        "runtime_origin": "EXTERNAL_DERIVED",
        "candidate_batch_id": required(payload, "batch_id", "candidate payload")?,
        "candidate_count": required(payload, "candidate_count", "candidate payload")?,
        "evidence_root_sha256": required(payload, "evidence_root_sha256", "candidate payload")?,
        "policy_identity": policy,
        "claim_scope": required(payload, "claim_scope", "candidate payload")?,
        "required_computational_boundary": REQUIRED_COMPUTATIONAL_BOUNDARY,
        "consumed_hash_operands": {
            "candidate_evidence_payload_sha256": operands.payload_sha256,
            "candidate_qnanl_packet_sha256": operands.packet_sha256,
            "candidate_qnanl_receipt_sha256": operands.qnanl_receipt_sha256,
        },
        "consumed_artifacts": {
            "candidate_evidence_payload": {"path": payload_path, "size_bytes": payload_size},
            "candidate_qnanl_packet": {"path": packet_path, "size_bytes": packet_size},
            "candidate_qnanl_receipt": {"path": receipt_path, "size_bytes": receipt_size},
            "candidate_qnanl_canonical_source": {"path": canonical_source_path, "size_bytes": decoded.len()},
        },
        "source_artifact_binding_contract": source_contract,
        "qnanl_binding": {
            "artifact_id": required(receipt, "artifact_id", "QNANL receipt")?,
            "artifact_family": "CHRONOLOGY_ACCEPTANCE_DECISION",
            "logical_family": "GFLOWNET_BRANCH_RECOMMENDATION",
            "decoded_canonical_sha256": sha256_bytes(&decoded),
            "decoded_raw_hash_blake3": required(receipt, "raw_hash_blake3", "QNANL receipt")?,
            "packet_hash_blake3": required(receipt, "packet_hash_blake3", "QNANL receipt")?,
        },
        "checks": {
            "exact_hash_operands_consumed": true,
            "exact_accepted_v5_policy_identity": true,
            "candidate_rows_and_sources_bound": true,
            "all_declared_source_artifacts_reopened_and_hashed": true,
            "upstream_receipt_identity_corroborated": true,
            "candidate_payload_canonical_and_self_hashes_valid": true,
            "qnanl_packet_decoded_and_seal_verified": true,
            "qnanl_receipt_hashes_paths_and_provenance_verified": true,
            "decoded_packet_contains_exact_payload": true,
        },
    }))
}

fn bind_execution_to_candidate_evidence(
    mut binding: Value,
    input: &Path,
    output: &Path,
    execution: &ExecutionReport,
) -> PcResult<Value> {
    let input_path = canonical_regular_file(input, "PrismCode input packet")?;
    let output_path = canonical_regular_file(output, "PrismCode output PCOUT")?;
    let object = binding
        .as_object_mut()
        .ok_or_else(|| PcError::Internal("candidate binding must be an object".to_string()))?;
    object.insert(
        "prismcode_execution".to_string(),
        json!({
            "status": execution.status,
            "backend": execution.backend,
            "deterministic": execution.deterministic,
            "input_pcode": {
                "path": input_path,
                "sha256": sha256_file_hex(input).map_err(|err| PcError::Io(err.to_string()))?,
                "packet_hash": execution.input_packet_hash,
            },
            "output_pcout": {
                "path": output_path,
                "sha256": sha256_file_hex(output).map_err(|err| PcError::Io(err.to_string()))?,
                "output_state_hash": execution.output_state_hash,
                "seal_chain_hash": execution.seal_chain_hash,
            },
            "instruction_count": execution.instruction_count,
            "state_count": execution.state_count,
            "event_count": execution.event_count,
            "checkpoint_count": execution.checkpoint_count,
        }),
    );
    Ok(binding)
}

fn write_json_no_clobber(path: &Path, value: &Value) -> PcResult<()> {
    if !path.is_absolute() {
        return Err(security(format!(
            "candidate consumption receipt path must be absolute: {}",
            path.display()
        )));
    }
    let parent = path
        .parent()
        .ok_or_else(|| security("candidate consumption receipt has no parent"))?;
    fs::create_dir_all(parent)?;
    let parent = fs::canonicalize(parent)?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| security("candidate consumption receipt has invalid filename"))?;
    let temp_path = parent.join(format!(".{file_name}.tmp.{}", std::process::id()));
    let bytes = canonical_json_bytes(value)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp_path)
        .map_err(|err| {
            PcError::Io(format!(
                "cannot create receipt staging file {}: {err}",
                temp_path.display()
            ))
        })?;
    if let Err(err) = file.write_all(&bytes).and_then(|_| file.sync_all()) {
        let _ = fs::remove_file(&temp_path);
        return Err(PcError::Io(format!(
            "cannot stage candidate consumption receipt: {err}"
        )));
    }
    drop(file);
    if let Err(err) = fs::hard_link(&temp_path, path) {
        let _ = fs::remove_file(&temp_path);
        return Err(PcError::Io(format!(
            "cannot commit no-clobber candidate consumption receipt {}: {err}",
            path.display()
        )));
    }
    fs::remove_file(&temp_path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use prismcode::qnanl_artifact::{encode_bytes, QnanlArtifactFamily, QnanlEncodeOptions};
    use tempfile::tempdir;

    fn write_json(path: &Path, value: &Value) {
        fs::write(path, canonical_json_bytes(value).expect("canonical JSON")).expect("write JSON");
    }

    fn fixture(root: &Path) -> (CandidateEvidenceOperands, Value) {
        let run_id = ACCEPTED_RUN_ID;
        let source_row = json!({
            "candidate_id": "candidate_a",
            "canonical_smiles": "CCO",
            "execution_order": 0,
            "score_components": {"effective_reward": 13.0, "nested": {"active": [0.4, 0.8]}},
        });
        let source_path = root.join("accepted_candidate_rows.jsonl");
        let mut source_bytes = serde_json::to_vec(&source_row).expect("source row JSONL");
        source_bytes.push(b'\n');
        fs::write(&source_path, source_bytes).expect("write source");
        let training_path = root.join("accepted_training_receipt.json");
        let training = json!({
            "schema": "prism.gflownet_v5.train_convergence_acceptance.v1",
            "status": "PASS",
            "run_id": run_id,
            "method": ACCEPTED_METHOD,
            "policy_sha256": ACCEPTED_POLICY_SHA256,
            "checkpoint_sha256": ACCEPTED_CHECKPOINT_SHA256,
            "terminal_epoch": ACCEPTED_TERMINAL_EPOCH,
            "convergence_reason": ACCEPTED_CONVERGENCE_REASON,
            "canonical_ranking_reward_method": ACCEPTED_RANKING_METHOD,
        });
        write_json(&training_path, &training);
        let source_sha = sha256_file_hex(&source_path).expect("source sha");
        let training_sha = sha256_file_hex(&training_path).expect("training sha");

        let mut candidate = json!({
            "candidate_id": "candidate_a",
            "canonical_smiles": "CCO",
            "execution_order": 0,
            "source_row_lineage": {
                "source_artifact_binding_id": "accepted_candidate_rows",
                "source_row_index": 0,
                "source_artifact_sha256": source_sha,
                "source_row_sha256": sha256_bytes(&canonical_json_bytes(&source_row).expect("row bytes")),
            },
            "score_components": {"effective_reward": 13.0, "nested": {"active": [0.4, 0.8]}},
        });
        let candidate_row_sha =
            sha256_bytes(&canonical_json_bytes(&candidate).expect("candidate bytes"));
        candidate.as_object_mut().expect("candidate object").insert(
            "canonical_row_sha256".to_string(),
            Value::String(candidate_row_sha),
        );
        let policy = json!({
            "run_id": run_id,
            "method": ACCEPTED_METHOD,
            "policy_sha256": ACCEPTED_POLICY_SHA256,
            "checkpoint_sha256": ACCEPTED_CHECKPOINT_SHA256,
            "terminal_epoch": ACCEPTED_TERMINAL_EPOCH,
            "convergence_reason": ACCEPTED_CONVERGENCE_REASON,
            "canonical_ranking_reward_method": ACCEPTED_RANKING_METHOD,
        });
        let contract = json!({
            "schema_kind": CANDIDATE_BINDING_SCHEMA,
            "schema_version": 1,
            "status": "VERIFIED",
            "binding_count": 2,
            "bindings": [
                {
                    "binding_id": "accepted_candidate_rows",
                    "artifact_class": "CANDIDATE_SOURCE_ARTIFACT",
                    "path": source_path,
                    "format": "JSONL",
                    "sha256": source_sha,
                    "size_bytes": fs::metadata(&source_path).expect("source metadata").len(),
                    "verification_status": "SHA256_SIZE_FILE_IDENTITY_VERIFIED",
                },
                {
                    "binding_id": "accepted_training_receipt",
                    "artifact_class": "UPSTREAM_RECEIPT",
                    "path": training_path,
                    "format": "JSON",
                    "sha256": training_sha,
                    "size_bytes": fs::metadata(&training_path).expect("receipt metadata").len(),
                    "verification_status": "SHA256_SIZE_FILE_IDENTITY_VERIFIED",
                }
            ],
            "row_binding_count": 1,
            "row_bindings": [{
                "candidate_id": "candidate_a",
                "execution_order": 0,
                "source_artifact_binding_id": "accepted_candidate_rows",
                "source_row_index": 0,
                "source_row_sha256": candidate["source_row_lineage"]["source_row_sha256"],
                "source_identity_field": "candidate_id",
                "source_smiles_field": "canonical_smiles",
                "identity_smiles_order_scores_verified": true,
            }],
            "receipt_content_verification": {
                "status": "VERIFIED",
                "receipt_count": 1,
                "receipts": [{"binding_id": "accepted_training_receipt", "run_id_corroborated": true}],
                "accepted_identity_corroboration": {
                    "run_id": true,
                    "method": true,
                    "policy_sha256": true,
                    "checkpoint_sha256": true,
                    "terminal_epoch": true,
                    "convergence_reason": true,
                    "canonical_ranking_reward_method": true,
                },
            },
        });
        let core = json!({
            "policy_identity": policy,
            "candidates": [candidate],
            "claim_scope": {
                "evidence_class": "COMPUTATIONAL_PRISM_DERIVED",
                "claim_promotion_allowed_effective": false,
                "package_route": "CDA_GATED_OPERATOR_REVIEW",
            },
            "receipt_hashes": {"accepted_training_receipt": training_sha},
            "source_artifact_binding_contract": contract,
        });
        let evidence_root = sha256_bytes(&canonical_json_bytes(&core).expect("core bytes"));
        let batch_id = format!("gfcand_{}", &evidence_root[..32]);
        let mut payload_object = core.as_object().expect("core object").clone();
        payload_object.insert(
            "schema_kind".to_string(),
            Value::String(CANDIDATE_PAYLOAD_SCHEMA.to_string()),
        );
        payload_object.insert("schema_version".to_string(), json!(1));
        payload_object.insert("batch_id".to_string(), Value::String(batch_id.clone()));
        payload_object.insert("candidate_count".to_string(), json!(1));
        payload_object.insert(
            "evidence_root_sha256".to_string(),
            Value::String(evidence_root),
        );
        payload_object.insert(
            "required_computational_boundary".to_string(),
            Value::String(REQUIRED_COMPUTATIONAL_BOUNDARY.to_string()),
        );
        let payload = Value::Object(payload_object);
        let payload_path = root.join("candidate_payload.json");
        write_json(&payload_path, &payload);
        let payload_sha = sha256_file_hex(&payload_path).expect("payload sha");

        let body = json!({
            "decision": batch_id,
            "decision_basis": format!("candidate_payload_sha256={payload_sha}"),
            "decision_id": batch_id,
            "accepted": true,
            "lifecycle_stage": "candidate_evidence_packet",
            "candidate_payload_sha256": payload_sha,
            "candidate_evidence_payload": payload,
        });
        let state_hash = format!(
            "sha256:{}",
            sha256_bytes(&canonical_json_bytes(&body).expect("body bytes"))
        );
        let decoded = json!({
            "schema_kind": "prismos_gflownet_branch_recommendation",
            "schema_version": 1,
            "adapter": "prismos_route_decision_v1",
            "adapter_version": 1,
            "adapter_logical_family": "GFLOWNET_BRANCH_RECOMMENDATION",
            "adapter_carrier_family": "CHRONOLOGY_ACCEPTANCE_DECISION",
            "adapter_carrier_family_code": 29,
            "adapter_carrier_transform": "CHRONOLOGY_ACCEPTANCE_DECISION_CANONICAL_BYTES",
            "adapter_rationale": "fixture",
            "decision": batch_id,
            "decision_basis": body["decision_basis"],
            "accepted": true,
            "lifecycle_stage": "candidate_evidence_packet",
            "acceptance_decision_id": batch_id,
            "route_decision_id": batch_id,
            "state_hash": state_hash,
            "runtime_origin": "POST_RUN_SEAL_ONLY",
            "runtime_native": false,
            "decision_body": body,
        });
        let decoded_bytes = canonical_json_bytes(&decoded).expect("decoded bytes");
        let canonical_source = root.join("candidate.qnanl.canonical.json");
        fs::write(&canonical_source, &decoded_bytes).expect("canonical source");
        let packet_path = root.join("candidate.qnanl");
        let qnanl_receipt = root.join("candidate.qnanl.receipt.json");
        encode_bytes(
            &decoded_bytes,
            &packet_path,
            &qnanl_receipt,
            QnanlEncodeOptions::post_run_seal_only(
                batch_id,
                QnanlArtifactFamily::ChronologyAcceptanceDecision,
                Some(canonical_source),
                "prismrun_test_fixture",
            ),
        )
        .expect("encode QNANL fixture");

        (
            CandidateEvidenceOperands {
                payload: payload_path.clone(),
                packet: packet_path.clone(),
                qnanl_receipt: qnanl_receipt.clone(),
                payload_sha256: sha256_file_hex(&payload_path).expect("payload sha"),
                packet_sha256: sha256_file_hex(&packet_path).expect("packet sha"),
                qnanl_receipt_sha256: sha256_file_hex(&qnanl_receipt).expect("receipt sha"),
                consumption_receipt: root.join("candidate.consumption.json"),
            },
            payload,
        )
    }

    fn refresh_payload_integrity(payload: &mut Value) {
        for candidate in payload["candidates"]
            .as_array_mut()
            .expect("candidate array")
        {
            candidate
                .as_object_mut()
                .expect("candidate object")
                .remove("canonical_row_sha256");
            let row_sha = sha256_bytes(&canonical_json_bytes(candidate).expect("candidate row"));
            candidate
                .as_object_mut()
                .expect("candidate object")
                .insert("canonical_row_sha256".to_string(), Value::String(row_sha));
        }
        let object = payload.as_object().expect("payload object");
        let core = json!({
            "policy_identity": object["policy_identity"],
            "candidates": object["candidates"],
            "claim_scope": object["claim_scope"],
            "receipt_hashes": object["receipt_hashes"],
            "source_artifact_binding_contract": object["source_artifact_binding_contract"],
        });
        let evidence_root = sha256_bytes(&canonical_json_bytes(&core).expect("evidence core"));
        payload["evidence_root_sha256"] = Value::String(evidence_root.clone());
        payload["batch_id"] = Value::String(format!("gfcand_{}", &evidence_root[..32]));
    }

    fn reseal_payload(operands: &mut CandidateEvidenceOperands, payload: &Value) {
        write_json(&operands.payload, payload);
        operands.payload_sha256 = sha256_file_hex(&operands.payload).expect("payload sha");
        let batch_id = payload["batch_id"].as_str().expect("batch id");
        let body = json!({
            "decision": batch_id,
            "decision_basis": format!("candidate_payload_sha256={}", operands.payload_sha256),
            "decision_id": batch_id,
            "accepted": true,
            "lifecycle_stage": "candidate_evidence_packet",
            "candidate_payload_sha256": operands.payload_sha256,
            "candidate_evidence_payload": payload,
        });
        let state_hash = format!(
            "sha256:{}",
            sha256_bytes(&canonical_json_bytes(&body).expect("body bytes"))
        );
        let decoded = json!({
            "schema_kind": "prismos_gflownet_branch_recommendation",
            "schema_version": 1,
            "adapter": "prismos_route_decision_v1",
            "adapter_version": 1,
            "adapter_logical_family": "GFLOWNET_BRANCH_RECOMMENDATION",
            "adapter_carrier_family": "CHRONOLOGY_ACCEPTANCE_DECISION",
            "adapter_carrier_family_code": 29,
            "adapter_carrier_transform": "CHRONOLOGY_ACCEPTANCE_DECISION_CANONICAL_BYTES",
            "adapter_rationale": "malicious self-consistent sealed fixture",
            "decision": batch_id,
            "decision_basis": body["decision_basis"],
            "accepted": true,
            "lifecycle_stage": "candidate_evidence_packet",
            "acceptance_decision_id": batch_id,
            "route_decision_id": batch_id,
            "state_hash": state_hash,
            "runtime_origin": "POST_RUN_SEAL_ONLY",
            "runtime_native": false,
            "decision_body": body,
        });
        let decoded_bytes = canonical_json_bytes(&decoded).expect("decoded bytes");
        let canonical_source = operands.packet.with_extension("qnanl.canonical.json");
        fs::write(&canonical_source, &decoded_bytes).expect("canonical source");
        encode_bytes(
            &decoded_bytes,
            &operands.packet,
            &operands.qnanl_receipt,
            QnanlEncodeOptions::post_run_seal_only(
                batch_id,
                QnanlArtifactFamily::ChronologyAcceptanceDecision,
                Some(canonical_source),
                "prismrun_malicious_fixture",
            ),
        )
        .expect("reseal QNANL fixture");
        operands.packet_sha256 = sha256_file_hex(&operands.packet).expect("packet sha");
        operands.qnanl_receipt_sha256 =
            sha256_file_hex(&operands.qnanl_receipt).expect("receipt sha");
    }

    #[test]
    fn exact_candidate_packet_hashes_rows_sources_policy_and_qnanl_are_consumed() {
        let temp = tempdir().expect("tempdir");
        let (operands, _) = fixture(temp.path());
        let receipt = verify_candidate_evidence(&operands).expect("verified candidate evidence");
        assert_eq!(receipt["status"], "PASS");
        assert_eq!(receipt["policy_identity"]["terminal_epoch"], 459);
        assert_eq!(receipt["candidate_count"], 1);
        assert!(receipt["checks"]
            .as_object()
            .expect("checks")
            .values()
            .all(|value| value == &Value::Bool(true)));
    }

    #[test]
    fn altered_policy_is_rejected_even_when_payload_hash_operand_is_updated() {
        let temp = tempdir().expect("tempdir");
        let (mut operands, mut payload) = fixture(temp.path());
        payload["policy_identity"]["terminal_epoch"] = json!(500);
        write_json(&operands.payload, &payload);
        operands.payload_sha256 = sha256_file_hex(&operands.payload).expect("tampered payload sha");
        let error = verify_candidate_evidence(&operands).expect_err("altered policy must fail");
        assert!(error.to_string().contains("terminal_epoch"));
    }

    #[test]
    fn altered_bound_source_is_rejected_before_consumption() {
        let temp = tempdir().expect("tempdir");
        let (operands, payload) = fixture(temp.path());
        let source_path = PathBuf::from(
            payload["source_artifact_binding_contract"]["bindings"][0]["path"]
                .as_str()
                .expect("source path"),
        );
        fs::write(source_path, b"tampered\n").expect("tamper source");
        let error = verify_candidate_evidence(&operands).expect_err("source tamper must fail");
        assert!(error.to_string().contains("source binding"));
    }

    #[test]
    fn altered_qnanl_packet_is_rejected_even_when_sha_operand_is_updated() {
        let temp = tempdir().expect("tempdir");
        let (mut operands, _) = fixture(temp.path());
        let mut bytes = fs::read(&operands.packet).expect("packet bytes");
        let last = bytes.len() - 1;
        bytes[last] ^= 0x01;
        fs::write(&operands.packet, bytes).expect("tamper packet");
        operands.packet_sha256 = sha256_file_hex(&operands.packet).expect("tampered packet sha");
        let error = verify_candidate_evidence(&operands).expect_err("packet tamper must fail");
        assert!(error.to_string().contains("decode failed"));
    }

    #[test]
    fn self_consistent_sealed_false_source_row_assertion_is_rejected() {
        let temp = tempdir().expect("tempdir");
        let (mut operands, mut payload) = fixture(temp.path());
        let source_path = PathBuf::from(
            payload["source_artifact_binding_contract"]["bindings"][0]["path"]
                .as_str()
                .expect("source path"),
        );
        let substituted_row = json!({
            "candidate_id": "candidate_a",
            "canonical_smiles": "CCC",
            "execution_order": 0,
            "score_components": {"effective_reward": 13.0, "nested": {"active": [0.4, 0.8]}},
        });
        let mut row_bytes = serde_json::to_vec(&substituted_row).expect("JSONL row");
        row_bytes.push(b'\n');
        fs::write(&source_path, row_bytes).expect("substitute source row");
        let source_sha = sha256_file_hex(&source_path).expect("source sha");
        let source_row_sha =
            sha256_bytes(&canonical_json_bytes(&substituted_row).expect("source row"));
        payload["source_artifact_binding_contract"]["bindings"][0]["sha256"] =
            Value::String(source_sha.clone());
        payload["source_artifact_binding_contract"]["bindings"][0]["size_bytes"] =
            json!(fs::metadata(&source_path).expect("source metadata").len());
        payload["source_artifact_binding_contract"]["row_bindings"][0]["source_row_sha256"] =
            Value::String(source_row_sha.clone());
        payload["candidates"][0]["source_row_lineage"]["source_artifact_sha256"] =
            Value::String(source_sha);
        payload["candidates"][0]["source_row_lineage"]["source_row_sha256"] =
            Value::String(source_row_sha);
        refresh_payload_integrity(&mut payload);
        reseal_payload(&mut operands, &payload);

        let error = verify_candidate_evidence(&operands)
            .expect_err("false source-row assertion must fail despite valid sealing/hashes");
        assert!(error.to_string().contains("canonical_smiles"));
    }

    #[test]
    fn self_consistent_sealed_false_upstream_identity_assertion_is_rejected() {
        let temp = tempdir().expect("tempdir");
        let (mut operands, mut payload) = fixture(temp.path());
        let receipt_path = PathBuf::from(
            payload["source_artifact_binding_contract"]["bindings"][1]["path"]
                .as_str()
                .expect("receipt path"),
        );
        let mut receipt: Value =
            serde_json::from_slice(&fs::read(&receipt_path).expect("receipt bytes"))
                .expect("receipt JSON");
        receipt["method"] = Value::String("modified_reward_method".to_string());
        receipt["irrelevant_nested_decoy"] = json!({"method": ACCEPTED_METHOD});
        write_json(&receipt_path, &receipt);
        let receipt_sha = sha256_file_hex(&receipt_path).expect("receipt sha");
        payload["source_artifact_binding_contract"]["bindings"][1]["sha256"] =
            Value::String(receipt_sha.clone());
        payload["source_artifact_binding_contract"]["bindings"][1]["size_bytes"] =
            json!(fs::metadata(&receipt_path).expect("receipt metadata").len());
        payload["receipt_hashes"]["accepted_training_receipt"] = Value::String(receipt_sha);
        refresh_payload_integrity(&mut payload);
        reseal_payload(&mut operands, &payload);

        let error = verify_candidate_evidence(&operands)
            .expect_err("false upstream identity assertion must fail despite valid sealing/hashes");
        assert!(error
            .to_string()
            .contains("missing or conflicting top-level method"));
    }

    #[test]
    fn partial_candidate_operand_group_is_fail_closed() {
        let error = CandidateEvidenceOperands::from_options(
            Some(PathBuf::from("/tmp/payload")),
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .expect_err("partial group must fail");
        assert!(error.to_string().contains("all-or-none operand group"));
    }
}
