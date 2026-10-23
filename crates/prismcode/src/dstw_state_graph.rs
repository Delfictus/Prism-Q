//! DSTW-T4 scientific-state vertex/transition emitters (DSTW-T4 Step 4).
//!
//! Emits the lifecycle graph for one prismos `md-run`:
//!   * `pre_md_state_vertex.json`   — initial state (topology-grounded)
//!   * `post_md_state_vertex.json`  — post-integration state (engine artifacts)
//!   * `md_state_transition.json`   — the MD integration edge (pre -> post)
//!   * `next_operator_recommendation.json`          — prioritized engine-owner work
//!   * (conditional) `post_pathb_state_vertex.json` + transition, and
//!     `post_tokenizer_state_vertex.json` + transition, ONLY when the upstream
//!     PathB / tokenizer artifacts already exist on disk.
//!
//! Every emitted artifact gets the triple required by the directive:
//!   (a) semantic JSON per `scientific_state_*.schema.json`, with a MANDATORY
//!       honest `runtime_origin`;
//!   (b) a QNANL seal via [`crate::qnanl_artifact::encode_bytes`] that is
//!       byte-exact decode-PROVEN here (re-decode from disk + compare), with a
//!       `<file>.decode_proof.json` side-car;
//!   (c) a typed PrismCode operand record in the [`OperandLedger`]
//!       (`STATE.VERTEX_CREATE` / `STATE.OPERATOR_APPLY` / `STATE.EMIT_TRANSITION`
//!       / `STATE.CLASSIFY_BLOCKED` / `STATE.NEXT_OPERATOR_REC`), VALUE-verified
//!       sha256 (the ledger OPENS each file).
//!
//! HONESTY BOUNDARIES (mirror the FORBIDDEN GUARDRAILS):
//!   * `runtime_origin` is on every artifact. Vertices/transitions that this
//!     prismos lifecycle orchestrated are `RUNTIME_SUBPROCESS_ORCHESTRATED_BY_PRISMOS`
//!     (the engine subprocess produced the underlying tensors); the planning
//!     recommendation prismos authors in-process is `RUNTIME_INLINE`; PathB/tokenizer
//!     vertices read from runtime child stages are explicitly origin-labeled. NONE
//!     claim in-MD-loop emission unless the engine itself writes that receipt.
//!   * BLOCKED tensors carry `value=null` + a named blocker; raw dense bytes are
//!     never inlined as semantic evidence (Guardrail 3).
//!   * POPULATABLE tensor values are VALUE-grounded: when an engine artifact is
//!     present its `value` carries the real {bytes, sha256}; when absent the tensor
//!     is BLOCKED, never null-padded into a fake populated value.
//!   * Apo/holo are execution states, not labels. A valid holo run is represented
//!     as holo/GDP state because `prismos md-run` already rejected non-GDP
//!     topologies before the engine launch.

use crate::operand_ledger::{
    OperandLedger, OperandRecord, OperandTarget, OperatorClass, RuntimeOrigin,
};
use crate::qnanl_artifact::{
    decode_packet_bytes, encode_bytes, QnanlArtifactFamily, QnanlEncodeOptions,
};
use serde::Serialize;
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub const DSTW_T4_SCHEMA_VERSION: u32 = 1;
pub const DSTW_T4_VERTEX_SCHEMA_KIND: &str = "dstw_t4_scientific_state_vertex";
pub const DSTW_T4_TRANSITION_SCHEMA_KIND: &str = "dstw_t4_scientific_state_transition";
pub const DSTW_T4_NEXT_OPERATOR_SCHEMA_KIND: &str = "dstw_t4_next_operator_recommendation";
pub const DSTW_T4_VALIDATION_STATE_SCHEMA_KIND: &str = "dstw_t4_validation_state";
pub const DSTW_T4_EPOCH_OVERLAY_SCHEMA_KIND: &str = "dstw_t4_epoch_overlay";
pub const DSTW_T4_CHRONOLOGY_ACCEPTANCE_DECISION_SCHEMA_KIND: &str =
    "dstw_t4_chronology_acceptance_decision";

pub type DstwResult<T> = Result<T, String>;

/// Inputs the prismos `run_md` lifecycle hands to the emitter. All paths are
/// already resolved; `holo_topology` is `Some` only when the GDP-parameterized
/// topology was located (its sha256 is VALUE-verified here, never trusted by name).
#[derive(Clone, Debug)]
pub struct DstwStateGraphInputs {
    pub output_dir: PathBuf,
    pub topology_path: PathBuf,
    pub target_id: String,
    pub ligand_mode: String,
    pub detection_mode: String,
    pub site_ranker: String,
    pub md_only_evidence: bool,
    pub engine_path: PathBuf,
    pub operator_version: String,
    pub command_argv: Vec<String>,
    pub holo_topology: Option<PathBuf>,
    pub opcode_counts: BTreeMap<String, u64>,
    pub dense_tensor_runtime_origin: RuntimeOrigin,
}

#[derive(Clone, Debug)]
pub struct DstwEmittedArtifact {
    pub kind: String,
    pub json_path: PathBuf,
    pub qnanl_packet_path: PathBuf,
    pub decode_proof_pass: bool,
    pub runtime_origin: String,
}

#[derive(Clone, Debug)]
pub struct DstwStateGraphReport {
    pub emitted: Vec<DstwEmittedArtifact>,
    pub operand_ledger_path: PathBuf,
    pub operand_record_count: usize,
    pub all_decode_proofs_pass: bool,
    /// DSTW-T4 Step h5: number of dense-tensor semantic summary records emitted
    /// (one per dense engine tensor: signal_grid / warp / DSTW / superresolution / TWIN16 / sensor).
    pub dense_tensor_summary_count: usize,
    /// DSTW-T4 Step h6: number of artifacts the PRISM-native merge manifest classified.
    pub merge_manifest_artifact_count: usize,
    /// DSTW-T4 Step h6: artifacts that did NOT map to a state-object class (NOT PRISM-native).
    pub merge_manifest_orphan_count: usize,
    /// DSTW-T4 Step h6: rows that passed EVERY merge-readiness gate (TRAINING_READY_PRISMOS_NATIVE).
    pub merge_manifest_training_ready_native_count: usize,
    /// DSTW-T4 Step h6: native-origin rows blocked by a PENDING (tokenizer-owned) gate.
    pub merge_manifest_native_candidate_blocked_count: usize,
    /// DSTW-T4 Step h6: derived / post-run / blocked rows, SEPARATED from native candidates.
    pub merge_manifest_derived_or_post_run_count: usize,
    /// DSTW-T4 Step h6: the nine merge-manifest acceptance artifact paths.
    pub merge_manifest_paths: Vec<PathBuf>,
}

// ---- identity ----------------------------------------------------------------

fn blake3_hex(parts: &[&str]) -> String {
    let joined = parts.join("|");
    blake3::hash(joined.as_bytes()).to_hex().to_string()
}

fn vertex_id(target: &str, ligand_state: &str, detection: &str, topo_sha: &str) -> String {
    format!(
        "vtx_{}",
        &blake3_hex(&[target, ligand_state, detection, topo_sha])[..32]
    )
}

fn transition_id(source_vertex: &str, operator: &str, version: &str, argv: &[String]) -> String {
    let argv_join = argv.join(" ");
    format!(
        "txn_{}",
        &blake3_hex(&[source_vertex, operator, version, &argv_join])[..32]
    )
}

// ---- value-verified hashing --------------------------------------------------

fn sha256_of(path: &Path) -> Option<String> {
    crate::operand_ledger::sha256_file_hex(path).ok()
}

fn file_bytes(path: &Path) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

fn topology_gdp_present(topology: &Path) -> bool {
    std::fs::read(topology)
        .ok()
        .map(|bytes| String::from_utf8_lossy(&bytes).contains("GDP"))
        .unwrap_or(false)
}

fn normalized_ligand_state(ligand_mode: &str, gdp_present: bool) -> &'static str {
    let mode = ligand_mode.to_ascii_lowercase();
    if mode.contains("holo") || mode.contains("gdp") {
        if gdp_present {
            "holo_gdp"
        } else {
            "holo_missing_gdp"
        }
    } else if gdp_present {
        "holo_gdp"
    } else {
        "apo"
    }
}

fn ligand_engine_effect(ligand_state: &str) -> &'static str {
    match ligand_state {
        "holo_gdp" => "holo_gdp_topology_drives_engine_input",
        "holo_missing_gdp" => "blocked_before_engine_non_gdp_topology",
        _ => "apo_md_only_evidence",
    }
}

fn ligand_feasibility(ligand_state: &str) -> &'static str {
    match ligand_state {
        "holo_missing_gdp" => "BLOCKED",
        _ => "POPULATABLE_NOW",
    }
}

fn ligand_blocker(ligand_state: &str) -> Option<&'static str> {
    match ligand_state {
        "holo_missing_gdp" => Some(
            "HOLO_TOPOLOGY_GDP_ABSENT: LigandMode::Holo requires the engine-driving topology \
             to contain GDP; prismos md-run rejects this before launch.",
        ),
        _ => None,
    }
}

/// First file in `dir` whose name ends with `suffix`, sorted for determinism.
fn find_artifact(dir: &Path, suffix: &str) -> Option<PathBuf> {
    let mut matches: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            !crate::qnanl_artifact::is_native_continuation_recovery_path(path)
                && path.is_file()
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .map(|name| name.ends_with(suffix))
                    .unwrap_or(false)
        })
        .collect();
    matches.sort();
    matches.into_iter().next()
}

fn find_artifact_recursive(dir: &Path, suffix: &str) -> Option<PathBuf> {
    fn walk(root: &Path, suffix: &str, depth: usize, out: &mut Vec<PathBuf>) {
        if depth > 4 {
            return;
        }
        let Ok(entries) = std::fs::read_dir(root) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if crate::qnanl_artifact::is_native_continuation_recovery_path(&path) { continue; }
            if path.is_dir() {
                if matches!(
                    name,
                    "qnanl_runtime_stream" | "qnanl_runtime_receipts" | ".git" | "target"
                ) {
                    continue;
                }
                walk(&path, suffix, depth + 1, out);
            } else if path.is_file() && name.ends_with(suffix) {
                out.push(path);
            }
        }
    }
    let mut matches = Vec::new();
    walk(dir, suffix, 0, &mut matches);
    matches.sort();
    matches.into_iter().next()
}

// ---- tensor envelopes --------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage {
    PreMd,
    PostMd,
}

/// A decoded engine-artifact-grounded tensor value, or BLOCKED with a named
/// blocker. Never null-padded: POPULATABLE iff a real artifact/value exists.
fn engine_artifact_tensor(output_dir: &Path, family: &str, source: &str, suffix: &str) -> Value {
    match find_artifact(output_dir, suffix) {
        Some(path) => {
            let bytes = file_bytes(&path);
            let sha = sha256_of(&path).unwrap_or_default();
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string();
            json!({
                "feasibility": "POPULATABLE_NOW",
                "tokenizer_family": family,
                "source_artifact": source,
                "runtime_origin": RuntimeOrigin::RuntimeSubprocessOrchestratedByPrismos.as_str(),
                "blocker": Value::Null,
                "value": {
                    "artifact": name,
                    "bytes": bytes,
                    "sha256": sha,
                    "decode_note": "byte-grounded artifact reference (VALUE-verified sha256); \
                                    full per-residue decode is the offline tok_export/decode-before-tokenize path (INV-1)"
                }
            })
        }
        None => json!({
            "feasibility": "BLOCKED",
            "tokenizer_family": family,
            "source_artifact": source,
            "runtime_origin": RuntimeOrigin::Blocked.as_str(),
            "blocker": format!(
                "engine artifact *{suffix} absent in this run's output dir; native runtime emission \
                 gated on MdNativeRuntimeState::write_receipt wiring (md_native_runtime.rs:306-317)"
            ),
            "value": Value::Null
        }),
    }
}

fn blocked_tensor(family: Option<&str>, source: &str, blocker: &str) -> Value {
    json!({
        "feasibility": "BLOCKED",
        "tokenizer_family": family,
        "source_artifact": source,
        "runtime_origin": RuntimeOrigin::Blocked.as_str(),
        "blocker": blocker,
        "value": Value::Null
    })
}

fn offline_tensor(family: &str, source: &str, scope: &str) -> Value {
    json!({
        "feasibility": "POPULATABLE_NOW",
        "tokenizer_family": family,
        "source_artifact": source,
        "runtime_origin": RuntimeOrigin::ExternalDerived.as_str(),
        "blocker": Value::Null,
        "value": {
            "scope": scope,
            "engine_native": false,
            "pipeline": "canonical offline python (feature_registry.py asserts 14/14; site_ranker.py lexicographic)"
        }
    })
}

fn topology_tensor(topology: &Path) -> Value {
    let bytes = file_bytes(topology);
    let sha = sha256_of(topology).unwrap_or_default();
    let name = topology
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("topology.json")
        .to_string();
    json!({
        "feasibility": "POPULATABLE_NOW",
        "tokenizer_family": Value::Null,
        "source_artifact": "<target>_clean.topology.json (positions/masses/elements/charges/lj_params/bonds/angles/dihedrals)",
        "runtime_origin": RuntimeOrigin::RuntimeSubprocessOrchestratedByPrismos.as_str(),
        "blocker": Value::Null,
        "value": {
            "artifact": name,
            "bytes": bytes,
            "sha256": sha,
            "decode_note": "topology is decoded JSON; positions/charges/bonded terms present"
        }
    })
}

/// The 13 schema tensor fields for one vertex at a given lifecycle `stage`.
fn build_tensor_block(stage: Stage, output_dir: &Path, topology: &Path) -> Map<String, Value> {
    let mut m = Map::new();

    m.insert("topology_tensor".to_string(), topology_tensor(topology));

    // coordinate: topology positions (always) + per-residue net displacement
    // (kcc_v2full) — populatable at PostMd if the engine artifact exists.
    let coordinate = match stage {
        Stage::PreMd => json!({
            "feasibility": "POPULATABLE_NOW",
            "tokenizer_family": "KccDescriptor",
            "source_artifact": "topology positions (initial coordinates, pre-integration)",
            "runtime_origin": RuntimeOrigin::RuntimeSubprocessOrchestratedByPrismos.as_str(),
            "blocker": Value::Null,
            "value": {
                "stage": "pre_md_initial",
                "net_displacement": "zero (initial frame; net_dx/dy/dz materialize at post_md)",
                "source": "topology.positions"
            }
        }),
        Stage::PostMd => engine_artifact_tensor(
            output_dir,
            "KccDescriptor",
            "topology positions + kcc_v2full.bin net_dx/net_dy/net_dz (nhs_rt_full.rs:2542-2561)",
            "_kcc_v2full.bin",
        ),
    };
    m.insert("coordinate_tensor".to_string(), coordinate);

    m.insert(
        "force_tensor".to_string(),
        blocked_tensor(
            None,
            "forces_final.bin + asc_vectors.bin (NumericArray opaque); mechanical_load.rs reducer",
            "no scalar-summary materializer (per-atom n_atoms x 6 f32); no MechanicalLoadSummary tokenizer family (mechanical_load.rs:144-158)",
        ),
    );
    m.insert(
        "signal_grid_tensor".to_string(),
        blocked_tensor(
            None,
            "signal_grid.bin (PRSGD001); SignalGridSummary reducer",
            "SignalGridSummary reducer is POST_RUN only (re-reads raw .bin); no typed family; not wired to teardown (dispatch_analysis_common/mod.rs:12-26)",
        ),
    );
    m.insert(
        "warp_tensor".to_string(),
        blocked_tensor(
            None,
            "warp_matrix.bin (DenseGpuField 136B/voxel ~285MB/stream); WarpSummary reducer",
            "sealed DENSE_FIELD_CANONICAL_BYTES opaque; WarpSummary offline-only; no WarpFieldSummary family (warp_jacobian.rs:25-31; qnanl_artifact.rs:126)",
        ),
    );

    // kcc: READY (PRKCC001, RUNTIME_INLINE at teardown) — populatable at PostMd.
    let kcc = match stage {
        Stage::PreMd => blocked_tensor(
            Some("KccDescriptor"),
            "kcc_v2full.bin (PRKCC001)",
            "pre-MD initial state: KCC descriptors not yet materialized (emitted at post_md teardown)",
        ),
        Stage::PostMd => engine_artifact_tensor(
            output_dir,
            "KccDescriptor",
            "kcc_v2full.bin (PRKCC001), 14 fields -> KccDescriptor (md_evidence_write_kcc_v2full nhs_rt_full.rs:2530)",
            "_kcc_v2full.bin",
        ),
    };
    m.insert("kcc_tensor".to_string(), kcc);

    m.insert(
        "kinetic_tensor".to_string(),
        blocked_tensor(
            Some("MsmKineticSummary"),
            "MsmKineticSummary / FiberBundleDstw.adaptive_dt",
            "MSM sealed stack is PROXY (ADR-0007 PENDING); kinetic needs trajectories + multi-variant (not md-only); adaptive_dt reduction materializer missing",
        ),
    );
    m.insert(
        "thermodynamic_tensor".to_string(),
        blocked_tensor(
            None,
            "prism_therm (offline therm_class); signal_grid hysteresis_score (BLOCKED)",
            "--emit-runtime-hysteresis-coverage has no coverage OUTPUT field (parse_hysteresis_ratio parses INPUT only, nhs_rt_full.rs:1329); hysteresis lives in BLOCKED SignalGridSummary",
        ),
    );

    // spike: READY (PRSPK001, always-on sensor) — populatable at PostMd.
    let spike = match stage {
        Stage::PreMd => blocked_tensor(
            Some("SpikeEvent"),
            "spikes.bin (PRSPK001) Vec<GpuSpikeEvent>",
            "pre-MD initial state: no spike events emitted yet (always-on sensor begins at integration)",
        ),
        Stage::PostMd => engine_artifact_tensor(
            output_dir,
            "SpikeEvent",
            "spikes.bin (PRSPK001) Vec<GpuSpikeEvent> -> SpikeEvent (md_evidence_write_spikes nhs_rt_full.rs:1772)",
            "_spikes.bin",
        ),
    };
    m.insert("spike_sensor_tensor".to_string(), spike);

    let epistemic = match stage {
        Stage::PreMd => blocked_tensor(
            Some("FiberBundleDstw"),
            "FiberBundleDstw{fisher_info,failure_probability,confidence_identity} + BocpdLog{posterior_max,map_run_length}",
            "pre-MD initial state: DSTW/BOCPD epistemic scalars not yet accumulated",
        ),
        Stage::PostMd => engine_artifact_tensor(
            output_dir,
            "FiberBundleDstw",
            "FiberBundleDstw + BocpdLog runtime-inline scalars (dstw_dispatch/dtsg_parquet_emitter.rs)",
            "_adaptive_dt.bin",
        ),
    };
    m.insert("epistemic_tensor".to_string(), epistemic);

    m.insert(
        "lineage_tensor".to_string(),
        blocked_tensor(
            None,
            "lineage_dag (NOT emitted); STATE.HASH/STATE.SNAPSHOT opcodes",
            "lineage_dag absent everywhere (C2); STATE.HASH has_dispatch_handler=N (def-only, no execution)",
        ),
    );
    m.insert(
        "validation_tensor".to_string(),
        offline_tensor(
            "FalsifiablePrediction",
            "canonical 14-feature gating (feature_registry.py) + FalsifiablePrediction",
            "VALIDATE_ONLY_OFFLINE",
        ),
    );
    m.insert(
        "prediction_tensor".to_string(),
        offline_tensor(
            "RankerRow",
            "site_ranker.py lexicographic rows -> FalsifiablePrediction / RankerRow",
            "VALIDATE_ONLY_OFFLINE",
        ),
    );

    m
}

/// BLOCKED tensors of a tensor block -> missing_preconditions rows.
fn missing_preconditions_from(tensors: &Map<String, Value>) -> Vec<Value> {
    let mut rows = Vec::new();
    for (field, env) in tensors {
        let feasibility = env.get("feasibility").and_then(Value::as_str).unwrap_or("");
        if feasibility != "BLOCKED" {
            continue;
        }
        let blocker = env.get("blocker").and_then(Value::as_str).unwrap_or("");
        rows.push(json!({
            "tensor": field,
            "blocker": blocker,
            "unblock_action": "wire MdNativeRuntimeState::write_receipt + flip the DEFERRED readout opcodes (md_native_runtime.rs:306-317), or add the missing scalar-summary materializer + tokenizer family",
            "evidence": env.get("source_artifact").cloned().unwrap_or(Value::Null)
        }));
    }
    rows
}

/// Engine outputs present in the dir that map to a QNANL family + already have a
/// sibling `.prismqnanl` seal -> qnanl_receipts rows. Self-seal is NOT listed
/// (avoids seal/content recursion). artifact_hashes captured alongside.
fn scan_engine_artifacts(output_dir: &Path) -> (Vec<Value>, Map<String, Value>, Vec<Value>) {
    let mut qnanl_receipts = Vec::new();
    let mut artifact_hashes = Map::new();
    let mut output_artifacts = Vec::new();
    let interesting = [
        "_kcc_v2full.bin",
        "_spikes.bin",
        "_adaptive_dt.bin",
        "_forces_final.bin",
        "_signal_grid.bin",
        "_warp_matrix.bin",
    ];
    if let Ok(rd) = std::fs::read_dir(output_dir) {
        let mut paths: Vec<PathBuf> = rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_file())
            .collect();
        paths.sort();
        for path in paths {
            let name = match path.file_name().and_then(|n| n.to_str()) {
                Some(n) => n.to_string(),
                None => continue,
            };
            if !interesting.iter().any(|suf| name.ends_with(suf)) {
                continue;
            }
            let sha = sha256_of(&path).unwrap_or_default();
            artifact_hashes.insert(name.clone(), Value::String(sha.clone()));
            output_artifacts.push(json!({
                "path": path.display().to_string(),
                "sha256": sha,
                "bytes": file_bytes(&path)
            }));
            let sealed = PathBuf::from(format!("{}.prismqnanl", path.display()));
            if sealed.exists() {
                let seal_sha = sha256_of(&sealed).unwrap_or_default();
                qnanl_receipts.push(json!({
                    "path": sealed.display().to_string(),
                    "qnanl_family": "PER_ENGINE_ARTIFACT_SEAL",
                    "seal_hash": seal_sha
                }));
            }
        }
    }
    (qnanl_receipts, artifact_hashes, output_artifacts)
}

fn operator_receipts_value() -> Value {
    json!([{
        "receipt": "OPERATOR_CONTROL/PRISMOS_NATIVE_MD_EVIDENCE_PIPELINE_20260627",
        "kind": "operator_decision_ledger",
        "note": "this pipeline's deliverable ledgers (vertex/transition schemas, feasibility ledger, blocker ledger)"
    }])
}

fn tokenizer_projection_value() -> Value {
    json!({
        "feasibility": "POPULATABLE_NOW",
        "families_projected": [
            "DstwStateVertex",
            "DstwStateTransition",
            "DstwNextOperator",
            "DenseTensorSemanticSummary"
        ],
        "matched_fields_by_family": {},
        "projection_path": "prismos md-run --emit-runtime-tokenizer-manifest invokes tok_export_full_prism_corpus over this state graph after emission",
        "dense_tensor_policy": "dense tensors enter the tokenizer only through semantic summary rows, never raw bytes"
    })
}

// ---- DSTW-T4 Step h4: typed BlockedState + falsifiable validation gate -------

/// Typed blocked-state record (DSTW-T4 Step h4). A `BlockedState` says which
/// operator/precondition is unavailable, what evidence exists, the precise
/// runtime-capability gap, and the next step.
#[derive(Clone, Debug, Serialize)]
pub struct BlockedState {
    pub blocker_id: String,
    pub required_operator: String,
    pub available_evidence: String,
    pub missing_runtime_capability: String,
    pub next_operator: String,
}

impl BlockedState {
    /// The fail-closed ligand-preflight blocker used only if the state graph is
    /// invoked directly with a holo mode but a non-GDP topology. Normal prismos
    /// `md-run` rejects this before engine launch.
    pub fn holo_topology_gdp_absent() -> Self {
        BlockedState {
            blocker_id: "HOLO_TOPOLOGY_GDP_ABSENT".to_string(),
            required_operator: "GDP_CONTAINING_ENGINE_TOPOLOGY".to_string(),
            available_evidence:
                "selected engine topology was opened and scanned for GDP before launch".to_string(),
            missing_runtime_capability:
                "the selected topology does not contain GDP, so holo execution is invalid"
                    .to_string(),
            next_operator: "SUPPLY_GDP_PARAMETERIZED_TOPOLOGY_AND_RERUN".to_string(),
        }
    }

    pub fn to_value(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
}

/// One expected-change row for a falsifiable transition:
/// `(tensor, expectation, required, evidence_suffix)`.
type ExpectedChange<'a> = (&'a str, &'a str, bool, &'a str);

/// Result of computing the falsifiable validation gate for a transition.
struct GateResult {
    verdict: String,
    expected_change: Value,
    observed_change: Value,
    failed_preconditions: Vec<Value>,
    contradiction_refs: Vec<Value>,
    validation_gate: Value,
}

/// `operator_id` block — the engine ISA opcode id if registered, else an honest
/// "this is an orchestration verb, not an ISA opcode" record (no fabricated id).
fn operator_id_block(operator_name: &str) -> Value {
    match crate::binary_runtime::opcode_id_for_name(operator_name) {
        Some(id) => json!({ "opcode_id": format!("0x{id:04X}"), "registered": true }),
        None => json!({
            "opcode_id": Value::Null,
            "registered": false,
            "absent_reason": "operator is a prismos-orchestration verb, not a registered engine \
                              ISA opcode (binary_runtime opcode registry, binary_runtime.rs:4366)"
        }),
    }
}

fn operator_family(operator_name: &str) -> &'static str {
    OperatorClass::classify(operator_name).as_str()
}

/// Compute the FALSIFIABLE validation gate for a transition (DSTW-T4 Step h4).
///
/// The verdict is derived from an expected-vs-observed evidence comparison, NOT
/// from the existence of the transition record. For each expected change we look
/// for the backing engine artifact and compute its REAL sha256; a REQUIRED
/// expected change with no backing artifact FAILS the gate (and emits a typed
/// contradiction), even though the transition JSON itself was written. A BLOCKED
/// transition (no required operator in the engine) is `BLOCKED_PRECONDITION`.
fn compute_validation_gate(
    output_dir: &Path,
    pre_baseline_hash: &str,
    expected: &[ExpectedChange],
    blocked: Option<&BlockedState>,
) -> GateResult {
    let expected_rows: Vec<Value> = expected
        .iter()
        .map(|(t, e, req, suf)| {
            json!({ "tensor": t, "expectation": e, "required": req, "evidence_suffix": suf })
        })
        .collect();

    if let Some(b) = blocked {
        let observed_change = json!({
            "feasibility": "BLOCKED",
            "coordinate_hash": Value::Null,
            "force_hash": Value::Null,
            "evidence_hash": Value::Null,
            "absent_reason": format!(
                "{}: no bound-state MD executed; {}",
                b.blocker_id, b.missing_runtime_capability
            )
        });
        let failed = vec![b.to_value()];
        let contradiction = vec![json!({
            "contradiction_id": b.blocker_id,
            "claim": "requested ligand/precondition state is executable",
            "observation": b.missing_runtime_capability,
            "blocker_ref": b.blocker_id
        })];
        let validation_gate = json!({
            "verdict": "BLOCKED_PRECONDITION",
            "passed": false,
            "basis": "expected bound-state change cannot be OBSERVED: required_operator is not \
                      available in the engine (verdict is NOT a function of record existence)",
            "required_operator": b.required_operator,
            "next_operator": b.next_operator
        });
        return GateResult {
            verdict: "BLOCKED_PRECONDITION".to_string(),
            expected_change: Value::Array(expected_rows),
            observed_change,
            failed_preconditions: failed,
            contradiction_refs: contradiction,
            validation_gate,
        };
    }

    let mut observed_rows: Vec<Value> = Vec::new();
    let mut failed: Vec<Value> = Vec::new();
    let mut contradiction: Vec<Value> = Vec::new();
    let mut present_pairs: Vec<String> = Vec::new();
    let mut coordinate_hash = Value::Null;
    let mut any_observed = false;

    for (tensor, expectation, required, suffix) in expected {
        match find_artifact(output_dir, suffix) {
            Some(path) => {
                any_observed = true;
                let sha = sha256_of(&path).unwrap_or_default();
                let bytes = file_bytes(&path);
                let distinct = sha != pre_baseline_hash;
                let name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("")
                    .to_string();
                present_pairs.push(format!("{name}:{sha}"));
                if *tensor == "coordinate_tensor" {
                    coordinate_hash = json!(sha);
                }
                observed_rows.push(json!({
                    "tensor": tensor,
                    "observed": true,
                    "evidence_artifact": name,
                    "post_hash": sha,
                    "bytes": bytes,
                    "distinct_from_pre_baseline": distinct
                }));
                if !distinct {
                    contradiction.push(json!({
                        "contradiction_id": "POST_EVIDENCE_IDENTICAL_TO_PRE_BASELINE",
                        "tensor": tensor,
                        "claim": expectation,
                        "observation": "post evidence hash equals the pre-state baseline hash \
                                        (operator produced no observable change)"
                    }));
                }
            }
            None => {
                observed_rows.push(json!({
                    "tensor": tensor,
                    "observed": false,
                    "evidence_artifact": Value::Null,
                    "absent_reason": format!(
                        "expected evidence artifact *{suffix} not present in output dir"
                    )
                }));
                if *required {
                    failed.push(json!({
                        "tensor": tensor,
                        "expectation": expectation,
                        "failed_precondition": format!(
                            "REQUIRED expected change unobserved: no *{suffix} evidence artifact"
                        ),
                        "unblock_action": "ensure the engine subprocess emitted the backing \
                                           artifact (native write_receipt) or rerun"
                    }));
                    contradiction.push(json!({
                        "contradiction_id": "EXPECTED_CHANGE_UNOBSERVED",
                        "tensor": tensor,
                        "claim": expectation,
                        "observation": format!(
                            "required evidence *{suffix} absent — transition cannot be confirmed \
                             by record existence alone"
                        )
                    }));
                }
            }
        }
    }

    let (force_hash, force_absent) = match find_artifact(output_dir, "_forces_final.bin") {
        Some(p) => (json!(sha256_of(&p).unwrap_or_default()), Value::Null),
        None => (
            Value::Null,
            json!(
                "forces_final.bin not materialized; MechanicalLoadSummary reducer is \
                   offline-only (mechanical_load.rs:144-158)"
            ),
        ),
    };

    present_pairs.sort();
    let evidence_hash = if present_pairs.is_empty() {
        Value::Null
    } else {
        let refs: Vec<&str> = present_pairs.iter().map(String::as_str).collect();
        json!(blake3_hex(&refs))
    };

    let required_unobserved = failed.len();
    let verdict = if required_unobserved > 0 {
        "FAIL_EXPECTED_CHANGE_UNOBSERVED"
    } else if any_observed {
        "PASS_OBSERVED_CHANGE_VERIFIED"
    } else {
        "INCONCLUSIVE_NO_EVIDENCE"
    };

    let observed_change = json!({
        "feasibility": "POPULATABLE_NOW",
        "coordinate_hash": coordinate_hash,
        "force_hash": force_hash,
        "force_hash_absent_reason": force_absent,
        "evidence_hash": evidence_hash,
        "evidence_hash_algo": "blake3(sorted name:sha256 of present evidence artifacts)",
        "observations": observed_rows
    });
    let validation_gate = json!({
        "verdict": verdict,
        "passed": verdict == "PASS_OBSERVED_CHANGE_VERIFIED",
        "basis": "verdict computed from expected-vs-observed evidence comparison; NOT from \
                  transition-record existence",
        "required_expected_changes": expected.iter().filter(|(_, _, r, _)| *r).count(),
        "required_unobserved": required_unobserved
    });

    GateResult {
        verdict: verdict.to_string(),
        expected_change: Value::Array(expected_rows),
        observed_change,
        failed_preconditions: failed,
        contradiction_refs: contradiction,
        validation_gate,
    }
}

// ---- vertex / transition JSON builders --------------------------------------

fn detection_mode_value(detection: &str, site_ranker: &str, md_only: bool) -> Value {
    // §1.E correction: --detection-mode is a REAL clap flag (prismos.rs:825-826,
    // DetectionMode: ValueEnum) consumed into engine ranking argv -> POPULATABLE.
    json!({
        "feasibility": "POPULATABLE_NOW",
        "posture": {
            "detection_mode": detection,
            "site_ranker": site_ranker,
            "md_only_evidence": md_only
        },
        "blocker": Value::Null
    })
}

#[allow(clippy::too_many_arguments)]
fn build_vertex(
    inputs: &DstwStateGraphInputs,
    stage: Stage,
    state: &str,
    ligand_feasibility: &str,
    gdp_present: bool,
    ligand_input_sha: Option<String>,
    ligand_blocker: Option<&str>,
    runtime_origin: RuntimeOrigin,
) -> (String, Value, Map<String, Value>) {
    let topo_sha = ligand_input_sha
        .clone()
        .or_else(|| sha256_of(&inputs.topology_path))
        .unwrap_or_default();
    let vid = vertex_id(&inputs.target_id, state, &inputs.detection_mode, &topo_sha);
    let tensors = build_tensor_block(stage, &inputs.output_dir, &inputs.topology_path);
    let missing = missing_preconditions_from(&tensors);
    let (qnanl_receipts, artifact_hashes, _outputs) = scan_engine_artifacts(&inputs.output_dir);

    let mut obj = Map::new();
    obj.insert("schema_kind".to_string(), json!(DSTW_T4_VERTEX_SCHEMA_KIND));
    obj.insert("schema_version".to_string(), json!(DSTW_T4_SCHEMA_VERSION));
    obj.insert(
        "lifecycle_stage".to_string(),
        json!(match stage {
            Stage::PreMd => "pre_md",
            Stage::PostMd => "post_md",
        }),
    );
    obj.insert("vertex_id".to_string(), json!(vid));
    obj.insert("target_id".to_string(), json!(inputs.target_id));
    obj.insert(
        "ligand_state".to_string(),
        json!({
            "state": state,
            "feasibility": ligand_feasibility,
            "gdp_present": gdp_present,
            "input_sha256": ligand_input_sha,
            "blocker": ligand_blocker
        }),
    );
    obj.insert(
        "detection_mode".to_string(),
        detection_mode_value(
            &inputs.detection_mode,
            &inputs.site_ranker,
            inputs.md_only_evidence,
        ),
    );
    for (k, v) in tensors.iter() {
        obj.insert(k.clone(), v.clone());
    }
    obj.insert(
        "tokenizer_projection".to_string(),
        tokenizer_projection_value(),
    );
    obj.insert("qnanl_receipts".to_string(), Value::Array(qnanl_receipts));
    obj.insert("operator_receipts".to_string(), operator_receipts_value());
    obj.insert(
        "artifact_hashes".to_string(),
        Value::Object(artifact_hashes),
    );
    obj.insert("missing_preconditions".to_string(), Value::Array(missing));
    obj.insert("runtime_origin".to_string(), json!(runtime_origin.as_str()));
    obj.insert(
        "runtime_native".to_string(),
        json!(runtime_origin.is_prismos_native()),
    );

    (vid, Value::Object(obj.clone()), tensors)
}

#[allow(clippy::too_many_arguments)]
fn build_transition(
    inputs: &DstwStateGraphInputs,
    source_vid: &str,
    target_vid: &str,
    operator_name: &str,
    operator_mode: Value,
    runtime_origin: RuntimeOrigin,
    changed: Vec<&str>,
    unchanged: Vec<&str>,
    failed: &Map<String, Value>,
    next_recommended_operator: &str,
    pre_baseline_hash: &str,
    expected: &[ExpectedChange],
    blocked: Option<BlockedState>,
) -> (String, Value) {
    let tid = transition_id(
        source_vid,
        operator_name,
        &inputs.operator_version,
        &inputs.command_argv,
    );
    let (_qnanl, _hashes, output_artifacts) = scan_engine_artifacts(&inputs.output_dir);
    let topo_sha = sha256_of(&inputs.topology_path).unwrap_or_default();

    let failed_rows: Vec<Value> = failed
        .iter()
        .filter(|(_, env)| env.get("feasibility").and_then(Value::as_str) == Some("BLOCKED"))
        .map(|(field, env)| {
            json!({
                "tensor": field,
                "reason": env.get("blocker").cloned().unwrap_or(Value::Null)
            })
        })
        .collect();

    let opcode_counts: Map<String, Value> = inputs
        .opcode_counts
        .iter()
        .map(|(k, v)| (k.clone(), json!(v)))
        .collect();

    // DSTW-T4 Step h4: FALSIFIABLE gate — expected-vs-observed evidence
    // comparison drives the verdict, not the existence of this record.
    let gate = compute_validation_gate(
        &inputs.output_dir,
        pre_baseline_hash,
        expected,
        blocked.as_ref(),
    );

    let obj = json!({
        "schema_kind": DSTW_T4_TRANSITION_SCHEMA_KIND,
        "schema_version": DSTW_T4_SCHEMA_VERSION,
        "transition_id": tid,
        // h4: explicit pre/post identity (mirror of source/target vertex ids).
        "pre_state_id": source_vid,
        "post_state_id": target_vid,
        "source_vertex_id": source_vid,
        "target_vertex_id": target_vid,
        "operator_name": operator_name,
        // h4: operator identity + family.
        "operator_id": operator_id_block(operator_name),
        "operator_family": operator_family(operator_name),
        "operator_version": inputs.operator_version,
        "operator_mode": operator_mode,
        "command_argv": inputs.command_argv,
        "input_artifacts": [{
            "path": inputs.topology_path.display().to_string(),
            "sha256": topo_sha
        }],
        "output_artifacts": output_artifacts,
        "changed_tensors": changed,
        "unchanged_tensors": unchanged,
        "failed_tensors": failed_rows,
        // h4: falsifiable proof block — expected change, observed change (real
        // computed hashes), the gate verdict, contradictions and failed
        // preconditions. A transition does NOT pass merely because files exist.
        "expected_change": gate.expected_change,
        "observed_change": gate.observed_change,
        "validation_gate": gate.validation_gate,
        "pre_state_baseline_hash": pre_baseline_hash,
        "contradiction_refs": gate.contradiction_refs,
        "failed_preconditions": gate.failed_preconditions,
        "validation_result": {
            "feasibility": "POPULATABLE_NOW",
            "scope": "FALSIFIABLE_EVIDENCE_COMPARISON",
            "verdict": gate.verdict,
            "source": "compute_validation_gate (expected-vs-observed); offline 14-feature gating \
                       (feature_registry.py / gating_stack.py) remains the canonical scientific gate"
        },
        "next_recommended_operator": next_recommended_operator,
        "next_operator_recommendation": next_recommended_operator,
        "runtime_origin": runtime_origin.as_str(),
        "qnanl_receipt": {
            "feasibility": "POPULATABLE_NOW",
            "seal_family": "JSON_TELEMETRY",
            "seal_hash": Value::Null,
            "seal_note": "byte-exact seal lives in the sibling <name>.prismqnanl packet; self-hash omitted to avoid seal/content recursion (see <name>.decode_proof.json)",
            "is_opcode_trace_grounded": false
        },
        "opcode_operand_refs": {
            "feasibility": "POPULATABLE_NOW",
            "opcode_counts": opcode_counts,
            "operand_refs": Value::Null,
            "blocker": "per-OPERAND refs BLOCKED-ENGINE: MdNativeKernelDispatch.opcodes is Vec<String> with no operand vector (md_native_runtime.rs:72)"
        }
    });
    (tid, obj)
}

// ---- DSTW-T4 Step h4: ValidationState record ---------------------------------

/// Summary of one engine artifact for the ValidationState: real bytes + sha256
/// when present, BLOCKED with a named gap otherwise. Never null-padded.
fn artifact_summary(output_dir: &Path, suffix: &str, family: &str, note: &str) -> Value {
    match find_artifact(output_dir, suffix) {
        Some(p) => json!({
            "feasibility": "POPULATABLE_NOW",
            "tokenizer_family": family,
            "artifact": p.file_name().and_then(|n| n.to_str()).unwrap_or(""),
            "bytes": file_bytes(&p),
            "sha256": sha256_of(&p).unwrap_or_default(),
            "note": note
        }),
        None => json!({
            "feasibility": "BLOCKED",
            "tokenizer_family": family,
            "value": Value::Null,
            "blocker": format!(
                "*{suffix} absent in run output (native write_receipt not wired, \
                 md_native_runtime.rs:306-317)"
            )
        }),
    }
}

/// The DSTW-T4 ValidationState consolidation record (Step h4). Aggregates the
/// target identity, hashes, evidence summaries (all VALUE-grounded or BLOCKED
/// with a reason — never null-padded), the unresolved evidence requirements, and
/// the contradiction graph references collected across the transitions.
#[derive(Clone, Debug, Serialize)]
struct ValidationStateRecord {
    schema_kind: String,
    schema_version: u32,
    target_id: String,
    ligand_state: String,
    topology_hash: Option<String>,
    coordinate_hash: Value,
    spike_event_count: Value,
    kcc_summary: Value,
    signal_summary: Value,
    warp_dstw_summary: Value,
    pathb_site_records: Value,
    known_site_overlap: Value,
    unresolved_evidence_requirements: Vec<Value>,
    contradiction_graph_refs: Vec<Value>,
    runtime_origin: String,
    runtime_native: bool,
}

fn build_validation_state(
    inputs: &DstwStateGraphInputs,
    topology_hash: Option<String>,
    contradiction_graph_refs: Vec<Value>,
) -> ValidationStateRecord {
    let dir = &inputs.output_dir;

    // coordinate evidence = post-MD kcc displacement artifact (or BLOCKED).
    let coordinate_hash = match find_artifact(dir, "_kcc_v2full.bin") {
        Some(p) => json!({
            "feasibility": "POPULATABLE_NOW",
            "source": "kcc_v2full.bin net_dx/dy/dz (nhs_rt_full.rs:2542-2561)",
            "sha256": sha256_of(&p).unwrap_or_default(),
            "bytes": file_bytes(&p)
        }),
        None => json!({
            "feasibility": "BLOCKED",
            "value": Value::Null,
            "blocker": "post-MD coordinate displacement artifact (kcc_v2full.bin) absent"
        }),
    };

    // spike/event COUNT: on-disk byte length is real; exact count needs decode.
    let spike_event_count = match find_artifact(dir, "_spikes.bin") {
        Some(p) => json!({
            "feasibility": "POPULATABLE_NOW",
            "on_disk_bytes": file_bytes(&p),
            "sha256": sha256_of(&p).unwrap_or_default(),
            "exact_event_count": Value::Null,
            "absent_reason": "exact event count requires decoding Vec<GpuSpikeEvent> records \
                              (offline decode-before-tokenize INV-1); on-disk byte length \
                              recorded, never fabricated"
        }),
        None => json!({
            "feasibility": "BLOCKED",
            "exact_event_count": Value::Null,
            "blocker": "spikes.bin (PRSPK001) absent in run output"
        }),
    };

    let kcc_summary = artifact_summary(
        dir,
        "_kcc_v2full.bin",
        "KccDescriptor",
        "PRKCC001, 14 KccDescriptor fields (nhs_rt_full.rs:2530)",
    );
    let signal_summary = json!({
        "feasibility": "BLOCKED",
        "value": Value::Null,
        "blocker": "SignalGridSummary reducer is POST_RUN only (re-reads raw signal_grid.bin); \
                    no typed family, not wired to teardown (dispatch_analysis_common/mod.rs:12-26)"
    });
    let warp_dstw_summary = json!({
        "warp": {
            "feasibility": "BLOCKED",
            "value": Value::Null,
            "blocker": "warp_matrix.bin DENSE_FIELD_CANONICAL_BYTES opaque; WarpSummary offline-only \
                        (warp_jacobian.rs:25-31)"
        },
        "dstw": artifact_summary(
            dir,
            "_adaptive_dt.bin",
            "FiberBundleDstw",
            "FiberBundleDstw + BocpdLog runtime-inline scalars",
        )
    });

    // Path-B site records: prefer the prismos-orchestrated runtime stage. Fall
    // back to legacy/direct binding_sites.json only as derived prior-stage data.
    let runtime_pathb = dir
        .join("pathb_runtime_stage")
        .join("binding_sites.materialized.json");
    let pathb_site_records = match runtime_pathb
        .exists()
        .then_some(runtime_pathb.clone())
        .or_else(|| find_artifact_recursive(dir, "binding_sites.materialized.json"))
        .or_else(|| find_artifact_recursive(dir, "binding_sites.json"))
    {
        Some(p) => json!({
            "feasibility": "POPULATABLE_NOW",
            "runtime_origin": if p.starts_with(dir.join("pathb_runtime_stage")) {
                RuntimeOrigin::RuntimeSubprocessOrchestratedByPrismos.as_str()
            } else {
                RuntimeOrigin::ExternalDerived.as_str()
            },
            "runtime_native": p.starts_with(dir.join("pathb_runtime_stage")),
            "artifact": p.display().to_string(),
            "sha256": sha256_of(&p).unwrap_or_default(),
            "note": if p.starts_with(dir.join("pathb_runtime_stage")) {
                "Path-B/V2 materialized by prismos md-run as a runtime subprocess stage"
            } else {
                "Path-B site records found as a derived/prior-stage artifact, not this runtime stage"
            }
        }),
        None => json!({
            "feasibility": "BLOCKED",
            "value": Value::Null,
            "blocker": "no runtime Path-B binding_sites.materialized.json in run output"
        }),
    };

    let known_site_overlap = json!({
        "feasibility": "NOT_AVAILABLE",
        "value": Value::Null,
        "absent_reason": "no curated known-site reference loaded in run_md scope; overlap is \
                          computed offline (P2Rank rerank / literature cross-ref), not fabricated here"
    });

    // unresolved evidence requirements = the BLOCKED post-MD tensors. Holo is
    // not an unresolved engine blocker here: valid holo runs use the GDP topology
    // as the engine input, and invalid holo requests are rejected before launch.
    let post_tensors = build_tensor_block(Stage::PostMd, dir, &inputs.topology_path);
    let mut unresolved = missing_preconditions_from(&post_tensors);
    let gdp_present = topology_gdp_present(&inputs.topology_path);
    let ligand_state = normalized_ligand_state(&inputs.ligand_mode, gdp_present);
    if ligand_state == "holo_missing_gdp" {
        unresolved.push(json!({
            "requirement": "GDP_CONTAINING_ENGINE_TOPOLOGY",
            "blocker": BlockedState::holo_topology_gdp_absent().to_value(),
            "unblock_action": "supply the GDP-parameterized topology as --topology; prismos md-run rejects intent-only holo execution"
        }));
    }

    ValidationStateRecord {
        schema_kind: DSTW_T4_VALIDATION_STATE_SCHEMA_KIND.to_string(),
        schema_version: DSTW_T4_SCHEMA_VERSION,
        target_id: inputs.target_id.clone(),
        ligand_state: inputs.ligand_mode.clone(),
        topology_hash,
        coordinate_hash,
        spike_event_count,
        kcc_summary,
        signal_summary,
        warp_dstw_summary,
        pathb_site_records,
        known_site_overlap,
        unresolved_evidence_requirements: unresolved,
        contradiction_graph_refs,
        // ValidationState is a prismos-inline consolidation of this run's evidence.
        runtime_origin: RuntimeOrigin::RuntimeInline.as_str().to_string(),
        runtime_native: RuntimeOrigin::RuntimeInline.is_prismos_native(),
    }
}

// ---- seal + decode-proof -----------------------------------------------------

/// Write `json` to `<output>/<basename>`, QNANL-seal it (SECTION_ZSTD body +
/// NANL-Q4 seal) via `encode_bytes`, then re-decode the packet from disk and
/// ASSERT byte-exact identity. Writes a `<file>.decode_proof.json` side-car.
/// Errors on any decode mismatch.
fn write_seal_and_prove(
    output_dir: &Path,
    basename: &str,
    json: &Value,
    runtime_origin: RuntimeOrigin,
) -> DstwResult<(PathBuf, PathBuf, bool)> {
    write_seal_and_prove_with_family(
        output_dir,
        basename,
        json,
        runtime_origin,
        QnanlArtifactFamily::JsonTelemetry,
    )
}

fn write_seal_and_prove_with_family(
    output_dir: &Path,
    basename: &str,
    json: &Value,
    runtime_origin: RuntimeOrigin,
    artifact_family: QnanlArtifactFamily,
) -> DstwResult<(PathBuf, PathBuf, bool)> {
    let json_path = output_dir.join(basename);
    let pretty = serde_json::to_string_pretty(json).map_err(|e| e.to_string())?;
    let raw = pretty.into_bytes();
    std::fs::write(&json_path, &raw).map_err(|e| e.to_string())?;

    let packet_path = PathBuf::from(format!("{}.prismqnanl", json_path.display()));
    let receipt_path = PathBuf::from(format!("{}.prismqnanl.receipt.json", json_path.display()));
    let receipt = encode_bytes(
        &raw,
        &packet_path,
        &receipt_path,
        QnanlEncodeOptions::new(
            basename.to_string(),
            artifact_family,
            Some(json_path.clone()),
            runtime_origin,
            "dstw_state_graph::write_seal_and_prove",
        ),
    )
    .map_err(|e| format!("qnanl seal failed for {basename}: {e}"))?;

    // Byte-exact decode proof (executed, not the embedded bool only).
    let packet_bytes = std::fs::read(&packet_path).map_err(|e| e.to_string())?;
    let decoded = decode_packet_bytes(&packet_bytes)
        .map_err(|e| format!("decode failed for {basename}: {e}"))?;
    let pass = decoded == raw;
    let proof_path = PathBuf::from(format!("{}.decode_proof.json", json_path.display()));
    let proof = json!({
        "schema_kind": "dstw_t4_decode_proof",
        "artifact": basename,
        "runtime_origin": runtime_origin.as_str(),
        "runtime_native": runtime_origin.is_prismos_native(),
        "packet_path": packet_path.display().to_string(),
        "raw_bytes": raw.len(),
        "decoded_bytes": decoded.len(),
        "raw_hash_blake3": receipt.raw_hash_blake3,
        "decode_byte_exact": pass
    });
    std::fs::write(
        &proof_path,
        serde_json::to_string_pretty(&proof).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;

    if !pass {
        return Err(format!(
            "DSTW-T4 decode proof FAILED (decoded != raw) for {basename}"
        ));
    }
    Ok((json_path, packet_path, pass))
}

// ---- operand records ---------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn push_vertex_record(
    ledger: &mut OperandLedger,
    inputs: &DstwStateGraphInputs,
    json_path: &Path,
    packet_path: &Path,
    input_artifact: Option<&Path>,
    origin: RuntimeOrigin,
    blocker: Option<&str>,
    parent_vid: Option<&str>,
) -> DstwResult<()> {
    let id = receipt_name(json_path);
    // DSTW-T4 Step h3: a vertex row targets a State; parent is the upstream
    // vertex id, child is the emitted vertex artifact (auto from the output path).
    let mut rec = OperandRecord::typed_from_artifacts(
        "STATE.VERTEX_CREATE",
        crate::binary_runtime::opcode_id_for_name("STATE.VERTEX_CREATE"),
        origin,
        input_artifact,
        Some(json_path),
        Some(packet_path),
        Some(id),
    )
    .map_err(|e| e.to_string())?
    .with_operand_target(crate::operand_ledger::OperandTarget::State)
    .with_state_lineage(parent_vid.map(str::to_string), None)
    .with_units("json_bytes")
    .with_run_context(
        None,
        Some(inputs.target_id.clone()),
        Some(inputs.ligand_mode.clone()),
        Some(inputs.detection_mode.clone()),
    );
    if let Some(reason) = blocker {
        rec = rec.with_blocker(reason);
    }
    ledger.push(rec);
    Ok(())
}

fn push_transition_records(
    ledger: &mut OperandLedger,
    inputs: &DstwStateGraphInputs,
    json_path: &Path,
    packet_path: &Path,
    source_json: Option<&Path>,
    operator_name: &str,
    origin: RuntimeOrigin,
    blocker: Option<&str>,
) -> DstwResult<()> {
    let id = receipt_name(json_path);
    // STATE.EMIT_TRANSITION — the edge-record emission.
    let mut emit = OperandRecord::typed_from_artifacts(
        "STATE.EMIT_TRANSITION",
        crate::binary_runtime::opcode_id_for_name("STATE.EMIT_TRANSITION"),
        origin,
        source_json,
        Some(json_path),
        Some(packet_path),
        Some(id.clone()),
    )
    .map_err(|e| e.to_string())?
    .with_run_context(
        None,
        Some(inputs.target_id.clone()),
        Some(inputs.ligand_mode.clone()),
        Some(inputs.detection_mode.clone()),
    );
    if let Some(reason) = blocker {
        emit = emit.with_blocker(reason);
    }
    ledger.push(emit);

    // STATE.OPERATOR_APPLY — the operator that drove the transition. Records the
    // operator name as coordinate_frame metadata (e.g. MD.INTEGRATE_VELOCITY_VERLET,
    // a real CUDA dispatch) so the edge is traceable to its operator.
    let mut apply = OperandRecord::typed_from_artifacts(
        "STATE.OPERATOR_APPLY",
        crate::binary_runtime::opcode_id_for_name("STATE.OPERATOR_APPLY"),
        origin,
        source_json,
        Some(json_path),
        Some(packet_path),
        Some(id),
    )
    .map_err(|e| e.to_string())?
    .with_coordinate_frame(format!("operator={operator_name}"));
    if let Some(reason) = blocker {
        apply = apply.with_blocker(reason);
    }
    ledger.push(apply);
    Ok(())
}

fn receipt_name(path: &Path) -> String {
    path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_string()
}

#[allow(clippy::too_many_arguments)]
fn build_t4_epoch_overlay(
    inputs: &DstwStateGraphInputs,
    pre_vid: &str,
    post_vid: &str,
    transition_id: &str,
    post_path: &Path,
    post_packet: &Path,
    emitted: &[DstwEmittedArtifact],
    ligand_state: &str,
) -> (String, String, Value) {
    let epoch_id = format!(
        "epoch_{}",
        &blake3_hex(&[&inputs.target_id, pre_vid, post_vid, transition_id])[..32]
    );
    let overlay_id = format!(
        "overlay_{}",
        &blake3_hex(&[&epoch_id, "dstw_t4_epoch_overlay"])[..32]
    );
    let start_event_id = format!(
        "cevt_{}",
        &blake3_hex(&[&epoch_id, pre_vid, "epoch_started"])[..32]
    );
    let end_event_id = format!(
        "cevt_{}",
        &blake3_hex(&[&epoch_id, post_vid, "epoch_closed"])[..32]
    );
    let qnanl_packet_ref = sha256_of(post_packet)
        .map(|sha| format!("packet_sha256:{sha}"))
        .unwrap_or_else(|| format!("packet_path:{}", post_packet.display()));
    let state_hash = sha256_of(post_path)
        .map(|sha| format!("sha256:{sha}"))
        .unwrap_or_else(|| format!("path:{}", post_path.display()));

    json!({
        "schema_kind": DSTW_T4_EPOCH_OVERLAY_SCHEMA_KIND,
        "schema_version": DSTW_T4_SCHEMA_VERSION,
        "overlay_id": overlay_id,
        "epoch_id": epoch_id,
        "target_id": inputs.target_id,
        "source_vertex_id": pre_vid,
        "target_vertex_id": post_vid,
        "transition_id": transition_id,
        "start_event_id": start_event_id,
        "end_event_id": end_event_id,
        "overlay_kind": "runtime_closeout",
        "epoch_kind": "md_runtime",
        "lifecycle_stage": "post_md",
        "ligand_state": ligand_state,
        "event_count": 3u64,
        "artifact_count": emitted.len() as u64,
        "timestep_min": 0u64,
        "timestep_max": 0u64,
        "physical_time_start_fs": 0.0,
        "physical_time_end_fs": 0.0,
        "physical_time_duration_fs": 0.0,
        "qnanl_packet_ref": qnanl_packet_ref,
        "operand_ref": receipt_name(post_path),
        "state_hash": state_hash,
        "runtime_origin": RuntimeOrigin::RuntimeInline.as_str(),
        "runtime_native": RuntimeOrigin::RuntimeInline.is_prismos_native(),
        "closed_interval": true
    })
    .as_object()
    .and_then(|obj| {
        let overlay_id = obj.get("overlay_id")?.as_str()?.to_string();
        let epoch_id = obj.get("epoch_id")?.as_str()?.to_string();
        Some((overlay_id, epoch_id, Value::Object(obj.clone())))
    })
    .expect("epoch overlay builder produced object")
}

fn build_chronology_acceptance_decision(
    inputs: &DstwStateGraphInputs,
    overlay_id: &str,
    epoch_id: &str,
    overlay_path: &Path,
    overlay_packet: &Path,
    emitted: &[DstwEmittedArtifact],
    operand_ledger_record_count: usize,
    all_decode_proofs_pass: bool,
) -> Value {
    let acceptance_decision_id = format!(
        "cacc_{}",
        &blake3_hex(&[overlay_id, epoch_id, "chronology_acceptance"])[..32]
    );
    let runtime_native_artifact_count = emitted
        .iter()
        .filter(|art| {
            RuntimeOrigin::from_label(&art.runtime_origin)
                .map(|origin| origin.is_prismos_native())
                .unwrap_or(false)
        })
        .count();
    let decode_proof_pass_count = emitted.iter().filter(|art| art.decode_proof_pass).count();
    let qnanl_packet_ref = sha256_of(overlay_packet)
        .map(|sha| format!("packet_sha256:{sha}"))
        .unwrap_or_else(|| format!("packet_path:{}", overlay_packet.display()));
    let state_hash = sha256_of(overlay_path)
        .map(|sha| format!("sha256:{sha}"))
        .unwrap_or_else(|| format!("path:{}", overlay_path.display()));
    let accepted = all_decode_proofs_pass;

    json!({
        "schema_kind": DSTW_T4_CHRONOLOGY_ACCEPTANCE_DECISION_SCHEMA_KIND,
        "schema_version": DSTW_T4_SCHEMA_VERSION,
        "acceptance_decision_id": acceptance_decision_id,
        "epoch_overlay_id": overlay_id,
        "epoch_id": epoch_id,
        "target_id": inputs.target_id,
        "decision": if accepted { "accepted" } else { "blocked" },
        "decision_basis": "decode_proofs_and_operand_ledger",
        "accepted": accepted,
        "all_decode_proofs_pass": all_decode_proofs_pass,
        "operand_ledger_record_count": operand_ledger_record_count as u64,
        "artifact_count": emitted.len() as u64,
        "qnanl_packet_count": emitted.len() as u64,
        "decode_proof_count": emitted.len() as u64,
        "decode_proof_pass_count": decode_proof_pass_count as u64,
        "runtime_native_artifact_count": runtime_native_artifact_count as u64,
        "post_run_artifact_count": emitted.len().saturating_sub(runtime_native_artifact_count) as u64,
        "blocked_gate_count": if accepted { 0u64 } else { 1u64 },
        "lifecycle_stage": "post_md",
        "runtime_origin": RuntimeOrigin::RuntimeInline.as_str(),
        "runtime_native": RuntimeOrigin::RuntimeInline.is_prismos_native(),
        "qnanl_packet_ref": qnanl_packet_ref,
        "operand_ref": receipt_name(overlay_path),
        "state_hash": state_hash
    })
}

// ---- DSTW-T4 Step h5: dense-tensor semantic summaries -------------------------
//
// Dense tensors must NOT enter training raw. For each dense engine tensor we emit
// one semantic record carrying the tensor's identity + reduced semantics so the
// model never sees raw bytes. Dimension/finite fields derive ONLY from a DECODED
// reducer summary (dispatch_analysis_common SignalGridSummary/WarpSummary, or a
// decoded JSON artifact) — never by re-reading raw tensor bytes (prismcode has no
// parquet/arrow reader by design). When no decoded summary exists those fields
// carry an explicit `*_absent_reason` (never null-padded).
//
// Runtime origin is caller-assigned before QNANL sealing. Existing-run demo paths
// pass POST_RUN_DERIVED; `prismos md-run` passes
// RUNTIME_SUBPROCESS_ORCHESTRATED_BY_PRISMOS while it owns the runtime pipeline.

pub const DSTW_T4_DENSE_TENSOR_SUMMARY_SCHEMA_KIND: &str = "dstw_t4_dense_tensor_semantic_summary";

/// One dense engine tensor's identity + decoded-summary probe shape. Suffixes are GROUNDED in the
/// engine writers cited per row (no guessing). `artifact_is_decoded_json` is true when the artifact
/// itself is already decoded JSON (TWIN16), so it doubles as the decoded summary source.
struct DenseTensorSpec {
    tensor_role: &'static str, // MUST equal a DENSE_TENSOR_ROLE label (routing.rs)
    modality: &'static str,    // MUST equal a DENSE_TENSOR_MODALITY label (routing.rs)
    source_kind: &'static str, // canonical engine artifact kind
    artifact_suffixes: &'static [&'static str],
    summary_suffixes: &'static [&'static str],
    artifact_is_decoded_json: bool,
    reducer_scalar_keys: &'static [&'static str],
    engine_writer: &'static str, // file:line citation
    reducer: &'static str,       // decoded-summary reducer provenance
}

/// The 6 dense tensors h5 covers. Order is the DENSE_TENSOR_ROLE enum order (so role enum codes are
/// 0..5 in the tokenizer).
const DENSE_TENSOR_SPECS: &[DenseTensorSpec] = &[
    DenseTensorSpec {
        tensor_role: "SIGNAL_GRID",
        modality: "thermodynamic_occupancy_grid",
        source_kind: "engine:_signal_grid.bin (PRSGD001)",
        artifact_suffixes: &["_signal_grid.bin", "signal_grid.bin"],
        summary_suffixes: &["_signal_grid_summary.json", "hysteresis_analysis.json", "pathway_analysis.json"],
        artifact_is_decoded_json: false,
        reducer_scalar_keys: &["hysteresis_score", "mean_abs_delta", "reversibility_mean", "warm_mean"],
        engine_writer: "nhs_rt_full.rs:1225,2273 (md_evidence_write_signal_grid nhs_rt_full.rs:2475)",
        reducer: "dispatch_analysis_common::read_signal_grid_summary (SignalGridSummary, dispatch_analysis_common/mod.rs:39-139)",
    },
    DenseTensorSpec {
        tensor_role: "WARP_FIELD",
        modality: "mechanical_warp_field",
        source_kind: "engine:_warp_matrix.bin (DenseGpuField 136B/voxel)",
        artifact_suffixes: &["_warp_matrix.bin", "warp_matrix.bin"],
        summary_suffixes: &["_warp_summary.json", "pathway_analysis.json"],
        artifact_is_decoded_json: false,
        reducer_scalar_keys: &["metric_mean", "metric_max"],
        engine_writer: "nhs_rt_full.rs:15402 ({stem}_stream{i}_warp_matrix.bin)",
        reducer: "dispatch_analysis_common::read_warp_summary (WarpSummary, dispatch_analysis_common/mod.rs:142-190)",
    },
    DenseTensorSpec {
        tensor_role: "DSTW_ADAPTIVE_DT",
        modality: "dstw_temporal",
        source_kind: "engine:_adaptive_dt.bin (DSTW per-stream adaptive timestep)",
        artifact_suffixes: &["_adaptive_dt.bin", "adaptive_dt.bin"],
        summary_suffixes: &["_adaptive_dt_summary.json", "_dstw_summary.json"],
        artifact_is_decoded_json: false,
        reducer_scalar_keys: &["adaptive_dt_mean", "dt_mean", "mean"],
        engine_writer: "nhs_rt_full.rs:10136 ({stem}_stream{i}_adaptive_dt.bin)",
        reducer: "DSTW adaptive-dt scalar reduction (dstw_dispatch/dtsg_parquet_emitter.rs) — NO native scalar materializer (offline-only)",
    },
    DenseTensorSpec {
        tensor_role: "SUPERRESOLUTION",
        modality: "superresolution_fused",
        source_kind: "external:prism_postmd_superresolution fused tensor (Path B)",
        artifact_suffixes: &["_superresolution_tensor.parquet", "_superresolution.parquet", "_superresolution_tensor.bin"],
        summary_suffixes: &["_superresolution_summary.json", "superresolution_summary.json"],
        artifact_is_decoded_json: false,
        reducer_scalar_keys: &["fused_scalar_mean", "tensor_mean", "mean"],
        engine_writer: "prismos.rs:6263 (prism_postmd_superresolution; scripts/build_prism_superresolution_postmd_tensor.py)",
        reducer: "post-MD fused superresolution tensor reduction — OFFLINE EXTERNAL (python tensor_fusion); not engine-native",
    },
    DenseTensorSpec {
        tensor_role: "TWIN_INTERFEROMETRIC",
        modality: "interferometric_twin",
        source_kind: "engine:TWIN16_INTERFEROMETRIC_TRANSITION_VIEW_RESIDUALS.json",
        artifact_suffixes: &["TWIN16_INTERFEROMETRIC_TRANSITION_VIEW_RESIDUALS.json"],
        summary_suffixes: &["bounded_thermo_temporal_ccns_sdst_summary.json"],
        artifact_is_decoded_json: true,
        reducer_scalar_keys: &["transition_view_l1_residual", "mean_residual", "l1_residual_mean"],
        engine_writer: "nhs_rt_full.rs:19128 (TWIN16 interferometric transition-view residuals)",
        reducer: "TWIN16 bounded thermo/temporal CCNS/SDST aggregate (nhs_rt_full.rs:20612) — artifact is already decoded JSON",
    },
    DenseTensorSpec {
        tensor_role: "NEUROMORPHIC_SENSOR",
        modality: "neuromorphic_sensor",
        source_kind: "engine:_spikes.bin (PRSPK001 Vec<GpuSpikeEvent>)",
        artifact_suffixes: &["_spikes.bin", "spikes.bin"],
        summary_suffixes: &["_spike_summary.json", "spike_events.json"],
        artifact_is_decoded_json: false,
        reducer_scalar_keys: &["spike_rate", "n_spikes", "spike_count", "mean_intensity"],
        engine_writer: "nhs_rt_full.rs:1688 ({stem}_stream{i}_spikes.bin; md_evidence_write_spikes nhs_rt_full.rs:1772)",
        reducer: "Vec<GpuSpikeEvent> count/role reduction — OFFLINE decode-before-tokenize (INV-1); no native scalar materializer",
    },
];

/// Parse the per-stream index from an engine artifact name (`..._stream07_...`). Returns None for
/// non-per-stream artifacts (e.g. the run-level TWIN16 json) — recorded with an absent_reason, never
/// null-padded.
fn parse_stream_id(name: &str) -> Option<u64> {
    let idx = name.find("stream")? + "stream".len();
    let digits: String = name[idx..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    if digits.is_empty() {
        return None;
    }
    digits.parse::<u64>().ok()
}

/// Recursive finite/NaN/Inf scan over a DECODED summary value (never raw bytes). serde_json Numbers
/// are always finite (JSON has no NaN/Inf); a non-finite reducer scalar therefore surfaces either as
/// a stringified "NaN"/"Infinity" (some emitters) or as `null` (serde f64 NaN→null). Both are
/// counted here so the finite gate is honest rather than vacuous.
#[derive(Default, Clone, Copy)]
struct FiniteScan {
    numbers: u64,
    nan: u64,
    inf: u64,
    nulls: u64,
}

fn finite_scan(v: &Value, acc: &mut FiniteScan) {
    match v {
        Value::Number(_) => acc.numbers += 1, // serde_json Number is always finite
        Value::String(s) => {
            let l = s.trim().to_ascii_lowercase();
            match l.as_str() {
                "nan" | "+nan" | "-nan" => acc.nan += 1,
                "inf" | "+inf" | "-inf" | "infinity" | "+infinity" | "-infinity" => acc.inf += 1,
                _ => {}
            }
        }
        Value::Null => acc.nulls += 1,
        Value::Array(a) => a.iter().for_each(|x| finite_scan(x, acc)),
        Value::Object(o) => o.values().for_each(|x| finite_scan(x, acc)),
        Value::Bool(_) => {}
    }
}

/// Probe a decoded summary for `row_count` (dimension) under common key aliases.
fn probe_row_count(summary: &Value) -> Option<u64> {
    for k in ["row_count", "n_rows", "rows", "voxel_count", "n_events"] {
        if let Some(n) = summary.get(k).and_then(Value::as_u64) {
            return Some(n);
        }
    }
    None
}

/// Probe a decoded summary for the column count (length of a `columns` array).
fn probe_column_count(summary: &Value) -> Option<u64> {
    for k in ["columns", "column_names", "fields"] {
        if let Some(arr) = summary.get(k).and_then(Value::as_array) {
            return Some(arr.len() as u64);
        }
    }
    summary.get("column_count").and_then(Value::as_u64)
}

/// Probe a decoded summary for a representative reduced scalar, by spec-ordered candidate keys then a
/// generic first-finite-top-level-number fallback.
fn probe_reducer_scalar(summary: &Value, keys: &[&str]) -> Option<f64> {
    for k in keys {
        if let Some(x) = summary.get(*k).and_then(Value::as_f64) {
            if x.is_finite() {
                return Some(x);
            }
        }
    }
    if let Some(obj) = summary.as_object() {
        for (_k, v) in obj {
            if let Some(x) = v.as_f64() {
                if x.is_finite() {
                    return Some(x);
                }
            }
        }
    }
    None
}

/// Build the array of dense-tensor semantic summary records for this run. Identity
/// fields are ALWAYS present (matched_fields>0 on the tokenizer side even when the artifact/summary
/// is absent); every absent data field carries a sibling `*_absent_reason` (never null-padded).
fn build_dense_tensor_semantic_summaries(
    output_dir: &Path,
    parent_vertex_id: &str,
    runtime_origin: RuntimeOrigin,
) -> Vec<Value> {
    let mut out = Vec::with_capacity(DENSE_TENSOR_SPECS.len());
    for spec in DENSE_TENSOR_SPECS {
        let mut rec = Map::new();
        rec.insert(
            "schema_kind".into(),
            json!(DSTW_T4_DENSE_TENSOR_SUMMARY_SCHEMA_KIND),
        );
        rec.insert("schema_version".into(), json!(DSTW_T4_SCHEMA_VERSION));
        // ---- identity (ALWAYS present) ----
        rec.insert("tensor_role".into(), json!(spec.tensor_role));
        rec.insert("modality".into(), json!(spec.modality));
        rec.insert("source_kind".into(), json!(spec.source_kind));
        rec.insert(
            "tokenizer_family".into(),
            json!("DenseTensorSemanticSummary"),
        );
        rec.insert("runtime_origin".into(), json!(runtime_origin.as_str()));
        rec.insert(
            "runtime_native".into(),
            json!(runtime_origin.is_prismos_native()),
        );
        rec.insert("parent_vertex_id".into(), json!(parent_vertex_id));

        // ---- locate the engine artifact ----
        let artifact = spec
            .artifact_suffixes
            .iter()
            .find_map(|suf| find_artifact(output_dir, suf));

        match &artifact {
            Some(path) => {
                let name = receipt_name(path);
                rec.insert("artifact_present".into(), json!(true));
                rec.insert("artifact_name".into(), json!(name));
                // source_hash = REAL sha256 (VALUE-verified), on_disk_bytes = REAL metadata len.
                match sha256_of(path) {
                    Some(sha) => {
                        rec.insert("source_hash".into(), json!(sha));
                    }
                    None => {
                        rec.insert("source_hash".into(), Value::Null);
                        rec.insert(
                            "source_hash_absent_reason".into(),
                            json!(
                                "sha256_file_hex failed to read the artifact (I/O); not fabricated"
                            ),
                        );
                    }
                }
                rec.insert("on_disk_bytes".into(), json!(file_bytes(path)));
                match parse_stream_id(&name) {
                    Some(sid) => {
                        rec.insert("stream_id".into(), json!(sid));
                    }
                    None => {
                        rec.insert("stream_id".into(), Value::Null);
                        rec.insert(
                            "stream_id_absent_reason".into(),
                            json!("artifact name carries no _streamNN_ index (run-level artifact, e.g. TWIN16 json)"),
                        );
                    }
                }
            }
            None => {
                rec.insert("artifact_present".into(), json!(false));
                rec.insert("source_hash".into(), Value::Null);
                rec.insert(
                    "source_hash_absent_reason".into(),
                    json!(format!(
                        "no engine artifact matching {:?} in this run's output dir; native runtime emission gated on MdNativeRuntimeState::write_receipt (md_native_runtime.rs:306-317)",
                        spec.artifact_suffixes
                    )),
                );
                rec.insert("on_disk_bytes".into(), Value::Null);
                rec.insert(
                    "on_disk_bytes_absent_reason".into(),
                    json!("artifact absent; no bytes to measure"),
                );
                rec.insert("stream_id".into(), Value::Null);
                rec.insert("stream_id_absent_reason".into(), json!("artifact absent"));
            }
        }

        // ---- decoded summary (ONLY through decoded summaries, never raw bytes) ----
        // For a JSON-native artifact (TWIN16) the artifact itself is the decoded summary; otherwise
        // we probe for a decoded reducer-summary sidecar. We NEVER parse a raw .bin here (prismcode
        // has no parquet/arrow reader: the SignalGridSummary/WarpSummary reducers run offline and
        // write a JSON summary that THIS path reads).
        let summary_source: Option<PathBuf> = if spec.artifact_is_decoded_json {
            artifact.clone()
        } else {
            spec.summary_suffixes
                .iter()
                .find_map(|suf| find_artifact(output_dir, suf))
        };
        let decoded: Option<Value> = summary_source
            .as_ref()
            .and_then(|p| std::fs::read(p).ok())
            .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok());

        rec.insert("reducer".into(), json!(spec.reducer));
        rec.insert("engine_writer".into(), json!(spec.engine_writer));

        match &decoded {
            Some(summary) => {
                rec.insert("summary_present".into(), json!(true));
                rec.insert(
                    "decoded_summary_artifact".into(),
                    json!(summary_source.as_ref().map(|p| receipt_name(p))),
                );
                // dimensions
                match probe_row_count(summary) {
                    Some(n) => {
                        rec.insert("row_count".into(), json!(n));
                    }
                    None => {
                        rec.insert("row_count".into(), Value::Null);
                        rec.insert(
                            "row_count_absent_reason".into(),
                            json!("decoded summary carries no row_count/n_rows/voxel_count key"),
                        );
                    }
                }
                match probe_column_count(summary) {
                    Some(c) => {
                        rec.insert("column_count".into(), json!(c));
                    }
                    None => {
                        rec.insert("column_count".into(), Value::Null);
                        rec.insert(
                            "column_count_absent_reason".into(),
                            json!("decoded summary carries no columns[]/column_count key"),
                        );
                    }
                }
                // finite / NaN / Inf scan over the DECODED summary
                let mut scan = FiniteScan::default();
                finite_scan(summary, &mut scan);
                let all_finite = scan.nan == 0 && scan.inf == 0;
                rec.insert("all_finite".into(), json!(all_finite));
                rec.insert("nan_count".into(), json!(scan.nan));
                rec.insert("inf_count".into(), json!(scan.inf));
                rec.insert(
                    "nonfinite_or_null_count".into(),
                    json!(scan.nan + scan.inf + scan.nulls),
                );
                rec.insert("finite_numbers_scanned".into(), json!(scan.numbers));
                // representative reduced scalar
                match probe_reducer_scalar(summary, spec.reducer_scalar_keys) {
                    Some(x) => {
                        rec.insert("reducer_scalar".into(), json!(x));
                    }
                    None => {
                        rec.insert("reducer_scalar".into(), Value::Null);
                        rec.insert(
                            "reducer_scalar_absent_reason".into(),
                            json!("decoded summary carries no finite reducer scalar under the candidate keys"),
                        );
                    }
                }
            }
            None => {
                rec.insert("summary_present".into(), json!(false));
                let reason = json!(format!(
                    "no decoded reducer summary {:?} present; dimension/finite fields are derived ONLY from a decoded summary (never from raw {} bytes — prismcode has no parquet/arrow reader; the {} reducer runs offline). RECOMMENDED OPERATIONAL IMPROVEMENT: native in-graph reduction at teardown (engine-owner).",
                    spec.summary_suffixes, spec.tensor_role, spec.reducer
                ));
                for f in [
                    "row_count",
                    "column_count",
                    "all_finite",
                    "nan_count",
                    "inf_count",
                    "nonfinite_or_null_count",
                    "reducer_scalar",
                ] {
                    rec.insert(f.into(), Value::Null);
                }
                rec.insert("decoded_summary_absent_reason".into(), reason);
            }
        }
        out.push(Value::Object(rec));
    }
    out
}

// ---- orchestration -----------------------------------------------------------

/// Emit the full DSTW-T4 state graph for one prismos `md-run`.
pub fn emit_dstw_t4_state_graph(inputs: &DstwStateGraphInputs) -> DstwResult<DstwStateGraphReport> {
    std::fs::create_dir_all(&inputs.output_dir).map_err(|e| e.to_string())?;
    let mut emitted: Vec<DstwEmittedArtifact> = Vec::new();
    let mut ledger = OperandLedger::new();
    let mut all_pass = true;

    let topology_sha = sha256_of(&inputs.topology_path);
    let topology_has_gdp = topology_gdp_present(&inputs.topology_path);
    let ligand_state = normalized_ligand_state(&inputs.ligand_mode, topology_has_gdp);
    let ligand_state_feasibility = ligand_feasibility(ligand_state);
    let ligand_state_blocker = ligand_blocker(ligand_state);

    // ---- V1: pre-MD topology-grounded vertex -----------------------------------
    let (pre_vid, pre_json, _pre_tensors) = build_vertex(
        inputs,
        Stage::PreMd,
        ligand_state,
        ligand_state_feasibility,
        topology_has_gdp,
        topology_sha.clone(),
        ligand_state_blocker,
        RuntimeOrigin::RuntimeSubprocessOrchestratedByPrismos,
    );
    let (pre_path, pre_packet, pass) = write_seal_and_prove(
        &inputs.output_dir,
        "pre_md_state_vertex.json",
        &pre_json,
        RuntimeOrigin::RuntimeSubprocessOrchestratedByPrismos,
    )?;
    all_pass &= pass;
    push_vertex_record(
        &mut ledger,
        inputs,
        &pre_path,
        &pre_packet,
        Some(&inputs.topology_path),
        RuntimeOrigin::RuntimeSubprocessOrchestratedByPrismos,
        None,
        None, // pre-MD vertex: parent is the topology input, not a vertex.
    )?;
    emitted.push(DstwEmittedArtifact {
        kind: "pre_md_state_vertex".to_string(),
        json_path: pre_path.clone(),
        qnanl_packet_path: pre_packet,
        decode_proof_pass: pass,
        runtime_origin: RuntimeOrigin::RuntimeSubprocessOrchestratedByPrismos
            .as_str()
            .to_string(),
    });

    // ---- V2: post-MD topology-grounded vertex ----------------------------------
    let (post_vid, post_json, post_tensors) = build_vertex(
        inputs,
        Stage::PostMd,
        ligand_state,
        ligand_state_feasibility,
        topology_has_gdp,
        topology_sha.clone(),
        ligand_state_blocker,
        RuntimeOrigin::RuntimeSubprocessOrchestratedByPrismos,
    );
    let (post_path, post_packet, pass) = write_seal_and_prove(
        &inputs.output_dir,
        "post_md_state_vertex.json",
        &post_json,
        RuntimeOrigin::RuntimeSubprocessOrchestratedByPrismos,
    )?;
    all_pass &= pass;
    push_vertex_record(
        &mut ledger,
        inputs,
        &post_path,
        &post_packet,
        Some(&inputs.topology_path),
        RuntimeOrigin::RuntimeSubprocessOrchestratedByPrismos,
        None,
        Some(pre_vid.as_str()), // post-MD vertex's parent is the pre-MD vertex.
    )?;
    emitted.push(DstwEmittedArtifact {
        kind: "post_md_state_vertex".to_string(),
        json_path: post_path.clone(),
        qnanl_packet_path: post_packet.clone(),
        decode_proof_pass: pass,
        runtime_origin: RuntimeOrigin::RuntimeSubprocessOrchestratedByPrismos
            .as_str()
            .to_string(),
    });

    // ---- T1: MD integration transition pre -> post ------------------------------
    let md_mode = json!({
        "feasibility": "POPULATABLE_NOW",
        "mode": {
            "detection_mode": inputs.detection_mode,
            "ligand_mode": inputs.ligand_mode,
            "ligand_state": ligand_state,
            "gdp_present": topology_has_gdp,
            "ligand_engine_effect": ligand_engine_effect(ligand_state)
        },
        "blocker": Value::Null
    });
    let (t1_id, t1_json) = build_transition(
        inputs,
        &pre_vid,
        &post_vid,
        "MD.INTEGRATE_VELOCITY_VERLET",
        md_mode,
        RuntimeOrigin::RuntimeSubprocessOrchestratedByPrismos,
        vec![
            "coordinate_tensor",
            "kcc_tensor",
            "spike_sensor_tensor",
            "epistemic_tensor",
        ],
        vec!["topology_tensor"],
        &post_tensors,
        "engine-owner: wire MdNativeRuntimeState::write_receipt + flip DEFERRED readout opcodes (md_native_runtime.rs:306-317) for native force/signal/warp/kinetic/thermo tensor emission",
        // h4 falsifiable gate: pre-baseline = selected topology hash; the MD edge is
        // confirmed ONLY if the post-MD evidence artifacts are observed.
        &topology_sha.clone().unwrap_or_default(),
        &[
            (
                "coordinate_tensor",
                "post-MD net displacement materializes (kcc_v2full.bin net_dx/dy/dz), distinct from the pre-integration topology",
                true,
                "_kcc_v2full.bin",
            ),
            (
                "kcc_tensor",
                "KCC descriptors PRKCC001 emitted at teardown (nhs_rt_full.rs:2530)",
                true,
                "_kcc_v2full.bin",
            ),
            (
                "spike_sensor_tensor",
                "always-on spike sensor emits spikes.bin PRSPK001 (nhs_rt_full.rs:1772)",
                false,
                "_spikes.bin",
            ),
            (
                "epistemic_tensor",
                "DSTW/BOCPD epistemic scalars accumulate (adaptive_dt.bin)",
                false,
                "_adaptive_dt.bin",
            ),
        ],
        None,
    );
    let (t1_path, t1_packet, pass) = write_seal_and_prove(
        &inputs.output_dir,
        "md_state_transition.json",
        &t1_json,
        RuntimeOrigin::RuntimeSubprocessOrchestratedByPrismos,
    )?;
    all_pass &= pass;
    push_transition_records(
        &mut ledger,
        inputs,
        &t1_path,
        &t1_packet,
        Some(&post_path),
        "MD.INTEGRATE_VELOCITY_VERLET",
        RuntimeOrigin::RuntimeSubprocessOrchestratedByPrismos,
        None,
    )?;
    emitted.push(DstwEmittedArtifact {
        kind: "md_state_transition".to_string(),
        json_path: t1_path,
        qnanl_packet_path: t1_packet,
        decode_proof_pass: pass,
        runtime_origin: RuntimeOrigin::RuntimeSubprocessOrchestratedByPrismos
            .as_str()
            .to_string(),
    });

    // ---- next_operator_recommendation.json (STATE.NEXT_OPERATOR_REC) ------------
    let recommendation = json!({
        "schema_kind": DSTW_T4_NEXT_OPERATOR_SCHEMA_KIND,
        "schema_version": DSTW_T4_SCHEMA_VERSION,
        "runtime_origin": RuntimeOrigin::RuntimeInline.as_str(),
        "runtime_native": RuntimeOrigin::RuntimeInline.is_prismos_native(),
        "priority_ordered_recommendations": [
            {
                "rank": 1,
                "operator": "ENGINE.NATIVE_TENSOR_EMISSION",
                "owner": "engine",
                "rationale": "wire MdNativeRuntimeState::write_receipt + native semantic reducers so force/signal_grid/warp/kinetic/thermodynamic tensors are emitted as value-verified runtime summaries, not post-run decoded summaries",
                "evidence": "md_native_runtime.rs runtime receipt path; prismos md-run now directly launches the selected apo/holo topology"
            },
            {
                "rank": 2,
                "operator": "ENGINE.IN_MD_LOOP_QNANL_STREAM",
                "owner": "engine",
                "rationale": "thread RuntimeQnanlStreamWriter into the per-stream MD closure so chunks seal DURING stepping (gate 11); today SpikeEventStreamWriter runs post-scope-join",
                "evidence": "nhs_rt_full.rs:18358-18381"
            },
            {
                "rank": 3,
                "operator": "ENGINE.TYPED_OPERAND_VECTOR",
                "owner": "engine",
                "rationale": "replace MdNativeKernelDispatch.opcodes Vec<String> with a per-dispatch typed operand vector so per-OPERAND refs become populatable (today only opcode-COUNT)",
                "evidence": "md_native_runtime.rs:72"
            }
        ]
    });
    let (rec_path, rec_packet, pass) = write_seal_and_prove(
        &inputs.output_dir,
        "next_operator_recommendation.json",
        &recommendation,
        RuntimeOrigin::RuntimeInline,
    )?;
    all_pass &= pass;
    let next_rec = OperandRecord::typed_from_artifacts(
        "STATE.NEXT_OPERATOR_REC",
        crate::binary_runtime::opcode_id_for_name("STATE.NEXT_OPERATOR_REC"),
        RuntimeOrigin::RuntimeInline,
        None,
        Some(&rec_path),
        Some(&rec_packet),
        Some(receipt_name(&rec_path)),
    )
    .map_err(|e| e.to_string())?;
    ledger.push(next_rec);
    emitted.push(DstwEmittedArtifact {
        kind: "next_operator_recommendation".to_string(),
        json_path: rec_path,
        qnanl_packet_path: rec_packet,
        decode_proof_pass: pass,
        runtime_origin: RuntimeOrigin::RuntimeInline.as_str().to_string(),
    });

    // ---- VS: ValidationState consolidation record (STATE.VALIDATION_SNAPSHOT) ---
    // Collect contradiction graph references discovered across falsifiable
    // transitions, annotated with their source. No synthetic GDP-blocker edge is
    // emitted for valid holo runs: the selected topology already drives the MD.
    let mut contradiction_graph_refs: Vec<Value> = Vec::new();
    if let Some(arr) = t1_json.get("contradiction_refs").and_then(Value::as_array) {
        for c in arr {
            let mut annotated = c.clone();
            if let Value::Object(map) = &mut annotated {
                map.insert(
                    "source_transition".to_string(),
                    json!("md_state_transition"),
                );
            }
            contradiction_graph_refs.push(annotated);
        }
    }
    // ---- DSTW-T4 Step h5: dense-tensor semantic summaries -----------------------
    // Dense tensors must NOT enter training raw — emit one semantic record per dense engine tensor.
    // Sealed + decode-proven like every other artifact; one STATE.DENSE_TENSOR_SUMMARY operand row.
    let dense_origin = inputs.dense_tensor_runtime_origin;
    let dense_summaries =
        build_dense_tensor_semantic_summaries(&inputs.output_dir, &post_vid, dense_origin);
    let dense_summary_count = dense_summaries.len();
    let dense_summaries_json = json!({
        "schema_kind": DSTW_T4_DENSE_TENSOR_SUMMARY_SCHEMA_KIND,
        "schema_version": DSTW_T4_SCHEMA_VERSION,
        "parent_vertex_id": post_vid,
        "runtime_origin": dense_origin.as_str(),
        "runtime_native": dense_origin.is_prismos_native(),
        "tokenizer_family": "DenseTensorSemanticSummary",
        "policy": "dense tensors must NOT enter training raw; dimension/finite fields derive ONLY from decoded summaries (never raw bytes); runtime origin is assigned by the caller before QNANL sealing",
        "summaries": dense_summaries
    });
    let (dts_path, dts_packet, pass) = write_seal_and_prove(
        &inputs.output_dir,
        "dense_tensor_semantic_summaries.json",
        &dense_summaries_json,
        dense_origin,
    )?;
    all_pass &= pass;
    let dts_rec = OperandRecord::typed_from_artifacts(
        "STATE.DENSE_TENSOR_SUMMARY",
        crate::binary_runtime::opcode_id_for_name("STATE.DENSE_TENSOR_SUMMARY"),
        dense_origin,
        Some(&post_path),
        Some(&dts_path),
        Some(&dts_packet),
        Some(receipt_name(&dts_path)),
    )
    .map_err(|e| e.to_string())?
    .with_operand_target(OperandTarget::Receipt)
    .with_state_lineage(Some(post_vid.clone()), None)
    .with_units("json_bytes")
    .with_run_context(
        None,
        Some(inputs.target_id.clone()),
        Some(inputs.ligand_mode.clone()),
        Some(inputs.detection_mode.clone()),
    );
    ledger.push(dts_rec);
    emitted.push(DstwEmittedArtifact {
        kind: "dense_tensor_semantic_summaries".to_string(),
        json_path: dts_path,
        qnanl_packet_path: dts_packet,
        decode_proof_pass: pass,
        runtime_origin: dense_origin.as_str().to_string(),
    });

    let validation_state =
        build_validation_state(inputs, topology_sha.clone(), contradiction_graph_refs);
    let validation_state_json =
        serde_json::to_value(&validation_state).map_err(|e| e.to_string())?;
    let (vs_path, vs_packet, pass) = write_seal_and_prove(
        &inputs.output_dir,
        "validation_state.json",
        &validation_state_json,
        RuntimeOrigin::RuntimeInline,
    )?;
    all_pass &= pass;
    let vs_rec = OperandRecord::typed_from_artifacts(
        "VALIDATION.STATE_SNAPSHOT",
        crate::binary_runtime::opcode_id_for_name("VALIDATION.STATE_SNAPSHOT"),
        RuntimeOrigin::RuntimeInline,
        Some(&post_path),
        Some(&vs_path),
        Some(&vs_packet),
        Some(receipt_name(&vs_path)),
    )
    .map_err(|e| e.to_string())?
    .with_operand_target(OperandTarget::Receipt)
    .with_state_lineage(Some(post_vid.clone()), None)
    .with_run_context(
        None,
        Some(inputs.target_id.clone()),
        Some(inputs.ligand_mode.clone()),
        Some(inputs.detection_mode.clone()),
    );
    ledger.push(vs_rec);
    emitted.push(DstwEmittedArtifact {
        kind: "validation_state".to_string(),
        json_path: vs_path,
        qnanl_packet_path: vs_packet,
        decode_proof_pass: pass,
        runtime_origin: RuntimeOrigin::RuntimeInline.as_str().to_string(),
    });

    // ---- conditional: PathB / tokenizer vertices (only if upstream exists) ------
    emit_downstream_if_present(
        inputs,
        &post_vid,
        &post_path,
        &mut ledger,
        &mut emitted,
        &mut all_pass,
    )?;

    // ---- chronology closeout: T4 epoch overlay + acceptance decision ------------
    let (overlay_id, epoch_id, epoch_overlay) = build_t4_epoch_overlay(
        inputs,
        &pre_vid,
        &post_vid,
        &t1_id,
        &post_path,
        &post_packet,
        &emitted,
        ligand_state,
    );
    let (overlay_path, overlay_packet, pass) = write_seal_and_prove_with_family(
        &inputs.output_dir,
        "dstw_t4_epoch_overlay.json",
        &epoch_overlay,
        RuntimeOrigin::RuntimeInline,
        QnanlArtifactFamily::T4EpochOverlay,
    )?;
    all_pass &= pass;
    let overlay_rec = OperandRecord::typed_from_artifacts(
        "STATE.EMIT_TRANSITION",
        crate::binary_runtime::opcode_id_for_name("STATE.EMIT_TRANSITION"),
        RuntimeOrigin::RuntimeInline,
        Some(&post_path),
        Some(&overlay_path),
        Some(&overlay_packet),
        Some(receipt_name(&overlay_path)),
    )
    .map_err(|e| e.to_string())?
    .with_operand_target(OperandTarget::Receipt)
    .with_state_lineage(Some(post_vid.clone()), None)
    .with_units("json_bytes")
    .with_run_context(
        None,
        Some(inputs.target_id.clone()),
        Some(inputs.ligand_mode.clone()),
        Some(inputs.detection_mode.clone()),
    );
    ledger.push(overlay_rec);
    emitted.push(DstwEmittedArtifact {
        kind: "dstw_t4_epoch_overlay".to_string(),
        json_path: overlay_path.clone(),
        qnanl_packet_path: overlay_packet.clone(),
        decode_proof_pass: pass,
        runtime_origin: RuntimeOrigin::RuntimeInline.as_str().to_string(),
    });

    let chronology_acceptance = build_chronology_acceptance_decision(
        inputs,
        &overlay_id,
        &epoch_id,
        &overlay_path,
        &overlay_packet,
        &emitted,
        ledger.len(),
        all_pass,
    );
    let (accept_path, accept_packet, pass) = write_seal_and_prove_with_family(
        &inputs.output_dir,
        "chronology_acceptance_decision.json",
        &chronology_acceptance,
        RuntimeOrigin::RuntimeInline,
        QnanlArtifactFamily::ChronologyAcceptanceDecision,
    )?;
    all_pass &= pass;
    let accept_rec = OperandRecord::typed_from_artifacts(
        "VALIDATION.STATE_SNAPSHOT",
        crate::binary_runtime::opcode_id_for_name("VALIDATION.STATE_SNAPSHOT"),
        RuntimeOrigin::RuntimeInline,
        Some(&overlay_path),
        Some(&accept_path),
        Some(&accept_packet),
        Some(receipt_name(&accept_path)),
    )
    .map_err(|e| e.to_string())?
    .with_operand_target(OperandTarget::Receipt)
    .with_state_lineage(Some(post_vid.clone()), None)
    .with_units("json_bytes")
    .with_run_context(
        None,
        Some(inputs.target_id.clone()),
        Some(inputs.ligand_mode.clone()),
        Some(inputs.detection_mode.clone()),
    );
    ledger.push(accept_rec);
    emitted.push(DstwEmittedArtifact {
        kind: "chronology_acceptance_decision".to_string(),
        json_path: accept_path,
        qnanl_packet_path: accept_packet,
        decode_proof_pass: pass,
        runtime_origin: RuntimeOrigin::RuntimeInline.as_str().to_string(),
    });

    // ---- operand ledger TSV -----------------------------------------------------
    let ledger_path = inputs.output_dir.join("dstw_t4_operand_ledger.tsv");
    ledger.write_tsv(&ledger_path).map_err(|e| e.to_string())?;
    // DSTW-T4 Step h3: coverage ledger aggregated by (operator_class,
    // operand_target). DSTW rows auto-classify (STATE.VERTEX_* -> State,
    // STATE.*TRANSITION/CLASSIFY/NEXT_OPERATOR -> Transition).
    let coverage_path = inputs.output_dir.join("OPCODE_OPERAND_COVERAGE_LEDGER.tsv");
    ledger
        .write_coverage_tsv(&coverage_path)
        .map_err(|e| e.to_string())?;

    // ---- DSTW-T4 Step h6: PRISM-native merge manifest + state graph -------------
    // Build the nine merge-readiness acceptance artifacts from the artifacts just
    // emitted + this run's typed ledger + a PROCESS-PROVEN provenance captured from
    // the EMITTING process (run_pid + packaged-binary sha256 — never a bare label).
    let provenance = crate::runtime_provenance::RuntimeProvenance::capture(
        &inputs.output_dir,
        inputs.command_argv.clone(),
    );
    let manifest = crate::merge_manifest::emit_prism_native_merge_manifest(
        &emitted,
        &ledger,
        &provenance,
        &inputs.target_id,
        &inputs.output_dir,
    )?;

    Ok(DstwStateGraphReport {
        operand_record_count: ledger.len(),
        emitted,
        operand_ledger_path: ledger_path,
        all_decode_proofs_pass: all_pass,
        dense_tensor_summary_count: dense_summary_count,
        merge_manifest_artifact_count: manifest.artifact_count,
        merge_manifest_orphan_count: manifest.orphan_count(),
        merge_manifest_training_ready_native_count: manifest.training_ready_native_count,
        merge_manifest_native_candidate_blocked_count: manifest.native_candidate_blocked_count,
        merge_manifest_derived_or_post_run_count: manifest.derived_or_post_run_count,
        merge_manifest_paths: manifest.output_paths,
    })
}

/// Emit a downstream vertex + transition for PathB or tokenizer ONLY when the
/// upstream artifact is present. Runtime child-stage artifacts are native
/// (`RUNTIME_SUBPROCESS_ORCHESTRATED_BY_PRISMOS`); legacy prior-stage artifacts
/// remain `EXTERNAL_DERIVED`.
fn emit_downstream_if_present(
    inputs: &DstwStateGraphInputs,
    post_vid: &str,
    post_path: &Path,
    ledger: &mut OperandLedger,
    emitted: &mut Vec<DstwEmittedArtifact>,
    all_pass: &mut bool,
) -> DstwResult<()> {
    let downstream: &[(&str, &str, &str, &str, &str)] = &[
        (
            "binding_sites",
            "post_pathb_state_vertex.json",
            "post_pathb_state_transition.json",
            "PATHB.MATERIALIZE_BINDING_SITE",
            "BindingSiteMaterialization",
        ),
        (
            "tokenizer_runtime_manifest",
            "post_tokenizer_state_vertex.json",
            "post_tokenizer_state_transition.json",
            "CSE.TOKENIZE",
            "TokenizerProjection",
        ),
    ];
    for (probe, vertex_name, transition_name, operator, family) in downstream {
        let probe_path = match *probe {
            "binding_sites" => inputs
                .output_dir
                .join("pathb_runtime_stage")
                .join("binding_sites.materialized.json")
                .exists()
                .then(|| {
                    inputs
                        .output_dir
                        .join("pathb_runtime_stage")
                        .join("binding_sites.materialized.json")
                })
                .or_else(|| {
                    find_artifact_recursive(&inputs.output_dir, "binding_sites.materialized.json")
                })
                .or_else(|| find_artifact_recursive(&inputs.output_dir, "binding_sites.json")),
            "tokenizer_runtime_manifest" => inputs
                .output_dir
                .join("prismos_runtime_tokenizer_export_receipt.json")
                .exists()
                .then(|| {
                    inputs
                        .output_dir
                        .join("prismos_runtime_tokenizer_export_receipt.json")
                })
                .or_else(|| {
                    inputs
                        .output_dir
                        .join("tokenizer_runtime_export_manifest.json")
                        .exists()
                        .then(|| {
                            inputs
                                .output_dir
                                .join("tokenizer_runtime_export_manifest.json")
                        })
                })
                .or_else(|| {
                    inputs
                        .output_dir
                        .join("tokenizer_runtime_corpus")
                        .join("corpus_card.json")
                        .exists()
                        .then(|| {
                            inputs
                                .output_dir
                                .join("tokenizer_runtime_corpus")
                                .join("corpus_card.json")
                        })
                }),
            _ => find_artifact_recursive(&inputs.output_dir, &format!("{probe}.json")),
        };
        let Some(probe_path) = probe_path else {
            continue;
        };
        let runtime_origin = if probe_path
            .starts_with(inputs.output_dir.join("pathb_runtime_stage"))
            || probe_path.file_name().and_then(|name| name.to_str())
                == Some("prismos_runtime_tokenizer_export_receipt.json")
        {
            RuntimeOrigin::RuntimeSubprocessOrchestratedByPrismos
        } else {
            RuntimeOrigin::ExternalDerived
        };
        let probe_sha = sha256_of(&probe_path).unwrap_or_default();
        let vid = vertex_id(&inputs.target_id, probe, &inputs.detection_mode, &probe_sha);
        let vertex = json!({
            "schema_kind": DSTW_T4_VERTEX_SCHEMA_KIND,
            "schema_version": DSTW_T4_SCHEMA_VERSION,
            "lifecycle_stage": format!("post_{probe}"),
            "vertex_id": vid,
            "target_id": inputs.target_id,
            "stage_family": family,
            "source_artifact": probe_path.display().to_string(),
            "source_artifact_sha256": probe_sha,
            "runtime_origin": runtime_origin.as_str(),
            "runtime_native": runtime_origin.is_prismos_native(),
            "note": if runtime_origin == RuntimeOrigin::RuntimeSubprocessOrchestratedByPrismos {
                "downstream stage was invoked by prismos md-run and is part of the runtime lineage"
            } else {
                "downstream stage is a prior/derived artifact, not this run's native runtime emission"
            }
        });
        let (v_path, v_packet, pass) =
            write_seal_and_prove(&inputs.output_dir, vertex_name, &vertex, runtime_origin)?;
        *all_pass &= pass;
        push_vertex_record(
            ledger,
            inputs,
            &v_path,
            &v_packet,
            Some(&probe_path),
            runtime_origin,
            if runtime_origin == RuntimeOrigin::RuntimeSubprocessOrchestratedByPrismos {
                None
            } else {
                Some("downstream stage read-only (EXTERNAL_DERIVED); not run_md-orchestrated")
            },
            Some(post_vid),
        )?;
        emitted.push(DstwEmittedArtifact {
            kind: format!("post_{probe}_state_vertex"),
            json_path: v_path.clone(),
            qnanl_packet_path: v_packet.clone(),
            decode_proof_pass: pass,
            runtime_origin: runtime_origin.as_str().to_string(),
        });

        let transition = json!({
            "schema_kind": DSTW_T4_TRANSITION_SCHEMA_KIND,
            "schema_version": DSTW_T4_SCHEMA_VERSION,
            "transition_id": transition_id(post_vid, operator, &inputs.operator_version, &inputs.command_argv),
            "source_vertex_id": post_vid,
            "target_vertex_id": vid,
            "operator_name": operator,
            "operator_version": inputs.operator_version,
            "runtime_origin": runtime_origin.as_str(),
            "runtime_native": runtime_origin.is_prismos_native(),
            "input_artifacts": [{"path": post_path.display().to_string(), "sha256": sha256_of(post_path)}],
            "output_artifacts": [{"path": probe_path.display().to_string(), "sha256": probe_sha}],
            "note": if runtime_origin == RuntimeOrigin::RuntimeSubprocessOrchestratedByPrismos {
                "runtime child-stage transition owned by prismos md-run"
            } else {
                "EXTERNAL_DERIVED edge to a prior-stage artifact; not run_md-orchestrated runtime emission"
            }
        });
        let (t_path, t_packet, pass) = write_seal_and_prove(
            &inputs.output_dir,
            transition_name,
            &transition,
            runtime_origin,
        )?;
        *all_pass &= pass;
        push_transition_records(
            ledger,
            inputs,
            &t_path,
            &t_packet,
            Some(post_path),
            operator,
            runtime_origin,
            if runtime_origin == RuntimeOrigin::RuntimeSubprocessOrchestratedByPrismos {
                None
            } else {
                Some("downstream stage read-only (EXTERNAL_DERIVED); not run_md-orchestrated")
            },
        )?;
        emitted.push(DstwEmittedArtifact {
            kind: format!("post_{probe}_state_transition"),
            json_path: t_path,
            qnanl_packet_path: t_packet,
            decode_proof_pass: pass,
            runtime_origin: runtime_origin.as_str().to_string(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("dstw_state_graph_{tag}_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_topology(dir: &Path) -> PathBuf {
        let topo = dir.join("4lpk_clean.topology.json");
        std::fs::write(&topo, br#"{"n_atoms":2467,"residue_names":["GLY","ALA"]}"#).unwrap();
        topo
    }

    fn base_inputs(dir: &Path, topo: &Path) -> DstwStateGraphInputs {
        DstwStateGraphInputs {
            output_dir: dir.to_path_buf(),
            topology_path: topo.to_path_buf(),
            target_id: "4LPK:KRAS_WT".to_string(),
            ligand_mode: "apo".to_string(),
            detection_mode: "mixed".to_string(),
            site_ranker: "phase-manifold".to_string(),
            md_only_evidence: true,
            engine_path: PathBuf::from("nhs_rt_full"),
            operator_version: "b498a5f7".to_string(),
            command_argv: vec!["--md-only-evidence".to_string()],
            holo_topology: None,
            opcode_counts: BTreeMap::from([("MD.INTEGRATE_VELOCITY_VERLET".to_string(), 6u64)]),
            dense_tensor_runtime_origin: RuntimeOrigin::PostRunDerived,
        }
    }

    #[test]
    fn emits_core_graph_with_byte_exact_decode_proofs() {
        let dir = temp_dir("core");
        let topo = write_topology(&dir);
        // a present engine artifact -> spike tensor POPULATABLE at post_md.
        std::fs::write(dir.join("run_spikes.bin"), b"\x00\x01\x02\x03").unwrap();
        let inputs = base_inputs(&dir, &topo);

        let report = emit_dstw_t4_state_graph(&inputs).unwrap();
        assert!(report.all_decode_proofs_pass, "all decode proofs must pass");
        // Core artifacts (no downstream present): pre, post, MD transition,
        // next-op, dense summaries, validation, and chronology closeout. No synthetic GDP-blocker rows.
        let kinds: Vec<&str> = report.emitted.iter().map(|e| e.kind.as_str()).collect();
        assert!(kinds.contains(&"pre_md_state_vertex"));
        assert!(kinds.contains(&"post_md_state_vertex"));
        assert!(kinds.contains(&"md_state_transition"));
        assert!(kinds.contains(&"next_operator_recommendation"));
        assert!(kinds.contains(&"dense_tensor_semantic_summaries"));
        assert!(kinds.contains(&"validation_state"));
        assert!(kinds.contains(&"dstw_t4_epoch_overlay"));
        assert!(kinds.contains(&"chronology_acceptance_decision"));
        assert!(!kinds.contains(&"blocked_gdp_bound_state_vertex"));
        assert!(!kinds.contains(&"gdp_forcefield_precondition_transition"));

        // every emitted JSON has a sibling .prismqnanl + .decode_proof.json that
        // re-decodes byte-exactly.
        for art in &report.emitted {
            assert!(art.json_path.exists(), "{:?} json missing", art.kind);
            assert!(
                art.qnanl_packet_path.exists(),
                "{:?} packet missing",
                art.kind
            );
            let raw = std::fs::read(&art.json_path).unwrap();
            let packet = std::fs::read(&art.qnanl_packet_path).unwrap();
            let decoded = decode_packet_bytes(&packet).unwrap();
            assert_eq!(decoded, raw, "{:?} decode not byte-exact", art.kind);
            let proof = PathBuf::from(format!("{}.decode_proof.json", art.json_path.display()));
            assert!(
                proof.exists(),
                "{:?} decode_proof side-car missing",
                art.kind
            );
        }

        // post_md spike tensor is POPULATABLE (artifact present) and VALUE-grounded.
        let post: Value =
            serde_json::from_slice(&std::fs::read(dir.join("post_md_state_vertex.json")).unwrap())
                .unwrap();
        let spike = &post["spike_sensor_tensor"];
        assert_eq!(spike["feasibility"], json!("POPULATABLE_NOW"));
        assert!(spike["value"]["sha256"].as_str().unwrap().len() == 64);
        // BLOCKED tensor carries null value + named blocker (never null-padded populated).
        let warp = &post["warp_tensor"];
        assert_eq!(warp["feasibility"], json!("BLOCKED"));
        assert!(warp["value"].is_null());
        assert!(!warp["blocker"].as_str().unwrap().is_empty());

        let overlay_receipt: Value = serde_json::from_slice(
            &std::fs::read(dir.join("dstw_t4_epoch_overlay.json.prismqnanl.receipt.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            overlay_receipt["artifact_family"],
            json!("T4_EPOCH_OVERLAY")
        );
        let decision_receipt: Value = serde_json::from_slice(
            &std::fs::read(dir.join("chronology_acceptance_decision.json.prismqnanl.receipt.json"))
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            decision_receipt["artifact_family"],
            json!("CHRONOLOGY_ACCEPTANCE_DECISION")
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn holo_gdp_topology_is_a_real_executed_ligand_state() {
        let dir = temp_dir("gdp");
        let holo = dir.join("4lpk_holo_gdp.topology.json");
        std::fs::write(&holo, br#"{"n_atoms":2507,"residue_names":["GDP","GLY"]}"#).unwrap();
        let mut inputs = base_inputs(&dir, &holo);
        inputs.ligand_mode = "holo".to_string();

        let report = emit_dstw_t4_state_graph(&inputs).unwrap();
        assert!(report.all_decode_proofs_pass);

        let pre: Value =
            serde_json::from_slice(&std::fs::read(dir.join("pre_md_state_vertex.json")).unwrap())
                .unwrap();
        let post: Value =
            serde_json::from_slice(&std::fs::read(dir.join("post_md_state_vertex.json")).unwrap())
                .unwrap();
        assert_eq!(pre["ligand_state"]["state"], json!("holo_gdp"));
        assert_eq!(post["ligand_state"]["state"], json!("holo_gdp"));
        assert_eq!(pre["ligand_state"]["feasibility"], json!("POPULATABLE_NOW"));
        assert_eq!(pre["ligand_state"]["gdp_present"], json!(true));
        let sha = pre["ligand_state"]["input_sha256"].as_str().unwrap();
        assert_eq!(sha.len(), 64);
        assert_eq!(sha, &super::sha256_of(&holo).unwrap());

        let md: Value =
            serde_json::from_slice(&std::fs::read(dir.join("md_state_transition.json")).unwrap())
                .unwrap();
        assert_eq!(
            md["operator_mode"]["mode"]["ligand_engine_effect"],
            json!("holo_gdp_topology_drives_engine_input")
        );
        assert!(!dir.join("blocked_gdp_bound_state_vertex.json").exists());
        assert!(!dir
            .join("gdp_forcefield_precondition_transition.json")
            .exists());

        let tsv = std::fs::read_to_string(&report.operand_ledger_path).unwrap();
        assert!(tsv.contains("STATE.VERTEX_CREATE"));
        assert!(tsv.contains("STATE.EMIT_TRANSITION"));
        assert!(tsv.contains("STATE.NEXT_OPERATOR_REC"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn step_h4_md_transition_verdict_is_falsifiable_not_file_existence() {
        // WITHOUT the required kcc evidence the MD transition must FAIL even though
        // the transition record file itself is written (verdict is NOT a function
        // of record existence).
        let dir = temp_dir("h4_fail");
        let topo = write_topology(&dir);
        let inputs = base_inputs(&dir, &topo);
        let report = emit_dstw_t4_state_graph(&inputs).unwrap();
        assert!(report.all_decode_proofs_pass);

        let md: Value =
            serde_json::from_slice(&std::fs::read(dir.join("md_state_transition.json")).unwrap())
                .unwrap();
        // record file exists, yet verdict is FAIL (required kcc evidence unobserved).
        assert_eq!(
            md["validation_gate"]["verdict"],
            json!("FAIL_EXPECTED_CHANGE_UNOBSERVED")
        );
        assert_eq!(md["validation_gate"]["passed"], json!(false));
        // h4 falsifiable fields present.
        assert_eq!(md["pre_state_id"], md["source_vertex_id"]);
        assert_eq!(md["post_state_id"], md["target_vertex_id"]);
        assert_eq!(md["operator_family"], json!("MD"));
        assert!(md["operator_id"].is_object());
        // failed_preconditions name the REQUIRED unobserved coordinate/kcc change.
        let failed = md["failed_preconditions"].as_array().unwrap();
        assert!(!failed.is_empty());
        assert!(failed.iter().any(|f| f["tensor"] == json!("kcc_tensor")));
        // a typed EXPECTED_CHANGE_UNOBSERVED contradiction was emitted.
        let contras = md["contradiction_refs"].as_array().unwrap();
        assert!(contras
            .iter()
            .any(|c| c["contradiction_id"] == json!("EXPECTED_CHANGE_UNOBSERVED")));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn step_h4_md_transition_passes_when_evidence_observed() {
        // WITH the required kcc evidence present + distinct from pre baseline, the
        // gate observes the change and PASSES.
        let dir = temp_dir("h4_pass");
        let topo = write_topology(&dir);
        std::fs::write(dir.join("run_kcc_v2full.bin"), b"\x10\x20\x30\x40\x50").unwrap();
        std::fs::write(dir.join("run_spikes.bin"), b"\x00\x01").unwrap();
        let inputs = base_inputs(&dir, &topo);
        let report = emit_dstw_t4_state_graph(&inputs).unwrap();
        assert!(report.all_decode_proofs_pass);

        let md: Value =
            serde_json::from_slice(&std::fs::read(dir.join("md_state_transition.json")).unwrap())
                .unwrap();
        assert_eq!(
            md["validation_gate"]["verdict"],
            json!("PASS_OBSERVED_CHANGE_VERIFIED")
        );
        assert_eq!(md["validation_gate"]["passed"], json!(true));
        // observed_change carries a REAL computed coordinate hash + evidence hash.
        let coord = md["observed_change"]["coordinate_hash"].as_str().unwrap();
        assert_eq!(coord.len(), 64);
        let ev = md["observed_change"]["evidence_hash"].as_str().unwrap();
        assert_eq!(ev.len(), 64);
        // observed coordinate is distinct from the pre topology baseline.
        let kcc_obs = md["observed_change"]["observations"]
            .as_array()
            .unwrap()
            .iter()
            .find(|o| o["tensor"] == json!("kcc_tensor"))
            .unwrap();
        assert_eq!(kcc_obs["observed"], json!(true));
        assert_eq!(kcc_obs["distinct_from_pre_baseline"], json!(true));
        assert!(md["failed_preconditions"].as_array().unwrap().is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn step_h4_invalid_holo_request_is_blocked_by_topology_evidence() {
        let dir = temp_dir("h4_gdp");
        let topo = write_topology(&dir);
        let mut inputs = base_inputs(&dir, &topo);
        inputs.ligand_mode = "holo".to_string();
        let report = emit_dstw_t4_state_graph(&inputs).unwrap();
        assert!(report.all_decode_proofs_pass);

        let pre: Value =
            serde_json::from_slice(&std::fs::read(dir.join("pre_md_state_vertex.json")).unwrap())
                .unwrap();
        assert_eq!(pre["ligand_state"]["state"], json!("holo_missing_gdp"));
        assert_eq!(pre["ligand_state"]["feasibility"], json!("BLOCKED"));
        assert_eq!(
            pre["ligand_state"]["blocker"],
            json!("HOLO_TOPOLOGY_GDP_ABSENT: LigandMode::Holo requires the engine-driving topology to contain GDP; prismos md-run rejects this before launch.")
        );
        assert!(!dir.join("blocked_gdp_bound_state_vertex.json").exists());
        assert!(!dir
            .join("gdp_forcefield_precondition_transition.json")
            .exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn step_h4_validation_state_emitted_and_not_null_padded() {
        let dir = temp_dir("h4_vs");
        let topo = write_topology(&dir);
        std::fs::write(dir.join("run_spikes.bin"), b"\x00\x01\x02").unwrap();
        let inputs = base_inputs(&dir, &topo);
        let report = emit_dstw_t4_state_graph(&inputs).unwrap();
        assert!(report.all_decode_proofs_pass);
        // validation_state is in the emitted set.
        assert!(report.emitted.iter().any(|e| e.kind == "validation_state"));

        let vs: Value =
            serde_json::from_slice(&std::fs::read(dir.join("validation_state.json")).unwrap())
                .unwrap();
        assert_eq!(
            vs["schema_kind"],
            json!(DSTW_T4_VALIDATION_STATE_SCHEMA_KIND)
        );
        assert_eq!(vs["target_id"], json!("4LPK:KRAS_WT"));
        // topology hash is the REAL sha256 (64 hex).
        assert_eq!(vs["topology_hash"].as_str().unwrap().len(), 64);
        // spike count is NOT fabricated: exact count null WITH an absent_reason +
        // a real on-disk byte length.
        assert!(vs["spike_event_count"]["exact_event_count"].is_null());
        assert!(vs["spike_event_count"]["absent_reason"]
            .as_str()
            .unwrap()
            .contains("decoding Vec<GpuSpikeEvent>"));
        assert_eq!(vs["spike_event_count"]["on_disk_bytes"], json!(3));
        // known_site_overlap is honestly NOT_AVAILABLE with a reason, not null-padded.
        assert_eq!(
            vs["known_site_overlap"]["feasibility"],
            json!("NOT_AVAILABLE")
        );
        assert!(vs["known_site_overlap"]["absent_reason"]
            .as_str()
            .unwrap()
            .contains("not fabricated"));
        // unresolved evidence requirements include real missing tensors, not a
        // stale holo-ingest blocker for an apo run.
        let unresolved = vs["unresolved_evidence_requirements"].as_array().unwrap();
        assert!(!unresolved
            .iter()
            .any(|u| u["requirement"] == json!("HOLO_TOPOLOGY_INGEST")
                || u["blocker"]["blocker_id"] == json!("HOLO_PHYSICS_INGEST_ABSENT_IN_ENGINE")));
        assert!(unresolved.iter().any(|u| u["tensor"].is_string()));
        // contradiction graph refs do not include the old label-only holo contradiction.
        let contras = vs["contradiction_graph_refs"].as_array().unwrap();
        assert!(
            !contras
                .iter()
                .any(|c| c["contradiction_id"]
                    == json!("HOLO_INTENT_RECORDED_ENGINE_INPUT_UNCHANGED"))
        );
        // operand ledger carries the VALIDATION.STATE_SNAPSHOT row (Validation class).
        let tsv = std::fs::read_to_string(&report.operand_ledger_path).unwrap();
        assert!(tsv.contains("VALIDATION.STATE_SNAPSHOT"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn step_h5_dense_tensor_summaries_identity_always_present_and_never_null_padded() {
        // With NO dense artifacts present, every one of the 6 records still carries its identity
        // (tensor_role/modality/source_kind/tokenizer_family/parent_vertex_id) + POST_RUN_DERIVED
        // origin, and every absent data field carries a sibling *_absent_reason (NEVER null-padded).
        let dir = temp_dir("h5_absent");
        std::fs::create_dir_all(&dir).unwrap();
        let summaries = build_dense_tensor_semantic_summaries(
            &dir,
            "vtx_parent_abc",
            RuntimeOrigin::PostRunDerived,
        );
        assert_eq!(summaries.len(), 6, "one record per dense tensor");
        let roles: Vec<&str> = summaries
            .iter()
            .map(|s| s["tensor_role"].as_str().unwrap())
            .collect();
        for r in [
            "SIGNAL_GRID",
            "WARP_FIELD",
            "DSTW_ADAPTIVE_DT",
            "SUPERRESOLUTION",
            "TWIN_INTERFEROMETRIC",
            "NEUROMORPHIC_SENSOR",
        ] {
            assert!(roles.contains(&r), "missing dense tensor role {r}");
        }
        for s in &summaries {
            // identity ALWAYS present.
            assert!(s["tensor_role"].is_string());
            assert!(s["modality"].is_string());
            assert!(s["source_kind"].is_string());
            assert_eq!(s["tokenizer_family"], json!("DenseTensorSemanticSummary"));
            assert_eq!(s["runtime_origin"], json!("POST_RUN_DERIVED"));
            assert_eq!(s["runtime_native"], json!(false)); // POST_RUN_DERIVED is NOT native
            assert_eq!(s["parent_vertex_id"], json!("vtx_parent_abc"));
            assert_eq!(s["artifact_present"], json!(false));
            assert_eq!(s["summary_present"], json!(false));
            // absent data fields are null WITH a reason (never null-padded silently).
            assert!(s["source_hash"].is_null());
            assert!(
                s["source_hash_absent_reason"]
                    .as_str()
                    .unwrap()
                    .contains("no engine artifact"),
                "source_hash must carry a typed absent_reason"
            );
            assert!(s["row_count"].is_null());
            assert!(s["all_finite"].is_null());
            assert!(s["reducer_scalar"].is_null());
            assert!(s["decoded_summary_absent_reason"]
                .as_str()
                .unwrap()
                .contains("never from raw"));
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn step_h5_dense_tensor_summary_derives_real_hash_dims_and_finite_from_decoded_summary() {
        // A present engine artifact (signal_grid.bin) + its DECODED reducer summary
        // (signal_grid_summary.json, the SignalGridSummary shape) ⇒ source_hash REAL 64-hex,
        // dimensions + finite-scan derived ONLY from the decoded summary (never from the raw .bin).
        let dir = temp_dir("h5_present");
        std::fs::create_dir_all(&dir).unwrap();
        // raw dense tensor (bytes are NEVER parsed for semantics — only sha256/len).
        std::fs::write(dir.join("4lpk_stream07_signal_grid.bin"), vec![7u8; 4096]).unwrap();
        // DECODED reducer summary (dispatch_analysis_common SignalGridSummary JSON projection).
        let summary = json!({
            "row_count": 32768,
            "columns": ["hit_count_cold_mean", "hit_count_warm_mean", "hit_count_delta", "variance_classification"],
            "cold_mean": 1.5,
            "warm_mean": 9.25,
            "mean_abs_delta": 7.75,
            "hysteresis_score": 0.42,
            "reversibility_mean": 0.58,
            "thermally_activated_count": 120
        });
        std::fs::write(
            dir.join("4lpk_stream07_signal_grid_summary.json"),
            serde_json::to_vec_pretty(&summary).unwrap(),
        )
        .unwrap();

        let summaries = build_dense_tensor_semantic_summaries(
            &dir,
            "vtx_parent_xyz",
            RuntimeOrigin::PostRunDerived,
        );
        let sg = summaries
            .iter()
            .find(|s| s["tensor_role"] == json!("SIGNAL_GRID"))
            .unwrap();
        assert_eq!(sg["artifact_present"], json!(true));
        assert_eq!(sg["summary_present"], json!(true));
        // source_hash is a REAL 64-hex sha256 (VALUE-verified), never fabricated.
        assert_eq!(sg["source_hash"].as_str().unwrap().len(), 64);
        assert_eq!(sg["on_disk_bytes"], json!(4096));
        // stream_id parsed REAL from the _stream07_ artifact name.
        assert_eq!(sg["stream_id"], json!(7));
        // dimensions + finite-scan derived from the DECODED summary.
        assert_eq!(sg["row_count"], json!(32768));
        assert_eq!(sg["column_count"], json!(4));
        assert_eq!(sg["all_finite"], json!(true));
        assert_eq!(sg["nan_count"], json!(0));
        assert_eq!(sg["inf_count"], json!(0));
        // representative reduced scalar pulled by spec-ordered key (hysteresis_score).
        assert_eq!(sg["reducer_scalar"], json!(0.42));
        // NO absent_reason siblings on the populated path.
        assert!(sg.get("source_hash_absent_reason").is_none());
        assert!(sg.get("decoded_summary_absent_reason").is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn step_h5_dense_tensor_finite_scan_flags_stringified_nonfinite() {
        // A decoded summary that stringifies a non-finite reducer scalar ("NaN") must FAIL the
        // finite gate (all_finite=false) — the gate is honest, not vacuous.
        let dir = temp_dir("h5_nan");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("4lpk_stream03_warp_matrix.bin"), vec![3u8; 256]).unwrap();
        let summary = json!({
            "row_count": 100,
            "columns": ["shear_stress"],
            "metric_mean": "NaN",
            "metric_max": 12.0
        });
        std::fs::write(
            dir.join("4lpk_stream03_warp_summary.json"),
            serde_json::to_vec(&summary).unwrap(),
        )
        .unwrap();
        let summaries =
            build_dense_tensor_semantic_summaries(&dir, "vtx_p", RuntimeOrigin::PostRunDerived);
        let w = summaries
            .iter()
            .find(|s| s["tensor_role"] == json!("WARP_FIELD"))
            .unwrap();
        assert_eq!(w["summary_present"], json!(true));
        assert_eq!(
            w["all_finite"],
            json!(false),
            "stringified NaN must fail the finite gate"
        );
        assert_eq!(w["nan_count"], json!(1));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn step_h5_emit_seals_dense_tensor_summaries_and_grounds_opcode() {
        // The full emit path seals dense_tensor_semantic_summaries.json (byte-exact decode proof),
        // reports dense_tensor_summary_count==6, and the operand ledger carries the opcode-GROUNDED
        // STATE.DENSE_TENSOR_SUMMARY row (real ISA opcode 0x010E, caller-assigned origin).
        let dir = temp_dir("h5_emit");
        let topo = write_topology(&dir);
        let inputs = base_inputs(&dir, &topo);
        let report = emit_dstw_t4_state_graph(&inputs).unwrap();
        assert!(report.all_decode_proofs_pass);
        assert_eq!(report.dense_tensor_summary_count, 6);
        assert!(report
            .emitted
            .iter()
            .any(|e| e.kind == "dense_tensor_semantic_summaries"
                && e.runtime_origin == "POST_RUN_DERIVED"));
        // sealed artifact decodes byte-exact.
        let dts_path = dir.join("dense_tensor_semantic_summaries.json");
        assert!(dts_path.exists());
        let raw = std::fs::read(&dts_path).unwrap();
        let packet = std::fs::read(format!("{}.prismqnanl", dts_path.display())).unwrap();
        assert_eq!(decode_packet_bytes(&packet).unwrap(), raw);
        let doc: Value = serde_json::from_slice(&raw).unwrap();
        assert_eq!(doc["summaries"].as_array().unwrap().len(), 6);
        // opcode-grounded ledger row.
        let tsv = std::fs::read_to_string(&report.operand_ledger_path).unwrap();
        assert!(tsv.contains("STATE.DENSE_TENSOR_SUMMARY"));
        // the opcode IS registered (real ISA id), not an honest-absent None.
        assert_eq!(
            crate::binary_runtime::opcode_id_for_name("STATE.DENSE_TENSOR_SUMMARY"),
            Some(0x010E)
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn step_h5_native_md_caller_origin_is_stamped_before_dense_qnanl_seal() {
        let dir = temp_dir("h5_native_origin");
        let topo = write_topology(&dir);
        let mut inputs = base_inputs(&dir, &topo);
        inputs.dense_tensor_runtime_origin = RuntimeOrigin::RuntimeSubprocessOrchestratedByPrismos;

        let report = emit_dstw_t4_state_graph(&inputs).unwrap();
        assert!(report.all_decode_proofs_pass);
        assert!(report
            .emitted
            .iter()
            .any(|e| e.kind == "dense_tensor_semantic_summaries"
                && e.runtime_origin == "RUNTIME_SUBPROCESS_ORCHESTRATED_BY_PRISMOS"));

        let dts_path = dir.join("dense_tensor_semantic_summaries.json");
        let raw = std::fs::read(&dts_path).unwrap();
        let packet = std::fs::read(format!("{}.prismqnanl", dts_path.display())).unwrap();
        assert_eq!(decode_packet_bytes(&packet).unwrap(), raw);

        let doc: Value = serde_json::from_slice(&raw).unwrap();
        assert_eq!(
            doc["runtime_origin"],
            json!("RUNTIME_SUBPROCESS_ORCHESTRATED_BY_PRISMOS")
        );
        assert_eq!(doc["runtime_native"], json!(true));
        for summary in doc["summaries"].as_array().unwrap() {
            assert_eq!(
                summary["runtime_origin"],
                json!("RUNTIME_SUBPROCESS_ORCHESTRATED_BY_PRISMOS")
            );
            assert_eq!(summary["runtime_native"], json!(true));
        }

        let receipt_path = format!("{}.prismqnanl.receipt.json", dts_path.display());
        let receipt: Value = serde_json::from_slice(&std::fs::read(receipt_path).unwrap()).unwrap();
        assert_eq!(
            receipt["runtime_origin"],
            json!("RUNTIME_SUBPROCESS_ORCHESTRATED_BY_PRISMOS")
        );
        assert!(
            receipt["runtime_origin_assigned_unix_ns"]
                .as_u64()
                .unwrap_or(0)
                > 0
        );
        assert_eq!(
            receipt["runtime_origin_assignment_stage"],
            json!("dstw_state_graph::write_seal_and_prove")
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}

#[cfg(test)]
mod continuation_recovery_tests {
    use super::*;
    #[test]
    fn recursive_and_direct_dstw_discovery_ignore_retained_future_sources() {
        let root = tempfile::tempdir().unwrap();
        let future = root.path().join(".continuation_absent_owners/epoch/payload/future_spikes.bin");
        std::fs::create_dir_all(future.parent().unwrap()).unwrap();
        std::fs::write(&future, b"retained future").unwrap();
        assert!(find_artifact_recursive(root.path(), "_spikes.bin").is_none());
        assert!(find_artifact(future.parent().unwrap(), "_spikes.bin").is_none());
        let active = root.path().join("active_spikes.bin");
        std::fs::write(&active, b"committed active").unwrap();
        assert_eq!(find_artifact_recursive(root.path(), "_spikes.bin"), Some(active.clone()));
        assert_eq!(find_artifact(root.path(), "_spikes.bin"), Some(active));
        assert_eq!(std::fs::read(&future).unwrap(), b"retained future");
    }
}
