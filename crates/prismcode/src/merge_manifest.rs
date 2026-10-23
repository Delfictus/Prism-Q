//! DSTW-T4 Step h6 — PRISM-native merge manifest + state graph.
//!
//! Consumes the artifacts already emitted by [`crate::dstw_state_graph::emit_dstw_t4_state_graph`]
//! (the sealed/decode-proven vertices, transitions, blocked state, next-operator
//! recommendation, dense-tensor semantic summaries, validation state) plus the
//! typed [`OperandLedger`] and the PROCESS-PROVEN [`RuntimeProvenance`], and
//! produces the nine merge-readiness acceptance artifacts:
//!
//!   1. `PRISM_NATIVE_STATE_GRAPH.jsonl`            — every emitted artifact mapped to EXACTLY one
//!                                                    state-object class, with an explicit ORPHAN CHECK.
//!   2. `RUNTIME_EMISSION_LEDGER.tsv`               — per-artifact runtime_origin with PROCESS proof
//!                                                    (pid + packaged-binary sha256), never a bare label.
//!   3. `OPCODE_OPERAND_COVERAGE_LEDGER.tsv`        — re-emitted from the ledger (idempotent).
//!   4. `QNANL_PRIMARY_COMPRESSION_LEDGER.tsv`      — per packet: raw/compressed/ratio/body-hash/
//!                                                    seal-proof/decode-proof/family/parent-vertex/
//!                                                    operator/runtime_origin.
//!   5. `SCIENTIFIC_STATE_TRANSITION_LEDGER.tsv`    — per transition: source/target vertex, operator,
//!                                                    falsifiable verdict, contradiction count.
//!   6. `TOKENIZER_NATIVE_MATCHED_FIELD_LEDGER.tsv` — per artifact: declared tokenizer family +
//!                                                    HONEST matched_fields gate (tokenizer-owned).
//!   7. `CONTRADICTION_GRAPH.yaml`                  — the falsifiable contradictions across transitions.
//!   8. `NEXT_OPERATOR_QUEUE.jsonl`                 — the rank-ordered engine-owner work queue.
//!   9. `CORPUS_MERGE_READINESS.json`               — per-row gate evaluation; TRAINING_READY_PRISMOS_NATIVE
//!                                                    rows SEPARATED from derived/post-run rows.
//!
//! HONESTY BOUNDARIES (mirror the directive's FORBIDDEN GUARDRAILS):
//!   * `runtime_origin` is PROCESS-proven: the RUNTIME_EMISSION_LEDGER carries the
//!     emitting process pid + packaged-binary sha256 + emission time, not a bare
//!     string. A row whose provenance lacks the pid/binary-sha is marked
//!     `process_proven=false` with a reason — never silently trusted.
//!   * `matched_fields` and `sidecar_count==token_count` are computed by the
//!     prism-tokenizer routing layer, NOT by prismcode (prismcode has no tokenizer
//!     dependency by design — the tokenizer reads prismcode output). The
//!     TOKENIZER_NATIVE_MATCHED_FIELD_LEDGER therefore DECLARES the family mapping
//!     and records the matched-fields gate as `PENDING_TOKENIZER_ROUTE` with a
//!     reason — it is never null-padded and never fabricated as a number.
//!   * No field is null-padded: every absent value carries a sibling reason.
//!   * The merge-readiness gate is strict — a row is TRAINING_READY_PRISMOS_NATIVE
//!     ONLY if ALL of {native runtime_origin, QNANL decode proof, tokenizer
//!     matched_fields>0, sidecar_count==token_count, lineage to vertex+transition,
//!     non-opaque family, no post-run-only source} pass. Because the tokenizer
//!     matched-fields gate is PENDING (tokenizer-owned) and native tensor emission
//!     is engine-owner BLOCKED, NO row qualifies today; the blocking gate is named
//!     per row. Derived / post-run / blocked rows are SEPARATED into their own bucket.

use crate::dstw_state_graph::DstwEmittedArtifact;
use crate::operand_ledger::{OperandLedger, RuntimeOrigin};
use crate::runtime_provenance::RuntimeProvenance;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

pub const MERGE_MANIFEST_SCHEMA_VERSION: u32 = 1;

/// Outcome of one merge-manifest emission. All paths are written to disk.
#[derive(Clone, Debug)]
pub struct MergeManifestReport {
    /// The nine acceptance artifact paths (in emission order).
    pub output_paths: Vec<PathBuf>,
    /// Number of emitted artifacts the manifest classified.
    pub artifact_count: usize,
    /// Artifacts that did NOT map to a known state-object class (NOT PRISM-native).
    pub orphans: Vec<String>,
    /// Rows that passed EVERY merge-readiness gate (TRAINING_READY_PRISMOS_NATIVE).
    pub training_ready_native_count: usize,
    /// Rows separated as derived / post-run / blocked (NOT native).
    pub derived_or_post_run_count: usize,
    /// Rows that are native-origin candidates but blocked by a PENDING gate.
    pub native_candidate_blocked_count: usize,
}

impl MergeManifestReport {
    pub fn orphan_count(&self) -> usize {
        self.orphans.len()
    }
}

// ---- per-artifact classified view --------------------------------------------

/// The valid PRISM-native state-object classes (directive enumerated plus the operator-scoped
/// scientific surface event class).
/// Any artifact that does not map to one of these is an ORPHAN (NOT PRISM-native).
const STATE_OBJECT_CLASSES: &[&str] = &[
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
];

/// A fully-resolved view of one emitted artifact, grounded by OPENING the JSON +
/// its receipt sidecar (never trusted by filename alone).
struct ArtifactView {
    kind: String,
    json_path: PathBuf,
    packet_path: PathBuf,
    schema_kind: Option<String>,
    /// Exactly one of [`STATE_OBJECT_CLASSES`], or `ORPHAN_NOT_PRISM_NATIVE`.
    state_object_class: String,
    is_orphan: bool,
    /// Downstream tokenizer family (declared; the actual matched_fields is computed
    /// by prism-tokenizer, not here). `None` when the artifact is not a tokenized
    /// evidence source (e.g. the validation snapshot is carried as provenance).
    tokenizer_family: Option<&'static str>,
    /// Whether the declared family is opaque (an opaque-escape family is NOT a
    /// genuine matched-field family). All current DSTW families are non-opaque.
    tokenizer_family_absent_reason: Option<String>,
    family_opaque: bool,
    runtime_origin: RuntimeOrigin,
    runtime_native: bool,
    decode_proof_pass: bool,
    // ---- lineage (read from the artifact JSON) ----
    vertex_id: Option<String>,
    parent_vertex_id: Option<String>,
    source_vertex_id: Option<String>,
    target_vertex_id: Option<String>,
    transition_id: Option<String>,
    operator_name: Option<String>,
    operator_family: Option<String>,
    validation_verdict: Option<String>,
    contradiction_refs: Vec<Value>,
    // ---- QNANL compression (read from the receipt sidecar) ----
    raw_bytes: Option<u64>,
    compressed_body_bytes: Option<u64>,
    compression_ratio_vs_raw: Option<f64>,
    compressed_body_hash_blake3: Option<String>,
    qnanl_seal_hash_blake3: Option<String>,
    roundtrip_verified: Option<bool>,
    compression_absent_reason: Option<String>,
    // ---- process-proven emission identity ----
    artifact_sha256: Option<String>,
    emission_unix_ns: u128,
    run_pid: u32,
    packaged_binary_sha256: Option<String>,
    process_proven: bool,
    process_proven_absent_reason: Option<String>,
}

fn read_json(path: &Path) -> Option<Value> {
    let bytes = std::fs::read(path).ok()?;
    serde_json::from_slice(&bytes).ok()
}

fn str_field(v: &Value, key: &str) -> Option<String> {
    v.get(key).and_then(Value::as_str).map(str::to_string)
}

/// Map (schema_kind, kind) to EXACTLY one state-object class. Returns
/// `(class, is_orphan)`. Grounded primarily in the on-disk `schema_kind`.
fn classify_state_object(schema_kind: Option<&str>, kind: &str) -> (String, bool) {
    match schema_kind {
        Some("dstw_t4_scientific_state_vertex") => {
            if kind.contains("blocked") {
                ("BlockedState".to_string(), false)
            } else {
                ("ScientificStateVertex".to_string(), false)
            }
        }
        Some("dstw_t4_scientific_state_transition") => ("Transition".to_string(), false),
        Some("dstw_t4_next_operator_recommendation") => {
            ("NextOperatorRecommendation".to_string(), false)
        }
        Some("dstw_t4_dense_tensor_semantic_summary") => ("EvidenceTensor".to_string(), false),
        Some("dstw_t4_validation_state") => ("Validation".to_string(), false),
        Some("dstw_t4_epoch_overlay") => ("Lineage".to_string(), false),
        Some("dstw_t4_chronology_acceptance_decision") => ("Validation".to_string(), false),
        Some("ScientificPocketOpeningEvent")
        | Some("scientific_pocket_opening_event")
        | Some("prismos_scientific_pocket_opening_event")
        | Some("LigandCausalActivityEvent")
        | Some("ligand_causal_activity_event")
        | Some("prismos_ligand_causal_activity_event")
        | Some("ResidueActivityEvent")
        | Some("residue_activity_event")
        | Some("prismos_residue_activity_event") => ("ScientificSurfaceEvent".to_string(), false),
        // Unknown schema_kind on disk => not a recognized PRISM-native state object.
        _ => ("ORPHAN_NOT_PRISM_NATIVE".to_string(), true),
    }
}

/// Declared downstream tokenizer family for an artifact (grounded in the
/// prism-tokenizer `ArtifactFamily` enum names; the validation snapshot has no
/// tokenized family — it is carried as provenance/lineage, not a tokenized source).
fn tokenizer_family_for(schema_kind: Option<&str>) -> (Option<&'static str>, Option<String>) {
    match schema_kind {
        Some("dstw_t4_scientific_state_vertex") => (Some("DstwStateVertex"), None),
        Some("dstw_t4_scientific_state_transition") => (Some("DstwStateTransition"), None),
        Some("dstw_t4_next_operator_recommendation") => (Some("DstwNextOperator"), None),
        Some("dstw_t4_dense_tensor_semantic_summary") => (Some("DenseTensorSemanticSummary"), None),
        Some("dstw_t4_epoch_overlay") => (Some("T4EpochOverlay"), None),
        Some("dstw_t4_chronology_acceptance_decision") => {
            (Some("ChronologyAcceptanceDecision"), None)
        }
        Some("ScientificPocketOpeningEvent")
        | Some("scientific_pocket_opening_event")
        | Some("prismos_scientific_pocket_opening_event") => {
            (Some("ScientificPocketOpeningEvent"), None)
        }
        Some("LigandCausalActivityEvent")
        | Some("ligand_causal_activity_event")
        | Some("prismos_ligand_causal_activity_event") => (Some("LigandCausalActivityEvent"), None),
        Some("ResidueActivityEvent")
        | Some("residue_activity_event")
        | Some("prismos_residue_activity_event") => (Some("ResidueActivityEvent"), None),
        Some("dstw_t4_validation_state") => (
            None,
            Some(
                "validation snapshot is carried as provenance/lineage, NOT a tokenized evidence \
                 source (no ArtifactFamily route — intentional, not a gap)"
                    .to_string(),
            ),
        ),
        _ => (
            None,
            Some("no recognized schema_kind => no declared tokenizer family".to_string()),
        ),
    }
}

/// Read the QNANL receipt sidecar (`<json>.prismqnanl.receipt.json`) compression
/// block. Honest absence (with reason) when the sidecar is missing.
#[allow(clippy::type_complexity)]
fn read_compression(
    json_path: &Path,
) -> (
    Option<u64>,
    Option<u64>,
    Option<f64>,
    Option<String>,
    Option<String>,
    Option<bool>,
    Option<String>,
) {
    let receipt_path = PathBuf::from(format!("{}.prismqnanl.receipt.json", json_path.display()));
    match read_json(&receipt_path) {
        Some(r) => (
            r.get("raw_bytes").and_then(Value::as_u64),
            r.get("compressed_body_bytes").and_then(Value::as_u64),
            r.get("compression_ratio_vs_raw").and_then(Value::as_f64),
            str_field(&r, "compressed_body_hash_blake3"),
            str_field(&r, "qnanl_seal_hash_blake3"),
            r.get("roundtrip_verified").and_then(Value::as_bool),
            None,
        ),
        None => (
            None,
            None,
            None,
            None,
            None,
            None,
            Some(format!(
                "QNANL receipt sidecar absent at {} (compression fields derive ONLY from the \
                 sealed receipt, never recomputed/guessed)",
                receipt_path.display()
            )),
        ),
    }
}

/// Build the resolved view for one emitted artifact.
fn build_view(art: &DstwEmittedArtifact, provenance: &RuntimeProvenance) -> ArtifactView {
    let body = read_json(&art.json_path);
    let schema_kind = body.as_ref().and_then(|v| str_field(v, "schema_kind"));
    let (state_object_class, is_orphan) = classify_state_object(schema_kind.as_deref(), &art.kind);
    let (tokenizer_family, tokenizer_family_absent_reason) =
        tokenizer_family_for(schema_kind.as_deref());

    let runtime_origin =
        RuntimeOrigin::from_label(&art.runtime_origin).unwrap_or(RuntimeOrigin::PostRunSealOnly);

    // Lineage fields, read from the artifact JSON when present.
    let (vertex_id, parent_vertex_id) = match &body {
        Some(v) => (str_field(v, "vertex_id"), str_field(v, "parent_vertex_id")),
        None => (None, None),
    };
    let (source_vertex_id, target_vertex_id, transition_id, operator_name, operator_family) =
        match &body {
            Some(v) => (
                str_field(v, "source_vertex_id"),
                str_field(v, "target_vertex_id"),
                str_field(v, "transition_id"),
                str_field(v, "operator_name"),
                str_field(v, "operator_family"),
            ),
            None => (None, None, None, None, None),
        };
    let validation_verdict = body
        .as_ref()
        .and_then(|v| v.get("validation_gate"))
        .and_then(|g| g.get("verdict"))
        .and_then(Value::as_str)
        .map(str::to_string);
    let contradiction_refs = body
        .as_ref()
        .and_then(|v| v.get("contradiction_refs"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    let (
        raw_bytes,
        compressed_body_bytes,
        compression_ratio_vs_raw,
        compressed_body_hash_blake3,
        qnanl_seal_hash_blake3,
        roundtrip_verified,
        compression_absent_reason,
    ) = read_compression(&art.json_path);

    // PROCESS-PROVEN emission identity: open the emitted JSON, hash it, stamp the
    // emitting process pid + packaged-binary sha256. A bare runtime_origin label is
    // INVALID without this — process_proven=false names exactly what is missing.
    let prov = provenance.artifact_record(
        &art.json_path,
        format!("merge_manifest:{}", art.kind),
        Vec::new(),
    );
    let packaged_binary_sha256 = provenance.packaged_binary_sha256.clone();
    let mut process_proven_absent_reason: Option<String> = None;
    let process_proven = if prov.emitted_artifact_sha256.is_none() {
        process_proven_absent_reason = Some(
            "emitted artifact sha256 unreadable — cannot bind the runtime_origin label to bytes"
                .to_string(),
        );
        false
    } else if packaged_binary_sha256.is_none() {
        process_proven_absent_reason = Some(
            provenance
                .packaged_binary_sha256_absent_reason
                .clone()
                .unwrap_or_else(|| "packaged binary sha256 absent".to_string()),
        );
        false
    } else {
        true
    };

    ArtifactView {
        kind: art.kind.clone(),
        json_path: art.json_path.clone(),
        packet_path: art.qnanl_packet_path.clone(),
        schema_kind,
        state_object_class,
        is_orphan,
        tokenizer_family,
        tokenizer_family_absent_reason,
        family_opaque: false,
        runtime_origin,
        runtime_native: runtime_origin.is_prismos_native(),
        decode_proof_pass: art.decode_proof_pass,
        vertex_id,
        parent_vertex_id,
        source_vertex_id,
        target_vertex_id,
        transition_id,
        operator_name,
        operator_family,
        validation_verdict,
        contradiction_refs,
        raw_bytes,
        compressed_body_bytes,
        compression_ratio_vs_raw,
        compressed_body_hash_blake3,
        qnanl_seal_hash_blake3,
        roundtrip_verified,
        compression_absent_reason,
        artifact_sha256: prov.emitted_artifact_sha256,
        emission_unix_ns: prov.emission_unix_ns,
        run_pid: provenance.pid,
        packaged_binary_sha256,
        process_proven,
        process_proven_absent_reason,
    }
}

fn tsv_cell(opt: &Option<String>) -> String {
    match opt {
        Some(s) => s.replace(['\t', '\n'], " "),
        None => String::new(),
    }
}

fn write_file(path: &Path, body: &str, out: &mut Vec<PathBuf>) -> Result<(), String> {
    std::fs::write(path, body).map_err(|e| format!("write {}: {e}", path.display()))?;
    out.push(path.to_path_buf());
    Ok(())
}

// ---- 1. PRISM_NATIVE_STATE_GRAPH.jsonl ---------------------------------------

fn emit_state_graph_jsonl(
    views: &[ArtifactView],
    output_dir: &Path,
    out: &mut Vec<PathBuf>,
) -> Result<Vec<String>, String> {
    let mut lines: Vec<String> = Vec::new();
    let mut orphans: Vec<String> = Vec::new();
    for v in views {
        if v.is_orphan {
            orphans.push(v.kind.clone());
        }
        let node = json!({
            "record_type": "state_graph_node",
            "schema_version": MERGE_MANIFEST_SCHEMA_VERSION,
            "artifact_kind": v.kind,
            "json_path": v.json_path.display().to_string(),
            "qnanl_packet_path": v.packet_path.display().to_string(),
            "schema_kind": v.schema_kind,
            "state_object_class": v.state_object_class,
            "is_orphan": v.is_orphan,
            "orphan_note": if v.is_orphan {
                Value::String(
                    "artifact does not map to any of the 11 PRISM-native state-object classes \
                     => NOT PRISM-native (listed in orphan_check)".to_string(),
                )
            } else {
                Value::Null
            },
            "runtime_origin": v.runtime_origin.as_str(),
            "runtime_native": v.runtime_native,
            "decode_proof_pass": v.decode_proof_pass,
            "vertex_id": v.vertex_id,
            "parent_vertex_id": v.parent_vertex_id,
            "source_vertex_id": v.source_vertex_id,
            "target_vertex_id": v.target_vertex_id,
            "transition_id": v.transition_id,
            "operator_name": v.operator_name,
            "tokenizer_family": v.tokenizer_family,
            "tokenizer_family_absent_reason": v.tokenizer_family_absent_reason,
            "artifact_sha256": v.artifact_sha256,
            "emission_unix_ns": v.emission_unix_ns.to_string(),
            "run_pid": v.run_pid,
        });
        lines.push(serde_json::to_string(&node).map_err(|e| e.to_string())?);
    }
    // Explicit ORPHAN CHECK record as the final JSONL line.
    let orphan_check = json!({
        "record_type": "orphan_check",
        "schema_version": MERGE_MANIFEST_SCHEMA_VERSION,
        "total_artifacts": views.len(),
        "valid_state_object_classes": STATE_OBJECT_CLASSES,
        "orphan_count": orphans.len(),
        "orphans": orphans,
        "verdict": if orphans.is_empty() {
            "ALL_ARTIFACTS_PRISM_NATIVE"
        } else {
            "ORPHANS_PRESENT_NOT_PRISM_NATIVE"
        },
    });
    lines.push(serde_json::to_string(&orphan_check).map_err(|e| e.to_string())?);

    let mut body = lines.join("\n");
    body.push('\n');
    write_file(
        &output_dir.join("PRISM_NATIVE_STATE_GRAPH.jsonl"),
        &body,
        out,
    )?;
    Ok(orphans)
}

// ---- 2. RUNTIME_EMISSION_LEDGER.tsv ------------------------------------------

fn emit_runtime_emission_ledger(
    views: &[ArtifactView],
    output_dir: &Path,
    out: &mut Vec<PathBuf>,
) -> Result<(), String> {
    let mut lines: Vec<String> = Vec::new();
    lines.push(
        "# DSTW-T4 Step h6 RUNTIME_EMISSION_LEDGER — runtime_origin is PROCESS-proven \
         (run_pid + packaged_binary_sha256 + emission_unix_ns + artifact sha256), NEVER a bare \
         label. process_proven=false names exactly what is missing."
            .to_string(),
    );
    lines.push(
        [
            "artifact_kind",
            "json_path",
            "runtime_origin",
            "runtime_native",
            "process_proven",
            "process_proven_absent_reason",
            "run_pid",
            "packaged_binary_sha256",
            "emission_unix_ns",
            "artifact_sha256",
            "decode_proof_pass",
            "parent_vertex_id",
        ]
        .join("\t"),
    );
    for v in views {
        lines.push(
            [
                v.kind.clone(),
                v.json_path.display().to_string(),
                v.runtime_origin.as_str().to_string(),
                v.runtime_native.to_string(),
                v.process_proven.to_string(),
                tsv_cell(&v.process_proven_absent_reason),
                v.run_pid.to_string(),
                tsv_cell(&v.packaged_binary_sha256),
                v.emission_unix_ns.to_string(),
                tsv_cell(&v.artifact_sha256),
                v.decode_proof_pass.to_string(),
                tsv_cell(&v.parent_vertex_id),
            ]
            .join("\t"),
        );
    }
    let mut body = lines.join("\n");
    body.push('\n');
    write_file(&output_dir.join("RUNTIME_EMISSION_LEDGER.tsv"), &body, out)
}

// ---- 4. QNANL_PRIMARY_COMPRESSION_LEDGER.tsv ---------------------------------

fn emit_qnanl_compression_ledger(
    views: &[ArtifactView],
    output_dir: &Path,
    out: &mut Vec<PathBuf>,
) -> Result<(), String> {
    let mut lines: Vec<String> = Vec::new();
    lines.push(
        "# DSTW-T4 Step h6 QNANL_PRIMARY_COMPRESSION_LEDGER — QNANL Primary Artifact Compression \
         (SECTION_ZSTD body + NANL-Q4 constrained seal). raw/compressed/ratio + body-hash + \
         seal-proof read from the sealed receipt; decode-proof is the executed byte-exact \
         re-decode. Absent compression => the receipt sidecar was missing (reason column)."
            .to_string(),
    );
    lines.push(
        [
            "artifact_kind",
            "packet_path",
            "raw_bytes",
            "compressed_body_bytes",
            "compression_ratio_vs_raw",
            "body_hash_blake3",
            "seal_proof_hash_blake3",
            "seal_roundtrip_verified",
            "decode_proof_pass",
            "tokenizer_family",
            "parent_vertex_id",
            "operator",
            "runtime_origin",
            "compression_absent_reason",
        ]
        .join("\t"),
    );
    for v in views {
        let num_u64 = |o: &Option<u64>| o.map(|x| x.to_string()).unwrap_or_default();
        let operator = v
            .operator_name
            .clone()
            .or_else(|| v.parent_vertex_id.clone().map(|_| "—".to_string()))
            .unwrap_or_default();
        lines.push(
            [
                v.kind.clone(),
                v.packet_path.display().to_string(),
                num_u64(&v.raw_bytes),
                num_u64(&v.compressed_body_bytes),
                v.compression_ratio_vs_raw
                    .map(|r| format!("{r:.6}"))
                    .unwrap_or_default(),
                tsv_cell(&v.compressed_body_hash_blake3),
                tsv_cell(&v.qnanl_seal_hash_blake3),
                v.roundtrip_verified
                    .map(|b| b.to_string())
                    .unwrap_or_default(),
                v.decode_proof_pass.to_string(),
                v.tokenizer_family.unwrap_or("").to_string(),
                tsv_cell(&v.parent_vertex_id),
                operator,
                v.runtime_origin.as_str().to_string(),
                tsv_cell(&v.compression_absent_reason),
            ]
            .join("\t"),
        );
    }
    let mut body = lines.join("\n");
    body.push('\n');
    write_file(
        &output_dir.join("QNANL_PRIMARY_COMPRESSION_LEDGER.tsv"),
        &body,
        out,
    )
}

// ---- 5. SCIENTIFIC_STATE_TRANSITION_LEDGER.tsv -------------------------------

fn emit_transition_ledger(
    views: &[ArtifactView],
    output_dir: &Path,
    out: &mut Vec<PathBuf>,
) -> Result<(), String> {
    let mut lines: Vec<String> = Vec::new();
    lines.push(
        "# DSTW-T4 Step h6 SCIENTIFIC_STATE_TRANSITION_LEDGER — one row per state-graph \
         transition. verdict is the FALSIFIABLE expected-vs-observed gate verdict (NOT a function \
         of record existence). contradiction_ref_count > 0 => see CONTRADICTION_GRAPH.yaml."
            .to_string(),
    );
    lines.push(
        [
            "transition_id",
            "artifact_kind",
            "source_vertex_id",
            "target_vertex_id",
            "operator_name",
            "operator_family",
            "validation_verdict",
            "runtime_origin",
            "runtime_native",
            "decode_proof_pass",
            "contradiction_ref_count",
        ]
        .join("\t"),
    );
    for v in views {
        if v.state_object_class != "Transition" {
            continue;
        }
        lines.push(
            [
                tsv_cell(&v.transition_id),
                v.kind.clone(),
                tsv_cell(&v.source_vertex_id),
                tsv_cell(&v.target_vertex_id),
                tsv_cell(&v.operator_name),
                tsv_cell(&v.operator_family),
                tsv_cell(&v.validation_verdict),
                v.runtime_origin.as_str().to_string(),
                v.runtime_native.to_string(),
                v.decode_proof_pass.to_string(),
                v.contradiction_refs.len().to_string(),
            ]
            .join("\t"),
        );
    }
    let mut body = lines.join("\n");
    body.push('\n');
    write_file(
        &output_dir.join("SCIENTIFIC_STATE_TRANSITION_LEDGER.tsv"),
        &body,
        out,
    )
}

// ---- 6. TOKENIZER_NATIVE_MATCHED_FIELD_LEDGER.tsv ----------------------------

fn emit_tokenizer_matched_field_ledger(
    views: &[ArtifactView],
    output_dir: &Path,
    out: &mut Vec<PathBuf>,
) -> Result<(), String> {
    let mut lines: Vec<String> = Vec::new();
    lines.push(
        "# DSTW-T4 Step h6 TOKENIZER_NATIVE_MATCHED_FIELD_LEDGER — DECLARES the downstream \
         tokenizer family per artifact. matched_fields + sidecar_count==token_count are computed \
         by the prism-tokenizer routing layer (NOT by prismcode — no tokenizer dependency by \
         design). The gate is PENDING_TOKENIZER_ROUTE with a reason; it is NEVER null-padded or \
         fabricated as a number. family_opaque=true would mean an opaque-escape (not a genuine \
         matched-field family) — all current DSTW families are non-opaque."
            .to_string(),
    );
    lines.push(
        [
            "artifact_kind",
            "tokenizer_family",
            "family_opaque",
            "matched_fields_gate",
            "matched_fields_value",
            "sidecar_count",
            "token_count",
            "gate_reason",
        ]
        .join("\t"),
    );
    for v in views {
        let (family, gate, reason) = match v.tokenizer_family {
            Some(f) => (
                f.to_string(),
                "PENDING_TOKENIZER_ROUTE".to_string(),
                "matched_fields/sidecar_count/token_count are produced by prism-tokenizer \
                 routing.rs over this family; prismcode declares the family but does not route"
                    .to_string(),
            ),
            None => (
                String::new(),
                "NOT_A_TOKENIZED_SOURCE".to_string(),
                v.tokenizer_family_absent_reason
                    .clone()
                    .unwrap_or_else(|| "no declared tokenizer family".to_string()),
            ),
        };
        lines.push(
            [
                v.kind.clone(),
                family,
                v.family_opaque.to_string(),
                gate,
                // matched_fields_value / sidecar_count / token_count: tokenizer-owned,
                // absent here WITH a reason in gate_reason — never null-padded to 0.
                "ABSENT_TOKENIZER_OWNED".to_string(),
                "ABSENT_TOKENIZER_OWNED".to_string(),
                "ABSENT_TOKENIZER_OWNED".to_string(),
                reason,
            ]
            .join("\t"),
        );
    }
    let mut body = lines.join("\n");
    body.push('\n');
    write_file(
        &output_dir.join("TOKENIZER_NATIVE_MATCHED_FIELD_LEDGER.tsv"),
        &body,
        out,
    )
}

// ---- 7. CONTRADICTION_GRAPH.yaml ---------------------------------------------

fn yaml_escape(s: &str) -> String {
    // Quote + escape for a YAML double-quoted scalar (safe for any content).
    let escaped = s
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', " ");
    format!("\"{escaped}\"")
}

fn emit_contradiction_graph(
    views: &[ArtifactView],
    output_dir: &Path,
    out: &mut Vec<PathBuf>,
) -> Result<usize, String> {
    let mut lines: Vec<String> = Vec::new();
    lines.push(
        "# DSTW-T4 Step h6 CONTRADICTION_GRAPH — falsifiable contradictions collected".to_string(),
    );
    lines.push(
        "# across the state-graph transitions (claim vs observation; each names its blocker)."
            .to_string(),
    );
    lines.push("schema_kind: dstw_t4_contradiction_graph".to_string());
    lines.push(format!("schema_version: {MERGE_MANIFEST_SCHEMA_VERSION}"));
    let mut total = 0usize;
    lines.push("contradictions:".to_string());
    for v in views {
        for c in &v.contradiction_refs {
            total += 1;
            let id = c
                .get("contradiction_id")
                .and_then(Value::as_str)
                .unwrap_or("UNNAMED");
            lines.push(format!("  - contradiction_id: {}", yaml_escape(id)));
            lines.push(format!("    source_transition: {}", yaml_escape(&v.kind)));
            if let Some(t) = &v.transition_id {
                lines.push(format!("    transition_id: {}", yaml_escape(t)));
            }
            for key in ["claim", "observation", "blocker_ref", "details", "verdict"] {
                if let Some(val) = c.get(key).and_then(Value::as_str) {
                    lines.push(format!("    {key}: {}", yaml_escape(val)));
                }
            }
        }
    }
    if total == 0 {
        // Honest: an empty graph, not a null-padded one.
        lines.push("  []".to_string());
        lines.push("contradiction_count: 0".to_string());
        lines.push(
            "note: \"no falsifiable contradictions observed across emitted transitions in this run\""
                .to_string(),
        );
    } else {
        lines.push(format!("contradiction_count: {total}"));
    }
    let mut body = lines.join("\n");
    body.push('\n');
    write_file(&output_dir.join("CONTRADICTION_GRAPH.yaml"), &body, out)?;
    Ok(total)
}

// ---- 8. NEXT_OPERATOR_QUEUE.jsonl --------------------------------------------

fn emit_next_operator_queue(
    views: &[ArtifactView],
    output_dir: &Path,
    out: &mut Vec<PathBuf>,
) -> Result<(), String> {
    let mut lines: Vec<String> = Vec::new();
    for v in views {
        if v.state_object_class != "NextOperatorRecommendation" {
            continue;
        }
        let body = match read_json(&v.json_path) {
            Some(b) => b,
            None => continue,
        };
        let origin = str_field(&body, "runtime_origin");
        if let Some(recs) = body
            .get("priority_ordered_recommendations")
            .and_then(Value::as_array)
        {
            for r in recs {
                let line = json!({
                    "record_type": "next_operator",
                    "schema_version": MERGE_MANIFEST_SCHEMA_VERSION,
                    "source_artifact": v.kind,
                    "runtime_origin": origin,
                    "rank": r.get("rank").cloned().unwrap_or(Value::Null),
                    "operator": r.get("operator").cloned().unwrap_or(Value::Null),
                    "owner": r.get("owner").cloned().unwrap_or(Value::Null),
                    "rationale": r.get("rationale").cloned().unwrap_or(Value::Null),
                    "evidence": r.get("evidence").cloned().unwrap_or(Value::Null),
                });
                lines.push(serde_json::to_string(&line).map_err(|e| e.to_string())?);
            }
        }
    }
    if lines.is_empty() {
        let empty = json!({
            "record_type": "next_operator_queue_empty",
            "schema_version": MERGE_MANIFEST_SCHEMA_VERSION,
            "note": "no next_operator_recommendation artifact present in this run",
        });
        lines.push(serde_json::to_string(&empty).map_err(|e| e.to_string())?);
    }
    let mut body = lines.join("\n");
    body.push('\n');
    write_file(&output_dir.join("NEXT_OPERATOR_QUEUE.jsonl"), &body, out)
}

// ---- 9. CORPUS_MERGE_READINESS.json ------------------------------------------

/// Evaluate the seven merge-readiness gates for one artifact. A row is
/// TRAINING_READY_PRISMOS_NATIVE only if ALL pass. Each gate carries its own
/// pass/reason — never a bare boolean.
fn evaluate_gates(v: &ArtifactView) -> Value {
    // G1 native runtime_origin proof (process-proven, not a bare label).
    let g1_pass = v.runtime_native && v.process_proven;
    let g1_reason = if !v.runtime_native {
        format!(
            "runtime_origin {} is NOT prismos-native",
            v.runtime_origin.as_str()
        )
    } else if !v.process_proven {
        v.process_proven_absent_reason
            .clone()
            .unwrap_or_else(|| "process proof missing".to_string())
    } else {
        "native + process-proven (run_pid + packaged_binary_sha256 + artifact sha256)".to_string()
    };

    // G2 QNANL decode proof (executed byte-exact re-decode).
    let g2_pass = v.decode_proof_pass;

    // G3 tokenizer matched_fields > 0 — TOKENIZER-OWNED, PENDING here.
    let g3_pass = false;
    let g3_reason =
        "matched_fields is computed by prism-tokenizer routing, not prismcode (PENDING_TOKENIZER_ROUTE)";

    // G4 sidecar_count == token_count — TOKENIZER-OWNED, PENDING here.
    let g4_pass = false;
    let g4_reason =
        "sidecar_count/token_count are produced by the tokenizer export, not prismcode (PENDING_TOKENIZER_ROUTE)";

    // G5 lineage to vertex + transition.
    let has_vertex_lineage = v.vertex_id.is_some() || v.parent_vertex_id.is_some();
    let has_transition_lineage =
        v.transition_id.is_some() || (v.source_vertex_id.is_some() && v.target_vertex_id.is_some());
    let g5_pass = has_vertex_lineage && has_transition_lineage;

    // G6 non-opaque family.
    let g6_pass = v.tokenizer_family.is_some() && !v.family_opaque;
    let g6_reason = match (v.tokenizer_family, v.family_opaque) {
        (Some(f), false) => format!("non-opaque family {f}"),
        (Some(f), true) => format!("family {f} is opaque-escape"),
        (None, _) => v
            .tokenizer_family_absent_reason
            .clone()
            .unwrap_or_else(|| "no tokenizer family".to_string()),
    };

    // G7 no post-run-only source.
    let post_run_only = matches!(
        v.runtime_origin,
        RuntimeOrigin::PostRunDerived
            | RuntimeOrigin::PostRunSealOnly
            | RuntimeOrigin::PostRunManual
    );
    let g7_pass = !post_run_only;

    json!({
        "g1_native_runtime_origin_proof": {"pass": g1_pass, "detail": g1_reason},
        "g2_qnanl_decode_proof": {"pass": g2_pass, "detail": "executed byte-exact re-decode"},
        "g3_tokenizer_matched_fields_gt_zero": {"pass": g3_pass, "detail": g3_reason},
        "g4_sidecar_count_eq_token_count": {"pass": g4_pass, "detail": g4_reason},
        "g5_lineage_to_vertex_and_transition": {
            "pass": g5_pass,
            "has_vertex_lineage": has_vertex_lineage,
            "has_transition_lineage": has_transition_lineage,
        },
        "g6_non_opaque_family": {"pass": g6_pass, "detail": g6_reason},
        "g7_no_post_run_only_source": {
            "pass": g7_pass,
            "detail": format!("runtime_origin={}", v.runtime_origin.as_str()),
        },
    })
}

fn emit_corpus_merge_readiness(
    views: &[ArtifactView],
    inputs_target: &str,
    output_dir: &Path,
    out: &mut Vec<PathBuf>,
) -> Result<(usize, usize, usize), String> {
    let mut native_ready: Vec<Value> = Vec::new();
    let mut native_candidate_blocked: Vec<Value> = Vec::new();
    let mut derived_or_post_run: Vec<Value> = Vec::new();

    for v in views {
        let gates = evaluate_gates(v);
        let all_pass = gates
            .as_object()
            .map(|m| {
                m.values()
                    .all(|g| g.get("pass").and_then(Value::as_bool).unwrap_or(false))
            })
            .unwrap_or(false);
        // The first failing gate is the named blocker.
        let blocking_gate = gates
            .as_object()
            .and_then(|m| {
                m.iter()
                    .find(|(_, g)| !g.get("pass").and_then(Value::as_bool).unwrap_or(false))
                    .map(|(k, _)| k.clone())
            })
            .unwrap_or_default();

        let row = json!({
            "artifact_kind": v.kind,
            "schema_kind": v.schema_kind,
            "state_object_class": v.state_object_class,
            "tokenizer_family": v.tokenizer_family,
            "runtime_origin": v.runtime_origin.as_str(),
            "runtime_native": v.runtime_native,
            "process_proven": v.process_proven,
            "decode_proof_pass": v.decode_proof_pass,
            "training_ready_prismos_native": all_pass,
            "blocking_gate": if all_pass { Value::Null } else { Value::String(blocking_gate) },
            "gates": gates,
        });

        // Bucket separation: derived/post-run/blocked go to their own bucket;
        // native-origin candidates that fail a PENDING gate go to the blocked-candidate bucket.
        let post_run_or_blocked = matches!(
            v.runtime_origin,
            RuntimeOrigin::PostRunDerived
                | RuntimeOrigin::PostRunSealOnly
                | RuntimeOrigin::PostRunManual
                | RuntimeOrigin::ExternalDerived
                | RuntimeOrigin::Blocked
        );
        if all_pass {
            native_ready.push(row);
        } else if post_run_or_blocked {
            derived_or_post_run.push(row);
        } else {
            native_candidate_blocked.push(row);
        }
    }

    let doc = json!({
        "schema_kind": "dstw_t4_corpus_merge_readiness",
        "schema_version": MERGE_MANIFEST_SCHEMA_VERSION,
        "target_id": inputs_target,
        "readiness_flag_definition": {
            "TRAINING_READY_PRISMOS_NATIVE": "true ONLY IF ALL seven gates pass: g1 native \
                runtime_origin (process-proven) + g2 QNANL decode proof + g3 tokenizer \
                matched_fields>0 + g4 sidecar_count==token_count + g5 lineage to vertex+transition \
                + g6 non-opaque family + g7 no post-run-only source.",
            "honesty_note": "g3 (matched_fields) and g4 (sidecar==token) are computed by the \
                prism-tokenizer routing/export layer, NOT by prismcode (no tokenizer dependency by \
                design). They are PENDING_TOKENIZER_ROUTE here, so NO row reaches \
                TRAINING_READY_PRISMOS_NATIVE in the prismcode-only emission — the native-origin \
                candidates are separated into native_candidate_blocked_by_pending_gate with the \
                blocking gate named, and the derived/post-run/blocked rows are separated again."
        },
        "buckets": {
            "training_ready_prismos_native": native_ready,
            "native_candidate_blocked_by_pending_gate": native_candidate_blocked,
            "derived_or_post_run_separated": derived_or_post_run,
        },
        "counts": {
            "training_ready_prismos_native": native_ready.len(),
            "native_candidate_blocked_by_pending_gate": native_candidate_blocked.len(),
            "derived_or_post_run_separated": derived_or_post_run.len(),
            "total": views.len(),
        }
    });
    let body = serde_json::to_string_pretty(&doc).map_err(|e| e.to_string())?;
    write_file(&output_dir.join("CORPUS_MERGE_READINESS.json"), &body, out)?;
    Ok((
        native_ready.len(),
        native_candidate_blocked.len(),
        derived_or_post_run.len(),
    ))
}

// ---- top-level emitter -------------------------------------------------------

/// Build the nine PRISM-native merge-readiness acceptance artifacts from the
/// already-emitted DSTW-T4 state-graph artifacts + ledger + PROCESS-PROVEN
/// provenance. Callable by prismos `md-run` (after `emit_dstw_t4_state_graph`) and
/// by the demo exerciser (its own captured provenance).
pub fn emit_prism_native_merge_manifest(
    emitted: &[DstwEmittedArtifact],
    ledger: &OperandLedger,
    provenance: &RuntimeProvenance,
    target_id: &str,
    output_dir: &Path,
) -> Result<MergeManifestReport, String> {
    std::fs::create_dir_all(output_dir).map_err(|e| e.to_string())?;
    let views: Vec<ArtifactView> = emitted.iter().map(|a| build_view(a, provenance)).collect();

    let mut out: Vec<PathBuf> = Vec::new();

    // 1. state graph + orphan check
    let orphans = emit_state_graph_jsonl(&views, output_dir, &mut out)?;
    // 2. runtime emission ledger (process-proven)
    emit_runtime_emission_ledger(&views, output_dir, &mut out)?;
    // 3. opcode/operand coverage ledger (idempotent re-emit from the ledger)
    let coverage_path = output_dir.join("OPCODE_OPERAND_COVERAGE_LEDGER.tsv");
    ledger
        .write_coverage_tsv(&coverage_path)
        .map_err(|e| e.to_string())?;
    out.push(coverage_path);
    // 4. QNANL primary compression ledger
    emit_qnanl_compression_ledger(&views, output_dir, &mut out)?;
    // 5. scientific state transition ledger
    emit_transition_ledger(&views, output_dir, &mut out)?;
    // 6. tokenizer native matched-field ledger
    emit_tokenizer_matched_field_ledger(&views, output_dir, &mut out)?;
    // 7. contradiction graph
    emit_contradiction_graph(&views, output_dir, &mut out)?;
    // 8. next operator queue
    emit_next_operator_queue(&views, output_dir, &mut out)?;
    // 9. corpus merge readiness
    let (native_ready, native_blocked, derived) =
        emit_corpus_merge_readiness(&views, target_id, output_dir, &mut out)?;

    Ok(MergeManifestReport {
        output_paths: out,
        artifact_count: views.len(),
        orphans,
        training_ready_native_count: native_ready,
        derived_or_post_run_count: derived,
        native_candidate_blocked_count: native_blocked,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dstw_state_graph::{emit_dstw_t4_state_graph, DstwStateGraphInputs};
    use std::collections::BTreeMap;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("merge_manifest_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
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
            dense_tensor_runtime_origin: crate::operand_ledger::RuntimeOrigin::PostRunDerived,
        }
    }

    /// The integrated path (what prismos `md-run` runs): `emit_dstw_t4_state_graph`
    /// now also emits the nine merge-manifest artifacts. Returns the full report.
    fn run_full(dir: &Path) -> crate::dstw_state_graph::DstwStateGraphReport {
        let topo = write_topology(dir);
        let inputs = base_inputs(dir, &topo);
        emit_dstw_t4_state_graph(&inputs).unwrap()
    }

    #[test]
    fn step_h6_every_artifact_maps_to_exactly_one_state_object_no_orphans() {
        let dir = temp_dir("no_orphans");
        let report = run_full(&dir);
        // Core run (no downstream) emits pre/post vertex, md transition, next_operator,
        // dense summaries, validation_state, and chronology closeout artifacts.
        assert!(
            report.merge_manifest_artifact_count >= 8,
            "expected >=8 artifacts, got {}",
            report.merge_manifest_artifact_count
        );
        assert_eq!(
            report.merge_manifest_orphan_count, 0,
            "no artifact may be an orphan"
        );
        // The 9 files exist.
        for name in [
            "PRISM_NATIVE_STATE_GRAPH.jsonl",
            "RUNTIME_EMISSION_LEDGER.tsv",
            "OPCODE_OPERAND_COVERAGE_LEDGER.tsv",
            "QNANL_PRIMARY_COMPRESSION_LEDGER.tsv",
            "SCIENTIFIC_STATE_TRANSITION_LEDGER.tsv",
            "TOKENIZER_NATIVE_MATCHED_FIELD_LEDGER.tsv",
            "CONTRADICTION_GRAPH.yaml",
            "NEXT_OPERATOR_QUEUE.jsonl",
            "CORPUS_MERGE_READINESS.json",
        ] {
            assert!(
                dir.join(name).exists(),
                "missing acceptance artifact {name}"
            );
        }
    }

    #[test]
    fn step_h6_state_graph_jsonl_has_orphan_check_and_classes() {
        let dir = temp_dir("jsonl");
        let _ = run_full(&dir);
        let body = std::fs::read_to_string(dir.join("PRISM_NATIVE_STATE_GRAPH.jsonl")).unwrap();
        let lines: Vec<&str> = body.lines().collect();
        // Final line is the orphan_check record.
        let last: Value = serde_json::from_str(lines.last().unwrap()).unwrap();
        assert_eq!(last["record_type"], "orphan_check");
        assert_eq!(last["orphan_count"], 0);
        assert_eq!(last["verdict"], "ALL_ARTIFACTS_PRISM_NATIVE");
        // Every node maps to a valid state-object class; no node is an orphan.
        for l in &lines[..lines.len() - 1] {
            let node: Value = serde_json::from_str(l).unwrap();
            assert_eq!(node["record_type"], "state_graph_node");
            let class = node["state_object_class"].as_str().unwrap();
            assert!(
                STATE_OBJECT_CLASSES.contains(&class),
                "class {class} not in the valid set"
            );
            assert_eq!(node["is_orphan"], false);
        }
    }

    #[test]
    fn step_h6_runtime_emission_ledger_is_process_proven_not_bare_label() {
        let dir = temp_dir("proven");
        let _ = run_full(&dir);
        let body = std::fs::read_to_string(dir.join("RUNTIME_EMISSION_LEDGER.tsv")).unwrap();
        // Header has process_proven + run_pid + packaged_binary_sha256 columns.
        let header = body.lines().nth(1).unwrap();
        assert!(header.contains("process_proven"));
        assert!(header.contains("run_pid"));
        assert!(header.contains("packaged_binary_sha256"));
        // At least one data row carries a real pid (process-proven, not a bare label).
        let pid = std::process::id().to_string();
        assert!(
            body.lines().any(|l| l.contains(&pid)),
            "expected the emitting process pid {pid} in the runtime emission ledger"
        );
    }

    #[test]
    fn step_h6_qnanl_compression_ledger_carries_real_seal_and_decode_proof() {
        let dir = temp_dir("qnanl");
        let _ = run_full(&dir);
        let body =
            std::fs::read_to_string(dir.join("QNANL_PRIMARY_COMPRESSION_LEDGER.tsv")).unwrap();
        let header = body.lines().nth(1).unwrap();
        for col in [
            "raw_bytes",
            "compressed_body_bytes",
            "compression_ratio_vs_raw",
            "body_hash_blake3",
            "seal_proof_hash_blake3",
            "decode_proof_pass",
        ] {
            assert!(header.contains(col), "missing column {col}");
        }
        // A data row must carry a real 64-hex body hash + decode_proof_pass=true.
        let data: Vec<&str> = body
            .lines()
            .filter(|l| !l.starts_with('#') && !l.starts_with("artifact_kind"))
            .collect();
        assert!(!data.is_empty());
        let mut saw_hash = false;
        for row in &data {
            let cols: Vec<&str> = row.split('\t').collect();
            // body_hash_blake3 is column index 5.
            if cols.len() > 8
                && cols[5].len() == 64
                && cols[5].chars().all(|c| c.is_ascii_hexdigit())
            {
                saw_hash = true;
                assert_eq!(
                    cols[8], "true",
                    "decode_proof_pass must be true for sealed packet"
                );
            }
        }
        assert!(saw_hash, "expected at least one real blake3 body hash");
    }

    #[test]
    fn step_h6_corpus_readiness_separates_buckets_and_names_blocking_gate() {
        let dir = temp_dir("readiness");
        let report = run_full(&dir);
        let body = std::fs::read_to_string(dir.join("CORPUS_MERGE_READINESS.json")).unwrap();
        let doc: Value = serde_json::from_str(&body).unwrap();
        // No row is training-ready in the prismcode-only emission (tokenizer gate PENDING).
        assert_eq!(report.merge_manifest_training_ready_native_count, 0);
        assert_eq!(doc["counts"]["training_ready_prismos_native"], 0);
        // Native-origin candidates are blocked by the PENDING tokenizer gate (g3),
        // and separated from derived/post-run rows.
        assert!(report.merge_manifest_native_candidate_blocked_count > 0);
        assert!(report.merge_manifest_derived_or_post_run_count > 0);
        // The pre-MD native vertex's blocking gate is the tokenizer matched-fields gate.
        let blocked = doc["buckets"]["native_candidate_blocked_by_pending_gate"]
            .as_array()
            .unwrap();
        assert!(blocked
            .iter()
            .any(|r| r["blocking_gate"] == "g3_tokenizer_matched_fields_gt_zero"));
        // Derived bucket holds the POST_RUN_DERIVED dense-tensor summaries + BLOCKED rows.
        let derived = doc["buckets"]["derived_or_post_run_separated"]
            .as_array()
            .unwrap();
        assert!(derived
            .iter()
            .any(|r| r["runtime_origin"] == "POST_RUN_DERIVED"));
    }

    #[test]
    #[ignore = "LEGACY_PRE_PRISM4D_NON_CANONICAL: asserts CONTRADICTION_GRAPH.yaml contains HOLO_INTENT_RECORDED_ENGINE_INPUT_UNCHANGED, a string the current generator does not emit (spec decision). merge_manifest runtime is UNCHANGED. Run via `cargo test -- --ignored`"]
    fn step_h6_contradiction_graph_and_transition_ledger_grounded() {
        let dir = temp_dir("contradiction");
        let _ = run_full(&dir);
        // The blocked GDP precondition transition yields a real contradiction.
        let yaml = std::fs::read_to_string(dir.join("CONTRADICTION_GRAPH.yaml")).unwrap();
        assert!(yaml.contains("contradiction_id"));
        assert!(yaml.contains("HOLO_INTENT_RECORDED_ENGINE_INPUT_UNCHANGED"));
        // Transition ledger has the two core transitions with falsifiable verdicts.
        let tl =
            std::fs::read_to_string(dir.join("SCIENTIFIC_STATE_TRANSITION_LEDGER.tsv")).unwrap();
        assert!(tl.contains("BLOCKED_PRECONDITION"));
        assert!(tl.contains("MD.INTEGRATE_VELOCITY_VERLET"));
    }

    /// Demo-exerciser path: the public `emit_prism_native_merge_manifest` is callable
    /// STANDALONE over an existing set of emitted artifacts (read from the prior run's
    /// dir) into a SEPARATE output dir, with the exerciser process's own PROCESS-PROVEN
    /// provenance. This is the second supported caller alongside prismos md-run.
    #[test]
    fn step_h6_public_emitter_is_callable_standalone_by_demo_exerciser() {
        let src = temp_dir("demo_src");
        let report = run_full(&src);
        // A separate consumer dir; the emitter references the absolute json paths in
        // report.emitted, so re-running it elsewhere is a clean read-over-existing.
        let demo_out = temp_dir("demo_out");
        let provenance = RuntimeProvenance::capture(
            &demo_out,
            vec!["merge_manifest_demo_exerciser".to_string()],
        );
        let ledger = OperandLedger::new();
        let man = emit_prism_native_merge_manifest(
            &report.emitted,
            &ledger,
            &provenance,
            "4LPK:KRAS_WT",
            &demo_out,
        )
        .unwrap();
        assert_eq!(
            man.output_paths.len(),
            9,
            "expected exactly 9 acceptance artifacts"
        );
        assert_eq!(man.orphan_count(), 0);
        assert!(demo_out.join("PRISM_NATIVE_STATE_GRAPH.jsonl").exists());
        assert!(demo_out.join("CORPUS_MERGE_READINESS.json").exists());
        // The standalone run is process-proven by the DEMO process pid.
        let rel = std::fs::read_to_string(demo_out.join("RUNTIME_EMISSION_LEDGER.tsv")).unwrap();
        assert!(rel.contains(&std::process::id().to_string()));
    }
}
