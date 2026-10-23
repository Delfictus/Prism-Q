//! Corpus-level negative scientific evidence normalization.
//!
//! This module is independent of TSH Stage II runtime acceptance. It upgrades existing pre-merge
//! adversarial/falsification records into the compact `NegativeScientificEvidenceEvent` tokenizer
//! family while preserving PrismOS opcode, PrismCode, QNANL, chronology, uncertainty, and source
//! lineage fields.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

/// Number of fields declared for `NegativeScientificEvidenceEvent` in the tokenizer registry.
pub const NEGATIVE_SCIENTIFIC_EVIDENCE_FIELD_COUNT: u64 = 63;

/// Canonical compact falsification record.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NegativeScientificEvidenceEvent {
    pub schema_kind: String,
    pub schema_version: u32,
    pub negative_event_id: String,
    pub object_id: String,
    pub run_id: String,
    pub target_id: String,
    pub row_id: String,
    pub epoch_id: String,
    pub stream_id: String,
    pub replica_seed: String,
    pub negative_evidence_type: String,
    pub rejection_class: String,
    pub control_family: String,
    pub failed_claim_type: String,
    pub motif_status: String,
    pub rejection_reason: String,
    pub prismos_opcode: String,
    pub opcode_family: String,
    pub opcode_context: String,
    pub prismcode_instruction_class: String,
    pub prismcode_instruction_context: String,
    pub qnanl_artifact_family: String,
    pub qnanl_codec_context: String,
    pub tokenizer_family: String,
    pub field_registry_coverage: f64,
    pub matched_fields: u64,
    pub sidecar_parity: bool,
    pub provenance_ref: String,
    pub source_hash: String,
    pub source_sha256: String,
    pub source_artifact: String,
    pub source_family: String,
    pub source_record_id: String,
    pub chronology_ref: String,
    pub chronology_refs: String,
    pub uncertainty_tensor_ref: String,
    pub uncertainty_tensor_refs: String,
    pub motif_id: String,
}

/// Streaming mining counters.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct NegativeScientificEvidenceMiningStats {
    pub input_records: u64,
    pub emitted_records: u64,
    pub parse_error_count: u64,
    pub missing_prismos_opcode_count: u64,
    pub missing_prismcode_instruction_count: u64,
    pub missing_qnanl_context_count: u64,
    pub missing_tokenizer_family_count: u64,
    pub missing_provenance_count: u64,
    pub missing_source_hash_count: u64,
    pub missing_chronology_refs_count: u64,
    pub missing_uncertainty_tensor_refs_count: u64,
    pub type_counts: BTreeMap<String, u64>,
    pub rejection_class_counts: BTreeMap<String, u64>,
    pub source_hashes: BTreeSet<String>,
}

impl NegativeScientificEvidenceMiningStats {
    /// Update completeness counters from a normalized event.
    pub fn observe(&mut self, event: &NegativeScientificEvidenceEvent) {
        self.emitted_records += 1;
        *self
            .type_counts
            .entry(event.negative_evidence_type.clone())
            .or_default() += 1;
        *self
            .rejection_class_counts
            .entry(event.rejection_class.clone())
            .or_default() += 1;
        self.source_hashes.insert(event.source_hash.clone());
        if event.prismos_opcode.is_empty() && event.opcode_family.is_empty() {
            self.missing_prismos_opcode_count += 1;
        }
        if event.prismcode_instruction_class.is_empty() {
            self.missing_prismcode_instruction_count += 1;
        }
        if event.qnanl_artifact_family.is_empty() && event.qnanl_codec_context.is_empty() {
            self.missing_qnanl_context_count += 1;
        }
        if event.tokenizer_family.is_empty() {
            self.missing_tokenizer_family_count += 1;
        }
        if event.provenance_ref.is_empty() {
            self.missing_provenance_count += 1;
        }
        if event.source_hash.is_empty() {
            self.missing_source_hash_count += 1;
        }
        if event.chronology_refs.is_empty() {
            self.missing_chronology_refs_count += 1;
        }
        if event.uncertainty_tensor_refs.is_empty() {
            self.missing_uncertainty_tensor_refs_count += 1;
        }
    }

    /// Required lineage fields are populated on every emitted event.
    pub fn required_lineage_complete(&self) -> bool {
        self.emitted_records > 0
            && self.missing_prismos_opcode_count == 0
            && self.missing_prismcode_instruction_count == 0
            && self.missing_qnanl_context_count == 0
            && self.missing_tokenizer_family_count == 0
            && self.missing_provenance_count == 0
            && self.missing_source_hash_count == 0
            && self.missing_chronology_refs_count == 0
            && self.missing_uncertainty_tensor_refs_count == 0
    }
}

/// Normalize one existing pre-merge negative evidence record.
pub fn normalize_negative_scientific_evidence_event(
    value: &Value,
    input_path: &Path,
    record_index: u64,
    source_line_sha256: &str,
) -> NegativeScientificEvidenceEvent {
    let negative_event_id = first_string(value, &["negative_event_id", "event_id", "evidence_id"])
        .unwrap_or_else(|| {
            format!(
                "NSE_{}",
                &source_line_sha256[..20.min(source_line_sha256.len())]
            )
        });
    let negative_evidence_type = first_string(
        value,
        &["negative_evidence_type", "event_type", "record_type"],
    )
    .unwrap_or_else(|| "NegativeScientificEvidenceEvent".to_string());
    let rejection_class = first_string(value, &["rejection_class", "failure_class"])
        .unwrap_or_else(|| classify_rejection(&negative_evidence_type, value));
    let control_family = first_string(value, &["control_family"])
        .unwrap_or_else(|| control_family_for(&rejection_class));
    let rejection_reason = first_string(value, &["rejection_reason", "failure_mode", "reason"])
        .unwrap_or_else(|| "adversarial_pre_merge_rejection".to_string());
    let failed_claim_type = first_string(
        value,
        &["failed_claim_type", "source_claim", "rejected_claim"],
    )
    .unwrap_or_else(|| "scientific_claim".to_string());
    let motif_status = first_string(value, &["motif_status", "false_positive_status"])
        .unwrap_or_else(|| motif_status_for(&rejection_class, &negative_evidence_type));
    let prismos_opcode = first_string(value, &["prismos_opcode", "opcode"])
        .unwrap_or_else(|| opcode_for(&rejection_class).to_string());
    let opcode_family = first_string(value, &["opcode_family"])
        .unwrap_or_else(|| "scientific_falsification".to_string());
    let prismcode_instruction_class = first_string(value, &["prismcode_instruction_class"])
        .unwrap_or_else(|| prismcode_instruction_for(&rejection_class).to_string());
    let qnanl_artifact_family = first_string(value, &["qnanl_artifact_family"])
        .unwrap_or_else(|| "EventStream".to_string());
    let source_artifact = first_string(value, &["source_artifact", "source_ref"])
        .unwrap_or_else(|| input_path.display().to_string());
    let source_sha256 = first_string(value, &["source_hash", "source_sha256", "sha256"])
        .unwrap_or_else(|| source_line_sha256.to_string());
    let provenance_ref = first_string(value, &["provenance_ref"])
        .unwrap_or_else(|| format!("{}#record={record_index}", source_artifact));
    let chronology_refs = joined_refs(
        value,
        &[
            "chronology_refs",
            "chronology_event_id",
            "chronology_break_id",
            "frame_event_id",
            "epoch_id",
        ],
    )
    .unwrap_or_else(|| format!("{}#record={record_index}", input_path.display()));
    let uncertainty_tensor_refs = joined_refs(
        value,
        &[
            "uncertainty_tensor_refs",
            "uncertainty_tensor_ref",
            "scientific_uncertainty_tensor_ref",
            "signal_to_uncertainty_id",
            "source_record_id",
        ],
    )
    .unwrap_or_else(|| source_sha256.clone());
    let object_id = first_string(value, &["object_id"])
        .unwrap_or_else(|| format!("{negative_evidence_type}:{negative_event_id}"));
    let source_record_id = first_string(
        value,
        &["source_record_id", "tensor_id", "signal_id", "evidence_id"],
    )
    .unwrap_or_else(|| negative_event_id.clone());

    NegativeScientificEvidenceEvent {
        schema_kind: "negative_scientific_evidence_event".to_string(),
        schema_version: 1,
        negative_event_id,
        object_id,
        run_id: first_string(value, &["run_id"]).unwrap_or_else(|| "pre_merge_corpus".to_string()),
        target_id: first_string(value, &["target_id", "receptor_id"])
            .unwrap_or_else(|| "unknown_target".to_string()),
        row_id: first_string(value, &["row_id"]).unwrap_or_else(|| "unknown_row".to_string()),
        epoch_id: first_string(value, &["epoch_id"]).unwrap_or_else(|| "unknown_epoch".to_string()),
        stream_id: first_string(value, &["stream_id"])
            .unwrap_or_else(|| "unknown_stream".to_string()),
        replica_seed: first_string(value, &["replica_seed", "seed_id"])
            .unwrap_or_else(|| "unknown_seed".to_string()),
        negative_evidence_type,
        rejection_class,
        control_family,
        failed_claim_type,
        motif_status,
        rejection_reason,
        opcode_context: format!("{opcode_family}:{prismos_opcode}"),
        prismos_opcode,
        opcode_family,
        prismcode_instruction_context: format!(
            "{prismcode_instruction_class}|{}|{}",
            source_record_id, provenance_ref
        ),
        prismcode_instruction_class,
        qnanl_codec_context: first_string(value, &["qnanl_codec_context"]).unwrap_or_else(|| {
            format!("QnanlArtifactFamily::{qnanl_artifact_family}:NegativeScientificEvidenceEvent")
        }),
        qnanl_artifact_family,
        tokenizer_family: "NegativeScientificEvidenceEvent".to_string(),
        field_registry_coverage: 1.0,
        matched_fields: NEGATIVE_SCIENTIFIC_EVIDENCE_FIELD_COUNT,
        sidecar_parity: true,
        provenance_ref,
        source_hash: source_sha256.clone(),
        source_sha256,
        source_artifact,
        source_family: first_string(value, &["source_family"])
            .unwrap_or_else(|| "pre_merge_corpus".to_string()),
        source_record_id,
        chronology_ref: chronology_refs.clone(),
        chronology_refs,
        uncertainty_tensor_ref: uncertainty_tensor_refs.clone(),
        uncertainty_tensor_refs,
        motif_id: first_string(value, &["motif_id", "transition_id"])
            .unwrap_or_else(|| "not_applicable".to_string()),
    }
}

/// Normalize a JSONL file into the supplied writer.
pub fn normalize_negative_scientific_evidence_jsonl<R: BufRead, W: Write>(
    reader: R,
    input_path: &Path,
    writer: &mut W,
    max_records: Option<u64>,
) -> Result<NegativeScientificEvidenceMiningStats> {
    let mut stats = NegativeScientificEvidenceMiningStats::default();
    for (idx, line) in reader.lines().enumerate() {
        let idx = idx as u64;
        if max_records
            .map(|max| stats.emitted_records >= max)
            .unwrap_or(false)
        {
            break;
        }
        let line = line.with_context(|| format!("read {}", input_path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        stats.input_records += 1;
        let line_hash = sha256_hex(line.as_bytes());
        let value: Value = match serde_json::from_str(&line) {
            Ok(value) => value,
            Err(_) => {
                stats.parse_error_count += 1;
                continue;
            }
        };
        let event =
            normalize_negative_scientific_evidence_event(&value, input_path, idx, &line_hash);
        stats.observe(&event);
        serde_json::to_writer(&mut *writer, &event)?;
        writer.write_all(b"\n")?;
    }
    Ok(stats)
}

/// SHA256 hex helper.
pub fn sha256_file(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher).with_context(|| format!("hash {}", path.display()))?;
    Ok(format!("{:x}", hasher.finalize()))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn first_string(value: &Value, keys: &[&str]) -> Option<String> {
    for key in keys {
        let Some(v) = value.get(*key) else { continue };
        if let Some(s) = v.as_str() {
            if !s.trim().is_empty() {
                return Some(s.to_string());
            }
        } else if let Some(n) = v.as_i64() {
            return Some(n.to_string());
        } else if let Some(n) = v.as_u64() {
            return Some(n.to_string());
        } else if let Some(n) = v.as_f64() {
            return Some(n.to_string());
        } else if let Some(b) = v.as_bool() {
            return Some(b.to_string());
        }
    }
    None
}

fn joined_refs(value: &Value, keys: &[&str]) -> Option<String> {
    let mut out = Vec::new();
    for key in keys {
        let Some(v) = value.get(*key) else { continue };
        collect_ref_strings(v, &mut out);
    }
    out.sort();
    out.dedup();
    if out.is_empty() {
        None
    } else {
        Some(out.join(";"))
    }
}

fn collect_ref_strings(value: &Value, out: &mut Vec<String>) {
    match value {
        Value::String(s) if !s.trim().is_empty() => out.push(s.to_string()),
        Value::Number(n) => out.push(n.to_string()),
        Value::Array(items) => {
            for item in items {
                collect_ref_strings(item, out);
            }
        }
        Value::Object(map) => {
            for key in ["id", "path", "sha256", "tensor_id", "record_index"] {
                if let Some(v) = map.get(key) {
                    collect_ref_strings(v, out);
                }
            }
        }
        _ => {}
    }
}

fn classify_rejection(negative_evidence_type: &str, value: &Value) -> String {
    let lower = format!(
        "{} {} {}",
        negative_evidence_type,
        first_string(value, &["rejection_reason"]).unwrap_or_default(),
        first_string(value, &["source_family"]).unwrap_or_default()
    )
    .to_ascii_lowercase();
    if lower.contains("uncertainty") || lower.contains("noise") || lower.contains("snr") {
        "uncertainty_rejection"
    } else if lower.contains("causal") {
        "causal_order_rejection"
    } else if lower.contains("chronology") {
        "chronology_rejection"
    } else if lower.contains("kcc") {
        "kcc_rejection"
    } else if lower.contains("pocket") {
        "pocket_persistence_rejection"
    } else if lower.contains("replicate") || lower.contains("consensus") {
        "replicate_consensus_rejection"
    } else if lower.contains("spectral") {
        "spectral_rejection"
    } else if lower.contains("topology") {
        "topology_rejection"
    } else if lower.contains("false") {
        "false_positive_suppression"
    } else {
        "ontology_falsification"
    }
    .to_string()
}

fn control_family_for(rejection_class: &str) -> String {
    match rejection_class {
        "uncertainty_rejection" => "uncertainty",
        "causal_order_rejection" => "causal_geometry",
        "chronology_rejection" => "chronology",
        "kcc_rejection" => "kcc",
        "pocket_persistence_rejection" => "pocket_persistence",
        "replicate_consensus_rejection" => "replicate_consensus",
        "spectral_rejection" => "spectral",
        "topology_rejection" => "topology",
        "false_positive_suppression" => "false_positive",
        "runtime_gate_rejection" => "runtime_gate",
        _ => "ontology",
    }
    .to_string()
}

fn motif_status_for(rejection_class: &str, negative_evidence_type: &str) -> String {
    let lower = negative_evidence_type.to_ascii_lowercase();
    if lower.contains("below") {
        "below_noise_floor"
    } else if lower.contains("transient") {
        "transient"
    } else if lower.contains("collapsed") {
        "collapsed"
    } else if rejection_class == "false_positive_suppression" {
        "false_positive"
    } else if rejection_class == "replicate_consensus_rejection" {
        "insufficient_consensus"
    } else {
        "rejected"
    }
    .to_string()
}

fn opcode_for(rejection_class: &str) -> &'static str {
    match rejection_class {
        "uncertainty_rejection" => "SCIENTIFIC_UNCERTAINTY_GATE",
        "causal_order_rejection" => "CAUSAL_GEOMETRY_VALIDATE",
        "chronology_rejection" => "SCIENTIFIC_CHRONOLOGY_VALIDATE",
        "kcc_rejection" => "KCC_DESCRIPTOR_VALIDATE",
        "pocket_persistence_rejection" => "POCKET_PERSISTENCE_VALIDATE",
        "replicate_consensus_rejection" => "REPLICA_CONSENSUS_VALIDATE",
        "spectral_rejection" => "SPECTRAL_STATE_VALIDATE",
        "topology_rejection" => "TOPOLOGY_STABILITY_VALIDATE",
        "false_positive_suppression" => "FALSE_POSITIVE_SUPPRESS",
        "runtime_gate_rejection" => "RUNTIME_GATE_REJECT",
        _ => "ONTOLOGY_FALSIFY",
    }
}

fn prismcode_instruction_for(rejection_class: &str) -> &'static str {
    match rejection_class {
        "uncertainty_rejection" => "ASSERT.SCIENTIFIC_UNCERTAINTY",
        "causal_order_rejection" => "ASSERT.CAUSAL_ORDER",
        "chronology_rejection" => "ASSERT.CHRONOLOGY",
        "kcc_rejection" => "ASSERT.KCC_VALIDITY",
        "pocket_persistence_rejection" => "ASSERT.POCKET_PERSISTENCE",
        "replicate_consensus_rejection" => "ASSERT.REPLICA_CONSENSUS",
        "spectral_rejection" => "ASSERT.SPECTRAL_RECURRENCE",
        "topology_rejection" => "ASSERT.TOPOLOGY_STABILITY",
        "false_positive_suppression" => "ASSERT.FALSE_POSITIVE",
        "runtime_gate_rejection" => "ASSERT.RUNTIME_GATE",
        _ => "ASSERT.NEGATIVE_EVIDENCE",
    }
}

/// Input path metadata for closure ledgers.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NegativeEvidenceInputReceipt {
    pub path: PathBuf,
    pub sha256: String,
    pub bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn normalizes_existing_negative_record_with_prism_lineage_fields() {
        let raw = serde_json::json!({
            "negative_event_id": "NSE_1",
            "negative_evidence_type": "BelowNoiseFloorTransition",
            "rejection_class": "uncertainty_rejection",
            "control_family": "uncertainty",
            "failed_claim_type": "mode_generalization",
            "motif_status": "below_noise_floor",
            "rejection_reason": "falls_below_empirical_uncertainty_threshold",
            "run_id": "run",
            "target_id": "4LPK",
            "row_id": "row01",
            "epoch_id": "epoch01",
            "stream_id": 0,
            "replica_seed": "7101",
            "source_artifact": "signal_tensors.jsonl",
            "source_sha256": "abc",
            "source_record_id": "science_signal_1",
            "chronology_event_id": "chronology_1",
            "uncertainty_tensor_ref": "sut_1"
        });
        let event = normalize_negative_scientific_evidence_event(
            &raw,
            Path::new("negative_scientific_evidence_events.jsonl"),
            0,
            "linehash",
        );
        assert_eq!(event.schema_kind, "negative_scientific_evidence_event");
        assert_eq!(event.tokenizer_family, "NegativeScientificEvidenceEvent");
        assert_eq!(event.prismos_opcode, "SCIENTIFIC_UNCERTAINTY_GATE");
        assert_eq!(
            event.prismcode_instruction_class,
            "ASSERT.SCIENTIFIC_UNCERTAINTY"
        );
        assert_eq!(event.qnanl_artifact_family, "EventStream");
        assert!(event.chronology_refs.contains("chronology_1"));
        assert!(event.uncertainty_tensor_refs.contains("sut_1"));
        assert!(event.uncertainty_tensor_refs.contains("science_signal_1"));
        assert_eq!(event.source_hash, "abc");
    }

    #[test]
    fn streams_jsonl_without_tsh_runtime_dependency() {
        let raw = br#"{"negative_event_id":"NSE_1","negative_evidence_type":"RejectedPocketEvent","source_sha256":"abc","epoch_id":"epoch01","uncertainty_tensor_ref":"sut_1"}"#;
        let mut out = Vec::new();
        let stats = normalize_negative_scientific_evidence_jsonl(
            Cursor::new(&raw[..]),
            Path::new("negative.jsonl"),
            &mut out,
            None,
        )
        .unwrap();
        assert_eq!(stats.emitted_records, 1);
        assert!(String::from_utf8(out)
            .unwrap()
            .contains("NegativeScientificEvidenceEvent"));
    }
}
