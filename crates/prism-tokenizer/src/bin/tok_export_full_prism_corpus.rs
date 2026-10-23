//! tok_export_full_prism_corpus — Directive v6 Agent 3/4: full-PRISM sharded corpus exporter.
//!
//! Tokenizes the GENERATIVE / molecule tables (enamine synthons + V-space products + GFlowNet
//! policy samples) that the python decode harness (`scripts/decode_parquet_to_records.py`) has
//! already DECODED parquet→JSONL (decode-before-tokenize, INV-1; the columnar/compressed bytes are
//! never tokenized as semantics). It reads a MANIFEST of decoded sources, routes each exact record
//! through `route_object` into the packed-`u64` token stream + lossless sidecar, and emits SHARDED
//! outputs. Deterministic: stable source ordering (manifest order), stable shard ordering, no RNG,
//! no timestamps in content → rerun hashes match byte-for-byte.
//!
//! Manifest (JSON array):
//! ```json
//! [ {"jsonl_records_path": "…/enamine.jsonl", "family": "MoleculeGeneration", "source_id": "enamine_115k"}, … ]
//! ```
//! family ∈ {MoleculeGeneration, GflownetTrajectory, …} — any ArtifactFamily name (Debug form).
//!
//! ```bash
//! cargo run -p prism-tokenizer --bin tok_export_full_prism_corpus -- \
//!   --manifest /tmp/full_prism_manifest.json \
//!   --out data/tokenized_training_corpus/full_prism_v1 \
//!   --shard-size 50000
//! ```

use prism_tokenizer::sidecar::ExactValue;
use prism_tokenizer::{
    detect_family, detect_family_from_filename_hint, route_artifact, route_kcc_sites,
    route_kcc_sites_with_binding, route_kcc_sites_with_binding_and_therm,
    route_kcc_sites_with_therm, route_object, route_scientific_surface_event_record,
    route_twin_per_residue, ArtifactFamily, Encoder, FieldGeometry, SiteRouteReport, TokenStream,
    TokenizerBackend,
};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

const SEAL: [u8; 32] = [0u8; 32]; // export seal placeholder; per-source sha lives in source_lineage

struct Args {
    manifest: PathBuf,
    out: PathBuf,
    shard_size: usize,
}

fn print_usage() {
    println!(
        "tok_export_full_prism_corpus\n\
\n\
Usage:\n\
  tok_export_full_prism_corpus --manifest <manifest.json> --out <output-dir> [--shard-size <n>]\n\
\n\
Options:\n\
  --manifest <path>    JSON array describing decoded runtime/source records\n\
  --out <path>         Output directory for token shards, sidecars, and corpus card\n\
  --shard-size <n>     Records per shard [default: 50000]\n\
  -h, --help           Show this help text"
    );
}

fn require_value(argv: &[String], i: usize, flag: &str) -> String {
    argv.get(i + 1)
        .cloned()
        .unwrap_or_else(|| panic!("{flag} requires a value"))
}

fn parse_args() -> Args {
    let mut a = Args {
        manifest: PathBuf::from("/tmp/full_prism_manifest.json"),
        out: PathBuf::from("data/tokenized_training_corpus/full_prism_v1"),
        shard_size: 50_000,
    };
    let argv: Vec<String> = std::env::args().collect();
    if argv.iter().any(|s| s == "--help" || s == "-h") {
        print_usage();
        std::process::exit(0);
    }
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--manifest" => {
                a.manifest = PathBuf::from(require_value(&argv, i, "--manifest"));
                i += 2;
            }
            "--out" => {
                a.out = PathBuf::from(require_value(&argv, i, "--out"));
                i += 2;
            }
            "--shard-size" => {
                a.shard_size = require_value(&argv, i, "--shard-size")
                    .parse()
                    .unwrap_or(50_000);
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }
    a
}

#[derive(Clone)]
enum SourceKind {
    JsonlRecords {
        jsonl_records_path: PathBuf,
        family: Option<ArtifactFamily>,
    },
    JsonArtifact {
        json_path: PathBuf,
        family: Option<ArtifactFamily>,
    },
    KccSiteJoin {
        kcc_visualization_path: PathBuf,
        binding_sites_path: Option<PathBuf>,
        prism_therm_path: Option<PathBuf>,
    },
    /// REAL per-residue TWIN coupling evidence (twin_per_residue.json — a top-level array of
    /// per-residue records). Routed via `route_twin_per_residue` into the TwinCoupled family so the
    /// Cα VQ is fit on the records' own positions FIRST (non-escape spatial codes), unlike a generic
    /// JsonArtifact pass which would route each residue with no coordinate fit.
    TwinPerResidue { json_path: PathBuf },
    /// DSTW-T4 scientific-state graph (crates/prismcode/src/dstw_state_graph.rs). The directory holds
    /// the emitted lifecycle artifacts (pre/post_md + blocked_gdp vertices, md + gdp-precondition
    /// transitions, next_operator_recommendation, chronology closeout, and conditional
    /// EXTERNAL_DERIVED downstream vertices/transitions). Each is tokenized as a FIRST-CLASS row
    /// by flattening nested DSTW vertex/transition/recommendation blocks to the family's flat field
    /// specs. Every row's lineage references the sibling
    /// QNANL packet (path + sha), the PrismCode operand ledger TSV, the runtime QNANL receipt, and
    /// the vertex/transition id (the directive's tokenizer-row provenance requirement).
    DstwStateGraph {
        dstw_dir: PathBuf,
        operand_ledger_path: PathBuf,
    },
}

#[derive(Clone)]
struct Source {
    source_id: String,
    kind: SourceKind,
}

fn family_from_str(s: &str) -> Option<ArtifactFamily> {
    use ArtifactFamily::*;
    Some(match s {
        "SpikeEvent" => SpikeEvent,
        "NanlSpikePacket" => NanlSpikePacket,
        "MsmKineticSummary" => MsmKineticSummary,
        "TeacherStudentConsensus" => TeacherStudentConsensus,
        "FiberBundleDstw" => FiberBundleDstw,
        "BocpdEvent" => BocpdEvent,
        "GflownetTrajectory" => GflownetTrajectory,
        "QnanlArtifact" => QnanlArtifact,
        "GpuTelemetry" => GpuTelemetry,
        "ReceiptStatus" => ReceiptStatus,
        "HoloBoundState" => HoloBoundState,
        "CapabilityGate" => CapabilityGate,
        "DenseTensorSemanticSummary" => DenseTensorSemanticSummary,
        "TwinCoupled" => TwinCoupled,
        "MoleculeGeneration" => MoleculeGeneration,
        "SpikeTransitionEvent" => SpikeTransitionEvent,
        "BocpdLog" => BocpdLog,
        "ProtocolState" => ProtocolState,
        "NoiseFloor" => NoiseFloor,
        "TwinTransitionResidual" => TwinTransitionResidual,
        "TwinConditionEdge" => TwinConditionEdge,
        "NmaCoupledEdge" => NmaCoupledEdge,
        "KccDescriptor" => KccDescriptor,
        "KccSite" => KccSite,
        "DstwStateVertex" => DstwStateVertex,
        "DstwStateTransition" => DstwStateTransition,
        "DstwNextOperator" => DstwNextOperator,
        "FalsifiablePrediction" => FalsifiablePrediction,
        "RankerRow" => RankerRow,
        "SrcCuda" => SrcCuda,
        "SrcEngine" => SrcEngine,
        "SrcCodec" => SrcCodec,
        "SrcTwin" => SrcTwin,
        "SrcGraph" => SrcGraph,
        "SrcGflownet" => SrcGflownet,
        "ValidationState" => ValidationState,
        "CapturedGraphGpuTelemetry" => CapturedGraphGpuTelemetry,
        "LineageDag" => LineageDag,
        "ThermoPhaseManifold" => ThermoPhaseManifold,
        "ScientificChronologyEvent" => ScientificChronologyEvent,
        "ScientificStateEpoch" => ScientificStateEpoch,
        "T4EpochOverlay" => T4EpochOverlay,
        "ChronologyAcceptanceDecision" => ChronologyAcceptanceDecision,
        "ScientificPocketOpeningEvent" => ScientificPocketOpeningEvent,
        "LigandCausalActivityEvent" => LigandCausalActivityEvent,
        "ResidueActivityEvent" => ResidueActivityEvent,
        "SpectralExtractor" => SpectralExtractor,
        "IsaRegistrySummary" => IsaRegistrySummary,
        "KccDescriptorNative" => KccDescriptorNative,
        "SpikeSummary" => SpikeSummary,
        "PredictionState" => PredictionState,
        "PhaseManifoldEdgeValidation" => PhaseManifoldEdgeValidation,
        "ScientificUncertaintyTensor" | "scientific_uncertainty_tensor" => {
            ScientificUncertaintyTensor
        }
        "RuntimeNumericalUncertaintyTensor" | "runtime_numerical_uncertainty_tensor" => {
            RuntimeNumericalUncertaintyTensor
        }
        "ThermalUncertaintyTensor" | "thermal_uncertainty_tensor" => ThermalUncertaintyTensor,
        "IntegratorUncertaintyTensor" | "integrator_uncertainty_tensor" => {
            IntegratorUncertaintyTensor
        }
        "ObservableUncertaintyTensor" | "observable_uncertainty_tensor" => {
            ObservableUncertaintyTensor
        }
        "ReplicaUncertaintyTensor" | "replica_uncertainty_tensor" => ObservableUncertaintyTensor,
        "TemporalUncertaintyTensor" | "temporal_uncertainty_tensor" => EpochUncertaintyTensor,
        "ModeUncertaintyTensor" | "mode_uncertainty_tensor" => ScientificUncertaintyTensor,
        "PhysicsConsistencyTensor" | "physics_consistency_tensor" => {
            RuntimeNumericalUncertaintyTensor
        }
        "ObservableEvidenceTensor" | "observable_evidence_tensor" => ObservableUncertaintyTensor,
        "SignalTensor" | "signal_tensor" => SignalToUncertaintyTensor,
        "SignalToUncertaintyTensor" | "signal_to_uncertainty_tensor" => SignalToUncertaintyTensor,
        "SignalToUncertaintyChannelTensor" | "signal_to_uncertainty_channel_tensor" => {
            SignalToUncertaintyTensor
        }
        "ScientificEvidenceTensor" | "scientific_evidence_tensor" => ScientificUncertaintyTensor,
        "EpochUncertaintyTensor" | "epoch_uncertainty_tensor" => EpochUncertaintyTensor,
        "ThermodynamicSalienceEvent" | "thermodynamic_salience_event" => ThermodynamicSalienceEvent,
        "TransitionMotif"
        | "transition_motif"
        | "TransitionMotifCandidate"
        | "transition_motif_candidate" => TransitionMotif,
        "PersistentTransitionObject" | "persistent_transition_object" => PersistentTransitionObject,
        "TransitionConsensusEvent" | "transition_consensus_event" => TransitionConsensusEvent,
        "TransitionEvidenceLedger"
        | "transition_evidence_ledger"
        | "ResidentTransitionCache"
        | "resident_transition_cache"
        | "TSHRuntimeNonInterferenceLedger"
        | "tsh_runtime_non_interference_ledger"
        | "TSHStageIIICompatibilityManifest"
        | "tsh_stage_iii_compatibility_manifest"
        | "TSHRuntimeObserverManifest"
        | "tsh_runtime_observer_manifest" => TransitionEvidenceLedger,
        "ScientificTransitionConsensusEvent"
        | "scientific_transition_consensus_event"
        | "scientific_transition_consensus_events" => ScientificTransitionConsensusEvent,
        "TransitionFalsificationProfile"
        | "transition_falsification_profile"
        | "transition_falsification_profiles" => TransitionFalsificationProfile,
        "RejectedPersistentTransitionCandidate"
        | "rejected_persistent_transition_candidate"
        | "rejected_persistent_transition_candidates" => RejectedPersistentTransitionCandidate,
        "PersistentTransitionAcceptanceLedger"
        | "persistent_transition_acceptance_ledger"
        | "StageIIIEligibilityLedger"
        | "stage_iii_eligibility_ledger" => PersistentTransitionAcceptanceLedger,
        "EpistemicUncertainty"
        | "epistemic_uncertainty"
        | "EPISTEMIC_UNCERTAINTY"
        | "SigmaVariance"
        | "sigma_variance" => EpistemicUncertainty,
        "TeacherConsensus" | "teacher_consensus" | "TEACHER_CONSENSUS" | "ConsensusEntropy"
        | "consensus_entropy" => TeacherConsensus,
        "ThermodynamicAction"
        | "thermodynamic_action"
        | "THERMODYNAMIC_ACTION"
        | "TSHAction"
        | "tsh_action" => ThermodynamicAction,
        "NegativeScientificEvidenceEvent"
        | "negative_scientific_evidence_event"
        | "negative_scientific_evidence_events"
        | "NegativeEvidenceLedger"
        | "negative_evidence_ledger"
        | "FalsePositiveSuppressionLedger"
        | "false_positive_suppression_ledger"
        | "RecoverableNegativeEvidenceAudit"
        | "recoverable_negative_evidence_audit" => NegativeScientificEvidenceEvent,
        "Opaque" => Opaque,
        _ => return None,
    })
}

fn family_from_manifest(e: &serde_json::Value) -> Option<Option<ArtifactFamily>> {
    let Some(family) = e.get("family").and_then(|v| v.as_str()) else {
        return Some(None);
    };
    if family.eq_ignore_ascii_case("auto") {
        Some(None)
    } else {
        family_from_str(family).map(Some)
    }
}

fn family_with_filename_hint(family: Option<ArtifactFamily>, path: &str) -> Option<ArtifactFamily> {
    family.or_else(|| detect_family_from_filename_hint(path).map(|(family, _)| family))
}

fn is_uncertainty_tensor_family(family: ArtifactFamily) -> bool {
    matches!(
        family,
        ArtifactFamily::ScientificUncertaintyTensor
            | ArtifactFamily::RuntimeNumericalUncertaintyTensor
            | ArtifactFamily::ThermalUncertaintyTensor
            | ArtifactFamily::IntegratorUncertaintyTensor
            | ArtifactFamily::ObservableUncertaintyTensor
            | ArtifactFamily::SignalToUncertaintyTensor
            | ArtifactFamily::EpochUncertaintyTensor
    )
}

fn is_tsh_transition_family(family: ArtifactFamily) -> bool {
    matches!(
        family,
        ArtifactFamily::ThermodynamicSalienceEvent
            | ArtifactFamily::TransitionMotif
            | ArtifactFamily::PersistentTransitionObject
            | ArtifactFamily::TransitionConsensusEvent
            | ArtifactFamily::TransitionEvidenceLedger
            | ArtifactFamily::ScientificTransitionConsensusEvent
            | ArtifactFamily::TransitionFalsificationProfile
            | ArtifactFamily::RejectedPersistentTransitionCandidate
            | ArtifactFamily::PersistentTransitionAcceptanceLedger
    )
}

fn is_records_wrapper_family(family: ArtifactFamily) -> bool {
    is_uncertainty_tensor_family(family)
        || is_tsh_transition_family(family)
        || family == ArtifactFamily::NegativeScientificEvidenceEvent
}

fn parse_manifest(p: &Path) -> Vec<Source> {
    let v: serde_json::Value =
        serde_json::from_slice(&std::fs::read(p).expect("read manifest")).expect("parse manifest");
    let arr = v.as_array().cloned().unwrap_or_default();
    arr.iter()
        .filter_map(|e| {
            let kind = e
                .get("source_kind")
                .or_else(|| e.get("kind"))
                .and_then(|v| v.as_str())
                .unwrap_or("jsonl_records");
            if matches!(kind, "DstwStateGraph" | "dstw_state_graph")
                || e.get("dstw_dir").is_some()
                || e.get("dstw_state_graph_dir").is_some()
            {
                let dir = e
                    .get("dstw_dir")
                    .or_else(|| e.get("dstw_state_graph_dir"))
                    .and_then(|v| v.as_str())?;
                let dstw_dir = PathBuf::from(dir);
                let operand_ledger_path = e
                    .get("operand_ledger_path")
                    .and_then(|v| v.as_str())
                    .map(PathBuf::from)
                    .unwrap_or_else(|| dstw_dir.join("dstw_t4_operand_ledger.tsv"));
                let sid = e
                    .get("source_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("dstw_state_graph")
                    .to_string();
                return Some(Source {
                    source_id: sid,
                    kind: SourceKind::DstwStateGraph {
                        dstw_dir,
                        operand_ledger_path,
                    },
                });
            }

            if matches!(kind, "TwinPerResidue" | "twin_per_residue")
                || e.get("twin_per_residue_path").is_some()
            {
                let twin_path = e
                    .get("twin_per_residue_path")
                    .or_else(|| e.get("twin_path"))
                    .and_then(|v| v.as_str())?;
                let sid = e
                    .get("source_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("twin_per_residue")
                    .to_string();
                return Some(Source {
                    source_id: sid,
                    kind: SourceKind::TwinPerResidue {
                        json_path: PathBuf::from(twin_path),
                    },
                });
            }

            if matches!(kind, "KccSiteJoin" | "kcc_site_join")
                || e.get("kcc_visualization_path").is_some()
            {
                let kcc_path = e
                    .get("kcc_visualization_path")
                    .or_else(|| e.get("kcc_path"))?
                    .as_str()?;
                let sid = e
                    .get("source_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("kcc_site_join")
                    .to_string();
                let binding_sites_path = e
                    .get("binding_sites_path")
                    .or_else(|| e.get("binding_path"))
                    .and_then(|v| v.as_str())
                    .map(PathBuf::from);
                let prism_therm_path = e
                    .get("prism_therm_path")
                    .or_else(|| e.get("therm_path"))
                    .and_then(|v| v.as_str())
                    .map(PathBuf::from);
                return Some(Source {
                    source_id: sid,
                    kind: SourceKind::KccSiteJoin {
                        kcc_visualization_path: PathBuf::from(kcc_path),
                        binding_sites_path,
                        prism_therm_path,
                    },
                });
            }

            let family = family_from_manifest(e)?;
            let sid = e
                .get("source_id")
                .and_then(|v| v.as_str())
                .or_else(|| e.get("family").and_then(|v| v.as_str()))
                .unwrap_or("auto")
                .to_string();

            if let Some(path) = e
                .get("json_path")
                .or_else(|| e.get("json_artifact_path"))
                .and_then(|v| v.as_str())
            {
                let family = family_with_filename_hint(family, path);
                return Some(Source {
                    source_id: sid,
                    kind: SourceKind::JsonArtifact {
                        json_path: PathBuf::from(path),
                        family,
                    },
                });
            }

            let path = e
                .get("jsonl_records_path")
                .or_else(|| e.get("jsonl_path"))
                .or_else(|| e.get("records_path"))?
                .as_str()?;
            Some(Source {
                source_id: sid,
                kind: SourceKind::JsonlRecords {
                    jsonl_records_path: PathBuf::from(path),
                    family: family_with_filename_hint(family, path),
                },
            })
        })
        .collect()
}

fn exact_to_json(e: &ExactValue) -> serde_json::Value {
    match e {
        ExactValue::Scalar(x) => serde_json::json!({"kind": "scalar", "value": x}),
        ExactValue::Vec3(v) => serde_json::json!({"kind": "vec3", "value": v}),
        ExactValue::Text(t) => serde_json::json!({"kind": "text", "value": t}),
    }
}

fn read_json_file(
    path: &Path,
    label: &str,
    missing: &mut Vec<String>,
) -> Option<serde_json::Value> {
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(_) => {
            missing.push(format!("{label}:missing:{}", path.display()));
            return None;
        }
    };
    match serde_json::from_slice(&bytes) {
        Ok(json) => Some(json),
        Err(err) => {
            missing.push(format!("{label}:parse_failed:{}:{err}", path.display()));
            None
        }
    }
}

fn read_optional_json_file(
    path: &Option<PathBuf>,
    label: &str,
    missing: &mut Vec<String>,
) -> Option<serde_json::Value> {
    path.as_ref()
        .and_then(|p| read_json_file(p, label, missing))
}

fn optional_path_string(path: &Option<PathBuf>) -> serde_json::Value {
    path.as_ref()
        .map(|p| serde_json::Value::String(p.display().to_string()))
        .unwrap_or(serde_json::Value::Null)
}

fn single_site_kcc_json(kcc: &serde_json::Value, site: serde_json::Value) -> serde_json::Value {
    let mut obj = serde_json::Map::new();
    obj.insert("sites".to_string(), serde_json::Value::Array(vec![site]));
    for key in [
        "residues",
        "vector_field_definition",
        "semantics",
        "interpretation_guidelines",
    ] {
        if let Some(v) = kcc.get(key) {
            obj.insert(key.to_string(), v.clone());
        }
    }
    serde_json::Value::Object(obj)
}

fn route_kcc_join_record(
    enc: &Encoder,
    stream: &mut TokenStream,
    kcc_site_json: &serde_json::Value,
    binding_sites: Option<&serde_json::Value>,
    prism_therm: Option<&serde_json::Value>,
) -> SiteRouteReport {
    match (binding_sites, prism_therm) {
        (Some(binding), Some(therm)) => route_kcc_sites_with_binding_and_therm(
            enc,
            stream,
            kcc_site_json,
            binding,
            therm,
            true,
            SEAL,
            true,
        ),
        (Some(binding), None) => {
            route_kcc_sites_with_binding(enc, stream, kcc_site_json, binding, true, SEAL, true)
        }
        (None, Some(therm)) => {
            route_kcc_sites_with_therm(enc, stream, kcc_site_json, therm, true, SEAL, true)
        }
        (None, None) => route_kcc_sites(enc, stream, kcc_site_json, true, SEAL, true),
    }
}

fn site_report_json(rep: &SiteRouteReport) -> serde_json::Value {
    serde_json::json!({
        "sites": rep.sites,
        "matched": rep.matched,
        "null_kcc": rep.null_kcc,
        "members_total": rep.members_total,
        "members_resolved": rep.members_resolved,
        "members_unresolved": rep.members_unresolved,
        "join_validated": rep.join_validated,
        "array_length_mismatches": rep.array_length_mismatches,
        "members_enriched": rep.members_enriched,
        "members_te_enriched": rep.members_te_enriched,
        "sites_binding_enriched": rep.sites_binding_enriched,
    })
}

#[derive(Clone)]
struct DecodedRecord {
    value: serde_json::Value,
    lineage: serde_json::Value,
    family_hint: Option<ArtifactFamily>,
}

fn record_with_lineage(value: serde_json::Value) -> Option<DecodedRecord> {
    record_with_lineage_and_family(value, None)
}

fn record_with_lineage_and_family(
    mut value: serde_json::Value,
    family_hint: Option<ArtifactFamily>,
) -> Option<DecodedRecord> {
    let obj = value.as_object_mut()?;
    let lineage = obj.remove("__lineage__").unwrap_or(serde_json::Value::Null);
    Some(DecodedRecord {
        value,
        lineage,
        family_hint,
    })
}

fn load_jsonl_records(path: &Path) -> Vec<DecodedRecord> {
    let f = std::fs::File::open(path).unwrap_or_else(|_| panic!("open {}", path.display()));
    let reader = BufReader::new(f);
    let mut records = Vec::new();
    for line in reader.lines() {
        let line = line.unwrap();
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value = match serde_json::from_str(&line) {
            Ok(value) => value,
            Err(_) => continue,
        };
        if let Some(record) = record_with_lineage(value) {
            records.push(record);
        }
    }
    records
}

fn load_json_records(path: &Path) -> Vec<DecodedRecord> {
    let value: serde_json::Value = serde_json::from_slice(
        &std::fs::read(path).unwrap_or_else(|_| panic!("open {}", path.display())),
    )
    .unwrap_or_else(|err| panic!("parse {}: {err}", path.display()));
    let records_wrapper_family = detect_family(&value).and_then(|(family, record_path)| {
        ((record_path == "/records" && is_records_wrapper_family(family))
            || (record_path == "/transition_objects"
                && family == ArtifactFamily::PersistentTransitionObject))
            .then_some(family)
    });
    match value {
        serde_json::Value::Array(values) => {
            values.into_iter().filter_map(record_with_lineage).collect()
        }
        serde_json::Value::Object(map)
            if records_wrapper_family.is_some()
                && map
                    .get("records")
                    .or_else(|| map.get("transition_objects"))
                    .map(|records| records.is_array())
                    .unwrap_or(false) =>
        {
            map.get("records")
                .or_else(|| map.get("transition_objects"))
                .and_then(|records| records.as_array())
                .into_iter()
                .flatten()
                .cloned()
                .filter_map(|record| record_with_lineage_and_family(record, records_wrapper_family))
                .collect()
        }
        serde_json::Value::Object(mut obj)
            if matches!(
                obj.get("schema_kind").and_then(|v| v.as_str()),
                Some("prismos_scientific_chronology_events")
                    | Some("prismos_scientific_state_epochs")
                    | Some("prismos_t4_epoch_overlays")
                    | Some("prismos_chronology_acceptance_decisions")
                    | Some("ScientificPocketOpeningEvents")
                    | Some("scientific_pocket_opening_events")
                    | Some("prismos_scientific_pocket_opening_events")
                    | Some("LigandCausalActivityEvents")
                    | Some("ligand_causal_activity_events")
                    | Some("prismos_ligand_causal_activity_events")
                    | Some("ResidueActivityEvents")
                    | Some("residue_activity_events")
                    | Some("prismos_residue_activity_events")
            ) =>
        {
            match obj.remove("records") {
                Some(serde_json::Value::Array(values)) => {
                    values.into_iter().filter_map(record_with_lineage).collect()
                }
                _ => Vec::new(),
            }
        }
        value => record_with_lineage(value).into_iter().collect(),
    }
}

fn record_family(
    explicit_family: Option<ArtifactFamily>,
    value: &serde_json::Value,
) -> Option<ArtifactFamily> {
    explicit_family.or_else(|| detect_family(value).map(|(family, _)| family))
}

fn fit_decoded_records(
    enc: &mut Encoder,
    explicit_family: Option<ArtifactFamily>,
    records: &[DecodedRecord],
) {
    let mut families = Vec::new();
    for record in records {
        if let Some(family) = record_family(explicit_family.or(record.family_hint), &record.value) {
            if !families.contains(&family) {
                families.push(family);
            }
        }
    }
    for family in families {
        let family_records: Vec<serde_json::Map<String, serde_json::Value>> = records
            .iter()
            .filter(|record| {
                record_family(explicit_family.or(record.family_hint), &record.value) == Some(family)
            })
            .filter_map(|record| {
                let obj = record.value.as_object()?;
                if family == ArtifactFamily::SpectralExtractor {
                    Some(flatten_spectral_extractor_manifest(obj))
                } else {
                    Some(obj.clone())
                }
            })
            .collect();
        for (field, vals) in harvest_fit_samples(enc, family, &family_records) {
            enc.fit_field(family, field, &vals);
        }
    }
}

fn route_decoded_record(
    enc: &Encoder,
    stream: &mut TokenStream,
    explicit_family: Option<ArtifactFamily>,
    value: &serde_json::Value,
    record_index: usize,
) -> Option<(ArtifactFamily, usize)> {
    if let Some(family) = explicit_family {
        if matches!(
            family,
            ArtifactFamily::ScientificPocketOpeningEvent
                | ArtifactFamily::LigandCausalActivityEvent
                | ArtifactFamily::ResidueActivityEvent
        ) {
            let report = route_scientific_surface_event_record(
                enc,
                stream,
                family,
                value,
                record_index,
                true,
                SEAL,
                true,
            );
            return Some((family, report.matched));
        }
        // NEXT_OPERATOR_QUEUE.jsonl per-recommendation rows carry the BARE keys
        // (operator/owner/rank) rather than the aggregate's flat top_* keys. Without
        // this by-value remap a raw route matches only runtime_origin (1 field) = a
        // forbidden shallow wrap. Lift each row into the DstwNextOperator flat schema
        // so route_object matches top_operator/top_owner/top_rank/recommendation_count.
        let obj_owned;
        let obj = if family == ArtifactFamily::DstwNextOperator
            && value.get("operator").is_some()
            && value.get("top_operator").is_none()
        {
            let src = value.as_object()?;
            let mut f = serde_json::Map::new();
            if let Some(v) = src.get("runtime_origin") {
                f.insert("runtime_origin".into(), v.clone());
            }
            if let Some(v) = src.get("operator") {
                f.insert("top_operator".into(), v.clone());
            }
            if let Some(v) = src.get("owner") {
                f.insert("top_owner".into(), v.clone());
            }
            if let Some(v) = src.get("rank") {
                f.insert("top_rank".into(), v.clone());
            }
            f.insert("recommendation_count".into(), serde_json::json!(1));
            obj_owned = serde_json::Value::Object(f);
            obj_owned.as_object()?
        } else if family == ArtifactFamily::ThermoPhaseManifold {
            // Lift the nested CCNS/SDST/phase_bucket/trigger rollup to flat scalar keys so
            // route_object value-matches the thermodynamic content (the raw nested dicts would
            // otherwise be opaque escapes). The whole object is preserved for lossless sidecar
            // capture; only the flat keys are value-matched.
            obj_owned =
                serde_json::Value::Object(flatten_thermo_phase_manifold(value.as_object()?));
            obj_owned.as_object()?
        } else if family == ArtifactFamily::PhaseManifoldEdgeValidation {
            // Lift the nested stats.*/config.* scalars to flat keys AND derive the component
            // aggregates (n_components, sum_component_n_nodes, total_intra_edges,
            // mean/min/max_intra_component_mean_cosine) so route_object value-matches the real edge
            // counts + SO(3) coherence (the raw nested dicts + components[] would otherwise be opaque
            // escapes). The whole object is preserved for lossless §10 sidecar capture (INV-1); only
            // the flat keys are value-matched.
            obj_owned = serde_json::Value::Object(flatten_phase_manifold_edge_validation(
                value.as_object()?,
            ));
            obj_owned.as_object()?
        } else if family == ArtifactFamily::SpectralExtractor {
            // Lift the run-specific spectral closure manifest's nested `summary`, `source_inputs`,
            // `closure_outputs`, and `qnanl_source_state` into flat keys so the closure routes by
            // value. The manifest itself is preserved in the sidecar; this is not a filename wrap.
            obj_owned =
                serde_json::Value::Object(flatten_spectral_extractor_manifest(value.as_object()?));
            obj_owned.as_object()?
        } else {
            value.as_object()?
        };
        if matches!(
            family,
            ArtifactFamily::ScientificPocketOpeningEvent
                | ArtifactFamily::LigandCausalActivityEvent
                | ArtifactFamily::ResidueActivityEvent
        ) {
            let report = prism_tokenizer::routing::route_scientific_surface_event_record(
                enc,
                stream,
                family,
                value,
                record_index,
                true,
                SEAL,
                true,
            );
            return Some((family, report.matched));
        }
        let seg = stream.begin_record(family, record_index);
        let report = route_object(enc, stream, family, obj, true, SEAL, true);
        stream.end_record(seg);
        Some((family, report.matched))
    } else {
        let (family, report) = route_artifact(enc, stream, value, true, SEAL, true);
        family.map(|family| (family, report.matched))
    }
}

/// One shard writer set: six aligned JSONL shards (token/segment/sidecar/provenance/receipt/lineage)
/// rotate together every `shard_size` records, so a record's slices stay in the same shard ordinal.
struct ShardWriters {
    out: PathBuf,
    shard_size: usize,
    shard_idx: usize,
    records_in_shard: usize,
    tokens: std::fs::File,
    segments: std::fs::File,
    sidecar: std::fs::File,
    provenance: std::fs::File,
    receipts: std::fs::File,
    lineage: std::fs::File,
    files_written: Vec<String>,
    // running token position offset (so `pos`/sidecar position is a stable GLOBAL index across shards)
    token_offset: usize,
}

fn open_shard(out: &Path, name: &str, idx: usize, files: &mut Vec<String>) -> std::fs::File {
    let fname = format!("{name}.{idx:04}.jsonl");
    files.push(fname.clone());
    std::fs::File::create(out.join(&fname)).unwrap()
}

impl ShardWriters {
    fn new(out: PathBuf, shard_size: usize) -> Self {
        let mut files_written = Vec::new();
        let tokens = open_shard(&out, "tokens.u64", 0, &mut files_written);
        let segments = open_shard(&out, "segments", 0, &mut files_written);
        let sidecar = open_shard(&out, "sidecar", 0, &mut files_written);
        let provenance = open_shard(&out, "provenance", 0, &mut files_written);
        let receipts = open_shard(&out, "receipts", 0, &mut files_written);
        let lineage = open_shard(&out, "source_lineage", 0, &mut files_written);
        Self {
            out,
            shard_size,
            shard_idx: 0,
            records_in_shard: 0,
            tokens,
            segments,
            sidecar,
            provenance,
            receipts,
            lineage,
            files_written,
            token_offset: 0,
        }
    }

    fn rotate_if_needed(&mut self) {
        if self.records_in_shard >= self.shard_size {
            self.shard_idx += 1;
            self.records_in_shard = 0;
            self.tokens = open_shard(
                &self.out,
                "tokens.u64",
                self.shard_idx,
                &mut self.files_written,
            );
            self.segments = open_shard(
                &self.out,
                "segments",
                self.shard_idx,
                &mut self.files_written,
            );
            self.sidecar = open_shard(
                &self.out,
                "sidecar",
                self.shard_idx,
                &mut self.files_written,
            );
            self.provenance = open_shard(
                &self.out,
                "provenance",
                self.shard_idx,
                &mut self.files_written,
            );
            self.receipts = open_shard(
                &self.out,
                "receipts",
                self.shard_idx,
                &mut self.files_written,
            );
            self.lineage = open_shard(
                &self.out,
                "source_lineage",
                self.shard_idx,
                &mut self.files_written,
            );
        }
    }

    /// Flush ONE record's freshly produced stream slice into the current shard, then clear the
    /// in-memory stream so per-record memory stays bounded (we never hold the whole corpus).
    #[allow(clippy::too_many_arguments)]
    fn emit_record(
        &mut self,
        stream: &mut TokenStream,
        family: ArtifactFamily,
        source_id: &str,
        global_record_idx: usize,
        lineage: &serde_json::Value,
        matched: usize,
    ) {
        self.rotate_if_needed();
        // tokens (global pos index)
        for (i, t) in stream.tokens.iter().enumerate() {
            let line = serde_json::json!({"pos": self.token_offset + i, "token_u64": t.0});
            writeln!(self.tokens, "{}", line).unwrap();
        }
        // segments (rebased to global token offset; record_index is the GLOBAL record ordinal).
        for s in stream.segments.iter() {
            let line = serde_json::json!({
                "start": self.token_offset + s.start,
                "len": s.len,
                "record_index": global_record_idx,
                "family": format!("{:?}", s.family),
                "source_id": source_id,
            });
            writeln!(self.segments, "{}", line).unwrap();
        }
        // sidecar (rebased global position)
        for e in stream.sidecar.iter() {
            let line = serde_json::json!({
                "position": self.token_offset + e.position,
                "field": e.field,
                "exact": exact_to_json(&e.exact),
            });
            writeln!(self.sidecar, "{}", line).unwrap();
        }
        // provenance (one row per record: family + source + token span)
        let prov = serde_json::json!({
            "record_index": global_record_idx,
            "source_id": source_id,
            "family": format!("{:?}", family),
            "token_start": self.token_offset,
            "n_tokens": stream.tokens.len(),
            "n_sidecar": stream.sidecar.len(),
        });
        writeln!(self.provenance, "{}", prov).unwrap();
        // receipt (epistemic gate: generative families are firewalled, verified-gated; INV-3)
        let canonical_safe = stream
            .segments
            .first()
            .map(|s| s.family)
            .map(|_| family_is_canonical_safe(family))
            .unwrap_or(false);
        let receipt = serde_json::json!({
            "record_index": global_record_idx,
            "source_id": source_id,
            "family": format!("{:?}", family),
            "matched_fields": matched,
            "n_tokens": stream.tokens.len(),
            "canonical_safe": canonical_safe,
            "epistemic_class": if canonical_safe { "canonical_evidence" } else { "generative_firewall" },
        });
        writeln!(self.receipts, "{}", receipt).unwrap();
        // source lineage (carries the python harness __lineage__ block verbatim)
        let lin = serde_json::json!({
            "record_index": global_record_idx,
            "source_id": source_id,
            "lineage": lineage.clone(),
        });
        writeln!(self.lineage, "{}", lin).unwrap();

        self.token_offset += stream.tokens.len();
        self.records_in_shard += 1;
        // clear the stream for the next record (bounded memory)
        stream.tokens.clear();
        stream.segments.clear();
        stream.sidecar = Default::default();
    }
}

/// All v6 generative/molecule families are GENERATIVE (canonical_safe=false in their specs). A
/// record's epistemic class follows the family's spec firewall flag (INV-3). MoleculeGeneration and
/// GflownetTrajectory are both generative — never canonical evidence.
fn family_is_canonical_safe(family: ArtifactFamily) -> bool {
    !matches!(
        family,
        ArtifactFamily::MoleculeGeneration | ArtifactFamily::GflownetTrajectory
    )
}

// --- DSTW-T4 scientific-state graph row materialization -------------------------------------

/// One emitted DSTW-T4 lifecycle artifact (crates/prismcode/src/dstw_state_graph.rs filenames) and
/// the tokenizer family it routes to.
struct DstwArtifactSpec {
    filename: &'static str,
    kind: &'static str,
    family: ArtifactFamily,
}

/// The full DSTW-T4 emitted state-graph: the core artifacts emit_dstw_t4_state_graph always
/// writes, plus the 4 conditional EXTERNAL_DERIVED downstream artifacts (present only if the
/// upstream PathB / tokenizer artifact existed). Missing files are skipped — never faked.
fn dstw_t4_artifacts() -> Vec<DstwArtifactSpec> {
    use ArtifactFamily::*;
    vec![
        DstwArtifactSpec {
            filename: "pre_md_state_vertex.json",
            kind: "pre_md_state_vertex",
            family: DstwStateVertex,
        },
        DstwArtifactSpec {
            filename: "post_md_state_vertex.json",
            kind: "post_md_state_vertex",
            family: DstwStateVertex,
        },
        DstwArtifactSpec {
            filename: "blocked_gdp_bound_state_vertex.json",
            kind: "blocked_gdp_bound_state_vertex",
            family: DstwStateVertex,
        },
        DstwArtifactSpec {
            filename: "md_state_transition.json",
            kind: "md_state_transition",
            family: DstwStateTransition,
        },
        DstwArtifactSpec {
            filename: "gdp_forcefield_precondition_transition.json",
            kind: "gdp_forcefield_precondition_transition",
            family: DstwStateTransition,
        },
        DstwArtifactSpec {
            filename: "next_operator_recommendation.json",
            kind: "next_operator_recommendation",
            family: DstwNextOperator,
        },
        DstwArtifactSpec {
            filename: "dstw_t4_epoch_overlay.json",
            kind: "dstw_t4_epoch_overlay",
            family: T4EpochOverlay,
        },
        DstwArtifactSpec {
            filename: "chronology_acceptance_decision.json",
            kind: "chronology_acceptance_decision",
            family: ChronologyAcceptanceDecision,
        },
        // conditional EXTERNAL_DERIVED downstream (emit_downstream_if_present)
        DstwArtifactSpec {
            filename: "post_pathb_state_vertex.json",
            kind: "post_pathb_state_vertex",
            family: DstwStateVertex,
        },
        DstwArtifactSpec {
            filename: "post_pathb_state_transition.json",
            kind: "post_pathb_state_transition",
            family: DstwStateTransition,
        },
        DstwArtifactSpec {
            filename: "post_tokenizer_state_vertex.json",
            kind: "post_tokenizer_state_vertex",
            family: DstwStateVertex,
        },
        DstwArtifactSpec {
            filename: "post_tokenizer_state_transition.json",
            kind: "post_tokenizer_state_transition",
            family: DstwStateTransition,
        },
    ]
}

fn read_dstw_json(path: &Path) -> Option<serde_json::Value> {
    let bytes = std::fs::read(path).ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// VALUE-verified sha256 of a file (hex). None if the file is missing/unreadable — never fabricated.
fn sha256_file_hex(path: &Path) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    let mut h = Sha256::new();
    h.update(&bytes);
    Some(format!("{:x}", h.finalize()))
}

fn obj_get<'a>(
    obj: &'a serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Option<&'a serde_json::Value> {
    obj.get(key).filter(|v| !v.is_null())
}

fn array_len(obj: &serde_json::Map<String, serde_json::Value>, key: &str) -> usize {
    obj.get(key)
        .and_then(|v| v.as_array())
        .map(|a| a.len())
        .unwrap_or(0)
}

/// Count tensor blocks (keys ending in `_tensor`) whose `feasibility` equals `want`.
fn count_tensor_feasibility(obj: &serde_json::Map<String, serde_json::Value>, want: &str) -> usize {
    obj.iter()
        .filter(|(k, _)| k.ends_with("_tensor"))
        .filter(|(_, v)| {
            v.as_object()
                .and_then(|m| m.get("feasibility"))
                .and_then(|f| f.as_str())
                == Some(want)
        })
        .count()
}

/// Flatten a DSTW-T4 vertex JSON (apo pre/post, BLOCKED holo_gdp, or EXTERNAL_DERIVED downstream)
/// into the DstwStateVertex flat field keys. Nested ligand_state / detection_mode.posture scalars
/// are lifted; null fields are simply omitted (route_object only matches present keys).
fn flatten_dstw_vertex(
    obj: &serde_json::Map<String, serde_json::Value>,
) -> serde_json::Map<String, serde_json::Value> {
    let mut f = serde_json::Map::new();
    if let Some(v) = obj_get(obj, "lifecycle_stage") {
        f.insert("lifecycle_stage".into(), v.clone());
    }
    if let Some(v) = obj_get(obj, "vertex_id") {
        f.insert("vertex_id".into(), v.clone());
    }
    if let Some(v) = obj_get(obj, "target_id") {
        f.insert("target_id".into(), v.clone());
    }
    if let Some(ls) = obj.get("ligand_state").and_then(|v| v.as_object()) {
        if let Some(v) = obj_get(ls, "state") {
            f.insert("ligand_state".into(), v.clone());
        }
        if let Some(v) = obj_get(ls, "feasibility") {
            f.insert("feasibility".into(), v.clone());
        }
        if let Some(v) = obj_get(ls, "gdp_present") {
            f.insert("gdp_present".into(), v.clone());
        }
        if let Some(v) = obj_get(ls, "input_sha256") {
            f.insert("input_sha256".into(), v.clone());
        }
        if let Some(v) = obj_get(ls, "blocker") {
            f.insert("blocker".into(), v.clone());
        }
    }
    // downstream EXTERNAL_DERIVED vertex carries source_artifact_sha256 instead of ligand input sha.
    if !f.contains_key("input_sha256") {
        if let Some(v) = obj_get(obj, "source_artifact_sha256") {
            f.insert("input_sha256".into(), v.clone());
        }
    }
    if let Some(posture) = obj
        .get("detection_mode")
        .and_then(|v| v.as_object())
        .and_then(|m| m.get("posture"))
        .and_then(|v| v.as_object())
    {
        if let Some(v) = obj_get(posture, "detection_mode") {
            f.insert("detection_mode".into(), v.clone());
        }
        if let Some(v) = obj_get(posture, "site_ranker") {
            f.insert("site_ranker".into(), v.clone());
        }
        if let Some(v) = obj_get(posture, "md_only_evidence") {
            f.insert("md_only_evidence".into(), v.clone());
        }
    }
    if let Some(v) = obj_get(obj, "stage_family") {
        f.insert("stage_family".into(), v.clone());
    }
    if let Some(v) = obj_get(obj, "runtime_origin") {
        f.insert("runtime_origin".into(), v.clone());
    }
    if let Some(v) = obj_get(obj, "runtime_native") {
        f.insert("runtime_native".into(), v.clone());
    }
    if obj.contains_key("missing_preconditions") {
        f.insert(
            "missing_precondition_count".into(),
            serde_json::json!(array_len(obj, "missing_preconditions")),
        );
    }
    if obj.contains_key("qnanl_receipts") {
        f.insert(
            "qnanl_receipt_count".into(),
            serde_json::json!(array_len(obj, "qnanl_receipts")),
        );
    }
    let populatable = count_tensor_feasibility(obj, "POPULATABLE_NOW");
    let blocked = count_tensor_feasibility(obj, "BLOCKED");
    if populatable + blocked > 0 {
        f.insert(
            "populatable_tensor_count".into(),
            serde_json::json!(populatable),
        );
        f.insert("blocked_tensor_count".into(), serde_json::json!(blocked));
    }
    f
}

/// Flatten a DSTW-T4 transition JSON (MD integration, BLOCKED GDP precondition, or downstream) into
/// the DstwStateTransition flat field keys. opcode_total is the SUM of the real engine opcode COUNTS
/// (never null-padded operands).
fn flatten_dstw_transition(
    obj: &serde_json::Map<String, serde_json::Value>,
) -> serde_json::Map<String, serde_json::Value> {
    let mut f = serde_json::Map::new();
    for key in [
        "transition_id",
        "source_vertex_id",
        "target_vertex_id",
        "operator_name",
        "operator_version",
        "runtime_origin",
        "next_recommended_operator",
    ] {
        if let Some(v) = obj_get(obj, key) {
            f.insert(key.into(), v.clone());
        }
    }
    if let Some(v) = obj
        .get("validation_result")
        .and_then(|v| v.as_object())
        .and_then(|m| obj_get(m, "feasibility"))
    {
        f.insert("validation_feasibility".into(), v.clone());
    }
    if let Some(v) = obj
        .get("qnanl_receipt")
        .and_then(|v| v.as_object())
        .and_then(|m| obj_get(m, "is_opcode_trace_grounded"))
    {
        f.insert("is_opcode_trace_grounded".into(), v.clone());
    }
    for (key, field) in [
        ("changed_tensors", "changed_tensor_count"),
        ("unchanged_tensors", "unchanged_tensor_count"),
        ("failed_tensors", "failed_tensor_count"),
        ("blocked_preconditions", "blocked_precondition_count"),
    ] {
        if obj.contains_key(key) {
            f.insert(field.into(), serde_json::json!(array_len(obj, key)));
        }
    }
    if let Some(counts) = obj
        .get("opcode_operand_refs")
        .and_then(|v| v.as_object())
        .and_then(|m| m.get("opcode_counts"))
        .and_then(|v| v.as_object())
    {
        let total: u64 = counts.values().filter_map(|v| v.as_u64()).sum();
        f.insert("opcode_total".into(), serde_json::json!(total));
    }
    f
}

/// Flatten the next_operator_recommendation JSON into the DstwNextOperator flat field keys. The full
/// rank-ordered list rides the lossless sidecar (route_object capture); the rank-1 entry is lifted.
fn flatten_dstw_recommendation(
    obj: &serde_json::Map<String, serde_json::Value>,
) -> serde_json::Map<String, serde_json::Value> {
    let mut f = serde_json::Map::new();
    if let Some(v) = obj_get(obj, "runtime_origin") {
        f.insert("runtime_origin".into(), v.clone());
    }
    if let Some(v) = obj_get(obj, "runtime_native") {
        f.insert("runtime_native".into(), v.clone());
    }
    let recs = obj
        .get("priority_ordered_recommendations")
        .and_then(|v| v.as_array());
    if let Some(recs) = recs {
        f.insert("recommendation_count".into(), serde_json::json!(recs.len()));
        if let Some(top) = recs.first().and_then(|v| v.as_object()) {
            if let Some(v) = obj_get(top, "rank") {
                f.insert("top_rank".into(), v.clone());
            }
            if let Some(v) = obj_get(top, "operator") {
                f.insert("top_operator".into(), v.clone());
            }
            if let Some(v) = obj_get(top, "owner") {
                f.insert("top_owner".into(), v.clone());
            }
        }
    }
    f
}

/// Flatten a `bounded_thermo_temporal_ccns_sdst_summary.json` aggregate object into the
/// ThermoPhaseManifold flat field keys. The whole original object is preserved (so the un-lifted
/// nested dicts — phase_edge_counts / temporal_extent / condition_ledger / cells[] / event_sources[]
/// with sha256 — ride the lossless §10 sidecar via route_object capture, INV-1/INV-4); on top of
/// that the nested CCNS/SDST/phase_bucket/trigger rollups are LIFTED to flat scalar keys so
/// route_object value-matches them (matched_fields>0). The flat top-level scalars (event_count,
/// schema_version, status, target, run_id, source_surface, site_materialized_prism_therm_status) are
/// already present verbatim and matched directly. n_cells / n_event_sources are derived counts.
fn flatten_thermo_phase_manifold(
    obj: &serde_json::Map<String, serde_json::Value>,
) -> serde_json::Map<String, serde_json::Value> {
    let mut f = obj.clone();
    // Derived array-length counts (invariant cross-check anchors: 4 cells, 8 event_sources).
    f.insert("n_cells".into(), serde_json::json!(array_len(obj, "cells")));
    f.insert(
        "n_event_sources".into(),
        serde_json::json!(array_len(obj, "event_sources")),
    );
    // Lift nested `<dict>.<key>` scalars to `<prefix>_<key>` flat keys (skip absent/null).
    let lift =
        |f: &mut serde_json::Map<String, serde_json::Value>, dict: &str, key: &str, flat: &str| {
            if let Some(v) = obj
                .get(dict)
                .and_then(|v| v.as_object())
                .and_then(|m| obj_get(m, key))
            {
                f.insert(flat.into(), v.clone());
            }
        };
    lift(
        &mut f,
        "ccns",
        "cold_hold_fraction",
        "ccns_cold_hold_fraction",
    );
    lift(&mut f, "ccns", "ramp_fraction", "ccns_ramp_fraction");
    lift(
        &mut f,
        "ccns",
        "warm_hold_fraction",
        "ccns_warm_hold_fraction",
    );
    lift(&mut f, "ccns", "cooling_fraction", "ccns_cooling_fraction");
    lift(&mut f, "sdst", "cold_fraction", "sdst_cold_fraction");
    lift(&mut f, "sdst", "hot_fraction", "sdst_hot_fraction");
    lift(
        &mut f,
        "sdst",
        "delta_hot_minus_cold",
        "sdst_delta_hot_minus_cold",
    );
    lift(
        &mut f,
        "phase_bucket_counts",
        "cold_hold",
        "phase_bucket_cold_hold",
    );
    lift(&mut f, "phase_bucket_counts", "ramp", "phase_bucket_ramp");
    lift(
        &mut f,
        "trigger_counts",
        "BOCPD_changepoint",
        "trigger_bocpd_changepoint",
    );
    lift(
        &mut f,
        "trigger_counts",
        "ring_buffer_push",
        "trigger_ring_buffer_push",
    );
    lift(
        &mut f,
        "trigger_counts",
        "ring_buffer_read_and_adapt",
        "trigger_ring_buffer_read_and_adapt",
    );
    lift(
        &mut f,
        "trigger_counts",
        "cold_hold_start",
        "trigger_cold_hold_start",
    );
    lift(
        &mut f,
        "trigger_counts",
        "cold_hold_to_ramp",
        "trigger_cold_hold_to_ramp",
    );
    lift(
        &mut f,
        "trigger_counts",
        "final_teardown",
        "trigger_final_teardown",
    );
    f
}

/// PhaseManifoldEdgeValidation flat field keys ({stem}_binding_sites_lattice.json). The whole original
/// object is preserved (so `components[]` + the nested `stats`/`config`/per-component blocks ride the
/// lossless §10 sidecar via route_object capture, INV-1); on top of that the nested `stats.*`/`config.*`
/// scalars are LIFTED to flat `stats_<k>`/`config_<k>` keys and the component aggregates are DERIVED so
/// route_object value-matches the real edge counts + SO(3) coherence (matched_fields>0). The derived
/// aggregates are the by-value invariant anchors: `sum_component_n_nodes` (== stats.n_nodes),
/// `total_intra_edges` (== stats.n_directed_edges), `n_components` (== len(components)), and the
/// min/mean/max intra-component mean cosine (∈[-1,1]).
fn flatten_phase_manifold_edge_validation(
    obj: &serde_json::Map<String, serde_json::Value>,
) -> serde_json::Map<String, serde_json::Value> {
    let mut f = obj.clone();
    if let Some(stats) = obj.get("stats").and_then(|v| v.as_object()) {
        for k in [
            "n_nodes",
            "n_lattice_cells",
            "n_pairs_evaluated",
            "n_pairs_phase_legal",
            "n_pairs_aabb_overlap",
            "n_directed_edges",
            "min_step_idx",
            "max_step_idx",
            "kernel_elapsed_ms",
            "host_elapsed_ms",
        ] {
            if let Some(v) = stats.get(k) {
                f.insert(format!("stats_{k}"), v.clone());
            }
        }
    }
    if let Some(config) = obj.get("config").and_then(|v| v.as_object()) {
        for k in [
            "so3_threshold",
            "spatial_cell_size_a",
            "max_temporal_edge_steps",
            "step_bucket_size",
        ] {
            if let Some(v) = config.get(k) {
                f.insert(format!("config_{k}"), v.clone());
            }
        }
    }
    if let Some(components) = obj.get("components").and_then(|v| v.as_array()) {
        f.insert("n_components".into(), serde_json::json!(components.len()));
        let mut sum_nodes: u64 = 0;
        let mut total_intra: u64 = 0;
        let mut cos_vals: Vec<f64> = Vec::new();
        for c in components {
            if let Some(n) = c.get("n_nodes").and_then(|v| v.as_u64()) {
                sum_nodes += n;
            }
            if let Some(so3) = c.get("so3_manifold").and_then(|v| v.as_object()) {
                if let Some(e) = so3.get("n_intra_edges").and_then(|v| v.as_u64()) {
                    total_intra += e;
                }
                if let Some(cos) = so3
                    .get("intra_component_mean_cosine")
                    .and_then(|v| v.as_f64())
                {
                    if cos.is_finite() {
                        cos_vals.push(cos);
                    }
                }
            }
        }
        f.insert("sum_component_n_nodes".into(), serde_json::json!(sum_nodes));
        f.insert("total_intra_edges".into(), serde_json::json!(total_intra));
        if !cos_vals.is_empty() {
            let mean = cos_vals.iter().sum::<f64>() / cos_vals.len() as f64;
            let mn = cos_vals.iter().cloned().fold(f64::INFINITY, f64::min);
            let mx = cos_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            f.insert(
                "mean_intra_component_mean_cosine".into(),
                serde_json::json!(mean),
            );
            f.insert(
                "min_intra_component_mean_cosine".into(),
                serde_json::json!(mn),
            );
            f.insert(
                "max_intra_component_mean_cosine".into(),
                serde_json::json!(mx),
            );
        }
    }
    f
}

/// SpectralExtractor closure manifest flat field keys
/// (`DSTW_RUN_SPECIFIC_SPECTRAL_CLOSURE_MANIFEST.json`). The producer writes a nested, auditable
/// manifest because the closure owns multiple output files. The tokenizer family needs the measured
/// values, so lift only invariant-bearing summary/input/output fields into flat keys while leaving
/// the full nested manifest in the lossless sidecar.
fn flatten_spectral_extractor_manifest(
    obj: &serde_json::Map<String, serde_json::Value>,
) -> serde_json::Map<String, serde_json::Value> {
    let mut f = obj.clone();
    let copy_top = |f: &mut serde_json::Map<String, serde_json::Value>, key: &str| {
        if let Some(v) = obj_get(obj, key) {
            f.insert(key.into(), v.clone());
        }
    };
    for key in [
        "schema_version",
        "status",
        "run_root",
        "materialization_root",
        "output_root",
    ] {
        copy_top(&mut f, key);
    }

    if let Some(q) = obj.get("qnanl_source_state").and_then(|v| v.as_object()) {
        for key in [
            "spectral_basis",
            "site_rank_policy",
            "site_ranking_methodology",
        ] {
            if let Some(v) = obj_get(q, key) {
                f.insert(key.into(), v.clone());
            }
        }
    }

    if let Some(summary) = obj.get("summary").and_then(|v| v.as_object()) {
        for key in [
            "residue_count",
            "contact_edge_count",
            "materialized_site_count",
            "ranked_site_rows",
            "twin16_condition_cells",
            "twin16_pairwise_residuals",
            "twin16_nonzero_pairwise_residuals",
            "twin16_max_l1_residual",
            "blocking_issue_count",
        ] {
            if let Some(v) = obj_get(summary, key) {
                f.insert(key.into(), v.clone());
            }
        }
        if let Some(values) = summary
            .get("laplacian_eigenvalues_first_8")
            .and_then(|v| v.as_array())
        {
            for (idx, value) in values.iter().take(8).enumerate() {
                if !value.is_null() {
                    f.insert(format!("laplacian_eigenvalue_{idx}"), value.clone());
                }
            }
        }
    }

    if let Some(inputs) = obj.get("source_inputs").and_then(|v| v.as_object()) {
        let mut lift_source = |block: &str, rows_key: Option<&str>, sha_key: &str| {
            if let Some(src) = inputs.get(block).and_then(|v| v.as_object()) {
                if let (Some(rows_key), Some(rows)) = (rows_key, src.get("rows")) {
                    f.insert(rows_key.into(), rows.clone());
                }
                if let Some(sha) = src.get("sha256") {
                    f.insert(sha_key.into(), sha.clone());
                }
            }
        };
        lift_source(
            "spectral_arrow",
            Some("spectral_arrow_rows"),
            "spectral_arrow_sha256",
        );
        lift_source(
            "physical_profile",
            Some("physical_profile_rows"),
            "physical_profile_sha256",
        );
        lift_source(
            "contact_edges",
            Some("contact_edges_rows"),
            "contact_edges_sha256",
        );
        lift_source("materialized_sites", None, "materialized_sites_sha256");
        lift_source("twin16_residuals", None, "twin16_residuals_sha256");
    }

    if let Some(outputs) = obj.get("closure_outputs").and_then(|v| v.as_object()) {
        for (block, flat_key) in [
            ("residue_coordinates", "residue_coordinates_sha256"),
            ("site_overlay", "site_overlay_sha256"),
            ("site_overlay_arrow", "site_overlay_arrow_sha256"),
            ("edge_validation", "edge_validation_sha256"),
            ("twin16_condition_view", "twin16_condition_view_sha256"),
            (
                "twin16_pairwise_residuals",
                "twin16_pairwise_residuals_sha256",
            ),
            ("sha256_manifest", "sha256_manifest_sha256"),
        ] {
            if let Some(sha) = outputs
                .get(block)
                .and_then(|v| v.as_object())
                .and_then(|m| m.get("sha256"))
            {
                f.insert(flat_key.into(), sha.clone());
            }
        }
    }

    f
}

/// Harvest scalar samples per (logm/lin) field over one source's records for corpus-fit. Returns a
/// map field→values. Bounded by reading the file twice is avoided: we cap the fit sample to keep the
/// pass cheap + deterministic (first `max_fit` records, in file order).
fn harvest_fit_samples(
    enc: &Encoder,
    family: ArtifactFamily,
    records: &[serde_json::Map<String, serde_json::Value>],
) -> Vec<(&'static str, Vec<f64>)> {
    let logm_lin_fields: Vec<&'static str> = enc
        .registry()
        .iter()
        .filter(|s| {
            s.family == family
                && matches!(
                    s.geometry,
                    FieldGeometry::LogMagnitude { .. } | FieldGeometry::LinearBounded { .. }
                )
        })
        .map(|s| s.field)
        .collect();
    let mut out: Vec<(&'static str, Vec<f64>)> = Vec::new();
    for field in logm_lin_fields {
        let mut vals = Vec::new();
        for obj in records {
            if let Some(x) = obj.get(field).and_then(|v| v.as_f64()) {
                if x.is_finite() {
                    vals.push(x);
                }
            }
        }
        if !vals.is_empty() {
            out.push((field, vals));
        }
    }
    out
}

fn main() {
    let args = parse_args();
    std::fs::create_dir_all(&args.out).unwrap();
    let sources = parse_manifest(&args.manifest);

    // GPU requirement (Directive v3/v6): prefer the GPU tokenizer when compiled with --features
    // gpu-tokenizer (bit-exact parity per TOKENIZER_GPU_PARITY_LEDGER.json). MoleculeGeneration has
    // no LatticeVq3d field (coords ride sidecar), so the VQ backend only matters for any future
    // coordinate field; scalar corpus-fit is backend-independent (bit-identical).
    #[cfg(feature = "gpu-tokenizer")]
    let backend_name = "gpu-tokenizer";
    #[cfg(not(feature = "gpu-tokenizer"))]
    let backend_name = "cpu-parity-fallback";
    #[cfg(feature = "gpu-tokenizer")]
    let _vq_backend = TokenizerBackend::Gpu;
    #[cfg(not(feature = "gpu-tokenizer"))]
    let _vq_backend = TokenizerBackend::Cpu;

    let mut writers = ShardWriters::new(args.out.clone(), args.shard_size);
    let mut per_source: Vec<serde_json::Value> = Vec::new();
    let mut global_record_idx = 0usize;
    let mut total_tokens = 0usize;
    let mut total_sidecar = 0usize;
    let mut total_matched = 0usize;

    for src in &sources {
        match &src.kind {
            SourceKind::JsonlRecords {
                jsonl_records_path,
                family,
            } => {
                let records = load_jsonl_records(jsonl_records_path);

                // per-source corpus-fit: refit each logm/lin quantizer on THIS source's own distribution so
                // the magnitude ranges reflect the real data (deterministic; same input ⇒ same fit).
                let mut enc = Encoder::new();
                fit_decoded_records(&mut enc, *family, &records);

                let mut source_tokens = 0usize;
                let mut source_matched = 0usize;
                let mut source_families: BTreeMap<String, usize> = BTreeMap::new();
                for (i, record) in records.iter().enumerate() {
                    // route ONE record into a fresh per-record stream (bounded memory), then flush.
                    let mut stream = TokenStream::new();
                    let Some((family, matched)) = route_decoded_record(
                        &enc,
                        &mut stream,
                        (*family).or(record.family_hint),
                        &record.value,
                        i,
                    ) else {
                        continue;
                    };
                    source_tokens += stream.tokens.len();
                    source_matched += matched;
                    *source_families.entry(format!("{:?}", family)).or_insert(0) += 1;
                    total_tokens += stream.tokens.len();
                    total_sidecar += stream.sidecar.len();
                    total_matched += matched;
                    writers.emit_record(
                        &mut stream,
                        family,
                        &src.source_id,
                        global_record_idx,
                        &record.lineage,
                        matched,
                    );
                    global_record_idx += 1;
                }

                per_source.push(serde_json::json!({
                    "source_id": src.source_id,
                    "source_kind": "jsonl_records",
                    "jsonl_records_path": jsonl_records_path.display().to_string(),
                    "family": family.map(|family| format!("{:?}", family)).unwrap_or_else(|| "auto".to_string()),
                    "routed_families": source_families,
                    "rows": records.len(),
                    "tokens": source_tokens,
                    "matched_fields": source_matched,
                    "canonical_safe": family.map(family_is_canonical_safe),
                }));
                eprintln!(
                    "  source {}: {} JSONL rows → {} tokens ({} matched fields), family {}",
                    src.source_id,
                    records.len(),
                    source_tokens,
                    source_matched,
                    family
                        .map(|family| format!("{:?}", family))
                        .unwrap_or_else(|| "auto".to_string())
                );
            }
            SourceKind::JsonArtifact { json_path, family } => {
                let records = load_json_records(json_path);
                let mut enc = Encoder::new();
                fit_decoded_records(&mut enc, *family, &records);

                let mut source_tokens = 0usize;
                let mut source_matched = 0usize;
                let mut source_families: BTreeMap<String, usize> = BTreeMap::new();
                for (i, record) in records.iter().enumerate() {
                    let mut stream = TokenStream::new();
                    let Some((family, matched)) = route_decoded_record(
                        &enc,
                        &mut stream,
                        (*family).or(record.family_hint),
                        &record.value,
                        i,
                    ) else {
                        continue;
                    };
                    source_tokens += stream.tokens.len();
                    source_matched += matched;
                    *source_families.entry(format!("{:?}", family)).or_insert(0) += 1;
                    total_tokens += stream.tokens.len();
                    total_sidecar += stream.sidecar.len();
                    total_matched += matched;
                    writers.emit_record(
                        &mut stream,
                        family,
                        &src.source_id,
                        global_record_idx,
                        &record.lineage,
                        matched,
                    );
                    global_record_idx += 1;
                }

                per_source.push(serde_json::json!({
                    "source_id": src.source_id,
                    "source_kind": "json_artifact",
                    "json_path": json_path.display().to_string(),
                    "family": family.map(|family| format!("{:?}", family)).unwrap_or_else(|| "auto".to_string()),
                    "routed_families": source_families,
                    "rows": records.len(),
                    "tokens": source_tokens,
                    "matched_fields": source_matched,
                    "canonical_safe": family.map(family_is_canonical_safe),
                }));
                eprintln!(
                    "  source {}: {} JSON artifact records → {} tokens ({} matched fields), family {}",
                    src.source_id,
                    records.len(),
                    source_tokens,
                    source_matched,
                    family.map(|family| format!("{:?}", family)).unwrap_or_else(|| "auto".to_string())
                );
            }
            SourceKind::KccSiteJoin {
                kcc_visualization_path,
                binding_sites_path,
                prism_therm_path,
            } => {
                let mut missing_siblings = Vec::new();
                let Some(kcc_json) = read_json_file(
                    kcc_visualization_path,
                    "kcc_visualization",
                    &mut missing_siblings,
                ) else {
                    per_source.push(serde_json::json!({
                        "source_id": src.source_id,
                        "source_kind": "kcc_site_join",
                        "family": "KccSite",
                        "kcc_visualization_path": kcc_visualization_path.display().to_string(),
                        "binding_sites_path": optional_path_string(binding_sites_path),
                        "prism_therm_path": optional_path_string(prism_therm_path),
                        "rows": 0,
                        "tokens": 0,
                        "matched_fields": 0,
                        "canonical_safe": true,
                        "site_materialization_status": "KCC_SOURCE_MISSING_OR_UNREADABLE",
                        "missing_siblings": missing_siblings.clone(),
                    }));
                    eprintln!(
                        "  source {}: KCC join skipped, missing {:?}",
                        src.source_id, missing_siblings
                    );
                    continue;
                };
                let binding_json = read_optional_json_file(
                    binding_sites_path,
                    "binding_sites",
                    &mut missing_siblings,
                );
                let therm_json =
                    read_optional_json_file(prism_therm_path, "prism_therm", &mut missing_siblings);

                let sites: Vec<serde_json::Value> = kcc_json
                    .get("sites")
                    .and_then(|v| v.as_array())
                    .cloned()
                    .unwrap_or_default();
                let enc = Encoder::new();
                let mut source_tokens = 0usize;
                let mut source_matched = 0usize;
                let mut source_binding_enriched = 0usize;
                let mut source_therm_enriched = 0usize;
                let mut source_members_total = 0usize;
                let mut join_validated_records = 0usize;

                for (site_idx, site) in sites.iter().cloned().enumerate() {
                    let one_site = single_site_kcc_json(&kcc_json, site);
                    let mut stream = TokenStream::new();
                    let rep = route_kcc_join_record(
                        &enc,
                        &mut stream,
                        &one_site,
                        binding_json.as_ref(),
                        therm_json.as_ref(),
                    );
                    source_tokens += stream.tokens.len();
                    source_matched += rep.matched;
                    source_binding_enriched += rep.sites_binding_enriched;
                    source_therm_enriched += rep.members_te_enriched;
                    source_members_total += rep.members_total;
                    if rep.join_validated {
                        join_validated_records += 1;
                    }
                    total_tokens += stream.tokens.len();
                    total_sidecar += stream.sidecar.len();
                    total_matched += rep.matched;
                    let lineage = serde_json::json!({
                        "source_kind": "kcc_site_join",
                        "kcc_visualization_path": kcc_visualization_path.display().to_string(),
                        "binding_sites_path": optional_path_string(binding_sites_path),
                        "prism_therm_path": optional_path_string(prism_therm_path),
                        "site_index": site_idx,
                        "missing_siblings": missing_siblings.clone(),
                        "join_report": site_report_json(&rep),
                    });
                    writers.emit_record(
                        &mut stream,
                        ArtifactFamily::KccSite,
                        &src.source_id,
                        global_record_idx,
                        &lineage,
                        rep.matched,
                    );
                    global_record_idx += 1;
                }

                per_source.push(serde_json::json!({
                    "source_id": src.source_id,
                    "source_kind": "kcc_site_join",
                    "family": "KccSite",
                    "kcc_visualization_path": kcc_visualization_path.display().to_string(),
                    "binding_sites_path": optional_path_string(binding_sites_path),
                    "prism_therm_path": optional_path_string(prism_therm_path),
                    "rows": sites.len(),
                    "tokens": source_tokens,
                    "matched_fields": source_matched,
                    "canonical_safe": true,
                    "site_materialization_status": if sites.is_empty() { "ZERO_ROUTABLE_SITES" } else { "ROUTED_SITES" },
                    "sites_binding_enriched": source_binding_enriched,
                    "members_te_enriched": source_therm_enriched,
                    "members_total": source_members_total,
                    "join_validated_records": join_validated_records,
                    "missing_siblings": missing_siblings.clone(),
                }));
                eprintln!(
                    "  source {}: {} KCC sites → {} tokens ({} matched fields), binding joins {}, therm member joins {}",
                    src.source_id,
                    sites.len(),
                    source_tokens,
                    source_matched,
                    source_binding_enriched,
                    source_therm_enriched
                );
            }
            SourceKind::TwinPerResidue { json_path } => {
                let mut missing_siblings = Vec::new();
                let Some(twin_json) =
                    read_json_file(json_path, "twin_per_residue", &mut missing_siblings)
                else {
                    per_source.push(serde_json::json!({
                        "source_id": src.source_id,
                        "source_kind": "twin_per_residue",
                        "family": "TwinCoupled",
                        "json_path": json_path.display().to_string(),
                        "rows": 0,
                        "tokens": 0,
                        "matched_fields": 0,
                        "canonical_safe": true,
                        "twin_materialization_status": "TWIN_SOURCE_MISSING_OR_UNREADABLE",
                        "missing_siblings": missing_siblings.clone(),
                    }));
                    eprintln!(
                        "  source {}: twin_per_residue skipped, missing {:?}",
                        src.source_id, missing_siblings
                    );
                    continue;
                };
                // route the WHOLE per-residue array in one call: route_twin_per_residue fits the Cα
                // VQ on the records' own positions FIRST, then routes each residue as its own segment
                // within a single stream. One emit_record covers the artifact (mirrors a single JSON
                // artifact), with per-residue segments preserved inside the token stream.
                let mut enc = Encoder::new();
                let mut stream = TokenStream::new();
                let rep = route_twin_per_residue(&mut enc, &mut stream, &twin_json, true, SEAL);
                let source_tokens = stream.tokens.len();
                let source_matched = rep.matched;
                total_tokens += stream.tokens.len();
                total_sidecar += stream.sidecar.len();
                total_matched += rep.matched;
                let lineage = serde_json::json!({
                    "source_kind": "twin_per_residue",
                    "json_path": json_path.display().to_string(),
                    "residues": rep.records,
                    "matched": rep.matched,
                    "skipped_unknown": rep.skipped_unknown,
                    "skipped_type": rep.skipped_type,
                    "null_fields": rep.null_fields,
                    "missing_siblings": missing_siblings.clone(),
                });
                writers.emit_record(
                    &mut stream,
                    ArtifactFamily::TwinCoupled,
                    &src.source_id,
                    global_record_idx,
                    &lineage,
                    rep.matched,
                );
                global_record_idx += 1;

                per_source.push(serde_json::json!({
                    "source_id": src.source_id,
                    "source_kind": "twin_per_residue",
                    "family": "TwinCoupled",
                    "json_path": json_path.display().to_string(),
                    "rows": rep.records,
                    "tokens": source_tokens,
                    "matched_fields": source_matched,
                    "canonical_safe": true,
                    "twin_materialization_status": if rep.records == 0 { "ZERO_ROUTABLE_RESIDUES" } else { "ROUTED_RESIDUES" },
                    "missing_siblings": missing_siblings.clone(),
                }));
                eprintln!(
                    "  source {}: {} TWIN residues → {} tokens ({} matched fields)",
                    src.source_id, rep.records, source_tokens, source_matched
                );
            }
            SourceKind::DstwStateGraph {
                dstw_dir,
                operand_ledger_path,
            } => {
                let enc = Encoder::new();
                let mut rows_routed = 0usize;
                let mut rows_missing = 0usize;
                let mut source_tokens = 0usize;
                let mut source_matched = 0usize;
                let mut routed_families: BTreeMap<String, usize> = BTreeMap::new();
                let operand_ledger_present = operand_ledger_path.exists();
                for art in dstw_t4_artifacts() {
                    let json_path = dstw_dir.join(art.filename);
                    let Some(value) = read_dstw_json(&json_path) else {
                        rows_missing += 1;
                        continue;
                    };
                    let Some(obj) = value.as_object() else {
                        rows_missing += 1;
                        continue;
                    };
                    // FLAT projection: nested ligand_state/detection_mode/tensor blocks lifted to the
                    // family's flat field keys so route_object matches them (matched_fields>0).
                    let flat = match art.family {
                        ArtifactFamily::DstwStateVertex => flatten_dstw_vertex(obj),
                        ArtifactFamily::DstwStateTransition => flatten_dstw_transition(obj),
                        ArtifactFamily::DstwNextOperator => flatten_dstw_recommendation(obj),
                        ArtifactFamily::T4EpochOverlay
                        | ArtifactFamily::ChronologyAcceptanceDecision => obj.clone(),
                        _ => serde_json::Map::new(),
                    };
                    let mut stream = TokenStream::new();
                    let seg = stream.begin_record(art.family, rows_routed);
                    let rep = route_object(&enc, &mut stream, art.family, &flat, true, SEAL, true);
                    stream.end_record(seg);

                    // VALUE-verified provenance refs for the row's sidecar lineage: QNANL packet
                    // (path + sha), the PrismCode operand ledger TSV, the runtime QNANL receipt, the
                    // decode-proof, and the vertex/transition id read from the artifact itself.
                    let json_sha = sha256_file_hex(&json_path);
                    let packet_path = PathBuf::from(format!("{}.prismqnanl", json_path.display()));
                    let packet_sha = sha256_file_hex(&packet_path);
                    let receipt_path =
                        PathBuf::from(format!("{}.prismqnanl.receipt.json", json_path.display()));
                    let decode_proof_path =
                        PathBuf::from(format!("{}.decode_proof.json", json_path.display()));
                    let node_id = obj
                        .get("vertex_id")
                        .or_else(|| obj.get("transition_id"))
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    let lineage = serde_json::json!({
                        "source_kind": "dstw_state_graph",
                        "artifact_kind": art.kind,
                        "schema_kind": obj.get("schema_kind").cloned().unwrap_or(serde_json::Value::Null),
                        "json_path": json_path.display().to_string(),
                        "json_sha256": json_sha,
                        "qnanl_packet_path": packet_path.display().to_string(),
                        "qnanl_packet_sha256": packet_sha,
                        "qnanl_packet_present": packet_sha.is_some(),
                        "runtime_qnanl_receipt_path": receipt_path.display().to_string(),
                        "runtime_qnanl_receipt_present": receipt_path.exists(),
                        "decode_proof_path": decode_proof_path.display().to_string(),
                        "decode_proof_present": decode_proof_path.exists(),
                        "operand_ledger_path": operand_ledger_path.display().to_string(),
                        "operand_ledger_present": operand_ledger_present,
                        "vertex_or_transition_id": node_id,
                        "runtime_origin": obj.get("runtime_origin").cloned().unwrap_or(serde_json::Value::Null),
                        "matched": rep.matched,
                        "skipped_unknown": rep.skipped_unknown,
                        "skipped_type": rep.skipped_type,
                    });

                    source_tokens += stream.tokens.len();
                    source_matched += rep.matched;
                    total_tokens += stream.tokens.len();
                    total_sidecar += stream.sidecar.len();
                    total_matched += rep.matched;
                    *routed_families
                        .entry(format!("{:?}", art.family))
                        .or_insert(0) += 1;
                    writers.emit_record(
                        &mut stream,
                        art.family,
                        &src.source_id,
                        global_record_idx,
                        &lineage,
                        rep.matched,
                    );
                    global_record_idx += 1;
                    rows_routed += 1;
                }

                per_source.push(serde_json::json!({
                    "source_id": src.source_id,
                    "source_kind": "dstw_state_graph",
                    "dstw_dir": dstw_dir.display().to_string(),
                    "operand_ledger_path": operand_ledger_path.display().to_string(),
                    "operand_ledger_present": operand_ledger_present,
                    "routed_families": routed_families,
                    "rows": rows_routed,
                    "rows_missing_or_unreadable": rows_missing,
                    "tokens": source_tokens,
                    "matched_fields": source_matched,
                    "canonical_safe": true,
                    "dstw_materialization_status": if rows_routed == 0 { "ZERO_ROUTABLE_ARTIFACTS" } else { "ROUTED_STATE_GRAPH" },
                }));
                eprintln!(
                    "  source {}: {} DSTW-T4 state-graph rows → {} tokens ({} matched fields), {} missing",
                    src.source_id, rows_routed, source_tokens, source_matched, rows_missing
                );
            }
        }
    }

    writers.tokens.flush().ok();
    writers.segments.flush().ok();
    writers.sidecar.flush().ok();
    writers.provenance.flush().ok();
    writers.receipts.flush().ok();
    writers.lineage.flush().ok();

    let n_shards = writers.shard_idx + 1;
    // corpus name = output dir basename (so full_prism_v1 and full_prism_v2 self-identify); stable
    // per run ⇒ determinism preserved. Falls back to "full_prism" if the path has no file name.
    let corpus_name = args
        .out
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("full_prism")
        .to_string();
    let card = serde_json::json!({
        "schema_kind": "full_prism_tokenized_corpus_card",
        "corpus": corpus_name,
        "schema_version": prism_tokenizer::SCHEMA_VERSION,
        "tokenizer_backend": backend_name,
        "gpu_parity_ledger": "OPERATOR_CONTROL/TOKENIZER_GPU_PARITY_LEDGER.json",
        "shard_size": args.shard_size,
        "n_shards": n_shards,
        "n_records": global_record_idx,
        "n_tokens": total_tokens,
        "n_sidecar_entries": total_sidecar,
        "n_matched_fields": total_matched,
        "per_source": per_source,
        "shard_files": ["tokens.u64.<NNNN>.jsonl", "segments.<NNNN>.jsonl", "sidecar.<NNNN>.jsonl",
                        "provenance.<NNNN>.jsonl", "receipts.<NNNN>.jsonl", "source_lineage.<NNNN>.jsonl"],
        "determinism": "no RNG; manifest source order + sequential record index; rerun is byte-identical",
        "epistemic": "per-record epistemic_class follows the FAMILY firewall flag (INV-3): MoleculeGeneration + GflownetTrajectory are canonical_safe=false (generative firewall, never canonical evidence); SpikeEvent + TeacherStudentConsensus are canonical_safe=true (engine/label evidence). The per-record receipt records the exact class.",
    });
    std::fs::write(
        args.out.join("corpus_card.json"),
        serde_json::to_string_pretty(&card).unwrap(),
    )
    .unwrap();

    // sha256sums.txt over every emitted shard file + the card (sorted for determinism).
    let mut sums: BTreeMap<String, String> = BTreeMap::new();
    let mut all_files = writers.files_written.clone();
    all_files.push("corpus_card.json".to_string());
    all_files.sort();
    all_files.dedup();
    for f in &all_files {
        let bytes = std::fs::read(args.out.join(f)).unwrap_or_default();
        let mut h = Sha256::new();
        h.update(&bytes);
        sums.insert(f.clone(), format!("{:x}", h.finalize()));
    }
    let sums_txt: String = sums
        .iter()
        .map(|(f, h)| format!("{}  {}\n", h, f))
        .collect();
    std::fs::write(args.out.join("sha256sums.txt"), sums_txt).unwrap();

    eprintln!(
        "tok_export_full_prism_corpus: {} records, {} tokens, {} sidecar, {} shards → {}",
        global_record_idx,
        total_tokens,
        total_sidecar,
        n_shards,
        args.out.display()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_manifest(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "prism_tok_export_{name}_{}.json",
            std::process::id()
        ))
    }

    #[test]
    fn dstw_state_graph_manifest_entry_parses_with_default_ledger() {
        let p = temp_manifest("dstw_manifest");
        let manifest = serde_json::json!([
            {
                "source_kind": "DstwStateGraph",
                "source_id": "fourlpk_dstw_state_graph",
                "dstw_dir": "output/4lpk_md_evidence"
            }
        ]);
        std::fs::write(&p, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let sources = parse_manifest(&p);
        std::fs::remove_file(&p).ok();
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].source_id, "fourlpk_dstw_state_graph");
        match &sources[0].kind {
            SourceKind::DstwStateGraph {
                dstw_dir,
                operand_ledger_path,
            } => {
                assert!(dstw_dir.ends_with("4lpk_md_evidence"));
                // defaulted to <dir>/dstw_t4_operand_ledger.tsv
                assert!(operand_ledger_path.ends_with("dstw_t4_operand_ledger.tsv"));
            }
            _ => panic!("expected DstwStateGraph source kind"),
        }
    }

    #[test]
    fn dstw_artifact_table_covers_core_closeout_plus_four_downstream() {
        let arts = dstw_t4_artifacts();
        assert_eq!(
            arts.len(),
            12,
            "6 core + 2 chronology closeout + 4 conditional downstream artifacts"
        );
        // the BLOCKED holo_gdp vertex + GDP precondition transition (the blocked-precondition
        // records) are first-class rows in the table.
        assert!(arts
            .iter()
            .any(|a| a.kind == "blocked_gdp_bound_state_vertex"
                && matches!(a.family, ArtifactFamily::DstwStateVertex)));
        assert!(arts
            .iter()
            .any(|a| a.kind == "gdp_forcefield_precondition_transition"
                && matches!(a.family, ArtifactFamily::DstwStateTransition)));
        assert!(arts.iter().any(|a| a.kind == "next_operator_recommendation"
            && matches!(a.family, ArtifactFamily::DstwNextOperator)));
        assert!(arts.iter().any(|a| a.kind == "dstw_t4_epoch_overlay"
            && matches!(a.family, ArtifactFamily::T4EpochOverlay)));
        assert!(arts
            .iter()
            .any(|a| a.kind == "chronology_acceptance_decision"
                && matches!(a.family, ArtifactFamily::ChronologyAcceptanceDecision)));
    }

    #[test]
    fn flatten_dstw_vertex_lifts_nested_blocks_to_flat_keys() {
        // a realistic BLOCKED holo_gdp vertex with the nested ligand_state / detection_mode / tensor
        // blocks the emitter writes — the flattener must lift them so route_object matches them.
        let v = serde_json::json!({
            "schema_kind": "dstw_t4_scientific_state_vertex",
            "lifecycle_stage": "post_md",
            "vertex_id": "vtx_deadbeef",
            "target_id": "4lpk",
            "ligand_state": {"state": "holo_gdp", "feasibility": "BLOCKED", "gdp_present": true,
                             "input_sha256": "d9f4a653", "blocker": "HOLO_PHYSICS_INGEST_ABSENT_IN_ENGINE"},
            "detection_mode": {"feasibility": "POPULATABLE_NOW",
                               "posture": {"detection_mode": "cryptic", "site_ranker": "phase-manifold", "md_only_evidence": true}},
            "kcc_tensor": {"feasibility": "POPULATABLE_NOW"},
            "warp_tensor": {"feasibility": "BLOCKED"},
            "force_tensor": {"feasibility": "BLOCKED"},
            "missing_preconditions": [1, 2, 3],
            "qnanl_receipts": [1],
            "runtime_origin": "BLOCKED",
            "runtime_native": false
        });
        let flat = flatten_dstw_vertex(v.as_object().unwrap());
        assert_eq!(
            flat.get("ligand_state").and_then(|x| x.as_str()),
            Some("holo_gdp")
        );
        assert_eq!(
            flat.get("feasibility").and_then(|x| x.as_str()),
            Some("BLOCKED")
        );
        assert_eq!(
            flat.get("input_sha256").and_then(|x| x.as_str()),
            Some("d9f4a653")
        );
        assert_eq!(
            flat.get("detection_mode").and_then(|x| x.as_str()),
            Some("cryptic")
        );
        assert_eq!(
            flat.get("site_ranker").and_then(|x| x.as_str()),
            Some("phase-manifold")
        );
        assert_eq!(
            flat.get("missing_precondition_count")
                .and_then(|x| x.as_u64()),
            Some(3)
        );
        assert_eq!(
            flat.get("populatable_tensor_count")
                .and_then(|x| x.as_u64()),
            Some(1)
        );
        assert_eq!(
            flat.get("blocked_tensor_count").and_then(|x| x.as_u64()),
            Some(2)
        );
        // route the flat object to prove matched_fields>0.
        let enc = Encoder::new();
        let mut s = TokenStream::new();
        let r = route_object(
            &enc,
            &mut s,
            ArtifactFamily::DstwStateVertex,
            &flat,
            true,
            SEAL,
            true,
        );
        assert!(
            r.matched >= 13,
            "flattened blocked vertex matched fields (got {})",
            r.matched
        );
    }

    #[test]
    fn flatten_dstw_transition_sums_opcode_counts() {
        let t = serde_json::json!({
            "transition_id": "txn_cafe",
            "source_vertex_id": "vtx_a",
            "target_vertex_id": "vtx_b",
            "operator_name": "MD.INTEGRATE_VELOCITY_VERLET",
            "operator_version": "blake3abc",
            "runtime_origin": "RUNTIME_SUBPROCESS_ORCHESTRATED_BY_PRISMOS",
            "changed_tensors": ["coordinate_tensor", "kcc_tensor"],
            "unchanged_tensors": ["topology_tensor"],
            "failed_tensors": [{"tensor": "warp_tensor"}],
            "blocked_preconditions": [{"blocker": "x"}],
            "validation_result": {"feasibility": "POPULATABLE_NOW"},
            "qnanl_receipt": {"is_opcode_trace_grounded": false},
            "opcode_operand_refs": {"opcode_counts": {"MD.STEP": 100, "STATE.HASH": 28}}
        });
        let flat = flatten_dstw_transition(t.as_object().unwrap());
        assert_eq!(
            flat.get("changed_tensor_count").and_then(|x| x.as_u64()),
            Some(2)
        );
        assert_eq!(
            flat.get("blocked_precondition_count")
                .and_then(|x| x.as_u64()),
            Some(1)
        );
        assert_eq!(flat.get("opcode_total").and_then(|x| x.as_u64()), Some(128));
        assert_eq!(
            flat.get("validation_feasibility").and_then(|x| x.as_str()),
            Some("POPULATABLE_NOW")
        );
        let enc = Encoder::new();
        let mut s = TokenStream::new();
        let r = route_object(
            &enc,
            &mut s,
            ArtifactFamily::DstwStateTransition,
            &flat,
            true,
            SEAL,
            true,
        );
        assert!(
            r.matched >= 11,
            "flattened transition matched fields (got {})",
            r.matched
        );
    }

    #[test]
    fn next_operator_queue_row_remaps_bare_keys_to_flat_top_fields() {
        // A real NEXT_OPERATOR_QUEUE.jsonl row carries BARE keys (operator/owner/rank)
        // not the aggregate's flat top_* keys. route_decoded_record with an explicit
        // DstwNextOperator family must remap by-value so route_object matches >1 field
        // (a raw route would match only runtime_origin = forbidden shallow wrap).
        let row = serde_json::json!({
            "evidence": "md_native_runtime.rs runtime receipt path",
            "operator": "ENGINE.NATIVE_TENSOR_EMISSION",
            "owner": "engine",
            "rank": 1,
            "rationale": "wire MdNativeRuntimeState::write_receipt",
            "record_type": "next_operator",
            "runtime_origin": "RUNTIME_INLINE",
            "schema_version": 1,
            "source_artifact": "next_operator_recommendation"
        });
        let enc = Encoder::new();
        let mut stream = TokenStream::new();
        let routed = route_decoded_record(
            &enc,
            &mut stream,
            Some(ArtifactFamily::DstwNextOperator),
            &row,
            0,
        );
        let (family, matched) = routed.expect("queue row routes");
        assert_eq!(family, ArtifactFamily::DstwNextOperator);
        // runtime_origin + top_operator + top_owner + top_rank + recommendation_count.
        assert!(
            matched >= 4,
            "queue row must match the flat top_* fields by-value (got {matched})"
        );
    }

    #[test]
    fn thermo_phase_manifold_flatten_lifts_rollup_and_routes_with_matched_fields_gt_zero() {
        // Minimal fixture of the REAL bounded_thermo_temporal_ccns_sdst_summary.json shape.
        // BY-VALUE invariant: event_count == Σ cells[].event_count == Σ event_sources[].event_count
        // == phase_bucket_counts.cold_hold + .ramp.
        let summary = serde_json::json!({
            "schema_kind": "bounded_thermo_temporal_ccns_sdst_summary",
            "schema_version": 1,
            "run_id": "4LPK_apo.topology_20260701",
            "target": "4LPK_apo",
            "status": "EMITTED",
            "source_surface": "SPIKE_TRANSITION_EVENT always-on view anchors",
            "site_materialized_prism_therm_status": "materialization_required_site_or_full_prism_therm_pass",
            "event_count": 8,
            "ccns": {"surface": "bounded_transition_view", "cold_hold_fraction": 0.75, "ramp_fraction": 0.25, "warm_hold_fraction": 0.0, "cooling_fraction": 0.0},
            "sdst": {"surface": "bounded_transition_view", "cold_fraction": 0.75, "hot_fraction": 0.0, "delta_hot_minus_cold": -0.75},
            "phase_bucket_counts": {"cold_hold": 6, "ramp": 2},
            "phase_edge_counts": {"cold_hold->cold_hold": 5},
            "trigger_counts": {"BOCPD_changepoint": 2, "ring_buffer_push": 4, "ring_buffer_read_and_adapt": 4, "cold_hold_start": 1, "cold_hold_to_ramp": 1, "final_teardown": 1},
            "temporal_extent": {"min_step": 0, "max_step": 10500, "min_chunk_idx": -1, "max_chunk_idx": 20},
            "condition_ledger": {"path": "cl.json", "present": true, "sha256": "abc"},
            "cells": [{"event_count": 4}, {"event_count": 4}],
            "event_sources": [{"event_count": 4, "path": "a", "present": true, "sha256": "x"}, {"event_count": 4, "path": "b", "present": true, "sha256": "y"}],
            "missing_event_sources": []
        });

        // detect_family must recognize the schema_kind (auto/route_artifact safety net).
        assert_eq!(
            detect_family(&summary),
            Some((ArtifactFamily::ThermoPhaseManifold, ""))
        );

        // Flatten lifts the nested rollup + derived counts by value.
        let flat = flatten_thermo_phase_manifold(summary.as_object().unwrap());
        assert_eq!(flat.get("n_cells").and_then(|x| x.as_u64()), Some(2));
        assert_eq!(
            flat.get("n_event_sources").and_then(|x| x.as_u64()),
            Some(2)
        );
        assert_eq!(
            flat.get("phase_bucket_cold_hold").and_then(|x| x.as_u64()),
            Some(6)
        );
        assert_eq!(
            flat.get("trigger_ring_buffer_push")
                .and_then(|x| x.as_u64()),
            Some(4)
        );
        assert_eq!(
            flat.get("ccns_cold_hold_fraction").and_then(|x| x.as_f64()),
            Some(0.75)
        );

        // Full explicit-family route (family_from_str resolves the manifest string).
        assert_eq!(
            family_from_str("ThermoPhaseManifold"),
            Some(ArtifactFamily::ThermoPhaseManifold)
        );
        let enc = Encoder::new();
        let mut stream = TokenStream::new();
        let (family, matched) = route_decoded_record(
            &enc,
            &mut stream,
            Some(ArtifactFamily::ThermoPhaseManifold),
            &summary,
            0,
        )
        .expect("thermo summary routes");
        assert_eq!(family, ArtifactFamily::ThermoPhaseManifold);
        // event_count + schema_version + status(enum) + 4 join-key ids + n_cells + n_event_sources
        // + 4 ccns_* + 3 sdst_* + 2 phase_bucket_* + 6 trigger_* == 22 lifted flat matches.
        assert!(
            matched >= 19,
            "thermo manifold must value-match the lifted rollup (got {matched})"
        );

        // BY-VALUE invariant holds on the fixture.
        let ec = summary["event_count"].as_u64().unwrap();
        let sum_cells: u64 = summary["cells"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c["event_count"].as_u64().unwrap())
            .sum();
        let sum_srcs: u64 = summary["event_sources"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s["event_count"].as_u64().unwrap())
            .sum();
        let sum_buckets: u64 = summary["phase_bucket_counts"]
            .as_object()
            .unwrap()
            .values()
            .map(|v| v.as_u64().unwrap())
            .sum();
        assert_eq!(ec, sum_cells);
        assert_eq!(ec, sum_srcs);
        assert_eq!(ec, sum_buckets);
    }

    #[test]
    fn spectral_extractor_manifest_flatten_lifts_geometry_and_routes_by_value() {
        let manifest = serde_json::json!({
            "schema_kind": "prismos_dstw_run_specific_spectral_closure_manifest",
            "schema_version": 2,
            "status": "IMPLEMENTATION_VERIFIED",
            "run_root": "/runs/row01/run",
            "materialization_root": "/runs/row01/run/pathb_runtime_stage",
            "output_root": "/runs/spectral",
            "qnanl_source_state": {
                "spectral_basis": "normalized_contact_thermodynamic_laplacian",
                "site_rank_policy": "PRISMOS_SITE_MATERIALIZATION_LEXICOGRAPHIC_V2",
                "site_ranking_methodology": "lexicographic"
            },
            "source_inputs": {
                "spectral_arrow": {"rows": 1, "sha256": "sha_spectral_arrow"},
                "physical_profile": {"rows": 156, "sha256": "sha_physical"},
                "contact_edges": {"rows": 873, "sha256": "sha_contacts"},
                "materialized_sites": {"site_count": 20, "sha256": "sha_sites"},
                "twin16_residuals": {"condition_cell_count": 4, "pairwise_residual_count": 16, "sha256": "sha_twin16"}
            },
            "closure_outputs": {
                "residue_coordinates": {"sha256": "sha_coords"},
                "site_overlay": {"sha256": "sha_overlay"},
                "site_overlay_arrow": {"sha256": "sha_overlay_arrow"},
                "edge_validation": {"sha256": "sha_edges"},
                "twin16_condition_view": {"sha256": "sha_twin16_view"},
                "twin16_pairwise_residuals": {"sha256": "sha_twin16_pairs"},
                "sha256_manifest": {"sha256": "sha_manifest"}
            },
            "summary": {
                "residue_count": 156,
                "contact_edge_count": 873,
                "materialized_site_count": 20,
                "ranked_site_rows": 20,
                "twin16_condition_cells": 4,
                "twin16_pairwise_residuals": 16,
                "twin16_nonzero_pairwise_residuals": 12,
                "twin16_max_l1_residual": 1042.361862334741,
                "blocking_issue_count": 0,
                "laplacian_eigenvalues_first_8": [0.0, 0.0796, 0.1120, 0.1313, 0.1968, 0.2098, 0.2565, 0.2778]
            }
        });
        assert_eq!(
            detect_family(&manifest),
            Some((ArtifactFamily::SpectralExtractor, ""))
        );
        assert_eq!(
            family_from_str("SpectralExtractor"),
            Some(ArtifactFamily::SpectralExtractor)
        );
        let flat = flatten_spectral_extractor_manifest(manifest.as_object().unwrap());
        assert_eq!(
            flat.get("contact_edge_count").and_then(|x| x.as_u64()),
            Some(873)
        );
        assert_eq!(
            flat.get("physical_profile_rows").and_then(|x| x.as_u64()),
            Some(156)
        );
        assert_eq!(
            flat.get("laplacian_eigenvalue_1").and_then(|x| x.as_f64()),
            Some(0.0796)
        );
        assert_eq!(
            flat.get("site_overlay_sha256").and_then(|x| x.as_str()),
            Some("sha_overlay")
        );

        let enc = Encoder::new();
        let mut stream = TokenStream::new();
        let (family, matched) = route_decoded_record(
            &enc,
            &mut stream,
            Some(ArtifactFamily::SpectralExtractor),
            &manifest,
            0,
        )
        .expect("spectral extractor manifest routes");
        assert_eq!(family, ArtifactFamily::SpectralExtractor);
        assert!(
            matched >= 30,
            "spectral closure must route by-value; got {matched}"
        );
    }

    #[test]
    fn manifest_family_parser_accepts_all_artifact_families_used_by_v3_ledgers() {
        let cases = [
            ("SpikeEvent", ArtifactFamily::SpikeEvent),
            ("NanlSpikePacket", ArtifactFamily::NanlSpikePacket),
            ("MsmKineticSummary", ArtifactFamily::MsmKineticSummary),
            (
                "TeacherStudentConsensus",
                ArtifactFamily::TeacherStudentConsensus,
            ),
            ("FiberBundleDstw", ArtifactFamily::FiberBundleDstw),
            ("BocpdEvent", ArtifactFamily::BocpdEvent),
            ("GflownetTrajectory", ArtifactFamily::GflownetTrajectory),
            ("QnanlArtifact", ArtifactFamily::QnanlArtifact),
            ("GpuTelemetry", ArtifactFamily::GpuTelemetry),
            ("ReceiptStatus", ArtifactFamily::ReceiptStatus),
            ("HoloBoundState", ArtifactFamily::HoloBoundState),
            ("CapabilityGate", ArtifactFamily::CapabilityGate),
            ("TwinCoupled", ArtifactFamily::TwinCoupled),
            ("MoleculeGeneration", ArtifactFamily::MoleculeGeneration),
            (
                "FalsifiablePrediction",
                ArtifactFamily::FalsifiablePrediction,
            ),
            ("RankerRow", ArtifactFamily::RankerRow),
            ("KccDescriptor", ArtifactFamily::KccDescriptor),
            ("SrcCuda", ArtifactFamily::SrcCuda),
            ("SrcEngine", ArtifactFamily::SrcEngine),
            ("SrcCodec", ArtifactFamily::SrcCodec),
            ("SrcTwin", ArtifactFamily::SrcTwin),
            ("SrcGraph", ArtifactFamily::SrcGraph),
            ("SrcGflownet", ArtifactFamily::SrcGflownet),
            ("Opaque", ArtifactFamily::Opaque),
            ("KccSite", ArtifactFamily::KccSite),
            ("SpikeTransitionEvent", ArtifactFamily::SpikeTransitionEvent),
            ("BocpdLog", ArtifactFamily::BocpdLog),
            ("ProtocolState", ArtifactFamily::ProtocolState),
            ("NoiseFloor", ArtifactFamily::NoiseFloor),
            (
                "TwinTransitionResidual",
                ArtifactFamily::TwinTransitionResidual,
            ),
            ("TwinConditionEdge", ArtifactFamily::TwinConditionEdge),
            ("NmaCoupledEdge", ArtifactFamily::NmaCoupledEdge),
            ("ValidationState", ArtifactFamily::ValidationState),
            (
                "CapturedGraphGpuTelemetry",
                ArtifactFamily::CapturedGraphGpuTelemetry,
            ),
            ("KccDescriptorNative", ArtifactFamily::KccDescriptorNative),
            ("SpikeSummary", ArtifactFamily::SpikeSummary),
            ("PredictionState", ArtifactFamily::PredictionState),
            (
                "ScientificChronologyEvent",
                ArtifactFamily::ScientificChronologyEvent,
            ),
            ("ScientificStateEpoch", ArtifactFamily::ScientificStateEpoch),
            ("T4EpochOverlay", ArtifactFamily::T4EpochOverlay),
            (
                "ChronologyAcceptanceDecision",
                ArtifactFamily::ChronologyAcceptanceDecision,
            ),
            (
                "ScientificPocketOpeningEvent",
                ArtifactFamily::ScientificPocketOpeningEvent,
            ),
            (
                "LigandCausalActivityEvent",
                ArtifactFamily::LigandCausalActivityEvent,
            ),
            ("ResidueActivityEvent", ArtifactFamily::ResidueActivityEvent),
            ("SpectralExtractor", ArtifactFamily::SpectralExtractor),
            (
                "ScientificUncertaintyTensor",
                ArtifactFamily::ScientificUncertaintyTensor,
            ),
            (
                "scientific_uncertainty_tensor",
                ArtifactFamily::ScientificUncertaintyTensor,
            ),
            (
                "RuntimeNumericalUncertaintyTensor",
                ArtifactFamily::RuntimeNumericalUncertaintyTensor,
            ),
            (
                "runtime_numerical_uncertainty_tensor",
                ArtifactFamily::RuntimeNumericalUncertaintyTensor,
            ),
            (
                "ThermalUncertaintyTensor",
                ArtifactFamily::ThermalUncertaintyTensor,
            ),
            (
                "IntegratorUncertaintyTensor",
                ArtifactFamily::IntegratorUncertaintyTensor,
            ),
            (
                "ObservableUncertaintyTensor",
                ArtifactFamily::ObservableUncertaintyTensor,
            ),
            (
                "SignalToUncertaintyTensor",
                ArtifactFamily::SignalToUncertaintyTensor,
            ),
            (
                "EpochUncertaintyTensor",
                ArtifactFamily::EpochUncertaintyTensor,
            ),
            (
                "NegativeScientificEvidenceEvent",
                ArtifactFamily::NegativeScientificEvidenceEvent,
            ),
            (
                "NegativeEvidenceLedger",
                ArtifactFamily::NegativeScientificEvidenceEvent,
            ),
            (
                "FalsePositiveSuppressionLedger",
                ArtifactFamily::NegativeScientificEvidenceEvent,
            ),
            (
                "recoverable_negative_evidence_audit",
                ArtifactFamily::NegativeScientificEvidenceEvent,
            ),
            ("EpistemicUncertainty", ArtifactFamily::EpistemicUncertainty),
            ("teacher_consensus", ArtifactFamily::TeacherConsensus),
            ("THERMODYNAMIC_ACTION", ArtifactFamily::ThermodynamicAction),
        ];

        for (name, expected) in cases {
            assert_eq!(family_from_str(name), Some(expected), "{name}");
        }
        assert_eq!(family_from_str("NotARealFamily"), None);
    }

    #[test]
    fn auto_manifest_uses_uncertainty_tensor_filename_hints() {
        let p = temp_manifest("uncertainty_tensor_hint_manifest");
        let manifest = serde_json::json!([
            {
                "source_id": "sut_rows",
                "jsonl_records_path": "output/run/scientific_uncertainty_tensor.rows.jsonl",
                "family": "auto"
            },
            {
                "source_id": "runtime_num_tensor",
                "json_path": "output/run/runtime_numerical_uncertainty_tensor.json"
            },
            {
                "source_id": "epoch_tensor",
                "json_path": "output/run/epoch_uncertainty_tensor.json"
            },
            {
                "source_id": "signal_channel_tensor",
                "jsonl_records_path": "output/run/signal_to_uncertainty_channel_tensors.jsonl"
            }
        ]);
        std::fs::write(&p, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let sources = parse_manifest(&p);
        std::fs::remove_file(&p).ok();
        assert_eq!(sources.len(), 4);
        match &sources[0].kind {
            SourceKind::JsonlRecords {
                jsonl_records_path,
                family,
            } => {
                assert!(jsonl_records_path.ends_with("scientific_uncertainty_tensor.rows.jsonl"));
                assert_eq!(*family, Some(ArtifactFamily::ScientificUncertaintyTensor));
            }
            _ => panic!("expected uncertainty JSONL source"),
        }
        match &sources[1].kind {
            SourceKind::JsonArtifact { json_path, family } => {
                assert!(json_path.ends_with("runtime_numerical_uncertainty_tensor.json"));
                assert_eq!(
                    *family,
                    Some(ArtifactFamily::RuntimeNumericalUncertaintyTensor)
                );
            }
            _ => panic!("expected uncertainty JSON source"),
        }
        match &sources[2].kind {
            SourceKind::JsonArtifact { json_path, family } => {
                assert!(json_path.ends_with("epoch_uncertainty_tensor.json"));
                assert_eq!(*family, Some(ArtifactFamily::EpochUncertaintyTensor));
            }
            _ => panic!("expected epoch uncertainty JSON source"),
        }
        match &sources[3].kind {
            SourceKind::JsonlRecords {
                jsonl_records_path,
                family,
            } => {
                assert!(jsonl_records_path.ends_with("signal_to_uncertainty_channel_tensors.jsonl"));
                assert_eq!(*family, Some(ArtifactFamily::SignalToUncertaintyTensor));
            }
            _ => panic!("expected signal-to-uncertainty channel JSONL source"),
        }
    }

    #[test]
    fn nested_uncertainty_json_records_keep_schema_family_hint() {
        let p = temp_manifest("neutral_name_nested_records");
        let artifact = serde_json::json!({
            "schema_kind": "epoch_uncertainty_tensor",
            "schema_version": 1,
            "records": [{
                "row_id": "epoch_uncertainty_row_0001",
                "run_id": "run_4lpk_apo",
                "epoch_id": "epoch_0007",
                "stream_id": 0,
                "mode": "epoch",
                "tensor_kind": "epoch_uncertainty_tensor",
                "observable": "event_recurrence",
                "source_family": "ScientificChronologyEvent",
                "source_event_id": "chronology_event_0007",
                "provenance_ref": "sha256:epoch",
                "mu": 7.0,
                "sigma": 0.5,
                "variance": 0.25,
                "n_samples": 32,
                "measurement_status": "derived",
                "tokenizer_family": "EpochUncertaintyTensor"
            }]
        });
        std::fs::write(&p, serde_json::to_vec(&artifact).unwrap()).unwrap();
        let records = load_json_records(&p);
        std::fs::remove_file(&p).ok();

        assert_eq!(records.len(), 1);
        assert_eq!(
            records[0].family_hint,
            Some(ArtifactFamily::EpochUncertaintyTensor)
        );

        let enc = Encoder::new();
        let mut stream = TokenStream::new();
        let (family, matched) = route_decoded_record(
            &enc,
            &mut stream,
            records[0].family_hint,
            &records[0].value,
            0,
        )
        .expect("nested uncertainty row routes");
        assert_eq!(family, ArtifactFamily::EpochUncertaintyTensor);
        assert!(
            matched > 0,
            "nested uncertainty row must not fall back to generic JSON"
        );
    }

    #[test]
    fn kcc_site_join_manifest_entry_parses() {
        let p = temp_manifest("kcc_join_manifest");
        let manifest = serde_json::json!([
            {
                "source_kind": "KccSiteJoin",
                "source_id": "fivevex_compact_kcc_join",
                "kcc_visualization_path": "output/5VEX_coupled_nma_amp10/5VEX_chain_B_fixed.kcc_visualization.json",
                "binding_sites_path": "output/5VEX_coupled_nma_amp10/5VEX_chain_B_fixed.binding_sites.json",
                "prism_therm_path": "output/5VEX_coupled_nma_amp10/5VEX_chain_B_fixed.topology.prism_therm.json"
            }
        ]);
        std::fs::write(&p, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let sources = parse_manifest(&p);
        std::fs::remove_file(&p).ok();
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].source_id, "fivevex_compact_kcc_join");
        match &sources[0].kind {
            SourceKind::KccSiteJoin {
                kcc_visualization_path,
                binding_sites_path,
                prism_therm_path,
            } => {
                assert!(
                    kcc_visualization_path.ends_with("5VEX_chain_B_fixed.kcc_visualization.json")
                );
                assert!(binding_sites_path
                    .as_ref()
                    .unwrap()
                    .ends_with("5VEX_chain_B_fixed.binding_sites.json"));
                assert!(prism_therm_path
                    .as_ref()
                    .unwrap()
                    .ends_with("5VEX_chain_B_fixed.topology.prism_therm.json"));
            }
            _ => panic!("expected KccSiteJoin source kind"),
        }
    }

    #[test]
    fn kcc_join_helper_reports_zero_for_missing_site_materialization() {
        let enc = Encoder::new();
        let mut stream = TokenStream::new();
        let binding_status = serde_json::json!({
            "status": "evidence_emitted_not_fully_materialized",
            "binding_sites": [],
            "path_b_required": true
        });
        let rep = route_kcc_join_record(
            &enc,
            &mut stream,
            &serde_json::json!({}),
            Some(&binding_status),
            None,
        );
        assert_eq!(rep.sites, 0);
        assert_eq!(rep.sites_binding_enriched, 0);
        assert_eq!(stream.len(), 0);
    }

    #[test]
    fn decoded_control_manifest_entries_parse_json_jsonl_and_auto_family() {
        let p = temp_manifest("control_manifest");
        let manifest = serde_json::json!([
            {
                "source_id": "fivevex_bocpd_stream0",
                "jsonl_records_path": "output/5VEX_multidiff_a1/5VEX_chain_B_engine_stream0_bocpd.jsonl",
                "family": "BocpdLog"
            },
            {
                "source_id": "fivevex_protocol_stream0",
                "json_path": "output/5VEX_multidiff_a1/5VEX_chain_B_engine_stream0_protocol_state.json",
                "family": "ProtocolState"
            },
            {
                "source_id": "fivevex_noise_stream00",
                "json_path": "output/5VEX_multidiff_a1/5VEX_chain_B_engine_stream00_noise_floor.json",
                "family": "auto"
            }
        ]);
        std::fs::write(&p, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let sources = parse_manifest(&p);
        std::fs::remove_file(&p).ok();
        assert_eq!(sources.len(), 3);
        match &sources[0].kind {
            SourceKind::JsonlRecords {
                jsonl_records_path,
                family,
            } => {
                assert!(jsonl_records_path.ends_with("5VEX_chain_B_engine_stream0_bocpd.jsonl"));
                assert_eq!(*family, Some(ArtifactFamily::BocpdLog));
            }
            _ => panic!("expected BOCPD JSONL source"),
        }
        match &sources[1].kind {
            SourceKind::JsonArtifact { json_path, family } => {
                assert!(json_path.ends_with("5VEX_chain_B_engine_stream0_protocol_state.json"));
                assert_eq!(*family, Some(ArtifactFamily::ProtocolState));
            }
            _ => panic!("expected protocol JSON source"),
        }
        match &sources[2].kind {
            SourceKind::JsonArtifact { json_path, family } => {
                assert!(json_path.ends_with("5VEX_chain_B_engine_stream00_noise_floor.json"));
                assert_eq!(*family, None);
            }
            _ => panic!("expected auto-detected noise JSON source"),
        }
    }

    #[test]
    fn fivevex_control_fixtures_route_tokenize_and_stay_distinct() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let fixtures = [
            (
                ArtifactFamily::BocpdLog,
                root.join("output/5VEX_multidiff_a1/5VEX_chain_B_engine_stream0_bocpd.jsonl"),
                true,
            ),
            (
                ArtifactFamily::ProtocolState,
                root.join(
                    "output/5VEX_multidiff_a1/5VEX_chain_B_engine_stream0_protocol_state.json",
                ),
                false,
            ),
            (
                ArtifactFamily::NoiseFloor,
                root.join("output/5VEX_multidiff_a1/5VEX_chain_B_engine_stream00_noise_floor.json"),
                false,
            ),
        ];
        let missing: Vec<_> = fixtures
            .iter()
            .filter(|(_, path, _)| !path.exists())
            .map(|(_, path, _)| path.display().to_string())
            .collect();
        if !missing.is_empty() {
            eprintln!("skip: decoded control fixtures absent: {missing:?}");
            return;
        }

        for (expected_family, path, is_jsonl) in fixtures {
            let records = if is_jsonl {
                let mut records = load_jsonl_records(&path);
                records.truncate(1);
                records
            } else {
                load_json_records(&path)
            };
            assert_eq!(
                records.len(),
                1,
                "fixture smoke uses exactly one decoded record from {}",
                path.display()
            );
            assert_eq!(
                record_family(None, &records[0].value),
                Some(expected_family),
                "{} auto-detects as its control family",
                path.display()
            );
            assert_ne!(
                record_family(None, &records[0].value),
                Some(ArtifactFamily::SpikeEvent)
            );
            assert_ne!(
                record_family(None, &records[0].value),
                Some(ArtifactFamily::TwinConditionEdge)
            );
            assert_ne!(
                record_family(None, &records[0].value),
                Some(ArtifactFamily::NmaCoupledEdge)
            );

            let mut enc = Encoder::new();
            fit_decoded_records(&mut enc, Some(expected_family), &records);
            let mut stream = TokenStream::new();
            let (routed_family, matched) = route_decoded_record(
                &enc,
                &mut stream,
                Some(expected_family),
                &records[0].value,
                0,
            )
            .expect("route fixture");
            assert_eq!(routed_family, expected_family);
            assert!(
                matched > 0,
                "{} matched at least one typed field",
                path.display()
            );
            assert!(
                !stream.tokens.is_empty(),
                "{} emitted tokenized output",
                path.display()
            );
            assert!(
                stream
                    .segments
                    .iter()
                    .all(|seg| seg.family == expected_family),
                "{} emitted only the expected family",
                path.display()
            );
        }
    }
}
