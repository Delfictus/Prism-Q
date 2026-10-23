//! Runtime ScientificStateEpoch and ScientificChronologyEvent emitters.
//!
//! The writer is append-only JSONL for epoch/event streams. Manifest and
//! acceptance files are written as ordinary JSON sidecars after the runtime has
//! finished emitting the evidence available to that stage.

use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Runtime JSONL sinks for scientific chronology surfaces.
#[derive(Clone)]
pub struct ScientificChronologySinks {
    inner: Arc<ScientificChronologySinksInner>,
}

struct ScientificChronologySinksInner {
    epochs_path: PathBuf,
    events_path: PathBuf,
    manifest_path: PathBuf,
    acceptance_decision_path: PathBuf,
    state: Mutex<ScientificChronologyState>,
}

#[derive(Clone,Debug,Default,PartialEq,Eq,serde::Serialize,serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ScientificChronologyState {
    epoch_record_count: u64,
    event_record_count: u64,
    event_counts_by_kind: BTreeMap<String, u64>,
    event_counts_by_type: BTreeMap<String, u64>,
    event_counts_by_status: BTreeMap<String, u64>,
    event_counts_by_scientific_state: BTreeMap<String, u64>,
    event_counts_by_type_and_scientific_state: BTreeMap<String, u64>,
    epoch_counts_by_status: BTreeMap<String, u64>,
}

pub const EVENT_TYPE_POCKET_GEOMETRY_EVAL: &str = "POCKET_GEOMETRY_EVAL";
pub const EVENT_TYPE_SCIENTIFIC_POCKET_OPENING: &str = "SCIENTIFIC_POCKET_OPENING_EVENT";
pub const EVENT_TYPE_LIGAND_CAUSAL_ACTIVITY: &str = "LIGAND_CAUSAL_ACTIVITY_EVENT";
pub const EVENT_TYPE_RESIDUE_ACTIVITY: &str = "RESIDUE_ACTIVITY_EVENT";

pub const SCHEMA_SCIENTIFIC_POCKET_OPENING: &str = "ScientificPocketOpeningEvent";
pub const SCHEMA_LIGAND_CAUSAL_ACTIVITY: &str = "LigandCausalActivityEvent";
pub const SCHEMA_RESIDUE_ACTIVITY: &str = "ResidueActivityEvent";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LigandContactType {
    HBond,
    Hydrophobic,
    PiStack,
    SaltBridge,
    WaterBridge,
}

impl LigandContactType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::HBond => "H_BOND",
            Self::Hydrophobic => "HYDROPHOBIC",
            Self::PiStack => "PI_STACK",
            Self::SaltBridge => "SALT_BRIDGE",
            Self::WaterBridge => "WATER_BRIDGE",
        }
    }
}

impl Serialize for LigandContactType {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResidueActivityType {
    PocketLining,
    ContactSwitch,
    HingeMotion,
    AllostericSignal,
    KccSpike,
    ForceHotspot,
    ExposureChange,
}

impl ResidueActivityType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::PocketLining => "POCKET_LINING",
            Self::ContactSwitch => "CONTACT_SWITCH",
            Self::HingeMotion => "HINGE_MOTION",
            Self::AllostericSignal => "ALLOSTERIC_SIGNAL",
            Self::KccSpike => "KCC_SPIKE",
            Self::ForceHotspot => "FORCE_HOTSPOT",
            Self::ExposureChange => "EXPOSURE_CHANGE",
        }
    }
}

impl Serialize for ResidueActivityType {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct PocketLiningContact {
    pub residue_id: i64,
    pub contact_type: LigandContactType,
    pub distance_a: f64,
    pub duration_frames: Option<u64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct DistanceGate {
    pub residue_id: i64,
    pub gate_id: String,
    pub distance_a: f64,
    pub status: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct ScientificPocketOpeningEvent {
    pub epoch_id: String,
    pub frame_event_id: Option<String>,
    pub residue_ids: Option<Vec<i64>>,
    pub pocket_volume: Option<f64>,
    pub pocket_surface_area: Option<f64>,
    pub lining_contacts: Option<Vec<PocketLiningContact>>,
    pub distance_gates: Option<Vec<DistanceGate>>,
    pub persistence_frames: Option<u64>,
    pub apo_holo_contrast: Value,
    pub confidence: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct LigandContact {
    pub residue_id: i64,
    pub contact_type: LigandContactType,
    pub distance_a: f64,
    pub duration_frames: u64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct LigandCausalSignals {
    pub contact_delta: Option<f64>,
    pub pocket_volume_delta: Option<f64>,
    pub kcc_delta: Option<f64>,
    pub warp_tensor_delta: Option<f64>,
    pub force_delta: Option<f64>,
    pub residue_motion_delta: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct LigandCausalActivityEvent {
    pub epoch_id: String,
    pub frame_event_id: Option<String>,
    pub ligand_id: Option<String>,
    pub atom_ids: Option<Vec<i64>>,
    pub pose_cluster_id: Option<String>,
    pub binding_site_id: Option<String>,
    pub contacts: Option<Vec<LigandContact>>,
    pub causal_signals: LigandCausalSignals,
    pub confidence: Option<f64>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ResidueActivityMetrics {
    pub sasa_delta: Option<f64>,
    pub rmsf_delta: Option<f64>,
    pub distance_gate_delta: Option<f64>,
    pub contact_map_delta: Option<f64>,
    pub kcc_score: Option<f64>,
    pub force_norm: Option<f64>,
    pub displacement_a: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ResidueActivityEvent {
    pub epoch_id: String,
    pub frame_event_id: Option<String>,
    pub chain_id: Option<String>,
    pub residue_id: Option<i64>,
    pub residue_name: Option<String>,
    pub activity_type: Option<ResidueActivityType>,
    pub metrics: ResidueActivityMetrics,
    pub linked_ligand_event_id: Option<String>,
    pub confidence: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RequiredScientificInput {
    pub operator_term: String,
    pub source_schema_term: String,
    pub required_stage: String,
    pub reason: String,
}

impl RequiredScientificInput {
    fn new(
        operator_term: &str,
        source_schema_term: &str,
        required_stage: &str,
        reason: &str,
    ) -> Self {
        Self {
            operator_term: operator_term.to_string(),
            source_schema_term: source_schema_term.to_string(),
            required_stage: required_stage.to_string(),
            reason: reason.to_string(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ScientificEventContext {
    epoch_id: String,
    frame_event_id: Option<String>,
    event_id_base: Option<String>,
    run_id: Option<String>,
    runtime_origin: Option<String>,
    runtime_native: bool,
    emitted_unix_ns: Option<u128>,
    stream_id: Option<u32>,
    target: Option<String>,
    frame_index: Option<u64>,
    step: Option<u64>,
    physical_time_fs: Option<f64>,
    topology_path: Option<String>,
    topology_sha256: Option<String>,
    source_artifact_kind: Option<String>,
    capture_cadence_class: Option<String>,
    chronology_fidelity_level: Option<String>,
    interferometric_mode: Option<String>,
    interferometric_marker_kind: Option<String>,
    chronology_marker_scope: Option<String>,
    refs: Value,
    byte_refs: Value,
    byte_binding: Option<Value>,
}

/// A runtime-native per-stream scientific epoch boundary.
///
/// The enum prevents callers from supplying an arbitrary event kind while the
/// sink validates the stream identity and trajectory-capture contract before
/// appending either lifecycle marker.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScientificEpochBoundary {
    Open,
    Close,
}

impl ScientificEpochBoundary {
    fn event_kind(self) -> &'static str {
        match self {
            Self::Open => "EPOCH_OPEN",
            Self::Close => "EPOCH_CLOSE",
        }
    }

    fn event_id_suffix(self) -> &'static str {
        match self {
            Self::Open => "epoch_open",
            Self::Close => "epoch_close",
        }
    }

    fn lifecycle_status(self) -> &'static str {
        match self {
            Self::Open => "OPEN_BEFORE_INTEGRATION",
            Self::Close => "CLOSE_SUCCESSFUL_WORKER_RETURN",
        }
    }
}

impl ScientificEventContext {
    pub fn new(epoch_id: impl Into<String>) -> Self {
        Self {
            epoch_id: epoch_id.into(),
            frame_event_id: None,
            event_id_base: None,
            run_id: None,
            runtime_origin: None,
            runtime_native: true,
            emitted_unix_ns: None,
            stream_id: None,
            target: None,
            frame_index: None,
            step: None,
            physical_time_fs: None,
            topology_path: None,
            topology_sha256: None,
            source_artifact_kind: None,
            capture_cadence_class: None,
            chronology_fidelity_level: None,
            interferometric_mode: None,
            interferometric_marker_kind: None,
            chronology_marker_scope: None,
            refs: Value::Null,
            byte_refs: Value::Null,
            byte_binding: None,
        }
    }

    pub fn frame_event_id(mut self, value: impl Into<String>) -> Self {
        self.frame_event_id = Some(value.into());
        self
    }

    pub fn event_id_base(mut self, value: impl Into<String>) -> Self {
        self.event_id_base = Some(value.into());
        self
    }

    pub fn run_id(mut self, value: impl Into<String>) -> Self {
        self.run_id = Some(value.into());
        self
    }

    pub fn runtime_origin(mut self, value: impl Into<String>) -> Self {
        self.runtime_origin = Some(value.into());
        self
    }

    pub fn runtime_native(mut self, value: bool) -> Self {
        self.runtime_native = value;
        self
    }

    pub fn emitted_unix_ns(mut self, value: u128) -> Self {
        self.emitted_unix_ns = Some(value);
        self
    }

    pub fn stream_id(mut self, value: u32) -> Self {
        self.stream_id = Some(value);
        self
    }

    pub fn target(mut self, value: impl Into<String>) -> Self {
        self.target = Some(value.into());
        self
    }

    pub fn frame_index(mut self, value: u64) -> Self {
        self.frame_index = Some(value);
        self
    }

    pub fn step(mut self, value: u64) -> Self {
        self.step = Some(value);
        self
    }

    pub fn physical_time_fs(mut self, value: f64) -> Self {
        self.physical_time_fs = finite_f64(value);
        self
    }

    pub fn topology(mut self, path: impl Into<String>, sha256: impl Into<String>) -> Self {
        self.topology_path = Some(path.into());
        self.topology_sha256 = Some(sha256.into());
        self
    }

    pub fn source_artifact_kind(mut self, value: impl Into<String>) -> Self {
        self.source_artifact_kind = Some(value.into());
        self
    }

    pub fn capture_cadence_class(mut self, value: impl Into<String>) -> Self {
        self.capture_cadence_class = Some(value.into());
        self
    }

    pub fn chronology_fidelity_level(mut self, value: impl Into<String>) -> Self {
        self.chronology_fidelity_level = Some(value.into());
        self
    }

    pub fn interferometric_mode(mut self, value: impl Into<String>) -> Self {
        self.interferometric_mode = Some(value.into());
        self
    }

    pub fn interferometric_marker_kind(mut self, value: impl Into<String>) -> Self {
        self.interferometric_marker_kind = Some(value.into());
        self
    }

    pub fn chronology_marker_scope(mut self, value: impl Into<String>) -> Self {
        self.chronology_marker_scope = Some(value.into());
        self
    }

    pub fn refs(mut self, value: Value) -> Self {
        self.refs = value;
        self
    }

    pub fn byte_refs(mut self, value: Value) -> Self {
        self.byte_refs = value;
        self
    }

    pub fn byte_binding(mut self, value: Value) -> Self {
        self.byte_binding = Some(value);
        self
    }
}

impl ScientificChronologySinks {
    /// Build canonical chronology paths under `output_dir`.
    pub fn new(output_dir: &Path) -> Self {
        Self {
            inner: Arc::new(ScientificChronologySinksInner {
                epochs_path: output_dir.join("scientific_state_epochs.jsonl"),
                events_path: output_dir.join("scientific_chronology_events.jsonl"),
                manifest_path: output_dir.join("scientific_chronology_manifest.json"),
                acceptance_decision_path: output_dir
                    .join("scientific_chronology_acceptance_decision.json"),
                state: Mutex::new(ScientificChronologyState::default()),
            }),
        }
    }

    /// Path to `scientific_state_epochs.jsonl`.
    pub fn epochs_path(&self) -> &Path {
        &self.inner.epochs_path
    }

    /// Path to `scientific_chronology_events.jsonl`.
    pub fn events_path(&self) -> &Path {
        &self.inner.events_path
    }

    /// Path to `scientific_chronology_manifest.json`.
    pub fn manifest_path(&self) -> &Path {
        &self.inner.manifest_path
    }

    /// Path to `scientific_chronology_acceptance_decision.json`.
    pub fn acceptance_decision_path(&self) -> &Path {
        &self.inner.acceptance_decision_path
    }

    /// Initialize fresh JSONL streams for the current run.
    pub fn initialize(&self) -> Result<()> {
        create_parent(&self.inner.epochs_path)?;
        create_parent(&self.inner.events_path)?;
        truncate_file(&self.inner.epochs_path)?;
        truncate_file(&self.inner.events_path)?;
        remove_if_present(&self.inner.manifest_path)?;
        remove_if_present(&self.inner.acceptance_decision_path)?;
        let mut state = self
            .inner
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("scientific chronology lock poisoned"))?;
        *state = ScientificChronologyState::default();
        Ok(())
    }

    /// Append a ScientificStateEpoch JSONL record.
    pub fn append_epoch(&self, epoch: Value) -> Result<()> {
        let mut record = object_with_defaults(
            epoch,
            "prismos_scientific_state_epoch",
            Some("OPEN"),
            "OPEN",
        );
        if !record.get("epoch_lifecycle").is_some() {
            record["epoch_lifecycle"] = Value::String("OPEN".to_string());
        }
        normalize_epoch_aliases(&mut record);
        let status = record
            .get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("UNKNOWN")
            .to_string();
        let mut state = self
            .inner
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("scientific chronology lock poisoned"))?;
        append_jsonl_locked(&self.inner.epochs_path, &record)?;
        state.epoch_record_count = state.epoch_record_count.saturating_add(1);
        *state.epoch_counts_by_status.entry(status).or_insert(0) += 1;
        Ok(())
    }

    /// Append a ScientificChronologyEvent JSONL record.
    pub fn append_event(&self, event: Value) -> Result<()> {
        let mut record = object_with_defaults(
            event,
            "prismos_scientific_chronology_event",
            Some("PASS"),
            "EVENT",
        );
        let kind = record
            .get("event_kind")
            .or_else(|| record.get("event_type"))
            .and_then(|v| v.as_str())
            .unwrap_or("UNKNOWN")
            .to_string();
        let event_type = record
            .get("event_type")
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .unwrap_or_else(|| canonical_event_type_for_kind(&kind).to_string());
        let status = record
            .get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("UNKNOWN")
            .to_string();
        let scientific_state = record
            .get("scientific_event_state")
            .or_else(|| record.get("event_availability_status"))
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .unwrap_or_else(|| scientific_state_for_status(&status).to_string());
        if let Value::Object(map) = &mut record {
            map.entry("event_kind".to_string())
                .or_insert_with(|| Value::String(kind.clone()));
            map.entry("event_type".to_string())
                .or_insert_with(|| Value::String(event_type.clone()));
            map.entry("scientific_event_state".to_string())
                .or_insert_with(|| Value::String(scientific_state.clone()));
            if !map.contains_key("artifact") {
                if let Some(artifact) = artifact_from_byte_binding(map) {
                    map.insert("artifact".to_string(), artifact);
                }
            }
        }
        normalize_event_aliases(&mut record);
        let mut state = self
            .inner
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("scientific chronology lock poisoned"))?;
        append_jsonl_locked(&self.inner.events_path, &record)?;
        state.event_record_count = state.event_record_count.saturating_add(1);
        *state.event_counts_by_kind.entry(kind).or_insert(0) += 1;
        *state
            .event_counts_by_type
            .entry(event_type.clone())
            .or_insert(0) += 1;
        *state.event_counts_by_status.entry(status).or_insert(0) += 1;
        *state
            .event_counts_by_scientific_state
            .entry(scientific_state.clone())
            .or_insert(0) += 1;
        *state
            .event_counts_by_type_and_scientific_state
            .entry(format!("{}::{}", event_type, scientific_state))
            .or_insert(0) += 1;
        Ok(())
    }

    /// Append one actual MD-worker lifecycle boundary for one stream.
    ///
    /// This helper is deliberately fail-closed: an event is rejected before
    /// any JSONL write unless it carries an explicit runtime timestamp,
    /// per-stream identity, run identity, and a self-consistent trajectory
    /// contract. A zero save interval must report trajectory capture as
    /// disabled with null paths; a nonzero interval must bind both runtime
    /// trajectory paths. Callers emit `Open` before integration and `Close`
    /// only on the successful worker-return path.
    pub fn append_stream_epoch_boundary(
        &self,
        boundary: ScientificEpochBoundary,
        context: ScientificEventContext,
        boundary_evidence: Value,
    ) -> Result<()> {
        anyhow::ensure!(
            !context.epoch_id.trim().is_empty(),
            "scientific epoch boundary requires a nonempty epoch_id"
        );
        let run_id = required_nonempty_context_value(
            context.run_id.as_deref(),
            "scientific epoch boundary requires a nonempty run_id",
        )?;
        let runtime_origin = required_nonempty_context_value(
            context.runtime_origin.as_deref(),
            "scientific epoch boundary requires a nonempty runtime_origin",
        )?;
        anyhow::ensure!(
            context.runtime_native,
            "scientific epoch boundary must be runtime-native"
        );
        let emitted_unix_ns = context
            .emitted_unix_ns
            .context("scientific epoch boundary requires an explicit worker lifecycle timestamp")?;
        let stream_id = context
            .stream_id
            .context("scientific epoch boundary requires a stream_id")?;
        let target = required_nonempty_context_value(
            context.target.as_deref(),
            "scientific epoch boundary requires a nonempty target",
        )?;
        let evidence = boundary_evidence
            .as_object()
            .context("scientific epoch boundary evidence must be a JSON object")?;
        validate_stream_epoch_boundary_evidence(boundary, evidence)?;

        let event_kind = boundary.event_kind();
        let event_id_stem = context
            .event_id_base
            .as_deref()
            .unwrap_or(context.epoch_id.as_str());
        anyhow::ensure!(
            !event_id_stem.trim().is_empty(),
            "scientific epoch boundary requires a nonempty event id stem"
        );

        let mut record = serde_json::Map::new();
        record.insert(
            "schema_kind".to_string(),
            Value::String("prismos_scientific_chronology_event".to_string()),
        );
        record.insert("schema_version".to_string(), serde_json::json!(1));
        record.insert(
            "event_id".to_string(),
            Value::String(format!("{}_{}", event_id_stem, boundary.event_id_suffix())),
        );
        record.insert(
            "event_kind".to_string(),
            Value::String(event_kind.to_string()),
        );
        record.insert(
            "event_type".to_string(),
            Value::String(event_kind.to_string()),
        );
        record.insert("status".to_string(), Value::String("PASS".to_string()));
        record.insert("epoch_id".to_string(), Value::String(context.epoch_id));
        record.insert("run_id".to_string(), Value::String(run_id.to_string()));
        record.insert(
            "runtime_origin".to_string(),
            Value::String(runtime_origin.to_string()),
        );
        record.insert("runtime_native".to_string(), Value::Bool(true));
        record.insert(
            "emitted_unix_ns".to_string(),
            serde_json::json!(emitted_unix_ns),
        );
        record.insert("stream_id".to_string(), serde_json::json!(stream_id));
        record.insert("target".to_string(), Value::String(target.to_string()));
        insert_option_string(
            &mut record,
            "topology_path",
            context.topology_path.as_deref(),
        );
        insert_option_string(
            &mut record,
            "topology_sha256",
            context.topology_sha256.as_deref(),
        );
        insert_option_string(
            &mut record,
            "capture_cadence_class",
            context.capture_cadence_class.as_deref(),
        );
        insert_option_string(
            &mut record,
            "chronology_fidelity_level",
            context.chronology_fidelity_level.as_deref(),
        );
        insert_option_string(
            &mut record,
            "interferometric_mode",
            context.interferometric_mode.as_deref(),
        );
        insert_option_string(
            &mut record,
            "interferometric_marker_kind",
            context.interferometric_marker_kind.as_deref(),
        );
        insert_option_string(
            &mut record,
            "chronology_marker_scope",
            context.chronology_marker_scope.as_deref(),
        );
        record.insert("refs".to_string(), context.refs);
        record.insert("byte_refs".to_string(), context.byte_refs);
        record.insert(
            "boundary_evidence".to_string(),
            Value::Object(evidence.clone()),
        );
        self.append_event(Value::Object(record))
    }

    /// Emit fail-closed scientific geometry/causality rows for a captured frame.
    ///
    /// These are the exact operator-requested first-class scientific event
    /// objects. Numeric science fields remain null until a runtime stage supplies
    /// the required geometry/contrast/causality inputs.
    pub fn append_pending_frame_scientific_events(
        &self,
        context: ScientificEventContext,
    ) -> Result<()> {
        self.append_event(build_pocket_geometry_eval_pending(&context))?;
        self.append_event(build_scientific_pocket_opening_pending(&context))?;
        self.append_event(build_ligand_causal_activity_pending(&context))?;
        self.append_event(build_residue_activity_pending(&context))?;
        Ok(())
    }

    /// Attach available runtime artifact refs to the exact scientific event lane.
    ///
    /// The artifact row itself remains the source-of-truth byte binding; these
    /// rows state which required scientific event objects are still pending
    /// downstream numeric evaluation from that byte-bound surface.
    pub fn append_pending_artifact_scientific_events(
        &self,
        context: ScientificEventContext,
    ) -> Result<()> {
        let source = context.source_artifact_kind.as_deref().unwrap_or("");
        if artifact_supports_pocket_geometry(source) {
            self.append_event(build_pocket_geometry_eval_pending(&context))?;
            self.append_event(build_scientific_pocket_opening_pending(&context))?;
        }
        if artifact_supports_ligand_causality(source) {
            self.append_event(build_ligand_causal_activity_pending(&context))?;
        }
        if artifact_supports_residue_activity(source) {
            self.append_event(build_residue_activity_pending(&context))?;
        }
        Ok(())
    }

    pub fn append_scientific_pocket_opening_event(
        &self,
        context: ScientificEventContext,
        event: ScientificPocketOpeningEvent,
    ) -> Result<()> {
        self.append_event(build_scientific_pocket_opening_event(&context, event))
    }

    pub fn append_ligand_causal_activity_event(
        &self,
        context: ScientificEventContext,
        event: LigandCausalActivityEvent,
    ) -> Result<()> {
        self.append_event(build_ligand_causal_activity_event(&context, event))
    }

    pub fn append_residue_activity_event(
        &self,
        context: ScientificEventContext,
        event: ResidueActivityEvent,
    ) -> Result<()> {
        self.append_event(build_residue_activity_event(&context, event))
    }

    /// Append a closing/update ScientificStateEpoch JSONL record.
    pub fn update_epoch(&self, epoch_id: &str, patch: Value) -> Result<()> {
        let mut record = object_with_defaults(
            patch,
            "prismos_scientific_state_epoch",
            Some("UPDATE_WITHOUT_STATUS"),
            "CLOSE",
        );
        record["epoch_id"] = Value::String(epoch_id.to_string());
        if !record.get("epoch_lifecycle").is_some() {
            record["epoch_lifecycle"] = Value::String("CLOSE".to_string());
        }
        self.append_epoch(record)
    }

    /// Write the scientific chronology manifest sidecar and return it.
    pub fn write_manifest(&self, context: Value) -> Result<Value> {
        let state = self
            .inner
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("scientific chronology lock poisoned"))?;
        let manifest = serde_json::json!({
            "schema_kind": "prismos_scientific_chronology_manifest",
            "schema_version": 1,
            "status": "PASS",
            "outputs": {
                "scientific_state_epochs": file_report(&self.inner.epochs_path, state.epoch_record_count),
                "scientific_chronology_events": file_report(&self.inner.events_path, state.event_record_count),
                "scientific_chronology_manifest": path_identity_pending(&self.inner.manifest_path),
                "scientific_chronology_acceptance_decision": path_identity_or_absent(&self.inner.acceptance_decision_path),
            },
            "epoch_counts_by_status": state.epoch_counts_by_status.clone(),
            "event_counts_by_kind": state.event_counts_by_kind.clone(),
            "event_counts_by_type": state.event_counts_by_type.clone(),
            "event_counts_by_status": state.event_counts_by_status.clone(),
            "event_counts_by_scientific_state": state.event_counts_by_scientific_state.clone(),
            "event_counts_by_type_and_scientific_state": state.event_counts_by_type_and_scientific_state.clone(),
            "context": context,
        });
        write_pretty_json(&self.inner.manifest_path, &manifest)?;
        drop(state);
        Ok(manifest)
    }

    /// Write the fail-closed scientific chronology acceptance decision.
    pub fn write_acceptance_decision(
        &self,
        required_event_counts: &BTreeMap<String, u64>,
        blockers: &[String],
        context: Value,
    ) -> Result<Value> {
        let state = self
            .inner
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("scientific chronology lock poisoned"))?;
        let mut missing_required_events = Vec::new();
        for (kind, expected) in required_event_counts {
            let observed = state.event_counts_by_kind.get(kind).copied().unwrap_or(0);
            if observed < *expected {
                missing_required_events.push(serde_json::json!({
                    "event_kind": kind,
                    "expected_min_count": expected,
                    "observed_count": observed,
                    "status": "MISSING_REQUIRED_EVENT",
                }));
            }
        }
        let status = if missing_required_events.is_empty() && blockers.is_empty() {
            "PASS"
        } else {
            "FAIL"
        };
        let decision = serde_json::json!({
            "schema_kind": "prismos_scientific_chronology_acceptance_decision",
            "schema_version": 1,
            "status": status,
            "fail_closed": true,
            "required_event_counts": required_event_counts,
            "observed_event_counts_by_kind": state.event_counts_by_kind.clone(),
            "observed_event_counts_by_type": state.event_counts_by_type.clone(),
            "observed_event_counts_by_status": state.event_counts_by_status.clone(),
            "observed_event_counts_by_scientific_state": state.event_counts_by_scientific_state.clone(),
            "observed_event_counts_by_type_and_scientific_state": state.event_counts_by_type_and_scientific_state.clone(),
            "scientific_state_gate_values": ["PRODUCED", "PENDING_INPUT", "BLOCKED_POLICY"],
            "missing_required_events": missing_required_events,
            "blockers": blockers,
            "outputs": {
                "scientific_state_epochs": file_report(&self.inner.epochs_path, state.epoch_record_count),
                "scientific_chronology_events": file_report(&self.inner.events_path, state.event_record_count),
            },
            "context": context,
        });
        write_pretty_json(&self.inner.acceptance_decision_path, &decision)?;
        drop(state);
        Ok(decision)
    }
}

fn required_nonempty_context_value<'a>(
    value: Option<&'a str>,
    error_message: &str,
) -> Result<&'a str> {
    let value = value.ok_or_else(|| anyhow::anyhow!(error_message.to_string()))?;
    anyhow::ensure!(!value.trim().is_empty(), "{}", error_message);
    Ok(value)
}

fn validate_stream_epoch_boundary_evidence(
    boundary: ScientificEpochBoundary,
    evidence: &serde_json::Map<String, Value>,
) -> Result<()> {
    let lifecycle_status = evidence
        .get("worker_lifecycle_status")
        .and_then(Value::as_str)
        .context("scientific epoch boundary evidence lacks worker_lifecycle_status")?;
    anyhow::ensure!(
        lifecycle_status == boundary.lifecycle_status(),
        "scientific epoch boundary lifecycle mismatch: boundary={} status={}",
        boundary.event_kind(),
        lifecycle_status
    );

    let trajectory = evidence
        .get("trajectory_capture")
        .and_then(Value::as_object)
        .context("scientific epoch boundary evidence lacks trajectory_capture object")?;
    let enabled = trajectory
        .get("enabled")
        .and_then(Value::as_bool)
        .context("trajectory_capture lacks enabled boolean")?;
    let interval = trajectory
        .get("save_trajectory_interval")
        .and_then(Value::as_u64)
        .context("trajectory_capture lacks save_trajectory_interval")?;
    let status = trajectory
        .get("status")
        .and_then(Value::as_str)
        .context("trajectory_capture lacks status")?;
    let trajectory_path = trajectory
        .get("trajectory_path")
        .context("trajectory_capture lacks trajectory_path")?;
    let origin_path = trajectory
        .get("trajectory_origin_receipt_path")
        .context("trajectory_capture lacks trajectory_origin_receipt_path")?;

    if interval == 0 {
        anyhow::ensure!(
            !enabled,
            "zero save_trajectory_interval cannot claim trajectory capture enabled"
        );
        anyhow::ensure!(
            status == "DISABLED_SAVE_TRAJECTORY_INTERVAL_ZERO",
            "zero save_trajectory_interval requires disabled trajectory status"
        );
        anyhow::ensure!(
            trajectory_path.is_null() && origin_path.is_null(),
            "zero save_trajectory_interval requires null trajectory paths"
        );
    } else {
        anyhow::ensure!(
            enabled,
            "nonzero save_trajectory_interval requires trajectory capture enabled"
        );
        anyhow::ensure!(
            status == "ENABLED",
            "nonzero save_trajectory_interval requires enabled trajectory status"
        );
        required_nonempty_context_value(
            trajectory_path.as_str(),
            "enabled trajectory capture requires a nonempty trajectory_path",
        )?;
        required_nonempty_context_value(
            origin_path.as_str(),
            "enabled trajectory capture requires a nonempty trajectory_origin_receipt_path",
        )?;
    }
    Ok(())
}

fn canonical_event_type_for_kind(kind: &str) -> &'static str {
    match kind {
        "EPOCH_OPEN" => "EPOCH_OPEN",
        "EPOCH_CLOSE" => "EPOCH_CLOSE",
        EVENT_TYPE_POCKET_GEOMETRY_EVAL => EVENT_TYPE_POCKET_GEOMETRY_EVAL,
        SCHEMA_SCIENTIFIC_POCKET_OPENING | EVENT_TYPE_SCIENTIFIC_POCKET_OPENING => {
            EVENT_TYPE_SCIENTIFIC_POCKET_OPENING
        }
        SCHEMA_LIGAND_CAUSAL_ACTIVITY | EVENT_TYPE_LIGAND_CAUSAL_ACTIVITY => {
            EVENT_TYPE_LIGAND_CAUSAL_ACTIVITY
        }
        SCHEMA_RESIDUE_ACTIVITY | EVENT_TYPE_RESIDUE_ACTIVITY => EVENT_TYPE_RESIDUE_ACTIVITY,
        "TRAJECTORY_FRAME_CAPTURED" => "FRAME_CAPTURE",
        "SUBCHUNK_SCIENTIFIC_UPDATE_MARKER" => "SUBCHUNK_SCIENTIFIC_UPDATE_MARKER",
        "TRAJECTORY_FRAME_STREAM_EMITTED" => "TRAJECTORY_FRAME_STREAM_EMIT",
        "SPIKE_STREAM_EMITTED" | "SPIKE_TRANSITION_EVENTS_EMITTED" => "SPIKE_STREAM_EMIT",
        "SIGNAL_GRID_EMITTED" => "SIGNAL_GRID_EMIT",
        "KCC_V2FULL_EMITTED" => "KCC_V2FULL_EMIT",
        "WARP_MATRIX_EMITTED" => "WARP_TENSOR_EMIT",
        "FORCE_FIELD_EMITTED" => "FORCE_FINAL_EMIT",
        "ASC_VECTORS_EMITTED" => "ASC_VECTOR_EMIT",
        "AROMATIC_CENTROIDS_EMITTED"
        | "ADAPTIVE_DT_EMITTED"
        | "BOCPD_LOG_EMITTED"
        | "PROTOCOL_STATE_EMITTED"
        | "NOISE_FLOOR_EMITTED" => "TENSOR_CAPTURE",
        "PRISMCODE_RECEIPT_EMITTED" => "PRISMCODE_RECEIPT",
        "QNANL_PACKET_EMITTED"
        | "QNANL_SEAL_EMITTED"
        | "SPIKE_NANL_Q4_SEAL_EMITTED"
        | "SPIKE_CODEC_PACKET_EMITTED"
        | "SPIKE_CODEC_RECEIPT_EMITTED" => "QNANL_SEAL",
        "TOKENIZER_HANDOFF_DECLARED" => "TOKENIZER_EXPORT",
        "TRAJECTORY_AUDIT_EMITTED" | "TRAJECTORY_ORIGIN_RECEIPT_EMITTED" => "VALIDATION_PASS",
        "TRAJECTORY_AUDIT_BLOCKER" => "BLOCKER_RAISED",
        _ => "TENSOR_CAPTURE",
    }
}

fn build_pocket_geometry_eval_pending(ctx: &ScientificEventContext) -> Value {
    let required_inputs = pocket_geometry_required_inputs();
    let mut record = base_scientific_event(
        ctx,
        EVENT_TYPE_POCKET_GEOMETRY_EVAL,
        EVENT_TYPE_POCKET_GEOMETRY_EVAL,
        "pocket_geometry_eval",
        &geometry_status_for_context(ctx),
        required_inputs,
    );
    insert_value(&mut record, "residue_ids", Value::Null);
    insert_value(&mut record, "pocket_volume", Value::Null);
    insert_value(&mut record, "pocket_surface_area", Value::Null);
    insert_value(&mut record, "lining_contacts", Value::Null);
    insert_value(&mut record, "distance_gates", Value::Null);
    insert_value(&mut record, "persistence_frames", Value::Null);
    insert_value(
        &mut record,
        "apo_holo_contrast",
        serde_json::json!({
            "status": "PENDING_SPECTRAL_EXTRACTOR_INPUTS",
            "operator_term": "apo_holo_contrast",
            "source_schema_terms": [
                "spectral_extractor.apo_holo_contrast",
                "spectral_fiber_bundle.apo_holo_contrast",
                "path_b.binding_site_physiology_surface.apo_holo_contrast"
            ],
        }),
    );
    insert_value(&mut record, "confidence", Value::Null);
    insert_value(
        &mut record,
        "geometry_eval_role",
        Value::String("required chain stage before ScientificPocketOpeningEvent".to_string()),
    );
    record
}

fn build_scientific_pocket_opening_pending(ctx: &ScientificEventContext) -> Value {
    let event = ScientificPocketOpeningEvent {
        epoch_id: ctx.epoch_id.clone(),
        frame_event_id: ctx.frame_event_id.clone(),
        residue_ids: None,
        pocket_volume: None,
        pocket_surface_area: None,
        lining_contacts: None,
        distance_gates: None,
        persistence_frames: None,
        apo_holo_contrast: serde_json::json!({
            "status": "PENDING_SPECTRAL_EXTRACTOR_INPUTS",
            "operator_term": "apo_holo_contrast",
            "source_schema_terms": [
                "spectral_extractor.apo_holo_contrast",
                "spectral_fiber_bundle.apo_holo_contrast",
                "path_b.binding_site_physiology_surface.apo_holo_contrast"
            ],
        }),
        confidence: None,
    };
    build_scientific_pocket_opening_with_status(
        ctx,
        event,
        geometry_status_for_context(ctx),
        pocket_geometry_required_inputs(),
    )
}

fn build_ligand_causal_activity_pending(ctx: &ScientificEventContext) -> Value {
    let event = LigandCausalActivityEvent {
        epoch_id: ctx.epoch_id.clone(),
        frame_event_id: ctx.frame_event_id.clone(),
        ligand_id: None,
        atom_ids: None,
        pose_cluster_id: None,
        binding_site_id: None,
        contacts: None,
        causal_signals: LigandCausalSignals::default(),
        confidence: None,
    };
    build_ligand_causal_activity_with_status(
        ctx,
        event,
        geometry_status_for_context(ctx),
        ligand_causality_required_inputs(),
    )
}

fn build_residue_activity_pending(ctx: &ScientificEventContext) -> Value {
    let event = ResidueActivityEvent {
        epoch_id: ctx.epoch_id.clone(),
        frame_event_id: ctx.frame_event_id.clone(),
        chain_id: None,
        residue_id: None,
        residue_name: None,
        activity_type: None,
        metrics: ResidueActivityMetrics::default(),
        linked_ligand_event_id: None,
        confidence: None,
    };
    build_residue_activity_with_status(
        ctx,
        event,
        geometry_status_for_context(ctx),
        residue_activity_required_inputs(),
    )
}

fn build_scientific_pocket_opening_event(
    ctx: &ScientificEventContext,
    event: ScientificPocketOpeningEvent,
) -> Value {
    let mut missing = missing_pocket_inputs(&event);
    let mut status = if missing.is_empty() {
        produced_or_geometry_blocked_status(ctx)
    } else {
        geometry_status_for_context(ctx)
    };
    if status == "PASS" {
        missing.clear();
    }
    let required_inputs = if status == "PASS" {
        Vec::new()
    } else if missing.is_empty() {
        pocket_geometry_required_inputs()
    } else {
        missing
    };
    build_scientific_pocket_opening_with_status(ctx, event, status, required_inputs)
}

fn build_ligand_causal_activity_event(
    ctx: &ScientificEventContext,
    event: LigandCausalActivityEvent,
) -> Value {
    let mut missing = missing_ligand_inputs(&event);
    let status = if missing.is_empty() {
        produced_or_geometry_blocked_status(ctx)
    } else {
        geometry_status_for_context(ctx)
    };
    if status == "PASS" {
        missing.clear();
    }
    let required_inputs = if status == "PASS" {
        Vec::new()
    } else if missing.is_empty() {
        ligand_causality_required_inputs()
    } else {
        missing
    };
    build_ligand_causal_activity_with_status(ctx, event, status, required_inputs)
}

fn build_residue_activity_event(
    ctx: &ScientificEventContext,
    event: ResidueActivityEvent,
) -> Value {
    let mut missing = missing_residue_inputs(&event);
    let status = if missing.is_empty() {
        produced_or_geometry_blocked_status(ctx)
    } else {
        geometry_status_for_context(ctx)
    };
    if status == "PASS" {
        missing.clear();
    }
    let required_inputs = if status == "PASS" {
        Vec::new()
    } else if missing.is_empty() {
        residue_activity_required_inputs()
    } else {
        missing
    };
    build_residue_activity_with_status(ctx, event, status, required_inputs)
}

fn build_scientific_pocket_opening_with_status(
    ctx: &ScientificEventContext,
    event: ScientificPocketOpeningEvent,
    status: &'static str,
    required_inputs: Vec<RequiredScientificInput>,
) -> Value {
    let mut record = base_scientific_event(
        ctx,
        SCHEMA_SCIENTIFIC_POCKET_OPENING,
        EVENT_TYPE_SCIENTIFIC_POCKET_OPENING,
        "scientific_pocket_opening",
        status,
        required_inputs,
    );
    insert_value(
        &mut record,
        "operator_event_object",
        Value::String(SCHEMA_SCIENTIFIC_POCKET_OPENING.to_string()),
    );
    merge_payload(&mut record, event_value(event));
    record
}

fn build_ligand_causal_activity_with_status(
    ctx: &ScientificEventContext,
    event: LigandCausalActivityEvent,
    status: &'static str,
    required_inputs: Vec<RequiredScientificInput>,
) -> Value {
    let mut record = base_scientific_event(
        ctx,
        SCHEMA_LIGAND_CAUSAL_ACTIVITY,
        EVENT_TYPE_LIGAND_CAUSAL_ACTIVITY,
        "ligand_causal_activity",
        status,
        required_inputs,
    );
    insert_value(
        &mut record,
        "operator_event_object",
        Value::String(SCHEMA_LIGAND_CAUSAL_ACTIVITY.to_string()),
    );
    merge_payload(&mut record, event_value(event));
    record
}

fn build_residue_activity_with_status(
    ctx: &ScientificEventContext,
    event: ResidueActivityEvent,
    status: &'static str,
    required_inputs: Vec<RequiredScientificInput>,
) -> Value {
    let mut record = base_scientific_event(
        ctx,
        SCHEMA_RESIDUE_ACTIVITY,
        EVENT_TYPE_RESIDUE_ACTIVITY,
        "residue_activity",
        status,
        required_inputs,
    );
    insert_value(
        &mut record,
        "operator_event_object",
        Value::String(SCHEMA_RESIDUE_ACTIVITY.to_string()),
    );
    merge_payload(&mut record, event_value(event));
    record
}

fn base_scientific_event(
    ctx: &ScientificEventContext,
    event_kind: &str,
    event_type: &str,
    event_id_suffix: &str,
    status: &'static str,
    required_inputs: Vec<RequiredScientificInput>,
) -> Value {
    let event_id_stem = ctx
        .event_id_base
        .as_deref()
        .or(ctx.frame_event_id.as_deref())
        .unwrap_or(ctx.epoch_id.as_str());
    let mut obj = serde_json::Map::new();
    obj.insert(
        "schema_kind".to_string(),
        Value::String("prismos_scientific_chronology_event".to_string()),
    );
    obj.insert("schema_version".to_string(), serde_json::json!(1));
    obj.insert(
        "event_id".to_string(),
        Value::String(format!("{}_{}", event_id_stem, event_id_suffix)),
    );
    obj.insert(
        "event_kind".to_string(),
        Value::String(event_kind.to_string()),
    );
    obj.insert(
        "event_type".to_string(),
        Value::String(event_type.to_string()),
    );
    obj.insert("status".to_string(), Value::String(status.to_string()));
    obj.insert(
        "scientific_event_state".to_string(),
        Value::String(scientific_state_for_status(status).to_string()),
    );
    obj.insert("epoch_id".to_string(), Value::String(ctx.epoch_id.clone()));
    obj.insert(
        "runtime_origin".to_string(),
        Value::String(
            ctx.runtime_origin
                .clone()
                .unwrap_or_else(|| "RUNTIME_INLINE".to_string()),
        ),
    );
    obj.insert(
        "runtime_native".to_string(),
        Value::Bool(ctx.runtime_native),
    );
    obj.insert(
        "emitted_unix_ns".to_string(),
        serde_json::json!(ctx.emitted_unix_ns.unwrap_or_else(current_unix_ns)),
    );
    insert_option_string(&mut obj, "frame_event_id", ctx.frame_event_id.as_deref());
    insert_option_string(&mut obj, "run_id", ctx.run_id.as_deref());
    if let Some(stream_id) = ctx.stream_id {
        obj.insert("stream_id".to_string(), serde_json::json!(stream_id));
    }
    insert_option_string(&mut obj, "target", ctx.target.as_deref());
    if let Some(frame_index) = ctx.frame_index {
        obj.insert("frame_index".to_string(), serde_json::json!(frame_index));
    }
    if let Some(step) = ctx.step {
        obj.insert("step".to_string(), serde_json::json!(step));
        obj.insert(
            "step_binding_status".to_string(),
            Value::String("BOUND".to_string()),
        );
    } else {
        obj.insert(
            "step_binding_status".to_string(),
            Value::String("PENDING_SOURCE_ARTIFACT_TEMPORAL_INDEX".to_string()),
        );
    }
    if let Some(physical_time_fs) = ctx.physical_time_fs {
        obj.insert(
            "physical_time_fs".to_string(),
            serde_json::json!(physical_time_fs),
        );
        obj.insert(
            "physical_time_binding_status".to_string(),
            Value::String("BOUND".to_string()),
        );
    } else {
        obj.insert(
            "physical_time_binding_status".to_string(),
            Value::String("PENDING_SOURCE_ARTIFACT_TEMPORAL_INDEX".to_string()),
        );
    }
    insert_option_string(&mut obj, "topology_path", ctx.topology_path.as_deref());
    insert_option_string(&mut obj, "topology_sha256", ctx.topology_sha256.as_deref());
    insert_option_string(
        &mut obj,
        "source_artifact_kind",
        ctx.source_artifact_kind.as_deref(),
    );
    insert_option_string(
        &mut obj,
        "capture_cadence_class",
        ctx.capture_cadence_class.as_deref(),
    );
    insert_option_string(
        &mut obj,
        "chronology_fidelity_level",
        ctx.chronology_fidelity_level.as_deref(),
    );
    insert_option_string(
        &mut obj,
        "interferometric_mode",
        ctx.interferometric_mode.as_deref(),
    );
    insert_option_string(
        &mut obj,
        "interferometric_marker_kind",
        ctx.interferometric_marker_kind.as_deref(),
    );
    insert_option_string(
        &mut obj,
        "chronology_marker_scope",
        ctx.chronology_marker_scope.as_deref(),
    );
    obj.insert("refs".to_string(), ctx.refs.clone());
    obj.insert("byte_refs".to_string(), ctx.byte_refs.clone());
    if let Some(byte_binding) = &ctx.byte_binding {
        obj.insert("byte_binding".to_string(), byte_binding.clone());
    }
    obj.insert(
        "required_chain".to_string(),
        serde_json::json!([
            "FRAME_CAPTURE",
            EVENT_TYPE_POCKET_GEOMETRY_EVAL,
            EVENT_TYPE_SCIENTIFIC_POCKET_OPENING,
            "T4_EPOCH_OVERLAY",
            "SITE_RANKING"
        ]),
    );
    obj.insert(
        "geometry_authority_order".to_string(),
        geometry_authority_order(),
    );
    obj.insert(
        "geometry_authority_status".to_string(),
        Value::String(geometry_status_for_context(ctx).to_string()),
    );
    obj.insert(
        "operator_source_field_mapping".to_string(),
        operator_source_field_mapping(),
    );
    obj.insert(
        "allowed_contact_type_values".to_string(),
        serde_json::json!([
            "H_BOND",
            "HYDROPHOBIC",
            "PI_STACK",
            "SALT_BRIDGE",
            "WATER_BRIDGE"
        ]),
    );
    obj.insert(
        "allowed_activity_type_values".to_string(),
        serde_json::json!([
            "POCKET_LINING",
            "CONTACT_SWITCH",
            "HINGE_MOTION",
            "ALLOSTERIC_SIGNAL",
            "KCC_SPIKE",
            "FORCE_HOTSPOT",
            "EXPOSURE_CHANGE"
        ]),
    );
    obj.insert(
        "missing_required_inputs".to_string(),
        serde_json::to_value(required_inputs).unwrap_or_else(|_| Value::Array(Vec::new())),
    );
    Value::Object(obj)
}

fn event_value<T: Serialize>(event: T) -> Value {
    serde_json::to_value(event).unwrap_or_else(|err| {
        serde_json::json!({
            "serialization_status": "BLOCKED_POLICY",
            "serialization_error": err.to_string(),
        })
    })
}

fn merge_payload(record: &mut Value, payload: Value) {
    let (Some(record_obj), Some(payload_obj)) = (record.as_object_mut(), payload.as_object())
    else {
        return;
    };
    for (key, value) in payload_obj {
        record_obj.insert(key.clone(), value.clone());
    }
}

fn insert_value(record: &mut Value, key: &str, value: Value) {
    if let Some(obj) = record.as_object_mut() {
        obj.insert(key.to_string(), value);
    }
}

fn insert_option_string(obj: &mut serde_json::Map<String, Value>, key: &str, value: Option<&str>) {
    if let Some(value) = value {
        obj.insert(key.to_string(), Value::String(value.to_string()));
    }
}

fn geometry_authority_order() -> Value {
    serde_json::json!([
        {
            "rank": 1,
            "operator_term": "spectral extractor",
            "source_schema_term": "SPECTRAL_EXTRACTOR",
            "authority": "highest_fidelity_geometry_adjudication"
        },
        {
            "rank": 2,
            "operator_term": "spectral fiber bundle",
            "source_schema_term": "SPECTRAL_FIBER_BUNDLE",
            "authority": "highest_fidelity_geometry_adjudication"
        },
        {
            "rank": 3,
            "operator_term": "Path-B pocket morphology",
            "source_schema_term": "prismos_geometric_pocket_morphology_surface",
            "authority": "candidate_site_context_not_authoritative_without_spectral_geometry"
        },
        {
            "rank": 4,
            "operator_term": "Path-B binding-site physiology",
            "source_schema_term": "prismos_binding_site_physiology_surface",
            "authority": "candidate_site_context_not_authoritative_without_spectral_geometry"
        }
    ])
}

fn operator_source_field_mapping() -> Value {
    serde_json::json!({
        "pocket_volume": {
            "operator_term": "pocket_volume",
            "source_schema_terms": [
                "spectral_extractor.pocket_volume_a3",
                "spectral_fiber_bundle.pocket_volume_a3",
                "pocket_morphology_surface.pocket_volume"
            ],
            "unit": "angstrom^3",
            "authority_note": "spectral extractor / fiber bundle must be preferred before Path-B morphology"
        },
        "pocket_surface_area": {
            "operator_term": "pocket_surface_area",
            "source_schema_terms": [
                "spectral_extractor.pocket_surface_area_a2",
                "spectral_fiber_bundle.pocket_surface_area_a2",
                "pocket_morphology_surface.pocket_surface_area"
            ],
            "unit": "angstrom^2"
        },
        "apo_holo_contrast": {
            "operator_term": "apo_holo_contrast",
            "source_schema_terms": [
                "spectral_extractor.apo_holo_contrast_status",
                "spectral_fiber_bundle.apo_holo_contrast_status",
                "binding_site_physiology_surface.apo_holo_contrast"
            ]
        },
        "causal_signals.kcc_delta": {
            "operator_term": "kcc_delta",
            "source_schema_terms": ["kcc_v2full", "spectral_fiber_bundle.kcc_delta"]
        },
        "causal_signals.warp_tensor_delta": {
            "operator_term": "warp_tensor_delta",
            "source_schema_terms": ["warp_matrix", "spectral_fiber_bundle.warp_tensor_delta"]
        },
        "causal_signals.force_delta": {
            "operator_term": "force_delta",
            "source_schema_terms": ["forces_final", "spectral_fiber_bundle.force_delta"]
        },
        "metrics.force_norm": {
            "operator_term": "force_norm",
            "source_schema_terms": ["forces_final.force_norm", "spectral_fiber_bundle.force_norm"]
        },
        "metrics.displacement_a": {
            "operator_term": "displacement_a",
            "source_schema_terms": [
                "trajectory_frames.displacement_a",
                "spectral_fiber_bundle.displacement_a"
            ]
        }
    })
}

fn pocket_geometry_required_inputs() -> Vec<RequiredScientificInput> {
    vec![
        RequiredScientificInput::new(
            "spectral extractor",
            "SPECTRAL_EXTRACTOR",
            "spectral geometry adjudication",
            "highest-fidelity geometry authority is absent",
        ),
        RequiredScientificInput::new(
            "spectral fiber bundle",
            "SPECTRAL_FIBER_BUNDLE",
            "spectral geometry adjudication",
            "highest-fidelity geometry authority is absent",
        ),
        RequiredScientificInput::new(
            "pocket_volume",
            "spectral_extractor.pocket_volume_a3",
            "POCKET_GEOMETRY_EVAL",
            "required numeric pocket volume is not exposed at this runtime boundary",
        ),
        RequiredScientificInput::new(
            "pocket_surface_area",
            "spectral_extractor.pocket_surface_area_a2",
            "POCKET_GEOMETRY_EVAL",
            "required numeric pocket surface area is not exposed at this runtime boundary",
        ),
        RequiredScientificInput::new(
            "apo_holo_contrast",
            "spectral_extractor.apo_holo_contrast_status",
            "POCKET_GEOMETRY_EVAL",
            "apo/holo contrast requires spectral or Path-B multi-state inputs",
        ),
    ]
}

fn ligand_causality_required_inputs() -> Vec<RequiredScientificInput> {
    let mut inputs = pocket_geometry_required_inputs();
    inputs.extend([
        RequiredScientificInput::new(
            "ligand_id",
            "topology_ligand_state.ligand_id",
            "topology/ligand state materialization",
            "PrismPrepTopology does not expose ligand truth to this frame hook",
        ),
        RequiredScientificInput::new(
            "pose_cluster_id",
            "spectral_fiber_bundle.pose_cluster_id",
            "spectral ligand pose clustering",
            "pose clusters require spectral/Path-B ligand context",
        ),
        RequiredScientificInput::new(
            "contacts",
            "spectral_fiber_bundle.contacts",
            "ligand contact adjudication",
            "typed ligand contacts require ligand atoms and residue contact durations",
        ),
    ]);
    inputs
}

fn residue_activity_required_inputs() -> Vec<RequiredScientificInput> {
    let mut inputs = pocket_geometry_required_inputs();
    inputs.extend([
        RequiredScientificInput::new(
            "activity_type",
            "residue_activity.activity_type",
            "residue activity classification",
            "activity classification requires spectral/Path-B residue features",
        ),
        RequiredScientificInput::new(
            "metrics",
            "residue_activity.metrics",
            "residue activity metric extraction",
            "sasa/rmsf/contact-map/KCC/force/displacement metrics are not all exposed at this boundary",
        ),
    ]);
    inputs
}

fn missing_pocket_inputs(event: &ScientificPocketOpeningEvent) -> Vec<RequiredScientificInput> {
    let mut missing = Vec::new();
    if event
        .residue_ids
        .as_ref()
        .map(Vec::is_empty)
        .unwrap_or(true)
    {
        missing.push(RequiredScientificInput::new(
            "residue_ids",
            "spectral_fiber_bundle.residue_ids",
            "POCKET_GEOMETRY_EVAL",
            "pocket opening residue support missing",
        ));
    }
    if event.pocket_volume.is_none() {
        missing.push(RequiredScientificInput::new(
            "pocket_volume",
            "spectral_extractor.pocket_volume_a3",
            "POCKET_GEOMETRY_EVAL",
            "pocket volume missing",
        ));
    }
    if event.pocket_surface_area.is_none() {
        missing.push(RequiredScientificInput::new(
            "pocket_surface_area",
            "spectral_extractor.pocket_surface_area_a2",
            "POCKET_GEOMETRY_EVAL",
            "pocket surface area missing",
        ));
    }
    if event
        .lining_contacts
        .as_ref()
        .map(Vec::is_empty)
        .unwrap_or(true)
    {
        missing.push(RequiredScientificInput::new(
            "lining_contacts",
            "spectral_fiber_bundle.lining_contacts",
            "POCKET_GEOMETRY_EVAL",
            "lining contact set missing",
        ));
    }
    if event
        .distance_gates
        .as_ref()
        .map(Vec::is_empty)
        .unwrap_or(true)
    {
        missing.push(RequiredScientificInput::new(
            "distance_gates",
            "spectral_fiber_bundle.distance_gates",
            "POCKET_GEOMETRY_EVAL",
            "distance gates missing",
        ));
    }
    if event.persistence_frames.is_none() {
        missing.push(RequiredScientificInput::new(
            "persistence_frames",
            "spectral_fiber_bundle.persistence_frames",
            "T4_EPOCH_OVERLAY",
            "persistence requires multiple frame observations",
        ));
    }
    if event.apo_holo_contrast.is_null() {
        missing.push(RequiredScientificInput::new(
            "apo_holo_contrast",
            "spectral_extractor.apo_holo_contrast_status",
            "POCKET_GEOMETRY_EVAL",
            "apo/holo contrast missing",
        ));
    }
    if event.confidence.is_none() {
        missing.push(RequiredScientificInput::new(
            "confidence",
            "spectral_fiber_bundle.confidence",
            "SITE_RANKING",
            "confidence missing",
        ));
    }
    missing
}

fn missing_ligand_inputs(event: &LigandCausalActivityEvent) -> Vec<RequiredScientificInput> {
    let mut missing = Vec::new();
    if event.ligand_id.as_deref().unwrap_or("").is_empty() {
        missing.push(RequiredScientificInput::new(
            "ligand_id",
            "topology_ligand_state.ligand_id",
            "topology/ligand state materialization",
            "ligand id missing",
        ));
    }
    if event.atom_ids.as_ref().map(Vec::is_empty).unwrap_or(true) {
        missing.push(RequiredScientificInput::new(
            "atom_ids",
            "topology_ligand_state.atom_ids",
            "topology/ligand state materialization",
            "ligand atom ids missing",
        ));
    }
    if event.pose_cluster_id.as_deref().unwrap_or("").is_empty() {
        missing.push(RequiredScientificInput::new(
            "pose_cluster_id",
            "spectral_fiber_bundle.pose_cluster_id",
            "spectral ligand pose clustering",
            "pose cluster id missing",
        ));
    }
    if event.binding_site_id.as_deref().unwrap_or("").is_empty() {
        missing.push(RequiredScientificInput::new(
            "binding_site_id",
            "spectral_fiber_bundle.binding_site_id",
            "SITE_RANKING",
            "binding site id missing",
        ));
    }
    if event.contacts.as_ref().map(Vec::is_empty).unwrap_or(true) {
        missing.push(RequiredScientificInput::new(
            "contacts",
            "spectral_fiber_bundle.contacts",
            "ligand contact adjudication",
            "ligand contacts missing",
        ));
    }
    if event.confidence.is_none() {
        missing.push(RequiredScientificInput::new(
            "confidence",
            "spectral_fiber_bundle.confidence",
            "SITE_RANKING",
            "confidence missing",
        ));
    }
    missing
}

fn missing_residue_inputs(event: &ResidueActivityEvent) -> Vec<RequiredScientificInput> {
    let mut missing = Vec::new();
    if event.chain_id.as_deref().unwrap_or("").is_empty() {
        missing.push(RequiredScientificInput::new(
            "chain_id",
            "topology.chain_ids",
            "topology binding",
            "chain id missing",
        ));
    }
    if event.residue_id.is_none() {
        missing.push(RequiredScientificInput::new(
            "residue_id",
            "topology.residue_ids",
            "topology binding",
            "residue id missing",
        ));
    }
    if event.residue_name.as_deref().unwrap_or("").is_empty() {
        missing.push(RequiredScientificInput::new(
            "residue_name",
            "topology.residue_names",
            "topology binding",
            "residue name missing",
        ));
    }
    if event.activity_type.is_none() {
        missing.push(RequiredScientificInput::new(
            "activity_type",
            "residue_activity.activity_type",
            "residue activity classification",
            "activity type missing",
        ));
    }
    if event.confidence.is_none() {
        missing.push(RequiredScientificInput::new(
            "confidence",
            "spectral_fiber_bundle.confidence",
            "SITE_RANKING",
            "confidence missing",
        ));
    }
    missing
}

fn produced_or_geometry_blocked_status(ctx: &ScientificEventContext) -> &'static str {
    if context_has_spectral_geometry(ctx) {
        "PASS"
    } else if context_has_pathb_geometry(ctx) {
        "BLOCKED_MISSING_SPECTRAL_GEOMETRY"
    } else {
        "PENDING_SPECTRAL_EXTRACTOR_INPUTS"
    }
}

fn geometry_status_for_context(ctx: &ScientificEventContext) -> &'static str {
    if context_has_spectral_geometry(ctx) {
        "PENDING_SPECTRAL_NUMERIC_EXTRACTION"
    } else if context_has_pathb_geometry(ctx) {
        "BLOCKED_MISSING_SPECTRAL_GEOMETRY"
    } else {
        "PENDING_SPECTRAL_EXTRACTOR_INPUTS"
    }
}

fn context_has_spectral_geometry(ctx: &ScientificEventContext) -> bool {
    ctx.source_artifact_kind
        .as_deref()
        .map(artifact_supports_spectral_geometry)
        .unwrap_or(false)
        || value_contains_key(&ctx.refs, &["spectral_extractor", "spectral_fiber_bundle"])
        || value_contains_key(
            &ctx.byte_refs,
            &["spectral_extractor", "spectral_fiber_bundle"],
        )
}

fn context_has_pathb_geometry(ctx: &ScientificEventContext) -> bool {
    ctx.source_artifact_kind
        .as_deref()
        .map(artifact_supports_pathb_geometry)
        .unwrap_or(false)
        || value_contains_key(
            &ctx.refs,
            &[
                "pocket_morphology_surface",
                "binding_site_physiology_surface",
            ],
        )
        || value_contains_key(
            &ctx.byte_refs,
            &[
                "pocket_morphology_surface",
                "binding_site_physiology_surface",
            ],
        )
}

fn value_contains_key(value: &Value, keys: &[&str]) -> bool {
    match value {
        Value::Object(map) => map.iter().any(|(key, value)| {
            let actual_key_match = !key.contains("expected")
                && !value.is_null()
                && keys.iter().any(|needle| key.contains(needle));
            actual_key_match || value_contains_key(value, keys)
        }),
        Value::Array(values) => values.iter().any(|value| value_contains_key(value, keys)),
        _ => false,
    }
}

fn artifact_supports_spectral_geometry(kind: &str) -> bool {
    let k = kind.to_ascii_lowercase();
    k.contains("spectral_extractor")
        || k.contains("spectral_fiber_bundle")
        || k.contains("fiber_bundle")
        || k.contains("spectral_bundle")
}

fn artifact_supports_pathb_geometry(kind: &str) -> bool {
    let k = kind.to_ascii_lowercase();
    k.contains("pocket_morphology")
        || k.contains("binding_site_physiology")
        || k.contains("binding_site_morphology")
}

fn artifact_supports_pocket_geometry(kind: &str) -> bool {
    matches!(
        kind,
        "trajectory_frames"
            | "trajectory_audit"
            | "signal_grid"
            | "kcc_v2full"
            | "warp_matrix"
            | "forces_final"
            | "aromatic_centroids_final"
            | "prismcode_md_native_receipt"
            | "qnanl_artifact_packet"
            | "qnanl_artifact_receipt"
    ) || artifact_supports_spectral_geometry(kind)
        || artifact_supports_pathb_geometry(kind)
}

fn artifact_supports_ligand_causality(kind: &str) -> bool {
    matches!(
        kind,
        "trajectory_frames"
            | "trajectory_audit"
            | "signal_grid"
            | "kcc_v2full"
            | "warp_matrix"
            | "forces_final"
            | "spike_transition_events"
            | "prismcode_md_native_receipt"
            | "qnanl_artifact_packet"
            | "qnanl_artifact_receipt"
    ) || artifact_supports_spectral_geometry(kind)
        || artifact_supports_pathb_geometry(kind)
}

fn artifact_supports_residue_activity(kind: &str) -> bool {
    matches!(
        kind,
        "trajectory_frames"
            | "trajectory_audit"
            | "signal_grid"
            | "kcc_v2full"
            | "warp_matrix"
            | "forces_final"
            | "asc_vectors"
            | "aromatic_centroids_final"
            | "spike_transition_events"
            | "prismcode_md_native_receipt"
            | "qnanl_artifact_packet"
            | "qnanl_artifact_receipt"
    ) || artifact_supports_spectral_geometry(kind)
        || artifact_supports_pathb_geometry(kind)
}

fn scientific_state_for_status(status: &str) -> &'static str {
    if status == "PASS" || status == "PRODUCED" {
        "PRODUCED"
    } else if status.starts_with("BLOCKED") || status.contains("BLOCKED") {
        "BLOCKED_POLICY"
    } else {
        "PENDING_INPUT"
    }
}

fn finite_f64(value: f64) -> Option<f64> {
    if value.is_finite() {
        Some(value)
    } else {
        None
    }
}

fn current_unix_ns() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

fn artifact_from_byte_binding(map: &serde_json::Map<String, Value>) -> Option<Value> {
    let binding = map.get("byte_binding")?.as_object()?;
    let byte_range = binding.get("byte_range").and_then(|v| v.as_object());
    let hashes = binding.get("hashes").and_then(|v| v.as_object());
    let path = binding
        .get("path")
        .cloned()
        .unwrap_or(Value::String("".to_string()));
    let family = map
        .get("source_artifact_kind")
        .cloned()
        .unwrap_or(Value::String("unknown".to_string()));
    Some(serde_json::json!({
        "path": path,
        "family": family,
        "byte_start": byte_range
            .and_then(|range| range.get("start"))
            .cloned()
            .unwrap_or(Value::Null),
        "byte_end": byte_range
            .and_then(|range| range.get("end_inclusive"))
            .cloned()
            .unwrap_or(Value::Null),
        "byte_end_exclusive": byte_range
            .and_then(|range| range.get("end_exclusive"))
            .cloned()
            .unwrap_or(Value::Null),
        "raw_bytes": binding.get("logical_artifact")
            .and_then(|logical| logical.get("size_bytes"))
            .or_else(|| byte_range.and_then(|range| range.get("size_bytes")))
            .cloned().unwrap_or(Value::Null),
        "stored_bytes": byte_range.and_then(|range| range.get("size_bytes"))
            .cloned().unwrap_or(Value::Null),
        "sha256": hashes
            .and_then(|hash| hash.get("sha256"))
            .cloned()
            .unwrap_or(Value::Null),
        "record_count": map
            .get("source_artifact_record_count")
            .cloned()
            .unwrap_or(Value::Null),
        "storage_representation": binding.get("storage_representation").cloned().unwrap_or(Value::Null),
        "byte_range_domain": binding.get("byte_range_domain").cloned().unwrap_or(Value::Null),
        "logical_artifact": binding.get("logical_artifact").cloned().unwrap_or(Value::Null),
    }))
}

fn normalize_epoch_aliases(record: &mut Value) {
    let Some(obj) = record.as_object_mut() else {
        return;
    };
    alias_key(obj, "target", "target_id");
    alias_key(obj, "epoch_lifecycle", "lifecycle_stage");
    alias_key(obj, "epoch_scope", "epoch_kind");
    alias_key(
        obj,
        "save_trajectory_interval",
        "requested_save_interval_steps",
    );
    if !obj.contains_key("closed_interval") {
        let closed = matches!(
            obj.get("epoch_lifecycle").and_then(|v| v.as_str()),
            Some("CLOSE" | "CLOSED")
        );
        obj.insert("closed_interval".to_string(), Value::Bool(closed));
    }
    let inferred_fidelity = infer_fidelity_level(obj);
    obj.entry("chronology_fidelity_level".to_string())
        .or_insert_with(|| inferred_fidelity.into());
    obj.entry("chronology_marker_scope".to_string())
        .or_insert_with(|| "SCIENTIFIC_STATE_EPOCH".into());
    let inferred_interferometric_mode = infer_interferometric_mode(obj);
    obj.entry("interferometric_mode".to_string())
        .or_insert_with(|| inferred_interferometric_mode.into());
    obj.entry("interferometric_marker_kind".to_string())
        .or_insert_with(|| "EPOCH_MARKER".into());
}

fn normalize_event_aliases(record: &mut Value) {
    let Some(obj) = record.as_object_mut() else {
        return;
    };
    alias_key(obj, "event_id", "chronology_event_id");
    alias_key(obj, "frame_index", "frame_idx");
    alias_key(obj, "source_artifact_kind", "source_kind");
    alias_key(obj, "source_artifact_kind", "artifact_family");
    if !obj.contains_key("operator_name") {
        obj.insert("operator_name".to_string(), "nhs_rt_full".into());
    }
    if !obj.contains_key("opcode_name") {
        let opcode = obj
            .get("event_type")
            .and_then(|v| v.as_str())
            .map(opcode_for_event_type)
            .unwrap_or("MD.SCIENTIFIC_CHRONOLOGY_EVENT");
        obj.insert("opcode_name".to_string(), opcode.into());
    }
    if !obj.contains_key("state_hash") {
        if let Some(v) = obj
            .get("rolling_hash64")
            .or_else(|| obj.get("frame_hash64"))
            .cloned()
        {
            obj.insert("state_hash".to_string(), v);
        }
    }
    let inferred_fidelity = infer_fidelity_level(obj);
    obj.entry("chronology_fidelity_level".to_string())
        .or_insert_with(|| inferred_fidelity.into());
    obj.entry("chronology_marker_scope".to_string())
        .or_insert_with(|| "SCIENTIFIC_CHRONOLOGY_EVENT".into());
    let inferred_interferometric_mode = infer_interferometric_mode(obj);
    obj.entry("interferometric_mode".to_string())
        .or_insert_with(|| inferred_interferometric_mode.into());
    let inferred_marker_kind = infer_marker_kind(obj);
    obj.entry("interferometric_marker_kind".to_string())
        .or_insert_with(|| inferred_marker_kind.into());
}

fn alias_key(obj: &mut serde_json::Map<String, Value>, from: &str, to: &str) {
    if obj.contains_key(to) {
        return;
    }
    if let Some(value) = obj.get(from).cloned() {
        obj.insert(to.to_string(), value);
    }
}

fn infer_fidelity_level(obj: &serde_json::Map<String, Value>) -> &'static str {
    match obj
        .get("chronology_fidelity_level")
        .or_else(|| obj.get("temporal_fidelity"))
        .and_then(|v| v.as_str())
    {
        Some("exhaustive") | Some("EXHAUSTIVE") => "EXHAUSTIVE",
        Some("high") | Some("HIGH") => "HIGH",
        Some("standard") | Some("STANDARD") => "STANDARD",
        Some("low") | Some("LOW") => "LOW",
        Some(_) => "CUSTOM",
        None => match obj.get("capture_cadence_class").and_then(|v| v.as_str()) {
            Some("CHUNK_BOUNDARY") => "HIGH_CHUNK_BOUNDARY",
            Some("EXACT_INTERVAL") => "EXACT_INTERVAL",
            Some("SUB_CHUNK_INTERVAL_MARKER") => "HIGH_SUBCHUNK_MARKER",
            Some("SUB_CHUNK_INTERVAL_MARKERS_WITH_CHUNK_FRAME_FALLBACK") => "HIGH_SUBCHUNK_MARKER",
            Some("CHUNK_CLOSE_FRAME_FALLBACK") => "HIGH_SUBCHUNK_MARKER_WITH_CHUNK_FRAME_FALLBACK",
            Some("RUNTIME_ARTIFACT_EMISSION") => "RUNTIME_ARTIFACT",
            _ => "RUNTIME_NATIVE",
        },
    }
}

fn infer_interferometric_mode(obj: &serde_json::Map<String, Value>) -> &'static str {
    if obj
        .get("multi_differential")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        return "MULTI_DIFFERENTIAL_TWIN16";
    }
    if obj
        .get("coupled_twin")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        return "COUPLED_TWIN";
    }
    if obj.contains_key("condition_id")
        || obj.contains_key("phase_from")
        || obj.contains_key("phase_to")
    {
        return "TWIN16_CONDITION_MARKER";
    }
    match obj.get("source_artifact_kind").and_then(|v| v.as_str()) {
        Some(kind) if kind.contains("twin") || kind.contains("thermodynamic_phase") => {
            "TWIN16_PHASE_MANIFOLD"
        }
        _ => "MD_NATIVE",
    }
}

fn infer_marker_kind(obj: &serde_json::Map<String, Value>) -> &'static str {
    match obj.get("event_type").and_then(|v| v.as_str()) {
        Some("FRAME_CAPTURE") => "TRAJECTORY_FRAME_MARKER",
        Some("SUBCHUNK_SCIENTIFIC_UPDATE_MARKER") => "SUBCHUNK_SCIENTIFIC_UPDATE_MARKER",
        Some("TRAJECTORY_FRAME_STREAM_EMIT") => "TRAJECTORY_FRAME_STREAM_ARTIFACT_MARKER",
        Some(EVENT_TYPE_POCKET_GEOMETRY_EVAL) => "POCKET_GEOMETRY_MARKER",
        Some(EVENT_TYPE_SCIENTIFIC_POCKET_OPENING) => "SCIENTIFIC_POCKET_OPENING_MARKER",
        Some(EVENT_TYPE_LIGAND_CAUSAL_ACTIVITY) => "LIGAND_CAUSAL_ACTIVITY_MARKER",
        Some(EVENT_TYPE_RESIDUE_ACTIVITY) => "RESIDUE_ACTIVITY_MARKER",
        Some("CUDA_GRAPH_CAPTURE") | Some("CUDA_GRAPH_REPLAY") => "CUDA_GRAPH_MARKER",
        Some("QNANL_SEAL") => "QNANL_PACKET_MARKER",
        Some("TOKENIZER_EXPORT") => "TOKENIZER_HANDOFF_MARKER",
        Some("PRISMCODE_RECEIPT") => "OPCODE_RECEIPT_MARKER",
        Some("EPOCH_OPEN") | Some("EPOCH_CLOSE") => "EPOCH_BOUNDARY_MARKER",
        _ => "ARTIFACT_EMISSION_MARKER",
    }
}

fn opcode_for_event_type(event_type: &str) -> &'static str {
    match event_type {
        "EPOCH_OPEN" => "STATE.CREATE_EPOCH",
        "EPOCH_CLOSE" => "STATE.CLOSE_EPOCH",
        "FRAME_CAPTURE" => "MD.FRAME_CAPTURE",
        "SUBCHUNK_SCIENTIFIC_UPDATE_MARKER" => "MD.SUBCHUNK_SCIENTIFIC_UPDATE_MARKER",
        "TRAJECTORY_FRAME_STREAM_EMIT" => "MD.TRAJECTORY_FRAME_STREAM_EMIT",
        EVENT_TYPE_POCKET_GEOMETRY_EVAL => "MD.POCKET_GEOMETRY_EVAL",
        EVENT_TYPE_SCIENTIFIC_POCKET_OPENING => "MD.SCIENTIFIC_POCKET_OPENING_EVENT",
        EVENT_TYPE_LIGAND_CAUSAL_ACTIVITY => "MD.LIGAND_CAUSAL_ACTIVITY_EVENT",
        EVENT_TYPE_RESIDUE_ACTIVITY => "MD.RESIDUE_ACTIVITY_EVENT",
        "SPIKE_STREAM_EMIT" => "MD.SPIKE_STREAM_EMIT",
        "SIGNAL_GRID_EMIT" => "MD.SIGNAL_GRID_UPDATE",
        "KCC_V2FULL_EMIT" => "MD.KCC_UPDATE",
        "WARP_TENSOR_EMIT" => "MD.WARP_TENSOR_UPDATE",
        "FORCE_FINAL_EMIT" => "MD.FORCE_FINAL_EMIT",
        "ASC_VECTOR_EMIT" => "MD.ASC_VECTOR_EMIT",
        "PRISMCODE_RECEIPT" => "PRISMCODE.RECEIPT",
        "QNANL_SEAL" => "QNANL.SEAL",
        "TOKENIZER_EXPORT" => "TOKENIZER.EXPORT",
        "BLOCKER_RAISED" => "VALIDATION.BLOCKER_RAISED",
        _ => "MD.SCIENTIFIC_CHRONOLOGY_EVENT",
    }
}

/// Return a byte-range and SHA-256 binding for an existing file.
pub fn path_binding(path: &Path) -> Result<Value> {
    let meta = path
        .metadata()
        .with_context(|| format!("stat chronology-bound file {}", path.display()))?;
    let sha256 = sha256_file(path)?;
    Ok(serde_json::json!({
        "path": path.display().to_string(),
        "byte_range": {
            "start": 0,
            "end_exclusive": meta.len(),
            "end_inclusive": meta.len().saturating_sub(1),
            "size_bytes": meta.len(),
        },
        "hashes": {
            "sha256": sha256,
        },
    }))
}

fn object_with_defaults(
    value: Value,
    schema_kind: &str,
    default_status: Option<&str>,
    default_record_kind: &str,
) -> Value {
    let mut obj = match value {
        Value::Object(map) => map,
        other => {
            let mut map = serde_json::Map::new();
            map.insert("payload".to_string(), other);
            map
        }
    };
    obj.entry("schema_kind".to_string())
        .or_insert_with(|| Value::String(schema_kind.to_string()));
    obj.entry("schema_version".to_string())
        .or_insert_with(|| serde_json::json!(1));
    obj.entry("record_kind".to_string())
        .or_insert_with(|| Value::String(default_record_kind.to_string()));
    if let Some(status) = default_status {
        obj.entry("status".to_string())
            .or_insert_with(|| Value::String(status.to_string()));
    }
    Value::Object(obj)
}

fn append_jsonl_locked(path: &Path, value: &Value) -> Result<()> {
    create_parent(path)?;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("open append {}", path.display()))?;
    serde_json::to_writer(&mut file, value)
        .with_context(|| format!("serialize JSONL record {}", path.display()))?;
    file.write_all(b"\n")
        .with_context(|| format!("terminate JSONL record {}", path.display()))?;
    file.flush()
        .with_context(|| format!("flush JSONL record {}", path.display()))?;
    Ok(())
}

fn create_parent(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    Ok(())
}

fn truncate_file(path: &Path) -> Result<()> {
    create_parent(path)?;
    std::fs::File::create(path)
        .with_context(|| format!("initialize JSONL stream {}", path.display()))?;
    Ok(())
}

fn remove_if_present(path: &Path) -> Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e).with_context(|| format!("remove stale {}", path.display())),
    }
}

fn write_pretty_json(path: &Path, value: &Value) -> Result<()> {
    create_parent(path)?;
    let file = std::fs::File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut writer = std::io::BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, value)
        .with_context(|| format!("serialize {}", path.display()))?;
    writer
        .write_all(b"\n")
        .with_context(|| format!("terminate {}", path.display()))?;
    writer
        .flush()
        .with_context(|| format!("flush {}", path.display()))?;
    Ok(())
}

fn file_report(path: &Path, record_count: u64) -> Value {
    let mut value = path_identity_or_absent(path);
    if let Value::Object(map) = &mut value {
        map.insert("record_count".to_string(), serde_json::json!(record_count));
    }
    value
}

fn path_identity_pending(path: &Path) -> Value {
    serde_json::json!({
        "path": path.display().to_string(),
        "status": "PENDING_WRITE",
    })
}

fn path_identity_or_absent(path: &Path) -> Value {
    match path_binding(path) {
        Ok(value) => value,
        Err(_) => serde_json::json!({
            "path": path.display().to_string(),
            "status": "ABSENT",
        }),
    }
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1024 * 1024];
    loop {
        let n = file
            .read(&mut buf)
            .with_context(|| format!("read {}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eight_stream_worker_epoch_boundaries_satisfy_fail_closed_acceptance() {
        let directory = tempfile::tempdir().expect("create chronology test directory");
        let sinks = ScientificChronologySinks::new(directory.path());
        sinks.initialize().expect("initialize chronology sinks");

        for stream_id in 0..8u32 {
            let epoch_id = format!("run-1_stream{stream_id:02}_md_worker_epoch");
            let context = |emitted_unix_ns| {
                ScientificEventContext::new(epoch_id.clone())
                    .event_id_base(epoch_id.clone())
                    .run_id("run-1")
                    .runtime_origin("RUNTIME_INLINE")
                    .runtime_native(true)
                    .emitted_unix_ns(emitted_unix_ns)
                    .stream_id(stream_id)
                    .target("path-b-target")
                    .topology("/verified/topology.json", "topology-sha256")
                    .capture_cadence_class("MD_WORKER_LIFECYCLE")
                    .chronology_fidelity_level("RUNTIME_NATIVE")
                    .interferometric_marker_kind("EPOCH_BOUNDARY_MARKER")
                    .chronology_marker_scope("PER_STREAM_MD_WORKER_LIFECYCLE")
                    .refs(serde_json::json!({
                        "scientific_state_epochs_path": sinks.epochs_path(),
                        "scientific_chronology_events_path": sinks.events_path(),
                    }))
            };
            let trajectory_capture = || {
                serde_json::json!({
                    "enabled": false,
                    "status": "DISABLED_SAVE_TRAJECTORY_INTERVAL_ZERO",
                    "save_trajectory_interval": 0,
                    "trajectory_path": null,
                    "trajectory_origin_receipt_path": null,
                })
            };

            sinks
                .append_stream_epoch_boundary(
                    ScientificEpochBoundary::Open,
                    context(1_000 + u128::from(stream_id)),
                    serde_json::json!({
                        "worker_lifecycle_status": "OPEN_BEFORE_INTEGRATION",
                        "trajectory_capture": trajectory_capture(),
                    }),
                )
                .expect("append worker epoch open");
            sinks
                .append_stream_epoch_boundary(
                    ScientificEpochBoundary::Close,
                    context(2_000 + u128::from(stream_id)),
                    serde_json::json!({
                        "worker_lifecycle_status": "CLOSE_SUCCESSFUL_WORKER_RETURN",
                        "trajectory_capture": trajectory_capture(),
                    }),
                )
                .expect("append worker epoch close");
        }

        let required_event_counts = BTreeMap::from([
            ("EPOCH_OPEN".to_string(), 8),
            ("EPOCH_CLOSE".to_string(), 8),
        ]);
        let decision = sinks
            .write_acceptance_decision(
                &required_event_counts,
                &[],
                serde_json::json!({"stream_count": 8, "save_trajectory_interval": 0}),
            )
            .expect("write chronology acceptance");
        assert_eq!(decision.get("status").and_then(Value::as_str), Some("PASS"));
        assert_eq!(
            decision
                .pointer("/observed_event_counts_by_kind/EPOCH_OPEN")
                .and_then(Value::as_u64),
            Some(8)
        );
        assert_eq!(
            decision
                .pointer("/observed_event_counts_by_kind/EPOCH_CLOSE")
                .and_then(Value::as_u64),
            Some(8)
        );
        assert_eq!(
            decision
                .get("missing_required_events")
                .and_then(Value::as_array)
                .map(Vec::len),
            Some(0)
        );

        let events_text =
            std::fs::read_to_string(sinks.events_path()).expect("read emitted chronology events");
        let events = events_text
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).expect("parse chronology event"))
            .collect::<Vec<_>>();
        assert_eq!(events.len(), 16);
        assert!(events.iter().all(|event| {
            event
                .pointer("/boundary_evidence/trajectory_capture/enabled")
                .and_then(Value::as_bool)
                == Some(false)
                && event
                    .pointer("/boundary_evidence/trajectory_capture/trajectory_path")
                    .is_some_and(Value::is_null)
                && event
                    .pointer("/boundary_evidence/trajectory_capture/trajectory_origin_receipt_path")
                    .is_some_and(Value::is_null)
        }));
        let event_ids = events
            .iter()
            .filter_map(|event| event.get("event_id").and_then(Value::as_str))
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            event_ids.len(),
            16,
            "worker boundary events must not duplicate"
        );
    }

    #[test]
    fn pending_pocket_event_keeps_exact_operator_fields() {
        let ctx = ScientificEventContext::new("epoch-1")
            .frame_event_id("frame-1")
            .event_id_base("frame-1")
            .run_id("run-1")
            .stream_id(7)
            .step(128)
            .physical_time_fs(256.0)
            .topology("/tmp/topology.json", "abc123")
            .refs(serde_json::json!({
                "spectral_geometry_authority": {
                    "expected_spectral_extractor_stage": "spectral geometry adjudication",
                    "path_b_candidate_context_only": {
                        "pocket_morphology_surface_expected_path": "/tmp/pocket_morphology_surface.prismos_v1.json"
                    }
                }
            }));
        let event = build_scientific_pocket_opening_pending(&ctx);

        assert_eq!(
            event.get("event_type").and_then(Value::as_str),
            Some(EVENT_TYPE_SCIENTIFIC_POCKET_OPENING)
        );
        assert_eq!(
            event.get("operator_event_object").and_then(Value::as_str),
            Some(SCHEMA_SCIENTIFIC_POCKET_OPENING)
        );
        assert_eq!(
            event.get("epoch_id").and_then(Value::as_str),
            Some("epoch-1")
        );
        assert_eq!(
            event.get("frame_event_id").and_then(Value::as_str),
            Some("frame-1")
        );
        assert!(event.get("residue_ids").is_some());
        assert!(event.get("pocket_volume").is_some());
        assert!(event.get("pocket_surface_area").is_some());
        assert!(event.get("lining_contacts").is_some());
        assert!(event.get("distance_gates").is_some());
        assert!(event.get("persistence_frames").is_some());
        assert!(event.get("apo_holo_contrast").is_some());
        assert!(event.get("confidence").is_some());
        assert_eq!(
            event
                .get("geometry_authority_status")
                .and_then(Value::as_str),
            Some("PENDING_SPECTRAL_EXTRACTOR_INPUTS")
        );
        assert_eq!(
            event.get("scientific_event_state").and_then(Value::as_str),
            Some("PENDING_INPUT")
        );
    }

    #[test]
    fn expected_paths_do_not_count_as_present_geometry_authority() {
        let ctx = ScientificEventContext::new("epoch-1").refs(serde_json::json!({
            "expected_spectral_extractor_stage": "spectral geometry adjudication",
            "pocket_morphology_surface_expected_path": "/tmp/pocket.json"
        }));
        assert_eq!(
            geometry_status_for_context(&ctx),
            "PENDING_SPECTRAL_EXTRACTOR_INPUTS"
        );
    }

    #[test]
    fn actual_pathb_context_blocks_without_spectral_geometry() {
        let ctx = ScientificEventContext::new("epoch-1").refs(serde_json::json!({
            "pocket_morphology_surface": {
                "path": "/tmp/pocket_morphology_surface.prismos_v1.json"
            }
        }));
        assert_eq!(
            geometry_status_for_context(&ctx),
            "BLOCKED_MISSING_SPECTRAL_GEOMETRY"
        );
    }

    #[test]
    fn enum_values_match_operator_allowed_values() {
        assert_eq!(LigandContactType::HBond.as_str(), "H_BOND");
        assert_eq!(LigandContactType::Hydrophobic.as_str(), "HYDROPHOBIC");
        assert_eq!(LigandContactType::PiStack.as_str(), "PI_STACK");
        assert_eq!(LigandContactType::SaltBridge.as_str(), "SALT_BRIDGE");
        assert_eq!(LigandContactType::WaterBridge.as_str(), "WATER_BRIDGE");
        assert_eq!(ResidueActivityType::PocketLining.as_str(), "POCKET_LINING");
        assert_eq!(
            ResidueActivityType::ContactSwitch.as_str(),
            "CONTACT_SWITCH"
        );
        assert_eq!(ResidueActivityType::HingeMotion.as_str(), "HINGE_MOTION");
        assert_eq!(
            ResidueActivityType::AllostericSignal.as_str(),
            "ALLOSTERIC_SIGNAL"
        );
        assert_eq!(ResidueActivityType::KccSpike.as_str(), "KCC_SPIKE");
        assert_eq!(ResidueActivityType::ForceHotspot.as_str(), "FORCE_HOTSPOT");
        assert_eq!(
            ResidueActivityType::ExposureChange.as_str(),
            "EXPOSURE_CHANGE"
        );
    }
}

/// Bind retained physical bytes while preserving the identity of a deleted,
/// byte-exactly recoverable logical artifact. Packet offsets never stand for
/// offsets into its decoded event stream.
pub fn logical_artifact_path_binding(
    raw_path: &Path,
    expected_raw_bytes: u64,
    receipt_paths: &[PathBuf],
) -> Result<Value> {
    match raw_path.metadata() {
        Ok(_) => return path_binding(raw_path),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error).with_context(|| format!("stat logical artifact {}", raw_path.display())),
    }
    anyhow::ensure!(receipt_paths.len() == 1,
        "missing logical artifact {} requires exactly one retained-packet receipt, found {}",
        raw_path.display(), receipt_paths.len());
    let receipt_path = &receipt_paths[0];
    let receipt: Value = serde_json::from_slice(&std::fs::read(receipt_path)?)
        .context("parse retained-packet chronology receipt")?;
    let kind = receipt.get("schema_kind").and_then(Value::as_str).unwrap_or("");
    anyhow::ensure!(matches!(kind, "prismos_qnanl_artifact_receipt" | "prismcode_nanl_spike_codec_receipt"),
        "unsupported retained-packet receipt schema {kind}");
    anyhow::ensure!(receipt.get("status").and_then(Value::as_str) == Some("PASS")
        && receipt.get("roundtrip_verified").and_then(Value::as_bool) == Some(true),
        "retained-packet receipt lacks verified byte-exact roundtrip");
    let source_key = if kind == "prismcode_nanl_spike_codec_receipt" { "raw_path" } else { "source_path" };
    anyhow::ensure!(receipt.get(source_key).and_then(Value::as_str) == raw_path.to_str(),
        "retained-packet receipt logical source mismatch");
    anyhow::ensure!(receipt.get("receipt_path").and_then(Value::as_str) == receipt_path.to_str(),
        "retained-packet receipt self identity mismatch");
    anyhow::ensure!(receipt.get("raw_bytes").and_then(Value::as_u64) == Some(expected_raw_bytes),
        "retained-packet receipt logical size mismatch");
    let raw_hash = receipt.get("raw_hash_blake3").and_then(Value::as_str)
        .context("missing logical raw BLAKE3")?;
    anyhow::ensure!(raw_hash.len() == 64 && raw_hash.bytes().all(|b| b.is_ascii_hexdigit()),
        "invalid logical raw BLAKE3");
    let packet_path = PathBuf::from(receipt.get("packet_path").and_then(Value::as_str)
        .context("missing retained packet path")?);
    anyhow::ensure!(packet_path != raw_path, "packet and logical source paths coincide");
    let mut packet = std::fs::File::open(&packet_path)
        .with_context(|| format!("open retained chronology packet {}", packet_path.display()))?;
    anyhow::ensure!(packet.metadata()?.is_file(), "retained packet must be a regular file");
    let mut sha = Sha256::new();
    let mut b3 = blake3::Hasher::new();
    let mut bytes = 0_u64;
    let mut buffer = [0_u8; 65536];
    loop {
        let n = packet.read(&mut buffer)?;
        if n == 0 { break; }
        sha.update(&buffer[..n]); b3.update(&buffer[..n]); bytes += n as u64;
    }
    let packet_hash = b3.finalize().to_hex().to_string();
    anyhow::ensure!(receipt.get("packet_bytes").and_then(Value::as_u64) == Some(bytes)
        && receipt.get("packet_hash_blake3").and_then(Value::as_str) == Some(packet_hash.as_str()),
        "retained chronology packet bytes differ from verified receipt");
    let receipt_binding = path_binding(receipt_path)?;
    Ok(serde_json::json!({
        "path": packet_path.display().to_string(),
        "byte_range": {"start": 0, "end_exclusive": bytes,
            "end_inclusive": bytes.saturating_sub(1), "size_bytes": bytes},
        "hashes": {"sha256": format!("{:x}", sha.finalize()), "blake3": packet_hash},
        "storage_representation": "RETAINED_BYTE_EXACT_QNANL_PACKET",
        "byte_range_domain": "ENCODED_PACKET_BYTES",
        "logical_artifact": {
            "path": raw_path.display().to_string(),
            "size_bytes": expected_raw_bytes,
            "hashes": {"blake3": raw_hash},
            "storage_status": "RAW_ABSENT_BYTE_EXACT_REHYDRATION_AVAILABLE",
            "byte_range_domain": "DECODED_LOGICAL_ARTIFACT_BYTES",
            "codec_receipt_schema": kind,
            "roundtrip_verified": true,
            "receipt_binding": receipt_binding
        }
    }))
}

#[cfg(test)]
mod retained_packet_binding_tests {
    use super::*;
    fn fixture(dir: &Path, spike: bool) -> (PathBuf, PathBuf, PathBuf) {
        let raw = dir.join("events.bin"); let packet = dir.join("events.packet"); let receipt = dir.join("receipt.json");
        std::fs::write(&packet, b"retained packet bytes").unwrap();
        let data = std::fs::read(&packet).unwrap();
        let mut value = serde_json::json!({"schema_kind": if spike {"prismcode_nanl_spike_codec_receipt"} else {"prismos_qnanl_artifact_receipt"},
            "status":"PASS","roundtrip_verified":true,"receipt_path":receipt,
            "packet_path":packet,"raw_bytes":123,"packet_bytes":data.len(),
            "raw_hash_blake3":blake3::hash(b"logical original").to_hex().to_string(),
            "packet_hash_blake3":blake3::hash(&data).to_hex().to_string()});
        value[if spike {"raw_path"} else {"source_path"}] = serde_json::json!(raw);
        std::fs::write(&receipt, serde_json::to_vec(&value).unwrap()).unwrap();
        (raw,packet,receipt)
    }
    #[test]
    fn retained_packet_preserves_encoded_and_logical_domains() {
        for spike in [false,true] {
            let dir=tempfile::tempdir().unwrap(); let (raw,packet,receipt)=fixture(dir.path(),spike);
            let value=logical_artifact_path_binding(&raw,123,&[receipt]).unwrap();
            assert_eq!(value["path"],packet.to_str().unwrap());
            assert_eq!(value["logical_artifact"]["path"],raw.to_str().unwrap());
            assert_eq!(value["logical_artifact"]["size_bytes"],123);
            assert_eq!(value["byte_range_domain"],"ENCODED_PACKET_BYTES");
            let event=serde_json::json!({"byte_binding":value,"source_artifact_kind":"spikes"});
            let projected=artifact_from_byte_binding(event.as_object().unwrap()).unwrap();
            assert_eq!(projected["logical_artifact"]["size_bytes"],123);
            assert_eq!(projected["raw_bytes"],123);
            assert_eq!(projected["stored_bytes"],21);
            assert_eq!(projected["storage_representation"],"RETAINED_BYTE_EXACT_QNANL_PACKET");
        }
    }
    #[test]
    fn retained_packet_rejects_corruption_missing_ambiguous_and_false_receipts() {
        let dir=tempfile::tempdir().unwrap(); let (raw,packet,receipt)=fixture(dir.path(),false);
        assert!(logical_artifact_path_binding(&raw,123,&[]).is_err());
        assert!(logical_artifact_path_binding(&raw,123,&[receipt.clone(),receipt.clone()]).is_err());
        assert!(logical_artifact_path_binding(&raw,124,&[receipt.clone()]).is_err());
        let pristine=std::fs::read(&receipt).unwrap();
        for (key,value) in [("status",serde_json::json!("FAIL")),("roundtrip_verified",serde_json::json!(false)),("source_path",serde_json::json!("wrong")),("raw_hash_blake3",serde_json::json!("bad")),("packet_hash_blake3",serde_json::json!("bad"))] {
            let mut v:Value=serde_json::from_slice(&pristine).unwrap();v[key]=value;
            std::fs::write(&receipt,serde_json::to_vec(&v).unwrap()).unwrap();
            assert!(logical_artifact_path_binding(&raw,123,&[receipt.clone()]).is_err(),"{key}");
        }
        std::fs::write(&receipt,&pristine).unwrap();std::fs::write(&packet,b"corrupt packet").unwrap();
        assert!(logical_artifact_path_binding(&raw,123,&[receipt.clone()]).is_err());
        std::fs::remove_file(&packet).unwrap();assert!(logical_artifact_path_binding(&raw,123,&[receipt]).is_err());
    }
    #[test]
    fn existing_raw_uses_ordinary_byte_binding() {
        let dir=tempfile::tempdir().unwrap();let p=dir.path().join("raw");std::fs::write(&p,b"raw").unwrap();
        assert_eq!(logical_artifact_path_binding(&p,3,&[]).unwrap(),path_binding(&p).unwrap());
    }
}


#[derive(Clone,Debug,serde::Serialize,serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ChronologyMetadataContinuationV1 {
    relative_path:String,
    #[serde(with="crate::checkpoint_codec::optional_bytes")]
    contents:Option<Vec<u8>>,
}
#[derive(Clone,Debug,serde::Serialize,serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScientificChronologyContinuationV1 {
    schema_version:u32,
    epochs:crate::scientific_checkpoint::AppendCursor,
    events:crate::scientific_checkpoint::AppendCursor,
    state:ScientificChronologyState,
    metadata:[ChronologyMetadataContinuationV1;2],
}
fn open_chronology_checkpoint_file(path:&Path)->Result<std::fs::File> {
    use std::os::unix::fs::OpenOptionsExt;
    let file=std::fs::OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW|libc::O_CLOEXEC).open(path)?;
    anyhow::ensure!(file.metadata()?.is_file(),"chronology checkpoint input is not a regular file");Ok(file)
}
fn chronology_prefix(file:&std::fs::File,cursor:&crate::scientific_checkpoint::AppendCursor)->Result<()> {
    anyhow::ensure!(file.metadata()?.len()>=cursor.committed_bytes,"chronology prefix shorter than checkpoint");
    let mut remaining=cursor.committed_bytes;let mut reader=file.try_clone()?;let mut hash=Sha256::new();let mut buffer=[0u8;65536];
    use std::io::{Seek,SeekFrom};reader.seek(SeekFrom::Start(0))?;
    while remaining>0{let count=reader.read(&mut buffer[..remaining.min(65536) as usize])?;
        anyhow::ensure!(count>0,"chronology prefix unexpectedly short");hash.update(&buffer[..count]);remaining-=count as u64;}
    anyhow::ensure!(format!("{:x}",hash.finalize())==cursor.committed_sha256,"chronology committed prefix changed");Ok(())
}
fn recount_chronology_prefixes(epochs:&std::fs::File,epoch_bytes:u64,events:&std::fs::File,event_bytes:u64)->Result<ScientificChronologyState> {
    use std::io::{BufRead,BufReader,Seek,SeekFrom};
    let mut state=ScientificChronologyState::default();
    let mut tally=|map:&mut BTreeMap<String,u64>,key:String|->Result<()>{let count=map.entry(key).or_insert(0);*count=count.checked_add(1).context("chronology counter overflow")?;Ok(())};
    for (file,size,event_stream) in [(epochs,epoch_bytes,false),(events,event_bytes,true)] {
        let mut file=file.try_clone()?;file.seek(SeekFrom::Start(0))?;let mut reader=BufReader::new(file.take(size));let mut line=Vec::new();
        loop {line.clear();if reader.read_until(b'\n',&mut line)?==0{break;}
            anyhow::ensure!(line.last()==Some(&b'\n'),"chronology checkpoint contains a partial record");
            let record:Value=serde_json::from_slice(&line)?;
            anyhow::ensure!(record.is_object(),"chronology persisted record is not an object");
            let status=record.get("status").and_then(Value::as_str).unwrap_or("UNKNOWN").to_string();
            if !event_stream {
                state.epoch_record_count=state.epoch_record_count.checked_add(1).context("chronology epoch count overflow")?;
                tally(&mut state.epoch_counts_by_status,status)?;
            } else {
                let kind=record.get("event_kind").or_else(||record.get("event_type")).and_then(Value::as_str).unwrap_or("UNKNOWN").to_string();
                let event_type=record.get("event_type").and_then(Value::as_str).map(str::to_string).unwrap_or_else(||canonical_event_type_for_kind(&kind).to_string());
                let scientific=record.get("scientific_event_state").or_else(||record.get("event_availability_status")).and_then(Value::as_str)
                    .map(str::to_string).unwrap_or_else(||scientific_state_for_status(&status).to_string());
                state.event_record_count=state.event_record_count.checked_add(1).context("chronology event count overflow")?;
                tally(&mut state.event_counts_by_kind,kind)?;tally(&mut state.event_counts_by_type,event_type.clone())?;
                tally(&mut state.event_counts_by_status,status)?;tally(&mut state.event_counts_by_scientific_state,scientific.clone())?;
                tally(&mut state.event_counts_by_type_and_scientific_state,format!("{event_type}::{scientific}"))?;
            }
        }
    }
    Ok(state)
}
impl ScientificChronologySinks {
    /// Call once from the global native checkpoint owner while every producer
    /// is held. The mutex binds both durable prefixes and all eight host maps/
    /// counters to one exact chronology boundary.
    pub fn checkpoint_continuation(&self,run_root:&Path)->Result<ScientificChronologyContinuationV1> {
        let state=self.inner.state.lock().map_err(|_|anyhow::anyhow!("chronology checkpoint lock poisoned"))?;
        let seal=|path:&Path|->Result<_>{let file=open_chronology_checkpoint_file(path)?;
            let relative=path.strip_prefix(run_root)?.to_str().context("non-UTF8 chronology path")?;
            let cursor=crate::scientific_checkpoint::seal_append_file(&file,relative)?;Ok((file,cursor))};
        let (epoch_file,epochs)=seal(&self.inner.epochs_path)?;let(event_file,events)=seal(&self.inner.events_path)?;
        let observed=recount_chronology_prefixes(&epoch_file,epochs.committed_bytes,&event_file,events.committed_bytes)?;
        anyhow::ensure!(observed==*state,"chronology host counts/maps differ from complete persisted population");
        let metadata=|path:&Path|->Result<_>{
            let contents=match std::fs::symlink_metadata(path){
                Ok(m)=>{anyhow::ensure!(m.is_file()&&!m.file_type().is_symlink(),"chronology metadata is not a regular file");
                    let bytes=crate::checkpoint_output::read_metadata(path)?;open_chronology_checkpoint_file(path)?.sync_all()?;Some(bytes)},
                Err(e)if e.kind()==std::io::ErrorKind::NotFound=>None,Err(e)=>return Err(e.into()),};
            Ok(ChronologyMetadataContinuationV1{relative_path:path.strip_prefix(run_root)?.to_str().context("non-UTF8 chronology metadata path")?.into(),contents})};
        Ok(ScientificChronologyContinuationV1{schema_version:1,epochs,events,state:state.clone(),
            metadata:[metadata(&self.inner.manifest_path)?,metadata(&self.inner.acceptance_decision_path)?]})
    }
    /// Fresh-process ingress. This replaces initialize() for continuation and
    /// returns the shared Arc owner with exact saved counters and file prefixes.
    pub fn reopen_checkpoint(run_root:&Path,saved:&ScientificChronologyContinuationV1)->Result<Self> {
        anyhow::ensure!(saved.schema_version==1 && saved.epochs.relative_path=="scientific_state_epochs.jsonl"
            && saved.events.relative_path=="scientific_chronology_events.jsonl"
            && saved.metadata[0].relative_path=="scientific_chronology_manifest.json"
            && saved.metadata[1].relative_path=="scientific_chronology_acceptance_decision.json",
            "chronology continuation schema/owner paths differ");
        let output=Self::new(run_root);
        // Validate both full committed populations before retiring any tail.
        let epochs=open_chronology_checkpoint_file(output.epochs_path())?;
        let events=open_chronology_checkpoint_file(output.events_path())?;
        chronology_prefix(&epochs,&saved.epochs)?;chronology_prefix(&events,&saved.events)?;
        let observed=recount_chronology_prefixes(&epochs,saved.epochs.committed_bytes,&events,saved.events.committed_bytes)?;
        anyhow::ensure!(observed==saved.state,"chronology saved counters/maps do not match the exact committed population");
        drop(epochs);drop(events);
        drop(crate::scientific_checkpoint::reopen_append_file(run_root,&saved.epochs)?);
        drop(crate::scientific_checkpoint::reopen_append_file(run_root,&saved.events)?);
        for item in &saved.metadata {
            let path=run_root.join(&item.relative_path);
            let current=match std::fs::symlink_metadata(&path){
                Ok(m)=>{anyhow::ensure!(m.is_file()&&!m.file_type().is_symlink(),"chronology metadata restore target invalid");Some(crate::checkpoint_output::read_metadata(&path)?)},
                Err(e)if e.kind()==std::io::ErrorKind::NotFound=>None,Err(e)=>return Err(e.into()),};
            if current==item.contents{continue;}
            if current.is_some(){crate::checkpoint_output::retain_later_file(run_root,&path)?;}
            if let Some(bytes)=&item.contents {
                let digest=format!("{:x}",Sha256::digest(bytes));
                let temp=run_root.join(format!(".{}.{}.{}.{}.restore",item.relative_path,digest,std::process::id(),current_unix_ns()));
                let mut file=std::fs::OpenOptions::new().write(true).create_new(true).open(&temp)?;
                file.write_all(bytes)?;file.sync_all()?;drop(file);std::fs::rename(&temp,&path)?;
                std::fs::File::open(run_root)?.sync_all()?;
            }
        }
        *output.inner.state.lock().map_err(|_|anyhow::anyhow!("chronology restored lock poisoned"))?=saved.state.clone();
        Ok(output)
    }
}

#[cfg(test)]
mod continuation_tests {
    use super::*;
    #[test]
    fn chronology_restart_retires_later_tail_and_preserves_all_maps_and_metadata() {
        let root=tempfile::tempdir().unwrap();let sink=ScientificChronologySinks::new(root.path());sink.initialize().unwrap();
        sink.append_epoch(serde_json::json!({"epoch_id":"e0","status":"OPEN"})).unwrap();
        sink.append_event(serde_json::json!({"event_kind":"EPOCH_OPEN","status":"PASS"})).unwrap();
        sink.append_event(serde_json::json!({"event_kind":"RESIDUE_ACTIVITY","status":"BLOCKED_POLICY","scientific_event_state":"BLOCKED_POLICY"})).unwrap();
        sink.write_manifest(serde_json::json!({"stage":"checkpoint"})).unwrap();
        let saved=sink.checkpoint_continuation(root.path()).unwrap();let state=crate::checkpoint_codec::to_bytes(&saved).unwrap();
        let expected_epochs=std::fs::read(sink.epochs_path()).unwrap();let expected_events=std::fs::read(sink.events_path()).unwrap();
        let expected_manifest=std::fs::read(sink.manifest_path()).unwrap();
        sink.append_event(serde_json::json!({"event_kind":"LATER_EVENT","status":"PASS"})).unwrap();
        sink.update_epoch("e0",serde_json::json!({"status":"CLOSED"})).unwrap();
        sink.write_manifest(serde_json::json!({"stage":"later"})).unwrap();
        sink.write_acceptance_decision(&BTreeMap::new(),&[],Value::Null).unwrap();drop(sink);
        let decoded=crate::checkpoint_codec::from_bytes(&state).unwrap();
        let reopened=ScientificChronologySinks::reopen_checkpoint(root.path(),&decoded).unwrap();
        assert_eq!(std::fs::read(reopened.epochs_path()).unwrap(),expected_epochs);
        assert_eq!(std::fs::read(reopened.events_path()).unwrap(),expected_events);
        assert_eq!(std::fs::read(reopened.manifest_path()).unwrap(),expected_manifest);
        assert!(!reopened.acceptance_decision_path().exists());
        assert_eq!(crate::checkpoint_codec::to_bytes(&reopened.checkpoint_continuation(root.path()).unwrap()).unwrap(),state);
        reopened.append_event(serde_json::json!({"event_kind":"RESUMED_EVENT","status":"PENDING_INPUT"})).unwrap();
        assert_eq!(reopened.checkpoint_continuation(root.path()).unwrap().state.event_record_count,3);
        assert!(root.path().join(".continuation_tails").exists());
    }
    #[test]
    fn chronology_refuses_changed_counts_and_corrupt_prefix_before_truncating_any_tail() {
        let root=tempfile::tempdir().unwrap();let sink=ScientificChronologySinks::new(root.path());sink.initialize().unwrap();
        sink.append_event(serde_json::json!({"event_kind":"EPOCH_OPEN"})).unwrap();
        let mut saved=sink.checkpoint_continuation(root.path()).unwrap();saved.state.event_record_count+=1;
        assert!(ScientificChronologySinks::reopen_checkpoint(root.path(),&saved).is_err());
        saved.state.event_record_count-=1;
        sink.append_epoch(serde_json::json!({"epoch_id":"later"})).unwrap();let later=std::fs::read(sink.epochs_path()).unwrap();
        std::fs::write(sink.events_path(),b"corrupt\n").unwrap();
        assert!(ScientificChronologySinks::reopen_checkpoint(root.path(),&saved).is_err());
        assert_eq!(std::fs::read(sink.epochs_path()).unwrap(),later);
    }
}

impl ScientificChronologyContinuationV1 {
    pub fn branch_copy_allocation_bound(&self)->Result<u64> {
        use crate::scientific_checkpoint::restart_transaction::branch_copy_member_bound;
        let mut bytes=branch_copy_member_bound(self.epochs.committed_bytes)?.checked_add(branch_copy_member_bound(self.events.committed_bytes)?).context("chronology branch bound overflow")?;
        for item in &self.metadata {if let Some(contents)=&item.contents {bytes=bytes.checked_add(branch_copy_member_bound(contents.len() as u64)?).context("chronology metadata branch bound overflow")?;}}
        Ok(bytes)
    }

    pub fn fork_outputs(&self,source:&Path,destination:&Path)->Result<()> {
        crate::scientific_checkpoint::restart_transaction::copy_committed_append_prefix(source,destination,&self.epochs)?;
        crate::scientific_checkpoint::restart_transaction::copy_committed_append_prefix(source,destination,&self.events)?;
        // Existing owner performs exact recount and restores its metadata/state.
        drop(ScientificChronologySinks::reopen_checkpoint(destination,self)?);Ok(())
    }
}
