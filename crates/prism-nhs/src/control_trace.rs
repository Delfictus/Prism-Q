//! Control-plane trace schema for audit records emitted outside the
//! runtime hot path.
//!
//! This module is deliberately data-only. It defines the typed wire
//! records and a small NDJSON writer helper; it does not read device
//! state, alter graph topology, or install runtime hooks.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::io::{self, Write};

/// Chronometric gearbox gear identifier.
///
/// The documented encoding is 0 = 0.5 fs, 1 = 2.0 fs, 2 = 4.0 fs (HMR),
/// 3 = abort-trap. Every legal gear is a small integer, so the type has NO
/// in-band way to say "unknown" — which is precisely why an unmeasured gear
/// must be carried as [`GearObservation`] with `gear: None`, never as a
/// plausible-looking `0`.
pub type GearId = u8;

/// Current control trace schema version emitted by this module.
///
/// **v2 (GEAR-PROVENANCE-002)** — `ControlPlaneRecord::G26` no longer carries
/// bare `previous_gear`/`current_gear` integers. Each gear is a
/// [`GearObservation`] that states where the value came from, and the
/// REQUESTED / AUTHORIZED / DEVICE-MEASURED classes are separate fields so a
/// later analyst cannot read "policy asked for gear 0" as "device integrated
/// at gear 0". `dt_ps` and `switch_selector` became explicitly optional with
/// their own origins for the same reason.
///
/// The enum is `deny_unknown_fields`, so a v1 reader handed a v2 record fails
/// loudly on the new keys and a v2 reader handed a v1 record fails loudly on
/// the missing ones. Neither silently misparses.
pub const CONTROL_TRACE_SCHEMA_VERSION: u32 = 2;

/// How a gear value in a control record was obtained.
///
/// This is the field that separates measurement from intent. Only
/// [`GearOrigin::DeviceReadback`] asserts that the device actually held the
/// gear; everything else is a request, an authorization, a derivation, or an
/// explicit absence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GearOrigin {
    /// Copied out of `ChronometricStateTensor.current_gear` /
    /// `.previous_gear` on the device. THE ONLY MEASUREMENT CLASS.
    DeviceReadback,
    /// A host-side control policy asked for this gear. Says nothing about
    /// what the device integrated at.
    HostPolicyRequest,
    /// A host override was written to `adj->gear_override`.
    HostOverride,
    /// Derived from other measured device state (for example, the predicate
    /// bridge's `override == 0xFF ? cruise.current_gear : override & 3`).
    DerivedFromDeviceState,
    /// No value of this class exists for this record. The accompanying
    /// `detail` names why.
    Unmeasured,
    /// A readback was attempted and failed. The accompanying `detail` names
    /// the terminal condition.
    ReadFailed,
}

impl GearOrigin {
    /// True only for the class that asserts the device actually held the
    /// gear. Downstream temporal normalization must gate on this.
    pub fn is_measurement(self) -> bool {
        matches!(self, Self::DeviceReadback)
    }
}

/// A gear value together with its provenance, in which absence is
/// representable.
///
/// `gear: None` means "no value of this class exists"; it is never rendered
/// as a legal gear. `detail` is a stable machine-readable label naming the
/// mechanism or the failure, so a record is auditable without reading source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GearObservation {
    /// The gear, or `None` when this class has no value for this record.
    pub gear: Option<GearId>,
    /// How the value was obtained, or why it is absent.
    pub origin: GearOrigin,
    /// Stable label naming the exact mechanism or terminal condition.
    pub detail: String,
}

impl GearObservation {
    /// A gear copied off the device — the only constructor that yields a
    /// measurement.
    pub fn device_measured(gear: GearId, detail: impl Into<String>) -> Self {
        Self {
            gear: Some(gear),
            origin: GearOrigin::DeviceReadback,
            detail: detail.into(),
        }
    }

    /// A gear a host policy asked for. Explicitly not a measurement.
    pub fn requested(gear: GearId, detail: impl Into<String>) -> Self {
        Self {
            gear: Some(gear),
            origin: GearOrigin::HostPolicyRequest,
            detail: detail.into(),
        }
    }

    /// A gear computed from other measured device state.
    pub fn derived(gear: GearId, detail: impl Into<String>) -> Self {
        Self {
            gear: Some(gear),
            origin: GearOrigin::DerivedFromDeviceState,
            detail: detail.into(),
        }
    }

    /// No value of this class exists.
    pub fn unmeasured(detail: impl Into<String>) -> Self {
        Self {
            gear: None,
            origin: GearOrigin::Unmeasured,
            detail: detail.into(),
        }
    }

    /// A readback was attempted and failed.
    pub fn read_failed(detail: impl Into<String>) -> Self {
        Self {
            gear: None,
            origin: GearOrigin::ReadFailed,
            detail: detail.into(),
        }
    }

    /// True only when this observation is a device measurement carrying a
    /// value.
    pub fn is_measurement(&self) -> bool {
        self.gear.is_some() && self.origin.is_measurement()
    }
}

/// Provenance of a non-gear scalar (timestep, switch selector) on a control
/// record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarOrigin {
    /// Copied off the device at a synchronized boundary.
    DeviceReadback,
    /// Computed from other measured device state.
    DerivedFromDeviceState,
    /// Supplied by host configuration; not a measurement of what ran.
    HostConfigured,
    /// No value exists for this record.
    Unmeasured,
    /// A readback was attempted and failed.
    ReadFailed,
}

/// The bound that governed a device readback, plus where the bound came from.
///
/// Recorded on every G26 record so the wait is auditable: a run that reports
/// a timeout is distinguishable from a run that never waited, and the budget
/// is reproducible without reading source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReadbackBound {
    /// Effective bound in milliseconds. Never zero — a zero bound would mean
    /// an unbounded wait.
    pub budget_ms: u32,
    /// Stable label naming where `budget_ms` was resolved from.
    pub budget_origin: String,
    /// Microseconds actually spent waiting.
    pub waited_us: u64,
    /// True when the bound expired and the readback produced no measurement.
    pub expired: bool,
}

/// Required per-record control trace envelope.
///
/// These fields identify the trace schema, the logical run, the captured graph
/// launch/control epoch, and the per-run record order without requiring any
/// runtime wiring in this data-only module.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ControlTraceEnvelope {
    /// Version of the serialized control trace schema.
    pub schema_version: u32,
    /// Stable identifier for the simulation or analysis run.
    pub run_id: String,
    /// Stable identifier for the graph launch/control epoch being audited.
    pub graph_launch_id: String,
    /// Monotonic record index within `run_id`.
    pub record_idx: u64,
}

impl ControlTraceEnvelope {
    /// Construct an envelope using the current schema version.
    pub fn current(
        run_id: impl Into<String>,
        graph_launch_id: impl Into<String>,
        record_idx: u64,
    ) -> Self {
        Self {
            schema_version: CONTROL_TRACE_SCHEMA_VERSION,
            run_id: run_id.into(),
            graph_launch_id: graph_launch_id.into(),
            record_idx,
        }
    }
}

/// Serde bridge for scalars whose "not measured" state must be explicit.
///
/// A non-finite value serializes as JSON `null` and deserializes back to NaN.
/// Without this, an unmeasured scalar either becomes an unreadable bare `NaN`
/// token or gets replaced by a plausible number — and a plausible number is
/// the failure mode this module exists to prevent.
mod nullable_f64 {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(value: &f64, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if value.is_finite() {
            serializer.serialize_f64(*value)
        } else {
            serializer.serialize_none()
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<f64, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(Option::<f64>::deserialize(deserializer)?.unwrap_or(f64::NAN))
    }
}

/// Raw energy/control state cited by control-plane decisions.
///
/// The matching hash is stored beside this struct on each record that cites
/// energy state. Keeping both preserves replay diagnostics while allowing
/// downstream tamper checks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceEnergyState {
    /// Potential energy scalar observed by the decision source.
    ///
    /// Non-finite means "not measured" and serializes as an explicit JSON
    /// `null`; it round-trips back to NaN. An emitter that could not read the
    /// device MUST leave this non-finite rather than substitute a plausible
    /// number, and name the condition in `source`.
    #[serde(with = "nullable_f64")]
    pub potential_energy: f64,
    /// External work scalar observed by the decision source. Same
    /// not-measured convention as `potential_energy`.
    #[serde(with = "nullable_f64")]
    pub external_work: f64,
    /// Stable label naming the source reducer, ring, or control surface.
    pub source: String,
}

impl SourceEnergyState {
    /// Canonical `sha256:<hex>` digest of this state's own JSON encoding.
    ///
    /// Field order is the struct's declaration order, so the digest is
    /// reproducible from the serialized record alone. Emitters use this so
    /// the hash beside a record is derived from the record, never asserted
    /// independently of it.
    pub fn canonical_hash(&self) -> String {
        use sha2::{Digest, Sha256};
        let encoded = serde_json::to_vec(self).unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(&encoded);
        format!("sha256:{:x}", hasher.finalize())
    }
}

/// Stable route names for the F1 adjudicator SWITCH.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum F1Route {
    /// Noise cluster discarded before construction.
    Prune,
    /// Candidate cluster admitted to the construction branch.
    Construct,
    /// Invariant violation branch.
    Violation,
}

/// Host-visible WHILE loop exit state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WhileExitReason {
    /// Loop has not exited at the point this record was emitted.
    NotExited,
    /// Loop exited because the evaluated condition became false.
    ConditionFalse,
    /// Loop exited because the configured iteration bound was reached.
    MaxIterations,
    /// Loop exited because the watchdog tripped.
    WatchdogExpired,
    /// Loop exited due to an external host-side stop/cancel request.
    ExternalStop,
}

/// Top-level control-plane audit event.
///
/// The enum uses an externally visible `kind` tag so NDJSON consumers can
/// branch on record type without probing nullable fields. Every variant
/// carries only fields required for that control-plane event class.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", deny_unknown_fields)]
pub enum ControlPlaneRecord {
    /// G26 chronometric gearbox state transition or steady-state observation.
    ///
    /// **Provenance contract (schema v2).** Three gear classes are kept in
    /// separate fields and must never be conflated:
    ///
    /// * `requested_gear` — what a host control policy asked for.
    /// * `authorized_gear` — what the predicate bridge was allowed to route
    ///   (host override composed with the device SFA's choice).
    /// * `current_gear` / `previous_gear` — the DEVICE-MEASURED pair, copied
    ///   out of `ChronometricStateTensor` at a synchronized boundary.
    ///
    /// Only an observation whose [`GearOrigin`] is `DeviceReadback` states
    /// what the integrator actually ran at. A consumer computing physical
    /// time MUST gate on [`GearObservation::is_measurement`] and fall back to
    /// `dt_ps` (which carries its own origin) otherwise.
    G26 {
        /// Required per-record trace envelope.
        envelope: ControlTraceEnvelope,
        /// Monotonic host-side frame index associated with the observation.
        frame_idx: u64,
        /// Runtime stream identifier that owns the observed control decision.
        stream_id: u32,
        /// Gear a host control policy REQUESTED. Not a measurement.
        requested_gear: GearObservation,
        /// Gear AUTHORIZED for the switch after override composition.
        /// Not a measurement of what was integrated.
        authorized_gear: GearObservation,
        /// DEVICE-MEASURED gear before the decision was applied.
        previous_gear: GearObservation,
        /// DEVICE-MEASURED gear held after the decision — the only field that
        /// states the resolution the integrator actually ran at.
        current_gear: GearObservation,
        /// Effective timestep in picoseconds actually bound to
        /// `ProtocolState.dt`, or `None` when it was not read.
        dt_ps: Option<f64>,
        /// Provenance of `dt_ps`.
        dt_ps_origin: ScalarOrigin,
        /// Gear selection predicate value forwarded to the G26 switch, or
        /// `None` when it was not established.
        switch_selector: Option<u32>,
        /// Provenance of `switch_selector`.
        switch_selector_origin: ScalarOrigin,
        /// Bound that governed the device readback for this record.
        readback_bound: ReadbackBound,
        /// Raw source energy state used or cited by the G26 decision.
        source_energy_state: SourceEnergyState,
        /// Hash of `source_energy_state` for tamper detection.
        source_energy_state_hash: String,
    },

    /// F1 adjudicator SWITCH routing decision.
    F1 {
        /// Required per-record trace envelope.
        envelope: ControlTraceEnvelope,
        /// Monotonic host-side frame index associated with the decision.
        frame_idx: u64,
        /// Runtime stream identifier that owns the adjudicator instance.
        stream_id: u32,
        /// Cluster identifier consumed by the F1 adjudicator.
        cluster_id: u32,
        /// Raw adjudication code written by the upstream classifier.
        adjudication_code: u32,
        /// Semantic branch selected by the F1 SWITCH.
        route: F1Route,
        /// Bitmask of violated invariants observed by the adjudicator.
        violation_bits: u32,
        /// Raw source energy state used or cited by the F1 decision.
        source_energy_state: SourceEnergyState,
        /// Hash of `source_energy_state` for tamper detection.
        source_energy_state_hash: String,
    },

    /// Host-visible WHILE control record for bounded loop progress.
    While {
        /// Required per-record trace envelope.
        envelope: ControlTraceEnvelope,
        /// Stable loop name chosen by the caller.
        loop_name: String,
        /// Monotonic iteration index inside the named loop.
        iteration: u64,
        /// Maximum iteration count allowed for this loop instance.
        max_iterations: u64,
        /// Boolean condition value evaluated for this iteration.
        condition_value: bool,
        /// Caller-supplied action label taken after condition evaluation.
        action: String,
        /// Exit state observed after this condition/action checkpoint.
        exit_reason: WhileExitReason,
        /// Maximum host watchdog duration allowed for this loop instance.
        watchdog_limit_ms: u64,
        /// Host watchdog duration consumed by this loop instance.
        watchdog_elapsed_ms: u64,
        /// True when the watchdog forced or will force loop termination.
        watchdog_tripped: bool,
        /// Raw source energy state used or cited by the WHILE predicate or exit.
        source_energy_state: SourceEnergyState,
        /// Hash of `source_energy_state` for tamper detection.
        source_energy_state_hash: String,
    },

    /// Adaptive sensing controller decision record.
    Asc {
        /// Required per-record trace envelope.
        envelope: ControlTraceEnvelope,
        /// Monotonic ASC chunk index.
        chunk_idx: u64,
        /// Runtime stream identifier for the ASC observation source.
        stream_id: u32,
        /// Residues selected as ASC steering focus targets.
        focus_residues: Vec<u32>,
        /// Steering gain applied to the focus set.
        alpha_gain: f64,
        /// Number of independent observer groups contributing evidence.
        observer_group_count: u32,
        /// Scalar surprise score used by the controller for this decision.
        surprise_score: f64,
        /// Raw source energy state used or cited by the ASC decision.
        source_energy_state: SourceEnergyState,
        /// Hash of `source_energy_state` for tamper detection.
        source_energy_state_hash: String,
    },

    /// Manual or supervisory gearbox override.
    GearOverride {
        /// Required per-record trace envelope.
        envelope: ControlTraceEnvelope,
        /// Monotonic host-side frame index associated with the override.
        frame_idx: u64,
        /// Runtime stream identifier affected by the override.
        stream_id: u32,
        /// Gear active before the override request.
        previous_gear: GearId,
        /// Gear requested by the override.
        requested_gear: GearId,
        /// Stable source label, such as a CLI, test harness, or operator.
        source: String,
        /// Human-readable reason supplied by the override source.
        reason: String,
        /// Gear overrides are host mutations by definition and must serialize as `true`.
        #[serde(
            serialize_with = "serialize_true_bool",
            deserialize_with = "deserialize_true_bool"
        )]
        host_mutation: bool,
        /// Raw source energy state cited by the current gear/dt context.
        source_energy_state: SourceEnergyState,
        /// Hash of `source_energy_state` for tamper detection.
        source_energy_state_hash: String,
    },
}

fn serialize_true_bool<S>(value: &bool, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    if *value {
        serializer.serialize_bool(true)
    } else {
        Err(serde::ser::Error::custom(
            "GearOverride.host_mutation must be true",
        ))
    }
}

fn deserialize_true_bool<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: Deserializer<'de>,
{
    let value = bool::deserialize(deserializer)?;
    if value {
        Ok(value)
    } else {
        Err(serde::de::Error::custom(
            "GearOverride.host_mutation must be true",
        ))
    }
}

/// Write control-plane records as newline-delimited JSON.
///
/// The helper accepts any writer so callers can use a buffered file,
/// memory buffer, or test sink without this module owning filesystem policy.
pub fn write_control_plane_ndjson<W, I>(writer: &mut W, records: I) -> io::Result<()>
where
    W: Write,
    I: IntoIterator<Item = ControlPlaneRecord>,
{
    for record in records {
        serde_json::to_writer(&mut *writer, &record).map_err(io::Error::other)?;
        writer.write_all(b"\n")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn envelope(record_idx: u64) -> ControlTraceEnvelope {
        ControlTraceEnvelope::current("run-001", "graph-launch-001", record_idx)
    }

    fn source_energy_state() -> SourceEnergyState {
        SourceEnergyState {
            potential_energy: -12345.625,
            external_work: 0.03125,
            source: "adj_pe_wext_reducer".to_string(),
        }
    }

    fn source_energy_state_hash() -> String {
        "sha256:source-energy-state-v1".to_string()
    }

    fn source_energy_state_json() -> serde_json::Value {
        serde_json::json!({
            "potential_energy": -12345.625,
            "external_work": 0.03125,
            "source": "adj_pe_wext_reducer"
        })
    }

    fn record_energy_fields(record: &ControlPlaneRecord) -> (&SourceEnergyState, &str) {
        match record {
            ControlPlaneRecord::G26 {
                source_energy_state,
                source_energy_state_hash,
                ..
            }
            | ControlPlaneRecord::F1 {
                source_energy_state,
                source_energy_state_hash,
                ..
            }
            | ControlPlaneRecord::While {
                source_energy_state,
                source_energy_state_hash,
                ..
            }
            | ControlPlaneRecord::Asc {
                source_energy_state,
                source_energy_state_hash,
                ..
            }
            | ControlPlaneRecord::GearOverride {
                source_energy_state,
                source_energy_state_hash,
                ..
            } => (source_energy_state, source_energy_state_hash.as_str()),
        }
    }

    fn sample_records() -> Vec<ControlPlaneRecord> {
        vec![
            ControlPlaneRecord::G26 {
                envelope: envelope(0),
                frame_idx: 42,
                stream_id: 2,
                requested_gear: GearObservation::requested(1, "test_host_policy"),
                authorized_gear: GearObservation::derived(2, "test_predicate_bridge"),
                previous_gear: GearObservation::device_measured(1, "cruise.previous_gear"),
                current_gear: GearObservation::device_measured(2, "cruise.current_gear"),
                dt_ps: Some(0.004),
                dt_ps_origin: ScalarOrigin::DeviceReadback,
                switch_selector: Some(2),
                switch_selector_origin: ScalarOrigin::DerivedFromDeviceState,
                readback_bound: ReadbackBound {
                    budget_ms: 2000,
                    budget_origin: "source_literal_default".to_string(),
                    waited_us: 300,
                    expired: false,
                },
                source_energy_state: source_energy_state(),
                source_energy_state_hash: source_energy_state_hash(),
            },
            ControlPlaneRecord::F1 {
                envelope: envelope(1),
                frame_idx: 43,
                stream_id: 2,
                cluster_id: 7,
                adjudication_code: 1,
                route: F1Route::Construct,
                violation_bits: 0,
                source_energy_state: source_energy_state(),
                source_energy_state_hash: source_energy_state_hash(),
            },
            ControlPlaneRecord::While {
                envelope: envelope(2),
                loop_name: "chunk_window".to_string(),
                iteration: 3,
                max_iterations: 16,
                condition_value: true,
                action: "continue".to_string(),
                exit_reason: WhileExitReason::NotExited,
                watchdog_limit_ms: 250,
                watchdog_elapsed_ms: 37,
                watchdog_tripped: false,
                source_energy_state: source_energy_state(),
                source_energy_state_hash: source_energy_state_hash(),
            },
            ControlPlaneRecord::Asc {
                envelope: envelope(3),
                chunk_idx: 9,
                stream_id: 0,
                focus_residues: vec![12, 47],
                alpha_gain: 0.01,
                observer_group_count: 4,
                surprise_score: 2.5,
                source_energy_state: source_energy_state(),
                source_energy_state_hash: source_energy_state_hash(),
            },
            ControlPlaneRecord::GearOverride {
                envelope: envelope(4),
                frame_idx: 44,
                stream_id: 2,
                previous_gear: 2,
                requested_gear: 0,
                source: "test_harness".to_string(),
                reason: "deterministic coverage".to_string(),
                host_mutation: true,
                source_energy_state: source_energy_state(),
                source_energy_state_hash: source_energy_state_hash(),
            },
        ]
    }

    #[test]
    fn records_round_trip_with_kind_tags() {
        let expected_kinds = ["G26", "F1", "While", "Asc", "GearOverride"];

        for (record, expected_kind) in sample_records().into_iter().zip(expected_kinds) {
            let encoded = serde_json::to_string(&record).unwrap();
            let json: serde_json::Value = serde_json::from_str(&encoded).unwrap();
            assert_eq!(json["kind"], serde_json::json!(expected_kind));
            assert!(encoded.contains("\"schema_version\":"));
            assert!(encoded.contains("\"run_id\":"));
            assert!(encoded.contains("\"graph_launch_id\":"));

            let decoded: ControlPlaneRecord = serde_json::from_str(&encoded).unwrap();
            assert_eq!(decoded, record);
        }
    }

    #[test]
    fn ndjson_writer_emits_one_record_per_line() {
        let records = sample_records();
        let mut buf = Vec::new();

        write_control_plane_ndjson(&mut buf, records.clone()).unwrap();

        let text = String::from_utf8(buf).unwrap();
        assert!(text.ends_with('\n'));
        assert_eq!(
            text.bytes().filter(|byte| *byte == b'\n').count(),
            records.len()
        );
        assert!(!text.contains("\n\n"));

        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), records.len());

        for (line, expected) in lines.into_iter().zip(records) {
            let decoded: ControlPlaneRecord = serde_json::from_str(line).unwrap();
            assert_eq!(decoded, expected);
        }
    }

    #[test]
    fn serialization_includes_raw_source_energy_state_and_hash() {
        for record in sample_records() {
            let json = serde_json::to_value(&record).unwrap();
            assert_eq!(json["source_energy_state"], source_energy_state_json());
            assert_eq!(
                json["source_energy_state_hash"],
                serde_json::json!("sha256:source-energy-state-v1")
            );
        }
    }

    #[test]
    fn source_energy_state_round_trip_preserves_raw_values_and_hash() {
        for record in sample_records() {
            let encoded = serde_json::to_string(&record).unwrap();
            let decoded: ControlPlaneRecord = serde_json::from_str(&encoded).unwrap();

            assert_eq!(decoded, record);
            let (source_energy_state, source_energy_state_hash) = record_energy_fields(&decoded);
            assert_eq!(source_energy_state.potential_energy, -12345.625);
            assert_eq!(source_energy_state.external_work, 0.03125);
            assert_eq!(source_energy_state.source, "adj_pe_wext_reducer");
            assert_eq!(source_energy_state_hash, "sha256:source-energy-state-v1");
        }
    }

    #[test]
    fn missing_required_variant_fields_are_rejected() {
        let json = serde_json::json!({
            "kind": "G26",
            "envelope": {
                "schema_version": CONTROL_TRACE_SCHEMA_VERSION,
                "run_id": "run-001",
                "graph_launch_id": "graph-launch-001",
                "record_idx": 0
            },
            "frame_idx": 42,
            "stream_id": 2,
            "requested_gear": { "gear": 1, "origin": "host_policy_request", "detail": "t" },
            "authorized_gear": { "gear": 2, "origin": "derived_from_device_state", "detail": "t" },
            "previous_gear": { "gear": 1, "origin": "device_readback", "detail": "t" },
            "current_gear": { "gear": 2, "origin": "device_readback", "detail": "t" },
            "dt_ps": 0.004,
            "dt_ps_origin": "device_readback",
            "switch_selector": 2,
            "switch_selector_origin": "derived_from_device_state",
            "readback_bound": {
                "budget_ms": 2000,
                "budget_origin": "source_literal_default",
                "waited_us": 300,
                "expired": false
            },
            "source_energy_state": source_energy_state_json()
        });

        let err = serde_json::from_value::<ControlPlaneRecord>(json).unwrap_err();
        assert!(err
            .to_string()
            .contains("missing field `source_energy_state_hash`"));
    }

    /// A v1 G26 record must FAIL LOUDLY against the v2 reader rather than be
    /// silently accepted with its bare integer gears reinterpreted as
    /// measurements.
    #[test]
    fn v1_g26_record_is_rejected_by_the_v2_reader() {
        let json = serde_json::json!({
            "kind": "G26",
            "envelope": {
                "schema_version": 1,
                "run_id": "run-001",
                "graph_launch_id": "graph-launch-001",
                "record_idx": 0
            },
            "frame_idx": 42,
            "stream_id": 2,
            "previous_gear": 1,
            "current_gear": 2,
            "dt_ps": 0.001,
            "switch_selector": 2,
            "source_energy_state": source_energy_state_json(),
            "source_energy_state_hash": source_energy_state_hash()
        });

        let err = serde_json::from_value::<ControlPlaneRecord>(json).unwrap_err();
        let text = err.to_string();
        assert!(
            text.contains("invalid type")
                || text.contains("missing field")
                || text.contains("unknown field"),
            "v1 G26 must not decode into the v2 schema; got: {text}"
        );
    }

    /// Only a device readback may be treated as a statement about the
    /// resolution the integrator ran at.
    #[test]
    fn only_device_readback_counts_as_a_gear_measurement() {
        assert!(GearObservation::device_measured(2, "cruise.current_gear").is_measurement());
        assert!(!GearObservation::requested(0, "host_policy").is_measurement());
        assert!(!GearObservation::derived(1, "predicate_bridge").is_measurement());
        assert!(!GearObservation::unmeasured("gearbox_inactive").is_measurement());
        assert!(!GearObservation::read_failed("bounded_wait_expired").is_measurement());

        // An absent gear must serialize as an explicit null, never as 0.
        let json = serde_json::to_value(GearObservation::unmeasured("gearbox_inactive")).unwrap();
        assert_eq!(json["gear"], serde_json::Value::Null);
        assert_eq!(json["origin"], serde_json::json!("unmeasured"));
    }

    /// The energy-state hash must be derived from the state it accompanies.
    #[test]
    fn canonical_hash_is_derived_from_the_state() {
        let a = source_energy_state();
        let mut b = source_energy_state();
        b.potential_energy = -12345.5;

        let ha = a.canonical_hash();
        assert!(ha.starts_with("sha256:"));
        assert_eq!(ha, a.canonical_hash());
        assert_ne!(ha, b.canonical_hash());
    }

    #[test]
    fn flat_option_bag_fields_are_rejected() {
        let json = serde_json::json!({
            "kind": "F1",
            "envelope": {
                "schema_version": CONTROL_TRACE_SCHEMA_VERSION,
                "run_id": "run-001",
                "graph_launch_id": "graph-launch-001",
                "record_idx": 1
            },
            "frame_idx": 43,
            "stream_id": 2,
            "cluster_id": 7,
            "adjudication_code": 1,
            "route": "construct",
            "violation_bits": 0,
            "source_energy_state": source_energy_state_json(),
            "source_energy_state_hash": source_energy_state_hash(),
            "g26_branch": null
        });

        let err = serde_json::from_value::<ControlPlaneRecord>(json).unwrap_err();
        assert!(err.to_string().contains("unknown field `g26_branch`"));
    }

    #[test]
    fn gear_override_rejects_non_host_mutation() {
        let json = serde_json::json!({
            "kind": "GearOverride",
            "envelope": {
                "schema_version": CONTROL_TRACE_SCHEMA_VERSION,
                "run_id": "run-001",
                "graph_launch_id": "graph-launch-001",
                "record_idx": 0
            },
            "frame_idx": 44,
            "stream_id": 2,
            "previous_gear": 2,
            "requested_gear": 0,
            "source": "test_harness",
            "reason": "must be rejected",
            "host_mutation": false,
            "source_energy_state": source_energy_state_json(),
            "source_energy_state_hash": source_energy_state_hash()
        });

        let err = serde_json::from_value::<ControlPlaneRecord>(json).unwrap_err();
        assert!(err.to_string().contains("host_mutation must be true"));
    }

    #[test]
    fn gear_override_rejects_false_host_mutation_on_serialize() {
        let record = ControlPlaneRecord::GearOverride {
            envelope: envelope(4),
            frame_idx: 44,
            stream_id: 2,
            previous_gear: 2,
            requested_gear: 0,
            source: "test_harness".to_string(),
            reason: "must be rejected".to_string(),
            host_mutation: false,
            source_energy_state: source_energy_state(),
            source_energy_state_hash: source_energy_state_hash(),
        };

        let err = serde_json::to_string(&record).unwrap_err();
        assert!(err.to_string().contains("host_mutation must be true"));
    }
}
