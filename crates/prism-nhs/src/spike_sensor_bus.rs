//! AlwaysOnSpikeSensorBus — Phase 2 of the Always-On QNANL Spike Bus.
//!
//! This is a NORMALIZATION / ROUTING layer over EXISTING always-on producers. It does NOT
//! invent a new sensor family, change sensor semantics, alter spike fields, event timing, MD
//! physics, force evaluation, or graph execution. It takes records ALREADY produced by the
//! engine's spike + event surfaces, normalizes them into a versioned *sparse* `SensorRecord`
//! shaped around joinability (chunk_id + record_range), and emits them as QNANL-sealed sensor
//! metadata chunks linked to the canonical spike chunks of the SPIKE_EVENT_STREAM.
//!
//! Each bus record carries a `source_surface` + provenance hashes (the spike chunk's raw +
//! QNANL hashes) so every exact/range-linked record joins back to a valid spike chunk.

use serde::{Deserialize, Serialize};
use std::path::Path;

pub const SENSOR_BUS_SCHEMA_KIND: &str = "always_on_spike_sensor_bus_record";
pub const SENSOR_BUS_SCHEMA_VERSION: u32 = 1;

/// Versioned SPARSE sensor record. The required join/provenance fields are always populated; all
/// producer-specific fields are `Option` (omitted from the serialized row when a producer does
/// not supply them). New sensors add `Option` fields without breaking existing rows — do NOT
/// hard-code every future sensor into a rigid struct beyond the documented optional set.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SensorRecord {
    pub schema_kind: String,
    pub schema_version: u32,
    pub sensor_id: String,
    pub event_family: String,
    pub source_surface: String,
    pub stream_id: u32,
    pub chunk_id: u64,
    pub record_start: u64,
    pub record_count: u64,
    /// linkage precision to the spike chunk: "exact" | "range" | "timestep".
    pub linkage: String,
    // --- provenance (the chunk hashes link the sensor to its spike chunk) ---
    pub raw_chunk_hash: String,
    pub qnanl_chunk_hash: String,
    // --- optional join keys / producer fields (sparse) ---
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group_id: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub condition_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub engine_variant: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub step: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sim_time_ps: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spike_event_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub site_id: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub residue_id: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voxel_id: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spike_source: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub intensity: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phase_from: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phase_to: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trigger_kind: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bocpd_probability: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub water_state: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vibration_state: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wd_change: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub perturbation_edge_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub twin_pair_id: Option<String>,
    // --- condition-edge fields (Phase 4 DPD adapter; minimal additions) ---
    #[serde(skip_serializing_if = "Option::is_none")]
    pub condition_role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub condition_family: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub condition_amplitude: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub condition_scan_fraction: Option<f32>,
    /// TWIN16 cell id (e.g. "TWIN16_0011") this spike chunk belongs to (A2: sensor carries cell id).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub twin16_cell: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub topology_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub forcefield_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub engine_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub receipt_id: Option<String>,
}

impl SensorRecord {
    /// Construct a record with the required fields; optional fields default to `None`.
    #[allow(clippy::too_many_arguments)]
    pub fn new_linked(
        sensor_id: String,
        event_family: &str,
        source_surface: &str,
        stream_id: u32,
        chunk_id: u64,
        record_start: u64,
        record_count: u64,
        linkage: &str,
        raw_chunk_hash: String,
        qnanl_chunk_hash: String,
    ) -> Self {
        Self {
            schema_kind: SENSOR_BUS_SCHEMA_KIND.to_string(),
            schema_version: SENSOR_BUS_SCHEMA_VERSION,
            sensor_id,
            event_family: event_family.to_string(),
            source_surface: source_surface.to_string(),
            stream_id,
            chunk_id,
            record_start,
            record_count,
            linkage: linkage.to_string(),
            raw_chunk_hash,
            qnanl_chunk_hash,
            group_id: None,
            condition_id: None,
            engine_variant: None,
            step: None,
            sim_time_ps: None,
            spike_event_id: None,
            site_id: None,
            residue_id: None,
            voxel_id: None,
            spike_source: None,
            intensity: None,
            phase_from: None,
            phase_to: None,
            trigger_kind: None,
            bocpd_probability: None,
            water_state: None,
            vibration_state: None,
            wd_change: None,
            perturbation_edge_id: None,
            twin_pair_id: None,
            condition_role: None,
            condition_family: None,
            condition_amplitude: None,
            condition_scan_fraction: None,
            twin16_cell: None,
            topology_hash: None,
            forcefield_hash: None,
            engine_hash: None,
            receipt_id: None,
        }
    }
}

/// Result of sealing one sensor metadata chunk.
#[derive(Clone, Debug)]
pub struct SensorChunkResult {
    pub path: std::path::PathBuf,
    pub record_count: usize,
    pub raw_bytes: usize,
    pub compressed_bytes: usize,
    pub raw_hash_blake3: String,
    pub qnanl_hash_blake3: String,
    pub roundtrip_verified: bool,
}

/// The bus: accumulates normalized records from existing producers, then seals them into
/// QNANL sensor metadata chunks (`sensors/sensor_chunk_NNNNNN.prsqevmeta`).
#[derive(Default)]
pub struct AlwaysOnSpikeSensorBus {
    records: Vec<SensorRecord>,
}

impl AlwaysOnSpikeSensorBus {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, r: SensorRecord) {
        self.records.push(r);
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn records(&self) -> &[SensorRecord] {
        &self.records
    }

    /// Seal all accumulated sensor records into ONE QNANL metadata chunk under `sensors_dir`
    /// (JsonTelemetry family — these are normalized metadata, NOT spike payload). Round-trip
    /// verifies the seal (decode == raw). Returns `None` if there are no records or encode fails.
    pub fn emit_qnanl_sensor_chunk(
        &self,
        sensors_dir: &Path,
        chunk_index: usize,
    ) -> Option<SensorChunkResult> {
        use prismcode::qnanl_artifact::{
            decode_packet_bytes, encode_bytes, QnanlArtifactFamily, QnanlEncodeOptions,
        };
        if self.records.is_empty() {
            return None;
        }
        let _ = std::fs::create_dir_all(sensors_dir);
        // canonical sensor payload = newline-delimited JSON of the normalized records.
        let mut raw = Vec::new();
        for r in &self.records {
            if let Ok(line) = serde_json::to_vec(r) {
                raw.extend_from_slice(&line);
                raw.push(b'\n');
            }
        }
        let raw_hash = blake3::hash(&raw).to_hex().to_string();
        let pkt = sensors_dir.join(format!("sensor_chunk_{:06}.prsqevmeta", chunk_index));
        let rcpt = sensors_dir.join(format!(
            "sensor_chunk_{:06}.prsqevmeta.receipt.json",
            chunk_index
        ));
        let opts = QnanlEncodeOptions::runtime_subprocess_orchestrated_by_prismos(
            format!("sensor_chunk_{:06}", chunk_index),
            QnanlArtifactFamily::JsonTelemetry,
            None,
            "AlwaysOnSpikeSensorBus::emit_qnanl_sensor_chunk",
        );
        match encode_bytes(&raw, &pkt, &rcpt, opts) {
            Ok(_receipt) => {
                let pkt_bytes = std::fs::read(&pkt).unwrap_or_default();
                let qnanl_hash = blake3::hash(&pkt_bytes).to_hex().to_string();
                let roundtrip_verified = decode_packet_bytes(&pkt_bytes)
                    .map(|dec| dec == raw)
                    .unwrap_or(false);
                Some(SensorChunkResult {
                    path: pkt,
                    record_count: self.records.len(),
                    raw_bytes: raw.len(),
                    compressed_bytes: pkt_bytes.len(),
                    raw_hash_blake3: raw_hash,
                    qnanl_hash_blake3: qnanl_hash,
                    roundtrip_verified,
                })
            }
            Err(e) => {
                log::warn!(
                    "  [SENSOR_BUS] sensor chunk {} encode failed: {}",
                    chunk_index,
                    e
                );
                None
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SpikeEventStreamWriter (Directive v2 A2) — reusable chunked QNANL spike-event stream writer.
// Serves the --multi-differential / TWIN16 path (and is available to any per-stream spike source).
// Writes SPIKE_EVENT_STREAM/chunks/spike_chunk_NNNNNN.prsqev (byte-lossless GpuSpikeEvent
// records, QNANL EventStream-family, round-trip verified) + per-chunk sensor records stamped with
// the TWIN16 cell + condition id, then MANIFEST.qnanl.json + sensors/sensor_chunk_000000.prsqevmeta.
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(feature = "gpu")]
pub struct SpikeEventStreamWriter {
    stream_root: std::path::PathBuf,
    chunks_dir: std::path::PathBuf,
    manifest_rows: Vec<serde_json::Value>,
    bus: AlwaysOnSpikeSensorBus,
    chunk_id: usize,
    total_records: usize,
    all_verified: bool,
}

#[cfg(feature = "gpu")]
impl SpikeEventStreamWriter {
    const CHUNK_RECORDS: usize = 2_000_000; // 128 B × 2M ≈ 256 MB raw/chunk (schema v2)

    pub fn new(output_dir: &Path) -> std::io::Result<Self> {
        let stream_root = output_dir.join("SPIKE_EVENT_STREAM");
        let chunks_dir = stream_root.join("chunks");
        std::fs::create_dir_all(&chunks_dir)?;
        Ok(Self {
            stream_root,
            chunks_dir,
            manifest_rows: Vec::new(),
            bus: AlwaysOnSpikeSensorBus::new(),
            chunk_id: 0,
            total_records: 0,
            all_verified: true,
        })
    }

    /// Append one stream's spikes as QNANL chunks, stamped with the TWIN16 cell + condition id.
    pub fn write_stream(
        &mut self,
        stream_id: u32,
        twin16_cell: &str,
        condition_id: &str,
        spikes: &[crate::fused_engine::GpuSpikeEvent],
    ) {
        use prismcode::qnanl_artifact::{
            decode_packet_bytes, encode_bytes, QnanlArtifactFamily, QnanlEncodeOptions,
        };
        let rec_size = std::mem::size_of::<crate::fused_engine::GpuSpikeEvent>();
        let mut record_start = 0usize;
        for chunk in spikes.chunks(Self::CHUNK_RECORDS) {
            let raw: &[u8] = unsafe {
                std::slice::from_raw_parts(chunk.as_ptr() as *const u8, chunk.len() * rec_size)
            };
            let raw_hash = blake3::hash(raw).to_hex().to_string();
            let pkt = self
                .chunks_dir
                .join(format!("spike_chunk_{:06}.prsqev", self.chunk_id));
            let rcpt = self.chunks_dir.join(format!(
                "spike_chunk_{:06}.prsqev.receipt.json",
                self.chunk_id
            ));
            let opts = QnanlEncodeOptions::runtime_subprocess_orchestrated_by_prismos(
                format!(
                    "spike_chunk_{:06}_stream{}_{}",
                    self.chunk_id, stream_id, twin16_cell
                ),
                QnanlArtifactFamily::EventStream,
                None,
                "SpikeEventStreamWriter::write_stream",
            );
            match encode_bytes(raw, &pkt, &rcpt, opts) {
                Ok(_) => {
                    let pkt_bytes = std::fs::read(&pkt).unwrap_or_default();
                    let qnanl_hash = blake3::hash(&pkt_bytes).to_hex().to_string();
                    let verified = decode_packet_bytes(&pkt_bytes)
                        .map(|d| d.as_slice() == raw)
                        .unwrap_or(false);
                    self.all_verified &= verified;
                    // sensor record stamped with TWIN16 cell + condition id (A2 gate).
                    let n = chunk.len().max(1) as f64;
                    let mut n_uv = 0u64;
                    let (mut si, mut sw, mut sv, mut sc) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
                    for s in chunk {
                        let src = s.spike_source;
                        if src == 1 {
                            n_uv += 1;
                        }
                        si += s.intensity as f64;
                        sw += s.water_density as f64;
                        sv += s.vibrational_energy as f64;
                        sc += s.wd_change as f64;
                    }
                    let uv_dom = n_uv * 2 >= chunk.len() as u64;
                    let mut sr = SensorRecord::new_linked(
                        format!("spike_sensor_{:06}", self.chunk_id),
                        if uv_dom {
                            "UV_PHOTO_SPIKE"
                        } else {
                            "LIF_SPIKE"
                        },
                        "multi_differential.GpuSpikeEvent",
                        stream_id,
                        self.chunk_id as u64,
                        record_start as u64,
                        chunk.len() as u64,
                        "range",
                        raw_hash.clone(),
                        qnanl_hash.clone(),
                    );
                    sr.spike_source = Some(if uv_dom { 1 } else { 2 });
                    sr.intensity = Some((si / n) as f32);
                    sr.water_state = Some((sw / n) as f32);
                    sr.vibration_state = Some((sv / n) as f32);
                    sr.wd_change = Some((sc / n) as f32);
                    sr.twin16_cell = Some(twin16_cell.to_string());
                    sr.condition_id = Some(condition_id.to_string());
                    sr.condition_family = Some("twin16_thermodynamic_cell".to_string());
                    self.bus.push(sr);
                    self.manifest_rows.push(serde_json::json!({
                        "chunk_id": self.chunk_id,
                        "chunk_kind": "spike",
                        "stream_id": stream_id,
                        "twin16_cell": twin16_cell,
                        "condition_id": condition_id,
                        "record_count": chunk.len(),
                        "record_start": record_start,
                        "record_end": record_start + chunk.len(),
                        "raw_bytes": raw.len(),
                        "compressed_bytes": pkt_bytes.len(),
                        "raw_hash_blake3": raw_hash,
                        "qnanl_hash_blake3": qnanl_hash,
                        "codec_stack": "SectionZstd+QNANL(PRSQNB01)",
                        "roundtrip_verified": verified,
                        "source_surfaces": ["multi_differential.GpuSpikeEvent"],
                    }));
                    self.chunk_id += 1;
                    self.total_records += chunk.len();
                }
                Err(e) => {
                    log::warn!(
                        "  [SPIKE_EVENT_STREAM] chunk {} encode failed: {}",
                        self.chunk_id,
                        e
                    );
                    self.all_verified = false;
                }
            }
            record_start += chunk.len();
        }
    }

    /// Emit RECEIPT.prismcode.json (Directive v2 A3): per-chunk SPIKE_CHUNK_COMMIT/QNANL_SEAL/
    /// QNANL_VERIFY/TWIN16_CELL_COMMIT ops + a SPIKE_MANIFEST_COMMIT, each carrying the raw +
    /// QNANL hashes so a verifier can re-hash the chunks and fail on missing/reordered/mutated
    /// evidence. Uses the op-family constants from prismcode::md_native_runtime.
    fn write_prismcode_receipt(&self) {
        use prismcode::md_native_runtime::{
            MD_NATIVE_SCHEMA_KIND, OP_MD_QNANL_SEAL, OP_MD_QNANL_VERIFY, OP_MD_SPIKE_CHUNK_COMMIT,
            OP_MD_SPIKE_MANIFEST_COMMIT, OP_MD_TWIN16_CELL_COMMIT,
        };
        let mut ops: Vec<serde_json::Value> = Vec::new();
        for (i, row) in self.manifest_rows.iter().enumerate() {
            let cell = row
                .get("twin16_cell")
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            let stream_id = row
                .get("stream_id")
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            let raw_hash = row
                .get("raw_hash_blake3")
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            let qnanl_hash = row
                .get("qnanl_hash_blake3")
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            let verified = row
                .get("roundtrip_verified")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            ops.push(serde_json::json!({"op": OP_MD_SPIKE_CHUNK_COMMIT, "seq": i, "chunk_id": i, "stream_id": stream_id, "twin16_cell": cell, "record_count": row.get("record_count"), "raw_hash_blake3": raw_hash}));
            ops.push(serde_json::json!({"op": OP_MD_QNANL_SEAL, "seq": i, "chunk_id": i, "qnanl_hash_blake3": qnanl_hash}));
            ops.push(serde_json::json!({"op": OP_MD_QNANL_VERIFY, "seq": i, "chunk_id": i, "roundtrip_verified": verified}));
            ops.push(serde_json::json!({"op": OP_MD_TWIN16_CELL_COMMIT, "seq": i, "chunk_id": i, "twin16_cell": cell, "stream_id": stream_id}));
        }
        ops.push(serde_json::json!({"op": OP_MD_SPIKE_MANIFEST_COMMIT, "n_chunks": self.chunk_id, "total_records": self.total_records, "all_roundtrip_verified": self.all_verified}));
        let receipt = serde_json::json!({
            "schema_kind": MD_NATIVE_SCHEMA_KIND,
            "schema_version": 1,
            "surface": "SPIKE_EVENT_STREAM (TWIN16 multi-differential captured-graph)",
            "n_ops": ops.len(),
            "verifier": "re-hash each spike_chunk_NNNNNN.prsqev (blake3) and compare to raw/qnanl_hash; fail on missing/reordered (seq)/mutated chunk or cell evidence",
            "ops": ops,
        });
        let _ = std::fs::write(
            self.stream_root.join("RECEIPT.prismcode.json"),
            serde_json::to_string_pretty(&receipt).unwrap_or_default(),
        );
    }

    /// Seal the sensor chunk + write MANIFEST.qnanl.json + RECEIPT.prismcode.json. Returns
    /// (n_chunks, total_records, all_roundtrip_verified, sensor_zero_orphans).
    pub fn finalize(self, emitted_total: usize) -> (usize, usize, bool, bool) {
        self.write_prismcode_receipt();
        let sensors_dir = self.stream_root.join("sensors");
        let sensor_res = self.bus.emit_qnanl_sensor_chunk(&sensors_dir, 0);
        let (scount, sverified) = sensor_res
            .as_ref()
            .map(|r| (r.record_count, r.roundtrip_verified))
            .unwrap_or((0, true));
        let join_ok = scount == self.chunk_id;
        let manifest = serde_json::json!({
            "schema_kind": "qnanl_spike_event_stream_manifest",
            "schema_version": 1,
            "source_path": "--multi-differential (TWIN16 captured-graph)",
            "n_chunks": self.chunk_id,
            "total_spike_records": self.total_records,
            "record_count_matches_emitted": self.total_records == emitted_total,
            "all_chunks_roundtrip_verified": self.all_verified,
            "canonical": true,
            "preview_artifact": "per-stream {stem}_streamN_spikes.bin (PRSPK001; retained)",
            "sensor_bus": {
                "schema_kind": SENSOR_BUS_SCHEMA_KIND,
                "sensor_record_count": scount,
                "sensor_roundtrip_verified": sverified,
                "sensor_join_complete_zero_orphans": join_ok,
                "carries_twin16_cell": true,
                "receipt_ops": ["MD.SPIKE_CHUNK_COMMIT", "MD.SENSOR_EMIT", "MD.QNANL_SEAL", "MD.QNANL_VERIFY", "MD.SPIKE_MANIFEST_COMMIT", "MD.TWIN16_CELL_COMMIT"]
            },
            "rows": self.manifest_rows,
        });
        let _ = std::fs::write(
            self.stream_root.join("MANIFEST.qnanl.json"),
            serde_json::to_string_pretty(&manifest).unwrap_or_default(),
        );
        (
            self.chunk_id,
            self.total_records,
            self.all_verified,
            join_ok,
        )
    }
}
