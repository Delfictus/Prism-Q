//! Rust-native Parquet writer with PRISM-DSTW propagation sidecars.

use std::{
    collections::{BTreeMap, HashMap},
    fs::{self, File, OpenOptions},
    io::{BufReader, Read, Seek, Write},
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context, Result};
use arrow_array::{
    Array, ArrayRef, Float32Array, Float64Array, Int32Array, LargeStringArray, RecordBatch,
    StringArray, UInt32Array, UInt64Array, UInt8Array,
};
use arrow_schema::{DataType, Field, Schema};
use chrono::Utc;
use parquet::{
    arrow::{arrow_reader::ParquetRecordBatchReaderBuilder, ArrowWriter},
    basic::{Compression, ZstdLevel},
    file::{metadata::KeyValue, properties::WriterProperties},
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use uuid::Uuid;

/// Canonical n80 integrated output directory.
pub const DEFAULT_OUTPUT_DIR: &str =
    "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale";

/// GLP-1R campaign identifier.
pub const CAMPAIGN_ID: &str = "glp1r_aleniglipron";

/// Stream identifier boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StreamId(pub u8);

/// Replica identifier boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ReplicaId(pub u16);

/// Voxel index boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VoxelIdx(pub u32);

/// Atom index boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AtomIdx(pub u32);

/// Aromatic-ring index boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RingIdx(pub u32);

/// Signal-grid hit count boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HitCount(pub i32);

/// Hit-count mean boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HitMean(pub f64);

/// Mechanical-load scalar boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MechanicalLoad(pub f32);

/// Shear-stress scalar boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShearStress(pub f32);

/// Strict three-vector boundary for raw f32 vector math.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    /// X component.
    pub x: f32,
    /// Y component.
    pub y: f32,
    /// Z component.
    pub z: f32,
}

impl Vec3 {
    /// Construct a vector from components.
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
}

/// Force-vector boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ForceVector(pub Vec3);

/// Allosteric-signaling-corridor vector boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AscVector(pub Vec3);

/// Compute mechanical load through a typed dot product.
pub fn mechanical_dot(force: ForceVector, asc: AscVector) -> MechanicalLoad {
    let f = force.0;
    let a = asc.0;
    MechanicalLoad(f.x * a.x + f.y * a.y + f.z * a.z)
}

/// Parsed stream path with campaign identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamPath {
    /// Raw artifact path.
    pub path: PathBuf,
    /// Condition / topology label.
    pub condition_id: String,
    /// Replica index.
    pub replica_id: ReplicaId,
    /// Stream index.
    pub stream_id: StreamId,
}

/// Per-stream thermal phase row loaded from protocol summary parquet.
#[derive(Debug, Clone)]
pub struct ProtocolPhase {
    /// Exact protocol phase.
    pub thermal_phase: String,
    /// Coarse cold/warm class.
    pub thermal_class: String,
    /// Protocol step represented by this stream.
    pub current_step: i32,
    /// Current temperature in Kelvin.
    pub current_temperature_k: f64,
}

/// Options for provenance parquet writing.
#[derive(Debug, Clone)]
pub struct ProvenanceOptions {
    /// Producing module name.
    pub module: String,
    /// Schema version string.
    pub schema_version: String,
    /// Producer script/binary path label.
    pub producer: String,
    /// Input raw files consumed.
    pub input_paths: Vec<PathBuf>,
    /// Additional source parquet inputs.
    pub source_parquets: Vec<PathBuf>,
    /// Partition key names.
    pub partition_keys: Vec<String>,
    /// Physical/math parameters.
    pub parameters: Value,
}

/// Streaming parquet writer that appends a Python-compatible propagation sidecar at close.
pub struct ProvenanceParquetWriter {
    writer: ArrowWriter<File>,
    output_path: PathBuf,
    options: ProvenanceOptions,
    row_count: u64,
}

impl ProvenanceParquetWriter {
    /// Create a provenance writer for a schema.
    pub fn try_new(
        output_path: &Path,
        schema: Arc<Schema>,
        options: ProvenanceOptions,
    ) -> Result<Self> {
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create output dir {}", parent.display()))?;
        }
        let generator_hash = current_generator_hash()?;
        let source_checksums = checksum_map(&options.source_parquets)?;
        let raw_checksums = checksum_map(&options.input_paths)?;
        let metadata = vec![
            KeyValue::new(
                "created_by".to_string(),
                Some("prism-nhs-rust-provenance/0.1.0".to_string()),
            ),
            KeyValue::new(
                "generator_script".to_string(),
                Some(options.producer.clone()),
            ),
            KeyValue::new("generator_hash".to_string(), Some(generator_hash)),
            KeyValue::new(
                "schema_version".to_string(),
                Some(options.schema_version.clone()),
            ),
            KeyValue::new("pipeline_stage".to_string(), Some(options.module.clone())),
            KeyValue::new(
                "partition_keys".to_string(),
                Some(serde_json::to_string(&options.partition_keys)?),
            ),
            KeyValue::new(
                "source_parquets".to_string(),
                Some(serde_json::to_string(&source_checksums)?),
            ),
            KeyValue::new(
                "raw_input_checksums".to_string(),
                Some(serde_json::to_string(&raw_checksums)?),
            ),
        ];
        let props = WriterProperties::builder()
            .set_compression(Compression::ZSTD(ZstdLevel::try_new(6)?))
            .set_key_value_metadata(Some(metadata))
            .build();
        let file = File::create(output_path)
            .with_context(|| format!("create parquet {}", output_path.display()))?;
        let writer = ArrowWriter::try_new(file, schema, Some(props))?;
        Ok(Self {
            writer,
            output_path: output_path.to_path_buf(),
            options,
            row_count: 0,
        })
    }

    /// Write one record batch.
    pub fn write(&mut self, batch: &RecordBatch) -> Result<()> {
        self.row_count += batch.num_rows() as u64;
        self.writer.write(batch)?;
        self.writer.flush()?;
        Ok(())
    }

    /// Close parquet and append the propagation JSONL sidecar.
    pub fn close(self) -> Result<u64> {
        self.writer.close()?;
        append_propagation_entries(&self.output_path, &self.options, self.row_count)?;
        Ok(self.row_count)
    }
}

/// Write a single-batch output with provenance.
pub fn write_single_batch_with_provenance(
    output_path: &Path,
    schema: Arc<Schema>,
    batch: RecordBatch,
    options: ProvenanceOptions,
) -> Result<u64> {
    let mut writer = ProvenanceParquetWriter::try_new(output_path, schema, options)?;
    writer.write(&batch)?;
    writer.close()
}

/// Compute SHA-256 for a file.
pub fn sha256_file(path: &Path) -> Result<String> {
    let file = File::open(path).with_context(|| format!("open for sha256 {}", path.display()))?;
    let mut reader = BufReader::with_capacity(1 << 20, file);
    let mut digest = Sha256::new();
    let mut buf = [0u8; 1 << 20];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        digest.update(&buf[..n]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

/// Raw URI form matching the Python sidecar convention.
pub fn raw_uri(path: &Path) -> Result<String> {
    let abs = path
        .canonicalize()
        .with_context(|| format!("canonicalize {}", path.display()))?;
    Ok(format!(
        "raw://{}",
        abs.display().to_string().trim_start_matches('/')
    ))
}

/// Compute a raw-URI keyed checksum map.
pub fn checksum_map(paths: &[PathBuf]) -> Result<BTreeMap<String, String>> {
    let mut out = BTreeMap::new();
    let mut sorted = paths.to_vec();
    sorted.sort();
    sorted.dedup();
    for path in sorted {
        if path.exists() {
            out.insert(raw_uri(&path)?, sha256_file(&path)?);
        }
    }
    Ok(out)
}

/// Discover stream files by suffix under the raw root.
pub fn discover_stream_files(raw_root: &Path, suffix: &str) -> Result<Vec<StreamPath>> {
    let mut out = Vec::new();
    visit_dirs(raw_root, &mut |path| {
        if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(suffix))
        {
            if let Some(parsed) = parse_stream_path(path, suffix) {
                out.push(parsed);
            }
        }
    })?;
    out.sort_by(|a, b| {
        a.condition_id
            .cmp(&b.condition_id)
            .then(a.replica_id.0.cmp(&b.replica_id.0))
            .then(a.stream_id.0.cmp(&b.stream_id.0))
            .then(a.path.cmp(&b.path))
    });
    Ok(out)
}

fn visit_dirs(dir: &Path, cb: &mut dyn FnMut(&Path)) -> Result<()> {
    for entry in fs::read_dir(dir).with_context(|| format!("read_dir {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            visit_dirs(&path, cb)?;
        } else {
            cb(&path);
        }
    }
    Ok(())
}

fn parse_stream_path(path: &Path, suffix: &str) -> Option<StreamPath> {
    let file_name = path.file_name()?.to_str()?;
    let marker = "_stream";
    let marker_idx = file_name.find(marker)?;
    let after = &file_name[marker_idx + marker.len()..];
    let digits: String = after.chars().take_while(|ch| ch.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    let stream_id = digits.parse::<u8>().ok()?;
    if !after[digits.len()..].starts_with('_') || !file_name.ends_with(suffix) {
        return None;
    }
    let replica_dir = path.parent()?.file_name()?.to_str()?;
    let replica_id = replica_dir.strip_prefix("replica_")?.parse::<u16>().ok()?;
    let condition_id = path.parent()?.parent()?.file_name()?.to_str()?.to_string();
    Some(StreamPath {
        path: path.to_path_buf(),
        condition_id,
        replica_id: ReplicaId(replica_id),
        stream_id: StreamId(stream_id),
    })
}

/// Filter stream paths using common CLI selectors.
pub fn filter_streams(
    files: &[StreamPath],
    condition_id: Option<&str>,
    replica_id: Option<u16>,
    max_streams: Option<usize>,
) -> Vec<StreamPath> {
    let iter = files.iter().filter(|item| {
        let condition_matches = match condition_id {
            Some(condition) => item.condition_id == condition,
            None => true,
        };
        let replica_matches = match replica_id {
            Some(replica) => item.replica_id.0 == replica,
            None => true,
        };
        condition_matches && replica_matches
    });
    match max_streams {
        Some(max) => iter.take(max).cloned().collect(),
        None => iter.cloned().collect(),
    }
}

/// Load an `f32` vector file whose layout is raw little-endian f32.
pub fn read_f32_file(path: &Path) -> Result<Vec<f32>> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    if bytes.len() % 4 != 0 {
        anyhow::bail!("{} byte length is not divisible by f32", path.display());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

/// Envelope header for `PRSGD001` signal grids.
#[derive(Debug, Clone)]
pub struct SignalGridHeader {
    /// Stream identifier from envelope.
    pub stream_id: StreamId,
    /// Record count from envelope.
    pub record_count: u64,
    /// Payload byte offset.
    pub payload_offset: u64,
    /// Payload size in bytes.
    pub payload_size: u64,
    /// Cubic grid dimension.
    pub grid_dim: u64,
    /// Voxel count.
    pub voxel_count: u64,
}

/// Read a PRSGD001 header and the first voxel-hit grid.
pub fn read_signal_hit_grid(path: &Path) -> Result<(SignalGridHeader, Vec<HitCount>)> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut magic = [0u8; 8];
    file.read_exact(&mut magic)?;
    if &magic != b"PRSGD001" {
        anyhow::bail!("{} expected PRSGD001 envelope", path.display());
    }
    let schema_version = read_u32(&mut file)?;
    if schema_version != 1 {
        anyhow::bail!(
            "{} unsupported signal schema {}",
            path.display(),
            schema_version
        );
    }
    let endian = read_u32(&mut file)?;
    if endian != 0x01020304 {
        anyhow::bail!("{} unsupported endian marker {}", path.display(), endian);
    }
    let stream_id = StreamId(read_u32(&mut file)? as u8);
    let run_len = read_u64(&mut file)? as usize;
    skip_exact(&mut file, run_len)?;
    let stem_len = read_u64(&mut file)? as usize;
    skip_exact(&mut file, stem_len)?;
    let record_count = read_u64(&mut file)?;
    let byte_stride = read_u64(&mut file)?;
    let payload_size = read_u64(&mut file)?;
    if byte_stride != 4 {
        anyhow::bail!(
            "{} expected i32 byte_stride=4, got {}",
            path.display(),
            byte_stride
        );
    }
    let payload_offset = file.stream_position()?;
    let grid_dim = read_u64(&mut file)?;
    let voxel_count = read_u64(&mut file)?;
    if voxel_count != record_count {
        anyhow::bail!("{} voxel_count != record_count", path.display());
    }
    let mut hits = Vec::with_capacity(voxel_count as usize);
    for _ in 0..voxel_count {
        hits.push(HitCount(read_i32(&mut file)?));
    }
    Ok((
        SignalGridHeader {
            stream_id,
            record_count,
            payload_offset,
            payload_size,
            grid_dim,
            voxel_count,
        },
        hits,
    ))
}

/// Load protocol-state phase map from `protocol_state_summary.parquet`.
pub fn load_protocol_phase_map(path: &Path) -> Result<HashMap<(String, u16, u8), ProtocolPhase>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
    let mut out = HashMap::new();
    while let Some(batch) = reader.next() {
        let batch = batch?;
        let condition = string_values(&batch, "condition_id")?;
        let replica = u8_col(&batch, "replica_id")?;
        let stream = u8_col(&batch, "stream_id")?;
        let thermal_phase = string_values(&batch, "thermal_phase")?;
        let thermal_class = string_values(&batch, "thermal_class")?;
        let current_step = i32_col(&batch, "current_step")?;
        let temp = f64_col(&batch, "current_temperature_K")?;
        for row in 0..batch.num_rows() {
            out.insert(
                (
                    condition[row].clone(),
                    replica.value(row) as u16,
                    stream.value(row),
                ),
                ProtocolPhase {
                    thermal_phase: thermal_phase[row].clone(),
                    thermal_class: thermal_class[row].clone(),
                    current_step: current_step.value(row),
                    current_temperature_k: temp.value(row),
                },
            );
        }
    }
    Ok(out)
}

fn append_propagation_entries(
    output_path: &Path,
    options: &ProvenanceOptions,
    row_count: u64,
) -> Result<()> {
    let ledger_path = output_path.with_extension("propagation.jsonl");
    let source_checksums = checksum_map(&options.source_parquets)?;
    let raw_checksums = checksum_map(&options.input_paths)?;
    let source_inputs: BTreeMap<String, Value> = options
        .source_parquets
        .iter()
        .enumerate()
        .map(|(idx, path)| (format!("source_{}", idx), json!(path.display().to_string())))
        .collect();
    append_jsonl(
        &ledger_path,
        &json!({
            "entry_id": Uuid::new_v4().to_string(),
            "module": options.module,
            "operation": "write_provenance_parquet",
            "inputs": source_inputs,
            "input_checksums": source_checksums,
            "parameters": options.parameters,
            "output_value": {"output_path": output_path.display().to_string(), "row_count": row_count},
            "output_uncertainty": null,
            "timestamp": Utc::now().to_rfc3339(),
            "gate_status": {
                "append_only_ledger": true,
                "arrow_rust_writer": true,
                "provenance_metadata": true,
                "repo_relative_paths": true
            },
            "supersedes": null
        }),
    )?;
    let raw_inputs: BTreeMap<String, Value> = options
        .input_paths
        .iter()
        .enumerate()
        .map(|(idx, path)| Ok((format!("raw_{}", idx), json!(raw_uri(path)?))))
        .collect::<Result<_>>()?;
    append_jsonl(
        &ledger_path,
        &json!({
            "entry_id": Uuid::new_v4().to_string(),
            "module": options.module,
            "operation": "raw_input_checksum_capture",
            "inputs": raw_inputs,
            "input_checksums": raw_checksums,
            "parameters": options.parameters,
            "output_value": {"output_path": output_path.display().to_string(), "row_count": row_count},
            "output_uncertainty": null,
            "timestamp": Utc::now().to_rfc3339(),
            "gate_status": {
                "raw_sha256": true,
                "external_raw_uri": true,
                "write_provenance_parquet_used": true
            },
            "supersedes": null
        }),
    )?;
    Ok(())
}

fn append_jsonl(path: &Path, value: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("append ledger {}", path.display()))?;
    file.write_all(serde_json::to_string(value)?.as_bytes())?;
    file.write_all(b"\n")?;
    Ok(())
}

fn current_generator_hash() -> Result<String> {
    let current = std::env::current_exe().context("current_exe")?;
    sha256_file(&current)
}

fn read_u32(file: &mut File) -> Result<u32> {
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32(file: &mut File) -> Result<i32> {
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_u64(file: &mut File) -> Result<u64> {
    let mut buf = [0u8; 8];
    file.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn skip_exact(file: &mut File, bytes: usize) -> Result<()> {
    let mut buf = vec![0u8; bytes];
    file.read_exact(&mut buf)?;
    Ok(())
}

fn string_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray> {
    let idx = batch.schema().index_of(name)?;
    batch
        .column(idx)
        .as_any()
        .downcast_ref::<StringArray>()
        .with_context(|| format!("column {} is not String", name))
}

fn string_values(batch: &RecordBatch, name: &str) -> Result<Vec<String>> {
    let idx = batch.schema().index_of(name)?;
    let column = batch.column(idx);
    if let Some(values) = column.as_any().downcast_ref::<StringArray>() {
        return Ok((0..values.len())
            .map(|row| values.value(row).to_string())
            .collect());
    }
    if let Some(values) = column.as_any().downcast_ref::<LargeStringArray>() {
        return Ok((0..values.len())
            .map(|row| values.value(row).to_string())
            .collect());
    }
    anyhow::bail!("column {} is not String or LargeString", name)
}

fn u8_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a UInt8Array> {
    let idx = batch.schema().index_of(name)?;
    batch
        .column(idx)
        .as_any()
        .downcast_ref::<UInt8Array>()
        .with_context(|| format!("column {} is not UInt8", name))
}

fn i32_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Int32Array> {
    let idx = batch.schema().index_of(name)?;
    batch
        .column(idx)
        .as_any()
        .downcast_ref::<Int32Array>()
        .with_context(|| format!("column {} is not Int32", name))
}

fn f64_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Float64Array> {
    let idx = batch.schema().index_of(name)?;
    batch
        .column(idx)
        .as_any()
        .downcast_ref::<Float64Array>()
        .with_context(|| format!("column {} is not Float64", name))
}

/// Convenience schema builder.
pub fn schema(fields: Vec<(&str, DataType, bool)>) -> Arc<Schema> {
    Arc::new(Schema::new(
        fields
            .into_iter()
            .map(|(name, data_type, nullable)| Field::new(name, data_type, nullable))
            .collect::<Vec<_>>(),
    ))
}

/// Build a record batch from typed columns.
pub fn record_batch(schema: Arc<Schema>, arrays: Vec<ArrayRef>) -> Result<RecordBatch> {
    RecordBatch::try_new(schema, arrays).context("RecordBatch::try_new")
}

/// Convert strings to an Arrow array.
pub fn strings(values: Vec<String>) -> ArrayRef {
    Arc::new(StringArray::from(values)) as ArrayRef
}

/// Convert f32 values to an Arrow array.
pub fn f32s(values: Vec<f32>) -> ArrayRef {
    Arc::new(Float32Array::from(values)) as ArrayRef
}

/// Convert f64 values to an Arrow array.
pub fn f64s(values: Vec<f64>) -> ArrayRef {
    Arc::new(Float64Array::from(values)) as ArrayRef
}

/// Convert i32 values to an Arrow array.
pub fn i32s(values: Vec<i32>) -> ArrayRef {
    Arc::new(Int32Array::from(values)) as ArrayRef
}

/// Convert u8 values to an Arrow array.
pub fn u8s(values: Vec<u8>) -> ArrayRef {
    Arc::new(UInt8Array::from(values)) as ArrayRef
}

/// Convert u32 values to an Arrow array.
pub fn u32s(values: Vec<u32>) -> ArrayRef {
    Arc::new(UInt32Array::from(values)) as ArrayRef
}

/// Convert u64 values to an Arrow array.
pub fn u64s(values: Vec<u64>) -> ArrayRef {
    Arc::new(UInt64Array::from(values)) as ArrayRef
}

/// Validate all arrays in a batch share row count.
pub fn assert_array_len(array: &ArrayRef, expected: usize, name: &str) -> Result<()> {
    if array.len() != expected {
        anyhow::bail!(
            "column {} has {} rows, expected {}",
            name,
            array.len(),
            expected
        );
    }
    Ok(())
}

#[derive(Debug, Serialize, Deserialize)]
struct _LedgerShapeDocumentation {
    entry_id: String,
    module: String,
    operation: String,
    inputs: BTreeMap<String, Value>,
    input_checksums: BTreeMap<String, String>,
    parameters: Value,
    output_value: Value,
    output_uncertainty: Option<f64>,
    timestamp: String,
    gate_status: BTreeMap<String, bool>,
    supersedes: Option<String>,
}
