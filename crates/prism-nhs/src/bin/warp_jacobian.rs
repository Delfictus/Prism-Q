use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{bail, Context, Result};
use arrow_array::RecordBatch;
use arrow_schema::{DataType, Field, Schema};
use clap::Parser;
use prism_nhs::io::provenance::{
    f64s, filter_streams, i32s, load_protocol_phase_map, read_f32_file, record_batch, strings,
    u32s, u64s, ProvenanceOptions, ProvenanceParquetWriter, StreamPath, VoxelIdx, CAMPAIGN_ID,
    DEFAULT_OUTPUT_DIR,
};
use rayon::prelude::*;
use serde_json::{json, Value};

const DEFAULT_PROTOCOL_STATE_SUMMARY: &str =
    "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/protocol_state_summary.parquet";
const WARP_ENTRY_BYTES: usize = 136;
const VOXEL_INDEX_OFFSET: usize = 0;
const ATOM_INDEX_OFFSET: usize = 4;
const ATOM_WEIGHT_OFFSET: usize = 68;
const N_ATOMS_OFFSET: usize = 132;
const MAX_WARP_ATOMS: usize = 16;
const ROW_GROUP_SIZE: usize = 65_536;

#[derive(Parser, Debug)]
struct Args {
    /// Raw campaign root (required; there is no compiled-in host default).
    #[arg(long)]
    raw_root: PathBuf,
    #[arg(long, default_value = DEFAULT_OUTPUT_DIR)]
    out_dir: PathBuf,
    /// Campaign topology root (required; there is no compiled-in host default).
    #[arg(long)]
    topology_root: PathBuf,
    #[arg(long, default_value = DEFAULT_PROTOCOL_STATE_SUMMARY)]
    protocol_state_summary: PathBuf,
    #[arg(long)]
    condition_id: Option<String>,
    #[arg(long)]
    replica_id: Option<u16>,
    #[arg(long)]
    max_streams: Option<usize>,
    #[arg(long)]
    frames: Option<String>,
    #[arg(long)]
    bifurcate: Option<String>,
}

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
struct StreamKey {
    condition_id: String,
    replica_id: u16,
    stream_id: u8,
}

#[derive(Debug, Clone)]
struct StreamInput {
    warp: StreamPath,
    signal_grid: StreamPath,
    vector_source_path: PathBuf,
}

#[derive(Debug, Clone, Copy)]
struct Vector3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Vector3 {
    const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    fn scale(self, scalar: f64) -> Self {
        Self::new(self.x * scalar, self.y * scalar, self.z * scalar)
    }

    fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }

    fn subtract(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }

    fn squared_norm(self) -> f64 {
        self.x
            .mul_add(self.x, self.y.mul_add(self.y, self.z * self.z))
    }
}

#[derive(Debug, Clone, Copy)]
struct ShearStress(f64);

impl ShearStress {
    fn new(value: f64) -> Result<Self> {
        if !value.is_finite() {
            bail!("ShearStress must be finite, got {value}");
        }
        if value < 0.0 {
            bail!("ShearStress must be non-negative, got {value}");
        }
        Ok(Self(value))
    }
}

#[derive(Debug, Clone, Default)]
struct WarpReadStats {
    empty_voxel_count: u64,
    invalid_atom_ref_count: u64,
    nonfinite_weight_count: u64,
    voxel_idx_mismatch_count: u64,
}

impl WarpReadStats {
    fn merge(&mut self, other: &Self) {
        self.empty_voxel_count += other.empty_voxel_count;
        self.invalid_atom_ref_count += other.invalid_atom_ref_count;
        self.nonfinite_weight_count += other.nonfinite_weight_count;
        self.voxel_idx_mismatch_count += other.voxel_idx_mismatch_count;
    }
}

#[derive(Debug, Clone)]
struct WarpAccumulator {
    grid_dim: usize,
    gradient_sum: Vec<f64>,
    gradient_max: Vec<f64>,
    stream_count: u64,
    read_stats: WarpReadStats,
}

impl WarpAccumulator {
    fn new(grid_dim: usize) -> Self {
        let voxels = grid_dim * grid_dim * grid_dim;
        Self {
            grid_dim,
            gradient_sum: vec![0.0; voxels],
            gradient_max: vec![0.0; voxels],
            stream_count: 0,
            read_stats: WarpReadStats::default(),
        }
    }

    fn add_gradient(&mut self, voxel: VoxelIdx, gradient: ShearStress) {
        let idx = voxel.0 as usize;
        self.gradient_sum[idx] += gradient.0;
        self.gradient_max[idx] = self.gradient_max[idx].max(gradient.0);
    }

    fn merge(&mut self, other: WarpAccumulator) -> Result<()> {
        if self.grid_dim != other.grid_dim || self.gradient_sum.len() != other.gradient_sum.len() {
            bail!("warp grid dimension mismatch during reduce");
        }
        self.stream_count += other.stream_count;
        self.read_stats.merge(&other.read_stats);
        for idx in 0..self.gradient_sum.len() {
            self.gradient_sum[idx] += other.gradient_sum[idx];
            self.gradient_max[idx] = self.gradient_max[idx].max(other.gradient_max[idx]);
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
enum VectorSourceMode {
    TopologyPositions {
        positions_by_condition: HashMap<String, Arc<Vec<Vector3>>>,
    },
    StreamBinary {
        files_by_key: BTreeMap<StreamKey, StreamPath>,
    },
}

#[derive(Debug, Clone)]
struct VectorSourceSelection {
    option_number: u8,
    source_label: &'static str,
    tensor_name: &'static str,
    metric_column: &'static str,
    reason: String,
    mode: VectorSourceMode,
    source_paths_by_condition: HashMap<String, PathBuf>,
}

impl VectorSourceSelection {
    fn vector_source_path(&self, stream: &StreamPath) -> Result<PathBuf> {
        match &self.mode {
            VectorSourceMode::TopologyPositions { .. } => self
                .source_paths_by_condition
                .get(&stream.condition_id)
                .cloned()
                .with_context(|| {
                    format!("missing topology vector source for {}", stream.condition_id)
                }),
            VectorSourceMode::StreamBinary { files_by_key } => {
                let key = stream_key(stream);
                files_by_key
                    .get(&key)
                    .map(|item| item.path.clone())
                    .with_context(|| {
                        format!("missing stream vector source for {}", stream.path.display())
                    })
            }
        }
    }

    fn vectors_for_stream(&self, stream: &StreamPath) -> Result<Arc<Vec<Vector3>>> {
        match &self.mode {
            VectorSourceMode::TopologyPositions {
                positions_by_condition,
            } => positions_by_condition
                .get(&stream.condition_id)
                .cloned()
                .with_context(|| format!("missing topology positions for {}", stream.condition_id)),
            VectorSourceMode::StreamBinary { files_by_key } => {
                let key = stream_key(stream);
                let source = files_by_key.get(&key).with_context(|| {
                    format!("missing stream vector source for {}", stream.path.display())
                })?;
                Ok(Arc::new(read_binary_vector_source(&source.path)?))
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct SignalGridShape {
    grid_dim: usize,
    voxel_count: usize,
}

fn stream_key(stream: &StreamPath) -> StreamKey {
    StreamKey {
        condition_id: stream.condition_id.clone(),
        replica_id: stream.replica_id.0,
        stream_id: stream.stream_id.0,
    }
}

fn files_by_key(files: Vec<StreamPath>) -> BTreeMap<StreamKey, StreamPath> {
    files
        .into_iter()
        .map(|item| (stream_key(&item), item))
        .collect::<BTreeMap<_, _>>()
}

fn covers_all_streams(files: &BTreeMap<StreamKey, StreamPath>, selected: &[StreamPath]) -> bool {
    selected
        .iter()
        .all(|stream| files.contains_key(&stream_key(stream)))
}

fn selected_conditions(selected: &[StreamPath]) -> HashSet<String> {
    selected
        .iter()
        .map(|stream| stream.condition_id.clone())
        .collect()
}

fn parse_frame_filter(frames: Option<&str>) -> Result<Vec<i32>> {
    let Some(raw) = frames.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(Vec::new());
    };
    raw.split(',')
        .map(|item| {
            let frame = item
                .trim()
                .parse::<i32>()
                .with_context(|| format!("parse frame filter value {item:?}"))?;
            if frame < 0 {
                bail!("frame filter values must be non-negative, got {frame}");
            }
            Ok(frame)
        })
        .collect()
}

fn filter_warps_by_frames(
    selected: Vec<StreamPath>,
    protocol_state_summary: &Path,
    frames: &[i32],
) -> Result<Vec<StreamPath>> {
    if frames.is_empty() {
        return Ok(selected);
    }
    let phase_map = load_protocol_phase_map(protocol_state_summary)?;
    let tolerance = 2_500_i32;
    let filtered = selected
        .into_iter()
        .filter(|stream| {
            phase_map
                .get(&(
                    stream.condition_id.clone(),
                    stream.replica_id.0,
                    stream.stream_id.0,
                ))
                .map(|phase| {
                    frames
                        .iter()
                        .any(|frame| (phase.current_step - *frame).abs() <= tolerance)
                })
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    if filtered.is_empty() {
        bail!(
            "frames {:?} selected no warp streams using {} with tolerance {}",
            frames,
            protocol_state_summary.display(),
            tolerance
        );
    }
    Ok(filtered)
}

fn select_vector_source(
    raw_root: &Path,
    topology_root: &Path,
    selected: &[StreamPath],
) -> Result<VectorSourceSelection> {
    let coords_final = files_by_key(
        prism_nhs::io::provenance::discover_stream_files(raw_root, "coords_final.bin")
            .unwrap_or_default(),
    );
    if covers_all_streams(&coords_final, selected) {
        return Ok(VectorSourceSelection {
            option_number: 1,
            source_label: "coords_final.bin",
            tensor_name: "ShearStress",
            metric_column: "shear_stress",
            reason: "final-timestep coordinate binaries are present for every selected stream"
                .to_string(),
            mode: VectorSourceMode::StreamBinary {
                files_by_key: coords_final,
            },
            source_paths_by_condition: HashMap::new(),
        });
    }

    let positions_bin = files_by_key(
        prism_nhs::io::provenance::discover_stream_files(raw_root, "positions.bin")
            .unwrap_or_default(),
    );
    if covers_all_streams(&positions_bin, selected) {
        return Ok(VectorSourceSelection {
            option_number: 1,
            source_label: "positions.bin",
            tensor_name: "ShearStress",
            metric_column: "shear_stress",
            reason: "stream position binaries are present for every selected stream".to_string(),
            mode: VectorSourceMode::StreamBinary {
                files_by_key: positions_bin,
            },
            source_paths_by_condition: HashMap::new(),
        });
    }

    let mut positions_by_condition = HashMap::new();
    let mut source_paths_by_condition = HashMap::new();
    let mut topology_missing = Vec::new();
    for condition in selected_conditions(selected) {
        let path = topology_root.join(format!("{condition}.topology.json"));
        match read_topology_positions(&path) {
            Ok(positions) => {
                positions_by_condition.insert(condition.clone(), Arc::new(positions));
                source_paths_by_condition.insert(condition, path);
            }
            Err(err) => topology_missing.push(format!("{condition}: {err}")),
        }
    }
    if topology_missing.is_empty() {
        return Ok(VectorSourceSelection {
            option_number: 1,
            source_label: "topology.positions",
            tensor_name: "ShearStress",
            metric_column: "shear_stress",
            reason: "no coordinate sidecar binaries were discovered; every selected condition has topology JSON positions, the highest available coordinate source".to_string(),
            mode: VectorSourceMode::TopologyPositions {
                positions_by_condition,
            },
            source_paths_by_condition,
        });
    }

    let forces = files_by_key(
        prism_nhs::io::provenance::discover_stream_files(raw_root, "forces_final.bin")
            .unwrap_or_default(),
    );
    if covers_all_streams(&forces, selected) {
        return Ok(VectorSourceSelection {
            option_number: 2,
            source_label: "forces_final.bin",
            tensor_name: "ForceConcentrationGradient",
            metric_column: "force_concentration_gradient",
            reason: format!(
                "coordinate sources were incomplete ({:?}); forces_final.bin covers every selected stream",
                topology_missing
            ),
            mode: VectorSourceMode::StreamBinary {
                files_by_key: forces,
            },
            source_paths_by_condition: HashMap::new(),
        });
    }

    let asc = files_by_key(
        prism_nhs::io::provenance::discover_stream_files(raw_root, "asc_vectors.bin")
            .unwrap_or_default(),
    );
    if covers_all_streams(&asc, selected) {
        return Ok(VectorSourceSelection {
            option_number: 3,
            source_label: "asc_vectors.bin",
            tensor_name: "ASCDirectionalShear",
            metric_column: "asc_directional_shear",
            reason: format!(
                "coordinate and force sources were incomplete ({:?}); asc_vectors.bin covers every selected stream",
                topology_missing
            ),
            mode: VectorSourceMode::StreamBinary { files_by_key: asc },
            source_paths_by_condition: HashMap::new(),
        });
    }

    bail!(
        "no vector source covers the selected streams: no coords_final.bin, no positions.bin, incomplete topology.positions ({:?}), incomplete forces_final.bin, incomplete asc_vectors.bin",
        topology_missing
    )
}

fn read_topology_positions(path: &Path) -> Result<Vec<Vector3>> {
    let file = File::open(path).with_context(|| format!("open topology {}", path.display()))?;
    let value: Value = serde_json::from_reader(file)
        .with_context(|| format!("parse topology {}", path.display()))?;
    let positions = value
        .get("positions")
        .and_then(Value::as_array)
        .with_context(|| format!("{} missing positions array", path.display()))?;
    if positions.is_empty() || positions.len() % 3 != 0 {
        bail!(
            "{} positions length {} is not a non-empty xyz vector array",
            path.display(),
            positions.len()
        );
    }
    positions
        .chunks_exact(3)
        .map(|chunk| {
            let x = chunk[0]
                .as_f64()
                .with_context(|| format!("{} contains non-float x coordinate", path.display()))?;
            let y = chunk[1]
                .as_f64()
                .with_context(|| format!("{} contains non-float y coordinate", path.display()))?;
            let z = chunk[2]
                .as_f64()
                .with_context(|| format!("{} contains non-float z coordinate", path.display()))?;
            Ok(Vector3::new(x, y, z))
        })
        .collect()
}

fn read_binary_vector_source(path: &Path) -> Result<Vec<Vector3>> {
    let values = read_f32_file(path)?;
    if values.is_empty() || values.len() % 3 != 0 {
        bail!(
            "{} vector source length {} is not a non-empty f32 xyz array",
            path.display(),
            values.len()
        );
    }
    Ok(values
        .chunks_exact(3)
        .map(|chunk| {
            Vector3::new(
                f64::from(chunk[0]),
                f64::from(chunk[1]),
                f64::from(chunk[2]),
            )
        })
        .collect())
}

fn read_signal_grid_shape(path: &Path) -> Result<SignalGridShape> {
    let mut file =
        File::open(path).with_context(|| format!("open signal grid {}", path.display()))?;
    let mut magic = [0u8; 8];
    file.read_exact(&mut magic)?;
    if &magic != b"PRSGD001" {
        bail!("{} expected PRSGD001 envelope", path.display());
    }
    let schema_version = read_u32(&mut file)?;
    if schema_version != 1 {
        bail!(
            "{} unsupported signal-grid schema {}",
            path.display(),
            schema_version
        );
    }
    let endian = read_u32(&mut file)?;
    if endian != 0x0102_0304 {
        bail!("{} unsupported endian marker {}", path.display(), endian);
    }
    let _stream_id = read_u32(&mut file)?;
    let run_len = read_u64(&mut file)? as usize;
    skip_exact(&mut file, run_len)?;
    let stem_len = read_u64(&mut file)? as usize;
    skip_exact(&mut file, stem_len)?;
    let record_count = read_u64(&mut file)?;
    let byte_stride = read_u64(&mut file)?;
    let _payload_size = read_u64(&mut file)?;
    if byte_stride != 4 {
        bail!(
            "{} expected i32 byte_stride=4, got {}",
            path.display(),
            byte_stride
        );
    }
    let grid_dim = read_u64(&mut file)?;
    let voxel_count = read_u64(&mut file)?;
    if voxel_count != record_count {
        bail!(
            "{} voxel_count {} != record_count {}",
            path.display(),
            voxel_count,
            record_count
        );
    }
    let dim = grid_dim as usize;
    if dim < 2 {
        bail!(
            "{} grid_dim {} cannot produce derivatives",
            path.display(),
            dim
        );
    }
    if dim * dim * dim != voxel_count as usize {
        bail!(
            "{} grid_dim {} does not match voxel_count {}",
            path.display(),
            grid_dim,
            voxel_count
        );
    }
    Ok(SignalGridShape {
        grid_dim: dim,
        voxel_count: voxel_count as usize,
    })
}

fn read_u32(file: &mut File) -> Result<u32> {
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32_from(record: &[u8], offset: usize) -> i32 {
    i32::from_le_bytes(record[offset..offset + 4].try_into().expect("aligned i32"))
}

fn read_f32_from(record: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(record[offset..offset + 4].try_into().expect("aligned f32"))
}

fn read_u64(file: &mut File) -> Result<u64> {
    let mut buf = [0u8; 8];
    file.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn skip_exact(file: &mut File, bytes: usize) -> Result<()> {
    file.seek(SeekFrom::Current(bytes as i64))?;
    Ok(())
}

fn pair_streams(
    warps: Vec<StreamPath>,
    signals: Vec<StreamPath>,
    selection: &VectorSourceSelection,
) -> Result<Vec<StreamInput>> {
    let signals_by_key = files_by_key(signals);
    warps
        .into_iter()
        .map(|warp| {
            let signal_grid = signals_by_key
                .get(&stream_key(&warp))
                .cloned()
                .with_context(|| format!("missing signal_grid.bin for {}", warp.path.display()))?;
            let vector_source_path = selection.vector_source_path(&warp)?;
            Ok(StreamInput {
                warp,
                signal_grid,
                vector_source_path,
            })
        })
        .collect()
}

fn parse_warp_vector(
    record: &[u8],
    expected_voxel_idx: usize,
    vectors: &[Vector3],
    stats: &mut WarpReadStats,
) -> Vector3 {
    let stored_voxel_idx = read_i32_from(record, VOXEL_INDEX_OFFSET);
    if stored_voxel_idx >= 0 && stored_voxel_idx as usize != expected_voxel_idx {
        stats.voxel_idx_mismatch_count += 1;
    }
    let n_atoms = read_i32_from(record, N_ATOMS_OFFSET).clamp(0, MAX_WARP_ATOMS as i32) as usize;
    if n_atoms == 0 {
        stats.empty_voxel_count += 1;
        return Vector3::ZERO;
    }
    let mut weighted = Vector3::ZERO;
    let mut valid_atoms = 0usize;
    for slot in 0..n_atoms {
        let atom_idx = read_i32_from(record, ATOM_INDEX_OFFSET + slot * 4);
        let weight = read_f32_from(record, ATOM_WEIGHT_OFFSET + slot * 4);
        if atom_idx < 0 {
            continue;
        }
        if !weight.is_finite() {
            stats.nonfinite_weight_count += 1;
            continue;
        }
        let Some(vector) = vectors.get(atom_idx as usize) else {
            stats.invalid_atom_ref_count += 1;
            continue;
        };
        weighted = weighted.add(vector.scale(f64::from(weight)));
        valid_atoms += 1;
    }
    if valid_atoms == 0 {
        stats.empty_voxel_count += 1;
    }
    weighted
}

fn read_voxel_vector_field(
    warp_path: &Path,
    grid_dim: usize,
    voxel_count: usize,
    vectors: &[Vector3],
) -> Result<(Vec<Vector3>, WarpReadStats)> {
    let len = warp_path.metadata()?.len() as usize;
    if len % WARP_ENTRY_BYTES != 0 {
        bail!(
            "{} is not aligned to 136-byte GpuWarpEntry records",
            warp_path.display()
        );
    }
    let warp_records = len / WARP_ENTRY_BYTES;
    if warp_records != voxel_count {
        bail!(
            "{} warp record count {} != signal grid voxel_count {}",
            warp_path.display(),
            warp_records,
            voxel_count
        );
    }
    if grid_dim * grid_dim * grid_dim != warp_records {
        bail!(
            "{} grid_dim {} does not match warp record count {}",
            warp_path.display(),
            grid_dim,
            warp_records
        );
    }

    let records_per_slice = grid_dim * grid_dim;
    let mut file =
        File::open(warp_path).with_context(|| format!("open {}", warp_path.display()))?;
    let mut field = Vec::with_capacity(warp_records);
    let mut stats = WarpReadStats::default();
    for z in 0..grid_dim {
        let mut bytes = vec![0u8; records_per_slice * WARP_ENTRY_BYTES];
        file.read_exact(&mut bytes)?;
        for (xy, record) in bytes.chunks_exact(WARP_ENTRY_BYTES).enumerate() {
            let expected = z * records_per_slice + xy;
            field.push(parse_warp_vector(record, expected, vectors, &mut stats));
        }
    }
    Ok((field, stats))
}

fn field_index(x: usize, y: usize, z: usize, dim: usize) -> usize {
    z * dim * dim + y * dim + x
}

fn derivative_x(field: &[Vector3], x: usize, y: usize, z: usize, dim: usize) -> Vector3 {
    if x == 0 {
        field[field_index(1, y, z, dim)].subtract(field[field_index(0, y, z, dim)])
    } else if x + 1 == dim {
        field[field_index(x, y, z, dim)].subtract(field[field_index(x - 1, y, z, dim)])
    } else {
        field[field_index(x + 1, y, z, dim)]
            .subtract(field[field_index(x - 1, y, z, dim)])
            .scale(0.5)
    }
}

fn derivative_y(field: &[Vector3], x: usize, y: usize, z: usize, dim: usize) -> Vector3 {
    if y == 0 {
        field[field_index(x, 1, z, dim)].subtract(field[field_index(x, 0, z, dim)])
    } else if y + 1 == dim {
        field[field_index(x, y, z, dim)].subtract(field[field_index(x, y - 1, z, dim)])
    } else {
        field[field_index(x, y + 1, z, dim)]
            .subtract(field[field_index(x, y - 1, z, dim)])
            .scale(0.5)
    }
}

fn derivative_z(field: &[Vector3], x: usize, y: usize, z: usize, dim: usize) -> Vector3 {
    if z == 0 {
        field[field_index(x, y, 1, dim)].subtract(field[field_index(x, y, 0, dim)])
    } else if z + 1 == dim {
        field[field_index(x, y, z, dim)].subtract(field[field_index(x, y, z - 1, dim)])
    } else {
        field[field_index(x, y, z + 1, dim)]
            .subtract(field[field_index(x, y, z - 1, dim)])
            .scale(0.5)
    }
}

fn frobenius_norm_jacobian(dw_dx: Vector3, dw_dy: Vector3, dw_dz: Vector3) -> Result<ShearStress> {
    ShearStress::new((dw_dx.squared_norm() + dw_dy.squared_norm() + dw_dz.squared_norm()).sqrt())
}

fn process_stream(
    input: &StreamInput,
    selection: &VectorSourceSelection,
) -> Result<WarpAccumulator> {
    let shape = read_signal_grid_shape(&input.signal_grid.path)?;
    let vectors = selection.vectors_for_stream(&input.warp)?;
    let (field, stats) = read_voxel_vector_field(
        &input.warp.path,
        shape.grid_dim,
        shape.voxel_count,
        &vectors,
    )?;
    let mut acc = WarpAccumulator::new(shape.grid_dim);
    acc.read_stats = stats;
    for z in 0..shape.grid_dim {
        for y in 0..shape.grid_dim {
            for x in 0..shape.grid_dim {
                let idx = field_index(x, y, z, shape.grid_dim);
                let dw_dx = derivative_x(&field, x, y, z, shape.grid_dim);
                let dw_dy = derivative_y(&field, x, y, z, shape.grid_dim);
                let dw_dz = derivative_z(&field, x, y, z, shape.grid_dim);
                acc.add_gradient(
                    VoxelIdx(idx as u32),
                    frobenius_norm_jacobian(dw_dx, dw_dy, dw_dz)?,
                );
            }
        }
    }
    acc.stream_count = 1;
    Ok(acc)
}

fn merge_maps(
    mut left: HashMap<String, WarpAccumulator>,
    right: HashMap<String, WarpAccumulator>,
) -> Result<HashMap<String, WarpAccumulator>> {
    for (condition, acc) in right {
        if let Some(existing) = left.get_mut(&condition) {
            existing.merge(acc)?;
        } else {
            left.insert(condition, acc);
        }
    }
    Ok(left)
}

fn output_schema(metric_column: &str) -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("campaign_id", DataType::Utf8, false),
        Field::new("condition_id", DataType::Utf8, false),
        Field::new("voxel_idx", DataType::UInt32, false),
        Field::new("x_idx", DataType::Int32, false),
        Field::new("y_idx", DataType::Int32, false),
        Field::new("z_idx", DataType::Int32, false),
        Field::new(metric_column, DataType::Float64, false),
        Field::new(format!("{metric_column}_max"), DataType::Float64, false),
        Field::new("stream_count", DataType::UInt64, false),
    ]))
}

fn condition_summaries(accumulators: &HashMap<String, WarpAccumulator>) -> Vec<Value> {
    let mut out = accumulators
        .iter()
        .map(|(condition, acc)| {
            json!({
                "condition_id": condition,
                "grid_dim": acc.grid_dim,
                "voxel_count": acc.gradient_sum.len(),
                "stream_count": acc.stream_count,
                "empty_voxel_count": acc.read_stats.empty_voxel_count,
                "invalid_atom_ref_count": acc.read_stats.invalid_atom_ref_count,
                "nonfinite_weight_count": acc.read_stats.nonfinite_weight_count,
                "voxel_idx_mismatch_count": acc.read_stats.voxel_idx_mismatch_count
            })
        })
        .collect::<Vec<_>>();
    out.sort_by_key(|item| {
        item["condition_id"]
            .as_str()
            .unwrap_or_default()
            .to_string()
    });
    out
}

fn topology_atom_counts(selection: &VectorSourceSelection) -> BTreeMap<String, usize> {
    match &selection.mode {
        VectorSourceMode::TopologyPositions {
            positions_by_condition,
        } => positions_by_condition
            .iter()
            .map(|(condition, positions)| (condition.clone(), positions.len()))
            .collect(),
        VectorSourceMode::StreamBinary { .. } => BTreeMap::new(),
    }
}

fn stream_input_paths(inputs: &[StreamInput]) -> Vec<PathBuf> {
    inputs
        .iter()
        .flat_map(|input| {
            [
                input.warp.path.clone(),
                input.signal_grid.path.clone(),
                input.vector_source_path.clone(),
            ]
        })
        .collect()
}

fn write_output_batches(
    writer: &mut ProvenanceParquetWriter,
    schema_ref: Arc<Schema>,
    metric_column: &str,
    accumulators: HashMap<String, WarpAccumulator>,
) -> Result<()> {
    let _ = metric_column;
    for (condition, acc) in accumulators {
        let dim = acc.grid_dim;
        for start in (0..acc.gradient_sum.len()).step_by(ROW_GROUP_SIZE) {
            let end = (start + ROW_GROUP_SIZE).min(acc.gradient_sum.len());
            let mut campaign_id = Vec::with_capacity(end - start);
            let mut condition_id = Vec::with_capacity(end - start);
            let mut voxel_idx = Vec::with_capacity(end - start);
            let mut x_idx = Vec::with_capacity(end - start);
            let mut y_idx = Vec::with_capacity(end - start);
            let mut z_idx = Vec::with_capacity(end - start);
            let mut gradient_mean = Vec::with_capacity(end - start);
            let mut gradient_max = Vec::with_capacity(end - start);
            let mut stream_count = Vec::with_capacity(end - start);
            for idx in start..end {
                campaign_id.push(CAMPAIGN_ID.to_string());
                condition_id.push(condition.clone());
                voxel_idx.push(idx as u32);
                x_idx.push((idx % dim) as i32);
                y_idx.push(((idx / dim) % dim) as i32);
                z_idx.push((idx / (dim * dim)) as i32);
                gradient_mean.push(if acc.stream_count == 0 {
                    0.0
                } else {
                    acc.gradient_sum[idx] / acc.stream_count as f64
                });
                gradient_max.push(acc.gradient_max[idx]);
                stream_count.push(acc.stream_count);
            }
            let batch: RecordBatch = record_batch(
                schema_ref.clone(),
                vec![
                    strings(campaign_id),
                    strings(condition_id),
                    u32s(voxel_idx),
                    i32s(x_idx),
                    i32s(y_idx),
                    i32s(z_idx),
                    f64s(gradient_mean),
                    f64s(gradient_max),
                    u64s(stream_count),
                ],
            )?;
            writer.write(&batch)?;
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let frame_filter = parse_frame_filter(args.frames.as_deref())?;
    let all_warps =
        prism_nhs::io::provenance::discover_stream_files(&args.raw_root, "warp_matrix.bin")?;
    let selected_warps = filter_streams(
        &all_warps,
        args.condition_id.as_deref(),
        args.replica_id,
        args.max_streams,
    );
    let selected_warps =
        filter_warps_by_frames(selected_warps, &args.protocol_state_summary, &frame_filter)?;
    if selected_warps.is_empty() {
        bail!("no warp_matrix.bin inputs selected");
    }
    let selection = select_vector_source(&args.raw_root, &args.topology_root, &selected_warps)?;
    let all_signals =
        prism_nhs::io::provenance::discover_stream_files(&args.raw_root, "signal_grid.bin")?;
    let selected_signals = filter_streams(
        &all_signals,
        args.condition_id.as_deref(),
        args.replica_id,
        None,
    );
    let inputs = pair_streams(selected_warps, selected_signals, &selection)?;

    let accumulators: HashMap<String, WarpAccumulator> = inputs
        .par_iter()
        .try_fold(
            HashMap::<String, WarpAccumulator>::new,
            |mut local, input| -> Result<_> {
                let acc = process_stream(input, &selection)?;
                if let Some(existing) = local.get_mut(&input.warp.condition_id) {
                    existing.merge(acc)?;
                } else {
                    local.insert(input.warp.condition_id.clone(), acc);
                }
                Ok(local)
            },
        )
        .try_reduce(HashMap::new, merge_maps)?;

    let summaries = condition_summaries(&accumulators);
    let out = args
        .out_dir
        .join(if selection.metric_column == "shear_stress" {
            "shear_stress_field.parquet"
        } else {
            "warp_gradient_field.parquet"
        });
    let schema_ref = output_schema(selection.metric_column);
    let module = if selection.metric_column == "shear_stress" {
        "phase4_shear_stress_field"
    } else {
        "phase4_warp_gradient_field"
    };
    let schema_version = if selection.metric_column == "shear_stress" {
        "prism.shear_stress_field.v1"
    } else {
        "prism.warp_gradient_field.v1"
    };
    let mut writer = ProvenanceParquetWriter::try_new(
        &out,
        schema_ref.clone(),
        ProvenanceOptions {
            module: module.to_string(),
            schema_version: schema_version.to_string(),
            producer: "crates/prism-nhs/src/bin/warp_jacobian.rs".to_string(),
            input_paths: stream_input_paths(&inputs),
            source_parquets: vec![args.protocol_state_summary],
            partition_keys: vec!["condition_id".to_string(), "voxel_idx".to_string()],
            parameters: json!({
                "vector_source_option": selection.option_number,
                "vector_source": selection.source_label,
                "vector_source_reason": selection.reason,
                "tensor_name": selection.tensor_name,
                "metric_column": selection.metric_column,
                "coordinate_lookup": {
                    "layout": "warp atom_indices[16] and atom_weights[16] multiplied against selected N_atoms x 3 vector lookup",
                    "topology_atom_counts": topology_atom_counts(&selection)
                },
                "grid_dim_source": "signal_grid.bin PRSGD001 header only; signal-grid payload is not loaded",
                "grid_dim_by_condition": summaries,
                "warp_entry_layout": "GpuWarpEntry: i32 voxel_idx, i32 atom_indices[16], f32 atom_weights[16], i32 n_atoms; 136 bytes",
                "jacobian": "3D central finite differences over x/y/z lattice strides; boundaries use one-sided differences",
                "magnitude": "Frobenius norm of the 3x3 Jacobian tensor",
                "selected_input_count": inputs.len(),
                "voxel_output_row_group_size": ROW_GROUP_SIZE,
                "max_streams": args.max_streams,
                "frames": args.frames,
                "frame_filter_selected_stream_count": inputs.len(),
                "bifurcate": args.bifurcate
            }),
        },
    )?;
    write_output_batches(
        &mut writer,
        schema_ref,
        selection.metric_column,
        accumulators,
    )?;
    writer.close()?;
    println!(
        "WROTE {} using option {} ({}) as {}",
        out.display(),
        selection.option_number,
        selection.source_label,
        selection.tensor_name
    );
    Ok(())
}
