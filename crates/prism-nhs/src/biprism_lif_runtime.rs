//! Device ownership and exact launch ABI for the versioned Voronoi observer.
//!
//! These operations are capture-compatible and do not alter the existing AMBER
//! schedule. A foliated propagation owner must establish exclusive core drift,
//! consistent units, true masses and post-kick RATTLE before installing them in
//! a production graph. A successful enqueue is not trajectory acceptance.

use crate::biprism_spike::{
    decode_native_lif_observation_queue, BiprismDecodeLimits, BiprismEventFrame,
    BiprismProducerGrid, NativeLifObservation, NATIVE_LIF_OBSERVATION_BYTES,
};
use crate::biprism_spike_arrow::{
    decode_voronoi_spike_export_in_frame, validate_leaf_stream_ids, VoronoiSpikeExport,
};
use crate::sdst_bridge::SdstGridGeometry;
use cudarc::driver::{
    sys, CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, DevicePtrMut,
    DeviceRepr, LaunchConfig, PushKernelArg,
};
use std::{any::Any, collections::HashSet, ffi::c_void, sync::Arc};

pub const FOLIATED_LEAVES: usize = 8;
pub const TILE_VERTICES: usize = 256;
pub const NATIVE_OBSERVATION_MAGIC: u32 = 0x4e4f_4232;

/// Exact native-observer device ABI. This points at native Å and native density
/// planes, independently of the V10S event producer's selected coordinate frame.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
pub struct NativeObservationLeafFfi {
    pub positions: u64,
    pub atom_types: u64,
    pub charges: u64,
    pub warp_matrix: u64,
    pub aromatic_centroids: u64,
    pub is_excited: u64,
    pub electronic_population: u64,
    pub atom_to_aromatic: u64,
    pub protocol: u64,
    pub exclusion_field: u64,
    pub water_density: u64,
    pub water_density_prev: u64,
    pub density_epoch: u64,
    pub selected_atom_ids: u64,
    pub active_tiles: u64,
    pub n_atoms: u32,
    pub n_aromatics: u32,
    pub grid_dim: u32,
    pub n_active_tiles: u32,
    pub grid_spacing: f32,
    pub origin_x: f32,
    pub origin_y: f32,
    pub origin_z: f32,
    pub native_coordinate_owner: u64,
    pub au_coordinate_owner: u64,
    pub reserved: u64,
}
// All176bytes consist of explicitly initialized integer/float fields.
unsafe impl DeviceRepr for NativeObservationLeafFfi {}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeObservationCaptureStateFfi {
    pub abi_magic: u32,
    pub cursor: u32,
    pub last_step: u32,
    pub fault: u32,
    pub bad_record: u32,
    pub captured_records: u32,
    pub capture_epochs: u32,
    pub reserved: u32,
}
const _: () = {
    assert!(core::mem::size_of::<NativeObservationLeafFfi>() == 176);
    assert!(core::mem::offset_of!(NativeObservationLeafFfi, n_atoms) == 120);
    assert!(core::mem::offset_of!(NativeObservationLeafFfi, grid_spacing) == 136);
    assert!(core::mem::offset_of!(NativeObservationLeafFfi, native_coordinate_owner) == 152);
    assert!(core::mem::offset_of!(NativeObservationLeafFfi, au_coordinate_owner) == 160);
    assert!(core::mem::size_of::<NativeObservationCaptureStateFfi>() == 32);
    assert!(core::mem::offset_of!(NativeObservationCaptureStateFfi, cursor) == 4);
    assert!(core::mem::offset_of!(NativeObservationCaptureStateFfi, fault) == 12);
};

/// Keeps the real native allocations alive. Sparse tile coordinates are copied
/// into immutable owned device storage; an empty list selects the full native grid.
pub struct NativeObservationLeafBinding {
    descriptor: NativeObservationLeafFfi,
    active_tiles: Vec<[i32; 3]>,
    context: Arc<CudaContext>,
    coordinate_generation: u64,
    _allocation_owner: Arc<dyn Any>,
}
impl NativeObservationLeafBinding {
    /// # Safety
    /// Every nonzero pointer must refer to the declared native allocation in
    /// `context` and remain stable for this owner's lifetime. Protocol points to
    /// the complete native ProtocolState, not merely its current_step member.
    /// selected_atom_ids addresses the current tile's256 packed identity words.
    /// The physical owner exclusively orders native arrays, retile and protocol
    /// changes. density_epoch/active_tiles/count are supplied by this owner and
    /// must be zero in the input descriptor.
    pub unsafe fn new(
        descriptor: NativeObservationLeafFfi,
        active_tiles: Vec<[i32; 3]>,
        context: Arc<CudaContext>,
        allocation_owner: Arc<dyn Any>,
        coordinate_generation: u64,
    ) -> Result<Self, String> {
        if coordinate_generation==0 {return Err("native observation coordinate generation is zero".into());}
        validate_native_observation_input(&descriptor, &active_tiles)?;
        Ok(Self {
            descriptor,
            active_tiles,
            context,
            coordinate_generation,
            _allocation_owner: allocation_owner,
        })
    }
}

pub fn validate_native_observation_input(
    d: &NativeObservationLeafFfi,
    active_tiles: &[[i32; 3]],
) -> Result<u32, String> {
    let required = [
        d.positions,
        d.atom_types,
        d.charges,
        d.warp_matrix,
        d.atom_to_aromatic,
        d.protocol,
        d.exclusion_field,
        d.water_density,
        d.water_density_prev,
        d.selected_atom_ids,
    ];
    if required.iter().any(|p| *p == 0 || *p % 4 != 0)
        || !(256..=0x0100_0000).contains(&d.n_atoms)
        || d.n_aromatics > i32::MAX as u32
        || d.density_epoch != 0
        || d.active_tiles != 0
        || d.n_active_tiles != 0
        || d.reserved != 0
        || [d.native_coordinate_owner,d.au_coordinate_owner].iter().any(|p|*p==0 || *p%16!=0)
        || (d.n_aromatics != 0
            && [d.aromatic_centroids, d.is_excited, d.electronic_population]
                .iter()
                .any(|p| *p == 0 || *p % 4 != 0))
    {
        return Err("invalid native observer pointers, shape or owner-initialized fields".into());
    }
    let geometry = observation_geometry(d);
    geometry.validate().map_err(|e| e.to_string())?;
    let tiles = [
        (d.grid_dim + 3) / 4,
        (d.grid_dim + 1) / 2,
        (d.grid_dim + 1) / 2,
    ];
    let dense = tiles.iter().product::<u32>();
    let mut seen = HashSet::new();
    if active_tiles.len() > dense as usize
        || active_tiles.iter().any(|tile| {
            !seen.insert(*tile)
                || (0..3).any(|axis| tile[axis] < 0 || tile[axis] as u32 >= tiles[axis])
        })
    {
        return Err("native active voxel tiles must be unique and within the4x2x2 grid".into());
    }
    Ok(if active_tiles.is_empty() {
        dense
    } else {
        active_tiles.len() as u32
    })
}
fn observation_geometry(d: &NativeObservationLeafFfi) -> SdstGridGeometry {
    SdstGridGeometry {
        dimension: d.grid_dim,
        spacing_a: d.grid_spacing,
        origin_a: [d.origin_x, d.origin_y, d.origin_z],
    }
}

pub struct NativeObservationFunctions {
    _module: Arc<CudaModule>,
    context: Arc<CudaContext>,
    density: CudaFunction,
    capture: CudaFunction,
}
impl NativeObservationFunctions {
    /// # Safety
    /// The existing governed loader must supply the actual native module and
    /// owning context, including its real WarpEntry/ProtocolState/helper ABI.
    pub unsafe fn from_governed_module(
        module: Arc<CudaModule>,
        context: Arc<CudaContext>,
    ) -> Result<Self, String> {
        context.bind_to_thread().map_err(|e| e.to_string())?;
        let density = module
            .load_function("nhs_biprism_native_density_only_v1")
            .map_err(|e| e.to_string())?;
        let capture = module
            .load_function("nhs_biprism_capture_native_observations_v1")
            .map_err(|e| e.to_string())?;
        for (f, threads) in [(&density, 128), (&capture, 256)] {
            if f.max_threads_per_block().map_err(|e| e.to_string())? < threads
                || f.occupancy_max_active_blocks_per_multiprocessor(threads as u32, 0, None)
                    .map_err(|e| e.to_string())?
                    == 0
            {
                return Err("native observation function cannot schedule its required CTA".into());
            }
        }
        Ok(Self {
            _module: module,
            context,
            density,
            capture,
        })
    }
}

struct NativeObservationDeviceState {
    functions: NativeObservationFunctions,
    _bindings: [NativeObservationLeafBinding; 8],
    descriptors_host: [NativeObservationLeafFfi; 8],
    descriptors: CudaSlice<NativeObservationLeafFfi>,
    _density_epochs: [CudaSlice<u32>; 8],
    _active_tiles: [Option<CudaSlice<i32>>; 8],
    records: CudaSlice<u8>,
    state: CudaSlice<u32>,
    blocks: [u32; 8],
    density_enqueued: bool,
    lif_enqueued: bool,
    export_state: Option<Vec<u32>>,
}

pub fn decode_native_observation_state(
    words: &[u32],
) -> Result<NativeObservationCaptureStateFfi, String> {
    if words.len() != 8 {
        return Err("native observation capture state requires32bytes".into());
    }
    Ok(NativeObservationCaptureStateFfi {
        abi_magic: words[0],
        cursor: words[1],
        last_step: words[2],
        fault: words[3],
        bad_record: words[4],
        captured_records: words[5],
        capture_epochs: words[6],
        reserved: words[7],
    })
}
pub fn validate_native_observation_state(
    s: NativeObservationCaptureStateFfi,
    count: u32,
    capacity: u32,
    step: u32,
) -> Result<(), String> {
    if step == u32::MAX
        || count > capacity
        || s.abi_magic != NATIVE_OBSERVATION_MAGIC
        || s.fault != 0
        || s.reserved != 0
        || s.cursor != count
        || s.last_step != step
        || s.bad_record != u32::MAX
        || s.captured_records < count
        || (count != 0 && s.capture_epochs == 0)
    {
        return Err(format!(
            "native observation state does not cover this retained queue epoch: {s:?}"
        ));
    }
    Ok(())
}

fn native_observation_ack_state(current: &[u32], exported: &[u32]) -> Result<Vec<u32>, String> {
    if current.len() != 8 || current != exported {
        return Err("native companion state changed after retained-export receipt".into());
    }
    let mut reset = current.to_vec();
    reset[1] = 0; // Only the ordinal cursor resets; faults and cumulative history remain.
    Ok(reset)
}

fn validate_native_observation_lif_join(
    descriptors: &[NativeObservationLeafFfi; 8],
    geometry: SdstGridGeometry,
    optional_water: u64,
    optional_exclusion: u64,
) -> Result<(), String> {
    geometry.validate().map_err(|e| e.to_string())?;
    let extent = u64::from(geometry.dimension).pow(3) * 4;
    for d in descriptors {
        bind_geometry(
            Some(BoundBiprismGeometry {
                native: observation_geometry(d),
                frame: BiprismEventFrame::NativeAngstrom,
            }),
            geometry,
            BiprismEventFrame::NativeAngstrom,
        )?;
        for target in [optional_water, optional_exclusion]
            .into_iter()
            .filter(|p| *p != 0)
        {
            let target_end = target
                .checked_add(extent)
                .ok_or("optional LIF plane address overflow")?;
            for protected in [d.exclusion_field, d.water_density, d.water_density_prev] {
                let protected_end = protected
                    .checked_add(extent)
                    .ok_or("native density plane address overflow")?;
                if target < protected_end && protected < target_end {
                    return Err("optional LIF heuristic float outputs must not alias native density/exclusion planes".into());
                }
            }
        }
    }
    Ok(())
}

/// Unfiltered companion diagnostics. A fault never becomes an empty accepted
/// export; callers can retain these exact bytes alongside the raw V10S queue.
pub struct NativeObservationDiagnostics {
    pub state: NativeObservationCaptureStateFfi,
    pub count: u32,
    pub capacity: u32,
    pub raw_records: Vec<u8>,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
pub struct BiprismLifCarryoverFfi {
    pub pending: u32,
    pub produced_step: u32,
    pub delivered_step: u32,
    pub fault: u32,
    pub produced_committed_substeps: u32,
    pub reserved: [u32; 3],
    pub atom_id: [u32; TILE_VERTICES],
    pub force: [[f32; 4]; TILE_VERTICES],
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
pub struct SignalGridSpecFfi {
    pub origin_xyz: [f32; 3],
    pub inv_voxel_spacing: f32,
    pub dim_xyz: [u32; 3],
    pub total_voxels: u32,
}

// All32bytes are explicitly initialized fields, including the by-value CUDA grid ABI.
unsafe impl DeviceRepr for SignalGridSpecFfi {}

const _: () = {
    assert!(core::mem::size_of::<BiprismLifCarryoverFfi>() == 5152);
    assert!(core::mem::offset_of!(BiprismLifCarryoverFfi, produced_committed_substeps) == 16);
    assert!(core::mem::offset_of!(BiprismLifCarryoverFfi, atom_id) == 32);
    assert!(core::mem::offset_of!(BiprismLifCarryoverFfi, force) == 1056);
    assert!(core::mem::size_of::<SignalGridSpecFfi>() == 32);
};

impl SignalGridSpecFfi {
    pub fn from_native_geometry(geometry: SdstGridGeometry) -> Result<Self, String> {
        Self::from_native_geometry_in_frame(geometry, BiprismEventFrame::NativeAngstrom)
    }
    pub fn from_native_geometry_in_frame(
        geometry: SdstGridGeometry,
        frame: BiprismEventFrame,
    ) -> Result<Self, String> {
        let producer = BiprismProducerGrid::from_native_geometry(geometry, frame)?;
        let value = Self {
            origin_xyz: producer.origin_xyz,
            inv_voxel_spacing: producer.inv_voxel_spacing,
            dim_xyz: [producer.dimension; 3],
            total_voxels: producer.total_voxels,
        };
        value.validate()?;
        Ok(value)
    }
    pub fn validate(&self) -> Result<(), String> {
        let product = self
            .dim_xyz
            .iter()
            .try_fold(1u64, |n, &d| n.checked_mul(d.into()));
        if self.origin_xyz.iter().any(|x| !x.is_finite())
            || !self.inv_voxel_spacing.is_finite()
            || self.inv_voxel_spacing <= 0.0
            || self.dim_xyz.contains(&0)
            || self.total_voxels == 0
            || self.total_voxels > i32::MAX as u32
            || product != Some(self.total_voxels.into())
        {
            return Err("invalid Voronoi signal grid extent, spacing or origin".into());
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct BoundBiprismGeometry {
    native: SdstGridGeometry,
    frame: BiprismEventFrame,
}
fn bind_geometry(
    previous: Option<BoundBiprismGeometry>,
    next: SdstGridGeometry,
    frame: BiprismEventFrame,
) -> Result<SignalGridSpecFfi, String> {
    let grid = SignalGridSpecFfi::from_native_geometry_in_frame(next, frame)?;
    if let Some(old) = previous {
        if old.frame != frame
            || old.native.dimension != next.dimension
            || old.native.spacing_a.to_bits() != next.spacing_a.to_bits()
            || old.native.origin_a.map(f32::to_bits) != next.origin_a.map(f32::to_bits)
        {
            return Err(
                "an owned V10S queue cannot change its captured producer geometry or frame".into(),
            );
        }
    }
    Ok(grid)
}

/// Decode the complete carry payload even when it contains a fault. Diagnostics
/// can retain this result independently of accepted consumer exports.
pub fn decode_carryover_words(words: &[u32]) -> Result<Vec<BiprismLifCarryoverFfi>, String> {
    const STRIDE: usize = 5152 / 4;
    if words.len() != FOLIATED_LEAVES * STRIDE {
        return Err("carryover readback size mismatch".into());
    }
    Ok(words
        .chunks_exact(STRIDE)
        .map(|w| BiprismLifCarryoverFfi {
            pending: w[0],
            produced_step: w[1],
            delivered_step: w[2],
            fault: w[3],
            produced_committed_substeps: w[4],
            reserved: [w[5], w[6], w[7]],
            atom_id: std::array::from_fn(|v| w[8 + v]),
            force: std::array::from_fn(|v| {
                std::array::from_fn(|d| f32::from_bits(w[264 + 4 * v + d]))
            }),
        })
        .collect())
}

fn validate_carryover_epoch(
    carry: &[BiprismLifCarryoverFfi],
    atom_count: u32,
) -> Result<u32, String> {
    if carry.len() != FOLIATED_LEAVES {
        return Err("all eight carryover leaves are required".into());
    }
    let first = &carry[0];
    for (leaf, c) in carry.iter().enumerate() {
        if c.fault != 0 {
            return Err(format!(
                "LIF carryover leaf {leaf} has fault 0x{:08x}",
                c.fault
            ));
        }
        if c.pending > 1
            || c.pending != first.pending
            || c.produced_step != first.produced_step
            || c.delivered_step != first.delivered_step
            || c.produced_committed_substeps != first.produced_committed_substeps
            || c.produced_committed_substeps < 4
            || c.produced_committed_substeps > u32::MAX - 4
            || c.produced_committed_substeps % 4 != 0
            || c.reserved != [0; 3]
            || c.produced_step == u32::MAX
            || (c.pending == 1 && c.delivered_step > c.produced_step)
            || (c.pending == 0 && c.delivered_step != c.produced_step + 1)
        {
            return Err(format!(
                "LIF carryover leaf {leaf} has an incomplete or mixed epoch"
            ));
        }
        for v in 0..TILE_VERTICES {
            if c.atom_id[v] >= atom_count
                || c.atom_id[..v].contains(&c.atom_id[v])
                || c.force[v].iter().any(|x| !x.is_finite())
                || c.force[v][3].to_bits() != 0
            {
                return Err(format!(
                    "LIF carryover leaf {leaf} vertex {v} has invalid identity or force"
                ));
            }
        }
    }
    Ok(first.produced_step)
}

/// A private allocation token prevents acknowledging a different owner's queue.
/// It is intentionally not Clone and cannot be constructed by consumers.
pub struct QueueExportReceipt {
    allocation: Arc<()>,
    generation: u64,
    count: u32,
    produced_step: u32,
    carry_words: Vec<u32>,
}

#[derive(Default)]
struct QueueEpoch {
    allocation: Arc<()>,
    generation: u64,
    outstanding: Option<(u32, u32)>,
    acknowledged_step: Option<u32>,
}
impl QueueEpoch {
    fn require_available(&self) -> Result<(), String> {
        if self.outstanding.is_some() {
            return Err(
                "retain and acknowledge the outstanding V10S export before queue reuse".into(),
            );
        }
        Ok(())
    }
    fn reserve(
        &mut self,
        count: u32,
        step: u32,
        carry_words: Vec<u32>,
    ) -> Result<QueueExportReceipt, String> {
        self.require_available()?;
        self.generation
            .checked_add(1)
            .ok_or("V10S queue generation exhausted")?;
        if self.acknowledged_step.is_some_and(|old| step <= old) {
            return Err("V10S producer epoch did not advance after acknowledgement".into());
        }
        self.outstanding = Some((count, step));
        Ok(QueueExportReceipt {
            allocation: self.allocation.clone(),
            generation: self.generation,
            count,
            produced_step: step,
            carry_words,
        })
    }
    fn validate_ack(
        &self,
        receipt: &QueueExportReceipt,
        count: u32,
        carry: &[u32],
    ) -> Result<(), String> {
        if !Arc::ptr_eq(&self.allocation, &receipt.allocation)
            || receipt.generation != self.generation
            || self.outstanding != Some((receipt.count, receipt.produced_step))
            || count != receipt.count
            || carry != receipt.carry_words
        {
            return Err("V10S acknowledgement is stale, belongs to another allocation, or producer state changed".into());
        }
        Ok(())
    }
    fn complete_ack(&mut self, receipt: &QueueExportReceipt) {
        self.generation += 1; // checked before receipt publication
        self.outstanding = None;
        self.acknowledged_step = Some(receipt.produced_step);
    }
}

fn completed_dtoh<T: DeviceRepr + Default + Clone, Src: DevicePtr<T>>(
    stream: &Arc<CudaStream>, source: &Src,
) -> Result<Vec<T>, String> {
    let mut result=vec![T::default();source.len()];
    let copy=stream.memcpy_dtoh(source,&mut result).map_err(|e|e.to_string());
    let drain=stream.synchronize().map_err(|e|e.to_string());
    if let Err(e)=copy{return Err(format!("device readback failed: {e}; drain={drain:?}"));}
    drain?;Ok(result)
}
fn completed_htod<T:DeviceRepr>(stream:&Arc<CudaStream>,source:&[T])->Result<CudaSlice<T>,String>{
    let mut result=unsafe{stream.alloc(source.len())}.map_err(|e|e.to_string())?;
    let copy=stream.memcpy_htod(source,&mut result).map_err(|e|e.to_string());
    let drain=stream.synchronize().map_err(|e|e.to_string());
    if let Err(e)=copy{return Err(format!("device upload failed: {e}; drain={drain:?}"));}
    drain?;Ok(result)
}

/// External planes remain owned by the physical engine. All integer grid
/// addresses refer to i32 allocations, matching the real serializer; primary
/// IDs start at -1. Observer float planes are separate and may be null.
#[derive(Clone, Copy)]
pub struct VoronoiLifPlanes {
    pub streams: sys::CUdeviceptr,
    pub tile: sys::CUdeviceptr,
    pub convergence: sys::CUdeviceptr,
    pub protocol_step: sys::CUdeviceptr,
    pub atom_residues: sys::CUdeviceptr,
    pub voxel_hits: sys::CUdeviceptr,
    pub coupled_spikes: sys::CUdeviceptr,
    pub primary_residue: sys::CUdeviceptr,
    pub primary_count: sys::CUdeviceptr,
    pub water_density: sys::CUdeviceptr,
    pub exclusion: sys::CUdeviceptr,
}

/// Read-only bindings for the collective fault/latch owner. These addresses
/// remain valid only while the owning BiprismLifDeviceState is retained.
#[derive(Clone, Copy, Debug)]
pub struct BiprismLifControlPlanes {
    pub carry: u64,
    pub queue: u64,
    pub count: u64,
    pub capacity: u32,
    pub native_observation_state: Option<u64>,
}

enum LifFunction<'a> {
    Raw(sys::CUfunction),
    Retained(&'a CudaFunction),
}

/// Owns every persistent observer allocation. Keep this object and all external
/// planes alive until the captured graph has been destroyed and synchronized.
/// No method clears pending carryover or a fault as a recovery shortcut.
pub struct BiprismLifDeviceState {
    stream: Arc<CudaStream>,
    voltage: CudaSlice<f32>,
    carry: CudaSlice<u32>,
    queue: CudaSlice<u8>,
    count: CudaSlice<u32>,
    capacity: u32,
    geometry: Option<BoundBiprismGeometry>,
    native_observation: Option<NativeObservationDeviceState>,
    leaf_stream_ids: [u8; FOLIATED_LEAVES],
    export_identity: String,
    epoch: QueueEpoch,
}

impl BiprismLifDeviceState {
    /// export_identity must identify this allocation uniquely within the run and
    /// across restarts; this in-memory owner does not claim durable crash replay.
    pub fn allocate(
        stream: Arc<CudaStream>,
        capacity: u32,
        leaf_stream_ids: [u8; 8],
        export_identity: String,
    ) -> Result<Self, String> {
        validate_leaf_stream_ids(&leaf_stream_ids)?;
        if export_identity.trim().is_empty() {
            return Err("V10S queue needs a run-unique export identity".into());
        }
        if capacity == 0 || capacity > i32::MAX as u32 {
            return Err("Voronoi queue capacity must fit the native positive signed domain".into());
        }
        let queue_bytes = (capacity as usize)
            .checked_mul(96)
            .ok_or("queue allocation overflow")?;
        Ok(Self {
            voltage: stream.alloc_zeros(8 * 256).map_err(|e| e.to_string())?,
            carry: stream
                .alloc_zeros(8 * 5152 / 4)
                .map_err(|e| e.to_string())?,
            queue: stream.alloc_zeros(queue_bytes).map_err(|e| e.to_string())?,
            count: stream.alloc_zeros(1).map_err(|e| e.to_string())?,
            stream,
            capacity,
            geometry: None,
            native_observation: None,
            leaf_stream_ids,
            export_identity,
            epoch: QueueEpoch::default(),
        })
    }

    /// Allocate the real native companion path before the first observer capture.
    /// The provided bindings retain native storage; per-voxel epochs, sparse tile
    /// copies, descriptor table, record queue and cursor/fault state are owned here.
    pub fn bind_native_observation(
        &mut self,
        functions: NativeObservationFunctions,
        bindings: [NativeObservationLeafBinding; 8],
    ) -> Result<(), String> {
        self.epoch.require_available()?;
        let mut capture = sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        let result = unsafe { sys::cuStreamIsCapturing(self.stream.cu_stream(), &mut capture) };
        if result != sys::CUresult::CUDA_SUCCESS
            || capture != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
        {
            return Err("native companion binding requires a valid uncaptured stream".into());
        }
        if self.native_observation.is_some() || self.geometry.is_some() {
            return Err(
                "native companion must be bound once before the first event capture".into(),
            );
        }
        if functions.context.cu_ctx() != self.stream.context().cu_ctx() {
            return Err("native companion module and observer contexts differ".into());
        }
        let geometry = observation_geometry(&bindings[0].descriptor);
        let mut blocks = [0; 8];
        let mut mutable_regions = Vec::new();
        let extent =
            geometry.dimension as u64 * geometry.dimension as u64 * geometry.dimension as u64 * 4;
        for (leaf, binding) in bindings.iter().enumerate() {
            if binding.context.cu_ctx() != self.stream.context().cu_ctx() {
                return Err("native companion leaf context differs from the observer".into());
            }
            blocks[leaf] =
                validate_native_observation_input(&binding.descriptor, &binding.active_tiles)?;
            bind_geometry(
                Some(BoundBiprismGeometry {
                    native: geometry,
                    frame: BiprismEventFrame::NativeAngstrom,
                }),
                observation_geometry(&binding.descriptor),
                BiprismEventFrame::NativeAngstrom,
            )?;
            for pointer in [
                binding.descriptor.exclusion_field,
                binding.descriptor.water_density,
                binding.descriptor.water_density_prev,
            ] {
                let end = pointer
                    .checked_add(extent)
                    .ok_or("native density plane address overflow")?;
                if mutable_regions
                    .iter()
                    .any(|&(start, prior_end)| pointer < prior_end && start < end)
                {
                    return Err(
                        "native density/exclusion planes alias across the observer population"
                            .into(),
                    );
                }
                mutable_regions.push((pointer, end));
            }
        }
        let voxels = geometry.dimension.pow(3) as usize;
        let epoch_host = vec![u32::MAX; voxels];
        let density_epochs: [CudaSlice<u32>; 8] = (0..8)
            .map(|_| completed_htod(&self.stream, &epoch_host))
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| "native density epoch leaf count")?;
        let active_host: [Vec<i32>; 8] = std::array::from_fn(|leaf| {
            bindings[leaf]
                .active_tiles
                .iter()
                .flatten()
                .copied()
                .collect()
        });
        let active_tiles: [Option<CudaSlice<i32>>; 8] = active_host
            .iter()
            .map(|v| {
                if v.is_empty() {
                    Ok(None)
                } else {
                    completed_htod(&self.stream, v).map(Some)
                }
            })
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| "native active tile leaf count")?;
        let descriptors_host = std::array::from_fn(|leaf| {
            let mut d = bindings[leaf].descriptor;
            d.density_epoch = density_epochs[leaf].device_ptr(&self.stream).0;
            d.active_tiles = active_tiles[leaf]
                .as_ref()
                .map_or(0, |a| a.device_ptr(&self.stream).0);
            d.n_active_tiles = bindings[leaf].active_tiles.len() as u32;
            d
        });
        let descriptors = completed_htod(&self.stream, &descriptors_host)?;
        let records = self
            .stream
            .alloc_zeros(self.capacity as usize * NATIVE_LIF_OBSERVATION_BYTES)
            .map_err(|e| e.to_string())?;
        let state_host = [NATIVE_OBSERVATION_MAGIC, 0, 0, 0, u32::MAX, 0, 0, 0];
        let state = completed_htod(&self.stream, &state_host)?;
        // Retain every pageable upload source through completion.
        self.stream.synchronize().map_err(|e| e.to_string())?;
        self.native_observation = Some(NativeObservationDeviceState {
            functions,
            _bindings: bindings,
            descriptors_host,
            descriptors,
            _density_epochs: density_epochs,
            _active_tiles: active_tiles,
            records,
            state,
            blocks,
            density_enqueued: false,
            lif_enqueued: false,
            export_state: None,
        });
        Ok(())
    }

    /// # Safety
    /// Native coordinates/aromatics/protocol/warp arrays must be current and
    /// exclusively ordered on this stream. Run once per due observation epoch,
    /// before LIF and before any Director or retile writer can advance the epoch.
    pub unsafe fn capture_native_density(&mut self) -> Result<(), String> {
        self.epoch.require_available()?;
        let native = self
            .native_observation
            .as_mut()
            .ok_or("native companion is not bound")?;
        if native.density_enqueued || native.lif_enqueued {
            return Err(
                "previous native density/LIF interval has not captured its companion".into(),
            );
        }
        for leaf in 0u32..8 {
            self.stream
                .launch_builder(&native.functions.density)
                .arg(&native.descriptors)
                .arg(&leaf)
                .arg(&mut native.state)
                .launch(LaunchConfig {
                    grid_dim: (native.blocks[leaf as usize], 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| format!("native density leaf{leaf}: {e}"))?;
        }
        native.density_enqueued = true;
        Ok(())
    }

    /// Capture same-emission native fields for every newly produced raw96 row.
    /// # Safety
    /// Enqueue immediately after the matching LIF producer and all eight native
    /// density launches, with protocol, density, grid and queue inputs immutable.
    pub unsafe fn capture_native_observations(&mut self) -> Result<(), String> {
        self.epoch.require_available()?;
        if self.geometry.is_none() {
            return Err("LIF must be captured before its native companion".into());
        }
        let native = self
            .native_observation
            .as_mut()
            .ok_or("native companion is not bound")?;
        if !native.density_enqueued || !native.lif_enqueued {
            return Err("current native density and LIF must precede companion capture".into());
        }
        self.stream
            .launch_builder(&native.functions.capture)
            .arg(&self.queue)
            .arg(&self.count)
            .arg(&native.descriptors)
            .arg(&mut native.records)
            .arg(&self.capacity)
            .arg(&mut native.state)
            .launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
            .map_err(|e| format!("same-emission native observation capture: {e}"))?;
        native.density_enqueued = false;
        native.lif_enqueued = false;
        Ok(())
    }

    /// Device fault/state address for the actual physical conditional owner.
    /// It must be observed by that owner; this getter installs no failure policy.
    pub fn native_observation_state_pointer(&self) -> Option<u64> {
        self.native_observation
            .as_ref()
            .map(|n| n.state.device_ptr(&self.stream).0)
    }

    pub fn control_planes(&self) -> BiprismLifControlPlanes {
        let carry = {
            let (p, _guard) = self.carry.device_ptr(&self.stream);
            p
        };
        let queue = {
            let (p, _guard) = self.queue.device_ptr(&self.stream);
            p
        };
        let count = {
            let (p, _guard) = self.count.device_ptr(&self.stream);
            p
        };
        let native_observation_state = self.native_observation_state_pointer();
        BiprismLifControlPlanes {
            carry,
            queue,
            count,
            capacity: self.capacity,
            native_observation_state,
        }
    }

    /// # Safety
    /// External planes and the function must match the actual 21-argument
    /// CUDA signature, remain exclusive and outlive graph replay. Launch only
    /// after successful physical propagation. Immediate writes are diagnostic.
    pub unsafe fn capture_observation(
        &mut self,
        function: sys::CUfunction,
        planes: VoronoiLifPlanes,
        geometry: SdstGridGeometry,
        decay: f32,
        threshold: f32,
        kick: f32,
        immediate_diagnostic: bool,
    ) -> Result<(), String> {
        self.capture_observation_in_frame(
            function,
            planes,
            geometry,
            BiprismEventFrame::NativeAngstrom,
            decay,
            threshold,
            kick,
            immediate_diagnostic,
        )
    }

    /// Same exact CUDA ABI, with an explicit frame bound to this queue for its
    /// entire allocation lifetime. Native geometry remains the consumer datum.
    /// # Safety
    /// All event-producing coordinates, the force model and the native unit
    /// bridge must obey `frame`. The physical owner retains all external planes
    /// and enforces post-step ordering and exclusivity through graph replay.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn capture_observation_in_frame(
        &mut self,
        function: sys::CUfunction,
        planes: VoronoiLifPlanes,
        geometry: SdstGridGeometry,
        frame: BiprismEventFrame,
        decay: f32,
        threshold: f32,
        kick: f32,
        immediate_diagnostic: bool,
    ) -> Result<(), String> {
        self.capture_observation_impl(
            LifFunction::Raw(function),
            planes,
            geometry,
            frame,
            decay,
            threshold,
            kick,
            immediate_diagnostic,
        )
    }

    /// Retained module/function variant of the exact21-argument producer.
    /// # Safety
    /// The governed loader must bind this function to the owner's CUDA context.
    /// All other physical ownership obligations match capture_observation_in_frame.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn capture_observation_function_in_frame(
        &mut self,
        function: &CudaFunction,
        planes: VoronoiLifPlanes,
        geometry: SdstGridGeometry,
        frame: BiprismEventFrame,
        decay: f32,
        threshold: f32,
        kick: f32,
        immediate_diagnostic: bool,
    ) -> Result<(), String> {
        self.capture_observation_impl(
            LifFunction::Retained(function),
            planes,
            geometry,
            frame,
            decay,
            threshold,
            kick,
            immediate_diagnostic,
        )
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn capture_observation_impl(
        &mut self,
        function: LifFunction<'_>,
        planes: VoronoiLifPlanes,
        geometry: SdstGridGeometry,
        frame: BiprismEventFrame,
        decay: f32,
        threshold: f32,
        kick: f32,
        immediate_diagnostic: bool,
    ) -> Result<(), String> {
        self.epoch.require_available()?;
        let grid = bind_geometry(self.geometry, geometry, frame)?;
        if let Some(native) = &self.native_observation {
            if !native.density_enqueued || native.lif_enqueued {
                return Err("native density must precede exactly one matching LIF capture".into());
            }
            validate_native_observation_lif_join(
                &native.descriptors_host,
                geometry,
                planes.water_density,
                planes.exclusion,
            )?;
        }
        let (voltage, _v) = self.voltage.device_ptr_mut(&self.stream);
        let (carry, _c) = self.carry.device_ptr_mut(&self.stream);
        let (queue, _q) = self.queue.device_ptr_mut(&self.stream);
        let (count, _n) = self.count.device_ptr_mut(&self.stream);
        validate_lif_launch_arguments(
            planes,
            voltage,
            carry,
            queue,
            count,
            grid,
            self.capacity,
            decay,
            threshold,
            kick,
            u32::from(immediate_diagnostic),
        )?;
        match function {
            LifFunction::Raw(function) => {
                launch_voronoi_dual_lif_and_4grid_bypass_v2_on_stream(
                    self.stream.cu_stream(),
                    function,
                    planes,
                    voltage,
                    carry,
                    queue,
                    count,
                    grid,
                    self.capacity,
                    decay,
                    threshold,
                    kick,
                    u32::from(immediate_diagnostic),
                )?;
            }
            LifFunction::Retained(function) => {
                let immediate = u32::from(immediate_diagnostic);
                self.stream
                    .launch_builder(function)
                    .arg(&planes.streams)
                    .arg(&planes.tile)
                    .arg(&planes.convergence)
                    .arg(&planes.protocol_step)
                    .arg(&planes.atom_residues)
                    .arg(&voltage)
                    .arg(&planes.voxel_hits)
                    .arg(&planes.coupled_spikes)
                    .arg(&planes.primary_residue)
                    .arg(&planes.primary_count)
                    .arg(&planes.water_density)
                    .arg(&planes.exclusion)
                    .arg(&queue)
                    .arg(&count)
                    .arg(&grid)
                    .arg(&self.capacity)
                    .arg(&decay)
                    .arg(&threshold)
                    .arg(&kick)
                    .arg(&carry)
                    .arg(&immediate)
                    .launch(LaunchConfig {
                        grid_dim: (8, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .map_err(|e| format!("retained Voronoi LIF launch: {e}"))?;
            }
        }
        self.geometry = Some(BoundBiprismGeometry {
            native: geometry,
            frame,
        });
        if let Some(native) = &mut self.native_observation {
            native.lif_enqueued = true;
        }
        Ok(())
    }

    /// # Safety
    /// Schedule after the next successful v10 commit and before RATTLE. The
    /// physical owner must supply disjoint eight-leaf velocities, physical
    /// inverse masses and mutually consistent force/mass/time units. Never pass
    /// the AMBER step in ps as an atomic-unit step without its physical bridge.
    pub unsafe fn capture_delivery(
        &mut self,
        function: sys::CUfunction,
        streams: sys::CUdeviceptr,
        convergence: sys::CUdeviceptr,
        protocol_step: sys::CUdeviceptr,
        atom_count: u32,
        dt_feedback_au: f32,
    ) -> Result<(), String> {
        self.capture_delivery_impl(
            LifFunction::Raw(function),
            streams,
            convergence,
            protocol_step,
            atom_count,
            dt_feedback_au,
        )
    }

    /// Retained module/function variant of the exact six-argument delivery.
    /// # Safety
    /// Same post-v10/pre-RATTLE physical contract as capture_delivery; the
    /// governed loader must bind this function to the owner's CUDA context.
    pub unsafe fn capture_delivery_function(
        &mut self,
        function: &CudaFunction,
        streams: sys::CUdeviceptr,
        convergence: sys::CUdeviceptr,
        protocol_step: sys::CUdeviceptr,
        atom_count: u32,
        dt_feedback_au: f32,
    ) -> Result<(), String> {
        self.capture_delivery_impl(
            LifFunction::Retained(function),
            streams,
            convergence,
            protocol_step,
            atom_count,
            dt_feedback_au,
        )
    }

    unsafe fn capture_delivery_impl(
        &mut self,
        function: LifFunction<'_>,
        streams: sys::CUdeviceptr,
        convergence: sys::CUdeviceptr,
        protocol_step: sys::CUdeviceptr,
        atom_count: u32,
        dt_feedback_au: f32,
    ) -> Result<(), String> {
        self.epoch.require_available()?;
        let (carry, _c) = self.carry.device_ptr_mut(&self.stream);
        validate_delivery_launch_arguments(
            streams,
            carry,
            convergence,
            protocol_step,
            atom_count,
            dt_feedback_au,
        )?;
        match function {
            LifFunction::Raw(function) => launch_deliver_persisted_lif_forces_on_stream(
                self.stream.cu_stream(),
                function,
                streams,
                carry,
                convergence,
                protocol_step,
                atom_count,
                dt_feedback_au,
            ),
            LifFunction::Retained(function) => self
                .stream
                .launch_builder(function)
                .arg(&streams)
                .arg(&carry)
                .arg(&convergence)
                .arg(&protocol_step)
                .arg(&atom_count)
                .arg(&dt_feedback_au)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map(|_| ())
                .map_err(|e| format!("retained LIF velocity delivery launch: {e}")),
        }
    }

    /// Read diagnostic carryover state, including every fault, without clearing it.
    /// # Safety
    /// All graph producers using these allocations must be quiescent until return.
    pub unsafe fn readback_carryover(&self) -> Result<Vec<BiprismLifCarryoverFfi>, String> {
        decode_carryover_words(&completed_dtoh(&self.stream, &self.carry)?)
    }

    /// Raw owned queue, including an overflow count. No bytes or fault are hidden.
    /// # Safety
    /// All graph producers must remain quiescent throughout diagnosis/readback.
    pub unsafe fn readback_raw_queue(&self) -> Result<(u32, Vec<u8>), String> {
        let count = completed_dtoh(&self.stream, &self.count)?[0];
        Ok((count, completed_dtoh(&self.stream, &self.queue)?))
    }

    /// # Safety
    /// All producers and native density writers must remain quiescent.
    pub unsafe fn readback_native_observation_diagnostics(
        &self,
    ) -> Result<NativeObservationDiagnostics, String> {
        let native = self
            .native_observation
            .as_ref()
            .ok_or("native companion is not bound")?;
        Ok(NativeObservationDiagnostics {
            state: decode_native_observation_state(&completed_dtoh(&self.stream, &native.state)?)?,
            count: completed_dtoh(&self.stream, &self.count)?[0],
            capacity: self.capacity,
            raw_records: completed_dtoh(&self.stream, &native.records)?,
        })
    }

    /// Read the actual same-emission device companion, then the real paired
    /// native V5/ledger export. Caller-supplied observer quantities cannot replace
    /// the captured water, phase or aromatic population on this route.
    /// # Safety
    /// Quiesce graph replay, protocol, density and retile writers until the
    /// retained export is acknowledged. Classifications bind each emitted row.
    pub unsafe fn readback_with_captured_native_observations(
        &mut self,
        limits: BiprismDecodeLimits,
        classifications: &[crate::spike_arrow_writer::SpikeClassification],
        replica_seed: u64,
    ) -> Result<(VoronoiSpikeExport, QueueExportReceipt), String> {
        if self.native_observation.is_none() {
            return Err("native companion is not bound".into());
        }
        self.readback_export_impl(limits, None, classifications, replica_seed)
    }

    /// Reads a validated queue prefix and publishes one acknowledgement receipt.
    /// No queue or physical state is reset by readback.
    /// # Safety
    /// Quiesce all graph replays and retain exclusive ownership until both output
    /// artifacts have been retained and acknowledged. Native observer contexts
    /// and classifications must be measured from this queue's emission epochs.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn readback_for_arrow_and_pathb(
        &mut self,
        limits: BiprismDecodeLimits,
        observations: &[NativeLifObservation],
        classifications: &[crate::spike_arrow_writer::SpikeClassification],
        replica_seed: u64,
    ) -> Result<(VoronoiSpikeExport, QueueExportReceipt), String> {
        if self.native_observation.is_some() {
            return Err(
                "bound native companion requires the captured-device readback route".into(),
            );
        }
        self.readback_export_impl(limits, Some(observations), classifications, replica_seed)
    }

    unsafe fn readback_export_impl(
        &mut self,
        limits: BiprismDecodeLimits,
        observations: Option<&[NativeLifObservation]>,
        classifications: &[crate::spike_arrow_writer::SpikeClassification],
        replica_seed: u64,
    ) -> Result<(VoronoiSpikeExport, QueueExportReceipt), String> {
        self.epoch.require_available()?;
        let geometry = self
            .geometry
            .ok_or("V10S producer geometry has not been captured")?;
        let carry_words = completed_dtoh(&self.stream, &self.carry)?;
        let carry = decode_carryover_words(&carry_words)?;
        let produced_step = validate_carryover_epoch(&carry, limits.num_atoms)?;
        let count = completed_dtoh(&self.stream, &self.count)?[0];
        if count > self.capacity {
            return Err("owned V10S queue overflow".into());
        }
        let mut observed_state = None;
        let owned_observations;
        let observations = if let Some(given) = observations {
            given
        } else {
            let native = self
                .native_observation
                .as_ref()
                .ok_or("native companion is not bound")?;
            if native.density_enqueued || native.lif_enqueued {
                return Err("native companion interval is incomplete".into());
            }
            let words = completed_dtoh(&self.stream, &native.state)?;
            validate_native_observation_state(
                decode_native_observation_state(&words)?,
                count,
                self.capacity,
                produced_step,
            )?;
            owned_observations = decode_native_lif_observation_queue(
                &completed_dtoh(&self.stream, &native.records)?,
                count,
                self.capacity,
                limits,
            )?;
            for row in &owned_observations {
                if row.coordinate_generation!=native._bindings[row.leaf_id as usize].coordinate_generation {
                    return Err("NOB2 row belongs to a different coordinate allocation generation".into());
                }
            }
            observed_state = Some(words);
            &owned_observations
        };
        let raw = completed_dtoh(&self.stream, &self.queue)?;
        let mut export = decode_voronoi_spike_export_in_frame(
            &raw,
            count,
            self.capacity,
            limits,
            observations,
            geometry.native,
            classifications,
            replica_seed,
            &self.leaf_stream_ids,
            geometry.frame,
        )?;
        for row in &export.decoded.ledger {
            let ev = &row.event;
            if ev.step_tau > produced_step
                || self
                    .epoch
                    .acknowledged_step
                    .is_some_and(|old| ev.step_tau <= old)
            {
                return Err(
                    "V10S event clock lies outside the unacknowledged producer epoch".into(),
                );
            }
            if ev.step_tau == produced_step {
                let c = &carry[usize::from(ev.leaf_id)];
                let v = usize::from(ev.vertex_id);
                if ev.atom_or_site_id != c.atom_id[v]
                    || ev.coupled_force_xyz.map(f32::to_bits)
                        != [c.force[v][0], c.force[v][1], c.force[v][2]].map(f32::to_bits)
                {
                    return Err("V10S latest event differs from its carryover endpoint".into());
                }
            }
        }
        export.bind_queue_export(&self.export_identity, self.epoch.generation)?;
        let receipt = self.epoch.reserve(count, produced_step, carry_words)?;
        if let Some(native) = &mut self.native_observation {
            native.export_state = observed_state;
        }
        Ok((export, receipt))
    }

    /// Clear the acknowledged queue count and matching native companion cursor;
    /// membrane, carry, faults and cumulative companion counters are retained.
    /// Failed validation leaves the prefix available for diagnosis. A CUDA
    /// failure keeps the receipt outstanding, preventing further producer capture.
    /// # Safety
    /// Both Arrow artifacts and the native consumer result must already be
    /// durably retained. All graph replays must remain quiescent and exclusive.
    /// This is an in-process acknowledgement, not a durable restart protocol.
    pub unsafe fn acknowledge_retained_export(
        &mut self,
        receipt: &QueueExportReceipt,
    ) -> Result<(), String> {
        let count = completed_dtoh(&self.stream, &self.count)?[0];
        let carry = completed_dtoh(&self.stream, &self.carry)?;
        self.epoch.validate_ack(receipt, count, &carry)?;
        let reset_state = if let Some(native) = &self.native_observation {
            let current = completed_dtoh(&self.stream, &native.state)?;
            Some(native_observation_ack_state(
                &current,
                native
                    .export_state
                    .as_deref()
                    .ok_or("native companion export snapshot is absent")?,
            )?)
        } else {
            None
        };
        if let Some(reset) = &reset_state {
            self.stream
                .memcpy_htod(reset, &mut self.native_observation.as_mut().unwrap().state)
                .map_err(|e| e.to_string())?;
            // Keep the pageable reset buffer alive even if a later count reset fails.
            self.stream.synchronize().map_err(|e| e.to_string())?;
        }
        self.stream
            .memset_zeros(&mut self.count)
            .map_err(|e| e.to_string())?;
        self.stream.synchronize().map_err(|e| e.to_string())?;
        self.epoch.complete_ack(receipt);
        if let Some(native) = &mut self.native_observation {
            native.export_state = None;
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_lif_launch_arguments(
    p: VoronoiLifPlanes,
    voltage: u64,
    carry: u64,
    queue: u64,
    count: u64,
    grid: SignalGridSpecFfi,
    capacity: u32,
    decay: f32,
    threshold: f32,
    kick: f32,
    immediate: u32,
) -> Result<(), String> {
    grid.validate()?;
    if [
        p.streams,
        p.tile,
        p.convergence,
        p.protocol_step,
        p.atom_residues,
        p.voxel_hits,
        p.coupled_spikes,
        p.primary_residue,
        p.primary_count,
        voltage,
        carry,
        queue,
        count,
    ]
    .contains(&0)
        || capacity == 0
        || immediate > 1
        || !decay.is_finite()
        || !threshold.is_finite()
        || threshold <= 0.0
        || !kick.is_finite()
        || kick < 0.0
    {
        return Err("invalid required Voronoi launch argument".into());
    }
    Ok(())
}

fn validate_delivery_launch_arguments(
    streams: u64,
    carry: u64,
    convergence: u64,
    protocol_step: u64,
    atom_count: u32,
    dt_feedback_au: f32,
) -> Result<(), String> {
    if [streams, carry, convergence, protocol_step].contains(&0)
        || atom_count < 256
        || !dt_feedback_au.is_finite()
        || dt_feedback_au <= 0.0
    {
        return Err("invalid required LIF delivery argument".into());
    }
    Ok(())
}

/// Exact 21-argument CUDA ABI. The carry pointer and immediate flag are the
/// final two arguments, preserving the preceding nineteen argument positions.
/// # Safety
/// See BiprismLifDeviceState::capture_observation; raw pointers are caller-owned.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_voronoi_dual_lif_and_4grid_bypass_v2_on_stream(
    stream: sys::CUstream,
    function: sys::CUfunction,
    mut p: VoronoiLifPlanes,
    mut voltage: sys::CUdeviceptr,
    mut carry: sys::CUdeviceptr,
    mut queue: sys::CUdeviceptr,
    mut count: sys::CUdeviceptr,
    mut grid: SignalGridSpecFfi,
    mut capacity: u32,
    mut decay: f32,
    mut threshold: f32,
    mut kick: f32,
    mut immediate: u32,
) -> Result<(), String> {
    if function.is_null() {
        return Err("null Voronoi LIF function".into());
    }
    validate_lif_launch_arguments(
        p, voltage, carry, queue, count, grid, capacity, decay, threshold, kick, immediate,
    )?;
    macro_rules! arg {
        ($v:expr) => {
            core::ptr::addr_of_mut!($v).cast::<c_void>()
        };
    }
    let mut args = [
        arg!(p.streams),
        arg!(p.tile),
        arg!(p.convergence),
        arg!(p.protocol_step),
        arg!(p.atom_residues),
        arg!(voltage),
        arg!(p.voxel_hits),
        arg!(p.coupled_spikes),
        arg!(p.primary_residue),
        arg!(p.primary_count),
        arg!(p.water_density),
        arg!(p.exclusion),
        arg!(queue),
        arg!(count),
        arg!(grid),
        arg!(capacity),
        arg!(decay),
        arg!(threshold),
        arg!(kick),
        arg!(carry),
        arg!(immediate),
    ];
    let result = sys::cuLaunchKernel(
        function,
        8,
        1,
        1,
        256,
        1,
        1,
        0,
        stream,
        args.as_mut_ptr(),
        core::ptr::null_mut(),
    );
    if result != sys::CUresult::CUDA_SUCCESS {
        return Err(format!("Voronoi LIF launch: {result:?}"));
    }
    Ok(())
}

/// Exact six-argument, one-CTA delivery ABI; all eight leaves are prevalidated
/// before the kernel changes any velocity or clears a pending descriptor.
/// # Safety
/// See BiprismLifDeviceState::capture_delivery, including physical unit ownership.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_deliver_persisted_lif_forces_on_stream(
    stream: sys::CUstream,
    function: sys::CUfunction,
    mut streams: sys::CUdeviceptr,
    mut carry: sys::CUdeviceptr,
    mut convergence: sys::CUdeviceptr,
    mut protocol_step: sys::CUdeviceptr,
    mut atom_count: u32,
    mut dt_feedback_au: f32,
) -> Result<(), String> {
    if function.is_null() {
        return Err("null LIF delivery function".into());
    }
    validate_delivery_launch_arguments(
        streams,
        carry,
        convergence,
        protocol_step,
        atom_count,
        dt_feedback_au,
    )?;
    macro_rules! arg {
        ($v:expr) => {
            core::ptr::addr_of_mut!($v).cast::<c_void>()
        };
    }
    let mut args = [
        arg!(streams),
        arg!(carry),
        arg!(convergence),
        arg!(protocol_step),
        arg!(atom_count),
        arg!(dt_feedback_au),
    ];
    let result = sys::cuLaunchKernel(
        function,
        1,
        1,
        1,
        256,
        1,
        1,
        0,
        stream,
        args.as_mut_ptr(),
        core::ptr::null_mut(),
    );
    if result != sys::CUresult::CUDA_SUCCESS {
        return Err(format!("LIF velocity delivery launch: {result:?}"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn native_descriptor_fixture() -> NativeObservationLeafFfi {
        NativeObservationLeafFfi {
            positions: 0x1000,
            atom_types: 0x2000,
            charges: 0x3000,
            warp_matrix: 0x4000,
            aromatic_centroids: 0,
            is_excited: 0,
            electronic_population: 0,
            atom_to_aromatic: 0x5000,
            protocol: 0x6000,
            exclusion_field: 0x10000,
            water_density: 0x20000,
            water_density_prev: 0x30000,
            density_epoch: 0,
            selected_atom_ids: 0x7000,
            active_tiles: 0,
            n_atoms: 256,
            n_aromatics: 0,
            grid_dim: 8,
            n_active_tiles: 0,
            grid_spacing: 0.5,
            origin_x: 0.0,
            origin_y: 0.0,
            origin_z: 0.0,
            native_coordinate_owner: 0x8000,
            au_coordinate_owner: 0x9000,
            reserved: 0,
        }
    }
    #[test]
    fn native_companion_descriptor_enforces_sparse_population_and_owned_fields() {
        let d = native_descriptor_fixture();
        assert_eq!(validate_native_observation_input(&d, &[]).unwrap(), 32);
        assert_eq!(
            validate_native_observation_input(&d, &[[0, 0, 0], [1, 3, 3]]).unwrap(),
            2
        );
        for bad in [
            vec![[0, 0, 0], [0, 0, 0]],
            vec![[-1, 0, 0]],
            vec![[2, 0, 0]],
            vec![[0, 4, 0]],
            vec![[0, 0, 4]],
        ] {
            assert!(validate_native_observation_input(&d, &bad).is_err());
        }
        let mut bad = d;
        bad.grid_dim = 129;
        assert!(validate_native_observation_input(&bad, &[]).is_err());
        bad = d;
        bad.n_atoms = 255;
        assert!(validate_native_observation_input(&bad, &[]).is_err());
        bad = d;
        bad.n_aromatics = 1;
        assert!(validate_native_observation_input(&bad, &[]).is_err());
        bad = d;
        bad.density_epoch = 0x8000;
        assert!(validate_native_observation_input(&bad, &[]).is_err());
        bad = d;
        bad.active_tiles = 0x8000;
        assert!(validate_native_observation_input(&bad, &[]).is_err());
        bad = d;
        bad.n_active_tiles = 1;
        assert!(validate_native_observation_input(&bad, &[]).is_err());
        bad = d;
        bad.protocol = 0;
        assert!(validate_native_observation_input(&bad, &[]).is_err());
    }
    #[test]
    fn native_companion_rejects_density_aliases_and_changed_physical_geometry() {
        let d = native_descriptor_fixture();
        let descriptors = [d; 8];
        let geometry = observation_geometry(&d);
        assert!(validate_native_observation_lif_join(&descriptors, geometry, 0, 0).is_ok());
        assert!(
            validate_native_observation_lif_join(&descriptors, geometry, 0x40000, 0x50000).is_ok()
        );
        for p in [
            d.exclusion_field,
            d.water_density + 4,
            d.water_density_prev - 4,
        ] {
            assert!(validate_native_observation_lif_join(&descriptors, geometry, p, 0).is_err());
            assert!(validate_native_observation_lif_join(&descriptors, geometry, 0, p).is_err());
        }
        assert!(
            validate_native_observation_lif_join(&descriptors, geometry, u64::MAX - 3, 0).is_err()
        );
        let mut changed = geometry;
        changed.origin_a[0] = -0.0;
        assert!(validate_native_observation_lif_join(&descriptors, changed, 0, 0).is_err());
    }
    #[test]
    fn native_companion_state_covers_exact_queue_and_preserves_fault_history_on_ack() {
        let words = [NATIVE_OBSERVATION_MAGIC, 3, 17, 0, u32::MAX, 9, 2, 0];
        let good = decode_native_observation_state(&words).unwrap();
        assert!(validate_native_observation_state(good, 3, 4, 17).is_ok());
        for (index, value) in [
            (0, 0),
            (1, 2),
            (2, 16),
            (3, 2),
            (4, 0),
            (5, 2),
            (6, 0),
            (7, 1),
        ] {
            let mut bad = words;
            bad[index] = value;
            assert!(validate_native_observation_state(
                decode_native_observation_state(&bad).unwrap(),
                3,
                4,
                17
            )
            .is_err());
            assert!(native_observation_ack_state(&bad, &words).is_err());
        }
        assert!(validate_native_observation_state(good, 3, 2, 17).is_err());
        assert!(validate_native_observation_state(good, 3, 4, u32::MAX).is_err());
        assert!(decode_native_observation_state(&words[..7]).is_err());
        let reset = native_observation_ack_state(&words, &words).unwrap();
        assert_eq!(
            reset,
            [NATIVE_OBSERVATION_MAGIC, 0, 17, 0, u32::MAX, 9, 2, 0]
        );
        let empty = [NATIVE_OBSERVATION_MAGIC, 0, 18, 0, u32::MAX, 9, 2, 0];
        assert!(validate_native_observation_state(
            decode_native_observation_state(&empty).unwrap(),
            0,
            4,
            18
        )
        .is_ok());
        assert!(native_observation_ack_state(&empty, &words).is_err());
    }
    fn carry_fixture() -> Vec<u32> {
        let mut words = vec![0; 8 * 5152 / 4];
        for w in words.chunks_exact_mut(5152 / 4) {
            w[0] = 1;
            w[1] = 17;
            w[2] = 17;
            w[4] = 72;
            for v in 0..256 {
                w[8 + v] = v as u32;
            }
        }
        words
    }
    #[test]
    fn biprism_grid_rejects_overflow_and_nonfinite_geometry() {
        let mut g = SignalGridSpecFfi {
            origin_xyz: [0.0; 3],
            inv_voxel_spacing: 1.0,
            dim_xyz: [16; 3],
            total_voxels: 4096,
        };
        assert!(g.validate().is_ok());
        g.dim_xyz = [u32::MAX; 3];
        assert!(g.validate().is_err());
        g.dim_xyz = [16; 3];
        g.inv_voxel_spacing = f32::NAN;
        assert!(g.validate().is_err());
    }
    #[test]
    fn captured_grid_refuses_same_cell_with_different_physical_center() {
        let first = SdstGridGeometry {
            dimension: 2,
            spacing_a: 1.0,
            origin_a: [0.0; 3],
        };
        let mut second = first;
        second.spacing_a = 2.0;
        let frame = BiprismEventFrame::NativeAngstrom;
        let bound = BoundBiprismGeometry {
            native: first,
            frame,
        };
        assert!(bind_geometry(None, first, frame).is_ok());
        assert!(bind_geometry(Some(bound), first, frame).is_ok());
        assert!(bind_geometry(Some(bound), second, frame).is_err());
        second = first;
        second.origin_a[0] = -0.0;
        assert!(bind_geometry(Some(bound), second, frame).is_err());
        assert!(bind_geometry(Some(bound), first, BiprismEventFrame::AtomicBohr).is_err());
        let atomic = BoundBiprismGeometry {
            native: first,
            frame: BiprismEventFrame::AtomicBohr,
        };
        let grid = bind_geometry(Some(atomic), first, BiprismEventFrame::AtomicBohr).unwrap();
        let spacing = crate::biprism_spike::BIPRISM_ANGSTROM_TO_BOHR as f32;
        assert_eq!(
            grid.inv_voxel_spacing.to_bits(),
            (1.0f32 / spacing).to_bits()
        );
    }
    #[test]
    fn carryover_export_requires_all_eight_successful_epochs_and_retains_faults() {
        let words = carry_fixture();
        let good = decode_carryover_words(&words).unwrap();
        assert_eq!(validate_carryover_epoch(&good, 256).unwrap(), 17);
        for leaf in 0..8 {
            let mut failed = good.clone();
            failed[leaf].fault = 32;
            assert!(validate_carryover_epoch(&failed, 256).is_err());
            assert_eq!(failed[leaf].fault, 32);
            let mut missing = good.clone();
            missing[leaf].pending = 0;
            assert!(validate_carryover_epoch(&missing, 256).is_err());
            let mut stale = good.clone();
            stale[leaf].produced_step = 16;
            assert!(validate_carryover_epoch(&stale, 256).is_err());
        }
        let mut delivered = good.clone();
        for c in &mut delivered {
            c.pending = 0;
            c.delivered_step = 18;
        }
        assert!(validate_carryover_epoch(&delivered, 256).is_ok());
        delivered[7].atom_id[255] = delivered[7].atom_id[254];
        assert!(validate_carryover_epoch(&delivered, 256).is_err());
        let mut invalid = good.clone();
        invalid[3].force[192][0] = f32::NAN;
        assert!(validate_carryover_epoch(&invalid, 256).is_err());
        let mut mismatched_delivery = good.clone();
        mismatched_delivery[7].delivered_step = 16;
        assert!(validate_carryover_epoch(&mismatched_delivery, 256).is_err());
        let mut exhausted = good.clone();
        for c in &mut exhausted {
            c.produced_committed_substeps = u32::MAX - 3;
        }
        assert!(validate_carryover_epoch(&exhausted, 256).is_err());
        assert!(decode_carryover_words(&words[..words.len() - 1]).is_err());
        assert!(validate_carryover_epoch(
            &decode_carryover_words(&vec![0; words.len()]).unwrap(),
            256
        )
        .is_err());
    }
    #[test]
    fn export_acknowledgement_rejects_duplicates_changed_producer_and_other_allocations() {
        let carry = carry_fixture();
        let mut epoch = QueueEpoch::default();
        let receipt = epoch.reserve(3, 17, carry.clone()).unwrap();
        assert!(epoch.reserve(3, 17, carry.clone()).is_err());
        assert!(epoch.validate_ack(&receipt, 4, &carry).is_err());
        let mut changed = carry.clone();
        changed[3] = 1;
        assert!(epoch.validate_ack(&receipt, 3, &changed).is_err());
        let mut other = QueueEpoch::default();
        other.reserve(3, 17, carry.clone()).unwrap();
        assert!(other.validate_ack(&receipt, 3, &carry).is_err());
        epoch.validate_ack(&receipt, 3, &carry).unwrap();
        epoch.complete_ack(&receipt);
        assert!(epoch.validate_ack(&receipt, 3, &carry).is_err());
        assert!(epoch.reserve(0, 17, carry.clone()).is_err());
        let next = epoch.reserve(0, 18, carry.clone()).unwrap();
        assert_eq!(next.generation, 1);
        epoch.validate_ack(&next, 0, &carry).unwrap();
    }
}
