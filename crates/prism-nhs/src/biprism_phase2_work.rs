//! Device measurement of the actual Phase2 operator refresh at fixed state.
//! These electronic-model work terms stay separate from native AMBER work.
use crate::biprism_v10_runtime::{BiprismDeviceViews, BIPRISM_TILE_BYTES};
use cudarc::driver::{
    CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, DeviceRepr, LaunchConfig,
    PushKernelArg,
};
use serde::Serialize;
use std::sync::Arc;
pub const PHASE2_WORK_MAGIC: u32 = 0x42505735;
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Phase2VertexWork {
    pub packed_atom: u32,
    pub leaf: u32,
    pub local_vertex: u32,
    pub state: u32,
    pub position: [f32; 3],
    pub reference: [f32; 3],
    pub psi: [f32; 4],
    pub amber_charge: f32,
    pub omega_before: f32,
    pub omega_after: f32,
    pub potential_alignment:u32,
    pub phi_before: f64,
    pub phi_same_halo: f64,
    pub phi_after: f64,
    pub berry_before: f32,
    pub berry_after: f32,
    pub charge: f64,
    pub rho00: f64,
    pub rho11: f64,
    pub population: f64,
    pub onsite_delta: f64,
    pub anchor_delta: f64,
    pub pair_delta: f64,
    pub boundary_work: f64,
    pub stationarity_operator: f64,
    pub stationarity_boundary: f64,
    pub onsite_eps_s: f32,
    pub onsite_eps_p: f32,
    pub position_lo: [f32; 3],
    pub reference_lo: [f32; 3],
    pub phi_entry:f64,
    pub psi_precise:[f64;4],
}
unsafe impl DeviceRepr for Phase2VertexWork {}
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Phase2EdgeWork {
    pub leaf: u32,
    pub edge: u32,
    pub u: u32,
    pub v: u32,
    pub base_before: f32,
    pub base_after: f32,
    pub live_before: f64,
    pub live_after: f64,
    pub operator_cross_work: f64,
    pub energy_before: f64,
    pub energy_same_halo: f64,
    pub energy_after: f64,
}
unsafe impl DeviceRepr for Phase2EdgeWork {}
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Phase2WorkLedger {
    pub magic: u32,
    pub leaf: u32,
    pub state: u32,
    pub sticky_fault: u32,
    pub scheduled_step: u32,
    pub entry_committed: u32,
    pub edges: u32,
    pub tets: u32,
    pub generation: u64,
    pub preparation_generation: u64,
    pub body_ordinal: u64,
    pub kappa: f64,
    pub residual: [f32; 3],
    pub iterations: [u32; 3],
    pub totals: [f64; 16],
    pub completed_measurements: u64,
    pub immutable_mismatches: u32,
    pub reserved: u32,
    pub entry_refinement_delta: f64,
}
unsafe impl DeviceRepr for Phase2WorkLedger {}
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Phase2WorkWindow {
    pub magic: u32,
    pub capacity: u32,
    pub first_body: u32,
    pub requested_bodies: u32,
    pub first_scheduled_step: u32,
    pub reserved: u32,
    pub generation: u64,
    pub preparation_generation: u64,
    pub reserved_tail: u64,
}
unsafe impl DeviceRepr for Phase2WorkWindow {}
const _: () = {
    assert!(core::mem::size_of::<Phase2VertexWork>() == 256);
    assert!(core::mem::offset_of!(Phase2VertexWork,phi_before)==72);
    assert!(core::mem::offset_of!(Phase2VertexWork, charge) == 104);
    assert!(core::mem::offset_of!(Phase2VertexWork, position_lo) == 192);
    assert!(core::mem::offset_of!(Phase2VertexWork,phi_entry)==216);
    assert!(core::mem::offset_of!(Phase2EdgeWork,live_before)==24&&core::mem::offset_of!(Phase2EdgeWork,operator_cross_work)==40);
    assert!(core::mem::size_of::<Phase2EdgeWork>() == 80);
    assert!(core::mem::size_of::<Phase2WorkLedger>() == 240);
    assert!(core::mem::offset_of!(Phase2WorkLedger, totals) == 88);
    assert!(core::mem::offset_of!(Phase2WorkLedger,entry_refinement_delta)==232);
    assert!(core::mem::size_of::<Phase2WorkWindow>() == 48);
};
pub const PHASE2_TOTAL_NAMES: [&str; 16] = [
    "dec_before_hartree",
    "dec_new_operator_old_halo_hartree",
    "dec_after_hartree",
    "dec_fixed_halo_energy_delta_hartree",
    "dec_cross_operator_work_hartree",
    "dec_operator_finite_solve_correction_hartree",
    "dec_boundary_energy_delta_hartree",
    "dec_boundary_reaction_work_hartree",
    "dec_boundary_finite_solve_correction_hartree",
    "onsite_omega_energy_delta_hartree",
    "ring_anchor_delta_hartree",
    "owned_directed_ring_pair_delta_hartree",
    "total_operator_refresh_energy_delta_hartree",
    "operator_identity_roundoff_hartree",
    "boundary_identity_roundoff_hartree",
    "absolute_core_berry_weight_change",
];
pub struct Phase2WorkReadback {
    pub window: Option<Phase2WorkWindow>,
    pub ledgers: Vec<Phase2WorkLedger>,
    pub vertices: Vec<Phase2VertexWork>,
    pub edges: Vec<Phase2EdgeWork>,
    pub current_attempt: bool,
}
impl Phase2WorkReadback {
    pub fn ledger_bytes(&self) -> Result<&[u8], String> {
        bytes_le(&self.ledgers)
    }
    pub fn vertex_bytes(&self) -> Result<&[u8], String> {
        bytes_le(&self.vertices)
    }
    pub fn edge_bytes(&self) -> Result<&[u8], String> {
        bytes_le(&self.edges)
    }
    pub fn index(&self) -> serde_json::Value {
        serde_json::json!({"schema":"biprism.phase2.operator_work.v3","window":self.window,
        "current_attempt":self.current_attempt,"ledger_bytes":240,"vertex_bytes":256,"edge_bytes":80,"totals":PHASE2_TOTAL_NAMES,
        "potential":"full IEEE754 binary64 entry/before/same-halo/after; tile FP32 projection is not authority",
        "entry_refinement_delta":"numerical solved-minus-entry functional diagnostic, excluded from total declared work",
        "order":"body,leaf,local_vertex or active edge;1536 edge slots per leaf, inactive slots zero",
        "state_scope":"measurement3 is published operator refresh; collective outer-body acceptance is separately required",
        "fixed_state":"actual imported x,unchanged psi/charges/selected IDs/incidence/reference during Phase2; no physical clock advances",
        "energy_functional":"q_C^T phi_C -0.5 phi^T L phi; raw omega onsite; directed-ring pair J=kappa*(omega_i+omega_next)/2",
        "finite_solver_scope":"three actual true-residual-qualified solves; finite-solve corrections retained separately from cross work",
        "state_transport":"represented psi held fixed; Berry mass=max(raw omega,1e-4f) before/after recorded, no hidden normalization",
        "native_AMBER_work_injection":false,"scientific_restart":false})
    }
}
pub struct Phase2InitializationReadback {
    pub generation: u64,
    pub preparation_generation: u64,
    pub native_completed_step:u32,
    pub before_tiles: Vec<u8>,
    pub after_tiles: Vec<u8>,
    pub ledgers: Option<[Phase2WorkLedger; 8]>,
}
impl Phase2InitializationReadback {
    pub fn ledger_bytes(&self) -> Result<&[u8], String> {
        bytes_le(
            self.ledgers
                .as_ref()
                .ok_or("initial Phase2 ledgers not read")?,
        )
    }
    pub fn validate(&self) -> Result<(), String> {
        if self.generation == 0 || self.preparation_generation == 0 {
            return Err("initial operator generation missing".into());
        }
        validate_phase2_initial_identity(&self.before_tiles, &self.after_tiles)?;
        validate_phase2_initial_ledgers_at_step(
            self.ledgers
                .as_ref()
                .ok_or("initial Phase2 ledgers not read")?,self.native_completed_step,
        )
    }
    pub fn index(&self) -> serde_json::Value {
        serde_json::json!({
        "schema":"biprism.initial_operator.v1","generation":self.generation,"preparation_generation":self.preparation_generation,
        "native_completed_step":self.native_completed_step,"physical_step_advanced":false,"accepted_main_substeps":0,
        "before_scope":"actual admitted BW tile prior to first actual Phase2 assembly",
        "after_scope":"actual Phase2 operator and true-residual-qualified DEC solve before first physical energy baseline",
        "tile_bytes":BIPRISM_TILE_BYTES,"leaves":8,"ledger_bytes":240,"ledgers":self.ledgers,
        "immutable_identity":"all bytes before phi_dec, active tet/edge counts and pocket identity unchanged",
        "subsequent_refresh":"must retain separate operator-refresh work; never reset the accumulated physical baseline",
        "scientific_restart":false})
    }
}
fn bytes_le<T>(x: &[T]) -> Result<&[u8], String> {
    if !cfg!(target_endian = "little") {
        return Err("Phase2 work requires little endian".into());
    }
    Ok(unsafe { std::slice::from_raw_parts(x.as_ptr().cast(), core::mem::size_of_val(x)) })
}
fn extents(capacity: u32) -> Result<(usize, usize, usize), String> {
    if !(1..=500).contains(&capacity) {
        return Err("Phase2 journal capacity outside1..500".into());
    }
    let l = (capacity as usize)
        .checked_mul(8)
        .ok_or("Phase2 ledger extent")?;
    Ok((
        l,
        l.checked_mul(256).ok_or("Phase2 vertex extent")?,
        l.checked_mul(1536).ok_or("Phase2 edge extent")?,
    ))
}
pub(crate) struct BiprismPhase2WorkOwner {
    stream: Arc<CudaStream>,
    snapshots: CudaSlice<u8>,
    ledgers: CudaSlice<Phase2WorkLedger>,
    vertices: CudaSlice<Phase2VertexWork>,
    edges: CudaSlice<Phase2EdgeWork>,
    window: CudaSlice<Phase2WorkWindow>,
    journal_ledgers: CudaSlice<Phase2WorkLedger>,
    journal_vertices: CudaSlice<Phase2VertexWork>,
    journal_edges: CudaSlice<Phase2EdgeWork>,
    initialize: CudaFunction,
    initialize_publish:CudaFunction,
    capture: CudaFunction,
    measure: CudaFunction,
    publish: CudaFunction,
    selected: [[u32; 256]; 8],
    capacity: u32,
    generation: u64,
    preparation_generation: Option<u64>,
    current: Option<Phase2WorkWindow>,
    kappa: f32,
}
impl BiprismPhase2WorkOwner {
    pub(crate) fn new(
        stream: Arc<CudaStream>,
        module: &Arc<CudaModule>,
        selected: [[u32; 256]; 8],
        capacity: u32,
        generation: u64,
        kappa: f32,
    ) -> Result<Self, String> {
        let (l, v, e) = extents(capacity)?;
        if generation == 0 || !kappa.is_finite() {
            return Err("Phase2 owner generation/coupling invalid".into());
        }
        Ok(Self {
            initialize: module
                .load_function("nhs_biprism_phase2_work_initialize_v1")
                .map_err(|e| e.to_string())?,
            initialize_publish:module.load_function("nhs_biprism_phase2_work_initialize_publish_v1").map_err(|e|e.to_string())?,
            capture: module
                .load_function("nhs_biprism_phase2_work_capture_v1")
                .map_err(|e| e.to_string())?,
            measure: module
                .load_function("nhs_biprism_phase2_work_measure_v1")
                .map_err(|e| e.to_string())?,
            publish: module
                .load_function("nhs_biprism_phase2_work_publish_v1")
                .map_err(|e| e.to_string())?,
            snapshots: upload(&stream, &vec![0u8; 8 * BIPRISM_TILE_BYTES])?,
            ledgers: upload(&stream, &[Phase2WorkLedger::default(); 8])?,
            vertices: upload(&stream, &vec![Phase2VertexWork::default(); 8 * 256])?,
            edges: upload(&stream, &vec![Phase2EdgeWork::default(); 8 * 1536])?,
            window: upload(&stream, &[Phase2WorkWindow::default()])?,
            journal_ledgers: upload(&stream, &vec![Phase2WorkLedger::default(); l])?,
            journal_vertices: upload(&stream, &vec![Phase2VertexWork::default(); v])?,
            journal_edges: upload(&stream, &vec![Phase2EdgeWork::default(); e])?,
            stream,
            selected,
            capacity,
            generation,
            preparation_generation: None,
            current: None,
            kappa,
        })
    }
    /// Cold only, after the actual Phase2 initializer and before capture. The
    /// caller compares pre/post tile identity and retains the resulting operator.
    pub(crate) unsafe fn initialize(
        &self,
        v: BiprismDeviceViews,
        atoms: u32,
        configured_step:u64,
    ) -> Result<[Phase2WorkLedger; 8], String> {
        let rows = self.vertices.device_ptr(&self.stream).0;
        let l = self.ledgers.device_ptr(&self.stream).0;
        self.stream
            .launch_builder(&self.initialize)
            .arg(&v.tiles)
            .arg(&v.stream_table)
            .arg(&v.charges)
            .arg(&atoms)
            .arg(&v.convergence)
            .arg(&rows)
            .arg(&l)
            .launch(LaunchConfig {
                grid_dim: (8, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
            .map_err(|e| e.to_string())?;
        self.stream.launch_builder(&self.initialize_publish)
            .arg(&v.tiles).arg(&v.stream_table).arg(&v.convergence).arg(&rows).arg(&l).arg(&configured_step)
            .launch(LaunchConfig{grid_dim:(1,1,1),block_dim:(256,1,1),shared_mem_bytes:0})
            .map_err(|e|e.to_string())?;
        self.read_ledgers()
    }

    pub(crate) fn begin_window(
        &mut self,
        first_body: u32,
        step: u32,
        count: u32,
        preparation_generation: u64,
    ) -> Result<(), String> {
        if self.current.is_some()
            || preparation_generation == 0
            || count == 0
            || count > self.capacity
            || first_body
                .checked_add(count)
                .filter(|x| *x <= 0x3fffffff)
                .is_none()
            || step.checked_add(count).filter(|x| *x < u32::MAX).is_none()
            || self
                .preparation_generation
                .is_some_and(|old| old != preparation_generation)
        {
            return Err("Phase2 journal epoch refused".into());
        }
        let w = Phase2WorkWindow {
            magic: PHASE2_WORK_MAGIC,
            capacity: self.capacity,
            first_body,
            requested_bodies: count,
            first_scheduled_step: step,
            reserved: 0,
            generation: self.generation,
            preparation_generation,
            reserved_tail: 0,
        };
        let source = [w];
        let copy = self
            .stream
            .memcpy_htod(&source, &mut self.window)
            .map_err(|e| e.to_string());
        let drain = self.stream.synchronize().map_err(|e| e.to_string());
        completed(copy, drain)?;
        self.preparation_generation = Some(preparation_generation);
        self.current = Some(w);
        Ok(())
    }
    pub(crate) unsafe fn capture_before(
        &self,
        v: BiprismDeviceViews,
        shared: u64,
        atoms: u32,
    ) -> Result<(), String> {
        let snapshots = self.snapshots.device_ptr(&self.stream).0;
        let rows = self.vertices.device_ptr(&self.stream).0;
        let edges = self.edges.device_ptr(&self.stream).0;
        let ledgers = self.ledgers.device_ptr(&self.stream).0;
        let window = self.window.device_ptr(&self.stream).0;
        self.stream
            .launch_builder(&self.capture)
            .arg(&v.tiles)
            .arg(&v.stream_table)
            .arg(&v.charges)
            .arg(&v.unit_bridge_status)
            .arg(&v.convergence)
            .arg(&shared)
            .arg(&atoms)
            .arg(&self.kappa)
            .arg(&snapshots)
            .arg(&rows)
            .arg(&edges)
            .arg(&ledgers)
            .arg(&window)
            .launch(LaunchConfig {
                grid_dim: (8, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
            .map_err(|e| e.to_string())?;
        Ok(())
    }
    pub(crate) unsafe fn measure_after(
        &self,
        v: BiprismDeviceViews,
        shared: u64,
    ) -> Result<(), String> {
        let snapshots = self.snapshots.device_ptr(&self.stream).0;
        let rows = self.vertices.device_ptr(&self.stream).0;
        let edges = self.edges.device_ptr(&self.stream).0;
        let ledgers = self.ledgers.device_ptr(&self.stream).0;
        let window = self.window.device_ptr(&self.stream).0;
        let jr = self.journal_vertices.device_ptr(&self.stream).0;
        let je = self.journal_edges.device_ptr(&self.stream).0;
        let jl = self.journal_ledgers.device_ptr(&self.stream).0;
        self.stream
            .launch_builder(&self.measure)
            .arg(&v.tiles)
            .arg(&v.stream_table)
            .arg(&v.charges)
            .arg(&snapshots)
            .arg(&v.convergence)
            .arg(&shared)
            .arg(&rows)
            .arg(&edges)
            .arg(&ledgers)
            .launch(LaunchConfig {
                grid_dim: (8, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
            .map_err(|e| e.to_string())?;
        self.stream
            .launch_builder(&self.publish)
            .arg(&v.convergence)
            .arg(&shared)
            .arg(&rows)
            .arg(&edges)
            .arg(&ledgers)
            .arg(&window)
            .arg(&jr)
            .arg(&je)
            .arg(&jl)
            .arg(&v.tiles).arg(&v.stream_table)
            .launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
            .map_err(|e| e.to_string())?;
        Ok(())
    }
    pub(crate) fn guard_pointers(&self) -> (u64, u64, u64) {
        (
            self.ledgers.device_ptr(&self.stream).0,
            self.window.device_ptr(&self.stream).0,
            self.journal_ledgers.device_ptr(&self.stream).0,
        )
    }
    pub(crate) fn read_ledgers(&self) -> Result<[Phase2WorkLedger; 8], String> {
        download(&self.stream, &self.ledgers)?
            .try_into()
            .map_err(|_| "Phase2 ledger extent".into())
    }
    pub(crate) fn read_window(&self) -> Result<Phase2WorkReadback, String> {
        let w = self.current.ok_or("no Phase2 work window")?;
        let mut ledgers = download(&self.stream, &self.journal_ledgers)?;
        let mut vertices = download(&self.stream, &self.journal_vertices)?;
        let mut edges = download(&self.stream, &self.journal_edges)?;
        ledgers.truncate(w.requested_bodies as usize * 8);
        vertices.truncate(ledgers.len() * 256);
        edges.truncate(ledgers.len() * 1536);
        Ok(Phase2WorkReadback {
            window: Some(w),
            ledgers,
            vertices,
            edges,
            current_attempt: false,
        })
    }
    pub(crate) fn read_current(&self) -> Result<Phase2WorkReadback, String> {
        Ok(Phase2WorkReadback {
            window: self.current,
            ledgers: download(&self.stream, &self.ledgers)?,
            vertices: download(&self.stream, &self.vertices)?,
            edges: download(&self.stream, &self.edges)?,
            current_attempt: true,
        })
    }
    pub(crate) fn validate(&self, r: &Phase2WorkReadback) -> Result<(), String> {
        validate_phase2_work(r, &self.selected, self.kappa)
    }
    pub(crate) fn acknowledge(&mut self) {
        self.current = None;
    }
}
fn completed<T>(copy: Result<T, String>, drain: Result<(), String>) -> Result<T, String> {
    match (copy, drain) {
        (Ok(x), Ok(())) => Ok(x),
        (Err(a), Ok(())) => Err(a),
        (Ok(_), Err(b)) => Err(format!("Phase2 transfer completion: {b}")),
        (Err(a), Err(b)) => Err(format!("Phase2 copy: {a}; completion: {b}")),
    }
}
fn upload<T: DeviceRepr + Copy>(s: &Arc<CudaStream>, x: &[T]) -> Result<CudaSlice<T>, String> {
    let copy = s.clone_htod(x).map_err(|e| e.to_string());
    let drain = s.synchronize().map_err(|e| e.to_string());
    completed(copy, drain)
}
fn download<T: DeviceRepr + Copy + Default>(
    s: &Arc<CudaStream>,
    x: &CudaSlice<T>,
) -> Result<Vec<T>, String> {
    let mut out = vec![T::default(); x.len()];
    let copy = s.memcpy_dtoh(x, &mut out).map_err(|e| e.to_string());
    let drain = s.synchronize().map_err(|e| e.to_string());
    completed(copy, drain)?;
    Ok(out)
}
fn close(a: f64, b: f64) -> bool {
    a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-12 * (1. + a.abs().max(b.abs()))
}
/// This check validates actual initialization, not a fabricated restart state.
/// Phase2 may change only operator/solve/cache fields after the immutable prefix.
pub fn validate_phase2_initial_ledgers(read: &[Phase2WorkLedger; 8]) -> Result<(), String> {
    validate_phase2_initial_ledgers_at_step(read,0)
}
pub fn validate_phase2_initial_ledgers_at_step(read:&[Phase2WorkLedger;8],step:u32)->Result<(),String>{
    for (i, a) in read.iter().enumerate() {
        if a.magic != PHASE2_WORK_MAGIC
            || a.leaf != i as u32
            || a.state != 4
            || a.sticky_fault != 0
            || a.scheduled_step != step
            || a.entry_committed != 0
            || a.body_ordinal != 0
            || a.completed_measurements != 0
            || a.edges == 0
            || a.edges > 1536
            || a.tets == 0
            || a.tets > 1024
            || a.immutable_mismatches != 0
            || a.reserved != 0
            || !a.entry_refinement_delta.is_finite()
            || !a.residual[0].is_finite()
            || a.residual[0] < 0.
            || a.residual[0] > 1e-6
            || a.iterations[0] > 192
        {
            return Err(format!(
                "initial actual DEC qualification refused leaf{i}: {a:?}"
            ));
        }
    }
    Ok(())
}
pub fn validate_phase2_initial_identity(before: &[u8], after: &[u8]) -> Result<(), String> {
    if before.len() != 8 * BIPRISM_TILE_BYTES || after.len() != before.len() {
        return Err("initial tile population extent".into());
    }
    for leaf in 0..8 {
        let a = &before[leaf * BIPRISM_TILE_BYTES..(leaf + 1) * BIPRISM_TILE_BYTES];
        let b = &after[leaf * BIPRISM_TILE_BYTES..(leaf + 1) * BIPRISM_TILE_BYTES];
        if a[..94208] != b[..94208]
            || a[225280..225288] != b[225280..225288]
            || a[225292..225296] != b[225292..225296]
        {
            return Err(format!(
                "initial Phase2 changed immutable tile state leaf{leaf}"
            ));
        }
    }
    Ok(())
}
pub fn validate_phase2_work(
    r: &Phase2WorkReadback,
    selected: &[[u32; 256]; 8],
    kappa: f32,
) -> Result<(), String> {
    let w = r.window.ok_or("missing Phase2 window")?;
    extents(w.capacity)?;
    if r.current_attempt
        || w.magic != PHASE2_WORK_MAGIC
        || w.generation == 0
        || w.preparation_generation == 0
        || w.reserved != 0
        || w.reserved_tail != 0
        || w.requested_bodies == 0
        || w.requested_bodies > w.capacity
        || r.ledgers.len() != w.requested_bodies as usize * 8
        || r.vertices.len() != r.ledgers.len() * 256
        || r.edges.len() != r.ledgers.len() * 1536
    {
        return Err("Phase2 journal population mismatch".into());
    }
    for (i, l) in r.ledgers.iter().enumerate() {
        let leaf = i % 8;
        let body = u64::from(w.first_body) + i as u64 / 8;
        let step = u64::from(w.first_scheduled_step) + i as u64 / 8;
        if l.magic != PHASE2_WORK_MAGIC
            || l.leaf != leaf as u32
            || l.state != 3
            || l.sticky_fault != 0
            || l.immutable_mismatches != 0
            || l.reserved != 0
            || !l.entry_refinement_delta.is_finite()
            || l.generation != w.generation
            || l.preparation_generation != w.preparation_generation
            || l.body_ordinal != body
            || l.completed_measurements != body + 1
            || u64::from(l.entry_committed) != body * 4
            || u64::from(l.scheduled_step) != step
            || step >= u32::MAX as u64
            || l.kappa != f64::from(kappa)
            || l.edges == 0
            || l.edges > 1536
            || l.tets == 0
            || l.tets > 1024
            || l.residual
                .iter()
                .any(|x| !x.is_finite() || *x < 0. || *x > 1e-6)
            || l.iterations.iter().any(|x| *x > 192)
            || l.totals.iter().any(|x| !x.is_finite())
        {
            return Err(format!("Phase2 ledger identity/solve invalid row{i}"));
        }
        let vr = &r.vertices[i * 256..(i + 1) * 256];
        let er = &r.edges[i * 1536..i * 1536 + l.edges as usize];
        let next_i = i - leaf + (leaf + 1) % 8;
        let next = &r.vertices[next_i * 256..(next_i + 1) * 256];
        let mut flux = [[0.0f64; 256]; 3];
        let mut total = [0.0f64; 16];
        let mut entry_energy=0.0f64;
        for (e, a) in er.iter().enumerate() {
            if a.leaf != leaf as u32
                || a.edge != e as u32
                || a.u >= 256
                || a.v >= 256
                || a.u == a.v
                || [f64::from(a.base_before), f64::from(a.base_after), a.live_before, a.live_after]
                    .iter()
                    .any(|x| !x.is_finite() || *x <= 0.)
            {
                return Err(format!("Phase2 active edge invalid row{i} edge{e}"));
            }
            let (u, v) = (a.u as usize, a.v as usize);
            let entry_difference=vr[u].phi_entry-vr[v].phi_entry;
            entry_energy+=-0.5*f64::from(a.live_before)*entry_difference*entry_difference;
            let d = [
                f64::from(vr[u].phi_before) - f64::from(vr[v].phi_before),
                f64::from(vr[u].phi_same_halo) - f64::from(vr[v].phi_same_halo),
                f64::from(vr[u].phi_after) - f64::from(vr[v].phi_after),
            ];
            let weights = [
                f64::from(a.live_before),
                f64::from(a.live_after),
                f64::from(a.live_after),
            ];
            for k in 0..3 {
                let f = weights[k] * d[k];
                flux[k][u] += f;
                flux[k][v] -= f;
            }
            let energy = [
                -0.5 * weights[0] * d[0] * d[0],
                -0.5 * weights[1] * d[1] * d[1],
                -0.5 * weights[2] * d[2] * d[2],
            ];
            if !close(
                a.operator_cross_work,
                -0.5 * (weights[1] - weights[0]) * d[0] * d[1],
            ) || !close(a.energy_before, energy[0])
                || !close(a.energy_same_halo, energy[1])
                || !close(a.energy_after, energy[2])
            {
                return Err("Phase2 edge primitive energy mismatch".into());
            }
            for k in 0..3 {
                total[k] += energy[k];
            }
            total[4] += a.operator_cross_work;
        }
        let mut rr = [0.0f64; 3];
        for (u, v) in vr.iter().enumerate() {
            let core = u < 192;
            if v.leaf != leaf as u32
                || v.local_vertex != u as u32
                || v.state != 3
                || v.potential_alignment != 0
                || (v.packed_atom & 0x00ffffff) != selected[leaf][u]
                || v.position
                    .iter()
                    .chain(&v.reference)
                    .chain(&v.position_lo)
                    .chain(&v.reference_lo)
                    .chain(&v.psi)
                    .any(|x| !x.is_finite())
                || [
                    v.amber_charge,
                    v.omega_before,
                    v.omega_after,
                    v.berry_before,
                    v.berry_after,
                    v.onsite_eps_s,
                    v.onsite_eps_p,
                ]
                .iter()
                .any(|x| !x.is_finite())
                ||[v.phi_entry,v.phi_before,v.phi_same_halo,v.phi_after].iter().any(|x|!x.is_finite())
                || v.psi_precise.iter().any(|x|!x.is_finite())
                || (core&&(0..4).any(|j|(v.psi_precise[j] as f32).to_bits()!=v.psi[j].to_bits()))
                || v.omega_before <= 0.
                || v.omega_after <= 0.
                || v.berry_before != v.omega_before.max(1e-4)
                || v.berry_after != v.omega_after.max(1e-4)
            {
                return Err(format!("Phase2 vertex identity invalid row{i} local{u}"));
            }
            let rho0 = if core {
                v.psi_precise[0].mul_add(
                    v.psi_precise[0],
                    v.psi_precise[1] * v.psi_precise[1],
                )
            } else {
                1.
            };
            let rho1 = if core {
                v.psi_precise[2].mul_add(
                    v.psi_precise[2],
                    v.psi_precise[3] * v.psi_precise[3],
                )
            } else {
                0.
            };
            let q = if core {
                f64::from(0.15f32).mul_add(rho1, f64::from(v.amber_charge))
            } else {
                f64::from(v.amber_charge)
            };
            if !close(v.rho00, rho0)
                || !close(v.rho11, rho1)
                || !close(v.charge, q)
                || !close(v.population, rho0 - rho1)
            {
                return Err("Phase2 represented state quadratic mismatch".into());
            }
            let ph = [
                f64::from(v.phi_before),
                f64::from(v.phi_same_halo),
                f64::from(v.phi_after),
            ];
            let residual = [q - flux[0][u], q - flux[1][u], q - flux[2][u]];
            let op = if core {
                0.5 * (ph[1] - ph[0]) * (residual[0] + residual[1])
            } else {
                0.
            };
            let boundary = if core {
                0.
            } else {
                -0.5 * (flux[1][u] + flux[2][u]) * (ph[2] - ph[1])
            };
            let bc = if core {
                0.5 * (ph[2] - ph[1]) * (residual[1] + residual[2])
            } else {
                0.
            };
            let dw = f64::from(v.omega_after) - f64::from(v.omega_before);
            let onsite = if core {
                dw * (f64::from(v.onsite_eps_s) * rho0 + f64::from(v.onsite_eps_p) * rho1)
            } else {
                0.
            };
            let anchor = if core { f64::from(kappa) * dw } else { 0. };
            let pair = if core {
                let n = &next[u];
                if (n.packed_atom & 0x00ffffff) != (v.packed_atom & 0x00ffffff) {
                    return Err("Phase2 ring atom correspondence changed".into());
                }
                -0.5 * f64::from(kappa)
                    * (dw + f64::from(n.omega_after) - f64::from(n.omega_before))
                    * (rho0 - rho1)
                    * n.population
            } else {
                0.
            };
            if !close(v.stationarity_operator, op)
                || !close(v.boundary_work, boundary)
                || !close(v.stationarity_boundary, bc)
                || !close(v.onsite_delta, onsite)
                || !close(v.anchor_delta, anchor)
                || !close(v.pair_delta, pair)
            {
                return Err("Phase2 vertex primitive work mismatch".into());
            }
            if core {
                entry_energy+=q*v.phi_entry;
                for k in 0..3 {
                    total[k] += q * ph[k];
                    rr[k] += residual[k] * residual[k];
                }
                total[15] += (f64::from(v.berry_after) - f64::from(v.berry_before)).abs();
            } else if v.phi_before.to_bits() != v.phi_same_halo.to_bits() {
                return Err("operator solve changed held halo".into());
            }
            total[5] += op;
            total[7] += boundary;
            total[8] += bc;
            total[9] += onsite;
            total[10] += anchor;
            total[11] += pair;
        }
        // The true residual is reconstructed from the retained graph, not inferred from solver success.
        if rr
            .iter()
            .any(|x| !x.is_finite() || *x > f64::from(1e-12f32))
        {
            return Err("reconstructed Phase2 true residual exceeds existing gate".into());
        }
        total[3] = total[1] - total[0];
        total[6] = total[2] - total[1];
        total[12] = total[2] - total[0] + total[9] + total[10] + total[11];
        total[13] = total[3] - total[4] - total[5];
        total[14] = total[6] - total[7] - total[8];
        if !close(l.entry_refinement_delta,total[0]-entry_energy){return Err("Phase2 entry refinement diagnostic mismatch".into());}
        if total[13].abs()>1e-6 || total[14].abs()>1e-6
            || l.totals[13].abs()>1e-6 || l.totals[14].abs()>1e-6 {
            return Err("Phase2 discrete identity exceeds device acceptance tolerance".into());
        }
        for k in 0..16 {
            if !close(total[k], l.totals[k]) {
                return Err(format!(
                    "Phase2 ledger differs from primitive rows row{i} total{k}"
                ));
            }
        }
        for a in &r.edges[i * 1536 + l.edges as usize..(i + 1) * 1536] {
            if bytes_le(std::slice::from_ref(a))?.iter().any(|x| *x != 0) {
                return Err("Phase2 inactive edge storage presented as produced".into());
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (Phase2WorkReadback, [[u32; 256]; 8]) {
        let selected = std::array::from_fn(|_| std::array::from_fn(|u| u as u32));
        let w = Phase2WorkWindow {
            magic: PHASE2_WORK_MAGIC,
            capacity: 1,
            first_body: 0,
            requested_bodies: 1,
            first_scheduled_step: 1,
            generation: 7,
            preparation_generation: 3,
            ..Default::default()
        };
        let mut r = Phase2WorkReadback {
            window: Some(w),
            ledgers: vec![Phase2WorkLedger::default(); 8],
            vertices: vec![Phase2VertexWork::default(); 8 * 256],
            edges: vec![Phase2EdgeWork::default(); 8 * 1536],
            current_attempt: false,
        };
        for leaf in 0..8 {
            let l = &mut r.ledgers[leaf];
            *l = Phase2WorkLedger {
                magic: PHASE2_WORK_MAGIC,
                leaf: leaf as u32,
                state: 3,
                scheduled_step: 1,
                edges: 192,
                tets: 1,
                generation: 7,
                preparation_generation: 3,
                kappa: f64::from(0.2f32),
                completed_measurements: 1,
                ..Default::default()
            };
            for u in 0..256 {
                let core = u < 192;
                let v = &mut r.vertices[leaf * 256 + u];
                *v = Phase2VertexWork {
                    packed_atom: u as u32,
                    leaf: leaf as u32,
                    local_vertex: u as u32,
                    state: 3,
                    position: [u as f32, 0., 0.],
                    reference: [u as f32, 0., 0.],
                    psi: [1., 0., 0., 0.],psi_precise:[1.,0.,0.,0.],
                    amber_charge: if core { 1. } else { 0. },
                    omega_before: 2.,
                    omega_after: 3.,
                    berry_before: 2.,
                    berry_after: 3.,
                    phi_before: if core { 3.5 } else { 3. },
                    phi_entry: if core { 3.5 } else { 3. },
                    phi_same_halo: if core { 3.25 } else { 3. },
                    phi_after: if core { 5.25 } else { 5. },
                    charge: if core { 1. } else { 0. },
                    rho00: 1.,
                    population: 1.,
                    onsite_eps_s: -0.5,
                    onsite_eps_p: -0.25,
                    onsite_delta: if core { -0.5 } else { 0. },
                    anchor_delta: if core { f64::from(0.2f32) } else { 0. },
                    pair_delta: if core { -f64::from(0.2f32) } else { 0. },
                    boundary_work: if core { 0. } else { 6. },
                    ..Default::default()
                };
            }
            for u in 0..192 {
                r.edges[leaf * 1536 + u] = Phase2EdgeWork {
                    leaf: leaf as u32,
                    edge: u as u32,
                    u: u as u32,
                    v: (192 + u % 64) as u32,
                    base_before: 2.,
                    base_after: 4.,
                    live_before: 2.,
                    live_after: 4.,
                    operator_cross_work: -0.125,
                    energy_before: -0.25,
                    energy_same_halo: -0.125,
                    energy_after: -0.125,
                };
            }
            l.totals = [
                624.,
                600.,
                984.,
                -24.,
                -24.,
                0.,
                384.,
                384.,
                0.,
                -96.,
                192. * f64::from(0.2f32),
                -192. * f64::from(0.2f32),
                264.,
                0.,
                0.,
                192.,
            ];
        }
        (r, selected)
    }
    #[test]
    fn represented_dirichlet_refresh_passes() {
        let (r, s) = fixture();
        assert!(validate_phase2_work(&r, &s, 0.2).is_ok());
    }

    #[test]
    fn full_potential_row_preserves_bits_beyond_fp32() {
        let mut row=Phase2VertexWork::default();row.phi_entry=1.0+2.0f64.powi(-40);
        row.phi_before=row.phi_entry+2.0f64.powi(-39);
        let bytes=bytes_le(std::slice::from_ref(&row)).unwrap();
        assert_eq!(bytes.len(),256);
        assert_eq!(u64::from_le_bytes(bytes[216..224].try_into().unwrap()),row.phi_entry.to_bits());
        assert_eq!(u64::from_le_bytes(bytes[72..80].try_into().unwrap()),row.phi_before.to_bits());
        assert_ne!(row.phi_before,f64::from(row.phi_before as f32));
    }

    #[test]
    fn numerical_refinement_is_reconstructed_not_credited() {
        let (mut r,s)=fixture();
        // The entry value is deliberately displaced from the independently
        // qualified solve. For q=1,w=2,d=0.5 the functional change is -delta².
        r.vertices[0].phi_entry+=0.125;
        r.ledgers[0].entry_refinement_delta=0.015625;
        assert!(validate_phase2_work(&r,&s,0.2).is_ok());
        assert_eq!(r.ledgers[0].totals[12],264.);
        r.ledgers[0].totals[12]+=r.ledgers[0].entry_refinement_delta;
        assert!(validate_phase2_work(&r,&s,0.2).is_err());
        let (mut r,s)=fixture();r.ledgers[0].entry_refinement_delta=0.01;
        assert!(validate_phase2_work(&r,&s,0.2).is_err());
    }
    #[test]
    fn altered_work_total_refuses() {
        let (mut r, s) = fixture();
        r.ledgers[3].totals[12] += 0.01;
        assert!(validate_phase2_work(&r, &s, 0.2).is_err());
    }
    #[test]
    fn changed_halo_in_fixed_boundary_solve_refuses() {
        let (mut r, s) = fixture();
        r.vertices[192].phi_same_halo = 3.1;
        assert!(validate_phase2_work(&r, &s, 0.2).is_err());
    }
    #[test]
    fn stale_generation_refuses() {
        let (mut r, s) = fixture();
        r.ledgers[2].preparation_generation = 2;
        assert!(validate_phase2_work(&r, &s, 0.2).is_err());
    }
    #[test]
    fn unproduced_edge_slot_refuses() {
        let (mut r, s) = fixture();
        r.edges[192].u = 1;
        assert!(validate_phase2_work(&r, &s, 0.2).is_err());
    }
    #[test]
    fn changed_physical_correspondence_refuses() {
        let (mut r, s) = fixture();
        r.vertices[256].packed_atom = 7;
        assert!(validate_phase2_work(&r, &s, 0.2).is_err());
    }
    #[test]
    fn nonfinite_primitive_refuses() {
        let (mut r, s) = fixture();
        r.vertices[13].psi[2] = f32::NAN;
        assert!(validate_phase2_work(&r, &s, 0.2).is_err());
    }
    #[test]
    fn terminal_attempt_is_not_accepted_journal() {
        let (mut r, s) = fixture();
        r.current_attempt = true;
        assert!(validate_phase2_work(&r, &s, 0.2).is_err());
    }
    #[test]
    fn earlier_measurement_does_not_fake_new_body() {
        let (mut r, s) = fixture();
        r.ledgers[7].completed_measurements = 0;
        assert!(validate_phase2_work(&r, &s, 0.2).is_err());
    }
    #[test]
    fn cold_initialization_preserves_immutable_population() {
        let a = vec![0u8; 8 * BIPRISM_TILE_BYTES];
        let mut b = a.clone();
        b[100000] = 1;
        assert!(validate_phase2_initial_identity(&a, &b).is_ok());
        b[0] = 1;
        assert!(validate_phase2_initial_identity(&a, &b).is_err());
    }
    #[test]
    fn transfer_error_combinations_are_retained() {
        assert_eq!(completed(Ok(3), Ok(())), Ok(3));
        assert_eq!(
            completed::<u8>(Err("copy".into()), Ok(())),
            Err("copy".into())
        );
        assert!(completed(Ok(3), Err("drain".into()))
            .unwrap_err()
            .contains("drain"));
        let e = completed::<u8>(Err("copy".into()), Err("drain".into())).unwrap_err();
        assert!(e.contains("copy") && e.contains("drain"));
    }
}
