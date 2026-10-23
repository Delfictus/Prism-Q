//! Explicit Biprism physical-leaf adapter over the actual fused-engine owners.
//!
//! This module is a child of fused_engine so every launch borrows the real
//! native planes. It does not run an alternative force model or add a second
//! clock. The caller owns graph capture/replay, all-leaf acceptance and core
//! AU propagation/scatter. Keep this adapter alive until its graphs retire.

use super::*;
use std::sync::{Mutex, MutexGuard};
use crate::biprism_coordinates::{BiprismCoordinateOwnerFfi, BiprismCoordinateIdentity,
    COORDINATE_MAGIC, COORDINATE_VERSION, NATIVE_ANGSTROM, represented_positions};

/// All lows are physical state. The separate mesh reference must not replace
/// the native CA restraint reference. Legacy engines never allocate this owner.
pub(super) struct BiprismNativeCoordinateStorage {
    pub(super) current_low: CudaSlice<f32>,
    pub(super) step_start_low: CudaSlice<f32>,
    pub(super) reference_high: CudaSlice<f32>,
    pub(super) reference_low: CudaSlice<f32>,
    pub(super) descriptor: CudaSlice<BiprismCoordinateOwnerFfi>,
    pub(super) identity: BiprismCoordinateIdentity,
    pub(super) fault_sink: u64,
    pub(super) native_owner_array: u64,
}

fn coordinate_download<T:cudarc::driver::DeviceRepr+Default+Clone>(stream: &Arc<CudaStream>, plane: &CudaSlice<T>) -> Result<Vec<T>> {
    let mut out = vec![T::default(); plane.len()];
    let copied = stream.memcpy_dtoh(plane, &mut out);
    let drained = stream.synchronize();
    copied?; drained?; Ok(out)
}

pub const BIPRISM_CORE_ATOMS: usize = 192;
const CLUSTER_BYTES: usize = 40;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BiprismConstraintPolicy {
    /// Every component touching the core must be wholly in the192 core IDs.
    /// Those whole components are excluded from AMBER projection and coverage.
    FlexibleCore,
    /// Core OU/kicks/drift remain external, but existing constraints may project
    /// them. Such projection changes the accepted core state and needs work/energy
    /// accounting by the caller; no pre-projection energy claim is retained here.
    ConstrainedCore,
}

#[derive(Clone, Debug, Serialize)]
pub struct BiprismOwnershipPlan {
    pub n_atoms: usize,
    pub policy: BiprismConstraintPolicy,
    pub core_atom_ids: Vec<u32>,
    pub ownership: Vec<u8>,
    pub original_constraint_count: usize,
    pub retained_original_constraint_ids: Vec<usize>,
    pub excluded_original_constraint_ids: Vec<usize>,
    #[serde(skip)]
    pub retained_cluster_bytes: Vec<u8>,
}

impl BiprismOwnershipPlan {
    /// Decode the actual native40B cluster bytes, preserving every retained row.
    /// No topology, membership, masses, labels or constraints are reconstructed.
    pub fn from_native_clusters(n_atoms: usize, core: &[u32], raw: &[u8],
        n_clusters: usize, policy: BiprismConstraintPolicy) -> Result<Self> {
        anyhow::ensure!(n_atoms >= BIPRISM_CORE_ATOMS && n_atoms <= i32::MAX as usize,
            "Biprism atom population is outside the native ABI");
        anyhow::ensure!(core.len() == BIPRISM_CORE_ATOMS,
            "Biprism motion owner requires exactly192 selected core IDs");
        let bytes = n_clusters.checked_mul(CLUSTER_BYTES).context("constraint extent overflow")?;
        anyhow::ensure!(raw.len() >= bytes && n_clusters <= i32::MAX as usize,
            "actual constraint byte population is incomplete");
        let mut selected = vec![false; n_atoms];
        for &atom in core {
            let i=atom as usize;
            anyhow::ensure!(i<n_atoms && !selected[i], "core atom IDs are duplicate or out of range");
            selected[i]=true;
        }
        let mut used=HashSet::new();
        let mut retained=Vec::new();let mut excluded=Vec::new();let mut kept=Vec::new();
        for (id,row) in raw[..bytes].chunks_exact(CLUSTER_BYTES).enumerate() {
            let i32_at=|p:usize|i32::from_ne_bytes(row[p..p+4].try_into().unwrap());
            let f32_at=|p:usize|f32::from_bits(u32::from_ne_bytes(row[p..p+4].try_into().unwrap()));
            let nh=i32_at(28);
            anyhow::ensure!((1..=3).contains(&nh), "constraint{id} has invalid cardinality");
            anyhow::ensure!(f32_at(32).is_finite() && f32_at(32)>0.0 &&
                f32_at(36).is_finite() && f32_at(36)>0.0, "constraint{id} has invalid inverse masses");
            let central=i32_at(0);
            anyhow::ensure!(central>=0 && (central as usize)<n_atoms,
                "constraint{id} central atom is out of range");
            let mut members=vec![central as usize];
            for h in 0..nh as usize {
                let atom=i32_at(4+4*h);let bond=f32_at(16+4*h);
                anyhow::ensure!(atom>=0 && (atom as usize)<n_atoms && bond.is_finite() && bond>0.0,
                    "constraint{id} active hydrogen/bond is malformed");
                members.push(atom as usize);
            }
            for &atom in &members {
                anyhow::ensure!(used.insert(atom),"native constraint components overlap at atom{atom}");
            }
            let inside=members.iter().filter(|&&i|selected[i]).count();
            if policy==BiprismConstraintPolicy::FlexibleCore && inside>0 {
                anyhow::ensure!(inside==members.len(),
                    "FlexibleCore selection cuts constraint{id}; select a complete component without expanding the192-ID owner implicitly");
                excluded.push(id);
            } else {
                retained.push(id);kept.extend_from_slice(row);
            }
        }
        let mut ownership=vec![0u8;n_atoms];
        let mask=if policy==BiprismConstraintPolicy::FlexibleCore {3} else {1};
        for &atom in core {ownership[atom as usize]=mask;}
        Ok(Self{n_atoms,policy,core_atom_ids:core.to_vec(),ownership,
            original_constraint_count:n_clusters,retained_original_constraint_ids:retained,
            excluded_original_constraint_ids:excluded,retained_cluster_bytes:kept})
    }
    pub fn retained_constraint_count(&self)->usize {self.retained_original_constraint_ids.len()}
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[repr(i32)]
pub enum BiprismAmberPhase {
    ForceN=0, OuHalfN=10, RattleAfterOuN=11, KickN=12,
    RattlePreDrift=2, DriftSaveXn=3, ShakePosition=4,
    ForceNp1=5, KickNp1=13, RattleBeforeOuNp1=14, OuHalfNp1=15,
}
pub const BIPRISM_AMBER_PHASES:[BiprismAmberPhase;11]=[
    BiprismAmberPhase::ForceN,BiprismAmberPhase::OuHalfN,
    BiprismAmberPhase::RattleAfterOuN,BiprismAmberPhase::KickN,
    BiprismAmberPhase::RattlePreDrift,BiprismAmberPhase::DriftSaveXn,
    BiprismAmberPhase::ShakePosition,BiprismAmberPhase::ForceNp1,
    BiprismAmberPhase::KickNp1,BiprismAmberPhase::RattleBeforeOuNp1,
    BiprismAmberPhase::OuHalfNp1,
];

/// Pointer observations do not transfer allocation lifetime or prove execution.
#[derive(Clone, Debug, Serialize)]
pub struct BiprismNativeLeafPointers {
    pub n_atoms:usize, pub positions:u64, pub velocities:u64, pub forces:u64,
    pub masses:u64, pub charges:u64, pub residue_ids:u64, pub atom_types:u64,
    pub warp_matrix:u64, pub aromatic_centroids:u64, pub is_excited:u64,
    pub electronic_population:u64, pub atom_to_aromatic:u64,
    pub exclusion_field:u64, pub water_density:u64, pub water_density_prev:u64,
    pub grid_dim:usize, pub grid_origin:[f32;3], pub grid_spacing:f32, pub n_aromatics:usize,
    pub positions_step_start:u64, pub potential_energy:u64, pub external_work:u64, pub external_work_window:u64,
    pub protocol:u64, pub protocol_current_step:u64, pub protocol_dt:u64,
    pub ownership:u64, pub retained_constraints:u64,
    pub neighbor_list:u64, pub neighbor_counts:u64,
    pub voxel_hit_grid:u64, pub coupled_spike_grid:u64,
    pub primary_residue_id:u64, pub primary_residue_count:u64,
    pub total_voxels:usize, pub n_residues:usize,
    pub coordinate_owner: BiprismCoordinateOwnerFfi,
    pub coordinate_owner_device: u64,
}


/// Read-only native endpoint owner identities. Pointers retain their existing
/// ABI and lifetimes; zero explicitly denotes an unbound optional owner.
#[repr(C,align(16))]
#[derive(Clone,Copy,Debug,Serialize)]
pub struct BiprismNativeEndpointOwnersFfi {
    pub governor:u64, pub physical_guard:u64, pub asc:u64,
    pub governor_generation:u64, pub initial_physical_epoch:u64,
    pub asc_generation:u64, pub initial_asc_epoch:u64, pub protocol_dt:u64,
    pub source:u32, pub asc_source:u32, pub retained_clusters:u32, pub reserved:u32,
}
unsafe impl cudarc::driver::DeviceRepr for BiprismNativeEndpointOwnersFfi {}
const _:()=assert!(core::mem::size_of::<BiprismNativeEndpointOwnersFfi>()==80);

impl NhsAmberFusedEngine {
    fn enable_biprism_coordinate_precision(&mut self, leaf: u32) -> Result<()> {
        anyhow::ensure!(leaf < 8 && self.biprism_coordinates.is_none(), "coordinate owner must bind once to an eight-leaf identity");
        self.stream.synchronize()?;
        let n3 = self.n_atoms.checked_mul(3).context("coordinate extent overflow")?;
        let current = coordinate_download(&self.stream, &self.d_positions)?;
        anyhow::ensure!(current.len() == n3 && current.iter().all(|v|v.is_finite()), "cold coordinate population invalid");
        let current_low = self.stream.alloc_zeros(n3)?;
        let step_start_low = self.stream.alloc_zeros(n3)?;
        let mut reference_high = self.stream.alloc_zeros(n3)?;
        let reference_low = self.stream.alloc_zeros(n3)?;
        let copied = self.stream.memcpy_htod(&current, &mut reference_high);
        let drained = self.stream.synchronize(); copied?; drained?;
        let s = &self.stream;
        let value = BiprismCoordinateOwnerFfi { magic: COORDINATE_MAGIC, version: COORDINATE_VERSION,
            leaf_id: leaf, atom_count: self.n_atoms as u32, generation: 1,
            unit_frame: NATIVE_ANGSTROM, initialized_mask: 7,
            current_hi: self.d_positions.device_ptr(s).0, current_lo: current_low.device_ptr(s).0,
            step_start_hi: self.d_positions_step_start.device_ptr(s).0, step_start_lo: step_start_low.device_ptr(s).0,
            reference_hi: reference_high.device_ptr(s).0, reference_lo: reference_low.device_ptr(s).0, reserved: [0; 2] };
        value.validate(leaf, self.n_atoms as u32, NATIVE_ANGSTROM, value.current_hi).map_err(anyhow::Error::msg)?;
        let retained = [value];
        let mut descriptor = unsafe{s.alloc(1)}?;
        let copied = s.memcpy_htod(&retained, &mut descriptor);
        let drained = s.synchronize(); copied?; drained?;
        self.biprism_coordinates = Some(BiprismNativeCoordinateStorage { current_low, step_start_low,
            reference_high, reference_low, descriptor, identity: value.into(), fault_sink: 0, native_owner_array: 0 });
        Ok(())
    }
    pub fn biprism_coordinate_descriptor(&self) -> Result<BiprismCoordinateOwnerFfi> {
        let c = self.biprism_coordinates.as_ref().context("Biprism precision authority absent")?;
        let s = &self.stream;
        let value = BiprismCoordinateOwnerFfi { magic: COORDINATE_MAGIC, version: c.identity.version,
            leaf_id: c.identity.leaf_id, atom_count: c.identity.atom_count, generation: c.identity.generation,
            unit_frame: NATIVE_ANGSTROM, initialized_mask: 7,
            current_hi: self.d_positions.device_ptr(s).0, current_lo: c.current_low.device_ptr(s).0,
            step_start_hi: self.d_positions_step_start.device_ptr(s).0, step_start_lo: c.step_start_low.device_ptr(s).0,
            reference_hi: c.reference_high.device_ptr(s).0, reference_lo: c.reference_low.device_ptr(s).0, reserved: [0;2] };
        value.validate(c.identity.leaf_id, self.n_atoms as u32, NATIVE_ANGSTROM, value.current_hi).map_err(anyhow::Error::msg)?;
        Ok(value)
    }
    /// Retain both represented components; callers may derive an f64 sum, but
    /// must retain these bytes for same-build continuation and precise outputs.
    pub fn get_biprism_coordinate_split(&self) -> Result<(Vec<f32>, Vec<f32>)> {
        let c = self.biprism_coordinates.as_ref().context("Biprism precision authority absent")?;
        let hi = coordinate_download(&self.stream, &self.d_positions)?;
        let lo = coordinate_download(&self.stream, &c.current_low)?;
        represented_positions(&hi, &lo).map_err(anyhow::Error::msg)?;
        Ok((hi, lo))
    }
    pub fn get_biprism_coordinate_snapshot(&self)->Result<crate::biprism_coordinates::BiprismCoordinateSnapshot>{
        let c=self.biprism_coordinates.as_ref().context("Biprism coordinate authority absent")?;
        let image=crate::biprism_coordinates::BiprismCoordinateSnapshot{identity:c.identity,
            current_high:coordinate_download(&self.stream,&self.d_positions)?,current_low:coordinate_download(&self.stream,&c.current_low)?,
            step_start_high:coordinate_download(&self.stream,&self.d_positions_step_start)?,step_start_low:coordinate_download(&self.stream,&c.step_start_low)?,
            reference_high:coordinate_download(&self.stream,&c.reference_high)?,reference_low:coordinate_download(&self.stream,&c.reference_low)?};
        image.validate().map_err(anyhow::Error::msg)?;Ok(image)
    }
    pub fn get_biprism_positions_f64(&self) -> Result<Vec<f64>> {
        let (hi,lo) = self.get_biprism_coordinate_split()?;
        represented_positions(&hi,&lo).map_err(anyhow::Error::msg)
    }
    /// The descriptor is one leaf, not an eight-leaf array. The sink belongs to
    /// the retained V10 convergence owner and is bound before any capture.
    pub(super) fn biprism_coordinate_binding(&self) -> Result<Option<(u64,u64)>> {
        self.biprism_coordinates.as_ref().map(|c| {
            anyhow::ensure!(c.fault_sink != 0 && c.fault_sink % 4 == 0, "coordinate fault sink not bound");
            Ok((c.descriptor.device_ptr(&self.stream).0, c.fault_sink))
        }).transpose()
    }
    /// ASC's additive ABI indexes the retained eight-descriptor native array.
    pub(super) fn biprism_coordinate_array_binding(&self) -> Result<Option<(u64,u32,u64)>> {
        self.biprism_coordinates.as_ref().map(|c| {
            anyhow::ensure!(c.native_owner_array!=0 && c.native_owner_array%16==0 && c.fault_sink!=0,
                "coordinate array/fault owner not bound");
            Ok((c.native_owner_array,c.identity.leaf_id,c.fault_sink))
        }).transpose()
    }
    /// Download the actual native N*1024 neighbor rows and N counts for cold
    /// selection. No fabricated core IDs or independently rebuilt neighborhoods.
    pub fn download_native_neighbor_population(&self)->Result<(Vec<i32>,Vec<i32>)> {
        anyhow::ensure!(self.use_neighbor_list,"native neighbor owner is not selected");
        let rows=coordinate_download(&self.stream,&self.d_neighbor_list)?;
        let counts=coordinate_download(&self.stream,&self.d_n_neighbors)?;
        anyhow::ensure!(rows.len()==self.n_atoms.checked_mul(1024).context("neighbor extent overflow")?
            && counts.len()==self.n_atoms,"native neighbor population extent mismatch");
        Ok((rows,counts))
    }
}

/// Allocation/stream owner for an explicitly selected new physical profile.
/// Other engine stepping must not run concurrently with this adapter. Its
/// host cursor validates enqueue order only; it never claims GPU completion.
pub struct BiprismPhysicalLeafAdapter {
    engine:Arc<Mutex<NhsAmberFusedEngine>>,
    stream:Arc<CudaStream>,
    plan:BiprismOwnershipPlan,
    d_ownership:CudaSlice<u8>,
    d_retained_clusters:CudaSlice<u8>,
    d_bath_ca_mask:CudaSlice<i32>,
    phase_kernels:Vec<(i32,CudaFunction,&'static str)>,
    rattle_kernel:CudaFunction,
    clamp_kernel:CudaFunction,
    uv_charge_kernel:CudaFunction,
    ca_begin_kernel:CudaFunction,
    ca_asc_kernel:CudaFunction,
    nma_compose_kernel:CudaFunction,
    aromatic_geometry_kernel:CudaFunction,
    director_kernel:CudaFunction,
    collective_director_kernel:CudaFunction,
    selected_director_kernel:CudaFunction,
    selected_request_pending:bool,
    governor:Option<Arc<crate::governor_device::GovernorDevice>>,
    // None=outside body; Some(0..11)=next physical phase,11=endpoint.
    enqueue_cursor:Option<usize>,
}

fn lock_engine(owner:&Arc<Mutex<NhsAmberFusedEngine>>)->Result<MutexGuard<'_,NhsAmberFusedEngine>> {
    owner.lock().map_err(|_|anyhow::anyhow!("Biprism native engine owner mutex poisoned"))
}

impl BiprismPhysicalLeafAdapter {
    /// Call quiescently before capture; it reads existing cluster bytes once and
    /// allocates stable policy planes. It never changes the engine's originals.
    pub fn bind(engine:Arc<Mutex<NhsAmberFusedEngine>>,core_atom_ids:&[u32],
        leaf_id:u32,
        policy:BiprismConstraintPolicy)->Result<Self> {
        let mut e=lock_engine(&engine)?;
        Self::check_profile(&e)?;
        e.enable_biprism_coordinate_precision(leaf_id)?;
        let stream=e.stream.clone();
        let raw=coordinate_download(&stream,&e.d_h_clusters)?;
        let plan=BiprismOwnershipPlan::from_native_clusters(e.n_atoms,core_atom_ids,&raw,e.n_clusters,policy)?;
        let governor=e.governor_device.clone();
        if let Some(g)=&governor {
            anyhow::ensure!(g.owns_stream(&stream) && g.n_constraint_clusters()==plan.retained_constraint_count(),
                "existing governor expects original constraints; bind a retained-population owner explicitly before this new profile");
        }
        let mut d_ownership=stream.alloc_zeros::<u8>(plan.n_atoms)?;
        let copy=stream.memcpy_htod(&plan.ownership,&mut d_ownership);
        let drain=stream.synchronize();copy?;drain?;
        let mut d_retained_clusters=stream.alloc_zeros::<u8>(plan.retained_cluster_bytes.len().max(CLUSTER_BYTES))?;
        if !plan.retained_cluster_bytes.is_empty() {
            let copy=stream.memcpy_htod(&plan.retained_cluster_bytes,&mut d_retained_clusters);
            let drain=stream.synchronize();copy?;drain?;
        }
        anyhow::ensure!(e.ca_mask_cpu.len()==e.n_atoms,"native CA mask population differs");
        let mut bath_ca=e.ca_mask_cpu.clone();
        for &i in core_atom_ids {bath_ca[i as usize]=0;}
        let mut d_bath_ca_mask=stream.alloc_zeros::<i32>(e.n_atoms)?;
        let copy=stream.memcpy_htod(&bath_ca,&mut d_bath_ca_mask);
        let drain=stream.synchronize();copy?;drain?;
        let mut phase_kernels=Vec::new();
        for &(phase,name) in &[
            (0,"nhs_biprism_amber_phase_force_n_compensated_v1"),
            (5,"nhs_biprism_amber_phase_force_np1_compensated_v1"),
            (10,"nhs_biprism_amber_phase_ou_half_n_compensated_v1"),
            (11,"nhs_biprism_amber_phase_rattle_after_ou_n_compensated_v1"),
            (12,"nhs_biprism_amber_phase_kick_n_compensated_v1"),
            (2,"nhs_biprism_amber_phase_rattle_pre_drift_compensated_v1"),
            (3,"nhs_biprism_amber_phase_drift_save_xn_compensated_v1"),
            (4,"nhs_biprism_amber_phase_shake_position_compensated_v1"),
            (13,"nhs_biprism_amber_phase_kick_np1_compensated_v1"),
            (14,"nhs_biprism_amber_phase_rattle_before_ou_np1_compensated_v1"),
            (15,"nhs_biprism_amber_phase_ou_half_np1_compensated_v1"),
        ] {phase_kernels.push((phase,e._fused_module.load_function(name)?,name));}
        let director_ptx=crate::twin_kernels::find_twin_ptx("protocol_director.ptx")?;
        let director_module=e.context.load_module(cudarc::nvrtc::Ptx::from_file(&director_ptx))?;
        let director_kernel=director_module.load_function("protocol_director")?;
        let collective_director_kernel=director_module.load_function("protocol_director_biprism_v1")?;
        let rattle_kernel=e._fused_module.load_function("nhs_biprism_rattle_velocity_constraints_compensated_v1")?;
        let clamp_kernel=e._fused_module.load_function("nhs_biprism_velocity_safety_clamp_v1")?;
        let selected_director_kernel=director_module.load_function("protocol_director_biprism_selected_v1")?;
        let uv_charge_kernel=e._fused_module.load_function("nhs_uv_charge_parameter_work_compensated_v1")?;
        let ca_begin_kernel=e._fused_module.load_function("nhs_ca_intervention_begin_compensated_v1")?;
        let ca_asc_kernel=e._fused_module.load_function("nhs_ca_intervention_asc_endpoint_compensated_v1")?;
        let nma_compose_kernel=e._fused_module.load_function("nhs_nma_compose_force_compensated_v1")?;
        let aromatic_geometry_kernel=e._fused_module.load_function("nhs_refresh_aromatic_geometry_compensated_v1")?;
        stream.synchronize()?;
        drop(e);
        Ok(Self{engine,stream,plan,d_ownership,d_retained_clusters,d_bath_ca_mask,
            phase_kernels,rattle_kernel,clamp_kernel,uv_charge_kernel,ca_begin_kernel,ca_asc_kernel,nma_compose_kernel,aromatic_geometry_kernel,director_kernel,collective_director_kernel,selected_director_kernel,selected_request_pending:false,governor,enqueue_cursor:None})
    }
    fn check_profile(e:&NhsAmberFusedEngine)->Result<()> {
        anyhow::ensure!(!e.historical_may_literal_execution && e.exact_nonbonded_exceptions_enabled>=0,
            "Biprism adapter requires the ordinary418.4 AMBER convention, not historical May");
        anyhow::ensure!(e.ultimate_engine.is_none() && e.replica_states.is_empty() &&
            e.lambda_ncmc_reversible_guard_receipt.is_none(),"conflicting physical engine owner");
        anyhow::ensure!(e.fused_inner_steps==1 && e.use_neighbor_list,
            "Biprism adapter requires one physical Director step and native deterministic neighbor forces");
        Ok(())
    }
    pub fn stream(&self)->Arc<CudaStream> {self.stream.clone()}
    pub fn engine_owner(&self)->Arc<Mutex<NhsAmberFusedEngine>> {self.engine.clone()}
    pub(crate) fn bind_precision_fault_sink(&mut self, sink:u64,native_array:u64)->Result<()> {
        anyhow::ensure!(sink!=0 && sink%4==0 && self.enqueue_cursor.is_none(),"invalid precision fault binding");
        let mut e=lock_engine(&self.engine)?;
        let c=e.biprism_coordinates.as_mut().context("coordinate authority absent")?;
        anyhow::ensure!(c.fault_sink==0 || c.fault_sink==sink,"coordinate fault owner cannot be rebound");
        anyhow::ensure!(native_array!=0 && native_array%16==0 && (c.native_owner_array==0 || c.native_owner_array==native_array),"coordinate array owner cannot be rebound");
        c.fault_sink=sink;c.native_owner_array=native_array;Ok(())
    }
    pub fn ownership_plan(&self)->&BiprismOwnershipPlan {&self.plan}
    /// Capture-time source identities for the collective endpoint guard.
    pub fn native_endpoint_owners(&self)->Result<BiprismNativeEndpointOwnersFfi> {
        let e=lock_engine(&self.engine)?;
        let mut out=BiprismNativeEndpointOwnersFfi{governor:0,physical_guard:0,asc:0,
            governor_generation:0,initial_physical_epoch:0,asc_generation:0,initial_asc_epoch:0,
            protocol_dt:e.d_protocol_dt_dev_ptr(&self.stream),source:0,asc_source:0,
            retained_clusters:u32::try_from(self.plan.retained_constraint_count())?,reserved:0};
        if let Some(g)=&self.governor {
            let (snapshot,_)=g.snapshot().map_err(|rc|anyhow::anyhow!("native governor snapshot rc={rc}"))?;
            out.governor=g.device_ptr();out.physical_guard=g.physical_guard_device_ptr();
            out.governor_generation=g.generation();out.source=g.source_stream_id();
            out.initial_physical_epoch=snapshot.last_physical_epoch;
        }
        if let Some(a)=&e.asc_device {
            let snapshot=a.snapshot().map_err(|rc|anyhow::anyhow!("native ASC snapshot rc={rc}"))?;
            anyhow::ensure!(snapshot.physical_epoch==snapshot.completed_epoch && snapshot.coverage_failures==0,
                "native ASC is not a quiescent accepted source");
            out.asc=a.device_ptr();out.asc_generation=snapshot.generation;out.asc_source=snapshot.source;
            out.initial_asc_epoch=snapshot.completed_epoch;
        }
        Ok(out)
    }
    /// Actual configured owners plus executed hooks; absence is explicit and
    /// cannot be presented as execution of an enabled optional feature.
    pub fn feature_execution_map(&self)->Result<serde_json::Value> {
        let e=lock_engine(&self.engine)?;
        Ok(serde_json::json!({"profile":"biprism_lagged_split", "all_atom_native_force_n_and_np1":true,
            "bath_ou_rng_kicks_drift":true,"core_motion_owner":"v10_atomic_unit_planes",
            "native_protocol_uv_charge_work":true,"native_constraint_work_and_endpoint_rattle":true,
            "native_bath_ca_intervention":true,"retained_constraint_count":self.plan.retained_constraint_count(),
            "excluded_original_constraint_ids":self.plan.excluded_original_constraint_ids,
            "nma_enabled":e.nma_enabled,"asc_bound":e.asc_device.is_some(),
            "governor_bound":self.governor.is_some(),"physical_guard_bound":self.governor.as_ref().is_some_and(|g|g.has_physical_guard()),
            "observer_clock_bound":e.observer_physical_clock_bound(),"lambda_slots":e.n_lambda_slots,
            "lambda_dual_endpoint_enabled":e.lambda_dual_endpoint_enabled,"pme_bound":e.lambda_pme.is_some(),
            "force_composition_hooks":["lambda_dual_endpoint","lambda_pme","nma","asc"],
            "legacy_multilif_and_coupling":"replaced_by_typed_v10_lif_native_density_companion",
            "active_asc_core_consumption":"explicit_mixed_v10_core_rows_and_native_bath_kicks_required; producer_decision_publication_is_a_separate_bound_surface"}))
    }
    /// This owner must already bind the exact retained population and native
    /// physical arrays. Observer-only clock and governor are mutually exclusive.
    pub fn bind_retained_governor(&mut self,g:Arc<crate::governor_device::GovernorDevice>)->Result<()> {
        let e=lock_engine(&self.engine)?;
        anyhow::ensure!(self.enqueue_cursor.is_none() && self.governor.is_none() &&
            e.governor_device.is_none() && !e.observer_physical_clock_bound(),
            "governor binding is not an idle exclusive retained-population owner");
        anyhow::ensure!(g.owns_stream(&self.stream) && g.n_constraint_clusters()==self.plan.retained_constraint_count(),
            "retained governor stream/population mismatch");
        self.governor=Some(g);Ok(())
    }
    pub fn pointers(&self)->Result<BiprismNativeLeafPointers> {
        let e=lock_engine(&self.engine)?;let s=&self.stream;
        let pointers=BiprismNativeLeafPointers{n_atoms:e.n_atoms,
            positions:e.d_positions.device_ptr(s).0,velocities:e.d_velocities.device_ptr(s).0,
            forces:e.d_forces.device_ptr(s).0,masses:e.d_masses.device_ptr(s).0,
            charges:e.d_charges.device_ptr(s).0,residue_ids:e.d_residue_ids.device_ptr(s).0,
            atom_types:e.d_atom_types.device_ptr(s).0,
            warp_matrix:e.d_warp_matrix.as_ref().context("native immutable warp owner absent")?.as_ref().device_ptr(s).0,
            aromatic_centroids:e.d_aromatic_centroids.device_ptr(s).0,
            is_excited:e.d_is_excited.device_ptr(s).0,electronic_population:e.d_electronic_population.device_ptr(s).0,
            atom_to_aromatic:e.d_atom_to_aromatic.device_ptr(s).0,
            exclusion_field:e.d_exclusion_field.device_ptr(s).0,water_density:e.d_water_density.device_ptr(s).0,
            water_density_prev:e.d_water_density_prev.device_ptr(s).0,
            grid_dim:e.grid_dim,grid_origin:e.grid_origin,grid_spacing:e.grid_spacing,n_aromatics:e.n_aromatics,
            positions_step_start:e.d_positions_step_start.device_ptr(s).0,
            potential_energy:e.d_potential_energy_components.device_ptr(s).0,
            external_work:e.d_external_work_buffer.device_ptr(s).0,
            external_work_window:e.d_external_work_window.device_ptr(s).0,
            protocol:e.d_protocol_state.device_ptr(s).0,
            protocol_current_step:e.d_protocol_current_step_dev_ptr(s),protocol_dt:e.d_protocol_dt_dev_ptr(s),
            ownership:self.d_ownership.device_ptr(s).0,retained_constraints:self.d_retained_clusters.device_ptr(s).0,
            neighbor_list:e.d_neighbor_list.device_ptr(s).0,neighbor_counts:e.d_n_neighbors.device_ptr(s).0,
            voxel_hit_grid:e.d_voxel_hit_grid.device_ptr(s).0,coupled_spike_grid:e.d_coupled_spike_grid.device_ptr(s).0,
            primary_residue_id:e.d_primary_residue_id.device_ptr(s).0,
            primary_residue_count:e.d_primary_residue_count.device_ptr(s).0,
            total_voxels:e.grid_dim*e.grid_dim*e.grid_dim,n_residues:e.n_residues,
            coordinate_owner:e.biprism_coordinate_descriptor()?,
            coordinate_owner_device:e.biprism_coordinates.as_ref().context("coordinate authority absent")?.descriptor.device_ptr(s).0};
        Ok(pointers)
    }
    /// Cold exact native control snapshot for the retained request owner.
    pub fn protocol_snapshot(&self)->Result<crate::protocol_state::ProtocolState>{
        let e=lock_engine(&self.engine)?;
        let mut bytes=vec![0u8;core::mem::size_of::<crate::protocol_state::ProtocolState>()];
        anyhow::ensure!(e.d_protocol_state.len()==bytes.len(),"native ProtocolState extent");
        let copy=self.stream.memcpy_dtoh(&e.d_protocol_state,&mut bytes);
        let drain=self.stream.synchronize();
        match(copy,drain){(Ok(()),Ok(()))=>{},(Err(a),Ok(()))=>anyhow::bail!("protocol copy: {a}"),
            (Ok(()),Err(b))=>anyhow::bail!("protocol completion: {b}"),
            (Err(a),Err(b))=>anyhow::bail!("protocol copy: {a}; completion: {b}")}
        Ok(unsafe{core::ptr::read_unaligned(bytes.as_ptr().cast::<crate::protocol_state::ProtocolState>())})
    }
    /// All eight requests must finish before the shared minimum is published.
    pub(crate) unsafe fn enqueue_selected_request(&mut self,scheduled:u64,b:crate::biprism_shared_step::BiprismSelectedRequestBinding)->Result<()>{
        anyhow::ensure!(self.enqueue_cursor.is_none()&&!self.selected_request_pending&&scheduled!=0&&scheduled%4==0,
            "selected request is not at an idle boundary");
        let owner=self.engine.clone();let mut e=lock_engine(&owner)?;Self::check_profile(&e)?;
        anyhow::ensure!(self.governor.is_some()&&!e.observer_physical_clock_bound(),"selected physical time requires actual native guard");
        e.rebuild_neighbor_lists_graph_safe(&self.stream)?;
        self.stream.launch_builder(&self.selected_director_kernel).arg(&mut e.d_protocol_state).arg(&scheduled)
            .arg(&b.terminal).arg(&b.baseline).arg(&b.original_fixed_request_ps).arg(&b.leaf).arg(&b.shared)
            .launch(LaunchConfig{grid_dim:(1,1,1),block_dim:(1,1,1),shared_mem_bytes:0})?;
        self.selected_request_pending=true;Ok(())
    }
    /// Runs only after collective selection wrote the same represented dt to
    /// every ProtocolState. No second Director evaluation or counter increment.
    pub(crate) unsafe fn enqueue_selected_physical_begin(&mut self,b:crate::biprism_shared_step::BiprismSelectedRequestBinding)->Result<()>{
        anyhow::ensure!(self.enqueue_cursor.is_none()&&self.selected_request_pending,"physical begin precedes selected request");
        let owner=self.engine.clone();let mut e=lock_engine(&owner)?;Self::check_profile(&e)?;let s=&self.stream;
        s.memset_zeros(&mut e.d_potential_energy_components)?;s.memset_zeros(&mut e.d_external_work_buffer)?;
        self.governor.as_ref().context("native selected governor absent")?.begin_selected_step(s,b.shared,b.generation,b.leaf)
            .map_err(|rc|anyhow::anyhow!("selected native governor begin rc={rc}"))?;
        if let Some(a)=&e.asc_device{a.begin_step(s).map_err(|rc|anyhow::anyhow!("Biprism ASC begin rc={rc}"))?;}
        e.enqueue_observer_physical_clock_begin(s)?;e.begin_conservative_constraint_step(s)?;
        e.observer_physical_clock_capture_seen=true;self.selected_request_pending=false;self.enqueue_cursor=Some(0);Ok(())
    }
    /// Enqueue the sole existing Director plus original accumulator/owner begins.
    /// Safety: caller holds exclusive physical graph/engine use, and must retain
    /// all owners until replay retires. Failure is not atomic whole-step rollback.
    pub unsafe fn enqueue_begin(&mut self)->Result<()> {self.enqueue_begin_impl(None)}
    /// Evaluate each native ProtocolState exactly once at the collective scheduled
    /// step. This does not increment that word or the native protocol step. The
    /// collective owner advances its scheduled word only after accepted physics;
    /// native protocol step then identifies the most recently executed body.
    /// Safety: the nonnull aligned pointer must identify a live, stream-ordered
    /// device u32 until every captured replay retires, with one collective writer.
    pub unsafe fn enqueue_begin_at_collective_step(&mut self,scheduled_step:u64)->Result<()> {
        anyhow::ensure!(scheduled_step!=0 && scheduled_step%4==0,"invalid collective scheduled-step pointer");
        self.enqueue_begin_impl(Some(scheduled_step))
    }
    unsafe fn enqueue_begin_impl(&mut self,scheduled_step:Option<u64>)->Result<()> {
        anyhow::ensure!(self.enqueue_cursor.is_none()&&!self.selected_request_pending,"Biprism physical body already open");
        let owner=self.engine.clone();let mut guard=lock_engine(&owner)?;let e=&mut *guard;
        Self::check_profile(e)?;let s=&self.stream;
        e.rebuild_neighbor_lists_graph_safe(s)?;
        let serial=LaunchConfig{grid_dim:(1,1,1),block_dim:(1,1,1),shared_mem_bytes:0};
        if let Some(step)=scheduled_step {
            s.launch_builder(&self.collective_director_kernel).arg(&mut e.d_protocol_state).arg(&step).launch(serial)?;
        } else {
            s.launch_builder(&self.director_kernel).arg(&mut e.d_protocol_state).launch(serial)?;
        }
        s.memset_zeros(&mut e.d_potential_energy_components)?;
        s.memset_zeros(&mut e.d_external_work_buffer)?;
        if let Some(g)=&self.governor {g.begin_step(s).map_err(|rc|anyhow::anyhow!("Biprism governor begin rc={rc}"))?;}
        if let Some(a)=&e.asc_device {a.begin_step(s).map_err(|rc|anyhow::anyhow!("Biprism ASC begin rc={rc}"))?;}
        e.enqueue_observer_physical_clock_begin(s)?;
        e.begin_conservative_constraint_step(s)?;
        e.observer_physical_clock_capture_seen=true;
        e.prismcode_md_record(e.timestep.max(0) as u64,0,"biprism_lagged_split_enqueue",
            if scheduled_step.is_some(){"protocol_director_biprism_v1"}else{"protocol_director"},
            &[pcmd::OP_MD_PROTOCOL_DIRECTOR,pcmd::OP_MD_ACCUMULATOR_ZERO],
            (1,1,1),(1,1,1),0,Some("Existing Director equations; collective variant reads scheduled step without increment"));
        self.enqueue_cursor=Some(0);Ok(())
    }
    /// Phases may be interleaved across leaf adapters. Enqueue accepted core
    /// scatter after DriftSaveXn and before ShakePosition. Cross-leaf events and
    /// all-leaf acceptance belong to the collective controller.
    pub unsafe fn enqueue_phase(&mut self,phase:BiprismAmberPhase)->Result<()> {
        let cursor=self.enqueue_cursor.context("physical body has not begun")?;
        anyhow::ensure!(BIPRISM_AMBER_PHASES.get(cursor)==Some(&phase),"Biprism physical phase out of order");
        let owner=self.engine.clone();let mut guard=lock_engine(&owner)?;let e=&mut *guard;
        self.enqueue_phase_raw(e,phase as i32,false)?;
        let s=&self.stream;
        if matches!(phase,BiprismAmberPhase::ForceN|BiprismAmberPhase::ForceNp1) {
            e.enqueue_lambda_dual_endpoint_hamiltonian()?;e.enqueue_lambda_pme_composition()?;
            e.compose_nma_force_phase(phase as i32,false)?;e.compose_asc_force_phase(phase as i32)?;
        }
        self.enqueue_energy_diagnostic(e,phase as i32)?;
        if phase==BiprismAmberPhase::DriftSaveXn {self.enqueue_isolated_shake_diagnostic(e,30)?;}
        if phase==BiprismAmberPhase::ShakePosition {
            self.enqueue_isolated_shake_diagnostic(e,40)?;
            e.rebuild_neighbor_lists_graph_safe(s)?;
        }
        self.enqueue_cursor=Some(cursor+1);Ok(())
    }

    unsafe fn enqueue_phase_raw(&self,e:&mut NhsAmberFusedEngine,execution_phase:i32,intervention_work:bool)->Result<()> {
        let stream=&self.stream;let physics_cfg=atom_owned_launch_config(e.n_atoms);
        let n_atoms_i32=e.n_atoms as i32;let n_bonds_i32=e.n_bonds as i32;
        let n_angles_i32=e.n_angles as i32;let n_dihedrals_i32=e.n_dihedrals as i32;
        let n_clusters_i32=self.plan.retained_constraint_count() as i32;
        let grid_dim_i32=e.grid_dim as i32;let n_uv_targets_i32=e.n_uv_targets as i32;
        let n_aromatics_i32=e.n_aromatics as i32;let use_neighbor_list=if e.use_neighbor_list{1i32}else{0i32};
        let max_spikes_i32=MAX_SPIKES_PER_STEP as i32;
        let constraint_owner=self.governor.as_ref().map_or(0u64,|g|g.constraint_window_ptr());
        let asc_owner=e.asc_device.as_ref().map_or(0u64,|a|a.device_ptr());
        let (_,kernel,name)=self.phase_kernels.iter().find(|(id,_,_)|*id==execution_phase)
            .context("unbound compensated Biprism phase")?;
        let (phase_kernel,phase_kernel_name)=(kernel.clone(),*name);
        let (coordinate_owner,precision_fault)=e.biprism_coordinate_binding()?.context("compensated phase requires coordinate authority")?;
        let work_sink=if intervention_work {&mut e.d_external_work_buffer} else {match execution_phase {
            MD_PHASE_RATTLE_PRE_DRIFT=>&mut e.d_conservative_constraint_pre,
            MD_PHASE_SHAKE_POSITION=>&mut e.d_conservative_constraint_shake,
            MD_PHASE_RATTLE_BEFORE_OU_NP1=>&mut e.d_conservative_constraint_post,
            _=>&mut e.d_external_work_buffer,
        }};
        let mut launch=stream.launch_builder(&phase_kernel);
        launch
                    .arg(&mut e.d_positions)
                    .arg(&mut e.d_velocities)
                    .arg(&mut e.d_forces)
                    .arg(&e.d_masses)
                    .arg(&e.d_charges)
                    .arg(&e.d_atom_types)
                    .arg(&e.d_residue_ids)
                    .arg(&n_atoms_i32)
                    .arg(&e.d_bonds)
                    .arg(&n_bonds_i32)
                    .arg(&e.d_angles)
                    .arg(&n_angles_i32)
                    .arg(&e.d_dihedrals)
                    .arg(&n_dihedrals_i32)
                    .arg(&e.d_lj_params)
                    .arg(&e.d_exclusion_list)
                    .arg(&e.d_exclusion_offsets)
                    .arg(&e.d_pairs14_list)
                    .arg(&e.d_pairs14_offsets)
                    .arg(&e.d_pairs14_charge_product)
                    .arg(&e.d_pairs14_sigma)
                    .arg(&e.d_pairs14_epsilon)
                    .arg(&e.exact_nonbonded_exceptions_enabled)
                    .arg(&self.d_retained_clusters)
                    .arg(&n_clusters_i32)
                    .arg(&mut e.d_exclusion_field)
                    .arg(&mut e.d_water_density)
                    .arg(&mut e.d_water_density_prev)
                    .arg(&mut e.d_lif_potential)
                    .arg(&mut e.d_spike_grid)
                    .arg(&e.grid_origin[0])
                    .arg(&e.grid_origin[1])
                    .arg(&e.grid_origin[2])
                    .arg(&e.grid_spacing)
                    .arg(&grid_dim_i32)
                    .arg(e.d_warp_matrix.as_ref().expect("initialized immutable warp").as_ref())
                    .arg(&e.d_uv_targets)
                    .arg(&n_uv_targets_i32)
                    .arg(&e.d_protocol_state)
                    .arg(&mut e.d_is_excited)
                    .arg(&mut e.d_time_since_excitation)
                    .arg(&mut e.d_electronic_population)
                    .arg(&mut e.d_vibrational_energy)
                    .arg(&mut e.d_franck_condon_progress)
                    .arg(&e.d_ground_state_charges)
                    .arg(&e.d_atom_to_aromatic)
                    .arg(&e.d_aromatic_type)
                    .arg(&e.d_ring_normals)
                    .arg(&e.d_aromatic_centroids)
                    .arg(&mut e.d_uv_signal_prev)
                    .arg(&e.d_aromatic_neighbors)
                    .arg(&n_aromatics_i32)
                    .arg(&mut e.d_spike_events)
                    .arg(&mut e.d_spike_count)
                    .arg(&max_spikes_i32)
                    .arg(&e.cutoff)
                    .arg(&mut e.d_rng_states)
                    .arg(&e.d_neighbor_list)
                    .arg(&e.d_n_neighbors)
                    .arg(&use_neighbor_list)
                    .arg(&e.d_efp_potential)
                    .arg(&e.d_efp_potential_prev)
                    .arg(&e.d_efp_lif_potential)
                    .arg(&mut e.d_spike_grid_efp)
                    .arg(&e.d_atom_lambda)
                    .arg(&mut e.d_voxel_hit_grid)
                    .arg(&mut e.d_last_uv_step)
                    .arg(&mut e.d_coupled_spike_grid)
                    .arg(&mut e.d_primary_residue_id)
                    .arg(&mut e.d_primary_residue_count)
                    .arg(&mut e.d_residue_step_causal)
                    // M1.2.17 — per-atom potential-energy components.
                    .arg(&mut e.d_potential_energy_components)
                    // M1.2.18.5 — VRAM-native total-external-work scalar.
                    .arg(work_sink)
                    .arg(&e.d_atom_bond_list)
                    .arg(&e.d_atom_bond_count)
                    .arg(&e.d_atom_bond_position)
                    .arg(&e.max_bonds_per_atom)
                    .arg(&e.d_atom_angle_list)
                    .arg(&e.d_atom_angle_count)
                    .arg(&e.d_atom_angle_position)
                    .arg(&e.max_angles_per_atom)
                    .arg(&e.d_atom_dihedral_list)
                    .arg(&e.d_atom_dihedral_count)
                    .arg(&e.d_atom_dihedral_position)
                    .arg(&e.max_dihedrals_per_atom)
                    .arg(&e.d_lambda_atom_slot)
                    .arg(&e.d_lambda_slot_values)
                    .arg(&e.d_lambda_current_charges)
                    .arg(&e.d_lambda_charge_endpoint_a)
                    .arg(&e.d_lambda_charge_endpoint_b)
                    .arg(&e.d_lambda_charge_derivatives)
                    .arg(&e.d_lambda_endpoint_occupancy_masks)
                    .arg(&e.n_lambda_slots)
                    .arg(&e.lambda_alpha_lj)
                    .arg(&e.lambda_alpha_q_a2)
                    .arg(&e.lambda_softcore_power)
                    .arg(&e.lambda_coupling_power)
                    .arg(&e.lambda_v2_ewald_enabled)
                    .arg(&e.lambda_pme_beta_a_inv)
                    .arg(&e.lambda_periodic_box_lengths_a[0])
                    .arg(&e.lambda_periodic_box_lengths_a[1])
                    .arg(&e.lambda_periodic_box_lengths_a[2])
                    .arg(&mut e.d_lambda_dudl_atom)
                    .arg(&mut e.d_lambda_energy_atom)
                    .arg(&mut e.d_positions_step_start)
                    .arg(&execution_phase)
                    .arg(&e.d_aromatic_charge_set_mean)
                    .arg(&constraint_owner)
                    .arg(&asc_owner);

        launch.arg(&self.d_ownership).arg(&coordinate_owner).arg(&precision_fault);
        launch.launch(physics_cfg).context("Biprism physical native phase launch failed")?;
        e.prismcode_md_record(e.timestep.max(0) as u64,execution_phase as u32,
            "biprism_lagged_split_enqueue",phase_kernel_name,amber_specialized_phase_opcodes(execution_phase),
            physics_cfg.grid_dim,physics_cfg.block_dim,physics_cfg.shared_mem_bytes,
            Some("Actual native112 argument compensated body; full force population and explicit masked motion/constraint owner"));
        Ok(())
    }

    /// Finish the actual physical endpoint without the old voxel-LIF/coupling
    /// producer. UV remains an explicitly external full-domain intervention.
    /// Safety: the caller must have queued/accepted core scatter and all-leaf
    /// failure handling; this host cursor proves only enqueue order.
    pub unsafe fn enqueue_endpoint(&mut self)->Result<()> {
        anyhow::ensure!(self.enqueue_cursor==Some(BIPRISM_AMBER_PHASES.len()),"physical phase sequence is incomplete");
        let owner=self.engine.clone();let mut guard=lock_engine(&owner)?;let e=&mut *guard;
        self.commit_constraint_step(e)?;
        self.enqueue_endpoint_raw(e,None,None)?;
        self.enqueue_cursor=None;Ok(())
    }
    /// Explicit new profile ASC owner; the ordinary endpoint retains the old
    /// seal when no mixed binding is supplied. No native core kick is invented.
    pub unsafe fn enqueue_endpoint_with_asc(&mut self,asc:&crate::biprism_asc_runtime::BiprismAscEndpointBinding<'_>)->Result<()> {
        anyhow::ensure!(self.enqueue_cursor==Some(BIPRISM_AMBER_PHASES.len()),"physical phase sequence is incomplete");
        let owner=self.engine.clone();let mut guard=lock_engine(&owner)?;let e=&mut *guard;
        self.commit_constraint_step(e)?;
        self.enqueue_endpoint_raw(e,Some(asc),None)?;self.enqueue_cursor=None;Ok(())
    }
    pub(crate) unsafe fn enqueue_endpoint_with_owners(&mut self,asc:&crate::biprism_asc_runtime::BiprismAscEndpointBinding<'_>,
        mixed:&crate::biprism_mixed_energy::MixedEndpointBinding<'_>)->Result<()> {
        anyhow::ensure!(self.enqueue_cursor==Some(BIPRISM_AMBER_PHASES.len()),"physical phase sequence is incomplete");
        let owner=self.engine.clone();let mut guard=lock_engine(&owner)?;let e=&mut *guard;
        self.commit_constraint_step(e)?;
        self.enqueue_endpoint_raw(e,Some(asc),Some(mixed))?;self.enqueue_cursor=None;Ok(())
    }
    unsafe fn enqueue_aromatic_geometry_raw(&self,e:&mut NhsAmberFusedEngine)->Result<()> {
        if e.n_aromatics==0{return Ok(());}
        let (owner,fault)=e.biprism_coordinate_binding()?.context("aromatic coordinate authority absent")?;
        let aromatics=e.n_aromatics as i32;let atoms=e.n_atoms as i32;
        self.stream.launch_builder(&self.aromatic_geometry_kernel).arg(&e.d_positions)
            .arg(&e.d_aromatic_atom_indices).arg(&e.d_aromatic_n_atoms).arg(&aromatics).arg(&atoms)
            .arg(&mut e.d_aromatic_centroids).arg(&mut e.d_ring_normals).arg(&owner).arg(&fault)
            .launch(LaunchConfig{grid_dim:(1,1,1),block_dim:(256,1,1),shared_mem_bytes:0})?;
        Ok(())
    }
    /// Refresh the existing float summaries from the current full coordinate
    /// authority. Aromatic recipient membership is unchanged.
    pub unsafe fn enqueue_refresh_aromatic_geometry(&self)->Result<()> {
        let mut e=lock_engine(&self.engine)?;self.enqueue_aromatic_geometry_raw(&mut e)
    }
    unsafe fn enqueue_endpoint_raw(&self,e:&mut NhsAmberFusedEngine,
        mixed_asc:Option<&crate::biprism_asc_runtime::BiprismAscEndpointBinding<'_>>,
        mixed_energy:Option<&crate::biprism_mixed_energy::MixedEndpointBinding<'_>>)->Result<()> {
        let stream=&self.stream;
        let (coordinate_owner,precision_fault)=e.biprism_coordinate_binding()?.context("endpoint coordinate authority absent")?;
        let context_label="biprism_lagged_split_endpoint";
        let mut capture_tagger:Option<&mut dyn crate::graph_capture::CaptureTagger>=None;
        let tagger=&mut capture_tagger;
        let n_atoms_i32 = e.n_atoms as i32;
        let n_aromatics_i32 = e.n_aromatics as i32;
        let n_clusters_i32 = self.plan.retained_constraint_count() as i32;
        let receipt_phase = if context_label == "step_autonomous" {
            "cuda_graph_capture_post_fused"
        } else {
            "post_fused"
        };

        let serial_cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        self.enqueue_aromatic_geometry_raw(e)?;
        // Preserve the actual native UV suppression and external-work sequence.
        if crate::protocol_state::uv_burst_suppressed() {
            e.prismcode_md_record(
                e.timestep.max(0) as u64,
                0,
                receipt_phase,
                "nhs_uv_pump_probe_step_suppressed",
                &[pcmd::OP_MD_PHOTO_UV_PUMP_SUPPRESSED],
                serial_cfg.grid_dim,
                serial_cfg.block_dim,
                serial_cfg.shared_mem_bytes,
                Some("P2 UV-off control: pump kernel not launched"),
            );
            if let Some(t) = tagger.as_deref_mut() {
                let _ = t.record_node("uv_pump_probe_suppressed")?;
            }
        } else {
            if !e.historical_may_literal_execution {
            let charge_snapshot_cfg = LaunchConfig {
                grid_dim: ((e.n_aromatics.max(1) as u32).div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                stream
                    .launch_builder(&e.uv_charge_state_snapshot_kernel)
                    .arg(&e.d_is_excited)
                    .arg(&e.d_electronic_population)
                    .arg(&mut e.d_uv_charge_excited_before)
                    .arg(&mut e.d_uv_charge_population_before)
                    .arg(&n_aromatics_i32)
                    .launch(charge_snapshot_cfg)
            }
            .with_context(|| format!("{context_label}: UV charge-state snapshot failed"))?;
            e.prismcode_md_record(
                e.timestep.max(0) as u64,
                0,
                receipt_phase,
                "nhs_uv_charge_state_snapshot",
                &[pcmd::OP_MD_PHOTO_UV_PUMP],
                charge_snapshot_cfg.grid_dim,
                charge_snapshot_cfg.block_dim,
                charge_snapshot_cfg.shared_mem_bytes,
                Some("Capture pre-pump charge parameters for fixed-coordinate work"),
            );
            if let Some(t) = tagger.as_deref_mut() {
                let _ = t.record_node("uv_charge_state_snapshot")?;
            }
            }
            unsafe {
                stream
                    .launch_builder(&e.uv_pump_probe_kernel)
                    .arg(&mut e.d_protocol_state)
                    .arg(&mut e.d_velocities)
                    .arg(&e.d_masses)
                    .arg(&mut e.d_is_excited)
                    .arg(&mut e.d_time_since_excitation)
                    .arg(&mut e.d_electronic_population)
                    .arg(&mut e.d_vibrational_energy)
                    .arg(&mut e.d_franck_condon_progress)
                    .arg(&e.d_aromatic_type)
                    .arg(&e.d_ring_normals)
                    .arg(&e.d_aromatic_neighbors)
                    .arg(&n_aromatics_i32)
                    .arg(&mut e.d_external_work_buffer)
                    .launch(serial_cfg)
            }
            .with_context(|| format!("{context_label}: nhs_uv_pump_probe_step failed"))?;
            e.prismcode_md_record(
                e.timestep.max(0) as u64,
                0,
                receipt_phase,
                "nhs_uv_pump_probe_step",
                &[pcmd::OP_MD_PHOTO_UV_PUMP],
                serial_cfg.grid_dim,
                serial_cfg.block_dim,
                serial_cfg.shared_mem_bytes,
                Some(context_label),
            );
            if let Some(t) = tagger.as_deref_mut() {
                let _ = t.record_node("uv_pump_probe")?;
            }
            if !e.historical_may_literal_execution {
            let charge_work_cfg = atom_owned_launch_config(e.n_atoms);
            // Coordinates are unchanged since FORCE_NP1. Correct only the
            // charge-dependent energy and record that same change as external
            // work before the existing clamp/RATTLE/work-reduction/guard chain.
            unsafe {
                stream
                    .launch_builder(&self.uv_charge_kernel)
                    .arg(&e.d_positions)
                    .arg(&e.d_charges)
                    .arg(&e.d_ground_state_charges)
                    .arg(&e.d_atom_to_aromatic)
                    .arg(&e.d_aromatic_type)
                    .arg(&e.d_aromatic_charge_set_mean)
                    .arg(&e.d_uv_charge_excited_before)
                    .arg(&e.d_uv_charge_population_before)
                    .arg(&e.d_is_excited)
                    .arg(&e.d_electronic_population)
                    .arg(&e.d_neighbor_list)
                    .arg(&e.d_n_neighbors)
                    .arg(&e.d_pairs14_list)
                    .arg(&e.d_pairs14_offsets)
                    .arg(&e.d_pairs14_charge_product)
                    .arg(&e.exact_nonbonded_exceptions_enabled)
                    .arg(&n_atoms_i32)
                    .arg(&e.cutoff)
                    .arg(&e.d_lambda_atom_slot)
                    .arg(&e.d_lambda_slot_values)
                    .arg(&e.n_lambda_slots)
                    .arg(&e.lambda_alpha_q_a2)
                    .arg(&e.lambda_softcore_power)
                    .arg(&e.lambda_coupling_power)
                    .arg(&e.lambda_v2_ewald_enabled)
                    .arg(&mut e.d_potential_energy_components)
                    .arg(&mut e.d_external_work_buffer)
                    .arg(&mut e.d_lambda_dudl_atom)
                    .arg(&mut e.d_lambda_energy_atom)
                    .arg(&coordinate_owner).arg(&precision_fault).launch(charge_work_cfg)
            }
            .with_context(|| format!("{context_label}: UV charge parameter work failed"))?;
            e.prismcode_md_record(
                e.timestep.max(0) as u64,
                0,
                receipt_phase,
                "nhs_uv_charge_parameter_work_compensated_v1",
                &[pcmd::OP_MD_PHOTO_UV_PUMP, pcmd::OP_MD_EXTERNAL_WORK_REDUCE],
                charge_work_cfg.grid_dim,
                charge_work_cfg.block_dim,
                charge_work_cfg.shared_mem_bytes,
                Some(
                    "Post-pump Coulomb parameter delta added once to endpoint PE and external work",
                ),
            );
            if let Some(t) = tagger.as_deref_mut() {
                let _ = t.record_node("uv_charge_parameter_work")?;
            }            }

        }

        if !e.historical_may_literal_execution {
        let atom_cfg = LaunchConfig {
            grid_dim: (
                (e.n_atoms.max(1) as u32).div_ceil(BLOCK_SIZE_1D as u32),
                1,
                1,
            ),
            block_dim: (BLOCK_SIZE_1D as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.clamp_kernel)
                .arg(&mut e.d_velocities)
                .arg(&e.d_masses)
                .arg(&n_atoms_i32)
                .arg(&e.d_protocol_state)
                .arg(&mut e.d_external_work_buffer)
                .arg(&self.d_ownership)
                .launch(atom_cfg)
        }
        .with_context(|| format!("{context_label}: nhs_velocity_safety_clamp failed"))?;
        e.prismcode_md_record(
            e.timestep.max(0) as u64,
            0,
            receipt_phase,
            "nhs_biprism_velocity_safety_clamp_v1",
            &[pcmd::OP_MD_VELOCITY_CLAMP],
            atom_cfg.grid_dim,
            atom_cfg.block_dim,
            atom_cfg.shared_mem_bytes,
            Some(context_label),
        );
        if let Some(t) = tagger.as_deref_mut() {
            let _ = t.record_node("velocity_safety_clamp")?;
        }

        let constraint_cfg = LaunchConfig {
            grid_dim: (
                (self.plan.retained_constraint_count().max(1) as u32).div_ceil(BLOCK_SIZE_1D as u32),
                1,
                1,
            ),
            block_dim: (BLOCK_SIZE_1D as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.rattle_kernel)
                .arg(&e.d_positions)
                .arg(&mut e.d_velocities)
                .arg(&self.d_retained_clusters)
                .arg(&n_clusters_i32)
                .arg(&mut e.d_external_work_buffer)
                .arg(
                    &self.governor
                        .as_ref()
                        .map_or(0u64, |g| g.constraint_window_ptr()),
                )
                .arg(&self.d_ownership)
                .arg(&n_atoms_i32)
                .arg(&coordinate_owner)
                .arg(&precision_fault)
                .launch(constraint_cfg)
        }
        .with_context(|| format!("{context_label}: nhs_rattle_velocity_constraints failed"))?;
        e.prismcode_md_record(
            e.timestep.max(0) as u64,
            0,
            receipt_phase,
            "nhs_biprism_rattle_velocity_constraints_compensated_v1",
            &[pcmd::OP_MD_SHAKE_CONSTRAIN],
            constraint_cfg.grid_dim,
            constraint_cfg.block_dim,
            constraint_cfg.shared_mem_bytes,
            Some(context_label),
        );
        if let Some(t) = tagger.as_deref_mut() {
            let _ = t.record_node("rattle_velocity")?;
        }

        }
        if e.historical_may_literal_execution {
            anyhow::ensure!(e.exact_nonbonded_exceptions_enabled==-1 && self.governor.is_none() && e.asc_device.is_none(),
                "historical numerical operator requires exact admitted model without modern intervention owners");
            unsafe {stream.launch_builder(&e.external_work_reduce_kernel)
                .arg(&mut e.d_external_work_buffer).arg(&n_atoms_i32).arg(&mut e.d_external_work_window)
                .launch(LaunchConfig{grid_dim:(1,1,1),block_dim:(256,1,1),shared_mem_bytes:0})}?;
            e.enqueue_amber_phase_on_stream(stream,MD_PHASE_KICK_NP1)?;
            e.prismcode_md_record(e.timestep.max(0) as u64,MD_PHASE_KICK_NP1 as u32,
                "historical_may_post_uv_second_half","native_may_original_stale_force_halfkick",
                &[pcmd::OP_MD_VELOCITY_SECOND_HALF],atom_owned_launch_config(e.n_atoms).grid_dim,
                atom_owned_launch_config(e.n_atoms).block_dim,0,
                Some("Actual post-UV original raw stale-force halfkick, no418.4, no second force, no post bath or RATTLE"));
            if let Some(t)=tagger.as_deref_mut(){let _=t.record_node("may_original_post_uv_stale_force_second_half")?;}
        }
        // The matching interval starts at the post-LIF snapshot. These exact
        // UV/clamp/endpoint-RATTLE operations publish native nonham work.
        if let Some(mixed)=mixed_energy {mixed.after_native_nonham().map_err(anyhow::Error::msg)?;}
        if e.ca_restraint_fn.is_some() && (context_label != "step_batch" || e.historical_may_literal_execution) {
            self.enqueue_ca_intervention_owned(e)?;
        }
        if let Some(mixed)=mixed_energy {mixed.after_native_ca().map_err(anyhow::Error::msg)?;}
        if e.historical_may_literal_execution && context_label!="may_original_step_final" {
            e.enqueue_may_original_nma_record(stream,false)?;
        }

        if !e.historical_may_literal_execution {
        let reduce_cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&e.external_work_reduce_kernel)
                .arg(&mut e.d_external_work_buffer)
                .arg(&n_atoms_i32)
                .arg(&mut e.d_external_work_window)
                .launch(reduce_cfg)
        }
        .with_context(|| format!("{context_label}: nhs_reduce_external_work_components failed"))?;
        e.prismcode_md_record(
            e.timestep.max(0) as u64,
            0,
            receipt_phase,
            "nhs_reduce_external_work_components",
            &[pcmd::OP_MD_EXTERNAL_WORK_REDUCE],
            reduce_cfg.grid_dim,
            reduce_cfg.block_dim,
            reduce_cfg.shared_mem_bytes,
            Some(context_label),
        );
        }
        if let Some(mixed)=mixed_asc {
            mixed.capture(e.asc_device.as_ref().map_or(0,|a|a.device_ptr()),
                self.d_ownership.device_ptr(stream).0,e.n_atoms as u32,stream)
                .map_err(anyhow::Error::msg)?;
        } else if let Some(asc) = &e.asc_device {
            asc.seal_step(stream)
                .map_err(|rc| anyhow::anyhow!("ASC physical seal failed rc={}", rc))?;
        }
        if let Some(governor) = &self.governor {
            let (work, _work) = e.d_external_work_window.device_ptr(stream);
            unsafe { governor.accumulate_step_work(stream, work) }.map_err(|rc| {
                anyhow::anyhow!("governor observation work accumulation failed rc={}", rc)
            })?;
            governor
                .seal_step(stream)
                .map_err(|rc| anyhow::anyhow!("governor physical seal failed rc={}", rc))?;
            let (forces, _forces) = e.d_forces.device_ptr(stream);
            unsafe {
                governor.measure_force(
                    stream,
                    forces as *const std::ffi::c_void,
                    e.n_atoms as u32,
                )
            }
            .map_err(|rc| anyhow::anyhow!("physical guard force measurement failed rc={}", rc))?;
            self.enqueue_energy_diagnostic(e,7)?;
            governor
                .evaluate_physical_step(stream)
                .map_err(|rc| anyhow::anyhow!("physical guard evaluation failed rc={}", rc))?;
        }
        if let Some(t) = tagger.as_deref_mut() {
            // Preserve this existing tag at its original dependency frontier.
            let _ = t.record_node("external_work_reduce")?;
        }
        e.enqueue_observer_physical_clock_commit(stream)?;
        if let Some(t) = tagger.as_deref_mut() {
            if e.observer_physical_clock_bound() {
                let _ = t.record_node("observer_physical_clock_commit")?;
            }
            // The complete physical endpoint includes the observer clock commit.
            // With the observer unbound both existing tags retain their frontier.
            let _ = t.record_node("physical_step_complete")?;
        }

        Ok(())
    }
    unsafe fn enqueue_ca_intervention_owned(&self,e:&mut NhsAmberFusedEngine)->Result<()> {
        let stream=&self.stream;
        let (coordinate_owner,precision_fault)=e.biprism_coordinate_binding()?.context("CA coordinate authority absent")?;
        let context_label="biprism_bath_ca_coordinate_intervention";
        let mut capture_tagger:Option<&mut dyn crate::graph_capture::CaptureTagger>=None;
        let tagger=&mut capture_tagger;
        e.ca_intervention_capture_admitted=true;
        let historical=e.historical_may_literal_execution;
        let observation_only=historical&&(context_label.starts_with("step_autonomous")||context_label=="step_batch");
        let apply_mask=if observation_only {0u64} else {self.d_bath_ca_mask.device_ptr(stream).0};
        let record_atoms=if observation_only {-i32::try_from(e.n_atoms)?} else {i32::try_from(e.n_atoms)?};
        let atoms=e.n_atoms as i32;
        unsafe {stream.launch_builder(&e.ca_intervention_validate_kernel)
            .arg(&e.d_protocol_state).arg(&e.d_ca_intervention_summary)
            .arg(&e.ca_intervention_journal_capacity).arg(&e.d_ca_intervention_drained)
            .arg(&e.ca_intervention_window_capacity)
            .launch(LaunchConfig{grid_dim:(1,1,1),block_dim:(1,1,1),shared_mem_bytes:0})}
            .context("CA intervention pre-mutation chronology/capacity validation failed")?;
        let atom_cfg=atom_owned_launch_config(e.n_atoms);
        unsafe {stream.launch_builder(&self.ca_begin_kernel)
            .arg(&mut e.d_positions).arg(&mut e.d_positions_step_start)
            .arg(&e.d_velocities).arg(&e.d_masses).arg(&e.d_potential_energy_components)
            .arg(&e.d_external_work_buffer).arg(&mut e.d_ca_intervention_before)
            .arg(&e.d_reference_positions).arg(&apply_mask).arg(&atoms)
            .arg(&e.d_protocol_state).arg(&coordinate_owner).arg(&precision_fault).launch(atom_cfg)}
            .context("CA coordinate intervention snapshot/projection failed")?;
        if let Some(t)=tagger.as_deref_mut(){let _=t.record_node("ca_coordinate_projection")?;}
        // This reuses the actual SHAKE solver with the pre-projection reference.
        // Its temporary work increment is superseded by exact deltaH below.
        if !historical {self.enqueue_phase_raw(e,MD_PHASE_SHAKE_POSITION,true)?;}
        if !historical {
        e.rebuild_neighbor_lists_graph_safe(stream)?;
        stream.memset_zeros(&mut e.d_potential_energy_components)?;
        stream.memset_zeros(&mut e.d_lambda_dudl_atom)?;
        stream.memset_zeros(&mut e.d_lambda_energy_atom)?;
        self.enqueue_phase_raw(e,MD_PHASE_FORCE_NP1,false)?;
        e.enqueue_lambda_dual_endpoint_hamiltonian()?;
        e.enqueue_lambda_pme_composition()?;
        e.compose_nma_force_phase(MD_PHASE_FORCE_NP1,true)?;
        if let Some(asc)=e.asc_device.as_ref() {
            let owner=asc.device_ptr();
            unsafe {stream.launch_builder(&self.ca_asc_kernel)
                .arg(&owner).arg(&e.d_positions).arg(&mut e.d_forces)
                .arg(&mut e.d_potential_energy_components).arg(&atoms).arg(&coordinate_owner).arg(&precision_fault).launch(atom_cfg)}
                .context("CA fixed-parameter ASC endpoint evaluation failed")?;
        }
        }
        if !historical {
        let clusters=self.plan.retained_constraint_count() as i32;
        let constraint_cfg=LaunchConfig{grid_dim:((self.plan.retained_constraint_count().max(1) as u32).div_ceil(256),1,1),block_dim:(256,1,1),shared_mem_bytes:0};
        let constraint_owner=self.governor.as_ref().map_or(0u64,|g|g.constraint_window_ptr());
        unsafe {stream.launch_builder(&self.rattle_kernel)
            .arg(&e.d_positions).arg(&mut e.d_velocities).arg(&self.d_retained_clusters)
            .arg(&clusters).arg(&mut e.d_external_work_buffer).arg(&constraint_owner).arg(&self.d_ownership).arg(&atoms)
            .arg(&coordinate_owner)
            .arg(&precision_fault).launch(constraint_cfg)}
            .context("CA intervention endpoint RATTLE failed")?;
        }
        let atom_window=e.d_ca_intervention_atom_window.as_ref().map_or(0u64,|w|w.device_ptr(stream).0);
        unsafe {stream.launch_builder(&e.ca_intervention_finish_kernel)
            .arg(&e.d_velocities).arg(&e.d_masses).arg(&e.d_potential_energy_components)
            .arg(&e.d_ca_intervention_before).arg(&mut e.d_external_work_buffer)
            .arg(&mut e.d_ca_intervention_isolated).arg(&atoms).arg(&atom_window)
            .arg(&e.d_ca_intervention_summary).arg(&e.ca_intervention_window_capacity).launch(atom_cfg)}
            .context("CA intervention exact Hamiltonian work failed")?;
        let journal=e.d_ca_intervention_journal.as_ref().map_or(0u64,|j|j.device_ptr(stream).0);
        let asc_owner=e.asc_device.as_ref().map_or(0u64,|a|a.device_ptr());
        unsafe {stream.launch_builder(&e.ca_intervention_record_kernel)
            .arg(&e.d_ca_intervention_isolated).arg(&record_atoms).arg(&e.d_protocol_state)
            .arg(&mut e.d_ca_intervention_summary).arg(&journal)
            .arg(&e.ca_intervention_journal_capacity).arg(&asc_owner)
            .launch(LaunchConfig{grid_dim:(1,1,1),block_dim:(256,1,1),shared_mem_bytes:0})}
            .context("CA intervention chronological work record failed")?;
        if let Some(t)=tagger.as_deref_mut(){let _=t.record_node("ca_coordinate_intervention_complete")?;}
        e.prismcode_md_record(e.timestep.max(0) as u64,0,context_label,
            if historical {"nhs_ca_original_projection_or_stored_buffer_observation"} else {"nhs_ca_coordinate_intervention_constrained_exact_work"},
            if historical {if observation_only {&[]} else {&[pcmd::OP_MD_CA_RESTRAINT]}}
                else {&[pcmd::OP_MD_CA_RESTRAINT,pcmd::OP_MD_SHAKE_CONSTRAIN,pcmd::OP_MD_EXTERNAL_WORK_REDUCE]},
            atom_cfg.grid_dim,atom_cfg.block_dim,0,
            Some(if historical {if observation_only {"No coordinate action; complete stored-buffer/clock population, applied alpha0"}
                else {"Actual original alpha projection; no added constraints or force/PE refresh; stored-buffer delta observer, no physical deltaH claim"}}
                else {"Original alpha; native constrained endpoint AMBER/lambda/PME/ASC reevaluation; isolated deltaH work before physical guard"}));
        Ok(())
    }
    fn commit_constraint_step(&self,e:&mut NhsAmberFusedEngine)->Result<()> {
        let stream=&self.stream;
        use cudarc::driver::DevicePtr;
        let window=e.d_conservative_constraint_window.as_ref().map_or(0u64,|v|v.device_ptr(stream).0);
        let atoms=u64::try_from(e.n_atoms)?;
        let source=self.governor.as_ref().map_or(u32::MAX,|v|v.source_stream_id());
        let diagnose=u32::from(std::env::var("PRISM_DIAGNOSTIC_PHASE_ENERGY").ok().as_deref()==Some("1"));
        unsafe {stream.launch_builder(&e.conservative_constraint_commit_kernel)
            .arg(&e.d_conservative_constraint_pre).arg(&e.d_conservative_constraint_shake)
            .arg(&e.d_conservative_constraint_post).arg(&mut e.d_conservative_constraint_state)
            .arg(&e.d_protocol_state).arg(&window).arg(&atoms)
            .arg(&e.conservative_constraint_window_steps).arg(&source).arg(&diagnose)
            .launch(LaunchConfig{grid_dim:(1,1,1),block_dim:(256,1,1),shared_mem_bytes:0})}?;
        Ok(())
        }
    fn enqueue_isolated_shake_diagnostic(&self,e:&mut NhsAmberFusedEngine,phase:i32)->Result<()> {
        let stream=&self.stream;
        if e.phase_energy_diagnostic_scratch.is_none() {return Ok(());}
        anyhow::ensure!(phase==30 || phase==40,"isolated SHAKE diagnostic phase invalid");
        anyhow::ensure!(e.n_lambda_slots==0 && e.lambda_dual_endpoint_enabled==0
            && e.lambda_v2_ewald_enabled==0 && e.lambda_pme.is_none(),
            "isolated SHAKE energy diagnostic has no alchemical/PME scratch authority");
        let (coordinate_owner,precision_fault)=e.biprism_coordinate_binding()?.context("diagnostic coordinate authority absent")?;
        let mut scratch=e.phase_energy_diagnostic_scratch.take().context("diagnostic scratch owner absent")?;
        macro_rules! swap_outputs { () => {{
            std::mem::swap(&mut e.d_forces,&mut scratch.d_forces);
            std::mem::swap(&mut e.d_potential_energy_components,&mut scratch.d_potential_energy_components);
            std::mem::swap(&mut e.d_lambda_dudl_atom,&mut scratch.d_lambda_dudl_atom);
            std::mem::swap(&mut e.d_lambda_energy_atom,&mut scratch.d_lambda_energy_atom);
            std::mem::swap(&mut e.d_cell_list,&mut scratch.d_cell_list);
            std::mem::swap(&mut e.d_cell_counts,&mut scratch.d_cell_counts);
            std::mem::swap(&mut e.d_atom_cell,&mut scratch.d_atom_cell);
            std::mem::swap(&mut e.d_neighbor_list,&mut scratch.d_neighbor_list);
            std::mem::swap(&mut e.d_n_neighbors,&mut scratch.d_n_neighbors);
        }}; }
        // These swaps change capture-time host argument bindings only. Both
        // allocation sets remain retained by this physical engine throughout
        // replay; no device copy or allocation occurs in active capture.
        swap_outputs!();
        let result=(||->Result<()> {
            e.rebuild_neighbor_lists_graph_safe(stream)?;
            stream.memset_zeros(&mut e.d_forces)?;
            stream.memset_zeros(&mut e.d_potential_energy_components)?;
            unsafe{self.enqueue_phase_raw(e,MD_PHASE_FORCE_NP1,false)}?;
            if e.nma_enabled {
                // Physical phase0 has selected/frozen the actual modal force.
                // Validate its pre-phase1 chronology, then use the existing
                // phase2 read-only evaluation without committing either phase.
                let validation_phase=1u32;let read_only_phase=2u32;
                let serial=LaunchConfig{grid_dim:(1,1,1),block_dim:(1,1,1),shared_mem_bytes:0};
                unsafe {stream.launch_builder(&e.nma_validate_kernel)
                    .arg(&e.d_protocol_state).arg(e.d_nma_phase_state.as_ref().context("NMA phase owner absent")?)
                    .arg(&e.nma_work_capacity).arg(&e.nma_work_window_steps).arg(&validation_phase).launch(serial)}?;
                let residues=e.nma_n_residues as i32;let modes=e.nma_n_modes as i32;let cap=10f32;
                let cfg=LaunchConfig{grid_dim:((e.nma_n_residues as u32).div_ceil(256),1,1),block_dim:(256,1,1),shared_mem_bytes:0};
                unsafe {stream.launch_builder(&self.nma_compose_kernel)
                    .arg(&e.d_positions).arg(&e.d_reference_positions).arg(&mut e.d_forces)
                    .arg(&mut e.d_potential_energy_components).arg(&e.d_external_work_buffer)
                    .arg(e.d_nma_displacements.as_ref().context("NMA displacements absent")?)
                    .arg(e.d_nma_force_scales.as_ref().context("NMA scales absent")?)
                    .arg(e.d_nma_ca_indices.as_ref().context("NMA indices absent")?)
                    .arg(&residues).arg(&modes).arg(&e.d_nma_controls).arg(&cap)
                    .arg(&e.d_protocol_state).arg(&read_only_phase)
                    .arg(e.d_nma_current_force.as_ref().context("NMA frozen force absent")?)
                    .arg(e.d_nma_force_rows.as_ref().context("NMA rows absent")?).arg(&coordinate_owner).arg(&precision_fault).launch(cfg)}?;
            }
            if let Some(asc)=e.asc_device.as_ref() {
                // Existing read-only active ASC potential, not a third kick.
                let owner=asc.device_ptr();let atoms=e.n_atoms as i32;
                unsafe {stream.launch_builder(&self.ca_asc_kernel)
                    .arg(&owner).arg(&e.d_positions).arg(&mut e.d_forces)
                    .arg(&mut e.d_potential_energy_components).arg(&atoms)
                    .arg(&coordinate_owner).arg(&precision_fault).launch(atom_owned_launch_config(e.n_atoms))}?;
            }
            self.enqueue_energy_diagnostic(e,phase)
        })();
        // Restore physical bindings and retain diagnostic allocations on BOTH
        // success and fallible capture admission. No early graph-borrow release.
        swap_outputs!();
        e.phase_energy_diagnostic_scratch=Some(scratch);
        result
        }
    fn enqueue_energy_diagnostic(&self,e:&NhsAmberFusedEngine,phase:i32)->Result<()> {
        let stream=&self.stream;
        if std::env::var("PRISM_DIAGNOSTIC_PHASE_ENERGY").ok().as_deref()!=Some("1") {return Ok(());}
        let kernel=e._fused_module.load_function("nhs_diagnostic_phase_energy_compensated_v1")?;
        let (coordinate_owner,precision_fault)=e.biprism_coordinate_binding()?.context("missing compensated diagnostic coordinate authority")?;
        let n=e.n_atoms as i32;let aromatics=e.n_aromatics as i32;
        let source=self.governor.as_ref().map_or(u32::MAX,|g|g.source_stream_id());
        let lambda_slots=e.n_lambda_slots as i32;
        unsafe {stream.launch_builder(&kernel)
            .arg(&e.d_positions).arg(&e.d_positions_step_start)
            .arg(&e.d_velocities).arg(&e.d_forces).arg(&e.d_masses)
            .arg(&e.d_potential_energy_components).arg(&e.d_external_work_buffer)
            .arg(&e.d_external_work_window).arg(&e.d_protocol_state)
            .arg(&n).arg(&phase).arg(&e.cutoff).arg(&e.d_charges).arg(&e.d_lj_params)
            .arg(&e.d_exclusion_list).arg(&e.d_exclusion_offsets)
            .arg(&e.d_pairs14_list).arg(&e.d_pairs14_offsets)
            .arg(&e.exact_nonbonded_exceptions_enabled)
            .arg(&e.d_is_excited).arg(&aromatics).arg(&lambda_slots)
            .arg(&e.d_atom_to_aromatic).arg(&e.d_ground_state_charges)
            .arg(&e.d_aromatic_charge_set_mean).arg(&e.d_aromatic_type)
            .arg(&e.d_electronic_population).arg(&source)
            .arg(&coordinate_owner).arg(&precision_fault)
            .launch(LaunchConfig{grid_dim:(1,1,1),block_dim:(256,1,1),shared_mem_bytes:0})}?;
        Ok(())
        }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn row(c:i32,h:i32)->Vec<u8> {
        let mut r=vec![0u8;40];
        for (p,v) in [(0,c),(4,h),(8,-1),(12,-1),(28,1)] {r[p..p+4].copy_from_slice(&v.to_ne_bytes());}
        for (p,v) in [(16,1.0f32),(32,1.0f32/12.0),(36,1.0f32)] {r[p..p+4].copy_from_slice(&v.to_ne_bytes());}r
    }
    fn core()->Vec<u32>{(0..192).collect()}
    #[test] fn flexible_retains_exact_bytes_and_ids_without_fake_coverage() {
        let raw=[row(0,1),row(200,201)].concat();
        let p=BiprismOwnershipPlan::from_native_clusters(256,&core(),&raw,2,BiprismConstraintPolicy::FlexibleCore).unwrap();
        assert_eq!(p.excluded_original_constraint_ids,[0]);assert_eq!(p.retained_original_constraint_ids,[1]);
        assert_eq!(p.retained_cluster_bytes,row(200,201));assert_eq!(p.ownership.iter().filter(|&&x|x==3).count(),192);
        assert!(p.ownership[192..].iter().all(|&x|x==0));
    }
    #[test] fn flexible_refuses_cut_component_and_constrained_retains_it() {
        let raw=row(0,200);
        assert!(BiprismOwnershipPlan::from_native_clusters(256,&core(),&raw,1,BiprismConstraintPolicy::FlexibleCore).is_err());
        let p=BiprismOwnershipPlan::from_native_clusters(256,&core(),&raw,1,BiprismConstraintPolicy::ConstrainedCore).unwrap();
        assert_eq!(p.retained_cluster_bytes,raw);assert_eq!(p.ownership[0],1);assert_eq!(p.ownership[200],0);
    }
    #[test] fn exact_population_and_native_cluster_validation() {
        let raw=row(200,201);let mut ids=core();
        ids[1]=0;assert!(BiprismOwnershipPlan::from_native_clusters(256,&ids,&raw,1,BiprismConstraintPolicy::FlexibleCore).is_err());
        assert!(BiprismOwnershipPlan::from_native_clusters(256,&core()[..191],&raw,1,BiprismConstraintPolicy::FlexibleCore).is_err());
        assert!(BiprismOwnershipPlan::from_native_clusters(256,&core(),&raw[..39],1,BiprismConstraintPolicy::FlexibleCore).is_err());
        assert!(BiprismOwnershipPlan::from_native_clusters(256,&core(),&row(200,256),1,BiprismConstraintPolicy::FlexibleCore).is_err());
        assert!(BiprismOwnershipPlan::from_native_clusters(256,&core(),&[row(200,201),row(201,202)].concat(),2,BiprismConstraintPolicy::FlexibleCore).is_err());
        let mut nan=raw.clone();nan[32..36].copy_from_slice(&f32::NAN.to_ne_bytes());
        assert!(BiprismOwnershipPlan::from_native_clusters(256,&core(),&nan,1,BiprismConstraintPolicy::FlexibleCore).is_err());
    }
    #[test] fn declared_schedule_preserves_one_force_pair_and_snapshot_before_endpoint() {
        assert_eq!(BIPRISM_AMBER_PHASES.map(|p|p as i32),[0,10,11,12,2,3,4,5,13,14,15]);
    }
}
