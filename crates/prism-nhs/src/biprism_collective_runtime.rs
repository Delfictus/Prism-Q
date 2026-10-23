//! Eight native AMBER leaves interleaved with the owned AU v10 planes in one
//! actual conditional WHILE graph. Every leaf uses the same CUDA stream; native
//! allocation identities and logical leaf identities remain distinct. Native
//! protocol words report the attempted/completed body, while the separate
//! scheduled word advances only after the collective guard accepts that body.
//!
//! This owner does not roll back a failed physical body or provide a durable
//! restart. Failure permanently poisons replay; diagnostics remain readable.
//! The four integer signal grids are explicitly cumulative across all8 leaves,
//! not native per-leaf counts or a claim of UV causality.

use crate::biprism_shared_step::{BiprismSharedStepOwner,BiprismSharedStepAttempt,BiprismSharedStepReadback,validate_shared_time_journal};
use crate::biprism_lif_work::{BiprismLifWorkOwner,LifWorkLedger,LifWorkReadback};
use crate::biprism_phase2_work::{BiprismPhase2WorkOwner,Phase2WorkLedger,Phase2WorkReadback,Phase2InitializationReadback};
use crate::biprism_mixed_energy::{BiprismMixedEnergyOwner,MixedBinding,MixedStage,MixedReadback,MixedCurrentAttempt,MixedEnergyState,validate_mixed_readback};
use crate::biprism_asc_runtime::{BiprismAscWorkOwner, BiprismAscLedgerFfi, BiprismAscWindowReadback};
use crate::biprism_lif_runtime::{
    BiprismLifCarryoverFfi, NativeObservationCaptureStateFfi, NativeObservationDiagnostics,
    NativeObservationFunctions, NativeObservationLeafBinding, NativeObservationLeafFfi,
    QueueExportReceipt, VoronoiLifPlanes,
};
use crate::biprism_spike::{BiprismDecodeLimits, BiprismEventFrame};
use crate::biprism_spike_arrow::VoronoiSpikeExport;
use crate::biprism_v10_runtime::{
    validate_convergence, validate_unit_bridge_status, BiprismConvergenceRecordFfi,
    BiprismPreparationReadback, BiprismUnitBridgeStatusFfi, BiprismV10DeviceState,
    BorrowedProtocolClock, DeviceRegion, GovernedModuleIdentity, UnitBridgeCheckpoint,
    UnitExportKind, V10AcceptanceLimits, V10Step, BIPRISM_TILE_BYTES,
};
use crate::fused_engine::{BiprismAmberPhase as Phase, BiprismPhysicalLeafAdapter,
    BiprismNativeLeafPointers, BiprismNativeEndpointOwnersFfi, SignalPreservationData};
use crate::graph_capture::{capture_pure_template, CapturedTemplate};
use crate::graph_node::{add_child_graph_node_v3, add_while_node_v3, create_while_handle};
use crate::sdst_bridge::SdstGridGeometry;
use crate::spike_arrow_writer::SpikeClassification;
use cudarc::driver::{sys, CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream,
    DevicePtr, DeviceRepr, LaunchConfig, PushKernelArg};
use std::{any::Any, ffi::c_void, ptr, sync::Arc};

pub const BIPRISM_COLLECTIVE_MAGIC:u32=0x4243_4331;
pub const BIPRISM_COLLECTIVE_TERMINAL:u32=0x2000_0000;
pub const BIPRISM_GRID_AGGREGATION:&str="cumulative_direct_device_sum_across_eight_foliated_leaves";

#[repr(C,align(16))]
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct BiprismCollectiveControlFfi {
    pub abi_magic:u32, pub sticky_fault:u32, pub last_completed_step:u32, pub successful_bodies:u32,
    pub last_committed_substeps:u32, pub failed_scheduled_step:u32,
    pub native_steps:[u32;8], pub reserved:[u32;2],
}
unsafe impl DeviceRepr for BiprismCollectiveControlFfi {}
#[repr(C)]
#[derive(Clone,Copy,Debug)]
struct CollectiveLimitsFfi {picard:f32,dec:f32,energy:f32,unitarity:f32}
unsafe impl DeviceRepr for CollectiveLimitsFfi {}
const _:()={
    assert!(core::mem::size_of::<BiprismCollectiveControlFfi>()==64);
    assert!(core::mem::offset_of!(BiprismCollectiveControlFfi,native_steps)==24);
    assert!(core::mem::size_of::<CollectiveLimitsFfi>()==16);
};

#[derive(Clone,Copy,Debug)]
pub struct BiprismCollectiveConfig {
    pub step:V10Step,
    pub dt_feedback_au:f32,
    pub lif_decay:f32,
    pub lif_threshold:f32,
    pub lif_kick:f32,
    pub geometry:SdstGridGeometry,
    pub acceptance:V10AcceptanceLimits,
    pub work_window_capacity:u32,
    pub physical_time_generation:u64,
    pub adaptive_dt_bounds:Option<crate::biprism_shared_step::AdaptiveDtBounds>,
}
impl BiprismCollectiveConfig {
    pub fn validate(&self)->Result<(),String> {
        self.step.validate()?;
        if let Some(b)=self.adaptive_dt_bounds{b.validate()?;}
        if self.physical_time_generation==0||self.step.integration_mode&4!=0{return Err("production requires an explicit nonzero shared-time generation, not fixed-dt diagnostic mode".into());}
        crate::biprism_lif_work::journal_extents(self.work_window_capacity)?;
        crate::biprism_asc_runtime::journal_extents(self.work_window_capacity)?;
        self.geometry.validate().map_err(|e|e.to_string())?;
        if !self.dt_feedback_au.is_finite() || self.dt_feedback_au<=0.0
            || !self.lif_decay.is_finite() || !(0.0..=1.0).contains(&self.lif_decay)
            || !self.lif_threshold.is_finite() || self.lif_threshold<=0.0
            || !self.lif_kick.is_finite()
            || [self.acceptance.picard_residual,self.acceptance.dec_residual,
                self.acceptance.absolute_energy_drift,self.acceptance.unitarity_defect]
                .iter().any(|x|!x.is_finite()||*x<0.0) {
            return Err("invalid explicit collective numerical/observer limits".into());
        }
        Ok(())
    }
}

/// Functions come from the governed native biprism bundle. Native density and
/// same-emission capture remain in their real fused-module owner, supplied
/// separately to the constructor.
pub struct BiprismCollectiveFunctions {
    _module:Arc<CudaModule>, context:Arc<CudaContext>,
    pub identity:GovernedModuleIdentity,
    observer:CudaFunction, delivery:CudaFunction, guard:CudaFunction, latch:CudaFunction,
}
impl BiprismCollectiveFunctions {
    /// # Safety
    /// The admitted module and identity must be returned by the real governed
    /// loader for this context. Strings do not prove loaded image provenance.
    pub unsafe fn from_governed_module(module:Arc<CudaModule>,context:Arc<CudaContext>,
        identity:GovernedModuleIdentity)->Result<Self,String> {
        if identity.member.trim().is_empty() || identity.image_sha256.len()!=64
            || !identity.image_sha256.bytes().all(|b|b.is_ascii_hexdigit()) {
            return Err("governed collective module identity is required".into());
        }
        context.bind_to_thread().map_err(|e|e.to_string())?;
        Ok(Self{observer:module.load_function("nhs_voronoi_dual_lif_and_4grid_bypass_v2").map_err(|e|e.to_string())?,
            delivery:module.load_function("nhs_deliver_persisted_lif_forces_v1").map_err(|e|e.to_string())?,
            guard:module.load_function("nhs_biprism_collective_guard_v1").map_err(|e|e.to_string())?,
            latch:module.load_function("nhs_chunk_while_latch_v10").map_err(|e|e.to_string())?,
            _module:module,context,identity})
    }
}

struct ScheduledClock {word:CudaSlice<u32>}
struct CollectiveGraphs {
    root:sys::CUgraph, exec:sys::CUgraphExec, _template:CapturedTemplate,
    stream:Arc<CudaStream>,
}
impl Drop for CollectiveGraphs {
    fn drop(&mut self) {
        let _=self.stream.synchronize();
        let _=self.stream.context().bind_to_thread();
        unsafe {
            if !self.exec.is_null(){let _=sys::cuGraphExecDestroy(self.exec);}
            if !self.root.is_null(){let _=sys::cuGraphDestroy(self.root);}
        }
    }
}
struct GraphBuildGuard {root:sys::CUgraph,exec:sys::CUgraphExec}
impl Drop for GraphBuildGuard {
    fn drop(&mut self){unsafe {
        if !self.exec.is_null(){let _=sys::cuGraphExecDestroy(self.exec);}
        if !self.root.is_null(){let _=sys::cuGraphDestroy(self.root);}
    }}
}

/// Actual unfiltered completed readbacks; values may describe a rejected body.
#[derive(Debug)]
pub struct BiprismCollectiveStatus {
    pub control:BiprismCollectiveControlFfi,
    pub next_scheduled_step:u32,
    pub remaining_steps:u32,
    pub convergence:[BiprismConvergenceRecordFfi;8],
    pub unit_bridge:[BiprismUnitBridgeStatusFfi;8],
    pub carryover:Vec<BiprismLifCarryoverFfi>,
    pub observation:NativeObservationCaptureStateFfi,
    pub raw_queue_count:u32,
    pub asc:[BiprismAscLedgerFfi;8],
    pub lif_work:[LifWorkLedger;8],
    pub phase2_work:[Phase2WorkLedger;8],
    pub mixed_energy:MixedEnergyState,
    pub shared_time:BiprismSharedStepAttempt,
    pub electronic_exchange:[crate::biprism_v10_runtime::FoliatedCouplingFooterFfi;8],
}
#[derive(Debug)]
pub struct BiprismChunkOutcome {
    pub requested_steps:u32,
    pub completed_before:u32,
    pub accepted:bool,
    pub refusal:Option<String>,
    pub graph_launches:u32,
    pub status:BiprismCollectiveStatus,
}
/// Compatibility container for the existing signal serializer. The explicit
/// aggregation label overrides the legacy per-stream/UV field interpretation.
pub struct BiprismAggregateSignalReadback {
    pub aggregation:&'static str,
    pub through_completed_step:u32,
    pub data:SignalPreservationData,
}

pub struct BiprismCollectiveRuntime {
    // Graph handles retire before any captured allocation or module owner.
    graphs:Option<CollectiveGraphs>,
    adapters:[BiprismPhysicalLeafAdapter;8], state:BiprismV10DeviceState,
    functions:BiprismCollectiveFunctions, stream:Arc<CudaStream>,
    config:BiprismCollectiveConfig, native:[BiprismNativeLeafPointers;8],
    clock:Arc<ScheduledClock>, remaining:CudaSlice<u32>, control:CudaSlice<BiprismCollectiveControlFfi>,
    native_step_pointers:CudaSlice<u64>, endpoint_owners:CudaSlice<BiprismNativeEndpointOwnersFfi>,
    voxel_hits:CudaSlice<i32>, coupled_spikes:CudaSlice<i32>,
    primary_residue:CudaSlice<i32>, primary_count:CudaSlice<i32>,
    lif_water_heuristic:CudaSlice<f32>, lif_exclusion_heuristic:CudaSlice<f32>,
    prepared:bool, poisoned:bool, completed_step:u32, committed_substeps:u32,
    total_successful_bodies:u32, launches:u64, export_pending:bool,
    asc:BiprismAscWorkOwner, asc_readback:Option<BiprismAscWindowReadback>,
    shared_time:BiprismSharedStepOwner,shared_readback:Option<BiprismSharedStepReadback>,
    lif_work:BiprismLifWorkOwner,lif_work_readback:Option<LifWorkReadback>,
    phase2_work:BiprismPhase2WorkOwner,phase2_readback:Option<Phase2WorkReadback>,phase2_initial:Option<Phase2InitializationReadback>,
    mixed_energy:BiprismMixedEnergyOwner,mixed_readback:Option<MixedReadback>,
}
impl BiprismCollectiveRuntime {
    /// # Safety
    /// Caller must exclusively schedule all retained native engines/planes until
    /// this owner drops. No engine clone may be stepped independently, and no
    /// concurrent protocol, graph, density, retile or consumer writer is allowed.
    /// Native observation functions must be the actual fused module's exports.
    /// This constructor does not launch a physical step or claim geometry.
    pub unsafe fn new(mut adapters:[BiprismPhysicalLeafAdapter;8],mut state:BiprismV10DeviceState,
        functions:BiprismCollectiveFunctions,native_observation_functions:NativeObservationFunctions,
        config:BiprismCollectiveConfig,native_completed_step:u32)->Result<Self,String> {
        config.validate()?;
        if native_completed_step!=0{return Err("selected native governor time begins only at a fresh zero epoch; continuation requires full owned state".into());}
        validate_core_correspondence(state.selected_atom_ids(),state.atom_count(),config.step.kappa_foliated)?;
        let stream=state.stream().clone();
        if !Arc::ptr_eq(stream.context(),&functions.context){return Err("collective functions context differs".into());}
        let native=adapters.iter().map(|a|a.pointers().map_err(|e|e.to_string()))
            .collect::<Result<Vec<_>,_>>()?.try_into().map_err(|_|"expected eight native adapters")?;
        let native:[BiprismNativeLeafPointers;8]=native;
        for leaf in 0..8 {
            let a=&adapters[leaf];let p=&native[leaf];let borrowed=&state.native_leaves()[leaf];
            if !Arc::ptr_eq(&a.stream(),&stream) || !Arc::ptr_eq(&borrowed.native_stream,&stream)
                || p.n_atoms!=state.atom_count() as usize
                || borrowed.native_coordinates.address!=p.positions
                || borrowed.native_velocities.address!=p.velocities
                || borrowed.native_forces.address!=p.forces
                || p.grid_dim!=config.geometry.dimension as usize
                || p.grid_spacing.to_bits()!=config.geometry.spacing_a.to_bits()
                || p.grid_origin.map(f32::to_bits)!=config.geometry.origin_a.map(f32::to_bits)
                || p.n_residues!=native[0].n_residues
                || a.ownership_plan().core_atom_ids.as_slice()!=&state.selected_atom_ids()[leaf][..192] {
                return Err(format!("leaf {leaf} actual ownership/stream/population/grid mismatch"));
            }
            if read_raw_u32(&stream,p.protocol_current_step)?!=native_completed_step {
                return Err(format!("leaf {leaf} native completed clock differs"));
            }
        }
        let protocols=adapters.iter().map(|a|a.protocol_snapshot().map_err(|e|e.to_string())).collect::<Result<Vec<_>,_>>()?.try_into().map_err(|_|"eight protocols required")?;
        let shared_time=BiprismSharedStepOwner::new(stream.clone(),&functions._module,&native,protocols,config.physical_time_generation,config.work_window_capacity,config.adaptive_dt_bounds)?;
        state.bind_shared_physical_step(shared_time.retained_record())?;
        let clock=Arc::new(ScheduledClock{word:upload(&stream,&[native_completed_step])?});
        let clock_ptr=clock.word.device_ptr(&stream).0;
        let retained:Arc<dyn Any+Send+Sync>=clock.clone();
        state.bind_protocol_clock(BorrowedProtocolClock::new(retained,stream.context().clone(),
            DeviceRegion{address:clock_ptr,bytes:4})?)?;
        let views=state.device_views();
        for (leaf, adapter) in adapters.iter_mut().enumerate() {
            adapter.bind_precision_fault_sink(views.convergence + leaf as u64 * 128 + 16,views.native_coordinate_owners)
                .map_err(|e|e.to_string())?;
        }
        let mut bindings=Vec::with_capacity(8);
        for leaf in 0..8 {
            let p=&native[leaf];
            let descriptor=NativeObservationLeafFfi{positions:p.positions,atom_types:p.atom_types,
                charges:p.charges,warp_matrix:p.warp_matrix,aromatic_centroids:p.aromatic_centroids,
                is_excited:p.is_excited,electronic_population:p.electronic_population,
                atom_to_aromatic:p.atom_to_aromatic,protocol:p.protocol,exclusion_field:p.exclusion_field,
                water_density:p.water_density,water_density_prev:p.water_density_prev,density_epoch:0,
                selected_atom_ids:views.tiles+(leaf*BIPRISM_TILE_BYTES) as u64,active_tiles:0,
                n_atoms:state.atom_count(),n_aromatics:u32::try_from(p.n_aromatics).map_err(|_|"aromatic extent")?,
                grid_dim:config.geometry.dimension,n_active_tiles:0,grid_spacing:p.grid_spacing,
                origin_x:p.grid_origin[0],origin_y:p.grid_origin[1],origin_z:p.grid_origin[2],native_coordinate_owner:views.native_coordinate_owners+leaf as u64*96,
                au_coordinate_owner:views.coordinate_owners+leaf as u64*96,reserved:0};
            // Empty sparse list deliberately means the actual complete native grid.
            let owner:Arc<dyn Any>=adapters[leaf].engine_owner();
            bindings.push(NativeObservationLeafBinding::new(descriptor,Vec::new(),stream.context().clone(),owner,p.coordinate_owner.generation)?);
        }
        state.lif_mut().bind_native_observation(native_observation_functions,
            bindings.try_into().map_err(|_|"eight native observation bindings required")?)?;
        let voxels=(config.geometry.dimension as usize).pow(3);
        let control_initial=BiprismCollectiveControlFfi{abi_magic:BIPRISM_COLLECTIVE_MAGIC,
            sticky_fault:0,last_completed_step:native_completed_step,successful_bodies:0,
            last_committed_substeps:0,failed_scheduled_step:0,native_steps:[native_completed_step;8],reserved:[0;2]};
        let remaining=upload(&stream,&[0u32])?;
        let control=upload(&stream,&[control_initial])?;
        let endpoint_rows=adapters.iter().map(|a|a.native_endpoint_owners().map_err(|e|e.to_string())).collect::<Result<Vec<_>,_>>()?;
        let asc=BiprismAscWorkOwner::new(stream.clone(),&functions._module,
            endpoint_rows.clone().try_into().map_err(|_|"eight endpoint owners required")?,
            native.clone(),*state.selected_atom_ids(),config.work_window_capacity)?;
        if endpoint_rows.iter().any(|r|r.governor==0||r.physical_guard==0||r.initial_physical_epoch!=0)||endpoint_rows.iter().enumerate().any(|(i,r)|endpoint_rows[..i].iter().any(|p|p.source==r.source)){
            return Err("all eight distinct expected native sources need selected physical guards".into());}
        let lif_work=BiprismLifWorkOwner::new(stream.clone(),&functions._module,&native,*state.selected_atom_ids(),config.work_window_capacity)?;
        let phase2_work=BiprismPhase2WorkOwner::new(stream.clone(),&functions._module,*state.selected_atom_ids(),config.work_window_capacity,config.physical_time_generation,config.step.kappa_foliated)?;
        let mixed_bindings=std::array::from_fn(|leaf|MixedBinding{native_hmr_masses:native[leaf].masses,
            native_potential:native[leaf].potential_energy,native_atom_work:native[leaf].external_work,
            native_cumulative_work:native[leaf].external_work_window,atoms:state.atom_count(),source:endpoint_rows[leaf].source,reserved:0});
        let mixed_energy=BiprismMixedEnergyOwner::new(stream.clone(),&functions._module,mixed_bindings,
            config.work_window_capacity,config.physical_time_generation,config.step.integration_mode,config.step.kappa_foliated)?;
        let endpoint_owners=upload(&stream,&endpoint_rows)?;
        let native_step_pointers=upload(&stream,&native.iter().map(|p|p.protocol_current_step).collect::<Vec<_>>())?;
        let voxel_hits=upload(&stream,&vec![0i32;voxels])?;
        let coupled_spikes=upload(&stream,&vec![0i32;voxels])?;
        let primary_residue=upload(&stream,&vec![-1i32;voxels])?;
        let primary_count=upload(&stream,&vec![0i32;voxels])?;
        let lif_water_heuristic=upload(&stream,&vec![0f32;voxels])?;
        let lif_exclusion_heuristic=upload(&stream,&vec![0f32;voxels])?;
        Ok(Self{graphs:None,adapters,state,functions,stream,config,native,clock,remaining,control,
            native_step_pointers,endpoint_owners,voxel_hits,coupled_spikes,primary_residue,primary_count,
            lif_water_heuristic,lif_exclusion_heuristic,
            prepared:false,poisoned:false,completed_step:native_completed_step,committed_substeps:0,
            total_successful_bodies:0,launches:0,export_pending:false,asc,asc_readback:None,shared_time,shared_readback:None,lif_work,lif_work_readback:None,phase2_work,phase2_readback:None,phase2_initial:None,mixed_energy,mixed_readback:None})
    }

    pub fn stream(&self)->Arc<CudaStream>{self.stream.clone()}
    pub fn config(&self)->BiprismCollectiveConfig{self.config}
    pub fn completed_step(&self)->u32{self.completed_step}
    pub fn graph_launch_count(&self)->u64{self.launches}
    pub fn is_poisoned(&self)->bool{self.poisoned}
    pub fn native_pointers(&self)->&[BiprismNativeLeafPointers;8]{&self.native}
    pub fn state(&self)->&BiprismV10DeviceState{&self.state}

    /// Cold full-N import at the native completed clock, followed by the actual
    /// device BW seeder and caller's strong geometry gate. No Director evaluation
    /// occurs during preparation. Only successful preparation sets scheduled=N+1.
    pub unsafe fn prepare<F>(&mut self,target_tets:u32,validator:F)->Result<(),String>
        where F:FnOnce(&BiprismPreparationReadback)->Result<(),String> {
        if self.prepared||self.poisoned||self.graphs.is_some(){return Err("collective preparation is single-use".into());}
        let result=(|| {
            self.state.capture_import_native_state()?;
            let readback=self.state.prepare_bowyer_watson(target_tets)?;
            validate_initial_electronic_population(readback.tile_bytes())?;
            let before_tiles=readback.tile_bytes().to_vec();
            self.state.accept_preparation(readback,validator)?;
            let preparation_generation=self.state.peer_edge_map_generation().ok_or("cold operator requires admitted topology generation")?;
            self.phase2_initial=Some(Phase2InitializationReadback{generation:self.config.physical_time_generation,preparation_generation,
                native_completed_step:self.completed_step,before_tiles,after_tiles:Vec::new(),ledgers:None});
            // BW carries placeholder weights. Assemble the actual declared operator
            // before the first physical baseline, without a Director/clock step.
            self.state.capture_phase2_per_leaf()?;
            let initial=self.phase2_work.initialize(self.state.device_views(),self.state.atom_count(),self.clock.word.device_ptr(&self.stream).0)?;
            self.phase2_initial.as_mut().unwrap().ledgers=Some(initial);
            self.phase2_initial.as_mut().unwrap().after_tiles=self.state.read_tiles()?;
            self.phase2_initial.as_ref().unwrap().validate()?;
            let potential=self.state.read_potential_state()?;
            self.state.validate_potential_endpoint(&potential,0,self.completed_step)?;
            set_word(&self.stream,self.clock.word.device_ptr(&self.stream).0,self.completed_step+1)?;
            self.stream.synchronize().map_err(|e|e.to_string())?;
            self.prepared=true;Ok(())
        })();
        if result.is_err(){self.poisoned=true;}result
    }

    /// Capture one complete physical body and place its pure template inside a
    /// parent-owned conditional WHILE. Replays do not enqueue host step loops.
    pub unsafe fn capture_once(&mut self)->Result<(),String> {
        if !self.prepared||self.poisoned||self.graphs.is_some(){return Err("capture requires one accepted fresh preparation".into());}
        let result=(|| {
            self.stream.context().bind_to_thread().map_err(|e|e.to_string())?;
            let mut build=GraphBuildGuard{root:ptr::null_mut(),exec:ptr::null_mut()};
            cuda_ok(sys::cuGraphCreate(&mut build.root,0),"collective root create")?;
            let conditional=create_while_handle(build.root,1).map_err(|e|format!("WHILE handle: {e:?}"))?;
            let (_,body)=add_while_node_v3(build.root,&[],conditional).map_err(|e|format!("WHILE body: {e:?}"))?;
            let template=capture_pure_template(self.stream.clone(),|_|self.enqueue_body(conditional).map_err(anyhow::Error::msg))
                .map_err(|e|e.to_string())?;
            add_child_graph_node_v3(body,&[],template.cu_graph()).map_err(|e|format!("pure body splice: {e:?}"))?;
            cuda_ok(sys::cuGraphInstantiateWithFlags(&mut build.exec,build.root,0),"collective graph instantiate")?;
            let root=build.root;let exec=build.exec;build.root=ptr::null_mut();build.exec=ptr::null_mut();
            self.graphs=Some(CollectiveGraphs{root,exec,_template:template,stream:self.stream.clone()});
            Ok(())
        })();
        if result.is_err(){self.poisoned=true;}result
    }

    unsafe fn enqueue_body(&mut self,conditional:u64)->Result<(),String> {
        let scheduled=self.clock.word.device_ptr(&self.stream).0;
        for (leaf,a) in self.adapters.iter_mut().enumerate(){a.enqueue_selected_request(scheduled,self.shared_time.request(leaf)?).map_err(|e|e.to_string())?;}
        self.shared_time.select(scheduled,self.state.device_views().convergence)?;
        for (leaf,a) in self.adapters.iter_mut().enumerate(){a.enqueue_selected_physical_begin(self.shared_time.request(leaf)?).map_err(|e|e.to_string())?;}
        self.enqueue_phase(Phase::ForceN)?;
        self.state.capture_import_native_state()?;
        let asc_view=self.state.device_views();
        self.mixed_energy.sample_all(asc_view,self.shared_time.pointer(),MixedStage::Start)?;
        self.asc.capture_force(asc_view,self.state.atom_count(),scheduled,self.shared_time.dt_au_pointer())?;
        self.phase2_work.capture_before(asc_view,self.shared_time.pointer(),self.state.atom_count())?;
        self.state.capture_phase2_per_leaf()?;
        self.phase2_work.measure_after(self.state.device_views(),self.shared_time.pointer())?;
        for phase in [Phase::OuHalfN,Phase::RattleAfterOuN,Phase::KickN,Phase::RattlePreDrift,Phase::DriftSaveXn] {
            self.enqueue_phase(phase)?;
        }
        // Core native x_n has been saved; v10 modifies only separate AU planes.
        self.state.capture_v10(self.config.step)?;
        self.asc.finish_core(self.state.device_views(),self.state.atom_count(),scheduled)?;
        self.state.capture_export_native_state(UnitExportKind::CoreState)?;
        for phase in [Phase::ShakePosition,Phase::ForceNp1,Phase::KickNp1,Phase::RattleBeforeOuNp1,Phase::OuHalfNp1] {
            self.enqueue_phase(phase)?;
        }
        self.state.capture_refresh_native_velocities()?;
        let view=self.state.device_views();let atoms=self.state.atom_count();
        let delivery_carry=self.state.lif_mut().control_planes().carry;
        self.mixed_energy.sample_all(view,self.shared_time.pointer(),MixedStage::PreLif)?;
        self.lif_work.capture(view,delivery_carry,scheduled,self.shared_time.dt_au_pointer(),atoms,self.config.step.integration_mode)?;
        self.state.lif_mut().capture_delivery_function(&self.functions.delivery,view.stream_table,
            view.convergence,scheduled,atoms,self.config.dt_feedback_au)?;
        self.state.capture_export_native_state(UnitExportKind::DeliveredVelocity)?;
        self.lif_work.measure_and_publish(view,delivery_carry,scheduled)?;
        self.mixed_energy.sample_all(view,self.shared_time.pointer(),MixedStage::PostLif)?;
        for (leaf,a) in self.adapters.iter_mut().enumerate(){
            let asc_endpoint=self.asc.endpoint(leaf,view.convergence);
            let mixed_endpoint=self.mixed_energy.endpoint(view,self.shared_time.pointer(),leaf as u32);
            a.enqueue_endpoint_with_owners(&asc_endpoint,&mixed_endpoint).map_err(|e|e.to_string())?;
        }
        self.state.capture_refresh_observation_state()?;
        self.mixed_energy.sample_all(view,self.shared_time.pointer(),MixedStage::End)?;
        self.mixed_energy.sample_all(view,self.shared_time.pointer(),MixedStage::HaloCounterfactual)?;
        self.mixed_energy.evaluate(view,self.shared_time.pointer(),self.phase2_work.guard_pointers().0,self.lif_work.guard_pointers().0)?;
        for a in &self.adapters{a.enqueue_refresh_aromatic_geometry().map_err(|e|e.to_string())?;}
        self.state.lif_mut().capture_native_density()?;
        let planes=VoronoiLifPlanes{streams:view.stream_table,tile:view.tiles,convergence:view.convergence,
            protocol_step:scheduled,atom_residues:view.atom_residues,
            voxel_hits:self.voxel_hits.device_ptr(&self.stream).0,
            coupled_spikes:self.coupled_spikes.device_ptr(&self.stream).0,
            primary_residue:self.primary_residue.device_ptr(&self.stream).0,
            primary_count:self.primary_count.device_ptr(&self.stream).0,
            water_density:self.lif_water_heuristic.device_ptr(&self.stream).0,
            exclusion:self.lif_exclusion_heuristic.device_ptr(&self.stream).0};
        self.state.lif_mut().capture_observation_function_in_frame(&self.functions.observer,planes,
            self.config.geometry,BiprismEventFrame::AtomicBohr,self.config.lif_decay,
            self.config.lif_threshold,self.config.lif_kick,false)?;
        self.state.lif_mut().capture_native_observations()?;
        let lif=self.state.lif_mut().control_planes();
        let observation=lif.native_observation_state.ok_or("native companion was not bound")?;
        let control=self.control.device_ptr(&self.stream).0;
        let native_steps=self.native_step_pointers.device_ptr(&self.stream).0;
        let endpoint_owners=self.endpoint_owners.device_ptr(&self.stream).0;
        let shared=self.shared_time.pointer();let shared_generation=self.shared_time.generation();
        let (lif_ledgers,lif_window,lif_journal)=self.lif_work.guard_pointers();
        let (phase2_ledgers,phase2_window,phase2_journal)=self.phase2_work.guard_pointers();
        let mixed=self.mixed_energy.guard_pointers();
        let (shared_window,shared_journal)=self.shared_time.acceptance_pointers();
        let limits=CollectiveLimitsFfi{picard:self.config.acceptance.picard_residual,
            dec:self.config.acceptance.dec_residual,energy:self.config.acceptance.absolute_energy_drift,
            unitarity:self.config.acceptance.unitarity_defect};
        let (asc_ledgers,asc_window,asc_journal)=self.asc.guard_pointers();
        self.stream.launch_builder(&self.functions.guard).arg(&control).arg(&view.convergence)
            .arg(&view.unit_bridge_status).arg(&lif.carry).arg(&observation).arg(&lif.count)
            .arg(&lif.capacity).arg(&scheduled).arg(&native_steps).arg(&atoms).arg(&limits)
            .arg(&endpoint_owners).arg(&shared).arg(&shared_generation).arg(&shared_window).arg(&shared_journal).arg(&asc_ledgers).arg(&asc_window).arg(&asc_journal)
            .arg(&lif_ledgers).arg(&lif_window).arg(&lif_journal)
            .arg(&phase2_ledgers).arg(&phase2_window).arg(&phase2_journal)
            .arg(&mixed.state).arg(&mixed.pending).arg(&mixed.leaves).arg(&mixed.samples).arg(&mixed.window)
            .arg(&mixed.globals_journal).arg(&mixed.leaves_journal).arg(&mixed.samples_journal)
            .launch(LaunchConfig{grid_dim:(1,1,1),block_dim:(1,1,1),shared_mem_bytes:0}).map_err(|e|e.to_string())?;
        // Shared time, mixed reference and control publish in the same final guard.
        let remaining=self.remaining.device_ptr(&self.stream).0;
        self.stream.launch_builder(&self.functions.latch).arg(&remaining).arg(&scheduled)
            .arg(&view.convergence).arg(&conditional)
            .launch(LaunchConfig{grid_dim:(1,1,1),block_dim:(1,1,1),shared_mem_bytes:0}).map_err(|e|e.to_string())?;
        Ok(())
    }
    unsafe fn enqueue_phase(&mut self,phase:Phase)->Result<(),String> {
        for a in &mut self.adapters{a.enqueue_phase(phase).map_err(|e|e.to_string())?;}Ok(())
    }

    /// One actual graph launch for a positive exact chunk. All device producers
    /// are quiescent before return. Rejected outcomes retain complete diagnostics
    /// and permanently forbid replay; they never become an accepted empty export.
    pub unsafe fn run_chunk(&mut self,steps:u32)->Result<BiprismChunkOutcome,String> {
        if self.poisoned||self.export_pending{return Err("replay refused: poisoned owner or retained export awaiting acknowledgement".into());}
        let graph=self.graphs.as_ref().ok_or("collective graph is not captured")?;
        validate_chunk_extent(self.completed_step,self.committed_substeps,steps)?;
        let before=self.completed_step;let previous=self.committed_substeps;
        if let Err(e)=self.shared_time.begin_window(self.total_successful_bodies,before+1,steps){self.poisoned=true;return Err(e);}
        if let Err(e)=self.lif_work.begin_window(self.total_successful_bodies,before+1,steps){self.poisoned=true;return Err(e);}
        if let Err(e)=self.asc.begin_window(self.total_successful_bodies,before+1,steps){self.poisoned=true;return Err(e);}
        let preparation_generation=self.state.peer_edge_map_generation().ok_or("Phase2 refresh lost admitted topology generation")?;
        if let Err(e)=self.phase2_work.begin_window(self.total_successful_bodies,before+1,steps,preparation_generation){self.poisoned=true;return Err(e);}
        if let Err(e)=self.mixed_energy.begin_window(u64::from(self.total_successful_bodies),before+1,steps,preparation_generation){self.poisoned=true;return Err(e);}
        set_word(&self.stream,self.remaining.device_ptr(&self.stream).0,steps)?;
        let result=cuda_ok(sys::cuGraphLaunch(graph.exec,self.stream.cu_stream()),"collective WHILE launch");
        if result.is_err(){self.poisoned=true;return result.map(|_|unreachable!());}
        self.launches+=1;
        if let Err(e)=self.stream.synchronize(){self.poisoned=true;return Err(format!("collective physical execution: {e}"));}
        let status=match self.read_status(){Ok(x)=>x,Err(e)=>{self.poisoned=true;return Err(e);}};
        let asc_readback=match self.asc.read_window(){Ok(x)=>x,Err(e)=>{self.poisoned=true;return Err(e);}};
        let shared_readback=match self.shared_time.read_window(){Ok(x)=>x,Err(e)=>{self.poisoned=true;return Err(e);}};let lif_work_readback=match self.lif_work.read_window(){Ok(x)=>x,Err(e)=>{self.poisoned=true;return Err(e);}};
        let phase2_readback=match self.phase2_work.read_window(){Ok(x)=>x,Err(e)=>{self.poisoned=true;return Err(e);}};
        let mixed_readback=match self.mixed_energy.read_window(){Ok(x)=>x,Err(e)=>{self.poisoned=true;return Err(e);}};
        let potential=match self.state.read_potential_state(){Ok(x)=>x,Err(e)=>{self.poisoned=true;return Err(e);}};
        let acceptance=validate_completed_chunk(&status,before,previous,self.total_successful_bodies,steps,self.config.acceptance)
            .and_then(|_|self.asc.validate_readback(&asc_readback))
            .and_then(|_|self.lif_work.validate(&lif_work_readback)).and_then(|_|validate_shared_time_journal(&shared_readback))
            .and_then(|_|validate_work_time_join(&asc_readback,&lif_work_readback,&shared_readback))
            .and_then(|_|self.phase2_work.validate(&phase2_readback))
            .and_then(|_|validate_phase2_time_join(&phase2_readback,&shared_readback))
            .and_then(|_|validate_mixed_readback(&mixed_readback))
            .and_then(|_|validate_mixed_work_time_join(&mixed_readback,&phase2_readback,&shared_readback))
            .and_then(|_|self.state.validate_potential_endpoint(&potential,previous+steps*4,before+steps))
            .and_then(|_|crate::biprism_v10_runtime::validate_electronic_exchange(&status.electronic_exchange,previous+steps*4,self.config.step.integration_mode,self.config.acceptance));
        self.asc_readback=Some(asc_readback);self.shared_readback=Some(shared_readback);self.lif_work_readback=Some(lif_work_readback);self.phase2_readback=Some(phase2_readback);self.mixed_readback=Some(mixed_readback);
        let refusal=acceptance.err();let accepted=refusal.is_none();
        if accepted{self.completed_step=before+steps;self.committed_substeps=previous+steps*4;self.total_successful_bodies+=steps;}
        else{self.poisoned=true;}
        Ok(BiprismChunkOutcome{requested_steps:steps,completed_before:before,accepted,refusal,graph_launches:1,status})
    }

    pub fn coordinate_snapshots(&self)->Result<Vec<crate::biprism_coordinates::BiprismCoordinateSnapshot>,String>{
        self.state.read_coordinate_snapshots()
    }
    pub fn spinor_snapshot(&self)->Result<Vec<crate::biprism_v10_runtime::BiprismSpinorOwnerFfi>,String>{self.state.read_spinor_state()}
    pub fn potential_snapshot(&self)->Result<crate::biprism_v10_runtime::BiprismPotentialReadback,String>{self.state.read_potential_state()}

    /// Read even after refusal. Native clocks below were sampled in the ordered
    /// device guard; next_scheduled_step is the separate latch-owned4B word.
    pub fn read_status(&mut self)->Result<BiprismCollectiveStatus,String> {
        self.stream.synchronize().map_err(|e|e.to_string())?;
        let control=download(&self.stream,&self.control)?[0];
        let next_scheduled_step=download(&self.stream,&self.clock.word)?[0];
        let remaining_steps=download(&self.stream,&self.remaining)?[0];
        let convergence=self.state.read_convergence()?;
        let unit_bridge=self.state.read_unit_bridge_status()?;
        let carryover=unsafe{self.state.lif_mut().readback_carryover()?};
        let diagnostics=unsafe{self.state.lif_mut().readback_native_observation_diagnostics()?};
        Ok(BiprismCollectiveStatus{control,next_scheduled_step,remaining_steps,convergence,
            unit_bridge,carryover,observation:diagnostics.state,raw_queue_count:diagnostics.count,asc:self.asc.read_current()?,lif_work:self.lif_work.read_ledgers()?,phase2_work:self.phase2_work.read_ledgers()?,mixed_energy:self.mixed_energy.read_state()?,shared_time:self.shared_time.read_current()?,electronic_exchange:self.state.read_electronic_exchange()?.footers()})
    }

    /// Completed-seal journal storage for the attempted window. The caller must retain the
    /// binary records/index before the paired-export acknowledgement. No copy is
    /// synthesized from endpoint aggregates, and failed windows remain available.
    pub fn asc_work_window(&self)->Result<&BiprismAscWindowReadback,String> {
        self.asc_readback.as_ref().ok_or_else(||"no completed ASC readback window".into())
    }

    /// Unfiltered attempted-current-body storage, including rows which failed
    /// before journal publication. Available on the terminal refusal path.
    pub fn asc_current_attempt(&self)->Result<crate::biprism_asc_runtime::BiprismAscAttemptReadback,String> {
        self.asc.read_current_attempt()
    }

    /// Exact derived exchange banks at the quiescent endpoint or failed attempt.
    pub fn electronic_exchange(&self)->Result<crate::biprism_v10_runtime::BiprismElectronicExchangeReadback,String>{
        self.state.read_electronic_exchange()
    }

    pub fn lif_work_window(&self)->Result<&LifWorkReadback,String>{self.lif_work_readback.as_ref().ok_or_else(||"no attempted LIF work window".into())}
    pub fn shared_time_window(&self)->Result<&BiprismSharedStepReadback,String>{self.shared_readback.as_ref().ok_or_else(||"no attempted shared time window".into())}
    pub fn lif_work_current_attempt(&self)->Result<LifWorkReadback,String>{self.lif_work.read_current()}
    pub fn shared_time_current_attempt(&self)->Result<BiprismSharedStepAttempt,String>{self.shared_time.read_current()}
    pub fn phase2_work_window(&self)->Result<&Phase2WorkReadback,String>{self.phase2_readback.as_ref().ok_or_else(||"no attempted Phase2 work window".into())}
    pub fn phase2_work_current_attempt(&self)->Result<Phase2WorkReadback,String>{self.phase2_work.read_current()}
    pub fn initial_operator(&self)->Option<&Phase2InitializationReadback>{self.phase2_initial.as_ref()}
    pub fn mixed_energy_window(&self)->Result<&MixedReadback,String>{self.mixed_readback.as_ref().ok_or_else(||"no attempted mixed energy window".into())}
    pub fn mixed_energy_current_attempt(&self)->Result<MixedCurrentAttempt,String>{self.mixed_energy.read_current()}
    /// Exact raw producer queue and independently captured native context for
    /// per-row classification. Available for failure artifacts; no filtering.
    pub unsafe fn raw_observation_inputs(&mut self)->Result<(u32,Vec<u8>,NativeObservationDiagnostics),String> {
        self.stream.synchronize().map_err(|e|e.to_string())?;
        let (count,bytes)=self.state.lif_mut().readback_raw_queue()?;
        let native=self.state.lif_mut().readback_native_observation_diagnostics()?;
        Ok((count,bytes,native))
    }
    /// Produces the actual paired nativeV5 and versioned ledger; both must be
    /// retained by the consumer before acknowledge_export is called.
    pub unsafe fn export_captured(&mut self,classifications:&[SpikeClassification],replica_seed:u64)
        ->Result<(VoronoiSpikeExport,QueueExportReceipt),String> {
        if self.poisoned||self.export_pending||self.total_successful_bodies==0 {
            return Err("export requires accepted physical execution and an available queue epoch".into());
        }
        let limits=BiprismDecodeLimits{num_atoms:self.state.atom_count(),
            num_residues:u32::try_from(self.native[0].n_residues).map_err(|_|"residue extent")?,
            total_voxels:self.config.geometry.dimension.pow(3),expected_step:None};
        let out=self.state.lif_mut().readback_with_captured_native_observations(limits,classifications,replica_seed)?;
        self.export_pending=true;Ok(out)
    }
    pub unsafe fn acknowledge_export(&mut self,receipt:&QueueExportReceipt)->Result<(),String> {
        if self.poisoned||!self.export_pending{return Err("no accepted retained export awaits acknowledgement".into());}
        self.state.lif_mut().acknowledge_retained_export(receipt)?;
        self.asc.acknowledge();self.asc_readback=None;self.lif_work.acknowledge();self.lif_work_readback=None;self.shared_time.acknowledge();self.shared_readback=None;self.phase2_work.acknowledge();self.phase2_readback=None;self.mixed_energy.acknowledge();self.mixed_readback=None;
        self.export_pending=false;Ok(())
    }
    /// Separate optional LIF float accumulators, never the native density model.
    pub fn lif_heuristic_grids(&self)->Result<(Vec<f32>,Vec<f32>),String> {
        self.stream.synchronize().map_err(|e|e.to_string())?;
        Ok((download(&self.stream,&self.lif_water_heuristic)?,download(&self.stream,&self.lif_exclusion_heuristic)?))
    }
    pub fn aggregate_signal_grids(&self)->Result<BiprismAggregateSignalReadback,String> {
        self.stream.synchronize().map_err(|e|e.to_string())?;
        Ok(BiprismAggregateSignalReadback{aggregation:BIPRISM_GRID_AGGREGATION,
            through_completed_step:self.completed_step,
            data:SignalPreservationData{voxel_hit_grid:download(&self.stream,&self.voxel_hits)?,
                coupled_spike_grid:download(&self.stream,&self.coupled_spikes)?,
                primary_residue_id:download(&self.stream,&self.primary_residue)?,
                primary_residue_count:download(&self.stream,&self.primary_count)?,
                grid_dim:self.config.geometry.dimension as usize}})
    }
}
impl Drop for BiprismCollectiveRuntime {
    fn drop(&mut self){let _=self.stream.synchronize();self.graphs.take();}
}

/// The coupling model pairs the same physical atom across leaves. Different
/// membership or ordering is an input error; this owner never silently retiles.
pub fn validate_core_correspondence(selected:&[[u32;256];8],atoms:u32,kappa:f32)->Result<(),String> {
    if !kappa.is_finite(){return Err("nonfinite foliated coupling".into());}
    for leaf in 0..8 {
        let mut seen=std::collections::HashSet::new();
        for (slot,&atom) in selected[leaf][..192].iter().enumerate(){
            if atom>=atoms||!seen.insert(atom){return Err(format!("invalid core identity leaf{leaf} slot{slot} atom{atom}"));}
            if kappa!=0.0 && atom!=selected[0][slot] {
                return Err(format!("foliated physical correspondence mismatch leaf{leaf} core_slot{slot}: atom{atom} != reference atom{}; explicit common population required",selected[0][slot]));
            }
        }
    }Ok(())
}
/// The actual BW producer currently has a fixed |0> initializer; this check binds
/// its produced FP32 bytes. It does not invent a configurable electronic state.
pub fn validate_initial_electronic_population(bytes:&[u8])->Result<(),String> {
    const PSI_OFFSET:usize=90112;
    if bytes.len()!=8*BIPRISM_TILE_BYTES{return Err("initial tile population extent".into());}
    for leaf in 0..8 {for vertex in 0..256 {for component in 0..4 {
        let at=leaf*BIPRISM_TILE_BYTES+PSI_OFFSET+vertex*16+component*4;
        let actual=u32::from_le_bytes(bytes[at..at+4].try_into().unwrap());
        let expected=if component==0{1.0f32.to_bits()}else{0};
        if actual!=expected{return Err(format!("initial BW spinor differs at leaf{leaf} vertex{vertex} component{component}: bits{actual:#x}"));}
    }}}Ok(())
}

pub fn validate_chunk_extent(completed:u32,committed:u32,steps:u32)->Result<(),String> {
    if steps==0||committed%4!=0||completed.checked_add(steps).filter(|x|*x<u32::MAX).is_none()
        ||steps.checked_mul(4).and_then(|x|committed.checked_add(x)).is_none(){
        return Err("chunk would overflow its physical clocks or success counters".into());
    }Ok(())
}
pub fn validate_completed_chunk(status:&BiprismCollectiveStatus,previous_step:u32,previous_committed:u32,
    previous_bodies:u32,steps:u32,limits:V10AcceptanceLimits)->Result<(),String> {
    validate_chunk_extent(previous_step,previous_committed,steps)?;
    let expected_step=previous_step+steps;let expected_committed=previous_committed+steps*4;
    let c=&status.control;
    if c.abi_magic!=BIPRISM_COLLECTIVE_MAGIC || c.sticky_fault!=0 || c.failed_scheduled_step!=0
        || c.reserved!=[0;2] || c.last_completed_step!=expected_step
        || c.successful_bodies!=previous_bodies.checked_add(steps).ok_or("body counter overflow")?
        || c.last_committed_substeps!=expected_committed || c.native_steps!=[expected_step;8]
        || status.remaining_steps!=0 || status.next_scheduled_step!=expected_step+1 {
        return Err("collective guard/clock identity refused the physical window".into());
    }
    let shared=&status.shared_time.shared;
    if shared.state!=2||shared.sticky_fault!=0||shared.last_accepted_step!=expected_step||shared.accepted_bodies!=previous_bodies+steps
        ||shared.scheduled_step!=expected_step{return Err("shared accepted time did not commit this collective window".into());}
    validate_convergence(&status.convergence,previous_committed,steps,limits)?;
    validate_unit_bridge_status(&status.unit_bridge,status.unit_bridge[0].atom_count,
        expected_step,expected_committed,UnitBridgeCheckpoint::ObservationRefresh)?;
    if status.carryover.len()!=8{return Err("incomplete carryover readback".into());}
    for carry in &status.carryover {
        let delivered=if c.successful_bodies>1{expected_step}else{0};
        if carry.pending!=1||carry.produced_step!=expected_step||carry.delivered_step!=delivered
            ||carry.fault!=0||carry.produced_committed_substeps!=expected_committed||carry.reserved!=[0;3] {
            return Err("carryover epoch or fault refused the physical window".into());
        }
    }
    let o=&status.observation;
    if o.abi_magic!=crate::biprism_lif_runtime::NATIVE_OBSERVATION_MAGIC||o.fault!=0||o.reserved!=0
        ||o.last_step!=expected_step||o.cursor!=status.raw_queue_count||o.bad_record!=u32::MAX {
        return Err("same-emission native observation refused the physical window".into());
    }
    Ok(())
}
fn cuda_ok(rc:sys::CUresult,context:&str)->Result<(),String>{
    if rc==sys::CUresult::CUDA_SUCCESS{Ok(())}else{Err(format!("{context}: {rc:?}"))}
}
unsafe fn set_word(stream:&Arc<CudaStream>,word:u64,value:u32)->Result<(),String>{
    cuda_ok(sys::cuMemsetD32Async(word,value,1,stream.cu_stream()),"collective word initialization")
}
fn upload<T:DeviceRepr>(stream:&Arc<CudaStream>,source:&[T])->Result<CudaSlice<T>,String>{
    let out=stream.clone_htod(source).map_err(|e|e.to_string())?;
    stream.synchronize().map_err(|e|e.to_string())?;Ok(out)
}
fn download<T:DeviceRepr>(stream:&Arc<CudaStream>,source:&CudaSlice<T>)->Result<Vec<T>,String>{
    let out=stream.clone_dtoh(source).map_err(|e|e.to_string())?;
    stream.synchronize().map_err(|e|e.to_string())?;Ok(out)
}
unsafe fn read_raw_u32(stream:&Arc<CudaStream>,address:u64)->Result<u32,String>{
    if address==0||address%4!=0{return Err("invalid native clock pointer".into());}
    let mut value=0u32;
    cuda_ok(sys::cuMemcpyDtoHAsync_v2((&mut value as *mut u32).cast::<c_void>(),address,4,stream.cu_stream()),"native clock read")?;
    stream.synchronize().map_err(|e|e.to_string())?;Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]fn physical_ring_requires_equal_atom_order_without_ordinal_substitution(){
        let mut ids=std::array::from_fn(|_|std::array::from_fn(|i|i as u32));
        assert!(validate_core_correspondence(&ids,256,0.2).is_ok());
        ids[3].swap(10,11);
        let err=validate_core_correspondence(&ids,256,0.2).unwrap_err();assert!(err.contains("leaf3 core_slot10"));
        assert!(validate_core_correspondence(&ids,256,0.0).is_ok());
        ids[3][10]=ids[3][9];assert!(validate_core_correspondence(&ids,256,0.0).is_err());
    }
    #[test]fn actual_bw_population_refuses_nonfinite_or_changed_spinor(){
        let mut b=vec![0u8;8*BIPRISM_TILE_BYTES];
        for leaf in 0..8 {for v in 0..256 {let at=leaf*BIPRISM_TILE_BYTES+90112+v*16;b[at..at+4].copy_from_slice(&1.0f32.to_le_bytes());}}
        assert!(validate_initial_electronic_population(&b).is_ok());
        b[90112..90116].copy_from_slice(&f32::NAN.to_le_bytes());assert!(validate_initial_electronic_population(&b).is_err());
    }
    #[test]fn zero_chunk_is_never_launched(){assert!(validate_chunk_extent(0,0,0).is_err());}
    #[test]fn exact_clock_and_counter_bounds(){
        assert!(validate_chunk_extent(7,0,500).is_ok());
        assert!(validate_chunk_extent(u32::MAX-1,0,1).is_err());
        assert!(validate_chunk_extent(0,u32::MAX-3,1).is_err());
        assert!(validate_chunk_extent(0,3,1).is_err());
    }
}

/// Exact owner join; it is not an energy-conservation threshold.
fn validate_work_time_join(asc:&BiprismAscWindowReadback,lif:&LifWorkReadback,time:&BiprismSharedStepReadback)->Result<(),String>{
    if asc.ledgers.len()!=time.records.len()*8||lif.ledgers.len()!=asc.ledgers.len(){return Err("work/time record population differs".into());}
    for(i,t)in time.records.iter().enumerate(){for leaf in 0..8{let a=&asc.ledgers[i*8+leaf];let l=&lif.ledgers[i*8+leaf];
        if a.scheduled_step!=t.step.scheduled_step||l.scheduled_step!=t.step.scheduled_step
            ||a.completed_steps!=u64::from(t.step.accepted_bodies)||l.completed_steps!=u64::from(t.step.accepted_bodies)
            ||a.dt_outer_au.to_bits()!=f64::from(t.step.outer_dt_au).to_bits()
            ||l.dt_feedback_au.to_bits()!=f64::from(t.step.outer_dt_au).to_bits(){return Err(format!("ASC/LIF/shared physical-time mismatch body{i} leaf{leaf}"));}}}
    Ok(())
}

// A complete operator measurement is qualified as a physical-body record only
// when its generation/body/step is joined to the accepted common-time journal.
fn validate_phase2_time_join(work:&Phase2WorkReadback,time:&BiprismSharedStepReadback)->Result<(),String>{
    if work.ledgers.len()!=time.records.len()*8{return Err("Phase2/time population mismatch".into());}
    for(i,t)in time.records.iter().enumerate(){for leaf in 0..8{let w=&work.ledgers[i*8+leaf];
        if w.generation!=time.generation||w.scheduled_step!=t.step.scheduled_step
            ||w.completed_measurements!=u64::from(t.step.accepted_bodies)||w.body_ordinal+1!=u64::from(t.step.accepted_bodies)
            ||w.entry_committed.checked_add(4)!=t.step.accepted_bodies.checked_mul(4){
            return Err(format!("Phase2/accepted time identity mismatch body{i} leaf{leaf}"));}
    }}Ok(())
}

fn validate_mixed_work_time_join(mixed:&MixedReadback,phase2:&Phase2WorkReadback,time:&BiprismSharedStepReadback)->Result<(),String>{
    if mixed.globals.len()!=time.records.len()||mixed.leaves.len()!=phase2.ledgers.len(){return Err("mixed/work/time population mismatch".into());}
    for(body,t)in time.records.iter().enumerate(){let g=&mixed.globals[body];
        if g.generation!=time.generation||g.scheduled_step!=t.step.scheduled_step||g.accepted_bodies!=u64::from(t.step.accepted_bodies)
            ||g.committed_substeps!=t.step.accepted_bodies.checked_mul(4).ok_or("mixed committed overflow")?{return Err("mixed accepted-time identity".into());}
        for leaf in 0..8{let l=&mixed.leaves[body*8+leaf];let p=&phase2.ledgers[body*8+leaf];
            if l.generation!=p.generation||l.preparation_generation!=p.preparation_generation||l.body_ordinal!=p.body_ordinal
                ||l.values[1]!=p.totals[12]{return Err("mixed Phase2 parameter work is not the actual accepted measurement".into());}}
    }Ok(())
}
