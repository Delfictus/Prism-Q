//! Per-stream governor storage and its capture lifetime. Allocation contents
//! are mutated only by the owning CUDA stream; Arc clones are lifetime leases.
#![cfg(feature = "gpu")]
use crate::gearbox::{GearGovernorPolicy, GearGovernorState};
use cudarc::driver::CudaStream;
use std::{
    ffi::c_void,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

static NEXT_GENERATION: AtomicU64 = AtomicU64::new(1);
const CLOSE_TIMEOUT_MS: u32 = 30_000;
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
pub struct ConstraintWindow {
    pub schema_version: u32,
    pub status: u32,
    pub source_stream_id: u32,
    pub phase_mask: u32,
    pub n_constraint_clusters: u32,
    pub expected_physical_steps: u32,
    pub completed_physical_steps: u32,
    pub fatal_failure_count: u32,
    pub allocation_generation: u64,
    pub window_epoch: u64,
    pub first_step_epoch: u64,
    pub last_step_epoch: u64,
    pub expected_cluster_solves: [u64; 3],
    pub completed_cluster_solves: [u64; 3],
    pub sealed_epoch: u64,
    pub phase_cluster_epochs: u64,
}
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
pub struct GovernorSnapshot {
    pub abi_version: u32,
    pub byte_size: u32,
    pub source_stream_id: u32,
    pub lifecycle: u32,
    pub allocation_generation: u64,
    pub binding_epoch: u64,
    pub state: GearGovernorState,
    pub decision_epoch: u64,
    pub observed_force_epoch: u64,
    pub observed_constraint_epoch: u64,
    pub last_physical_epoch: u64,
    pub force_status: u32,
    pub constraint_status: u32,
    pub reset_reason: u32,
    pub reserved: u32,
}
const _: () = assert!(std::mem::size_of::<GovernorSnapshot>() == 208);
const _: () = assert!(std::mem::size_of::<ConstraintWindow>() == 128);
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct PhysicalGuardBindings {
    pub abi: u32, pub byte_size: u32, pub source: u32, pub n_atoms: u32,
    pub generation: u64, pub governor: u64, pub forces: u64, pub potential: u64,
    pub velocities: u64, pub masses: u64, pub dt: u64, pub step_work: u64,
}
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct PhysicalGuardSnapshot {
    pub abi:u32,pub byte_size:u32,pub source:u32,pub status:u32,
    pub generation:u64,pub physical_eval_epoch:u64,pub science_epoch:u64,pub science_publications:u64,
    pub science_status:u32,pub science_code:u32,pub science_request:u32,pub science_override:u32,
    pub initial_gear:u32,pub applied_gear:u32,pub dt_status:u32,pub energy_valid:u32,
    pub previous_pe:f64,pub current_pe:f64,pub observed_work:f64,pub observed_drift:f64,
    pub last_transition_work:f64,pub total_transition_work:f64,pub pending_interval_work:f64,pub admitted_dt:f64,
    pub force_status:u32,pub constraint_status:u32,pub transition_pending:u32,pub host_ceiling:u32,
    pub physical_decisions:u64,pub transition_count:u64,pub science_age_physical_steps:u64,pub max_evaluation_gap:u64,
}
const _: () = assert!(std::mem::size_of::<PhysicalGuardBindings>() == 80);
const _: () = assert!(std::mem::size_of::<PhysicalGuardSnapshot>() == 192);
const _: () = assert!(std::mem::offset_of!(PhysicalGuardSnapshot, previous_pe) == 80);
const _: () = assert!(std::mem::offset_of!(PhysicalGuardSnapshot, physical_decisions) == 160);
/// One completed ordinary physical step, captured after pre-kick dt admission
/// and committed before post-step gear changes. Times are picoseconds.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct PhysicalTimeRecord {
    pub step:u64,
    pub applied_dt_ps:f64,
    pub cumulative_simulated_time_ps:f64,
}
const _:()=assert!(std::mem::size_of::<PhysicalTimeRecord>()==24);
/// Content fields are native readbacks. Allocation generation identifies the
/// live operational owner; continuation already preserves/rebinds its history.
#[derive(Clone,Debug,serde::Serialize)]
pub struct UnarmedPhysicalClockStateV1 {
    pub schema_version:u32,pub source_stream_id:u32,pub allocation_generation:u64,
    pub capacity_steps:u64,pub completed_step:u64,pub cumulative_simulated_time_ps_bits:u64,
    pub applied_dt_ps_bits:u64,pub clock_policy_armed:u32,
}
pub const PHYSICAL_ENERGY_NORMALIZATION_CONTRACT: &str = "prism.physical_energy_balance.previous_endpoint_kinetic.v1";
pub const PHYSICAL_ENERGY_BALANCE_RECORD_BYTES: usize = 160;
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct PhysicalEnergyBalanceRecord {
    pub abi:u32,pub byte_size:u32,pub source:u32,pub history_status:u32,
    pub normalization_contract:u32,pub reserved:u32,
    pub generation:u64,pub step:u64,pub physical_epoch:u64,pub previous_epoch:u64,
    pub previous_potential:f64,pub current_potential:f64,pub previous_kinetic:f64,pub current_kinetic:f64,
    pub body_work:f64,pub carried_transition_work:f64,pub observed_work:f64,pub signed_residual:f64,
    pub kinetic_reference:f64,pub normalized_residual:f64,pub legacy_h_ratio:f64,
    pub applied_dt_ps:f64,pub cumulative_simulated_time_ps:f64,
}
const _:()=assert!(std::mem::size_of::<PhysicalEnergyBalanceRecord>()==PHYSICAL_ENERGY_BALANCE_RECORD_BYTES);
const _:()=assert!(std::mem::offset_of!(PhysicalEnergyBalanceRecord,previous_potential)==56);
pub fn physical_energy_balance_contract()->serde_json::Value {
    serde_json::json!({
        "contract":PHYSICAL_ENERGY_NORMALIZATION_CONTRACT,"normalization_contract":1,
        "record_abi":1,"record_bytes":PHYSICAL_ENERGY_BALANCE_RECORD_BYTES,
        "formula":"abs((U_current-U_previous)+(K_current-K_previous)-observed_work)/K_previous",
        "reference":"actual previous accepted post-integrator/pre-transition endpoint kinetic energy; finite and strictly positive",
        "work":"body_work + carried_transition_work; deterministic constraint impulses remain separate and are not added to work",
        "history_status":{"0":"BOOTSTRAP_NO_PREVIOUS_ACCEPTED_ENDPOINT","1":"ESTABLISHED_VALID","2":"ESTABLISHED_INVALID_KINETIC_REFERENCE"},
        "bootstrap":"only first physical epoch; previous quantities and normalized residual are unavailable, not zero",
        "invalid_reference":"fatal physical measurement state independently of armed policy; never bootstrap or denominator floor",
        "legacy_h_ratio":"diagnostic only; abs(H_current-(H_previous+work))/abs(H_previous+work); unavailable when legacy denominator is unusable",
        "signed_residual":"direct component difference; retained independently of normalized metric and legacy rounded-H diagnostic",
        "energy_units":"kcal/mol","time_units":"ps",
        "source":"existing native fixed-lane PE and KE reductions; no added atom scan or GPU launch",
        "journal":"same complete horizon, count and step authority as native physical time; every completed physical step before next gear transition",
        "failure_visibility":"terminal guard printf retains failing components; context trap may prevent host retrieval of current incomplete observation window",
        "numerical_bands":"unchanged drift_abort and all fractional bands; explicitly new measurement contract, not equivalent prior calibration",
        "scientific_calibration":"not established by this numerical source repair"
    })
}

extern "C" {
    fn prism_physical_guard_bind_time_v1(ptr:u64,step:u64,capacity:u64,stream:*mut c_void)->i32;
    fn prism_physical_guard_read_energy_v1(ptr:u64,start:u64,count:u64,out:*mut PhysicalEnergyBalanceRecord,stream:*mut c_void,timeout_ms:u32)->i32;
    fn prism_physical_guard_read_time_v1(ptr:u64,start:u64,count:u64,out:*mut PhysicalTimeRecord,stream:*mut c_void,timeout_ms:u32)->i32;
    fn prism_physical_guard_create_v1(out:*mut u64,bindings:*const PhysicalGuardBindings,stream:*mut c_void)->i32;
    fn prism_physical_guard_begin_step_v1(ptr:u64,stream:*mut c_void)->i32;
    fn prism_physical_guard_begin_selected_step_v1(ptr:u64,shared_step:u64,generation:u64,leaf:u32,source:u32,stream:*mut c_void)->i32;
    fn prism_physical_guard_evaluate_step_v1(ptr:u64,stream:*mut c_void)->i32;
    fn prism_physical_guard_publish_ceiling_v1(ptr:u64,ceiling:u32,stream:*mut c_void)->i32;
    fn prism_physical_guard_snapshot_v1(ptr:u64,stream:*mut c_void,timeout:u32,out:*mut PhysicalGuardSnapshot)->i32;
    fn prism_physical_guard_destroy_v1(ptr:u64,stream:*mut c_void,timeout:u32)->i32;
    pub fn prism_gearbox_launch_guard_bridge_v1(handle:u64,cruise:*const crate::gearbox::ChronometricStateTensor,governor:u64,stream:*mut c_void,trace:u64)->i32;
    pub fn prism_gearbox_populate_guard_bodies_v1(bodies:*mut cudarc::driver::sys::CUgraph,governor:u64)->i32;
    fn prism_gearbox_governor_accumulate_work_v2(device:u64,step_work:*const f64,stream:*mut c_void)->i32;
    fn prism_gearbox_governor_create_v2(
        out: *mut u64,
        policy: *const GearGovernorPolicy,
        source: u32,
        generation: u64,
        clusters: u32,
        stream: *mut c_void,
    ) -> i32;
    fn prism_gearbox_governor_destroy_v2(device: u64, stream: *mut c_void, timeout_ms: u32) -> i32;
    fn prism_gearbox_governor_snapshot_v2(
        device: u64,
        stream: *mut c_void,
        timeout_ms: u32,
        out: *mut GovernorSnapshot,
        constraints: *mut ConstraintWindow,
    ) -> i32;
    fn prism_gearbox_governor_begin_step_v2(device: u64, stream: *mut c_void) -> i32;
    fn prism_gearbox_governor_seal_step_v2(device: u64, stream: *mut c_void) -> i32;
    fn prism_gearbox_governor_force_complete_v2(
        device: u64,
        producer: u32,
        stream: *mut c_void,
    ) -> i32;
    fn prism_gearbox_governor_measure_force_v2(
        device: u64,
        forces: *const c_void,
        n: u32,
        stream: *mut c_void,
    ) -> i32;
    pub fn prism_gearbox_launch_sfa_v2(
        adj: *const crate::interferometric_adjudicator::InterferometricAdjudicatorFfi,
        cruise: *mut crate::gearbox::ChronometricStateTensor,
        device: u64,
        stream: *mut c_void,
    ) -> i32;
}

pub struct GovernorDevice {
    device: u64,
    stream: Arc<CudaStream>,
    source_stream_id: u32,
    generation: u64,
    clusters: u32,
    physical_guard: AtomicU64,
    // Exact retained energy-record generation through each closed prefix.
    continuation_energy_generations: std::sync::Mutex<Vec<(u64,u64)>>,
}
impl GovernorDevice {
    pub fn new(
        stream: Arc<CudaStream>,
        policy: &GearGovernorPolicy,
        source_stream_id: u32,
        clusters: usize,
    ) -> Result<Arc<Self>, i32> {
        let context_rc = unsafe { cudarc::driver::sys::cuCtxSetCurrent(stream.context().cu_ctx()) };
        if context_rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(context_rc as i32);
        }
        let clusters = u32::try_from(clusters).map_err(|_| 1)?;
        let generation = NEXT_GENERATION.fetch_add(1, Ordering::Relaxed);
        if generation == 0 || stream.cu_stream().is_null() {
            return Err(1);
        }
        let mut device = 0;
        let rc = unsafe {
            prism_gearbox_governor_create_v2(
                &mut device,
                policy,
                source_stream_id,
                generation,
                clusters,
                stream.cu_stream().cast(),
            )
        };
        let owner = Self {
            device,
            stream,
            source_stream_id,
            generation,
            clusters,
            physical_guard: AtomicU64::new(0),
            continuation_energy_generations: std::sync::Mutex::new(Vec::new()),
        };
        if rc != 0 {
            drop(owner);
            return Err(rc);
        }
        Ok(Arc::new(owner))
    }
    pub fn device_ptr(&self) -> u64 {
        self.device
    }
    pub fn has_physical_guard(&self)->bool { self.physical_guard.load(Ordering::Acquire)!=0 }
    /// Actual retained guard address for an ordered same-stream collective reader.
    pub fn physical_guard_device_ptr(&self)->u64 { self.physical_guard.load(Ordering::Acquire) }
    /// Bind live engine arrays once, before any capture. Their owner must outlive all graph leases.
    pub unsafe fn bind_physical_guard(&self,mut bindings:PhysicalGuardBindings)->Result<(),i32> {
        if self.has_physical_guard() {return Err(1);}
        bindings.abi=1;bindings.byte_size=std::mem::size_of::<PhysicalGuardBindings>() as u32;
        bindings.source=self.source_stream_id;bindings.generation=self.generation;bindings.governor=self.device;
        let mut ptr=0;
        let rc=prism_physical_guard_create_v1(&mut ptr,&bindings,self.stream.cu_stream().cast());
        self.physical_guard.store(ptr,Ordering::Release);
        result(rc)
    }
    /// Caller keeps the owning ProtocolState alive through all captured readers.
    /// This binds only at physical step zero, before any graph capture.
    pub unsafe fn bind_physical_time(&self,step:u64,capacity:u64)->Result<(),i32> {
        result(cudarc::driver::sys::cuCtxSetCurrent(self.stream.context().cu_ctx()) as i32)?;
        let ptr=self.physical_guard.load(Ordering::Acquire);
        if ptr==0 {return Err(1);}
        result(prism_physical_guard_bind_time_v1(ptr,step,capacity,self.stream.cu_stream().cast()))
    }
    /// Read-only ownership proof for the existing single physical-time journal.
    /// Offsets are the same sealed416B guard ABI used by continuation below;
    /// no policy, dt, propagation plane, journal, RNG or clock state is written.
    pub fn unarmed_physical_clock_state(&self,expected:&PhysicalGuardBindings,step_pointer:u64,
        capacity_steps:u64,completed_step:u64)->anyhow::Result<UnarmedPhysicalClockStateV1> {
        use crate::scientific_checkpoint::device::read_raw;
        use anyhow::{ensure,Context};
        self.stream.synchronize()?;
        let pointer=self.physical_guard.load(Ordering::Acquire);
        ensure!(pointer!=0 && capacity_steps>0 && completed_step<=capacity_steps,"unarmed native clock lacks its admitted guard/horizon");
        let governor=unsafe{read_raw(&self.stream,self.device,624)?};
        ensure!(checkpoint_u32(&governor,0)?==2 && checkpoint_u32(&governor,4)?==384
            && checkpoint_u32(&governor,8)?==self.source_stream_id && checkpoint_u64(&governor,16)?==self.generation
            && checkpoint_u64(&governor,200)?==pointer,
            "native physical clock governor identity/ABI differs");
        // policy is the existing144B member at32; validate its actual device
        // armed word, not a host declaration or only the mutable echo.
        let armed=32+std::mem::offset_of!(GearGovernorPolicy,armed);
        ensure!(checkpoint_u32(&governor,armed)?==0,"raw Ghost clock requires the actual unarmed physical policy");
        let header=unsafe{read_raw(&self.stream,pointer,416)?};
        ensure!(checkpoint_u32(&header,0)?==1 && checkpoint_u32(&header,4)?==192
            && checkpoint_u32(&header,8)?==self.source_stream_id && checkpoint_u64(&header,16)?==self.generation
            && checkpoint_u32(&header,192)?==1 && checkpoint_u32(&header,196)?==80
            && checkpoint_u32(&header,200)?==self.source_stream_id && checkpoint_u32(&header,204)?==expected.n_atoms
            && checkpoint_u64(&header,208)?==self.generation && checkpoint_u64(&header,216)?==self.device,
            "native physical clock guard/provider/source/atom binding differs");
        for (offset,wanted) in [(224,expected.forces),(232,expected.potential),(240,expected.velocities),
            (248,expected.masses),(256,expected.dt),(264,expected.step_work),(336,step_pointer)] {
            ensure!(wanted!=0 && checkpoint_u64(&header,offset)?==wanted,"native physical clock live pointer binding differs at{}",offset);
        }
        let capacity=checkpoint_u64(&header,344)?;let count=checkpoint_u64(&header,352)?;
        let total=f64::from_bits(checkpoint_u64(&header,360)?);
        ensure!(capacity==capacity_steps && count==completed_step && checkpoint_u64(&header,24)?==count
            && checkpoint_u64(&header,280)?==count && checkpoint_u32(&header,296)?==0
            && checkpoint_u32(&header,12)?==if count==0{0}else{2}
            && total.is_finite() && total>=0.0 && (count!=0 || total.to_bits()==0),
            "native physical clock is unbound, open, failed, or differs from its actual physical endpoint");
        let time=checkpoint_u64(&header,328)?;let energy=checkpoint_u64(&header,400)?;
        let time_bytes=capacity.checked_mul(24).context("native clock span overflow")?;
        let journal_bytes=capacity.checked_mul(184).context("native time/energy span overflow")?;
        let journal_end=time.checked_add(journal_bytes).context("native clock pointer overflow")?;
        ensure!(time!=0 && energy==time.checked_add(time_bytes).context("native energy tail pointer overflow")?,
            "native clock/energy journal is not the original single owning allocation");
        let atoms=u64::from(expected.n_atoms);
        for (start,bytes) in [(expected.forces,atoms*12),(expected.potential,atoms*8),(expected.velocities,atoms*12),
            (expected.masses,atoms*4),(expected.dt,4),(expected.step_work,8),(step_pointer,4)] {
            let end=start.checked_add(bytes).context("native physical plane span overflow")?;
            ensure!(end<=time || journal_end<=start,"native clock journal aliases a physical state/control plane");
        }
        Ok(UnarmedPhysicalClockStateV1{schema_version:1,source_stream_id:self.source_stream_id,allocation_generation:self.generation,
            capacity_steps:capacity,completed_step:count,cumulative_simulated_time_ps_bits:total.to_bits(),
            applied_dt_ps_bits:checkpoint_u64(&header,368)?,clock_policy_armed:0})
    }
    pub fn physical_time_records(&self,start:u64,end:u64)->Result<Vec<PhysicalTimeRecord>,i32> {
        result(unsafe {cudarc::driver::sys::cuCtxSetCurrent(self.stream.context().cu_ctx())} as i32)?;
        let ptr=self.physical_guard.load(Ordering::Acquire);
        if ptr==0 || end<=start {return Err(1);}
        let count=usize::try_from(end-start).map_err(|_|1)?;
        let mut rows=vec![PhysicalTimeRecord::default();count];
        result(unsafe{prism_physical_guard_read_time_v1(ptr,start,end-start,rows.as_mut_ptr(),self.stream.cu_stream().cast(),CLOSE_TIMEOUT_MS)})?;
        for (index,row) in rows.iter().enumerate() {
            if row.step!=start+index as u64+1 || !row.applied_dt_ps.is_finite() || row.applied_dt_ps<=0.0 ||
                !row.cumulative_simulated_time_ps.is_finite() || row.cumulative_simulated_time_ps<=0.0 {return Err(1);}
            if index>0 && rows[index-1].cumulative_simulated_time_ps+row.applied_dt_ps!=row.cumulative_simulated_time_ps {return Err(1);}
        }
        Ok(rows)
    }
    /// Read the original native energy rows for completed steps (start,end].
    /// No interpolation, potential-reference substitution or invented bootstrap.
    pub fn physical_energy_balance_records(&self,start:u64,end:u64)->Result<Vec<PhysicalEnergyBalanceRecord>,i32> {
        result(unsafe {cudarc::driver::sys::cuCtxSetCurrent(self.stream.context().cu_ctx())} as i32)?;
        let ptr=self.physical_guard.load(Ordering::Acquire);
        if ptr==0 || end<=start {return Err(1);}
        let count=usize::try_from(end-start).map_err(|_|1)?;
        count.checked_mul(PHYSICAL_ENERGY_BALANCE_RECORD_BYTES).ok_or(1)?;
        let mut rows=vec![PhysicalEnergyBalanceRecord::default();count];
        result(unsafe{prism_physical_guard_read_energy_v1(ptr,start,end-start,rows.as_mut_ptr(),self.stream.cu_stream().cast(),CLOSE_TIMEOUT_MS)})?;
        for (index,row) in rows.iter().enumerate() {
            if row.abi!=1 || row.byte_size as usize!=PHYSICAL_ENERGY_BALANCE_RECORD_BYTES ||
                row.source!=self.source_stream_id || row.generation!=self.energy_generation_for_step(row.step)? ||
                row.normalization_contract!=1 || row.reserved!=0 ||
                row.step!=start+index as u64+1 || row.physical_epoch!=row.step ||
                row.previous_epoch.checked_add(1)!=Some(row.physical_epoch) ||
                !row.current_potential.is_finite() || !row.current_kinetic.is_finite() || row.current_kinetic<0.0 ||
                !row.body_work.is_finite() || !row.carried_transition_work.is_finite() || !row.observed_work.is_finite() ||
                row.body_work+row.carried_transition_work!=row.observed_work ||
                !row.applied_dt_ps.is_finite() || row.applied_dt_ps<=0.0 ||
                !row.cumulative_simulated_time_ps.is_finite() || row.cumulative_simulated_time_ps<=0.0 {return Err(1);}
            if row.previous_epoch==0 {
                if row.history_status!=0 || row.step!=1 ||
                    ![row.previous_potential,row.previous_kinetic,row.signed_residual,row.kinetic_reference,
                      row.normalized_residual,row.legacy_h_ratio].iter().all(|x|x.is_nan()) {return Err(1);}
            } else {
                if !row.previous_potential.is_finite() {return Err(1);}
                let valid_reference=row.previous_kinetic.is_finite()&&row.previous_kinetic>0.0;
                if valid_reference {
                    if row.history_status!=1 || row.kinetic_reference!=row.previous_kinetic ||
                        !row.signed_residual.is_finite() || !row.normalized_residual.is_finite() || row.normalized_residual<0.0 {return Err(1);}
                    let residual=((row.current_potential-row.previous_potential)+(row.current_kinetic-row.previous_kinetic))-row.observed_work;
                    if residual!=row.signed_residual {return Err(1);}
                    let expected=residual.abs()/row.previous_kinetic;
                    if (expected-row.normalized_residual).abs()>16.0*f64::EPSILON*expected.abs().max(f64::MIN_POSITIVE) {return Err(1);}
                } else if row.history_status!=2 || !row.normalized_residual.is_nan() {return Err(1);}
                // Preserve legacy finite/nonfinite diagnostics without gating this contract.
            }
            if index>0 {
                let previous=&rows[index-1];
                if previous.cumulative_simulated_time_ps+row.applied_dt_ps!=row.cumulative_simulated_time_ps ||
                    row.previous_potential!=previous.current_potential || row.previous_kinetic!=previous.current_kinetic {return Err(1);}
            }
        }
        Ok(rows)
    }

    pub fn evaluate_physical_step(&self,stream:&CudaStream)->Result<(),i32> {
        if !self.owns_stream(stream) {return Err(1);}
        let ptr=self.physical_guard.load(Ordering::Acquire);
        if ptr==0 {return Ok(());}
        result(unsafe{prism_physical_guard_evaluate_step_v1(ptr,stream.cu_stream().cast())})
    }
    pub fn publish_physical_ceiling(&self,ceiling:u32)->Result<(),i32> {
        let ptr=self.physical_guard.load(Ordering::Acquire);
        if ptr==0 {return Err(1);}
        result(unsafe{prism_physical_guard_publish_ceiling_v1(ptr,ceiling,self.stream.cu_stream().cast())})
    }
    pub fn physical_guard_snapshot(&self)->Result<Option<PhysicalGuardSnapshot>,i32> {
        let ptr=self.physical_guard.load(Ordering::Acquire);
        if ptr==0 {return Ok(None);}
        let mut snap=unsafe{std::mem::zeroed()};
        result(unsafe{prism_physical_guard_snapshot_v1(ptr,self.stream.cu_stream().cast(),CLOSE_TIMEOUT_MS,&mut snap)})?;
        Ok(Some(snap))
    }
    pub fn receipt(&self)->Result<serde_json::Value,i32> {
        let (snapshot,constraints)=self.snapshot()?;
        let mut receipt=snapshot.receipt(&constraints);
        if let Some(guard)=self.physical_guard_snapshot()? {
            receipt["physical_guard"]=serde_json::to_value(guard).map_err(|_|1)?;
            receipt["physical_guard"]["science_measurement_class"]=serde_json::json!(if guard.science_status==0 {"UNMEASURED"} else {"COMPLETED_OBSERVER_PUBLICATION"});
            receipt["physical_guard"]["energy_measurement_policy_identity"]=serde_json::json!({
                "normalization":physical_energy_balance_contract(),
                "governor_policy_digest":snapshot.state.policy_digest,
                "identity_scope":"ordered association of explicit measurement contract and existing numerical-band policy digest"});
            receipt["physical_guard"]["evaluation_scope"]=serde_json::json!("EVERY_COMPLETED_ORDINARY_PHYSICAL_STEP");
            receipt["physical_guard"]["transition_work_scope"]=serde_json::json!("EXACT_DELTA_K_CARRIED_TO_NEXT_PHYSICAL_INTERVAL");
        }
        Ok(receipt)
    }
    pub fn observation_work_ptr(&self)->u64 { self.device+584 }
    /// Read the completed per-step work scalar; the engine owns its capture lifetime.
    pub unsafe fn accumulate_step_work(&self,stream:&CudaStream,step_work:u64)->Result<(),i32> {
        if !self.owns_stream(stream) {return Err(1);}
        result(prism_gearbox_governor_accumulate_work_v2(self.device,step_work as *const f64,stream.cu_stream().cast()))
    }
    pub fn cruise_state_ptr(&self) -> u64 {
        self.device + 592
    }
    pub fn constraint_window_ptr(&self) -> u64 {
        self.device + 448
    }
    pub fn force_value_ptr(&self) -> u64 {
        self.device + 384 + 48
    }
    pub fn source_stream_id(&self) -> u32 {
        self.source_stream_id
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }
    pub fn n_constraint_clusters(&self) -> usize {
        self.clusters as usize
    }
    pub fn owns_stream(&self, stream: &CudaStream) -> bool {
        self.stream.cu_stream() == stream.cu_stream()
    }
    pub fn owns_raw_stream(&self, stream: *mut c_void) -> bool {
        self.stream.cu_stream().cast::<c_void>() == stream
    }
    pub fn begin_step(&self, stream: &CudaStream) -> Result<(), i32> {
        if !self.owns_stream(stream) {
            return Err(1);
        }
        result(unsafe {
            prism_gearbox_governor_begin_step_v2(self.device, stream.cu_stream().cast())
        })?;
        let ptr=self.physical_guard.load(Ordering::Acquire);
        if ptr==0 {return Ok(());}
        result(unsafe{prism_physical_guard_begin_step_v1(ptr,stream.cu_stream().cast())})
    }
    /// Explicit selected measurement profile; ordinary begin remains unchanged.
    /// The retained record must stay selected and immutable through evaluation.
    pub unsafe fn begin_selected_step(&self,stream:&CudaStream,shared:u64,generation:u64,leaf:u32)->Result<(),i32>{
        if !self.owns_stream(stream)||shared==0||shared%128!=0||generation==0||leaf>=8{return Err(1);}
        let ptr=self.physical_guard.load(Ordering::Acquire);if ptr==0{return Err(1);}
        result(prism_gearbox_governor_begin_step_v2(self.device,stream.cu_stream().cast()))?;
        result(prism_physical_guard_begin_selected_step_v1(ptr,shared,generation,leaf,self.source_stream_id(),stream.cu_stream().cast()))
    }
    pub fn seal_step(&self, stream: &CudaStream) -> Result<(), i32> {
        if !self.owns_stream(stream) {
            return Err(1);
        }
        result(unsafe {
            prism_gearbox_governor_seal_step_v2(self.device, stream.cu_stream().cast())
        })
    }
    pub fn force_complete(&self, stream: &CudaStream, producer: u32) -> Result<(), i32> {
        if !self.owns_stream(stream) {
            return Err(1);
        }
        result(unsafe {
            prism_gearbox_governor_force_complete_v2(
                self.device,
                producer,
                stream.cu_stream().cast(),
            )
        })
    }
    /// The caller owns the live native AoS float3 force buffer for the capture
    /// lifetime; no values are read or substituted on the host.
    pub unsafe fn measure_force(
        &self,
        stream: &CudaStream,
        forces: *const c_void,
        n: u32,
    ) -> Result<(), i32> {
        if !self.owns_stream(stream) {
            return Err(1);
        }
        result(prism_gearbox_governor_measure_force_v2(
            self.device,
            forces,
            n,
            stream.cu_stream().cast(),
        ))
    }
    pub fn snapshot(&self) -> Result<(GovernorSnapshot, ConstraintWindow), i32> {
        let context_rc =
            unsafe { cudarc::driver::sys::cuCtxSetCurrent(self.stream.context().cu_ctx()) };
        if context_rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(context_rc as i32);
        }
        let mut snap = unsafe { std::mem::zeroed() };
        let mut constraints = unsafe { std::mem::zeroed() };
        result(unsafe {
            prism_gearbox_governor_snapshot_v2(
                self.device,
                self.stream.cu_stream().cast(),
                CLOSE_TIMEOUT_MS,
                &mut snap,
                &mut constraints,
            )
        })?;
        Ok((snap, constraints))
    }
}
fn result(rc: i32) -> Result<(), i32> {
    if rc == 0 {
        Ok(())
    } else {
        Err(rc)
    }
}
impl Drop for GovernorDevice {
    fn drop(&mut self) {
        if self.device == 0 {
            return;
        }
        let context_rc =
            unsafe { cudarc::driver::sys::cuCtxSetCurrent(self.stream.context().cu_ctx()) };
        let rc = if context_rc == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            let guard=self.physical_guard.load(Ordering::Acquire);
            let guard_rc=if guard!=0 {unsafe{prism_physical_guard_destroy_v1(guard,self.stream.cu_stream().cast(),CLOSE_TIMEOUT_MS)}} else {0};
            if guard_rc!=0 {
                log::error!("PHYSICAL_GUARD_QUARANTINED source={} generation={} device={:#x} rc={}",self.source_stream_id,self.generation,guard,guard_rc);
                std::mem::forget(self.stream.clone());
                return;
            }
            unsafe {
                prism_gearbox_governor_destroy_v2(
                    self.device,
                    self.stream.cu_stream().cast(),
                    CLOSE_TIMEOUT_MS,
                )
            }
        } else {
            context_rc as i32
        };
        if rc != 0 {
            // Keep the context/stream alive along with the deliberately unfreed
            // device allocation. Unknown completion is not permission to free.
            log::error!(
                "GOVERNOR_QUARANTINED source={} generation={} device={:#x} rc={}",
                self.source_stream_id,
                self.generation,
                self.device,
                rc
            );
            std::mem::forget(self.stream.clone());
        }
    }
}

pub fn window_status_label(status: u32) -> &'static str {
    match status {
        0 => "UNMEASURED",
        1 => "OPEN",
        2 => "COMPLETE",
        3 => "NOT_APPLICABLE",
        4 => "FAILED_DIAGNOSTIC",
        5 => "BAD_SCHEMA",
        6 => "OWNER_MISMATCH",
        7 => "STALE",
        8 => "INCOMPLETE_COVERAGE",
        _ => "UNKNOWN_STATUS",
    }
}
impl GovernorSnapshot {
    pub fn receipt(&self, constraints: &ConstraintWindow) -> serde_json::Value {
        let norm = self.state.observed_force_norm;
        serde_json::json!({
            "abi_version": self.abi_version, "source_stream_id": self.source_stream_id,
            "allocation_generation": self.allocation_generation, "binding_epoch": self.binding_epoch,
            "decision_epoch": self.decision_epoch, "physical_epoch": self.last_physical_epoch,
            "observer_latency_bound_physical_steps": self.state.downshift_latency_steps,
            "measurement_window_scope": "LATEST_COMPLETED_PHYSICAL_STEP",
            "external_work_scope": "SUM_OF_COMPLETED_STEP_WORK_SINCE_PREVIOUS_SFA_CONSUMPTION",
            "policy_digest": format!("{:016x}", self.state.policy_digest),
            "armed_echo": self.state.governor_armed, "authorized_gear": self.state.authorized_gear,
            "binding_gate_code": self.state.binding_gate,
            "binding_gate": crate::gearbox::gear_gate_label(self.state.binding_gate),
            "force_status": window_status_label(self.force_status),
            "force_epoch": self.observed_force_epoch,
            "force_value_class": if self.force_status != 2 { "UNMEASURED" } else if norm.is_finite() { "FINITE" } else { "MEASURED_NONFINITE" },
            "observed_force_norm": if norm.is_finite() && self.force_status == 2 { Some(norm) } else { None },
            "constraint_status": window_status_label(self.constraint_status),
            "constraint_epoch": self.observed_constraint_epoch,
            "constraint_phase_mask": constraints.phase_mask,
            "constraint_clusters": constraints.n_constraint_clusters,
            "expected_cluster_solves": constraints.expected_cluster_solves,
            "completed_cluster_solves": constraints.completed_cluster_solves,
            "constraint_fatal_failures": constraints.fatal_failure_count,
            "reset_reason": self.reset_reason, "cuda_result": 0,
        })
    }
}

extern "C" {
    fn prism_gearbox_stream_drain_v2(stream: *mut c_void, timeout_ms: u32) -> i32;
}
/// A checked, bounded stream join that does not require a governor allocation.
/// It is also used while building graphs, before a governor has been attached.
pub fn drain_stream(stream: &CudaStream) -> Result<(), i32> {
    let context_rc = unsafe { cudarc::driver::sys::cuCtxSetCurrent(stream.context().cu_ctx()) };
    if context_rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        return Err(context_rc as i32);
    }
    result(unsafe { prism_gearbox_stream_drain_v2(stream.cu_stream().cast(), CLOSE_TIMEOUT_MS) })
}

/// Keep any CUDA pointee owner alive on an early `?`/panic as well as normal
/// teardown. The inner value is released only after its stream completed.
/// A failed join quarantines the allocation and its context for process life.
/// This is deliberately independent of whether a governor was constructed.
/// Additional GPU graph users of an allocation owner must retire before it drops.
/// Implementations must prevent future launches, fence in-flight graph use and
/// destroy graph executables. An error requests existing process-life quarantine.
pub trait CudaRetirementLease: Send + Sync {
    fn fence_and_retire(&self) -> Result<(), i32>;
}

pub struct CudaOwned<T> {
    inner: std::mem::ManuallyDrop<T>,
    stream: Arc<CudaStream>,
    retirement_leases: Vec<Arc<dyn CudaRetirementLease>>,
    quarantine_error: Option<i32>,
}
impl<T> CudaOwned<T> {
    pub fn new(inner: T, stream: Arc<CudaStream>) -> Self {
        Self {
            inner: std::mem::ManuallyDrop::new(inner),
            stream,
            retirement_leases: Vec::new(),
            quarantine_error: None,
        }
    }
}
impl<T> CudaOwned<T> {
    /// Preserve this allocation for process lifetime after an enclosing graph
    /// retirement failure. A later successful stream query cannot revoke it.
    pub fn quarantine(&mut self, error: i32) {
        self.quarantine_error.get_or_insert(error);
    }
    pub fn retain_retirement(&mut self, lease: Arc<dyn CudaRetirementLease>) {
        if !self.retirement_leases.iter().any(|old| Arc::ptr_eq(old, &lease)) {
            self.retirement_leases.push(lease);
        }
    }
}
impl<T> std::ops::Deref for CudaOwned<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.inner
    }
}
impl<T> std::ops::DerefMut for CudaOwned<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.inner
    }
}
impl<T> Drop for CudaOwned<T> {
    fn drop(&mut self) {
        let retired = self.retirement_leases.iter().try_for_each(|lease| lease.fence_and_retire());
        match retired
            .and_then(|()| self.quarantine_error.map_or(Ok(()), Err))
            .and_then(|()| drain_stream(&self.stream)) {
            Ok(()) => unsafe { std::mem::ManuallyDrop::drop(&mut self.inner) },
            Err(rc) => {
                log::error!(
                    "CUDA_OWNER_QUARANTINED type={} stream={:p} rc={}",
                    std::any::type_name::<T>(),
                    self.stream.cu_stream(),
                    rc
                );
                std::mem::forget(self.stream.clone());
                for lease in &self.retirement_leases { std::mem::forget(lease.clone()); }
            }
        }
    }
}


/// Complete private governor allocation and guard history at a synchronized
/// scientific boundary. Native addresses are zeroed on export and rebound to
/// fresh owners on import. ABI sizes/offsets are pinned by the static_asserts in
/// gearbox.cuh, prism_governor_measurement.cuh and prism_physical_guard.cuh.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GovernorContinuationV1 {
    schema_version:u32, source:u32, clusters:u32, original_generation:u64,
    #[serde(with="crate::checkpoint_codec::bytes")]
    allocation:Vec<u8>, physical_guard:Option<PhysicalGuardContinuationV1>,
    energy_generations:Vec<(u64,u64)>,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct PhysicalGuardContinuationV1 {
    #[serde(with="crate::checkpoint_codec::bytes")] header:Vec<u8>,
    #[serde(with="crate::checkpoint_codec::bytes")] transition_rows:Vec<u8>,
    #[serde(with="crate::checkpoint_codec::bytes")] time_rows:Vec<u8>,
    #[serde(with="crate::checkpoint_codec::bytes")] energy_rows:Vec<u8>,
}
fn checkpoint_u32(bytes:&[u8],offset:usize)->anyhow::Result<u32> {
    Ok(u32::from_le_bytes(bytes.get(offset..offset+4).ok_or_else(||anyhow::anyhow!("truncated governor state"))?.try_into()?))
}
fn checkpoint_u64(bytes:&[u8],offset:usize)->anyhow::Result<u64> {
    Ok(u64::from_le_bytes(bytes.get(offset..offset+8).ok_or_else(||anyhow::anyhow!("truncated governor state"))?.try_into()?))
}
impl GovernorDevice {
    pub fn continuation_allocation_plan(&self)->anyhow::Result<Vec<(String,u64)>> {
        self.stream.synchronize()?;
        let mut rows=vec![("governor_allocation".into(),624u64.checked_add(u64::from(self.clusters).checked_mul(24)
            .ok_or_else(||anyhow::anyhow!("governor extent overflow"))?).ok_or_else(||anyhow::anyhow!("governor extent overflow"))?)];
        let pointer=self.physical_guard.load(Ordering::Acquire);
        if pointer!=0 {
            // Only this fixed416-byte owner header is read before admission.
            let header=unsafe{crate::scientific_checkpoint::device::read_raw(&self.stream,pointer,416)?};
            let atoms=u64::from(checkpoint_u32(&header,204)?);let count=checkpoint_u64(&header,352)?;
            anyhow::ensure!(count<=checkpoint_u64(&header,344)? && checkpoint_u32(&header,296)?==0,"guard allocation plan crosses an unfinished physical step");
            rows.push(("guard_header".into(),416));
            for(name,bytes)in [("transition_rows",atoms.checked_mul(8)),("original_time_rows",count.checked_mul(24)),("original_energy_rows",count.checked_mul(160))] {
                rows.push((name.into(),bytes.ok_or_else(||anyhow::anyhow!("guard allocation plan overflow"))?));
            }
        }
        let generations=self.continuation_energy_generations.lock().map_err(|_|anyhow::anyhow!("energy lineage poisoned"))?;
        rows.push(("energy_generation_lineage_wire".into(),crate::checkpoint_codec::encoded_size(&*generations,u64::MAX)?));Ok(rows)
    }
    pub fn capture_continuation(&self)->anyhow::Result<GovernorContinuationV1> {
        use crate::scientific_checkpoint::device::read_raw;
        self.stream.synchronize()?;
        let nbytes=624usize.checked_add((self.clusters as usize).checked_mul(24)
            .ok_or_else(||anyhow::anyhow!("governor checkpoint extent overflow"))?)
            .ok_or_else(||anyhow::anyhow!("governor checkpoint extent overflow"))?;
        let mut allocation=unsafe{read_raw(&self.stream,self.device,nbytes)?};
        anyhow::ensure!(checkpoint_u32(&allocation,0)?==2 && checkpoint_u32(&allocation,4)?==384
            && checkpoint_u32(&allocation,8)?==self.source_stream_id
            && checkpoint_u64(&allocation,16)?==self.generation
            && checkpoint_u32(&allocation,464)?==self.clusters,"governor checkpoint ABI/owner mismatch");
        // Validate every internal link against this native contiguous owner.
        for (offset,target) in [(176,384),(184,448),(192,576),(568,624)] {
            anyhow::ensure!(checkpoint_u64(&allocation,offset)?==self.device+target,"governor pointer lies outside declared owner");
            allocation[offset..offset+8].fill(0);
        }
        let guard_backlink=self.physical_guard.load(Ordering::Acquire);
        anyhow::ensure!(checkpoint_u64(&allocation,200)?==guard_backlink,"governor physical-guard backlink differs from owning allocation");
        allocation[200..208].fill(0);
        // An open physical window is not a completed integrator boundary.
        anyhow::ensure!(checkpoint_u32(&allocation,388)?!=1 && checkpoint_u32(&allocation,452)?!=1,
            "governor force/constraint window remains open");
        let pointer=self.physical_guard.load(Ordering::Acquire);
        let physical_guard=if pointer==0 {None} else {
            let mut header=unsafe{read_raw(&self.stream,pointer,416)?};
            let atoms=checkpoint_u32(&header,204)? as usize;
            anyhow::ensure!(checkpoint_u32(&header,0)?==1 && checkpoint_u32(&header,4)?==192
                && checkpoint_u32(&header,192)?==1 && checkpoint_u32(&header,196)?==80
                && checkpoint_u32(&header,200)?==self.source_stream_id
                && checkpoint_u64(&header,208)?==self.generation && atoms>0,
                "physical guard private state ABI/owner mismatch");
            anyhow::ensure!(checkpoint_u32(&header,296)?==0,"physical guard step is in flight");
            let count=checkpoint_u64(&header,352)?;let capacity=checkpoint_u64(&header,344)?;
            anyhow::ensure!(count<=capacity,"physical guard journal count exceeds capacity");
            let transition_rows=unsafe{read_raw(&self.stream,checkpoint_u64(&header,272)?,atoms.checked_mul(8)
                .ok_or_else(||anyhow::anyhow!("transition state extent overflow"))?)?};
            let time_rows=unsafe{read_raw(&self.stream,checkpoint_u64(&header,328)?,usize::try_from(count)?.checked_mul(24)
                .ok_or_else(||anyhow::anyhow!("clock state extent overflow"))?)?};
            let energy_rows=unsafe{read_raw(&self.stream,checkpoint_u64(&header,400)?,usize::try_from(count)?.checked_mul(160)
                .ok_or_else(||anyhow::anyhow!("energy state extent overflow"))?)?};
            for offset in [216,224,232,240,248,256,264,272,328,336,400] {header[offset..offset+8].fill(0);}
            Some(PhysicalGuardContinuationV1{header,transition_rows,time_rows,energy_rows})
        };
        Ok(GovernorContinuationV1{schema_version:1,source:self.source_stream_id,clusters:self.clusters,
            original_generation:self.generation,allocation,physical_guard,
            energy_generations:self.continuation_energy_generations.lock().map_err(|_|anyhow::anyhow!("poisoned energy generation lineage"))?.clone()})
    }
    /// Rebind allocation generation and all device addresses. Scientific
    /// counters/history remain unchanged; the new allocation identity is an
    /// explicit native restart event, not a reset of the physical epoch.
    pub fn restore_continuation(&self,saved:&GovernorContinuationV1)->anyhow::Result<()> {
        use crate::scientific_checkpoint::device::{read_raw,write_raw};
        let fresh=self.capture_continuation()?;
        anyhow::ensure!(saved.schema_version==1 && saved.source==self.source_stream_id && saved.clusters==self.clusters
            && saved.allocation.len()==fresh.allocation.len(),"governor continuation identity/extent mismatch");
        anyhow::ensure!(saved.allocation[32..176]==fresh.allocation[32..176],"governor policy differs on restart");
        let mut energy_generations=saved.energy_generations.clone();
        let mut previous_end=0u64;
        for &(generation,end) in &energy_generations {
            anyhow::ensure!(generation!=0 && end>previous_end,"invalid saved energy generation lineage");previous_end=end;
        }
        let restored_count=match &saved.physical_guard{Some(guard)=>checkpoint_u64(&guard.header,352)?,None=>0};
        anyhow::ensure!(previous_end<=restored_count,"generation lineage exceeds committed physical history");
        if restored_count>previous_end {energy_generations.push((saved.original_generation,restored_count));}
        if let Some(guard)=&saved.physical_guard {
            for (index,row) in guard.energy_rows.chunks_exact(160).enumerate(){
                let step=index as u64+1;
                let generation=energy_generations.iter().find(|(_,end)|step<=*end).map(|(g,_)|*g)
                    .ok_or_else(||anyhow::anyhow!("energy row outside bound historical generation"))?;
                anyhow::ensure!(checkpoint_u64(row,24)?==generation && checkpoint_u64(row,32)?==step
                    && checkpoint_u32(row,8)?==self.source_stream_id,"energy history generation/step/source mismatch");
            }
        }
        let mut allocation=saved.allocation.clone();
        for (offset,target) in [(176,384),(184,448),(192,576),(568,624)] {
            anyhow::ensure!(checkpoint_u64(&allocation,offset)?==0,"serialized governor contains a device pointer");
            allocation[offset..offset+8].copy_from_slice(&(self.device+target).to_le_bytes());
        }
        for offset in [16,400,480] {
            anyhow::ensure!(checkpoint_u64(&allocation,offset)?==saved.original_generation,"governor saved generation inconsistent");
            allocation[offset..offset+8].copy_from_slice(&self.generation.to_le_bytes());
        }
        let guard_pointer=self.physical_guard.load(Ordering::Acquire);
        anyhow::ensure!(checkpoint_u64(&allocation,200)?==0,"serialized governor contains a live physical-guard backlink");
        allocation[200..208].copy_from_slice(&guard_pointer.to_le_bytes());
        let prepared=match (&saved.physical_guard,&fresh.physical_guard) {
            (None,None)=>None,
            (Some(saved),Some(_))=>{
                anyhow::ensure!(saved.header.len()==416,"physical guard header truncated");
                let current=unsafe{read_raw(&self.stream,guard_pointer,416)?};
                anyhow::ensure!(checkpoint_u32(&saved.header,204)?==checkpoint_u32(&current,204)?
                    && checkpoint_u64(&saved.header,344)?==checkpoint_u64(&current,344)?,
                    "physical guard population/capacity changed on restart");
                let count=checkpoint_u64(&saved.header,352)?;let capacity=checkpoint_u64(&saved.header,344)?;
                let atoms=checkpoint_u32(&saved.header,204)? as usize;
                anyhow::ensure!(count<=capacity && saved.time_rows.len()==usize::try_from(count)?.checked_mul(24)
                    .ok_or_else(||anyhow::anyhow!("clock extent overflow"))?
                    && saved.energy_rows.len()==usize::try_from(count)?.checked_mul(160).ok_or_else(||anyhow::anyhow!("energy extent overflow"))?
                    && saved.transition_rows.len()==atoms.checked_mul(8).ok_or_else(||anyhow::anyhow!("transition extent overflow"))?,
                    "physical guard history population mismatch");
                let mut header=saved.header.clone();
                for offset in [216,224,232,240,248,256,264,272,328,336,400] {
                    anyhow::ensure!(checkpoint_u64(&header,offset)?==0,"serialized guard contains a device pointer");
                    header[offset..offset+8].copy_from_slice(&current[offset..offset+8]);
                }
                // Immutable live bindings carry the freshly reconstructed graph
                // pointers, source and allocation generation as a single unit.
                header[192..272].copy_from_slice(&current[192..272]);
                header[16..24].copy_from_slice(&self.generation.to_le_bytes());
                Some((header,saved,checkpoint_u64(&current,272)?,checkpoint_u64(&current,328)?,checkpoint_u64(&current,400)?))
            },
            _=>anyhow::bail!("physical guard owner presence changed on restart"),
        };
        // All schemas, lengths and links were checked before the first write.
        unsafe{write_raw(&self.stream,self.device,&allocation)?;}
        if let Some((header,saved,transition,time,energy))=prepared {
            unsafe {
                write_raw(&self.stream,transition,&saved.transition_rows)?;
                write_raw(&self.stream,time,&saved.time_rows)?;
                write_raw(&self.stream,energy,&saved.energy_rows)?;
                write_raw(&self.stream,guard_pointer,&header)?;
            }
        }
        self.stream.synchronize()?;
        *self.continuation_energy_generations.lock().map_err(|_|anyhow::anyhow!("poisoned energy generation lineage"))?=energy_generations;
        Ok(())
    }
}

impl GovernorDevice {
    fn energy_generation_for_step(&self,step:u64)->Result<u64,i32>{
        let lineage=self.continuation_energy_generations.lock().map_err(|_|1)?;
        Ok(lineage.iter().find(|(_,end)|step<=*end).map_or(self.generation,|(generation,_)|*generation))
    }
    /// Original row generations remain unchanged; this maps their closed
    /// physical prefixes to the current freshly allocated owner for consumers.
    pub fn continuation_generation_lineage(&self)->Result<Vec<(u64,u64)>,i32>{
        self.continuation_energy_generations.lock().map(|v|v.clone()).map_err(|_|1)
    }
}
