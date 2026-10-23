//! Stream-owned ASC controls; production applies them at both physical forces.
#![cfg(feature = "gpu")]
use crate::asc_support::{AscSelectedSupport, AscSelectedSupportPolicy};
use crate::input::PrismPrepTopology;
use cudarc::driver::CudaStream;
use std::{
    ffi::c_void,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};
pub mod measurement;
static GENERATION: AtomicU64 = AtomicU64::new(1);
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct AscControl {
    pub source: u32,
    pub mode: u32,
    pub valid: u32,
    pub status: u32,
    pub producer_epoch: u64,
    pub producer_frame: u64,
    pub support_identity: u64,
    pub generation: u64,
    pub strength: f32,
    pub center: [f32; 3],
    pub topology: [u64; 4],
}
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct AscPhase {
    pub composed_atoms: u64,
    pub consumed_atoms: u64,
    pub nonzero_force_atoms: u64,
    pub nonzero_impulse_atoms: u64,
    pub force: [f64; 3],
    pub torque: [f64; 3],
    pub impulse: [f64; 3],
    pub velocity_delta: [f64; 3],
    pub differential_kick_energy: f64,
    pub potential: f64,
}
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
pub struct AscSnapshot {
    pub abi: u32,
    pub byte_size: u32,
    pub source: u32,
    pub n_atoms: u32,
    pub generation: u64,
    pub physical_epoch: u64,
    pub completed_epoch: u64,
    pub published_epoch: u64,
    pub topology: [u64; 4],
    pub alpha: f32,
    pub mode: u32,
    pub status: u32,
    pub reserved: u32,
    pub active: AscControl,
    pub previous: AscControl,
    pub pending: AscControl,
    pub phases: [AscPhase; 2],
    pub control_work: f64,
    pub coverage_failures: u64,
    pub decisions_produced: u64,
    pub composed_steps: u64,
    pub consumed_steps: u64,
    pub reserved2: u64,
    pub rows: u64,
    pub reserved3: u64,
}
const _: () = assert!(std::mem::size_of::<AscSnapshot>() == 736);
const _: () = assert!(std::mem::size_of::<AscControl>() == 96);
const _: () = assert!(std::mem::size_of::<AscPhase>() == 144);
extern "C" {
    fn prism_asc_create_v1(
        out: *mut u64,
        source: u32,
        generation: u64,
        atoms: u32,
        topology: *const u64,
        alpha: f32,
        mode: u32,
        stream: *mut c_void,
    ) -> i32;
    fn prism_asc_create_selected_v2(
        out: *mut u64,
        source: u32,
        generation: u64,
        atoms: u32,
        topology: *const u64,
        alpha: f32,
        support_identity: u64,
        membership_host: *const u8,
        selected_count: u32,
        stream: *mut c_void,
    ) -> i32;
    fn prism_asc_destroy_v1(ptr: u64, stream: *mut c_void) -> i32;
    fn prism_asc_begin_step_v1(ptr: u64, stream: *mut c_void) -> i32;
    fn prism_asc_seal_step_v1(ptr: u64, stream: *mut c_void) -> i32;
    fn prism_asc_publish_decision_v1(
        ptr: u64,
        adj: *const crate::interferometric_adjudicator::InterferometricAdjudicatorFfi,
        stream: *mut c_void,
    ) -> i32;
    fn prism_asc_publish_decision_selected_v2(
        ptr: u64,
        adj: *const crate::interferometric_adjudicator::InterferometricAdjudicatorFfi,
        stream: *mut c_void,
    ) -> i32;
    fn prism_asc_compose_force_v1(
        ptr: u64,
        n: u32,
        phase: u32,
        x: *const f32,
        f: *mut f32,
        pe: *mut f64,
        work: *mut f64,
        stream: *mut c_void,
    ) -> i32;
    fn prism_asc_compose_force_selected_v2(
        ptr: u64,
        n: u32,
        phase: u32,
        x: *const f32,
        f: *mut f32,
        pe: *mut f64,
        work: *mut f64,
        stream: *mut c_void,
    ) -> i32;
    fn prism_asc_compose_force_compensated_v1(ptr:u64,n:u32,phase:u32,x:*const f32,f:*mut f32,pe:*mut f64,work:*mut f64,stream:*mut c_void,
        owners:*const crate::biprism_coordinates::BiprismCoordinateOwnerFfi,leaf:u32,fault:*mut u32)->i32;
    fn prism_asc_compose_force_selected_compensated_v2(ptr:u64,n:u32,phase:u32,x:*const f32,f:*mut f32,pe:*mut f64,work:*mut f64,stream:*mut c_void,
        owners:*const crate::biprism_coordinates::BiprismCoordinateOwnerFfi,leaf:u32,fault:*mut u32)->i32;
    fn prism_asc_seal_step_selected_v2(ptr: u64, stream: *mut c_void) -> i32;
    fn prism_asc_snapshot_v1(ptr: u64, stream: *mut c_void, out: *mut AscSnapshot) -> i32;
}
fn checked(rc: i32) -> Result<(), i32> {
    if rc == 0 {
        Ok(())
    } else {
        Err(rc)
    }
}
pub struct AscDevice {
    device: u64,
    stream: Arc<CudaStream>,
    source: u32,
    generation: u64,
    atoms: u32,
    topology_sha256: String,
    selected_support: Option<AscSelectedSupport>,
}
impl AscDevice {
    pub fn new_legacy_whole_system(
        stream: Arc<CudaStream>,
        source: u32,
        atoms: usize,
        topology_sha256: &str,
        alpha: f32,
    ) -> anyhow::Result<Arc<Self>> {
        anyhow::ensure!(
            topology_sha256.len() == 64 && topology_sha256.bytes().all(|b| b.is_ascii_hexdigit()),
            "ASC requires exact topology/atom-order SHA-256"
        );
        anyhow::ensure!(
            atoms > 0 && alpha.is_finite(),
            "invalid ASC population/gain"
        );
        let atoms = u32::try_from(atoms)?;
        let mut topology = [0u64; 4];
        for (j, word) in topology.iter_mut().enumerate() {
            *word = u64::from_str_radix(&topology_sha256[j * 16..j * 16 + 16], 16)?;
        }
        let generation = GENERATION.fetch_add(1, Ordering::Relaxed);
        anyhow::ensure!(generation != 0, "ASC generation overflow");
        let context_rc = unsafe { cudarc::driver::sys::cuCtxSetCurrent(stream.context().cu_ctx()) };
        anyhow::ensure!(
            context_rc == cudarc::driver::sys::CUresult::CUDA_SUCCESS,
            "ASC context bind failed: {:?}",
            context_rc
        );
        let mut device = 0;
        let rc = unsafe {
            prism_asc_create_v1(
                &mut device,
                source,
                generation,
                atoms,
                topology.as_ptr(),
                alpha,
                1,
                stream.cu_stream().cast(),
            )
        };
        let owner = Self {
            device,
            stream,
            source,
            generation,
            atoms,
            topology_sha256: topology_sha256.to_owned(),
            selected_support: None,
        };
        anyhow::ensure!(rc == 0, "ASC allocation/initialization failed rc={rc}");
        Ok(Arc::new(owner))
    }
    pub fn new_selected_support(
        stream: Arc<CudaStream>,
        source: u32,
        topology: &PrismPrepTopology,
        topology_sha256: &str,
        policy: &AscSelectedSupportPolicy,
        alpha: f32,
    ) -> anyhow::Result<Arc<Self>> {
        anyhow::ensure!(alpha.is_finite(), "invalid ASC selected-support gain");
        // All policy, topology, source and membership failures occur before context
        // binding or device allocation.
        let support = policy.validate_and_expand(topology, topology_sha256)?;
        let atoms = u32::try_from(topology.n_atoms)?;
        let selected_count = u32::try_from(support.atom_indices().len())?;
        let mut topology_words = [0u64; 4];
        for (j, word) in topology_words.iter_mut().enumerate() {
            *word = u64::from_str_radix(&topology_sha256[j * 16..j * 16 + 16], 16)?;
        }
        let generation = GENERATION.fetch_add(1, Ordering::Relaxed);
        anyhow::ensure!(generation != 0, "ASC generation overflow");
        let context_rc = unsafe { cudarc::driver::sys::cuCtxSetCurrent(stream.context().cu_ctx()) };
        anyhow::ensure!(
            context_rc == cudarc::driver::sys::CUresult::CUDA_SUCCESS,
            "ASC context bind failed: {:?}",
            context_rc
        );
        let mut device = 0;
        let rc = unsafe {
            prism_asc_create_selected_v2(
                &mut device,
                source,
                generation,
                atoms,
                topology_words.as_ptr(),
                alpha,
                support.support_identity(),
                support.mask().as_ptr(),
                selected_count,
                stream.cu_stream().cast(),
            )
        };
        let owner = Self {
            device,
            stream,
            source,
            generation,
            atoms,
            topology_sha256: topology_sha256.to_owned(),
            selected_support: Some(support),
        };
        anyhow::ensure!(
            rc == 0,
            "ASC selected-support allocation/initialization failed rc={rc}"
        );
        Ok(Arc::new(owner))
    }
    pub fn device_ptr(&self) -> u64 {
        self.device
    }
    pub fn n_atoms(&self) -> usize {
        self.atoms as usize
    }
    pub fn source_stream_id(&self) -> u32 {
        self.source
    }
    pub fn is_selected_support(&self) -> bool {
        self.selected_support.is_some()
    }
    pub fn selected_atom_indices(&self) -> Option<&[u32]> {
        self.selected_support.as_ref().map(|s| s.atom_indices())
    }
    pub fn membership_mask(&self) -> Option<&[u8]> {
        self.selected_support.as_ref().map(|s| s.mask())
    }
    pub fn owns_stream(&self, s: &CudaStream) -> bool {
        self.stream.cu_stream() == s.cu_stream()
    }
    pub fn owns_raw_stream(&self, s: *mut c_void) -> bool {
        self.stream.cu_stream().cast::<c_void>() == s
    }
    pub fn begin_step(&self, s: &CudaStream) -> Result<(), i32> {
        if !self.owns_stream(s) {
            return Err(1);
        }
        checked(unsafe { prism_asc_begin_step_v1(self.device, s.cu_stream().cast()) })
    }
    pub fn seal_step(&self, s: &CudaStream) -> Result<(), i32> {
        if !self.owns_stream(s) {
            return Err(1);
        }
        let rc = unsafe {
            if self.is_selected_support() {
                prism_asc_seal_step_selected_v2(self.device, s.cu_stream().cast())
            } else {
                prism_asc_seal_step_v1(self.device, s.cu_stream().cast())
            }
        };
        checked(rc)
    }
    /// Pointees must be the engine-owned live buffers, retained by the graph.
    pub unsafe fn compose(
        &self,
        s: &CudaStream,
        phase: u32,
        x: u64,
        f: u64,
        pe: u64,
        work: u64,
    ) -> Result<(), i32> {
        if !self.owns_stream(s) {
            return Err(1);
        }
        let rc = if self.is_selected_support() {
            prism_asc_compose_force_selected_v2(
                self.device,
                self.atoms,
                phase,
                x as _,
                f as _,
                pe as _,
                work as _,
                s.cu_stream().cast(),
            )
        } else {
            prism_asc_compose_force_v1(
                self.device,
                self.atoms,
                phase,
                x as _,
                f as _,
                pe as _,
                work as _,
                s.cu_stream().cast(),
            )
        };
        checked(rc)
    }
    /// Explicit native-Angstrom precision owner; legacy compose remains unchanged.
    pub unsafe fn compose_compensated(&self,s:&CudaStream,phase:u32,x:u64,f:u64,pe:u64,work:u64,
        owners:u64,leaf:u32,fault:u64)->Result<(),i32> {
        if !self.owns_stream(s) || owners==0 || owners%16!=0 || leaf>=8 || fault==0 || fault%4!=0 { return Err(1); }
        checked(if self.is_selected_support() {
            prism_asc_compose_force_selected_compensated_v2(self.device,self.atoms,phase,x as _,f as _,pe as _,work as _,s.cu_stream().cast(),owners as _,leaf,fault as _)
        } else {
            prism_asc_compose_force_compensated_v1(self.device,self.atoms,phase,x as _,f as _,pe as _,work as _,s.cu_stream().cast(),owners as _,leaf,fault as _)
        })
    }
    pub unsafe fn publish(
        &self,
        s: &CudaStream,
        adj: *const crate::interferometric_adjudicator::InterferometricAdjudicatorFfi,
    ) -> Result<(), i32> {
        if !self.owns_stream(s) {
            return Err(1);
        }
        let rc = if self.is_selected_support() {
            prism_asc_publish_decision_selected_v2(self.device, adj, s.cu_stream().cast())
        } else {
            prism_asc_publish_decision_v1(self.device, adj, s.cu_stream().cast())
        };
        checked(rc)
    }
    pub fn snapshot(&self) -> Result<AscSnapshot, i32> {
        crate::governor_device::drain_stream(&self.stream)?;
        let mut out = unsafe { std::mem::zeroed() };
        checked(unsafe {
            prism_asc_snapshot_v1(self.device, self.stream.cu_stream().cast(), &mut out)
        })?;
        Ok(out)
    }
    pub fn receipt(&self) -> Result<serde_json::Value, i32> {
        let s = self.snapshot()?;
        let selected = self.selected_support.as_ref();
        Ok(serde_json::json!({
             "schema_version":if selected.is_some(){2}else{1},"source_stream_id":self.source,"allocation_generation":self.generation,"topology_atom_order_sha256":self.topology_sha256,
             "support_mode":if selected.is_some(){"SELECTED_STRUCTURAL_ATOM_RESIDUE_SUPPORT"}else{"LEGACY_WHOLE_SYSTEM_ALL_ATOMS"},
             "support_identity":selected.map_or(1,|v|v.support_identity()),
             "center_authority":if selected.is_some(){"LATEST_COMPLETED_SELECTED_ATOM_CENTROID_HELD_FROM_DECISION_PUBLICATION"}else{"FIRST_RELAXED_TILE_AABB_MIDPOINT_LEGACY"},
             "selected_causal_support":false,
             "policy_source_sha256":selected.map(|v|v.policy_source_sha256()),"support_source_kind":selected.map(|v|v.source_kind()),
             "support_source_identity":selected.map(|v|v.source_identity()),"operator_declared_support_source_evidence_sha256":selected.map(|v|v.source_evidence_sha256()),
             "atom_order_sha256":selected.map(|v|v.atom_order_sha256()),"selected_atom_count":selected.map(|v|v.atom_indices().len()),
             "selected_physical_residue_ids":selected.map(|v|v.physical_residue_ids()),
         "physical_epoch":s.physical_epoch,"completed_epoch":s.completed_epoch,"published_epoch":s.published_epoch,"status_code":s.status,
         "active_decision":s.active,"pending_decision":s.pending,"phases":s.phases,"control_change_work_kcal_mol":s.control_work,
         "coverage_failures":s.coverage_failures,"decisions_produced":s.decisions_produced,"composed_steps":s.composed_steps,"consumed_steps":s.consumed_steps,
         "measurement_scope":"Last completed physical step, both pre-constraint half-kicks; actual scaled/clamped force minus same kick without ASC. Constraint, safety-clamp and thermostat changes are separate.",
         "producer_frame_clock":"owner-local completed ordinary physical-step ordinal, not legacy host capture-time frame index",
         "decision_age_physical_steps":s.completed_epoch.saturating_sub(s.active.producer_epoch),"control_cadence":"promote next physical begin; hold controls between completed-step/chunk publications","cuda_result":0
        }))
    }
}
impl Drop for AscDevice {
    fn drop(&mut self) {
        if self.device == 0 {
            return;
        }
        let rc = match crate::governor_device::drain_stream(&self.stream) {
            Ok(()) => unsafe { prism_asc_destroy_v1(self.device, self.stream.cu_stream().cast()) },
            Err(rc) => rc,
        };
        if rc != 0 {
            log::error!(
                "ASC_QUARANTINED source={} generation={} rc={}",
                self.source,
                self.generation,
                rc
            );
            std::mem::forget(self.stream.clone());
        }
    }
}


/// The native ASC state includes per-atom force/impulse/epoch rows and the
/// cumulative-work tail, not only the public summary. Captured pointer slots
/// are zeroed and rebound to the reconstructed native allocation on restore.
#[derive(serde::Serialize,serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AscContinuationV1 {
    schema_version:u32,source:u32,atoms:u32,topology_sha256:String,
    original_generation:u64,#[serde(with="crate::checkpoint_codec::bytes")] allocation:Vec<u8>,
}
impl AscDevice {
    pub fn continuation_allocation_plan(&self)->anyhow::Result<Vec<(String,u64)>> {
        let n=u64::from(self.atoms);let bytes=768u64.checked_add(n.checked_mul(320).ok_or_else(||anyhow::anyhow!("ASC extent overflow"))?)
            .and_then(|v|v.checked_add(if self.selected_support.is_some(){n}else{0})).ok_or_else(||anyhow::anyhow!("ASC extent overflow"))?;
        Ok(vec![("complete_native_allocation".into(),bytes)])
    }
    pub fn capture_continuation(&self)->anyhow::Result<AscContinuationV1> {
        use crate::scientific_checkpoint::device::read_raw;
        self.stream.synchronize()?;
        let selected=self.selected_support.is_some();
        let bytes=768usize.checked_add((self.atoms as usize).checked_mul(320)
            .ok_or_else(||anyhow::anyhow!("ASC extent overflow"))?).and_then(|n|n.checked_add(if selected{self.atoms as usize}else{0}))
            .ok_or_else(||anyhow::anyhow!("ASC extent overflow"))?;
        let mut allocation=unsafe{read_raw(&self.stream,self.device,bytes)?};
        let u32_at=|o:usize|u32::from_le_bytes(allocation[o..o+4].try_into().unwrap());
        let u64_at=|o:usize|u64::from_le_bytes(allocation[o..o+8].try_into().unwrap());
        anyhow::ensure!(u32_at(0)==1 && u32_at(4)==736 && u32_at(8)==self.source && u32_at(12)==self.atoms
            && u64_at(16)==self.generation,"ASC state ABI/owner mismatch");
        anyhow::ensure!(u64_at(24)==u64_at(32),"ASC physical step remains open");
        anyhow::ensure!(u64_at(720)==self.device+736,"ASC rows do not belong to allocation");
        if selected {anyhow::ensure!(u64_at(728)==self.device+768+self.atoms as u64*320,"ASC membership pointer mismatch");}
        else {anyhow::ensure!(u64_at(728)==0,"unexpected ASC membership pointer");}
        allocation[720..736].fill(0);
        Ok(AscContinuationV1{schema_version:1,source:self.source,atoms:self.atoms,topology_sha256:self.topology_sha256.clone(),
            original_generation:self.generation,allocation})
    }
    pub fn restore_continuation(&self,saved:&AscContinuationV1)->anyhow::Result<()> {
        use crate::scientific_checkpoint::device::write_raw;
        let fresh=self.capture_continuation()?;
        anyhow::ensure!(saved.schema_version==1 && saved.source==self.source && saved.atoms==self.atoms
            && saved.topology_sha256==self.topology_sha256 && saved.allocation.len()==fresh.allocation.len(),
            "ASC restart identity/extent mismatch");
        anyhow::ensure!(saved.allocation[48..88]==fresh.allocation[48..88] && saved.allocation[92..96]==fresh.allocation[92..96],
            "ASC topology/gain/mode/support count changed on restart");
        if self.selected_support.is_some() {
            let start=768+self.atoms as usize*320;
            anyhow::ensure!(saved.allocation[start..]==fresh.allocation[start..]
                && saved.allocation[712..720]==fresh.allocation[712..720],"ASC selected membership/support identity changed");
        }
        let mut allocation=saved.allocation.clone();
        anyhow::ensure!(allocation[720..736].iter().all(|&x|x==0),"serialized ASC state contains device addresses");
        allocation[720..728].copy_from_slice(&(self.device+736).to_le_bytes());
        if self.selected_support.is_some(){allocation[728..736].copy_from_slice(&(self.device+768+self.atoms as u64*320).to_le_bytes());}
        for offset in [16,136,232,328] {
            let generation=u64::from_le_bytes(allocation[offset..offset+8].try_into()?);
            anyhow::ensure!(generation==0 || generation==saved.original_generation,"ASC control generation differs from its owner");
            if generation!=0 {allocation[offset..offset+8].copy_from_slice(&self.generation.to_le_bytes());}
        }
        unsafe{write_raw(&self.stream,self.device,&allocation)?;}
        self.stream.synchronize()?;Ok(())
    }
}
