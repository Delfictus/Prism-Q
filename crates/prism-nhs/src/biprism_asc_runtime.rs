//! Actual ASC ownership for the additive native-bath/v10-core profile.
//!
//! Native bath kick rows and control-change work retain their ordinary owners.
//! Core rows bind the held force actually imported into v10, not a synthetic
//! AMBER kick or an additive endpoint velocity response. Completed seals are
//! journaled per step and leaf. A separate raw current-attempt snapshot retains
//! rows even when a failed body returns before journal publication.

use crate::biprism_v10_runtime::{BiprismDeviceViews, BiprismConvergenceRecordFfi};
use crate::fused_engine::{BiprismNativeEndpointOwnersFfi, BiprismNativeLeafPointers};
use cudarc::driver::{CudaFunction,CudaModule,CudaSlice,CudaStream,DevicePtr,DeviceRepr,LaunchConfig,PushKernelArg};
use serde::Serialize;
use std::sync::Arc;

pub const ASC_MAGIC:u32=0x4241_5331;
pub const ASC_PREPARED:u32=1;
pub const ASC_CORE_ACCEPTED:u32=2;
pub const ASC_SEALED:u32=3;
pub const ASC_DISABLED:u32=4;
pub const ASC_CORE:usize=192;

#[repr(C,align(16))]
#[derive(Clone,Copy,Debug,Default)]
struct AscBindingFfi { owner:u64, motion_ownership:u64 }
unsafe impl DeviceRepr for AscBindingFfi {}
#[repr(C,align(16))]
#[derive(Clone,Copy,Debug,Default,Serialize)]
pub struct BiprismAscCoreRowFfi {
    pub physical_epoch:u64,pub decision_epoch:u64,
    pub atom_id:u32,pub scheduled_step:u32,pub state:u32,pub selected:u32,
    pub x_start_au:[f32;3],pub x_end_au:[f32;3],
    pub requested_native_force:[f64;3],pub effective_au_force:[f64;3],pub canonical_impulse_au:[f64;3],
    pub path_work_hartree:f64,pub potential_start_kcal_mol:f64,pub potential_end_kcal_mol:f64,
    pub held_force_work_error_kcal_mol:f64,
    pub native_total_force:[f32;3],pub native_base_force:[f32;3],pub au_total_force:[f32;3],pub au_base_force:[f32;3],
}
unsafe impl DeviceRepr for BiprismAscCoreRowFfi {}
#[repr(C,align(16))]
#[derive(Clone,Copy,Debug,Default,Serialize)]
pub struct BiprismAscLedgerFfi {
    pub abi_magic:u32,pub leaf_id:u32,pub sticky_fault:u32,pub state:u32,
    pub generation:u64,pub physical_epoch:u64,pub decision_epoch:u64,pub initial_completed_epoch:u64,
    pub scheduled_step:u32,pub begin_substeps:u32,pub end_substeps:u32,pub selected_core_atoms:u32,
    pub selected_bath_atoms:u32,pub core_np1_unconsumed_atoms:u32,pub atom_count:u32,pub reserved:u32,
    pub completed_steps:u64,pub completed_epoch:u64,
    pub dt_outer_au:f64,pub path_work_hartree:f64,pub cumulative_path_work_hartree:f64,
    pub held_force_work_error_kcal_mol:f64,pub cumulative_work_error_kcal_mol:f64,
    pub canonical_impulse_au:[f64;3],pub cumulative_impulse_au:[f64;3],
    pub control_work_kcal_mol:f64,
}
unsafe impl DeviceRepr for BiprismAscLedgerFfi {}
#[repr(C,align(16))]
#[derive(Clone,Copy,Debug,Default,Serialize)]
pub struct BiprismAscWindowFfi {
    pub abi_magic:u32,pub capacity:u32,pub first_body:u32,pub requested_steps:u32,
    pub first_scheduled_step:u32,pub reserved:[u32;3],
}
unsafe impl DeviceRepr for BiprismAscWindowFfi {}
const _:()={
    assert!(core::mem::size_of::<AscBindingFfi>()==16);
    assert!(core::mem::size_of::<BiprismAscCoreRowFfi>()==208);
    assert!(core::mem::offset_of!(BiprismAscCoreRowFfi,requested_native_force)==56);
    assert!(core::mem::size_of::<BiprismAscLedgerFfi>()==192);
    assert!(core::mem::offset_of!(BiprismAscLedgerFfi,control_work_kcal_mol)==184);
    assert!(core::mem::size_of::<BiprismAscWindowFfi>()==32);
};

/// A window may contain an attempted prefix. Only validate_accepted_window
/// authorizes interpreting every requested slot as a completed body. Raw bytes
/// stay available after failures, with disabled slots explicitly identified.
#[derive(Debug)]
pub struct BiprismAscWindowReadback {
    pub window:BiprismAscWindowFfi,
    pub enabled:[bool;8],
    pub ledgers:Vec<BiprismAscLedgerFfi>,
    pub rows:Vec<BiprismAscCoreRowFfi>,
}
impl BiprismAscWindowReadback {
    /// ABI v1 contains no padding and this engine requires little-endian hosts.
    pub fn ledger_bytes(&self)->Result<&[u8],String>{bytes_le(&self.ledgers)}
    pub fn row_bytes(&self)->Result<&[u8],String>{bytes_le(&self.rows)}
    pub fn index(&self)->serde_json::Value {
        serde_json::json!({"schema":"biprism.asc.mixed_work.v1","window":self.window,"enabled_leaves":self.enabled,
            "record_order":"window_step,leaf,core_local_slot","ledger_record_bytes":192,"core_record_bytes":208,"core_slots":192,
            "ledger_records":self.ledgers.len(),"reserved_core_row_slots":self.rows.len(),
            "disabled_leaf_rows":"reserved storage only; excluded from produced row population",
            "native_force_units":"kcal/mol/Angstrom","au_force_units":"Hartree/bohr","impulse_units":"atomic momentum",
            "core_response":"held imported total AU float minus identically converted native base float; outer-step impulse and accepted core displacement work",
            "force_np1_core":"composed for the endpoint but not consumed by a second native half-kick",
            "bath_response":"actual native pre-constraint half-kick difference; no synthetic core kick stamps",
            "control_work":"native parameter intervention only; conservative core path work is not added again as external work",
            "projection_scope":"core rows end before native endpoint projection, thermostat, UV, delayed LIF delivery or CA intervention",
            "scientific_restart":false})
    }
}
fn bytes_le<T>(values:&[T])->Result<&[u8],String> {
    if !cfg!(target_endian="little"){return Err("ASC journal ABI requires a little-endian host".into());}
    Ok(unsafe{std::slice::from_raw_parts(values.as_ptr().cast(),core::mem::size_of_val(values))})
}

pub struct BiprismAscEndpointBinding<'a> {
    function:&'a CudaFunction,stream:&'a Arc<CudaStream>,owner:u64,mask:u64,
    ledger:u64,rows:u64,atom_count:u32,window:u64,journal_ledgers:u64,journal_rows:u64,
    leaf:u32,convergence:u64,
}
impl BiprismAscEndpointBinding<'_> {
    /// # Safety
    /// Called at the native ASC sealing frontier after the two force phases,
    /// bath kicks and endpoint work. The retained allocation owner must outlive
    /// capture/replay. Zero actual_owner denotes an explicitly disabled ASC leaf.
    pub(crate) unsafe fn capture(&self,actual_owner:u64,actual_mask:u64,atoms:u32,
        stream:&Arc<CudaStream>)->Result<(),String> {
        if actual_owner!=self.owner||actual_mask!=self.mask||atoms!=self.atom_count||!Arc::ptr_eq(stream,self.stream) {
            return Err("ASC mixed seal does not match actual native owner/motion population/stream".into());
        }
        stream.launch_builder(self.function).arg(&self.owner).arg(&self.mask).arg(&self.ledger)
            .arg(&self.rows).arg(&self.atom_count).arg(&self.window).arg(&self.journal_ledgers)
            .arg(&self.journal_rows).arg(&self.leaf).arg(&self.convergence)
            .launch(LaunchConfig{grid_dim:(1,1,1),block_dim:(1,1,1),shared_mem_bytes:0})
            .map_err(|e|e.to_string())?;
        Ok(())
    }
}

/// Unfiltered live device storage sampled after quiescence. A row can be stale
/// or only partly produced when a body refuses. Its own state/epoch and the live
/// ledger are evidence; this snapshot does not certify completed row coverage.
#[derive(Debug)]
pub struct BiprismAscAttemptReadback {
    pub enabled:[bool;8],pub ledgers:[BiprismAscLedgerFfi;8],
    pub rows:Vec<BiprismAscCoreRowFfi>,
}
impl BiprismAscAttemptReadback {
    pub fn ledger_bytes(&self)->Result<&[u8],String>{bytes_le(&self.ledgers)}
    pub fn row_bytes(&self)->Result<&[u8],String>{bytes_le(&self.rows)}
    pub fn index(&self)->serde_json::Value {
        serde_json::json!({"schema":"biprism.asc.attempted_current_body.v1",
            "enabled_leaves":self.enabled,"ledger_record_bytes":192,"core_record_bytes":208,
            "raw_core_storage_slots":self.rows.len(),"record_order":"leaf,core_local_slot",
            "accepted_row_population":null,"scope":"unfiltered live storage; row state and epoch must be interpreted with corresponding live ledger; stale and partial rows are preserved",
            "ledgers":self.ledgers,"scientific_restart":false})
    }
}

pub(crate) struct BiprismAscWorkOwner {
    stream:Arc<CudaStream>,bindings:CudaSlice<AscBindingFfi>,ledgers:CudaSlice<BiprismAscLedgerFfi>,
    rows:CudaSlice<BiprismAscCoreRowFfi>,window:CudaSlice<BiprismAscWindowFfi>,
    journal_ledgers:CudaSlice<BiprismAscLedgerFfi>,journal_rows:CudaSlice<BiprismAscCoreRowFfi>,
    capture:CudaFunction,finish:CudaFunction,seal:CudaFunction,
    owners:[BiprismNativeEndpointOwnersFfi;8],native:[BiprismNativeLeafPointers;8],
    selected:[[u32;256];8],capacity:u32,current_window:Option<BiprismAscWindowFfi>,
}
impl BiprismAscWorkOwner {
    pub(crate) fn new(stream:Arc<CudaStream>,module:&Arc<CudaModule>,
        owners:[BiprismNativeEndpointOwnersFfi;8],native:[BiprismNativeLeafPointers;8],
        selected:[[u32;256];8],capacity:u32)->Result<Self,String> {
        let (ledger_count,row_count)=journal_extents(capacity)?;
        let bindings=std::array::from_fn::<_,8,_>(|i|AscBindingFfi{owner:owners[i].asc,motion_ownership:native[i].ownership});
        let capture=module.load_function("nhs_biprism_asc_capture_force_v1").map_err(|e|e.to_string())?;
        let finish=module.load_function("nhs_biprism_asc_finish_core_v1").map_err(|e|e.to_string())?;
        let seal=module.load_function("nhs_biprism_asc_seal_mixed_v1").map_err(|e|e.to_string())?;
        Ok(Self{bindings:upload(&stream,&bindings)?,ledgers:upload(&stream,&[BiprismAscLedgerFfi::default();8])?,
            rows:upload(&stream,&vec![BiprismAscCoreRowFfi::default();8*ASC_CORE])?,
            window:upload(&stream,&[BiprismAscWindowFfi::default()])?,
            journal_ledgers:upload(&stream,&vec![BiprismAscLedgerFfi::default();ledger_count])?,
            journal_rows:upload(&stream,&vec![BiprismAscCoreRowFfi::default();row_count])?,
            stream,capture,finish,seal,owners,native,selected,capacity,current_window:None})
    }
    pub(crate) fn begin_window(&mut self,first_body:u32,first_scheduled_step:u32,steps:u32)->Result<(),String> {
        if self.current_window.is_some(){return Err("ASC work window awaits retained export acknowledgement".into());}
        let w=BiprismAscWindowFfi{abi_magic:ASC_MAGIC,capacity:self.capacity,first_body,requested_steps:steps,
            first_scheduled_step,reserved:[0;3]};
        validate_window_descriptor(&w)?;
        self.stream.memcpy_htod(&[w],&mut self.window).map_err(|e|e.to_string())?;
        self.stream.synchronize().map_err(|e|e.to_string())?;
        self.current_window=Some(w);Ok(())
    }
    pub(crate) unsafe fn capture_force(&self,view:BiprismDeviceViews,atoms:u32,scheduled:u64,dt:u64)->Result<(),String> {
        if dt==0||dt%4!=0{return Err("selected ASC dt pointer absent".into());}
        let b=self.bindings.device_ptr(&self.stream).0;let l=self.ledgers.device_ptr(&self.stream).0;
        let r=self.rows.device_ptr(&self.stream).0;let w=self.window.device_ptr(&self.stream).0;
        self.stream.launch_builder(&self.capture).arg(&b).arg(&view.tiles).arg(&view.stream_table)
            .arg(&view.native_stream_table).arg(&atoms).arg(&view.convergence).arg(&scheduled).arg(&l)
            .arg(&r).arg(&dt).arg(&w)
            .launch(LaunchConfig{grid_dim:(8,1,1),block_dim:(1,1,1),shared_mem_bytes:0}).map_err(|e|e.to_string())?;
        Ok(())
    }
    pub(crate) unsafe fn finish_core(&self,view:BiprismDeviceViews,atoms:u32,scheduled:u64)->Result<(),String> {
        let b=self.bindings.device_ptr(&self.stream).0;let l=self.ledgers.device_ptr(&self.stream).0;let r=self.rows.device_ptr(&self.stream).0;
        self.stream.launch_builder(&self.finish).arg(&b).arg(&view.stream_table).arg(&atoms).arg(&view.convergence)
            .arg(&scheduled).arg(&l).arg(&r)
            .launch(LaunchConfig{grid_dim:(8,1,1),block_dim:(1,1,1),shared_mem_bytes:0}).map_err(|e|e.to_string())?;
        Ok(())
    }
    pub(crate) fn endpoint(&self,leaf:usize,convergence:u64)->BiprismAscEndpointBinding<'_> {
        BiprismAscEndpointBinding{function:&self.seal,stream:&self.stream,owner:self.owners[leaf].asc,mask:self.native[leaf].ownership,
            ledger:self.ledgers.device_ptr(&self.stream).0+(leaf*192) as u64,
            rows:self.rows.device_ptr(&self.stream).0+(leaf*ASC_CORE*208) as u64,
            atom_count:self.native[leaf].n_atoms as u32,window:self.window.device_ptr(&self.stream).0,
            journal_ledgers:self.journal_ledgers.device_ptr(&self.stream).0,journal_rows:self.journal_rows.device_ptr(&self.stream).0,
            leaf:leaf as u32,convergence:convergence+(leaf*core::mem::size_of::<BiprismConvergenceRecordFfi>()) as u64}
    }
    pub(crate) fn guard_pointers(&self)->(u64,u64,u64) {
        (self.ledgers.device_ptr(&self.stream).0,self.window.device_ptr(&self.stream).0,self.journal_ledgers.device_ptr(&self.stream).0)
    }
    pub(crate) fn read_current(&self)->Result<[BiprismAscLedgerFfi;8],String> {
        download(&self.stream,&self.ledgers)?.try_into().map_err(|_|"ASC current ledger extent".into())
    }
    pub(crate) fn read_current_attempt(&self)->Result<BiprismAscAttemptReadback,String> {
        self.stream.synchronize().map_err(|e|e.to_string())?;
        Ok(BiprismAscAttemptReadback{enabled:self.owners.map(|o|o.asc!=0),
            ledgers:self.read_current()?,rows:download(&self.stream,&self.rows)?})
    }
    pub(crate) fn read_window(&self)->Result<BiprismAscWindowReadback,String> {
        let w=self.current_window.ok_or("no attempted ASC work window")?;
        let mut ledgers=download(&self.stream,&self.journal_ledgers)?;let mut rows=download(&self.stream,&self.journal_rows)?;
        ledgers.truncate(w.requested_steps as usize*8);rows.truncate(w.requested_steps as usize*8*ASC_CORE);
        Ok(BiprismAscWindowReadback{window:w,enabled:self.owners.map(|o|o.asc!=0),ledgers,rows})
    }
    pub(crate) fn validate_readback(&self,r:&BiprismAscWindowReadback)->Result<(),String> {
        validate_accepted_window(r,&self.owners,&self.selected,self.native[0].n_atoms as u32)
    }
    /// Called only after the controller's unsafe acknowledgement contract: all
    /// exact journal bytes and paired observations have been durably retained.
    pub(crate) fn acknowledge(&mut self){self.current_window=None;}
}

pub fn journal_extents(capacity:u32)->Result<(usize,usize),String> {
    if capacity==0||capacity>500{return Err("ASC work capacity must be1..500 physical bodies".into());}
    let ledgers=(capacity as usize).checked_mul(8).ok_or("ASC ledger extent overflow")?;
    Ok((ledgers,ledgers.checked_mul(ASC_CORE).ok_or("ASC row extent overflow")?))
}
pub fn validate_window_descriptor(w:&BiprismAscWindowFfi)->Result<(),String> {
    journal_extents(w.capacity)?;
    if w.abi_magic!=ASC_MAGIC||w.requested_steps==0||w.requested_steps>w.capacity||w.reserved!=[0;3]
        ||w.first_body.checked_add(w.requested_steps).is_none()
        ||w.first_scheduled_step.checked_add(w.requested_steps).filter(|x|*x<u32::MAX).is_none() {
        return Err("ASC work window identity/range refused".into());
    }Ok(())
}
fn ledger_finite(r:&BiprismAscLedgerFfi)->bool {
    [r.dt_outer_au,r.path_work_hartree,r.cumulative_path_work_hartree,r.held_force_work_error_kcal_mol,
        r.cumulative_work_error_kcal_mol,r.control_work_kcal_mol].iter().chain(r.canonical_impulse_au.iter())
        .chain(r.cumulative_impulse_au.iter()).all(|x|x.is_finite())&&r.dt_outer_au>0.0
}
fn row_finite(r:&BiprismAscCoreRowFfi)->bool {
    r.x_start_au.iter().chain(r.x_end_au.iter()).chain(r.native_total_force.iter()).chain(r.native_base_force.iter())
        .chain(r.au_total_force.iter()).chain(r.au_base_force.iter()).all(|x|x.is_finite())
        &&r.requested_native_force.iter().chain(r.effective_au_force.iter()).chain(r.canonical_impulse_au.iter())
        .chain([r.path_work_hartree,r.potential_start_kcal_mol,r.potential_end_kcal_mol,r.held_force_work_error_kcal_mol].iter()).all(|x|x.is_finite())
}
pub fn validate_accepted_window(r:&BiprismAscWindowReadback,owners:&[BiprismNativeEndpointOwnersFfi;8],
    selected:&[[u32;256];8],atoms:u32)->Result<(),String> {
    validate_window_descriptor(&r.window)?;
    let expected=r.window.requested_steps as usize*8;
    if r.ledgers.len()!=expected||r.rows.len()!=expected*ASC_CORE||r.enabled!=owners.map(|o|o.asc!=0) {
        return Err("ASC work journal population/owner mismatch".into());
    }
    for (record,l) in r.ledgers.iter().enumerate(){
        let leaf=record%8;let ordinal=record/8;let bodies=u64::from(r.window.first_body)+ordinal as u64+1;
        let step=r.window.first_scheduled_step+ordinal as u32;let native=owners[leaf];
        if l.abi_magic!=ASC_MAGIC||l.leaf_id!=leaf as u32||l.sticky_fault!=0||l.reserved!=0||l.scheduled_step!=step
            ||l.completed_steps!=bodies||l.begin_substeps as u64!=(bodies-1)*4||l.end_substeps as u64!=bodies*4
            ||l.atom_count!=atoms||!ledger_finite(l) {
            return Err(format!("ASC ledger identity/epoch/finiteness failed at window record{record}"));
        }
        if !r.enabled[leaf] {
            if l.state!=ASC_DISABLED||l.generation!=0||l.physical_epoch!=0||l.completed_epoch!=0
                ||l.selected_core_atoms!=0||l.selected_bath_atoms!=0||l.core_np1_unconsumed_atoms!=0
                ||l.path_work_hartree!=0.0||l.control_work_kcal_mol!=0.0 {
                return Err(format!("disabled ASC leaf{leaf} claims produced work"));
            }continue;
        }
        let epoch=native.initial_asc_epoch.checked_add(bodies).ok_or("ASC physical epoch overflow")?;
        if l.state!=ASC_SEALED||l.generation!=native.asc_generation||l.initial_completed_epoch!=native.initial_asc_epoch
            ||l.physical_epoch!=epoch||l.completed_epoch!=epoch||l.selected_core_atoms>192
            ||l.core_np1_unconsumed_atoms!=l.selected_core_atoms||l.selected_core_atoms+l.selected_bath_atoms>atoms {
            return Err(format!("ASC native source coverage failed at window record{record}"));
        }
        let rows=&r.rows[record*ASC_CORE..(record+1)*ASC_CORE];let mut count=0u32;
        for (local,row) in rows.iter().enumerate(){
            if row.state!=ASC_CORE_ACCEPTED||row.atom_id!=selected[leaf][local]||row.atom_id>=atoms
                ||row.physical_epoch!=epoch||row.decision_epoch!=l.decision_epoch||row.scheduled_step!=step
                ||row.selected>1||!row_finite(row) {
                return Err(format!("ASC core row identity/finiteness failed record{record} slot{local}"));
            }count+=row.selected;
        }
        if count!=l.selected_core_atoms{return Err("ASC selected row count differs from sealed ledger".into());}
    }Ok(())
}
fn upload<T:DeviceRepr>(s:&Arc<CudaStream>,v:&[T])->Result<CudaSlice<T>,String>{
    let d=s.clone_htod(v).map_err(|e|e.to_string())?;s.synchronize().map_err(|e|e.to_string())?;Ok(d)
}
fn download<T:DeviceRepr>(s:&Arc<CudaStream>,d:&CudaSlice<T>)->Result<Vec<T>,String>{
    let v=s.clone_dtoh(d).map_err(|e|e.to_string())?;s.synchronize().map_err(|e|e.to_string())?;Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]fn complete500_window_has_explicit_storage_population(){
        let(l,r)=journal_extents(500).unwrap();assert_eq!((l,r),(4000,768000));
        assert_eq!(l*192+r*208,160_512_000);assert!(journal_extents(0).is_err());assert!(journal_extents(501).is_err());
    }
    #[test]fn window_refuses_empty_reserved_and_wrap(){
        let w=BiprismAscWindowFfi{abi_magic:ASC_MAGIC,capacity:500,first_body:0,requested_steps:500,first_scheduled_step:1,reserved:[0;3]};
        assert!(validate_window_descriptor(&w).is_ok());
        for bad in [BiprismAscWindowFfi{requested_steps:0,..w},BiprismAscWindowFfi{reserved:[1,0,0],..w},
            BiprismAscWindowFfi{first_body:u32::MAX,..w},BiprismAscWindowFfi{first_scheduled_step:u32::MAX-499,..w}] {
            assert!(validate_window_descriptor(&bad).is_err());
        }
    }
}
