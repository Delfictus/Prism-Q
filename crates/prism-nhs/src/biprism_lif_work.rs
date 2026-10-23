//! Measured delayed-LIF intervention on actual AU/native velocity planes.
//! Native HMR work is published to the existing native atom work buffer only
//! after all eight measured leaves pass. Metric and lab quantities are separate.
use crate::biprism_v10_runtime::BiprismDeviceViews;
use crate::fused_engine::BiprismNativeLeafPointers;
use cudarc::driver::{CudaFunction,CudaModule,CudaSlice,CudaStream,DevicePtr,DeviceRepr,LaunchConfig,PushKernelArg};
use serde::Serialize;
use std::sync::Arc;
pub const LIF_WORK_MAGIC:u32=0x424c5733;
pub const LIF_WORK_ROW_BYTES:usize=464;
pub const LIF_WORK_LEDGER_BYTES:usize=288;
#[repr(C,align(16))]
#[derive(Clone,Copy,Debug,Default)]
struct Binding { native_hmr_mass_dalton:u64,native_atom_work_kcal:u64 }
unsafe impl DeviceRepr for Binding {}
#[repr(C,align(16))]
#[derive(Clone,Copy,Debug,Default,Serialize)]
pub struct LifWorkRow {
    pub atom_id:u32,pub leaf_id:u32,pub local_vertex:u32,pub state:u32,
    pub scheduled_step:u32,pub committed_substeps:u32,pub source_produced_step:u32,pub pending_before:u32,
    pub true_inverse_mass_au:f32,pub propagation_inverse_mass_au:f32,pub represented_main_mass_au:f32,pub native_hmr_mass_dalton:f32,
    pub metric_position_au:[f32;3],pub native_position_angstrom:[f32;3],pub accepted_main_velocity:[f32;3],pub pre_lif_scope:u32,
    pub au_velocity_before:[f32;3],pub au_velocity_after:[f32;3],pub native_velocity_before:[f32;3],pub native_velocity_after:[f32;3],
    pub requested_force_au:[f32;3],pub accepted_metric:[f32;6],pub reserved:u32,
    pub requested_impulse:[f64;3],pub realized_lab_impulse:[f64;3],pub realized_true_covariant_impulse:[f64;3],
    pub requested_lab_work:f64,pub realized_lab_work:f64,pub lab_kinetic_delta:f64,
    pub requested_true_metric_work:f64,pub realized_true_metric_work:f64,pub true_metric_kinetic_delta:f64,
    pub main_mass_metric_kinetic_delta:f64,pub native_hmr_kinetic_delta_kcal:f64,
    pub native_true_lab_kinetic_delta:f64,pub lab_bridge_kinetic_difference:f64,
    pub native_true_metric_kinetic_delta:f64,pub metric_bridge_kinetic_difference:f64,
    pub main_vs_true_metric_mass_difference:f64,pub native_phase_hmr_kinetic_delta_kcal:f64,pub native_phase_vs_guard_work_difference_kcal:f64,
    pub pre_lif_true_metric_delta:f64,pub pre_lif_main_metric_delta:f64,
    pub metric_position_au_lo:[f32;3],pub native_position_angstrom_lo:[f32;3],pub coordinate_reserved:[u32;2],
    pub accepted_metric_f64:[f64;6],
}
unsafe impl DeviceRepr for LifWorkRow {}
impl LifWorkRow {
    pub fn values(&self)->[f64;17]{[
        self.requested_lab_work,self.realized_lab_work,self.lab_kinetic_delta,self.requested_true_metric_work,
        self.realized_true_metric_work,self.true_metric_kinetic_delta,self.main_mass_metric_kinetic_delta,
        self.native_hmr_kinetic_delta_kcal,self.native_true_lab_kinetic_delta,self.lab_bridge_kinetic_difference,
        self.native_true_metric_kinetic_delta,self.metric_bridge_kinetic_difference,self.main_vs_true_metric_mass_difference,
        self.native_phase_hmr_kinetic_delta_kcal,self.native_phase_vs_guard_work_difference_kcal,
        self.pre_lif_true_metric_delta,self.pre_lif_main_metric_delta]}
    fn finite(&self)->bool {
        [self.true_inverse_mass_au,self.propagation_inverse_mass_au,self.represented_main_mass_au,self.native_hmr_mass_dalton]
            .iter().all(|x|x.is_finite()&&*x>0.0)
            &&self.accepted_metric_f64.iter().all(|x|x.is_finite())
            &&self.metric_position_au.iter().chain(&self.native_position_angstrom).chain(&self.accepted_main_velocity)
            .chain(&self.au_velocity_before).chain(&self.au_velocity_after).chain(&self.native_velocity_before)
            .chain(&self.native_velocity_after).chain(&self.metric_position_au_lo).chain(&self.native_position_angstrom_lo).chain(&self.requested_force_au).chain(&self.accepted_metric).all(|x|x.is_finite())
            &&self.values().iter().chain(&self.requested_impulse).chain(&self.realized_lab_impulse)
            .chain(&self.realized_true_covariant_impulse).all(|x|x.is_finite())
    }
}
#[repr(C,align(16))]
#[derive(Clone,Copy,Debug,Default,Serialize)]
pub struct LifWorkLedger {
    pub abi_magic:u32,pub leaf_id:u32,pub state:u32,pub sticky_fault:u32,
    pub scheduled_step:u32,pub committed_substeps:u32,pub produced_step:u32,pub delivered_step_before:u32,
    pub pending_before:u32,pub mode:u32,pub reserved:[u32;2],
    pub completed_steps:u64,pub body_ordinal:u64,pub dt_feedback_au:f64,pub totals:[f64;17],
    pub requested_impulse:[f64;3],pub realized_lab_impulse:[f64;3],pub realized_true_covariant_impulse:[f64;3],
    pub reserved_tail:u64,
}
unsafe impl DeviceRepr for LifWorkLedger {}
#[repr(C,align(16))]
#[derive(Clone,Copy,Debug,Default,Serialize)]
pub struct LifWorkWindow {
    pub abi_magic:u32,pub capacity:u32,pub first_body:u32,pub requested_steps:u32,
    pub first_scheduled_step:u32,pub reserved:[u32;3],
}
unsafe impl DeviceRepr for LifWorkWindow {}
const _:()={
    assert!(core::mem::size_of::<LifWorkRow>()==464);
    assert!(core::mem::offset_of!(LifWorkRow,accepted_metric_f64)==416);
    assert!(core::mem::offset_of!(LifWorkRow,requested_impulse)==176);
    assert!(core::mem::offset_of!(LifWorkRow,metric_position_au_lo)==384);
    assert!(core::mem::offset_of!(LifWorkRow,coordinate_reserved)==408);
    assert!(core::mem::offset_of!(LifWorkRow,requested_lab_work)==248);
    assert!(core::mem::size_of::<LifWorkLedger>()==288);
    assert!(core::mem::offset_of!(LifWorkLedger,totals)==72);
    assert!(core::mem::size_of::<LifWorkWindow>()==32);
};
pub const LIF_WORK_VALUE_NAMES:[&str;17]=[
    "requested_lab_work_hartree","realized_lab_work_hartree","true_lab_kinetic_delta_hartree",
    "requested_true_metric_work_hartree","realized_true_metric_work_hartree","true_metric_kinetic_delta_hartree",
    "core_main_mass_metric_kinetic_delta_hartree","native_guard_hmr_kinetic_delta_kcal_mol",
    "native_true_lab_kinetic_delta_hartree","lab_delivery_bridge_difference_hartree",
    "native_true_metric_kinetic_delta_hartree","metric_delivery_bridge_difference_hartree",
    "core_main_vs_true_metric_mass_difference_hartree","native_phase_hmr_kinetic_delta_kcal_mol",
    "native_phase_vs_guard_work_difference_kcal_mol","core_pre_lif_true_metric_gap_hartree","core_pre_lif_main_metric_gap_hartree"];
pub struct LifWorkReadback {pub window:Option<LifWorkWindow>,pub ledgers:Vec<LifWorkLedger>,pub rows:Vec<LifWorkRow>,pub current_attempt:bool}
impl LifWorkReadback {
    pub fn ledger_bytes(&self)->Result<&[u8],String>{bytes_le(&self.ledgers)}
    pub fn row_bytes(&self)->Result<&[u8],String>{bytes_le(&self.rows)}
    pub fn index(&self)->serde_json::Value {serde_json::json!({
        "schema":"biprism.lif.actual_work.v2","window":self.window,"attempted_current_body":self.current_attempt,
        "record_order":"window_body,leaf,local_vertex; live attempt omits window_body","row_bytes":464,"ledger_bytes":288,
        "reserved_row_slots":self.rows.len(),"ledger_slots":self.ledgers.len(),"summary_fields":LIF_WORK_VALUE_NAMES,
        "row_state":"1 captured,2 measured,3 collectively published; unproduced/stale storage remains raw telemetry",
        "native_work_consumer":"actual per-atom double work buffer, consumed by existing reduction and physical governor; divisor double418.4",
        "phase_work_precision":"promoted418.4f convention separately retained; no universal bitwise native-energy equivalence",
        "metric":"actual accepted C4 FP64 matrix (FP32 fields are diagnostic projections); core slots0..191, halo identity; only core main-mass fields belong to the main Hamiltonian",
        "masses":"true reciprocal of represented inverse in FP64 versus main FP32 reciprocal/floor result retained separately",
        "pre_lif_scope":"core gap between accepted C4 velocity and refreshed pre-delivery velocity; flexible mode records bridge gap, constrained mode also includes intervening native projection",
        "late_interventions":"UV,clamp,CA and RATTLE occur after this ledger and retain their existing work owners",
        "physical_momentum_gate":"qualified separately by actual native delivery; this ledger does not replace that gate",
        "scientific_restart":false})}
}
fn bytes_le<T>(x:&[T])->Result<&[u8],String>{if !cfg!(target_endian="little"){return Err("LIF work requires little endian".into());}Ok(unsafe{std::slice::from_raw_parts(x.as_ptr().cast(),core::mem::size_of_val(x))})}
pub fn journal_extents(capacity:u32)->Result<(usize,usize),String>{
    if !(1..=500).contains(&capacity){return Err("LIF work window must hold1..500 actual bodies".into());}
    let n=(capacity as usize).checked_mul(8).ok_or("LIF journal extent")?;Ok((n,n.checked_mul(256).ok_or("LIF row extent")?))
}
pub(crate) struct BiprismLifWorkOwner {
    stream:Arc<CudaStream>,bindings:CudaSlice<Binding>,ledgers:CudaSlice<LifWorkLedger>,rows:CudaSlice<LifWorkRow>,
    window:CudaSlice<LifWorkWindow>,journal_ledgers:CudaSlice<LifWorkLedger>,journal_rows:CudaSlice<LifWorkRow>,
    capture:CudaFunction,measure:CudaFunction,publish:CudaFunction,selected:[[u32;256];8],capacity:u32,current:Option<LifWorkWindow>,
}
impl BiprismLifWorkOwner {
    pub(crate) fn new(stream:Arc<CudaStream>,module:&Arc<CudaModule>,native:&[BiprismNativeLeafPointers;8],selected:[[u32;256];8],capacity:u32)->Result<Self,String>{
        let(n,r)=journal_extents(capacity)?;
        let bindings=std::array::from_fn::<_,8,_>(|i|Binding{native_hmr_mass_dalton:native[i].masses,native_atom_work_kcal:native[i].external_work});
        if bindings.iter().any(|b|b.native_hmr_mass_dalton==0||b.native_atom_work_kcal==0){return Err("actual native mass/work owner absent".into());}
        Ok(Self{capture:module.load_function("nhs_biprism_lif_work_capture_v1").map_err(|e|e.to_string())?,
            measure:module.load_function("nhs_biprism_lif_work_measure_v1").map_err(|e|e.to_string())?,
            publish:module.load_function("nhs_biprism_lif_work_publish_v1").map_err(|e|e.to_string())?,
            bindings:upload(&stream,&bindings)?,ledgers:upload(&stream,&[LifWorkLedger::default();8])?,
            rows:upload(&stream,&vec![LifWorkRow::default();8*256])?,window:upload(&stream,&[LifWorkWindow::default()])?,
            journal_ledgers:upload(&stream,&vec![LifWorkLedger::default();n])?,journal_rows:upload(&stream,&vec![LifWorkRow::default();r])?,
            stream,selected,capacity,current:None})
    }
    pub(crate) fn begin_window(&mut self,first_body:u32,step:u32,count:u32)->Result<(),String>{
        if self.current.is_some()||count==0||count>self.capacity||first_body.checked_add(count).is_none()
            ||step.checked_add(count).filter(|x|*x<u32::MAX).is_none(){return Err("LIF work window identity/range refused".into());}
        let w=LifWorkWindow{abi_magic:LIF_WORK_MAGIC,capacity:self.capacity,first_body,requested_steps:count,first_scheduled_step:step,reserved:[0;3]};
        let source=[w];
        let copy=self.stream.memcpy_htod(&source,&mut self.window).map_err(|e|e.to_string());
        let drain=self.stream.synchronize().map_err(|e|e.to_string());completed_transfer(copy,drain)?;
        self.current=Some(w);Ok(())
    }
    /// Device dt pointer belongs to the retained shared physical-step owner and
    /// stays frozen from common selection through accepted-time commit.
    pub(crate) unsafe fn capture(&self,v:BiprismDeviceViews,carry:u64,scheduled:u64,dt_au:u64,atoms:u32,mode:u32)->Result<(),String>{
        if dt_au==0||dt_au%4!=0{return Err("selected physical dt pointer absent".into());}
        let b=self.bindings.device_ptr(&self.stream).0;let l=self.ledgers.device_ptr(&self.stream).0;let r=self.rows.device_ptr(&self.stream).0;let w=self.window.device_ptr(&self.stream).0;
        self.stream.launch_builder(&self.capture).arg(&b).arg(&v.tiles).arg(&v.stream_table).arg(&v.native_stream_table)
            .arg(&carry).arg(&v.unit_bridge_status).arg(&v.convergence).arg(&scheduled).arg(&dt_au).arg(&atoms).arg(&mode)
            .arg(&l).arg(&r).arg(&w).launch(LaunchConfig{grid_dim:(8,1,1),block_dim:(256,1,1),shared_mem_bytes:0}).map_err(|e|e.to_string())?;Ok(())
    }
    pub(crate) unsafe fn measure_and_publish(&self,v:BiprismDeviceViews,carry:u64,scheduled:u64)->Result<(),String>{
        let b=self.bindings.device_ptr(&self.stream).0;let l=self.ledgers.device_ptr(&self.stream).0;let r=self.rows.device_ptr(&self.stream).0;let w=self.window.device_ptr(&self.stream).0;
        let jl=self.journal_ledgers.device_ptr(&self.stream).0;let jr=self.journal_rows.device_ptr(&self.stream).0;
        self.stream.launch_builder(&self.measure).arg(&v.stream_table).arg(&v.native_stream_table).arg(&carry)
            .arg(&v.unit_bridge_status).arg(&v.convergence).arg(&scheduled).arg(&l).arg(&r)
            .launch(LaunchConfig{grid_dim:(8,1,1),block_dim:(256,1,1),shared_mem_bytes:0}).map_err(|e|e.to_string())?;
        self.stream.launch_builder(&self.publish).arg(&b).arg(&v.convergence).arg(&l).arg(&r).arg(&w).arg(&jl).arg(&jr)
            .launch(LaunchConfig{grid_dim:(1,1,1),block_dim:(256,1,1),shared_mem_bytes:0}).map_err(|e|e.to_string())?;Ok(())
    }
    pub(crate) fn guard_pointers(&self)->(u64,u64,u64){(self.ledgers.device_ptr(&self.stream).0,self.window.device_ptr(&self.stream).0,self.journal_ledgers.device_ptr(&self.stream).0)}
    pub(crate) fn read_ledgers(&self)->Result<[LifWorkLedger;8],String>{download(&self.stream,&self.ledgers)?.try_into().map_err(|_|"LIF ledger extent".into())}
    pub(crate) fn read_window(&self)->Result<LifWorkReadback,String>{let w=self.current.ok_or("no attempted LIF work window")?;
        let mut ledgers=download(&self.stream,&self.journal_ledgers)?;let mut rows=download(&self.stream,&self.journal_rows)?;
        ledgers.truncate(w.requested_steps as usize*8);rows.truncate(w.requested_steps as usize*8*256);
        Ok(LifWorkReadback{window:Some(w),ledgers,rows,current_attempt:false})}
    pub(crate) fn read_current(&self)->Result<LifWorkReadback,String>{Ok(LifWorkReadback{window:self.current,
        ledgers:download(&self.stream,&self.ledgers)?,rows:download(&self.stream,&self.rows)?,current_attempt:true})}
    pub(crate) fn validate(&self,r:&LifWorkReadback)->Result<(),String>{validate_accepted(r,&self.selected)}
    pub(crate) fn acknowledge(&mut self){self.current=None;}
}
pub fn validate_accepted(r:&LifWorkReadback,selected:&[[u32;256];8])->Result<(),String>{
    let w=r.window.ok_or("missing work window")?;journal_extents(w.capacity)?;
    if r.current_attempt||w.abi_magic!=LIF_WORK_MAGIC||w.reserved!=[0;3]||w.requested_steps==0||w.requested_steps>w.capacity
        ||r.ledgers.len()!=w.requested_steps as usize*8||r.rows.len()!=r.ledgers.len()*256{return Err("LIF journal population mismatch".into());}
    for (i,l) in r.ledgers.iter().enumerate(){let leaf=i%8;let body=u64::from(w.first_body)+i as u64/8;let step=u64::from(w.first_scheduled_step)+i as u64/8;
        if step>=u32::MAX as u64||l.abi_magic!=LIF_WORK_MAGIC||l.leaf_id!=leaf as u32||l.state!=3||l.sticky_fault!=0
            ||l.body_ordinal!=body||l.completed_steps!=body+1||u64::from(l.scheduled_step)!=step
            ||u64::from(l.committed_substeps)!=(body+1)*4||l.pending_before>1||l.mode&!3!=0||(l.pending_before==1&&l.produced_step.checked_add(1)!=Some(l.scheduled_step))||l.reserved!=[0;2]||l.reserved_tail!=0
            ||!l.dt_feedback_au.is_finite()||l.dt_feedback_au<=0.0
            ||l.totals.iter().chain(&l.requested_impulse).chain(&l.realized_lab_impulse).chain(&l.realized_true_covariant_impulse).any(|x|!x.is_finite()){
            return Err(format!("LIF work ledger identity/finite check failed record{i}"));}
        let mut sum=[0.0f64;17];let mut impulse_sum=[[0.0f64;3];3];
        for local in 0..256{let p=&r.rows[i*256+local];
            if p.state!=3||p.leaf_id!=leaf as u32||p.local_vertex!=local as u32||p.atom_id!=selected[leaf][local]
                ||p.scheduled_step!=l.scheduled_step||p.committed_substeps!=l.committed_substeps
                ||p.source_produced_step!=l.produced_step||p.pending_before!=l.pending_before||p.reserved!=0||p.coordinate_reserved!=[0;2]||!p.finite()
                ||p.pre_lif_scope!=if local<192{if l.mode&1!=0{1}else{2}}else{0}{return Err(format!("LIF work row identity/finite failure record{i} vertex{local}"));}
            for(k,v)in p.values().into_iter().enumerate(){sum[k]+=v;}
            for(k,vector)in [p.requested_impulse,p.realized_lab_impulse,p.realized_true_covariant_impulse].into_iter().enumerate(){for a in 0..3{impulse_sum[k][a]+=vector[a];}}
            if local>=192&&(p.main_mass_metric_kinetic_delta!=0.0||p.main_vs_true_metric_mass_difference!=0.0
                ||p.pre_lif_true_metric_delta!=0.0||p.pre_lif_main_metric_delta!=0.0){return Err("halo mislabelled as core main energy".into());}
            if l.pending_before==0&&(p.requested_force_au!=[0.0;3]||p.au_velocity_after!=p.au_velocity_before||p.native_velocity_after!=p.native_velocity_before){return Err("no-delivery body changed selected velocities".into());}
        }
        for k in 0..17{if !sum[k].is_finite()||(sum[k]-l.totals[k]).abs()>1e-12*(1.0+sum[k].abs()){
            return Err(format!("LIF work summary differs from retained rows record{i} field{k}"));}}
        for(k,vector)in [l.requested_impulse,l.realized_lab_impulse,l.realized_true_covariant_impulse].into_iter().enumerate(){for a in 0..3{
            let v=impulse_sum[k][a];if !v.is_finite()||(v-vector[a]).abs()>1e-12*(1.0+v.abs()){
                return Err(format!("LIF impulse summary differs from retained rows record{i} vector{k} axis{a}"));}}}
    }Ok(())
}
fn completed_transfer<T>(copy:Result<T,String>,drain:Result<(),String>)->Result<T,String>{
    match(copy,drain){(Ok(v),Ok(()))=>Ok(v),(Err(copy),Ok(()))=>Err(copy),
        (Ok(_),Err(drain))=>Err(format!("transfer completion failed: {drain}")),
        (Err(copy),Err(drain))=>Err(format!("transfer failed: {copy}; completion failed: {drain}"))}
}
fn upload<T:DeviceRepr+Copy>(s:&Arc<CudaStream>,x:&[T])->Result<CudaSlice<T>,String>{
    let copy=s.clone_htod(x).map_err(|e|e.to_string());
    let drain=s.synchronize().map_err(|e|e.to_string());completed_transfer(copy,drain)
}
fn download<T:DeviceRepr+Copy+Default>(s:&Arc<CudaStream>,x:&CudaSlice<T>)->Result<Vec<T>,String>{
    let mut out=vec![T::default();x.len()];
    let copy=s.memcpy_dtoh(x,&mut out).map_err(|e|e.to_string());
    let drain=s.synchronize().map_err(|e|e.to_string());completed_transfer(copy,drain)?;Ok(out)
}

#[cfg(test)]mod transfer_error_tests{
 use super::completed_transfer;
 #[test]fn preserves_transfer_and_completion_errors(){
  assert_eq!(completed_transfer(Ok(7),Ok(())),Ok(7));
  assert_eq!(completed_transfer::<u32>(Err("copy".into()),Ok(())),Err("copy".into()));
  assert!(completed_transfer(Ok(7),Err("drain".into())).unwrap_err().contains("drain"));
  let e=completed_transfer::<u32>(Err("copy".into()),Err("drain".into())).unwrap_err();
  assert!(e.contains("copy")&&e.contains("drain"));
 }
}
