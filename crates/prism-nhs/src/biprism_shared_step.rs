//! Coupled physical-time owner: native requests, common selection, immutable
//! per-body dt, and an accepted-only cumulative clock. Native per-leaf attempted
//! journals may be ahead after refusal and are never relabelled or reset.
use crate::protocol_state::ProtocolState;
use crate::fused_engine::BiprismNativeLeafPointers;
use cudarc::driver::{CudaFunction,CudaModule,CudaSlice,CudaStream,DevicePtr,DeviceRepr,LaunchConfig,PushKernelArg};
use serde::{Serialize,Deserialize};
use std::sync::Arc;
pub const SHARED_STEP_MAGIC:u32=0x42505354;
/// Explicit common-time bounds. Requests below the floor refuse; no safety
/// request is increased. The upper bound caps the common minimum only.
#[derive(Clone,Copy,Debug,Serialize,Deserialize,PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AdaptiveDtBounds {pub dt_min_ps:f32,pub dt_max_ps:f32}
impl AdaptiveDtBounds {
    pub fn validate(self)->Result<(),String>{if !self.dt_min_ps.is_finite()||!self.dt_max_ps.is_finite()
        ||self.dt_min_ps<=0.||self.dt_max_ps<self.dt_min_ps {Err("invalid adaptive timestep bounds".into())}else{Ok(())}}
    fn words(self)->[u32;3]{[2,self.dt_min_ps.to_bits(),self.dt_max_ps.to_bits()]}
}
pub fn select_bounded_dt(requests:[f32;8],bounds:Option<AdaptiveDtBounds>)->Result<f32,String>{
    if requests.iter().any(|d|!d.is_finite()||*d<=0.){return Err("invalid native timestep request".into());}
    let minimum=requests.into_iter().fold(f32::INFINITY,f32::min);
    if let Some(b)=bounds{b.validate()?;if minimum<b.dt_min_ps{return Err("native safety request below governed minimum".into());}
        Ok(minimum.min(b.dt_max_ps))}else{Ok(minimum)}
}
fn window_bounds(w:&BiprismSharedStepWindowFfi)->Result<Option<AdaptiveDtBounds>,String>{
    if w.reserved==[0;3]{return Ok(None);}
    if w.reserved[0]!=2{return Err("unknown shared-time selection policy".into());}
    let b=AdaptiveDtBounds{dt_min_ps:f32::from_bits(w.reserved[1]),dt_max_ps:f32::from_bits(w.reserved[2])};b.validate()?;Ok(Some(b))
}
#[repr(C,align(128))]
#[derive(Clone,Copy,Debug,Default,Serialize)]
pub struct BiprismSharedStepFfi {
    pub abi_magic:u32,pub state:u32,pub sticky_fault:u32,pub scheduled_step:u32,
    pub last_accepted_step:u32,pub accepted_bodies:u32,pub terminal_frozen_mask:u32,pub policy:u32,
    pub native_dt_ps:f32,pub outer_dt_au:f32,pub minimum_request_ps:f32,pub maximum_request_ps:f32,
    pub cumulative_accepted_time_ps:f64,pub generation:u64,pub requested_dt_ps:[f32;8],pub protocol_control_step:[u32;8],
}
unsafe impl DeviceRepr for BiprismSharedStepFfi {}
#[repr(C,align(16))]
#[derive(Clone,Copy,Debug,Default,Serialize)]
pub struct BiprismTerminalControlsFfi {
    pub abi_magic:u32,pub leaf_id:u32,pub terminal_step:u32,pub captured_step:u32,
    pub temperature:f32,pub effective_gamma:f32,pub requested_dt_ps:f32,pub uv_wavelength_nm:f32,
    pub uv_burst_active:i32,pub uv_target_idx:i32,pub phase_bits:u32,pub reserved:u32,
}
unsafe impl DeviceRepr for BiprismTerminalControlsFfi {}
#[repr(C,align(16))]
#[derive(Clone,Copy,Debug,Default,Serialize)]
pub struct BiprismProtocolStepRowFfi {
    pub baseline:BiprismTerminalControlsFfi,
    pub current_temperature:f32,pub effective_gamma:f32,pub requested_dt_ps:f32,pub uv_wavelength_nm:f32,
    pub uv_burst_active:i32,pub uv_target_idx:i32,pub phase_bits:u32,pub steering_flags:u32,
    pub steering_uv_boost:f32,pub steering_temp_bias:f32,pub steering_focus_residue:i32,pub reserved:u32,
}
unsafe impl DeviceRepr for BiprismProtocolStepRowFfi {}
#[repr(C,align(128))]
#[derive(Clone,Copy,Debug,Default,Serialize)]
pub struct BiprismSharedStepJournalFfi {pub step:BiprismSharedStepFfi,pub leaves:[BiprismProtocolStepRowFfi;8]}
unsafe impl DeviceRepr for BiprismSharedStepJournalFfi {}
#[repr(C,align(16))]
#[derive(Clone,Copy,Debug,Default,Serialize)]
pub struct BiprismSharedStepWindowFfi {
    pub magic:u32,pub capacity:u32,pub first_body:u32,pub requested_bodies:u32,pub first_scheduled_step:u32,pub reserved:[u32;3],
}
unsafe impl DeviceRepr for BiprismSharedStepWindowFfi {}
const _:()={
    assert!(core::mem::size_of::<BiprismSharedStepFfi>()==128);
    assert!(core::mem::offset_of!(BiprismSharedStepFfi,native_dt_ps)==32);
    assert!(core::mem::offset_of!(BiprismSharedStepFfi,outer_dt_au)==36);
    assert!(core::mem::offset_of!(BiprismSharedStepFfi,generation)==56);
    assert!(core::mem::size_of::<BiprismTerminalControlsFfi>()==48);
    assert!(core::mem::size_of::<BiprismProtocolStepRowFfi>()==96);
    assert!(core::mem::size_of::<BiprismSharedStepJournalFfi>()==896);
    assert!(core::mem::size_of::<BiprismSharedStepWindowFfi>()==32);
};
#[derive(Clone,Copy,Debug)]
pub(crate) struct BiprismSelectedRequestBinding {
    pub shared:u64,pub terminal:u64,pub baseline:u64,pub original_fixed_request_ps:f32,pub generation:u64,pub leaf:u32,
}
#[derive(Debug,Serialize)]
pub struct BiprismSharedStepAttempt {
    pub shared:BiprismSharedStepFfi,pub terminal:[BiprismTerminalControlsFfi;8],pub current_baseline:[BiprismTerminalControlsFfi;8],
}
pub struct BiprismSharedStepReadback {
    pub window:BiprismSharedStepWindowFfi,pub before_time_ps:f64,pub generation:u64,
    pub terminal_steps:[u32;8],pub terminal_snapshots:[BiprismTerminalControlsFfi;8],pub records:Vec<BiprismSharedStepJournalFfi>,
}
impl BiprismSharedStepReadback {
    pub fn bytes(&self)->Result<&[u8],String>{if !cfg!(target_endian="little"){return Err("shared time journal requires little endian".into());}
        Ok(unsafe{std::slice::from_raw_parts(self.records.as_ptr().cast(),core::mem::size_of_val(self.records.as_slice()))})}
    pub fn index(&self)->serde_json::Value {serde_json::json!({"schema":"biprism.shared_physical_time.v1","record_bytes":896,
        "window":self.window,"adaptive_dt_bounds":window_bounds(&self.window).ok().flatten(),"selection_policy":"retain raw requests; min(requests, configured maximum); refuse below configured minimum", "generation":self.generation,"before_time_ps":self.before_time_ps,"terminal_steps":self.terminal_steps,
        "terminal_snapshots":self.terminal_snapshots,"records":self.records.len(),"clock_scope":"one common represented native dt per collectively accepted body; sum in double",
        "protocol_scope":"terminal exogenous baseline held; endogenous ASC modifiers and physical counters remain live",
        "native_journals":"per-leaf attempted/completed rows may precede collective acceptance; no all-eight rollback or retry",
        "per_step_lif_decay":"unchanged observation-based model; not silently rescaled by dt","scientific_restart":false})}
}
pub(crate) struct BiprismSharedStepOwner {
    stream:Arc<CudaStream>,record:Arc<CudaSlice<BiprismSharedStepFfi>>,terminal:CudaSlice<BiprismTerminalControlsFfi>,baseline:CudaSlice<BiprismTerminalControlsFfi>,
    protocol_pointers:CudaSlice<u64>,window:CudaSlice<BiprismSharedStepWindowFfi>,journal:CudaSlice<BiprismSharedStepJournalFfi>,
    select:CudaFunction,accept:CudaFunction,fixed_requests:[f32;8],terminal_steps:[u32;8],generation:u64,capacity:u32,
    current:Option<(BiprismSharedStepWindowFfi,f64)>,bounds:Option<AdaptiveDtBounds>,
}
impl BiprismSharedStepOwner {
    pub(crate) fn new(stream:Arc<CudaStream>,module:&Arc<CudaModule>,native:&[BiprismNativeLeafPointers;8],
        protocols:[ProtocolState;8],generation:u64,capacity:u32,bounds:Option<AdaptiveDtBounds>)->Result<Self,String>{
        if let Some(b)=bounds{b.validate()?;}
        if generation==0||!(1..=500).contains(&capacity){return Err("shared time generation/capacity invalid".into());}
        let mut fixed_requests=[0f32;8];let mut terminal_steps=[0u32;8];let mut terminal=[BiprismTerminalControlsFfi::default();8];
        for i in 0..8 {
            let p=&protocols[i];
            if p.current_step!=0||p.total_steps==0||p.total_steps>i32::MAX as u32
                ||!p.dt.is_finite()||p.dt<=0.||!p.base_dt.is_finite()||p.base_dt<=0.
                ||!(0..=1).contains(&p.adaptive_dt_enabled)||native[i].protocol==0
                ||native[..i].iter().any(|n|n.protocol==native[i].protocol){return Err(format!("shared time cold native protocol identity invalid leaf{i}"));}
            fixed_requests[i]=p.dt;terminal_steps[i]=p.total_steps;
            terminal[i]=BiprismTerminalControlsFfi{abi_magic:SHARED_STEP_MAGIC,leaf_id:i as u32,terminal_step:p.total_steps,..Default::default()};
        }
        let record=BiprismSharedStepFfi{abi_magic:SHARED_STEP_MAGIC,policy:if bounds.is_some(){2}else{1},generation,..Default::default()};
        Ok(Self{select:module.load_function("nhs_biprism_shared_step_select_v1").map_err(|e|e.to_string())?,
            accept:module.load_function("nhs_biprism_shared_step_accept_v1").map_err(|e|e.to_string())?,
            record:Arc::new(upload(&stream,&[record])?),terminal:upload(&stream,&terminal)?,baseline:upload(&stream,&[BiprismTerminalControlsFfi::default();8])?,
            protocol_pointers:upload(&stream,&native.iter().map(|p|p.protocol).collect::<Vec<_>>())?,
            window:upload(&stream,&[BiprismSharedStepWindowFfi::default()])?,journal:upload(&stream,&vec![BiprismSharedStepJournalFfi::default();capacity as usize])?,
            stream,fixed_requests,terminal_steps,generation,capacity,current:None,bounds})
    }
    pub(crate) fn retained_record(&self)->Arc<CudaSlice<BiprismSharedStepFfi>>{self.record.clone()}
    pub(crate) fn pointer(&self)->u64{self.record.device_ptr(&self.stream).0}
    pub(crate) fn acceptance_pointers(&self)->(u64,u64){(self.window.device_ptr(&self.stream).0,self.journal.device_ptr(&self.stream).0)}
    pub(crate) fn dt_au_pointer(&self)->u64{self.pointer()+36}
    pub(crate) fn generation(&self)->u64{self.generation}
    pub(crate) fn request(&self,leaf:usize)->Result<BiprismSelectedRequestBinding,String>{if leaf>=8{return Err("invalid request leaf".into());}
        Ok(BiprismSelectedRequestBinding{shared:self.pointer(),terminal:self.terminal.device_ptr(&self.stream).0+leaf as u64*48,
            baseline:self.baseline.device_ptr(&self.stream).0+leaf as u64*48,original_fixed_request_ps:self.fixed_requests[leaf],generation:self.generation,leaf:leaf as u32})}
    pub(crate) fn begin_window(&mut self,first_body:u32,step:u32,count:u32)->Result<(),String>{
        if self.current.is_some()||count==0||count>self.capacity||first_body.checked_add(count).filter(|x|*x<=0x3fffffff).is_none()
            ||step.checked_add(count).filter(|x|*x<u32::MAX).is_none(){return Err("invalid shared physical window".into());}
        let s=download(&self.stream,&self.record)?[0];
        if s.abi_magic!=SHARED_STEP_MAGIC||s.generation!=self.generation||s.sticky_fault!=0||s.accepted_bodies!=first_body
            ||s.last_accepted_step.checked_add(1)!=Some(step)||(first_body==0&&s.state!=0)||(first_body!=0&&s.state!=2)
            ||!s.cumulative_accepted_time_ps.is_finite()||s.cumulative_accepted_time_ps<0.{return Err("shared time not quiescent at expected accepted epoch".into());}
        let w=BiprismSharedStepWindowFfi{magic:SHARED_STEP_MAGIC,capacity:self.capacity,first_body,requested_bodies:count,first_scheduled_step:step,reserved:self.bounds.map_or([0;3],AdaptiveDtBounds::words)};
        let source=[w];
        let copy=self.stream.memcpy_htod(&source,&mut self.window).map_err(|e|e.to_string());
        let drain=self.stream.synchronize().map_err(|e|e.to_string());completed_transfer(copy,drain)?;
        self.current=Some((w,s.cumulative_accepted_time_ps));Ok(())
    }
    pub(crate) unsafe fn select(&self,scheduled:u64,convergence:u64)->Result<(),String>{let r=self.pointer();let p=self.protocol_pointers.device_ptr(&self.stream).0;
        let b=self.baseline.device_ptr(&self.stream).0;let t=self.terminal.device_ptr(&self.stream).0;let w=self.window.device_ptr(&self.stream).0;let j=self.journal.device_ptr(&self.stream).0;
        self.stream.launch_builder(&self.select).arg(&r).arg(&p).arg(&b).arg(&t).arg(&scheduled).arg(&self.generation)
            .arg(&convergence).arg(&w).arg(&j).launch(LaunchConfig{grid_dim:(1,1,1),block_dim:(1,1,1),shared_mem_bytes:0}).map_err(|e|e.to_string())?;Ok(())}
    pub(crate) unsafe fn accept(&self,control:u64,scheduled:u64,convergence:u64)->Result<(),String>{let r=self.pointer();let w=self.window.device_ptr(&self.stream).0;let j=self.journal.device_ptr(&self.stream).0;
        self.stream.launch_builder(&self.accept).arg(&r).arg(&control).arg(&convergence).arg(&scheduled).arg(&self.generation).arg(&w).arg(&j)
            .launch(LaunchConfig{grid_dim:(1,1,1),block_dim:(1,1,1),shared_mem_bytes:0}).map_err(|e|e.to_string())?;Ok(())}
    pub(crate) fn read_current(&self)->Result<BiprismSharedStepAttempt,String>{Ok(BiprismSharedStepAttempt{shared:download(&self.stream,&self.record)?[0],
        terminal:download(&self.stream,&self.terminal)?.try_into().map_err(|_|"terminal extent")?,current_baseline:download(&self.stream,&self.baseline)?.try_into().map_err(|_|"baseline extent")?})}
    pub(crate) fn read_window(&self)->Result<BiprismSharedStepReadback,String>{let(w,before_time_ps)=self.current.ok_or("no attempted shared time window")?;
        let mut records=download(&self.stream,&self.journal)?;records.truncate(w.requested_bodies as usize);
        Ok(BiprismSharedStepReadback{window:w,before_time_ps,generation:self.generation,terminal_steps:self.terminal_steps,terminal_snapshots:download(&self.stream,&self.terminal)?.try_into().map_err(|_|"terminal extent")?,records})}
    pub(crate) fn acknowledge(&mut self){self.current=None;}
}
pub fn validate_shared_time_journal(r:&BiprismSharedStepReadback)->Result<(),String>{let w=r.window;let bounds=window_bounds(&w)?;
    if w.magic!=SHARED_STEP_MAGIC||w.capacity==0||w.capacity>500||w.requested_bodies==0||w.requested_bodies>w.capacity
        ||r.records.len()!=w.requested_bodies as usize||r.generation==0||!r.before_time_ps.is_finite()||r.before_time_ps<0.{return Err("invalid shared time journal extent".into());}
    let mut time=r.before_time_ps;
    let last_step=u64::from(w.first_scheduled_step)+u64::from(w.requested_bodies)-1;
    for leaf in 0..8{let t=&r.terminal_snapshots[leaf];let target=r.terminal_steps[leaf];
        if t.abi_magic!=SHARED_STEP_MAGIC||t.leaf_id!=leaf as u32||t.terminal_step!=target||t.reserved!=0
            ||t.captured_step!=if last_step>=u64::from(target){target}else{0}{return Err("terminal snapshot identity differs from attempted window".into());}}

    for(i,j)in r.records.iter().enumerate(){let s=j.step;let body=u64::from(w.first_body)+i as u64+1;let step=u64::from(w.first_scheduled_step)+i as u64;
        if step>=u32::MAX as u64||body>0x3fffffff||s.abi_magic!=SHARED_STEP_MAGIC||s.state!=2||s.sticky_fault!=0||s.policy!=if bounds.is_some(){2}else{1}||s.generation!=r.generation
            ||u64::from(s.scheduled_step)!=step||u64::from(s.last_accepted_step)!=step||u64::from(s.accepted_bodies)!=body{return Err(format!("shared time accepted identity invalid body{i}"));}
        if s.requested_dt_ps.iter().any(|d|!d.is_finite()||*d<=0.){return Err("nonfinite/nonpositive requested dt".into());}
        let lo=s.requested_dt_ps.iter().copied().fold(f32::INFINITY,f32::min);let hi=s.requested_dt_ps.iter().copied().fold(0f32,f32::max);
        let chosen=select_bounded_dt(s.requested_dt_ps,bounds)?;
        if s.native_dt_ps.to_bits()!=chosen.to_bits()||s.minimum_request_ps.to_bits()!=lo.to_bits()||s.maximum_request_ps.to_bits()!=hi.to_bits()
            ||s.outer_dt_au.to_bits()!=((f64::from(chosen)*41341.37333517015)as f32).to_bits(){return Err("selected dt differs from original request minimum or represented AU conversion".into());}
        time+=f64::from(chosen);if !time.is_finite()||time.to_bits()!=s.cumulative_accepted_time_ps.to_bits(){return Err("accepted time does not sum actual represented chosen dt".into());}
        let mut mask=0;
        for leaf in 0..8{let p=&j.leaves[leaf];let b=&p.baseline;let terminal=r.terminal_steps[leaf];let control=(step as u32).min(terminal);
            if terminal==0||control==0||b.abi_magic!=SHARED_STEP_MAGIC||b.leaf_id!=leaf as u32||b.terminal_step!=terminal||b.captured_step!=control
                ||b.reserved!=0||p.reserved!=0||s.protocol_control_step[leaf]!=control||p.phase_bits!=b.phase_bits
                ||b.phase_bits!=(((control as f32/terminal as f32)*1024.)as u32)&0x3ff
                ||p.requested_dt_ps.to_bits()!=s.requested_dt_ps[leaf].to_bits()||b.requested_dt_ps.to_bits()!=p.requested_dt_ps.to_bits()
                ||p.effective_gamma.to_bits()!=b.effective_gamma.to_bits()||p.uv_wavelength_nm.to_bits()!=b.uv_wavelength_nm.to_bits()
                ||p.uv_burst_active!=b.uv_burst_active||p.uv_target_idx!=b.uv_target_idx
                ||!(0..=1).contains(&p.uv_burst_active)
                ||[b.temperature,b.effective_gamma,b.uv_wavelength_nm,p.current_temperature,p.steering_uv_boost,p.steering_temp_bias].iter().any(|x|!x.is_finite())
                ||b.temperature<=0.||p.current_temperature<=0.||b.effective_gamma<0.{return Err(format!("shared protocol baseline/current identity invalid body{i} leaf{leaf}"));}
            if step>=u64::from(terminal){let t=&r.terminal_snapshots[leaf];
                if b.temperature.to_bits()!=t.temperature.to_bits()||b.effective_gamma.to_bits()!=t.effective_gamma.to_bits()
                    ||b.requested_dt_ps.to_bits()!=t.requested_dt_ps.to_bits()||b.uv_wavelength_nm.to_bits()!=t.uv_wavelength_nm.to_bits()
                    ||b.uv_burst_active!=t.uv_burst_active||b.uv_target_idx!=t.uv_target_idx||b.phase_bits!=t.phase_bits{
                    return Err("terminal baseline changed relative to actual retained terminal snapshot".into());}}
            if step>u64::from(terminal){mask|=1u32<<leaf;}

        }
        if s.terminal_frozen_mask!=mask{return Err("terminal mask differs from actual control steps".into());}
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
