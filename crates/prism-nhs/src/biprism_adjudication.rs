//! Accepted-window native ASC decision producer. This is a child of
//! captured_pipeline so it reuses the existing native DynamicT7 entry point.
//! It does not replace the physical WHILE body or infer a decision from host scores.
use super::prism_dynamic_t7_launch;
use crate::{asc_device::{AscDevice,AscSnapshot}, fused_engine::{BiprismNativeLeafPointers,GpuSpikeEvent},
    graph_capture::{capture_pure_template,CaptureTagger}, interferometric_adjudicator::{self,InterferometricAdjudicatorFfi},
    rich_spike::{RichSpike,NativeV2SeedPopulation}, so3_project::{self,ContactShellTile}};
use cudarc::driver::{sys,CudaSlice,CudaStream,DevicePtr,DeviceRepr};
use std::{any::Any,ffi::c_void,sync::Arc};
use serde_json::{json,Value};

/// Exact existing resident Path-Ω4³ cell partition, retaining every input row
/// and its original queue ordinal. This is distinct from PathB site clustering.
pub struct BiprismAdjudicationSeeds {
    pub rich:Vec<RichSpike>,pub offsets:Vec<u32>,pub original_ordinals:Vec<u32>,
    pub cell_ids:Vec<u32>,pub bbox:[[f32;3];2],pub cell_size:[f32;3],
}
pub fn prepare_adjudication_seeds(events:&[(u32,GpuSpikeEvent)],begin:u32,end:u32)->Result<BiprismAdjudicationSeeds,String>{
    if begin>=end||events.is_empty(){return Err("adjudication requires a nonempty accepted observation window".into());}
    let mut values=Vec::with_capacity(events.len());let mut seen=std::collections::HashSet::new();
    let (mut lo,mut hi)=([f32::INFINITY;3],[f32::NEG_INFINITY;3]);
    for (ordinal,event) in events {
        if !seen.insert(*ordinal)||event.timestep<=i32::try_from(begin).map_err(|_|"begin clock exceeds native event domain")?
            ||u32::try_from(event.timestep).map_err(|_|"negative native clock")?>end{return Err("duplicate queue ordinal or event outside accepted interval".into());}
        let (rich,_)=crate::coupled_md::rich_spike_from_gpu_event(event,0).map_err(|e|e.to_string())?;
        for (d,x) in [rich.x,rich.y,rich.z].iter().enumerate(){lo[d]=lo[d].min(*x);hi[d]=hi[d].max(*x);}
        values.push((*ordinal,rich));
    }
    for d in 0..3{let span=(hi[d]-lo[d]).max(1.);lo[d]-=span*0.05;hi[d]+=span*0.05;}
    let cell_size=std::array::from_fn(|d|((hi[d]-lo[d])/4.).max(0.1));
    if lo.iter().chain(hi.iter()).chain(cell_size.iter()).any(|x|!x.is_finite()) {return Err("adjudication spatial extent overflow".into());}
    let mut paired:Vec<_>=values.into_iter().map(|(ordinal,s)|{
        let p=[s.x,s.y,s.z];let c:[u32;3]=std::array::from_fn(|d|(((p[d]-lo[d])/cell_size[d])as i32).clamp(0,3)as u32);
        (c[0]+4*c[1]+16*c[2],ordinal,s)
    }).collect();paired.sort_by_key(|x|x.0);
    let mut rich=Vec::with_capacity(paired.len());let mut original_ordinals=Vec::with_capacity(paired.len());
    let mut offsets=vec![0];let mut cell_ids=Vec::new();let mut prior=None;
    for (i,(cell,ordinal,mut s))in paired.into_iter().enumerate(){
        if prior!=Some(cell){if prior.is_some(){offsets.push(u32::try_from(i).map_err(|_|"CSR index overflow")?);}cell_ids.push(cell);prior=Some(cell);}
        s.cluster_id=(cell_ids.len()-1)as i32;rich.push(s);original_ordinals.push(ordinal);
    }
    offsets.push(u32::try_from(rich.len()).map_err(|_|"seed extent overflow")?);
    if cell_ids.is_empty()||cell_ids.len()>64{return Err("native NodeC cluster domain exceeded".into());}
    Ok(BiprismAdjudicationSeeds{rich,offsets,original_ordinals,cell_ids,bbox:[lo,hi],cell_size})
}

/// Analysis-residue anchors use the actual topology map. The established native
/// first-atom fallback is retained explicitly for analysis residues with no Cα.
pub fn adjudication_anchor_lut(atom_residues:&[i32],ca_atoms:&[usize],n_residues:usize)->Result<(Vec<u32>,Vec<u32>),String>{
    if n_residues==0||n_residues>1024{return Err("native GASP LUT requires1..1024 analysis residues".into());}
    let mut table=vec![u32::MAX;1024];let mut ca=vec![false;n_residues];
    for (atom,&rid)in atom_residues.iter().enumerate(){if rid>=0{let r=rid as usize;if r>=n_residues{return Err("analysis residue outside declared axis".into());}if table[r]==u32::MAX{table[r]=u32::try_from(atom).map_err(|_|"atom LUT overflow")?;}}}
    for &atom in ca_atoms {let &r=atom_residues.get(atom).ok_or("Cα atom outside topology")?;if r>=0{let r=r as usize;if ca[r]&&table[r]!=atom as u32{return Err("multiple Cα anchors on one analysis residue".into());}table[r]=atom as u32;ca[r]=true;}}
    if table[..n_residues].contains(&u32::MAX){return Err("analysis residue has no physical atom anchor".into());}
    let fallback=(0..n_residues).filter(|&r|!ca[r]).map(|r|r as u32).collect();Ok((table,fallback))
}

pub struct BiprismAscDecisionReadback {
    pub accepted:bool,pub error:Option<String>,pub index:Value,
    pub adjudicator:Vec<u8>,pub relaxed_tiles:Vec<u8>,pub perturbed_tiles:Vec<u8>,
    pub calibration:Vec<u8>,pub asc_snapshot:Option<AscSnapshot>,
}
/// Persistent calibration belongs to one actual native ASC owner/stream. Engine
/// retention is thread-bound. No constructor can manufacture completed epochs.
pub struct BiprismAscDecisionOwner {
    stream:Arc<CudaStream>,owner:Arc<AscDevice>,native:BiprismNativeLeafPointers,
    _engine:Arc<dyn Any>,adj:CudaSlice<u8>,calibration:CudaSlice<u64>,compat_index:CudaSlice<u32>,compat_stats:CudaSlice<f32>,
    anchors:Vec<u32>,fallback:Vec<u32>,last_boundary:u32,poisoned:bool,
}
impl BiprismAscDecisionOwner {
    /// # Safety
    /// engine owns native pointers, all work uses this one stream/host thread,
    /// and no competing campaign may overwrite the module-global anchor LUT.
    pub unsafe fn new(stream:Arc<CudaStream>,owner:Arc<AscDevice>,engine:Arc<dyn Any>,native:BiprismNativeLeafPointers,
        anchors:Vec<u32>,fallback:Vec<u32>,completed_step:u32)->Result<Self,String>{
        if !owner.owns_stream(&stream)||owner.n_atoms()!=native.n_atoms||anchors.len()!=1024
            ||native.protocol_current_step==0||native.protocol_dt==0||native.forces==0||native.masses==0{
            return Err("native ASC decision binding differs from physical owner".into());
        }
        stream.context().bind_to_thread().map_err(|e|e.to_string())?;
        let mut init=InterferometricAdjudicatorFfi::zero();init.d_dt=native.protocol_dt as *mut f32;
        init.d_external_work=native.external_work as *mut f64;
        let bytes=core::slice::from_raw_parts((&init as *const InterferometricAdjudicatorFfi).cast::<u8>(),256);
        let adj=stream.clone_htod(bytes).map_err(|e|e.to_string())?;
        let calibration=stream.alloc_zeros(4).map_err(|e|e.to_string())?;
        let compat_index=stream.alloc_zeros(1).map_err(|e|e.to_string())?;
        let compat_stats=stream.alloc_zeros(2).map_err(|e|e.to_string())?;
        check(interferometric_adjudicator::apply_t7_kl_calibration(ptr(&adj,&stream)as *mut _,stream.cu_stream().cast()),"native T7 bootstrap")?;
        stream.synchronize().map_err(|e|e.to_string())?;
        Ok(Self{stream,owner,native,_engine:engine,adj,calibration,compat_index,compat_stats,anchors,fallback,last_boundary:completed_step,poisoned:false})
    }
    pub fn seed_manifest(&self,seeds:&BiprismAdjudicationSeeds,begin:u32,end:u32)->Value{
        let packed=NativeV2SeedPopulation::prepare(&seeds.rich);
        json!({"source_stream":self.owner.source_stream_id(),"window_start_exclusive":begin,"window_end_inclusive":end,
            "all_native_rows":seeds.rich.len(),"original_queue_ordinals":seeds.original_ordinals,"csr_offsets":seeds.offsets,"dense_cell_ids":seeds.cell_ids,
            "bbox_angstrom":seeds.bbox,"cell_size_angstrom":seeds.cell_size,"seed_layout":format!("{:?}",packed.layout),
            "anchor_analysis_residue_to_atom":self.anchors,"anchor_fallback_residues":self.fallback,
            "causal_chemical_scope":"existing rich_spike_from_gpu_event sets unavailable causal_lag/chem_flags/kinetic_delta to0; no causal or chemical estimate invented",
            "execution":"one captured native boundary graph after accepted physical window; pending ASC applies at next native begin",
            "physical_integrator_clock_increment":false,"scientific_restart":false})
    }
    pub unsafe fn run_boundary(&mut self,seeds:&BiprismAdjudicationSeeds,begin:u32,end:u32,successful_bodies:u32)->Result<BiprismAscDecisionReadback,String>{
        if self.poisoned||begin!=self.last_boundary||end<=begin{return Err("ASC decision boundary stale or poisoned".into());}
        let before=self.owner.snapshot().map_err(|e|format!("ASC predecision snapshot rc={e}"))?;
        if before.completed_epoch!=u64::from(successful_bodies)||before.physical_epoch!=before.completed_epoch
            ||before.published_epoch>=before.completed_epoch||before.coverage_failures!=0
            ||(before.status!=0&&before.status!=1){return Err("ASC decision has no accepted completed native mixed epoch".into());}
        let clock=read_u32(&self.stream,self.native.protocol_current_step)?;
        if clock!=end{return Err("ASC producer protocol clock differs from accepted window".into());}
        for s in &seeds.rich {if s.residue_id<0||s.residue_id>=1024||self.anchors[s.residue_id as usize]==u32::MAX{return Err("native seed lacks retained analysis-residue anchor".into());}}
        check(so3_project::set_residue_to_calpha(self.anchors.as_ptr(),1024,self.stream.cu_stream().cast()),"native anchor upload")?;
        crate::sh_basis::init_device_basis(self.stream.cu_stream().cast()).map_err(|e|format!("native SH basis rc={e}"))?;
        let basis=crate::sh_basis::k_lm_device_ptr().map_err(|e|format!("native SH pointer rc={e}"))?;
        let packed=NativeV2SeedPopulation::prepare(&seeds.rich);
        let d_seed=self.stream.clone_htod(packed.bytes(&seeds.rich)).map_err(|e|e.to_string())?;
        let offsets=self.stream.clone_htod(&seeds.offsets).map_err(|e|e.to_string())?;
        let clusters=(seeds.offsets.len()-1)as u32;let count=u32::try_from(seeds.rich.len()).map_err(|_|"seed count overflow")?;
        let relaxed=self.stream.alloc_zeros::<u8>(clusters as usize*1280).map_err(|e|e.to_string())?;
        let perturbed=self.stream.alloc_zeros::<u8>(clusters as usize*1280).map_err(|e|e.to_string())?;
        let positions=self.stream.alloc_zeros::<f32>(count as usize*3).map_err(|e|e.to_string())?;
        let com=self.stream.alloc_zeros::<f32>(3).map_err(|e|e.to_string())?;
        let mass=self.stream.alloc_zeros::<f32>(1).map_err(|e|e.to_string())?;
        let correction=self.stream.alloc_zeros::<f32>(3).map_err(|e|e.to_string())?;
        let mut adj_host=self.stream.clone_dtoh(&self.adj).map_err(|e|e.to_string())?;
        self.stream.synchronize().map_err(|e|e.to_string())?;
        adj_host[56..64].copy_from_slice(&ptr(&relaxed,&self.stream).to_ne_bytes());
        adj_host[64..72].copy_from_slice(&ptr(&perturbed,&self.stream).to_ne_bytes());
        adj_host[144..148].copy_from_slice(&0u32.to_ne_bytes());
        self.stream.memcpy_htod(&adj_host,&mut self.adj).map_err(|e|e.to_string())?;
        self.stream.synchronize().map_err(|e|e.to_string())?;
        let raw=self.stream.cu_stream();let adj=ptr(&self.adj,&self.stream);let layout=packed.layout;
        let seed=ptr(&d_seed,&self.stream)as *const RichSpike;let off=ptr(&offsets,&self.stream)as *const u32;
        let rel=ptr(&relaxed,&self.stream)as *mut ContactShellTile;let per=ptr(&perturbed,&self.stream)as *mut ContactShellTile;
        let pos=ptr(&positions,&self.stream)as *mut f32;let com_p=ptr(&com,&self.stream)as *mut c_void;
        let mass_p=ptr(&mass,&self.stream)as *mut c_void;let correction_p=ptr(&correction,&self.stream)as *mut c_void;
        let result=(||->Result<(),String>{
            let graph=capture_pure_template(self.stream.clone(),|tag|{
                native(so3_project::ffi::prism_so3_project_seed_run(seed,off,clusters,basis,rel,end,self.owner.source_stream_id(),raw.cast(),layout),"native relaxed SO3")?;tag.record_node("relaxed_so3")?;
                native(so3_project::ffi::prism_apply_gradient_gasp_positions_seed_clock_launch(seed,pos,self.native.forces as *const f32,self.native.masses as *const f32,adj as *const c_void,com_p,mass_p,self.native.protocol_current_step as *const u32,count,self.native.n_atoms as u32,raw.cast(),layout),"native GASP")?;tag.record_node("gasp")?;
                native(so3_project::ffi::prism_momentum_guard_check_launch(com_p,mass_p,correction_p,adj as *mut c_void,raw.cast()),"native GASP COM guard")?;
                native(so3_project::ffi::prism_apply_com_positions_correction_launch(pos.cast(),correction_p,count,raw.cast()),"native GASP COM correction")?;
                native(so3_project::ffi::prism_so3_project_positions_seed_run(seed,pos,off,clusters,basis,per,end,self.owner.source_stream_id(),raw.cast(),layout),"native perturbed SO3")?;tag.record_node("perturbed_so3")?;
                native(interferometric_adjudicator::ffi::prism_interferometric_adjudicator_step(adj as *mut _,clusters,raw.cast()),"native NodeC")?;tag.record_node("node_c")?;
                native(prism_dynamic_t7_launch(adj as *const c_void,ptr(&self.calibration,&self.stream)as *mut c_void,ptr(&self.compat_index,&self.stream)as *mut c_void,ptr(&self.compat_stats,&self.stream)as *mut c_void,raw.cast()),"native DynamicT7")?;tag.record_node("dynamic_t7")?;
                self.owner.publish(&self.stream,adj as *const _).map_err(|e|anyhow::anyhow!("native ASC publication rc={e}"))?;tag.record_node("asc_pending_publication")?;Ok(())
            }).map_err(|e|e.to_string())?.instantiate().map_err(|e|e.to_string())?;
            graph.launch_on_stream(raw).map_err(|e|e.to_string())?;
            self.stream.synchronize().map_err(|e|e.to_string())?;Ok(())
        })();
        // Always attempt to retain actual bytes, including failed publications.
        let mut errors=Vec::new();if let Err(e)=result{errors.push(e);}
        let adjudicator=download_or_record(&self.stream,&self.adj,&mut errors);
        let relaxed_tiles=download_or_record(&self.stream,&relaxed,&mut errors);
        let perturbed_tiles=download_or_record(&self.stream,&perturbed,&mut errors);
        let calibration=download_or_record(&self.stream,&self.calibration,&mut errors).into_iter().flat_map(u64::to_ne_bytes).collect();
        let after=match self.owner.snapshot(){Ok(x)=>Some(x),Err(e)=>{errors.push(format!("ASC postdecision snapshot rc={e}"));None}};
        if let Some(a)=after.as_ref(){if a.published_epoch!=before.completed_epoch||a.completed_epoch!=before.completed_epoch||a.pending.producer_epoch!=before.completed_epoch
            ||a.generation!=before.generation||a.topology!=before.topology||a.source!=before.source||a.decisions_produced!=before.decisions_produced+1{
            errors.push("native ASC pending publication identity failed".into());}}
        if adjudicator.len()==256 {let code=u32::from_ne_bytes(adjudicator[52..56].try_into().unwrap());let div=f32::from_ne_bytes(adjudicator[48..52].try_into().unwrap());
            if code>=2||!div.is_finite(){errors.push(format!("native NodeC refused decision code={code} divergence={div}"));}}
        let accepted=errors.is_empty();let error=(!accepted).then(||errors.join("; "));
        if accepted{self.last_boundary=end;}else{self.poisoned=true;}
        let index=json!({"input":self.seed_manifest(seeds,begin,end),"accepted":accepted,"error":error,
            "before":format!("{before:?}"),"after":format!("{after:?}"),"calibration_bytes":32,"physical_graph_launches":0,"boundary_graph_launches":1,
            "scope":"native captured observation-to-pending-ASC producer; next physical begin consumes pending control; no trajectory acceptance inferred"});
        Ok(BiprismAscDecisionReadback{accepted,error,index,adjudicator,relaxed_tiles,perturbed_tiles,calibration,asc_snapshot:after})
    }
    /// An empty accepted event interval does not fabricate a NodeC sample.
    pub fn acknowledge_empty(&mut self,begin:u32,end:u32)->Result<Value,String>{
        if self.poisoned||begin!=self.last_boundary||end<=begin{return Err("empty ASC boundary stale or poisoned".into());}
        self.last_boundary=end;Ok(json!({"published":false,"native_event_count":0,"begin":begin,"end":end,"scope":"no observation; prior native ASC control remains held"}))
    }
}
impl Drop for BiprismAscDecisionOwner {fn drop(&mut self){let _=self.stream.synchronize();}}
fn ptr<T:DeviceRepr>(s:&CudaSlice<T>,stream:&CudaStream)->u64{s.device_ptr(stream).0}
fn check(rc:i32,stage:&str)->Result<(),String>{if rc==0{Ok(())}else{Err(format!("{stage} rc={rc}"))}}
fn native(rc:i32,stage:&str)->anyhow::Result<()>{check(rc,stage).map_err(anyhow::Error::msg)}
fn download_or_record<T:DeviceRepr>(stream:&Arc<CudaStream>,s:&CudaSlice<T>,errors:&mut Vec<String>)->Vec<T>{
    match stream.clone_dtoh(s).and_then(|v|stream.synchronize().map(|_|v)){Ok(x)=>x,Err(e)=>{errors.push(e.to_string());Vec::new()}}
}
unsafe fn read_u32(stream:&CudaStream,address:u64)->Result<u32,String>{let mut v=0u32;let rc=sys::cuMemcpyDtoHAsync_v2((&mut v as *mut u32).cast(),address,4,stream.cu_stream());
    if rc!=sys::CUresult::CUDA_SUCCESS{return Err(format!("native clock read {rc:?}"));}stream.synchronize().map_err(|e|e.to_string())?;Ok(v)}
