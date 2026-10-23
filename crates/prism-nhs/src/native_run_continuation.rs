//! Process-level native restart admission. A run lease protects output-prefix
//! recovery for the entire process, independently of short checkpoint commits.
use anyhow::{ensure,Context,Result};
use serde::{Serialize,Deserialize};
use std::{collections::{BTreeMap,BTreeSet},fs::{File,OpenOptions},path::{Path,PathBuf},sync::Arc,time::Duration};
use std::os::{fd::AsRawFd,unix::fs::OpenOptionsExt};
use crate::scientific_checkpoint::{CheckpointIdentity,CheckpointStore,ClosureContract,FieldContract,Head,RestoredCheckpoint,capture_host,restore_host};
use crate::checkpoint_epoch::CheckpointEpoch;
/// The host control window remains at its original start until its normal full
/// feedback/drain finishes. Physical progress inside that window is separate.
#[derive(Clone,Debug,serde::Serialize,serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OriginalDispatchCursorV1 {
    pub schema_version:u32,pub original_chunk_index:i32,
    pub window_start_step:u32,pub window_end_step:u32,pub completed_native_step:u32,
    pub original_body_width:u32,pub next_original_body_slot:u32,
    pub chunk_prelude_applied:bool,
}
impl OriginalDispatchCursorV1 {
    pub fn validate(&self,host_start:i32,native_step:u64,next_chunk:i32,width:u32)->Result<()> {
        anyhow::ensure!(self.schema_version==1 && self.original_body_width==width && width>0
            && self.original_chunk_index==next_chunk && self.window_start_step==u32::try_from(host_start)?
            && self.completed_native_step as u64==native_step && self.window_start_step<self.completed_native_step
            && self.completed_native_step<self.window_end_step && self.chunk_prelude_applied
            && self.next_original_body_slot==(self.completed_native_step-self.window_start_step)%width,
            "original native dispatch/host-window/body-slot continuation differs");Ok(())
    }
}

/// Exact saved startup/control chronology for the opt-in original May
/// profile. It never infers physical time from the algorithmic control tick.
#[derive(Clone,Debug,serde::Serialize,serde::Deserialize,PartialEq,Eq)]
#[serde(deny_unknown_fields)]
pub struct MayStartupContinuationV1 {
    pub schema_version:u32,pub profile_id:String,pub stream_id:usize,
    pub executed_physical_steps:u64,pub common_next_tick:i32,
    pub algorithmic_next_control_tick:i32,pub captured_body_width:u32,
    pub logical_feedback_steps:u32,pub controller_owner_ids:Vec<usize>,
    pub legacy_shared_tail:bool,
}
pub const MAY_ORIGINAL_PROFILE_ID:&str="historical_may_literal_adaptive_original_endpoints_warmup6_v1";
pub const MAY_ORIGINAL_HORIZONS:[i32;8]=[35000,35000,40000,40000,37000,37000,34000,34000];
impl MayStartupContinuationV1 {
    pub fn at(stream_id:usize,next_tick:i32,executed:u64,width:u32)->Result<Self>{
        ensure!(stream_id<8 && (0..=80).contains(&next_tick) && matches!(executed,0|6)
            && (next_tick==0 || executed==6) && width==1,"May startup/control continuation is not an admitted completed boundary");
        Ok(Self{schema_version:1,profile_id:MAY_ORIGINAL_PROFILE_ID.into(),stream_id,
            executed_physical_steps:executed,common_next_tick:next_tick,
            algorithmic_next_control_tick:next_tick.checked_mul(500).context("May control tick overflow")?,
            captured_body_width:width,logical_feedback_steps:500,controller_owner_ids:(0..8).collect(),legacy_shared_tail:true})
    }
    pub fn validate(&self,stream_id:usize,next_tick:i32,width:u32)->Result<()> {
        ensure!(self==&Self::at(stream_id,next_tick,self.executed_physical_steps,width)?,"May startup/control policy differs from exact saved owner");Ok(())
    }
}
/// Only the ORIGINAL direct six-body startup group owns this cursor. It is
/// never a captured body-width selector or a scheduled feedback-window cursor.
#[derive(Clone,Debug,serde::Serialize,serde::Deserialize,PartialEq,Eq)]
#[serde(deny_unknown_fields)]
pub struct MayStartupGroupCursorV1 {
    pub schema_version:u32,pub original_group_width:u32,pub next_inner_slot:u32,
    pub pending_original_group_tail:bool,pub run_accounting_pending:bool,
    pub start_temperature_bits:u32,pub last_temperature_bits:u32,
    pub last_uv_burst_active:bool,pub last_uv_wavelength_bits:u32,
    pub accumulated_owner_run_ns:u64,
}
impl MayStartupGroupCursorV1 {
    pub fn fresh(temperature:f32)->Result<Self>{
        ensure!(temperature.is_finite(),"startup initial temperature is not finite");
        Ok(Self{schema_version:1,original_group_width:6,next_inner_slot:0,
            pending_original_group_tail:true,run_accounting_pending:true,
            start_temperature_bits:temperature.to_bits(),last_temperature_bits:temperature.to_bits(),
            last_uv_burst_active:false,last_uv_wavelength_bits:0f32.to_bits(),accumulated_owner_run_ns:0})
    }
    pub fn validate_pending(&self)->Result<()> {
        ensure!(self.schema_version==1 && self.original_group_width==6 && self.next_inner_slot<6
            && self.pending_original_group_tail && self.run_accounting_pending
            && f32::from_bits(self.start_temperature_bits).is_finite()
            && f32::from_bits(self.last_temperature_bits).is_finite()
            && f32::from_bits(self.last_uv_wavelength_bits).is_finite(),
            "original startup cursor/group-tail/report state is incomplete");Ok(())
    }
}
/// Version2 explicitly admits completed direct-startup body boundaries1..5.
/// Engine/provider identity prevents importing a V1 wire from an older ELF.
#[derive(Clone,Debug,serde::Serialize,serde::Deserialize,PartialEq,Eq)]
#[serde(deny_unknown_fields)]
pub struct MayStartupContinuationV2 {
    pub schema_version:u32,pub profile_id:String,pub stream_id:usize,
    pub executed_physical_steps:u64,pub common_next_tick:i32,
    pub algorithmic_next_control_tick:i32,pub captured_body_width:u32,
    pub logical_feedback_steps:u32,pub controller_owner_ids:Vec<usize>,
    pub legacy_shared_tail:bool,pub pending_original_group:Option<MayStartupGroupCursorV1>,
}
impl MayStartupContinuationV2 {
    pub fn at(stream:usize,tick:i32,executed:u64,width:u32)->Result<Self>{Self::with_cursor(stream,tick,executed,width,None)}
    pub fn with_cursor(stream:usize,tick:i32,executed:u64,width:u32,cursor:Option<MayStartupGroupCursorV1>)->Result<Self>{
        ensure!(stream<8 && (0..=80).contains(&tick) && executed<=6 && width==1
            && (tick==0 || executed==6),"May startup/control V2 boundary is not admitted");
        if (1..6).contains(&executed){let c=cursor.as_ref().context("partial May startup lacks its pending group owner")?;
            c.validate_pending()?;ensure!(tick==0 && u64::from(c.next_inner_slot)==executed,"startup cursor/physical frontier differs");
        }else{ensure!(cursor.is_none(),"initial/completed startup retains a partial group cursor");}
        Ok(Self{schema_version:2,profile_id:MAY_ORIGINAL_PROFILE_ID.into(),stream_id:stream,
            executed_physical_steps:executed,common_next_tick:tick,
            algorithmic_next_control_tick:tick.checked_mul(500).context("May control tick overflow")?,captured_body_width:width,
            logical_feedback_steps:500,controller_owner_ids:(0..8).collect(),legacy_shared_tail:true,pending_original_group:cursor})
    }
    pub fn validate(&self,stream:usize,tick:i32,width:u32)->Result<()>{
        ensure!(self==&Self::with_cursor(stream,tick,self.executed_physical_steps,width,self.pending_original_group.clone())?,
            "May V2 startup policy or pending original tail differs");Ok(())
    }
}
pub const MAY_STARTUP_PREFIX_PHASE:&str="completed_original_may_startup_inner_body_before_pending_group_tail";

#[derive(Clone,Debug,serde::Serialize,serde::Deserialize,PartialEq,Eq)]
#[serde(deny_unknown_fields)]
pub struct OwnEndCompletionV1 {
    schema_version:u32,profile_id:String,stream_id:usize,
    owner_step_targets:Vec<i32>,collective_next_tick:i32,collective_tick_steps:i32,
    collective_tick_count:i32,physical_completed_steps:i32,physical_step_target:i32,
    physical_complete:bool,
    #[serde(default,skip_serializing_if="Option::is_none")]
    startup_warmup_physical_steps:Option<i32>,
    #[serde(default,skip_serializing_if="Option::is_none")]
    logical_scheduled_steps:Option<i32>,
}
pub const OWN_END_PROFILE_ID:&str="differential_initial_cold2000_fixed2_own_end_v1";
pub const OWN_END_HORIZONS:[i32;8]=[27000,27000,36000,36000,31000,31000,30000,30000];
pub const OWN_END_RESTART_PHASE:&str="completed_native_own_end_collective_tick_before_next_director";
impl OwnEndCompletionV1 {
    pub fn validate_join(&self,stream_id:u32,next_chunk:i32,physical:u64)->Result<()> {
        let historical=self.profile_id==MAY_ORIGINAL_PROFILE_ID;
        let targets=if historical{&MAY_ORIGINAL_HORIZONS}else{&OWN_END_HORIZONS};
        let startup=if historical{u64::try_from(self.startup_warmup_physical_steps.context("May completion lacks startup owner")?)?}else{0};
        let logical=if historical{self.logical_scheduled_steps.context("May completion lacks logical frontier")?}else{i32::try_from(physical)?};
        let expected=Self::at_with_startup(stream_id as usize,targets,next_chunk,logical,startup)?;
        ensure!(self==&expected && u64::try_from(expected.physical_completed_steps)?==physical,
            "own-end typed completion/header/common tick mismatch");Ok(())
    }
    pub fn at(stream_id:usize,targets:&[i32],next_tick:i32,logical:i32)->Result<Self> {
        Self::at_with_startup(stream_id,targets,next_tick,logical,0)
    }
    pub fn at_with_startup(stream_id:usize,targets:&[i32],next_tick:i32,logical:i32,startup:u64)->Result<Self> {
        let historical=targets==MAY_ORIGINAL_HORIZONS;
        ensure!((historical || targets==OWN_END_HORIZONS) && stream_id<targets.len(),"own-end continuation profile changed");
        let ticks=*targets.iter().max().context("own-end target vector absent")?/500;
        ensure!((0..=ticks).contains(&next_tick) && logical==(next_tick*500).min(targets[stream_id]),"own-end logical frontier/tick differs");
        if historical{ensure!(startup<=6 && (next_tick==0 || startup==6),"May own-end startup physical frontier differs");}else{ensure!(startup==0,"ordinary own-end acquired a May warmup");}
        let offset=i32::try_from(startup)?;let planned_offset=if historical{6}else{0};
        Ok(Self{schema_version:if historical{2}else{1},profile_id:if historical{MAY_ORIGINAL_PROFILE_ID}else{OWN_END_PROFILE_ID}.into(),
            stream_id,owner_step_targets:targets.iter().map(|v|v+planned_offset).collect(),
            collective_next_tick:next_tick,collective_tick_steps:500,collective_tick_count:ticks,
            physical_completed_steps:logical.checked_add(offset).context("own-end actual startup step overflow")?,
            physical_step_target:targets[stream_id]+planned_offset,physical_complete:logical==targets[stream_id] && offset==planned_offset,
            startup_warmup_physical_steps:historical.then_some(offset),logical_scheduled_steps:historical.then_some(logical)})
    }
}
/// Complete all-owner prefix of one ORIGINAL feedback window. A retired
/// owner participates with its real unchanged physical frontier, never fake
/// clock/event rows. Controller feedback for this window has not yet run.
#[derive(Clone,Debug,serde::Serialize,serde::Deserialize,PartialEq,Eq)]
#[serde(deny_unknown_fields)]
pub struct OwnEndPrefixContinuationV1 {
    pub schema_version:u32,pub profile_id:String,pub stream_id:usize,
    pub logical_owner_targets:Vec<i32>,pub common_chunk_index:i32,
    pub collective_logical_step:u64,pub host_window_start:i32,
    pub completed_native_step:u64,pub startup_physical_steps:u64,
    pub original_body_width:u32,pub original_body_slot:u32,
    pub chunk_prelude_applied:bool,pub original_feedback_pending:bool,
}
pub const OWN_END_PREFIX_PHASE:&str="own_end_original_dispatch_prefix_before_original_window_feedback";
pub fn own_end_phase(value:&str)->bool {value==OWN_END_RESTART_PHASE || value==OWN_END_PREFIX_PHASE || value==MAY_STARTUP_PREFIX_PHASE}
impl OwnEndPrefixContinuationV1 {
    pub fn at(stream_id:usize,targets:&[i32],chunk:i32,logical:u64,startup:u64,width:u32)->Result<Self>{
        let historical=targets==MAY_ORIGINAL_HORIZONS;
        ensure!((historical || targets==OWN_END_HORIZONS) && stream_id<8 && chunk>=0 && width>0,
            "unknown own-end original prefix population/body");
        let begin=u64::try_from(chunk)?.checked_mul(500).context("prefix collective window overflow")?;
        let end=begin.checked_add(500).context("prefix collective window overflow")?;
        ensure!(begin<logical && logical<end && logical<u64::try_from(*targets.iter().max().context("prefix target population absent")?)?,
            "prefix is not strictly inside the original collective feedback window");
        ensure!(if historical{startup==6 && width==1}else{startup==0},"prefix cannot enter an incomplete May warmup or change body shape");
        let target=u64::try_from(targets[stream_id])?;let host_start=begin.min(target);
        let completed=logical.min(target).checked_add(startup).context("prefix native step overflow")?;
        Ok(Self{schema_version:1,profile_id:if historical{MAY_ORIGINAL_PROFILE_ID}else{OWN_END_PROFILE_ID}.into(),stream_id,
            logical_owner_targets:targets.to_vec(),common_chunk_index:chunk,collective_logical_step:logical,
            host_window_start:i32::try_from(host_start)?,completed_native_step:completed,startup_physical_steps:startup,
            original_body_width:width,original_body_slot:if begin<target{u32::try_from((logical-begin)%u64::from(width))?}else{0},
            chunk_prelude_applied:true,original_feedback_pending:true})
    }
    pub fn validate_worker(&self,worker:&NativeWorkerContinuationV1,physical:u64)->Result<()> {
        ensure!(self==&Self::at(worker.stream_id as usize,&self.logical_owner_targets,worker.next_chunk,
            self.collective_logical_step,self.startup_physical_steps,self.original_body_width)?
            && worker.get::<i32>("host:steps_run")?==self.host_window_start && physical==self.completed_native_step,
            "own-end original prefix/header/logical pre-feedback state differs");
        let previous:OwnEndCompletionV1=worker.get("host:own_end_completion")?;
        previous.validate_join(worker.stream_id,worker.next_chunk,u64::try_from(self.host_window_start)?+self.startup_physical_steps)?;
        let cursor:Option<OriginalDispatchCursorV1>=worker.get_optional("host:original_dispatch_cursor")?;
        let active=self.host_window_start<self.logical_owner_targets[self.stream_id];
        ensure!(cursor.is_some()==active,"own-end prefix active/retired cursor population differs");
        if let Some(cursor)=cursor {
            cursor.validate(i32::try_from(u64::try_from(self.host_window_start)?+self.startup_physical_steps)?,physical,worker.next_chunk,self.original_body_width)?;
            let end=u64::try_from((worker.next_chunk+1).checked_mul(500).context("prefix feedback endpoint overflow")?.min(self.logical_owner_targets[self.stream_id]))?+self.startup_physical_steps;
            ensure!(u64::from(cursor.window_end_step)==end && cursor.next_original_body_slot==self.original_body_slot,
                "own-end prefix does not retain the original endpoint/body slot");
        }Ok(())
    }
}
/// Join saved logical control progress to actual native chronology; presence of
/// May state is fixed by exact CLI/input/module identity and complete population.
pub fn worker_completed_native_step(worker:&NativeWorkerContinuationV1)->Result<u64>{
    if let Some(prefix)=worker.get_optional::<OwnEndPrefixContinuationV1>("host:own_end_prefix")? {
        prefix.validate_worker(worker,prefix.completed_native_step)?;return Ok(prefix.completed_native_step);
    }
    if let Some(cursor)=worker.get_optional::<OriginalDispatchCursorV1>("host:original_dispatch_cursor")? {return Ok(cursor.completed_native_step as u64);}
    let logical=u64::try_from(worker.get::<i32>("host:steps_run")?)?;
    let startup:Option<MayStartupContinuationV2>=worker.get_optional("host:may_startup")?;
    let offset=if let Some(state)=startup{state.validate(worker.stream_id as usize,worker.next_chunk,1)?;state.executed_physical_steps}else{0};
    logical.checked_add(offset).context("native logical/startup frontier overflow")
}

/// Source-checked publication law, distinct from HEAD generation and physical
/// owner steps. Constructed by decoding the actual immutable host closures.
#[derive(Clone,Debug)]
pub struct OwnEndCollectivePublicationV1 {
    pub common_next_tick:i32,pub checkpoint_ordinal:u64,pub coupling_epoch:u64,
    pub raw_head:Option<Head>,pub profile_id:String,pub physical_owner_targets:Vec<u64>,pub startup_physical_steps:u64,
    pub collective_logical_step:u64,pub prefix_pending:bool,
}
/// Read-only admission projection of the complete saved raw owner. Other raw
/// fields remain in the original complete payload; this type never rewrites it.
#[derive(Deserialize)]
struct RawCollectivePublicationV2 {
    schema_version:u32,identity:CheckpointIdentity,head:Option<Head>,
    boundaries:Vec<RawPhysicalPublicationV1>,common_next_chunk:i32,checkpoint_ordinal:u64,
}
#[derive(Deserialize)]
struct RawPhysicalPublicationV1 {stream_id:u32,completed_step:u64,physical_time_ps_bits:u64}
pub fn validate_own_end_publication(identity:&CheckpointIdentity,boundaries:&[crate::scientific_checkpoint::StreamBoundary],
    mut read_worker:impl FnMut(u32)->Result<NativeWorkerContinuationV1>,
    read_shared:impl FnOnce()->Result<NativeSharedContinuationV1>)->Result<OwnEndCollectivePublicationV1> {
    ensure!(identity.stream_count==8 && boundaries.len()==8 && boundaries.iter().all(|b|own_end_phase(&b.next_integrator_phase) && b.next_integrator_phase==boundaries[0].next_integrator_phase),
        "own-end publication requires its complete exact-phase eight-worker population");
    let mut common=None;let mut population:Option<(String,Vec<u64>,u64)>=None;let mut collective_position=None;
    for stream in 0..identity.stream_count {
        // Drop each complete worker before decoding the next; no second full
        // generation of host closures is retained at publication.
        let worker=read_worker(stream)?;let boundary=&boundaries[stream as usize];
        ensure!(worker.schema_version==1 && worker.stream_id==stream && boundary.stream_id==stream,"own-end publication worker identity differs");
        let completion:OwnEndCompletionV1=worker.get("host:own_end_completion")?;
        let prefix:Option<OwnEndPrefixContinuationV1>=worker.get_optional("host:own_end_prefix")?;
        let prefix_pending=boundary.next_integrator_phase==OWN_END_PREFIX_PHASE;
        ensure!(prefix.is_some()==prefix_pending,"prefix phase/owner population differs");
        let position=if let Some(prefix)=&prefix {
            prefix.validate_worker(&worker,boundary.completed_step)?;prefix.collective_logical_step
        }else{completion.validate_join(stream,worker.next_chunk,boundary.completed_step)?;
            ensure!(worker.get_optional::<OriginalDispatchCursorV1>("host:original_dispatch_cursor")?.is_none(),"completed own-end anchor retains an original partial cursor");
            u64::try_from(worker.next_chunk)?.checked_mul(500).context("collective position overflow")?};
        ensure!(collective_position.is_none_or(|prior|prior==(position,prefix_pending)),"own-end prefix collective position differs across workers");
        collective_position=Some((position,prefix_pending));
        ensure!(worker_completed_native_step(&worker)?==boundary.completed_step
            && worker.get::<u64>("host:v2_frames_enqueued")?==boundary.committed_output_sequence,
            "own-end publication physical/cursor/writer closure differs");
        let startup:Option<MayStartupContinuationV2>=worker.get_optional("host:may_startup")?;
        if let Some(state)=startup.as_ref(){state.validate(stream as usize,worker.next_chunk,1)?;
            ensure!((boundary.next_integrator_phase==MAY_STARTUP_PREFIX_PHASE)==(state.pending_original_group.is_some()),
                "startup phase does not match the actual pending original group");}
        let startup_steps=startup.as_ref().map_or(0,|s|s.executed_physical_steps);
        ensure!(startup.is_some()==(completion.profile_id==MAY_ORIGINAL_PROFILE_ID)
            && completion.startup_warmup_physical_steps.map_or(startup_steps==0,|v|v as u64==startup_steps),"own-end startup/completion policy differs");
        let this_population=(completion.profile_id.clone(),completion.owner_step_targets.iter().copied().map(u64::try_from).collect::<std::result::Result<Vec<_>,_>>()?,startup_steps);
        ensure!(population.as_ref().is_none_or(|prior|prior==&this_population),"own-end profile/physical targets/startup transaction differs across workers");population=Some(this_population);
        let this=(worker.next_chunk,worker.checkpoint_ordinal,boundary.coupling_epoch);
        ensure!(common.is_none_or(|prior|prior==this),"own-end worker tick/checkpoint ordinal/barrier generations differ");common=Some(this);
    }
    let (tick,ordinal,epoch)=common.context("empty own-end publication population")?;
    let shared=read_shared()?;ensure!(shared.schema_version==1,"own-end shared schema differs");
    let joined:(i32,u64)=shared.get("native_collective_boundary")?;
    ensure!(joined==(tick,epoch),"own-end shared collective/barrier boundary differs");
    let raw:Option<RawCollectivePublicationV2>=shared.get_optional("raw_chronicle")?;
    let raw=raw.context("own-end publication lacks complete raw owner")?;
    ensure!(raw.schema_version==2 && raw.identity==*identity && raw.common_next_chunk==tick && raw.checkpoint_ordinal==ordinal
        && raw.boundaries.len()==boundaries.len() && raw.boundaries.iter().zip(boundaries).all(|(r,b)|
            r.stream_id==b.stream_id && r.completed_step==b.completed_step && r.physical_time_ps_bits==b.physical_time_ps_bits),
        "own-end raw/scientific/worker ordinal or physical frontier differs");
    let (profile_id,physical_owner_targets,startup_physical_steps)=population.context("own-end population absent")?;
    let (collective_logical_step,prefix_pending)=collective_position.context("own-end collective position absent")?;
    Ok(OwnEndCollectivePublicationV1{common_next_tick:tick,checkpoint_ordinal:ordinal,coupling_epoch:epoch,raw_head:raw.head,profile_id,physical_owner_targets,startup_physical_steps,collective_logical_step,prefix_pending})
}

/// Canonical absence has no owned body allocation. Decode that exact nine-byte
/// value as Option<()> first, preserving magic/tag/EOF validation; do not charge
/// an absent large resource owner's inline Some layout against its tiny wire.
/// Present values retain the original exact typed bounded decoder and budget.
fn decode_optional_native_host<T:serde::de::DeserializeOwned>(mut bytes:&[u8])->Result<Option<T>> {
    let wire_bytes=bytes.len() as u64;
    let budget=wire_bytes.checked_mul(64).context("native optional host allocation budget overflow")?;
    if bytes.len()==9 && bytes[8]==12 {
        let absent:Option<()>=crate::checkpoint_codec::from_reader_with_allocation_budget(
            &mut bytes,wire_bytes,budget,budget)?;
        ensure!(absent.is_none(),"native canonical optional absence differs");return Ok(None);
    }
    Ok(crate::checkpoint_codec::from_reader_with_allocation_budget(&mut bytes,wire_bytes,budget,budget)?)
}
#[derive(Serialize,Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeWorkerContinuationV1 {
    pub schema_version:u32,pub stream_id:u32,pub next_chunk:i32,pub checkpoint_ordinal:u64,
    #[serde(with="crate::checkpoint_codec::byte_map")]
    pub states:BTreeMap<String,Vec<u8>>,
}
impl NativeWorkerContinuationV1 {
    pub fn get_optional<T:serde::de::DeserializeOwned>(&self,name:&str)->Result<Option<T>> {
        let bytes=self.states.get(name).with_context(||format!("missing native worker optional owner {name}"))?;
        decode_optional_native_host(bytes).with_context(||format!("native worker optional owner {name}, type {}, wire_bytes {}",std::any::type_name::<T>(),bytes.len()))
    }

    pub fn get<T:serde::de::DeserializeOwned>(&self,name:&str)->Result<T>{
        let bytes=self.states.get(name).with_context(||format!("missing native worker owner {name}"))?;
        let budget=(bytes.len() as u64).checked_mul(64).context("native host allocation budget overflow")?;
        crate::checkpoint_codec::from_reader_with_allocation_budget(&mut bytes.as_slice(),bytes.len() as u64,budget,budget)
            .with_context(||format!("native worker owner {name}, type {}, wire_bytes {}, planned_budget {budget}",std::any::type_name::<T>(),bytes.len()))
    }
    pub fn validate_population(&self,expected:&[&str])->Result<()> {
        ensure!(self.schema_version==1 && self.next_chunk>=0,"unsupported native worker checkpoint");
        let actual=self.states.keys().map(String::as_str).collect::<BTreeSet<_>>();
        ensure!(actual==expected.iter().copied().collect::<BTreeSet<_>>(),"native worker owner population changed: missing={:?} extra={:?}",
            expected.iter().copied().filter(|v|!actual.contains(v)).collect::<Vec<_>>(),
            actual.iter().copied().filter(|v|!expected.contains(v)).collect::<Vec<_>>());Ok(())
    }
}
#[derive(Serialize,Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeSharedContinuationV1 {
    pub schema_version:u32,
    #[serde(with="crate::checkpoint_codec::byte_map")]
    pub states:BTreeMap<String,Vec<u8>>,
}
impl NativeSharedContinuationV1 {
    pub fn get_optional<T:serde::de::DeserializeOwned>(&self,name:&str)->Result<Option<T>> {
        let bytes=self.states.get(name).with_context(||format!("missing native shared optional owner {name}"))?;
        decode_optional_native_host(bytes).with_context(||format!("native shared optional owner {name}, type {}, wire_bytes {}",std::any::type_name::<T>(),bytes.len()))
    }

    pub fn validate_population(&self,expected:&[&str])->Result<()> {
        ensure!(self.schema_version==1,"unsupported native shared checkpoint");
        let actual=self.states.keys().map(String::as_str).collect::<BTreeSet<_>>();
        ensure!(actual==expected.iter().copied().collect::<BTreeSet<_>>(),"native shared owner population changed");Ok(())
    }
    pub fn get<T:serde::de::DeserializeOwned>(&self,name:&str)->Result<T>{
        let bytes=self.states.get(name).with_context(||format!("missing native shared owner {name}"))?;
        let budget=(bytes.len() as u64).checked_mul(64).context("native shared allocation budget overflow")?;
        crate::checkpoint_codec::from_reader_with_allocation_budget(&mut bytes.as_slice(),bytes.len() as u64,budget,budget)
            .with_context(||format!("native shared owner {name}, type {}, wire_bytes {}, planned_budget {budget}",std::any::type_name::<T>(),bytes.len()))
    }
}
pub fn put<T:Serialize+?Sized>(states:&mut BTreeMap<String,Vec<u8>>,name:&str,value:&T)->Result<()> {
    ensure!(!states.contains_key(name),"duplicate native continuation owner {name}");
    states.insert(name.into(),crate::checkpoint_codec::to_bytes(value)?);Ok(())
}
/// Count before allocating a host owner and enforce the cumulative live map
/// budget. The old unbounded compatibility entry point remains unchanged.
pub fn put_bounded<T:Serialize+?Sized>(states:&mut BTreeMap<String,Vec<u8>>,name:&str,value:&T,maximum:u64)->Result<()> {
    ensure!(!states.contains_key(name),"duplicate native continuation owner {name}");
    let size=crate::checkpoint_codec::encoded_size(value,maximum)?;
    let retained=states.iter().try_fold(0u64,|total,(key,value)|{
        total.checked_add(value.len() as u64).and_then(|v|v.checked_add(key.len() as u64)).context("host continuation map byte overflow")
    })?;
    ensure!(retained.checked_add(size).and_then(|v|v.checked_add(name.len() as u64)).is_some_and(|n|n<=maximum),
        "host continuation owner map exceeds preallocation budget");
    states.insert(name.into(),crate::checkpoint_codec::to_bytes_bounded(value,maximum)?);Ok(())
}
impl NativeContinuationRun {
    pub fn worker_field_bounded(value:&NativeWorkerContinuationV1,maximum:u64)->Result<crate::scientific_checkpoint::CapturedField>{
        crate::scientific_checkpoint::capture_host_bounded(&format!("stream_{}",value.stream_id),"native_owner_closure",value,maximum)
    }
    pub fn shared_field_bounded(value:&NativeSharedContinuationV1,maximum:u64)->Result<crate::scientific_checkpoint::CapturedField>{
        crate::scientific_checkpoint::capture_host_bounded("shared","native_owner_closure",value,maximum)
    }
}
pub struct NativeContinuationRun {
    pub streaming:bool,
    planned_stop_requested:std::sync::atomic::AtomicBool,
    pub generation:Option<crate::scientific_checkpoint::restart_transaction::ValidatedGeneration>,
    pub maximum_field_bytes:u64,
    pub maximum_generation_bytes:u64,
    // Operational fresh resource admission, recomputed on each restore; not
    // scientific state and never an inherited capacity credit.
    raw_retention_bound:u64,
    admitted_collective_chunk_count:u64,
    store:CheckpointStore,
    _reservation:std::sync::Mutex<Option<crate::scientific_checkpoint::restart_transaction::ReservationLease>>,
    pub identity:CheckpointIdentity,pub epoch:Arc<CheckpointEpoch>,
    pub saved_workers:Vec<NativeWorkerContinuationV1>,pub saved_shared:Option<NativeSharedContinuationV1>,
    pub head:Option<Head>,pub root:PathBuf,pub restore_barrier:crate::rescue_controller::CoupledWindowBarrier,_lease:RunOutputClaim,
}
impl NativeContinuationRun {
    pub fn open(root:&Path,output_root:&Path,max_bytes:u64,mut current:CheckpointIdentity,resume:bool)->Result<Arc<Self>> {
        let store=CheckpointStore::create(root,max_bytes)?;
        let lease=reserve_output_root(output_root)?;
        ensure!(lease.claimed.compare_exchange(false,true,std::sync::atomic::Ordering::AcqRel,std::sync::atomic::Ordering::Acquire).is_ok(),
            "another native run in this process owns the experiment output");
        let lease=RunOutputClaim(lease);
        let closure=Self::closure(current.stream_count);
        let saved=store.identity_at_head()?;
        ensure!(resume==saved.is_some(),if resume{"no durable native generation to resume"}else{"existing native generation requires explicit resume"});
        let (head,restored)=if let Some((head,prior))=saved {
            current.run_id=prior.run_id;
            let restored=store.restore(&head,&current,&closure)?;(Some(head),Some(restored))
        }else{(None,None)};
        let (saved_workers,saved_shared)=if let Some(restored)=restored {Self::decode(&current,restored)?}else{(Vec::new(),None)};
        let ordinal=saved_workers.first().map(|v|v.checkpoint_ordinal.checked_add(1).context("checkpoint ordinal overflow")).transpose()?.unwrap_or(0);
        let epoch=CheckpointEpoch::new(store,current.clone(),closure,head.clone(),ordinal,Duration::from_secs(900))?;
        let restore_barrier=crate::rescue_controller::CoupledWindowBarrier::new(current.stream_count as usize);
        Ok(Arc::new(Self{planned_stop_requested:std::sync::atomic::AtomicBool::new(false),identity:current,epoch,saved_workers,saved_shared,head,root:root.into(),restore_barrier,_lease:lease,
            streaming:false,raw_retention_bound:0,admitted_collective_chunk_count:0,generation:None,maximum_field_bytes:max_bytes,maximum_generation_bytes:max_bytes,store:CheckpointStore::create(root,max_bytes)?,_reservation:std::sync::Mutex::new(None)}))
    }
    /// Same native run/lease/epoch owner, with direct field CAS rather than a
    /// nested engine blob. Fresh runtime extents must register before replay.
    pub fn open_streaming(root:&Path,output_root:&Path,max_bytes:u64,maximum_field_bytes:u64,
        retained_bytes:u64,free_floor:u64,campaign_capacity:crate::scientific_checkpoint::restart_transaction::ScientificCampaignCapacityV1,
        mut current:CheckpointIdentity,resume:bool)->Result<Arc<Self>> {
        ensure!(maximum_field_bytes>0 && maximum_field_bytes<=max_bytes,"invalid scientific field memory limit");
        let store=CheckpointStore::create(root,max_bytes)?;
        let lease=reserve_output_root(output_root)?;
        ensure!(lease.claimed.compare_exchange(false,true,std::sync::atomic::Ordering::AcqRel,std::sync::atomic::Ordering::Acquire).is_ok(),"another native run owns output namespace");
        let lease=RunOutputClaim(lease);let base=Self::closure(current.stream_count);
        let saved=store.identity_at_head()?;ensure!(resume==saved.is_some(),"scientific resume/fresh mode differs from actual durable HEAD");
        let (head,generation,saved_workers,saved_shared)=if let Some((head,prior))=saved {
            current.run_id=prior.run_id;
            let index=store.validate_generation(&head,&current)?;
            let admission=store.admit_generation_host_capacity(&index,free_floor)?;
            publish_fresh_admission_receipt(output_root,"scientific_restart_host_capacity",&admission,Some(&head))?;
            for required in &base.fields {ensure!(index.closure().fields.iter().any(|v|v==required),"scientific generation lacks exact native host owner");}
            for field in &index.closure().fields {
                if base.fields.iter().any(|v|v==field){continue;}
                ensure!((0..current.stream_count).any(|i|field.owner==format!("stream_{i}/engine") || field.owner.starts_with(&format!("stream_{i}/engine/"))),"scientific generation contains unadmitted native state owner");
            }
            let (workers,shared)=Self::decode_with(&current,index.boundaries(),
                |i|store.read_generation_host(&index,&format!("stream_{i}"),"native_owner_closure",maximum_field_bytes),
                ||store.read_generation_host(&index,"shared","native_owner_closure",maximum_field_bytes))?;
            for worker in &workers {
                let step:i32=worker.get("host:steps_run")?;
                let boundary=index.boundaries().iter().find(|b|b.stream_id==worker.stream_id).context("worker boundary missing")?;
                let cursor:Option<OriginalDispatchCursorV1>=worker.get_optional("host:original_dispatch_cursor")?;
                if own_end_phase(&boundary.next_integrator_phase) {
                    let prefix:Option<OwnEndPrefixContinuationV1>=worker.get_optional("host:own_end_prefix")?;
                    if let Some(prefix)=prefix {prefix.validate_worker(worker,boundary.completed_step)?;}
                    else{worker.get::<OwnEndCompletionV1>("host:own_end_completion")?.validate_join(worker.stream_id,worker.next_chunk,boundary.completed_step)?;}
                    ensure!(worker_completed_native_step(worker)?==boundary.completed_step,"own-end streamed native frontier differs");
                }else if let Some(cursor)=cursor {
                    cursor.validate(step,boundary.completed_step,worker.next_chunk,cursor.original_body_width)?;
                    ensure!(boundary.next_integrator_phase=="ordinary_native_dispatch_prefix_before_original_window_feedback","partial native cursor/phase mismatch");
                }else{
                    ensure!(worker_completed_native_step(worker)?==boundary.completed_step,"worker completed physical/logical/startup state differs from committed native boundary");
                    ensure!(boundary.next_integrator_phase=="completed_native_coupled_window_before_next_director" || boundary.next_integrator_phase==OWN_END_RESTART_PHASE,"scientific anchor is not at complete native scheduler boundary");
                }
                let enqueued:u64=worker.get("host:v2_frames_enqueued")?;
                ensure!(boundary.committed_output_sequence==enqueued,"native writer producer/frontier differs from scientific boundary");
                ensure!(boundary.coupling_parity<2,"scientific coupling parity is invalid");
            }
            (Some(head),Some(index),workers,shared)
        }else{(None,None,Vec::new(),None)};
        let ordinal=saved_workers.first().map(|v|v.checkpoint_ordinal.checked_add(1).context("checkpoint ordinal overflow")).transpose()?.unwrap_or(0);
        let expected=generation.as_ref().map(|v|v.closure().digest()).transpose()?;
        let capacity=store.admit_remaining_campaign_capacity(&current,u64::try_from(saved_workers.first().map_or(0,|w|w.next_chunk))?,
            head.is_some(),max_bytes,retained_bytes,&campaign_capacity)?;
        publish_fresh_admission_receipt(output_root,"scientific_restart_campaign_capacity",&capacity,head.as_ref())?;
        let reservation=store.reserve_retained_capacity(crate::scientific_checkpoint::restart_transaction::ReservationContract{
            schema_version:1,identity:current.clone(),maximum_retained_bytes:retained_bytes,remaining_free_floor_bytes:free_floor})?;
        let epoch=CheckpointEpoch::new_runtime_registered(CheckpointStore::create(root,max_bytes)?,current.clone(),base,
            head.clone(),ordinal,Duration::from_secs(900),expected)?;
        let restore_barrier=crate::rescue_controller::CoupledWindowBarrier::new(current.stream_count as usize);
        Ok(Arc::new(Self{planned_stop_requested:std::sync::atomic::AtomicBool::new(false),identity:current,epoch,saved_workers,saved_shared,head,root:root.into(),restore_barrier,
            _lease:lease,streaming:true,raw_retention_bound:campaign_capacity.raw_store_retention_bound,admitted_collective_chunk_count:campaign_capacity.collective_chunk_count,generation,maximum_field_bytes,maximum_generation_bytes:max_bytes,store,_reservation:std::sync::Mutex::new(Some(reservation))}))
    }
    pub fn handoff_unused_scientific_capacity_to_pathb(&self,terminal:&serde_json::Value)->Result<serde_json::Value> {
        ensure!(self.streaming,"MD capacity handoff requires streaming native store");
        let head=self.epoch.quiescent_committed_head()?;
        let slot=self._reservation.lock().map_err(|_|anyhow::anyhow!("capacity lease poisoned"))?;
        let lease=slot.as_ref().context("MD capacity lease absent/already retired")?;
        self.store.handoff_unused_scientific_capacity(lease,&head,&self.identity,self.raw_retention_bound,self.admitted_collective_chunk_count,terminal)
    }
    pub fn retire_quiescent_capacity(&self,terminal:&serde_json::Value)->Result<serde_json::Value> {
        ensure!(self.streaming,"retained capacity handoff requires streaming native store");
        let head=self.epoch.quiescent_committed_head()?;
        let mut slot=self._reservation.lock().map_err(|_|anyhow::anyhow!("capacity lease poisoned"))?;
        let lease=slot.take().context("capacity lease already retired or absent")?;
        self.store.retire_reserved_capacity(lease,&head,&self.identity,terminal)
    }
    pub fn retire_empty_failed_capacity(&self,terminal:&serde_json::Value)->Result<serde_json::Value> {
        ensure!(self.streaming,"failed capacity retirement requires streaming store");
        let mut slot=self._reservation.lock().map_err(|_|anyhow::anyhow!("failed capacity lease poisoned"))?;
        let lease=slot.take().context("failed capacity lease already retired or absent")?;
        self.store.retire_empty_failed_capacity(lease,&self.identity,terminal)
    }
    pub fn request_planned_scientific_stop(&self){self.planned_stop_requested.store(true,std::sync::atomic::Ordering::Release);}
    pub fn planned_scientific_stop_requested(&self)->bool{self.planned_stop_requested.load(std::sync::atomic::Ordering::Acquire)}
    /// Reuse the committed metadata owner. Payload bytes were hashed/fsynced
    /// by its transaction; this receipt does not claim restart equivalence.
    pub fn committed_scientific_stop_proof(&self,common_next_tick:i32)->Result<serde_json::Value> {
        ensure!(self.streaming && common_next_tick>=0,"scientific stop requires the streamed native closure");
        let head=self.epoch.committed_head()?.context("scientific stop lacks committed HEAD")?;
        let generation=self.store.validate_generation(&head,&self.identity)?;
        ensure!(generation.boundaries().len()==self.identity.stream_count as usize,"scientific stop worker population differs");
        for stream in 0..self.identity.stream_count {
            let owner=format!("stream_{stream}/engine");
            ensure!(generation.closure().fields.iter().filter(|field|field.owner==owner || field.owner.starts_with(&format!("{owner}/"))).count()==351,
                "scientific stop lacks complete338+13 engine field population");
        }
        Ok(serde_json::json!({"schema":"prism.native_scientific_stop_capsule.v1","status":"SCIENTIFIC_STOP_CAPSULE_COMMITTED",
            "scientific_head":head,"identity":self.identity,"common_next_tick":common_next_tick,
            "boundaries":generation.boundaries(),"registered_field_count":generation.closure().fields.len(),
            "scope":"original native scientific/controller/writer/raw-cursor closure at the declared all-worker scheduler phase; typed prefixes retain pending original feedback",
            "scheduler_phase":generation.boundaries().first().map(|b|b.next_integrator_phase.as_str()),
            "restart_equivalence_qualified":false,"arbitrary_step_restart_qualified":false}))
    }
    pub fn restored_native_step(&self,stream:usize)->Result<u64> {
        if let Some(index)=&self.generation {
            return Ok(index.boundaries().iter().find(|b|b.stream_id==stream as u32).context("restored native boundary missing")?.completed_step);
        }
        self.saved_workers.get(stream).map(worker_completed_native_step).unwrap_or(Ok(0))
    }
    pub fn read_engine_field(&self,owner:&str,name:&str)->Result<crate::scientific_checkpoint::CapturedField> {
        let index=self.generation.as_ref().context("scientific run has no restored generation")?;
        self.store.read_generation_field(index,owner,name,self.maximum_field_bytes)
    }
    pub fn generation_reader(&self)->Result<crate::scientific_checkpoint::restart_transaction::ScientificGenerationReader<'_>> {
        Ok(crate::scientific_checkpoint::restart_transaction::ScientificGenerationReader{store:&self.store,index:self.generation.as_ref().context("scientific generation missing")?,maximum_device_field_bytes:self.maximum_field_bytes,maximum_owned_fragment_bytes:self.maximum_field_bytes})
    }
    pub fn engine_contracts(&self,stream:usize)->Result<Vec<FieldContract>> {
        let index=self.generation.as_ref().context("scientific run has no restored generation")?;
        let prefix=format!("stream_{stream}/engine");
        Ok(index.closure().fields.iter().filter(|f|f.owner==prefix || f.owner.starts_with(&format!("{prefix}/"))).cloned().collect())
    }
    pub fn closure(streams:u32)->ClosureContract {
        let mut fields=(0..streams).map(|i|FieldContract{owner:format!("stream_{i}"),name:"native_owner_closure".into(),
            encoding:"prism.typed-host-state.v1".into(),expected_bytes:None,may_be_absent:false}).collect::<Vec<_>>();
        fields.push(FieldContract{owner:"shared".into(),name:"native_owner_closure".into(),encoding:"prism.typed-host-state.v1".into(),expected_bytes:None,may_be_absent:false});
        ClosureContract{schema_version:1,required_owners:fields.iter().map(|f|f.owner.clone()).collect(),fields,unresolved_owners:BTreeSet::new()}
    }
    fn decode(identity:&CheckpointIdentity,restored:RestoredCheckpoint)->Result<(Vec<NativeWorkerContinuationV1>,Option<NativeSharedContinuationV1>)>{
        Self::decode_with(identity,&restored.boundaries,|i|{
            let owner=format!("stream_{i}");
            restore_host(restored.fields.iter().find(|f|f.contract.owner==owner).context("missing stream owner")?)
        },||restore_host(restored.fields.iter().find(|f|f.contract.owner=="shared").context("missing shared native owners")?))
    }
    fn decode_with(identity:&CheckpointIdentity,boundaries:&[crate::scientific_checkpoint::StreamBoundary],
        mut read_worker:impl FnMut(u32)->Result<NativeWorkerContinuationV1>,
        read_shared:impl FnOnce()->Result<NativeSharedContinuationV1>) ->Result<(Vec<NativeWorkerContinuationV1>,Option<NativeSharedContinuationV1>)> {
        let mut workers=Vec::new();for i in 0..identity.stream_count {
            let value=read_worker(i)?;
            ensure!(value.schema_version==1 && value.stream_id==i && value.next_chunk>=0,"native worker identity/next chunk mismatch");
            if let Some(first)=workers.first(){let first:&NativeWorkerContinuationV1=first;
                ensure!(value.next_chunk==first.next_chunk && value.checkpoint_ordinal==first.checkpoint_ordinal,"native workers resume at different boundaries");}
            let boundary=boundaries.iter().find(|b|b.stream_id==i).context("native worker scientific boundary absent")?;
            if own_end_phase(&boundary.next_integrator_phase) {
                let completion:OwnEndCompletionV1=value.get("host:own_end_completion")?;
                let prefix:Option<OwnEndPrefixContinuationV1>=value.get_optional("host:own_end_prefix")?;
                if let Some(prefix)=&prefix {ensure!(boundary.next_integrator_phase==OWN_END_PREFIX_PHASE,"prefix owner/phase mismatch");prefix.validate_worker(&value,boundary.completed_step)?;}
                else{
                    if boundary.next_integrator_phase==MAY_STARTUP_PREFIX_PHASE {
                        let state:MayStartupContinuationV2=value.get_optional("host:may_startup")?
                            .context("startup body capsule lacks its typed pending original group")?;
                        state.validate(i as usize,value.next_chunk,1)?;
                        ensure!(state.pending_original_group.is_some() && value.next_chunk==0
                            && state.executed_physical_steps==boundary.completed_step,
                            "startup body phase/cursor/physical frontier differs before restore");
                    }else{ensure!(boundary.next_integrator_phase==OWN_END_RESTART_PHASE,"completed owner/phase mismatch");}
                    completion.validate_join(i,value.next_chunk,boundary.completed_step)?;
                }
                let host_steps:i32=value.get("host:steps_run")?;
                let _=host_steps;ensure!(worker_completed_native_step(&value)?==boundary.completed_step,"own-end host logical/startup physical frontier differs from native header");
                let cursor:Option<OriginalDispatchCursorV1>=value.get_optional("host:original_dispatch_cursor")?;
                ensure!(prefix.is_some() || cursor.is_none(),"completed own-end capsule retains partial cursor");
            }else{
                ensure!(!value.states.contains_key("host:own_end_completion"),"own-end completion cannot be admitted as ordinary equal-step continuation");
            }
            workers.push(value);
        }
        let shared=read_shared()?;
        ensure!(shared.schema_version==1,"unsupported shared native owner schema");
        if shared.states.contains_key("native_collective_boundary") {
            let (next_tick,coupling_epoch):(i32,u64)=shared.get("native_collective_boundary")?;
            ensure!(workers.iter().all(|w|w.next_chunk==next_tick) && boundaries.iter().all(|b|b.coupling_epoch==coupling_epoch),
                "shared barrier generation/common tick differs from per-worker native boundary");
        }else{
            ensure!(!boundaries.iter().any(|b|own_end_phase(&b.next_integrator_phase)),"own-end capsule lacks explicit collective/barrier join");
        }
        Ok((workers,Some(shared)))
    }
    pub fn worker_field(value:&NativeWorkerContinuationV1)->Result<crate::scientific_checkpoint::CapturedField>{
        capture_host(&format!("stream_{}",value.stream_id),"native_owner_closure",value)
    }
    pub fn shared_field(value:&NativeSharedContinuationV1)->Result<crate::scientific_checkpoint::CapturedField>{capture_host("shared","native_owner_closure",value)}
}
/// The outer CLI reserves the namespace before any dispatch receipt/output is
/// written. The run then claims that reservation without opening a conflicting
/// second flock descriptor. Only one active native run may claim it.
pub struct OutputReservation {file:File,claimed:std::sync::atomic::AtomicBool}
struct RunOutputClaim(Arc<OutputReservation>);
impl Drop for RunOutputClaim {fn drop(&mut self){self.0.claimed.store(false,std::sync::atomic::Ordering::Release);}}
impl Drop for OutputReservation {fn drop(&mut self){unsafe{libc::flock(self.file.as_raw_fd(),libc::LOCK_UN);}}}
pub fn reserve_output_root(output_root:&Path)->Result<Arc<OutputReservation>> {
    static RESERVATIONS:std::sync::Mutex<Vec<(PathBuf,std::sync::Weak<OutputReservation>)>>=std::sync::Mutex::new(Vec::new());
    ensure!(output_root.is_absolute() && output_root.components().all(|c|matches!(c,std::path::Component::RootDir|std::path::Component::Normal(_))),
        "scientific continuation requires an absolute output path without traversal");
    let mut prefix=PathBuf::new();
    for component in output_root.components(){
        prefix.push(component.as_os_str());
        match std::fs::symlink_metadata(&prefix){
            Ok(metadata)=>ensure!(metadata.is_dir() && !metadata.file_type().is_symlink(),"native output ancestor is not a real directory"),
            Err(error) if error.kind()==std::io::ErrorKind::NotFound=>{
                match std::fs::create_dir(&prefix){Ok(())=>{},Err(e) if e.kind()==std::io::ErrorKind::AlreadyExists=>{},Err(e)=>return Err(e.into())}
                let metadata=std::fs::symlink_metadata(&prefix)?;
                ensure!(metadata.is_dir() && !metadata.file_type().is_symlink(),"native output creation raced with a different file type");
                File::open(prefix.parent().context("native output parent absent")?)?.sync_all()?;
            },
            Err(error)=>return Err(error.into()),
        }
    }
    let root=output_root.to_path_buf();
    let mut entries=RESERVATIONS.lock().map_err(|_|anyhow::anyhow!("output reservation lock poisoned"))?;
    entries.retain(|(_,v)|v.strong_count()>0);
    if let Some(existing)=entries.iter().find(|(p,_)|*p==root).and_then(|(_,v)|v.upgrade()){return Ok(existing);}
    let file=OpenOptions::new().read(true).write(true).create(true).truncate(false)
        .custom_flags(libc::O_NOFOLLOW|libc::O_CLOEXEC).open(root.join("NATIVE_EXPERIMENT_LEASE"))?;
    ensure!(file.metadata()?.is_file(),"native output lease is not a regular file");
    ensure!(unsafe{libc::flock(file.as_raw_fd(),libc::LOCK_EX|libc::LOCK_NB)}==0,
        "another process owns this native experiment output/checkpoint lease");
    let reservation=Arc::new(OutputReservation{file,claimed:std::sync::atomic::AtomicBool::new(false)});
    entries.push((root,Arc::downgrade(&reservation)));Ok(reservation)
}

/// Exact buffered output boundary. Capture checks that the readable descriptor
/// refers to the actual writer inode before hashing; write-only legacy owners
/// need not change their scientific serialization.
pub fn seal_writer(writer:&mut std::io::BufWriter<File>,root:&Path,path:&Path)->Result<crate::scientific_checkpoint::AppendCursor>{
    use std::io::Write;use std::os::unix::fs::MetadataExt;
    writer.flush()?;writer.get_ref().sync_all()?;
    let reader=OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW|libc::O_CLOEXEC).open(path)?;
    let a=writer.get_ref().metadata()?;let b=reader.metadata()?;
    ensure!(a.dev()==b.dev() && a.ino()==b.ino(),"checkpoint output writer path changed");
    crate::scientific_checkpoint::seal_append_file(&reader,path.strip_prefix(root)?.to_str().context("non-UTF8 output locator")?)
}
pub fn open_writer(saved:Option<&NativeWorkerContinuationV1>,key:&str,root:&Path,path:&Path,create:bool)->Result<Option<std::io::BufWriter<File>>>{
    if let Some(saved)=saved {
        let cursor:Option<crate::scientific_checkpoint::AppendCursor>=saved.get_optional(key)?;
        if let Some(cursor)=cursor {
            ensure!(root.join(&cursor.relative_path)==path,"checkpoint writer locator changed for {key}");
            return Ok(Some(std::io::BufWriter::new(crate::scientific_checkpoint::reopen_append_file(root,&cursor)?)))
        }
        if path.try_exists()? {crate::checkpoint_output::retain_later_file(root,path)?;}
    }
    if create {Ok(Some(std::io::BufWriter::new(OpenOptions::new().read(true).write(true).create_new(true)
        .custom_flags(libc::O_NOFOLLOW|libc::O_CLOEXEC).open(path)?)))}else{Ok(None)}
}
#[derive(Serialize,Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeMetadataBoundaryV1 {
    pub schema_version:u32,
    #[serde(with="crate::checkpoint_codec::byte_map")]
    pub files:BTreeMap<String,Vec<u8>>,
    pub absent:BTreeSet<String>,
}
impl NativeMetadataBoundaryV1 {
    pub fn capture(root:&Path,paths:&[&Path])->Result<Self>{
        let mut files=BTreeMap::new();let mut absent=BTreeSet::new();
        for path in paths {let rel=path.strip_prefix(root)?.to_str().context("non-UTF8 metadata locator")?.to_owned();
            ensure!(!files.contains_key(&rel) && !absent.contains(&rel),"duplicate metadata owner");
            if path.try_exists()? {let mut f=OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW|libc::O_CLOEXEC).open(path)?;
                ensure!(f.metadata()?.is_file() && f.metadata()?.len()<=64*1024*1024,"metadata is not bounded regular file");
                f.sync_all()?;let mut bytes=Vec::new();std::io::Read::read_to_end(&mut f,&mut bytes)?;files.insert(rel,bytes);
            }else{absent.insert(rel);}}
        Ok(Self{schema_version:1,files,absent})
    }
    pub fn restore(&self,root:&Path)->Result<()> {
        use std::io::Write;
        ensure!(self.schema_version==1,"unknown metadata boundary");
        for rel in self.files.keys().chain(self.absent.iter()) {
            ensure!(Path::new(rel).components().all(|c|matches!(c,std::path::Component::Normal(_))),"invalid metadata path");
            ensure!(!(self.files.contains_key(rel)&&self.absent.contains(rel)),"duplicate metadata presence");
            let path=root.join(rel);
            if path.try_exists()? {
                let exact=if let Some(bytes)=self.files.get(rel) {
                    let mut f=OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW|libc::O_CLOEXEC).open(&path)?;
                    let mut actual=Vec::new();ensure!(f.metadata()?.len()<=64*1024*1024,"oversize existing metadata");
                    std::io::Read::read_to_end(&mut f,&mut actual)?;&actual==bytes
                }else{false};
                if exact{continue;}crate::checkpoint_output::retain_later_file(root,&path)?;
            }
            if let Some(bytes)=self.files.get(rel) {
                let mut f=OpenOptions::new().write(true).create_new(true).custom_flags(libc::O_NOFOLLOW|libc::O_CLOEXEC).open(&path)?;
                f.write_all(bytes)?;f.sync_all()?;File::open(path.parent().context("metadata parent")?)?.sync_all()?;
            }
        }Ok(())
    }
}

/// Exact per-step allocation provenance. Generations are process-local native
/// allocation identifiers and may repeat across process boundaries. Intervals,
/// committed checkpoint heads and reconstruction receipts disambiguate them.
pub fn allocation_generation_intervals(closed:&[(u64,u64)],current:u64,completed:u64)->Result<serde_json::Value>{
    ensure!(current!=0,"native allocation generation is zero");
    let mut end=0u64;let mut rows=Vec::new();
    for &(generation,stop) in closed {
        ensure!(generation!=0 && stop>end && stop<=completed,"invalid native allocation history interval");
        rows.push(serde_json::json!({"first_step":end+1,"last_step":stop,"allocation_generation":generation,"allocation_scope":"prior_process_committed_prefix"}));
        end=stop;
    }
    if completed>end {rows.push(serde_json::json!({"first_step":end+1,"last_step":completed,"allocation_generation":current,"allocation_scope":"current_process"}));}
    Ok(serde_json::Value::Array(rows))
}

/// Bind every mapped native shared library to the actual open inode before
/// hashing. Absolute installation locators intentionally remain part of the
/// same-provider continuation identity; relocation requires new qualification.
pub fn mapped_native_library_identities() -> anyhow::Result<std::collections::BTreeMap<String, String>> {
    use anyhow::{Context, ensure};
    let before = std::fs::read_to_string("/proc/self/maps")
        .context("read actual native provider mappings")?;
    let libraries = parse_mapped_native_libraries(&before)?;
    let mut identities = std::collections::BTreeMap::new();
    for (path, mapping) in &libraries {
        let hash = hash_mapped_native_library(mapping)?;
        identities.insert(format!("mapped_library:{path}"), hash);
    }
    let after = std::fs::read_to_string("/proc/self/maps")
        .context("recheck actual native provider mappings")?;
    ensure!(libraries == parse_mapped_native_libraries(&after)?,
        "mapped native provider population changed while hashing");
    Ok(identities)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MappedNativeLibrary {
    path: String,
    device_major: u64,
    device_minor: u64,
    inode: u64,
}

fn parse_mapped_native_libraries(maps: &str)
    -> anyhow::Result<std::collections::BTreeMap<String, MappedNativeLibrary>> {
    use anyhow::{Context, ensure};
    let mut libraries = std::collections::BTreeMap::new();
    for row in maps.lines() {
        // Consume exactly the five procfs columns. Preserve the remaining
        // locator verbatim, including internal/repeated spaces in filenames.
        let mut rest = row;
        let mut columns = Vec::with_capacity(5);
        for _ in 0..5 {
            rest = rest.trim_start_matches(|c: char| c.is_ascii_whitespace());
            let end = rest.find(|c: char| c.is_ascii_whitespace()).unwrap_or(rest.len());
            ensure!(end > 0, "malformed native mapping row");
            columns.push(&rest[..end]);
            rest = &rest[end..];
        }
        let path = rest.trim_start_matches(|c: char| c.is_ascii_whitespace());
        if !path.starts_with('/') { continue; }
        let name = std::path::Path::new(path).file_name().and_then(|s| s.to_str()).unwrap_or("");
        if !name.contains(".so") { continue; }
        ensure!(!path.ends_with(" (deleted)"), "mapped native library has been removed: {path}");
        // Procfs escapes embedded newline as a backslash-octal sequence, which
        // is ambiguous with a literal such filename. Refuse rather than guess.
        ensure!(!path.contains("\\012"), "ambiguous escaped native library locator: {path}");
        let (major, minor) = columns[3].split_once(':').context("mapping device lacks major/minor")?;
        let mapping = MappedNativeLibrary {
            path: path.into(),
            device_major: u64::from_str_radix(major, 16).context("mapping device major invalid")?,
            device_minor: u64::from_str_radix(minor, 16).context("mapping device minor invalid")?,
            inode: columns[4].parse().context("mapping inode invalid")?,
        };
        ensure!(mapping.inode != 0, "mapped native provider lacks a file inode: {path}");
        if let Some(existing) = libraries.get(path) {
            ensure!(existing == &mapping, "one native provider locator maps different device/inodes: {path}");
        } else {
            libraries.insert(path.into(), mapping);
        }
    }
    Ok(libraries)
}

fn same_native_file_version(a: &std::fs::Metadata, b: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    a.is_file() && b.is_file() && a.dev() == b.dev() && a.ino() == b.ino()
        && a.len() == b.len() && a.mode() == b.mode() && a.nlink() == b.nlink()
        && a.mtime() == b.mtime() && a.mtime_nsec() == b.mtime_nsec()
        && a.ctime() == b.ctime() && a.ctime_nsec() == b.ctime_nsec()
}

fn hash_mapped_native_library(mapping: &MappedNativeLibrary) -> anyhow::Result<String> {
    use anyhow::{Context, ensure};
    use sha2::{Digest, Sha256};
    use std::io::Read;
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
    let mut file = std::fs::OpenOptions::new().read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC).open(&mapping.path)
        .with_context(|| format!("open mapped native provider {}", mapping.path))?;
    let before = file.metadata()?;
    ensure!(before.is_file() && before.nlink() > 0
        && before.ino() == mapping.inode
        && libc::major(before.dev()) as u64 == mapping.device_major
        && libc::minor(before.dev()) as u64 == mapping.device_minor,
        "mapped native provider device/inode differs from opened descriptor: {}", mapping.path);
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 65536];
    let mut bytes = 0u64;
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 { break; }
        digest.update(&buffer[..n]);
        bytes = bytes.checked_add(n as u64).context("native provider byte count overflow")?;
    }
    let after = file.metadata()?;
    ensure!(bytes == before.len() && same_native_file_version(&before, &after),
        "mapped native provider changed during descriptor hashing: {}", mapping.path);
    let locator = std::fs::symlink_metadata(&mapping.path)
        .with_context(|| format!("recheck mapped native provider locator {}", mapping.path))?;
    ensure!(same_native_file_version(&before, &locator),
        "mapped native provider locator changed during descriptor hashing: {}", mapping.path);
    Ok(format!("{:x}", digest.finalize()))
}

#[cfg(test)]
mod mapped_native_provider_tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::{fs::File, os::fd::AsRawFd};

    struct Mapping { address: *mut libc::c_void, bytes: usize }
    impl Mapping {
        fn new(file: &File) -> Self {
            let bytes = file.metadata().unwrap().len() as usize;
            let address = unsafe { libc::mmap(std::ptr::null_mut(), bytes, libc::PROT_READ,
                libc::MAP_PRIVATE, file.as_raw_fd(), 0) };
            assert_ne!(address, libc::MAP_FAILED);
            unsafe { std::ptr::read_volatile(address.cast::<u8>()); }
            Self { address, bytes }
        }
    }
    impl Drop for Mapping {
        fn drop(&mut self) { assert_eq!(unsafe { libc::munmap(self.address, self.bytes) }, 0); }
    }
    fn locate(path: &std::path::Path) -> MappedNativeLibrary {
        let maps = std::fs::read_to_string("/proc/self/maps").unwrap();
        // Restrict this filesystem test to its own real kernel mapping; other
        // test threads may intentionally hold removed mappings simultaneously.
        let rows = maps.lines().filter(|r| r.ends_with(path.to_str().unwrap())).collect::<Vec<_>>().join("\n");
        parse_mapped_native_libraries(&rows).unwrap().remove(path.to_str().unwrap()).unwrap()
    }

    #[test]
    fn actual_mapped_inode_hash_and_duplicate_segments_preserve_spaces() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("native  provider.so");
        let bytes = vec![0x5au8; 4096];
        std::fs::write(&path, &bytes).unwrap();
        let file = File::open(&path).unwrap();
        let _first = Mapping::new(&file); let _second = Mapping::new(&file);
        let mapping = locate(&path);
        assert_eq!(hash_mapped_native_library(&mapping).unwrap(), format!("{:x}", Sha256::digest(&bytes)));
    }

    #[test]
    fn actual_atomic_replacement_after_proc_snapshot_is_refused() {
        let dir = tempfile::tempdir().unwrap(); let path = dir.path().join("provider.so");
        std::fs::write(&path, vec![0x11u8; 4096]).unwrap();
        let file = File::open(&path).unwrap(); let _mapping = Mapping::new(&file);
        let saved = locate(&path);
        let next = dir.path().join("replacement"); std::fs::write(&next, vec![0x22u8; 4096]).unwrap();
        std::fs::rename(next, &path).unwrap();
        let err = hash_mapped_native_library(&saved).unwrap_err().to_string();
        assert!(err.contains("device/inode"), "{err}");
        let maps = std::fs::read_to_string("/proc/self/maps").unwrap();
        let row = maps.lines().find(|r| r.ends_with(&format!("{} (deleted)", path.display()))).unwrap();
        assert!(parse_mapped_native_libraries(row).unwrap_err().to_string().contains("removed"));
    }

    #[test]
    fn actual_symlink_replacement_is_refused_and_equal_size_rewrite_changes_version() {
        use std::os::unix::fs::symlink;
        let dir = tempfile::tempdir().unwrap(); let path = dir.path().join("provider.so");
        std::fs::write(&path, vec![0x11u8; 4096]).unwrap();
        let file = File::open(&path).unwrap(); let _mapping = Mapping::new(&file);
        let saved = locate(&path); let before = file.metadata().unwrap();
        std::fs::write(&path, vec![0x33u8; 4096]).unwrap();
        assert!(!same_native_file_version(&before, &file.metadata().unwrap()));
        let retained = dir.path().join("retained"); std::fs::rename(&path, &retained).unwrap();
        symlink(&retained, &path).unwrap();
        assert!(hash_mapped_native_library(&saved).is_err());
    }

    #[test]
    fn actual_process_loaded_provider_population_is_hashable() {
        let qualified = concat!(module_path!(), "::loaded_provider_child");
        let test_name = qualified.split_once("::").unwrap().1;
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", test_name, "--ignored", "--nocapture"])
            .env("PRISM_PROVIDER_IDENTITY_TEST_CHILD", "1").output().unwrap();
        assert!(output.status.success(), "child stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
        assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"),
            "isolated real provider child test was not executed");
    }

    #[test]
    #[ignore = "invoked in an isolated real process by actual_process_loaded_provider_population_is_hashable"]
    fn loaded_provider_child() {
        assert_eq!(std::env::var("PRISM_PROVIDER_IDENTITY_TEST_CHILD").as_deref(), Ok("1"));
        let identities = mapped_native_library_identities().unwrap();
        assert!(!identities.is_empty());
        assert!(identities.keys().any(|key| key.contains("libc.so")));
        assert!(identities.values().all(|value| value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())));
    }
}

pub struct QuiescentSourceLease { _claim:RunOutputClaim }
pub fn claim_quiescent_branch_source(root:&Path)->Result<QuiescentSourceLease> {
    let metadata=std::fs::symlink_metadata(root.join("NATIVE_EXPERIMENT_LEASE"))?;
    ensure!(metadata.is_file() && !metadata.file_type().is_symlink(),"historical source has no existing native execution lease owner");
    let lease=reserve_output_root(root)?;
    ensure!(lease.claimed.compare_exchange(false,true,std::sync::atomic::Ordering::AcqRel,std::sync::atomic::Ordering::Acquire).is_ok(),"historical source is currently executing in this process");
    Ok(QuiescentSourceLease{_claim:RunOutputClaim(lease)})
}

/// Admission is remeasured on every ingress. Preserve prior successful or
/// interrupted evidence and publish a fresh bounded attempt, never overwrite.
fn publish_fresh_admission_receipt(root:&Path,name:&str,value:&serde_json::Value,head:Option<&Head>)->Result<PathBuf> {
    use std::io::Write;
    ensure!(matches!(name,"scientific_restart_host_capacity"|"scientific_restart_campaign_capacity"),"foreign admission receipt owner");
    // Two owners x64 attempts x16KiB stay within the original4MiB
    // branch-control reservation, without assuming receipt compression.
    for attempt in 0..64u32 {
        let filename=if attempt==0{format!("{name}.json")}else{format!("{name}.attempt-{attempt:05}.json")};let path=root.join(filename);
        match OpenOptions::new().write(true).create_new(true).custom_flags(libc::O_NOFOLLOW|libc::O_CLOEXEC).open(&path) {
            Ok(mut file)=>{
                let mut measured=value.clone();let object=measured.as_object_mut().context("admission receipt must be an object")?;
                object.insert("admission_attempt".into(),serde_json::json!({"schema_version":1,"attempt":attempt,"scientific_head":head,
                    "process_id":std::process::id(),"prior_attempts_preserved":true,"fresh_capacity_measurement":true}));
                let encoded=serde_json::to_vec_pretty(&measured)?;ensure!(encoded.len()<=16384,"fresh admission receipt exceeds its branch-control byte bound");
                file.write_all(&encoded)?;file.sync_all()?;File::open(root)?.sync_all()?;return Ok(path);
            },
            Err(error)if error.kind()==std::io::ErrorKind::AlreadyExists=>{
                let metadata=std::fs::symlink_metadata(&path)?;
                ensure!(metadata.is_file()&&!metadata.file_type().is_symlink(),"prior admission attempt is not a regular preserved file");
            },Err(error)=>return Err(error.into()),
        }
    }anyhow::bail!("admission receipt attempt population exceeded bounded capacity")
}
