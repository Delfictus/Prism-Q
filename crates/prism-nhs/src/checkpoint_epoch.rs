//! All-worker scientific checkpoint rendezvous. Physical and consumer fences
//! execute inside the pause; peers cannot launch the next graph until the
//! immutable generation and output cursors have been durably published.
use crate::scientific_checkpoint::{CapturedField,CheckpointIdentity,CheckpointStore,ClosureContract,Head,StreamBoundary};
use anyhow::{ensure,Context,Result};
use std::sync::{Arc,Mutex,MutexGuard,Condvar};
use std::time::{Duration,Instant};

pub struct StreamSnapshot {pub boundary:StreamBoundary,pub fields:Vec<CapturedField>}
pub struct StagedStreamSnapshot {pub boundary:StreamBoundary,pub fields:Vec<crate::scientific_checkpoint::restart_transaction::StagedField>}
pub enum EpochSnapshot {Resident(StreamSnapshot),Staged(StagedStreamSnapshot)}
impl EpochSnapshot {fn boundary(&self)->&StreamBoundary{match self{Self::Resident(v)=>&v.boundary,Self::Staged(v)=>&v.boundary}}}
struct RegistrationState {fields:Vec<Option<Vec<crate::scientific_checkpoint::FieldContract>>>,closure:Option<ClosureContract>,expected_digest:Option<String>}
struct EpochState {
    ordinal:u64,arrived:Vec<bool>,projected_bytes:Vec<Option<(u64,u64)>>,projection_capacity_admitted:bool,snapshots:Vec<Option<EpochSnapshot>>,
    completed:usize,released:usize,departed:usize,head:Option<Head>,failure:Option<String>,published:bool,
}
pub struct CheckpointEpoch {
    store:CheckpointStore,identity:CheckpointIdentity,closure:ClosureContract,
    state:Mutex<EpochState>,changed:Condvar,timeout:Duration,
    registration:Mutex<RegistrationState>,registration_changed:Condvar,require_registration:bool,
    staged_projection:Mutex<()>,
    staged_lanes:Mutex<usize>,staged_lanes_changed:Condvar,
}
/// A pause proof exists only after every native worker has arrived. A caller
/// may perform its synchronous CUDA/consumer drains and snapshot while holding
/// this borrow; no peer can advance to the next physical dispatch.
pub struct PausedGeneration<'a>{epoch:&'a CheckpointEpoch,ordinal:u64,stream_id:usize}
enum ScientificProjectionGuard<'a>{Serial(MutexGuard<'a,()>),Parallel(ScientificStagingLane<'a>)}
struct ScientificStagingLane<'a>{epoch:&'a CheckpointEpoch}
impl Drop for ScientificStagingLane<'_>{fn drop(&mut self){if let Ok(mut lanes)=self.epoch.staged_lanes.lock(){*lanes=lanes.saturating_sub(1);self.epoch.staged_lanes_changed.notify_one();}}}
impl PausedGeneration<'_>{
    pub fn ordinal(&self)->u64{self.ordinal}
    pub fn stream_id(&self)->usize{self.stream_id}
    pub fn identity(&self)->&CheckpointIdentity{&self.epoch.identity}
    /// Read-only raw index capability while the exact scientific cohort is
    /// paused. The epoch's held writer fd is reused across same-root handles;
    /// ordinary readers/writers keep their original cross-process lock path.
    pub fn validate_chronicle_index(&self,receiving:&CheckpointStore,head:&Head,identity:&CheckpointIdentity)
        ->Result<crate::scientific_checkpoint::chronicle::ValidatedChronicleIndex> {
        ensure!(identity==&self.epoch.identity && receiving.root()==self.epoch.store.root(),
            "paused Chronicle read belongs to another scientific execution/store");
        let state=self.epoch.lock()?;CheckpointEpoch::check(&state)?;
        ensure!(state.ordinal==self.ordinal && self.stream_id<state.arrived.len()
            && state.arrived.iter().all(|v|*v) && !state.published,
            "paused Chronicle read requires the exact live all-owner capture ordinal");
        drop(state);
        self.epoch.store.validate_chronicle_index_with_paused_writer(receiving,head,identity)
    }
}
impl CheckpointEpoch {
    pub fn new(store:CheckpointStore,identity:CheckpointIdentity,closure:ClosureContract,
        head:Option<Head>,next_ordinal:u64,timeout:Duration)->Result<Arc<Self>> {
        closure.validate()?;ensure!(!timeout.is_zero(),"checkpoint rendezvous timeout must be positive");
        let n=identity.stream_count as usize;ensure!(n>0,"empty checkpoint participant set");
        ensure!(store.head()?==head,"checkpoint epoch starts from stale HEAD");
        Ok(Arc::new(Self{store,identity,closure,state:Mutex::new(EpochState{ordinal:next_ordinal,
            arrived:vec![false;n],projected_bytes:vec![None;n],projection_capacity_admitted:false,snapshots:(0..n).map(|_|None).collect(),completed:0,released:0,departed:0,
            head,failure:None,published:false}),changed:Condvar::new(),timeout,
            registration:Mutex::new(RegistrationState{fields:(0..n).map(|_|None).collect(),closure:None,expected_digest:None}),registration_changed:Condvar::new(),require_registration:false,staged_projection:Mutex::new(()),staged_lanes:Mutex::new(0),staged_lanes_changed:Condvar::new()}))
    }
    /// Exact constructor field contracts are registered at zero physical
    /// propagation, before checkpoint host allocation or any graph replay.
    pub fn new_runtime_registered(store:CheckpointStore,identity:CheckpointIdentity,base:ClosureContract,
        head:Option<Head>,next_ordinal:u64,timeout:Duration,expected_digest:Option<String>)->Result<Arc<Self>> {
        let mut epoch=Self::new(store,identity,base,head,next_ordinal,timeout)?;
        let value=Arc::get_mut(&mut epoch).context("new epoch unexpectedly shared")?;
        value.require_registration=true;
        value.registration.get_mut().map_err(|_|anyhow::anyhow!("registration poisoned"))?.expected_digest=expected_digest;
        Ok(epoch)
    }
    pub fn register_runtime_fields(&self,stream_id:usize,fields:Vec<crate::scientific_checkpoint::FieldContract>)->Result<()> {
        ensure!(self.require_registration,"runtime registration not selected");
        let deadline=Instant::now().checked_add(self.timeout).context("registration deadline overflow")?;
        let mut state=self.registration.lock().map_err(|_|anyhow::anyhow!("registration poisoned"))?;
        ensure!(stream_id<state.fields.len() && state.fields[stream_id].is_none(),"duplicate/unadmitted runtime owner registration");
        let prefix=format!("stream_{stream_id}/engine");
        ensure!(fields.iter().all(|f|f.owner==prefix || f.owner.starts_with(&format!("{prefix}/"))),"runtime owner field escapes worker authority");
        state.fields[stream_id]=Some(fields);
        if state.fields.iter().all(Option::is_some) {
            let mut closure=self.closure.clone();
            for field in state.fields.iter().flat_map(|v|v.as_ref().into_iter().flatten()) {closure.fields.push(field.clone());closure.required_owners.insert(field.owner.clone());}
            closure.validate()?;
            if let Some(expected)=&state.expected_digest {ensure!(&closure.digest()?==expected,"fresh complete owner extents differ from saved generation");}
            state.closure=Some(closure);self.registration_changed.notify_all();
        }
        while state.closure.is_none() {
            {let current=self.lock()?;Self::check(&current)?;}
            let remaining=deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero(){drop(state);self.abort("runtime owner registration timed out");anyhow::bail!("runtime owner registration timed out");}
            state=self.registration_changed.wait_timeout(state,remaining).map_err(|_|anyhow::anyhow!("registration wait poisoned"))?.0;
        }
        Ok(())
    }
    /// All workers declare their actual current engine extents plus bounded
    /// outer closure size before any field payload is allocated/written. This
    /// makes the generation cap a pre-write bound, including abort/orphan cost.
    pub fn admit_generation_projection(&self,proof:&PausedGeneration<'_>,bytes:u64,host_peak:u64)->Result<()> {
        ensure!(std::ptr::eq(proof.epoch,self),"foreign generation extent proof");
        let deadline=Instant::now().checked_add(self.timeout).context("projection admission deadline overflow")?;
        let mut state=self.lock()?;Self::check(&state)?;
        ensure!(state.ordinal==proof.ordinal && state.arrived.iter().all(|v|*v) && state.projected_bytes[proof.stream_id].is_none(),
            "generation extent admission outside exact all-worker pause");
        state.projected_bytes[proof.stream_id]=Some((bytes,host_peak));self.changed.notify_all();
        while !state.projected_bytes.iter().all(Option::is_some) {state=self.wait(state,deadline)?;}
        if !state.projection_capacity_admitted {
            let total=state.projected_bytes.iter().try_fold(0u64,|sum,value|sum.checked_add(value.context("missing generation extent")?.0).context("generation extent sum overflow"))?;
            let host_total=state.projected_bytes.iter().try_fold(128*1024*1024u64,|sum,value|
                sum.checked_add(value.context("missing generation host extent")?.1).context("capture host peak overflow"))?;
            if let Err(error)=crate::scientific_checkpoint::restart_transaction::admit_process_capture_host_peak(host_total,128*1024*1024) {
                state.failure=Some(format!("all-worker preallocation host admission: {error:#}"));
                self.changed.notify_all();Self::check(&state)?;
            }
            if total>self.store.maximum_generation_bytes() {
                state.failure=Some(format!("all-worker generation extents {} exceed pre-write generation cap {}",total,self.store.maximum_generation_bytes()));
                self.changed.notify_all();Self::check(&state)?;
            }
            // One collective measurement, before any producer is released.
            // Later arrivals must not recharge the same peak after another
            // admitted worker has legitimately started its readback.
            self.store.begin_parallel_scientific_staging()?;
            state.projection_capacity_admitted=true;self.changed.notify_all();
        }
        Ok(())
    }
    fn staged_lane(&self,owner:&str,field:&str)->Result<ScientificProjectionGuard<'_>> {
        ensure!(!self.failed(),"scientific checkpoint aborted before field projection");
        if !self.store.parallel_scientific_staging_active()? {
            let projection=self.staged_projection.lock().map_err(|_|anyhow::anyhow!("staged projection poisoned"))?;
            ensure!(!self.failed(),"scientific checkpoint aborted while waiting for serial field projection");
            return Ok(ScientificProjectionGuard::Serial(projection));
        }
        let deadline=Instant::now().checked_add(self.timeout).context("staging lane deadline overflow")?;
        let start=Instant::now();let mut lanes=self.staged_lanes.lock().map_err(|_|anyhow::anyhow!("scientific staging lanes poisoned"))?;
        while *lanes>=2 {
            ensure!(!self.failed(),"scientific checkpoint aborted before staging lane");
            let remaining=deadline.saturating_duration_since(Instant::now());ensure!(!remaining.is_zero(),"scientific staging lane timed out");
            lanes=self.staged_lanes_changed.wait_timeout(lanes,remaining).map_err(|_|anyhow::anyhow!("scientific staging lane wait poisoned"))?.0;
        }
        ensure!(!self.failed(),"scientific checkpoint aborted before field projection");*lanes+=1;
        log::info!("SCIENTIFIC_STAGING_LANE owner={} field={} active={} maximum=2 wait_ns={}",owner,field,*lanes,start.elapsed().as_nanos());
        Ok(ScientificProjectionGuard::Parallel(ScientificStagingLane{epoch:self}))
    }
    /// Reuse the same in-process projection admission during fresh restore;
    /// only one complete device-field temporary exists across worker restores.
    pub fn with_serial_restore<T>(&self,restore:impl FnOnce()->Result<T>)->Result<T> {
        let _projection=self.staged_projection.lock().map_err(|_|anyhow::anyhow!("restore projection poisoned"))?;
        restore()
    }
    pub fn stage_field(&self,field:&CapturedField,max_field_bytes:u64)->Result<crate::scientific_checkpoint::restart_transaction::StagedField> {
        // Projected all-owner capture shares one cross-process writer lease
        // through two bounded staging lanes. Legacy nonprojected APIs retain
        // their original serial same-epoch projection guard.
        let _lane=self.staged_lane(&field.contract.owner,&field.contract.name)?;
        self.store.stage_field_lossless(field,max_field_bytes)
    }
    #[cfg(feature="gpu")]
    pub fn stage_field_streaming(&self,plan:&crate::fused_engine::ScientificFieldPlan,
        producer:&mut dyn FnMut(&mut dyn std::io::Write)->Result<()>)->Result<crate::scientific_checkpoint::restart_transaction::StagedField> {
        let _lane=self.staged_lane(&plan.contract.owner,&plan.contract.name)?;
        self.store.stage_field_streaming(&plan.contract,plan.present_bytes,
            (plan.contract.encoding=="prism.typed-host-state.v1" || plan.contract.encoding==crate::scientific_checkpoint::native_spike_archive::NATIVE_SPIKE_ARCHIVE_ENCODING).then_some(plan.temporary_bytes),producer)
    }
    pub fn stage_native_spike_archive(&self,contract:&crate::scientific_checkpoint::FieldContract,head:Option<&Head>,
        producer:crate::scientific_checkpoint::chronicle::ProducerKey,seed:u64,instance:u64,mutation_epoch:u64,first:u64,next:u64,
        previous:Option<&crate::scientific_checkpoint::native_spike_archive::NativeSpikeArchiveWitness>,
        compare:impl FnMut(u64,&[u8])->Result<()>) ->Result<(crate::scientific_checkpoint::restart_transaction::StagedField,crate::scientific_checkpoint::native_spike_archive::NativeSpikeArchiveWitness)> {
        let _lane=self.staged_lane(&contract.owner,&contract.name)?;
        self.store.stage_native_spike_archive(contract,&self.identity,head,producer,seed,instance,mutation_epoch,first,next,previous,compare)
    }
    fn lock(&self)->Result<MutexGuard<'_,EpochState>> {self.state.lock().map_err(|_|anyhow::anyhow!("poisoned checkpoint epoch"))}
    pub fn failed(&self)->bool{self.state.lock().map_or(true,|state|state.failure.is_some())}
    pub fn committed_head(&self)->Result<Option<Head>> {
        let state=self.lock()?;Self::check(&state)?;Ok(state.head.clone())
    }
    /// Only after every participant has returned from its last durable release.
    pub fn quiescent_committed_head(&self)->Result<Head> {
        let state=self.lock()?;Self::check(&state)?;
        ensure!(!state.published && state.completed==0 && state.released==0 && state.departed==0
            && state.arrived.iter().all(|v|!*v) && state.snapshots.iter().all(Option::is_none)
            && state.projected_bytes.iter().all(Option::is_none),"reservation handoff requires a fully departed checkpoint epoch");
        state.head.clone().context("no committed scientific generation to hand off")
    }
    pub fn abort(&self,cause:impl Into<String>) {if let Ok(mut s)=self.state.lock(){if s.failure.is_none(){s.failure=Some(cause.into());}self.changed.notify_all();self.registration_changed.notify_all();self.staged_lanes_changed.notify_all();}self.store.abort_parallel_scientific_staging();}
    fn check(s:&EpochState)->Result<()> {if let Some(cause)=&s.failure{anyhow::bail!("scientific checkpoint aborted: {cause}");}Ok(())}
    fn wait<'a>(&'a self,mut s:MutexGuard<'a,EpochState>,deadline:Instant)->Result<MutexGuard<'a,EpochState>> {
        Self::check(&s)?;let remaining=deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero(){s.failure=Some("checkpoint rendezvous timed out before all native owners acknowledged".into());self.changed.notify_all();Self::check(&s)?;}
        let (s,_)=self.changed.wait_timeout(s,remaining).map_err(|_|anyhow::anyhow!("poisoned checkpoint epoch wait"))?;
        Self::check(&s)?;Ok(s)
    }
    /// Called by every worker for the same completed coupled-window ordinal.
    /// `capture` must drain that worker's streams and output consumers, then
    /// collect every admitted owner. Stream zero additionally collects shared
    /// coupling/controller state while the same pause remains in force.
    pub fn checkpoint(&self,stream_id:usize,ordinal:u64,
        capture:impl FnOnce(&PausedGeneration<'_>)->Result<StreamSnapshot>)->Result<Head> {
        self.checkpoint_with_consumer_release(stream_id,ordinal,capture,||Ok(()))
    }
    /// Consumer pause guards live outside `capture`. Their release runs only
    /// after HEAD is durable, and every release must complete before any worker
    /// can launch its next native graph.
    pub fn checkpoint_with_consumer_release(&self,stream_id:usize,ordinal:u64,
        capture:impl FnOnce(&PausedGeneration<'_>)->Result<StreamSnapshot>,
        release:impl FnOnce()->Result<()>)->Result<Head> {
        self.checkpoint_internal(stream_id,ordinal,|proof|capture(proof).map(EpochSnapshot::Resident),release)
    }
    pub fn checkpoint_streamed_with_consumer_release(&self,stream_id:usize,ordinal:u64,
        capture:impl FnOnce(&PausedGeneration<'_>)->Result<StagedStreamSnapshot>,
        release:impl FnOnce()->Result<()>)->Result<Head> {
        self.checkpoint_internal(stream_id,ordinal,|proof|capture(proof).map(EpochSnapshot::Staged),release)
    }
    pub fn checkpoint_projection_with_consumer_release(&self,stream_id:usize,ordinal:u64,
        capture:impl FnOnce(&PausedGeneration<'_>)->Result<EpochSnapshot>,release:impl FnOnce()->Result<()>)->Result<Head> {
        self.checkpoint_internal(stream_id,ordinal,capture,release)
    }
    fn checkpoint_internal(&self,stream_id:usize,ordinal:u64,
        capture:impl FnOnce(&PausedGeneration<'_>)->Result<EpochSnapshot>,
        release:impl FnOnce()->Result<()>)->Result<Head> {
        if self.require_registration {let reg=self.registration.lock().map_err(|_|anyhow::anyhow!("registration poisoned"))?;ensure!(reg.closure.is_some(),"scientific owner extents were not admitted before capture");}
        let deadline=Instant::now().checked_add(self.timeout).context("checkpoint deadline overflow")?;
        let mut state=self.lock()?;Self::check(&state)?;
        ensure!(stream_id<state.arrived.len(),"unadmitted checkpoint worker");
        // The fastest worker may arrive at the next checkpoint before the last
        // peer has returned from the previous one. Wait for departure reset.
        while ordinal>state.ordinal && state.published{state=self.wait(state,deadline)?;}
        if ordinal!=state.ordinal || state.arrived[stream_id] {
            state.failure=Some(format!("checkpoint worker {stream_id} arrived with duplicate/mismatched ordinal {ordinal}; expected {}",state.ordinal));
            self.changed.notify_all();Self::check(&state)?;
        }
        state.arrived[stream_id]=true;self.changed.notify_all();
        while !state.arrived.iter().all(|x|*x){state=self.wait(state,deadline)?;}
        drop(state);
        let proof=PausedGeneration{epoch:self,ordinal,stream_id};
        // Abort waiters even when an owner panics; never leave a live peer
        // indefinitely blocked in a scientific barrier.
        struct CaptureGuard<'a>{epoch:&'a CheckpointEpoch,armed:bool}
        impl Drop for CaptureGuard<'_>{fn drop(&mut self){if self.armed{self.epoch.abort("native checkpoint owner exited before acknowledgment");}}}
        let mut guard=CaptureGuard{epoch:self,armed:true};
        let snapshot=match capture(&proof){Ok(value)=>value,Err(error)=>{self.abort(format!("worker {stream_id} checkpoint capture: {error:#}"));return Err(error);}};
        ensure!(std::ptr::eq(proof.epoch,self),"foreign checkpoint pause proof");
        if snapshot.boundary().stream_id as usize!=stream_id {self.abort("snapshot worker identity differs from registered participant");anyhow::bail!("checkpoint stream identity mismatch");}
        let mut state=self.lock()?;Self::check(&state)?;
        state.snapshots[stream_id]=Some(snapshot);state.completed+=1;
        guard.armed=false;
        if state.completed==state.arrived.len() {
            let mut boundaries=Vec::with_capacity(state.snapshots.len());let mut fields=Vec::new();let mut staged=Vec::new();
            let mut mode:Option<bool>=None;
            for saved in &mut state.snapshots {
                match saved.take().context("missing acknowledged owner")? {
                    EpochSnapshot::Resident(value)=>{ensure!(mode.is_none_or(|m|!m),"mixed scientific capture modes");mode=Some(false);boundaries.push(value.boundary);fields.extend(value.fields);},
                    EpochSnapshot::Staged(value)=>{ensure!(mode.is_none_or(|m|m),"mixed scientific capture modes");mode=Some(true);boundaries.push(value.boundary);staged.extend(value.fields);},
                }
            }
            let closure=if self.require_registration {self.registration.lock().map_err(|_|anyhow::anyhow!("registration poisoned"))?.closure.clone().context("runtime closure missing")?}else{self.closure.clone()};
            if mode==Some(true) && state.projection_capacity_admitted {
                if let Err(error)=self.store.finish_parallel_scientific_staging(){
                    state.failure=Some(format!("scientific staging lease completion: {error:#}"));
                    self.store.abort_parallel_scientific_staging();self.changed.notify_all();return Err(error);
                }
            }
            let publication_start=Instant::now();
            let committed=if mode==Some(true){self.store.commit_staged(state.head.as_ref(),&self.identity,&closure,&boundaries,&staged)}else{self.store.commit(state.head.as_ref(),&self.identity,&closure,&boundaries,&fields)};
            match committed {
                Ok(head)=>{log::info!("SCIENTIFIC_GENERATION_PUBLICATION ordinal={} fields={} elapsed_ns={}",ordinal,closure.fields.len(),publication_start.elapsed().as_nanos());state.head=Some(head);state.published=true;},
                Err(error)=>{state.failure=Some(format!("durable native checkpoint publication: {error:#}"));self.changed.notify_all();return Err(error);}
            }
            self.changed.notify_all();
        }
        while !state.published{state=self.wait(state,deadline)?;}
        let head=state.head.clone().context("published checkpoint missing HEAD")?;
        drop(state);
        let mut release_guard=CaptureGuard{epoch:self,armed:true};
        if let Err(error)=release(){self.abort(format!("worker {stream_id} durable checkpoint consumer release: {error:#}"));return Err(error);}
        release_guard.armed=false;
        let mut state=self.lock()?;Self::check(&state)?;
        state.released+=1;self.changed.notify_all();
        while state.released<state.arrived.len(){state=self.wait(state,deadline)?;}
        state.departed+=1;
        if state.departed==state.arrived.len(){
            state.ordinal=state.ordinal.checked_add(1).context("checkpoint ordinal overflow")?;
            state.arrived.fill(false);state.projected_bytes.fill(None);state.projection_capacity_admitted=false;state.completed=0;state.released=0;state.departed=0;state.published=false;self.changed.notify_all();
        }Ok(head)
    }
    /// Guard each native worker from initial admission through its last common
    /// checkpoint. An unexpected Result exit cancels the rendezvous immediately.
    pub fn participant(self:&Arc<Self>,stream_id:usize)->Result<CheckpointParticipant> {
        ensure!(stream_id<self.identity.stream_count as usize,"unadmitted checkpoint worker");
        Ok(CheckpointParticipant{epoch:self.clone(),stream_id,closed:false})
    }
}
pub struct CheckpointParticipant {epoch:Arc<CheckpointEpoch>,stream_id:usize,closed:bool}
impl CheckpointParticipant {pub fn close_at_common_horizon(mut self){self.closed=true;}}
impl Drop for CheckpointParticipant {fn drop(&mut self){if !self.closed{self.epoch.abort(format!("native worker {} exited before common checkpoint horizon",self.stream_id));}}}

#[cfg(test)]
mod tests {
    use super::*;use crate::scientific_checkpoint::{capture_host,FieldContract};use std::collections::BTreeSet;
    fn create(timeout:Duration)->(tempfile::TempDir,Arc<CheckpointEpoch>){
        let root=tempfile::tempdir().unwrap();let store=CheckpointStore::create(&root.path().join("store"),1024*1024).unwrap();
        let fields=(0..2).map(|i|FieldContract{owner:format!("stream_{i}"),name:"whole_owner".into(),encoding:"prism.typed-host-state.v1".into(),expected_bytes:None,may_be_absent:false}).collect::<Vec<_>>();
        let closure=ClosureContract{schema_version:1,required_owners:fields.iter().map(|f|f.owner.clone()).collect(),fields,unresolved_owners:BTreeSet::new()};
        let identity=CheckpointIdentity{run_id:"epoch-storage-unit-test".into(),topology_sha256:"1".repeat(64),engine_sha256:"2".repeat(64),kernel_set_sha256:"3".repeat(64),protocol_sha256:"4".repeat(64),execution_request_sha256:"5".repeat(64),state_abi_sha256:"6".repeat(64),provider:"native checkpoint rendezvous test only".into(),stream_count:2};
        let epoch=CheckpointEpoch::new(store,identity,closure,None,0,timeout).unwrap();(root,epoch)
    }
    fn snapshot(i:usize,ordinal:u64)->StreamSnapshot{StreamSnapshot{boundary:StreamBoundary{stream_id:i as u32,replica_epoch:0,replica_seed:i as u64,completed_step:ordinal+1,physical_time_ps_bits:((ordinal+1)as f64).to_bits(),next_integrator_phase:"completed-window".into(),coupling_epoch:ordinal,coupling_parity:0,committed_output_sequence:ordinal+1},fields:vec![capture_host(&format!("stream_{i}"),"whole_owner",&(i,ordinal)).unwrap()]}}
    #[test]fn every_worker_stays_paused_through_publication_and_next_epoch(){
        let (_root,epoch)=create(Duration::from_secs(5));let arrived=Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let handles=(0..2).map(|i|{let e=epoch.clone();let arrived=arrived.clone();std::thread::spawn(move||{
            let participant=e.participant(i).unwrap();for ordinal in 0..4 {arrived.fetch_add(1,std::sync::atomic::Ordering::SeqCst);
                let head=e.checkpoint(i,ordinal,|proof|{assert_eq!(proof.ordinal(),ordinal);
                    assert!(arrived.load(std::sync::atomic::Ordering::SeqCst)>=(ordinal as usize+1)*2);
                    Ok(snapshot(i,ordinal))}).unwrap();assert_eq!(head.generation,ordinal+1);}
            participant.close_at_common_horizon();})}).collect::<Vec<_>>();
        for h in handles{h.join().unwrap();}assert_eq!(epoch.store.head().unwrap().unwrap().generation,4);
    }
    #[test]fn missing_worker_and_failed_owner_cannot_publish(){
        let (_root,epoch)=create(Duration::from_millis(20));
        assert!(epoch.checkpoint(0,0,|_|Ok(snapshot(0,0))).is_err());assert!(epoch.store.head().unwrap().is_none());
        let (_root,epoch)=create(Duration::from_secs(2));
        let e=epoch.clone();let peer=std::thread::spawn(move||e.checkpoint(1,0,|_|Ok(snapshot(1,0))));
        assert!(epoch.checkpoint(0,0,|_|anyhow::bail!("native output fsync failed")).is_err());
        assert!(peer.join().unwrap().is_err());assert!(epoch.store.head().unwrap().is_none());
    }
    #[test]fn participant_exit_wakes_peers_without_timeout(){
        let (_root,epoch)=create(Duration::from_secs(60));let participant=epoch.participant(1).unwrap();
        let e=epoch.clone();let peer=std::thread::spawn(move||e.checkpoint(0,0,|_|Ok(snapshot(0,0))));
        drop(participant);assert!(peer.join().unwrap().is_err());assert!(epoch.store.head().unwrap().is_none());
    }
}
