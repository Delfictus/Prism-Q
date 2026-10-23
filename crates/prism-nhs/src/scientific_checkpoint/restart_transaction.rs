//! Streaming projections and cumulative reservation in the original store.
//! Objects staged before a generation commits are unreachable forensic objects;
//! they cannot advance HEAD, a worker cursor, or scientific execution.
use super::*;
use anyhow::bail;
use std::os::unix::fs::MetadataExt;

#[derive(Clone)]
pub struct StagedField {pub(super) store_root:PathBuf,pub(super) stored:StoredField}
impl StagedField {
    pub fn contract(&self)->&FieldContract {&self.stored.contract}
    pub fn byte_count(&self)->u64 {self.stored.byte_count}
}
#[derive(Clone,Debug,Serialize,Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReservationContract {
    pub schema_version:u32,
    pub identity:CheckpointIdentity,
    pub maximum_retained_bytes:u64,
    pub remaining_free_floor_bytes:u64,
}
/// Same-store process lease; distinct from the short writer transaction lock.
pub struct ReservationLease {file:File,root:PathBuf,lease_inode:u64,lease_device:u64,reserved_inode:u64,reserved_device:u64,pub contract:ReservationContract}
impl Drop for ReservationLease {fn drop(&mut self){unsafe{libc::flock(self.file.as_raw_fd(),libc::LOCK_UN);}}}

impl CheckpointStore {
    /// Called under the original writer lock. Scientific state is frozen at
    /// the successful all-worker MD join; raw direct-consumer CAS remains live.
    pub(super) fn ensure_scientific_producer_open(&self)->Result<()> {
        ensure!(!self.root.join("RESTART_MD_CAPACITY_HANDOFF_PREPARED.json").exists()
            && !self.root.join("RESTART_MD_CAPACITY_HANDOFF.json").exists(),
            "scientific producers closed at MD-to-PathB capacity handoff");
        Ok(())
    }
    pub fn root(&self)->&Path {&self.root}
    pub fn maximum_generation_bytes(&self)->u64 {self.max_total_bytes}

    /// Reserve actual filesystem blocks before MD/checkpoint host allocation.
    /// The admitted amount includes both HEAD namespaces when sharing a store.
    /// This never deletes a prior generation or assumes compression/dedup.
    pub fn reserve_retained_capacity(&self,contract:ReservationContract)->Result<ReservationLease> {
        contract.identity.validate()?;
        ensure!(contract.schema_version==1 && contract.maximum_retained_bytes>0,"invalid cumulative store reservation");
        let _lock=self.lock()?;
        let file=OpenOptions::new().read(true).write(true).create(true).truncate(false)
            .custom_flags(libc::O_NOFOLLOW|libc::O_CLOEXEC).open(self.root.join("RESTART_RESERVATION_LEASE"))?;
        ensure!(file.metadata()?.is_file(),"reservation lease is not regular");
        ensure!(unsafe{libc::flock(file.as_raw_fd(),libc::LOCK_EX|libc::LOCK_NB)}==0,"store reservation already owned by another process");
        ensure!(!self.root.join("RESTART_RESERVATION_RETIREMENT.json").exists() && !self.root.join("RESTART_RESERVATION_RETIREMENT_PREPARED.json").exists()
            && !self.root.join("RESTART_MD_CAPACITY_HANDOFF_PREPARED.json").exists()
            && !self.root.join("RESTART_MD_CAPACITY_HANDOFF.json").exists()
            && !self.root.join("RESTART_EMPTY_FAILURE_RETIREMENT_PREPARED.json").exists()
            && !self.root.join("RESTART_EMPTY_FAILURE_RETIREMENT.json").exists(),
            "retired/unreconciled source store cannot reacquire capacity; restore requires its separately admitted historical branch");
        let metadata=file.metadata()?;
        let mut lease=ReservationLease{file,root:self.root.clone(),lease_inode:metadata.ino(),lease_device:metadata.dev(),reserved_inode:0,reserved_device:0,contract};
        let contract_path=self.root.join("RESTART_RESERVATION.json");
        if contract_path.exists() {
            let previous:ReservationContract=serde_json::from_slice(&read_bounded(&contract_path,16384)?)?;
            ensure!(previous.schema_version==lease.contract.schema_version && previous.identity==lease.contract.identity
                && previous.maximum_retained_bytes==lease.contract.maximum_retained_bytes
                && previous.remaining_free_floor_bytes==lease.contract.remaining_free_floor_bytes,
                "cumulative reservation contract changed");
        } else {
            write_new(&contract_path,&serde_json::to_vec(&lease.contract)?)?;
        }
        let used=self.retained_object_bytes()?;
        ensure!(used<=lease.contract.maximum_retained_bytes,"retained CAS population exceeds cumulative admission");
        let remaining=lease.contract.maximum_retained_bytes-used;
        let reserve_path=self.root.join("RESTART_RESERVED_SPACE");
        let reserve=OpenOptions::new().read(true).write(true).create(true).truncate(false)
            .custom_flags(libc::O_NOFOLLOW|libc::O_CLOEXEC).open(reserve_path)?;
        ensure!(reserve.metadata()?.is_file(),"reserved space is not regular");
        let allocated=reserve.metadata()?.blocks().checked_mul(512).context("reserved block count overflow")?;
        let additional=remaining.saturating_sub(allocated);
        let free=self.filesystem_available_bytes()?;
        ensure!(free>=additional.checked_add(lease.contract.remaining_free_floor_bytes).context("reservation free-space overflow")?,
            "insufficient real free space for cumulative checkpoint reservation");
        reserve.set_len(remaining)?;
        if remaining>0 {
            let length=libc::off_t::try_from(remaining)?;
            let rc=unsafe{libc::posix_fallocate(reserve.as_raw_fd(),0,length)};
            ensure!(rc==0,"filesystem refused real reserved blocks: {}",std::io::Error::from_raw_os_error(rc));
        }
        reserve.sync_all()?;sync_dir(&self.root)?;
        let metadata=reserve.metadata()?;lease.reserved_inode=metadata.ino();lease.reserved_device=metadata.dev();
        Ok(lease)
    }
    /// Release only surplus scientific scratch after final SCI closure. Keep
    /// the complete preadmitted raw-store bound and the SAME live lease for
    /// selection/Arrow acknowledgements. No datum or immutable CAS is removed.
    pub fn handoff_unused_scientific_capacity(&self,lease:&ReservationLease,head:&Head,
        identity:&CheckpointIdentity,raw_retention_bound:u64,collective_chunk_count:u64,terminal:&serde_json::Value)->Result<serde_json::Value> {
        ensure!(lease.root==self.root && lease.contract.identity==*identity && raw_retention_bound>0,
            "foreign/empty MD capacity handoff");
        let _lock=self.lock()?;
        self.ensure_scientific_producer_open()?;
        ensure!(!self.root.join("RESTART_RESERVATION_RETIREMENT_PREPARED.json").exists()
            && !self.root.join("RESTART_EMPTY_FAILURE_RETIREMENT_PREPARED.json").exists(),"capacity already retired");
        ensure!(self.head()?.as_ref()==Some(head),"MD capacity handoff SCIHEAD changed");
        let manifest=self.read_manifest(head)?;
        ensure!(manifest.identity==*identity,"MD capacity handoff final SCI identity changed");
        // Decode only the already committed native host closures, not another
        // full device generation. This is the SAME publication validator used
        // for exact own-end SCI commit/restore joins.
        let publication=self.own_end_publication(identity,&manifest.boundaries,&manifest.fields)?;
        ensure!(!publication.prefix_pending && u64::try_from(publication.common_next_tick)?==collective_chunk_count
            && manifest.boundaries.iter().zip(&publication.physical_owner_targets).all(|(boundary,target)|boundary.completed_step==*target),
            "normal MD handoff lacks the complete original collective horizon and all physical owner endpoints");
        ensure!(terminal["boundary"]=="NORMAL_MD_JOIN_BEFORE_PATHB"
            && terminal["all_worker_threads_joined"]==true && terminal["native_worker_failures"]==0
            && terminal["checkpoint_epoch_quiescent"]==true,"MD capacity handoff lacks successful quiescent all-worker join");
        let raw_head=self.chronicle_head()?;
        let saved_raw=publication.raw_head.as_ref().context("final SCI raw source absent")?;
        let final_raw=raw_head.as_ref().context("terminal raw HEAD absent")?;
        let raw_closure=self.verify_terminal_raw_control_suffix(saved_raw,final_raw,identity,&manifest.boundaries,terminal)?;
        let flow=&terminal["flow_summary"];
        ensure!(raw_head.is_some() && flow["identity"]==serde_json::to_value(identity)?
            && flow["foliage_head"]==serde_json::to_value(&raw_head)?,"MD capacity handoff raw HEAD/identity differs");
        let queues=flow["native_queue_prefixes"].as_array().context("MD handoff raw queue population absent")?;
        ensure!(queues.len()==identity.stream_count as usize && queues.iter().all(|q|q["captured_records"]==0
            && q["awaiting_existing_reset"]==false),"MD handoff has uncaptured/unretired native queue prefixes");
        let disk_contract:ReservationContract=serde_json::from_slice(&read_bounded(&self.root.join("RESTART_RESERVATION.json"),16384)?)?;
        ensure!(serde_json::to_vec(&disk_contract)?==serde_json::to_vec(&lease.contract)?,"MD handoff capacity contract changed");
        let lm=lease.file.metadata()?;let on_disk=fs::symlink_metadata(self.root.join("RESTART_RESERVATION_LEASE"))?;
        ensure!(lm.is_file() && lm.nlink()==1 && lm.ino()==lease.lease_inode && lm.dev()==lease.lease_device
            && on_disk.is_file() && on_disk.nlink()==1 && on_disk.ino()==lm.ino() && on_disk.dev()==lm.dev(),"MD handoff lease inode/device changed");
        let reserve=OpenOptions::new().read(true).write(true).custom_flags(libc::O_NOFOLLOW|libc::O_CLOEXEC)
            .open(self.root.join("RESTART_RESERVED_SPACE"))?;
        let before=reserve.metadata()?;
        ensure!(before.is_file() && before.nlink()==1 && before.ino()==lease.reserved_inode && before.dev()==lease.reserved_device,
            "MD handoff scratch inode/device changed");
        ensure!(before.len()>=raw_retention_bound && before.len()<=lease.contract.maximum_retained_bytes,
            "remaining reserve cannot retain the complete admitted raw-consumer bound");
        let mut receipt=serde_json::json!({"schema":"prism.scientific_md_pathb_capacity_handoff.v1","status":"PREPARED",
            "native_pid":std::process::id(),"scientific_head":head,"identity":identity,"raw_head":raw_head,
            "capacity_contract":lease.contract,"terminal":terminal,"scientific_producers_closed":true,
            "raw_consumer_commits_authorized":true,"pathb_complete":false,"lease_closed":false,
            "lease_inode":lm.ino(),"lease_device":lm.dev(),"reserved_inode":before.ino(),"reserved_device":before.dev(),
            "before_reserved_bytes":before.len(),"before_reserved_blocks":before.blocks(),
            "retained_raw_store_bound_bytes":raw_retention_bound,"scientific_objects_and_journals_retained":true,
            "common_next_tick":publication.common_next_tick,"checkpoint_ordinal":publication.checkpoint_ordinal,
            "coupling_epoch":publication.coupling_epoch,"physical_owner_targets":publication.physical_owner_targets,
            "startup_physical_steps":publication.startup_physical_steps,"all_physical_targets_verified":true,
            "terminal_raw_control_suffix":raw_closure});
        write_new(&self.root.join("RESTART_MD_CAPACITY_HANDOFF_PREPARED.json"),&serde_json::to_vec(&receipt)?)?;sync_dir(&self.root)?;
        reserve.set_len(raw_retention_bound)?;reserve.sync_all()?;sync_dir(&self.root)?;
        let after=reserve.metadata()?;
        ensure!(after.len()==raw_retention_bound && after.ino()==before.ino() && after.dev()==before.dev()
            && after.blocks()<=before.blocks(),"MD handoff scratch result differs");
        receipt["status"]=serde_json::json!("SCIENTIFIC_CAPACITY_RELEASED_RAW_CONSUMERS_LIVE");
        receipt["after_reserved_bytes"]=serde_json::json!(after.len());receipt["after_reserved_blocks"]=serde_json::json!(after.blocks());
        receipt["released_allocated_bytes"]=serde_json::json!(before.blocks().checked_sub(after.blocks()).context("released block count underflow")?.checked_mul(512).context("released byte count overflow")?);
        write_new(&self.root.join("RESTART_MD_CAPACITY_HANDOFF.json"),&serde_json::to_vec(&receipt)?)?;sync_dir(&self.root)?;
        Ok(receipt)
    }
    /// Final zero-new-datum readouts publish real queue/drain metadata AFTER
    /// the final scientific anchor. Authenticate that suffix without rereading
    /// the complete event population or pretending the two raw heads equal.
    /// Caller holds the original writer LOCK.
    fn verify_terminal_raw_control_suffix(&self,saved:&Head,current:&Head,identity:&CheckpointIdentity,
        boundaries:&[StreamBoundary],terminal:&serde_json::Value)->Result<serde_json::Value> {
        let steps=current.generation.checked_sub(saved.generation).context("terminal raw HEAD predates SCI source")?;
        ensure!(steps==identity.stream_count as u64,"terminal raw ancestry must include exactly one final readout per owner");
        let anchor=self.read_foliage(saved)?;let final_index=self.read_foliage(current)?;
        ensure!(anchor.identity==*identity && final_index.identity==*identity
            && anchor.dictionary==final_index.dictionary && anchor.consumers==final_index.consumers
            && anchor.restart_anchor==final_index.restart_anchor
            && final_index.leaves.len()>=anchor.leaves.len()
            && final_index.leaves[..anchor.leaves.len()]==anchor.leaves,
            "terminal raw suffix changed saved data/dictionary/consumer authority");
        let mut cursor=current.clone();
        for _ in 0..steps {
            let index=self.read_foliage(&cursor)?;
            ensure!(index.identity==*identity,"terminal raw ancestry identity changed");
            cursor=index.previous_foliage.context("terminal raw ancestry disconnected")?;
        }
        ensure!(cursor==*saved,"terminal raw HEAD is not a descendant of the selected SCI raw source");
        for producer in &final_index.producers {
            if !matches!(producer.producer.field_id,3|4) {
                ensure!(anchor.producers.iter().any(|old|old==producer),"terminal readout grew raw/clock/derived population");
            }
        }
        for producer in &anchor.producers {
            if !matches!(producer.producer.field_id,3|4) {
                ensure!(final_index.producers.iter().any(|new|new==producer),"terminal readout lost a saved raw/clock population");
            }
        }
        let appended=&final_index.leaves[anchor.leaves.len()..];
        ensure!(appended.len()==identity.stream_count as usize*2,"terminal raw control population lacks exact final readouts");
        let queues=terminal["flow_summary"]["native_queue_prefixes"].as_array().context("terminal live queue state absent")?;
        let mut queue_words=std::collections::BTreeMap::new();let mut drain_words=std::collections::BTreeMap::new();
        for leaf in appended {
            let sid=leaf.producer.stream_id;
            let boundary=boundaries.iter().find(|b|b.stream_id==sid).context("terminal raw owner absent")?;
            let width=match leaf.producer.field_id{3=>112,4=>80,_=>bail!("terminal suffix contains new event/clock/selection datum")};
            ensure!(leaf.lineage.is_none() && leaf.byte_count==width
                && leaf.coverage.next_ordinal==leaf.coverage.first_ordinal.checked_add(1).context("terminal ordinal overflow")?
                && leaf.coverage.first_step==boundary.completed_step && leaf.coverage.last_step==boundary.completed_step
                && leaf.coverage.first_physical_time_ps_bits==boundary.physical_time_ps_bits
                && leaf.coverage.last_physical_time_ps_bits==boundary.physical_time_ps_bits,
                "terminal control coverage differs from actual final SCI physical boundary");
            let bytes=read_bounded(&self.root.join("objects").join(&leaf.object_sha256),width)?;
            ensure!(bytes.len() as u64==width && hash(&bytes)==leaf.object_sha256,"terminal control bytes/hash differ");
            let words=bytes.chunks_exact(8).map(|b|u64::from_le_bytes(b.try_into().expect("eight-byte word"))).collect::<Vec<_>>();
            let raw_next=anchor.producers.iter().find(|p|p.producer.stream_id==sid && p.producer.field_id==0
                && p.producer.replica_epoch==leaf.producer.replica_epoch).map_or(0,|p|p.next_ordinal);
            let clock=anchor.producers.iter().find(|p|p.producer.stream_id==sid && p.producer.field_id==1
                && p.producer.replica_epoch==leaf.producer.replica_epoch).context("terminal original clock population absent")?;
            ensure!(clock.next_ordinal==boundary.completed_step && clock.last_step==boundary.completed_step
                && clock.physical_time_ps_bits==boundary.physical_time_ps_bits,"terminal clock differs from SCI native frontier");
            let prior=anchor.producers.iter().find(|p|p.producer==leaf.producer).map_or(0,|p|p.next_ordinal);
            ensure!(leaf.coverage.first_ordinal==prior,"terminal control producer advanced more than once");
            if leaf.producer.field_id==4 {
                ensure!(words[0]==boundary.completed_step && words[1]==words[4] && words[2]>=words[4]
                    && words[3]==words[4] && words[5].checked_add(words[4])==Some(raw_next) && words[6]==raw_next
                    && words[8]==1 && words[9]==boundary.physical_time_ps_bits
                    && queues.get(sid as usize).and_then(|q|q["epoch"].as_u64())==words[7].checked_add(1),
                    "terminal queue witness is not the existing preserved prefix followed by actual reset ACK");
                ensure!(queue_words.insert(sid,words).is_none(),"duplicate terminal queue readout");
            }else{
                ensure!(words[0]==boundary.completed_step && words[2]>=words[1] && words[3]==0
                    && words[4]==raw_next && words[5]==raw_next && words[6]==clock.next_ordinal && words[7]==clock.next_ordinal
                    && words[8]==boundary.completed_step && words[9]==boundary.completed_step
                    && words[10]==boundary.physical_time_ps_bits && words[11]==boundary.physical_time_ps_bits
                    && words[12]==0 && words[13]==0,"terminal drain appended new event/clock bytes or changed physical chronology");
                ensure!(drain_words.insert(sid,words).is_none(),"duplicate terminal drain readout");
            }
        }
        ensure!(queue_words.len()==identity.stream_count as usize && drain_words.len()==identity.stream_count as usize,"terminal queue/drain owner populations lack all final owners");
        for (sid,queue) in &queue_words {
            let drain=drain_words.get(sid).context("terminal queue lacks its actual zero-new-datum drain")?;
            ensure!(queue[1]==drain[1] && queue[2]==drain[2],"terminal queue/drain attempted/capacity words differ");
        }
        Ok(serde_json::json!({"schema":"prism.final_md_raw_control_suffix.v1","scientific_raw_source":saved,"terminal_raw_head":current,
            "authenticated_ancestry_generations":steps,"preserved_original_leaf_count":anchor.leaves.len(),"appended_control_leaves":appended.len(),
            "raw_event_and_physical_clock_populations_unchanged":true,"zero_new_event_bytes":true,"queue_drain_owner_count":queue_words.len()}))
    }
    /// Consume the exact in-process capacity lease only after native producers
    /// and the final checkpoint epoch have departed. This returns scratch blocks,
    /// never CAS objects, journal prefixes, fields or observations. PREPARED is
    /// retained if any syscall fails; callers must not infer successful transfer.
    pub fn retire_reserved_capacity(&self,lease:ReservationLease,head:&Head,identity:&CheckpointIdentity,
        terminal:&serde_json::Value)->Result<serde_json::Value> {
        ensure!(lease.root==self.root && lease.contract.identity==*identity,"foreign reservation owner");
        let _lock=self.lock()?;
        ensure!(self.head()?.as_ref()==Some(head),"reservation handoff SCIHEAD changed");
        ensure!(self.read_manifest(head)?.identity==*identity,"reservation handoff identity changed");
        ensure!(terminal["all_worker_threads_joined"]==true && terminal["checkpoint_epoch_quiescent"]==true
            && matches!(terminal["boundary"].as_str(),Some("AUTHENTICATED_PLANNED_SCIENTIFIC_STOP"|"NORMAL_POST_PATHB_CONSUMERS_CLOSED")),"reservation handoff lacks native terminal closure");
        let lm=lease.file.metadata()?;let on_disk=fs::symlink_metadata(self.root.join("RESTART_RESERVATION_LEASE"))?;
        ensure!(lm.ino()==lease.lease_inode && lm.dev()==lease.lease_device && on_disk.is_file()
            && on_disk.ino()==lm.ino() && on_disk.dev()==lm.dev(),"reservation lease inode/device changed");
        let reserve=OpenOptions::new().read(true).write(true).custom_flags(libc::O_NOFOLLOW|libc::O_CLOEXEC).open(self.root.join("RESTART_RESERVED_SPACE"))?;
        let before=reserve.metadata()?;
        ensure!(before.is_file() && before.ino()==lease.reserved_inode && before.dev()==lease.reserved_device,"reserved scratch inode/device changed");
        let raw_head=self.chronicle_head()?;
        if terminal["boundary"]=="AUTHENTICATED_PLANNED_SCIENTIFIC_STOP" {
            ensure!(terminal["final_foliage_head"]==serde_json::to_value(&raw_head)?,"raw HEAD changed after final planned ACK");
        }else {
            ensure!(terminal["flow_summary"]["identity"]==serde_json::to_value(identity)?
                && terminal["flow_summary"]["foliage_head"]==serde_json::to_value(&raw_head)?
                && terminal["flow_summary"]["pending_consumer_ranges"].as_array().is_some_and(|rows|rows.is_empty()),"normal retirement lacks actual final direct consumer closure");
        }
        let md_handoff=if self.root.join("RESTART_MD_CAPACITY_HANDOFF_PREPARED.json").exists() {
            let value:serde_json::Value=serde_json::from_slice(&read_bounded(&self.root.join("RESTART_MD_CAPACITY_HANDOFF.json"),MAX_MANIFEST_BYTES)?)?;
            ensure!(value["status"]=="SCIENTIFIC_CAPACITY_RELEASED_RAW_CONSUMERS_LIVE"
                && value["scientific_head"]==serde_json::to_value(head)? && value["identity"]==serde_json::to_value(identity)?
                && value["reserved_inode"]==before.ino() && value["reserved_device"]==before.dev()
                && value["lease_closed"]==false,"final retirement MD handoff differs");Some(value)
        }else{None};
        let mut receipt=serde_json::json!({"schema":"prism.scientific_reserved_capacity_retirement.v1","status":"PREPARED",
            "native_pid":std::process::id(),"scientific_head":head,"identity":identity,"raw_head":raw_head,
            "reserved_inode":before.ino(),"reserved_device":before.dev(),"before_reserved_bytes":before.len(),"before_reserved_blocks":before.blocks(),
            "lease_inode":lm.ino(),"lease_device":lm.dev(),"lease_closed":false,"terminal":terminal,"md_pathb_capacity_handoff":md_handoff,
            "scientific_objects_and_journals_retained":true,"capacity_contract":lease.contract});
        write_new(&self.root.join("RESTART_RESERVATION_RETIREMENT_PREPARED.json"),&serde_json::to_vec(&receipt)?)?;sync_dir(&self.root)?;
        reserve.set_len(0)?;reserve.sync_all()?;let after=reserve.metadata()?;
        ensure!(after.len()==0 && after.blocks()==0,"filesystem did not release reserved scratch blocks");
        sync_dir(&self.root)?;
        ensure!(unsafe{libc::flock(lease.file.as_raw_fd(),libc::LOCK_UN)}==0,"capacity lease unlock failed");
        drop(lease); // descriptor closes before the completed receipt is published
        receipt["status"]=serde_json::json!("RETIRED_UNUSED_CAPACITY");receipt["lease_closed"]=serde_json::json!(true);
        receipt["after_reserved_bytes"]=serde_json::json!(0);receipt["after_reserved_blocks"]=serde_json::json!(0);
        write_new(&self.root.join("RESTART_RESERVATION_RETIREMENT.json"),&serde_json::to_vec(&receipt)?)?;sync_dir(&self.root)?;
        Ok(receipt)
    }
    /// Explicit all-worker error cleanup before the first durable datum. This
    /// cannot release capacity from a committed or partially populated store.
    /// No CAS, metadata, journal or observation file is removed or rewritten.
    pub fn retire_empty_failed_capacity(&self,lease:ReservationLease,identity:&CheckpointIdentity,
        terminal:&serde_json::Value)->Result<serde_json::Value> {
        ensure!(lease.root==self.root && lease.contract.identity==*identity,"foreign failed-store reservation owner");
        ensure!(terminal["all_worker_threads_joined"]==true
            && terminal["boundary"]=="WORKER_FAILURE_AFTER_ALL_WORKERS_JOINED_BEFORE_FIRST_DURABLE_COMMIT"
            && terminal["worker_errors"].as_array().is_some_and(|v|!v.is_empty()),"empty retirement requires actual joined worker errors");
        let _lock=self.lock()?;
        for entry in fs::read_dir(&self.root)? {
            let entry=entry?;let name=entry.file_name();let name=name.to_str().context("invalid empty-store entry")?;
            let m=fs::symlink_metadata(entry.path())?;
            match name {
                "LOCK"|"RESTART_RESERVED_SPACE"|"RESTART_RESERVATION.json"|"RESTART_RESERVATION_LEASE"=>
                    ensure!(m.is_file() && !m.file_type().is_symlink() && m.nlink()==1,"invalid empty-store control file"),
                "objects"|"manifests"|"encoded_objects"=>ensure!(m.is_dir() && !m.file_type().is_symlink()
                    && fs::read_dir(entry.path())?.next().is_none(),"failed store contains scientific objects"),
                _=>bail!("failed store has a durable head, journal or unclassified data; unused-only retirement refused: {}",name),
            }
        }
        let disk_contract:ReservationContract=serde_json::from_slice(&read_bounded(&self.root.join("RESTART_RESERVATION.json"),16384)?)?;
        ensure!(serde_json::to_vec(&disk_contract)?==serde_json::to_vec(&lease.contract)?,"failed-store reservation contract changed");
        let lm=lease.file.metadata()?;let on_disk=fs::symlink_metadata(self.root.join("RESTART_RESERVATION_LEASE"))?;
        ensure!(lm.ino()==lease.lease_inode && lm.dev()==lease.lease_device && on_disk.ino()==lm.ino()
            && on_disk.dev()==lm.dev() && on_disk.nlink()==1,"failed-store lease inode/device changed");
        let reserve=OpenOptions::new().read(true).write(true).custom_flags(libc::O_NOFOLLOW|libc::O_CLOEXEC)
            .open(self.root.join("RESTART_RESERVED_SPACE"))?;
        let before=reserve.metadata()?;
        ensure!(before.is_file() && before.nlink()==1 && before.ino()==lease.reserved_inode
            && before.dev()==lease.reserved_device && before.len()==lease.contract.maximum_retained_bytes,
            "failed-store scratch identity/capacity changed or was already spent");
        let mut receipt=serde_json::json!({"schema":"prism.scientific_empty_failed_capacity_retirement.v1","status":"PREPARED",
            "native_pid":std::process::id(),"identity":identity,"capacity_contract":lease.contract,"terminal":terminal,
            "reserved_inode":before.ino(),"reserved_device":before.dev(),"before_reserved_bytes":before.len(),"before_reserved_blocks":before.blocks(),
            "lease_inode":lm.ino(),"lease_device":lm.dev(),"lease_closed":false,
            "no_scientific_or_raw_head":true,"scientific_object_and_journal_population_empty":true,
            "uncommitted_execution_may_have_occurred":true,"restart_or_MD_success_claim":false});
        write_new(&self.root.join("RESTART_EMPTY_FAILURE_RETIREMENT_PREPARED.json"),&serde_json::to_vec(&receipt)?)?;sync_dir(&self.root)?;
        reserve.set_len(0)?;reserve.sync_all()?;let after=reserve.metadata()?;
        ensure!(after.len()==0 && after.blocks()==0,"filesystem did not release empty failed-store scratch");sync_dir(&self.root)?;
        ensure!(unsafe{libc::flock(lease.file.as_raw_fd(),libc::LOCK_UN)}==0,"failed-store lease unlock failed");drop(lease);
        receipt["status"]=serde_json::json!("RETIRED_EMPTY_FAILED_STORE_UNUSED_CAPACITY");receipt["lease_closed"]=serde_json::json!(true);
        receipt["after_reserved_bytes"]=serde_json::json!(0);receipt["after_reserved_blocks"]=serde_json::json!(0);
        write_new(&self.root.join("RESTART_EMPTY_FAILURE_RETIREMENT.json"),&serde_json::to_vec(&receipt)?)?;sync_dir(&self.root)?;Ok(receipt)
    }
    pub fn filesystem_available_bytes(&self)->Result<u64> {
        use std::os::unix::ffi::OsStrExt;
        let path=std::ffi::CString::new(self.root.as_os_str().as_bytes())?;
        let mut stat=std::mem::MaybeUninit::<libc::statvfs>::uninit();
        ensure!(unsafe{libc::statvfs(path.as_ptr(),stat.as_mut_ptr())}==0,"filesystem capacity query failed");
        let stat=unsafe{stat.assume_init()};
        (stat.f_bavail as u64).checked_mul(stat.f_frsize as u64).context("filesystem free-space size overflow")
    }
    fn retained_object_bytes(&self)->Result<u64> {
        let mut total=0u64;
        for area in ["objects","manifests","encoded_objects"] {
            for entry in fs::read_dir(self.root.join(area))? {
                let entry=entry?;let name=entry.file_name();let name=name.to_str().context("invalid CAS filename")?;
                let metadata=fs::symlink_metadata(entry.path())?;
                ensure!(metadata.is_file() && !metadata.file_type().is_symlink(),"CAS contains nonregular entry");
                // Private crash temporaries consume disk too; never omit them.
                ensure!((name.len()==64 && name.bytes().all(|b|b.is_ascii_hexdigit()))
                    || (name.starts_with('.') && name.ends_with(".next")) || (name.starts_with("decoded_") && name.ends_with(".json")),"unknown CAS entry");
                total=total.checked_add(metadata.len()).context("retained CAS byte count overflow")?;
            }
        }Ok(total)
    }
    /// Called under original writer LOCK immediately before a new CAS write.
    /// A crash after release but before writing is reconciled by the next
    /// reserve_retained_capacity call before execution, never silently credited.
    pub(super) fn spend_reserved_capacity(&self,bytes:u64)->Result<()> {
        let _reservation=self.reservation_updates.lock().map_err(|_|anyhow::anyhow!("scientific reservation updates poisoned"))?;
        ensure!(!self.root.join("RESTART_RESERVATION_RETIREMENT_PREPARED.json").exists()
            && !self.root.join("RESTART_EMPTY_FAILURE_RETIREMENT_PREPARED.json").exists(),"scientific store capacity is terminally retired or retirement requires reconciliation; no later CAS mutation admitted");
        ensure!(!self.root.join("RESTART_MD_CAPACITY_HANDOFF_PREPARED.json").exists()
            || self.root.join("RESTART_MD_CAPACITY_HANDOFF.json").exists(),"MD capacity handoff requires reconciliation before raw-consumer mutation");
        let contract_path=self.root.join("RESTART_RESERVATION.json");
        if !contract_path.exists(){return Ok(());}
        let contract:ReservationContract=serde_json::from_slice(&read_bounded(&contract_path,16384)?)?;
        let reserve=OpenOptions::new().read(true).write(true).custom_flags(libc::O_NOFOLLOW|libc::O_CLOEXEC)
            .open(self.root.join("RESTART_RESERVED_SPACE"))?;
        ensure!(reserve.metadata()?.is_file(),"invalid retained-capacity reservation");
        let remaining=reserve.metadata()?.len();
        ensure!(bytes<=remaining,"cumulative retained checkpoint byte reservation exhausted");
        ensure!(self.filesystem_available_bytes()? >=contract.remaining_free_floor_bytes,"filesystem reserve floor exhausted");
        reserve.set_len(remaining-bytes)?;reserve.sync_all()?;Ok(())
    }

    /// Return only bytes proved unused after a durable compressed object/index
    /// and temporary-file retirement. Reallocate real blocks, not a logical credit.
    pub(super) fn return_reserved_capacity(&self,bytes:u64)->Result<()> {
        let _reservation=self.reservation_updates.lock().map_err(|_|anyhow::anyhow!("scientific reservation updates poisoned"))?;
        if bytes==0 || !self.root.join("RESTART_RESERVATION.json").exists(){return Ok(());}
        let reserve=OpenOptions::new().read(true).write(true).custom_flags(libc::O_NOFOLLOW|libc::O_CLOEXEC).open(self.root.join("RESTART_RESERVED_SPACE"))?;
        let old=reserve.metadata()?.len();old.checked_add(bytes).context("reservation credit overflow")?;
        let rc=unsafe{libc::posix_fallocate(reserve.as_raw_fd(),libc::off_t::try_from(old)?,libc::off_t::try_from(bytes)?)};
        ensure!(rc==0,"cannot restore actual unused reserved blocks: {}",std::io::Error::from_raw_os_error(rc));
        reserve.sync_all()?;Ok(())
    }
    /// Each field object is written and synced while its bounded host buffer is
    /// live, then callers drop it. An opaque staged reference is not a commit.
    pub fn stage_field(&self,field:&CapturedField,max_field_bytes:u64)->Result<StagedField> {
        if field.contract.encoding==super::native_spike_archive::NATIVE_SPIKE_ARCHIVE_ENCODING {
            return self.stage_field_lossless(field,max_field_bytes);
        }
        field.contract.validate()?;
        let count=field.bytes.as_ref().map(|v|v.len() as u64).unwrap_or(0);
        ensure!(count<=max_field_bytes && count<=self.max_total_bytes,"staged field exceeds preadmitted extent");
        ensure!(field.bytes.is_some() || field.contract.may_be_absent,"required staged field absent");
        ensure!(field.contract.expected_bytes.is_none_or(|v|v==count) || field.bytes.is_none(),"staged field length differs from contract");
        let _lock=self.lock()?;self.ensure_scientific_producer_open()?;
        let digest=field.bytes.as_ref().map(|v|hash(v));
        if let (Some(digest),Some(bytes))=(&digest,&field.bytes){self.write_immutable("objects",digest,bytes)?;}
        sync_dir(&self.root.join("objects"))?;
        Ok(StagedField{store_root:self.root.clone(),stored:StoredField{contract:field.contract.clone(),
            present:field.bytes.is_some(),byte_count:count,object_sha256:digest,representation:None,host_allocation_budget:None}})
    }
    pub fn stage_field_lossless(&self,field:&CapturedField,max_field_bytes:u64)->Result<StagedField> {
        field.contract.validate()?;
        let count=field.bytes.as_ref().map_or(0,|v|v.len() as u64);
        ensure!(count<=max_field_bytes && count<=self.max_total_bytes,"lossless field exceeds preallocation admission");
        ensure!(field.bytes.is_some() || field.contract.may_be_absent,"required lossless field absent");
        ensure!(field.contract.expected_bytes.is_none_or(|v|v==count) || field.bytes.is_none(),"lossless field contract extent mismatch");
        let _lock=self.scientific_staging_guard()?;self.ensure_scientific_producer_open()?;
        let representation=field.bytes.as_ref().map(|v|self.encode_lossless_object(v)).transpose()?;
        let digest=representation.as_ref().map(|v|v.decoded_sha256.clone());
        Ok(StagedField{store_root:self.root.clone(),stored:StoredField{contract:field.contract.clone(),present:field.bytes.is_some(),byte_count:count,object_sha256:digest,representation,
            host_allocation_budget:if field.contract.encoding=="prism.typed-host-state.v1" {
                Some(count.checked_mul(64).context("host blob allocation budget overflow")?)
            }else if field.contract.encoding==super::native_spike_archive::NATIVE_SPIKE_ARCHIVE_ENCODING {
                ensure!(count%96==0,"native96 field contains an incomplete record");
                Some(count.checked_mul(4).and_then(|n|n.checked_add(128)).context("native vector allocation budget overflow")?)
            }else{None}}})
    }
    pub fn commit_staged(&self,expected:Option<&Head>,identity:&CheckpointIdentity,closure:&ClosureContract,
        boundaries:&[StreamBoundary],fields:&[StagedField])->Result<Head> {
        identity.validate()?;closure.validate()?;validate_boundaries(identity,boundaries)?;
        ensure!(fields.len()==closure.fields.len(),"staged field population mismatch");
        let mut by_key=BTreeMap::new();let mut total=0u64;
        for field in fields {
            ensure!(field.store_root==self.root,"staged field belongs to a different store");
            ensure!(by_key.insert(field.stored.contract.key(),&field.stored).is_none(),"duplicate staged field");
            total=total.checked_add(field.stored.byte_count).context("generation byte count overflow")?;
        }
        ensure!(total<=self.max_total_bytes,"staged generation exceeds admitted byte budget");
        let _lock=self.lock()?;self.ensure_scientific_producer_open()?;
        let current=self.head()?;ensure!(current.as_ref()==expected,"stale staged checkpoint HEAD");
        let mut stored=Vec::with_capacity(fields.len());
        for contract in &closure.fields {
            let value=*by_key.get(&contract.key()).context("missing staged field contract")?;
            ensure!(&value.contract==contract,"staged field contract changed");
            if let Some(digest)=&value.object_sha256 {
                ensure!(value.present,"absent staged object has digest");
                if let Some(envelope)=&value.representation {
                    ensure!(&envelope.decoded_sha256==digest && envelope.decoded_bytes==value.byte_count,"lossless staged field identity mismatch");
                    self.verify_lossless_object(envelope)?;
                }else{self.verify_object_streaming(digest,value.byte_count)?;}
            } else {ensure!(!value.present && value.byte_count==0 && contract.may_be_absent,"required staged field absent");}
            stored.push(value.clone());
        }
        let own_end=boundaries.first().is_some_and(|b|crate::native_run_continuation::own_end_phase(&b.next_integrator_phase));
        let publication=if own_end {Some(self.own_end_publication(identity,boundaries,&stored)?)}else{None};
        if let Some(value)=&publication {ensure!(value.raw_head==self.chronicle_head()?,"own-end publication raw HEAD differs from current durable foliage");}
        if let Some(head)=&current {
            let previous=self.read_manifest(head)?;
            ensure!(previous.identity==*identity && previous.closure_contract_sha256==closure.digest()?,"staged identity/closure changed");
            validate_boundaries(identity,&previous.boundaries)?;
            let prior_own_end=previous.boundaries.first().is_some_and(|b|crate::native_run_continuation::own_end_phase(&b.next_integrator_phase));
            ensure!(own_end==prior_own_end,"scientific publication cannot switch own-end/ordinary chronology");
            if let Some(value)=&publication {
                let prior=self.own_end_publication(identity,&previous.boundaries,&previous.fields)?;
                ensure!(value.profile_id==prior.profile_id && value.physical_owner_targets==prior.physical_owner_targets,
                    "own-end profile/physical target vector changed across capsules");
                let startup_transition=value.profile_id==crate::native_run_continuation::MAY_ORIGINAL_PROFILE_ID
                    && prior.common_next_tick==0 && value.common_next_tick==0
                    && prior.startup_physical_steps<value.startup_physical_steps && value.startup_physical_steps<=6
                    && prior.collective_logical_step==0 && value.collective_logical_step==0
                    && value.coupling_epoch==prior.coupling_epoch;
                let ordinary_next=value.collective_logical_step>prior.collective_logical_step
                    && (if value.common_next_tick==prior.common_next_tick {value.prefix_pending && value.coupling_epoch==prior.coupling_epoch}
                        else{value.common_next_tick>prior.common_next_tick && value.coupling_epoch>prior.coupling_epoch})
                    && (value.startup_physical_steps==prior.startup_physical_steps || (prior.common_next_tick==0 && prior.startup_physical_steps<6 && value.startup_physical_steps==6));
                ensure!((ordinary_next || startup_transition) && value.checkpoint_ordinal==prior.checkpoint_ordinal.checked_add(1).context("own-end capsule ordinal overflow")?,
                    "own-end collective/startup transaction or checkpoint ordinal did not advance exactly");
            }
            for (old,new) in previous.boundaries.iter().zip(boundaries) {
                ensure!(old.stream_id==new.stream_id && old.replica_epoch==new.replica_epoch && old.replica_seed==new.replica_seed
                    && new.coupling_epoch>=old.coupling_epoch && new.committed_output_sequence>=old.committed_output_sequence,"staged scientific owner/cursor chronology regressed");
                if own_end {
                    ensure!(new.completed_step>=old.completed_step && f64::from_bits(new.physical_time_ps_bits)>=f64::from_bits(old.physical_time_ps_bits),"own-end physical chronology regressed");
                    if new.completed_step==old.completed_step {
                        ensure!(new.completed_step==*publication.as_ref().context("own-end publication absent")?.physical_owner_targets.get(new.stream_id as usize).context("own-end physical target missing")?
                            && new.physical_time_ps_bits==old.physical_time_ps_bits && new.coupling_parity==old.coupling_parity,
                            "own-end plateau is not an exact unchanged retired-owner frontier");
                    }else{ensure!(f64::from_bits(new.physical_time_ps_bits)>f64::from_bits(old.physical_time_ps_bits),"own-end active physical clock did not advance");}
                }else{ensure!(new.completed_step>old.completed_step && f64::from_bits(new.physical_time_ps_bits)>f64::from_bits(old.physical_time_ps_bits),"ordinary staged scientific chronology did not strictly advance");}
            }
        }else if let Some(value)=&publication {
            ensure!(value.common_next_tick==0 && value.checkpoint_ordinal==0 && value.coupling_epoch==0 && value.startup_physical_steps==0 && boundaries.iter().all(|b|b.completed_step==0 && b.physical_time_ps_bits==0f64.to_bits()),"fresh own-end run lacks its zero-state collective anchor");
        }
        let generation=current.as_ref().map_or(0,|h|h.generation).checked_add(1).context("generation overflow")?;
        let manifest=Manifest{schema_version:SCHEMA,generation,previous_head:current,identity:identity.clone(),
            closure_contract_sha256:closure.digest()?,boundaries:boundaries.to_vec(),fields:stored};
        let bytes=serde_json::to_vec(&manifest)?;ensure!(bytes.len() as u64<=MAX_MANIFEST_BYTES,"staged manifest exceeds limit");
        let digest=hash(&bytes);self.write_immutable("manifests",&digest,&bytes)?;sync_dir(&self.root.join("manifests"))?;
        let head=Head{generation,manifest_sha256:digest};self.publish_head_named("HEAD",&head,&mut |_|Ok(()))?;Ok(head)
    }
    fn own_end_publication(&self,identity:&CheckpointIdentity,boundaries:&[StreamBoundary],fields:&[StoredField])->Result<crate::native_run_continuation::OwnEndCollectivePublicationV1> {
        let read=|owner:&str|->Result<&StoredField>{fields.iter().find(|f|f.contract.owner==owner && f.contract.name=="native_owner_closure").context("own-end publication lacks its actual native host closure")};
        crate::native_run_continuation::validate_own_end_publication(identity,boundaries,
            |stream|{let field=read(&format!("stream_{stream}"))?;self.read_stored_host(field,field.byte_count)},
            ||{let field=read("shared")?;self.read_stored_host(field,field.byte_count)})
    }
    pub(super) fn verify_object_streaming(&self,digest:&str,expected:u64)->Result<()> {
        validate_digest(digest)?;
        let mut file=OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW|libc::O_CLOEXEC)
            .open(self.root.join("objects").join(digest))?;
        ensure!(file.metadata()?.is_file() && file.metadata()?.len()==expected,"CAS object extent/type changed");
        let mut hasher=Sha256::new();let mut buffer=[0u8;65536];let mut count=0u64;
        loop {let n=file.read(&mut buffer)?;if n==0{break;}hasher.update(&buffer[..n]);count=count.checked_add(n as u64).context("object length overflow")?;ensure!(count<=expected,"CAS object grew");}
        ensure!(count==expected && format!("{:x}",hasher.finalize())==digest,"CAS object content changed");Ok(())
    }
}

/// Hash-verified immutable generation metadata. Payloads remain individually
/// addressable and are checked/decoded under a caller's exact field budget.
pub struct ValidatedGeneration {root:PathBuf,head:Head,identity:CheckpointIdentity,closure:ClosureContract,boundaries:Vec<StreamBoundary>,fields:Vec<StoredField>,total_bytes:u64}
impl ValidatedGeneration {
    pub fn head(&self)->&Head{&self.head}
    pub fn identity(&self)->&CheckpointIdentity{&self.identity}
    pub fn closure(&self)->&ClosureContract{&self.closure}
    pub fn boundaries(&self)->&[StreamBoundary]{&self.boundaries}
    pub fn total_decoded_bytes(&self)->u64{self.total_bytes}
}
impl CheckpointStore {
    pub fn validate_generation(&self,expected:&Head,identity:&CheckpointIdentity)->Result<ValidatedGeneration> {
        let _lock=self.lock()?;
        ensure!(self.head()?.as_ref()==Some(expected),"stale scientific generation admission");
        self.validate_generation_metadata(expected,identity)
    }
    fn validate_generation_metadata(&self,head:&Head,identity:&CheckpointIdentity)->Result<ValidatedGeneration> {
        identity.validate()?;let manifest=self.read_manifest(head)?;
        ensure!(manifest.identity==*identity,"scientific generation provider/execution identity differs");
        validate_boundaries(identity,&manifest.boundaries)?;
        let fields=manifest.fields.iter().map(|f|f.contract.clone()).collect::<Vec<_>>();
        let closure=ClosureContract{schema_version:SCHEMA,required_owners:fields.iter().map(|f|f.owner.clone()).collect(),fields,unresolved_owners:BTreeSet::new()};
        ensure!(closure.digest()?==manifest.closure_contract_sha256,"scientific generation closure digest differs");
        let mut total=0u64;
        for field in &manifest.fields {
            ensure!(field.present || (field.contract.may_be_absent && field.byte_count==0 && field.object_sha256.is_none() && field.representation.is_none()),"invalid optional scientific field representation");
            ensure!(field.contract.expected_bytes.is_none_or(|n|n==field.byte_count) || !field.present,"scientific generation extent differs from field contract");
            if field.present {validate_digest(field.object_sha256.as_ref().context("present scientific field lacks content hash")?)?;}
            if let Some(envelope)=&field.representation {ensure!(field.present && Some(&envelope.decoded_sha256)==field.object_sha256.as_ref()
                && envelope.decoded_bytes==field.byte_count,"scientific lossless envelope differs from field identity");}
            total=total.checked_add(field.byte_count).context("scientific generation size overflow")?;
        }
        ensure!(total<=self.max_total_bytes,"scientific generation exceeds preallocation admission");
        Ok(ValidatedGeneration{root:self.root.clone(),head:head.clone(),identity:identity.clone(),closure,
            boundaries:manifest.boundaries,fields:manifest.fields,total_bytes:total})
    }
    pub fn read_generation_field(&self,index:&ValidatedGeneration,owner:&str,name:&str,maximum:u64)->Result<CapturedField> {
        ensure!(index.root==self.root && index.total_bytes<=self.max_total_bytes,"scientific generation belongs to another store/budget");
        let field=index.fields.iter().find(|f|f.contract.owner==owner && f.contract.name==name).context("scientific generation field absent")?;
        ensure!(field.byte_count<=maximum,"scientific field exceeds preallocation memory grant");
        let bytes=if !field.present {None} else if let Some(envelope)=&field.representation {Some(self.read_lossless_object(envelope,maximum)?)} else {
            let hash_expected=field.object_sha256.as_ref().context("scientific field lacks hash")?;
            let bytes=read_bounded(&self.root.join("objects").join(hash_expected),field.byte_count)?;
            ensure!(bytes.len() as u64==field.byte_count && hash(&bytes)==*hash_expected,"scientific field content/size differs");Some(bytes)
        };
        Ok(CapturedField{contract:field.contract.clone(),bytes})
    }
    /// Explicit fork: source HEAD never moves backwards. Only a verified ancestor
    /// is selected, in a fresh destination namespace with immutable object links.
    /// Output files are separately copied by exact committed prefix; never link
    /// mutable output writers across branches.
    pub fn fork_scientific_history(&self,destination:&Path,expected_current:&Head,anchor:&Head,
        identity:&CheckpointIdentity)->Result<Head> {
        ensure!(destination!=self.root && !destination.starts_with(&self.root) && !self.root.starts_with(destination),"scientific branch namespaces overlap");
        let target=CheckpointStore::create(destination,self.max_total_bytes)?;
        let _source_lock=self.lock()?;
        let target_lock_file=OpenOptions::new().read(true).write(true).create(true).truncate(false).custom_flags(libc::O_NOFOLLOW|libc::O_CLOEXEC).open(target.root.join("LOCK"))?;
        ensure!(unsafe{libc::flock(target_lock_file.as_raw_fd(),libc::LOCK_EX|libc::LOCK_NB)}==0,"scientific branch destination is concurrently owned");
        let _target_lock=WriterLock(target_lock_file);
        ensure!(self.head()?.as_ref()==Some(expected_current),"scientific branch source changed");
        if let Some(existing)=target.head()? {
            ensure!(&existing==anchor,"scientific branch destination already contains another generation");
            let expected=serde_json::json!({"schema_version":1,"kind":"explicit_historical_scientific_branch","source_root":self.root,"source_current_head":expected_current,"selected_anchor":anchor,"identity":identity,"source_head_unchanged":true});
            let actual:serde_json::Value=serde_json::from_slice(&read_bounded(&target.root.join("SCIENTIFIC_BRANCH_ORIGIN.json"),65536)?)?;
            ensure!(actual==expected,"existing scientific branch origin differs from exact ingress contract");
            target.validate_generation_metadata(anchor,identity)?;return Ok(anchor.clone());
        }
        let mut cursor=Some(expected_current.clone());let mut found=false;let mut inspected=0usize;
        while let Some(head)=cursor {
            inspected+=1;ensure!(inspected<=100000,"scientific ancestry exceeds admitted bound");
            let manifest=self.read_manifest(&head)?;ensure!(manifest.identity==*identity,"scientific ancestor identity differs");
            if head==*anchor {found=true;break;}cursor=manifest.previous_head;
        }
        ensure!(found,"requested scientific anchor is not a verified ancestor of current HEAD");
        let mut cursor=Some(anchor.clone());let mut copied=0usize;
        while let Some(head)=cursor {
            copied+=1;ensure!(copied<=100000,"scientific branch ancestry exceeds admitted bound");
            let generation=self.validate_generation_metadata(&head,identity)?;
            for field in &generation.fields {
                if let Some(envelope)=&field.representation {
                    self.verify_lossless_object(envelope)?;
                    self.link_lossless_closure_to(&target,envelope)?;
                    // The decoded-hash lookup is a derived cache. Recreate it from the
                    // verified manifest envelope, never trust an absent/stale source cache.

                }else if let Some(digest)=&field.object_sha256 {self.verify_object_streaming(digest,field.byte_count)?;self.link_immutable_to(&target,"objects",digest)?;}
            }
            self.link_immutable_to(&target,"manifests",&head.manifest_sha256)?;
            let previous=self.read_manifest(&head)?.previous_head;
            ensure!(previous.as_ref().is_none_or(|prior|prior.generation<head.generation),"scientific ancestry does not strictly decrease");
            cursor=previous;
        }
        for area in ["objects","manifests","encoded_objects"] {sync_dir(&target.root.join(area))?;}
        let origin=serde_json::json!({"schema_version":1,"kind":"explicit_historical_scientific_branch","source_root":self.root,"source_current_head":expected_current,"selected_anchor":anchor,"identity":identity,"source_head_unchanged":true});
        publish_branch_metadata(&target.root,Path::new("SCIENTIFIC_BRANCH_ORIGIN.json"),&serde_json::to_vec(&origin)?)?;
        target.publish_head_named("HEAD",anchor,&mut |_|Ok(()))?;Ok(anchor.clone())
    }
    pub(super) fn link_immutable_to(&self,target:&CheckpointStore,area:&str,name:&str)->Result<()> {
        ensure!(matches!(area,"objects"|"manifests"|"encoded_objects"),"invalid immutable namespace");
        ensure!(!name.contains('/') && !name.contains(".."),"invalid immutable locator");
        let from=self.root.join(area).join(name);let to=target.root.join(area).join(name);
        let metadata=fs::symlink_metadata(&from)?;ensure!(metadata.is_file()&&!metadata.file_type().is_symlink(),"immutable branch source is not regular");
        if to.exists() {
            use std::os::unix::fs::MetadataExt;
            let existing=fs::symlink_metadata(&to)?;
            ensure!(existing.is_file() && !existing.file_type().is_symlink() && existing.dev()==metadata.dev() && existing.ino()==metadata.ino(),"branch immutable destination differs");
        }else{spend_branch_copy_capacity(&target.root,0)?;fs::hard_link(from,to).context("immutable branch requires same-filesystem hard links; copy fallback is not implicit")?;}
        Ok(())
    }
}
impl CheckpointStore {
    pub fn stage_field_streaming(&self,contract:&FieldContract,present_bytes:Option<u64>,host_allocation_budget:Option<u64>,
        mut produce:impl FnMut(&mut dyn Write)->Result<()>)->Result<StagedField> {
        let stage_start=std::time::Instant::now();
        contract.validate()?;
        ensure!((contract.encoding=="prism.typed-host-state.v1" || contract.encoding==super::native_spike_archive::NATIVE_SPIKE_ARCHIVE_ENCODING)==host_allocation_budget.is_some(),"typed host field allocation plan missing/foreign");
        ensure!(present_bytes.is_some() || contract.may_be_absent,"required streamed scientific field absent");
        let count=present_bytes.unwrap_or(0);ensure!(count<=self.max_total_bytes,"streamed scientific field exceeds logical generation admission");
        ensure!(contract.expected_bytes.is_none_or(|n|n==count) || present_bytes.is_none(),"streamed field allocation differs from contract");
        let _lock=self.scientific_staging_guard()?;self.ensure_scientific_producer_open()?;
        let writer_acquired=stage_start.elapsed().as_nanos();
        let representation=if present_bytes.is_some(){Some(self.encode_lossless_stream(count,&mut produce)?)}else{None};
        log::info!("SCIENTIFIC_FIELD_STAGE owner={} field={} decoded_bytes={} present={} writer_wait_and_validation_ns={} encode_verify_publish_ns={} total_ns={}",contract.owner,contract.name,count,present_bytes.is_some(),writer_acquired,stage_start.elapsed().as_nanos().saturating_sub(writer_acquired),stage_start.elapsed().as_nanos());
        let digest=representation.as_ref().map(|v|v.decoded_sha256.clone());
        Ok(StagedField{store_root:self.root.clone(),stored:StoredField{contract:contract.clone(),present:present_bytes.is_some(),byte_count:count,object_sha256:digest,representation,host_allocation_budget}})
    }
}

impl ValidatedGeneration {
    pub(super) fn belongs_to_store(&self,store:&CheckpointStore)->bool {self.root==store.root && self.total_bytes<=store.max_total_bytes}
    pub(super) fn native_archive_field(&self,owner:&str,name:&str)->Result<&StoredField> {
        self.fields.iter().find(|field|field.contract.owner==owner&&field.contract.name==name).context("native archived host field absent")
    }
}
pub struct ScientificGenerationReader<'a> {pub store:&'a CheckpointStore,pub index:&'a ValidatedGeneration,pub maximum_device_field_bytes:u64,pub maximum_owned_fragment_bytes:u64}
impl ScientificGenerationReader<'_> {
    pub fn field(&self,owner:&str,name:&str)->Result<CapturedField>{self.store.read_generation_field(self.index,owner,name,self.maximum_device_field_bytes)}
    pub fn host<T:serde::de::DeserializeOwned>(&self,owner:&str,name:&str)->Result<T>{self.store.read_generation_host(self.index,owner,name,self.maximum_owned_fragment_bytes)}
}
struct ScientificHashReader<R:Read>{inner:R,hash:Sha256,count:u64,expected:u64}
impl<R:Read> Read for ScientificHashReader<R> {
    fn read(&mut self,bytes:&mut[u8])->std::io::Result<usize>{let n=self.inner.read(bytes)?;
        let next=self.count.checked_add(n as u64).ok_or_else(||std::io::Error::new(std::io::ErrorKind::InvalidData,"scientific read count overflow"))?;
        if next>self.expected{return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,"scientific decoded reader exceeded exact field extent"));}
        self.hash.update(&bytes[..n]);self.count=next;Ok(n)}
}
impl CheckpointStore {
    pub fn admit_generation_host_capacity(&self,index:&ValidatedGeneration,free_floor:u64)->Result<serde_json::Value> {
        ensure!(index.root==self.root && index.total_bytes<=self.max_total_bytes,"host admission generation/store budget differs");
        let mut host_budget=0u64;let mut largest_device=0u64;
        for field in &index.fields {
            if field.contract.encoding=="prism.typed-host-state.v1" || field.host_allocation_budget.is_some() {
                let budget=field.host_allocation_budget.unwrap_or(field.byte_count.checked_mul(64).context("legacy host bound overflow")?);
                host_budget=host_budget.checked_add(budget).context("host population budget overflow")?;
            }else{largest_device=largest_device.max(field.byte_count);}
        }
        let transfer_budget=largest_device.checked_mul(2).context("restore transfer budget overflow")?;
        let required=host_budget.checked_add(transfer_budget).and_then(|n|n.checked_add(128*1024*1024)).context("restore host capacity overflow")?;
        let meminfo=std::fs::read_to_string("/proc/meminfo")?;
        let available=meminfo.lines().find_map(|line|line.strip_prefix("MemAvailable:").and_then(|v|v.split_whitespace().next()).and_then(|v|v.parse::<u64>().ok())).context("host MemAvailable unavailable")?.checked_mul(1024).context("host free byte overflow")?;
        let mut admitted_available=available;
        let cgroup=std::fs::read_to_string("/proc/self/cgroup")?;
        let relative=cgroup.lines().find_map(|line|line.strip_prefix("0::")).context("cgroup v2 host capacity unavailable")?;
        ensure!(relative.starts_with('/') && !Path::new(relative).components().any(|c|matches!(c,std::path::Component::ParentDir)),"invalid cgroup resource path");
        let mut current=PathBuf::from("/sys/fs/cgroup").join(relative.trim_start_matches('/'));
        let mut cgroup_limits=Vec::new();
        loop {
            // Kernel cgroup-v2 memory interfaces are non-root files. The
            // global root is bounded by MemAvailable above, not a fabricated
            // memory.max. A namespace root exposing a real limit still has
            // that limit enforced below. Missing non-root files remain fatal.
            if current==Path::new("/sys/fs/cgroup") && !current.join("memory.max").try_exists()? {
                ensure!(current.is_dir() && current.join("cgroup.controllers").is_file(),"cgroup v2 controller root unavailable");
                break;
            }
            let maximum=std::fs::read_to_string(current.join("memory.max"))
                .with_context(||format!("read cgroup memory limit {}",current.display()))?;
            if maximum.trim()!="max" {
                let maximum:u64=maximum.trim().parse()?;let used:u64=std::fs::read_to_string(current.join("memory.current"))?.trim().parse()?;
                let remaining=maximum.checked_sub(used).context("existing cgroup already exceeds host cap")?;
                admitted_available=admitted_available.min(remaining);
                cgroup_limits.push(serde_json::json!({"path":current,"maximum":maximum,"current":used,"remaining":remaining}));
            }
            if current==Path::new("/sys/fs/cgroup"){break;}
            ensure!(current.pop() && current.starts_with("/sys/fs/cgroup"),"cgroup admission escaped root");
        }
        ensure!(required.checked_add(free_floor).is_some_and(|n|n<=admitted_available),"full scientific host population/restore workspace exceeds preallocation capacity: need {} + floor {}, available {}",required,free_floor,admitted_available);
        Ok(serde_json::json!({"schema":"prism.scientific_restart_host_capacity.v1","head":index.head,"host_typed_allocation_budget":host_budget,
            "serial_device_transfer_budget":transfer_budget,"codec_and_metadata_workspace":128*1024*1024u64,"required_bytes":required,"free_floor_bytes":free_floor,
            "mem_available_bytes":available,"admitted_available_bytes":admitted_available,"cgroup_limits":cgroup_limits,
            "checked_before_host_owner_decode_and_cuda_construction":true,"os_exclusive_memory_reservation":false,
            "allocation_policy":"typed sequence/map/string charges before allocation; size hints suppressed; exact registered population plus bounded transfer workspace"}))
    }
    pub fn read_generation_host<T:serde::de::DeserializeOwned>(&self,index:&ValidatedGeneration,owner:&str,name:&str,max_fragment:u64)->Result<T> {
        ensure!(index.root==self.root && index.total_bytes<=self.max_total_bytes,"scientific reader store/budget differs");
        let field=index.fields.iter().find(|f|f.contract.owner==owner && f.contract.name==name).context("host scientific field missing")?;
        self.read_stored_host(field,max_fragment)
    }
    fn read_stored_host<T:serde::de::DeserializeOwned>(&self,field:&StoredField,max_fragment:u64)->Result<T> {
        ensure!(field.present && field.contract.encoding=="prism.typed-host-state.v1","host scientific field type/presence differs");
        let expected_hash=field.object_sha256.as_ref().context("host scientific field has no hash")?;
        let input:Box<dyn Read>=if let Some(envelope)=&field.representation {
            ensure!(envelope.decoded_bytes==field.byte_count && &envelope.decoded_sha256==expected_hash,"host envelope identity differs");
            self.verify_lossless_object(envelope)?;
            self.open_lossless_reader(envelope)?
        }else{
            self.verify_object_streaming(expected_hash,field.byte_count)?;
            Box::new(OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW|libc::O_CLOEXEC).open(self.root.join("objects").join(expected_hash))?)
        };
        let mut reader=ScientificHashReader{inner:input,hash:Sha256::new(),count:0,expected:field.byte_count};
        let heap_budget=field.host_allocation_budget.unwrap_or(field.byte_count.checked_mul(64).context("legacy host allocation budget overflow")?);
        let value=crate::checkpoint_codec::from_reader_with_allocation_budget(&mut reader,field.byte_count,max_fragment,heap_budget)
            .with_context(||format!("scientific host field {}/{}, type {}, wire_bytes {}, planned_budget {}, max_fragment {}",
                field.contract.owner,field.contract.name,std::any::type_name::<T>(),field.byte_count,heap_budget,max_fragment))?;
        ensure!(reader.count==field.byte_count && format!("{:x}",reader.hash.finalize())==*expected_hash,"restored host field content differs");
        Ok(value)
    }
}
impl CheckpointStore {
    pub(super) fn branch_writer_lock(&self)->Result<WriterLock> {
        let file=OpenOptions::new().read(true).write(true).create(true).truncate(false).custom_flags(libc::O_NOFOLLOW|libc::O_CLOEXEC).open(self.root.join("LOCK"))?;
        ensure!(file.metadata()?.is_file(),"invalid branch LOCK");
        ensure!(unsafe{libc::flock(file.as_raw_fd(),libc::LOCK_EX|libc::LOCK_NB)}==0,"branch destination concurrently owned");Ok(WriterLock(file))
    }
}

/// Copy the exact committed mutable-output prefix into a separate branch. Never
/// hard-link an append writer: branch truncation/appends cannot mutate source.
pub fn copy_committed_append_prefix(source_root:&Path,target_root:&Path,cursor:&AppendCursor)->Result<()> {
    copy_committed_append_prefix_with_header(source_root,target_root,cursor,None)
}
/// Existing V2 owner closes its eight-byte count header after the anchor. Bind
/// the saved header plus exact original body bytes without mutating the source.
pub fn copy_committed_append_prefix_with_header(source_root:&Path,target_root:&Path,cursor:&AppendCursor,header:Option<&[u8]>)->Result<()> {
    use std::os::unix::fs::FileExt;
    validate_label(&cursor.relative_path)?;validate_digest(&cursor.committed_sha256)?;
    verify_directory_chain(source_root)?;verify_directory_chain(target_root)?;
    ensure!(source_root!=target_root && !target_root.starts_with(source_root) && !source_root.starts_with(target_root),"mutable output branch namespaces overlap");
    let source=source_root.join(&cursor.relative_path);verify_directory_chain(source.parent().context("source output parent missing")?)?;
    let file=OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW|libc::O_CLOEXEC).open(source)?;
    ensure!(file.metadata()?.is_file() && file.metadata()?.len()>=cursor.committed_bytes,"committed output source truncated/type changed");
    let normalized_hash=if let Some(header)=header {
        ensure!(header.len()<=4096 && header.len() as u64<=cursor.committed_bytes,"unsupported committed header extent");
        let mut hash=Sha256::new();hash.update(header);let mut offset=header.len() as u64;let mut slab=[0u8;65536];
        while offset<cursor.committed_bytes {let n=file.read_at(&mut slab[..usize::try_from((cursor.committed_bytes-offset).min(65536))?],offset)?;
            ensure!(n>0,"historical mutable-header body truncated");hash.update(&slab[..n]);offset+=n as u64;}
        format!("{:x}",hash.finalize())
    }else{hash_file_prefix(&file,cursor.committed_bytes)?};
    ensure!(normalized_hash==cursor.committed_sha256,"source output committed prefix hash differs");
    let mut parent=target_root.to_path_buf();
    let relative=Path::new(&cursor.relative_path);
    for component in relative.parent().context("target output relative parent missing")?.components() {
        ensure!(matches!(component,std::path::Component::Normal(_)),"invalid target output parent");
        parent.push(component.as_os_str());
        match fs::create_dir(&parent){Ok(())=>{sync_dir(parent.parent().context("created output parent missing")?)?;},Err(e)if e.kind()==std::io::ErrorKind::AlreadyExists=>{},Err(e)=>return Err(e.into())}
        verify_directory_chain(&parent)?;
    }
    let destination=target_root.join(relative);
    if destination.exists(){let existing=OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW|libc::O_CLOEXEC).open(&destination)?;
        ensure!(existing.metadata()?.is_file() && existing.metadata()?.len()==cursor.committed_bytes && hash_file_prefix(&existing,cursor.committed_bytes)?==cursor.committed_sha256,"existing branch output differs");return Ok(());}
    let temporary=destination.with_file_name(format!(".{}.{}.branch.next",destination.file_name().context("branch filename missing")?.to_string_lossy(),cursor.committed_sha256));
    if temporary.exists(){let m=fs::symlink_metadata(&temporary)?;ensure!(m.is_file()&&!m.file_type().is_symlink(),"invalid output branch temporary");fs::remove_file(&temporary)?;}
    spend_branch_copy_capacity(target_root,cursor.committed_bytes)?;
    let mut output=OpenOptions::new().write(true).create_new(true).mode(0o600).custom_flags(libc::O_NOFOLLOW|libc::O_CLOEXEC).open(&temporary)?;
    let mut offset=0u64;
    if let Some(header)=header {output.write_all(header)?;offset=header.len() as u64;}
    let mut slab=[0u8;65536];
    while offset<cursor.committed_bytes {let n=file.read_at(&mut slab[..usize::try_from((cursor.committed_bytes-offset).min(65536))?],offset)?;
        ensure!(n>0,"branch output copy was incomplete");output.write_all(&slab[..n])?;offset+=n as u64;}
    ensure!(offset==cursor.committed_bytes,"branch output copy extent differs");output.sync_all()?;drop(output);
    let check=OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW|libc::O_CLOEXEC).open(&temporary)?;
    ensure!(hash_file_prefix(&check,cursor.committed_bytes)?==cursor.committed_sha256,"branch output copy content mismatch");drop(check);
    // A concurrently created destination is never replaced.
    fs::hard_link(&temporary,&destination)?;sync_dir(destination.parent().context("branch output parent missing")?)?;
    fs::remove_file(&temporary)?;sync_dir(destination.parent().context("branch output parent missing")?)?;Ok(())
}
impl CheckpointStore {
    pub fn select_historical_anchor(&self,expected_current:&Head,identity:&CheckpointIdentity,requested_step:u64)->Result<Head> {
        let _lock=self.lock()?;ensure!(self.head()?.as_ref()==Some(expected_current),"scientific source HEAD changed during anchor selection");
        let mut cursor=Some(expected_current.clone());let mut count=0usize;
        while let Some(head)=cursor {count+=1;ensure!(count<=100000,"scientific anchor search exceeds ancestry budget");
            let generation=self.validate_generation_metadata(&head,identity)?;
            if generation.boundaries.iter().all(|b|b.completed_step<=requested_step){return Ok(head);}
            let previous=self.read_manifest(&head)?.previous_head;
            if let Some(prior)=&previous{ensure!(prior.generation<head.generation,"scientific ancestry does not decrease");}
            cursor=previous;
        }anyhow::bail!("no complete scientific anchor at/before requested step; an initial state capsule is required")
    }
    pub fn select_historical_own_end_anchor(&self,expected_current:&Head,identity:&CheckpointIdentity,next_tick:u32)->Result<Head> {
        ensure!(next_tick<=80 && identity.stream_count==8,"own-end historical tick/population differs");
        let _lock=self.lock()?;ensure!(self.head()?.as_ref()==Some(expected_current),"scientific source HEAD changed during own-end anchor selection");
        let mut cursor=Some(expected_current.clone());let mut count=0usize;
        while let Some(head)=cursor {count+=1;ensure!(count<=100000,"own-end anchor search exceeds ancestry budget");
            let generation=self.validate_generation_metadata(&head,identity)?;
            if generation.boundaries.iter().all(|boundary|boundary.next_integrator_phase==crate::native_run_continuation::OWN_END_RESTART_PHASE) {
                let manifest=self.read_manifest(&head)?;
                let field=manifest.fields.iter().find(|f|f.contract.owner=="stream_0" && f.contract.name=="native_owner_closure").context("own-end historical worker0 missing")?;
                let worker:crate::native_run_continuation::NativeWorkerContinuationV1=self.read_stored_host(field,field.byte_count)?;
                if worker.next_chunk==i32::try_from(next_tick)? {
                    let joined=self.own_end_publication(identity,&manifest.boundaries,&manifest.fields)?;
                    ensure!(joined.common_next_tick==worker.next_chunk,"historical own-end collective projection differs");return Ok(head);
                }
            }
            let previous=self.read_manifest(&head)?.previous_head;
            if let Some(prior)=&previous{ensure!(prior.generation<head.generation,"own-end scientific ancestry does not decrease");}cursor=previous;
        }anyhow::bail!("no scientific capsule at the exact requested completed collective tick; own-end body-slot replay is not admitted")
    }
    /// Startup physical1..5 is distinct from scheduled logicalL+6. Select
    /// only a complete all8 startup anchor at/before K, including initial0;
    /// the original direct body executor performs any required real replay.
    pub fn select_historical_may_startup_anchor(&self,expected_current:&Head,identity:&CheckpointIdentity,physical:u32)->Result<Head>{
        ensure!(identity.stream_count==8 && (1..6).contains(&physical),"startup historical selector must be physical1..5/all8");
        let _lock=self.lock()?;ensure!(self.head()?.as_ref()==Some(expected_current),"startup source SCIHEAD changed");
        let mut cursor=Some(expected_current.clone());let mut count=0usize;
        while let Some(head)=cursor {count+=1;ensure!(count<=100000,"startup ancestry budget exceeded");
            let manifest=self.read_manifest(&head)?;ensure!(manifest.identity==*identity,"startup source identity changed");
            if manifest.boundaries.iter().all(|b|crate::native_run_continuation::own_end_phase(&b.next_integrator_phase)) {
                let joined=self.own_end_publication(identity,&manifest.boundaries,&manifest.fields)?;
                if joined.profile_id==crate::native_run_continuation::MAY_ORIGINAL_PROFILE_ID
                    && joined.common_next_tick==0 && joined.collective_logical_step==0
                    && joined.startup_physical_steps<=u64::from(physical)
                    && manifest.boundaries.iter().all(|b|b.completed_step==joined.startup_physical_steps){return Ok(head);}
            }
            let previous=manifest.previous_head;
            if let Some(prior)=&previous{ensure!(prior.generation<head.generation,"startup ancestry did not decrease");}cursor=previous;
        }anyhow::bail!("no complete all8 startup anchor at/before requested physical step; initial0 anchor is required")
    }
    pub fn select_historical_own_end_prefix_anchor(&self,expected_current:&Head,identity:&CheckpointIdentity,logical:u64)->Result<Head>{
        ensure!(identity.stream_count==8 && logical>0 && logical%500!=0,"own-end requested prefix must be an interior collective logical step");
        let _lock=self.lock()?;ensure!(self.head()?.as_ref()==Some(expected_current),"source HEAD changed during own-end prefix selection");
        let mut cursor=Some(expected_current.clone());let mut count=0usize;
        while let Some(head)=cursor {count+=1;ensure!(count<=100000,"own-end prefix ancestry budget exceeded");
            let manifest=self.read_manifest(&head)?;ensure!(manifest.identity==*identity,"prefix source identity changed");
            if manifest.boundaries.iter().all(|b|crate::native_run_continuation::own_end_phase(&b.next_integrator_phase)) {
                let joined=self.own_end_publication(identity,&manifest.boundaries,&manifest.fields)?;
                if joined.collective_logical_step<=logical {return Ok(head);}
            }
            let previous=manifest.previous_head;
            if let Some(prior)=&previous{ensure!(prior.generation<head.generation,"prefix scientific ancestry does not decrease");}cursor=previous;
        }anyhow::bail!("no complete own-end scientific anchor before requested original prefix")
    }
    pub fn inspect_historical_generation(&self,expected_current:&Head,anchor:&Head,identity:&CheckpointIdentity)->Result<ValidatedGeneration> {
        let _lock=self.lock()?;ensure!(self.head()?.as_ref()==Some(expected_current),"scientific source HEAD changed");
        let mut cursor=Some(expected_current.clone());let mut count=0usize;
        while let Some(head)=cursor {count+=1;ensure!(count<=100000,"scientific historical search exceeds ancestry budget");
            let manifest=self.read_manifest(&head)?;ensure!(manifest.identity==*identity,"historical scientific identity differs");
            if head==*anchor{return self.validate_generation_metadata(anchor,identity);}
            if let Some(prior)=&manifest.previous_head{ensure!(prior.generation<head.generation,"historical ancestry does not decrease");}
            cursor=manifest.previous_head;
        }anyhow::bail!("requested historical generation is not a committed ancestor")
    }
}

/// Publish exact saved small metadata in a fresh branch without replacing any
/// concurrent/live writer. Same identity is idempotent; different bytes refuse.
pub fn publish_branch_metadata(root:&Path,relative:&Path,bytes:&[u8])->Result<()> {
    ensure!(bytes.len()<=64*1024*1024 && !relative.as_os_str().is_empty()
        && relative.components().all(|c|matches!(c,std::path::Component::Normal(_))),"invalid bounded branch metadata");
    verify_directory_chain(root)?;
    let mut parent=root.to_path_buf();
    for part in relative.parent().context("metadata relative parent missing")?.components() {
        parent.push(part.as_os_str());
        match fs::create_dir(&parent){Ok(())=>sync_dir(parent.parent().context("metadata parent missing")?)?,
            Err(e)if e.kind()==std::io::ErrorKind::AlreadyExists=>{},Err(e)=>return Err(e.into())}
        verify_directory_chain(&parent)?;
    }
    let path=root.join(relative);
    if path.try_exists()? {
        let existing=read_bounded(&path,64*1024*1024)?;
        ensure!(existing==bytes,"existing historical metadata differs");return Ok(());
    }
    let temporary=parent.join(format!(".metadata-{}.branch.next",hash(bytes)));
    if temporary.try_exists()?{ensure!(read_bounded(&temporary,64*1024*1024)?==bytes,"branch metadata temporary differs");}
    else{spend_branch_copy_capacity(root,bytes.len() as u64)?;write_new(&temporary,bytes)?;}
    fs::hard_link(&temporary,&path)?;sync_dir(&parent)?;fs::remove_file(&temporary)?;sync_dir(&parent)?;Ok(())
}


/// Physical reservation for independently copied output prefixes. Source data
/// is never hard-linked into a mutable writer namespace. Each target write
/// releases its admitted blocks immediately before the existing durable copy.
pub struct BranchCopyReservation {root:PathBuf,file:File}
impl Drop for BranchCopyReservation {
    fn drop(&mut self) {
        // Only our private allocation lease is removed; scientific output and
        // failed partial copies remain available for forensic recovery.
        if self.file.set_len(0).is_ok() {let _=self.file.sync_all();}
        let _=std::fs::remove_file(self.root.join(".scientific_branch_copy_reservation"));
        let _=File::open(&self.root).and_then(|f|f.sync_all());
    }
}
pub fn branch_copy_member_bound(bytes:u64)->Result<u64> {
    bytes.checked_add(4095).map(|n|n/4096*4096).and_then(|n|n.checked_mul(2))
        .and_then(|n|n.checked_add(8192)).context("historical copied-output physical bound overflow")
}
pub fn reserve_branch_copy_capacity(root:&Path,bytes:u64,free_floor:u64)->Result<BranchCopyReservation> {
    verify_directory_chain(root)?;
    use std::os::unix::ffi::OsStrExt;
    let path=std::ffi::CString::new(root.as_os_str().as_bytes())?;
    let mut stat=std::mem::MaybeUninit::<libc::statvfs>::uninit();
    ensure!(unsafe{libc::statvfs(path.as_ptr(),stat.as_mut_ptr())}==0,"branch output capacity unavailable");
    let stat=unsafe{stat.assume_init()};let free=(stat.f_bavail as u64).checked_mul(stat.f_frsize as u64).context("branch free capacity overflow")?;
    let file=OpenOptions::new().read(true).write(true).create(true).truncate(false).custom_flags(libc::O_NOFOLLOW|libc::O_CLOEXEC)
        .open(root.join(".scientific_branch_copy_reservation"))?;
    ensure!(file.metadata()?.is_file(),"branch copy reservation is not regular");
    ensure!(unsafe{libc::flock(file.as_raw_fd(),libc::LOCK_EX|libc::LOCK_NB)}==0,"branch copy reservation concurrently owned");
    let already_allocated=file.metadata()?.blocks().checked_mul(512).context("branch reserved blocks overflow")?;
    ensure!(bytes.checked_add(free_floor).is_some_and(|need|need<=free.saturating_add(already_allocated)),"exact committed output prefixes exceed branch preallocation capacity");
    file.set_len(bytes)?;
    if bytes>0 {let rc=unsafe{libc::posix_fallocate(file.as_raw_fd(),0,libc::off_t::try_from(bytes)?)};
        ensure!(rc==0,"branch copy physical reservation refused: {}",std::io::Error::from_raw_os_error(rc));}
    file.sync_all()?;sync_dir(root)?;Ok(BranchCopyReservation{root:root.into(),file})
}
pub(super) fn spend_branch_copy_capacity(root:&Path,bytes:u64)->Result<()> {
    let path=root.join(".scientific_branch_copy_reservation");
    if !path.try_exists()?{return Ok(());}
    let file=OpenOptions::new().read(true).write(true).custom_flags(libc::O_NOFOLLOW|libc::O_CLOEXEC).open(path)?;
    ensure!(file.metadata()?.is_file(),"branch copy reserve is not regular");
    let bound=branch_copy_member_bound(bytes)?;let remaining=file.metadata()?.len();
    ensure!(bound<=remaining,"committed branch output exceeds preallocated member population");
    file.set_len(remaining-bound)?;file.sync_all()?;Ok(())
}

impl CheckpointStore {
    pub fn scientific_branch_allocation_bound(&self,anchor:&Head,identity:&CheckpointIdentity)->Result<u64> {
        let mut cursor=Some(anchor.clone());let mut count=0usize;let mut bound=4*1024*1024u64;
        let mut objects=std::collections::BTreeSet::new();let mut indexes=std::collections::BTreeSet::new();
        while let Some(head)=cursor {count+=1;ensure!(count<=100000,"scientific branch allocation ancestry exceeded");
            let generation=self.validate_generation_metadata(&head,identity)?;
            bound=bound.checked_add(branch_copy_member_bound(0)?).context("scientific branch metadata bound overflow")?;
            for field in &generation.fields {
                if let Some(envelope)=&field.representation {self.accumulate_lossless_branch_bound(envelope,&mut objects,&mut indexes,&mut bound)?;}
                else if let Some(hash)=&field.object_sha256 {if objects.insert(("objects".to_string(),hash.clone())) {
                    bound=bound.checked_add(branch_copy_member_bound(0)?).context("raw CAS link bound overflow")?;}}
                ensure!(objects.len()<=262144 && indexes.len()<=262144,"branch allocation object metadata exceeds bounded admission");
            }
            let previous=self.read_manifest(&head)?.previous_head;
            if let Some(prior)=&previous {ensure!(prior.generation<head.generation,"branch allocation ancestry does not decrease");}cursor=previous;
        }Ok(bound)
    }
}

#[derive(Clone,Debug,Serialize,Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScientificCampaignCapacityV1 {
    pub schema_version:u32,pub collective_chunk_count:u64,pub capsule_every_chunks:u64,
    pub exact_engine_field_count_per_worker:u64,pub raw_store_retention_bound:u64,
    pub additional_requested_anchor:bool,pub planned_stop_anchor_upper:u64,
}
impl CheckpointStore {
    pub fn admit_remaining_campaign_capacity(&self,identity:&CheckpointIdentity,first_chunk:u64,
        has_scientific_head:bool,maximum_generation_bytes:u64,maximum_retained_bytes:u64,plan:&ScientificCampaignCapacityV1)->Result<serde_json::Value> {
        ensure!(plan.schema_version==1 && plan.planned_stop_anchor_upper==1 && plan.capsule_every_chunks>0 && plan.exact_engine_field_count_per_worker==351
            && first_chunk<=plan.collective_chunk_count,"scientific campaign capacity contract differs from exact H17 registry/cadence");
        let terminal=plan.collective_chunk_count;
        let regular=(terminal/plan.capsule_every_chunks).checked_sub(first_chunk/plan.capsule_every_chunks).context("remaining capsule count underflow")?;
        let terminal_extra=u64::from(terminal>first_chunk && terminal%plan.capsule_every_chunks!=0);
        let initial=u64::from(!has_scientific_head);
        let requested=u64::from(plan.additional_requested_anchor);
        let capsules=regular.checked_add(terminal_extra).and_then(|n|n.checked_add(initial)).and_then(|n|n.checked_add(requested)).and_then(|n|n.checked_add(plan.planned_stop_anchor_upper)).context("scientific capsule population overflow")?;
        let fields=plan.exact_engine_field_count_per_worker.checked_add(1).and_then(|n|n.checked_mul(identity.stream_count as u64)).and_then(|n|n.checked_add(1)).context("scientific field population overflow")?;
        let chunks=maximum_generation_bytes.div_ceil(4*1024*1024).checked_add(fields).context("scientific bounded chunk count overflow")?;
        // Encoder refuses a leaf whose zstd bound exceeds decoded+64 KiB;
        // every chunk/index envelope is bounded at16 KiB. The remaining64 KiB
        // per chunk covers its ordered field manifest and publication overhead.
        // No compression ratio, shared leaf or CAS dedup saving is assumed.
        let generation_upper=maximum_generation_bytes.checked_add(chunks.checked_mul(128*1024).context("scientific encoded overhead overflow")?)
            .and_then(|n|n.checked_add(fields.checked_mul(16384)?)).and_then(|n|n.checked_add(MAX_MANIFEST_BYTES*2)).context("scientific encoded generation upper overflow")?;
        let existing=self.retained_object_bytes()?;
        let required=existing.checked_add(capsules.checked_mul(generation_upper).context("scientific campaign encoded upper overflow")?)
            .and_then(|n|n.checked_add(plan.raw_store_retention_bound)).context("combined future scientific/raw retention bound overflow")?;
        ensure!(required<=maximum_retained_bytes,
            "complete remaining scientific/raw campaign exceeds declared retained reservation: require {} B, configured {} B; configure a separately explicit capsule cadence/generation cap/store capacity before MD",required,maximum_retained_bytes);
        Ok(serde_json::json!({"schema":"prism.scientific_campaign_capacity.v1","identity":identity,"first_collective_chunk":first_chunk,"capacity_contract":plan,
            "initial_anchors":initial,"remaining_regular_anchors":regular,"extra_terminal_anchor":terminal_extra,"requested_anchor_upper":requested,"planned_stop_anchor_upper":plan.planned_stop_anchor_upper,
            "remaining_capsules_upper":capsules,"registry_fields_per_generation":fields,"decoded_generation_cap_bytes":maximum_generation_bytes,
            "encoded_generation_upper_bytes":generation_upper,"retained_existing_bytes":existing,"remaining_raw_retention_upper_bytes":plan.raw_store_retention_bound,
            "required_retained_bytes":required,"configured_retained_bytes":maximum_retained_bytes,
            "compression_ratio_assumed":false,"dedup_savings_assumed":false,"physical_fallocate_next":true,
            "full_md_including_required_durable_commits_must_qualify_under_900_seconds":true,"measured_runtime_qualified":false}))
    }
}

/// Stateless preallocation admission under the existing all-worker pause.
/// This measures RAM/cgroup headroom; it is not an exclusive OS reservation.
pub fn admit_process_capture_host_peak(required:u64,free_floor:u64)->Result<serde_json::Value> {
        let meminfo=std::fs::read_to_string("/proc/meminfo")?;
        let available=meminfo.lines().find_map(|line|line.strip_prefix("MemAvailable:").and_then(|v|v.split_whitespace().next()).and_then(|v|v.parse::<u64>().ok())).context("host MemAvailable unavailable")?.checked_mul(1024).context("host free byte overflow")?;
        let mut admitted_available=available;
        let cgroup=std::fs::read_to_string("/proc/self/cgroup")?;
        let relative=cgroup.lines().find_map(|line|line.strip_prefix("0::")).context("cgroup v2 host capacity unavailable")?;
        ensure!(relative.starts_with('/') && !Path::new(relative).components().any(|c|matches!(c,std::path::Component::ParentDir)),"invalid cgroup resource path");
        let mut current=PathBuf::from("/sys/fs/cgroup").join(relative.trim_start_matches('/'));
        let mut cgroup_limits=Vec::new();
        loop {
            // Kernel cgroup-v2 memory interfaces are non-root files. The
            // global root is bounded by MemAvailable above, not a fabricated
            // memory.max. A namespace root exposing a real limit still has
            // that limit enforced below. Missing non-root files remain fatal.
            if current==Path::new("/sys/fs/cgroup") && !current.join("memory.max").try_exists()? {
                ensure!(current.is_dir() && current.join("cgroup.controllers").is_file(),"cgroup v2 controller root unavailable");
                break;
            }
            let maximum=std::fs::read_to_string(current.join("memory.max"))
                .with_context(||format!("read cgroup memory limit {}",current.display()))?;
            if maximum.trim()!="max" {
                let maximum:u64=maximum.trim().parse()?;let used:u64=std::fs::read_to_string(current.join("memory.current"))?.trim().parse()?;
                let remaining=maximum.checked_sub(used).context("existing cgroup already exceeds host cap")?;
                admitted_available=admitted_available.min(remaining);
                cgroup_limits.push(serde_json::json!({"path":current,"maximum":maximum,"current":used,"remaining":remaining}));
            }
            if current==Path::new("/sys/fs/cgroup"){break;}
            ensure!(current.pop() && current.starts_with("/sys/fs/cgroup"),"cgroup admission escaped root");
        }
        ensure!(required.checked_add(free_floor).is_some_and(|n|n<=admitted_available),
            "scientific capture host peak exceeds preallocation headroom: need {} plus floor {}, available {}",required,free_floor,admitted_available);
        Ok(serde_json::json!({"schema":"prism.scientific_capture_host_capacity.v1","required_bytes":required,
            "free_floor_bytes":free_floor,"mem_available_bytes":available,"admitted_available_bytes":admitted_available,
            "cgroup_limits":cgroup_limits,"checked_before_any_generation_payload_capture":true,"os_exclusive_memory_reservation":false}))
}

