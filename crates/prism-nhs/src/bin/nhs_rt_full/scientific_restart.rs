//! Production-owner historical fork and exact output-prefix ingress. No MD
//! surrogate is run here; native workers restore and replay their original path.
use super::*;
use prism_nhs::scientific_checkpoint::{CheckpointIdentity,CheckpointStore,Head,AppendCursor};
use prism_nhs::native_run_continuation::{NativeWorkerContinuationV1,NativeSharedContinuationV1,NativeMetadataBoundaryV1};

#[derive(Clone,Debug,serde::Serialize,serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FailureAuthorityPinV1{pub path:PathBuf,pub sha256:String}
#[derive(Clone,Debug,serde::Serialize,serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HistoricalFailureAuthorityV1{
    pub schema_version:u32,pub accounting_receipt:FailureAuthorityPinV1,
    pub reconciliation_result:FailureAuthorityPinV1,pub capacity_disposition:FailureAuthorityPinV1,
}
fn failed_pin(pin:&FailureAuthorityPinV1)->Result<serde_json::Value>{
    anyhow::ensure!(pin.path.is_absolute() && prism_nhs::ensemble::manifest::sha256_file(&pin.path)?==pin.sha256,
        "failed-source authority digest/path differs");
    let file=std::fs::OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW|libc::O_CLOEXEC).open(&pin.path)?;
    anyhow::ensure!(file.metadata()?.is_file() && file.metadata()?.len()<=16*1024*1024,"failed-source authority exceeds bounded regular metadata");
    Ok(serde_json::from_reader(file)?)
}
pub fn validate_failure_authority(authority:&HistoricalFailureAuthorityV1,source:&Path,head:&Head,
    identity:&CheckpointIdentity,verify_current_capacity:bool)->Result<serde_json::Value>{
    use std::os::unix::fs::MetadataExt;
    anyhow::ensure!(authority.schema_version==1,"unsupported failure authority schema");
    let receipt=failed_pin(&authority.accounting_receipt)?;let result=failed_pin(&authority.reconciliation_result)?;
    let capacity=failed_pin(&authority.capacity_disposition)?;
    let receipt_pin=serde_json::to_value(&authority.accounting_receipt)?;
    anyhow::ensure!(receipt["schema"]=="prism.continuation_md_failure_accounting.v2"
        && receipt["segment_outcome"]=="FAILED_RECONCILED" && receipt["eligible_remaining_budget"]==true
        && receipt["failure_quiescence_verified"]==true && receipt["native_md_complete"]==false
        && receipt["native_retirement_claimed"]==false && receipt["native_durable_ACK_claimed"]==false
        && receipt["scientific_head"]==serde_json::to_value(head)? && receipt["scientific_identity"]==serde_json::to_value(identity)?
        && receipt["source_store_root"]==serde_json::to_value(source)?
        && receipt["engine_sha256"]==identity.engine_sha256 && receipt["boot_id"]==std::fs::read_to_string("/proc/sys/kernel/random/boot_id")?.trim(),
        "failed-source timing/SCI/campaign authority is not exact eligible current-boot reconciliation");
    anyhow::ensure!(receipt["native_exit_code"].is_null() || receipt["native_exit_code"].as_i64().is_some_and(|v|v!=0),
        "failed-source reconciliation cannot declare native exit0");
    let prior=receipt["prior_consumed_md_ns"].as_u64().context("failed source prior charge absent")?;
    let start=receipt["current_segment_start_monotonic_ns"].as_u64().context("failed source start absent")?;
    let end=receipt["current_segment_end_monotonic_upper_ns"].as_u64().context("failed source bounded end absent")?;
    let elapsed=end.checked_sub(start).context("failed source time regressed")?;
    let consumed=prior.checked_add(elapsed).context("failed source charge overflow")?;
    anyhow::ensure!(consumed<900_000_000_000 && receipt["current_segment_conservative_md_ns"].as_u64()==Some(elapsed)
        && receipt["cumulative_consumed_md_ns"].as_u64()==Some(consumed)
        && result["schema"]=="prism.c1_open_failure_reconciliation.result.v1"
        && result["accounting_receipt"]==receipt_pin && result["eligible_remaining_budget"]==true
        && result["protected_source_verified"]==true && result["science_or_reserved_space_mutated"]==false
        && capacity["schema"]=="prism.failed_source_capacity_disposition.v1"
        && capacity["accounting_receipt"]==receipt_pin
        && capacity["failure_reconciliation_result"]==serde_json::to_value(&authority.reconciliation_result)?
        && capacity["scientific_head"]==receipt["scientific_head"] && capacity["scientific_identity"]==receipt["scientific_identity"]
        && capacity["source_store_root"]==receipt["source_store_root"] && capacity["native_retirement_claimed"]==false
        && capacity["native_closure_verified"]==false && capacity["source_capacity_was_not_mutated"]==true
        && capacity["all_other_source_membership_and_stats_unchanged"]==true,
        "failed-source conservative charge/result/capacity composition differs");
    if verify_current_capacity {
        let file=std::fs::OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW|libc::O_CLOEXEC).open(source.join("RESTART_RESERVED_SPACE"))?;
        let info=file.metadata()?;let r=&capacity["reserved"];
        anyhow::ensure!(info.is_file() && info.nlink()==1 && r["device"].as_u64()==Some(info.dev())
            && r["inode"].as_u64()==Some(info.ino()) && r["size"].as_u64()==Some(info.len())
            && r["allocated_bytes"].as_u64()==Some(info.blocks().checked_mul(512).context("source block bytes overflow")?),
            "failed-source observed scratch allocation changed; native retirement is not inferred");
        let raw_path=source.join("CHRONICLE_HEAD");
        let raw=if raw_path.exists(){serde_json::from_reader(std::fs::File::open(raw_path)?)?}else{serde_json::Value::Null};
        anyhow::ensure!(raw==capacity["raw_head"],"failed-source raw emergency frontier changed before branch copy");
    }
    Ok(receipt)
}
#[derive(Clone,Debug,serde::Serialize,serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HistoricalBranchContract {
    pub schema_version:u32,
    pub source_store_root:PathBuf,
    pub source_output_root:PathBuf,
    pub source_current_head:Head,
    pub destination_store_root:PathBuf,
    pub destination_output_root:PathBuf,
    pub requested_step:u64,
    /// When present, requested_step must be zero; collective ticks are never
    /// interpreted as a physical step/time shared by unequal owners.
    #[serde(default)]
    pub requested_collective_next_tick:Option<u32>,
    /// Version2 only: an ORIGINAL common logical feedback-window prefix. It
    /// is not a physical clock shared by unequal or retired worker owners.
    #[serde(default)]
    pub requested_collective_logical_step:Option<u64>,
    /// Version3: completed ORIGINAL startup physical body1..5, before its
    /// pending once-per-six tail; never the scheduledlogical selector.
    #[serde(default)]
    pub requested_startup_physical_step:Option<u32>,
    /// Version4 only: conservative failed-source C1 plus actual capacity
    /// observation. This never claims native planned retirement or ACK.
    #[serde(default)]
    pub source_failure_authority:Option<HistoricalFailureAuthorityV1>,
    pub expected_identity:CheckpointIdentity,
}
#[derive(Default)]
pub struct HistoricalReplayRequest {pub requested_native_step:Option<u64>,pub requested_collective_logical_step:Option<u64>,pub requested_startup_physical_step:Option<u32>,pub selected_startup_physical_steps:Option<u64>}
pub fn prepare_historical_branch(args:&Args,current:&CheckpointIdentity)->Result<HistoricalReplayRequest> {
    let Some(path)=args.scientific_restart_branch_contract.as_ref() else{return Ok(HistoricalReplayRequest::default());};
    anyhow::ensure!(args.resume_scientific_checkpoint,"historical branch requires explicit scientific resume mode");
    let file=std::fs::OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW|libc::O_CLOEXEC).open(path)?;
    anyhow::ensure!(file.metadata()?.is_file() && file.metadata()?.len()<=65536,"historical branch contract is not a bounded regular file");
    let contract:HistoricalBranchContract=serde_json::from_reader(file)?;
    anyhow::ensure!(matches!(contract.schema_version,1|2|3|4) && Some(&contract.destination_store_root)==args.scientific_checkpoint_root.as_ref()
        && Some(&contract.destination_store_root)==args.md_chronicle_root.as_ref() && contract.destination_output_root==args.output,
        "historical branch destination differs from admitted native output/CAS roots");
    anyhow::ensure!((contract.schema_version==4)==contract.source_failure_authority.is_some(),
        "failed-source continuation requires distinct schema4 explicit authority");
    let mut identity=current.clone();identity.run_id=contract.expected_identity.run_id.clone();
    anyhow::ensure!(identity==contract.expected_identity,"historical branch source scientific/provider/input identity differs");
    let selected_count=u8::from(contract.requested_collective_next_tick.is_some())+u8::from(contract.requested_collective_logical_step.is_some())
        +u8::from(contract.requested_startup_physical_step.is_some());
    let own_end_selector=selected_count>0;
    anyhow::ensure!(args.protocol_own_end==own_end_selector && selected_count<=1,
        "historical selector must explicitly choose exactly one own-end collective or ordinary physical chronology");
    if let Some(physical)=contract.requested_startup_physical_step {
        anyhow::ensure!(matches!(contract.schema_version,3|4) && args.may_original_physical_profile && contract.requested_step==0
            && (1..6).contains(&physical),"startup branch requires version3 explicit May physical1..5 selector");
    }
    if let Some(logical)=contract.requested_collective_logical_step {
        let targets=if args.may_original_physical_profile{prism_nhs::native_run_continuation::MAY_ORIGINAL_HORIZONS}else{prism_nhs::native_run_continuation::OWN_END_HORIZONS};
        anyhow::ensure!(matches!(contract.schema_version,2|4) && contract.requested_step==0 && logical>0 && logical%500!=0
            && logical<u64::try_from(*targets.iter().max().context("own-end logical target missing")?)?,
            "own-end prefix requires its version2 ORIGINAL logical interior selector");
    }
    if let Some(tick)=contract.requested_collective_next_tick {
        anyhow::ensure!(contract.requested_step==0 && tick<=if args.may_original_physical_profile{80}else{72},"own-end historical ingress requires an exact completed collective tick, with no scalar physical step substitution");
    }
    let totals=(0..identity.stream_count as usize).map(|stream|configured_differential_protocol(args,stream,identity.stream_count as usize)
        .map(|protocol|protocol.total_steps())).collect::<Result<Vec<_>>>()?;
    let physical_horizon=multi_stream_campaign_step_target(args.steps,false,&totals)?;
    anyhow::ensure!(own_end_selector || contract.requested_step<=u64::try_from(physical_horizon)?,"requested scientific step exceeds configured original physical horizon");
    let terminal_next_tick=physical_horizon.checked_add(499).context("terminal tick overflow")?/500;
    let _source_lease=prism_nhs::native_run_continuation::claim_quiescent_branch_source(&contract.source_output_root)?;
    if let Some(authority)=&contract.source_failure_authority {
        validate_failure_authority(authority,&contract.source_store_root,&contract.source_current_head,&identity,true)?;
    }
    let _target_lease=prism_nhs::native_run_continuation::reserve_output_root(&contract.destination_output_root)?;
    let store=CheckpointStore::create(&contract.source_store_root,args.scientific_checkpoint_max_bytes)?;
    let selected=if let Some(physical)=contract.requested_startup_physical_step {
        store.select_historical_may_startup_anchor(&contract.source_current_head,&identity,physical)?
    }else if let Some(tick)=contract.requested_collective_next_tick {
        store.select_historical_own_end_anchor(&contract.source_current_head,&identity,tick)?
    }else if let Some(logical)=contract.requested_collective_logical_step {
        store.select_historical_own_end_prefix_anchor(&contract.source_current_head,&identity,logical)?
    }else{store.select_historical_anchor(&contract.source_current_head,&identity,contract.requested_step)?};
    let generation=store.inspect_historical_generation(&contract.source_current_head,&selected,&identity)?;
    let _host_admission=store.admit_generation_host_capacity(&generation,args.scientific_checkpoint_free_floor_bytes)?;
    let mut workers=Vec::new();
    for stream in 0..identity.stream_count {
        let worker:NativeWorkerContinuationV1=store.read_generation_host(&generation,&format!("stream_{stream}"),"native_owner_closure",args.scientific_checkpoint_field_bytes)?;
        worker.validate_population(&super::native_worker_continuation_keys(args.protocol_own_end))?;
        if let Some(tick)=contract.requested_collective_next_tick {
            anyhow::ensure!(worker.next_chunk==i32::try_from(tick)?,"own-end saved host common tick differs from requested exact anchor");
            let completion:prism_nhs::native_run_continuation::OwnEndCompletionV1=worker.get("host:own_end_completion")?;
            let boundary=generation.boundaries().iter().find(|b|b.stream_id==stream).context("own-end branch lacks worker boundary")?;
            completion.validate_join(stream,worker.next_chunk,boundary.completed_step)?;
        }
        if let Some(physical)=contract.requested_startup_physical_step {
            let state:prism_nhs::native_run_continuation::MayStartupContinuationV2=worker.get_optional("host:may_startup")?
                .context("startup branch lacks typed original group owner")?;
            state.validate(stream as usize,worker.next_chunk,1)?;
            let boundary=generation.boundaries().iter().find(|b|b.stream_id==stream).context("startup branch boundary missing")?;
            anyhow::ensure!(worker.next_chunk==0 && worker.get::<i32>("host:steps_run")?==0
                && state.executed_physical_steps<=u64::from(physical) && boundary.completed_step==state.executed_physical_steps
                && prism_nhs::native_run_continuation::worker_completed_native_step(&worker)?==boundary.completed_step,
                "startup selected anchor/host/body/physical frontier differs before output or device writes");
        }
        anyhow::ensure!(worker.stream_id==stream,"historical worker identity differs");workers.push(worker);
    }
    let shared:NativeSharedContinuationV1=store.read_generation_host(&generation,"shared","native_owner_closure",args.scientific_checkpoint_field_bytes)?;
    shared.validate_population(super::NATIVE_SHARED_CONTINUATION_KEYS)?;
    let raw:Option<super::chronicle_raw::RawChronicleContinuationV2>=shared.get_optional("raw_chronicle")?;
    let raw=raw.context("historical native profile lacks the exact raw observation cursor owner")?;
    anyhow::ensure!(raw.identity==identity && generation.boundaries().iter().all(|b|raw.boundaries.get(b.stream_id as usize).is_some_and(|r|r.stream_id==b.stream_id && r.completed_step==b.completed_step && r.physical_time_ps_bits==b.physical_time_ps_bits))
        && workers.iter().all(|w|w.checkpoint_ordinal==raw.checkpoint_ordinal && w.next_chunk==raw.common_next_chunk),"historical scientific/host/raw ordinal boundaries disagree");
    // Ordinary ingress retains the original body-slot suffix and live queue.
    // Own-end ingress accepts exact completed collective anchors only.
    let anchor_step=raw.boundaries.iter().map(|b|b.completed_step).max().unwrap_or(0);
    anyhow::ensure!(own_end_selector || anchor_step<=contract.requested_step,"selected historical native anchor follows the requested step");
    let copied_output_bound=output_branch_copy_bound(&workers,&shared)?;
    let target_store=CheckpointStore::create(&contract.destination_store_root,args.scientific_checkpoint_max_bytes)?;
    use std::os::unix::fs::MetadataExt;
    anyhow::ensure!(std::fs::metadata(&contract.source_store_root)?.dev()==std::fs::metadata(&contract.destination_store_root)?.dev(),
        "historical immutable CAS branch requires an explicit same-filesystem destination; data-copy migration is a separate contract");
    let scientific_branch_bound=store.scientific_branch_allocation_bound(&selected,&identity)?;
    let raw_branch_bound=raw.head.as_ref().map(|head|store.chronicle_branch_allocation_bound(head,&identity)).transpose()?.unwrap_or(0);
    let cas_branch_bound=scientific_branch_bound.checked_add(raw_branch_bound).context("historical CAS metadata bound overflow")?;
    let same_reservation_root=target_store.root()==contract.destination_output_root;
    let output_reservation_bound=copied_output_bound.checked_add(if same_reservation_root {cas_branch_bound}else{0}).context("combined branch reservation overflow")?;
    let _output_reservation=prism_nhs::scientific_checkpoint::restart_transaction::reserve_branch_copy_capacity(
        &contract.destination_output_root,output_reservation_bound,args.scientific_checkpoint_free_floor_bytes)?;
    let _cas_metadata_reservation=if same_reservation_root {None}else{Some(prism_nhs::scientific_checkpoint::restart_transaction::reserve_branch_copy_capacity(
        target_store.root(),cas_branch_bound,args.scientific_checkpoint_free_floor_bytes)?)};
    for worker in &workers {
        for key in worker.states.keys().filter(|k|k.starts_with("writer:")) {
            let cursor:Option<AppendCursor>=worker.get_optional(key)?;
            if let Some(cursor)=cursor {prism_nhs::scientific_checkpoint::restart_transaction::copy_committed_append_prefix(
                &contract.source_output_root,&contract.destination_output_root,&cursor)?;}
        }
        let metadata:NativeMetadataBoundaryV1=worker.get("metadata")?;
        anyhow::ensure!(metadata.schema_version==1 && metadata.files.keys().all(|key|!metadata.absent.contains(key)),"historical metadata owner population differs");
        for (relative,bytes) in &metadata.files {prism_nhs::scientific_checkpoint::restart_transaction::publish_branch_metadata(
            &contract.destination_output_root,Path::new(relative),bytes)?;}
        metadata.restore(&contract.destination_output_root)?;
        let resident:Option<prism_nhs::gpu::TrajectoryContinuationV1>=worker.get_optional("resident_traj_writer")?;
        if let Some(saved)=resident {saved.fork_outputs(&contract.source_output_root,&contract.destination_output_root)?;}
        let cadence:Option<prism_nhs::fixed_cadence_coordinate::FixedCadenceContinuationV1>=worker.get_optional("native_pathb_cadence_writer")?;
        if let Some(saved)=cadence {saved.output.fork_outputs(&contract.source_output_root,&contract.destination_output_root)?;}
        let t7:Option<prism_nhs::t7_calibration_journal::T7ContinuationV1>=worker.get_optional("t7_journal")?;
        if let Some(saved)=t7 {prism_nhs::scientific_checkpoint::restart_transaction::copy_committed_append_prefix(
            &contract.source_output_root,&contract.destination_output_root,&saved.journal)?;}
        let asc:Option<prism_nhs::asc_device::measurement::AscMeasurementContinuationV1>=worker.get_optional("isolated_asc_journal")?;
        if let Some(saved)=asc {
            for cursor in [&saved.rows,&saved.records] {prism_nhs::scientific_checkpoint::restart_transaction::copy_committed_append_prefix(
                &contract.source_output_root,&contract.destination_output_root,cursor)?;}
        }
        // This owner exists even with zero trajectory frames: its eight-byte
        // header is still an actual append prefix, not an absent observer.
        let trajectory:Option<super::V2TrajectoryContinuationV1>=worker.get_optional("v2_trajectory_writer")?;
        if let Some(saved)=trajectory {
            anyhow::ensure!(saved.schema_version==1 && saved.stream_id==worker.stream_id as usize,
                "historical trajectory writer identity differs");
            prism_nhs::scientific_checkpoint::restart_transaction::copy_committed_append_prefix_with_header(
                &contract.source_output_root,&contract.destination_output_root,&saved.cursor,Some(&[0u8;8]))?;
        }
        let ghost:Option<prism_nhs::ghost_segment::GhostSegmentContinuationV1>=worker.get_optional("v2_ghost_segment_writer")?;
        if let Some(saved)=ghost {saved.output.fork_outputs(&contract.source_output_root,&contract.destination_output_root)?;}
        let coordinate:Option<prism_nhs::ghost_coordinate_segment::GhostCoordinateContinuationV1>=worker.get_optional("v2_ghost_coordinate_writer")?;
        if let Some(saved)=coordinate {saved.output.fork_outputs(&contract.source_output_root,&contract.destination_output_root)?;}
        let zstr:Option<prism_nhs::zstr::ZstrContinuationV1>=worker.get_optional("v2_zstr_consumer")?;
        if let Some(saved)=zstr {prism_nhs::scientific_checkpoint::restart_transaction::copy_committed_append_prefix(
            &contract.source_output_root,&contract.destination_output_root,&saved.output)?;}

    }
    let chronology:Option<prism_nhs::scientific_chronology::ScientificChronologyContinuationV1>=shared.get_optional("scientific_chronology")?;
    if let Some(saved)=chronology {saved.fork_outputs(&contract.source_output_root,&contract.destination_output_root)?;}
    // Every referenced output prefix is now durable and verified. Raw ancestry
    // commits before scientific HEAD so no published native capsule can point
    // at an unavailable writer prefix or a not-yet-forked raw foliage.
    // Publish the complete original raw-object/foliage closure first. A crash
    // before SCI HEAD publication leaves a raw-only branch, never a scientific
    // HEAD with unavailable observation references.
    if let Some(anchor)=&raw.head {
        let head=store.chronicle_head()?.context("historical source lost observational HEAD")?;
        store.fork_chronicle_history(&contract.destination_store_root,&head,anchor,&identity)?;
    }
    store.fork_scientific_history(&contract.destination_store_root,&contract.source_current_head,&selected,&identity)?;
    write_json_pretty_atomic(&args.output.join("scientific_historical_branch_receipt.json"),&serde_json::json!({
        "schema_version":1,"contract_sha256":prism_nhs::ensemble::manifest::sha256_file(path)?,"source_head":contract.source_current_head,
        "selected_anchor":selected,"anchor_max_physical_step":anchor_step,"per_owner_boundaries":raw.boundaries,"requested_step":if !own_end_selector{Some(contract.requested_step)}else{None},"requested_collective_next_tick":contract.requested_collective_next_tick,"requested_collective_logical_step":contract.requested_collective_logical_step,"requested_startup_physical_step":contract.requested_startup_physical_step,
        "source_failure_authority":contract.source_failure_authority,"source_unchanged":true,"mutable_output_prefixes_copied_not_linked":true,"copied_output_preallocation_bound":copied_output_bound,"cas_branch_metadata_preallocation_bound":cas_branch_bound,"identity":identity,
        "actual_requested_step_replay_qualified":false,"scope":"historical_anchor_plus_source_bound_original_body_slot_replay_pending_actual_qualification",
        "current_operational_argv_sha256":format!("{:x}",Sha256::digest(std::fs::read("/proc/self/cmdline")?)),
        "current_named_operational_environment_sha256":format!("{:x}",Sha256::digest(serde_json::to_vec(&std::env::vars().filter(|(name,_)|name.starts_with("PRISM_NATIVE_MD_")||name.starts_with("PRISM_DURABLE_STOP_")||name=="CUDA_MPS_PIPE_DIRECTORY"||name=="CUDA_MPS_LOG_DIRECTORY").collect::<BTreeMap<_,_>>())?)),"post_md_native_restore_can_skip_md":workers.iter().all(|w|w.next_chunk>=terminal_next_tick)
    }))?;Ok(HistoricalReplayRequest{requested_native_step:if let Some(logical)=contract.requested_collective_logical_step {
        Some(logical.checked_add(if args.may_original_physical_profile{6}else{0}).context("requested native prefix frontier overflow")?)
    }else if contract.requested_collective_next_tick.is_some() || contract.requested_startup_physical_step.is_some(){None}else{Some(contract.requested_step)},
        requested_collective_logical_step:contract.requested_collective_logical_step,requested_startup_physical_step:contract.requested_startup_physical_step,
        selected_startup_physical_steps:(args.may_original_physical_profile && workers.iter().all(|w|w.next_chunk==0) && anchor_step<=6).then_some(anchor_step)})
}
pub use prism_nhs::native_run_continuation::OriginalDispatchCursorV1;
pub type OriginalBodySlices=BTreeMap<(u32,u32),prism_nhs::graph_capture::AutonomousGraph>;
/// Build every additional requested prefix/suffix BEFORE the measured all-owner
/// admission barrier. The MD dispatcher refuses an absent cache entry; it never
/// captures a new graph after physical propagation has started.
pub fn prepare_original_body_slices(engine:&mut prism_nhs::persistent_engine::PersistentNhsEngine,
    width:u32,chunk:u32,requested:Option<u64>,restored:Option<&OriginalDispatchCursorV1>)->Result<OriginalBodySlices> {
    anyhow::ensure!(width>0 && chunk>0,"empty original native dispatch shape");
    let mut keys=BTreeSet::new();
    let requested_slot=requested.map(|step|((step%chunk as u64)%width as u64)as u32);
    if let Some(slot)=requested_slot.filter(|slot|*slot>0) {keys.insert((0,slot));keys.insert((slot,width-slot));}
    if let Some(saved)=restored {
        saved.validate(saved.window_start_step as i32,saved.completed_native_step as u64,saved.original_chunk_index,width)?;
        let slot=saved.next_original_body_slot;
        if slot>0 {keys.insert((slot,width-slot));}
        if let Some(target)=requested {
            if target>saved.completed_native_step as u64 && target<saved.window_end_step as u64
                && (target-saved.window_start_step as u64)/width as u64==(saved.completed_native_step-saved.window_start_step)as u64/width as u64 {
                keys.insert((slot,u32::try_from(target-saved.completed_native_step as u64)?));
            }
        }
    }
    let mut result=BTreeMap::new();
    for (slot,count) in keys {result.insert((slot,count),engine.capture_original_body_slice(count,count,slot,width)?);}
    Ok(result)
}
#[derive(Clone,Debug,serde::Serialize)]
pub struct OriginalDispatchReceipt {pub start_device_step:u32,pub end_device_step:u32,pub logical_steps:u32,
    pub graph_launches:u32,pub original_body_width:u32,pub requested_checkpoint_committed:bool}
/// Original physical body order and original feedback endpoint. Every executed
/// segment is fenced/read back by the existing production graph executor.
pub fn execute_original_dispatch(engine:&mut prism_nhs::persistent_engine::PersistentNhsEngine,
    primary:&prism_nhs::graph_capture::AutonomousGraph,slices:&OriginalBodySlices,
    chunk_index:i32,window_start:u32,window_end:u32,requested:Option<u64>,
    mut checkpoint:impl FnMut(&mut prism_nhs::persistent_engine::PersistentNhsEngine,OriginalDispatchCursorV1)->Result<()>)->Result<OriginalDispatchReceipt> {
    let width=primary.steps_per_launch();let start=engine.device_protocol_current_step()?;
    anyhow::ensure!(width>0 && window_start<=start && start<window_end,"original dispatch physical/host window differs");
    let target=requested.filter(|target|*target>start as u64 && *target<window_end as u64).map(u32::try_from).transpose()?;
    let mut current=start;let mut launches=0u32;let mut committed=false;
    while current<window_end {
        let end=if !committed{target.unwrap_or(window_end)}else{window_end};
        let slot=(current-window_start)%width;let remaining=end-current;
        let (graph,logical)=if slot==0 && remaining>=width {(primary,(remaining/width)*width)}else{
            let count=remaining.min(width-slot);
            (slices.get(&(slot,count)).context("original prefix/suffix graph was not preallocated before admission")?,count)
        };
        let receipt=engine.execute_nonresident_graph_replay_window(graph,logical)?;
        anyhow::ensure!(receipt.start_device_step==current && receipt.logical_steps==logical
            && receipt.end_device_step==current.checked_add(logical).context("original dispatch step overflow")?
            && receipt.steps_per_launch==graph.steps_per_launch(),"original dispatch native segment receipt differs");
        current=receipt.end_device_step;launches=launches.checked_add(receipt.graph_launches).context("original dispatch launch count overflow")?;
        if target==Some(current) && !committed {
            checkpoint(engine,OriginalDispatchCursorV1{schema_version:1,original_chunk_index:chunk_index,
                window_start_step:window_start,window_end_step:window_end,completed_native_step:current,
                original_body_width:width,next_original_body_slot:(current-window_start)%width,chunk_prelude_applied:true})?;
            committed=true;
        }
    }
    Ok(OriginalDispatchReceipt{start_device_step:start,end_device_step:current,logical_steps:current-start,graph_launches:launches,
        original_body_width:width,requested_checkpoint_committed:committed})
}

pub fn load_branch_provenance(root:Option<&Path>)->Result<Option<serde_json::Value>> {
    let Some(root)=root else{return Ok(None);};let path=root.join("SCIENTIFIC_BRANCH_ORIGIN.json");
    if !path.try_exists()?{return Ok(None);}
    let file=std::fs::OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW|libc::O_CLOEXEC).open(&path)?;
    anyhow::ensure!(file.metadata()?.is_file() && file.metadata()?.len()<=65536,"historical branch origin is not bounded regular metadata");
    Ok(Some(serde_json::json!({"origin_sha256":prism_nhs::ensemble::manifest::sha256_file(&path)?,"origin":serde_json::from_reader::<_,serde_json::Value>(file)?})))
}


/// Count the very same typed native output owners used by historical ingress,
/// before any destination HEAD is published. Duplicated references deliberately
/// over-reserve; no file/prefix is discarded to reduce this bound.
fn output_branch_copy_bound(workers:&[NativeWorkerContinuationV1],shared:&NativeSharedContinuationV1)->Result<u64> {
    fn add(total:&mut u64,bytes:u64)->Result<()> {*total=total.checked_add(bytes).context("historical output population bound overflow")?;Ok(())}
    fn cursor(total:&mut u64,saved:&AppendCursor)->Result<()> {
        add(total,prism_nhs::scientific_checkpoint::restart_transaction::branch_copy_member_bound(saved.committed_bytes)?)
    }
    // Receipts, directory publication and small branch-control metadata.
    let mut total=4*1024*1024u64;
    for worker in workers {
        for key in worker.states.keys().filter(|key|key.starts_with("writer:")) {
            if let Some(saved)=worker.get_optional::<AppendCursor>(key)? {cursor(&mut total,&saved)?;}
        }
        let metadata:NativeMetadataBoundaryV1=worker.get("metadata")?;
        for bytes in metadata.files.values() {add(&mut total,prism_nhs::scientific_checkpoint::restart_transaction::branch_copy_member_bound(bytes.len() as u64)?)?;}
        if let Some(saved)=worker.get_optional::<prism_nhs::gpu::TrajectoryContinuationV1>("resident_traj_writer")? {add(&mut total,saved.branch_copy_allocation_bound()?)?;}
        if let Some(saved)=worker.get_optional::<prism_nhs::fixed_cadence_coordinate::FixedCadenceContinuationV1>("native_pathb_cadence_writer")? {add(&mut total,saved.output.branch_copy_allocation_bound()?)?;}
        if let Some(saved)=worker.get_optional::<prism_nhs::t7_calibration_journal::T7ContinuationV1>("t7_journal")? {cursor(&mut total,&saved.journal)?;}
        if let Some(saved)=worker.get_optional::<prism_nhs::asc_device::measurement::AscMeasurementContinuationV1>("isolated_asc_journal")? {cursor(&mut total,&saved.rows)?;cursor(&mut total,&saved.records)?;}
        if let Some(saved)=worker.get_optional::<super::V2TrajectoryContinuationV1>("v2_trajectory_writer")? {cursor(&mut total,&saved.cursor)?;}
        if let Some(saved)=worker.get_optional::<prism_nhs::ghost_segment::GhostSegmentContinuationV1>("v2_ghost_segment_writer")? {add(&mut total,saved.output.branch_copy_allocation_bound()?)?;}
        if let Some(saved)=worker.get_optional::<prism_nhs::ghost_coordinate_segment::GhostCoordinateContinuationV1>("v2_ghost_coordinate_writer")? {add(&mut total,saved.output.branch_copy_allocation_bound()?)?;}
        if let Some(saved)=worker.get_optional::<prism_nhs::zstr::ZstrContinuationV1>("v2_zstr_consumer")? {cursor(&mut total,&saved.output)?;}
    }
    if let Some(saved)=shared.get_optional::<prism_nhs::scientific_chronology::ScientificChronologyContinuationV1>("scientific_chronology")? {add(&mut total,saved.branch_copy_allocation_bound()?)?;}
    Ok(total)
}
