//! Host cumulative wall-budget binding. Scientific state remains in the
//! original NativeContinuationRun/CheckpointEpoch; no propagation is performed.
use super::*;
use std::os::unix::fs::OpenOptionsExt;
use prism_nhs::native_run_continuation::NativeContinuationRun;
use prism_nhs::scientific_checkpoint::Head;
const CEILING_NS:u64=900_000_000_000;
#[derive(Clone,Debug,serde::Serialize,serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BudgetAtCaptureV1 {
    schema_version:u32,campaign_id:String,engine_sha256:String,topology_sha256:String,
    source_native_proposal_sha256:String,ceiling_ns:u64,minimum_consumed_md_ns:u64,
}
pub struct NativeBudget {
    launch:serde_json::Value,launch_sha256:String,policy_sha256:String,run_token:String,
    prior_ns:u64,start_ns:u64,deadline_ns:u64,identity:prism_nhs::scientific_checkpoint::CheckpointIdentity,
    selected:Option<Head>,output:PathBuf,
}
fn bounded(path:&Path)->Result<serde_json::Value>{
    let file=std::fs::OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW|libc::O_CLOEXEC).open(path)?;
    anyhow::ensure!(file.metadata()?.is_file() && file.metadata()?.len()<=65536,"MD budget input exceeds bound/type");
    Ok(serde_json::from_reader(file)?)
}
fn number(v:&serde_json::Value,key:&str)->Result<u64>{v[key].as_u64().with_context(||format!("MD budget missing integer {key}"))}
fn string<'a>(v:&'a serde_json::Value,key:&str)->Result<&'a str>{v[key].as_str().with_context(||format!("MD budget missing text {key}"))}
pub fn monotonic_ns()->Result<u64>{
    let mut value=std::mem::MaybeUninit::<libc::timespec>::uninit();
    anyhow::ensure!(unsafe{libc::clock_gettime(libc::CLOCK_MONOTONIC,value.as_mut_ptr())}==0,"native monotonic clock unavailable");
    let value=unsafe{value.assume_init()};
    u64::try_from(value.tv_sec)?.checked_mul(1_000_000_000).and_then(|v|v.checked_add(value.tv_nsec as u64)).context("monotonic clock overflow")
}
impl NativeBudget {
    pub fn from_env(run:Option<&std::sync::Arc<NativeContinuationRun>>,may:bool,output:&Path,startup_stop:Option<u32>)->Result<Option<std::sync::Arc<Self>>>{
        let path=std::env::var_os("PRISM_CONTINUATION_MD_BUDGET_PATH");let digest=std::env::var("PRISM_CONTINUATION_MD_BUDGET_SHA256").ok();
        if path.is_none() && digest.is_none(){anyhow::ensure!(!(may && run.is_some()),"May scientific continuation requires actual persistent cumulative MD900 accounting");return Ok(None);}
        let run=run.context("MD continuation budget requires actual native scientific owner")?;
        anyhow::ensure!(may && run.streaming,"cumulative MD policy currently admits only explicit May native fullcaps lane");
        let path=PathBuf::from(path.context("native budget path absent")?);let digest=digest.context("native budget digest absent")?;
        anyhow::ensure!(path.is_absolute() && prism_nhs::ensemble::manifest::sha256_file(&path)?==digest,"native budget launch file digest mismatch");
        let launch=bounded(&path)?;let policy=&launch["policy"];
        anyhow::ensure!(launch["schema"]=="prism.native_continuation_md_launch.v1" && policy["schema"]=="prism.continuation_md_budget.policy.v1"
            && number(policy,"ceiling_ns")?==CEILING_NS && string(policy,"engine_sha256")?==run.identity.engine_sha256
            && string(policy,"topology_sha256")?==run.identity.topology_sha256,"MD budget scientific identity/ceiling changed");
        anyhow::ensure!(string(&launch,"boot_id")?==std::fs::read_to_string("/proc/sys/kernel/random/boot_id")?.trim(),"native budget belongs to another monotonic boot");
        anyhow::ensure!(string(&launch,"native_argv_sha256")?==format!("{:x}",Sha256::digest(std::fs::read("/proc/self/cmdline")?)),"native budget actual argv differs");
        let prior=number(&launch,"prior_consumed_md_ns")?;let start=number(&launch,"segment_start_monotonic_ns")?;let deadline=number(&launch,"deadline_monotonic_ns")?;
        anyhow::ensure!(prior<CEILING_NS && deadline==start.checked_add(CEILING_NS-prior).context("native budget deadline overflow")? && start<=monotonic_ns()?,"native budget time charge/deadline inconsistent");
        anyhow::ensure!(policy["planned_stop_at_startup_physical_step"].as_u64()==startup_stop.map(u64::from)
            && startup_stop.is_none_or(|k|(1..6).contains(&k) && policy["planned_stop_after_collective_tick"].is_null()),
            "actual native startup stop selector differs from cumulative campaign policy");
        let mode=string(policy,"mode")?;
        anyhow::ensure!((mode=="RESTORE")==run.head.is_some() && matches!(mode,"FRESH_CAPTURE"|"RESTORE"),"native budget fresh/restored mode changed");
        let failure_pin=&policy["source_failure_reconciliation"];
        if mode=="RESTORE" {
            let path=PathBuf::from(string(&policy["historical_branch_contract"],"path")?);
            anyhow::ensure!(prism_nhs::ensemble::manifest::sha256_file(&path)?==string(&policy["historical_branch_contract"],"sha256")?,
                "restored branch budget pin changed");
            let branch=bounded(&path)?;
            anyhow::ensure!(branch["source_failure_authority"].is_null()==failure_pin.is_null(),
                "native failed-source branch cannot bypass its actual conservative campaign charge law");
        }
        if !failure_pin.is_null(){
            anyhow::ensure!(mode=="RESTORE" && policy["source_reservation_retirement"].is_null(),
                "failure reconciliation cannot impersonate native source retirement");
            let branch_path=PathBuf::from(string(&policy["historical_branch_contract"],"path")?);
            anyhow::ensure!(prism_nhs::ensemble::manifest::sha256_file(&branch_path)?==string(&policy["historical_branch_contract"],"sha256")?,
                "failed-source budget historical branch changed");
            let branch:super::scientific_restart::HistoricalBranchContract=serde_json::from_value(bounded(&branch_path)?)?;
            let authority=branch.source_failure_authority.as_ref().context("failure budget lacks native failed-source branch authority")?;
            let receipt=super::scientific_restart::validate_failure_authority(authority,&branch.source_store_root,&branch.source_current_head,&run.identity,false)?;
            anyhow::ensure!(serde_json::to_value(&authority.accounting_receipt)?==*failure_pin
                && policy["prior_segment_receipt"]==*failure_pin
                && policy["source_capacity_disposition"]==serde_json::to_value(&authority.capacity_disposition)?
                && receipt["campaign_id"]==policy["campaign_id"]
                && receipt["source_native_proposal_sha256"]==policy["source_native_proposal_sha256"]
                && receipt["cumulative_consumed_md_ns"].as_u64()==Some(prior)
                && policy["source_scientific_head"]==receipt["scientific_head"],
                "native failed-source restore would reset or substitute prior campaign charge/capacity");
        }
        if let Some(shared)=&run.saved_shared {
            let saved:Option<BudgetAtCaptureV1>=shared.get("native_md_budget_continuation")?;
            let saved=saved.context("restored capsule lacks actual persistent MD campaign owner")?;
            anyhow::ensure!(saved.schema_version==1 && saved.campaign_id==string(policy,"campaign_id")? && saved.engine_sha256==run.identity.engine_sha256
                && saved.topology_sha256==run.identity.topology_sha256 && saved.source_native_proposal_sha256==string(policy,"source_native_proposal_sha256")?
                && saved.ceiling_ns==CEILING_NS && saved.minimum_consumed_md_ns<=prior,"restored capsule would reset/rollback consumed MD budget");
            let native_elapsed:u128=shared.get("shared:active_wall_elapsed_ns")?;
            anyhow::ensure!(native_elapsed<=u128::from(prior),"host cumulative charge is below saved native active elapsed time");
        }
        let token=string(&launch,"run_token")?.to_owned();anyhow::ensure!(!token.is_empty() && token.len()<=128,"invalid native MD budget token");
        Ok(Some(std::sync::Arc::new(Self{policy_sha256:string(&launch,"policy_sha256")?.into(),run_token:token,prior_ns:prior,start_ns:start,deadline_ns:deadline,
            identity:run.identity.clone(),selected:run.head.clone(),output:output.into(),launch,launch_sha256:digest})))
    }
    pub fn at_capture(&self)->Result<BudgetAtCaptureV1>{
        let now=monotonic_ns()?;let elapsed=now.checked_sub(self.start_ns).context("native budget clock regressed")?;
        Ok(BudgetAtCaptureV1{schema_version:1,campaign_id:string(&self.launch["policy"],"campaign_id")?.into(),engine_sha256:self.identity.engine_sha256.clone(),
            topology_sha256:self.identity.topology_sha256.clone(),source_native_proposal_sha256:string(&self.launch["policy"],"source_native_proposal_sha256")?.into(),
            ceiling_ns:CEILING_NS,minimum_consumed_md_ns:self.prior_ns.checked_add(elapsed).context("native cumulative time overflow")?})
    }
    fn actual_process_identity(&self)->Result<serde_json::Value>{
        use std::os::unix::fs::MetadataExt;
        let stat=std::fs::read_to_string("/proc/self/stat")?;
        let end=stat.rfind(')').context("native process stat lacks command terminator")?;
        let fields=stat[end+1..].split_whitespace().collect::<Vec<_>>();
        let start:u64=fields.get(19).context("native process start tick absent")?.parse()?;
        let exe=std::fs::read_link("/proc/self/exe")?;
        let digest=prism_nhs::ensemble::manifest::sha256_file(Path::new("/proc/self/exe"))?;
        anyhow::ensure!(digest==self.identity.engine_sha256,"native failed-source identity ELF differs from actual engine");
        Ok(serde_json::json!({"pid":std::process::id(),"start_ticks":start,"uid":unsafe{libc::geteuid()},
            "exe":exe,"sha256":digest,"boot_id":std::fs::read_to_string("/proc/sys/kernel/random/boot_id")?.trim(),
            "time_namespace_inode":std::fs::metadata("/proc/self/ns/time")?.ino()}))
    }
    pub fn publish_admission(&self,run:&NativeContinuationRun)->Result<()> {
        if let Some(selected)=&self.selected {
            let branch=bounded(&self.output.join("scientific_historical_branch_receipt.json"))?;
            let policy=&self.launch["policy"];
            anyhow::ensure!(branch["contract_sha256"]==policy["historical_branch_contract"]["sha256"] && branch["source_head"]==policy["source_scientific_head"]
                && branch["selected_anchor"]==serde_json::to_value(selected)? && branch["identity"]==serde_json::to_value(&self.identity)?,"actual native historical branch differs from budget authority");
        }
        let next=run.saved_workers.first().map_or(0,|w|w.next_chunk);
        anyhow::ensure!(run.saved_workers.iter().all(|w|w.next_chunk==next),"budget admission lacks common saved host tick");
        let mut logical=Vec::new();let mut physical=Vec::new();
        for stream in 0..self.identity.stream_count as usize {
            logical.push(run.saved_workers.get(stream).map(|w|w.get::<i32>("host:steps_run")).transpose()?.unwrap_or(0));
            physical.push(run.restored_native_step(stream)?);
        }
        let value=serde_json::json!({"schema":"prism.native_continuation_md_budget_admission.v1","launch_contract_sha256":self.launch_sha256,
            "policy_sha256":self.policy_sha256,"run_token":self.run_token,"native_pid":std::process::id(),"native_process_identity":self.actual_process_identity()?,
            "actual_MPS_pipe_directory":std::env::var("CUDA_MPS_PIPE_DIRECTORY").ok(),
            "scientific_identity":self.identity,"scientific_store_root":run.root,"selected_scientific_anchor":self.selected,
            "initial_or_selected_scientific_head":run.epoch.committed_head()?,"common_next_tick":next,
            "logical_window_start_by_stream":logical,"restored_physical_frontiers_by_stream":physical,
            "prior_consumed_md_ns":self.prior_ns,"segment_start_monotonic_ns":self.start_ns,"deadline_monotonic_ns":self.deadline_ns,
            "molecular_clock_independent":true,"new_process_physical_steps_before_admission":0,
            "scope":"fresh all-owner constructor/restored-state resource barrier plus held SCI identity before first new physical propagation"});
        let path=self.output.join("native_continuation_md_budget_admission.json");super::write_json_pretty_atomic(&path,&value)?;
        std::fs::File::open(path)?.sync_all()?;std::fs::File::open(&self.output)?.sync_all()?;Ok(())
    }
    pub fn publish_join(&self,run:&NativeContinuationRun,planned:bool)->Result<()> {
        let head=run.epoch.quiescent_committed_head()?;
        let path=self.output.join("native_continuation_md_join.json");
        super::write_json_pretty_atomic(&path,&serde_json::json!({"schema":"prism.native_continuation_md_join.v1",
            "launch_contract_sha256":self.launch_sha256,"run_token":self.run_token,"native_pid":std::process::id(),
            "scientific_head":head,"scientific_identity":self.identity,"all_worker_threads_joined":true,
            "planned_scientific_stop":planned,"native_md_complete":!planned,"monotonic_ns":monotonic_ns()?,
            "clock_scope":"same-boot CLOCK_MONOTONIC after all MD/required scientific capture and worker joins, before PathB"}))?;
        std::fs::File::open(path)?.sync_all()?;std::fs::File::open(&self.output)?.sync_all()?;Ok(())
    }
    pub fn stop_if_expired(&self,stop:&super::chronicle_raw::DurableStop,run:&NativeContinuationRun,stream:usize)->Result<()> {
        if monotonic_ns()?>=self.deadline_ns {
            stop.request_native_budget_stop(stream)?;run.request_planned_scientific_stop();
        }Ok(())
    }
}
