//! Exact native CA/NMA work and separate conservative-constraint populations. The raw
//! atom/residue rows stay authoritative; no chemical lane or causal effect is inferred.
use super::*;
use super::physical_time::{PhysicalTimeJournalAuthority, VerifiedPhysicalTimeAxis};
#[path="constraint_impulse.rs"]
mod constraint_impulse;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind { Ca, Nma, Constraint }
impl Kind {
    fn key(self)->&'static str {match self {Self::Ca=>"ca_coordinate_work",Self::Nma=>"nma_force_work",Self::Constraint=>"conservative_constraint_impulse"}}
    fn width(self)->usize {match self {Self::Ca=>3,Self::Nma=>12,Self::Constraint=>3}}
    fn manifest(self)->&'static str {match self {Self::Ca=>"prismos_ca_coordinate_work_population_v1",Self::Nma=>"prismos_native_nma_control_work_v1",Self::Constraint=>"prismos_native_conservative_constraint_impulse_v1"}}
    fn window(self)->&'static str {match self {Self::Ca=>"prismos_ca_coordinate_work_window_v1",Self::Nma=>"prismos_native_nma_work_window_v1",Self::Constraint=>"prismos_native_conservative_constraint_window_v1"}}
    fn binary_suffix(self)->&'static str {if self==Self::Constraint {"f64le.zst"}else{"f64le"}}
    fn components(self)->Value {match self {
        Self::Constraint=>json!(["rattle_after_conservative_kick_n_delta_kinetic_kcal_mol","shake_position_delta_kinetic_kcal_mol","rattle_after_conservative_kick_np1_delta_kinetic_kcal_mol"]),
        Self::Ca=>json!(["native_atom_partition_delta_potential_kcal_mol","atom_delta_kinetic_kcal_mol","native_atom_partition_delta_hamiltonian_kcal_mol"]),
        Self::Nma=>json!(["force_n_x","force_n_y","force_n_z","force_np1_x","force_np1_y","force_np1_z",
            "potential_n_kcal_mol","potential_np1_kcal_mol","fixed_coordinate_parameter_work_kcal_mol",
            "scheduled_mode_index_minus_one_if_inactive","scheduled_amplitude","native_protocol_step"])
    }}
    fn axes(self)->Value {match self {
        Self::Constraint=>json!(["step","phase","atom"]),
        Self::Ca=>json!(["physical_step","loaded_topology_atom_index","component"]),
        Self::Nma=>json!(["physical_step","mode_file_residue_index","component"])
    }}
}
fn number(v:&Value)->Result<f64> {let x=v.as_f64().context("native control work number absent")?;ensure!(x.is_finite(),"nonfinite native control work");Ok(x)}

/// Verify the producer's complete native row population in its original order.
/// Exact sums are checks, never replacements for the retained atom/residue rows.
fn verify_step(kind:Kind, bytes:&[u8], population:usize, modes:u64, step:u64, receipt:&Value)->Result<()> {
    if kind==Kind::Constraint {return constraint_impulse::verify_atom_planes_step(bytes,population,step,receipt);}
    let width=kind.width();
    ensure!(population>0 && bytes.len()==population.checked_mul(width).and_then(|v|v.checked_mul(8)).context("native work shape overflow")?,"native work row shape mismatch");
    ensure!(u(&receipt["protocol_current_step"])?==step,"native work scalar step mismatch");
    let mut sums=[0.0f64;3];let mut nonzero=0usize;
    for raw in bytes.chunks_exact(width*8) {
        let mut row=[0.0f64;12];
        for (value,raw_value) in row.iter_mut().zip(raw.chunks_exact(8)) {*value=f64::from_le_bytes(raw_value.try_into().unwrap());}
        ensure!(row.iter().all(|v|v.is_finite()),"nonfinite native control work row");
        match kind {
            Kind::Constraint=>unreachable!("constraint planes handled before atom-major decoding"),
            Kind::Ca=>{for j in 0..3 {sums[j]+=row[j];}},
            Kind::Nma=>{
                ensure!(row[11]==step as f64 && row[..3]==row[3..6],"NMA exact step/two-kick force mismatch");
                ensure!(row[9].fract()==0.0 && row[9]>=-1.0 && row[9]<modes as f64,"NMA mode outside source population");
                for j in 0..3 {sums[j]+=row[6+j];}
                nonzero+=usize::from(row[..3].iter().any(|v|*v!=0.0));
            }
        }
    }
    let fields=match kind {Kind::Constraint=>unreachable!("constraint planes handled above"),Kind::Ca=>["delta_potential_kcal_mol","delta_kinetic_kcal_mol","coordinate_intervention_work_kcal_mol"],
        Kind::Nma=>["potential_n_kcal_mol","potential_np1_kcal_mol","fixed_coordinate_parameter_work_kcal_mol"]};
    for j in 0..3 {ensure!(sums[j].to_bits()==number(&receipt[fields[j]])?.to_bits(),"native work complete population sum differs for {} at step {}",fields[j],step);}
    if kind==Kind::Ca {ensure!(u(&receipt["physical_step_ordinal"])?==step,"CA ordinal differs from native step");}
    else {ensure!(u(&receipt["nonzero_force_residues"])?==nonzero as u64,"NMA active force population differs");}
    Ok(())
}

pub(super) struct NativeControlWork {
    pub summary:Value,
    pub sources_resolved:bool,
    arms:BTreeMap<String,BTreeMap<String,WorkPopulation>>,
}
struct WorkPopulation {
    origin:SourceOrigin,
    source:MeasurementSourcePin,
    rows:BTreeMap<u64,Value>,
    entity_residues:Vec<i32>,
}
impl NativeControlWork {
    pub(super) fn event(&self,arm:&str,step:u64,time:f64,event_clock:Option<&physical_time::RecordedPhysicalTimeEndpoint>,residues:&[u32])->Result<Value> {
        let mut result=serde_json::Map::new();
        for kind in [Kind::Ca,Kind::Nma,Kind::Constraint] {
            let Some(pop)=self.arms.get(arm).and_then(|a|a.get(kind.key())) else {
                let receipt=&self.summary[arm][kind.key()];
                result.insert(kind.key().into(),json!({"status":receipt["status"],"admission":receipt,
                    "raw_values_invented":false}));continue;
            };
            let Some(row)=pop.rows.get(&step) else {
                result.insert(kind.key().into(),json!({"status":"EVENT_STEP_OUTSIDE_RECORDED_WORK_POPULATION","all_work_steps_source":pop.source}));continue;
            };
            let work_time=number(&row["physical_time"]["native_record"]["cumulative_simulated_time_ps"])?;
            if work_time.to_bits()!=time.to_bits() {
                if pop.origin==SourceOrigin::AutoDiscovered {
                    result.insert(kind.key().into(),json!({"status":"EVENT_CLOCK_MISMATCH",
                        "origin":pop.origin,"native_work_population_status":"VERIFIED_RECORDED_POPULATION",
                        "complete_population_source":pop.source,"complete_step_record":row,
                        "baseline_event_time_ps":time,"native_work_time_ps":work_time,
                        "baseline_event_time_authority":event_clock,
                        "baseline_time_authority_class":if event_clock.is_some(){"EXACT_NATIVE_PHYSICAL_TIME"}else{"HISTORICAL_DECLARED_FRAME_TIME"},
                        "native_work_time_authority":row["physical_time"],
                        "association_performed":false,"primary_timebase_changed":false}));
                    continue;
                }
                bail!("required explicit work/event native physical times differ");
            }
            let wanted=residues.iter().copied().collect::<BTreeSet<_>>();
            let indices=pop.entity_residues.iter().enumerate().filter_map(|(i,r)|u32::try_from(*r).ok().filter(|r|wanted.contains(r)).map(|_|i)).collect::<Vec<_>>();
            result.insert(kind.key().into(),json!({"status":"EXACT_NATIVE_EVENT_STEP_ASSOCIATION",
                "complete_population_source":pop.source,"source_line_ordinal":step-1,
                "complete_step_record":row,"event_entity_indices":indices,
                "selection_semantics":if kind==Kind::Constraint {"event_entity_indices address the last atom axis of every native phase plane; no entity-major interleaving or filtering of the complete population"}else{"native event residue membership projected onto explicit original atom/mode-file residue axis; complete population retained"},
                "entity_axis_index":if kind==Kind::Constraint {2}else{1},
                "entity_axis_name":if kind==Kind::Nma {"mode_file_residue_index"}else{"loaded_topology_atom_index"},
                "force_basis":if kind==Kind::Constraint {Value::Null}else{json!("native laboratory coordinates; no unrecorded rotation applied")},
                "accounting_role":row["accounting_role"],"causal_response_inferred":false}));
        }
        Ok(Value::Object(result))
    }
}

#[derive(Clone,Copy,Debug,PartialEq,Eq,Serialize)]
#[serde(rename_all="SCREAMING_SNAKE_CASE")]
enum SourceOrigin {ExplicitManifest,AutoDiscovered}
#[derive(Clone,Copy,Debug,PartialEq,Eq,Serialize)]
#[serde(rename_all="SCREAMING_SNAKE_CASE")]
enum AdmissionStatus {Absent,IncompleteTriplet,MissingExactClock,RejectedRecordedData,VerifiedRecordedPopulation}
struct Candidate {origin:SourceOrigin,input:Option<Value>,receipt:Value}
fn unavailable(origin:SourceOrigin,status:AdmissionStatus,attempts:Value,error:Option<String>)->Value {
    json!({"available":false,"origin":origin,"status":status,"attempted_sources":attempts,
        "error_chain":error,"historical_records_changed":false,"values_synthesized":false})
}
fn arm_root(m:&Value,arm:&str,stream:u16)->Result<(PathBuf,String)> {
    let phase=source(&m["mechanistic_sources"]["arms"][arm]["ghost_phase_binding"])?;phase.verify()?;
    let suffix=format!("_stream{stream:02}_ghost_phase_binding.json");
    let name=phase.path.file_name().and_then(|n|n.to_str()).context("native phase filename absent")?;
    let stem=name.strip_suffix(suffix.as_str()).context("native phase filename/stream mismatch")?;
    Ok((phase.path.parent().context("native producer root absent")?.to_path_buf(),format!("{stem}_stream{stream:02}")))
}
fn probe_paths(paths:&[PathBuf;3])->Candidate {
    let origin=SourceOrigin::AutoDiscovered;
    let attempts=paths.iter().map(|path|match fs::symlink_metadata(path) {
        Ok(meta)=>json!({"attempted_path":path,"present":true,"regular_file":meta.is_file(),"symlink":meta.file_type().is_symlink(),"size_bytes":meta.len()}),
        Err(error) if error.kind()==std::io::ErrorKind::NotFound=>json!({"attempted_path":path,"present":false}),
        Err(error)=>json!({"attempted_path":path,"present":false,"inspection_error":error.to_string()})
    }).collect::<Vec<_>>();
    if attempts.iter().any(|a|a.get("inspection_error").is_some()) {
        return Candidate{origin,input:None,receipt:unavailable(origin,AdmissionStatus::RejectedRecordedData,json!(attempts),Some("optional source path inspection failed".into()))};
    }
    let count=attempts.iter().filter(|a|a["present"]==true).count();
    if count!=3 {
        return Candidate{origin,input:None,receipt:unavailable(origin,if count==0{AdmissionStatus::Absent}else{AdmissionStatus::IncompleteTriplet},json!(attempts),None)};
    }
    let pinned=(||->Result<Value>{
        ensure!(attempts.iter().all(|a|a["regular_file"]==true && a["symlink"]==false),"optional native work triplet requires regular non-symlink files");
        Ok(json!({"manifest":pin(&paths[0])?,"index":pin(&paths[1])?,"binary":pin(&paths[2])?}))
    })();
    match pinned {
        Ok(input)=>Candidate{origin,input:Some(input),receipt:unavailable(origin,AdmissionStatus::RejectedRecordedData,json!(attempts),Some("candidate awaiting full native population admission".into()))},
        Err(error)=>Candidate{origin,input:None,receipt:unavailable(origin,AdmissionStatus::RejectedRecordedData,json!(attempts),Some(format!("{error:#}")))}
    }
}
fn raw_native_clock_authority(path:&Path,stream:u16)->Result<PhysicalTimeJournalAuthority> {
    let source=pin(path)?;
    let mut raw=String::new();BufReader::new(File::open(path)?).read_line(&mut raw)?;
    let first:physical_time::NativePhysicalTimeRecord=serde_json::from_str(&raw)?;
    ensure!(first.schema_kind=="prism_native_physical_time" && first.schema_version==1
        && first.source_stream_id==u64::from(stream) && first.step==1,
        "raw native physical clock first-row identity differs");
    let authority=PhysicalTimeJournalAuthority{source,source_stream_id:first.source_stream_id,
        allocation_generation:first.allocation_generation,epoch_id:first.epoch_id,
        completion:physical_time::PhysicalTimeCompletion::default()};
    // The existing validator verifies every row, epoch/generation, exact dt sum
    // and contiguous native step population. No completion claim is generated.
    VerifiedPhysicalTimeAxis::load(&authority)?;
    Ok(authority)
}

fn probe_kind_sources(root:&Path,prefix:&str,kind:Kind)->Candidate {
    let mut candidate=probe_paths(&[root.join(format!("{prefix}_{}.manifest.json",kind.key())),
        root.join(format!("{prefix}_{}.index.jsonl",kind.key())),root.join(format!("{prefix}_{}.{}",kind.key(),kind.binary_suffix()))]);
    if kind!=Kind::Constraint || candidate.input.is_none() {return candidate;}
    let path=root.join(format!("{prefix}_{}.completion.json",kind.key()));
    let result=(||->Result<Option<MeasurementSourcePin>> {
        if !path.try_exists()? {return Ok(None);}
        let metadata=fs::symlink_metadata(&path)?;
        ensure!(metadata.is_file() && !metadata.file_type().is_symlink(),"constraint completion must be a regular file");
        Ok(Some(pin(&path)?))
    })();
    match result {
        Ok(Some(completion))=>candidate.input.as_mut().unwrap()["completion"]=serde_json::to_value(completion).unwrap(),
        Ok(None)=>{},
        Err(error)=>{
            candidate.input=None;
            candidate.receipt=unavailable(SourceOrigin::AutoDiscovered,AdmissionStatus::RejectedRecordedData,
                json!({"triplet":candidate.receipt["attempted_sources"],"completion_path":path}),Some(format!("{error:#}")));
        }
    }
    candidate
}

fn paired_clock_candidate(m:&Value,stream:u16,allow_constraint_raw_prefix:bool)->Result<Value> {
    let mut candidate=m.clone();
    for arm in ["treatment","control"] {
        let a=&mut candidate["mechanistic_sources"]["arms"][arm];
        if a["physical_time_journal"].is_null() && !a["native_control_work_clock_authority"].is_null() {
            a["physical_time_journal"]=a["native_control_work_clock_authority"].clone();
        }
        if a["physical_time_journal"].is_null() {
            let (root,prefix)=arm_root(m,arm,stream)?;
            let path=root.join(format!("{prefix}_physical_time.manifest.json"));
            if allow_constraint_raw_prefix && !path.try_exists()? {
                let authority=raw_native_clock_authority(&root.join(format!("{prefix}_physical_time.jsonl")),stream)?;
                a["physical_time_journal"]=serde_json::to_value(authority)?;
                a["native_control_work_clock_completion_status"]=json!("NOT_RECORDED_VERIFIED_RAW_NATIVE_JOURNAL_PREFIX");
            } else {
                let manifest=pin(&path).with_context(||format!("optional work lacks exact native clock manifest {}",path.display()))?;
                let value:Value=serde_json::from_reader(File::open(&path)?)?;
                ensure!(value["schema_kind"]=="prism_native_physical_time_manifest" && u(&value["schema_version"])?==1,"native clock manifest schema differs");
                let authority:PhysicalTimeJournalAuthority=serde_json::from_value(value)?;
                ensure!(authority.source_stream_id==stream as u64 && authority.source.path==root.join(format!("{prefix}_physical_time.jsonl")),"clock source differs from native arm root");
                a["physical_time_journal"]=serde_json::to_value(authority)?;
                a["native_control_work_physical_time_manifest"]=serde_json::to_value(manifest)?;
            }
        }
        let authority:PhysicalTimeJournalAuthority=serde_json::from_value(a["physical_time_journal"].clone())?;
        ensure!(authority.source_stream_id==stream as u64,"paired native clock stream differs");
        VerifiedPhysicalTimeAxis::load(&authority)?;
    }
    Ok(candidate)
}
fn optional_status(summary:&Value)->&'static str {
    let rejected=["treatment","control"].iter().any(|arm|["ca_coordinate_work","nma_force_work","conservative_constraint_impulse"].iter().any(|kind|
        matches!(summary[*arm][*kind]["status"].as_str(),Some("INCOMPLETE_TRIPLET"|"MISSING_EXACT_CLOCK"|"REJECTED_RECORDED_DATA"))
        || summary[*arm][*kind]["population_coverage_status"]=="PARTIAL_DURABLE_PREFIX"));
    if rejected {"IMPLEMENTATION_PARTIAL"} else {"IMPLEMENTATION_VERIFIED"}
}
fn write_admission_qc(out:&Path,summary:&Value)->Result<MeasurementSourcePin> {
    let dir=out.join("INTERNAL_QC");fs::create_dir_all(&dir)?;
    let path=dir.join("NATIVE_CONTROL_WORK_ADMISSION.json");
    write_json(&path,&json!({"schema_kind":"prism.native_control_work_optional_admission.v1",
        "status":"IMPLEMENTATION_VERIFIED","scope":"per-population source admission; rejected optional inputs do not invalidate other native measurements",
        "populations":summary,"optional_work_status":optional_status(summary),
        "optional_rejection_invalidates_baseline_measurements":false,
        "explicit_invalid_manifest_references_remain_fatal":true,
        "optional_sources_enter_active_manifest_only_after_full_verification":true,
        "paired_clock_publication_transactional":true}))?;pin(&path)
}

/// Admit each optional population before it can enter global source pins.
/// Discovery failures stay typed QC evidence. Explicit manifest references
/// remain required inputs and propagate errors after recording their reason.
pub(super) fn prepare(m:&mut Value,original:&MeasurementSourcePin,out:&Path)->Result<NativeControlWork> {
    let baseline=m.clone();let stream=manifest_stream_id(m)?;
    let mut candidates=BTreeMap::new();let mut summaries=json!({"treatment":{},"control":{}});
    for arm in ["treatment","control"] {
        for kind in [Kind::Ca,Kind::Nma,Kind::Constraint] {
            let input=&baseline["mechanistic_sources"]["arms"][arm][kind.key()];
            let candidate=if !input.is_null() {
                Candidate{origin:SourceOrigin::ExplicitManifest,input:Some(input.clone()),
                    receipt:unavailable(SourceOrigin::ExplicitManifest,AdmissionStatus::RejectedRecordedData,
                        json!({"declared_reference":input}),Some("required source awaiting admission".into()))}
            } else if baseline["mechanistic_sources"]["arms"][arm]["ghost_phase_binding"].is_null() {
                Candidate{origin:SourceOrigin::AutoDiscovered,input:None,receipt:unavailable(SourceOrigin::AutoDiscovered,AdmissionStatus::Absent,json!([]),Some("no declared native producer root".into()))}
            } else {
                // The phase pin is a pre-existing required source, not an
                // optional sidecar; its invalid identity remains fatal.
                let phase=source(&baseline["mechanistic_sources"]["arms"][arm]["ghost_phase_binding"])?;
                phase.verify()?;
                match arm_root(&baseline,arm,stream) {
                    Ok((root,prefix))=>probe_kind_sources(&root,&prefix,kind),
                    Err(error)=>Candidate{origin:SourceOrigin::AutoDiscovered,input:None,
                        receipt:unavailable(SourceOrigin::AutoDiscovered,AdmissionStatus::RejectedRecordedData,
                            json!({"verified_phase_source":phase,"producer_filename_discovery":false}),
                            Some(format!("optional producer path derivation unavailable: {error:#}")))}
                }
            };
            summaries[arm][kind.key()]=candidate.receipt.clone();
            candidates.insert((arm,kind.key()),candidate);
        }
    }
    let mut arms=BTreeMap::<String,BTreeMap<String,WorkPopulation>>::new();
    let any=candidates.values().any(|c|c.input.is_some());
    if any {
        let allow_constraint_raw_prefix=candidates.iter().any(|((_,key),candidate)|*key==Kind::Constraint.key() && candidate.input.is_some());
        let clock_result=paired_clock_candidate(&baseline,stream,allow_constraint_raw_prefix);
        let mut candidate_manifest=match clock_result {
            Ok(value)=>value,
            Err(error)=>{
                let mut explicit=false;
                for ((arm,key),candidate) in &candidates {
                    if candidate.input.is_none(){continue;}
                    summaries[*arm][*key]=unavailable(candidate.origin,AdmissionStatus::MissingExactClock,
                        candidate.receipt["attempted_sources"].clone(),Some(format!("{error:#}")));
                    explicit|=candidate.origin==SourceOrigin::ExplicitManifest;
                }
                let qc=write_admission_qc(out,&summaries)?;
                if explicit {return Err(error).context("required explicit native work lacks exact paired clock");}
                summaries["_optional_work_status"]=json!(optional_status(&summaries));
                summaries["_admission_qc"]=serde_json::to_value(qc)?;
                return Ok(NativeControlWork{summary:summaries,sources_resolved:false,arms});
            }
        };
        let topology_pin=source(&baseline["mechanistic_sources"]["topology"])?;topology_pin.verify()?;
        let topology:Value=serde_json::from_reader(File::open(&topology_pin.path)?)?;
        ensure!(topology_pin.sha256==text(&baseline["topology_sha256"])?,"native work baseline topology differs");
        let atom_residues:Vec<i32>=serde_json::from_value(topology["atom_analysis_residue_ids"].clone())?;
        let qc_root=out.join("INTERNAL_QC").join("NATIVE_CONTROL_WORK_CANDIDATES");
        fs::create_dir_all(&qc_root)?;
        let mut verified_any=false;
        for arm in ["treatment","control"] {
            for kind in [Kind::Ca,Kind::Nma,Kind::Constraint] {
                let candidate=&candidates[&(arm,kind.key())];let Some(input)=candidate.input.as_ref() else {continue;};
                candidate_manifest["mechanistic_sources"]["arms"][arm][kind.key()]=input.clone();
                let result=evaluate_population(&candidate_manifest,arm,kind,&atom_residues,&qc_root);
                match result {
                    Ok((mut population,mut receipt))=>{
                        let path=out.join(format!("{arm}_{}_measurements.jsonl.zst",kind.key()));
                        ensure!(!path.exists(),"preserve native work destination");
                        fs::rename(&population.source.path,&path)?;
                        population.source=pin(&path)?;
                        population.origin=candidate.origin;
                        receipt["evaluated_source"]=serde_json::to_value(&population.source)?;
                        receipt["origin"]=serde_json::to_value(candidate.origin)?;
                        receipt["status"]=serde_json::to_value(AdmissionStatus::VerifiedRecordedPopulation)?;
                        receipt["attempted_sources"]=candidate.receipt["attempted_sources"].clone();
                        summaries[arm][kind.key()]=receipt;
                        m["mechanistic_sources"]["arms"][arm][kind.key()]=input.clone();
                        arms.entry(arm.into()).or_default().insert(kind.key().into(),population);
                        verified_any=true;
                    }
                    Err(error)=>{
                        let mut receipt=unavailable(candidate.origin,AdmissionStatus::RejectedRecordedData,
                            candidate.receipt["attempted_sources"].clone(),Some(format!("{error:#}")));
                        let partial=qc_root.join(format!("{arm}_{}_measurements.jsonl.zst",kind.key()));
                        if let Ok(meta)=fs::metadata(&partial) {
                            receipt["retained_partial_output"]=json!({"output_path":partial,"size_bytes":meta.len(),
                                "observed_sha256":digest_file(&partial).ok(),"admitted_measurement_population":false});
                        }
                        summaries[arm][kind.key()]=receipt;
                        if candidate.origin==SourceOrigin::ExplicitManifest {
                            write_admission_qc(out,&summaries)?;
                            return Err(error).context("invalid required explicit native control-work population");
                        }
                    }
                }
            }
        }
        if verified_any {
            // Commit BOTH independently verified clocks together, never one
            // arm and never a guessed clock after a rejected optional source.
            // These are work-specific authorities: never change the primary
            // baseline trajectory timebase merely by discovering sidecars.
            for arm in ["treatment","control"] {
                m["mechanistic_sources"]["arms"][arm]["native_control_work_clock_authority"]=candidate_manifest["mechanistic_sources"]["arms"][arm]["physical_time_journal"].clone();
                if !candidate_manifest["mechanistic_sources"]["arms"][arm]["native_control_work_clock_completion_status"].is_null() {
                    m["mechanistic_sources"]["arms"][arm]["native_control_work_clock_completion_status"]=candidate_manifest["mechanistic_sources"]["arms"][arm]["native_control_work_clock_completion_status"].clone();
                }
                if !candidate_manifest["mechanistic_sources"]["arms"][arm]["native_control_work_physical_time_manifest"].is_null() {
                    m["mechanistic_sources"]["arms"][arm]["native_control_work_physical_time_manifest"]=candidate_manifest["mechanistic_sources"]["arms"][arm]["native_control_work_physical_time_manifest"].clone();
                }
            }
        }
    }
    let changed=*m!=baseline;
    if changed {m["native_control_work_source_resolution"]=json!({
        "schema_kind":"prism.native_control_work_source_resolution.v2","original_input_manifest":original,
        "resolution":"only fully verified native populations admitted; independent optional failures retained in INTERNAL_QC",
        "historical_inputs_modified":false,"missing_rows_synthesized":false});}
    summaries["_optional_work_status"]=json!(optional_status(&summaries));
    let qc=write_admission_qc(out,&summaries)?;
    summaries["_admission_qc"]=serde_json::to_value(qc)?;
    Ok(NativeControlWork{summary:summaries,sources_resolved:changed,arms})
}

fn evaluate_population(m:&Value,arm:&str,kind:Kind,atom_residues:&[i32],out:&Path)->Result<(WorkPopulation,Value)> {
let a=&m["mechanistic_sources"]["arms"][arm];let input=&a[kind.key()];
let manifest=source(&input["manifest"])?;let index=source(&input["index"])?;let binary=source(&input["binary"])?;
for p in [&manifest,&index,&binary] {p.verify()?;ensure!(fs::symlink_metadata(&p.path)?.is_file() && !fs::symlink_metadata(&p.path)?.file_type().is_symlink(),"native work source must be a regular nonsymlink file");}
let config:Value=serde_json::from_reader(File::open(&manifest.path)?)?;
let clock:PhysicalTimeJournalAuthority=serde_json::from_value(a["physical_time_journal"].clone()).context("supplied native work requires exact native physical time authority")?;
let axis=VerifiedPhysicalTimeAxis::load(&clock)?;
let stream=manifest_stream_id(m)? as u64;
ensure!(config["schema_kind"]==kind.manifest() && u(&config["schema_version"])?==1
    && (if kind==Kind::Constraint {
        constraint_impulse::verify_config(&config,atom_residues.len())?;
        config["run_id"]==m[format!("{arm}_run_id")]
    } else {config["encoding"]=="little_endian_f64" && config["components"]==kind.components()})
    && config["axis_order"]==kind.axes() && config["topology_sha256"]==m["topology_sha256"]
    && u(&config["source_stream_id"])?==stream && clock.source_stream_id==stream
    && config["epoch_id"]==clock.epoch_id && config["binary_path"]==serde_json::to_value(&binary.path)?
    && config["index_path"]==serde_json::to_value(&index.path)?,"native work schema/topology/stream/epoch/path mismatch");
// Epoch names are not globally unique. Require the same native run
// directory as the verified Ghost/phase and physical-time sources.
let phase=source(&a["ghost_phase_binding"])?;phase.verify()?;
let phase_binding:crate::ghost_phase_lattice::StreamProtocolPhaseBinding=
    serde_json::from_reader(File::open(&phase.path)?)?;phase_binding.verify()?;
let ghost_manifest_pin=source(&a["ghost_segment_manifest"])?;ghost_manifest_pin.verify()?;
let ghost_manifest:crate::ghost_segment::GhostSegmentManifest=
    serde_json::from_reader(File::open(&ghost_manifest_pin.path)?)?;
ghost_manifest.context.verify()?;
ensure!(ghost_manifest.context_digest_sha256==ghost_manifest.context.digest_sha256()?
    && ghost_manifest.context.run_id==text(&m[format!("{arm}_run_id")])?
    && ghost_manifest.context.topology_sha256==text(&m["topology_sha256"])?
    && ghost_manifest.context.stream_id as u64==stream && phase_binding.stream_id as u64==stream
    && ghost_manifest.context.phase_schedule_digest_sha256==phase_binding.phase_schedule_digest_sha256
    && ghost_manifest.context.protocol_state_hash==phase_binding.protocol_state_hash,
    "native control work output root is not bound to the declared run/phase");
let native_root=phase.path.parent().context("native phase root absent")?;
ensure!(ghost_manifest_pin.path.parent().and_then(Path::parent)==Some(native_root),
    "native work and exact Ghost context have different output roots");
for p in [&manifest,&index,&binary,&clock.source] {ensure!(p.path.parent()==Some(native_root),"native control work crosses run output roots");}
if kind==Kind::Constraint {
    return constraint_impulse::evaluate_recorded_population(arm,input,&config,&manifest,&index,&binary,&clock,&axis,atom_residues,out);
}
let (population,modes,entity_residues)=match kind {
    Kind::Constraint=>unreachable!("constraint phase-major population handled above"),
    Kind::Ca=>{ensure!(u(&config["n_atoms"])?==atom_residues.len() as u64,"CA atom axis differs from topology");(atom_residues.len(),0,atom_residues.to_vec())},
    Kind::Nma=>{
        require_sha256(text(&config["modes_sha256"])?,"NMA modes source")?;
        let ids=config["loaded_topology_atom_indices"].as_array().context("NMA loaded topology axis missing")?;
        let mut unique=BTreeSet::new();let mut residues=Vec::with_capacity(ids.len());
        for id in ids {let i=usize::try_from(u(id)?)?;ensure!(unique.insert(i),"duplicate NMA loaded atom index");residues.push(*atom_residues.get(i).context("NMA atom index outside topology")?);}
        ensure!(u(&config["n_residues"])?==ids.len() as u64,"NMA residue axis mismatch");
        (ids.len(),u(&config["n_modes"])?,residues)
    }
};
let configured=u(&config[if kind==Kind::Ca{"configured_physical_steps"}else{"configured_steps"}])?;
let window_capacity=u(&config["device_window_steps"])?;
ensure!(population>0 && configured>0 && window_capacity>0 && (kind==Kind::Ca || modes>0),"empty native work population");
let step_bytes=(population as u64).checked_mul(kind.width() as u64).and_then(|v|v.checked_mul(8)).context("native work byte width overflow")?;
let output=out.join(format!("{arm}_{}_measurements.jsonl.zst",kind.key()));
let mut writer=zstd::stream::write::Encoder::new(BufWriter::new(File::create(&output)?),1)?;
let mut raw=BufReader::new(File::open(&binary.path)?);let mut rows=BTreeMap::new();let mut completed=0u64;let mut byte_offset=0u64;
for (window_ordinal,line_) in BufReader::new(File::open(&index.path)?).lines().enumerate() {
    let line_=line_?;let window:Value=serde_json::from_str(&line_)?;
    let begin=u(&window["start_step_exclusive"])?;let end=u(&window["end_step_inclusive"])?;
    ensure!(window["schema_kind"]==kind.window() && window["encoding"]=="little_endian_f64"
        && window["topology_sha256"]==m["topology_sha256"] && u(&window["source_stream_id"])?==stream
        && window["epoch_id"]==clock.epoch_id && begin==completed && end>begin && end<=configured
        && end-begin<=window_capacity && u(&window["binary_byte_offset"])?==byte_offset,
        "native work index identity/gap/window mismatch");
    if kind==Kind::Ca {
        ensure!(u(&window["n_atoms"])?==population as u64 && u(&window["component_count"])?==3,
            "CA indexed atom/component axis differs");
    } else {
        ensure!(window["native_parameters"]["modes_sha256"]==config["modes_sha256"],"NMA indexed mode source differs");
        number(&window["native_parameters"]["amplification"])?;
        number(&window["native_parameters"]["scan_fraction"])?;
    }
    let length=step_bytes.checked_mul(end-begin).context("native work window size overflow")?;
    ensure!(u(&window["binary_byte_length"])?==length
        && byte_offset.checked_add(length).is_some_and(|end|end<=fs::metadata(&binary.path).map(|m|m.len()).unwrap_or(0)),
        "native work index length mismatch or beyond archive");
    let mut bytes=vec![0u8;usize::try_from(length)?];raw.read_exact(&mut bytes)?;
    ensure!(format!("{:x}",Sha256::digest(&bytes))==text(&window["binary_block_sha256"])?,"native work raw block hash differs");
    let receipts=window["native_step_rows"].as_array().context("native work scalar receipts absent")?;
    ensure!(receipts.len() as u64==end-begin,"native work step receipt population differs");
    for (offset,(step_raw,receipt)) in bytes.chunks_exact(usize::try_from(step_bytes)?).zip(receipts).enumerate() {
        let step=begin+offset as u64+1;verify_step(kind,step_raw,population,modes,step,receipt)?;
        let row=json!({"schema_kind":"prism.native_control_work_measurement.v1","kind":kind.key(),"arm":arm,
            "run_id":m[format!("{arm}_run_id")],"source_stream_id":stream,"epoch_id":clock.epoch_id,
            "completed_step":step,"physical_time":axis.endpoint(step)?,"population":population,
            "raw_source":binary,"raw_byte_offset":byte_offset+offset as u64*step_bytes,"raw_byte_length":step_bytes,
            "raw_step_sha256":format!("{:x}",Sha256::digest(step_raw)),"producer_manifest":manifest,
            "producer_index":index,"producer_index_line_ordinal":window_ordinal,
            "producer_index_line_sha256":format!("{:x}",Sha256::digest(line_.as_bytes())),
            "native_step_receipt":receipt,"native_parameters":window["native_parameters"],
            "native_axis_order":config["axis_order"],"native_components":config["components"],
            "work_identity":if kind==Kind::Ca{"CA coordinate intervention native atom energy partition; not independent atomic causal work"}else{"imposed NMA fixed-coordinate parameter work; both kick forces retained"},
            "time_semantics":if kind==Kind::Ca{"completed transition CA projection/constraints at fixed Hamiltonian parameters"}else{"FORCE_N start and FORCE_NP1 constrained endpoint before later CA projection"},
            "full_raw_population_retained":true,"chemical_lane_inferred":false});
        line(&mut writer,&row)?;rows.insert(step,row);
    }
    completed=end;byte_offset+=length;
}
ensure!(completed>0 && byte_offset==fs::metadata(&binary.path)?.len(),"native work archive has missing/unindexed/trailing bytes");
writer.finish()?.flush()?;
for p in [&manifest,&index,&binary] {p.verify()?;}
let output_pin=pin(&output)?;
let summary=json!({"available":true,"evaluated_source":output_pin,"complete_raw_source":binary,
    "manifest":manifest,"index":index,"physical_time_authority":clock,"recorded_steps":completed,
    "configured_steps":configured,"realized_full_configured_coverage":completed==configured,"entities_per_step":population,
    "component_count":kind.width(),"entity_analysis_residue_ids":entity_residues,
    "all_physical_steps_retained_without_Ghost_or_TSH_filter":true,"scientific_quantity_identity_preserved":true});
Ok((WorkPopulation{origin:SourceOrigin::ExplicitManifest,source:output_pin,rows,entity_residues},summary))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn source16_root()->Result<std::path::PathBuf> { Ok(std::env::var_os("PRISM_MD_RUNS_ROOT").map(std::path::PathBuf::from).ok_or_else(|| anyhow::anyhow!("PRISM_MD_RUNS_ROOT unset: set it to the root that holds 7F5N_source16_physical_diagnostic_records_20260924"))?.join("7F5N_source16_physical_diagnostic_records_20260924")) }
    fn durable_pin(ledger:&Value,path:&Path)->Result<MeasurementSourcePin> {
        let entry=ledger["copied_files"].as_array().context("durable copy ledger absent")?.iter()
            .find(|e|e["durable_copy"]["path"]==serde_json::to_value(path).unwrap()).context("fixture outside exact durable copy ledger")?;
        ensure!(entry["byte_identical"]==true && entry["source"]["sha256"]==entry["durable_copy"]["sha256"],"fixture copy is not identical");
        let p=MeasurementSourcePin{path:path.into(),sha256:text(&entry["durable_copy"]["sha256"])?.into()};p.verify()?;Ok(p)
    }
    #[test]
    #[ignore="requires the original immutable historical 7F5N source manifest"]
    fn historical_pair_discovery_preserves_unavailable_work_and_input_identity()->Result<()> {
        let path_buf=std::env::var_os("PRISM_CONTROL_ROOT").map(std::path::PathBuf::from).ok_or_else(|| anyhow::anyhow!("PRISM_CONTROL_ROOT unset: set it to the root that holds MATCHED_INTERVENTION_7F5N_20260922"))?.join("MATCHED_INTERVENTION_7F5N_20260922/M4_EVENT_INTEGRATION_20260924/mechanistic_inputs/manifest_event_components.json");
        let path=path_buf.as_path();
        let input=pin(path)?;let original:Value=serde_json::from_reader(File::open(path)?)?;
        let mut resolved=original.clone();
        let out=std::env::temp_dir().join(format!("prism_optional_work_absence_{}",std::process::id()));
        ensure!(!out.exists(),"preserve previous optional admission test output");
        let absent=prepare(&mut resolved,&input,&out)?;
        assert!(!absent.sources_resolved);
        assert_eq!(resolved,original);input.verify()?;
        let event=absent.event("treatment",10,0.02,None,&[1,2])?;
        assert_eq!(event["ca_coordinate_work"]["status"],"ABSENT");
        assert_eq!(event["nma_force_work"]["status"],"ABSENT");
        Ok(())
    }
    #[test]
    #[ignore="requires real completed source16 GPU work journals and their durable source ledger"]
    fn completed_source16_ca_nma_work_exact_rows_and_negatives()->Result<()> {
        let root_buf=source16_root()?;
        let root=root_buf.as_path();
        let ledger:Value=serde_json::from_reader(File::open(root.join("OPERATOR_CONTROL/SOURCE_TO_DURABLE_COPY_LEDGER.json"))?)?;
        let mut count=0;
        for (execution,stem) in [("NMA_FORCE_WORK_POPULATION_MEMCHECK_16","2859137_false"),("NMA_ACTIVE_CONTROLS_MEMCHECK_16","2859218_true")] {
            let dir=root.join("COMPLETED_NATIVE_RECORDS").join(execution);
            let receipt:Value=serde_json::from_reader(File::open(dir.join("COMPLETED_NATIVE_RECEIPT.json"))?)?;
            // The original GPU log independently contains the source receipt;
            // archive provenance is checked before reading any numerical rows.
            let log=durable_pin(&ledger,&dir.join(format!("{execution}.log")))?;
            let log_text=fs::read_to_string(&log.path)?;
            let receipt_line=log_text.lines().find_map(|line| {
                let start=line.find("{\"active_asc_and_armed_g26\"")?;
                serde_json::from_str::<Value>(&line[start..]).ok()
            }).context("original native receipt absent from pinned execution log")?;
            ensure!(receipt==receipt_line,"durable receipt differs from original GPU output");
            let ca_name=format!("prism_ca_work_{}_false_true",stem);
            let ca=durable_pin(&ledger,&dir.join(format!("{ca_name}.f64le")))?;
            let idx=durable_pin(&ledger,&dir.join(format!("{ca_name}.index.jsonl")))?;
            let nma=durable_pin(&ledger,&dir.join(format!("prism_nma_native_{stem}.f64le")))?;
            let ca_bytes=fs::read(&ca.path)?;let nma_bytes=fs::read(&nma.path)?;
            ensure!(ca_bytes.len()==70*4865*3*8 && nma_bytes.len()==70*300*12*8,"original fixture population changed");
            let mut ca_step=0usize;
            for line_ in BufReader::new(File::open(&idx.path)?).lines() {
                let window:Value=serde_json::from_str(&line_?)?;
                let start=usize::try_from(u(&window["first_step"])?)?-1;let end=usize::try_from(u(&window["last_step"])?)?;
                ensure!(start==ca_step,"fixture CA step gap");
                let bytes=&ca_bytes[start*4865*24..end*4865*24];
                ensure!(format!("{:x}",Sha256::digest(bytes))==text(&window["sha256"])?,"fixture CA block hash mismatch");
                for row in window["rows"].as_array().context("fixture CA rows")? {
                    verify_step(Kind::Ca,&ca_bytes[ca_step*4865*24..(ca_step+1)*4865*24],4865,0,(ca_step+1) as u64,row)?;ca_step+=1;
                }
            }
            let mut nma_step=0usize;
            for window in receipt["native_nma_work_windows"].as_array().context("native NMA work windows missing")? {
                let start=usize::try_from(u(&window["start"])?)?;let end=usize::try_from(u(&window["end"])?)?;
                ensure!(start==nma_step,"fixture NMA step gap");
                let bytes=&nma_bytes[start*300*96..end*300*96];
                ensure!(format!("{:x}",Sha256::digest(bytes))==text(&window["sha256"])?,"fixture NMA block hash mismatch");
                for row in window["rows"].as_array().context("fixture NMA rows")? {
                    let bytes=&nma_bytes[nma_step*300*96..(nma_step+1)*300*96];
                    verify_step(Kind::Nma,bytes,300,10,(nma_step+1) as u64,row)?;
                    let mut wrong_step=row.clone();wrong_step["protocol_current_step"]=json!(1000);
                    assert!(verify_step(Kind::Nma,bytes,300,10,(nma_step+1) as u64,&wrong_step).is_err());
                    let mut wrong_sum=row.clone();wrong_sum["potential_n_kcal_mol"]=json!(number(&row["potential_n_kcal_mol"])?+1.0);
                    assert!(verify_step(Kind::Nma,bytes,300,10,(nma_step+1) as u64,&wrong_sum).is_err());
                    let mut corrupt=bytes.to_vec();corrupt[0..8].copy_from_slice(&f64::NAN.to_le_bytes());
                    assert!(verify_step(Kind::Nma,&corrupt,300,10,(nma_step+1) as u64,row).is_err());
                    nma_step+=1;
                }
            }
            assert_eq!((ca_step,nma_step),(70,70));count+=ca_step+nma_step;
        }
        println!("source16_native_control_work_rows_verified step_populations={count} scientific_values_synthesized=0");
        Ok(())
    }
    #[test]
    #[ignore="requires pinned historical phase and completed native work bytes; tests file admission, not a new MD population"]
    fn optional_work_partial_missing_clock_and_explicit_strictness()->Result<()> {
        let original_buf=std::env::var_os("PRISM_CONTROL_ROOT").map(std::path::PathBuf::from).ok_or_else(|| anyhow::anyhow!("PRISM_CONTROL_ROOT unset: set it to the root that holds MATCHED_INTERVENTION_7F5N_20260922"))?.join("MATCHED_INTERVENTION_7F5N_20260922/M4_EVENT_INTEGRATION_20260924/mechanistic_inputs/manifest_event_components.json");
        let original_path=original_buf.as_path();
        let original_pin=pin(original_path)?;
        let mut manifest:Value=serde_json::from_reader(File::open(original_path)?)?;
        let directory=std::env::temp_dir().join(format!("prism_optional_work_contract_{}",std::process::id()));
        ensure!(!directory.exists(),"preserve previous optional work test artifacts");fs::create_dir_all(&directory)?;
        let stream=manifest_stream_id(&manifest)?;
        for arm in ["treatment","control"] {
            let p=source(&manifest["mechanistic_sources"]["arms"][arm]["ghost_phase_binding"])?;p.verify()?;
            let root=directory.join(arm);fs::create_dir_all(&root)?;
            let copy=root.join(p.path.file_name().context("phase filename")?);fs::copy(&p.path,&copy)?;
            let copied=pin(&copy)?;assert_eq!(copied.sha256,p.sha256);
            manifest["mechanistic_sources"]["arms"][arm]["ghost_phase_binding"]=serde_json::to_value(copied)?;
            // Only the required phase pin location changes. Scientific bytes
            // remain original; no clock or work manifest is invented.
        }
        let ledger:Value=serde_json::from_reader(File::open(source16_root()?.join("OPERATOR_CONTROL/SOURCE_TO_DURABLE_COPY_LEDGER.json"))?)?;
        let real=durable_pin(&ledger,&source16_root()?.join("COMPLETED_NATIVE_RECORDS/NMA_ACTIVE_CONTROLS_MEMCHECK_16/prism_ca_work_2859218_true_false_true.f64le"))?;
        let (root,prefix)=arm_root(&manifest,"treatment",stream)?;
        let paths=[root.join(format!("{prefix}_ca_coordinate_work.manifest.json")),root.join(format!("{prefix}_ca_coordinate_work.index.jsonl")),root.join(format!("{prefix}_ca_coordinate_work.f64le"))];
        assert_eq!(probe_paths(&paths).receipt["status"],"ABSENT");
        fs::copy(&real.path,&paths[2])?;
        let baseline=manifest.clone();
        let partial=prepare(&mut manifest,&original_pin,&directory.join("partial_output"))?;
        assert_eq!(manifest,baseline);assert!(!partial.sources_resolved);
        assert_eq!(partial.summary["treatment"]["ca_coordinate_work"]["status"],"INCOMPLETE_TRIPLET");
        assert_eq!(partial.summary["treatment"]["nma_force_work"]["status"],"ABSENT");
        assert_eq!(partial.summary["_optional_work_status"],"IMPLEMENTATION_PARTIAL");
        let event=partial.event("treatment",10,0.02,None,&[])?;
        assert_eq!(event["ca_coordinate_work"]["status"],"INCOMPLETE_TRIPLET");
        // Intentional invalid file contracts use original native bytes. These
        // files are negative inputs, never declared new scientific records.
        fs::copy(&real.path,&paths[0])?;fs::copy(&real.path,&paths[1])?;
        let complete=probe_paths(&paths);assert!(complete.input.is_some());
        let missing=prepare(&mut manifest,&original_pin,&directory.join("missing_clock_output"))?;
        assert_eq!(manifest,baseline);assert!(!missing.sources_resolved);
        assert_eq!(missing.summary["treatment"]["ca_coordinate_work"]["status"],"MISSING_EXACT_CLOCK");
        assert_eq!(missing.summary["control"]["ca_coordinate_work"]["status"],"ABSENT");
        let mut explicit=manifest.clone();
        explicit["mechanistic_sources"]["arms"]["treatment"]["ca_coordinate_work"]=complete.input.unwrap();
        assert!(prepare(&mut explicit,&original_pin,&directory.join("explicit_output")).is_err());
        assert!(directory.join("explicit_output/INTERNAL_QC/NATIVE_CONTROL_WORK_ADMISSION.json").is_file());
        let bad=[root.join("nonregular.manifest"),paths[1].clone(),paths[2].clone()];
        fs::create_dir(&bad[0])?;
        assert_eq!(probe_paths(&bad).receipt["status"],"REJECTED_RECORDED_DATA");
        let mut renamed=baseline.clone();
        let phase=source(&renamed["mechanistic_sources"]["arms"]["treatment"]["ghost_phase_binding"])?;
        let alias=root.join("operator_named_phase.json");fs::copy(&phase.path,&alias)?;
        let alias_pin=pin(&alias)?;assert_eq!(alias_pin.sha256,phase.sha256);
        renamed["mechanistic_sources"]["arms"]["treatment"]["ghost_phase_binding"]=serde_json::to_value(alias_pin)?;
        let renamed_before=renamed.clone();
        let alias_result=prepare(&mut renamed,&original_pin,&directory.join("renamed_phase_output"))?;
        assert_eq!(renamed,renamed_before);assert!(!alias_result.sources_resolved);
        assert_eq!(alias_result.summary["treatment"]["ca_coordinate_work"]["status"],"REJECTED_RECORDED_DATA");
        assert_eq!(alias_result.summary["control"]["ca_coordinate_work"]["status"],"ABSENT");
        original_pin.verify()?;real.verify()?;
        println!("optional_work_file_contracts_verified output={} source_bytes_synthesized=0",directory.display());
        Ok(())
    }
    #[test]
    #[ignore="uses an exact time record from the completed native GPU receipt; tests association policy only"]
    fn optional_event_clock_rejection_preserves_other_population()->Result<()> {
        let path=source16_root()?.join("COMPLETED_NATIVE_RECORDS/NMA_ACTIVE_CONTROLS_MEMCHECK_16/COMPLETED_NATIVE_RECEIPT.json");
        let receipt:Value=serde_json::from_reader(File::open(&path)?)?;
        let ledger:Value=serde_json::from_reader(File::open(source16_root()?.join("OPERATOR_CONTROL/SOURCE_TO_DURABLE_COPY_LEDGER.json"))?)?;
        let log=durable_pin(&ledger,&path.parent().unwrap().join("NMA_ACTIVE_CONTROLS_MEMCHECK_16.log"))?;
        let original:Value=fs::read_to_string(&log.path)?.lines().find_map(|l| {
            let start=l.find("{\"active_asc_and_armed_g26\"")?;serde_json::from_str(&l[start..]).ok()
        }).context("native receipt missing")?;assert_eq!(receipt,original);
        let time=&receipt["physical_time_records"][0];let t=number(&time["cumulative_simulated_time_ps"])?;
        let row=json!({"physical_time":{"native_record":time},"original_receipt":pin(&path)?});
        // Build only a consumer policy state from an original GPU row. This
        // does not assert that evaluate_population admitted a new campaign.
        let population=WorkPopulation{origin:SourceOrigin::AutoDiscovered,source:pin(&path)?,
            rows:BTreeMap::from([(1,row)]),entity_residues:vec![]};
        let mut work=NativeControlWork{summary:json!({"treatment":{"nma_force_work":{
            "status":"REJECTED_RECORDED_DATA","error_chain":"deliberate negative admission contract"}}}),
            sources_resolved:false,arms:BTreeMap::from([("treatment".into(),BTreeMap::from([("ca_coordinate_work".into(),population)]))])};
        let mismatch=work.event("treatment",1,t+1.0,None,&[])?;
        assert_eq!(mismatch["ca_coordinate_work"]["status"],"EVENT_CLOCK_MISMATCH");
        assert_eq!(mismatch["ca_coordinate_work"]["native_work_time_ps"],time["cumulative_simulated_time_ps"]);
        assert_eq!(mismatch["ca_coordinate_work"]["baseline_time_authority_class"],"HISTORICAL_DECLARED_FRAME_TIME");
        assert_eq!(mismatch["nma_force_work"]["status"],"REJECTED_RECORDED_DATA");
        let matched=work.event("treatment",1,t,None,&[])?;
        assert_eq!(matched["ca_coordinate_work"]["status"],"EXACT_NATIVE_EVENT_STEP_ASSOCIATION");
        assert_eq!(matched["nma_force_work"]["status"],"REJECTED_RECORDED_DATA");
        work.arms.get_mut("treatment").unwrap().get_mut("ca_coordinate_work").unwrap().origin=SourceOrigin::ExplicitManifest;
        assert!(work.event("treatment",1,t+1.0,None,&[]).is_err());
        Ok(())
    }
}
