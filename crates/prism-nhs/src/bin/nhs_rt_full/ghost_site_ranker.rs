//! Native Ghost component/selected-spike binding. No CCL identity or ligand truth input.
use anyhow::{ensure,Context,Result};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{collections::{BTreeMap,BTreeSet},fs::{File,OpenOptions},io::{BufReader,BufWriter,Read,Write},os::unix::fs::FileExt,path::{Path,PathBuf}};
use prism_nhs::{fused_engine::{GpuSpikeEvent,KccData},ghost_phase_lattice::{GhostPhaseLatticeNode,GhostPhaseLatticeComponent,GhostRecordByteLineage},sdst_bridge::LatticeBindingSiteProjection};

#[derive(Clone,Debug,serde::Serialize)]
pub(super) struct SelectedSourceMap {
    pub path:PathBuf,pub sha256:String,pub stream_id:usize,pub raw_records:u64,
    pub selected_records:u64,pub raw_wire_sha256:String,pub selected_wire_sha256:String,
}
fn file_sha(path:&Path)->Result<String>{let mut f=File::open(path)?;let mut h=Sha256::new();let mut b=[0u8;65536];loop{let n=f.read(&mut b)?;if n==0{break}h.update(&b[..n]);}Ok(format!("{:x}",h.finalize()))}
fn sync_parent(path:&Path)->Result<()>{File::open(path.parent().context("output lacks parent")?)?.sync_all()?;Ok(())}
fn write_json(path:&Path,value:&Value)->Result<()>{let mut f=OpenOptions::new().create_new(true).write(true).open(path)?;serde_json::to_writer_pretty(&mut f,value)?;f.write_all(b"\n")?;f.sync_all()?;sync_parent(path)}

/// This is a projection of the SAME selector indices already verified and
/// committed by ChronicleRaw::record_selection, never another selection policy.
pub(super) fn write_selection_map(output:&Path,stream_id:usize,raw:&[GpuSpikeEvent],selected:&[usize])->Result<SelectedSourceMap>{
    let path=output.join(format!("native_ghost_rank_stream{stream_id:02}_selected_raw_ordinals.u64le"));
    let mut f=BufWriter::new(OpenOptions::new().create_new(true).write(true).open(&path)?);
    f.write_all(b"PRGSMAP1")?;f.write_all(&(stream_id as u64).to_le_bytes())?;f.write_all(&(raw.len() as u64).to_le_bytes())?;f.write_all(&(selected.len() as u64).to_le_bytes())?;
    let mut raw_hash=Sha256::new();for s in raw{raw_hash.update(s.to_le_bytes());}
    let mut selected_hash=Sha256::new();let mut seen=vec![0u8;raw.len().checked_add(7).context("sourceordinalbitmapoverflow")?/8];
    for &index in selected{ensure!(index<raw.len(),"selection projection sourceordinaloutofrange");let mask=1u8<<(index%8);ensure!(seen[index/8]&mask==0,"selectionprojectionduplicateoriginalordinal");seen[index/8]|=mask;f.write_all(&(index as u64).to_le_bytes())?;selected_hash.update(raw[index].to_le_bytes());}
    f.flush()?;f.get_ref().sync_all()?;sync_parent(&path)?;
    Ok(SelectedSourceMap{sha256:file_sha(&path)?,path,stream_id,raw_records:raw.len() as u64,selected_records:selected.len() as u64,raw_wire_sha256:format!("{:x}",raw_hash.finalize()),selected_wire_sha256:format!("{:x}",selected_hash.finalize())})
}
fn finite(v:f64,name:&str)->Result<f64>{ensure!(v.is_finite(),"nonfinite native feature {name}");Ok(v)}
fn mean_ca(ids:&[usize],ca:&[[f32;3]],single_precision:bool)->Result<Option<[f32;3]>>{
    if ids.is_empty(){return Ok(None)}
    if single_precision{let mut a=[0.0f32;3];for &i in ids{let c=ca.get(i).context("CA residue identity unavailable")?;for k in 0..3{ensure!(c[k].is_finite(),"nonfinite prepared CA");a[k]+=c[k];}}Ok(Some([a[0]/ids.len()as f32,a[1]/ids.len()as f32,a[2]/ids.len()as f32]))}
    else{let mut a=[0.0f64;3];for &i in ids{let c=ca.get(i).context("CA residue identity unavailable")?;for k in 0..3{ensure!(c[k].is_finite(),"nonfinite prepared CA");a[k]+=c[k]as f64;}}Ok(Some([(a[0]/ids.len()as f64)as f32,(a[1]/ids.len()as f64)as f32,(a[2]/ids.len()as f64)as f32]))}
}
fn weighted_centroid(indices:&[usize],spikes:&[GpuSpikeEvent])->Result<Option<[f32;3]>>{
    let mut a=[0.0f64;3];let mut sum=0.0f64;for &i in indices{let s=spikes.get(i).context("selected spike identity missing")?;let w=finite(s.intensity as f64,"intensity")?.max(1.0e-9);for k in 0..3{a[k]+=finite(s.position[k]as f64,"position")?*w;}sum+=w;}
    Ok(if sum>0.0{Some([(a[0]/sum)as f32,(a[1]/sum)as f32,(a[2]/sum)as f32])}else{None})
}
fn kcc_score(k:&KccData,r:usize)->Result<f64>{
    ensure!(r<k.n_residues,"Ghost driver exceeds actual KCC residue axis");
    let lc=finite(*k.lag_corr_peak.get(r).context("KCC lag axis missing")?as f64,"KCC lag_corr_peak")?as f32;
    let count=*k.residue_count.get(r).context("KCC count axis missing")?;let active=*k.active_causal.get(r).context("KCC causal axis missing")?;
    ensure!(count>0,"Ghost rank driver has no actual KCC observations");
    let causality_frac=active as f32/count as f32;
    let bm=(finite(*k.burst_motion.get(r).context("KCC burst axis missing")?as f64,"KCC burst")?as f32).min(3.0)/3.0;
    let me=(finite(*k.motion_efficiency.get(r).context("KCC motion axis missing")?as f64,"KCC motion")?as f32).min(0.01)/0.01;
    let ds=finite(*k.direction_score.get(r).context("KCC direction axis missing")?as f64,"KCC direction")?as f32;
    Ok((0.3*lc+0.25*causality_frac+0.2*bm+0.15*me+0.1*ds)as f64)
}
fn owner(index:usize,offsets:&[usize],len:usize)->Result<usize>{let s=offsets.partition_point(|&x|x<=index).checked_sub(1).context("selected event lacks owner")?;ensure!(index<len&&index<offsets.get(s+1).copied().unwrap_or(len),"selected event owner interval mismatch");Ok(s)}

/// Bind every selected Ghost component to exact selected events. A source event
/// may overlap several components; membership is explicit for EACH component,
/// while several nodes of ONE component never duplicate the feature population.
/// The window is the declared ordinary observer cadence, not a fitted cutoff.
pub(super) fn enrich_and_rank<F:Fn(usize,i32)->Result<u8>>(
    rows:&mut [Value],projections:&[LatticeBindingSiteProjection],nodes:&[GhostPhaseLatticeNode],
    components:&[GhostPhaseLatticeComponent],lineage:&[GhostRecordByteLineage],spikes:&[GpuSpikeEvent],
    offsets:&[usize],maps:&[Option<SelectedSourceMap>],ca:&[[f32;3]],kcc:&KccData,
    phase:F,lining_cutoff:f32,output:&Path,run_id:&str,
)->Result<Value>{
    ensure!(offsets.len()==8&&maps.len()==8&&offsets.first()==Some(&0)&&offsets.windows(2).all(|w|w[0]<=w[1]),"Ghost ranking requires exact eight source intervals");
    ensure!(rows.len()==projections.len()&&!rows.is_empty(),"Ghost materialized population absent/mismatched");
    ensure!(nodes.len()==lineage.len()&&lining_cutoff.is_finite()&&lining_cutoff>0.0,"Ghost node lineage/lining quantity missing");
    let mut map_files=Vec::new();for (s,m) in maps.iter().enumerate(){let m=m.as_ref().context("missing exact Chronicle-selected original ordinal projection")?;let end=offsets.get(s+1).copied().unwrap_or(spikes.len());ensure!(m.stream_id==s&&m.selected_records==(end-offsets[s])as u64&&file_sha(&m.path)?==m.sha256,"selected map owner/count/hash mismatch");let mut f=File::open(&m.path)?;let mut header=[0u8;32];f.read_exact(&mut header)?;ensure!(&header[..8]==b"PRGSMAP1"&&u64::from_le_bytes(header[8..16].try_into()?)==s as u64&&u64::from_le_bytes(header[16..24].try_into()?)==m.raw_records&&u64::from_le_bytes(header[24..32].try_into()?)==m.selected_records&&f.metadata()?.len()==32+8*m.selected_records,"selected original map header/size mismatch");let mut h=Sha256::new();for e in &spikes[offsets[s]..end]{h.update(e.to_le_bytes());}ensure!(format!("{:x}",h.finalize())==m.selected_wire_sha256,"selected native bytes differ from verified raw selection");map_files.push(f);}
    for (i,(n,l)) in nodes.iter().zip(lineage).enumerate(){ensure!(n.tile_index as usize==i&&l.global_lattice_index==i as u64&&l.stream_id==n.stream_id&&l.step_idx==n.step_idx&&l.site_id==n.site_id&&l.frame_idx==n.frame_idx&&l.record_sha256.len()==64,"Ghost component lost authenticated native node identity");ensure!(n.step_idx>=500&&n.step_idx%500==0,"Ghost node clock incompatible with declared500 observer cadence");}
    let mut buckets:BTreeMap<(usize,u64),Vec<usize>>=BTreeMap::new();
    for (i,s) in spikes.iter().enumerate(){ensure!(s.timestep>=0,"negative selected event step");let sid=owner(i,offsets,spikes.len())?;let end=(s.timestep as u64).checked_add(499).context("event endpoint overflow")?/500*500;buckets.entry((sid,end)).or_default().push(i);phase(i,s.timestep)?;}
    let binding_path=output.join("native_ghost_spike_component_bindings.jsonl");let mut bindings=BufWriter::new(OpenOptions::new().create_new(true).write(true).open(&binding_path)?);let mut binding_count=0u64;let mut failures=Vec::new();
    for (row,p) in rows.iter_mut().zip(projections){
        let cid=usize::try_from(p.source_component_id.0)?;let comp=components.get(cid).context("typed Ghost component index unavailable")?;
        ensure!(row["materialized_site_id"]==serde_json::to_value(p.materialized_site_id)?&&row["source_component_id"]==serde_json::to_value(p.source_component_id)?,"Ghost typed row/projection identity mismatch");
        let mut groups:BTreeMap<(usize,u64),Vec<usize>>=BTreeMap::new();let mut lead_counts:BTreeMap<usize,u64>=BTreeMap::new();
        for &ni in &comp.node_indices{let n=nodes.get(ni as usize).context("component node unavailable")?;groups.entry((n.stream_id as usize,n.step_idx)).or_default().push(ni as usize);if n.causal_lead_residue!=u32::MAX{let r=n.causal_lead_residue as usize;ensure!(r<ca.len(),"Ghost causal lead not on topology residue axis");*lead_counts.entry(r).or_default()+=1;}}
        let mut selected=Vec::new();let mut component_hash=Sha256::new();
        for ((sid,end),nis) in &groups{if let Some(candidates)=buckets.get(&(*sid,*end)){for &i in candidates{let e=&spikes[i];if e.timestep as u64<=end-500||e.timestep as u64>*end{continue}let matched:Vec<usize>=nis.iter().copied().filter(|&ni|{let n=&nodes[ni];(0..3).all(|a|e.position[a].is_finite()&&e.position[a]>=n.aabb_min[a]&&e.position[a]<=n.aabb_max[a])}).collect();if matched.is_empty(){continue}let local=i-offsets[*sid];let mut b=[0u8;8];map_files[*sid].read_exact_at(&mut b,32+(local as u64)*8)?;let raw_ordinal=u64::from_le_bytes(b);ensure!(raw_ordinal<maps[*sid].as_ref().context("map vanished")?.raw_records,"source original ordinal exceeds original population");let evidence=json!({"schema":"prism.native_ghost_selected_spike_component_binding.v1","run_id":run_id,"materialized_site_id":p.materialized_site_id,"analyzed_site_identity":p.analyzed_site_identity,"source_component_id":p.source_component_id,"source_stream_id":sid,"selected_concatenated_ordinal":i,"selected_stream_ordinal":local,"original_raw_stream_ordinal":raw_ordinal,"native_event_sha256":format!("{:x}",Sha256::digest(e.to_le_bytes())),"native_event_protocol_step":e.timestep,"native_event_phase":phase(i,e.timestep)?,"observer_endpoint_step":end,"matched_authenticated_global_node_indices":matched,"membership":"same_native_owner; preceding500step_endpoint_interval; inclusive_actual_node_AABB; union_once_within_component","CCL_identity_used":false});let bytes=serde_json::to_vec(&evidence)?;bindings.write_all(&bytes)?;bindings.write_all(b"\n")?;component_hash.update(&bytes);component_hash.update(b"\n");binding_count+=1;selected.push(i);}}}
        selected.sort_unstable();ensure!(selected.windows(2).all(|w|w[0]!=w[1]),"same selected native event counted twice within one Ghost component");
        row["spike_count"]=json!(selected.len());row["native_spike_binding"]=json!({"schema":"prism.native_ghost_component_feature_binding.v1","binding_path":binding_path,"component_binding_sha256":format!("{:x}",component_hash.finalize()),"selected_spikes":selected.len(),"input_selection_maps":maps,"component_index":cid,"all_component_node_indices":comp.node_indices,"centroid_preserved":p.centroid,"source_identity":"typedGhostcomponent+authenticatedrecordlineage; neverCCLintID","overlap_policy":"one_event_once_per_component; cross_component_overlap_retained"});
        let result=(||->Result<()>{
            ensure!(!selected.is_empty(),"Ghost component has no selected raw event spatial/owner/window binding");ensure!(!lead_counts.is_empty(),"Ghost component has no native measured causal lead for KCC rank features");
            let mut cold=Vec::new();let mut hot=Vec::new();let mut ramp=0u64;let mut cooling=0u64;let mut channels=BTreeSet::new();let mut streams=BTreeSet::new();let mut intensities=Vec::with_capacity(selected.len());let mut total=0.0f64;let mut dn=0.0f64;let mut dd=0.0f64;let mut support=BTreeSet::new();
            for &i in &selected{let e=&spikes[i];match phase(i,e.timestep)?.min(3){0=>cold.push(i),1=>ramp+=1,2=>hot.push(i),_=>cooling+=1}channels.insert(e.spike_source);streams.insert(owner(i,offsets,spikes.len())?);let w=finite(e.intensity as f64,"intensity")?.max(1.0e-9);total+=e.intensity as f64;dn+=finite(e.wd_change as f64,"wd_change")?.abs()*w;dd+=w;intensities.push(e.intensity);ensure!((0..=8).contains(&e.n_residues),"selected event support count invalid");for &r in e.nearby_residues.iter().take(e.n_residues as usize){ensure!(r>=0&&(r as usize)<ca.len(),"selected event support residue unmapped");support.insert(r as usize);}}
            let residues:Vec<usize>=support.into_iter().collect();ensure!(!residues.is_empty(),"Ghost component aggregate selected-event structural support unavailable");let lining=mean_ca(&residues,ca,true)?;let structural=mean_ca(&residues,ca,false)?;
            let mut drivers:Vec<(usize,u64)>=lead_counts.iter().map(|(&r,&n)|(r,n)).collect();drivers.sort_by(|a,b|b.1.cmp(&a.1).then(a.0.cmp(&b.0)));drivers.truncate(3);let candidate_ids:Vec<usize>=drivers.iter().map(|x|x.0).collect();
            let mut scored=Vec::new();for &(r,_) in &drivers{scored.push((r,kcc_score(kcc,r)?));}let mut best=0;for i in 1..scored.len(){if scored[i].1>scored[best].1||(scored[i].1==scored[best].1&&scored[i].0<scored[best].0){best=i}}
            let driver=Some(*ca.get(scored[best].0).context("driverCA missing")?);let mut scores=vec![scored[best].1];for &(r,_) in &drivers{scores.push(kcc_score(kcc,r)?);}let driver_factor=scores.iter().sum::<f64>()/scores.len()as f64;
            let lead_total=drivers.iter().map(|d|d.1).sum::<u64>();let mut burst=0.0f32;let mut driver_weights=Vec::new();for &(r,n) in &drivers{let weight=n as f32/lead_total as f32;let bm=finite(*kcc.burst_motion.get(r).context("KCCburst missing")?as f64,"KCCburst")?as f32;burst+=weight*bm;driver_weights.push(weight);}
            intensities.sort_by(|a,b|a.partial_cmp(b).expect("finite intensity checked"));let q=((intensities.len()-1)as f64*0.90).round()as usize;let q90=intensities[q.min(intensities.len()-1)];let burst_indices:Vec<usize>=selected.iter().copied().filter(|&i|spikes[i].intensity>=q90).collect();let burst_centroid=weighted_centroid(&burst_indices,spikes)?;
            let cold_centroid=weighted_centroid(&cold,spikes)?;let hot_centroid=weighted_centroid(&hot,spikes)?;
            let cutoff_sq=lining_cutoff*lining_cutoff;let near:Vec<usize>=selected.iter().copied().filter(|&i|residues.iter().any(|&r|{let c=ca[r];let e=spikes[i].position;let dx=e[0]-c[0];let dy=e[1]-c[1];let dz=e[2]-c[2];dx*dx+dy*dy+dz*dz<=cutoff_sq})).collect();
            let adjacent=if near.is_empty(){None}else{let mut a=[0.0f64;3];for &i in &near{for k in 0..3{a[k]+=spikes[i].position[k]as f64;}}Some([(a[0]/near.len()as f64)as f32,(a[1]/near.len()as f64)as f32,(a[2]/near.len()as f64)as f32])};
            let views=[Some(p.centroid),lining,driver,adjacent,hot_centroid,cold_centroid,burst_centroid,structural];let points:Vec<[f32;3]>=views.into_iter().flatten().collect();let completeness=points.len()as f64/8.0;let coherence=if points.len()>=2{let mut c=[0.0f64;3];for q in &points{for k in 0..3{c[k]+=q[k]as f64;}}for k in 0..3{c[k]/=points.len()as f64;}let dispersion=points.iter().map(|q|{let dx=q[0]as f64-c[0];let dy=q[1]as f64-c[1];let dz=q[2]as f64-c[2];(dx*dx+dy*dy+dz*dz).sqrt()}).sum::<f64>()/points.len()as f64;1.0/(1.0+dispersion)}else{0.0};
            // Terminal test is on one-indexed topology residue order, matching
            // the existing penalty axis, never an inferred PDB author number.
            let terminal=residues.first().map(|r|r+1<=7).unwrap_or(false);let single=residues.len()<=1;let penalty=(if terminal{-0.35}else{0.0})+(if single{-0.25}else{0.0});let pn=(cold.len()+hot.len()+ramp as usize).max(1)as f64;
            row["phase_manifold_rank_features"]=json!({"total_energy":total,"n_spikes":selected.len(),"stream_consensus":streams.len()as f64/8.0,"phase_modulation":(hot.len()as f64-cold.len()as f64).abs()/pn+0.25*(ramp as f64/pn),"kcc_driver_factor":driver_factor,"desolvation_factor":dn/dd,"burst_factor":burst,"source_diversity":channels.len(),"manifold_completeness":completeness,"centroid_coherence":coherence,"cooling_phase_count":cooling,"terminal_like":terminal,"single_residue":single,"penalty_total":penalty,"legacy_ranker_fields_used":false,"ligand_truth_used":false});
            row["rank_feature_provenance"]=json!({"schema":"prism.native_ghost_rank_feature_provenance.v1","total_energy_quantity":"existingphase-manifold key; sum source-discriminated native intensity, not universal physicalenergy","KCC_source":"actual mergedbeststreamperresidue nativeKCC; selectedGhostcausal-lead counts choose top3 with stabletopologyIDties","driver_topology_residue_ids":candidate_ids,"driver_node_support":drivers,"driver_node_weights":driver_weights,"driver_score_primary_topology_residue":scored[best].0,"residue_ids_zero_based_topology":residues,"CA_coordinates_source":"prepared topology immutable structural reference; not endpointatomiccoordinates","no_unmeasured_lag_filled":true,"view_order":["ghost_native_centroid","lining_residue_CA","native_driver_CA","spikes_near_lining_CA","hot_phase_spikes","cold_phase_spikes","burst_motion_spikes","validation_structural_CA"],"geometry_unchanged":true,"native_event_phase_preserved":true,"node_endpoint_phase_not_substituted_for_event_phase":true});
            row["localization"]=json!({"geometric_voxel_mass":p.centroid,"lining_residues":lining,"driver_residues":driver,"ligand_adjacent_subcluster":adjacent,"hot_phase":hot_centroid,"cold_phase":cold_centroid,"burst_motion":burst_centroid,"validation_structural":structural,"ligand_truth_used":false});
            ensure!(row["therm_class"].as_str().is_some()&&row["hysteresis_asymmetry"].as_f64().is_some()&&row["relative_asymmetry"].as_f64().is_some(),"typedGhostrow lacks actualThermcryptic inputs");Ok(())
        })();
        if let Err(e)=result{row["ranker_status"]=json!("native_feature_binding_unavailable");row["ranker_error"]=json!(format!("{e:#}"));failures.push(json!({"materialized_site_id":p.materialized_site_id,"source_component_id":p.source_component_id,"error":format!("{e:#}")}));}else{row["ranker_status"]=json!("native_ghost_features_bound_pending_common_rank");}
    }
    bindings.flush()?;bindings.get_ref().sync_all()?;sync_parent(&binding_path)?;
    let receipt=json!({"schema":"prism.native_ghost_spike_feature_binding_receipt.v1","run_id":run_id,"component_rows":rows.len(),"assignment_rows":binding_count,"assignment_path":binding_path,"assignment_sha256":file_sha(&binding_path)?,"exact_selection_maps":maps,"selected_source_population":spikes.len(),"full_original_raw_preserved":true,"source_population_filtered_by_this_module":false,"component_overlap_preserved":true,"unavailable_components":failures,"status":if failures.is_empty(){"FEATURES_BOUND"}else{"FEATURE_BINDING_UNAVAILABLE"}});
    write_json(&output.join("native_ghost_spike_feature_binding_receipt.json"),&receipt)?;
    if !failures.is_empty(){write_json(&output.join("native_ghost_feature_rows.unranked.json"),&json!(rows))?;anyhow::bail!("nativeGhostrank featurebinding unavailable for{} components; alltypedrows/bindings retained",failures.len());}
    phase_manifold_rank_rows(rows,true)?;
    let mut cryptic=Vec::new();for(i,row)in rows.iter_mut().enumerate(){let base=row["final_phase_manifold_score"].as_f64().context("phase score missing")?;let h=finite(row["hysteresis_asymmetry"].as_f64().context("actualTherm hysteresis missing")?,"hysteresis")?;let a=finite(row["relative_asymmetry"].as_f64().context("actualTherm relative asymmetry missing")?,"relative_asymmetry")?;let score=base+cryptic_boost(h,a,row["therm_class"].as_str().context("actualThermclass missing")?);row["cryptic_score"]=json!(score);row["cryptic_base_score_source"]=json!("final_phase_manifold_score");row["ranker_status"]=json!("native_ghost_phase_manifold_v1");row["ranker_scope"]=json!("actual selectedrawspikes bound toauthenticatednativeGhostcomponent; commonninefeaturephaseformula+commonThermcrypticboost; noCCLidentity/GT/defaultmissingfeatures");cryptic.push((i,score));}
    cryptic.sort_by(|a,b|b.1.partial_cmp(&a.1).expect("finite strictscore").then(a.0.cmp(&b.0)));for(rank,(i,_))in cryptic.into_iter().enumerate(){rows[i]["cryptic_rank"]=json!(rank+1);}
    Ok(receipt)
}

/// Exact existing phase-manifold normalization, weighting, summation, penalty
/// and ordering body. Strict Ghost caller verifies every required input first;
/// legacy callers retain their existing missing-value policy.
pub(super) fn phase_manifold_rank_rows(ms_sites_json:&mut [Value],strict:bool)->Result<()>{
    if strict {for row in ms_sites_json.iter(){let f=row.get("phase_manifold_rank_features").context("nativeGhostphasefeaturesmissing")?;for key in ["total_energy","stream_consensus","phase_modulation","kcc_driver_factor","desolvation_factor","burst_factor","source_diversity","manifold_completeness","centroid_coherence","penalty_total"]{finite(f.get(key).and_then(Value::as_f64).with_context(||format!("missingactualnativephasefeature{key}"))?,key)?;}ensure!(f.get("terminal_like").and_then(Value::as_bool).is_some()&&f.get("single_residue").and_then(Value::as_bool).is_some(),"actualGhostpenaltyflagsmissing");}}
            let feature_specs: [(&str, f64); 9] = [
                ("total_energy", 0.16),
                ("stream_consensus", 0.14),
                ("phase_modulation", 0.13),
                ("kcc_driver_factor", 0.16),
                ("desolvation_factor", 0.12),
                ("burst_factor", 0.11),
                ("source_diversity", 0.07),
                ("manifold_completeness", 0.07),
                ("centroid_coherence", 0.04),
            ];
            let read_rank_feature = |site: &serde_json::Value, key: &str| -> f64 {
                site.get("phase_manifold_rank_features")
                    .and_then(|f| f.get(key))
                    .and_then(|v| v.as_f64())
                    .filter(|v| v.is_finite())
                    .unwrap_or(0.0)
            };
            let n_sites = ms_sites_json.len();
            let mut raw_scores = vec![0.0_f64; n_sites];
            let mut component_maps: Vec<serde_json::Map<String, serde_json::Value>> =
                (0..n_sites).map(|_| serde_json::Map::new()).collect();

            for &(key, weight) in &feature_specs {
                let values: Vec<f64> = ms_sites_json
                    .iter()
                    .map(|site| read_rank_feature(site, key))
                    .collect();
                let mean = if values.is_empty() {
                    0.0
                } else {
                    values.iter().sum::<f64>() / values.len() as f64
                };
                let var = if values.is_empty() {
                    0.0
                } else {
                    values
                        .iter()
                        .map(|v| {
                            let d = *v - mean;
                            d * d
                        })
                        .sum::<f64>()
                        / values.len() as f64
                };
                let sd = var.sqrt();
                for (i, raw) in values.iter().enumerate() {
                    let z = if sd.is_finite() && sd > 1.0e-12 {
                        (*raw - mean) / sd
                    } else {
                        0.0
                    };
                    let contribution = weight * z;
                    raw_scores[i] += contribution;
                    component_maps[i].insert(
                        key.to_string(),
                        serde_json::json!({
                            "raw": *raw,
                            "z": z,
                            "weight": weight,
                            "contribution": contribution,
                        }),
                    );
                }
            }

            for (i, site) in ms_sites_json.iter_mut().enumerate() {
                let features = site
                    .get("phase_manifold_rank_features")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({}));
                let penalty = features
                    .get("penalty_total")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                let score = raw_scores[i] + penalty;
                site["phase_manifold_score"] = serde_json::json!(score);
                site["final_phase_manifold_score"] = serde_json::json!(score);
                site["ranker_version"] = serde_json::json!("phase_manifold_v1");
                site["phase_manifold_score_explanation"] = serde_json::json!({
                    "components": component_maps[i].clone(),
                    "penalties": {
                        "terminal_like": features
                            .get("terminal_like")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false),
                        "single_residue": features
                            .get("single_residue")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false),
                        "penalty_total": penalty,
                    },
                    "legacy_ranker_fields_used": false,
                    "ligand_truth_used": false,
                    "normalization": "per-target zscore over emitted candidate sites",
                });
            }

            ms_sites_json.sort_by(|a, b| {
                let sa = a
                    .get("final_phase_manifold_score")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(f64::NEG_INFINITY);
                let sb = b
                    .get("final_phase_manifold_score")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(f64::NEG_INFINITY);
                let primary = sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal);
                if primary != std::cmp::Ordering::Equal {
                    return primary;
                }
                let ca = a.get("spike_count").and_then(|v| v.as_u64()).unwrap_or(0);
                let cb = b.get("spike_count").and_then(|v| v.as_u64()).unwrap_or(0);
                let secondary = cb.cmp(&ca);
                if secondary != std::cmp::Ordering::Equal {
                    return secondary;
                }
                let ia = a.get("id").and_then(|v| v.as_i64()).unwrap_or(i64::MAX);
                let ib = b.get("id").and_then(|v| v.as_i64()).unwrap_or(i64::MAX);
                ia.cmp(&ib)
            });
            for (rank, site) in ms_sites_json.iter_mut().enumerate() {
                site["rank"] = serde_json::json!(rank + 1);
                site["phase_manifold_rank"] = serde_json::json!(rank + 1);
            }

    Ok(())
}

pub(super) fn cryptic_boost(hysteresis:f64,rel_asym:f64,therm_class:&str)->f64{
if therm_class == "CRYPTIC" {
                    let hyst_factor = hysteresis.max(0.0).min(1.0);
                    let asym_factor = (rel_asym / 3.0).max(0.0).min(1.0);
                    0.15 * hyst_factor + 0.05 * asym_factor
                } else {
                    0.0
                }
}
