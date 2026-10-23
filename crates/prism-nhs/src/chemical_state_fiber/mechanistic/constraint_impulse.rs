//! Lossless native conservative constraint kinetic-impulse populations.
//! Native axes remain [step][phase][atom]; these are not intervention work.
use super::*;

const ACCOUNTING_ROLE:&str="deterministic_constraint_impulse_not_external_heat_or_control_work";
const REDUCTION:&str="host_f64_atom_index_order_from_identical_returned_rows";

pub(super) fn verify_atom_planes_step(bytes:&[u8],atoms:usize,step:u64,receipt:&Value)->Result<()> {
    let plane_bytes=atoms.checked_mul(8).context("constraint atom-plane byte overflow")?;
    ensure!(atoms>0 && bytes.len()==plane_bytes.checked_mul(3).context("constraint step byte overflow")?,"constraint atom/phase population differs");
    ensure!(u(&receipt["physical_step_ordinal"])?==step && u(&receipt["protocol_current_step"])?==step
        && u(&receipt["n_atoms"])?==atoms as u64,"constraint native step/atom identity differs");
    let totals=receipt["phase_delta_kinetic_kcal_mol"].as_array().context("constraint phase totals absent")?;
    ensure!(totals.len()==3,"constraint phase count differs");
    for (phase,plane) in bytes.chunks_exact(plane_bytes).enumerate() {
        let mut sum=0.0f64;
        for raw in plane.chunks_exact(8) {
            let value=f64::from_le_bytes(raw.try_into().unwrap());
            ensure!(value.is_finite(),"nonfinite native constraint atom value");sum+=value;
        }
        ensure!(sum.is_finite() && sum.to_bits()==number(&totals[phase])?.to_bits(),
            "constraint phase total differs from ordered original atom-plane sum");
    }
    Ok(())
}

fn decode_window(window:&Value,compressed:&[u8],atoms:usize,begin:u64,end:u64)->Result<Vec<u8>> {
    ensure!(end>begin && window["schema_kind"]==Kind::Constraint.window() && u(&window["schema_version"])?==1
        && window["encoding"]=="little_endian_f64" && window["codec"]=="zstd"
        && u(&window["compression_level"])?==1 && window["independent_zstd_frame"]==true
        && window["axis_order"]==json!(["physical_step","constraint_phase","loaded_topology_atom_index"])
        && window["component_order"]==Kind::Constraint.components() && u(&window["component_count"])?==3
        && window["native_guard_phase_ids"]==json!([2,4,14]) && window["accounting_role"]==ACCOUNTING_ROLE
        && window["phase_total_reduction_convention"]==REDUCTION && window["all_atoms_and_zero_values_retained"]==true
        && u(&window["n_atoms"])?==atoms as u64 && u(&window["start_step_exclusive"])?==begin
        && u(&window["end_step_inclusive"])?==end,"native constraint window schema/axes/phase identity differs");
    let step_bytes=atoms.checked_mul(24).context("constraint step byte overflow")?;
    let raw_bytes=usize::try_from(end-begin)?.checked_mul(step_bytes).context("constraint window byte overflow")?;
    ensure!(u(&window["compressed_byte_length"])?==compressed.len() as u64
        && u(&window["raw_byte_length"])?==raw_bytes as u64
        && format!("{:x}",Sha256::digest(compressed))==text(&window["compressed_block_sha256"])?,"constraint compressed source hash/size differs");
    let raw=zstd::bulk::decompress(compressed,raw_bytes)?;
    ensure!(raw.len()==raw_bytes && format!("{:x}",Sha256::digest(&raw))==text(&window["raw_block_sha256"])?,"constraint original byte hash/size differs");
    let receipts=window["native_step_rows"].as_array().context("constraint per-step receipts absent")?;
    ensure!(receipts.len()==usize::try_from(end-begin)?,"constraint step receipt population differs");
    for (offset,(step,receipt)) in raw.chunks_exact(step_bytes).zip(receipts).enumerate() {
        verify_atom_planes_step(step,atoms,begin+offset as u64+1,receipt)?;
    }
    Ok(raw)
}

pub(super) fn verify_config(config:&Value,atoms:usize)->Result<()> {
    ensure!(config["schema_kind"]==Kind::Constraint.manifest() && u(&config["schema_version"])?==1
        && config["axis_order"]==Kind::Constraint.axes() && config["component_order"]==Kind::Constraint.components()
        && config["dtype"]=="f64" && config["byte_order"]=="little-endian" && config["journal_bound"]==true
        && config["storage_encoding"]=="concatenated_independent_zstd_frames_of_little_endian_f64"
        && u(&config["compression_level"])?==1 && u(&config["n_atoms"])?==atoms as u64 && atoms>0
        && config["accounting_role"]==ACCOUNTING_ROLE && config["phase_total_reduction"]==REDUCTION
        && u(&config["physical_horizon_steps"])? >0 && u(&config["device_window_steps"])? >0,
        "native conservative constraint manifest identity/axes differ");
    Ok(())
}

/// Standalone population decoding needs real source/clock/topology authority;
/// Ghost or chemical records are only required by the later attachment caller.
#[allow(clippy::too_many_arguments)]
pub(super) fn evaluate_recorded_population(
    arm:&str,input:&Value,config:&Value,manifest:&MeasurementSourcePin,index:&MeasurementSourcePin,
    binary:&MeasurementSourcePin,clock:&PhysicalTimeJournalAuthority,axis:&VerifiedPhysicalTimeAxis,
    atom_residues:&[i32],out:&Path,
)->Result<(WorkPopulation,Value)> {
    verify_config(config,atom_residues.len())?;
    ensure!(serde_json::to_value(&axis.authority)?==serde_json::to_value(clock)?,"constraint decoded clock differs from supplied authority");
    for source in [manifest,index,binary] {source.verify()?;}
    ensure!(config["binary_path"]==serde_json::to_value(&binary.path)? && config["index_path"]==serde_json::to_value(&index.path)?
        && config["epoch_id"]==clock.epoch_id && u(&config["source_stream_id"])?==clock.source_stream_id
        && config["physical_time_journal_path"]==serde_json::to_value(&clock.source.path)?,"constraint raw population source/clock identity differs");
    let completion=if input["completion"].is_null(){None}else{
        let p=source(&input["completion"])?;p.verify()?;
        let metadata=fs::symlink_metadata(&p.path)?;
        ensure!(metadata.is_file() && !metadata.file_type().is_symlink(),"constraint completion must be a regular file");
        ensure!(config["completion_path"]==serde_json::to_value(&p.path)?,"constraint completion path differs");
        let value:Value=serde_json::from_reader(File::open(&p.path)?)?;Some((p,value))
    };
    let configured=u(&config["physical_horizon_steps"])?;let capacity=u(&config["device_window_steps"])?;
    let step_bytes=atom_residues.len().checked_mul(24).context("constraint step size overflow")?;
    let output=out.join(format!("{arm}_conservative_constraint_impulse_measurements.jsonl.zst"));
    let mut writer=zstd::stream::write::Encoder::new(BufWriter::new(File::create(&output)?),1)?;
    let mut archive=BufReader::new(File::open(&binary.path)?);let archive_bytes=fs::metadata(&binary.path)?.len();
    let mut lines=BufReader::new(File::open(&index.path)?);let mut line_raw=Vec::new();
    let mut rows=BTreeMap::new();let(mut completed,mut compressed_offset,mut raw_offset,mut windows,mut index_offset)=(0u64,0u64,0u64,0u64,0u64);
    let mut index_tail=Value::Null;
    loop {
        line_raw.clear();let length=lines.read_until(b'\n',&mut line_raw)?;if length==0 {break;}
        if !line_raw.ends_with(b"\n") {
            ensure!(completion.is_none(),"completed constraint index has an unterminated suffix");
            index_tail=json!({"source":index,"byte_offset":index_offset,"byte_length":length,
                "sha256":format!("{:x}",Sha256::digest(&line_raw)),"status":"UNCOMMITTED_UNTERMINATED_INDEX_SUFFIX_RETAINED"});break;
        }
        let window:Value=serde_json::from_slice(&line_raw)?;
        let begin=u(&window["start_step_exclusive"])?;let end=u(&window["end_step_inclusive"])?;
        ensure!(begin==completed && end>begin && end<=configured && end-begin<=capacity
            && u(&window["window_ordinal"])?==windows && u(&window["compressed_byte_offset"])?==compressed_offset
            && u(&window["raw_byte_offset"])?==raw_offset && window["run_id"]==config["run_id"]
            && window["epoch_id"]==config["epoch_id"] && window["topology_sha256"]==config["topology_sha256"]
            && u(&window["source_stream_id"])?==clock.source_stream_id,
            "constraint window population/offset/run identity differs");
        let length=u(&window["compressed_byte_length"])?;
        ensure!(length>0 && compressed_offset.checked_add(length).is_some_and(|end|end<=archive_bytes),"indexed constraint frame extends outside archive");
        let mut compressed=vec![0u8;usize::try_from(length)?];archive.read_exact(&mut compressed)?;
        let raw=decode_window(&window,&compressed,atom_residues.len(),begin,end)?;
        let clocks=window["native_physical_time_rows"].as_array().context("constraint window lacks exact physical clock rows")?;
        ensure!(clocks.len()==usize::try_from(end-begin)? && window["physical_time_status"]=="EXACT_NATIVE_PER_PHYSICAL_STEP_DT_SUM"
            && window["physical_time_journal_path"]==serde_json::to_value(&clock.source.path)?
            && u(&window["physical_time_allocation_generation"])?==clock.allocation_generation,
            "constraint window clock authority differs");
        for (offset,step_raw) in raw.chunks_exact(step_bytes).enumerate() {
            let step=begin+offset as u64+1;let endpoint=axis.endpoint(step)?;let recorded=&clocks[offset];
            ensure!(u(&recorded["step"])?==step && number(&recorded["applied_dt_ps"])?.to_bits()==endpoint.native_record.applied_dt_ps.to_bits()
                && number(&recorded["cumulative_simulated_time_ps"])?.to_bits()==endpoint.native_record.cumulative_simulated_time_ps.to_bits(),
                "constraint step differs from independently validated native clock");
            let row=json!({"schema_kind":"prism.native_conservative_constraint_measurement.v1","kind":"conservative_constraint_impulse",
                "arm":arm,"run_id":config["run_id"],"source_stream_id":clock.source_stream_id,"epoch_id":clock.epoch_id,
                "topology_sha256":config["topology_sha256"],"completed_step":step,"physical_time":endpoint,
                "population":atom_residues.len(),"native_axis_order":config["axis_order"],"native_components":config["component_order"],
                "accounting_role":ACCOUNTING_ROLE,"phase_total_reduction":REDUCTION,
                "compressed_source":binary,"compressed_window_byte_offset":compressed_offset,"compressed_window_byte_length":length,
                "compressed_window_sha256":window["compressed_block_sha256"],"codec":"zstd",
                "decoded_window_byte_length":raw.len(),"decoded_window_sha256":window["raw_block_sha256"],
                "decoded_step_byte_offset":offset*step_bytes,"decoded_step_byte_length":step_bytes,
                "decoded_step_sha256":format!("{:x}",Sha256::digest(step_raw)),
                "decoded_step_layout":"three contiguous atom planes in native phase order; atom index addresses the last axis",
                "producer_manifest":manifest,"producer_index":index,"producer_index_line_ordinal":windows,
                "producer_index_byte_offset":index_offset,"producer_index_line_byte_length":line_raw.len(),
                "producer_index_line_sha256":format!("{:x}",Sha256::digest(&line_raw)),
                "producer_index_line_hash_scope":"original bytes including line terminator",
                "native_step_receipt":window["native_step_rows"][offset],"native_guard_phase_ids":[2,4,14],
                "full_raw_population_retained":true,"physical_time_is_exact_native_record":true,
                "external_heat_or_intervention_work":false,"causal_response_inferred":false});
            line(&mut writer,&row)?;rows.insert(step,row);
        }
        completed=end;compressed_offset+=length;raw_offset=u(&window["raw_byte_length"])?.checked_add(raw_offset).context("constraint raw offset overflow")?;
        windows+=1;index_offset+=line_raw.len() as u64;
    }
    ensure!(completed>0,"no complete indexed native constraint window");
    let archive_tail=archive_bytes-compressed_offset;
    let completion_pin=if let Some((p,receipt))=&completion {
        ensure!(receipt["schema_kind"]=="prismos_native_conservative_constraint_completion_v1" && u(&receipt["schema_version"])?==1
            && source(&receipt["source_manifest"])?==*manifest && source(&receipt["compressed_population"])?==*binary
            && source(&receipt["window_index"])?==*index && receipt["run_id"]==config["run_id"]
            && receipt["epoch_id"]==clock.epoch_id && receipt["topology_sha256"]==config["topology_sha256"]
            && u(&receipt["source_stream_id"])?==clock.source_stream_id && u(&receipt["recorded_steps"])?==completed
            && u(&receipt["configured_steps"])?==configured && u(&receipt["window_count"])?==windows
            && u(&receipt["raw_bytes"])?==raw_offset && u(&receipt["compressed_bytes"])?==compressed_offset
            && receipt["accounting_role"]==ACCOUNTING_ROLE && receipt["native_phases"]==json!([2,4,14])
            && receipt["realized_full_configured_coverage"]==json!(completed==configured)
            && source(&receipt["physical_time_journal"]["source"])?==clock.source
            && archive_tail==0 && index_tail.is_null(),"constraint completion contradicts recorded population");
        p.verify()?;serde_json::to_value(p)?
    } else {Value::Null};
    writer.finish()?.flush()?;for p in [manifest,index,binary,&clock.source] {p.verify()?;}
    let output_pin=pin(&output)?;let full=completed==configured && completion.is_some();
    let summary=json!({"available":true,"evaluated_source":output_pin,"complete_compressed_source":binary,
        "manifest":manifest,"index":index,"producer_completion_receipt":completion_pin,
        "producer_completion_status":if completion.is_some(){"RECORDED_AND_VERIFIED"}else{"NOT_RECORDED"},
        "recorded_steps":completed,"configured_steps":configured,"realized_full_configured_coverage":completed==configured,
        "population_coverage_status":if full{"COMPLETE_CONFIGURED_POPULATION"}else{"PARTIAL_DURABLE_PREFIX"},
        "raw_bytes_verified":raw_offset,"compressed_bytes_verified":compressed_offset,"window_count":windows,
        "unindexed_compressed_tail":{"source":binary,"byte_offset":compressed_offset,"byte_length":archive_tail,"decoded":false},
        "unterminated_index_tail":index_tail,"physical_time_authority":clock,
        "entities_per_step":atom_residues.len(),"component_count":3,"entity_analysis_residue_ids":atom_residues,
        "entity_axis_index":2,"entity_axis_name":"loaded_topology_atom_index",
        "native_axis_order":config["axis_order"],"native_components":config["component_order"],"accounting_role":ACCOUNTING_ROLE,
        "all_physical_steps_retained_without_Ghost_or_TSH_filter":true,"scientific_quantity_identity_preserved":true,
        "complete_chemical_lane_or_campaign_inferred":false});
    Ok((WorkPopulation{origin:SourceOrigin::ExplicitManifest,source:output_pin,rows,entity_residues:atom_residues.to_vec()},summary))
}

#[cfg(test)]
mod recorded_population_tests {
    use super::*;
    fn actual_sources()->Result<(Value,MeasurementSourcePin,MeasurementSourcePin,MeasurementSourcePin,PhysicalTimeJournalAuthority,Vec<i32>,MeasurementSourcePin)> {
        let root=std::env::var_os("PRISM_CONSTRAINT_RECORDED_ROOT").map(std::path::PathBuf::from).ok_or_else(|| anyhow::anyhow!("PRISM_CONSTRAINT_RECORDED_ROOT unset: set it to the root that holds the recorded 7F5N_source28_production16_20260924_01/output population"))?;
        let prefix="7F5N_A.v3_stream00";
        let manifest=pin(&root.join(format!("{prefix}_conservative_constraint_impulse.manifest.json")))?;
        let index=pin(&root.join(format!("{prefix}_conservative_constraint_impulse.index.jsonl")))?;
        ensure!(manifest.sha256=="7013bc748ce3a1bd9053f88da4c0309f56235708ab7d6424363d24f8910d3929" && index.sha256=="9c16ead41190f503d4cd3be560d5c45ebe0b7ca6db3d46dcc101e5c41c7fed0b","actual source28 fixture receipt/index identity differs");
        let config:Value=serde_json::from_reader(File::open(&manifest.path)?)?;
        let binary=pin(&root.join(format!("{prefix}_conservative_constraint_impulse.f64le.zst")))?;
        let clock=raw_native_clock_authority(&root.join(format!("{prefix}_physical_time.jsonl")),0)?;
        ensure!(clock.source.sha256=="baec1a3517890ac80f3dac33a1e6b13f76ab10bdbcde456d0dbb9c7734455580","actual source28 clock fixture differs");
        let topology=pin(&std::env::var_os("PRISM_INPUTS_ROOT").map(std::path::PathBuf::from).ok_or_else(|| anyhow::anyhow!("PRISM_INPUTS_ROOT unset: set it to the root that holds 7F5N_A/7F5N_A.v3.topology.json"))?.join("7F5N_A/7F5N_A.v3.topology.json"))?;
        ensure!(topology.sha256=="861be5f35178fa5b5f15022bda7f6b6ffa9febf679d4ff1932669a4accf0b4d3" && config["topology_sha256"]==topology.sha256,"recorded topology authority differs");
        let body:Value=serde_json::from_reader(File::open(&topology.path)?)?;
        let atom_residues:Vec<i32>=serde_json::from_value(body["atom_analysis_residue_ids"].clone())?;
        ensure!(u(&body["n_atoms"])?==4865 && atom_residues.len()==4865 && config["run_id"]=="7F5N_A.v3.topology_20260925_064910","native topology population/run differs");
        verify_config(&config,atom_residues.len())?;
        Ok((config,manifest,index,binary,clock,atom_residues,topology))
    }

    #[test]
    #[ignore="requires immutable source28 owner00 journal and an exclusive PRISM_CONSTRAINT_RECORDED_OUTPUT directory"]
    fn source28_complete_recorded_owner_population_exact_phase_atom_clock()->Result<()> {
        let (config,manifest,index,binary,clock,atom_residues,topology)=actual_sources()?;
        let axis=VerifiedPhysicalTimeAxis::load(&clock)?;
        let out=PathBuf::from(std::env::var("PRISM_CONSTRAINT_RECORDED_OUTPUT").context("PRISM_CONSTRAINT_RECORDED_OUTPUT required")?);
        fs::create_dir(&out)?;
        let input=json!({"manifest":manifest,"index":index,"binary":binary});
        let (population,summary)=evaluate_recorded_population("recorded_stream00",&input,&config,&manifest,&index,&binary,&clock,&axis,&atom_residues,&out)?;
        ensure!(population.rows.len()==1870 && summary["recorded_steps"]==1870 && summary["window_count"]==187,"actual indexed prefix population differs");
        ensure!(summary["configured_steps"]==45000 && summary["producer_completion_status"]=="NOT_RECORDED" && summary["population_coverage_status"]=="PARTIAL_DURABLE_PREFIX" && summary["realized_full_configured_coverage"]==false,"interrupted prefix was promoted to full coverage");
        ensure!(summary["raw_bytes_verified"]==json!(1870u64*3*4865*8) && summary["unindexed_compressed_tail"]["byte_length"]==0 && summary["unterminated_index_tail"].is_null(),"raw population/tail preservation differs");
        ensure!(summary["entity_axis_index"]==2 && summary["entity_analysis_residue_ids"]==serde_json::to_value(&atom_residues)? && population.entity_residues==atom_residues,"original atom-axis residue authority differs");
        for (step,row) in &population.rows {
            let endpoint=axis.endpoint(*step)?;
            ensure!(row["physical_time"]==serde_json::to_value(endpoint)? && row["completed_step"]==json!(*step)
                && row["compressed_source"]==serde_json::to_value(&binary)? && row["native_axis_order"]==json!(["step","phase","atom"])
                && row["accounting_role"]==ACCOUNTING_ROLE && row["external_heat_or_intervention_work"]==false
                && row["causal_response_inferred"]==false && row["decoded_step_byte_length"]==json!(4865*3*8),"full population clock/source/quantity changed");
        }
        let mut wrong=clock.clone();wrong.allocation_generation+=1;
        let rejected=evaluate_recorded_population("wrong_clock",&input,&config,&manifest,&index,&binary,&wrong,&axis,&atom_residues,&out).is_err();
        let negatives=json!({"wrong_clock_generation_rejected":rejected});
        ensure!(negatives["wrong_clock_generation_rejected"]==true,"independent decoded-axis authority mismatch accepted");
        fs::write(out.join("ACTUAL_RECORDED_OWNER_RECEIPT.json"),serde_json::to_vec_pretty(&json!({
            "status":"IMPLEMENTATION_VERIFIED","scope":"complete recorded source28 owner00 conservative constraint prefix; no Ghost events or chemical lanes constructed",
            "source_topology":topology,"summary":summary,"negative_contracts":negatives,"MD_executed":false,
            "production_mechanistic_event_attachment_executed":false,"source_record_count":1870,"f64_values_verified":1870u64*3*4865
        }))?)?;
        Ok(())
    }

    #[test]
    #[ignore="requires immutable source28 owner00 original compressed journal"]
    fn source28_original_window_hash_phase_order_and_step_negatives()->Result<()> {
        let (_config,_manifest,index,binary,_clock,atom_residues,_topology)=actual_sources()?;
        let mut first=String::new();BufReader::new(File::open(&index.path)?).read_line(&mut first)?;
        let window:Value=serde_json::from_str(&first)?;
        let mut compressed=vec![0;usize::try_from(u(&window["compressed_byte_length"])?)?];
        File::open(&binary.path)?.read_exact(&mut compressed)?;
        let raw=decode_window(&window,&compressed,atom_residues.len(),0,10)?;
        let mut corrupted=compressed.clone();corrupted[0]^=1;
        ensure!(decode_window(&window,&corrupted,atom_residues.len(),0,10).is_err(),"altered compressed source accepted");
        let mut wrong_step=window.clone();wrong_step["native_step_rows"][0]["physical_step_ordinal"]=json!(2);
        ensure!(decode_window(&wrong_step,&compressed,atom_residues.len(),0,10).is_err(),"wrong original native step accepted");
        let atoms=atom_residues.len();let step_raw=&raw[..atoms*24];let receipt=&window["native_step_rows"][0];
        let mut interleaved=Vec::with_capacity(step_raw.len());
        for atom in 0..atoms {for phase in 0..3 {let offset=(phase*atoms+atom)*8;interleaved.extend_from_slice(&step_raw[offset..offset+8]);}}
        ensure!(verify_atom_planes_step(&interleaved,atoms,1,receipt).is_err(),"incorrect atom-major reinterpretation accepted");
        let mut nonfinite=step_raw.to_vec();nonfinite[..8].copy_from_slice(&f64::NAN.to_le_bytes());
        ensure!(verify_atom_planes_step(&nonfinite,atoms,1,receipt).is_err(),"nonfinite original atom quantity accepted");
        let mut wrong_total=receipt.clone();wrong_total["phase_delta_kinetic_kcal_mol"][0]=json!(0.0);
        ensure!(verify_atom_planes_step(step_raw,atoms,1,&wrong_total).is_err(),"wrong ordered atom sum accepted");
        Ok(())
    }
}
