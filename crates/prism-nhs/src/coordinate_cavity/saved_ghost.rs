//! Offline execution of the existing production coordinate-cavity estimator
//! on every original v4 lattice occurrence. Original source files are read only.
use anyhow::{bail, Context, Result};
use clap::Parser;
use crate::coordinate_cavity::{compute_coordinate_cavity_from_ghost_join, CoordinateCavityConfig};
use crate::ghost_coordinate_segment::{join_ghost_lattice_coordinates, load_verified_ghost_coordinate_segments, normalize_legacy_nonperiodic_boundary, GhostCoordinateKey};
use crate::ghost_phase_lattice::load_verified_ghost_segments_from_dir;
use crate::input::PrismPrepTopology;
use rayon::prelude::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Parser, Debug)]
pub struct Args {
    #[arg(long)] pub run_dir: PathBuf,
    #[arg(long)] pub lattice: PathBuf,
    #[arg(long)] pub topology: PathBuf,
    #[arg(long)] pub output: PathBuf,
    #[arg(long, default_value_t = 8)] pub threads: usize,
    /// Explicit component partition. Omitted means the entire original population.
    #[arg(long)] pub component_id: Option<u64>,
    /// Explicit occurrence partition for execution diagnostics, never full coverage.
    #[arg(long)] pub occurrence_limit: Option<usize>,
    /// Reuse only complete component outputs bound to identical inputs/configuration.
    #[arg(long)] pub resume: bool,
}

fn hash(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }
fn regular(path: &Path) -> Result<Vec<u8>> {
    let meta = std::fs::symlink_metadata(path)?;
    if !meta.is_file() || meta.file_type().is_symlink() { bail!("source must be a regular non-symlink file: {}", path.display()); }
    Ok(std::fs::read(path)?)
}
fn sync_dir(path: &Path) -> Result<()> { File::open(path)?.sync_all()?; Ok(()) }
fn publish(source: &Path, destination: &Path) -> Result<()> {
    // Atomic no-clobber publication on the same filesystem.
    std::fs::hard_link(source,destination)?;
    sync_dir(destination.parent().context("publication parent")?)?;
    std::fs::remove_file(source)?;
    sync_dir(source.parent().context("staging parent")?)?;
    Ok(())
}
fn write_new(path: &Path, value: &Value) -> Result<()> {
    let mut ordinal=0;
    let mut staged=path.with_extension("staged.json");
    while staged.exists() { ordinal+=1; staged=path.with_extension(format!("staged.{ordinal}.json")); }
    let mut file = OpenOptions::new().write(true).create_new(true).open(&staged)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?; file.sync_all()?; drop(file);
    publish(&staged,path)
}
fn admit_output(requested: &Path, original: &Path) -> Result<PathBuf> {
    // Validate existing ancestors before any directory or file creation.
    let mut ancestor=requested.to_path_buf(); let mut suffix=Vec::new();
    while !ancestor.exists() {
        suffix.push(ancestor.file_name().context("output has no existing ancestor")?.to_os_string());
        ancestor=ancestor.parent().context("output ancestor missing")?.to_path_buf();
    }
    let mut output=std::fs::canonicalize(ancestor)?;
    for name in suffix.iter().rev() { if name==".." || name=="." { bail!("output traversal rejected"); } output.push(name); }
    if output.starts_with(original) || original.starts_with(&output) { bail!("recovery output must be separate from the original dataset"); }
    Ok(output)
}
fn preserve_uncommitted(path: &Path) -> Result<()> {
    if !path.exists() {return Ok(());}
    let mut ordinal=0;
    loop {
        let archived=path.with_extension(format!("uncommitted.{ordinal}"));
        if !archived.exists() {publish(path,&archived)?;return Ok(());}
        ordinal+=1;
    }
}
fn resource_admission(output: &Path) -> Result<()> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    let path=CString::new(output.as_os_str().as_bytes())?;
    let mut stat=std::mem::MaybeUninit::<libc::statvfs>::uninit();
    if unsafe {libc::statvfs(path.as_ptr(),stat.as_mut_ptr())} != 0 {return Err(std::io::Error::last_os_error().into());}
    let stat=unsafe {stat.assume_init()};
    let available=stat.f_bavail as u128*stat.f_frsize as u128;
    if available < 100u128*1024*1024*1024 {bail!("recovery disk reserve: less than100GiB available; preserve partial outputs and free space before resume");}
    Ok(())
}

fn selected_centroid(n: u64,total:u64,packed:&[u8],expected:u64,seed:[f64;3],origin:f64,spacing:f64)->Result<Option<[f64;3]>> {
    if n==0 || n.checked_mul(n).and_then(|v|v.checked_mul(n))!=Some(total) || packed.len() as u64 != (total+7)/8 {bail!("invalid native grid dimensions/packed bytes");}
    let mut count=0;let mut sum=[0.0;3];
    for linear in 0..total {
        if packed[(linear/8) as usize] & (1 << (linear%8)) == 0 {continue;}
        let xyz=[linear/(n*n),(linear/n)%n,linear%n];
        for axis in 0..3 {sum[axis]+=seed[axis]+origin+xyz[axis] as f64*spacing;}
        count+=1;
    }
    if count!=expected {bail!("selected voxel population is not conserved");}
    Ok((count>0).then(||sum.map(|v|v/count as f64)))
}

/// A measured voxel centroid, not the recorded seed or any component AABB.
fn voxel_centroids(observation: &crate::coordinate_cavity::CoordinateCavityObservation, seed: [f64; 3]) -> Result<Value> {
    let mut rows = Vec::new();
    for grid in &observation.physical_lining.grids {
        if grid.packed_bit_semantics != "linear_index_xyz_z_fastest_lsb0_v1" { bail!("unsupported native grid ordering"); }
        let centroid=selected_centroid(grid.dimension,grid.voxel_count,&grid.selected_component_grid_packed_lsb0,
            grid.selected_cavity_voxel_count,seed,grid.origin_relative_angstrom,grid.spacing_a)?;
        let count=grid.selected_cavity_voxel_count;
        rows.push(json!({"grid_ordinal":grid.grid_ordinal,"spacing_a":grid.spacing_a,"phase_offset_fraction":grid.phase_offset_fraction,
            "selected_voxel_count":count,"volume_a3":grid.selected_cavity_volume_a3,"boundary_area_a2":grid.boundary_area_a2,
            "measured_cavity_centroid_xyz_a":centroid,"selected_grid_sha256":grid.selected_component_grid_sha256,
            "centroid_method":"mean_of_all_native_selected_voxel_coordinates_xyz_z_fastest_v1"}));
    }
    Ok(Value::Array(rows))
}

pub fn run(args: Args) -> Result<()> {
    let started = std::time::Instant::now();
    if args.threads == 0 || args.occurrence_limit == Some(0) { bail!("threads and optional occurrence limit must be positive"); }
    let run_dir = std::fs::canonicalize(&args.run_dir)?;
    let original_root=run_dir.parent().context("original run parent missing")?;
    let output=admit_output(&args.output,original_root)?;
    let lattice_bytes = regular(&args.lattice)?;
    let lattice: Value = serde_json::from_slice(&lattice_bytes)?;
    if lattice["schema_kind"] != "prismos_ghost_phase_lattice_components" || lattice["schema_version"] != 4 || lattice["backend"] != "ghost_phase_lattice_4d" { bail!("requires the original native v4 Ghost lattice"); }
    let target = lattice["target"].as_str().context("lattice target missing")?;
    let run_id = lattice["run_id"].as_str().context("lattice run identity missing")?;
    let recorded_topology = PathBuf::from(lattice["topology_input"].as_str().context("lattice topology missing")?);
    if std::fs::canonicalize(recorded_topology)? != std::fs::canonicalize(&args.topology)? { bail!("topology path differs from original lattice"); }
    let topology_bytes = regular(&args.topology)?;
    let mut topology: PrismPrepTopology = serde_json::from_slice(&topology_bytes)?;
    let topology_sha = hash(&topology_bytes);
    let source_topology=topology.clone();
    let boundary_normalization = normalize_legacy_nonperiodic_boundary(&mut topology)?;
    topology.validate_analysis_residue_axis().map_err(anyhow::Error::msg)?;
    let config = CoordinateCavityConfig::default(); config.validate()?;
    let step_bucket = lattice["config"]["step_bucket_size"].as_u64().context("recorded step bucket missing")?;
    eprintln!("loading verified Ghost records; recorded step bucket={step_bucket}");
    let ghost = load_verified_ghost_segments_from_dir(&run_dir, target, step_bucket)?;
    if !ghost.equality_pass || !ghost.overflow_conservation_pass || ghost.nodes.len() != ghost.record_lineage.len() { bail!("original Ghost population/conservation failed"); }
    for row in &ghost.record_lineage {
        if row.run_id != run_id || row.topology_sha256 != topology_sha { bail!("Ghost run/topology source identity mismatch"); }
    }
    eprintln!("loading exact coordinate manifests for {} records", ghost.nodes.len());
    let coordinates = load_verified_ghost_coordinate_segments(&run_dir, &source_topology)?;
    let joined = join_ghost_lattice_coordinates(&coordinates, &ghost)?;
    eprintln!("coordinate join conserved {} records", joined.conservation.exact_joined_records);
    let by_key = ghost.record_lineage.iter().enumerate().map(|(index,row)| ((row.stream_id,row.step_idx,row.site_id,row.frame_idx),index)).collect::<BTreeMap<_,_>>();
    if by_key.len() != ghost.nodes.len() { bail!("duplicate original Ghost observation identity"); }
    let original_components = lattice["components"].as_array().context("original components absent")?;
    let mut covered = BTreeSet::new();
    let mut component_ids=BTreeSet::new();
    let mut components = Vec::new();
    for component in original_components {
        let id = component["component_id"].as_u64().context("component id missing")?;
        if !component_ids.insert(id) {bail!("duplicate original component ID");}
        let members = component["node_lineage"].as_array().context("component lineage missing")?;
        let mut indices = Vec::new();
        for node in members {
            let stream = u16::try_from(node["stream_id"].as_u64().context("stream missing")?)?;
            let key = (stream,node["step_idx"].as_u64().context("step missing")?,u32::try_from(node["site_id"].as_u64().context("site missing")?)?,node["frame_idx"].as_u64().context("frame missing")?);
            let index = *by_key.get(&key).context("original component member has no authenticated Ghost source")?;
            let projected = &ghost.nodes[index];
            if projected.tile_index as u64 != node["tile_index"].as_u64().context("tile ordinal missing")? || projected.protocol_phase as u64 != node["protocol_phase"].as_u64().context("phase missing")? { bail!("original lattice member differs from the exact record projection"); }
            if !covered.insert(index) { bail!("original node belongs to multiple components"); }
            indices.push(index);
        }
        if indices.len() as u64 != component["n_nodes"].as_u64().context("component count missing")? { bail!("component member count mismatch"); }
        if args.component_id.is_none() || args.component_id == Some(id) { components.push((id,indices)); }
    }
    if covered.len() != ghost.nodes.len() || components.is_empty() { bail!("original full component population not conserved or requested component absent"); }
    let ghost_census=ghost.record_lineage.iter().map(|r| (r.stream_id,r.segment_manifest_sha256.clone(),r.segment_file.clone(),r.segment_file_sha256.clone(),r.segment_context_digest_sha256.clone())).collect::<BTreeSet<_>>();
    let coordinate_census=coordinates.index.iter().map(|(k,r)| (k.stream_id,r.manifest_sha256.clone(),r.segment_file.clone(),r.segment_file_sha256.clone())).collect::<BTreeSet<_>>();
    let executable=std::env::current_exe()?;
    let executable_sha=crate::ensemble::manifest::sha256_file(&executable)?;
    let binding = json!({"schema_kind":"prism_saved_ghost_cavity_input_binding","schema_version":1,"run_id":run_id,"target":target,
        "original_run_dir":run_dir,"original_lattice":args.lattice,"lattice_sha256":hash(&lattice_bytes),"topology":args.topology,"topology_sha256":topology_sha,
        "executing_estimator_binary_sha256":executable_sha,"ghost_source_census":ghost_census,"coordinate_source_census":coordinate_census,
        "coordinate_contexts":coordinates.contexts,"boundary_normalization":boundary_normalization,"estimator_config":config,"step_bucket_size":step_bucket,
        "original_component_count":original_components.len(),"original_occurrence_count":ghost.nodes.len(),"component_partition":args.component_id,
        "occurrence_limit_per_component":args.occurrence_limit,"join_conservation":joined.conservation,
        "scope":"native offline coordinate-derived cavity observations; original runtime finalization is not rewritten; no lambda/control/kinetic qualification invented"});
    let binding_sha = hash(&serde_json::to_vec(&binding)?);
    std::fs::create_dir_all(&output)?;
    let output=std::fs::canonicalize(output)?;
    if admit_output(&output,original_root)? != output {bail!("output identity changed");}
    let lease=OpenOptions::new().write(true).create(true).truncate(false).open(output.join(".recovery.lock"))?;
    if unsafe {libc::flock(lease.as_raw_fd(),libc::LOCK_EX | libc::LOCK_NB)} != 0 {bail!("another recovery owns this output directory");}
    resource_admission(&output)?;
    let binding_path = output.join("saved_ghost_input_binding.json");
    if binding_path.exists() {
        if !args.resume || serde_json::from_slice::<Value>(&regular(&binding_path)?)? != binding { bail!("existing output belongs to different input/scope/configuration"); }
    } else { write_new(&binding_path,&binding)?; }
    let component_dir=output.join("components");std::fs::create_dir_all(&component_dir)?;
    let processed=AtomicU64::new(0);
    let pool=rayon::ThreadPoolBuilder::new().num_threads(args.threads).build()?;
    let summaries: Vec<Value> = pool.install(|| components.par_iter().map(|(component_id,indices)| -> Result<Value> {
        let stem=format!("component_{component_id:06}");
        let rows_path=component_dir.join(format!("{stem}.observations.jsonl.zst"));
        let summary_path=component_dir.join(format!("{stem}.summary.json"));
        if args.resume && summary_path.exists() {
            let summary:Value=serde_json::from_slice(&regular(&summary_path)?)?;
            if summary["input_binding_sha256"] != binding_sha || summary["component_id"] != *component_id || summary["rows_sha256"] != crate::ensemble::manifest::sha256_file(&rows_path)? { bail!("resume component binding or output digest mismatch"); }
            return Ok(summary);
        }
        if summary_path.exists() {bail!("refusing to replace committed component output");}
        if rows_path.exists() {if !args.resume {bail!("uncommitted component rows require --resume");} preserve_uncommitted(&rows_path)?;}
        let temp_path=component_dir.join(format!("{stem}.incomplete.jsonl.zst"));
        // An interrupted uncommitted stream is retained with a new ordinal.
        let mut ordinal=0u64;
        let mut current=temp_path.clone();
        while current.exists() {ordinal+=1;current=component_dir.join(format!("{stem}.incomplete.{ordinal}.jsonl.zst"));}
        let mut writer=zstd::stream::write::Encoder::new(OpenOptions::new().write(true).create_new(true).open(&current)?,3)?;
        let mut positive=0u64;let mut streams=BTreeSet::new();let mut residues=BTreeSet::new();let mut first=u64::MAX;let mut last=0;
        let population=args.occurrence_limit.unwrap_or(indices.len()).min(indices.len());
        for index in indices.iter().take(population) {
            if processed.load(Ordering::Relaxed)%64==0 {resource_admission(&output)?;}
            let source=&ghost.record_lineage[*index];let node=&ghost.nodes[*index];
            let key=GhostCoordinateKey{stream_id:source.stream_id,step_idx:source.step_idx,ghost_site_id:source.site_id,ghost_record_sha256:source.record_sha256.clone()};
            let coordinate=joined.by_key.get(&key).context("exact joined coordinate absent")?;
            let context=coordinates.contexts.get(&source.stream_id).context("coordinate context absent")?;
            let members=source.cluster_residue_membership.cluster_residue_spike_counts.iter().map(|r|r.residue_id).collect::<Vec<_>>();
            let seed=node.centroid_xyz.map(f64::from);
            let observation=compute_coordinate_cavity_from_ghost_join(coordinate,context,seed,&members,&topology,&config)?;
            let centroids=voxel_centroids(&observation,seed)?;
            let has_cavity=observation.sidecar.body.grid_estimates.iter().any(|r|r.cavity_voxels>0);
            if has_cavity {positive+=1;}
            streams.insert(source.stream_id);residues.extend(members.iter().copied());first=first.min(source.step_idx);last=last.max(source.step_idx);
            let row=json!({"schema_kind":"prism_recovered_native_ghost_cavity_observation","schema_version":1,
                "input_binding_sha256":binding_sha,"run_id":run_id,"original_component_id":component_id,
                "status":if has_cavity {"COORDINATE_CAVITY_MEASURED"} else {"OBSERVED_NO_CAVITY_AT_RECORDED_SEED"},
                "recorded_grid_seed_xyz_a":seed,"seed_role":"recorded Ghost coordinate grid seed; never a measured cavity centroid",
                "measured_geometry_by_grid":centroids,"source_ghost_record":source,"exact_coordinate_frame":observation.coordinate_frame,
                "native_coordinate_cavity_sidecar":observation.sidecar,
                "native_ghost_so3_power_spectrum_4x6":node.so3_power_spectrum,
                "native_ghost_so3_cosine_evidence_mask":node.so3_plane_status,
                "native_ghost_so3_scope":"original authenticated record bands; mask excludes padding/invalid planes; values not recomputed or imputed",
                "graph_laplacian_spectral_descriptors":null,"spectral_status":"GRAPH_LAPLACIAN_REQUIRES_SEPARATE_NATIVE_PRODUCER",
                "site_qualification_status":"native physical cavity measurement; downstream native binding-site/chemical-state qualification remains separate"});
            serde_json::to_writer(&mut writer,&row)?;writer.write_all(b"\n")?;
            let n=processed.fetch_add(1,Ordering::Relaxed)+1;
            if n%128==0 {writer.flush()?;eprintln!("computed {n} exact occurrences; elapsed {:.1}s",started.elapsed().as_secs_f64());}
        }
        writer.flush()?;let file=writer.finish()?;file.sync_all()?;drop(file);
        publish(&current,&rows_path)?;
        let summary=json!({"schema_kind":"prism_recovered_native_ghost_component_geometry","schema_version":1,
            "input_binding_sha256":binding_sha,"component_id":component_id,"original_component_occurrences":indices.len(),
            "computed_occurrences":population,"positive_cavity_occurrences":positive,"observed_no_cavity_occurrences":population as u64-positive,
            "component_population_complete":population==indices.len(),"stream_ids":streams,"source_contact_analysis_residue_ids":residues,
            "first_step":first,"last_step":last,"observation_compression":"zstd_level3_lossless_jsonl","observation_rows":rows_path,"rows_sha256":crate::ensemble::manifest::sha256_file(&rows_path)?,
            "geometry_source":"compute_coordinate_cavity_from_ghost_join; exact signed topology and committed molecular coordinates",
            "aabb_geometry_used":false,"spectral_status":"separate producer required; not inferred"});
        write_new(&summary_path,&summary)?;eprintln!("component {component_id} complete: {population} occurrences");Ok(summary)
    }).collect::<Result<Vec<_>>>())?;
    let total: u64=summaries.iter().map(|r|r["computed_occurrences"].as_u64().unwrap()).sum();
    let report=json!({"schema_kind":"prism_saved_ghost_cavity_recovery","schema_version":1,"status":"NATIVE_COORDINATE_GEOMETRY_COMPUTED",
        "input_binding_sha256":binding_sha,"input_binding":binding_path,"original_component_count":original_components.len(),
        "computed_component_count":summaries.len(),"computed_occurrences":total,"original_occurrences":ghost.nodes.len(),
        "whole_original_population_complete":total==ghost.nodes.len() as u64,"elapsed_seconds":started.elapsed().as_secs_f64(),
        "components":summaries,"spectral_status":"separate native graph-spectral producer not executed by this operation"});
    let report_path=output.join("saved_ghost_cavity_recovery.json");
    if report_path.exists() {if !args.resume {bail!("refusing to replace recovery report");}} else {write_new(&report_path,&report)?;}
    println!("{}",json!({"report":report_path,"computed_occurrences":total,"whole_original_population_complete":report["whole_original_population_complete"]}));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn measured_grid_coordinates_and_conservation() {
        // n=2; selected xyz=(0,0,1),(1,0,0), independent expected mean.
        assert_eq!(selected_centroid(2,8,&[0b00010010],2,[10.0,20.0,30.0],-1.0,0.5).unwrap(),Some([9.25,19.0,29.25]));
        assert_eq!(selected_centroid(2,8,&[0],0,[10.0,20.0,30.0],-1.0,0.5).unwrap(),None);
        assert!(selected_centroid(2,8,&[1],0,[0.0;3],0.0,0.5).is_err());
        assert!(selected_centroid(2,8,&[],0,[0.0;3],0.0,0.5).is_err());
    }
    #[test]
    fn rejected_original_destination_causes_no_mutation() {
        let root=std::env::temp_dir().join(format!("prism_saved_ghost_path_{}",std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let original=root.join("original");std::fs::create_dir(&original).unwrap();
        let bad=original.join("must_not_exist/deeper");
        assert!(admit_output(&bad,&original).is_err());assert!(!original.join("must_not_exist").exists());
        assert!(admit_output(&root,&original).is_err());
        let good=root.join("recovered/new");assert_eq!(admit_output(&good,&original).unwrap(),good);
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn publication_never_clobbers_and_retains_uncommitted() {
        let root=std::env::temp_dir().join(format!("prism_saved_ghost_publish_{}",std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let destination=root.join("rows");std::fs::write(&destination,b"existing").unwrap();
        let source=root.join("new");std::fs::write(&source,b"new").unwrap();
        assert!(publish(&source,&destination).is_err());assert_eq!(std::fs::read(&destination).unwrap(),b"existing");
        preserve_uncommitted(&destination).unwrap();assert!(!destination.exists());
        assert_eq!(std::fs::read(root.join("rows.uncommitted.0")).unwrap(),b"existing");
        publish(&source,&destination).unwrap();assert_eq!(std::fs::read(&destination).unwrap(),b"new");
        std::fs::remove_dir_all(root).unwrap();
    }
}
