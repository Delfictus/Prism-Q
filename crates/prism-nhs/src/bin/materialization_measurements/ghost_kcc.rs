//! Exact saved Ghost/KCC association. This module neither detects causal edges
//! nor changes components, scores, promotion, or historical lead identities.
use super::{Manifest, ThermalSchedule, THERMAL_PHASE_NAMES};
use anyhow::{Context, Result};
use prism_nhs::ghost_phase_materializer::GhostNodeLineage;
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const FIELDS: [&str; 14] = [
    "temporal_corr",
    "direction_score",
    "motion_efficiency",
    "burst_motion",
    "phase_shift",
    "causal_lag",
    "lag_corr_peak",
    "local_cov",
    "net_dx",
    "net_dy",
    "net_dz",
    "sum_m",
    "residue_count",
    "active_causal",
];
const METHOD: &str = "argmax_abs_native_kcc_temporal_corr_tie_lowest_residue_id";

#[derive(Deserialize)]
struct Lattice {
    schema_kind: String,
    run_id: String,
    target: String,
    topology_input: String,
    components: Vec<Component>,
}
#[derive(Deserialize)]
struct Component {
    component_id: u64,
    n_nodes: usize,
    node_lineage: Vec<GhostNodeLineage>,
}
#[derive(Clone)]
struct Location {
    path: PathBuf,
    line: u64,
    offset: u64,
    len: usize,
    sha256: String,
}
impl Location {
    fn value(&self) -> Value {
        json!({"path": self.path, "line_1based": self.line,
        "byte_offset": self.offset, "byte_length": self.len, "record_sha256": self.sha256})
    }
}
struct IndexedChronology {
    locations: BTreeMap<(u32, u64), Vec<Location>>,
    sources: Vec<Value>,
}
struct HashingReader<R> {
    reader: R,
    hash: Sha256,
}
impl<R: Read> Read for HashingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.reader.read(buf)?;
        self.hash.update(&buf[..n]);
        Ok(n)
    }
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn same_path(a: &Path, b: &Path) -> Result<bool> {
    Ok(a.canonicalize()? == b.canonicalize()?)
}
fn u64_field(v: &Value, field: &str) -> Result<u64> {
    v.get(field)
        .and_then(Value::as_u64)
        .with_context(|| format!("missing/invalid {field}"))
}
fn stream(v: &Value) -> Result<u32> {
    Ok(u64_field(v, "stream_id")?.try_into()?)
}

fn check_binding(v: &Value, manifest: &Manifest, expected_root: &Path) -> Result<u32> {
    anyhow::ensure!(
        v["record_kind"] == "candidate_binding"
            && v["schema_kind"] == "prismos.kcc_causal_lead_observations.v1",
        "unsupported KCC binding schema"
    );
    anyhow::ensure!(
        v["residue_identity_domain"] == "zero_based_native_topology_residue_index"
            && v["unmeasured_device_sentinel"] == u32::MAX
            && v["selection_method"] == METHOD,
        "KCC identity/selection semantics mismatch"
    );
    let sid = stream(v)?;
    anyhow::ensure!(
        manifest
            .artifacts
            .iter()
            .any(|a| a.present && a.kind == "spikes" && a.stream_id == Some(sid)),
        "KCC binding stream absent from manifest"
    );
    let topo = v["topology_input"]
        .as_str()
        .context("KCC binding lacks topology")?;
    anyhow::ensure!(
        same_path(Path::new(topo), Path::new(&manifest.topology_input))?,
        "KCC topology mismatch"
    );
    if let Some(id) = v.get("run_id") {
        anyhow::ensure!(id.as_str() == Some(&manifest.run_id), "KCC run mismatch");
    }
    if let Some(target) = v.get("target") {
        anyhow::ensure!(
            target.as_str() == Some(&manifest.target),
            "KCC target mismatch"
        );
    }
    anyhow::ensure!(expected_root.exists(), "source evidence root absent");
    Ok(sid)
}

fn index_chronologies(
    paths: &[PathBuf],
    manifest: &Manifest,
    evidence_root: &Path,
) -> Result<IndexedChronology> {
    let mut locations: BTreeMap<(u32, u64), Vec<Location>> = BTreeMap::new();
    let mut sources = Vec::new();
    for path in paths {
        anyhow::ensure!(
            same_path(
                path.parent().context("chronology parent missing")?,
                evidence_root
            )?,
            "chronology is outside explicitly selected manifest evidence root: {}",
            path.display()
        );
        let mut reader = BufReader::new(File::open(path)?);
        let mut buf = Vec::new();
        let mut offset = 0u64;
        let mut line = 0u64;
        let mut digest = Sha256::new();
        let mut binding: Option<(u32, Value)> = None;
        loop {
            buf.clear();
            let n = reader.read_until(b'\n', &mut buf)?;
            if n == 0 {
                break;
            }
            line += 1;
            digest.update(&buf);
            let v: Value = serde_json::from_slice(&buf)
                .with_context(|| format!("{}:{line}", path.display()))?;
            if line == 1 {
                binding = Some((check_binding(&v, manifest, evidence_root)?, v));
            } else {
                let sid = stream(&v)?;
                anyhow::ensure!(
                    Some(sid) == binding.as_ref().map(|b| b.0),
                    "chronology measurement stream disagrees with binding"
                );
                anyhow::ensure!(
                    v["record_kind"] == "measurement",
                    "unsupported chronology record kind"
                );
                let step = u64_field(&v, "measurement_step")?;
                locations.entry((sid, step)).or_default().push(Location {
                    path: path.clone(),
                    line,
                    offset,
                    len: n,
                    sha256: hash(&buf),
                });
            }
            offset += n as u64;
        }
        let (sid, binding) = binding.context("empty chronology")?;
        sources.push(json!({"path":path,"sha256":format!("{:x}",digest.finalize()),"size_bytes":offset,
            "stream_id":sid,"record_count":line.saturating_sub(1),"binding":binding,
            "run_association":"same_manifest_evidence_root_and_topology_stream_binding; absent_embedded_run_id_not_invented"}));
    }
    Ok(IndexedChronology { locations, sources })
}

fn load_snapshot(location: &Location) -> Result<Value> {
    let mut file = File::open(&location.path)?;
    file.seek(SeekFrom::Start(location.offset))?;
    let mut bytes = vec![0; location.len];
    file.read_exact(&mut bytes)?;
    anyhow::ensure!(
        hash(&bytes) == location.sha256,
        "chronology changed between indexing and emission"
    );
    Ok(serde_json::from_slice(&bytes)?)
}
fn snapshot_problem(v: &Value, sid: u32, step: u64) -> Option<&'static str> {
    if stream(v).ok() != Some(sid) {
        return Some("snapshot_stream_mismatch");
    }
    if u64_field(v, "measurement_step").ok() != Some(step) {
        return Some("snapshot_step_mismatch");
    }
    if v["applies_to"] != "current_post_physics_ghost_overlay" {
        return Some("unsupported_snapshot_applies_to");
    }
    if v["selection_method"] != METHOD {
        return Some("unsupported_selection_method");
    }
    let Some(n) = v["kcc_per_residue"]["n_residues"].as_u64() else {
        return Some("invalid_descriptor_shape");
    };
    if n == 0
        || !FIELDS.iter().all(|name| {
            v["kcc_per_residue"][*name]
                .as_array()
                .map(|a| {
                    a.len() == n as usize
                        && a.iter().all(|x| {
                            if *name == "residue_count" || *name == "active_causal" {
                                x.as_u64().map(|i| i <= u32::MAX as u64).unwrap_or(false)
                            } else {
                                x.is_null() || x.as_f64().map(f64::is_finite).unwrap_or(false)
                            }
                        })
                })
                .unwrap_or(false)
        })
    {
        return Some("invalid_descriptor_shape");
    }
    if let Some(raw) = v["kcc_per_residue"].get("lag_provenance") {
        let Ok(rows) = serde_json::from_value::<Vec<prism_nhs::causome::KccLagProvenance>>(raw.clone()) else {
            return Some("invalid_lag_provenance_shape");
        };
        if rows.len()!=n as usize { return Some("invalid_lag_provenance_shape"); }
        for (rid,row) in rows.iter().enumerate() {
            if row.source_stream_id!=Some(sid)
                || Some(row.observation_count as u64)!=v["kcc_per_residue"]["residue_count"][rid].as_u64()
                || Some(row.active_causal_count as u64)!=v["kcc_per_residue"]["active_causal"][rid].as_u64() {
                return Some("lag_provenance_measurement_mismatch");
            }
            for (name,bits) in [("causal_lag",row.causal_lag_f32_bits),("lag_corr_peak",row.lag_corr_peak_f32_bits)] {
                if let Some(bits)=bits {
                    let native=f32::from_bits(bits); let value=&v["kcc_per_residue"][name][rid];
                    if (native.is_finite() && value.as_f64().map(|v|(v as f32).to_bits())!=Some(bits))
                        || (!native.is_finite() && !value.is_null()) {
                        return Some("lag_provenance_measurement_mismatch");
                    }
                }
            }
        }
    }
    None
}
fn descriptor(v: &Value, rid: Option<u32>) -> Option<Value> {
    let rid = rid.filter(|r| *r != u32::MAX)? as usize;
    if rid >= v["kcc_per_residue"]["n_residues"].as_u64()? as usize {
        return None;
    }
    let mut fields: serde_json::Map<String,Value> = FIELDS.iter()
        .map(|name| (name.to_string(),v["kcc_per_residue"][*name][rid].clone())).collect();
    if let Some(row)=v["kcc_per_residue"].get("lag_provenance").and_then(Value::as_array).and_then(|rows|rows.get(rid)) {
        fields.insert("lag_provenance".into(),row.clone());
    }
    Some(Value::Object(fields))
}
fn lead_result(
    v: &Value,
    lead: &Value,
    node: &GhostNodeLineage,
) -> (&'static str, Option<u32>, Option<Value>, Option<Value>) {
    if !lead
        .get("causal_lead_residue")
        .map(|x| x.is_null() || x.as_u64().map(|v| v <= u32::MAX as u64).unwrap_or(false))
        .unwrap_or(false)
    {
        return ("invalid_lead_identity", None, None, None);
    }
    let rid = lead
        .get("causal_lead_residue")
        .and_then(Value::as_u64)
        .and_then(|r| u32::try_from(r).ok())
        .filter(|r| *r != u32::MAX);
    if u64_field(lead, "measurement_step").ok() != Some(node.step_idx) {
        return ("lead_step_mismatch", rid, None, None);
    }
    let kcc = descriptor(v, rid);
    if rid.is_some() && kcc.is_none() {
        return ("lead_residue_out_of_range", rid, None, None);
    }
    if let Some(ref data) = kcc {
        if ["temporal_corr", "residue_count", "active_causal"]
            .iter()
            .any(|f| lead[*f] != data[*f])
        {
            return (
                "lead_descriptor_mismatch",
                rid,
                kcc,
                descriptor(v, node.causal_lead_residue),
            );
        }
    }
    let node_rid = node.causal_lead_residue.filter(|r| *r != u32::MAX);
    let status = if node_rid != rid {
        "lead_identity_mismatch"
    } else if rid.is_none() {
        "exact_key_unmeasured_lead"
    } else {
        "exact_key_and_lead"
    };
    let ghost = if node_rid == rid {
        None
    } else {
        descriptor(v, node_rid)
    };
    (status, rid, kcc, ghost)
}

pub(super) fn emit(
    lattice_path: &Path,
    chronology_paths: &[PathBuf],
    manifest: &Manifest,
    evidence_root: &Path,
    output: &Path,
    schedules: &HashMap<u32, ThermalSchedule>,
) -> Result<PathBuf> {
    let mut lattice_reader = HashingReader {
        reader: BufReader::new(File::open(lattice_path)?),
        hash: Sha256::new(),
    };
    let lattice: Lattice = serde_json::from_reader(&mut lattice_reader)?;
    let lattice_hash = format!("{:x}", lattice_reader.hash.finalize());
    anyhow::ensure!(
        lattice.schema_kind == "prismos_ghost_phase_lattice_components"
            && lattice.run_id == manifest.run_id
            && lattice.target == manifest.target
            && same_path(
                Path::new(&lattice.topology_input),
                Path::new(&manifest.topology_input)
            )?,
        "Ghost lattice run/target/topology/schema mismatch"
    );
    let index = index_chronologies(chronology_paths, manifest, evidence_root)?;
    let mut nodes: BTreeMap<(u32, u64, u32), Vec<(u64, usize, GhostNodeLineage)>> = BTreeMap::new();
    let mut components = BTreeMap::<u64, Value>::new();
    let mut frame_ids = BTreeSet::new();
    let mut steps = BTreeSet::new();
    for (source_component_ordinal, component) in lattice.components.into_iter().enumerate() {
        anyhow::ensure!(
            component.n_nodes == component.node_lineage.len(),
            "component node count mismatch"
        );
        anyhow::ensure!(
            !components.contains_key(&component.component_id),
            "duplicate component identity"
        );
        components.insert(component.component_id,json!({"component_id":component.component_id,"n_nodes":component.n_nodes,"source_component_ordinal":source_component_ordinal,"dispositions":{}}));
        for (ordinal, node) in component.node_lineage.into_iter().enumerate() {
            frame_ids.insert(node.frame_idx);
            steps.insert(node.step_idx);
            nodes
                .entry((u32::from(node.stream_id), node.step_idx, node.site_id))
                .or_default()
                .push((component.component_id, ordinal, node));
        }
    }
    let row_path = output.join("ghost_kcc_observations.prismos_v1.jsonl");
    let mut writer = BufWriter::new(File::create(&row_path)?);
    let mut output_hash = Sha256::new();
    let mut output_bytes = 0u64;
    let mut count = 0u64;
    let mut counts = BTreeMap::<String, u64>::new();
    let mut cached_key = None;
    let mut snapshot: Option<Value> = None;
    let mut cached_snapshot_problem = None;
    let mut leads: BTreeMap<u32, Vec<Value>> = BTreeMap::new();
    let mut used_snapshots = BTreeSet::new();
    let node_keys: BTreeSet<_> = nodes.keys().copied().collect();
    let mut extra_leads = Vec::<Value>::new();
    for ((sid, step, site), group) in nodes {
        let locations = index.locations.get(&(sid, step));
        if cached_key != Some((sid, step)) {
            cached_key = Some((sid, step));
            snapshot = None;
            cached_snapshot_problem = None;
            leads.clear();
            if let Some(locations) = locations {
                used_snapshots.insert((sid, step));
                if locations.len() == 1 {
                    let value = load_snapshot(&locations[0])?;
                    cached_snapshot_problem = snapshot_problem(&value, sid, step);
                    if let Some(rows) = value["leads"].as_array() {
                        for row in rows {
                            if let Some(cluster) = row["cluster_id"]
                                .as_u64()
                                .and_then(|i| u32::try_from(i).ok())
                            {
                                leads.entry(cluster).or_default().push(row.clone());
                            }
                        }
                    }
                    for cluster in leads.keys() {
                        if !node_keys.contains(&(sid, step, *cluster)) {
                            extra_leads
                                .push(json!({"stream_id":sid,"step_idx":step,"cluster_id":cluster,
                                "source":locations[0].value()}));
                        }
                    }
                    snapshot = Some(value);
                }
            }
        }
        let duplicate_nodes = group.len() > 1;
        for (component, ordinal, node) in group {
            let mut disposition = if duplicate_nodes {
                "duplicate_lattice_key"
            } else if locations.is_none() {
                "missing_snapshot"
            } else if locations.unwrap().len() != 1 {
                "duplicate_snapshot"
            } else {
                "missing_cluster_lead"
            };
            let mut rid = None;
            let mut data = None;
            let mut ghost_data = None;
            let mut lead_record = Value::Null;
            if !duplicate_nodes {
                if let Some(ref v) = snapshot {
                    if let Some(problem) = cached_snapshot_problem {
                        disposition = problem;
                    } else if let Some(rows) = leads.get(&site) {
                        if rows.len() != 1 {
                            disposition = "duplicate_cluster_lead";
                        } else {
                            lead_record = rows[0].clone();
                            (disposition, rid, data, ghost_data) = lead_result(v, &rows[0], &node);
                        }
                    }
                }
            }
            let phase = i32::try_from(step)
                .ok()
                .and_then(|s| schedules.get(&sid).map(|p| p.phase(s)))
                .unwrap_or(6);
            let zero_loss = node.causal_lead_residue.is_none()
                && rid == Some(0)
                && disposition == "lead_identity_mismatch";
            let row = json!({"schema_kind":"prismos_ghost_kcc_node_observation","schema_version":1,
                "component_id":component,"node_lineage_ordinal":ordinal,"node_lineage":node,
                "join_key":{"stream_id":sid,"step_idx":step,"site_id":site},"disposition":disposition,
                "ghost_causal_lead_residue":node.causal_lead_residue,"chronology_causal_lead_residue":rid,
                "chronology_lead_record":lead_record,"chronology_lead_descriptor":data,
                "ghost_lead_descriptor":ghost_data,
                "ghost_descriptor_relationship":if node.causal_lead_residue==rid && rid.is_some(){"same_as_chronology_lead_descriptor"}else{"separate_or_unavailable"},
                "known_parser_discrepancy":if zero_loss {Some("matches_historical_lattice_zero_sentinel_loss; original_node_preserved_and_raw_record_not_reverified")}else{None},
                "prescribed_thermal_phase":THERMAL_PHASE_NAMES[phase],
                "ghost_protocol_phase":node.protocol_phase,
                "chronological_authority":"step_idx; frame_idx_preserved_without_reinterpreting_as_timestamp",
                "snapshot_sources":locations.map(|l|l.iter().map(Location::value).collect::<Vec<_>>()).unwrap_or_default(),
                "descriptor_array_index":rid,"descriptor_scope":"whole_run_streaming_KCC_through_matched_endpoint",
                "unavailable_semantics":"source_JSON_null_preserved; original_NaN_or_infinity_bits_not_recoverable_from_JSON",
                "lag_estimator_origin":data.as_ref().and_then(|d|d.pointer("/lag_provenance/estimator_origin")).cloned().unwrap_or_else(||json!("legacy_unknown")),
                "causal_influence_edges":"not_computed",
                "lattice_source":{"path":lattice_path,"sha256":lattice_hash,
                    "json_pointer":format!("/components/{}/node_lineage/{ordinal}",components[&component]["source_component_ordinal"].as_u64().unwrap())}});
            let bytes = serde_json::to_vec(&row)?;
            writer.write_all(&bytes)?;
            writer.write_all(b"\n")?;
            output_hash.update(&bytes);
            output_hash.update(b"\n");
            output_bytes += bytes.len() as u64 + 1;
            count += 1;
            *counts.entry(disposition.into()).or_default() += 1;
            let c = components.get_mut(&component).unwrap();
            let previous = c["dispositions"][disposition].as_u64().unwrap_or(0);
            c["dispositions"][disposition] = json!(previous + 1);
        }
    }
    writer.flush()?;
    let summary_path = output.join("ghost_kcc_observation_manifest.prismos_v1.json");
    let summary = json!({"schema_kind":"prismos_ghost_kcc_observation_manifest","schema_version":1,
        "run_id":manifest.run_id,"target":manifest.target,"topology_input":manifest.topology_input,
        "population_scope":"every_original_lattice_node_independent_of_density_regions_or_promotion",
        "join_rule":"exact(stream_id,step_idx,site_id=cluster_id); no spatial_or_temporal_nearest_neighbor_fallback",
        "node_count":count,"component_count":components.len(),"dispositions":counts,
        "chronology_leads_without_lattice_nodes":extra_leads,
        "node_frame_idx_unique_count":frame_ids.len(),"node_step_idx_unique_count":steps.len(),
        "chronological_authority":"step_idx; legacy frame_idx is retained as raw identity only",
        "lattice_source":{"path":lattice_path,"sha256":lattice_hash},"chronology_sources":index.sources,
        "snapshot_keys_without_lattice_nodes":index.locations.keys().filter(|k|!used_snapshots.contains(k)).map(|k|json!({"stream_id":k.0,"step_idx":k.1})).collect::<Vec<_>>(),
        "node_observations":{"path":row_path,"sha256":format!("{:x}",output_hash.finalize()),"size_bytes":output_bytes,"format":"JSONL"},
        "components":components.into_values().collect::<Vec<_>>(),"descriptor_fields":FIELDS,
        "prescribed_thermal_phase_order":THERMAL_PHASE_NAMES,"directed_causal_inference":"not_performed",
        "historical_zero_provenance_limit":"archived zero-filled unmeasured records and measured residue0 require producer provenance; no raw historical record rewritten"});
    let mut out = BufWriter::new(File::create(&summary_path)?);
    serde_json::to_writer_pretty(&mut out, &summary)?;
    out.flush()?;
    Ok(summary_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn node(lead: Option<u32>) -> GhostNodeLineage {
        GhostNodeLineage {
            tile_index: 7,
            stream_id: 0,
            site_id: 0,
            frame_idx: 0,
            step_idx: 10,
            protocol_phase: 1,
            step_bucket: 0,
            causal_lead_residue: lead,
        }
    }
    fn snapshot() -> Value {
        let mut fields = serde_json::Map::new();
        for name in FIELDS {
            fields.insert(
                name.into(),
                if name == "residue_count" {
                    json!([2])
                } else if name == "active_causal" {
                    json!([1])
                } else if name == "temporal_corr" {
                    json!([-0.5])
                } else {
                    json!([0.0])
                },
            );
        }
        fields.insert("n_residues".into(), json!(1));
        json!({"record_kind":"measurement","stream_id":0,"measurement_step":10,
            "applies_to":"current_post_physics_ghost_overlay","selection_method":METHOD,
            "kcc_per_residue":fields,"leads":[{"cluster_id":0,"measurement_step":10,
                "causal_lead_residue":0,"temporal_corr":-0.5,"residue_count":2,"active_causal":1}]})
    }
    #[test]
    fn chronology_lag_provenance_is_validated_and_forwarded() {
        let mut v=snapshot();
        assert!(descriptor(&v,Some(0)).unwrap().get("lag_provenance").is_none());
        let mut p=prism_nhs::causome::KccLagProvenance::legacy_unknown(2,1);
        p.source_stream_id=Some(0); p.estimator_origin="gpu_MD_step_grid_search".into();
        p.causal_lag_f32_bits=Some(0.0f32.to_bits()); p.lag_corr_peak_f32_bits=Some(0.0f32.to_bits());
        v["kcc_per_residue"]["lag_provenance"]=json!([p]);
        assert_eq!(snapshot_problem(&v,0,10),None);
        assert_eq!(descriptor(&v,Some(0)).unwrap()["lag_provenance"]["estimator_origin"],"gpu_MD_step_grid_search");
        v["kcc_per_residue"]["lag_provenance"][0]["source_stream_id"]=json!(1);
        assert_eq!(snapshot_problem(&v,0,10),Some("lag_provenance_measurement_mismatch"));
        v["kcc_per_residue"]["lag_provenance"]=json!([]);
        assert_eq!(snapshot_problem(&v,0,10),Some("invalid_lag_provenance_shape"));
    }

    #[test]
    fn exact_zero_identity_and_historical_null_are_distinct() {
        let v = snapshot();
        assert_eq!(snapshot_problem(&v, 0, 10), None);
        let (status, rid, data, _) = lead_result(&v, &v["leads"][0], &node(Some(0)));
        assert_eq!(status, "exact_key_and_lead");
        assert_eq!(rid, Some(0));
        assert_eq!(data.unwrap()["temporal_corr"], -0.5);
        let (status, rid, data, ghost) = lead_result(&v, &v["leads"][0], &node(None));
        assert_eq!(status, "lead_identity_mismatch");
        assert_eq!(rid, Some(0));
        assert!(data.is_some());
        assert!(ghost.is_none());
    }
    #[test]
    fn wrong_stream_step_shape_and_overlay_never_join() {
        let mut v = snapshot();
        assert_eq!(
            snapshot_problem(&v, 1, 10),
            Some("snapshot_stream_mismatch")
        );
        assert_eq!(snapshot_problem(&v, 0, 20), Some("snapshot_step_mismatch"));
        v["applies_to"] = json!("next_monolithic_replays_until_next_observation");
        assert_eq!(
            snapshot_problem(&v, 0, 10),
            Some("unsupported_snapshot_applies_to")
        );
        v = snapshot();
        v["kcc_per_residue"]["causal_lag"] = json!([]);
        assert_eq!(
            snapshot_problem(&v, 0, 10),
            Some("invalid_descriptor_shape")
        );
        v = snapshot();
        v["leads"][0]["measurement_step"] = json!(20);
        assert_eq!(
            lead_result(&v, &v["leads"][0], &node(Some(0))).0,
            "lead_step_mismatch"
        );
    }
    fn fixture(root: &Path) -> (Manifest, PathBuf, PathBuf, Value, Value) {
        let topology = root.join("topology.json");
        std::fs::write(&topology, b"{}").unwrap();
        let m:Manifest=serde_json::from_value(json!({"schema_version":1,"schema_kind":"test_manifest",
            "run_id":"run","target":"target","topology_input":topology,"stream_count":1,"streams_serialized":1,
            "total_spikes_md":0,"artifacts":[{"kind":"spikes","stream_id":0,"path":"unused", "size_bytes":0,"present":true}],
            "compression_status":"test","checksum_algorithm":"test","endian":"little","v2_was_live":true,"path_b_required":true})).unwrap();
        let binding = json!({"record_kind":"candidate_binding","schema_kind":"prismos.kcc_causal_lead_observations.v1",
            "stream_id":0,"topology_input":topology,"residue_identity_domain":"zero_based_native_topology_residue_index",
            "unmeasured_device_sentinel":u32::MAX,"selection_method":METHOD});
        let lattice = root.join("lattice.json");
        std::fs::write(
            &lattice,
            serde_json::to_vec(
                &json!({"schema_kind":"prismos_ghost_phase_lattice_components",
            "run_id":"run","target":"target","topology_input":topology,
            "components":[{"component_id":99,"n_nodes":1,"node_lineage":[node(None)]}]}),
            )
            .unwrap(),
        )
        .unwrap();
        (
            m,
            lattice,
            root.join("chronology.jsonl"),
            binding,
            snapshot(),
        )
    }
    fn write_rows(path: &Path, rows: &[Value]) {
        let mut file = File::create(path).unwrap();
        for row in rows {
            serde_json::to_writer(&mut file, row).unwrap();
            file.write_all(b"\n").unwrap();
        }
    }
    #[test]
    fn full_emission_preserves_node_missing_duplicate_and_record_hashes() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        let (m, lattice, chronology, binding, v) = fixture(root);
        for (rows, expected) in [
            (vec![binding.clone(), v.clone()], "lead_identity_mismatch"),
            (vec![binding.clone()], "missing_snapshot"),
            (
                vec![binding.clone(), v.clone(), v.clone()],
                "duplicate_snapshot",
            ),
        ] {
            write_rows(&chronology, &rows);
            let manifest = emit(
                &lattice,
                &[chronology.clone()],
                &m,
                root,
                root,
                &HashMap::new(),
            )
            .unwrap();
            let summary: Value = serde_json::from_reader(File::open(manifest).unwrap()).unwrap();
            assert_eq!(summary["node_count"], 1);
            assert_eq!(summary["dispositions"][expected], 1);
            let data = std::fs::read(root.join("ghost_kcc_observations.prismos_v1.jsonl")).unwrap();
            assert_eq!(summary["node_observations"]["sha256"], hash(&data));
            let row: Value = serde_json::from_slice(&data).unwrap();
            assert_eq!(row["component_id"], 99);
            assert_eq!(row["node_lineage"]["tile_index"], 7);
            assert!(row["ghost_causal_lead_residue"].is_null());
            assert_eq!(
                row["lattice_source"]["json_pointer"],
                "/components/0/node_lineage/0"
            );
            if expected == "lead_identity_mismatch" {
                assert_eq!(row["descriptor_array_index"], 0);
                assert_eq!(
                    row["chronology_lead_descriptor"].as_object().unwrap().len(),
                    14
                );
                let index = index_chronologies(&[chronology.clone()], &m, root).unwrap();
                let location = &index.locations[&(0, 10)][0];
                assert_eq!(load_snapshot(location).unwrap(), v);
                assert_eq!(location.line, 2);
            }
        }
    }
    #[test]
    fn binding_stream_and_lattice_run_mismatches_are_rejected() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        let (m, lattice, chronology, binding, mut v) = fixture(root);
        v["stream_id"] = json!(1);
        write_rows(&chronology, &[binding.clone(), v]);
        assert!(emit(
            &lattice,
            &[chronology.clone()],
            &m,
            root,
            root,
            &HashMap::new()
        )
        .is_err());
        let mut l: Value = serde_json::from_reader(File::open(&lattice).unwrap()).unwrap();
        l["run_id"] = json!("other");
        std::fs::write(&lattice, serde_json::to_vec(&l).unwrap()).unwrap();
        write_rows(&chronology, &[binding, snapshot()]);
        assert!(emit(&lattice, &[chronology], &m, root, root, &HashMap::new()).is_err());
    }

    #[test]
    fn raw_native_projection_preserves_residue_zero_and_unavailable_sentinel() {
        use prism_nhs::ghost_phase_lattice::{
            project_ghost_v2_record, QuartilePhaseSchedule, NODE_CAUSAL_LEAD_NONE,
        };
        use prism_nhs::ghost_tile::*;
        assert_eq!(GhostTileFrame::zero().causal_lead_residue, u32::MAX);
        for lead in [0, u32::MAX] {
            let mut raw = vec![0u8; GHOST_RECORD_BYTES];
            raw[124..128].copy_from_slice(&lead.to_le_bytes());
            raw[128..132].copy_from_slice(&GHOST_FRAME_SCHEMA_V3.to_le_bytes());
            raw[GHOST_V2_OFFSET_STEP_IDX..GHOST_V2_OFFSET_STEP_IDX + 8]
                .copy_from_slice(&10u64.to_le_bytes());
            raw[GHOST_V2_OFFSET_DT_FS..GHOST_V2_OFFSET_DT_FS + 4]
                .copy_from_slice(&2.0f32.to_le_bytes());
            raw[GHOST_V2_OFFSET_FIELD_COMPLETENESS_FLAGS
                ..GHOST_V2_OFFSET_FIELD_COMPLETENESS_FLAGS + 2]
                .copy_from_slice(&GHOST_FCF_BIT_SPATIAL_NATIVE_AABB_MIDPOINT.to_le_bytes());
            for axis in 0..3 {
                let offset = GHOST_V2_OFFSET_AABB_MAX + axis * 4;
                raw[offset..offset + 4].copy_from_slice(&1.0f32.to_le_bytes());
            }
            // Supply a valid native AABB; test only the lead identity semantics.
            let schedule = QuartilePhaseSchedule {
                min_step: 0,
                max_step: 40,
            };
            let parsed = project_ghost_v2_record(&raw, 0, 0, &schedule, 10).unwrap();
            assert_eq!(parsed.causal_lead_residue, lead);
            if lead == 0 {
                assert_ne!(parsed.causal_lead_residue, NODE_CAUSAL_LEAD_NONE);
            }
        }
    }
}
