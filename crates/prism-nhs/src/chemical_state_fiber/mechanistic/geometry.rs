//! Additive native geometry consumption for an already evaluated measurement
//! artifact. Source observations remain immutable; no MD or tensor replay runs.
use super::*;
use rayon::prelude::*;
use crate::coordinate_cavity::{
    verify_physical_grid_lining, PhysicalCavityBoundaryFaceAttribution, PhysicalCavityBoundaryKind,
    PhysicalCavityGridLiningEvidence, PhysicalCavityLiningAtomIdentity,
};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GeometryPartition {
    geometry: MeasurementSourcePin,
    atom_identities: MeasurementSourcePin,
    execution_receipt: MeasurementSourcePin,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GeometryArm {
    partitions: Vec<GeometryPartition>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GeometryManifest {
    schema_kind: String,
    measurement_artifact: MeasurementSourcePin,
    materialized_sites: MeasurementSourcePin,
    body_fixed_anchors: MeasurementSourcePin,
    arms: BTreeMap<String, GeometryArm>,
}

/// A face is addressed by its exact ordinal within a pinned native grid. The
/// complete original face and atom identity remain in their source artifacts.
#[derive(Debug, Clone, Serialize)]
struct ReceptorResidueFaces {
    analysis_residue_id: u32,
    face_indices: Vec<usize>,
    atom_indices: Vec<usize>,
}

fn receptor_residue_faces(
    faces: &[PhysicalCavityBoundaryFaceAttribution],
    identities: &BTreeMap<usize, PhysicalCavityLiningAtomIdentity>,
) -> Result<Vec<ReceptorResidueFaces>> {
    let mut groups = BTreeMap::<u32, (Vec<usize>, BTreeSet<usize>)>::new();
    for (i, face) in faces.iter().enumerate() {
        let atom = identities
            .get(&face.nearest_atom_index)
            .context("unknown face atom")?;
        ensure!(
            atom.atom_identity_sha256 == face.nearest_atom_identity_sha256,
            "face atom identity mismatch"
        );
        if face.boundary_kind == PhysicalCavityBoundaryKind::ReceptorExcludedVolume {
            let entry = groups.entry(atom.analysis_residue_id).or_default();
            entry.0.push(i);
            entry.1.insert(atom.atom_index);
        }
    }
    Ok(groups
        .into_iter()
        .map(|(rid, (faces, atoms))| ReceptorResidueFaces {
            analysis_residue_id: rid,
            face_indices: faces,
            atom_indices: atoms.into_iter().collect(),
        })
        .collect())
}

/// Verify exact producer JSON bytes rather than changing floating-point text
/// through a second serialization. Native Value records have sorted keys.
fn verify_record_bytes(raw: &str, row: &Value) -> Result<()> {
    let digest = text(&row["record_sha256"])?;
    require_sha256(digest, "geometry record")?;
    let token = format!("\"record_sha256\":\"{digest}\",");
    ensure!(
        raw.matches(&token).count() == 1,
        "geometry hash field placement"
    );
    let body = raw.replacen(&token, "", 1);
    ensure!(
        format!("{:x}", Sha256::digest(body.as_bytes())) == digest,
        "geometry source record digest mismatch"
    );
    Ok(())
}

fn checked_frame_binding(
    b: &Value,
    artifact: &MechanisticMeasurementArtifact,
    arm_source: &Value,
    ordinal: usize,
    coordinate_digest: &str,
) -> Result<()> {
    ensure!(
        u(&b["frame_ordinal"])? as usize == ordinal
            && u(&b["step"])? == artifact.frame_steps[ordinal]
            && u(&b["stream_id"])? == artifact.stream_id as u64
            && b["trajectory_sha256"] == arm_source["trajectory"]["sha256"]
            && b["audit_sha256"] == arm_source["audit"]["sha256"]
            && text(&b["original_coordinate_payload_sha256"])? == coordinate_digest,
        "geometry coordinate/run/stream chronology mismatch"
    );
    Ok(())
}

type CompressedLines =
    std::io::Lines<BufReader<zstd::stream::read::Decoder<'static, BufReader<File>>>>;
fn compressed_lines(path: &Path) -> Result<CompressedLines> {
    Ok(BufReader::new(zstd::stream::read::Decoder::new(File::open(path)?)?).lines())
}

struct EventCursor {
    lines: CompressedLines,
    source: MeasurementSourcePin,
    next: Option<(usize, Value)>,
    read: usize,
    previous_step: Option<u64>,
}
impl EventCursor {
    fn open(source: MeasurementSourcePin) -> Result<Self> {
        let mut cursor = Self {
            lines: compressed_lines(&source.path)?,
            source,
            next: None,
            read: 0,
            previous_step: None,
        };
        cursor.advance()?;
        Ok(cursor)
    }
    fn advance(&mut self) -> Result<()> {
        self.next = match self.lines.next() {
            None => None,
            Some(raw) => {
                let row: Value = serde_json::from_str(&raw?)?;
                let step = u(&row["completed_step"])?;
                ensure!(
                    self.previous_step.is_none_or(|old| old <= step),
                    "unordered source events"
                );
                self.previous_step = Some(step);
                let ordinal = self.read;
                self.read += 1;
                Some((ordinal, row))
            }
        };
        Ok(())
    }
}

#[derive(Clone)]
struct RegionLink {
    region_id: u64,
    index_line: u64,
    index_sha256: String,
    record_sha256: String,
    receptor_residues: BTreeSet<u32>,
}

fn emit_event_links(
    events: &mut EventCursor,
    writer: &mut impl Write,
    artifact: &MechanisticMeasurementArtifact,
    arm: &str,
    frame: usize,
    regions: &[RegionLink],
    output_index_name: &str,
) -> Result<(u64, u64)> {
    let step = artifact.frame_steps[frame];
    let mut rows = 0;
    let mut matched = 0;
    while let Some((ordinal, event)) = &events.next {
        let event_step = u(&event["completed_step"])?;
        ensure!(
            event_step >= step,
            "event source endpoint lost from geometry chronology"
        );
        if event_step > step {
            break;
        }
        let lineage = &event["native_record_lineage"];
        ensure!(
            text(&event["arm"])? == arm
                && text(&lineage["run_id"])? == artifact.arm_run_ids[arm]
                && text(&lineage["topology_sha256"])? == artifact.topology_sha256
                && u(&lineage["stream_id"])? == artifact.stream_id as u64
                && u(&lineage["step_idx"])? == step
                && u(&event["coordinate_frame_ordinal"])? as usize == frame
                && text(&event["resolved_physical_phase"])? == artifact.frame_phases[frame],
            "event source differs from measurement arm/endpoint"
        );
        let mut members = BTreeSet::new();
        for row in event["residue_measurements"]
            .as_array()
            .context("event residue axis missing")?
        {
            let rid = u32::try_from(u(&row["analysis_residue_id"])?)?;
            ensure!(
                artifact.analysis_residue_ids.binary_search(&rid).is_ok() && members.insert(rid),
                "invalid or duplicated event residue"
            );
        }
        let links: Vec<_> = regions.iter().filter_map(|region| {
            let shared: Vec<_> = region.receptor_residues.intersection(&members).copied().collect();
            (!shared.is_empty()).then(|| json!({"region_id":region.region_id,
                "geometry_index_artifact":output_index_name,
                "geometry_index_line_ordinal":region.index_line,
                "geometry_index_record_sha256":region.index_sha256,
                "geometry_source_record_sha256":region.record_sha256,
                "shared_analysis_residue_ids":shared,
                "face_lookup":"geometry index grids.receptor_residue_faces, by exact analysis_residue_id; face_indices address original grid.boundary_faces"}))
        }).collect();
        matched += links.len() as u64;
        let mut row = json!({"schema_kind":"prism.native_event_receptor_face_links.v1",
            "source_event":events.source,"source_event_line_ordinal":ordinal,
            "native_record_lineage":lineage,"native_lattice_component":event["native_lattice_component"],
            "arm":arm,"run_id":artifact.arm_run_ids[arm],"stream_id":artifact.stream_id,
            "step":step,"frame_ordinal":frame,"phase":artifact.frame_phases[frame],
            "native_event_residue_ids":members,"geometry_links":links,
            "link_semantics":"exact endpoint and shared analysis residue between native event membership and receptor-excluded-volume face attribution; not spatial overlap, directed influence, or identity equivalence",
            "event_without_receptor_face_match_preserved":links.is_empty()});
        row["record_sha256"] = canonical_sha256(&row)?.into();
        line(writer, &row)?;
        rows += 1;
        events.advance()?;
    }
    Ok((rows, matched))
}

fn coordinate_payload(reader: &mut File, frame: usize, atoms: usize) -> Result<Vec<u8>> {
    let stride = atoms
        .checked_mul(12)
        .context("coordinate stride overflow")?;
    reader.seek(SeekFrom::Start(
        24 + (frame as u64)
            .checked_mul(stride as u64)
            .context("coordinate offset overflow")?,
    ))?;
    let mut bytes = vec![0; stride];
    reader.read_exact(&mut bytes)?;
    Ok(bytes)
}
fn point(bytes: &[u8], atom: usize) -> Result<[f64; 3]> {
    let offset = atom.checked_mul(12).context("atom offset overflow")?;
    let bytes = bytes
        .get(offset..offset + 12)
        .context("anchor atom outside coordinate population")?;
    let p = std::array::from_fn(|i| {
        f32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap()) as f64
    });
    ensure!(
        p.iter().all(|v| v.is_finite()),
        "nonfinite anchor coordinates"
    );
    Ok(p)
}

/// Only independent region work is parallel. Each row runs the same complete
/// native digest, chronology, body transform, grid and face validation body.
struct GeometryFrameContext<'a> {
    artifact: &'a MechanisticMeasurementArtifact,
    arm_source: &'a Value,
    manifest: &'a GeometryManifest,
    part: &'a GeometryPartition,
    declaration: &'a Value,
    site_rows: &'a BTreeMap<u64, &'a Value>,
    identities: &'a BTreeMap<usize, PhysicalCavityLiningAtomIdentity>,
    atom_hashes: &'a BTreeMap<String, usize>,
    reference: &'a [u8],
    payload: &'a [u8],
    payload_digest: &'a str,
    arm: &'a str,
    frame: usize,
    source_line: usize,
    geometry_rows: u64,
}
struct ValidatedRegionRow {
    index_row: Value,
    link: RegionLink,
    face_counts: [u64;3],
    clipped: bool,
}
impl GeometryFrameContext<'_> {
    fn validate_row(&self, raw: &str, offset: usize) -> Result<ValidatedRegionRow> {
        let artifact=self.artifact; let a=self.arm_source; let manifest=self.manifest;
        let part=self.part; let declaration=self.declaration; let site_rows=self.site_rows;
        let identities=self.identities; let atom_hashes=self.atom_hashes;
        let reference=self.reference; let payload=self.payload; let payload_digest=self.payload_digest;
        let arm=self.arm; let frame=self.frame;
        let source_line=self.source_line + offset;
        let geometry_rows=self.geometry_rows + offset as u64;
        let mut face_counts=[0u64;3];
        let row: Value = serde_json::from_str(&raw)?;
        verify_record_bytes(&raw, &row)?;
        let binding = &row["observation_binding"];
        checked_frame_binding(binding, &artifact, a, frame, &payload_digest)?;
        let region = u(&binding["region_id"])?;
        let site = site_rows
            .get(&region)
            .context("geometry region not in declared materializer population")?;
        ensure!(
            row["schema_kind"] == "prism_materializer_region_frame_cavity"
                && row["topology_sha256"] == artifact.topology_sha256
                && row["atom_identity_table_sha256"] == part.atom_identities.sha256
                && binding["materialized_sites_sha256"]
                    == manifest.materialized_sites.sha256
                && text(&binding["source_region_record_sha256"])?
                    == canonical_sha256(site)?
                && text(&row["observation_sha256"])? == canonical_sha256(binding)?
                && source(&binding["body_fixed_tracking"]["declaration"])?
                    == manifest.body_fixed_anchors,
            "geometry region/source binding mismatch"
        );
        let anchor_ids: Vec<usize> = serde_json::from_value(
            declaration["region_anchor_atom_indices"][region.to_string()].clone(),
        )?;
        ensure!(
            anchor_ids.len() >= 3
                && anchor_ids.iter().collect::<BTreeSet<_>>().len() == anchor_ids.len(),
            "invalid body-frame anchor population"
        );
        let reference_points = anchor_ids
            .iter()
            .map(|&id| point(&reference, id))
            .collect::<Result<Vec<_>>>()?;
        let current_points = anchor_ids
            .iter()
            .map(|&id| point(&payload, id))
            .collect::<Result<Vec<_>>>()?;
        let center: [f64; 3] = serde_json::from_value(site["centroid_xyz"].clone())?;
        let fit = crate::fixed_cadence_coordinate::measure_body_frame_fit(
            &reference_points,
            &current_points,
            center,
        )?;
        ensure!(
            serde_json::to_value(&fit)? == binding["body_fixed_tracking"]["transform"],
            "geometry body transform differs from native coordinate measurement"
        );
        let mut grid_index = Vec::new();
        let mut receptor_residues = BTreeSet::new();
        let mut clipped = false;
        let grid_values = row["physical_lining_grids"]
            .as_array()
            .context("native grids missing")?;
        let spacing_count = row["estimator_config"]["grid_spacings_a"]
            .as_array()
            .context("grid spacing axis")?
            .len();
        let phase_count = row["estimator_config"]["grid_phase_offsets"]
            .as_array()
            .context("grid phase axis")?
            .len();
        ensure!(
            !grid_values.is_empty() && grid_values.len() == spacing_count * phase_count,
            "native grid population incomplete"
        );
        for (ordinal, value) in grid_values.iter().enumerate() {
            let grid: PhysicalCavityGridLiningEvidence =
                serde_json::from_value(value.clone())?;
            verify_physical_grid_lining(&grid, ordinal as u64, &atom_hashes)?;
            let membership = receptor_residue_faces(&grid.boundary_faces, &identities)?;
            ensure!(
                membership
                    .iter()
                    .map(|g| g.face_indices.len() as u64)
                    .sum::<u64>()
                    == grid.receptor_excluded_volume_face_count,
                "receptor membership face population mismatch"
            );
            receptor_residues.extend(membership.iter().map(|r| r.analysis_residue_id));
            face_counts[0] += grid.receptor_excluded_volume_face_count;
            face_counts[1] += grid.estimator_mask_face_count;
            face_counts[2] += grid.local_window_face_count;
            clipped |= grid.local_window_face_count > 0;
            grid_index.push(json!({"grid_ordinal":ordinal,"grid_record_sha256":grid.grid_record_sha256,
                "grid_source_sha256":grid.grid_source_sha256,"boundary_faces_sha256":grid.boundary_faces_sha256,
                "selected_component_grid_sha256":grid.selected_component_grid_sha256,
                "receptor_residue_faces":membership,"boundary_face_count":grid.boundary_face_count,
                "receptor_excluded_volume_face_count":grid.receptor_excluded_volume_face_count,
                "estimator_mask_face_count":grid.estimator_mask_face_count,"local_window_face_count":grid.local_window_face_count,
                "scope":"source grid retains every face, full voxel masks and atom attribution; membership includes receptor-excluded-volume faces only"}));
        }
        let mut index_row = json!({"schema_kind":"prism.native_region_frame_geometry_index.v1",
            "arm":arm,"run_id":artifact.arm_run_ids[arm],"stream_id":artifact.stream_id,
            "step":artifact.frame_steps[frame],"frame_ordinal":frame,"phase":artifact.frame_phases[frame],
            "geometry_source":part.geometry,"geometry_source_line_ordinal":source_line,
            "geometry_source_record_sha256":row["record_sha256"],"observation_binding":binding,
            "atom_identity_source":part.atom_identities,"grids":grid_index,
            "local_window_clipping_observed":clipped,"retained_without_TSH_or_clipping_admission":true});
        let index_digest = canonical_sha256(&index_row)?;
        index_row["record_sha256"] = index_digest.clone().into();
        let link = RegionLink {
            region_id: region,
            index_line: geometry_rows,
            index_sha256: index_digest,
            record_sha256: text(&row["record_sha256"])?.into(),
            receptor_residues,
        };
        Ok(ValidatedRegionRow { index_row, link, face_counts, clipped })
    }

    fn validate_ordered(&self, pool: &rayon::ThreadPool, raw_rows: &[String]) -> Result<Vec<ValidatedRegionRow>> {
        // IndexedParallelIterator collect retains source order. Collect errors
        // in that order too, so the first failing source row is deterministic.
        let results: Vec<Result<ValidatedRegionRow>> = pool.install(||
            raw_rows.par_iter().enumerate().map(|(offset,raw)| self.validate_row(raw,offset)).collect());
        results.into_iter().collect()
    }
}

/// Bound default CPU concurrency without changing the scientific population.
pub fn default_geometry_workers() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get().min(8))
}

/// Run the actual native consuming integration against completed scientific
/// records. Every source event is preserved; every source region/frame is
/// indexed. It emits additive artifacts and a new measurement artifact with
/// the original manifest identity, then verifies that complete artifact.
pub fn evaluate_geometry_supplement(manifest_path: &Path, out: &Path) -> Result<Value> {
    evaluate_geometry_supplement_with_workers(manifest_path,out,default_geometry_workers())
}

pub fn evaluate_geometry_supplement_with_workers(manifest_path: &Path, out: &Path, workers: usize) -> Result<Value> {
    ensure!(workers > 0, "geometry worker count must be positive");
    let pool = rayon::ThreadPoolBuilder::new().num_threads(workers)
        .thread_name(|index| format!("prism-native-geometry-{index}"))
        .build().context("native geometry worker pool")?;
    ensure!(
        !out.exists(),
        "preserve existing geometry supplement output"
    );
    let manifest_value: Value = serde_json::from_reader(File::open(manifest_path)?)?;
    let manifest: GeometryManifest = serde_json::from_value(manifest_value.clone())?;
    ensure!(
        manifest.schema_kind == "prism.native_mechanistic_geometry_manifest.v1"
            && manifest.arms.keys().map(String::as_str).collect::<Vec<_>>()
                == ["control", "treatment"],
        "geometry manifest schema/arm mismatch"
    );
    let mut new_pins = BTreeMap::new();
    collect_pins(&manifest_value, &mut new_pins)?;
    new_pins.insert(manifest_path.to_path_buf(), pin(manifest_path)?);
    for source in new_pins.values() {
        source.verify()?;
    }
    let mut artifact =
        MechanisticMeasurementArtifact::load_verified(&manifest.measurement_artifact.path)?;
    let base_manifest = artifact
        .source_pins
        .iter()
        .find(|p| p.sha256 == artifact.manifest_sha256)
        .context("base measurement manifest missing")?;
    let m: Value = serde_json::from_reader(File::open(&base_manifest.path)?)?;
    let sites: Value = serde_json::from_reader(File::open(&manifest.materialized_sites.path)?)?;
    let mut site_rows = BTreeMap::new();
    for bucket in [
        "binding_sites",
        "spatiotemporal_partial_candidates",
        "density_only_candidates",
    ] {
        for row in sites[bucket].as_array().into_iter().flatten() {
            let rid = u(&row["native_causome"]["region_id"])?;
            ensure!(
                site_rows.insert(rid, row).is_none(),
                "duplicate source region identity"
            );
        }
    }
    ensure!(
        !site_rows.is_empty(),
        "empty source geometry region population"
    );
    let declaration: Value =
        serde_json::from_reader(File::open(&manifest.body_fixed_anchors.path)?)?;
    ensure!(
        declaration["schema_kind"] == "prism_recorded_region_body_anchors_v1"
            && declaration["topology_sha256"] == artifact.topology_sha256
            && declaration["sites_sha256"] == manifest.materialized_sites.sha256,
        "body-frame declaration identity mismatch"
    );
    let reference_pin = source(&declaration["reference_trajectory"])?;
    reference_pin.verify()?;
    new_pins.insert(reference_pin.path.clone(), reference_pin.clone());
    let reference_frame = usize::try_from(u(&declaration["reference_frame_ordinal"])?)?;
    let reference = coordinate_payload(
        &mut File::open(&reference_pin.path)?,
        reference_frame,
        artifact.atom_count,
    )?;
    ensure!(
        format!("{:x}", Sha256::digest(&reference))
            == text(&declaration["reference_coordinate_payload_sha256"])?
            && artifact.frame_steps.get(reference_frame).copied()
                == declaration["reference_step"].as_u64(),
        "body-frame reference endpoint mismatch"
    );
    fs::create_dir_all(out)?;
    let execution_config_path=out.join("GEOMETRY_EXECUTION_CONFIG.json");
    write_json(&execution_config_path,&json!({
        "schema_kind":"prism.native_geometry_execution_config.v1",
        "source_manifest":pin(manifest_path)?,"requested_geometry_workers":workers,
        "actual_geometry_workers":pool.current_num_threads(),
        "parallel_unit":"complete_region_frame_native_validation",
        "max_in_flight_frames":1,"regions_per_frame":site_rows.len(),
        "publication_order":"original_source_line_order",
        "face_or_row_population_reduction":false,
        "floating_point_reduction_reordered":false,
    }))?;

    let mut summaries = BTreeMap::new();
    let mut products = Vec::new();
    for arm in ["treatment", "control"] {
        let a = &m["mechanistic_sources"]["arms"][arm];
        let event_name = format!("{arm}_native_event_measurements.jsonl.zst");
        let event_pin = artifact
            .evaluated_artifacts
            .get(&event_name)
            .context("native event product missing")?
            .clone();
        let mut events = EventCursor::open(event_pin)?;
        let index_name = format!("{arm}_native_geometry_index.jsonl.zst");
        let links_name = format!("{arm}_native_event_geometry_links.jsonl.zst");
        let mut index_writer = zstd::stream::write::Encoder::new(
            BufWriter::new(File::create(out.join(&index_name))?),
            1,
        )?;
        let mut links_writer = zstd::stream::write::Encoder::new(
            BufWriter::new(File::create(out.join(&links_name))?),
            1,
        )?;
        let mut trajectory = File::open(source(&a["trajectory"])?.path)?;
        let mut expected_frame = 0usize;
        let mut geometry_rows = 0u64;
        let mut event_rows = 0u64;
        let mut association_rows = 0u64;
        let mut face_counts = [0u64; 3];
        let mut observed_frames_without_events = 0u64;
        let mut clipped_region_frames = 0u64;
        for part in &manifest.arms[arm].partitions {
            let receipt: Value =
                serde_json::from_reader(File::open(&part.execution_receipt.path)?)?;
            let start = usize::try_from(u(&receipt["frame_start"])?)?;
            let end = usize::try_from(u(&receipt["frame_end"])?)?;
            ensure!(
                start == expected_frame
                    && end > start
                    && end <= artifact.frame_steps.len()
                    && receipt["topology_sha256"] == artifact.topology_sha256
                    && receipt["trajectory_sha256"] == a["trajectory"]["sha256"]
                    && receipt["audit_sha256"] == a["audit"]["sha256"]
                    && receipt["sites_sha256"] == manifest.materialized_sites.sha256,
                "geometry partition coverage or identity mismatch"
            );
            let identities: BTreeMap<usize, PhysicalCavityLiningAtomIdentity> =
                serde_json::from_reader(File::open(&part.atom_identities.path)?)?;
            let mut atom_hashes = BTreeMap::new();
            let topology: Value = load(&m["mechanistic_sources"]["topology"])?;
            let typed_topology:crate::input::PrismPrepTopology=serde_json::from_value(topology)?;
            ensure!(typed_topology.n_atoms==artifact.atom_count,"geometry full coordinate atom population differs");
            let expected_identities=crate::coordinate_cavity::receptor_atom_identity_map(&typed_topology)?;
            ensure!(identities==expected_identities,
                "geometry receptor identity population differs from the native signed-topology producer");
            for (&atom, identity) in &identities {
                identity.verify()?;
                ensure!(
                    identity.atom_index == atom
                        && atom_hashes
                            .insert(identity.atom_identity_sha256.clone(), atom)
                            .is_none(),
                    "geometry atom/topology mapping mismatch"
                );
            }
            let mut rows = compressed_lines(&part.geometry.path)?;
            let mut source_line = 0usize;
            for frame in start..end {
                let payload = coordinate_payload(&mut trajectory, frame, artifact.atom_count)?;
                let payload_digest = format!("{:x}", Sha256::digest(&payload));
                let mut seen = BTreeSet::new();
                let mut links = Vec::new();
                // Retain at most one frame's complete raw region population.
                // JSON parsing and every expensive native validation run per row.
                let raw_rows = (0..site_rows.len()).map(|_| {
                    rows.next().context("geometry partition ended before full region population")?
                        .map_err(anyhow::Error::from)
                }).collect::<Result<Vec<_>>>()?;
                let context = GeometryFrameContext {
                    artifact:&artifact,arm_source:a,manifest:&manifest,part,declaration:&declaration,
                    site_rows:&site_rows,identities:&identities,atom_hashes:&atom_hashes,
                    reference:&reference,payload:&payload,payload_digest:&payload_digest,
                    arm,frame,source_line,geometry_rows,
                };
                for validated in context.validate_ordered(&pool,&raw_rows)? {
                    ensure!(seen.insert(validated.link.region_id), "geometry region/source binding mismatch");
                    // Publication and all population reductions remain serial.
                    // No floating-point reduction or record order changes.
                    for axis in 0..3 { face_counts[axis] += validated.face_counts[axis]; }
                    clipped_region_frames += u64::from(validated.clipped);
                    line(&mut index_writer,&validated.index_row)?;
                    links.push(validated.link);
                    geometry_rows += 1;
                    source_line += 1;
                }
                ensure!(
                    seen.len() == site_rows.len(),
                    "geometry frame lost region population"
                );
                let (count, matched) = emit_event_links(
                    &mut events,
                    &mut links_writer,
                    &artifact,
                    arm,
                    frame,
                    &links,
                    &index_name,
                )?;
                observed_frames_without_events += u64::from(count == 0);
                event_rows += count;
                association_rows += matched;
                if frame % 100 == 0 {
                    eprintln!("native geometry {arm} frame {frame}/{} regions {geometry_rows} events {event_rows}", artifact.frame_steps.len());
                }
            }
            ensure!(rows.next().is_none(), "extra geometry partition rows");
            ensure!(
                u(&receipt["observations"])? == source_line as u64
                    && u(&receipt["expected_observations"])? == source_line as u64,
                "geometry partition receipt population mismatch"
            );
            expected_frame = end;
        }
        ensure!(
            expected_frame == artifact.frame_steps.len()
                && events.next.is_none()
                && event_rows == events.read as u64,
            "geometry/event population incomplete"
        );
        index_writer.finish()?.flush()?;
        links_writer.finish()?.flush()?;
        products.extend([index_name, links_name]);
        summaries.insert(arm, json!({"frames":expected_frame,"regions_per_frame":site_rows.len(),
            "geometry_records":geometry_rows,"event_records":event_rows,"event_region_associations":association_rows,
            "frames_without_native_events_preserved":observed_frames_without_events,
            "receptor_excluded_volume_faces":face_counts[0],"estimator_mask_faces":face_counts[1],
            "local_window_faces":face_counts[2],"clipped_region_frames_preserved":clipped_region_frames}));
    }
    for p in new_pins.into_values() {
        if let Some(old) = artifact.source_pins.iter().find(|old| old.path == p.path) {
            ensure!(old == &p, "supplement attempted source epoch replacement");
        } else {
            artifact.source_pins.push(p);
        }
    }
    for name in products {
        ensure!(
            !artifact.evaluated_artifacts.contains_key(&name),
            "supplement product already present"
        );
        artifact
            .evaluated_artifacts
            .insert(name.clone(), pin(&out.join(&name))?);
    }
    artifact.scope["geometry_grids"] = json!({"mode":"native_body_fixed_geometry_consumed",
        "source_measurement_artifact":manifest.measurement_artifact,"supplement_manifest":pin(manifest_path)?,
        "full_native_grid_and_face_validation":true,"body_transforms_recomputed_from_exact_coordinates":true,
        "event_correspondence":"exact source run/stream/step plus shared actual receptor-face analysis residues",
        "all_geometry_frames_retained":true,"all_native_events_retained":true,
        "raw_faces_voxel_masks_and_original_events":"preserved in pinned immutable sources",
        "causal_influence_inferred":false,"rank_or_region_component_equivalence_used":false,
        "arms":summaries});
    artifact.verify()?;
    write_json(
        &out.join("mechanistic_measurement_artifact.json"),
        &artifact,
    )?;
    let receipt = json!({"schema_kind":"prism.native_mechanistic_geometry_supplement_receipt.v1",
        "status":"IMPLEMENTATION_VERIFIED","artifact":pin(&out.join("mechanistic_measurement_artifact.json"))?,
        "canonical_artifact_sha256":canonical_sha256(&artifact)?,"source_manifest_identity_preserved":artifact.manifest_sha256,
        "MD_executed":false,"tensor_or_event_measurements_regenerated":false,"geometry_reestimated":false,
        "geometry_consumption_completed":true,"execution_config":pin(&execution_config_path)?,
        "scope":artifact.scope["geometry_grids"]});
    write_json(&out.join("EXECUTION_RECEIPT.json"), &receipt)?;
    Ok(receipt)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn identity(atom: usize, residue: u32) -> PhysicalCavityLiningAtomIdentity {
        PhysicalCavityLiningAtomIdentity {
            atom_identity_sha256: format!("{atom:064x}"),
            atom_index: atom,
            atom_name: "CA".into(),
            element: "C".into(),
            physical_residue_id: residue as usize,
            analysis_residue_id: residue,
            pdb_residue_id: residue as i64,
            residue_name: "ALA".into(),
            chain_id: "A".into(),
            molecule_id: 0,
            molecule_identity: "A".into(),
            molecule_component_sha256: "a".repeat(64),
        }
    }
    fn face(
        atom: usize,
        kind: PhysicalCavityBoundaryKind,
    ) -> PhysicalCavityBoundaryFaceAttribution {
        PhysicalCavityBoundaryFaceAttribution {
            face_record_sha256: "a".repeat(64),
            grid_source_sha256: "b".repeat(64),
            voxel_linear_index: 0,
            voxel_xyz: [0; 3],
            axis: 0,
            orientation: 1,
            boundary_kind: kind,
            face_center_relative_angstrom: [0.; 3],
            face_area_a2: 1.,
            nearest_atom_identity_sha256: format!("{atom:064x}"),
            nearest_atom_index: atom,
            nearest_atom_center_distance_a: 1.,
            nearest_solvent_excluded_surface_distance_a: 0.,
        }
    }
    #[test]
    fn receptor_membership_keeps_actual_faces_and_excludes_window_mask_attribution() {
        let identities = [
            (0, identity(0, 71)),
            (1, identity(1, 71)),
            (2, identity(2, 83)),
        ]
        .into_iter()
        .collect();
        let faces = vec![
            face(0, PhysicalCavityBoundaryKind::LocalWindow),
            face(1, PhysicalCavityBoundaryKind::ReceptorExcludedVolume),
            face(2, PhysicalCavityBoundaryKind::EstimatorMask),
            face(0, PhysicalCavityBoundaryKind::ReceptorExcludedVolume),
        ];
        let groups = receptor_residue_faces(&faces, &identities).unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].analysis_residue_id, 71);
        assert_eq!(groups[0].face_indices, [1, 3]);
        assert_eq!(groups[0].atom_indices, [0, 1]);
        let mut bad = faces;
        bad[0].nearest_atom_identity_sha256 = "c".repeat(64);
        assert!(receptor_residue_faces(&bad, &identities).is_err());
    }
    #[test]
    fn exact_record_hash_rejects_tampered_geometry_without_float_reencoding() {
        let body = "{\"a\":0.00000000000000012345,\"z\":2}";
        let digest = format!("{:x}", Sha256::digest(body.as_bytes()));
        let raw =
            format!("{{\"a\":0.00000000000000012345,\"record_sha256\":\"{digest}\",\"z\":2}}");
        let row: Value = serde_json::from_str(&raw).unwrap();
        verify_record_bytes(&raw, &row).unwrap();
        assert!(verify_record_bytes(&raw.replace("\"z\":2", "\"z\":3"), &row).is_err());
    }
    #[test]
    fn exact_binding_rejects_cross_arm_step_stream_and_coordinate_substitution() {
        let artifact = MechanisticMeasurementArtifact {
            schema_kind: "prism.native_fiber_mechanistic_measurements.v1".into(),
            manifest_sha256: "a".repeat(64),
            topology_sha256: "b".repeat(64),
            arm_run_ids: BTreeMap::new(),
            stream_id: 0,
            frame_steps: vec![10, 20],
            frame_phases: vec!["cold_hold".into(), "heating".into()],
            atom_count: 1,
            analysis_residue_ids: vec![0],
            source_pins: vec![],
            evaluated_artifacts: BTreeMap::new(),
            scope: json!({}),
        };
        let a = json!({"trajectory":{"sha256":"trajectory"},"audit":{"sha256":"audit"}});
        let b = json!({"frame_ordinal":1,"step":20,"stream_id":0,"trajectory_sha256":"trajectory",
            "audit_sha256":"audit","original_coordinate_payload_sha256":"coordinates"});
        checked_frame_binding(&b, &artifact, &a, 1, "coordinates").unwrap();
        for (key, value) in [
            ("step", json!(10)),
            ("frame_ordinal", json!(0)),
            ("stream_id", json!(1)),
            ("trajectory_sha256", json!("other_arm")),
            ("audit_sha256", json!("other_audit")),
            ("original_coordinate_payload_sha256", json!("other_frame")),
        ] {
            let mut bad = b.clone();
            bad[key] = value;
            assert!(
                checked_frame_binding(&bad, &artifact, &a, 1, "coordinates").is_err(),
                "{key}"
            );
        }
    }
    /// Actual immutable producer record qualification, configured explicitly by
    /// the caller. This executes the complete grid verifier and receptor-face
    /// consumer; it does not claim complete campaign population qualification.
    #[test]
    #[ignore = "requires PRISM_GEOMETRY_RECORDS and PRISM_GEOMETRY_ATOM_IDENTITIES"]
    fn recorded_native_geometry_full_grid_consumer() {
        let path = std::env::var("PRISM_GEOMETRY_RECORDS").unwrap();
        let identities_path = std::env::var("PRISM_GEOMETRY_ATOM_IDENTITIES").unwrap();
        let raw = compressed_lines(Path::new(&path))
            .unwrap()
            .next()
            .unwrap()
            .unwrap();
        let row: Value = serde_json::from_str(&raw).unwrap();
        verify_record_bytes(&raw, &row).unwrap();
        assert_eq!(
            digest_file(Path::new(&identities_path)).unwrap(),
            text(&row["atom_identity_table_sha256"]).unwrap()
        );
        let identities: BTreeMap<usize, PhysicalCavityLiningAtomIdentity> =
            serde_json::from_reader(File::open(identities_path).unwrap()).unwrap();
        let mut hashes = BTreeMap::new();
        for (&atom, identity) in &identities {
            identity.verify().unwrap();
            assert_eq!(identity.atom_index, atom);
            hashes.insert(identity.atom_identity_sha256.clone(), atom);
        }
        let mut receptor_faces = 0;
        for (ordinal, value) in row["physical_lining_grids"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
        {
            let grid: PhysicalCavityGridLiningEvidence =
                serde_json::from_value(value.clone()).unwrap();
            verify_physical_grid_lining(&grid, ordinal as u64, &hashes).unwrap();
            let groups = receptor_residue_faces(&grid.boundary_faces, &identities).unwrap();
            assert_eq!(
                groups
                    .iter()
                    .map(|g| g.face_indices.len() as u64)
                    .sum::<u64>(),
                grid.receptor_excluded_volume_face_count
            );
            receptor_faces += grid.receptor_excluded_volume_face_count;
        }
        eprintln!("native complete geometry record verified: region={} step={} grids={} receptor_faces={}",row["observation_binding"]["region_id"],row["observation_binding"]["step"],row["physical_lining_grids"].as_array().unwrap().len(),receptor_faces);
    }
}

#[cfg(test)]
mod actual_geometry_parallel_tests {
    use super::*;

    #[test]
    #[ignore="requires PRISM_GEOMETRY_TEST_MANIFEST pinned actual native geometry and PRISM_GEOMETRY_TEST_OUTPUT fresh receipt path"]
    fn actual_native_region_rows_are_byte_identical_with_one_and_multiple_workers() -> Result<()> {
        let manifest_path=PathBuf::from(std::env::var("PRISM_GEOMETRY_TEST_MANIFEST")?);
        let receipt_path=PathBuf::from(std::env::var("PRISM_GEOMETRY_TEST_OUTPUT")?);
        ensure!(!receipt_path.exists(),"preserve previous equivalence receipt");
        let manifest:GeometryManifest=serde_json::from_reader(File::open(&manifest_path)?)?;
        ensure!(manifest.schema_kind=="prism.native_mechanistic_geometry_manifest.v1","actual geometry manifest schema");
        manifest.measurement_artifact.verify()?;manifest.materialized_sites.verify()?;manifest.body_fixed_anchors.verify()?;
        let artifact=MechanisticMeasurementArtifact::load_verified(&manifest.measurement_artifact.path)?;
        let base=artifact.source_pins.iter().find(|p|p.sha256==artifact.manifest_sha256).context("base measurement manifest")?;
        let m:Value=serde_json::from_reader(File::open(&base.path)?)?;
        let sites:Value=serde_json::from_reader(File::open(&manifest.materialized_sites.path)?)?;
        let mut site_rows=BTreeMap::new();
        for bucket in ["binding_sites","spatiotemporal_partial_candidates","density_only_candidates"] {
            for row in sites[bucket].as_array().into_iter().flatten() {
                ensure!(site_rows.insert(u(&row["native_causome"]["region_id"])?,row).is_none(),"duplicate actual source region");
            }
        }
        ensure!(!site_rows.is_empty(),"actual region population empty");
        let declaration:Value=serde_json::from_reader(File::open(&manifest.body_fixed_anchors.path)?)?;
        let reference_pin=source(&declaration["reference_trajectory"])?;reference_pin.verify()?;
        let reference=coordinate_payload(&mut File::open(&reference_pin.path)?,usize::try_from(u(&declaration["reference_frame_ordinal"])?)?,artifact.atom_count)?;
        let one=rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
        let many=rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
        let mut checks=Vec::new();
        for arm in ["treatment","control"] {
            let arm_source=&m["mechanistic_sources"]["arms"][arm];
            let trajectory_pin=source(&arm_source["trajectory"])?;
            let mut trajectory=File::open(&trajectory_pin.path)?;
            let parts=&manifest.arms[arm].partitions;
            ensure!(!parts.is_empty(),"actual partitions missing");
            let mut selected=vec![0usize];if parts.len()>1 {selected.push(parts.len()-1);}
            for part_id in selected {
                let part=&parts[part_id];part.geometry.verify()?;part.atom_identities.verify()?;part.execution_receipt.verify()?;
                let receipt:Value=serde_json::from_reader(File::open(&part.execution_receipt.path)?)?;
                let frame=usize::try_from(u(&receipt["frame_start"])?)?;
                let payload=coordinate_payload(&mut trajectory,frame,artifact.atom_count)?;
                let payload_digest=format!("{:x}",Sha256::digest(&payload));
                let identities:BTreeMap<usize,PhysicalCavityLiningAtomIdentity>=serde_json::from_reader(File::open(&part.atom_identities.path)?)?;
                let topology:crate::input::PrismPrepTopology=serde_json::from_value(load(&m["mechanistic_sources"]["topology"])?)?;
                ensure!(topology.n_atoms==artifact.atom_count && identities==crate::coordinate_cavity::receptor_atom_identity_map(&topology)?,"actual receptor identities incomplete");
                let mut atom_hashes=BTreeMap::new();
                for (&atom,identity) in &identities {identity.verify()?;ensure!(atom_hashes.insert(identity.atom_identity_sha256.clone(),atom).is_none(),"duplicate actual atom hash");}
                let mut lines=compressed_lines(&part.geometry.path)?;
                let raw_rows=(0..site_rows.len()).map(|_|lines.next().context("actual frame region missing")?.map_err(anyhow::Error::from)).collect::<Result<Vec<_>>>()?;
                let context=GeometryFrameContext {artifact:&artifact,arm_source,manifest:&manifest,part,declaration:&declaration,
                    site_rows:&site_rows,identities:&identities,atom_hashes:&atom_hashes,reference:&reference,
                    payload:&payload,payload_digest:&payload_digest,arm,frame,source_line:0,geometry_rows:(frame*site_rows.len()) as u64};
                let serial_started=std::time::Instant::now();let serial=context.validate_ordered(&one,&raw_rows)?;
                let serial_seconds=serial_started.elapsed().as_secs_f64();
                let parallel_started=std::time::Instant::now();let parallel=context.validate_ordered(&many,&raw_rows)?;
                let parallel_seconds=parallel_started.elapsed().as_secs_f64();
                ensure!(serial.len()==site_rows.len() && parallel.len()==serial.len(),"parallel region population mismatch");
                let mut scientific_sha=Sha256::new();let mut counts=[0u64;3];
                for (a,b) in serial.iter().zip(&parallel) {
                    let av=serde_json::to_vec(&a.index_row)?;let bv=serde_json::to_vec(&b.index_row)?;
                    ensure!(av==bv,"actual geometry serialized JSON differs by worker count");
                    scientific_sha.update(&av);scientific_sha.update(b"\n");
                    ensure!(a.link.region_id==b.link.region_id && a.link.index_line==b.link.index_line
                        && a.link.index_sha256==b.link.index_sha256 && a.link.record_sha256==b.link.record_sha256
                        && a.link.receptor_residues==b.link.receptor_residues && a.face_counts==b.face_counts
                        && a.clipped==b.clipped,"parallel link/count/clipping mismatch");
                    for axis in 0..3 {counts[axis]+=a.face_counts[axis];}
                }
                // Damaging an actual source digest must fail both modes; workers
                // never turn digest validation or source failure into an omission.
                let mut damaged=raw_rows.clone();let marker="\"record_sha256\":\"";
                let location=damaged[0].find(marker).context("actual record digest marker")?+marker.len();
                let old=damaged[0].as_bytes()[location] as char;damaged[0].replace_range(location..location+1,if old=='0'{"1"}else{"0"});
                ensure!(context.validate_ordered(&one,&damaged).is_err() && context.validate_ordered(&many,&damaged).is_err(),"damaged actual record passed parallel validation");
                checks.push(json!({"arm":arm,"frame_ordinal":frame,"geometry_source":part.geometry,
                    "regions":serial.len(),"face_counts":counts,"serialized_json_sha256":format!("{:x}",scientific_sha.finalize()),
                    "one_worker_seconds":serial_seconds,"four_worker_seconds":parallel_seconds,
                    "exact_json_link_count_clipping_equivalence":true,"damaged_source_rejected_both_modes":true}));
            }
        }
        write_json(&receipt_path,&json!({"schema_kind":"prism.native_geometry_parallel_equivalence.v1",
            "status":"IMPLEMENTATION_VERIFIED","manifest":pin(&manifest_path)?,"workers":[1,4],
            "scope":"every native region in first frame of first and last pinned partition in each arm",
            "full_native_row_verification":true,"synthetic_scientific_records_generated":false,"checks":checks}))?;
        Ok(())
    }
}
