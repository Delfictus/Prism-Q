//! Native observational fibers over recorded protocol, coordinates and events.
//! This producer shares the native Ghost and structural readout adapters. It
//! does not construct chemical lanes, absence journals, PME or Markov states.
use super::readout::{
    check_event, evaluated, geometry_configs, read_geometry_frame, GeometryRow, RowCursor,
};
use super::*;
use crate::chemical_state_fiber::GhostByteLineage;
use crate::coordinate_cavity::{
    PhysicalCavitySpatialMetricAuthority, PhysicalCavitySpatialMetricMode,
};
use crate::ghost_phase_lattice::GhostRecordByteLineage;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_kind: String,
    readout_receipt: MeasurementSourcePin,
}

/// Actual spatial and protocol authorities, without unavailable periodic or
/// alchemical scalar fields. Historical source identities remain unchanged.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecordedEnvironmentAuthority {
    pub schema_kind: String,
    pub topology: MeasurementSourcePin,
    pub spatial_metric: PhysicalCavitySpatialMetricAuthority,
    pub protocol_state: MeasurementSourcePin,
    pub ghost_phase_binding: MeasurementSourcePin,
    pub run_id: String,
    pub stream_id: u16,
    pub legacy_boundary_normalized_in_memory: bool,
}
impl RecordedEnvironmentAuthority {
    fn verify(&self, a: &MechanisticMeasurementArtifact, arm: &str) -> Result<()> {
        self.spatial_metric.verify()?;
        ensure!(
            self.schema_kind == "prism.recorded_environment_authority.v1"
                && self.topology.sha256 == a.topology_sha256
                && self.run_id == a.arm_run_ids[arm]
                && self.stream_id == a.stream_id,
            "recorded environment identity mismatch"
        );
        ensure!(
            matches!(
                self.spatial_metric.mode,
                PhysicalCavitySpatialMetricMode::DeclaredNonperiodicEuclidean
            ),
            "this recorded nonperiodic producer does not reinterpret periodic conditions"
        );
        for p in [
            &self.topology,
            &self.protocol_state,
            &self.ghost_phase_binding,
        ] {
            ensure!(
                a.source_pins.contains(p),
                "environment source not in verified measurement population"
            );
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NativeMeasurementRowReference {
    pub source: MeasurementSourcePin,
    pub line_ordinal: u64,
    pub canonical_row_sha256: String,
}
impl NativeMeasurementRowReference {
    fn new(source: &MeasurementSourcePin, ordinal: u64, row: &Value) -> Result<Self> {
        Ok(Self {
            source: source.clone(),
            line_ordinal: ordinal,
            canonical_row_sha256: canonical_sha256(row)?,
        })
    }
}
/// Exact source-indexed receptor boundary support. These are observed atom
/// and residue memberships, not inferred atom identity or site rank matches.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecordedReceptorStructuralSupport {
    pub native_region_id: u64,
    pub geometry_index: NativeMeasurementRowReference,
    pub shared_analysis_residue_ids: Vec<u32>,
    pub native_grid_receptor_support: Vec<Value>,
    pub body_fixed_tracking: Value,
    pub geometry_source_record_sha256: String,
    pub local_window_clipping_observed: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecordedMeasurementFiberBody {
    pub ghost: GhostByteLineage,
    pub environment_authority_sha256: String,
    pub arm: String,
    pub coordinate_frame_ordinal: usize,
    pub completed_step: u64,
    pub physical_time_ps: f64,
    pub resolved_physical_phase: String,
    pub recorded_lattice_phase_ordinal: u16,
    pub complete_native_readout_source: NativeMeasurementRowReference,
    pub complete_measurement_frame: NativeMeasurementRowReference,
    pub native_measurements: Value,
    pub receptor_structural_support: Vec<RecordedReceptorStructuralSupport>,
    pub receptor_face_correspondence: Value,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecordedMeasurementFiber {
    pub schema_kind: String,
    pub fiber_sha256: String,
    pub body: RecordedMeasurementFiberBody,
}
impl RecordedMeasurementFiber {
    /// Verify native typed serialization, preserving the producer's body
    /// field order rather than hashing a differently ordered generic JSON map.
    pub fn verify(&self) -> Result<()> {
        self.body.ghost.validate()?;
        ensure!(
            self.schema_kind == "prism.recorded_native_measurement_fiber.v1"
                && self.body.completed_step == self.body.ghost.key.step_idx
                && self.body.recorded_lattice_phase_ordinal == self.body.ghost.protocol_phase
                && self.body.physical_time_ps.is_finite()
                && self.fiber_sha256 == canonical_sha256(&self.body)?,
            "recorded measurement fiber identity mismatch"
        );
        require_sha256(
            &self.body.environment_authority_sha256,
            "recorded environment authority",
        )?;
        Ok(())
    }
    /// Resolve the full immutable native record against an exact Ghost occurrence.
    /// This does not infer chemical lane, independent sampling or causal influence.
    pub fn resolve_ghost(&self, ghost: &GhostByteLineage, phase: &str) -> Result<&Self> {
        self.verify()?;
        ghost.validate()?;
        ensure!(
            self.body.ghost.lineage_sha256()? == ghost.lineage_sha256()?
                && self.body.resolved_physical_phase == phase,
            "recorded measurement and consumer Ghost/phase differ"
        );
        let event = &self.body.native_measurements;
        let raw: GhostRecordByteLineage =
            serde_json::from_value(event["native_record_lineage"].clone())?;
        let recorded = crate::pathb_evidence_emitter::adapt_recorded_ghost_byte_lineage(
            &raw,
            &ghost.protocol_name,
        )?;
        ensure!(
            recorded.lineage_sha256()? == ghost.lineage_sha256()?
                && event["arm"] == self.body.arm
                && event["coordinate_frame_ordinal"].as_u64()
                    == Some(self.body.coordinate_frame_ordinal as u64)
                && event["completed_step"].as_u64() == Some(self.body.completed_step)
                && event["physical_time_ps"].as_f64() == Some(self.body.physical_time_ps)
                && event["resolved_physical_phase"] == phase,
            "recorded measurement body disagrees with its native event clock/identity"
        );
        Ok(self)
    }
    /// Borrow full native tensors, contacts, exact clock and receptor-face records
    /// for an already valid physical chemical fiber. Both original hashes remain
    /// unchanged; the caller separately verifies the recorded environment authority.
    pub fn resolve_physical(&self, fiber: &ChemicalStateFiber) -> Result<&Self> {
        fiber.verify()?;
        self.resolve_ghost(&fiber.body.ghost, &fiber.body.protocol_phase.phase_name)
    }
    fn materialize(body: RecordedMeasurementFiberBody) -> Result<Self> {
        body.ghost.validate()?;
        ensure!(
            body.completed_step == body.ghost.key.step_idx
                && body.recorded_lattice_phase_ordinal == body.ghost.protocol_phase
                && body.physical_time_ps.is_finite()
                && !body.resolved_physical_phase.is_empty(),
            "measurement fiber endpoint differs"
        );
        require_sha256(
            &body.environment_authority_sha256,
            "recorded environment authority",
        )?;
        Ok(Self {
            schema_kind: "prism.recorded_native_measurement_fiber.v1".into(),
            fiber_sha256: canonical_sha256(&body)?,
            body,
        })
    }
}
fn checked_source_reference(
    value: &Value,
    source: &MeasurementSourcePin,
    ordinal: u64,
    row: &Value,
) -> Result<()> {
    ensure!(
        serde_json::from_value::<MeasurementSourcePin>(value["source"].clone())? == *source
            && u(&value["line_ordinal"])? == ordinal
            && value["canonical_row_sha256"] == canonical_sha256(row)?,
        "saved readout row reference does not bind its complete native record"
    );
    Ok(())
}
fn structural_support(
    regions: &BTreeMap<u64, GeometryRow>,
    links: &Value,
    index_source: &MeasurementSourcePin,
) -> Result<Vec<RecordedReceptorStructuralSupport>> {
    links["geometry_links"].as_array().context("native geometry associations")?.iter().map(|link| {
        let id=u(&link["region_id"])?;
        let g=regions.get(&id).context("native region missing from full structural population")?;
        let shared:Vec<u32>=serde_json::from_value(link["shared_analysis_residue_ids"].clone())?;
        let members=shared.iter().copied().collect::<BTreeSet<_>>();
        let grids=g.row["grids"].as_array().context("native grid evidence")?.iter().map(|grid| {
            let exact_rows=grid["receptor_residue_faces"].as_array().context("native receptor memberships")?.iter()
                .filter(|r|r["analysis_residue_id"].as_u64().is_some_and(|id|members.contains(&(id as u32))))
                .cloned().collect::<Vec<_>>();
            Ok(json!({"grid_ordinal":grid["grid_ordinal"],"grid_record_sha256":grid["grid_record_sha256"],
                "boundary_faces_sha256":grid["boundary_faces_sha256"],"receptor_residue_faces":exact_rows,
                "face_indices_address":"original immutable native grid.boundary_faces; no face coordinates or chemical labels inferred"}))
        }).collect::<Result<Vec<_>>>()?;
        Ok(RecordedReceptorStructuralSupport{native_region_id:id,
            geometry_index:NativeMeasurementRowReference::new(index_source,g.ordinal,&g.row)?,shared_analysis_residue_ids:shared,
            native_grid_receptor_support:grids,body_fixed_tracking:g.row["observation_binding"]["body_fixed_tracking"].clone(),
            geometry_source_record_sha256:text(&g.row["geometry_source_record_sha256"])?.into(),
            local_window_clipping_observed:g.row["local_window_clipping_observed"].as_bool().context("clipping observation")?})
    }).collect()
}

/// Materialize every recorded event as an exact native observational fiber,
/// plus every recorded frame even when it has no native event. Shared native
/// adapters verify source lineage and structural measurements; no MD executes.
pub fn materialize_recorded_measurement_fibers(manifest_path: &Path, out: &Path) -> Result<Value> {
    ensure!(
        !out.exists(),
        "refusing to replace recorded measurement fibers"
    );
    let manifest_pin = pin(manifest_path)?;
    let m: Manifest = serde_json::from_reader(File::open(manifest_path)?)?;
    ensure!(
        m.schema_kind == "prism.recorded_measurement_fiber_manifest.v1",
        "recorded fiber manifest schema"
    );
    m.readout_receipt.verify()?;
    let receipt: Value = serde_json::from_reader(File::open(&m.readout_receipt.path)?)?;
    ensure!(
        receipt["schema_kind"] == "prism.native_paired_mechanistic_readout_receipt.v1"
            && receipt["status"] == "IMPLEMENTATION_VERIFIED",
        "completed native readout required"
    );
    let artifact_pin = source(&receipt["source_artifact"])?;
    artifact_pin.verify()?;
    let a = MechanisticMeasurementArtifact::load_verified(&artifact_pin.path)?;
    ensure!(
        receipt["source_manifest_sha256"] == a.manifest_sha256
            && u(&receipt["frames"])? as usize == a.frame_steps.len(),
        "recorded fiber readout chronology/source mismatch"
    );
    let products: BTreeMap<String, MeasurementSourcePin> =
        serde_json::from_value(receipt["outputs"].clone())?;
    for p in products.values() {
        p.verify()?;
    }
    let base_pin = a
        .source_pins
        .iter()
        .find(|p| p.sha256 == a.manifest_sha256)
        .context("source measurement manifest")?;
    let base: Value = serde_json::from_reader(File::open(&base_pin.path)?)?;
    let top_pin = source(&base["mechanistic_sources"]["topology"])?;
    let mut top: crate::input::PrismPrepTopology =
        serde_json::from_reader(File::open(&top_pin.path)?)?;
    let normalized =
        crate::ghost_coordinate_segment::normalize_legacy_nonperiodic_boundary(&mut top)?;
    top.validate_analysis_residue_axis()
        .map_err(anyhow::Error::msg)?;
    let spatial_metric = crate::coordinate_cavity::physical_spatial_metric_authority(&top)?;
    let axis = top
        .atom_analysis_residue_ids
        .as_ref()
        .context("prepared topology residue axis")?
        .clone();
    ensure!(
        axis.len() == a.atom_count,
        "recorded fiber atom axis mismatch"
    );
    let arms = ["treatment", "control"];
    let mut environments = Vec::new();
    let mut protocols = Vec::new();
    let mut phases = Vec::new();
    for arm in arms {
        let aa = &base["mechanistic_sources"]["arms"][arm];
        let env = RecordedEnvironmentAuthority {
            schema_kind: "prism.recorded_environment_authority.v1".into(),
            topology: top_pin.clone(),
            spatial_metric: spatial_metric.clone(),
            protocol_state: source(&aa["protocol_state"])?,
            ghost_phase_binding: source(&aa["ghost_phase_binding"])?,
            run_id: a.arm_run_ids[arm].clone(),
            stream_id: a.stream_id,
            legacy_boundary_normalized_in_memory: normalized,
        };
        env.verify(&a, arm)?;
        protocols.push(serde_json::from_reader::<_, Value>(File::open(
            &env.protocol_state.path,
        )?)?);
        let phase: crate::ghost_phase_lattice::StreamProtocolPhaseBinding =
            serde_json::from_reader(File::open(&env.ghost_phase_binding.path)?)?;
        phase.verify()?;
        phases.push(phase);
        environments.push(env);
    }
    let configs = geometry_configs(&a)?;
    let mut indices = arms
        .map(|arm| {
            RowCursor::open(evaluated(
                &a,
                &format!("{arm}_native_geometry_index.jsonl.zst"),
            )?)
        })
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    let mut records = arms
        .map(|arm| {
            RowCursor::open(
                products
                    .get(&format!("{arm}_native_event_readout.jsonl.zst"))
                    .context("complete event readout missing")?
                    .clone(),
            )
        })
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    let frames_pin = products
        .get("paired_measurement_frames.jsonl.zst")
        .context("complete frame readout missing")?;
    let mut frames = RowCursor::open(frames_pin.clone())?;
    let region_counts = arms
        .map(|arm| {
            usize::try_from(u(
                &a.scope["geometry_grids"]["arms"][arm]["regions_per_frame"]
            )?)
            .map_err(anyhow::Error::from)
        })
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    fs::create_dir_all(out)?;
    let environment_path = out.join("recorded_environment_authorities.json");
    let environment_hashes = environments
        .iter()
        .map(canonical_sha256)
        .collect::<Result<Vec<_>>>()?;
    write_json(
        &environment_path,
        &json!({"schema_kind":"prism.recorded_environment_authorities.v1","authorities":environments,
        "authority_sha256":environment_hashes,"pressure_PME_chemical_potential_inferred":false,"native_runtime_absence_journals":"not_supplied; none manufactured"}),
    )?;
    let mut writers = arms
        .map(|arm| {
            Ok(zstd::stream::write::Encoder::new(
                BufWriter::new(File::create(
                    out.join(format!("{arm}_native_measurement_fibers.jsonl.zst")),
                )?),
                1,
            )?)
        })
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    let mut frame_writer = zstd::stream::write::Encoder::new(
        BufWriter::new(File::create(
            out.join("recorded_measurement_frame_fibers.jsonl.zst"),
        )?),
        1,
    )?;
    let mut populations = [Sha256::new(), Sha256::new()];
    let mut counts = [0u64; 2];
    let mut unmatched = [0u64; 2];
    let mut no_event_frames = [0u64; 2];
    for frame in 0..a.frame_steps.len() {
        let (frame_ordinal, frame_record) = frames.take()?;
        let t = &frame_record["native_vector_frame"];
        ensure!(
            frame_ordinal as usize == frame
                && u(&t["frame_ordinal"])? as usize == frame
                && u(&t["step"])? == a.frame_steps[frame]
                && t["phase"] == a.frame_phases[frame],
            "measurement frame chronology differs"
        );
        let frame_ref =
            NativeMeasurementRowReference::new(frames_pin, frame_ordinal, &frame_record)?;
        let before = counts;
        for (i, arm) in arms.iter().enumerate() {
            let physical_phase = measured_protocol_phase(&protocols[i], a.frame_steps[frame])?;
            ensure!(
                physical_phase == a.frame_phases[frame],
                "recorded protocol state differs from exact frame phase"
            );
            let regions = read_geometry_frame(
                &mut indices[i],
                &a,
                &axis,
                arm,
                frame,
                region_counts[i],
                &configs,
            )?;
            while let Some((_, row)) = &records[i].next {
                let step = u(&row["native_event"]["completed_step"])?;
                ensure!(
                    step >= a.frame_steps[frame],
                    "native event outside recorded measurement chronology"
                );
                if step > a.frame_steps[frame] {
                    break;
                }
                let (ordinal, row) = records[i].take()?;
                let event = &row["native_event"];
                let links = &row["receptor_face_correspondence"];
                let event_pin =
                    evaluated(&a, &format!("{arm}_native_event_measurements.jsonl.zst"))?;
                let link_pin =
                    evaluated(&a, &format!("{arm}_native_event_geometry_links.jsonl.zst"))?;
                checked_source_reference(&row["source_event"], &event_pin, ordinal, event)?;
                checked_source_reference(&row["source_geometry_links"], &link_pin, ordinal, links)?;
                let kcc = &frame_record["native_kcc_endpoints"][*arm];
                ensure!(
                    !kcc.is_null(),
                    "native event lacks recorded exact KCC endpoint"
                );
                check_event(
                    &a,
                    event,
                    links,
                    arm,
                    frame,
                    ordinal,
                    &event_pin,
                    &indices[i].source,
                    &regions,
                    t,
                    kcc,
                )?;
                let source_lineage: GhostRecordByteLineage =
                    serde_json::from_value(event["native_record_lineage"].clone())?;
                let phase = &phases[i];
                ensure!(
                    source_lineage.condition_id == phase.condition_id
                        && source_lineage.phase_schedule_digest_sha256
                            == phase.phase_schedule_digest_sha256
                        && source_lineage.protocol_state_hash == phase.protocol_state_hash,
                    "native event phase authority differs"
                );
                let ghost = crate::pathb_evidence_emitter::adapt_recorded_ghost_byte_lineage(
                    &source_lineage,
                    &phase.protocol_name,
                )?;
                let support = structural_support(&regions, links, &indices[i].source)?;
                unmatched[i] += u64::from(support.is_empty());
                let fiber = RecordedMeasurementFiber::materialize(RecordedMeasurementFiberBody {
                    recorded_lattice_phase_ordinal: ghost.protocol_phase,
                    ghost,
                    environment_authority_sha256: environment_hashes[i].clone(),
                    arm: (*arm).into(),
                    coordinate_frame_ordinal: frame,
                    completed_step: a.frame_steps[frame],
                    physical_time_ps: super::physical_time::timeline_arm_time(t, arm)?,
                    resolved_physical_phase: physical_phase.into(),
                    complete_native_readout_source: NativeMeasurementRowReference::new(
                        &records[i].source,
                        ordinal,
                        &row,
                    )?,
                    complete_measurement_frame: frame_ref.clone(),
                    native_measurements: event.clone(),
                    receptor_structural_support: support,
                    receptor_face_correspondence: links.clone(),
                })?;
                // Ordered hash population is domain-separated and count-bound;
                // individual fibers retain full independent content identities.
                populations[i].update(fiber.fiber_sha256.as_bytes());
                populations[i].update(b"\n");
                line(&mut writers[i], &fiber)?;
                counts[i] += 1;
            }
            no_event_frames[i] += u64::from(counts[i] == before[i]);
        }
        line(
            &mut frame_writer,
            &json!({"schema_kind":"prism.recorded_measurement_frame_fiber.v1","source":frame_ref,
            "environment_authority_sha256":environment_hashes,"complete_native_frame_measurements":frame_record,
            "native_event_fibers":{"treatment":counts[0]-before[0],"control":counts[1]-before[1]},
            "frame_without_events_retained":counts==before,"TSH_admission_used":false}),
        )?;
    }
    ensure!(frames.next.is_none(), "extra recorded measurement frames");
    for i in 0..2 {
        ensure!(
            records[i].next.is_none()
                && indices[i].next.is_none()
                && counts[i] == u(&receipt["native_event_records"][arms[i]])?,
            "measurement fiber population not conserved"
        );
    }
    for w in writers {
        w.finish()?.flush()?;
    }
    frame_writer.finish()?.flush()?;
    let population_sha256=populations.into_iter().enumerate().map(|(i,h)|canonical_sha256(&json!({
        "ordered_fiber_sha256_ascii_lf_digest":format!("{:x}",h.finalize()),"count":counts[i],"arm":arms[i],"source_receipt_sha256":m.readout_receipt.sha256})))
        .collect::<Result<Vec<_>>>()?;
    for p in products.values() {
        p.verify()?;
    }
    a.verify()?;
    m.readout_receipt.verify()?;
    artifact_pin.verify()?;
    manifest_pin.verify()?;
    let mut outputs = BTreeMap::new();
    for name in [
        "recorded_environment_authorities.json",
        "recorded_measurement_frame_fibers.jsonl.zst",
        "treatment_native_measurement_fibers.jsonl.zst",
        "control_native_measurement_fibers.jsonl.zst",
    ] {
        outputs.insert(name, pin(&out.join(name))?);
    }
    let result = json!({"schema_kind":"prism.recorded_native_measurement_fiber_bundle.v1","status":"IMPLEMENTATION_VERIFIED",
        "source_manifest":manifest_pin,"source_readout_receipt":m.readout_receipt,"source_measurement_artifact":artifact_pin,"outputs":outputs,
        "event_fibers":{"treatment":counts[0],"control":counts[1]},"ordered_fiber_population_sha256":population_sha256,
        "recorded_frame_fibers":a.frame_steps.len(),"events_without_receptor_support_retained":{"treatment":unmatched[0],"control":unmatched[1]},
        "frames_without_native_events_retained":{"treatment":no_event_frames[0],"control":no_event_frames[1]},
        "native_chemical_control_lanes_created":false,"runtime_no_probe_absence_receipts_created":false,"pressure_or_PME_invented":false,
        "Markov_or_open_state_inferred":false,"MD_executed":false,"native_tensor_fields_collapsed":false,
        "enrichment_key":"exact complete GhostByteLineage::lineage_sha256; chemical lane/thermodynamic/Markov evidence remains separately required"});
    write_json(&out.join("MEASUREMENT_FIBER_BUNDLE.json"), &result)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "actual complete saved pair: PRISM_MEASUREMENT_FIBER_MANIFEST and unique PRISM_MEASUREMENT_FIBER_OUTPUT"]
    fn actual_saved_pair_materializes_complete_measurement_fibers() -> Result<()> {
        let manifest = PathBuf::from(std::env::var("PRISM_MEASUREMENT_FIBER_MANIFEST")?);
        let out = PathBuf::from(std::env::var("PRISM_MEASUREMENT_FIBER_OUTPUT")?);
        let result = materialize_recorded_measurement_fibers(&manifest, &out)?;
        ensure!(
            result["status"] == "IMPLEMENTATION_VERIFIED"
                && u(&result["recorded_frame_fibers"])? > 0,
            "actual frame fibers missing"
        );
        for arm in ["treatment", "control"] {
            ensure!(
                u(&result["event_fibers"][arm])? > 0,
                "actual event fibers missing"
            );
        }
        ensure!(
            result["pressure_or_PME_invented"] == false
                && result["native_chemical_control_lanes_created"] == false,
            "physical contract changed"
        );
        Ok(())
    }
}
