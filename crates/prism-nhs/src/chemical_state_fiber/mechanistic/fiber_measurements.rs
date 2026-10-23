//! Stream existing native measurement fibers into physical-lane consumers.
//! The source bodies and original chemical bundle identities remain immutable.
use super::super::measurement_fibers::{RecordedEnvironmentAuthority, RecordedMeasurementFiber};
use super::*;
use crate::coordinate_cavity::PhysicalCavitySpatialMetricMode;
#[path="native_graph_association.rs"]
mod native_graph;

struct HashingReader<R> {
    inner: R,
    hash: Sha256,
}
impl<R: Read> Read for HashingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.hash.update(&buf[..n]);
        Ok(n)
    }
}

/// Content-addressed position of a complete original measurement fiber.
/// Consumers receive a typed borrowed record during streaming; this address
/// permits later resolution without duplicating any native tensor payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordedFiberReference {
    pub source: MeasurementSourcePin,
    pub line_ordinal: u64,
    pub fiber_sha256: String,
    pub ghost_lineage_sha256: String,
}

/// Verify and consume a whole compressed native fiber source once. Verification
/// includes compressed source bytes, every typed fiber, exact event identity,
/// duplicates and the producer's ordered/count-bound population digest.
/// Visitors must defer committing external effects until this function succeeds.
pub fn visit_recorded_fibers<F>(
    source: &MeasurementSourcePin,
    arm: &str,
    expected_count: u64,
    expected_population: &str,
    source_readout_sha256: &str,
    mut visit: F,
) -> Result<u64>
where
    F: FnMut(&RecordedFiberReference, &RecordedMeasurementFiber) -> Result<()>,
{
    require_sha256(&source.sha256, "recorded fiber source")?;
    require_sha256(expected_population, "recorded fiber population")?;
    require_sha256(source_readout_sha256, "recorded fiber source readout")?;
    let reader = HashingReader {
        inner: File::open(&source.path)?,
        hash: Sha256::new(),
    };
    let mut rows = BufReader::new(zstd::stream::read::Decoder::new(reader)?);
    let mut ordered = Sha256::new();
    let mut seen = BTreeSet::new();
    let mut count = 0u64;
    let mut previous = None;
    for raw in rows.by_ref().lines() {
        let fiber: RecordedMeasurementFiber = serde_json::from_str(&raw?)?;
        fiber.resolve_ghost(&fiber.body.ghost, &fiber.body.resolved_physical_phase)?;
        ensure!(fiber.body.arm == arm, "recorded fiber arm differs");
        let order = (
            fiber.body.coordinate_frame_ordinal,
            fiber.body.ghost.key.record_index,
        );
        ensure!(
            previous.is_none_or(|p| p < order),
            "recorded fiber chronology duplicate/reversed"
        );
        previous = Some(order);
        let lineage = fiber.body.ghost.lineage_sha256()?;
        ensure!(
            seen.insert(lineage.clone()),
            "duplicate recorded Ghost occurrence"
        );
        let reference = RecordedFiberReference {
            source: source.clone(),
            line_ordinal: count,
            fiber_sha256: fiber.fiber_sha256.clone(),
            ghost_lineage_sha256: lineage,
        };
        visit(&reference, &fiber)?;
        ordered.update(fiber.fiber_sha256.as_bytes());
        ordered.update(b"\n");
        count += 1;
    }
    let hashed = rows.into_inner().finish().into_inner();
    ensure!(
        format!("{:x}", hashed.hash.finalize()) == source.sha256,
        "recorded compressed fiber source changed"
    );
    let population = canonical_sha256(&json!({
        "ordered_fiber_sha256_ascii_lf_digest":format!("{:x}", ordered.finalize()),
        "count":count,"arm":arm,"source_receipt_sha256":source_readout_sha256}))?;
    ensure!(
        count == expected_count && population == expected_population,
        "recorded fiber population count/order mismatch"
    );
    Ok(count)
}

fn verify_environment_record(
    env: &RecordedEnvironmentAuthority,
    authority_hash: &str,
    fiber: &RecordedMeasurementFiber,
    artifact: &MechanisticMeasurementArtifact,
) -> Result<()> {
    env.spatial_metric.verify()?;
    ensure!(
        env.schema_kind == "prism.recorded_environment_authority.v1"
            && canonical_sha256(env)? == authority_hash
            && fiber.body.environment_authority_sha256 == authority_hash
            && env.run_id == fiber.body.ghost.run_id
            && env.stream_id == fiber.body.ghost.key.stream_id
            && env.topology.sha256 == fiber.body.ghost.topology_sha256
            && artifact.source_pins.contains(&env.topology)
            && artifact.source_pins.contains(&env.protocol_state)
            && artifact.source_pins.contains(&env.ghost_phase_binding),
        "recorded environment/source authority differs"
    );
    artifact.validate_ghost(&fiber.body.ghost)?;
    let i = fiber.body.coordinate_frame_ordinal;
    ensure!(
        artifact.frame_steps.get(i) == Some(&fiber.body.completed_step)
            && artifact.frame_phases.get(i) == Some(&fiber.body.resolved_physical_phase)
            && artifact.arm_run_ids.get(&fiber.body.arm) == Some(&env.run_id),
        "recorded fiber coordinate/protocol source differs"
    );
    Ok(())
}

fn verify_source_references(
    fiber: &RecordedMeasurementFiber,
    event_ordinal: u64,
    event_source: &MeasurementSourcePin,
    frame_source: &MeasurementSourcePin,
    expected_physical_time_ps: f64,
) -> Result<()> {
    ensure!(
        fiber.body.complete_native_readout_source.source == *event_source
            && fiber.body.complete_native_readout_source.line_ordinal == event_ordinal
            && fiber.body.complete_measurement_frame.source == *frame_source
            && fiber.body.complete_measurement_frame.line_ordinal
                == fiber.body.coordinate_frame_ordinal as u64
            && fiber.body.physical_time_ps == expected_physical_time_ps,
        "recorded fiber source references or independent per-arm vector clock differ"
    );
    Ok(())
}

/// Consume all saved native records and attach exact addresses to an optional
/// verified chemical bundle. Full data are available to the native resolver;
/// the output index retains every matched and unmatched source record. Missing
/// chemical inputs cannot be replaced by the observation pair.
pub fn consume_recorded_measurement_bundle(
    recorded_pin: &MeasurementSourcePin,
    readout_pin: &MeasurementSourcePin,
    artifact_pin: &MeasurementSourcePin,
    artifact: &MechanisticMeasurementArtifact,
    physical: Option<&NativePathBFiberBundle>,
    physical_pin: Option<&MeasurementSourcePin>,
    manifest_pin: &MeasurementSourcePin,
    missing: &[&str],
    associate_native_graph: bool,
    out: &Path,
) -> Result<Value> {
    ensure!(
        !out.exists(),
        "refusing to replace recorded fiber consumer output"
    );
    let receipt: Value = read_pin(recorded_pin)?;
    ensure!(
        receipt["schema_kind"] == "prism.recorded_native_measurement_fiber_bundle.v1"
            && receipt["status"] == "IMPLEMENTATION_VERIFIED"
            && source(&receipt["source_readout_receipt"])? == *readout_pin
            && source(&receipt["source_measurement_artifact"])? == *artifact_pin
            && u(&receipt["recorded_frame_fibers"])? as usize == artifact.frame_steps.len(),
        "recorded fiber bundle has different source population"
    );
    let products: BTreeMap<String, MeasurementSourcePin> =
        serde_json::from_value(receipt["outputs"].clone())?;
    let environment_pin = products
        .get("recorded_environment_authorities.json")
        .context("recorded environments")?;
    let environments: Value = read_pin(environment_pin)?;
    ensure!(
        environments["schema_kind"] == "prism.recorded_environment_authorities.v1",
        "recorded environment schema"
    );
    let authorities: Vec<RecordedEnvironmentAuthority> =
        serde_json::from_value(environments["authorities"].clone())?;
    let authority_hashes: Vec<String> =
        serde_json::from_value(environments["authority_sha256"].clone())?;
    ensure!(
        authorities.len() == 2 && authority_hashes.len() == 2,
        "paired environment population differs"
    );
    let frame_pin = products
        .get("recorded_measurement_frame_fibers.jsonl.zst")
        .context("all recorded frame fibers")?;
    frame_pin.verify()?;
    let readout: Value = read_pin(readout_pin)?;
    let readout_products: BTreeMap<String, MeasurementSourcePin> =
        serde_json::from_value(readout["outputs"].clone())?;
    let readout_frames = readout_products
        .get("paired_measurement_frames.jsonl.zst")
        .context("source readout paired frames")?;
    let timeline_pin = artifact
        .evaluated_artifacts
        .get("vector_frames.jsonl")
        .context("source native vector chronology")?;
    timeline_pin.verify()?;
    let mut arm_times = Vec::with_capacity(artifact.frame_steps.len());
    for (ordinal, raw) in BufReader::new(File::open(&timeline_pin.path)?)
        .lines()
        .enumerate()
    {
        let row: Value = serde_json::from_str(&raw?)?;
        ensure!(
            u(&row["frame_ordinal"])? as usize == ordinal
                && artifact.frame_steps.get(ordinal) == Some(&u(&row["step"])?)
                && artifact.frame_phases.get(ordinal).map(String::as_str) == row["phase"].as_str(),
            "native vector chronology differs from artifact"
        );
        arm_times.push([
            super::super::physical_time::timeline_arm_time(&row, "treatment")?,
            super::super::physical_time::timeline_arm_time(&row, "control")?,
        ]);
    }
    ensure!(
        arm_times.len() == artifact.frame_steps.len(),
        "native vector frame count differs"
    );
    let mut by_lineage = BTreeMap::new();
    if let Some(bundle) = physical {
        bundle.verify()?;
        for (i, fiber) in bundle.fibers.iter().enumerate() {
            artifact.validate_fiber_phase(fiber)?;
            ensure!(
                by_lineage
                    .insert(fiber.body.ghost.lineage_sha256()?, i)
                    .is_none(),
                "duplicate physical fiber Ghost occurrence"
            );
        }
    }
    fs::create_dir_all(out)?;
    let links_path = out.join("native_fiber_measurement_links.jsonl.zst");
    let mut links =
        zstd::stream::write::Encoder::new(BufWriter::new(File::create(&links_path)?), 1)?;
    let mut matched = BTreeSet::new();
    let mut counts = BTreeMap::new();
    let mut native_graph_receipts=BTreeMap::new();
    for (i, arm) in ["treatment", "control"].iter().enumerate() {
        let p = products
            .get(&format!("{arm}_native_measurement_fibers.jsonl.zst"))
            .context("complete native fibers")?;
        let expected = u(&receipt["event_fibers"][*arm])?;
        let population = text(&receipt["ordered_fiber_population_sha256"][i])?;
        let mut native_graph=if associate_native_graph {
            Some(native_graph::NativeGraphAssociation::load(artifact,arm,out)?)
        } else {None};
        let count = visit_recorded_fibers(
            p,
            arm,
            expected,
            population,
            &readout_pin.sha256,
            |reference, observation| {
                verify_environment_record(
                    &authorities[i],
                    &authority_hashes[i],
                    observation,
                    artifact,
                )?;
                let event_source = readout_products
                    .get(&format!("{arm}_native_event_readout.jsonl.zst"))
                    .context("source arm event readout")?;
                verify_source_references(
                    observation,
                    reference.line_ordinal,
                    event_source,
                    readout_frames,
                    arm_times
                        .get(observation.body.coordinate_frame_ordinal)
                        .context("native fiber vector frame")?[i],
                )?;
                let native_graph_link=native_graph.as_mut()
                    .map(|graph|graph.associate(reference,observation)).transpose()?;
                let index = by_lineage.get(&reference.ghost_lineage_sha256).copied();
                let chemical = if let Some(index) = index {
                    ensure!(
                        matched.insert(index),
                        "physical fiber linked more than once"
                    );
                    let chemical = &physical.context("physical bundle")?.fibers[index];
                    observation.resolve_physical(chemical)?;
                    ensure!(matches!(authorities[i].spatial_metric.mode, PhysicalCavitySpatialMetricMode::PeriodicTriclinicMinimumImage)
                    && authorities[i].spatial_metric.box_vectors_angstrom == Some(chemical.body.thermodynamic_condition.periodic_box_vectors_a),
                    "physical chemical fiber spatial environment differs from recorded measurement");
                    Some(chemical)
                } else {
                    None
                };
                line(
                    &mut links,
                    &json!({"schema_kind":"prism.physical_fiber_native_measurement_link.v1",
                "recorded_measurement":reference,"physical_fiber_sha256":chemical.map(|f| &f.fiber_sha256),
                "native_graph_association":native_graph_link,
                "physical_control_lane":chemical.map(|f| f.body.control_lane),
                "match_status":if chemical.is_some(){"EXACT_NATIVE_LINEAGE"}else{"UNMATCHED_OBSERVATION_RETAINED"},
                "arm":observation.body.arm,"stream_id":observation.body.ghost.key.stream_id,
                "completed_step":observation.body.completed_step,"physical_time_ps":observation.body.physical_time_ps,
                "resolved_physical_phase":observation.body.resolved_physical_phase,
                "environment_authority_sha256":observation.body.environment_authority_sha256,
                "native_readout_source":observation.body.complete_native_readout_source,
                "complete_measurement_frame":observation.body.complete_measurement_frame}),
                )
            },
        )?;
        if let Some(graph)=native_graph {native_graph_receipts.insert(*arm,graph.finish(out)?);}
        counts.insert(*arm, count);
    }
    ensure!(
        matched.len() == by_lineage.len(),
        "physical fiber lacks exact recorded measurements"
    );
    links.finish()?.flush()?;
    recorded_pin.verify()?;
    readout_pin.verify()?;
    manifest_pin.verify()?;
    if let Some(p) = physical_pin {
        p.verify()?;
    }
    // A constructed physical bundle is retained exactly; no evidence insertion
    // changes the existing content identities. The association index is additive.
    let retained_physical = match (physical, physical_pin) {
        (Some(_), Some(p)) => Some(p.clone()),
        (Some(bundle), None) => {
            let path = out.join("native_pathb_fiber_bundle.json");
            write_json(&path, bundle)?;
            Some(pin(&path)?)
        }
        _ => None,
    };
    let result = json!({"schema_kind":"prism.native_fiber_measurement_consumer_receipt.v1",
        "status":if physical.is_some() || associate_native_graph {"IMPLEMENTATION_VERIFIED"}else{"IMPLEMENTATION_PARTIAL"},
        "requested_native_graph_association":associate_native_graph,
        "native_graph_association_status":if associate_native_graph {"IMPLEMENTATION_VERIFIED"}else{"NOT_REQUESTED"},
        "native_graph_association_receipts":native_graph_receipts,
        "native_measurement_consumption_status":"IMPLEMENTATION_VERIFIED",
        "physical_lane_attachment_status":if physical.is_some(){"IMPLEMENTATION_VERIFIED"}else{"REQUIRED_PHYSICAL_INPUTS_ABSENT"},
        "source_manifest":manifest_pin,"recorded_measurement_bundle":recorded_pin,
        "source_artifact":artifact_pin,"source_readout_receipt":readout_pin,
        "physical_bundle":retained_physical,"physical_fiber_population_sha256":physical.map(|p| &p.fiber_population_sha256),
        "native_measurement_links":pin(&links_path)?,"physical_fibers_linked":matched.len(),
        "observations_consumed":counts,"ordered_fiber_population_sha256":receipt["ordered_fiber_population_sha256"],
        "recorded_frame_fibers":receipt["recorded_frame_fibers"],"retained_original_products":products,
        "required_physical_inputs":missing,"source_fiber_identities_changed":false,
        "native_tensor_fields_collapsed":false,"unmatched_observations_retained":true,
        "chemical_lanes_created":false,"MD_executed":false});
    write_json(&out.join("EXECUTION_RECEIPT.json"), &result)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "actual saved original row: PRISM_RECORDED_FIBER_FIXTURE"]
    fn actual_measurement_resolves_exact_ghost_and_rejects_mismatch() -> Result<()> {
        let path = PathBuf::from(std::env::var("PRISM_RECORDED_FIBER_FIXTURE")?);
        let record: RecordedMeasurementFiber = serde_json::from_reader(File::open(path)?)?;
        let event_pin = record.body.complete_native_readout_source.source.clone();
        let frame_pin = record.body.complete_measurement_frame.source.clone();
        let event_ordinal = record.body.complete_native_readout_source.line_ordinal;
        verify_source_references(
            &record,
            event_ordinal,
            &event_pin,
            &frame_pin,
            record.body.physical_time_ps,
        )?;
        let mut wrong_pin = event_pin.clone();
        wrong_pin.sha256 = "0".repeat(64);
        ensure!(
            verify_source_references(
                &record,
                event_ordinal,
                &wrong_pin,
                &frame_pin,
                record.body.physical_time_ps
            )
            .is_err(),
            "unrelated event source accepted"
        );
        ensure!(
            verify_source_references(
                &record,
                event_ordinal + 1,
                &event_pin,
                &frame_pin,
                record.body.physical_time_ps
            )
            .is_err(),
            "wrong event ordinal accepted"
        );
        ensure!(
            verify_source_references(
                &record,
                event_ordinal,
                &event_pin,
                &frame_pin,
                record.body.physical_time_ps + 1.0
            )
            .is_err(),
            "independent clock mismatch accepted"
        );
        let mut wrong_frame = record.clone();
        wrong_frame.body.complete_measurement_frame.line_ordinal += 1;
        ensure!(
            verify_source_references(
                &wrong_frame,
                event_ordinal,
                &event_pin,
                &frame_pin,
                record.body.physical_time_ps
            )
            .is_err(),
            "wrong frame ordinal accepted"
        );
        let original = record.fiber_sha256.clone();
        let resolved =
            record.resolve_ghost(&record.body.ghost, &record.body.resolved_physical_phase)?;
        ensure!(
            std::ptr::eq(resolved, &record) && original == resolved.fiber_sha256,
            "typed resolver did not preserve full original record"
        );
        let mut wrong = record.body.ghost.clone();
        wrong.key.stream_id += 1;
        ensure!(
            record
                .resolve_ghost(&wrong, &record.body.resolved_physical_phase)
                .is_err(),
            "cross-stream join accepted"
        );
        ensure!(
            record
                .resolve_ghost(&record.body.ghost, "wrong_phase")
                .is_err(),
            "wrong phase accepted"
        );
        let mut wrong_time = record.clone();
        wrong_time.body.physical_time_ps += 1.0;
        wrong_time.fiber_sha256 = canonical_sha256(&wrong_time.body)?;
        ensure!(
            wrong_time
                .resolve_ghost(
                    &wrong_time.body.ghost,
                    &wrong_time.body.resolved_physical_phase
                )
                .is_err(),
            "changed native clock accepted despite valid replacement body hash"
        );
        Ok(())
    }
    #[test]
    #[ignore = "actual saved original row: PRISM_RECORDED_FIBER_FIXTURE"]
    fn actual_recorded_population_checks_source_count_and_duplicates() -> Result<()> {
        let original = fs::read(PathBuf::from(std::env::var(
            "PRISM_RECORDED_FIBER_FIXTURE",
        )?))?;
        let record: RecordedMeasurementFiber = serde_json::from_slice(&original)?;
        record.verify()?;
        let temp = std::env::temp_dir().join(format!(
            "prism_recorded_fiber_contract_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        fs::create_dir(&temp)?;
        let path = temp.join("actual_original_row.zst");
        fs::write(&path, zstd::stream::encode_all(original.as_slice(), 1)?)?;
        let source = pin(&path)?;
        let readout = "edd083f3b487d7c040b04f9f211831808715be384984593e997e2e46cb8e555d";
        let mut hash = Sha256::new();
        hash.update(record.fiber_sha256.as_bytes());
        hash.update(b"\n");
        let population = canonical_sha256(
            &json!({"ordered_fiber_sha256_ascii_lf_digest":format!("{:x}",hash.finalize()),
            "count":1,"arm":record.body.arm,"source_receipt_sha256":readout}),
        )?;
        let mut visited = 0;
        let count = visit_recorded_fibers(
            &source,
            &record.body.arm,
            1,
            &population,
            readout,
            |reference, full| {
                ensure!(
                    reference.line_ordinal == 0
                        && reference.fiber_sha256 == record.fiber_sha256
                        && full.body.native_measurements == record.body.native_measurements
                        && full.body.receptor_structural_support.len()
                            == record.body.receptor_structural_support.len(),
                    "full actual native record was altered or dropped"
                );
                visited += 1;
                Ok(())
            },
        )?;
        ensure!(
            count == 1 && visited == 1,
            "actual record visitor population"
        );
        ensure!(
            visit_recorded_fibers(
                &source,
                &record.body.arm,
                2,
                &population,
                readout,
                |_, _| Ok(())
            )
            .is_err(),
            "wrong expected population accepted"
        );
        let mut wrong_source = source.clone();
        wrong_source.sha256 = "0".repeat(64);
        ensure!(
            visit_recorded_fibers(
                &wrong_source,
                &record.body.arm,
                1,
                &population,
                readout,
                |_, _| Ok(())
            )
            .is_err(),
            "wrong source bytes accepted"
        );
        let mut duplicate = original.clone();
        duplicate.extend_from_slice(&original);
        fs::write(&path, zstd::stream::encode_all(duplicate.as_slice(), 1)?)?;
        let repeated = pin(&path)?;
        ensure!(
            visit_recorded_fibers(
                &repeated,
                &record.body.arm,
                2,
                &population,
                readout,
                |_, _| Ok(())
            )
            .is_err(),
            "duplicated actual Ghost occurrence accepted"
        );
        fs::remove_dir_all(temp)?;
        Ok(())
    }
}
