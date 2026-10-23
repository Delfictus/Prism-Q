//! Attach completed native readout to real, validated chemical-state fibers.
//! Physical lanes, Hamiltonians, thermodynamic states and sampling uncertainty
//! must come from existing native inputs; this adapter does not create them.
use super::*;
#[cfg(any(feature = "gpu", feature = "cpu-analysis"))]
#[path = "fiber_measurements.rs"]
pub mod recorded;
use crate::chemical_state_fiber::{
    ChemicalStateFiber, FragmentStateObservation, GhostByteLineage, GhostFiberObservation,
};
use crate::pathb_chemical_state::{
    materialize_native_pathb_fibers, NativePathBFiberBundle, NativePathBMaterialization,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FiberInputs {
    pathb_materialization: MeasurementSourcePin,
    ghost_observations: MeasurementSourcePin,
    fragment_observations: MeasurementSourcePin,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_kind: String,
    readout_receipt: MeasurementSourcePin,
    existing_bundle: Option<MeasurementSourcePin>,
    physical_inputs: Option<FiberInputs>,
    /// Optional immutable completed measurement-fiber population. Native records
    /// are consumed directly; none of the 467,400 historical bodies is rewritten.
    recorded_measurement_bundle: Option<MeasurementSourcePin>,
    /// Join recorded physical fibers to the existing full native Ghost/Path-B
    /// connectivity substrate. Does not request chemical-lane estimation.
    #[serde(default)]
    native_graph_association: bool,
}
fn read_pin<T: serde::de::DeserializeOwned>(pin: &MeasurementSourcePin) -> Result<T> {
    pin.verify()?;
    Ok(serde_json::from_reader(File::open(&pin.path)?)?)
}
fn prerequisite_reasons(topology: &Value, m: &Manifest) -> Vec<&'static str> {
    let mut reasons = Vec::new();
    if topology["periodic_cell"]["periodic"] == false {
        reasons
            .push("recorded_nonperiodic_topology_cannot_supply_existing_periodic_fiber_contract");
    }
    if m.existing_bundle.is_none() && m.physical_inputs.is_none() {
        reasons.push("native_bundle_or_exact_pathb_ghost_fragment_inputs_not_supplied");
    }
    reasons
}
fn occurrence_key(row: &Value) -> Result<String> {
    Ok(format!(
        "{}:{}:{}:{}",
        text(&row["run_id"])?,
        text(&row["segment_manifest_sha256"])?,
        text(&row["segment_file_sha256"])?,
        u(&row["payload_byte_offset"])?
    ))
}
fn fiber_occurrence_key(g: &GhostByteLineage) -> String {
    format!(
        "{}:{}:{}:{}",
        g.run_id, g.segment_manifest_sha256, g.segment_sha256, g.byte_offset
    )
}
/// Compare every recorded lineage field against the existing native fiber
/// lineage. No new lineage or physical lane is constructed by this join.
fn exact_ghost(g: &GhostByteLineage, row: &Value, protocol_name: &str) -> Result<()> {
    g.validate()?;
    let source = serde_json::to_value(g)?;
    for (fiber_field, event_field) in [
        ("run_id", "run_id"),
        ("replica_id", "replica_id"),
        ("replica_seed", "replica_seed"),
        ("stream_seed", "stream_seed"),
        ("topology_sha256", "topology_sha256"),
        ("record_abi_sha256", "record_abi_sha256"),
        ("record_source_sha256", "record_source_sha256"),
        (
            "segment_context_digest_sha256",
            "segment_context_digest_sha256",
        ),
        ("segment_manifest_sha256", "segment_manifest_sha256"),
        ("segment_sequence", "segment_ordinal"),
        ("segment_sha256", "segment_file_sha256"),
        ("segment_file", "segment_file"),
        ("byte_offset", "payload_byte_offset"),
        ("byte_length", "byte_length"),
        ("record_sha256", "record_sha256"),
        ("lattice_node_index", "global_lattice_index"),
        ("engine_sha256", "selected_engine_sha256"),
        ("condition_id", "condition_id"),
        ("protocol_phase", "protocol_phase"),
        (
            "phase_schedule_digest_sha256",
            "phase_schedule_digest_sha256",
        ),
        ("protocol_state_hash", "protocol_state_hash"),
    ] {
        ensure!(
            source[fiber_field] == row[event_field] && !row[event_field].is_null(),
            "fiber/readout lineage differs: {fiber_field}"
        );
    }
    for (fiber_field, event_field) in [
        ("stream_id", "stream_id"),
        ("frame_idx", "frame_idx"),
        ("step_idx", "step_idx"),
        ("site_id", "site_id"),
        ("record_index", "stream_record_ordinal"),
    ] {
        ensure!(
            source["key"][fiber_field] == row[event_field],
            "fiber/readout key differs: {fiber_field}"
        );
    }
    ensure!(
        g.protocol_name == protocol_name,
        "fiber protocol differs from recorded phase binding"
    );
    Ok(())
}

/// Execute the existing native constructors and attach exact, full readout
/// correspondence. Inapplicable recorded inputs produce explicit requirements
/// and preserve all prior artifacts; no physical fiber is synthesized.
pub fn integrate_native_fiber_readout(manifest_path: &Path, out: &Path) -> Result<Value> {
    ensure!(
        !out.exists(),
        "refusing to replace fiber integration output"
    );
    let manifest_pin = pin(manifest_path)?;
    // Keep the existing v1 recorded-measurement and graph-association contract
    // unchanged. V2 explicitly binds multiple paired artifacts to chemical lanes.
    let value: Value = serde_json::from_reader(File::open(manifest_path)?)?;
    if value["schema_kind"] == "prism.native_fiber_readout_manifest.v2" {
        // Parse directly to retain duplicate-key detection in the typed lane map.
        let multi: MultiSourceManifest = serde_json::from_reader(File::open(manifest_path)?)?;
        return integrate_multi_source_physical_readout(manifest_pin,multi,out);
    }
    let m: Manifest = serde_json::from_reader(File::open(manifest_path)?)?;
    ensure!(
        m.schema_kind == "prism.native_fiber_readout_manifest.v1",
        "fiber integration manifest schema"
    );
    ensure!(
        !(m.existing_bundle.is_some() && m.physical_inputs.is_some()),
        "choose existing bundle OR exact native constructor inputs"
    );
    let readout: Value = read_pin(&m.readout_receipt)?;
    ensure!(
        readout["schema_kind"] == "prism.native_paired_mechanistic_readout_receipt.v1"
            && readout["status"] == "IMPLEMENTATION_VERIFIED",
        "completed native readout required"
    );
    let a_pin: MeasurementSourcePin = serde_json::from_value(readout["source_artifact"].clone())?;
    a_pin.verify()?;
    let a = MechanisticMeasurementArtifact::load_verified(&a_pin.path)?;
    ensure!(
        readout["source_manifest_sha256"] == a.manifest_sha256
            && u(&readout["frames"])? as usize == a.frame_steps.len(),
        "readout/artifact source or chronology mismatch"
    );
    let products: BTreeMap<String, MeasurementSourcePin> =
        serde_json::from_value(readout["outputs"].clone())?;
    for p in products.values() {
        p.verify()?;
    }
    for name in [
        "paired_measurement_frames.jsonl.zst",
        "paired_receptor_membership.jsonl.zst",
        "treatment_native_event_readout.jsonl.zst",
        "control_native_event_readout.jsonl.zst",
    ] {
        ensure!(
            products.contains_key(name),
            "readout product missing: {name}"
        );
    }
    let source_manifest = a
        .source_pins
        .iter()
        .find(|p| p.sha256 == a.manifest_sha256)
        .context("source measurement manifest")?;
    let source: Value = serde_json::from_reader(File::open(&source_manifest.path)?)?;
    let topology = load(&source["mechanistic_sources"]["topology"])?;
    let reasons = prerequisite_reasons(&topology, &m);
    #[cfg(any(feature = "gpu", feature = "cpu-analysis"))]
    if let Some(recorded) = &m.recorded_measurement_bundle {
        // A supplied physical bundle must be verified even when other inputs
        // are inapplicable. Never silently drop requested physical attachment.
        let physical = if let Some(p) = &m.existing_bundle {
            Some(read_pin::<NativePathBFiberBundle>(p)?)
        } else if let Some(p) = &m.physical_inputs {
            Some(materialize_native_pathb_fibers(
                read_pin(&p.pathb_materialization)?,
                read_pin(&p.ghost_observations)?,
                read_pin(&p.fragment_observations)?,
            )?)
        } else {
            None
        };
        if physical.is_some() {
            ensure!(
                reasons.is_empty(),
                "physical bundle conflicts with recorded environment: {reasons:?}"
            );
        }
        return recorded::consume_recorded_measurement_bundle(
            recorded,
            &m.readout_receipt,
            &a_pin,
            &a,
            physical.as_ref(),
            m.existing_bundle.as_ref(),
            &manifest_pin,
            &reasons,
            m.native_graph_association,
            out,
        );
    }
    ensure!(!m.native_graph_association,
        "native graph association requires a supplied recorded measurement bundle");
    #[cfg(not(any(feature = "gpu", feature = "cpu-analysis")))]
    ensure!(
        m.recorded_measurement_bundle.is_none(),
        "recorded fiber consumer requires cpu-analysis or gpu feature"
    );
    if !reasons.is_empty() {
        fs::create_dir_all(out)?;
        let receipt = json!({"schema_kind":"prism.native_fiber_readout_integration_receipt.v1","status":"IMPLEMENTATION_BLOCKED",
            "manifest":manifest_pin,"readout_receipt":m.readout_receipt,"source_artifact":a_pin,
            "physical_fiber_prerequisites":reasons,"recorded_periodic_cell":topology["periodic_cell"],
            "required_native_input_contracts":["NativePathBFiberBundle OR NativePathBMaterialization + GhostFiberObservation[] + FragmentStateObservation[]",
                "existing ThermodynamicCondition validator requires actual nondegenerate periodic box and PME identity",
                "control lanes retain native chemical identities; UV treatment/control labels do not instantiate fragment lanes"],
            "retained_readout_products":products,"fibers_created":0,"existing_artifacts_modified":false,"MD_executed":false});
        write_json(&out.join("EXECUTION_RECEIPT.json"), &receipt)?;
        return Ok(receipt);
    }
    let mut inputs = vec![
        m.readout_receipt.clone(),
        a_pin.clone(),
        manifest_pin.clone(),
    ];
    let bundle = if let Some(p) = &m.existing_bundle {
        inputs.push(p.clone());
        let b: NativePathBFiberBundle = read_pin(p)?;
        b.verify()?;
        b
    } else {
        let p = m
            .physical_inputs
            .as_ref()
            .context("native physical inputs missing")?;
        inputs.extend([
            p.pathb_materialization.clone(),
            p.ghost_observations.clone(),
            p.fragment_observations.clone(),
        ]);
        let pathb: NativePathBMaterialization = read_pin(&p.pathb_materialization)?;
        let ghost: Vec<GhostFiberObservation> = read_pin(&p.ghost_observations)?;
        let fragment: Vec<FragmentStateObservation> = read_pin(&p.fragment_observations)?;
        materialize_native_pathb_fibers(pathb, ghost, fragment)?
    };
    // Existing validators preserve phase, lane, endpoint eligibility, Markov
    // model and complete population. Never reset any of their requirements.
    let mut attached = a.attach_to_bundle(&bundle)?;
    let mut by_occurrence = BTreeMap::new();
    for (i, f) in attached.fibers.iter().enumerate() {
        ensure!(
            by_occurrence
                .insert(fiber_occurrence_key(&f.body.ghost), i)
                .is_none(),
            "duplicate native fiber occurrence"
        );
    }
    fs::create_dir_all(out)?;
    let mut associations = zstd::stream::write::Encoder::new(
        BufWriter::new(File::create(
            out.join("fiber_native_readout_links.jsonl.zst"),
        )?),
        1,
    )?;
    let mut seen = BTreeSet::new();
    let mut read_counts = BTreeMap::new();
    for arm in ["treatment", "control"] {
        let phase = load(&source["mechanistic_sources"]["arms"][arm]["ghost_phase_binding"])?;
        let protocol_name = text(&phase["protocol_name"])?;
        let product = &products[&format!("{arm}_native_event_readout.jsonl.zst")];
        let rows = BufReader::new(zstd::stream::read::Decoder::new(File::open(
            &product.path,
        )?)?)
        .lines();
        let mut count = 0u64;
        for (ordinal, raw) in rows.enumerate() {
            let row: Value = serde_json::from_str(&raw?)?;
            count += 1;
            ensure!(
                row["schema_kind"] == "prism.native_event_readout.v1",
                "native event readout schema"
            );
            let event = &row["native_event"];
            let lineage = &event["native_record_lineage"];
            ensure!(
                event["arm"] == arm
                    && lineage["run_id"] == a.arm_run_ids[arm]
                    && lineage["topology_sha256"] == a.topology_sha256,
                "readout source arm/run/topology"
            );
            let key = occurrence_key(lineage)?;
            let Some(&i) = by_occurrence.get(&key) else {
                continue;
            };
            ensure!(seen.insert(i), "duplicate readout record for native fiber");
            let fiber = &attached.fibers[i];
            exact_ghost(&fiber.body.ghost, lineage, protocol_name)?;
            ensure!(
                fiber.body.protocol_phase.phase_name == event["resolved_physical_phase"],
                "fiber physical phase differs from native readout"
            );
            let event_hash = canonical_sha256(&row)?;
            let mut body = fiber.body.clone();
            for (name, digest) in [
                (
                    "native_paired_readout_receipt_v1",
                    m.readout_receipt.sha256.clone(),
                ),
                ("native_paired_readout_event_v1", event_hash.clone()),
            ] {
                ensure!(
                    body.evidence_artifact_sha256
                        .get(name)
                        .is_none_or(|existing| existing == &digest),
                    "existing fiber evidence name has different identity"
                );
                body.evidence_artifact_sha256.insert(name.into(), digest);
            }
            let updated = ChemicalStateFiber::materialize(body)?;
            line(
                &mut associations,
                &json!({"original_fiber_sha256":bundle.fibers[i].fiber_sha256,
                "fiber_sha256":updated.fiber_sha256,"ghost_lineage_sha256":updated.body.ghost.lineage_sha256()?,
                "control_lane":updated.body.control_lane,"readout_event_source":product,"readout_event_line_ordinal":ordinal,
                "readout_event_canonical_sha256":event_hash,"full_native_event_and_measurements":"complete immutable readout row retained; all tensor/vector/contact/KCC/ASC and receptor-face references unchanged",
                "paired_measurement_frame":row["paired_measurement_frame"],"receptor_face_correspondence":row["receptor_face_correspondence"]}),
            )?;
            attached.fibers[i] = updated;
        }
        ensure!(
            count == u(&readout["native_event_records"][arm])?,
            "native readout event population mismatch"
        );
        read_counts.insert(arm, count);
    }
    ensure!(
        seen.len() == attached.fibers.len(),
        "physical fiber lacks exact native readout occurrence"
    );
    let mut hashes = attached
        .fibers
        .iter()
        .map(|f| f.fiber_sha256.clone())
        .collect::<Vec<_>>();
    hashes.sort();
    attached.fiber_population_sha256 = canonical_sha256(&hashes)?;
    attached.verify()?;
    associations.finish()?.flush()?;
    for p in inputs.iter().chain(products.values()) {
        p.verify()?;
    }
    write_json(&out.join("native_pathb_fiber_bundle.json"), &attached)?;
    let receipt = json!({"schema_kind":"prism.native_fiber_readout_integration_receipt.v1","status":"IMPLEMENTATION_VERIFIED",
        "manifest":manifest_pin,"readout_receipt":m.readout_receipt,"input_native_fiber_population_sha256":bundle.fiber_population_sha256,
        "fiber_bundle":pin(&out.join("native_pathb_fiber_bundle.json"))?,"exact_native_readout_links":pin(&out.join("fiber_native_readout_links.jsonl.zst"))?,
        "fiber_count":attached.fibers.len(),"event_records_consumed":read_counts,"unmatched_readout_events":"preserved in immutable complete readout products",
        "control_lanes_changed":false,"new_physical_conditions_created":false,"MD_executed":false,"source_inputs":inputs});
    write_json(&out.join("EXECUTION_RECEIPT.json"), &receipt)?;
    Ok(receipt)
}

struct VerifiedReadoutFrame {
    step: u64,
    phase: String,
    window_id: Value,
    arm_times: [f64;2],
    vector_row_sha256: String,
}

fn read_verified_frame_authorities(
    artifact:&MechanisticMeasurementArtifact,
    products:&BTreeMap<String,MeasurementSourcePin>,
) -> Result<Vec<VerifiedReadoutFrame>> {
    let timeline=artifact.evaluated_artifacts.get("vector_frames.jsonl")
        .context("source native vector chronology")?;
    let mut frames=Vec::with_capacity(artifact.frame_steps.len());
    for (ordinal,raw) in BufReader::new(File::open(&timeline.path)?).lines().enumerate() {
        let row:Value=serde_json::from_str(&raw?)?;
        let step=u(&row["step"])?;let phase=text(&row["phase"])?;
        ensure!(u(&row["frame_ordinal"])?==ordinal as u64
            && artifact.frame_steps.get(ordinal)==Some(&step)
            && artifact.frame_phases.get(ordinal).map(String::as_str)==Some(phase),
            "native vector chronology differs from artifact");
        frames.push(VerifiedReadoutFrame{step,phase:phase.into(),window_id:row["window_id"].clone(),
            arm_times:[super::physical_time::timeline_arm_time(&row,"treatment")?,
                super::physical_time::timeline_arm_time(&row,"control")?],
            vector_row_sha256:canonical_sha256(&row)?});
    }
    ensure!(frames.len()==artifact.frame_steps.len(),"native vector frame population differs");
    let paired=&products["paired_measurement_frames.jsonl.zst"];
    let mut count=0usize;
    for (ordinal,raw) in BufReader::new(zstd::stream::read::Decoder::new(File::open(&paired.path)?)?).lines().enumerate() {
        let row:Value=serde_json::from_str(&raw?)?;
        let authority=frames.get(ordinal).context("extra paired measurement frame")?;
        let reference=&row["source_vector_frame"];
        ensure!(row["schema_kind"]=="prism.native_paired_measurement_readout_frame.v1"
            && row["topology_sha256"]==artifact.topology_sha256
            && row["run_ids"]==serde_json::to_value(&artifact.arm_run_ids)?
            && u(&row["stream_id"])?==u64::from(artifact.stream_id)
            && serde_json::from_value::<MeasurementSourcePin>(reference["source"].clone())?==*timeline
            && u(&reference["line_ordinal"])?==ordinal as u64
            && reference["canonical_row_sha256"]==authority.vector_row_sha256
            && canonical_sha256(&row["native_vector_frame"])?==authority.vector_row_sha256,
            "paired measurement frame differs from exact source vector chronology");
        count+=1;
    }
    ensure!(count==frames.len(),"paired measurement frame population differs");
    Ok(frames)
}

fn verify_event_frame_authority(row:&Value,arm:&str,frames:&[VerifiedReadoutFrame]) -> Result<usize> {
    let arm_index=match arm {"treatment"=>0,"control"=>1,_=>bail!("unknown measurement arm")};
    let event=&row["native_event"];
    let frame=usize::try_from(u(&event["coordinate_frame_ordinal"])?)?;
    let authority=frames.get(frame).context("event coordinate frame outside verified chronology")?;
    ensure!(event["schema_kind"]=="prism.native_event_mechanistic_observation.v1"
        && u(&event["completed_step"])?==authority.step
        && u(&event["native_record_lineage"]["step_idx"])?==authority.step
        && event["resolved_physical_phase"]==authority.phase
        && event["physical_time_ps"].as_f64()==Some(authority.arm_times[arm_index])
        && event["measurement_window_id"]==authority.window_id
        && row["paired_measurement_frame"]["artifact"]=="paired_measurement_frames.jsonl.zst"
        && u(&row["paired_measurement_frame"]["line_ordinal"])?==frame as u64,
        "native event frame/phase/time or paired measurement reference differs");
    Ok(frame)
}

/// V2 associates multiple existing paired readouts with explicit native chemical
/// lanes. Treatment/control remain measurement arms inside each pinned artifact.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MultiSourceManifest {
    schema_kind: String,
    existing_bundle: MeasurementSourcePin,
    #[serde(deserialize_with="deserialize_lane_readouts")]
    readout_receipts_by_lane: BTreeMap<crate::chemical_state_fiber::ControlLane, Vec<MeasurementSourcePin>>,
}

fn deserialize_lane_readouts<'de,D>(deserializer:D)
    -> std::result::Result<BTreeMap<crate::chemical_state_fiber::ControlLane,Vec<MeasurementSourcePin>>,D::Error>
where D:serde::Deserializer<'de> {
    struct Lanes;
    impl<'de> serde::de::Visitor<'de> for Lanes {
        type Value=BTreeMap<crate::chemical_state_fiber::ControlLane,Vec<MeasurementSourcePin>>;
        fn expecting(&self,f:&mut std::fmt::Formatter)->std::fmt::Result {f.write_str("unique typed native ControlLane keys")}
        fn visit_map<A:serde::de::MapAccess<'de>>(self,mut input:A)->std::result::Result<Self::Value,A::Error> {
            let mut result=BTreeMap::new();
            while let Some((lane,readouts))=input.next_entry()? {
                if result.insert(lane,readouts).is_some() {
                    return Err(serde::de::Error::custom("duplicate ControlLane manifest key"));
                }
            }
            Ok(result)
        }
    }
    deserializer.deserialize_map(Lanes)
}

struct VerifiedLaneReadout {
    lane: crate::chemical_state_fiber::ControlLane,
    receipt_pin: MeasurementSourcePin,
    receipt: Value,
    artifact_pin: MeasurementSourcePin,
    artifact: MechanisticMeasurementArtifact,
    artifact_digest: String,
    source: Value,
    spatial_metric: crate::coordinate_cavity::PhysicalCavitySpatialMetricAuthority,
    frames: Vec<VerifiedReadoutFrame>,
    products: BTreeMap<String, MeasurementSourcePin>,
}

fn admit_lane_atom_count(
    counts: &crate::pathb_chemical_state::LaneTopologyAtomCounts,
    lane: crate::chemical_state_fiber::ControlLane,
    topology: &str,
    atoms: usize,
) -> Result<()> {
    let expected = counts.get(&lane).and_then(|v| v.get(topology))
        .context("declared readout lane/topology absent from native atom-count authority")?;
    ensure!(atoms > 0 && *expected == u64::try_from(atoms)?,
        "readout atom_count differs from native lane/topology authority");
    Ok(())
}

fn insert_exact_evidence(evidence: &mut BTreeMap<String,String>, name: &str, digest: &str) -> Result<()> {
    require_sha256(digest, "attached native readout evidence")?;
    ensure!(evidence.get(name).is_none_or(|existing| existing == digest),
        "existing fiber evidence name has different identity: {name}");
    evidence.insert(name.into(),digest.into());
    Ok(())
}

fn integrate_multi_source_physical_readout(
    manifest_pin: MeasurementSourcePin, m: MultiSourceManifest, out: &Path,
) -> Result<Value> {
    ensure!(m.schema_kind == "prism.native_fiber_readout_manifest.v2", "multi-source manifest schema");
    ensure!(!out.exists(), "refusing to replace multi-source fiber output");
    let bundle: NativePathBFiberBundle = read_pin(&m.existing_bundle)?;
    bundle.verify()?;
    let counts = &bundle.pathb.receipt.observed_atom_counts_by_lane_topology;
    ensure!(!counts.is_empty(),
        "multi-source physical attachment requires native observed_atom_counts_by_lane_topology");
    let mut by_occurrence = BTreeMap::new();
    for (i,fiber) in bundle.fibers.iter().enumerate() {
        ensure!(counts.get(&fiber.body.control_lane)
            .is_some_and(|v|v.contains_key(&fiber.body.ghost.topology_sha256)),
            "native fiber lane/topology absent from atom-count authority");
        ensure!(by_occurrence.insert(fiber_occurrence_key(&fiber.body.ghost),i).is_none(),
            "duplicate native fiber occurrence across physical lanes");
    }
    let mut inputs = vec![manifest_pin.clone(),m.existing_bundle.clone()];
    let mut sources = Vec::new();
    let mut receipt_identities = BTreeSet::new();
    let mut artifact_identities = BTreeSet::new();
    let mut supplied_lane_topologies = BTreeSet::new();
    for (lane,receipts) in &m.readout_receipts_by_lane {
        ensure!(counts.contains_key(lane), "declared readout lane is absent from native bundle authority");
        for receipt_pin in receipts {
            ensure!(receipt_identities.insert(receipt_pin.sha256.clone()),
                "duplicate paired readout receipt assignment");
            let receipt: Value = read_pin(receipt_pin)?;
            ensure!(receipt["schema_kind"] == "prism.native_paired_mechanistic_readout_receipt.v1"
                && receipt["status"] == "IMPLEMENTATION_VERIFIED", "completed native readout required");
            let artifact_pin: MeasurementSourcePin = serde_json::from_value(receipt["source_artifact"].clone())?;
            artifact_pin.verify()?;
            let artifact = MechanisticMeasurementArtifact::load_verified(&artifact_pin.path)?;
            ensure!(receipt["source_manifest_sha256"] == artifact.manifest_sha256
                && u(&receipt["frames"])? == artifact.frame_steps.len() as u64,
                "readout/artifact source or chronology mismatch");
            let artifact_digest = canonical_sha256(&artifact)?;
            ensure!(artifact_identities.insert(artifact_digest.clone()),
                "duplicate paired artifact assigned through multiple receipts or chemical lanes");
            admit_lane_atom_count(counts,*lane,&artifact.topology_sha256,artifact.atom_count)?;
            supplied_lane_topologies.insert((*lane,artifact.topology_sha256.clone()));
            let source_pin = artifact.source_pins.iter().find(|p| p.sha256 == artifact.manifest_sha256)
                .context("source measurement manifest")?;
            let source: Value = serde_json::from_reader(File::open(&source_pin.path)?)?;
            let topology = load(&source["mechanistic_sources"]["topology"])?;
            let typed_topology:crate::input::PrismPrepTopology=serde_json::from_value(topology)?;
            let spatial_metric=crate::coordinate_cavity::physical_spatial_metric_authority(&typed_topology)?;
            spatial_metric.verify()?;
            ensure!(matches!(spatial_metric.mode,crate::coordinate_cavity::PhysicalCavitySpatialMetricMode::PeriodicTriclinicMinimumImage),
                "chemical fiber attachment requires actual periodic spatial authority; nonperiodic evidence is not reclassified");
            let products: BTreeMap<String,MeasurementSourcePin> = serde_json::from_value(receipt["outputs"].clone())?;
            for p in products.values() {p.verify()?;}
            for name in ["paired_measurement_frames.jsonl.zst","paired_receptor_membership.jsonl.zst",
                "treatment_native_event_readout.jsonl.zst","control_native_event_readout.jsonl.zst"] {
                ensure!(products.contains_key(name), "readout product missing: {name}");
            }
            let frames=read_verified_frame_authorities(&artifact,&products)?;
            inputs.extend([receipt_pin.clone(),artifact_pin.clone()]);
            inputs.extend(artifact.source_pins.iter().cloned());
            inputs.extend(artifact.evaluated_artifacts.values().cloned());
            inputs.extend(products.values().cloned());
            sources.push(VerifiedLaneReadout{lane:*lane,receipt_pin:receipt_pin.clone(),receipt,
                artifact_pin,artifact,artifact_digest,source,spatial_metric,frames,products});
        }
    }
    // One complete conserved bundle. No subset is constructed or qualified.
    let mut attached = bundle.clone();
    if let Some(parent)=out.parent().filter(|p|!p.as_os_str().is_empty()) {fs::create_dir_all(parent)?;}
    fs::create_dir(out).context("multi-source output already exists or cannot be created exclusively")?;
    let mut associations = zstd::stream::write::Encoder::new(BufWriter::new(File::create(
        out.join("fiber_native_readout_links.jsonl.zst"))?),1)?;
    let mut seen = BTreeSet::new();
    let mut readout_occurrences = BTreeSet::new();
    let mut source_reports = Vec::new();
    for source in &sources {
        let mut arm_reports = BTreeMap::new();
        for arm in ["treatment","control"] {
            let phase = load(&source.source["mechanistic_sources"]["arms"][arm]["ghost_phase_binding"])?;
            let protocol_name = text(&phase["protocol_name"])?;
            let product = &source.products[&format!("{arm}_native_event_readout.jsonl.zst")];
            let rows = BufReader::new(zstd::stream::read::Decoder::new(File::open(&product.path)?)?).lines();
            let mut count=0u64;let mut matched=0u64;
            for (ordinal,raw) in rows.enumerate() {
                let row: Value = serde_json::from_str(&raw?)?;count+=1;
                ensure!(row["schema_kind"] == "prism.native_event_readout.v1", "native event readout schema");
                let event=&row["native_event"];let lineage=&event["native_record_lineage"];
                ensure!(event["arm"] == arm && lineage["run_id"] == source.artifact.arm_run_ids[arm]
                    && lineage["topology_sha256"] == source.artifact.topology_sha256
                    && u(&lineage["stream_id"])? == u64::from(source.artifact.stream_id),
                    "readout source arm/run/topology/stream differs from verified artifact");
                let frame=verify_event_frame_authority(&row,arm,&source.frames)?;
                let key=occurrence_key(lineage)?;
                ensure!(readout_occurrences.insert(key.clone()),
                    "duplicate/ambiguous exact native event occurrence across declared readouts");
                let Some(&index)=by_occurrence.get(&key) else {continue;};
                let fiber=&bundle.fibers[index];
                ensure!(fiber.body.control_lane == source.lane,
                    "exact native event assigned to a different chemical lane");
                ensure!(seen.insert(index), "duplicate readout assignment for native fiber");
                ensure!(source.artifact.frame_steps.get(frame)==Some(&fiber.body.ghost.key.step_idx),
                    "fiber step differs from verified coordinate frame");
                ensure!(source.spatial_metric.box_vectors_angstrom==Some(fiber.body.thermodynamic_condition.periodic_box_vectors_a),
                    "physical chemical fiber box differs from recorded periodic spatial authority");
                source.artifact.validate_fiber_phase(fiber)?;
                exact_ghost(&fiber.body.ghost,lineage,protocol_name)?;
                ensure!(fiber.body.protocol_phase.phase_name == event["resolved_physical_phase"],
                    "fiber physical phase differs from exact native readout");
                let event_hash=canonical_sha256(&row)?;
                let mut body=fiber.body.clone();
                for (name,digest) in [
                    ("native_mechanistic_measurements_v1",source.artifact_digest.as_str()),
                    ("native_paired_readout_receipt_v1",source.receipt_pin.sha256.as_str()),
                    ("native_paired_readout_event_v1",event_hash.as_str()),
                ] {insert_exact_evidence(&mut body.evidence_artifact_sha256,name,digest)?;}
                let updated=ChemicalStateFiber::materialize(body)?;
                line(&mut associations,&json!({"original_fiber_sha256":fiber.fiber_sha256,
                    "fiber_sha256":updated.fiber_sha256,"ghost_lineage_sha256":updated.body.ghost.lineage_sha256()?,
                    "control_lane":source.lane,"measurement_arm":arm,
                    "source_artifact":source.artifact_pin,"readout_receipt":source.receipt_pin,
                    "verified_coordinate_frame_ordinal":frame,"verified_physical_time_ps":event["physical_time_ps"],
                    "spatial_metric_authority_sha256":source.spatial_metric.authority_sha256,
                    "readout_event_source":product,"readout_event_line_ordinal":ordinal,
                    "readout_event_canonical_sha256":event_hash,
                    "paired_measurement_frame":row["paired_measurement_frame"],
                    "receptor_face_correspondence":row["receptor_face_correspondence"],
                    "full_native_event_and_measurements":"complete immutable source row retained; all native tensor/vector/contact/KCC/ASC and receptor-face references unchanged"}))?;
                attached.fibers[index]=updated;matched+=1;
            }
            ensure!(count == u(&source.receipt["native_event_records"][arm])?,
                "native readout event population mismatch");
            arm_reports.insert(arm,json!({"records":count,"matched_fibers":matched,"unmatched_records":count-matched,"source":product}));
        }
        source_reports.push(json!({"control_lane":source.lane,"readout_receipt":source.receipt_pin,
            "source_artifact":source.artifact_pin,"topology_sha256":source.artifact.topology_sha256,
            "atom_count":source.artifact.atom_count,"arm_records":arm_reports,
            "source_spatial_metric_authority":source.spatial_metric}));
    }
    associations.finish()?.flush()?;
    let mut unmatched=BufWriter::new(File::create(out.join("unmatched_native_fibers.jsonl"))?);
    let mut unmatched_counts:BTreeMap<crate::chemical_state_fiber::ControlLane,BTreeMap<String,u64>>=BTreeMap::new();
    for (index,fiber) in bundle.fibers.iter().enumerate() {
        if seen.contains(&index) {continue;}
        let lane=fiber.body.control_lane;let topology=&fiber.body.ghost.topology_sha256;
        *unmatched_counts.entry(lane).or_default().entry(topology.clone()).or_default()+=1;
        line(&mut unmatched,&json!({"fiber_index":index,"fiber_sha256":fiber.fiber_sha256,
            "control_lane":lane,"topology_sha256":topology,"ghost_lineage_sha256":fiber.body.ghost.lineage_sha256()?,
            "ghost_occurrence_key":fiber_occurrence_key(&fiber.body.ghost),
            "status":if supplied_lane_topologies.contains(&(lane,topology.clone())) {"NO_EXACT_MATCH_IN_SUPPLIED_READOUTS"}else{"NO_READOUT_FOR_LANE_TOPOLOGY"},
            "original_fiber_preserved":true}))?;
    }
    unmatched.flush()?;
    let missing_sources:Vec<_>=counts.iter().flat_map(|(lane,topologies)|topologies.keys().map(move|topology|(*lane,topology.clone())))
        .filter(|identity|!supplied_lane_topologies.contains(identity))
        .map(|(lane,topology)|json!({"control_lane":lane,"topology_sha256":topology})).collect();
    let mut hashes=attached.fibers.iter().map(|f|f.fiber_sha256.clone()).collect::<Vec<_>>();hashes.sort();
    attached.fiber_population_sha256=canonical_sha256(&hashes)?;attached.verify()?;
    let mut pins_by_path=BTreeMap::new();
    for input in &inputs {
        if let Some(previous)=pins_by_path.insert(input.path.clone(),input.clone()) {
            ensure!(previous==*input,"conflicting active source pins");
        }
    }
    for input in pins_by_path.values() {input.verify()?;}
    write_json(&out.join("native_pathb_fiber_bundle.json"),&attached)?;
    let complete=seen.len()==bundle.fibers.len() && missing_sources.is_empty();
    let receipt=json!({"schema_kind":"prism.native_fiber_readout_integration_receipt.v2",
        "status":if complete {"IMPLEMENTATION_VERIFIED"}else{"IMPLEMENTATION_PARTIAL"},
        "manifest":manifest_pin,"input_bundle":m.existing_bundle,
        "input_native_fiber_population_sha256":bundle.fiber_population_sha256,
        "fiber_bundle":pin(&out.join("native_pathb_fiber_bundle.json"))?,
        "exact_native_readout_links":pin(&out.join("fiber_native_readout_links.jsonl.zst"))?,
        "unmatched_native_fibers":pin(&out.join("unmatched_native_fibers.jsonl"))?,
        "fiber_count":attached.fibers.len(),"matched_fibers":seen.len(),"unmatched_fibers":bundle.fibers.len()-seen.len(),
        "unmatched_fibers_by_lane_topology":unmatched_counts,"missing_lane_topology_readouts":missing_sources,
        "observed_atom_counts_by_lane_topology":counts,"readout_sources":source_reports,
        "unmatched_readout_events":"complete original pinned products retained; no source rows deleted",
        "control_lanes_changed":false,"new_physical_conditions_created":false,"MD_executed":false,
        "original_unmatched_fibers_preserved":true,"complete_physical_fiber_coverage":complete,
        "source_inputs":pins_by_path.values().collect::<Vec<_>>()});
    write_json(&out.join("EXECUTION_RECEIPT.json"),&receipt)?;Ok(receipt)
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn multi_source_lane_topology_admission_preserves_distinct_atom_populations() -> Result<()> {
        use crate::chemical_state_fiber::ControlLane::*;
        // Contract values only; these are not emitted molecular observations.
        let a="a".repeat(64);let b="b".repeat(64);
        let counts=crate::pathb_chemical_state::collect_lane_topology_atom_counts([
            (NoProbe,a.clone(),4865),(HypothesisFragment,b.clone(),4891),
            (OrthogonalChemicalControl,b.clone(),4891)])?;
        admit_lane_atom_count(&counts,NoProbe,&a,4865)?;
        admit_lane_atom_count(&counts,HypothesisFragment,&b,4891)?;
        ensure!(admit_lane_atom_count(&counts,HypothesisFragment,&b,4865).is_err(),"scalarized count admitted");
        ensure!(admit_lane_atom_count(&counts,NoProbe,&b,4891).is_err(),"wrong lane topology admitted");
        ensure!(admit_lane_atom_count(&counts,NoProbe,&a,0).is_err(),"zero atoms admitted");
        Ok(())
    }

    #[test]
    fn multi_source_rejects_evidence_identity_overwrite() -> Result<()> {
        let first="a".repeat(64);let second="b".repeat(64);let mut evidence=BTreeMap::new();
        insert_exact_evidence(&mut evidence,"native_mechanistic_measurements_v1",&first)?;
        insert_exact_evidence(&mut evidence,"native_mechanistic_measurements_v1",&first)?;
        let original=evidence.clone();
        ensure!(insert_exact_evidence(&mut evidence,"native_mechanistic_measurements_v1",&second).is_err(),"old evidence overwritten");
        ensure!(evidence==original,"rejected replacement mutated evidence");
        Ok(())
    }

    #[test]
    fn multi_source_manifest_rejects_duplicate_or_measurement_arm_lane_keys() -> Result<()> {
        let prefix=r#"{"schema_kind":"prism.native_fiber_readout_manifest.v2","existing_bundle":{"path":"contract-only","sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},"readout_receipts_by_lane": "#;
        let duplicate=format!("{prefix}{{\"no_probe\":[],\"no_probe\":[]}}}}");
        ensure!(serde_json::from_str::<MultiSourceManifest>(&duplicate).is_err(),"duplicate lane silently overwritten");
        let arm=format!("{prefix}{{\"treatment\":[]}}}}");
        ensure!(serde_json::from_str::<MultiSourceManifest>(&arm).is_err(),"measurement arm reclassified as chemical lane");
        let valid=format!("{prefix}{{\"no_probe\":[],\"hypothesis_fragment\":[],\"orthogonal_chemical_control\":[]}}}}");
        let parsed:MultiSourceManifest=serde_json::from_str(&valid)?;
        ensure!(parsed.readout_receipts_by_lane.len()==3,"typed physical lane population");
        Ok(())
    }

    #[test]
    #[ignore = "requires real PRISM_MULTI_SOURCE_FIBER_MANIFEST and fresh PRISM_MULTI_SOURCE_FIBER_OUTPUT; no synthetic scientific inputs"]
    fn actual_multi_source_physical_fiber_readout_preserves_complete_bundle() -> Result<()> {
        let path=PathBuf::from(std::env::var("PRISM_MULTI_SOURCE_FIBER_MANIFEST")?);
        let output=PathBuf::from(std::env::var("PRISM_MULTI_SOURCE_FIBER_OUTPUT")?);
        let m:MultiSourceManifest=serde_json::from_reader(File::open(&path)?)?;
        let original:NativePathBFiberBundle=read_pin(&m.existing_bundle)?;
        let result=integrate_native_fiber_readout(&path,&output)?;
        let attached:NativePathBFiberBundle=read_pin(&serde_json::from_value(result["fiber_bundle"].clone())?)?;
        attached.verify()?;
        ensure!(attached.pathb==original.pathb && attached.fibers.len()==original.fibers.len(),"native conservation changed");
        for (before,after) in original.fibers.iter().zip(&attached.fibers) {
            let mut restored=after.body.clone();restored.evidence_artifact_sha256=before.body.evidence_artifact_sha256.clone();
            ensure!(ChemicalStateFiber::materialize(restored)?==*before,"fiber measurements changed beyond evidence attachment");
        }
        let complete=result["complete_physical_fiber_coverage"]==true;
        ensure!(result["status"]==if complete {"IMPLEMENTATION_VERIFIED"}else{"IMPLEMENTATION_PARTIAL"},"coverage status differs");
        ensure!(u(&result["matched_fibers"])?+u(&result["unmatched_fibers"])?==original.fibers.len() as u64,"attachment population lost");
        Ok(())
    }

    #[test]
    fn multi_source_event_binds_exact_independent_time_and_frame_reference() -> Result<()> {
        // Consumer contract fixture only: no MD or scientific output is generated.
        let frames=vec![VerifiedReadoutFrame{step:10,phase:"cold_hold".into(),window_id:json!("w0"),
            arm_times:[0.02,0.018],vector_row_sha256:"a".repeat(64)}];
        let row=json!({"native_event":{"schema_kind":"prism.native_event_mechanistic_observation.v1",
            "coordinate_frame_ordinal":0,"completed_step":10,"native_record_lineage":{"step_idx":10},
            "resolved_physical_phase":"cold_hold","physical_time_ps":0.02,"measurement_window_id":"w0"},
            "paired_measurement_frame":{"artifact":"paired_measurement_frames.jsonl.zst","line_ordinal":0}});
        ensure!(verify_event_frame_authority(&row,"treatment",&frames)?==0,"exact treatment frame");
        ensure!(verify_event_frame_authority(&row,"control",&frames).is_err(),"one arm time substituted for another");
        for (field,bad) in [("completed_step",json!(11)),("coordinate_frame_ordinal",json!(1)),
            ("resolved_physical_phase",json!("warm_hold")),("physical_time_ps",json!(0.021)),("measurement_window_id",json!("w1"))] {
            let mut wrong=row.clone();wrong["native_event"][field]=bad;
            ensure!(verify_event_frame_authority(&wrong,"treatment",&frames).is_err(),"wrong {field} admitted");
        }
        let mut wrong=row.clone();wrong["paired_measurement_frame"]["line_ordinal"]=json!(1);
        ensure!(verify_event_frame_authority(&wrong,"treatment",&frames).is_err(),"wrong paired frame admitted");
        Ok(())
    }

    #[test]
    #[ignore = "requires PRISM_FIBER_READOUT_MANIFEST and PRISM_FIBER_READOUT_TEST_OUTPUT referencing real recorded pair"]
    fn actual_recorded_pair_preserves_missing_physical_contracts() -> Result<()> {
        let path = PathBuf::from(std::env::var("PRISM_FIBER_READOUT_MANIFEST")?);
        let out = PathBuf::from(std::env::var("PRISM_FIBER_READOUT_TEST_OUTPUT")?);
        let result = integrate_native_fiber_readout(&path, &out)?;
        ensure!(
            result["status"] == "IMPLEMENTATION_BLOCKED" && result["fibers_created"] == 0,
            "nonperiodic recorded pair fabricated physical fibers"
        );
        ensure!(
            result["recorded_periodic_cell"]["periodic"] == false,
            "fixture is not actual nonperiodic pair"
        );
        ensure!(
            !out.join("native_pathb_fiber_bundle.json").exists(),
            "inapplicable pair emitted physical bundle"
        );
        Ok(())
    }
}
