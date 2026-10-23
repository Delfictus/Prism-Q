//! Full-population paired readout of completed native measurement products.
//! No MD, geometry reconstruction, salience admission, causal classification,
//! or cross-arm event identity inference occurs in this consumer.
use super::*;

pub(crate) struct RowCursor {
    pub(crate) source: MeasurementSourcePin,
    lines: std::io::Lines<Box<dyn BufRead>>,
    pub(crate) next: Option<(u64, Value)>,
    pub(crate) count: u64,
}
impl RowCursor {
    pub(crate) fn open(source: MeasurementSourcePin) -> Result<Self> {
        let reader: Box<dyn BufRead> =
            if source.path.extension().and_then(|x| x.to_str()) == Some("zst") {
                Box::new(BufReader::new(zstd::stream::read::Decoder::new(
                    File::open(&source.path)?,
                )?))
            } else {
                Box::new(BufReader::new(File::open(&source.path)?))
            };
        let mut s = Self {
            source,
            lines: reader.lines(),
            next: None,
            count: 0,
        };
        s.advance()?;
        Ok(s)
    }
    fn advance(&mut self) -> Result<()> {
        self.next = match self.lines.next() {
            None => None,
            Some(line) => {
                let row = serde_json::from_str(&line?)?;
                let index = self.count;
                self.count += 1;
                Some((index, row))
            }
        };
        Ok(())
    }
    pub(crate) fn take(&mut self) -> Result<(u64, Value)> {
        let row = self
            .next
            .take()
            .context("native readout source ended early")?;
        self.advance()?;
        Ok(row)
    }
}
pub(crate) fn evaluated(a: &MechanisticMeasurementArtifact, name: &str) -> Result<MeasurementSourcePin> {
    a.evaluated_artifacts
        .get(name)
        .cloned()
        .with_context(|| format!("completed native product missing: {name}"))
}
fn checked_digest(row: &Value) -> Result<()> {
    let digest = text(&row["record_sha256"])?;
    let mut body = row.clone();
    body.as_object_mut()
        .context("record object")?
        .remove("record_sha256");
    ensure!(
        canonical_sha256(&body)? == digest,
        "native readout record digest mismatch"
    );
    Ok(())
}
fn pinned(a: &MechanisticMeasurementArtifact, value: &Value) -> Result<MeasurementSourcePin> {
    let p: MeasurementSourcePin = serde_json::from_value(value.clone())?;
    ensure!(
        a.source_pins.contains(&p) || a.evaluated_artifacts.values().any(|q| q == &p),
        "native readout reference outside verified source population"
    );
    Ok(p)
}
fn endpoint(
    a: &MechanisticMeasurementArtifact,
    row: &Value,
    arm: &str,
    frame: usize,
) -> Result<()> {
    ensure!(
        row["arm"] == arm
            && row["run_id"] == a.arm_run_ids[arm]
            && u(&row["frame_ordinal"])? as usize == frame
            && u(&row["step"])? == a.frame_steps[frame]
            && row["phase"] == a.frame_phases[frame],
        "native readout arm/run/frame/step/phase mismatch"
    );
    Ok(())
}
fn ref_row(source: &MeasurementSourcePin, ordinal: u64, row: &Value) -> Result<Value> {
    Ok(
        json!({"source":source,"line_ordinal":ordinal,"canonical_row_sha256":canonical_sha256(row)?}),
    )
}
// Each recipe is joined through the pinned supplement's partition object:
// geometry source and execution receipt are never matched by filename alone.
pub(crate) fn geometry_configs(a: &MechanisticMeasurementArtifact) -> Result<BTreeMap<String, Value>> {
    let manifest_pin = pinned(a, &a.scope["geometry_grids"]["supplement_manifest"])?;
    let manifest: Value = serde_json::from_reader(File::open(&manifest_pin.path)?)?;
    ensure!(
        manifest["schema_kind"] == "prism.native_mechanistic_geometry_manifest.v1",
        "geometry supplement manifest schema"
    );
    let mut configs = BTreeMap::new();
    for arm in ["treatment", "control"] {
        for part in manifest["arms"][arm]["partitions"]
            .as_array()
            .context("geometry partitions")?
        {
            let geometry = pinned(a, &part["geometry"])?;
            let receipt_pin = pinned(a, &part["execution_receipt"])?;
            let receipt: Value = serde_json::from_reader(File::open(&receipt_pin.path)?)?;
            let authority = &receipt["estimator_config_authority"];
            let config = &authority["effective_config"];
            let native_config: crate::coordinate_cavity::CoordinateCavityConfig =
                serde_json::from_value(config.clone())?;
            native_config.validate()?;
            let digest = canonical_sha256(&native_config)?;
            ensure!(
                config.is_object()
                    && authority["effective_config_sha256"] == digest
                    && receipt["native_estimator_sha256"] == digest
                    && receipt["topology_sha256"] == a.topology_sha256
                    && receipt["sites_sha256"] == manifest["materialized_sites"]["sha256"],
                "geometry recipe authority/identity mismatch"
            );
            ensure!(
                !config["grid_spacings_a"]
                    .as_array()
                    .context("grid spacing axis")?
                    .is_empty()
                    && !config["grid_phase_offsets"]
                        .as_array()
                        .context("grid phase axis")?
                        .is_empty(),
                "empty grid recipe"
            );
            ensure!(configs.insert(geometry.sha256.clone(), json!({"geometry_source":geometry,
                "execution_receipt":receipt_pin,"arm":arm,"frame_start":receipt["frame_start"],"frame_end":receipt["frame_end"],
                "trajectory_sha256":receipt["trajectory_sha256"],"audit_sha256":receipt["audit_sha256"],
                "effective_config":config,"effective_config_sha256":digest})).is_none(), "duplicate geometry recipe source");
        }
    }
    Ok(configs)
}
#[derive(Clone)]
struct GridMembership {
    residues: BTreeSet<u32>,
    atoms: BTreeSet<usize>,
}
pub(crate) struct GeometryRow {
    pub(crate) ordinal: u64,
    pub(crate) row: Value,
    grids: Vec<GridMembership>,
    residues: BTreeSet<u32>,
    estimator: Value,
}
fn geometry_row(
    a: &MechanisticMeasurementArtifact,
    axis: &[i32],
    arm: &str,
    frame: usize,
    ordinal: u64,
    row: Value,
) -> Result<GeometryRow> {
    checked_digest(&row)?;
    endpoint(a, &row, arm, frame)?;
    ensure!(
        row["schema_kind"] == "prism.native_region_frame_geometry_index.v1"
            && u(&row["stream_id"])? == a.stream_id as u64,
        "native geometry schema/stream"
    );
    pinned(a, &row["geometry_source"])?;
    pinned(a, &row["atom_identity_source"])?;
    let b = &row["observation_binding"];
    ensure!(
        u(&b["frame_ordinal"])? as usize == frame
            && u(&b["step"])? == a.frame_steps[frame]
            && u(&b["stream_id"])? == a.stream_id as u64,
        "native geometry nested chronology"
    );
    pinned(a, &b["body_fixed_tracking"]["declaration"])?;
    let mut grids = Vec::new();
    let mut all = BTreeSet::new();
    for (i, g) in row["grids"]
        .as_array()
        .context("geometry grid population")?
        .iter()
        .enumerate()
    {
        ensure!(
            u(&g["grid_ordinal"])? as usize == i,
            "geometry grid ordinal"
        );
        let mut residues = BTreeSet::new();
        let mut atoms = BTreeSet::new();
        let mut faces = BTreeSet::new();
        for m in g["receptor_residue_faces"]
            .as_array()
            .context("receptor face memberships")?
        {
            let rid = u32::try_from(u(&m["analysis_residue_id"])?)?;
            ensure!(
                a.analysis_residue_ids.binary_search(&rid).is_ok() && residues.insert(rid),
                "invalid duplicate receptor residue"
            );
            let ff = m["face_indices"]
                .as_array()
                .context("receptor face indices")?;
            let aa = m["atom_indices"]
                .as_array()
                .context("receptor atom indices")?;
            ensure!(
                !ff.is_empty() && !aa.is_empty(),
                "empty receptor membership"
            );
            for x in ff {
                let id = u(x)?;
                ensure!(
                    id < u(&g["boundary_face_count"])? && faces.insert(id),
                    "duplicate/outside receptor face"
                );
            }
            for x in aa {
                let id = usize::try_from(u(x)?)?;
                ensure!(
                    axis.get(id).copied().and_then(|id|u32::try_from(id).ok()) == Some(rid) && atoms.insert(id),
                    "receptor atom/residue topology mismatch"
                );
            }
        }
        ensure!(
            faces.len() as u64 == u(&g["receptor_excluded_volume_face_count"])?,
            "receptor face conservation"
        );
        all.extend(residues.iter().copied());
        grids.push(GridMembership { residues, atoms });
    }
    ensure!(!grids.is_empty(), "empty native grid population");
    Ok(GeometryRow {
        ordinal,
        row,
        grids,
        residues: all,
        estimator: Value::Null,
    })
}
pub(crate) fn read_geometry_frame(
    cursor: &mut RowCursor,
    a: &MechanisticMeasurementArtifact,
    axis: &[i32],
    arm: &str,
    frame: usize,
    expected: usize,
    configs: &BTreeMap<String, Value>,
) -> Result<BTreeMap<u64, GeometryRow>> {
    let mut regions = BTreeMap::new();
    for _ in 0..expected {
        let (ordinal, row) = cursor.take()?;
        let id = u(&row["observation_binding"]["region_id"])?;
        let source = source(&row["geometry_source"])?;
        let cfg = configs
            .get(&source.sha256)
            .context("geometry source absent from recipe partition")?;
        ensure!(
            serde_json::from_value::<MeasurementSourcePin>(cfg["geometry_source"].clone())?
                == source
                && cfg["arm"] == arm
                && u(&cfg["frame_start"])? <= frame as u64
                && (frame as u64) < u(&cfg["frame_end"])?
                && cfg["trajectory_sha256"] == row["observation_binding"]["trajectory_sha256"]
                && cfg["audit_sha256"] == row["observation_binding"]["audit_sha256"],
            "geometry recipe partition/endpoint binding differs"
        );
        let mut checked = geometry_row(a, axis, arm, frame, ordinal, row)?;
        ensure!(
            checked.grids.len()
                == cfg["effective_config"]["grid_spacings_a"]
                    .as_array()
                    .unwrap()
                    .len()
                    * cfg["effective_config"]["grid_phase_offsets"]
                        .as_array()
                        .unwrap()
                        .len(),
            "geometry grid count differs from recipe axes"
        );
        checked.estimator = cfg.clone();
        ensure!(
            regions.insert(id, checked).is_none(),
            "duplicate region at native frame"
        );
    }
    Ok(regions)
}
fn set_difference<T: Ord + Clone + Serialize>(a: &BTreeSet<T>, b: &BTreeSet<T>) -> Value {
    json!({"shared":a.intersection(b).cloned().collect::<Vec<_>>(),
        "treatment_only":a.difference(b).cloned().collect::<Vec<_>>(),
        "control_only":b.difference(a).cloned().collect::<Vec<_>>()})
}
fn paired_geometry(
    t: &GeometryRow,
    c: &GeometryRow,
    t_source: &MeasurementSourcePin,
    c_source: &MeasurementSourcePin,
) -> Result<Value> {
    ensure!(
        !t.estimator.is_null()
            && !c.estimator.is_null()
            && t.estimator["effective_config_sha256"] == c.estimator["effective_config_sha256"]
            && t.estimator["effective_config"] == c.estimator["effective_config"],
        "paired geometry estimator recipes differ"
    );
    let tb = &t.row["observation_binding"];
    let cb = &c.row["observation_binding"];
    for field in [
        "region_id",
        "materialized_sites_sha256",
        "source_region_record_sha256",
        "aligned_grid_center_angstrom",
    ] {
        ensure!(
            tb[field] == cb[field] && !tb[field].is_null(),
            "paired region definitions differ: {field}"
        );
    }
    ensure!(
        tb["body_fixed_tracking"]["declaration"] == cb["body_fixed_tracking"]["declaration"]
            && t.grids.len() == c.grids.len(),
        "paired native reference/grid populations differ"
    );
    let grids=t.grids.iter().zip(&c.grids).enumerate().map(|(i,(t,c))|json!({
        "grid_ordinal":i,"analysis_residue_membership":set_difference(&t.residues,&c.residues),
        "atom_membership":set_difference(&t.atoms,&c.atoms),
        "identity_semantics":"same declared native grid ordinal; exact receptor-boundary residue/atom presence; face ordinals are arm-local, not paired faces"})).collect::<Vec<_>>();
    Ok(
        json!({"region_id":tb["region_id"],"treatment_geometry":ref_row(t_source,t.ordinal,&t.row)?,
        "control_geometry":ref_row(c_source,c.ordinal,&c.row)?,
        "source_region_record_sha256":tb["source_region_record_sha256"],
        "materialized_sites_sha256":tb["materialized_sites_sha256"],
        "body_fixed_declaration":tb["body_fixed_tracking"]["declaration"],
        "estimator_recipe_bindings":{"treatment":t.estimator,"control":c.estimator},
        "body_fixed_transforms":{"treatment":tb["body_fixed_tracking"]["transform"],"control":cb["body_fixed_tracking"]["transform"]},
        "local_window_clipping_observed":{"treatment":t.row["local_window_clipping_observed"],"control":c.row["local_window_clipping_observed"]},
        "native_grids":{"treatment":t.row["grids"],"control":c.row["grids"]},
        "paired_grid_membership":grids,"admission_used":false,
        "scope":"descriptive endpoint membership differences; no direct voxel/face subtraction, cavity opening, causal influence or statistical significance inferred"}),
    )
}
fn check_kcc(a: &MechanisticMeasurementArtifact, k: &Value, arm: &str, frame: usize) -> Result<()> {
    endpoint(a, k, arm, frame)?;
    for values in k["native_kcc_per_residue"]
        .as_object()
        .context("native KCC descriptors")?
        .values()
    {
        if let Some(values) = values.as_array() {
            ensure!(
                values.len() == a.analysis_residue_ids.len(),
                "KCC residue axis"
            );
        }
    }
    pinned(a, &k["source_chronology"])?;
    Ok(())
}
/// KCC begins only where the producer recorded it. Row ordinal is not the
/// physical frame ordinal; absence never borrows a later or earlier endpoint.
fn take_kcc_frame(
    cursor: &mut RowCursor,
    a: &MechanisticMeasurementArtifact,
    arm: &str,
    frame: usize,
    timeline: &Value,
) -> Result<Option<(u64, Value)>> {
    let Some((_, next)) = &cursor.next else {
        return Ok(None);
    };
    let next_frame = usize::try_from(u(&next["frame_ordinal"])?)?;
    ensure!(
        next_frame >= frame && a.frame_steps.get(next_frame).copied() == Some(u(&next["step"])?),
        "stale or inconsistent sparse KCC endpoint"
    );
    if next_frame > frame {
        return Ok(None);
    }
    let (ordinal, row) = cursor.take()?;
    check_kcc(a, &row, arm, frame)?;
    ensure!(
        row["physical_time_ps"].as_f64() == Some(super::physical_time::timeline_arm_time(timeline, arm)?)
            && row["window_id"] == timeline["window_id"],
        "KCC physical time/window differs from vector frame"
    );
    if let Some((_, following)) = &cursor.next {
        ensure!(
            u(&following["frame_ordinal"])? > frame as u64
                && u(&following["step"])? > a.frame_steps[frame],
            "duplicate or out-of-order sparse KCC endpoint"
        );
    }
    Ok(Some((ordinal, row)))
}
pub(crate) fn check_event(
    a: &MechanisticMeasurementArtifact,
    e: &Value,
    l: &Value,
    arm: &str,
    frame: usize,
    event_ordinal: u64,
    event_source: &MeasurementSourcePin,
    index_source: &MeasurementSourcePin,
    regions: &BTreeMap<u64, GeometryRow>,
    timeline: &Value,
    kcc: &Value,
) -> Result<()> {
    let lineage = &e["native_record_lineage"];
    ensure!(
        e["schema_kind"] == "prism.native_event_mechanistic_observation.v1"
            && e["arm"] == arm
            && lineage["run_id"] == a.arm_run_ids[arm]
            && lineage["topology_sha256"] == a.topology_sha256
            && u(&lineage["stream_id"])? == a.stream_id as u64
            && u(&lineage["step_idx"])? == a.frame_steps[frame]
            && u(&e["completed_step"])? == a.frame_steps[frame]
            && u(&e["coordinate_frame_ordinal"])? as usize == frame
            && e["resolved_physical_phase"] == a.frame_phases[frame]
            && e["physical_time_ps"].as_f64() == Some(super::physical_time::timeline_arm_time(timeline, arm)?)
            && e["measurement_window_id"] == timeline["window_id"],
        "native event source endpoint mismatch"
    );
    checked_digest(l)?;
    endpoint(a, l, arm, frame)?;
    ensure!(
        l["schema_kind"] == "prism.native_event_receptor_face_links.v1"
            && u(&l["stream_id"])? == a.stream_id as u64
            && u(&l["source_event_line_ordinal"])? == event_ordinal
            && serde_json::from_value::<MeasurementSourcePin>(l["source_event"].clone())?
                == *event_source
            && l["native_record_lineage"] == *lineage
            && l["native_lattice_component"] == e["native_lattice_component"],
        "geometry/event correspondence mismatch"
    );
    let mut ids = BTreeSet::new();
    for r in e["residue_measurements"]
        .as_array()
        .context("event measured residue population")?
    {
        let id = u32::try_from(u(&r["analysis_residue_id"])?)?;
        let i = a
            .analysis_residue_ids
            .binary_search(&id)
            .map_err(|_| anyhow::anyhow!("unknown event residue"))?;
        ensure!(ids.insert(id), "duplicate event residue");
        let channels = timeline["channels"]
            .as_object()
            .context("vector channels")?;
        ensure!(
            r["vectors"].as_object().context("event vectors")?.len() == channels.len(),
            "event lost vector channels"
        );
        for (name, v) in channels {
            ensure!(
                r["vectors"][name]["value"] == v["analysis_residue_vectors"][i]
                    && r["vectors"][name]["unit"] == v["unit"],
                "event residue vector differs from exact frame"
            );
        }
        let descriptors = kcc["native_kcc_per_residue"]
            .as_object()
            .context("KCC descriptor fields")?;
        ensure!(
            r["native_kcc"].as_object().context("event KCC")?.len()
                == descriptors.values().filter(|v| v.is_array()).count(),
            "event lost KCC fields"
        );
        for (name, v) in descriptors {
            if let Some(values) = v.as_array() {
                ensure!(
                    r["native_kcc"][name] == values[i],
                    "event KCC differs from exact endpoint"
                );
            }
        }
    }
    ensure!(
        serde_json::from_value::<BTreeSet<u32>>(l["native_event_residue_ids"].clone())? == ids,
        "event membership differs from geometry link"
    );
    let expected = regions
        .iter()
        .filter_map(|(id, r)| {
            let shared = ids.intersection(&r.residues).copied().collect::<Vec<_>>();
            (!shared.is_empty()).then_some((*id, shared))
        })
        .collect::<BTreeMap<_, _>>();
    let links = l["geometry_links"]
        .as_array()
        .context("native geometry links")?;
    ensure!(
        links.len() == expected.len(),
        "event region association population mismatch"
    );
    let mut seen = BTreeSet::new();
    for link in links {
        let id = u(&link["region_id"])?;
        let g = regions
            .get(&id)
            .context("event references unknown geometry region")?;
        ensure!(
            seen.insert(id)
                && link["shared_analysis_residue_ids"]
                    == json!(expected
                        .get(&id)
                        .context("event region has no actual receptor membership")?)
                && u(&link["geometry_index_line_ordinal"])? == g.ordinal
                && link["geometry_index_record_sha256"] == g.row["record_sha256"]
                && link["geometry_source_record_sha256"] == g.row["geometry_source_record_sha256"]
                && link["geometry_index_artifact"]
                    == index_source
                        .path
                        .file_name()
                        .context("geometry filename")?
                        .to_string_lossy()
                        .as_ref(),
            "event geometry index/residue correspondence differs"
        );
    }
    ensure!(
        l["event_without_receptor_face_match_preserved"] == json!(links.is_empty()),
        "event no-match state differs"
    );
    for key in ["contact_measurements", "bonded_measurements"] {
        pinned(a, &e[key]["source"])?;
        ensure!(
            u(&e[key]["frame_ordinal"])? as usize == frame,
            "event contact/bonded chronology differs"
        );
    }
    pinned(a, &e["bonded_measurements"]["term_index"])?;
    ensure!(
        serde_json::from_value::<MeasurementSourcePin>(e["kcc_endpoint_index"].clone())?
            == evaluated(a, &format!("{arm}_kcc_endpoint_index.jsonl"))?,
        "event KCC endpoint source differs"
    );
    ensure!(
        e["kcc_endpoint_support"] == kcc["lag_support_ref"],
        "event KCC support reference differs"
    );
    let channels = timeline["channels"]
        .as_object()
        .context("vector channels")?;
    ensure!(
        e["full_atom_vector_channels"]
            .as_object()
            .context("full event vector references")?
            .len()
            == channels.len(),
        "event atom channels missing"
    );
    for (name, v) in channels {
        for field in ["byte_offset", "byte_length", "unit"] {
            ensure!(
                e["full_atom_vector_channels"][name][field] == v[field],
                "event atom vector reference differs"
            );
        }
    }
    if !e["isolated_ASC_measurements"].is_null() {
        pinned(a, &e["isolated_ASC_measurements"]["source"])?;
    }
    Ok(())
}

/// Consume all completed native outputs once, preserving arm populations and
/// basis identities. Output membership differences are exact set operations.
/// Existing measurements are read and checked, never re-estimated.
pub fn evaluate_native_readout(artifact_path: &Path, out: &Path) -> Result<Value> {
    ensure!(!out.exists(), "refusing to replace native paired readout");
    let artifact_pin = pin(artifact_path)?;
    let a = MechanisticMeasurementArtifact::load_verified(artifact_path)?;
    ensure!(
        a.scope["geometry_grids"]["mode"] == "native_body_fixed_geometry_consumed"
            && a.scope["geometry_grids"]["full_native_grid_and_face_validation"] == true
            && a.scope["geometry_grids"]["all_geometry_frames_retained"] == true
            && a.scope["geometry_grids"]["all_native_events_retained"] == true,
        "completed full geometry supplement required"
    );
    let source_manifest = a
        .source_pins
        .iter()
        .find(|p| p.sha256 == a.manifest_sha256)
        .context("source manifest")?;
    let manifest: Value = serde_json::from_reader(File::open(&source_manifest.path)?)?;
    let axes_pin = evaluated(&a, "measurement_axes.json")?;
    let axes: Value = serde_json::from_reader(File::open(&axes_pin.path)?)?;
    let atom_axis: Vec<i32> = serde_json::from_value(axes["atom_analysis_residue_ids"].clone())?;
    ensure!(
        atom_axis.len() == a.atom_count
            && axes["analysis_residue_ids"] == json!(a.analysis_residue_ids),
        "native readout measurement axis"
    );
    let topology: Value = load(&manifest["mechanistic_sources"]["topology"])?;
    ensure!(
        topology["atom_analysis_residue_ids"] == axes["atom_analysis_residue_ids"],
        "readout atom axis differs from topology"
    );
    let vector_pin = evaluated(&a, "aligned_vectors.f64le")?;
    let mut binary = File::open(&vector_pin.path)?;
    let mut timeline = RowCursor::open(evaluated(&a, "vector_frames.jsonl")?)?;
    let arms = ["treatment", "control"];
    let mut indices = arms
        .map(|arm| {
            RowCursor::open(evaluated(
                &a,
                &format!("{arm}_native_geometry_index.jsonl.zst"),
            )?)
        })
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    let mut events = arms
        .map(|arm| {
            RowCursor::open(evaluated(
                &a,
                &format!("{arm}_native_event_measurements.jsonl.zst"),
            )?)
        })
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    let mut links = arms
        .map(|arm| {
            RowCursor::open(evaluated(
                &a,
                &format!("{arm}_native_event_geometry_links.jsonl.zst"),
            )?)
        })
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    let mut kcc = arms
        .map(|arm| RowCursor::open(evaluated(&a, &format!("{arm}_kcc_endpoint_index.jsonl"))?))
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    let configs = geometry_configs(&a)?;
    let counts = arms
        .map(|arm| {
            usize::try_from(u(
                &a.scope["geometry_grids"]["arms"][arm]["regions_per_frame"]
            )?)
            .map_err(anyhow::Error::from)
        })
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        counts[0] > 0 && counts[0] == counts[1],
        "paired geometry population differs"
    );
    fs::create_dir_all(out)?;
    let mut frame_out = zstd::stream::write::Encoder::new(
        BufWriter::new(File::create(
            out.join("paired_measurement_frames.jsonl.zst"),
        )?),
        1,
    )?;
    let mut region_out = zstd::stream::write::Encoder::new(
        BufWriter::new(File::create(
            out.join("paired_receptor_membership.jsonl.zst"),
        )?),
        1,
    )?;
    let mut event_out = arms
        .map(|arm| {
            Ok(zstd::stream::write::Encoder::new(
                BufWriter::new(File::create(
                    out.join(format!("{arm}_native_event_readout.jsonl.zst")),
                )?),
                1,
            )?)
        })
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    let mut vector_end = 0u64;
    let mut event_counts = [0u64; 2];
    let mut unmatched = [0u64; 2];
    let mut no_event_frames = [0u64; 2];
    let mut paired_count = 0u64;
    for frame in 0..a.frame_steps.len() {
        let (ordinal, t) = timeline.take()?;
        ensure!(
            ordinal as usize == frame
                && u(&t["frame_ordinal"])? as usize == frame
                && u(&t["step"])? == a.frame_steps[frame]
                && t["phase"] == a.frame_phases[frame],
            "vector chronology differs"
        );
        let mut ranges = Vec::new();
        for (name, c) in t["channels"]
            .as_object()
            .context("timeline channel object")?
        {
            ensure!(
                u(&c["byte_length"])? == a.atom_count as u64 * 24
                    && c["analysis_residue_vectors"]
                        .as_array()
                        .context("residue vector rows")?
                        .len()
                        == a.analysis_residue_ids.len(),
                "native vector dimensions"
            );
            for v in c["analysis_residue_vectors"].as_array().unwrap() {
                let v: [f64; 3] = serde_json::from_value(v.clone())?;
                ensure!(v.iter().all(|v| v.is_finite()), "nonfinite residue vector");
            }
            ranges.push((u(&c["byte_offset"])?, name, u(&c["byte_length"])?));
        }
        ranges.sort_by_key(|x| x.0);
        let mut digests = BTreeMap::new();
        for (offset, name, bytes) in ranges {
            ensure!(offset == vector_end, "native vector overlap/gap/order");
            let mut raw = vec![0; usize::try_from(bytes)?];
            binary.read_exact(&mut raw)?;
            for v in raw.chunks_exact(8) {
                ensure!(
                    f64::from_le_bytes(v.try_into()?).is_finite(),
                    "nonfinite full atom vector"
                );
            }
            digests.insert(name.clone(), format!("{:x}", Sha256::digest(&raw)));
            vector_end = vector_end
                .checked_add(bytes)
                .context("vector offset overflow")?;
        }
        let mut kk = Vec::new();
        let mut gg = Vec::new();
        for (i, arm) in arms.iter().enumerate() {
            let endpoint = take_kcc_frame(&mut kcc[i], &a, arm, frame, &t)?;
            kk.push(endpoint);
            gg.push(read_geometry_frame(
                &mut indices[i],
                &a,
                &atom_axis,
                arm,
                frame,
                counts[i],
                &configs,
            )?);
        }
        ensure!(
            gg[0].keys().eq(gg[1].keys()),
            "paired region identities differ"
        );
        for (id, g) in &gg[0] {
            let pair = paired_geometry(g, &gg[1][id], &indices[0].source, &indices[1].source)?;
            line(
                &mut region_out,
                &json!({"schema_kind":"prism.native_paired_receptor_membership.v1",
                "paired_frame_ordinal":frame,"step":a.frame_steps[frame],"phase":a.frame_phases[frame],
                "arm_physical_time":t["arm_physical_time"],"shared_physical_time_ps":t["physical_time_ps"],"pair_correspondence":"same recorded step; no physical-time interpolation",
                "stream_id":a.stream_id,"run_ids":a.arm_run_ids,"topology_sha256":a.topology_sha256,
                "measurement_frame_ref":{"artifact":"paired_measurement_frames.jsonl.zst","line_ordinal":frame},
                "comparison":pair}),
            )?;
            paired_count += 1;
        }
        let before = event_counts;
        for (i, arm) in arms.iter().enumerate() {
            while let Some((_, e)) = &events[i].next {
                let step = u(&e["completed_step"])?;
                ensure!(
                    step >= a.frame_steps[frame],
                    "event outside recorded chronology"
                );
                if step > a.frame_steps[frame] {
                    break;
                }
                let (n, event) = events[i].take()?;
                let (ln, link) = links[i].take()?;
                ensure!(ln == n, "event and geometry-link populations differ");
                check_event(
                    &a,
                    &event,
                    &link,
                    arm,
                    frame,
                    n,
                    &events[i].source,
                    &indices[i].source,
                    &gg[i],
                    &t,
                    &kk[i]
                        .as_ref()
                        .context("recorded native event lacks exact recorded KCC endpoint")?
                        .1,
                )?;
                unmatched[i] += u64::from(link["geometry_links"].as_array().unwrap().is_empty());
                line(
                    &mut event_out[i],
                    &json!({"schema_kind":"prism.native_event_readout.v1",
                    "source_event":ref_row(&events[i].source,n,&event)?,"native_event":event,
                    "source_geometry_links":ref_row(&links[i].source,n,&link)?,"receptor_face_correspondence":link,
                    "paired_measurement_frame":{"artifact":"paired_measurement_frames.jsonl.zst","line_ordinal":frame},
                    "paired_region_readout":"paired_receptor_membership.jsonl.zst, exact frame and native region identity",
                    "coordinate_basis_scope":"full vectors use measurement_axes.frame_basis; geometry uses each region's declared body-fixed basis; isolated ASC retains laboratory basis; no cross-basis projection performed",
                    "TSH_admission_used":false,"cross_arm_event_pairing_inferred":false}),
                )?;
                event_counts[i] += 1;
            }
            no_event_frames[i] += u64::from(event_counts[i] == before[i]);
        }
        line(
            &mut frame_out,
            &json!({"schema_kind":"prism.native_paired_measurement_readout_frame.v1",
            "source_vector_frame":ref_row(&timeline.source,ordinal,&t)?,"native_vector_frame":t,
            "topology_sha256":a.topology_sha256,"run_ids":a.arm_run_ids,"stream_id":a.stream_id,
            "full_atom_vectors":vector_pin,"full_atom_channel_bytes_sha256":digests,"measurement_axes":axes_pin,
            "native_kcc_endpoints":{"treatment":kk[0].as_ref().map(|(_,row)|row),"control":kk[1].as_ref().map(|(_,row)|row)},
            "native_kcc_source_line_ordinals":{"treatment":kk[0].as_ref().map(|(ordinal,_)|ordinal),"control":kk[1].as_ref().map(|(ordinal,_)|ordinal)},
            "native_kcc_availability":{"treatment":if kk[0].is_some(){"exact_recorded_endpoint"}else{"no_recorded_KCC"},"control":if kk[1].is_some(){"exact_recorded_endpoint"}else{"no_recorded_KCC"}},
            "native_kcc_sources":{"treatment":kcc[0].source,"control":kcc[1].source},
            "contact_measurements":{"source":manifest["mechanistic_sources"]["pair_frames"],"axis":manifest["mechanistic_sources"]["pair_index"],"frame_ordinal":frame},
            "bonded_measurements":{"source":manifest["mechanistic_sources"]["bonded_frames"],"axis":manifest["mechanistic_sources"]["bonded_index"],"frame_ordinal":frame},
            "source_tensor_artifacts":{"vector_moments":a.evaluated_artifacts.get("vector_moments.jsonl"),"contact_moments":a.evaluated_artifacts.get("contact_moments.jsonl"),"bonded_moments":a.evaluated_artifacts.get("bonded_moments.jsonl")},
            "isolated_ASC_source_artifacts":{"treatment":a.evaluated_artifacts.get("treatment_isolated_asc_measurements.jsonl.zst"),"control":a.evaluated_artifacts.get("control_isolated_asc_measurements.jsonl.zst")},
            "native_event_counts":{"treatment":event_counts[0]-before[0],"control":event_counts[1]-before[1]},
            "full_frame_population_retained":true,"TSH_admission_used":false}),
        )?;
    }
    ensure!(
        timeline.next.is_none() && vector_end == binary.metadata()?.len(),
        "unconsumed vector frames/bytes"
    );
    for i in 0..2 {
        let arm = arms[i];
        ensure!(
            indices[i].next.is_none()
                && events[i].next.is_none()
                && links[i].next.is_none()
                && kcc[i].next.is_none(),
            "unconsumed native rows"
        );
        ensure!(
            event_counts[i] == u(&a.scope["geometry_grids"]["arms"][arm]["event_records"])?
                && event_counts[i]
                    == u(&a.scope["native_event_measurements"]["arms"][arm]["native_records"])?
                && indices[i].count
                    == u(&a.scope["geometry_grids"]["arms"][arm]["geometry_records"])?,
            "native readout population differs from completed artifacts"
        );
    }
    frame_out.finish()?.flush()?;
    region_out.finish()?.flush()?;
    for w in event_out {
        w.finish()?.flush()?;
    }
    // Reverify immutable source hashes after consumption; no successful receipt
    // may survive a source epoch replacement during readout.
    a.verify()?;
    artifact_pin.verify()?;
    let mut outputs = BTreeMap::new();
    for name in [
        "paired_measurement_frames.jsonl.zst",
        "paired_receptor_membership.jsonl.zst",
        "treatment_native_event_readout.jsonl.zst",
        "control_native_event_readout.jsonl.zst",
    ] {
        outputs.insert(name, pin(&out.join(name))?);
    }
    let receipt = json!({"schema_kind":"prism.native_paired_mechanistic_readout_receipt.v1","status":"IMPLEMENTATION_VERIFIED",
        "source_artifact":artifact_pin,"source_manifest_sha256":a.manifest_sha256,"outputs":outputs,
        "frames":a.frame_steps.len(),"paired_region_frames":paired_count,"regions_per_frame":counts[0],
        "native_event_records":{"treatment":event_counts[0],"control":event_counts[1]},
        "events_without_receptor_face_matches_retained":{"treatment":unmatched[0],"control":unmatched[1]},
        "frames_without_events_retained":{"treatment":no_event_frames[0],"control":no_event_frames[1]},
        "native_kcc_records":{"treatment":kcc[0].count,"control":kcc[1].count},
        "frames_without_recorded_KCC":{"treatment":a.frame_steps.len() as u64-kcc[0].count,"control":a.frame_steps.len() as u64-kcc[1].count},
        "full_atom_vector_bytes_consumed":vector_end,"MD_executed":false,"measurements_reestimated":false,
        "TSH_admission_used":false,"TSH_intervals_supplied":false,"chemical_state_fiber_bundle_instantiated":false,
        "scope":"exact descriptive native paired receptor atom/residue membership and full population event/vector/tensor readout; no causal label, salience rank, effect uncertainty, or cross-basis projection"});
    write_json(&out.join("EXECUTION_RECEIPT.json"), &receipt)?;
    Ok(receipt)
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Reads complete real first-frame region populations and the first real
    /// native event. No synthetic scientific observations are generated.
    #[test]
    #[ignore = "requires completed native artifact via PRISM_NATIVE_READOUT_ARTIFACT"]
    fn actual_native_readout_geometry_and_event_correspondence() -> Result<()> {
        let path = PathBuf::from(std::env::var("PRISM_NATIVE_READOUT_ARTIFACT")?);
        let a = MechanisticMeasurementArtifact::load_verified(&path)?;
        let axes: Value =
            serde_json::from_reader(File::open(evaluated(&a, "measurement_axes.json")?.path)?)?;
        let axis: Vec<i32> = serde_json::from_value(axes["atom_analysis_residue_ids"].clone())?;
        let count = usize::try_from(u(
            &a.scope["geometry_grids"]["arms"]["treatment"]["regions_per_frame"]
        )?)?;
        let mut t = RowCursor::open(evaluated(&a, "treatment_native_geometry_index.jsonl.zst")?)?;
        let mut c = RowCursor::open(evaluated(&a, "control_native_geometry_index.jsonl.zst")?)?;
        let configs = geometry_configs(&a)?;
        let tt = read_geometry_frame(&mut t, &a, &axis, "treatment", 0, count, &configs)?;
        let cc = read_geometry_frame(&mut c, &a, &axis, "control", 0, count, &configs)?;
        ensure!(
            tt.keys().eq(cc.keys()),
            "actual paired first frame identities"
        );
        for (id, tr) in &tt {
            paired_geometry(tr, &cc[id], &t.source, &c.source)?;
        }
        let first = tt.values().next().context("actual geometry empty")?;
        let mut wrong = first.row.clone();
        wrong["run_id"] = json!(a.arm_run_ids["control"]);
        wrong.as_object_mut().unwrap().remove("record_sha256");
        wrong["record_sha256"] = json!(canonical_sha256(&wrong)?);
        ensure!(
            geometry_row(&a, &axis, "treatment", 0, 0, wrong).is_err(),
            "cross-arm geometry accepted"
        );
        let mut events = RowCursor::open(evaluated(
            &a,
            "treatment_native_event_measurements.jsonl.zst",
        )?)?;
        let (n, event) = events.take()?;
        let frame = usize::try_from(u(&event["coordinate_frame_ordinal"])?)?;
        let mut timeline = RowCursor::open(evaluated(&a, "vector_frames.jsonl")?)?;
        let mut kcc = RowCursor::open(evaluated(&a, "treatment_kcc_endpoint_index.jsonl")?)?;
        let mut geometry =
            RowCursor::open(evaluated(&a, "treatment_native_geometry_index.jsonl.zst")?)?;
        for prior in 0..frame {
            let (_, prior_row) = timeline.take()?;
            let measured = take_kcc_frame(&mut kcc, &a, "treatment", prior, &prior_row)?;
            if prior == 0 {
                ensure!(
                    measured.is_none(),
                    "actual source frame zero unexpectedly has KCC"
                );
            }
            for _ in 0..count {
                geometry.take()?;
            }
        }
        let (_, v) = timeline.take()?;
        let (_, k) = take_kcc_frame(&mut kcc, &a, "treatment", frame, &v)?
            .context("actual first event KCC missing")?;
        // Mutate actual records in memory only; do not generate scientific data.
        let following = kcc.next.take();
        let mut wrong_kcc = k.clone();
        wrong_kcc["step"] = json!(a.frame_steps[frame] + 1);
        kcc.next = Some((0, wrong_kcc));
        ensure!(
            take_kcc_frame(&mut kcc, &a, "treatment", frame, &v).is_err(),
            "inconsistent KCC step accepted"
        );
        kcc.next = Some((1, k.clone()));
        ensure!(
            take_kcc_frame(&mut kcc, &a, "treatment", frame + 1, &v).is_err(),
            "duplicate stale KCC endpoint accepted"
        );
        kcc.next = following;
        let regions = read_geometry_frame(
            &mut geometry,
            &a,
            &axis,
            "treatment",
            frame,
            count,
            &configs,
        )?;
        let mut links = RowCursor::open(evaluated(
            &a,
            "treatment_native_event_geometry_links.jsonl.zst",
        )?)?;
        let (_, l) = links.take()?;
        check_event(
            &a,
            &event,
            &l,
            "treatment",
            frame,
            n,
            &events.source,
            &geometry.source,
            &regions,
            &v,
            &k,
        )?;
        let mut wrong = event.clone();
        wrong["completed_step"] = json!(a.frame_steps[frame] + 1);
        ensure!(
            check_event(
                &a,
                &wrong,
                &l,
                "treatment",
                frame,
                n,
                &events.source,
                &geometry.source,
                &regions,
                &v,
                &k
            )
            .is_err(),
            "wrong exact event endpoint accepted"
        );
        let mut wrong = event.clone();
        if let Some(row) = wrong["residue_measurements"]
            .as_array_mut()
            .and_then(|rows| rows.first_mut())
        {
            row["analysis_residue_id"] = json!(u32::MAX);
        }
        ensure!(
            check_event(
                &a,
                &wrong,
                &l,
                "treatment",
                frame,
                n,
                &events.source,
                &geometry.source,
                &regions,
                &v,
                &k
            )
            .is_err(),
            "outside residue accepted"
        );
        Ok(())
    }
}
