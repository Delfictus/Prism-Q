//! Offline materializer-region observations. These are deliberately not Ghost records.
//! Reuses the native cavity grid and complete physical face attribution unchanged.
use super::*;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

#[derive(clap::Parser, Debug)]
pub struct Args {
    #[arg(long)]
    pub trajectory: PathBuf,
    #[arg(long)]
    pub audit: PathBuf,
    #[arg(long)]
    pub topology: PathBuf,
    #[arg(long)]
    pub sites: PathBuf,
    #[arg(long)]
    pub transform_recipe: PathBuf,
    #[arg(long)]
    pub transforms: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
    /// Explicit retrospective estimator declaration; absent preserves native defaults.
    #[arg(long)]
    pub config_json: Option<PathBuf>,
    /// Explicit partition; default processes the complete recorded population.
    #[arg(long, default_value_t = 0)]
    pub frame_start: usize,
    #[arg(long)]
    pub frame_end: Option<usize>,
    /// Optional explicit benchmark or population partition, never inferred from rank.
    #[arg(long)]
    pub region_id: Option<u64>,
    /// Emit all topology bonded coordinate observables once per frame.
    #[arg(long)]
    pub bonded_coordinates: bool,
    /// Explicit residue-attached reference/anchor declaration. Without this
    /// option the historical fixed laboratory ROI semantics are unchanged.
    #[arg(long)]
    pub body_fixed_anchors: Option<PathBuf>,
}
fn python_hash(value: &Value) -> Result<String> {
    crate::input::python_canonical_sha256(value).map_err(anyhow::Error::msg)
}
fn verify_self_hash(v: &Value, key: &str) -> Result<String> {
    let mut body = v.clone();
    let expected = body
        .as_object_mut()
        .context("hash body must be object")?
        .remove(key)
        .and_then(|v| v.as_str().map(str::to_owned))
        .context("missing self hash")?;
    if python_hash(&body)? != expected {
        bail!("{key} mismatch");
    }
    Ok(expected)
}
fn arr3(v: &Value) -> Result<[f64; 3]> {
    Ok(serde_json::from_value(v.clone())?)
}
fn apply(p: [f64; 3], r: [[f64; 3]; 3], t: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|j| p[0] * r[0][j] + p[1] * r[1][j] + p[2] * r[2][j] + t[j])
}
fn xyz(payload: &[u8], atom: usize) -> [f64; 3] {
    std::array::from_fn(|j| {
        f32::from_le_bytes(
            payload[12 * atom + 4 * j..12 * atom + 4 * j + 4]
                .try_into()
                .unwrap(),
        ) as f64
    })
}
pub fn run(args: Args) -> Result<()> {
    let start = std::time::Instant::now();
    let raw = read_regular_nofollow(&args.trajectory, "V1 trajectory")?;
    let audit_raw = read_regular_nofollow(&args.audit, "V1 audit")?;
    let audit: Value = serde_json::from_slice(&audit_raw)?;
    let top_raw = read_regular_nofollow(&args.topology, "topology")?;
    let mut top: PrismPrepTopology = serde_json::from_slice(&top_raw)?;
    let top_sha = sha256_bytes(&top_raw);
    let normalized =
        crate::ghost_coordinate_segment::normalize_legacy_nonperiodic_boundary(&mut top)?;
    top.validate_analysis_residue_axis()
        .map_err(anyhow::Error::msg)?;
    let metric = PhysicalSpatialMetric::from_topology(&top)?;
    if !matches!(
        metric.authority.mode,
        PhysicalCavitySpatialMetricMode::DeclaredNonperiodicEuclidean
    ) {
        bail!("recorded body-frame adapter currently requires declared nonperiodic coordinates; no implicit unwrap");
    }
    let identities = receptor_atom_identity_map(&top)?;
    let mapping_sha = top
        .analysis_residue_contract
        .as_ref()
        .unwrap()
        .atom_analysis_residue_ids_sha256
        .clone();
    let (config, config_file_binding) = if let Some(path) = &args.config_json {
        let bytes = read_regular_nofollow(path, "coordinate cavity estimator configuration")?;
        let effective: CoordinateCavityConfig = serde_json::from_slice(&bytes)
            .context("invalid coordinate cavity estimator configuration JSON")?;
        (effective, json!({"path":std::fs::canonicalize(path)?,"sha256":sha256_bytes(&bytes),"size_bytes":bytes.len()}))
    } else {
        (CoordinateCavityConfig::default(), Value::Null)
    };
    config.validate()?;
    let estimator_sha = canonical_sha256(&config)?;
    let config_manifest = json!({"schema_kind":"prism_recorded_cavity_estimator_configuration",
        "configuration_origin":if args.config_json.is_some(){"explicit_config_json"}else{"native_default"},
        "source_file":config_file_binding,"effective_config":config,"effective_config_sha256":estimator_sha,
        "scope":"explicit retrospective coordinate estimator; no automatic optimization or convergence claim"});
    let recipe_raw = read_regular_nofollow(&args.transform_recipe, "transform recipe")?;
    let recipe: Value = serde_json::from_slice(&recipe_raw)?;
    let recipe_sha = verify_self_hash(&recipe, "recipe_sha256")?;
    let trajectory_sha = sha256_bytes(&raw);
    let audit_sha = sha256_bytes(&audit_raw);
    if recipe["trajectory_sha256"] != trajectory_sha
        || recipe["trajectory_audit_sha256"] != audit_sha
        || recipe["topology_sha256"] != top_sha
    {
        bail!("Kabsch recipe source identity mismatch");
    }
    if raw.len() < 24
        || &raw[..8] != b"PRISM4D\0"
        || u32::from_le_bytes(raw[8..12].try_into().unwrap()) != 1
        || u32::from_le_bytes(raw[12..16].try_into().unwrap()) as usize != top.n_atoms
    {
        bail!("expected exact PRISM4D V1 trajectory matching topology");
    }
    let header_interval = u32::from_le_bytes(raw[16..20].try_into().unwrap()) as u64;
    let header_dt = f32::from_le_bytes(raw[20..24].try_into().unwrap()) as f64;
    if audit["writer_format"] != "PRISM4D_FRAMES_BIN_V1"
        || audit["save_trajectory_interval"].as_u64() != Some(header_interval)
        || audit["dt_ps"].as_f64() != Some(header_dt)
        || audit["n_atoms"].as_u64() != Some(top.n_atoms as u64)
        || audit["expected_bytes"].as_u64() != Some(raw.len() as u64)
    {
        bail!("V1 header disagrees with original writer audit");
    }
    let stride = top.n_atoms.checked_mul(12).context("stride overflow")?;
    let frames = audit["frames"]
        .as_array()
        .context("missing audited chronology")?;
    let count = frames.len();
    if count == 0
        || raw.len() != 24 + count * stride
        || audit["frames_written"].as_u64() != Some(count as u64)
        || audit["bytes_written"].as_u64() != Some(raw.len() as u64)
        || recipe["frame_count"].as_u64() != Some(count as u64)
    {
        bail!("V1 count/length/recipe mismatch");
    }
    let mut rolling = FNV_OFFSET;
    let mut prev = None;
    for (i, fr) in frames.iter().enumerate() {
        let step = fr["step"].as_u64().context("missing frame step")?;
        if prev.is_some_and(|p| step <= p)
            || fr["frame_index"].as_u64() != Some(i as u64)
            || fr["n_floats"].as_u64() != Some((top.n_atoms * 3) as u64)
        {
            bail!("invalid audited frame sequence");
        }
        prev = Some(step);
        let h = fnv_update(
            fnv_update(
                fnv_update(FNV_OFFSET, &step.to_le_bytes()),
                &((top.n_atoms * 3) as u32).to_le_bytes(),
            ),
            &raw[24 + i * stride..24 + (i + 1) * stride],
        );
        if fr["frame_hash64"].as_str() != Some(format!("{h:016x}").as_str()) {
            bail!("V1 frame hash mismatch at {i}");
        }
        rolling = fnv_update(rolling, &h.to_le_bytes());
    }
    if audit["rolling_hash64"].as_str() != Some(format!("{rolling:016x}").as_str()) {
        bail!("V1 rolling hash mismatch");
    }
    let fit = recipe["fit_atom_indices"]
        .as_array()
        .context("missing fit population")?
        .iter()
        .map(|v| {
            v.as_u64()
                .context("invalid fit id")
                .and_then(|i| usize::try_from(i).map_err(Into::into))
        })
        .collect::<Result<Vec<_>>>()?;
    if fit.len() < 3
        || fit.iter().any(|i| *i >= top.n_atoms)
        || fit.iter().collect::<BTreeSet<_>>().len() != fit.len()
    {
        bail!("invalid fit population");
    }
    if recipe["reference_frame_ordinal"] != 0
        || recipe["reference_step"] != frames[0]["step"]
        || recipe["reference_coordinate_payload_sha256"] != sha256_bytes(&raw[24..24 + stride])
    {
        bail!("first-frame anchor mismatch");
    }
    let reference = fit
        .iter()
        .map(|i| xyz(&raw[24..24 + stride], *i))
        .collect::<Vec<_>>();
    let site_raw = read_regular_nofollow(&args.sites, "materialized sites")?;
    let site_sha = sha256_bytes(&site_raw);
    let sites: Value = serde_json::from_slice(&site_raw)?;
    // Preserve every native region, including qualified rows outside materialized top-N.
    let population = [
        "binding_sites",
        "spatiotemporal_partial_candidates",
        "density_only_candidates",
    ]
    .iter()
    .flat_map(|key| sites[*key].as_array().into_iter().flatten())
    .collect::<Vec<_>>();
    let mut region_ids = BTreeSet::new();
    for row in &population {
        let id = row["native_causome"]["region_id"]
            .as_u64()
            .context("region lacks native identity")?;
        if !region_ids.insert(id) {
            bail!("duplicate native region identity across source buckets");
        }
    }
    let selected = population
        .iter()
        .filter(|s| {
            args.region_id
                .is_none_or(|id| s["native_causome"]["region_id"].as_u64() == Some(id))
        })
        .collect::<Vec<_>>();
    if selected.is_empty() {
        bail!("requested region population absent");
    }
    let body_tracking = if let Some(path) = &args.body_fixed_anchors {
        let bytes=read_regular_nofollow(path,"body-fixed anchor declaration")?;
        let declaration:Value=serde_json::from_slice(&bytes)?;
        if declaration["schema_kind"]!="prism_recorded_region_body_anchors_v1"
            || declaration["topology_sha256"]!=top_sha || declaration["sites_sha256"]!=site_sha {
            bail!("body-fixed anchor declaration source mismatch");
        }
        let reference_path=PathBuf::from(declaration["reference_trajectory"]["path"].as_str().context("reference trajectory path")?);
        let reference_bytes=read_regular_nofollow(&reference_path,"body-fixed reference trajectory")?;
        if sha256_bytes(&reference_bytes)!=declaration["reference_trajectory"]["sha256"].as_str().context("reference SHA")?
            || reference_bytes.len()<24 || &reference_bytes[..8]!=b"PRISM4D\0"
            || u32::from_le_bytes(reference_bytes[12..16].try_into().unwrap()) as usize!=top.n_atoms {
            bail!("body-fixed reference trajectory identity mismatch");
        }
        let ref_ordinal=declaration["reference_frame_ordinal"].as_u64().context("reference ordinal")? as usize;
        let begin=ref_ordinal.checked_mul(stride).and_then(|x|x.checked_add(24)).context("reference offset overflow")?;
        let payload=reference_bytes.get(begin..begin.checked_add(stride).context("reference length overflow")?).context("reference frame absent")?.to_vec();
        if sha256_bytes(&payload)!=declaration["reference_coordinate_payload_sha256"].as_str().context("reference coordinate SHA")? {
            bail!("body-fixed reference coordinate identity mismatch");
        }
        for site in &selected {
            let id=site["native_causome"]["region_id"].as_u64().unwrap().to_string();
            let ids:Vec<usize>=serde_json::from_value(declaration["region_anchor_atom_indices"][&id].clone()).context("region anchor atom indices")?;
            if ids.len()<3 || ids.iter().any(|i|*i>=top.n_atoms) || ids.iter().collect::<BTreeSet<_>>().len()!=ids.len() {
                bail!("invalid exact body-fixed anchor population for region {id}");
            }
            let points=ids.iter().map(|i|xyz(&payload,*i)).collect::<Vec<_>>();
            crate::fixed_cadence_coordinate::measure_body_frame_fit(&points,&points,arr3(&site["centroid_xyz"])?)?;
        }
        Some((declaration,payload,json!({"path":path,"sha256":sha256_bytes(&bytes)})))
    } else {None};
    let end = args.frame_end.unwrap_or(count);
    if args.frame_start >= end || end > count {
        bail!("invalid explicit frame partition");
    }
    // Hash the producer's exact canonical JSONL bytes. Re-encoding floating-point
    // decimals with another language's shortest formatter is not byte identity.
    let transform_rows = BufReader::new(std::fs::File::open(&args.transforms)?)
        .lines()
        .enumerate()
        .map(|(i, line)| -> Result<Value> {
            let line = line?;
            let row: Value = serde_json::from_str(&line)?;
            let digest = row["record_sha256"]
                .as_str()
                .context("missing transform digest")?;
            require_sha256(digest, "transform digest")?;
            let token = format!("\"record_sha256\":\"{}\",", digest);
            if line.matches(&token).count() != 1 {
                bail!("transform canonical hash field placement invalid at {i}");
            }
            let original_body = line.replacen(&token, "", 1);
            if sha256_bytes(original_body.as_bytes()) != digest {
                bail!("exact transform body hash mismatch at {i}");
            }
            Ok(row)
        })
        .collect::<Result<Vec<_>>>()?;
    if transform_rows.len() != count {
        bail!("transform row count mismatch");
    }
    std::fs::create_dir_all(&args.output)?;
    let partial = args.output.join("region_frame_cavities.jsonl.zst.partial");
    let target = args.output.join("region_frame_cavities.jsonl.zst");
    if target.exists() {
        bail!("complete cavity output already exists");
    }
    let output_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&partial)?;
    std::fs::write(args.output.join("estimator_config_manifest.json"), serde_json::to_vec_pretty(&config_manifest)?)?;
    let mut out = zstd::stream::write::Encoder::new(BufWriter::new(output_file), 1)?;
    let identity_file = args.output.join("atom_identity_table.json");
    std::fs::write(&identity_file, serde_json::to_vec_pretty(&identities)?)?;
    let identity_sha = sha256_bytes(&std::fs::read(&identity_file)?);
    let mut bonded = if args.bonded_coordinates {
        let mut term_index = json!({"schema_kind":"prism_recorded_bonded_term_index","topology_sha256":top_sha,"atom_indices":"zero-based original topology atom order","term_ids":"zero-based array index; duplicate Fourier terms retained","bonds":top.bonds,"angles":top.angles,"dihedrals":top.dihedrals,"distance_unit":"angstrom","angle_unit":"radian","torsion_semantics":"signed atan2, exact native endpoint bonded convention","undefined_geometry":"null with native singularity status; never filled with zero","boundary":"declared nonperiodic original coordinates"});
        let declared_topology: Value = serde_json::from_slice(&top_raw)?;
        term_index["dihedral_classification_provenance"] = json!({
            "source_topology_sha256":top_sha,
            "per_term_improper_field_declared":declared_topology["dihedrals"].as_array()
                .context("source topology lacks dihedral declarations")?.iter()
                .map(|term|term.get("improper").is_some()).collect::<Vec<_>>(),
            "reader_default_when_absent":false,
            "absent_source_field_meaning":"unknown proper/improper classification; deserialization default is not a source measurement"
        });
        std::fs::write(
            args.output.join("bonded_term_index.json"),
            serde_json::to_vec_pretty(&term_index)?,
        )?;
        let target = args.output.join("bonded_coordinates.jsonl.zst");
        if target.exists() {
            bail!("bonded output exists");
        }
        Some(zstd::stream::write::Encoder::new(
            BufWriter::new(
                OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(args.output.join("bonded_coordinates.jsonl.zst.partial"))?,
            ),
            1,
        )?)
    } else {
        None
    };
    let mut observation_count = 0u64;
    let mut volumes = Vec::new();
    for (i, fr) in frames.iter().enumerate() {
        let row = &transform_rows[i];
        let transform_sha = row["record_sha256"].as_str().unwrap();
        let payload = &raw[24 + i * stride..24 + (i + 1) * stride];
        let payload_sha = sha256_bytes(payload);
        if row["recipe_sha256"] != recipe_sha
            || row["frame_ordinal"] != i
            || row["step"] != fr["step"]
            || row["coordinate_payload_sha256"] != payload_sha
            || row["coordinate_frame_hash64"] != fr["frame_hash64"]
            || row["coordinate_byte_offset"] != 24 + i * stride
            || row["coordinate_byte_length"] != stride
        {
            bail!("transform coordinate binding mismatch at {i}");
        }
        let r: [[f64; 3]; 3] = serde_json::from_value(row["rotation"].clone())?;
        let t = arr3(&row["translation_angstrom"])?;
        if r.iter().flatten().chain(t.iter()).any(|x| !x.is_finite()) {
            bail!("nonfinite transform");
        }
        let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
            - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
            + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
        let orth = (0..3)
            .flat_map(|a| {
                (0..3).map(move |b| {
                    ((0..3).map(|k| r[k][a] * r[k][b]).sum::<f64>()
                        - if a == b { 1.0 } else { 0.0 })
                    .abs()
                })
            })
            .fold(0.0, f64::max);
        // Numerical identity checks, never physical fit quality gates.
        if (det - 1.0).abs() > 1e-10 || orth > 1e-10 {
            bail!("improper Kabsch rotation");
        }
        let mut sum = 0.;
        let mut max: f64 = 0.;
        for (a, reference) in fit.iter().zip(&reference) {
            let p = apply(xyz(payload, *a), r, t);
            let d = (0..3).map(|j| (p[j] - reference[j]).powi(2)).sum::<f64>();
            sum += d;
            max = max.max(d.sqrt());
        }
        let rms = (sum / fit.len() as f64).sqrt();
        if (rms - row["fit_rmsd_angstrom"].as_f64().context("missing RMSD")?).abs() > 1e-9
            || (max
                - row["fit_max_residual_angstrom"]
                    .as_f64()
                    .context("missing fit residual")?)
            .abs()
                > 1e-9
        {
            bail!("transform residual inconsistency");
        }
        if i < args.frame_start || i >= end {
            continue;
        }
        if let Some(writer) = bonded.as_mut() {
            let original = (0..top.n_atoms)
                .map(|a| xyz(payload, a))
                .collect::<Vec<_>>();
            let values =
                crate::lambda_fragment::recorded_bonded_coordinate_observables(&original, &top)?;
            let mut record = json!({"schema_kind":"prism_recorded_bonded_coordinates","frame_ordinal":i,"step":fr["step"],"source_coordinate_payload_sha256":payload_sha,"topology_sha256":top_sha,"observables":values});
            record["record_sha256"] = canonical_sha256(&record)?.into();
            serde_json::to_writer(&mut *writer, &record)?;
            writer.write_all(b"\n")?;
        }
        let transformed = identities
            .keys()
            .map(|id| (*id, apply(xyz(payload, *id), r, t)))
            .collect::<BTreeMap<_, _>>();
        for site in &selected {
            let region = site["native_causome"]["region_id"]
                .as_u64()
                .context("missing native region identity")?;
            let lab_center = arr3(&site["centroid_xyz"])?;
            // Transform the recorded laboratory ROI together with all atoms.
            // This preserves its physical location; only grid axes are body-aligned.
            let transport=if let Some((declaration,reference_payload,_))=&body_tracking {
                let ids:Vec<usize>=serde_json::from_value(declaration["region_anchor_atom_indices"][region.to_string()].clone())?;
                let reference=ids.iter().map(|id|xyz(reference_payload,*id)).collect::<Vec<_>>();
                let current=ids.iter().map(|id|xyz(payload,*id)).collect::<Vec<_>>();
                Some(crate::fixed_cadence_coordinate::measure_body_frame_fit(&reference,&current,lab_center)?)
            } else {None};
            let center = if transport.is_some(){lab_center}else{apply(lab_center, r, t)};
            let mut binding = json!({"schema_kind":"prism_materializer_region_frame","schema_version":1,"region_id":region,"materialized_sites_sha256":site_sha,"source_region_record_sha256":canonical_sha256(site)?,"frame_ordinal":i,"step":fr["step"],"stream_id":recipe["stream_id"],"original_coordinate_payload_sha256":payload_sha,"trajectory_sha256":trajectory_sha,"audit_sha256":audit_sha,"body_frame_recipe_sha256":recipe_sha,"body_frame_transform_sha256":transform_sha,"recorded_laboratory_roi_centroid_angstrom":lab_center,"aligned_grid_center_angstrom":center,"roi_center_semantics":"same recorded laboratory region transformed by this frame R+t; axes evaluated in reference body frame; centroid was not measured in first reference frame","scope":"retrospective descriptive coordinate geometry"});
            if let Some(fit)=&transport {
                binding["roi_center_semantics"]=json!("declared region center transported from explicitly pinned reference coordinates by native ordered residue anchors; grid and atoms evaluated in that region reference frame");
                binding["body_fixed_tracking"]=json!({"declaration":body_tracking.as_ref().unwrap().2,"transform":fit,
                    "fit_residual_role":"measured deformation diagnostic; no retrospective quality/admission threshold", "center_is_new_detected_site":false});
            }
            let observation_sha = canonical_sha256(&binding)?;
            let atoms = identities
                .iter()
                .map(|(id, identity)| {
                    let pos = transport.as_ref().map_or_else(||transformed[id],|fit|fit.point_in_reference(xyz(payload,*id)));
                    let sigma = top.lj_params[*id].sigma;
                    if !sigma.is_finite() || sigma <= 0. {
                        bail!("invalid topology LJ sigma");
                    }
                    Ok(LocalReceptorAtom {
                        position_relative_angstrom: std::array::from_fn(|j| pos[j] - center[j]),
                        solvent_excluded_radius_angstrom: 0.5 * 2f64.powf(1. / 6.) * sigma
                            + config.solvent_probe_radius_a,
                        identity: identity.clone(),
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let source = PhysicalGridSourceBinding {
                region_frame_observation: true,
                ghost_record_sha256: &observation_sha,
                coordinate_frame_sha256: &payload_sha,
                topology_sha256: &top_sha,
                atom_residue_mapping_sha256: &mapping_sha,
                estimator_sha256: &estimator_sha,
            };
            let mut grids = Vec::new();
            let mut estimates = Vec::new();
            for spacing in &config.grid_spacings_a {
                for phase in &config.grid_phase_offsets {
                    let g = evaluate_grid_with_physical_lining(
                        &atoms,
                        *spacing,
                        *phase,
                        grids.len() as u64,
                        &config,
                        &metric,
                        &source,
                    )?;
                    estimates.push(g.estimate);
                    grids.push(g.lining);
                }
            }
            let mean = estimates.iter().map(|g| g.volume_a3).sum::<f64>() / estimates.len() as f64;
            let se = (estimates
                .iter()
                .map(|g| (g.volume_a3 - mean).powi(2))
                .sum::<f64>()
                / ((estimates.len() - 1) * estimates.len()) as f64)
                .sqrt();
            let mut result = json!({"schema_kind":"prism_materializer_region_frame_cavity","schema_version":1,"observation_binding":binding,"observation_sha256":observation_sha,"topology_sha256":top_sha,"atom_identity_table_sha256":identity_sha,"estimator_config":config,"estimator_config_authority":config_manifest,"estimator_sha256":estimator_sha,"spatial_metric":metric.authority,"mean_volume_a3":mean,"discretization_standard_error_a3":se,"grid_estimates":estimates,"physical_lining_grids":grids});
            result["record_sha256"] = canonical_sha256(&result)?.into();
            serde_json::to_writer(&mut out, &result)?;
            out.write_all(b"\n")?;
            observation_count += 1;
            volumes.push(mean);
        }
        if (i - args.frame_start) % 100 == 0 {
            out.flush()?;
            eprintln!(
                "frame {i}/{end}, observations {observation_count}, elapsed {:.3}s",
                start.elapsed().as_secs_f64()
            );
        }
    }
    out.flush()?;
    let mut file = out.finish()?;
    file.flush()?;
    file.get_ref().sync_all()?;
    std::fs::rename(&partial, &target)?;
    if let Some(mut writer) = bonded {
        writer.flush()?;
        let mut file = writer.finish()?;
        file.flush()?;
        file.get_ref().sync_all()?;
        std::fs::rename(
            args.output.join("bonded_coordinates.jsonl.zst.partial"),
            args.output.join("bonded_coordinates.jsonl.zst"),
        )?;
    }
    let receipt = json!({"status":"IMPLEMENTATION_VERIFIED","schema_kind":"prism_recorded_roi_cavity_execution","trajectory_sha256":trajectory_sha,"audit_sha256":audit_sha,"sites_sha256":site_sha,"topology_sha256":top_sha,"transform_recipe_sha256":recipe_sha,"native_estimator_sha256":estimator_sha,"estimator_config_authority":config_manifest,"recorded_frames_verified":count,"frame_start":args.frame_start,"frame_end":end,"region_population":selected.len(),"observations":observation_count,"expected_observations":(end-args.frame_start)*selected.len(),"all_recorded_frames_processed":args.frame_start==0&&end==count,"all_materialized_regions_processed":selected.len()==population.len(),"original_ghost_identity_preserved":true,"MD_rerun":false,"lossless_compression":"zstd level1; all grid bytes and physical face records retained","bonded_coordinates_emitted":args.bonded_coordinates,"legacy_nonperiodic_declaration_normalized_in_memory":normalized,"elapsed_seconds":start.elapsed().as_secs_f64(),"minimum_volume_a3":volumes.iter().copied().fold(f64::INFINITY,f64::min),"maximum_volume_a3":volumes.iter().copied().fold(f64::NEG_INFINITY,f64::max)});
    std::fs::write(
        args.output.join("execution_receipt.json"),
        serde_json::to_vec_pretty(&receipt)?,
    )?;
    eprintln!("{receipt}");
    Ok(())
}
