//! Native, source-bound measurements underlying chemical-state fibers.
//!
//! This evaluator is independent of TSH salience and cavity eligibility. It
//! does not manufacture a periodic/alchemical fiber for open-boundary data.
//! Population covariance describes the recorded trajectory, not uncertainty
//! of an independent-sample mean. Original lab-frame records remain pinned.
pub mod geometry;
pub mod physical_time;
mod control_work;
pub mod kcc_time;
#[cfg(any(feature = "gpu", feature = "cpu-analysis"))]
pub mod measurement_fibers;
pub mod fiber_readout;
pub mod readout;

use super::{canonical_sha256, require_sha256, ChemicalStateFiber, GhostByteLineage};
use anyhow::{bail, ensure, Context, Result};
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::{Field, Row};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeasurementSourcePin {
    pub path: PathBuf,
    pub sha256: String,
}
impl MeasurementSourcePin {
    pub fn verify(&self) -> Result<()> {
        require_sha256(&self.sha256, "measurement source")?;
        ensure!(
            digest_file(&self.path)? == self.sha256,
            "measurement source changed: {}",
            self.path.display()
        );
        Ok(())
    }
}
/// The prepared native topology reserves negative analysis IDs for atoms
/// outside the receptor. Preserve those signed IDs on the full atom axis;
/// only receptor reductions use the optional compact residue-axis index.
fn receptor_residue_axis(signed_atom_ids: &[i32]) -> (Vec<u32>, Vec<Option<usize>>) {
    let residues=signed_atom_ids.iter().filter_map(|id|u32::try_from(*id).ok())
        .collect::<BTreeSet<_>>().into_iter().collect::<Vec<_>>();
    let lookup=residues.iter().enumerate().map(|(i,id)|(*id,i)).collect::<BTreeMap<_,_>>();
    let membership=signed_atom_ids.iter().map(|id|
        u32::try_from(*id).ok().map(|id|lookup[&id])).collect();
    (residues,membership)
}

fn digest_file(path: &Path) -> Result<String> {
    let mut f =
        BufReader::new(File::open(path).with_context(|| format!("open {}", path.display()))?);
    let mut h = Sha256::new();
    let mut buf = vec![0; 1024 * 1024];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}
fn pin(path: &Path) -> Result<MeasurementSourcePin> {
    Ok(MeasurementSourcePin {
        path: path.to_path_buf(),
        sha256: digest_file(path)?,
    })
}
fn source(v: &Value) -> Result<MeasurementSourcePin> {
    Ok(serde_json::from_value(v.clone())?)
}
fn load(v: &Value) -> Result<Value> {
    Ok(serde_json::from_reader(File::open(source(v)?.path)?)?)
}
/// One explicit physical condition lane per paired artifact. Historical
/// manifests default to stream zero; both arms must use this declared lane.
fn manifest_stream_id(manifest: &Value) -> Result<u16> {
    let id = match manifest.get("stream_id") {
        None => 0,
        Some(value) => u16::try_from(u(value)?).context("measurement stream exceeds native identity range")?,
    };
    for arm in ["treatment", "control"] {
        if let Some(value) = manifest["mechanistic_sources"]["arms"][arm].get("stream_id") {
            ensure!(u(value)? == id as u64, "paired artifact mixes physical stream lanes");
        }
    }
    Ok(id)
}
fn u(v: &Value) -> Result<u64> {
    v.as_u64().context("expected unsigned integer")
}
fn text(v: &Value) -> Result<&str> {
    v.as_str().context("expected text")
}
fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}
fn line(w: &mut impl Write, value: &impl Serialize) -> Result<()> {
    serde_json::to_writer(&mut *w, value)?;
    w.write_all(b"\n")?;
    Ok(())
}
fn collect_pins(v: &Value, pins: &mut BTreeMap<PathBuf, MeasurementSourcePin>) -> Result<()> {
    if v.get("path").is_some() && v.get("sha256").is_some() {
        let p = source(v)?;
        if let Some(old) = pins.insert(p.path.clone(), p.clone()) {
            ensure!(old == p, "conflicting source pins");
        }
    }
    match v {
        Value::Object(o) => {
            for x in o.values() {
                collect_pins(x, pins)?
            }
        }
        Value::Array(a) => {
            for x in a {
                collect_pins(x, pins)?
            }
        }
        _ => {}
    }
    Ok(())
}

// Historical producer metadata may name the same path at an older epoch.
// Only the manifest's active inputs are dereferenced. Preserve historical
// identity separately instead of reinterpreting its path as the current file.
fn collect_manifest_pins(m: &Value) -> Result<BTreeMap<PathBuf, MeasurementSourcePin>> {
    let mut pins = BTreeMap::new();
    for key in [
        "source_files",
        "frames",
        "force_treatment",
        "force_control",
        "force_chronology_receipt",
        "mechanistic_sources",
    ] {
        ensure!(
            !m[key].is_null(),
            "required active input field missing: {key}"
        );
        collect_pins(&m[key], &mut pins)?;
    }
    Ok(pins)
}

/// Content-addressed evaluated evidence. This is an additional measurement
/// artifact, not a substitute for ChemicalStateFiber's physical contracts.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MechanisticMeasurementArtifact {
    pub schema_kind: String,
    pub manifest_sha256: String,
    pub topology_sha256: String,
    pub arm_run_ids: BTreeMap<String, String>,
    pub stream_id: u16,
    pub frame_steps: Vec<u64>,
    pub frame_phases: Vec<String>,
    pub atom_count: usize,
    pub analysis_residue_ids: Vec<u32>,
    pub source_pins: Vec<MeasurementSourcePin>,
    pub evaluated_artifacts: BTreeMap<String, MeasurementSourcePin>,
    pub scope: Value,
}
impl MechanisticMeasurementArtifact {
    /// Dereference and verify every measurement artifact before attachment.
    pub fn load_verified(path: &Path) -> Result<Self> {
        let a: Self = serde_json::from_reader(File::open(path)?)?;
        a.verify()?;
        Ok(a)
    }
    pub fn verify(&self) -> Result<()> {
        ensure!(
            self.schema_kind == "prism.native_fiber_mechanistic_measurements.v1",
            "unsupported mechanism schema"
        );
        require_sha256(&self.manifest_sha256, "manifest")?;
        require_sha256(&self.topology_sha256, "topology")?;
        ensure!(
            self.atom_count > 0 && !self.frame_steps.is_empty(),
            "empty mechanism axes"
        );
        ensure!(
            self.frame_steps.windows(2).all(|x| x[0] < x[1]),
            "nonmonotonic mechanism steps"
        );
        ensure!(
            self.analysis_residue_ids.windows(2).all(|x| x[0] < x[1]),
            "duplicate residue axis"
        );
        ensure!(
            !self.evaluated_artifacts.is_empty(),
            "unevaluated mechanism artifact"
        );
        for p in self
            .source_pins
            .iter()
            .chain(self.evaluated_artifacts.values())
        {
            p.verify()?
        }
        let manifest_pin = self
            .source_pins
            .iter()
            .find(|p| p.sha256 == self.manifest_sha256)
            .context("pinned source manifest missing")?;
        let manifest: Value = serde_json::from_reader(File::open(&manifest_pin.path)?)?;
        ensure!(
            manifest["topology_sha256"] == self.topology_sha256
                && u(&manifest["n_atoms"])? as usize == self.atom_count,
            "artifact axes differ from pinned manifest"
        );
        for arm in ["treatment", "control"] {
            ensure!(
                self.arm_run_ids.get(arm).context("arm identity missing")?
                    == text(&manifest[format!("{arm}_run_id")])?,
                "artifact run differs from source manifest"
            );
        }
        ensure!(
            manifest["topology_sha256"] == manifest["mechanistic_sources"]["topology"]["sha256"],
            "manifest topology differs from pinned topology source"
        );
        let expected = collect_manifest_pins(&manifest)?;
        for p in expected.values() {
            ensure!(
                self.source_pins.contains(p),
                "artifact lost required source pin"
            );
        }
        let topology = load(&manifest["mechanistic_sources"]["topology"])?;
        let ids: Vec<i32> = serde_json::from_value(topology["atom_analysis_residue_ids"].clone())?;
        let (unique,_) = receptor_residue_axis(&ids);
        ensure!(
            unique == self.analysis_residue_ids,
            "artifact residue axis differs from topology"
        );
        ensure!(
            self.frame_phases.len() == self.frame_steps.len(),
            "artifact phase axis missing"
        );
        let timeline = self
            .evaluated_artifacts
            .get("vector_frames.jsonl")
            .context("evaluated timeline missing")?;
        let mut count = 0;
        for (i, l) in BufReader::new(File::open(&timeline.path)?)
            .lines()
            .enumerate()
        {
            let v: Value = serde_json::from_str(&l?)?;
            ensure!(
                i < self.frame_steps.len()
                    && u(&v["step"])? == self.frame_steps[i]
                    && text(&v["phase"])? == self.frame_phases[i],
                "artifact metadata differs from evaluated chronology"
            );
            count += 1;
        }
        ensure!(
            count == self.frame_steps.len(),
            "artifact timeline length mismatch"
        );
        let mut count = 0;
        for (i, l) in BufReader::new(File::open(source(&manifest["frames"])?.path)?)
            .lines()
            .enumerate()
        {
            let v: Value = serde_json::from_str(&l?)?;
            ensure!(
                i < self.frame_steps.len()
                    && u(&v["step"])? == self.frame_steps[i]
                    && text(&v["phase"])? == self.frame_phases[i],
                "artifact metadata differs from source chronology"
            );
            count += 1;
        }
        ensure!(
            count == self.frame_steps.len() && self.stream_id == manifest_stream_id(&manifest)?,
            "artifact source chronology/stream mismatch"
        );
        Ok(())
    }
    fn validate_fiber_phase(&self, fiber: &ChemicalStateFiber) -> Result<()> {
        self.validate_ghost(&fiber.body.ghost)?;
        let i = self
            .frame_steps
            .binary_search(&fiber.body.ghost.key.step_idx)
            .ok()
            .context("fiber step")?;
        ensure!(
            self.frame_phases[i] == fiber.body.protocol_phase.phase_name,
            "fiber protocol phase differs from measured interval"
        );
        Ok(())
    }
    /// Match by exact run, stream, topology and recorded step, never rank or
    /// centroid. No periodic, Hamiltonian or endpoint eligibility is changed.
    pub fn validate_ghost(&self, ghost: &GhostByteLineage) -> Result<()> {
        ensure!(
            self.arm_run_ids.values().any(|r| r == &ghost.run_id),
            "mechanism run differs from Ghost"
        );
        ensure!(
            self.stream_id == ghost.key.stream_id,
            "mechanism stream differs from Ghost"
        );
        ensure!(
            self.topology_sha256 == ghost.topology_sha256,
            "mechanism topology differs from Ghost"
        );
        ensure!(
            self.frame_steps.binary_search(&ghost.key.step_idx).is_ok(),
            "Ghost step lacks exact measured coordinates"
        );
        Ok(())
    }
    /// Produce a new validated fiber; preserve the original content identity.
    pub fn attach_to_fiber(&self, fiber: &ChemicalStateFiber) -> Result<ChemicalStateFiber> {
        self.verify()?;
        fiber.verify()?;
        self.validate_fiber_phase(fiber)?;
        let mut body = fiber.body.clone();
        body.evidence_artifact_sha256.insert(
            "native_mechanistic_measurements_v1".into(),
            canonical_sha256(self)?,
        );
        ChemicalStateFiber::materialize(body)
    }
    /// Attach to an already valid bundle and conserve its exact population.
    pub fn attach_to_bundle(
        &self,
        bundle: &crate::pathb_chemical_state::NativePathBFiberBundle,
    ) -> Result<crate::pathb_chemical_state::NativePathBFiberBundle> {
        self.verify()?;
        bundle.verify()?;
        let digest = canonical_sha256(self)?;
        let mut result = bundle.clone();
        for fiber in &mut result.fibers {
            self.validate_fiber_phase(fiber)?;
            let mut body = fiber.body.clone();
            body.evidence_artifact_sha256
                .insert("native_mechanistic_measurements_v1".into(), digest.clone());
            *fiber = ChemicalStateFiber::materialize(body)?;
        }
        let mut hashes: Vec<_> = result
            .fibers
            .iter()
            .map(|f| f.fiber_sha256.clone())
            .collect();
        hashes.sort();
        result.fiber_population_sha256 = canonical_sha256(&hashes)?;
        result.verify()?;
        Ok(result)
    }
}

#[derive(Clone, Default, Debug, Serialize)]
struct VectorMoments {
    n: u64,
    mean: [f64; 3],
    m2: [f64; 6],
}
impl VectorMoments {
    fn add(&mut self, x: [f64; 3]) -> Result<()> {
        ensure!(x.iter().all(|x| x.is_finite()), "nonfinite vector");
        self.n += 1;
        let delta = std::array::from_fn::<_, 3, _>(|i| x[i] - self.mean[i]);
        for i in 0..3 {
            self.mean[i] += delta[i] / self.n as f64;
        }
        let after = std::array::from_fn::<_, 3, _>(|i| x[i] - self.mean[i]);
        for (k, (i, j)) in [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
            .into_iter()
            .enumerate()
        {
            self.m2[k] += delta[i] * after[j];
        }
        Ok(())
    }
    fn value(&self) -> Value {
        if self.n == 0 {
            json!({"n":0,"mean":null,"population_covariance_upper":null})
        } else {
            json!({"n":self.n,"mean":self.mean,"population_covariance_upper":self.m2.map(|x|x/self.n as f64)})
        }
    }
}
#[derive(Clone, Default)]
struct ScalarMoments {
    n: u64,
    missing: u64,
    mean: f64,
    m2: f64,
    min: f64,
    max: f64,
    circular: VectorMoments,
}
impl ScalarMoments {
    fn add(&mut self, x: Option<f64>, circular: bool) -> Result<()> {
        let Some(x) = x else {
            self.missing += 1;
            return Ok(());
        };
        ensure!(x.is_finite(), "nonfinite scalar");
        if self.n == 0 {
            self.min = x;
            self.max = x;
        }
        self.n += 1;
        let d = x - self.mean;
        self.mean += d / self.n as f64;
        self.m2 += d * (x - self.mean);
        self.min = self.min.min(x);
        self.max = self.max.max(x);
        if circular {
            self.circular.add([x.cos(), x.sin(), 0.0])?;
        }
        Ok(())
    }
    fn value(&self, circular: bool) -> Value {
        if self.n == 0 {
            return json!({"n":0,"missing":self.missing,"mean":null});
        }
        if circular {
            let c = self.circular.mean;
            json!({"n":self.n,"missing":self.missing,"circular_mean_radians":if c[0]==0.0 && c[1]==0.0 {None}else{Some(c[1].atan2(c[0]))},"resultant_length":c[0].hypot(c[1]),"cos_sin_moments":self.circular.value()})
        } else {
            json!({"n":self.n,"missing":self.missing,"mean":self.mean,"population_variance":self.m2/self.n as f64,"min":self.min,"max":self.max})
        }
    }
}
#[derive(Clone, Deserialize)]
struct Transform {
    frame_ordinal: usize,
    step: u64,
    rotation: [[f64; 3]; 3],
    translation_angstrom: [f64; 3],
    reference_coordinate_payload_sha256: String,
    coordinate_payload_sha256: String,
    coordinate_frame_hash64: String,
}
fn rotate(x: [f64; 3], r: [[f64; 3]; 3]) -> [f64; 3] {
    std::array::from_fn(|j| (0..3).map(|i| x[i] * r[i][j]).sum())
}
fn position(x: [f64; 3], t: &Transform) -> [f64; 3] {
    let y = rotate(x, t.rotation);
    std::array::from_fn(|j| y[j] + t.translation_angstrom[j])
}
fn validate_rotation(r: [[f64; 3]; 3]) -> Result<()> {
    ensure!(
        r.iter().flatten().all(|x| x.is_finite()),
        "nonfinite rotation"
    );
    for i in 0..3 {
        for j in 0..3 {
            let dot = (0..3).map(|k| r[i][k] * r[j][k]).sum::<f64>();
            ensure!(
                (dot - if i == j { 1.0 } else { 0.0 }).abs() < 1e-9,
                "nonorthogonal basis"
            );
        }
    }
    let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
        - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
        + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
    ensure!((det - 1.0).abs() < 1e-9, "improper rotation");
    Ok(())
}
fn raw_vectors(raw: &[u8]) -> Result<Vec<[f64; 3]>> {
    ensure!(raw.len() % 12 == 0, "partial vector");
    raw.chunks_exact(12)
        .map(|b| {
            let x = std::array::from_fn(|i| {
                f32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap()) as f64
            });
            ensure!(x.iter().all(|x| x.is_finite()), "nonfinite raw vector");
            Ok(x)
        })
        .collect()
}
fn force_frames(m: &Value, arm: &str, n: usize) -> Result<BTreeMap<u64, Vec<[f64; 3]>>> {
    let p = source(&m[format!("force_{arm}")])?;
    let receipt = load(&m["force_chronology_receipt"])?;
    let r = &receipt["arms"][if arm == "control" {
        "uv_off_control"
    } else {
        arm
    }];
    ensure!(
        r["full_force_payload_equality_ASC_vs_ZSTR"] == true
            && r["full_coordinate_payload_equality_ZSTR_vs_native_frames"] == true,
        "force correspondence was not verified"
    );
    ensure!(
        r["sources"]
            .as_array()
            .context("force receipt sources")?
            .iter()
            .any(|x| x["sha256"] == p.sha256),
        "force receipt does not bind this file"
    );
    let offset = u(&m["force_completed_step_offset"])?;
    let cadence = u(&m["cadence_steps"])?;
    ensure!(
        u(&r["actual_completed_step_first_last"][0])?
            .checked_sub(u(&r["stored_asc_step_first_last"][0])?)
            == Some(offset),
        "unverified force offset"
    );
    let mut f = BufReader::new(File::open(&p.path)?);
    let mut head = [0; 8];
    f.read_exact(&mut head)?;
    ensure!(u64::from_le_bytes(head) == n as u64, "force atom count");
    let bytes = 16 + n * 12;
    let size = fs::metadata(&p.path)?.len() as usize;
    ensure!((size - 8) % bytes == 0, "partial force record");
    let mut out = BTreeMap::new();
    let mut raw = vec![0; bytes];
    let mut prior = None;
    for _ in 0..(size - 8) / bytes {
        f.read_exact(&mut raw)?;
        let chunk = u64::from_le_bytes(raw[..8].try_into()?);
        let stored = u64::from_le_bytes(raw[8..16].try_into()?);
        ensure!(
            chunk.checked_mul(cadence) == Some(stored),
            "force chunk clock mismatch"
        );
        let step = stored.checked_add(offset).context("force clock overflow")?;
        if let Some(p) = prior {
            ensure!(p + cadence == step, "force chronology gap");
        }
        prior = Some(step);
        ensure!(
            out.insert(step, raw_vectors(&raw[16..])?).is_none(),
            "duplicate force step"
        );
    }
    ensure!(
        out.len() as u64 == u(&r["records"])?,
        "force receipt count mismatch"
    );
    ensure!(
        out.keys().next().copied() == Some(u(&r["actual_completed_step_first_last"][0])?)
            && out.keys().next_back().copied()
                == Some(u(&r["actual_completed_step_first_last"][1])?),
        "force receipt endpoint mismatch"
    );
    Ok(out)
}
#[derive(Clone, Serialize)]
struct Frame {
    ordinal: usize,
    step: u64,
    time_ps: Option<f64>,
    arm_physical_time: BTreeMap<String, physical_time::RecordedPhysicalTimeEndpoint>,
    phase: String,
    window: usize,
}
impl Frame {
    fn time_for_arm(&self, arm: &str) -> Result<f64> {
        match self.arm_physical_time.get(arm) {
            Some(row) => Ok(row.native_record.cumulative_simulated_time_ps),
            None if self.arm_physical_time.is_empty() => self.time_ps.context("historical frame time missing"),
            None => bail!("exact physical time missing for arm {arm}"),
        }
    }
}
#[derive(Serialize)]
struct Window {
    window_id: usize,
    phase: String,
    first_step: u64,
    last_step: u64,
    frames: usize,
}
fn field<'a>(row: &'a Row, name: &str) -> Result<&'a Field> {
    row.get_column_iter()
        .find(|(n, _)| n.as_str() == name)
        .map(|(_, v)| v)
        .with_context(|| format!("missing parquet field {name}"))
}
fn fi(row: &Row, name: &str) -> Result<u64> {
    match field(row, name)? {
        Field::Long(n) if *n >= 0 => Ok(*n as u64),
        _ => bail!("invalid parquet integer {name}"),
    }
}
fn fsx<'a>(row: &'a Row, name: &str) -> Result<&'a str> {
    match field(row, name)? {
        Field::Str(s) => Ok(s),
        _ => bail!("invalid parquet text {name}"),
    }
}
fn floats(row: &Row, name: &str) -> Result<Vec<Option<f64>>> {
    match field(row, name)? {
        Field::ListInternal(list) => list
            .elements()
            .iter()
            .map(|v| match v {
                Field::Double(x) if x.is_finite() => Ok(Some(*x)),
                Field::Null => Ok(None),
                _ => bail!("invalid parquet float {name}"),
            })
            .collect(),
        _ => bail!("invalid parquet list {name}"),
    }
}
fn frame_check(row: &Row, frames: &[Frame], i: usize) -> Result<()> {
    ensure!(i < frames.len(), "extra parquet frame");
    ensure!(
        fi(row, "frame_ordinal")? == i as u64
            && fi(row, "step")? == frames[i].step
            && fsx(row, "scheduled_phase")? == frames[i].phase,
        "parquet chronology mismatch at {i}"
    );
    Ok(())
}
fn binary_vectors(w: &mut impl Write, x: &[[f64; 3]]) -> Result<()> {
    for v in x {
        for a in v {
            w.write_all(&a.to_le_bytes())?;
        }
    }
    Ok(())
}

/// Evaluate full recorded tensors and geometric time series without MD or TSH
/// admission. Every source is pinned before decoding; outputs never overwrite.
pub fn evaluate_mechanistic_pair(manifest_path: &Path, out: &Path) -> Result<Value> {
    ensure!(
        !out.exists(),
        "preserve existing mechanistic output {}",
        out.display()
    );
    let original_input_manifest=pin(manifest_path)?;
    let mut m: Value = serde_json::from_reader(File::open(manifest_path)?)?;
    original_input_manifest.verify()?;
    let native_control_work=control_work::prepare(&mut m,&original_input_manifest,out)?;
    let native_work_sources_resolved=native_control_work.sources_resolved;
    let s = &m["mechanistic_sources"];
    ensure!(s.is_object(), "mechanistic sources missing");
    let mut pins = collect_manifest_pins(&m)?;
    pins.insert(manifest_path.to_path_buf(), pin(manifest_path)?);
    for p in pins.values() {
        p.verify()?;
    }
    let topology = load(&s["topology"])?;
    ensure!(
        m["topology_sha256"] == s["topology"]["sha256"],
        "manifest topology identity mismatch"
    );
    let n = u(&m["n_atoms"])? as usize;
    ensure!(
        u(&topology["n_atoms"])? as usize == n,
        "topology atom count"
    );
    let residue_atoms: Vec<i32> =
        serde_json::from_value(topology["atom_analysis_residue_ids"].clone())?;
    let masses: Vec<f64> = serde_json::from_value(topology["masses"].clone())?;
    ensure!(
        residue_atoms.len() == n
            && masses.len() == n
            && masses.iter().all(|x| x.is_finite() && *x > 0.0),
        "atom identity/mass coverage"
    );
    let (residue_ids,atom_to_residue) = receptor_residue_axis(&residue_atoms);
    let nr = residue_ids.len();
    let mut residue_masses = vec![0.0; nr];
    for i in 0..n {
        if let Some(residue)=atom_to_residue[i] {residue_masses[residue] += masses[i];}
    }
    let exact_time_requested = ["treatment", "control"].map(|arm|
        !s["arms"][arm]["physical_time_journal"].is_null());
    // One recorded arm must never be paired with an invented clock for the other.
    ensure!(exact_time_requested == [false, false] || exact_time_requested == [true, true],
        "paired physical time journals incomplete");
    let dt = if exact_time_requested == [false, false] {
        let dt = m["dt_ps"].as_f64().context("historical dt missing")?;
        ensure!(dt.is_finite() && dt > 0.0, "invalid historical dt");
        Some(dt)
    } else { None };
    let cadence = u(&m["cadence_steps"])?;
    ensure!(cadence > 0, "zero cadence");
    let requested_steps = (1..=u(&m["frame_count"])?).map(|i|
        i.checked_mul(cadence).context("physical time step overflow"))
        .collect::<Result<BTreeSet<_>>>()?;
    let mut arm_time_indices = BTreeMap::new();
    if exact_time_requested == [true, true] {
        for arm in ["treatment", "control"] {
            let authority: physical_time::PhysicalTimeJournalAuthority =
                serde_json::from_value(s["arms"][arm]["physical_time_journal"].clone())?;
            ensure!(authority.source_stream_id == manifest_stream_id(&m)? as u64, "native time authority differs from manifest physical stream");
            arm_time_indices.insert(arm, physical_time::load_physical_time_endpoints(&authority, &requested_steps)?);
        }
    }
    let mut frames = Vec::new();
    let mut windows: Vec<Window> = Vec::new();
    for (i, l) in BufReader::new(File::open(source(&m["frames"])?.path)?)
        .lines()
        .enumerate()
    {
        let f: Value = serde_json::from_str(&l?)?;
        let step = u(&f["step"])?;
        let phase = text(&f["phase"])?.to_string();
        ensure!(step == (i as u64 + 1) * cadence, "measured step chronology mismatch");
        let mut arm_physical_time = BTreeMap::new();
        let time = if let Some(dt) = dt {
            let time = f["time_ps"].as_f64().context("historical time missing")?;
            ensure!((time - step as f64 * dt).abs() < 1e-10, "historical measured chronology mismatch");
            Some(time)
        } else {
            for arm in ["treatment", "control"] {
                arm_physical_time.insert(arm.to_string(), arm_time_indices[arm].get(&step)
                    .context("missing exact physical time endpoint")?.clone());
            }
            let t = arm_physical_time["treatment"].native_record.cumulative_simulated_time_ps;
            let c = arm_physical_time["control"].native_record.cumulative_simulated_time_ps;
            // Shared step correspondence does not imply simultaneous physical time.
            (t == c).then_some(t)
        };
        if windows.last().is_none_or(|w| w.phase != phase) {
            windows.push(Window {
                window_id: windows.len(),
                phase: phase.clone(),
                first_step: step,
                last_step: step,
                frames: 0,
            });
        }
        let w = windows.last_mut().unwrap();
        w.last_step = step;
        w.frames += 1;
        frames.push(Frame {
            ordinal: i,
            step,
            time_ps: time,
            arm_physical_time,
            phase,
            window: w.window_id,
        });
    }
    ensure!(
        frames.len() == u(&m["frame_count"])? as usize && !frames.is_empty(),
        "frame count mismatch"
    );
    let common = text(&s["common_reference_coordinate_payload_sha256"])?;
    let mut transforms: Vec<Vec<Transform>> = Vec::new();
    let mut audits = Vec::new();
    let mut trajectories = Vec::new();
    for arm in ["treatment", "control"] {
        let a = &s["arms"][arm];
        let recipe = load(&a["transform_recipe"])?;
        ensure!(u(&recipe["stream_id"])? == manifest_stream_id(&m)? as u64,
            "coordinate transform belongs to another physical stream");
        ensure!(
            recipe["reference_coordinate_payload_sha256"] == common,
            "different common body reference"
        );
        let ts: Vec<Transform> = BufReader::new(File::open(source(&a["transforms"])?.path)?)
            .lines()
            .map(|l| Ok(serde_json::from_str(&l?)?))
            .collect::<Result<_>>()?;
        ensure!(ts.len() == frames.len(), "transform count");
        let audit = load(&a["audit"])?;
        ensure!(u(&audit["stream_id"])? == manifest_stream_id(&m)? as u64,
            "coordinate audit belongs to another physical stream");
        ensure!(
            audit["frames"].as_array().context("audit frames")?.len() == frames.len(),
            "audit count"
        );
        let p = source(&a["trajectory"])?;
        ensure!(
            fs::metadata(&p.path)?.len() == 24 + frames.len() as u64 * n as u64 * 12,
            "trajectory length"
        );
        let mut r = BufReader::new(File::open(&p.path)?);
        let mut header = [0; 24];
        r.read_exact(&mut header)?;
        ensure!(
            &header[..8] == b"PRISM4D\0"
                && u32::from_le_bytes(header[8..12].try_into()?) == 1
                && u32::from_le_bytes(header[12..16].try_into()?) == n as u32
                && u32::from_le_bytes(header[16..20].try_into()?) == cadence as u32,
            "trajectory ABI mismatch"
        );
        let header_dt = f32::from_le_bytes(header[20..24].try_into()?) as f64;
        ensure!(header_dt.is_finite() && header_dt > 0.0, "invalid trajectory header dt");
        if let Some(dt) = dt {
            ensure!(header_dt == dt, "historical trajectory dt mismatch");
        }
        // V1 stores one nominal header dt. In exact mode, only the native
        // per-physical-step journal determines elapsed time; the audited
        // frame's exact native clock must agree with that same endpoint.
        for (i, frame) in frames.iter().enumerate() {
            if let Some(native) = frame.arm_physical_time.get(arm) {
                ensure!(audit["frames"][i]["exact_cumulative_time_fs"].as_f64()
                    == Some(native.native_record.cumulative_simulated_time_ps * 1000.0),
                    "coordinate audit lacks/mismatches exact native journal time for {arm} frame {i}");
            }
        }
        for (i, t) in ts.iter().enumerate() {
            ensure!(
                t.frame_ordinal == i
                    && t.step == frames[i].step
                    && t.reference_coordinate_payload_sha256 == common,
                "transform chronology/reference mismatch"
            );
            validate_rotation(t.rotation)?;
            ensure!(
                t.translation_angstrom.iter().all(|x| x.is_finite()),
                "nonfinite translation"
            );
            ensure!(
                audit["frames"][i]["step"] == t.step
                    && audit["frames"][i]["frame_hash64"] == t.coordinate_frame_hash64,
                "transform/audit frame mismatch"
            );
        }
        transforms.push(ts);
        audits.push(audit);
        trajectories.push(r);
    }
    let forces = [
        force_frames(&m, "treatment", n)?,
        force_frames(&m, "control", n)?,
    ];
    for arm in &forces {
        for step in arm.keys() {
            ensure!(
                frames.binary_search_by_key(step, |f| f.step).is_ok(),
                "force outside coordinate chronology"
            );
        }
    }
    fs::create_dir_all(out)?;
    let effective_input_manifest=if native_work_sources_resolved {
        let resolved_path=out.join("native_control_work_resolved_input_manifest.json");
        write_json(&resolved_path,&m)?;
        let resolved=pin(&resolved_path)?;
        pins.insert(resolved.path.clone(),resolved.clone());
        resolved
    } else {original_input_manifest};
    write_json(&out.join("protocol_windows.json"), &windows)?;
    write_json(
        &out.join("measurement_axes.json"),
        &json!({"schema_kind":"prism.mechanistic_axes.v1","topology":s["topology"],"atom_count":n,"atom_analysis_residue_ids":residue_atoms,"atom_receptor_residue_axis_indices":atom_to_residue,"atom_membership_contract":"original signed topology analysis IDs; negative IDs have no receptor residue membership; every atom vector and moment remains present","analysis_residue_ids":residue_ids,"atom_names":topology["atom_names"],"residue_names":topology["residue_names"],"region_memberships":m["regions"],"contact_pair_index":s["pair_index"],"bonded_term_index":s["bonded_index"],"residue_position_reduction":"mass_weighted_center_of_mass; full atom vectors also retained","residue_force_reduction":"sum of recorded total AMBER atomic force vectors; full atom vectors also retained","frame_basis":"pinned common treatment reference; row-vector rotation; forces never translated","force_identity":"TOTAL_AMBER_FORCE_BUFFER","covariance_definition":"population temporal covariance in each protocol window; not confidence interval or independent-sample uncertainty","tensor_entries":["xx","xy","xz","yy","yz","zz"]}),
    )?;
    let mut binary = BufWriter::new(File::create(out.join("aligned_vectors.f64le"))?);
    let mut timeline = BufWriter::new(File::create(out.join("vector_frames.jsonl"))?);
    let mut offset = 0u64;
    let mut moments: BTreeMap<(usize, String), Vec<VectorMoments>> = BTreeMap::new();
    let mut raw = vec![0u8; n * 12];
    for f in &frames {
        let mut positions = Vec::new();
        for arm in 0..2 {
            trajectories[arm].read_exact(&mut raw)?;
            ensure!(
                format!("{:x}", Sha256::digest(&raw))
                    == transforms[arm][f.ordinal].coordinate_payload_sha256,
                "coordinate payload hash mismatch"
            );
            let p = raw_vectors(&raw)?
                .into_iter()
                .map(|v| position(v, &transforms[arm][f.ordinal]))
                .collect::<Vec<_>>();
            positions.push(p);
        }
        let delta: Vec<[f64; 3]> = positions[0]
            .iter()
            .zip(&positions[1])
            .map(|(a, b)| std::array::from_fn(|i| a[i] - b[i]))
            .collect();
        let mut values = vec![
            ("treatment_position", positions.remove(0)),
            ("control_position", positions.remove(0)),
            ("delta_position", delta),
        ];
        let mut aligned_force: Vec<Option<Vec<[f64; 3]>>> = Vec::new();
        for arm in 0..2 {
            aligned_force.push(forces[arm].get(&f.step).map(|x| {
                x.iter()
                    .map(|v| rotate(*v, transforms[arm][f.ordinal].rotation))
                    .collect()
            }));
        }
        if let (Some(a), Some(b)) = (&aligned_force[0], &aligned_force[1]) {
            values.push((
                "delta_force",
                a.iter()
                    .zip(b)
                    .map(|(a, b)| std::array::from_fn(|i| a[i] - b[i]))
                    .collect(),
            ));
        }
        for (arm, name) in ["treatment_force", "control_force"].into_iter().enumerate() {
            if let Some(x) = aligned_force[arm].take() {
                values.push((name, x));
            }
        }
        let mut channels = serde_json::Map::new();
        for (name, x) in values {
            let is_force = name.ends_with("force");
            let unit = if is_force {
                "kcal mol^-1 Angstrom^-1"
            } else {
                "Angstrom"
            };
            channels.insert(
                name.into(),
                json!({"byte_offset":offset,"byte_length":n*24,"unit":unit}),
            );
            binary_vectors(&mut binary, &x)?;
            offset += n as u64 * 24;
            let stats = moments
                .entry((f.window, name.into()))
                .or_insert_with(|| vec![VectorMoments::default(); n + nr]);
            let mut residues = vec![[0.0; 3]; nr];
            for i in 0..n {
                stats[i].add(x[i])?;
                if let Some(residue)=atom_to_residue[i] {
                    for a in 0..3 {
                        residues[residue][a] += x[i][a] * if is_force { 1.0 } else { masses[i] };
                    }
                }
            }
            for i in 0..nr {
                if !is_force {
                    for a in 0..3 {
                        residues[i][a] /= residue_masses[i];
                    }
                }
                stats[n + i].add(residues[i])?;
            }
            // Preserve the actual frame-resolved residue vectors, in addition
            // to full atom bytes and protocol-wide moments. Event consumers
            // must not reconstruct a time series from an averaged tensor.
            channels.get_mut(name).unwrap()["analysis_residue_vectors"] = json!(residues);
        }
        line(
            &mut timeline,
            &json!({"frame_ordinal":f.ordinal,"step":f.step,"physical_time_ps":f.time_ps,"arm_physical_time":f.arm_physical_time,
                "time_axis_scope":if f.arm_physical_time.is_empty(){"historical_fixed_dt_declared"}else{"exact_native_per_arm_step_sum"},
                "pair_correspondence":"same recorded MD step; physical times may differ; no interpolation",
                "phase":f.phase,"window_id":f.window,"channels":channels,"force_availability":{"treatment":forces[0].contains_key(&f.step),"control":forces[1].contains_key(&f.step)},"missing_force_reason":"not recorded at this step; unrelated to cavity clipping or TSH eligibility"}),
        )?;
    }
    binary.flush()?;
    timeline.flush()?;
    let mut moment_out = BufWriter::new(File::create(out.join("vector_moments.jsonl"))?);
    for ((w, channel), stats) in &moments {
        for (i, x) in stats.iter().enumerate() {
            let mut v = x.value();
            let o = v.as_object_mut().unwrap();
            o.insert("window_id".into(), json!(w));
            o.insert("phase".into(), json!(windows[*w].phase));
            o.insert("channel".into(), json!(channel));
            o.insert(
                "axis_kind".into(),
                json!(if i < n { "atom" } else { "analysis_residue" }),
            );
            o.insert(
                "axis_id".into(),
                json!(if i < n { i as u32 } else { residue_ids[i - n] }),
            );
            line(&mut moment_out, &v)?;
        }
    }
    moment_out.flush()?;
    drop(moments);
    let contact = evaluate_contacts(s, &frames, &windows, &audits, out)?;
    let bonded = evaluate_bonded(s, &frames, &windows, &transforms, out)?;
    let mut kcc = serde_json::Map::new();
    for arm in ["treatment", "control"] {
        kcc.insert(
            arm.into(),
            index_kcc(&s["arms"][arm], arm, &m, &frames, nr, out)?,
        );
    }
    let names = [
        "protocol_windows.json",
        "measurement_axes.json",
        "aligned_vectors.f64le",
        "vector_frames.jsonl",
        "vector_moments.jsonl",
        "contact_moments.jsonl",
        "bonded_moments.jsonl",
        "torsion_identity.json",
        "treatment_kcc_endpoint_index.jsonl",
        "control_kcc_endpoint_index.jsonl",
    ];
    let mut evaluated = BTreeMap::new();
    for name in names {
        evaluated.insert(name.into(), pin(&out.join(name))?);
    }
    for arm in ["treatment","control"] {
        let supplement=&kcc[arm]["winning_pair_physical_times"]["artifact"];
        if !supplement.is_null() {
            evaluated.insert(format!("{arm}_kcc_winning_pair_physical_times.jsonl.zst"),source(supplement)?);
        }
    }
    let isolated_asc = evaluate_isolated_asc_measurements(&m, &residue_atoms, out)?;
    for arm in ["treatment", "control"] {
        if isolated_asc[arm]["available"] == true {
            let name = format!("{arm}_isolated_asc_measurements.jsonl.zst");
            evaluated.insert(name.clone(), pin(&out.join(name))?);
        }
    }
    evaluated.insert("INTERNAL_QC/NATIVE_CONTROL_WORK_ADMISSION.json".into(),
        source(&native_control_work.summary["_admission_qc"])?);
    for arm in ["treatment","control"] {
        for kind in ["ca_coordinate_work","nma_force_work","conservative_constraint_impulse"] {
            if native_control_work.summary[arm][kind]["available"]==true {
                let p=source(&native_control_work.summary[arm][kind]["evaluated_source"])?;
                evaluated.insert(format!("{arm}_{kind}_measurements.jsonl.zst"),p);
            }
        }
    }
    let events = evaluate_native_event_measurements(&m, &frames, &residue_ids, &isolated_asc, &native_control_work, out)?;
    if events["available"] == true {
        for arm in ["treatment", "control"] {
            let name = format!("{arm}_native_event_measurements.jsonl.zst");
            evaluated.insert(name.clone(), pin(&out.join(name))?);
        }
    }
    let artifact = MechanisticMeasurementArtifact {
        schema_kind: "prism.native_fiber_mechanistic_measurements.v1".into(),
        manifest_sha256: effective_input_manifest.sha256,
        topology_sha256: text(&m["topology_sha256"])?.into(),
        arm_run_ids: BTreeMap::from([
            ("treatment".into(), text(&m["treatment_run_id"])?.into()),
            ("control".into(), text(&m["control_run_id"])?.into()),
        ]),
        stream_id: manifest_stream_id(&m)?,
        frame_steps: frames.iter().map(|f| f.step).collect(),
        frame_phases: frames.iter().map(|f| f.phase.clone()).collect(),
        atom_count: n,
        analysis_residue_ids: residue_ids,
        source_pins: pins.into_values().collect(),
        evaluated_artifacts: evaluated,
        scope: json!({"historical_comparison_producer":m["historical_comparison_producer"],"historical_producer_scope":"historical identity retained as recorded; current files verified only against active source_files and mechanistic_sources pins","TSH_admission_used":false,"cavity_clipping_filters_measurements":false,"coordinates":"all recorded atoms in all saved frames","forces":"all recorded total AMBER force vectors; absent steps remain absent","contacts":contact,"bonded":bonded,"KCC":kcc,"KCC_role":"exact endpoint association and byte-support verification; cumulative estimator descriptors are not interval-local causal effects","geometry_grids":"pinned existing atom-attributed geometry; historical paired voxel masks are not in common basis and not re-evaluated here","inference":"descriptive paired measurement; no new causal score, ranking, effective sample size or uncertainty estimate","complete_chemical_state_fiber_bundle_instantiated":false,"bundle_reason":"measurement artifact can attach to existing valid exact-lineage fibers; periodic/alchemical/Markov contracts are unchanged; no such bundle supplied"}),
    };
    let mut artifact = artifact;
    artifact.scope["physical_time"] = json!({
        "authority":if exact_time_requested == [true,true]{"exact_native_per_arm_step_sum"}else{"historical_fixed_dt_declared"},
        "per_arm_journals":{"treatment":s["arms"]["treatment"]["physical_time_journal"],"control":s["arms"]["control"]["physical_time_journal"]},
        "shared_time_null_when_different":true,"step_matched_comparison_is_time_matched":false,
        "historical_input_times_reestimated":false});
    artifact.scope["native_event_measurements"] = events;
    artifact.scope["isolated_ASC_measurements"] = isolated_asc;
    artifact.scope["native_control_work_measurements"] = native_control_work.summary;
    // Re-hash all inputs after evaluation to detect concurrent mutation.
    artifact.verify()?;
    write_json(
        &out.join("mechanistic_measurement_artifact.json"),
        &artifact,
    )?;
    let summary = json!({"schema_kind":"prism.native_mechanistic_replay_receipt.v1","status":"IMPLEMENTATION_PARTIAL","measurement_evaluation_completed":true,"artifact":pin(&out.join("mechanistic_measurement_artifact.json"))?,"canonical_artifact_sha256":canonical_sha256(&artifact)?,"frames_per_arm":frames.len(),"atoms":n,"residues":nr,"force_frames_per_arm":[forces[0].len(),forces[1].len()],"vector_bytes":offset,"protocol_windows":windows.len(),"mechanistic_inference_completed":false,"MD_executed":false,"TSH_replaced":false,"scope":artifact.scope});
    write_json(&out.join("EXECUTION_RECEIPT.json"), &summary)?;
    Ok(summary)
}

fn evaluate_contacts(
    s: &Value,
    frames: &[Frame],
    windows: &[Window],
    audits: &[Value],
    out: &Path,
) -> Result<Value> {
    let index = load(&s["pair_index"])?;
    ensure!(
        index["topology"] == s["topology"] && index["sites"] == s["sites"],
        "pair index source mismatch"
    );
    let pairs = index["pairs"].as_array().context("pair axis")?;
    for (i, p) in pairs.iter().enumerate() {
        ensure!(u(&p["pair_index"])? as usize == i, "pair axis order");
    }
    let channels = [
        "treatment_ca_distances_a",
        "control_ca_distances_a",
        "delta_ca_distances_a",
        "treatment_min_heavy_atom_distances_a",
        "control_min_heavy_atom_distances_a",
        "delta_min_heavy_atom_distances_a",
    ];
    let mut stats =
        vec![vec![vec![ScalarMoments::default(); pairs.len()]; channels.len()]; windows.len()];
    let mut count = 0;
    let r = SerializedFileReader::new(File::open(source(&s["pair_frames"])?.path)?)?;
    for (i, row) in r.get_row_iter(None)?.enumerate() {
        let row = row?;
        frame_check(&row, frames, i)?;
        for (a, name) in ["treatment_frame_hash64", "control_frame_hash64"]
            .into_iter()
            .enumerate()
        {
            ensure!(
                fsx(&row, name)? == text(&audits[a]["frames"][i]["frame_hash64"])?,
                "contact coordinate identity"
            );
        }
        let values = channels
            .iter()
            .map(|name| floats(&row, name))
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            values.iter().all(|x| x.len() == pairs.len()),
            "contact pair dimension"
        );
        for (a, b, d) in [(0, 1, 2), (3, 4, 5)] {
            for j in 0..pairs.len() {
                match (values[a][j], values[b][j], values[d][j]) {
                    (Some(x), Some(y), Some(z)) => {
                        ensure!((x - y - z).abs() < 1e-10, "contact delta disagrees")
                    }
                    (None, _, None) | (_, None, None) => {}
                    _ => bail!("contact missing-value disagreement"),
                }
            }
        }
        for c in 0..channels.len() {
            for j in 0..pairs.len() {
                stats[frames[i].window][c][j].add(values[c][j], false)?;
            }
        }
        count += 1;
    }
    ensure!(count == frames.len(), "contact chronology incomplete");
    let mut w = BufWriter::new(File::create(out.join("contact_moments.jsonl"))?);
    for wi in 0..windows.len() {
        for c in 0..channels.len() {
            for (j, pair) in pairs.iter().enumerate() {
                let mut v = stats[wi][c][j].value(false);
                let o = v.as_object_mut().unwrap();
                o.insert("window_id".into(), json!(wi));
                o.insert("phase".into(), json!(windows[wi].phase));
                o.insert("channel".into(), json!(channels[c]));
                o.insert("pair_index".into(), json!(j));
                o.insert(
                    "analysis_residue_ids".into(),
                    pair["analysis_residue_ids"].clone(),
                );
                o.insert(
                    "source_region_ids".into(),
                    pair["source_region_ids"].clone(),
                );
                line(&mut w, &v)?;
            }
        }
    }
    w.flush()?;
    Ok(
        json!({"frames":count,"pair_count":pairs.len(),"coverage":"exact historical lining-pair axis; unevaluated residue pairs are unobserved, not zero","time_series":s["pair_frames"],"nearest_atom_correspondence":"preserved in pinned source Parquet; distance moments do not replace identities"}),
    )
}
fn torsion_groups(terms: &[Value]) -> Result<BTreeMap<[u64; 4], Vec<usize>>> {
    let mut groups: BTreeMap<[u64; 4], Vec<usize>> = BTreeMap::new();
    for (i, t) in terms.iter().enumerate() {
        let q = [u(&t["i"])?, u(&t["j"])?, u(&t["k_idx"])?, u(&t["l"])?];
        groups.entry(q).or_default().push(i);
    }
    Ok(groups)
}
fn evaluate_bonded(
    s: &Value,
    frames: &[Frame],
    windows: &[Window],
    transforms: &[Vec<Transform>],
    out: &Path,
) -> Result<Value> {
    let index = load(&s["bonded_index"])?;
    let raw_topology = load(&s["topology"])?;
    ensure!(
        index["topology_sha256"] == s["topology"]["sha256"],
        "bonded topology mismatch"
    );
    let channels = [
        ("delta_bond_lengths_angstrom", "bonds", false),
        ("delta_angles_radians", "angles", false),
        ("delta_signed_dihedrals_radians", "dihedrals", true),
    ];
    let mut stats: Vec<Vec<Vec<ScalarMoments>>> = windows
        .iter()
        .map(|_| {
            channels
                .iter()
                .map(|(_, key, _)| {
                    vec![ScalarMoments::default(); index[*key].as_array().unwrap().len()]
                })
                .collect()
        })
        .collect();
    let terms = index["dihedrals"].as_array().context("dihedral terms")?;
    let groups = torsion_groups(terms)?;
    write_json(
        &out.join("torsion_identity.json"),
        &json!({"schema_kind":"prism.recorded_torsion_identity.v1","source":s["bonded_index"],"unique_ordered_atom_quadruples":groups.len(),"fourier_terms":terms.len(),"groups":groups.iter().map(|(q,t)|json!({"atom_indices":q,"term_indices":t,"representative_term_index":t[0]})).collect::<Vec<_>>(),"semantics":"Fourier terms on the same ordered quadruple share one geometric torsion, never independent evidence; each original term retained. Proper/improper flags retain source status; no electronic/quantum claim introduced."}),
    )?;
    let r = SerializedFileReader::new(File::open(source(&s["bonded_frames"])?.path)?)?;
    let mut count = 0;
    for (i, row) in r.get_row_iter(None)?.enumerate() {
        let row = row?;
        frame_check(&row, frames, i)?;
        for (a, name) in ["treatment_coordinate_sha256", "control_coordinate_sha256"]
            .into_iter()
            .enumerate()
        {
            ensure!(
                fsx(&row, name)? == transforms[a][i].coordinate_payload_sha256,
                "bonded coordinate identity mismatch"
            );
        }
        for (c, (name, key, circular)) in channels.iter().enumerate() {
            let values = floats(&row, name)?;
            ensure!(
                values.len() == index[*key].as_array().unwrap().len(),
                "bonded axis mismatch"
            );
            if *circular {
                for ids in groups.values() {
                    for j in &ids[1..] {
                        ensure!(
                            values[*j] == values[ids[0]],
                            "duplicate Fourier terms have different measured torsion"
                        );
                    }
                }
            }
            for (j, x) in values.into_iter().enumerate() {
                if *circular {
                    if let Some(x) = x {
                        ensure!(
                            x.abs() <= std::f64::consts::PI + 1e-12,
                            "signed torsion delta outside principal interval"
                        );
                    }
                }
                stats[frames[i].window][c][j].add(x, *circular)?;
            }
        }
        count += 1;
    }
    ensure!(count == frames.len(), "bonded chronology incomplete");
    let mut w = BufWriter::new(File::create(out.join("bonded_moments.jsonl"))?);
    for wi in 0..windows.len() {
        for (c, (name, key, circular)) in channels.iter().enumerate() {
            for (j, x) in stats[wi][c].iter().enumerate() {
                let mut v = x.value(*circular);
                let o = v.as_object_mut().unwrap();
                o.insert("window_id".into(), json!(wi));
                o.insert("phase".into(), json!(windows[wi].phase));
                o.insert("channel".into(), json!(name));
                o.insert("term_index".into(), json!(j));
                o.insert("term".into(), index[*key][j].clone());
                if *circular {
                    let declared = raw_topology["dihedrals"][j].get("improper").is_some();
                    o.insert("improper_declared".into(), json!(declared));
                    o.insert("proper_improper_classification_status".into(),json!(if declared {"recorded_in_topology"}else{"unavailable; recorded index improper flag may be a deserializer default"}));
                }
                line(&mut w, &v)?;
            }
        }
    }
    w.flush()?;
    Ok(
        json!({"frames":count,"bond_terms":index["bonds"].as_array().unwrap().len(),"angle_terms":index["angles"].as_array().unwrap().len(),"torsion_terms":terms.len(),"unique_ordered_torsions":groups.len(),"time_series":s["bonded_frames"],"circular_statistics":"signed torsion differences summarized through cos/sin; original signed per-frame differences preserved","fourier_duplicate_support_inflation":false}),
    )
}
fn index_kcc(
    a: &Value,
    arm: &str,
    m: &Value,
    frames: &[Frame],
    nr: usize,
    out: &Path,
) -> Result<Value> {
    let chronology = source(&a["kcc_chronology"])?;
    let support = source(&a["kcc_support"])?;
    let physical_axis = if a["physical_time_journal"].is_null() { None } else {
        let authority:physical_time::PhysicalTimeJournalAuthority=serde_json::from_value(a["physical_time_journal"].clone())?;
        ensure!(authority.source_stream_id==manifest_stream_id(m)? as u64,"KCC physical time stream mismatch");
        Some(physical_time::VerifiedPhysicalTimeAxis::load(&authority)?)
    };
    let physical_name=format!("{arm}_kcc_winning_pair_physical_times.jsonl.zst");
    let mut physical_writer=if physical_axis.is_some() {Some(zstd::stream::write::Encoder::new(
        BufWriter::new(File::create(out.join(&physical_name))?),1)?)}else{None};
    let mut total_physical_pairs=0u64;
    let mut pack = File::open(&support.path)?;
    let pack_len = pack.metadata()?.len();
    let mut input = BufReader::new(File::open(&chronology.path)?);
    let mut w = BufWriter::new(File::create(
        out.join(format!("{arm}_kcc_endpoint_index.jsonl")),
    )?);
    let mut raw = Vec::new();
    let mut offset = 0u64;
    let mut count = 0;
    let mut previous = None;
    let mut origins = BTreeMap::<String, u64>::new();
    let mut support_bytes = 0;
    loop {
        raw.clear();
        let bytes = input.read_until(b'\n', &mut raw)?;
        if bytes == 0 {
            break;
        }
        let v: Value = serde_json::from_slice(&raw)?;
        if v["record_kind"] == "measurement" {
            let step = u(&v["measurement_step"])?;
            ensure!(previous.is_none_or(|p| p < step), "KCC endpoint order");
            previous = Some(step);
            let i = frames
                .binary_search_by_key(&step, |f| f.step)
                .ok()
                .context("KCC endpoint lacks exact coordinates")?;
            ensure!(u(&v["stream_id"])? == manifest_stream_id(m)? as u64, "unexpected KCC stream");
            let k = &v["kcc_per_residue"];
            ensure!(
                u(&k["n_residues"])? as usize == nr,
                "KCC residue axis mismatch"
            );
            let provenance: Vec<crate::causome::KccLagProvenance> =
                serde_json::from_value(k["lag_provenance"].clone())?;
            ensure!(provenance.len() == nr, "KCC provenance coverage");
            for name in [
                "active_causal",
                "burst_motion",
                "causal_lag",
                "direction_score",
                "lag_corr_peak",
                "local_cov",
                "motion_efficiency",
                "net_dx",
                "net_dy",
                "net_dz",
                "phase_shift",
                "sum_m",
                "temporal_corr",
            ] {
                ensure!(
                    k[name].as_array().context("missing KCC array")?.len() == nr,
                    "KCC field dimension {name}"
                );
            }
            let source_stream_id = manifest_stream_id(m)? as u32;
            for (j, p) in provenance.iter().enumerate() {
                ensure!(
                    p.source_stream_id.is_none_or(|id| id == source_stream_id),
                    "KCC provenance stream mismatch"
                );
                ensure!(
                    !p.estimator_origin.is_empty()
                        && !p.sign_convention.is_empty()
                        && !p.lag_unit.is_empty(),
                    "KCC lag semantic origin absent"
                );
                *origins.entry(p.estimator_origin.clone()).or_default() += 1;
                if let (Some(bits), Some(value)) =
                    (p.causal_lag_f32_bits, k["causal_lag"][j].as_f64())
                {
                    ensure!(
                        f32::from_bits(bits) as f64 == value,
                        "KCC lag bits mismatch"
                    );
                }
            }
            let r = &v["lag_support_ref"];
            ensure!(
                text(&r["path"])? == support.path.to_str().context("UTF8 support path")?
                    && u(&r["measurement_step"])? == step,
                "KCC archive identity mismatch"
            );
            ensure!(
                u(&r["record_metadata"]["measurement_step"])? == step
                    && u(&r["record_metadata"]["stream_id"])? == manifest_stream_id(m)? as u64,
                "KCC support metadata endpoint/stream mismatch"
            );
            let start = u(&r["byte_offset"])?;
            let length = u(&r["byte_length"])?;
            ensure!(
                start.checked_add(length).is_some_and(|end| end <= pack_len),
                "KCC support out of bounds"
            );
            ensure!(
                r["record_metadata"]["run_id"] == m[format!("{arm}_run_id")]
                    && u(&r["record_metadata"]["row_count"])? as usize == nr,
                "KCC run/residue support mismatch"
            );
            pack.seek(SeekFrom::Start(start))?;
            let mut encoded = vec![0; usize::try_from(length)?];
            pack.read_exact(&mut encoded)?;
            ensure!(
                format!("{:x}", Sha256::digest(&encoded)) == text(&r["encoded_record_sha256"])?,
                "KCC support bytes mismatch"
            );
            support_bytes += length;
            let physical_times = if let Some(axis)=&physical_axis {
                let archive=kcc_time::decode_support_record(&encoded,&r["record_metadata"])?;
                let value=kcc_time::winning_pair_times(&archive,axis,text(&m[format!("{arm}_run_id")])?,r,nr)?;
                total_physical_pairs+=u(&value["winning_pair_count"])?;
                line(physical_writer.as_mut().unwrap(),&value)?;
                Some(json!({"artifact":physical_name,"line_ordinal":count,"record_sha256":value["record_sha256"],
                    "measurement_step":step,"winning_pair_count":value["winning_pair_count"]}))
            }else{None};
            line(
                &mut w,
                &json!({"frame_ordinal":i,"step":step,"physical_time_ps":frames[i].time_for_arm(arm)?,"physical_time_authority":frames[i].arm_physical_time.get(arm),"phase":frames[i].phase,"window_id":frames[i].window,"arm":arm,"run_id":m[format!("{arm}_run_id")],"source_chronology":chronology,"source_line_byte_offset":offset,"source_line_byte_length":bytes,"source_line_sha256":format!("{:x}",Sha256::digest(&raw)),"lag_support_ref":r,"winning_pair_physical_times":physical_times,"native_kcc_per_residue":k,"native_leads":v["leads"],"descriptor_scope":"cumulative native measurement ending at this exact coordinate step; not an interval-local estimate or demonstrated interresidue causal influence","vector_frame_join":"vector_frames.jsonl frame_ordinal and step","archive_verification":"encoded byte range and sha256 verified; archived event payload retained, not re-estimated"}),
            )?;
            count += 1;
        }
        offset += bytes as u64;
    }
    w.flush()?;
    ensure!(count > 0, "KCC measurement chronology empty");
    if let Some(writer)=physical_writer {writer.finish()?.flush()?;}
    let physical_receipt=if physical_axis.is_some(){json!({"artifact":pin(&out.join(&physical_name))?,
        "measurement_endpoints":count,"winning_pairs":total_physical_pairs,"native_estimator_reselected":false})}
        else{json!({"available":false,"reason":"native physical time journal absent; no historical duration reconstruction"})};
    Ok(
        json!({"endpoint_records":count,"winning_pair_physical_times":physical_receipt,"estimator_origin_residue_observations":origins,"verified_encoded_support_bytes":support_bytes,"source":chronology,"support":support,"consumption":"native lag values, descriptor dimensions, exact coordinate endpoints, typed estimator provenance and support byte correspondence checked; no new lag inference"}),
    )
}

/// Resolve the five physical protocol intervals without rewriting the native
/// four-phase lattice ontology (whose cooling phase includes cold return).
fn measured_protocol_phase(protocol: &Value, step: u64) -> Result<&'static str> {
    let bounds = [u(&protocol["cold_hold_end"])?, u(&protocol["ramp_end"])?,
        u(&protocol["warm_hold_end"])?, u(&protocol["ramp_down_end"])?,
        u(&protocol["total_steps"])?.checked_add(1).context("protocol end overflow")?];
    ensure!(bounds.windows(2).all(|w|w[0]<=w[1]), "unordered physical phase boundaries");
    let i = bounds.iter().position(|end| step < *end).context("event exceeds physical protocol")?;
    // Match the owning native director's explicit warm-tail override. Legacy
    // protocol exports without this optional field retain their recorded phases.
    if i == 4 && protocol["steering_flags"].as_u64().unwrap_or(0) & 0x10 != 0 {
        return Ok("warm_hold");
    }
    Ok(["cold_hold", "heating", "warm_hold", "cooling", "cold_return"][i])
}

/// Consume authenticated native observations at exact coordinate endpoints.
/// No spatial/score/TSH acceptance gate changes this event population. The
/// existing full atom vectors, contact/torsion time series and raw Ghost bytes
/// remain authoritative; shared series are referenced without duplicating or
/// flattening them into a score.
/// Preserve the producer's component identity using its exact node partition.
/// This checks correspondence; it does not re-run or re-label lattice connectivity.
fn component_membership(lattice: &Value, nodes: &[crate::ghost_phase_lattice::GhostPhaseLatticeNode]) -> Result<Vec<u64>> {
    let mut membership = vec![None; nodes.len()];
    let mut ids = BTreeSet::new();
    for component in lattice["components"].as_array().context("lattice components missing")? {
        let id = u(&component["component_id"])?;
        ensure!(ids.insert(id), "duplicate lattice component identity");
        let rows = component["node_lineage"].as_array().context("component node lineage missing")?;
        ensure!(u(&component["n_nodes"])? == rows.len() as u64, "component population mismatch");
        for row in rows {
            let index = usize::try_from(u(&row["tile_index"])?)?;
            let node = nodes.get(index).context("component node outside decoded population")?;
            ensure!(u(&row["step_idx"])? == node.step_idx
                && u(&row["frame_idx"])? == node.frame_idx
                && u(&row["stream_id"])? == node.stream_id as u64
                && u(&row["site_id"])? == node.site_id as u64
                && u(&row["protocol_phase"])? == node.protocol_phase as u64,
                "component node identity differs from decoded observation");
            ensure!(membership[index].replace(id).is_none(), "observation belongs to multiple components");
        }
    }
    membership.into_iter().map(|id| id.context("observation absent from component partition")).collect()
}

fn evaluate_isolated_asc_measurements(m: &Value, atom_residues: &[i32], out: &Path) -> Result<Value> {
    let mut result = serde_json::Map::new();
    for arm in ["treatment", "control"] {
        let journal = &m["mechanistic_sources"]["arms"][arm]["isolated_asc_journal"];
        if journal.is_null() {
            result.insert(arm.into(), json!({"available":false,"reason":"isolated native journal was not supplied; total AMBER force is not a substitute"}));
            continue;
        }
        #[cfg(not(feature="gpu"))]
        bail!("isolated ASC native consumer requires gpu feature");
        #[cfg(feature="gpu")]
        {
            let schema = source(&journal["schema.json"])?;
            let dir = schema.path.parent().context("isolated ASC journal directory")?;
            for name in ["schema.json", "seal.json", "atom_rows.bin", "endpoints.jsonl"] {
                let p = source(&journal[name])?;
                ensure!(p.path == dir.join(name), "isolated ASC manifest mixes journal roots");
                p.verify()?;
            }
            let path = out.join(format!("{arm}_isolated_asc_measurements.jsonl.zst"));
            let mut value = crate::asc_device::measurement::export_isolated_asc_measurements(
                dir, text(&m[format!("{arm}_run_id")])?, manifest_stream_id(m)? as u32,
                text(&m["topology_sha256"])?, atom_residues, &path)?;
            value["evaluated_source"] = serde_json::to_value(pin(&path)?)?;
            result.insert(arm.into(), value);
        }
    }
    Ok(Value::Object(result))
}

fn evaluate_native_event_measurements(m: &Value, frames: &[Frame], residues: &[u32], isolated_asc: &Value, native_control_work: &control_work::NativeControlWork, out: &Path) -> Result<Value> {
    use crate::ghost_phase_lattice::{load_verified_ghost_segments_from_dir, StreamProtocolPhaseBinding,
        NODE_CAUSAL_LEAD_NONE, NODE_GEAR_ID_NONE, NODE_CCNS_PHASE_NONE};
    let s = &m["mechanistic_sources"];
    let present = ["treatment", "control"].map(|arm| !s["arms"][arm]["ghost_segment_manifest"].is_null());
    if present == [false, false] {
        return Ok(json!({"available":false,"reason":"native Ghost inputs not supplied; original full measurement population retained"}));
    }
    ensure!(present == [true,true], "paired native event sources incomplete");
    let timeline: Vec<Value> = BufReader::new(File::open(out.join("vector_frames.jsonl"))?).lines()
        .map(|l| Ok(serde_json::from_str(&l?)?)).collect::<Result<_>>()?;
    ensure!(timeline.len()==frames.len(), "event vector timeline incomplete");
    let pairs = load(&s["pair_index"])?;
    let pairs = pairs["pairs"].as_array().context("event pair axis")?;
    let mut summary = serde_json::Map::new();
    for arm in ["treatment", "control"] {
        let a = &s["arms"][arm];
        let asc_endpoints: BTreeMap<u64, &Value> = isolated_asc[arm]["endpoint_index"].as_array()
            .into_iter().flatten().map(|row| Ok((u(&row["completed_step"])?, row))).collect::<Result<_>>()?;
        for key in ["ghost_segment_manifest", "ghost_lineage", "ghost_phase_binding", "protocol_state"] {
            source(&a[key])?.verify()?;
        }
        let manifest_pin = source(&a["ghost_segment_manifest"])?;
        let root = manifest_pin.path.parent().and_then(Path::parent).context("Ghost root")?;
        let binding_pin = source(&a["ghost_phase_binding"])?;
        let filename = binding_pin.path.file_name().and_then(|p|p.to_str()).context("phase filename")?;
        let stream_id = manifest_stream_id(m)?;
        let suffix = format!("_stream{stream_id:02}_ghost_phase_binding.json");
        let stem = filename.strip_suffix(suffix.as_str()).context("phase binding filename differs from explicit physical stream")?;
        let binding: StreamProtocolPhaseBinding = serde_json::from_reader(File::open(&binding_pin.path)?)?;
        binding.verify()?;
        ensure!(binding.stream_id == stream_id, "phase binding stream differs from measurement lane");
        let protocol = load(&a["protocol_state"])?;
        ensure!(u(&protocol["stream_index"])? == stream_id as u64, "protocol stream differs from measurement lane");
        let decoded = load_verified_ghost_segments_from_dir(root, stem, u(&m["cadence_steps"])?)?;
        ensure!(decoded.equality_pass && decoded.overflow_conservation_pass
            && decoded.nodes.len()==decoded.record_lineage.len(), "native event conservation failed");
        let components = if a["native_lattice"].is_null() { None } else {
            source(&a["native_lattice"])?.verify()?;
            let lattice = load(&a["native_lattice"])?;
            ensure!(lattice["run_id"] == a["ghost_producer_run_id"]
                && lattice["ghost_record_byte_lineage_artifact"]["sha256"] == a["ghost_lineage"]["sha256"]
                && lattice["ghost_lattice_accepted_edge_artifact"]["sha256"] == a["ghost_accepted_edges"]["sha256"]
                && u(&lattice["ghost_record_byte_lineage_artifact"]["row_count"])? == decoded.decoded_records,
                "lattice component source differs from authenticated observations");
            Some(component_membership(&lattice, &decoded.nodes)?)
        };
        let endpoint_index_pin = pin(&out.join(format!("{arm}_kcc_endpoint_index.jsonl")))?;
        let mut kcc_lines = BufReader::new(File::open(&endpoint_index_pin.path)?).lines();
        let mut kcc: Option<Value> = None;
        let mut segment: Option<(String, BufReader<File>)> = None;
        let mut seen = BTreeSet::new();
        let mut phase_counts = BTreeMap::<String,u64>::new();
        let mut adjudication_counts = BTreeMap::<u8,u64>::new();
        let mut tainted_records = 0u64;
        let mut contact_links = 0u64;
        let mut member_links = 0u64;
        let mut last_step = 0;
        let mut w = zstd::stream::write::Encoder::new(BufWriter::new(File::create(
            out.join(format!("{arm}_native_event_measurements.jsonl.zst")))?),1)?;
        let selected_records = decoded.record_lineage.iter().filter(|l| l.stream_id == stream_id).count();
        let mut selected_components = BTreeSet::new();
        for (node, lineage) in decoded.nodes.iter().zip(&decoded.record_lineage)
            .filter(|(_, lineage)| lineage.stream_id == stream_id) {
            ensure!(lineage.run_id == text(&a["ghost_producer_run_id"])?
                && lineage.run_id == text(&m[format!("{arm}_run_id")])?
                && lineage.topology_sha256 == text(&m["topology_sha256"])?
                && lineage.stream_id==stream_id && node.stream_id==stream_id
                && lineage.segment_manifest_sha256 == manifest_pin.sha256 && lineage.step_idx==node.step_idx
                && lineage.site_id==node.site_id, "event source identity differs from measurements");
            if let Some(ids) = &components {
                selected_components.insert(ids[lineage.global_lattice_index as usize]);
            }
            ensure!(node.step_idx>=last_step, "nonmonotonic native observation sequence");
            last_step=node.step_idx;
            if segment.as_ref().is_none_or(|(name,_)|name!=&lineage.segment_file) {
                segment=Some((lineage.segment_file.clone(),BufReader::new(File::open(root.join(&lineage.segment_file))?)));
            }
            let reader=&mut segment.as_mut().unwrap().1;
            reader.seek(SeekFrom::Start(lineage.payload_byte_offset))?;
            let mut record=vec![0;lineage.byte_length as usize];
            reader.read_exact(&mut record)?;
            ensure!(format!("{:x}",Sha256::digest(&record))==lineage.record_sha256,"event metadata raw record changed");
            use crate::ghost_tile::GhostTileFrame;
            let chain=record[std::mem::offset_of!(GhostTileFrame,chain_id)];
            let adjudication=record[std::mem::offset_of!(GhostTileFrame,adjudication_code)];
            let flag_offset=std::mem::offset_of!(GhostTileFrame,telemetry_flags);
            let flags=u16::from_le_bytes(record[flag_offset..flag_offset+2].try_into()?);
            *adjudication_counts.entry(adjudication).or_default()+=1;
            tainted_records+=u64::from(flags & 1 != 0);
            ensure!(seen.insert((lineage.stream_id,lineage.step_idx,lineage.site_id,lineage.record_sha256.clone())), "duplicate native observation identity");
            let i=frames.binary_search_by_key(&node.step_idx,|f|f.step).ok().context("native event lacks exact coordinate endpoint")?;
            let phase=measured_protocol_phase(&protocol,node.step_idx)?;
            ensure!(frames[i].phase==phase,"physical phase differs from measurement chronology");
            while kcc.as_ref().and_then(|v|v["step"].as_u64()).is_none_or(|step|step<node.step_idx) {
                kcc=Some(serde_json::from_str(&kcc_lines.next().context("native event lacks KCC endpoint")??)?);
            }
            let k=kcc.as_ref().unwrap();
            ensure!(u(&k["step"])?==node.step_idx && u(&k["frame_ordinal"])?==i as u64, "native event/KCC endpoint mismatch");
            let mut members=lineage.cluster_residue_membership.cluster_residue_spike_counts.iter()
                .map(|r|r.residue_id).collect::<BTreeSet<_>>();
            if node.causal_lead_residue!=NODE_CAUSAL_LEAD_NONE {members.insert(node.causal_lead_residue);}
            let mut measurements=Vec::new();
            let mut missing=Vec::new();
            for rid in &members {
                let Ok(ri)=residues.binary_search(rid) else {missing.push(*rid);continue};
                let channels=timeline[i]["channels"].as_object().context("event channels")?;
                let mut vectors=serde_json::Map::new();
                for (name,c) in channels {
                    let rows=c["analysis_residue_vectors"].as_array().context("actual residue vector time series missing")?;
                    ensure!(rows.len()==residues.len(), "event residue vector dimension");
                    vectors.insert(name.clone(),json!({"value":rows[ri],"unit":c["unit"]}));
                }
                let mut descriptors=serde_json::Map::new();
                for (name,values) in k["native_kcc_per_residue"].as_object().context("native KCC fields")? {
                    if let Some(rows)=values.as_array() {
                        ensure!(rows.len()==residues.len(),"native KCC axis differs at event");
                        descriptors.insert(name.clone(),rows[ri].clone());
                    }
                }
                measurements.push(json!({"analysis_residue_id":rid,"vectors":vectors,"native_kcc":descriptors}));
            }
            let contact_pairs:Vec<_>=pairs.iter().filter(|p| p["analysis_residue_ids"].as_array()
                .is_some_and(|ids|ids.iter().any(|id|id.as_i64().and_then(|id|u32::try_from(id).ok()).is_some_and(|id|members.contains(&id)))))
                .map(|p|p["pair_index"].clone()).collect();
            member_links+=measurements.len() as u64;
            contact_links+=contact_pairs.len() as u64;
            *phase_counts.entry(phase.into()).or_default()+=1;
            line(&mut w,&json!({"schema_kind":"prism.native_event_mechanistic_observation.v1",
                "arm":arm,"native_record_lineage":lineage,"coordinate_frame_ordinal":i,
                "completed_step":node.step_idx,"physical_time_ps":frames[i].time_for_arm(arm)?,"physical_time_authority":frames[i].arm_physical_time.get(arm),
                "raw_frame_idx":node.frame_idx,"raw_frame_idx_is_coordinate_clock":false,
                "native_record_tags":{"chain_id":chain,"adjudication_code":adjudication,"telemetry_flags":flags,
                    "adjudication_semantics":"0=Prune/observation,1=Construct,2=Violation; every record retained", "class_tainted":flags & 1 != 0},
                "native_lattice_phase":{"ordinal":node.protocol_phase,"binding":a["ghost_phase_binding"]},
                "native_lattice_component":components.as_ref().map(|ids|json!({
                    "component_id":ids[lineage.global_lattice_index as usize],"source":a["native_lattice"],
                    "node_index":lineage.global_lattice_index,
                    "identity_scope":"native lattice component; not materialized region rank; exact producer partition retained"})),
                "resolved_physical_phase":phase,"physical_protocol":a["protocol_state"],
                "measurement_window_id":frames[i].window,
                "native_spectral_geometry":{"aabb_min_angstrom":node.aabb_min,"aabb_max_angstrom":node.aabb_max,
                    "so3_power_spectrum":node.so3_power_spectrum,"so3_plane_status":node.so3_plane_status,
                    "kl_divergence":node.kl_divergence,"thermo_flux":node.thermo_flux,"water_density":node.water_density,
                    "causal_lead_residue":(node.causal_lead_residue!=NODE_CAUSAL_LEAD_NONE).then_some(node.causal_lead_residue),
                    "gear_id":(node.gear_id!=NODE_GEAR_ID_NONE).then_some(node.gear_id),
                    "ccns_phase_bin":(node.ccns_phase_bin!=NODE_CCNS_PHASE_NONE).then_some(node.ccns_phase_bin)},
                "residue_measurements":measurements,"unresolved_analysis_residue_ids":missing,
                "full_atom_vector_channels":timeline[i]["channels"].as_object().unwrap().iter().map(|(name,c)|
                    (name.clone(),json!({"byte_offset":c["byte_offset"],"byte_length":c["byte_length"],"unit":c["unit"]}))).collect::<BTreeMap<_,_>>(),
                "contact_measurements":{"source":s["pair_frames"],"frame_ordinal":i,"pair_indices":contact_pairs,
                    "scope":"measured fixed pair axis incident on event residues; missing pairs unmeasured; spike membership is not cavity lining"},
                "bonded_measurements":{"source":s["bonded_frames"],"term_index":s["bonded_index"],"frame_ordinal":i,
                    "scope":"all signed native bonded coordinate terms retained; event membership does not alter term identity"},
                "kcc_endpoint_support":k["lag_support_ref"],"kcc_endpoint_index":endpoint_index_pin,
                "kcc_winning_pair_physical_times":k["winning_pair_physical_times"],
                "native_control_work_measurements":native_control_work.event(arm,node.step_idx,
                    frames[i].time_for_arm(arm)?,frames[i].arm_physical_time.get(arm),&members.iter().copied().collect::<Vec<_>>())?,
                "isolated_ASC_measurements":asc_endpoints.get(&node.step_idx).map(|endpoint| json!({
                    "source":isolated_asc[arm]["evaluated_source"],"endpoint":endpoint,
                    "event_analysis_residue_ids":members,
                    "join":"same run, topology, stream and completed physical step; exact atom-to-residue axis",
                    "basis":"native laboratory coordinates; not the common-reference AMBER/vector basis",
                    "causal_response_inferred":false})),
                "isolated_ASC_availability":if asc_endpoints.contains_key(&node.step_idx) {"exact_endpoint"}
                    else if isolated_asc[arm]["available"] == true {"atom_rows_not_observed_at_this_step"}
                    else {"native_isolated_journal_unavailable"},
                "native_kcc_metadata":k["native_kcc_per_residue"].as_object().unwrap().iter()
                    .filter(|(_,v)|!v.is_array()).map(|(k,v)|(k.clone(),v.clone())).collect::<BTreeMap<_,_>>(),
                "TSH_admission_used":false,
                "scope":"exact observational correspondence; native cumulative KCC descriptors are not demonstrated directed influence; vectors retain total AMBER force identity"}))?;
        }
        w.finish()?.flush()?;
        ensure!(seen.len()==selected_records,"event consumer lost declared stream population");
        summary.insert(arm.into(),json!({"native_records":seen.len(),"exact_coordinate_joins":seen.len(),
            "resolved_physical_phase_counts":phase_counts,"residue_measurement_links":member_links,
            "adjudication_code_counts":adjudication_counts,"class_tainted_records":tainted_records,
            "component_joined_records":components.as_ref().map(|_|seen.len()),
            "component_count":components.as_ref().map(|_|selected_components.len()),
            "source_stream_id":stream_id,"verified_all_stream_source_records":decoded.decoded_records,
            "stream_selection":"explicit manifest physical condition lane; global lattice indices retained; other streams remain in immutable source",
            "measured_pair_links":contact_links,"first_cold_hold_observations_inferred":false,
            "segment_manifest":a["ghost_segment_manifest"],"raw_lineage":a["ghost_lineage"]}));
    }
    Ok(json!({"available":true,"arms":summary,"consumer":"native chemical-state measurement and DSTW saved-evidence adapter",
        "complete_chemical_state_fiber_bundle_instantiated":false,"spike_membership_relabelled_as_cavity_lining":false}))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_components_require_exact_once_identity_partition() {
        let node=crate::ghost_phase_lattice::GhostPhaseLatticeNode::empty();
        let row=json!({"tile_index":0,"step_idx":0,"frame_idx":0,"stream_id":0,"site_id":0,"protocol_phase":0});
        let component=json!({"component_id":47,"n_nodes":1,"node_lineage":[row]});
        let lattice=json!({"components":[component]});
        assert_eq!(component_membership(&lattice,&[node]).unwrap(),vec![47]);
        let mut bad=lattice.clone();bad["components"][0]["node_lineage"][0]["step_idx"]=json!(1);
        assert!(component_membership(&bad,&[node]).is_err());
        let mut bad=lattice.clone();bad["components"][0]["node_lineage"][0]["tile_index"]=json!(1);
        assert!(component_membership(&bad,&[node]).is_err());
        let mut other=component.clone();other["component_id"]=json!(48);
        assert!(component_membership(&json!({"components":[component,other]}),&[node]).is_err());
        assert!(component_membership(&json!({"components":[]}),&[node]).is_err());
    }
    #[test]
    fn explicit_measurement_stream_lanes_preserve_historical_zero_without_cross_stream_mix() {
        assert_eq!(manifest_stream_id(&json!({})).unwrap(),0);
        for id in 0..16 {
            let m=json!({"stream_id":id,"mechanistic_sources":{"arms":{"treatment":{"stream_id":id},"control":{"stream_id":id}}}});
            assert_eq!(manifest_stream_id(&m).unwrap(),id as u16);
            let mut wrong=m;wrong["mechanistic_sources"]["arms"]["control"]["stream_id"]=json!(id+1);
            assert!(manifest_stream_id(&wrong).is_err());
        }
        assert!(manifest_stream_id(&json!({"stream_id":null})).is_err());
        assert!(manifest_stream_id(&json!({"stream_id":65536})).is_err());
    }
    #[test]
    fn physical_cold_return_is_distinct_from_lattice_cooling() {
        let p=json!({"cold_hold_end":14,"ramp_end":20,"warm_hold_end":35,"ramp_down_end":41,"total_steps":55});
        for (step,phase) in [(0,"cold_hold"),(14,"heating"),(20,"warm_hold"),(35,"cooling"),(41,"cold_return"),(55,"cold_return")] {
            assert_eq!(measured_protocol_phase(&p,step).unwrap(),phase);
        }
        assert!(measured_protocol_phase(&p,56).is_err());
        let mut p=p;p["ramp_down_end"]=json!(30);
        assert!(measured_protocol_phase(&p,40).is_err());
    }
    #[test]
    fn physical_warm_tail_matches_device_override_without_extending_clock() {
        let p=json!({"cold_hold_end":14,"ramp_end":20,"warm_hold_end":35,"ramp_down_end":41,"total_steps":55,"steering_flags":16});
        assert_eq!(measured_protocol_phase(&p,19).unwrap(),"heating");
        for step in [20,34,41,55] {assert_eq!(measured_protocol_phase(&p,step).unwrap(),"warm_hold");}
        for step in [35,40] {assert_eq!(measured_protocol_phase(&p,step).unwrap(),"cooling");}
        assert!(measured_protocol_phase(&p,56).is_err());
    }
    #[test]
    fn vector_moments_preserve_cross_terms_and_exact_constants() {
        let mut m = VectorMoments::default();
        m.add([1., 2., 3.]).unwrap();
        m.add([3., 6., 9.]).unwrap();
        assert_eq!(m.mean, [2., 4., 6.]);
        assert_eq!(m.m2.map(|x| x / 2.), [1., 2., 3., 4., 6., 9.]);
        let mut m = VectorMoments::default();
        for _ in 0..5500 {
            m.add([0.015151515151515166, -2., 0.]).unwrap();
        }
        assert_eq!(m.m2, [0.; 6]);
        assert!(m.add([f64::NAN, 0., 0.]).is_err());
    }
    #[test]
    fn rigid_basis_rotates_forces_without_translation() {
        let t = Transform {
            frame_ordinal: 0,
            step: 10,
            rotation: [[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]],
            translation_angstrom: [10., 20., 30.],
            reference_coordinate_payload_sha256: String::new(),
            coordinate_payload_sha256: String::new(),
            coordinate_frame_hash64: String::new(),
        };
        validate_rotation(t.rotation).unwrap();
        assert_eq!(position([1., 2., 3.], &t), [8., 21., 33.]);
        assert_eq!(rotate([1., 2., 3.], t.rotation), [-2., 1., 3.]);
        assert!(validate_rotation([[-1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]).is_err());
        let a = [1., 2., 3.];
        let b = [4., 1., -2.];
        let distance =
            |x: [f64; 3], y: [f64; 3]| (0..3).map(|i| (x[i] - y[i]).powi(2)).sum::<f64>();
        assert_eq!(distance(a, b), distance(position(a, &t), position(b, &t)));
    }
    #[test]
    fn missing_scalar_is_not_zero_and_torsion_wrap_is_circular() {
        let mut m = ScalarMoments::default();
        m.add(None, true).unwrap();
        m.add(Some(std::f64::consts::PI - 0.01), true).unwrap();
        m.add(Some(-std::f64::consts::PI + 0.01), true).unwrap();
        assert_eq!(m.n, 2);
        assert_eq!(m.missing, 1);
        let v = m.value(true);
        assert!(
            (v["circular_mean_radians"].as_f64().unwrap().abs() - std::f64::consts::PI).abs()
                < 1e-10
        );
        assert!(v["resultant_length"].as_f64().unwrap() > 0.99);
        let terms = vec![
            json!({"i":1,"j":2,"k_idx":3,"l":4,"periodicity":1}),
            json!({"i":1,"j":2,"k_idx":3,"l":4,"periodicity":2}),
        ];
        assert_eq!(torsion_groups(&terms).unwrap().len(), 1);
    }
    #[test]
    fn undefined_circular_direction_remains_null() {
        let mut m = ScalarMoments::default();
        m.n = 2;
        m.circular.n = 2;
        assert!(m.value(true)["circular_mean_radians"].is_null());
    }
    #[test]
    fn historical_producer_epoch_does_not_override_active_source_pin() {
        let current = json!({"path":"same.rs","sha256":"current"});
        let m = json!({"source_files":[current],"frames":{},"force_treatment":{},"force_control":{},"force_chronology_receipt":{},"mechanistic_sources":{},"historical_comparison_producer":{"path":"same.rs","sha256":"old"}});
        let pins = collect_manifest_pins(&m).unwrap();
        assert_eq!(pins[&PathBuf::from("same.rs")].sha256, "current");
        assert_eq!(m["historical_comparison_producer"]["sha256"], "old");
    }
    #[test]
    fn conflicting_or_malformed_pins_are_rejected() {
        let mut pins = BTreeMap::new();
        assert!(collect_pins(
            &json!([{ "path":"a","sha256":"a"},{"path":"a","sha256":"b"}]),
            &mut pins
        )
        .is_err());
        assert!(MeasurementSourcePin {
            path: "absent".into(),
            sha256: "broken".into()
        }
        .verify()
        .is_err());
    }
}

#[cfg(test)]
mod signed_membership_tests {
    use super::*;
    #[test]
    fn negative_atoms_have_no_receptor_membership_but_zero_is_valid() {
        let original=vec![-1,0,4,-7,0,4];
        let (residues,membership)=receptor_residue_axis(&original);
        assert_eq!(residues,vec![0,4]);
        assert_eq!(membership,vec![None,Some(0),Some(1),None,Some(0),Some(1)]);
        assert_eq!(serde_json::from_value::<Vec<i32>>(json!(original)).unwrap(),original);
    }

    #[test]
    #[ignore="reads the original pinned periodic no-probe and fragment topologies; no molecular simulation"]
    fn actual_periodic_environment_and_fragment_atoms_preserve_full_signed_axis()->Result<()> {
        let root_buf=std::env::var_os("PRISM_CONTROL_ROOT").map(std::path::PathBuf::from).ok_or_else(|| anyhow::anyhow!("PRISM_CONTROL_ROOT unset: set it to the root that holds MYC_MAX_ADAPTIVE_PROBE_GHOST_FIBER_20260723"))?.join("MYC_MAX_ADAPTIVE_PROBE_GHOST_FIBER_20260723/LAMBDA_TOPOLOGIES_20260723_R8_PRODUCTION_DERIVED");
        let root=root_buf.as_path();
        for (file,digest,atoms,environment) in [
            ("mycmax_no_probe_2fs.topology.json","7c2e87ef3a2964662f55d9c5e163625132c54c9f5228c001e832a03ba11f7ca3",155898,153108),
            ("mycmax_lambda_species_001_precal.topology.json","9392e3ba8accc6a36144595dce3d29e4e6741cd5135c1ce5fbecc493e99bcc35",155994,153204),
        ] {
            let path=root.join(file);ensure!(digest_file(&path)?==digest,"actual periodic topology fixture changed");
            // These hash-pinned R8 fixtures predate the current periodic-cell schema.
            // Exercise their original signed atom/identity fields, not full runtime admission.
            // Production geometry keeps strict PrismPrepTopology parsing and validation.
            let topology:Value=serde_json::from_reader(File::open(&path)?)?;
            ensure!(topology["periodic_cell"]["periodic"]==true && topology["periodic_cell"]["boundary_condition"]=="ORTHORHOMBIC_TINFOIL_3D","original periodic authority changed");
            let signed:Vec<i32>=serde_json::from_value(topology["atom_analysis_residue_ids"].clone())?;
            let physical_ids:Vec<i32>=serde_json::from_value(topology["residue_ids"].clone())?;
            let chains:Vec<String>=serde_json::from_value(topology["chain_ids"].clone())?;
            let names:Vec<String>=serde_json::from_value(topology["residue_names"].clone())?;
            let identity_rows=topology["analysis_residue_contract"]["analysis_residues"].as_array().context("original analysis identity rows absent")?;
            let mut identities=BTreeMap::new();
            for row in identity_rows {
                let id=u32::try_from(row["analysis_residue_id"].as_u64().context("original analysis residue ID absent")?)?;
                ensure!(identities.insert(id,row).is_none(),"duplicate original receptor identity");
            }
            let (residues,membership)=receptor_residue_axis(&signed);
            ensure!(topology["n_atoms"].as_u64()==Some(atoms as u64) && signed.len()==atoms && membership.len()==atoms && physical_ids.len()==atoms && chains.len()==atoms && names.len()==atoms,"full loaded atom population truncated");
            ensure!(membership.iter().filter(|r|r.is_none()).count()==environment && residues==(0u32..163).collect::<Vec<_>>(),"environment/receptor partition changed");
            ensure!(identities.keys().copied().collect::<Vec<_>>()==residues && membership.iter().filter(|r|r.is_some()).count()==2790,"original receptor identity population changed");
            for (atom,raw) in signed.iter().enumerate() {
                match membership[atom] {
                    Some(ri)=>{
                        ensure!(*raw>=0 && residues[ri]==u32::try_from(*raw)?,"receptor atom lost exact native identity");
                        let row=identities[&residues[ri]];
                        ensure!(row["physical_residue_id"].as_i64()==Some(i64::from(physical_ids[atom])) && row["chain_id"].as_str()==Some(chains[atom].as_str()) && row["residue_name"].as_str()==Some(names[atom].as_str()) && row["role"]=="RECEPTOR","receptor atom differs from original physical/chain/residue identity");
                    }
                    None=>ensure!(*raw<0,"environment atom fabricated as receptor"),
                }
            }
            ensure!(serde_json::from_value::<Vec<i32>>(serde_json::to_value(&signed)?)?==signed,"signed native IDs changed on serialization");
        }
        Ok(())
    }

    #[test]
    #[ignore="reads the original pinned 7F5N nonnegative topology; no molecular simulation"]
    fn actual_7f5n_nonnegative_mapping_and_ordered_reductions_remain_identical()->Result<()> {
        let path_buf=std::env::var_os("PRISM_INPUTS_ROOT").map(std::path::PathBuf::from).ok_or_else(|| anyhow::anyhow!("PRISM_INPUTS_ROOT unset: set it to the root that holds 7F5N_A/7F5N_A.v3.topology.json"))?.join("7F5N_A/7F5N_A.v3.topology.json");
        let path=path_buf.as_path();
        ensure!(digest_file(path)?=="861be5f35178fa5b5f15022bda7f6b6ffa9febf679d4ff1932669a4accf0b4d3","7F5N fixture source changed");
        let topology:Value=serde_json::from_reader(File::open(path)?)?;
        let old:Vec<u32>=serde_json::from_value(topology["atom_analysis_residue_ids"].clone())?;
        let signed:Vec<i32>=serde_json::from_value(topology["atom_analysis_residue_ids"].clone())?;
        let masses:Vec<f64>=serde_json::from_value(topology["masses"].clone())?;
        let positions:Vec<f64>=serde_json::from_value(topology["positions"].clone())?;
        let old_residues=old.iter().copied().collect::<BTreeSet<_>>().into_iter().collect::<Vec<_>>();
        let old_lookup=old_residues.iter().enumerate().map(|(i,id)|(*id,i)).collect::<BTreeMap<_,_>>();
        let old_mapping=old.iter().map(|id|old_lookup[id]).collect::<Vec<_>>();
        let (residues,mapping)=receptor_residue_axis(&signed);
        ensure!(residues==old_residues && mapping.iter().copied().collect::<Option<Vec<_>>>()==Some(old_mapping.clone()) && serde_json::to_value(&signed)?==serde_json::to_value(&old)?,"positive source/membership behavior changed");
        let mut old_mass=vec![0.0f64;residues.len()];let mut new_mass=old_mass.clone();
        let mut old_position=vec![[0.0f64;3];residues.len()];let mut new_position=old_position.clone();
        for atom in 0..signed.len() {
            old_mass[old_mapping[atom]]+=masses[atom];
            if let Some(ri)=mapping[atom] {new_mass[ri]+=masses[atom];}
            for axis in 0..3 {
                old_position[old_mapping[atom]][axis]+=positions[atom*3+axis]*masses[atom];
                if let Some(ri)=mapping[atom] {new_position[ri][axis]+=positions[atom*3+axis]*masses[atom];}
            }
        }
        for ri in 0..residues.len() {
            ensure!(old_mass[ri].to_bits()==new_mass[ri].to_bits(),"ordered residue mass changed");
            for axis in 0..3 {ensure!((old_position[ri][axis]/old_mass[ri]).to_bits()==(new_position[ri][axis]/new_mass[ri]).to_bits(),"ordered receptor coordinate reduction changed");}
        }
        Ok(())
    }
}
