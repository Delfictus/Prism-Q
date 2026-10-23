//! Exact ASC endpoint records and complete physical-step control-work coverage.
//! Sampling atom rows does not imply that intervening atom rows were observed.
use super::{AscDevice, AscSnapshot};
use anyhow::{ensure, Context, Result};
use sha2::{Digest, Sha256};
use std::{
    ffi::c_void,
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
};
const ROW_BYTES: usize = 320;
const WORK_MAGIC: u64 = 0x415343574f524b31;
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default)]
struct CumulativeWork {
    magic: u64,
    completed_steps: u64,
    completed_epoch: u64,
    control_work_kcal_mol: f64,
}
const _: () = assert!(std::mem::size_of::<CumulativeWork>() == 32);

/// All fields of the native 320-byte atom record, in the producer's lab basis.
/// The two force/kick endpoints remain ordered; none is averaged or relabelled
/// as a total force, thermodynamic work, or demonstrated causal response.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AscAtomMeasurement {
    pub atom_index: usize,
    /// Original signed native topology value. A negative value preserves an
    /// environment atom record but grants no receptor residue membership.
    pub analysis_residue_id: i32,
    pub physical_epochs: [u64; 4],
    pub decision_epochs: [u64; 4],
    pub potential_kcal_mol: [f64; 2],
    pub control_change_work_kcal_mol: f64,
    pub base_force_kcal_mol_angstrom: [[f32; 3]; 2],
    pub impulse_kcal_ps_mol_angstrom: [[f64; 3]; 2],
    pub isolated_asc_force_kcal_mol_angstrom: [[f64; 3]; 2],
    pub torque_about_coordinate_origin_kcal_mol: [[f64; 3]; 2],
    pub velocity_delta_angstrom_ps: [[f64; 3]; 2],
    pub differential_kick_energy_kcal_mol: [f64; 2],
}

fn decode_atom(row: &[u8], atom_index: usize, analysis_residue_id: i32) -> AscAtomMeasurement {
    let doubles = |offset: usize| {
        std::array::from_fn(|half| {
            std::array::from_fn(|axis| f64::from_bits(u64_at(row, offset + (half * 3 + axis) * 8)))
        })
    };
    AscAtomMeasurement {
        atom_index,
        analysis_residue_id,
        physical_epochs: std::array::from_fn(|i| u64_at(row, i * 8)),
        decision_epochs: std::array::from_fn(|i| u64_at(row, 32 + i * 8)),
        potential_kcal_mol: std::array::from_fn(|i| f64::from_bits(u64_at(row, 64 + i * 8))),
        control_change_work_kcal_mol: f64::from_bits(u64_at(row, 80)),
        base_force_kcal_mol_angstrom: std::array::from_fn(|half| {
            std::array::from_fn(|axis| {
                let offset = 88 + (half * 3 + axis) * 4;
                f32::from_le_bytes(row[offset..offset + 4].try_into().unwrap())
            })
        }),
        impulse_kcal_ps_mol_angstrom: doubles(112),
        isolated_asc_force_kcal_mol_angstrom: doubles(160),
        torque_about_coordinate_origin_kcal_mol: doubles(208),
        velocity_delta_angstrom_ps: doubles(256),
        differential_kick_energy_kcal_mol: std::array::from_fn(|i| {
            f64::from_bits(u64_at(row, 304 + i * 8))
        }),
    }
}

/// Measured intervention support, kept separate from total AMBER force and
/// from any claim of residue mediation. Counts describe saved atom endpoints;
/// full vectors remain in their original journal. Missing schema fields retain
/// unavailable status for historical records.
#[derive(serde::Serialize)]
struct RecordedInterventionSupport {
    control_alpha: Option<f64>,
    selected_atom_indices: Option<Vec<usize>>,
    observed_atom_rows: u64,
    observed_nonzero_force_atom_halves: u64,
    observed_nonzero_impulse_atom_halves: u64,
    #[serde(skip)]
    selected_mask: Option<Vec<bool>>,
}
impl RecordedInterventionSupport {
    fn from_schema(schema: &serde_json::Value, atoms: usize) -> Result<Self> {
        let alpha = schema
            .get("control_alpha")
            .map(|v| {
                v.as_f64()
                    .filter(|x| x.is_finite())
                    .context("ASC control gain must be finite")
            })
            .transpose()?;
        let indices = if schema["support_mode"] == "SELECTED_STRUCTURAL_ATOM_RESIDUE_SUPPORT" {
            let mut mask = vec![false; atoms];
            let mut indices = Vec::new();
            for v in schema["selected_atom_indices"]
                .as_array()
                .context("ASC selected atom support absent")?
            {
                let i = usize::try_from(v.as_u64().context("ASC atom support index")?)?;
                ensure!(
                    i < atoms && !mask[i],
                    "ASC atom support duplicate or out of range"
                );
                mask[i] = true;
                indices.push(i);
            }
            ensure!(!indices.is_empty(), "ASC selected atom support empty");
            Some((indices, mask))
        } else {
            None
        };
        let (selected_atom_indices, selected_mask) = match indices {
            Some((indices, mask)) => (Some(indices), Some(mask)),
            None => (None, None),
        };
        Ok(Self {
            control_alpha: alpha,
            selected_atom_indices,
            selected_mask,
            observed_atom_rows: 0,
            observed_nonzero_force_atom_halves: 0,
            observed_nonzero_impulse_atom_halves: 0,
        })
    }
    fn observe(&mut self, raw: &[u8]) -> Result<()> {
        ensure!(
            raw.len() % ROW_BYTES == 0,
            "ASC intervention record alignment"
        );
        if let Some(mask) = &self.selected_mask {
            ensure!(
                mask.len() == raw.len() / ROW_BYTES,
                "ASC support/row atom population mismatch"
            );
        }
        for (atom, row) in raw.chunks_exact(ROW_BYTES).enumerate() {
            let value = |offset| f64::from_bits(u64_at(row, offset));
            // Producer prism_asc_compose writes zero potential/work/force outside
            // the selected mask. The kick comparison uses the same total/base
            // force there. Total AMBER base_force at bytes88..112 is unrestricted.
            let must_be_zero = self.control_alpha == Some(0.0)
                || self.selected_mask.as_ref().is_some_and(|mask| !mask[atom]);
            if must_be_zero {
                for offset in (64..88).step_by(8).chain((112..ROW_BYTES).step_by(8)) {
                    ensure!(value(offset) == 0.0,
                        "ASC nonzero isolated contribution on zero-gain or unselected atom {atom} at byte {offset}");
                }
            }
            self.observed_atom_rows += 1;
            for half in 0..2 {
                self.observed_nonzero_force_atom_halves +=
                    (0..3).any(|axis| value(160 + half * 24 + axis * 8) != 0.0) as u64;
                self.observed_nonzero_impulse_atom_halves +=
                    (0..3).any(|axis| value(112 + half * 24 + axis * 8) != 0.0) as u64;
            }
        }
        Ok(())
    }
}

/// Source-bound streaming consumer. The callback sees complete atom records
/// and the original endpoint/work metadata. Memory is bounded to one endpoint.
pub fn visit_isolated_asc_endpoints(
    dir: &Path,
    run_id: &str,
    stream: u32,
    topology_sha256: &str,
    atom_analysis_residue_ids: &[i32],
    mut visit: impl FnMut(&serde_json::Value, &[AscAtomMeasurement]) -> Result<()>,
) -> Result<serde_json::Value> {
    let verified = verify_sealed_journal(dir, run_id, stream)?;
    let schema = &verified["schema"];
    ensure!(
        schema["topology_atom_order_sha256"] == topology_sha256
            && schema["atoms"] == atom_analysis_residue_ids.len(),
        "ASC consumer topology/atom axis mismatch"
    );
    let mut reader = BufReader::new(File::open(dir.join("atom_rows.bin"))?);
    let mut raw = vec![
        0;
        atom_analysis_residue_ids
            .len()
            .checked_mul(ROW_BYTES)
            .context("ASC consumer size overflow")?
    ];
    let mut digest = Sha256::new();
    for record in BufReader::new(File::open(dir.join("endpoints.jsonl"))?).lines() {
        let record: serde_json::Value = serde_json::from_str(&record?)?;
        reader.read_exact(&mut raw)?;
        ensure!(
            record["row_sha256"] == hash(&raw),
            "ASC endpoint bytes changed during consumption"
        );
        validate_rows(
            &raw,
            atom_analysis_residue_ids.len(),
            record["completed_physical_step"]
                .as_u64()
                .context("ASC endpoint step")?,
            record["active_decision"]["producer_epoch"]
                .as_u64()
                .context("ASC decision epoch")?,
        )?;
        let atoms: Vec<_> = raw
            .chunks_exact(ROW_BYTES)
            .enumerate()
            .map(|(i, row)| decode_atom(row, i, atom_analysis_residue_ids[i]))
            .collect();
        visit(&record, &atoms)?;
        digest.update(&raw);
    }
    ensure!(
        reader.read(&mut [0])? == 0
            && verified["seal"]["rows_sha256"] == format!("{:x}", digest.finalize()),
        "ASC row population changed during consumption"
    );
    Ok(verified)
}

/// Native downstream representation: complete per-atom channels, exact residue
/// membership, and an endpoint index. Original binary rows remain authoritative.
pub fn export_isolated_asc_measurements(
    dir: &Path,
    run_id: &str,
    stream: u32,
    topology_sha256: &str,
    atom_analysis_residue_ids: &[i32],
    output: &Path,
) -> Result<serde_json::Value> {
    use serde_json::json;
    let mut writer =
        zstd::stream::write::Encoder::new(std::io::BufWriter::new(create_new(output)?), 1)?;
    let mut residues = std::collections::BTreeMap::<u32, Vec<usize>>::new();
    for (atom, residue) in atom_analysis_residue_ids.iter().enumerate() {
        if let Ok(receptor_residue) = u32::try_from(*residue) {
            residues.entry(receptor_residue).or_default().push(atom);
        }
    }
    let mut endpoints = Vec::new();
    let verified = visit_isolated_asc_endpoints(
        dir,
        run_id,
        stream,
        topology_sha256,
        atom_analysis_residue_ids,
        |record, atoms| {
            let ordinal = endpoints.len();
            serde_json::to_writer(
                &mut writer,
                &json!({
                    "schema_kind":"prism.native_isolated_asc_measurements.v1",
                    "run_id":run_id,"stream_id":stream,"topology_sha256":topology_sha256,
                    "decompressed_line_ordinal":ordinal,"endpoint":record,
                    "atom_measurements":atoms,"residue_atom_indices":residues,
                    "basis":"native laboratory coordinates; torque about coordinate origin",
                    "atom_fields_reduction":"none; both ordered force and preconstraint kick endpoints retained",
                    "scope":"isolated applied ASC contribution; event correspondence alone does not establish causal influence"
                }),
            )?;
            writer.write_all(b"\n")?;
            endpoints.push(json!({"completed_step":record["completed_physical_step"],
                "decompressed_line_ordinal":ordinal,"raw_byte_offset":record["byte_offset"],
                "raw_byte_length":record["byte_length"],"raw_row_sha256":record["row_sha256"],
                "active_decision":record["active_decision"],
                "work_interval_start_exclusive":record["work_interval_start_exclusive"],
                "work_interval_end_inclusive":record["work_interval_end_inclusive"],
                "interval_control_work_kcal_mol":record["interval_control_work_kcal_mol"],
                "unobserved_intermediate_atom_row_steps":record["unobserved_intermediate_atom_row_steps"]}));
            Ok(())
        },
    )?;
    let mut file = writer.finish()?;
    file.flush()?;
    file.get_ref().sync_all()?;
    Ok(
        json!({"available":true,"source":verified,"output_path":output,
        "endpoint_index":endpoints,"atom_count":atom_analysis_residue_ids.len(),
        "residue_count":residues.len(),"total_AMBER_force_replaced":false,
        "intermediate_atom_rows_reconstructed":false}),
    )
}
extern "C" {
    fn prism_asc_measurement_snapshot_v1(
        ptr: u64,
        stream: *mut c_void,
        snapshot: *mut AscSnapshot,
        work: *mut CumulativeWork,
        rows: *mut c_void,
        row_bytes: usize,
    ) -> i32;
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn u64_at(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}
fn validate_rows(rows: &[u8], atoms: usize, epoch: u64, decision: u64) -> Result<()> {
    ensure!(
        rows.len()
            == atoms
                .checked_mul(ROW_BYTES)
                .context("ASC row size overflow")?,
        "ASC row length mismatch"
    );
    for (atom, row) in rows.chunks_exact(ROW_BYTES).enumerate() {
        for k in 0..4 {
            ensure!(
                u64_at(row, k * 8) == epoch && u64_at(row, 32 + k * 8) == decision,
                "ASC atom {atom} has stale physical/decision epoch at event {k}"
            );
        }
        for offset in (64..88).step_by(8).chain((112..320).step_by(8)) {
            ensure!(
                f64::from_bits(u64_at(row, offset)).is_finite(),
                "ASC nonfinite atom {atom} field {offset}"
            );
        }
        for offset in (88..112).step_by(4) {
            ensure!(
                f32::from_le_bytes(row[offset..offset + 4].try_into().unwrap()).is_finite(),
                "ASC nonfinite baseline force atom {atom}"
            );
        }
    }
    Ok(())
}
impl AscDevice {
    fn isolated_snapshot(
        &self,
        expected_step: u64,
    ) -> Result<(AscSnapshot, CumulativeWork, Vec<u8>)> {
        ensure!(
            cfg!(target_endian = "little"),
            "ASC exact ABI export requires little-endian host"
        );
        let mut rows = vec![
            0u8;
            self.n_atoms()
                .checked_mul(ROW_BYTES)
                .context("ASC row allocation overflow")?
        ];
        let mut snapshot: AscSnapshot = unsafe { std::mem::zeroed() };
        let mut work = CumulativeWork::default();
        let rc = unsafe {
            prism_asc_measurement_snapshot_v1(
                self.device,
                self.stream.cu_stream().cast(),
                &mut snapshot,
                &mut work,
                rows.as_mut_ptr().cast(),
                rows.len(),
            )
        };
        ensure!(
            rc == 0,
            "isolated ASC snapshot failed rc={rc}; completion not certified"
        );
        ensure!(
            snapshot.abi == 1
                && snapshot.byte_size == 736
                && snapshot.source == self.source
                && snapshot.generation == self.generation
                && snapshot.n_atoms as usize == self.n_atoms(),
            "isolated ASC allocation identity mismatch"
        );
        ensure!(
            expected_step > 0
                && snapshot.physical_epoch == expected_step
                && snapshot.completed_epoch == expected_step
                && snapshot.coverage_failures == 0
                && matches!(snapshot.status, 0 | 1),
            "isolated ASC physical endpoint mismatch"
        );
        ensure!(
            work.magic == WORK_MAGIC
                && work.completed_epoch == expected_step
                && work.completed_steps == expected_step
                && work.control_work_kcal_mol.is_finite(),
            "isolated ASC cumulative work lacks full physical-step coverage"
        );
        validate_rows(
            &rows,
            self.n_atoms(),
            expected_step,
            snapshot.active.producer_epoch,
        )?;
        Ok((snapshot, work, rows))
    }
}
/// Append-only raw rows plus source-bound endpoint records. A seal is written
/// only after both streams are synced. Incomplete output remains inspectable.
pub struct AscMeasurementJournal {
    dir: PathBuf,
    rows: File,
    records: File,
    run_id: String,
    stream_id: u32,
    schema_sha256: String,
    generation: u64,
    active_device_generation: u64,
    allocation_rebinds: Vec<AscAllocationRebind>,
    previous_step: u64,
    previous_work: f64,
    endpoints: u64,
    byte_offset: u64,
    rows_hash: Sha256,
    records_hash: Sha256,
    finished: bool,
}
fn create_new(path: &Path) -> Result<File> {
    Ok(OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("create exclusive ASC artifact {}", path.display()))?)
}
fn json_bytes(v: &serde_json::Value) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec(v)?;
    bytes.push(b'\n');
    Ok(bytes)
}
/// Allocation addresses/generations are process-local. Every rebind is explicit;
/// original journal identity and numerical epochs remain unchanged.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AscAllocationRebind {
    pub previous_device_generation: u64,
    pub restored_device_generation: u64,
    pub before_endpoint: u64,
    pub completed_physical_step: u64,
    pub rows_checkpoint_sha256: String,
    pub records_checkpoint_sha256: String,
}
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AscMeasurementContinuationV1 {
    pub schema_version: u32,
    pub relative_dir: PathBuf,
    pub rows: crate::scientific_checkpoint::AppendCursor,
    pub records: crate::scientific_checkpoint::AppendCursor,
    pub run_id: String,
    pub stream_id: u32,
    pub schema_sha256: String,
    pub generation: u64,
    pub active_device_generation: u64,
    pub allocation_rebinds: Vec<AscAllocationRebind>,
    pub previous_step: u64,
    pub previous_work_bits: u64,
    pub endpoints: u64,
    pub byte_offset: u64,
}
fn generation_for_endpoint(history: &[AscAllocationRebind], original: u64, index: u64) -> u64 {
    history
        .iter()
        .filter(|h| h.before_endpoint <= index)
        .last()
        .map_or(original, |h| h.restored_device_generation)
}
fn validate_generation_history(
    history: &[AscAllocationRebind],
    original: u64,
    current: u64,
    endpoints: u64,
    step: u64,
) -> Result<()> {
    let mut generation = original;
    let mut last_endpoint = 0;
    let mut last_step = 0;
    for item in history {
        ensure!(
            item.previous_device_generation == generation
                && item.before_endpoint >= last_endpoint
                && item.before_endpoint <= endpoints
                && item.completed_physical_step >= last_step
                && item.completed_physical_step <= step
                && [
                    &item.rows_checkpoint_sha256,
                    &item.records_checkpoint_sha256
                ]
                .iter()
                .all(|h| h.len() == 64 && h.bytes().all(|b| b.is_ascii_hexdigit())),
            "ASC allocation continuation lineage mismatch"
        );
        generation = item.restored_device_generation;
        last_endpoint = item.before_endpoint;
        last_step = item.completed_physical_step;
    }
    ensure!(
        generation == current,
        "ASC current allocation generation unbound"
    );
    Ok(())
}
impl AscMeasurementJournal {
    pub fn checkpoint(&mut self, run_root: &Path) -> Result<AscMeasurementContinuationV1> {
        ensure!(
            !self.finished,
            "sealed ASC journal cannot become a live continuation"
        );
        let rows = crate::scientific_checkpoint::seal_append_file(
            &self.rows,
            self.dir
                .join("atom_rows.bin")
                .strip_prefix(run_root)?
                .to_str()
                .context("ASC path encoding")?,
        )?;
        let records = crate::scientific_checkpoint::seal_append_file(
            &self.records,
            self.dir
                .join("endpoints.jsonl")
                .strip_prefix(run_root)?
                .to_str()
                .context("ASC path encoding")?,
        )?;
        ensure!(
            rows.committed_bytes == self.byte_offset
                && rows.committed_sha256 == format!("{:x}", self.rows_hash.clone().finalize())
                && records.committed_sha256
                    == format!("{:x}", self.records_hash.clone().finalize()),
            "ASC output hash state differs from committed files"
        );
        ensure!(
            hash(&crate::checkpoint_output::read_metadata(
                &self.dir.join("schema.json")
            )?) == self.schema_sha256,
            "ASC schema changed"
        );
        File::open(&self.dir)?.sync_all()?;
        validate_generation_history(
            &self.allocation_rebinds,
            self.generation,
            self.active_device_generation,
            self.endpoints,
            self.previous_step,
        )?;
        Ok(AscMeasurementContinuationV1 {
            schema_version: 1,
            relative_dir: self.dir.strip_prefix(run_root)?.into(),
            rows,
            records,
            run_id: self.run_id.clone(),
            stream_id: self.stream_id,
            schema_sha256: self.schema_sha256.clone(),
            generation: self.generation,
            active_device_generation: self.active_device_generation,
            allocation_rebinds: self.allocation_rebinds.clone(),
            previous_step: self.previous_step,
            previous_work_bits: self.previous_work.to_bits(),
            endpoints: self.endpoints,
            byte_offset: self.byte_offset,
        })
    }
    pub fn reopen_checkpoint(
        run_root: &Path,
        state: &AscMeasurementContinuationV1,
        owner: &AscDevice,
    ) -> Result<Self> {
        ensure!(
            state.schema_version == 1 && state.stream_id == owner.source,
            "ASC continuation owner/format differs"
        );
        ensure!(
            !state.relative_dir.as_os_str().is_empty()
                && state
                    .relative_dir
                    .components()
                    .all(|c| matches!(c, std::path::Component::Normal(_))),
            "ASC directory locator invalid"
        );
        let dir = run_root.join(&state.relative_dir);
        ensure!(
            run_root.join(&state.rows.relative_path) == dir.join("atom_rows.bin")
                && run_root.join(&state.records.relative_path) == dir.join("endpoints.jsonl"),
            "ASC output cursors have wrong owners"
        );
        let schema_bytes = crate::checkpoint_output::read_metadata(&dir.join("schema.json"))?;
        ensure!(
            hash(&schema_bytes) == state.schema_sha256,
            "ASC schema prefix changed"
        );
        let schema: serde_json::Value = serde_json::from_slice(&schema_bytes)?;
        ensure!(
            schema["run_id"] == state.run_id
                && schema["source_stream_id"] == state.stream_id
                && schema["allocation_generation"] == state.generation
                && schema["topology_atom_order_sha256"] == owner.topology_sha256
                && schema["atoms"] == owner.atoms,
            "ASC restored topology/source differs"
        );
        validate_generation_history(
            &state.allocation_rebinds,
            state.generation,
            state.active_device_generation,
            state.endpoints,
            state.previous_step,
        )?;
        if dir.join("seal.json").exists() {
            crate::checkpoint_output::retain_later_file(&dir, &dir.join("seal.json"))?;
        }
        let rows = crate::scientific_checkpoint::reopen_append_file(run_root, &state.rows)?;
        let records = crate::scientific_checkpoint::reopen_append_file(run_root, &state.records)?;
        let mut rows_hash = Sha256::new();
        let mut records_hash = Sha256::new();
        let mut row_reader = BufReader::new(File::open(dir.join("atom_rows.bin"))?);
        let mut record_reader = BufReader::new(File::open(dir.join("endpoints.jsonl"))?);
        let mut encoded = Vec::new();
        let mut raw = vec![
            0;
            usize::try_from(owner.atoms)?
                .checked_mul(ROW_BYTES)
                .context("ASC row size overflow")?
        ];
        let (mut count, mut offset, mut previous, mut work) = (0u64, 0u64, 0u64, 0.0f64);
        loop {
            encoded.clear();
            if record_reader.read_until(b'\n', &mut encoded)? == 0 {
                break;
            }
            ensure!(
                encoded.last() == Some(&b'\n'),
                "ASC partial record in committed prefix"
            );
            let r: serde_json::Value = serde_json::from_slice(&encoded)?;
            let step = r["completed_physical_step"]
                .as_u64()
                .context("ASC record step")?;
            let next = r["cumulative_control_work_after_kcal_mol"]
                .as_f64()
                .context("ASC work")?;
            ensure!(
                step > previous
                    && next.is_finite()
                    && r["run_id"] == state.run_id
                    && r["source_stream_id"] == state.stream_id
                    && r["allocation_generation"] == state.generation
                    && r["schema_sha256"] == state.schema_sha256
                    && r["byte_offset"] == offset
                    && r["byte_length"] == raw.len()
                    && r["work_interval_start_exclusive"] == previous
                    && r["cumulative_control_work_before_kcal_mol"].as_f64() == Some(work)
                    && r["device_allocation_generation"]
                        .as_u64()
                        .unwrap_or(state.generation)
                        == generation_for_endpoint(
                            &state.allocation_rebinds,
                            state.generation,
                            count
                        ),
                "ASC checkpoint record lineage differs"
            );
            row_reader.read_exact(&mut raw)?;
            ensure!(
                r["row_sha256"] == hash(&raw),
                "ASC checkpoint row hash differs"
            );
            validate_rows(
                &raw,
                usize::try_from(owner.atoms)?,
                step,
                r["active_decision"]["producer_epoch"]
                    .as_u64()
                    .context("ASC producer epoch")?,
            )?;
            rows_hash.update(&raw);
            records_hash.update(&encoded);
            count += 1;
            offset += raw.len() as u64;
            previous = step;
            work = next;
        }
        ensure!(
            count == state.endpoints
                && offset == state.byte_offset
                && offset == state.rows.committed_bytes
                && previous == state.previous_step
                && work.to_bits() == state.previous_work_bits,
            "ASC checkpoint counters differ from output prefix"
        );
        let mut allocation_rebinds = state.allocation_rebinds.clone();
        allocation_rebinds.push(AscAllocationRebind {
            previous_device_generation: state.active_device_generation,
            restored_device_generation: owner.generation,
            before_endpoint: state.endpoints,
            completed_physical_step: state.previous_step,
            rows_checkpoint_sha256: state.rows.committed_sha256.clone(),
            records_checkpoint_sha256: state.records.committed_sha256.clone(),
        });
        Ok(Self {
            dir,
            rows,
            records,
            run_id: state.run_id.clone(),
            stream_id: state.stream_id,
            schema_sha256: state.schema_sha256.clone(),
            generation: state.generation,
            active_device_generation: owner.generation,
            allocation_rebinds,
            previous_step: previous,
            previous_work: work,
            endpoints: count,
            byte_offset: offset,
            rows_hash,
            records_hash,
            finished: false,
        })
    }

    pub fn create(dir: &Path, run_id: &str, owner: &AscDevice) -> Result<Self> {
        ensure!(!run_id.trim().is_empty(), "ASC journal needs run identity");
        fs::create_dir(dir)
            .with_context(|| format!("create fresh ASC journal {}", dir.display()))?;
        let selected = owner.selected_support.as_ref();
        let schema = serde_json::json!({
            "schema_kind":"prism.isolated_asc_endpoint_journal.v1","run_id":run_id,
            "source_stream_id":owner.source,"allocation_generation":owner.generation,
            "topology_atom_order_sha256":owner.topology_sha256,"atoms":owner.atoms,"row_bytes":ROW_BYTES,
            "byte_order":"little_endian","content_identity":"ISOLATED_ASC_FORCE_AND_PRECONSTRAINT_KICK_DIFFERENCE",
            "support_mode":if selected.is_some(){"SELECTED_STRUCTURAL_ATOM_RESIDUE_SUPPORT"}else{"LEGACY_WHOLE_SYSTEM_ALL_ATOMS"},
            "control_alpha":owner.snapshot().map_err(|rc|anyhow::anyhow!("ASC schema owner snapshot rc={rc}"))?.alpha,
            "policy_source_sha256":selected.map(|v|v.policy_source_sha256()),
            "support_identity":selected.map_or(1,|v|v.support_identity()),
            "selected_atom_indices":selected.map(|v|v.atom_indices()),
            "support_source_evidence_sha256":selected.map(|v|v.source_evidence_sha256()),
            "selected_causal_support":false,
            "row_layout":[
                {"offset":0,"type":"u64[4]","field":"physical_epoch_force_n_force_np1_kick_n_kick_np1"},
                {"offset":32,"type":"u64[4]","field":"decision_epoch"},
                {"offset":64,"type":"f64[2]","field":"potential","units":"kcal/mol"},
                {"offset":80,"type":"f64","field":"control_change_work","units":"kcal/mol"},
                {"offset":88,"type":"f32[2][3]","field":"base_force","units":"kcal/mol/angstrom"},
                {"offset":112,"type":"f64[2][3]","field":"impulse","units":"kcal*ps/mol/angstrom"},
                {"offset":160,"type":"f64[2][3]","field":"isolated_ASC_force","units":"kcal/mol/angstrom"},
                {"offset":208,"type":"f64[2][3]","field":"torque_about_coordinate_origin","units":"kcal/mol"},
                {"offset":256,"type":"f64[2][3]","field":"velocity_delta","units":"angstrom/ps"},
                {"offset":304,"type":"f64[2]","field":"differential_kick_energy","units":"kcal/mol"}],
            "kick_scope":"same pre-constraint half-kick with/without ASC; scaling/clamp fixed; no thermostat/RATTLE attribution",
            "work_scope":"fixed-coordinate change of ASC control potential; complete successful physical seals; distinct from total external work",
            "durability":"rows and records synced at explicit closeout; absent seal means incomplete, never silently complete"});
        let mut schema_file = create_new(&dir.join("schema.json"))?;
        let schema_bytes = json_bytes(&schema)?;
        let schema_sha256 = hash(&schema_bytes);
        schema_file.write_all(&schema_bytes)?;
        schema_file.sync_all()?;
        Ok(Self {
            dir: dir.to_path_buf(),
            rows: create_new(&dir.join("atom_rows.bin"))?,
            records: create_new(&dir.join("endpoints.jsonl"))?,
            run_id: run_id.into(),
            stream_id: owner.source,
            schema_sha256,
            generation: owner.generation,
            active_device_generation: owner.generation,
            allocation_rebinds: Vec::new(),
            previous_step: 0,
            previous_work: 0.0,
            endpoints: 0,
            byte_offset: 0,
            rows_hash: Sha256::new(),
            records_hash: Sha256::new(),
            finished: false,
        })
    }
    pub fn append_endpoint(
        &mut self,
        owner: &AscDevice,
        completed_step: u64,
        chunk_idx: u64,
    ) -> Result<()> {
        ensure!(
            !self.finished
                && owner.source == self.stream_id
                && owner.generation == self.active_device_generation,
            "ASC journal owner changed or already sealed"
        );
        ensure!(
            completed_step > self.previous_step,
            "ASC duplicate or reversed measurement endpoint"
        );
        let (s, w, rows) = owner.isolated_snapshot(completed_step)?;
        let interval_steps = completed_step - self.previous_step;
        let interval_work = w.control_work_kcal_mol - self.previous_work;
        ensure!(
            interval_work.is_finite(),
            "ASC interval work subtraction is nonfinite"
        );
        let mut work_bytes = Vec::with_capacity(32);
        work_bytes.extend(w.magic.to_le_bytes());
        work_bytes.extend(w.completed_steps.to_le_bytes());
        work_bytes.extend(w.completed_epoch.to_le_bytes());
        work_bytes.extend(w.control_work_kcal_mol.to_le_bytes());
        let record = serde_json::json!({"schema_kind":"prism.isolated_asc_endpoint.v1",
            "run_id":self.run_id,"source_stream_id":self.stream_id,"allocation_generation":self.generation,
            "schema_sha256":self.schema_sha256,"device_allocation_generation":self.active_device_generation,
            "chunk_idx":chunk_idx,"completed_physical_step":completed_step,"physical_epoch":s.physical_epoch,
            "active_decision":s.active,"pending_decision":s.pending,"status_code":s.status,
            "phases":s.phases,"observed_atom_row_endpoints":[completed_step],
            "unobserved_intermediate_atom_row_steps":if interval_steps>1 {Some([self.previous_step+1,completed_step-1])}else{None},
            "unobserved_intermediate_atom_row_step_count":interval_steps-1,
            "work_interval_start_exclusive":self.previous_step,"work_interval_end_inclusive":completed_step,
            "work_interval_completed_step_count":interval_steps,"work_missing_physical_steps":0,
            "cumulative_control_work_before_kcal_mol":self.previous_work,
            "cumulative_control_work_after_kcal_mol":w.control_work_kcal_mol,
            "interval_control_work_kcal_mol":interval_work,
            "interval_work_arithmetic":"difference of f64 cumulative sums; includes every physical seal, not sampled-row summation",
            "cumulative_work_exact_bytes_hex":work_bytes.iter().map(|v|format!("{v:02x}")).collect::<String>(),
            "last_step_control_work_kcal_mol":s.control_work,
            "row_artifact":"atom_rows.bin","byte_offset":self.byte_offset,"byte_length":rows.len(),
            "row_sha256":hash(&rows),"row_bytes":ROW_BYTES,"atoms":s.n_atoms,
            "coverage_failures":s.coverage_failures});
        let record_bytes = json_bytes(&record)?;
        self.rows.write_all(&rows)?;
        self.records.write_all(&record_bytes)?;
        self.rows_hash.update(&rows);
        self.records_hash.update(&record_bytes);
        self.byte_offset += rows.len() as u64;
        self.previous_step = completed_step;
        self.previous_work = w.control_work_kcal_mol;
        self.endpoints += 1;
        Ok(())
    }
    pub fn finish(&mut self, expected_completed_step: u64) -> Result<()> {
        ensure!(
            !self.finished && self.endpoints > 0 && self.previous_step == expected_completed_step,
            "ASC journal has no final physical endpoint or was sealed twice"
        );
        self.rows.sync_all()?;
        self.records.sync_all()?;
        let seal = serde_json::json!({"schema_kind":"prism.isolated_asc_journal_seal.v1",
            "run_id":self.run_id,"source_stream_id":self.stream_id,"allocation_generation":self.generation,
            "schema_sha256":self.schema_sha256,"device_allocation_generation":self.active_device_generation,"allocation_rebinds":self.allocation_rebinds,
            "completed_physical_steps":self.previous_step,"observed_atom_row_endpoints":self.endpoints,
            "unobserved_atom_row_steps":self.previous_step-self.endpoints,
            "cumulative_control_work_kcal_mol":self.previous_work,"work_missing_physical_steps":0,
            "rows_bytes":self.byte_offset,"rows_sha256":format!("{:x}",self.rows_hash.clone().finalize()),
            "records_sha256":format!("{:x}",self.records_hash.clone().finalize()),"fsync_complete":true});
        let mut f = create_new(&self.dir.join("seal.json"))?;
        f.write_all(&json_bytes(&seal)?)?;
        f.sync_all()?;
        File::open(&self.dir)?.sync_all()?;
        self.finished = true;
        Ok(())
    }
}
/// Verify the sealed native journal before inventory or downstream attachment.
/// Raw rows remain a separate intervention channel, never substituted for AMBER.
pub fn verify_sealed_journal(dir: &Path, run_id: &str, stream: u32) -> Result<serde_json::Value> {
    use serde_json::{json, Value};
    fn file_pin(path: &Path) -> Result<Value> {
        let mut reader = BufReader::new(File::open(path)?);
        let mut digest = Sha256::new();
        let mut buffer = vec![0u8; 1024 * 1024];
        let mut size = 0u64;
        loop {
            let n = reader.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            digest.update(&buffer[..n]);
            size += n as u64;
        }
        Ok(json!({"path":path,"sha256":format!("{:x}",digest.finalize()),"size_bytes":size}))
    }
    let schema: Value = serde_json::from_reader(File::open(dir.join("schema.json"))?)?;
    let seal: Value = serde_json::from_reader(File::open(dir.join("seal.json"))?)?;
    ensure!(
        schema["schema_kind"] == "prism.isolated_asc_endpoint_journal.v1"
            && seal["schema_kind"] == "prism.isolated_asc_journal_seal.v1"
            && schema["byte_order"] == "little_endian"
            && schema["content_identity"] == "ISOLATED_ASC_FORCE_AND_PRECONSTRAINT_KICK_DIFFERENCE",
        "isolated ASC schema identity"
    );
    for record in [&schema, &seal] {
        ensure!(
            record["run_id"] == run_id
                && record["source_stream_id"] == stream
                && record["allocation_generation"] == schema["allocation_generation"],
            "isolated ASC source identity"
        );
    }
    ensure!(
        seal["fsync_complete"] == true && seal["work_missing_physical_steps"] == 0,
        "isolated ASC journal lacks completed durable coverage"
    );
    let mut files = serde_json::Map::new();
    for (name, field) in [
        ("schema.json", "schema_sha256"),
        ("atom_rows.bin", "rows_sha256"),
        ("endpoints.jsonl", "records_sha256"),
        ("seal.json", ""),
    ] {
        let pin = file_pin(&dir.join(name))?;
        if !field.is_empty() {
            ensure!(
                pin["sha256"] == seal[field],
                "isolated ASC sealed bytes changed: {name}"
            );
        }
        files.insert(name.into(), pin);
    }
    let atoms = schema["atoms"].as_u64().context("ASC atom dimension")?;
    let endpoints = seal["observed_atom_row_endpoints"]
        .as_u64()
        .context("ASC endpoint population")?;
    let expected = atoms
        .checked_mul(ROW_BYTES as u64)
        .and_then(|v| v.checked_mul(endpoints))
        .context("ASC byte count overflow")?;
    ensure!(
        atoms > 0
            && endpoints > 0
            && schema["row_bytes"] == ROW_BYTES
            && files["atom_rows.bin"]["size_bytes"] == expected
            && seal["rows_bytes"] == expected,
        "ASC row population mismatch"
    );
    let original_generation = schema["allocation_generation"]
        .as_u64()
        .context("ASC original generation")?;
    let allocation_rebinds: Vec<AscAllocationRebind> = serde_json::from_value(
        seal.get("allocation_rebinds")
            .cloned()
            .unwrap_or_else(|| serde_json::json!([])),
    )?;
    let final_generation = seal["device_allocation_generation"]
        .as_u64()
        .unwrap_or(original_generation);
    validate_generation_history(
        &allocation_rebinds,
        original_generation,
        final_generation,
        endpoints,
        seal["completed_physical_steps"]
            .as_u64()
            .context("ASC final step")?,
    )?;
    let mut intervention_support =
        RecordedInterventionSupport::from_schema(&schema, usize::try_from(atoms)?)?;
    let mut previous = 0u64;
    let mut count = 0u64;
    let mut offset = 0u64;
    let mut work = 0.0f64;
    let mut atom_rows = BufReader::new(File::open(dir.join("atom_rows.bin"))?);
    let mut raw = vec![0; usize::try_from(atoms * ROW_BYTES as u64)?];
    for line in BufReader::new(File::open(dir.join("endpoints.jsonl"))?).lines() {
        let row: Value = serde_json::from_str(&line?)?;
        let step = row["completed_physical_step"]
            .as_u64()
            .context("ASC endpoint step")?;
        let next = row["cumulative_control_work_after_kcal_mol"]
            .as_f64()
            .context("ASC cumulative work")?;
        ensure!(
            step > previous
                && next.is_finite()
                && row["schema_kind"] == "prism.isolated_asc_endpoint.v1"
                && row["row_artifact"] == "atom_rows.bin"
                && row["row_bytes"] == ROW_BYTES
                && row["run_id"] == run_id
                && row["source_stream_id"] == stream
                && row["allocation_generation"] == schema["allocation_generation"]
                && row["schema_sha256"] == seal["schema_sha256"]
                && row["physical_epoch"] == step
                && row["byte_offset"] == offset
                && row["byte_length"] == atoms * ROW_BYTES as u64
                && row["atoms"] == atoms
                && row["work_interval_start_exclusive"] == previous
                && row["work_interval_end_inclusive"] == step
                && row["work_interval_completed_step_count"] == step - previous
                && row["work_missing_physical_steps"] == 0
                && row["coverage_failures"] == 0
                && row["cumulative_control_work_before_kcal_mol"].as_f64() == Some(work)
                && row["interval_control_work_kcal_mol"].as_f64() == Some(next - work)
                && row["unobserved_intermediate_atom_row_step_count"] == step - previous - 1,
            "isolated ASC endpoint lineage/coverage mismatch"
        );
        ensure!(
            row["device_allocation_generation"]
                .as_u64()
                .unwrap_or(original_generation)
                == generation_for_endpoint(&allocation_rebinds, original_generation, count),
            "ASC endpoint device allocation generation lacks an explicit continuation binding"
        );
        atom_rows.read_exact(&mut raw)?;
        ensure!(
            row["row_sha256"] == hash(&raw),
            "isolated ASC endpoint row hash mismatch"
        );
        validate_rows(
            &raw,
            usize::try_from(atoms)?,
            step,
            row["active_decision"]["producer_epoch"]
                .as_u64()
                .context("ASC decision epoch")?,
        )?;
        intervention_support.observe(&raw)?;
        if intervention_support.control_alpha == Some(0.0) {
            ensure!(
                next == 0.0 && row["interval_control_work_kcal_mol"].as_f64() == Some(0.0),
                "zero-gain ASC control has nonzero recorded cumulative work"
            );
        }
        let mut expected_work = Vec::with_capacity(32);
        expected_work.extend(WORK_MAGIC.to_le_bytes());
        expected_work.extend(step.to_le_bytes());
        expected_work.extend(step.to_le_bytes());
        expected_work.extend(next.to_le_bytes());
        ensure!(
            row["cumulative_work_exact_bytes_hex"]
                == expected_work
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect::<String>(),
            "ASC cumulative work bytes disagree with endpoint"
        );
        previous = step;
        work = next;
        count += 1;
        offset += atoms * ROW_BYTES as u64;
    }
    ensure!(
        count == endpoints
            && seal["completed_physical_steps"] == previous
            && seal["unobserved_atom_row_steps"] == previous - count
            && seal["cumulative_control_work_kcal_mol"].as_f64() == Some(work),
        "ASC final endpoint differs from seal"
    );
    Ok(
        json!({"schema_kind":"prism.verified_isolated_asc_journal.v1","run_id":run_id,
        "source_stream_id":stream,"files":files,"seal":seal,"schema":schema,
        "intervention_support_check":intervention_support,
        "intervention_support_scope":"exact saved atom endpoints and recorded cumulative work; full vectors preserved; unsampled atom steps remain unsampled; observed nonzero dose is not demonstrated residue mediation",

        "scope":"source-bound isolated ASC endpoint rows and cumulative control-change work; no causal-response attribution inferred"}),
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    fn row(epoch: u64, decision: u64) -> Vec<u8> {
        let mut r = vec![0; ROW_BYTES];
        for i in 0..4 {
            r[i * 8..i * 8 + 8].copy_from_slice(&epoch.to_le_bytes());
            r[32 + i * 8..40 + i * 8].copy_from_slice(&decision.to_le_bytes());
        }
        r
    }
    #[test]
    fn zero_gain_and_unselected_support_validate_actual_isolated_channels() {
        use serde_json::json;
        let schema = json!({"control_alpha":0.0,"support_mode":"SELECTED_STRUCTURAL_ATOM_RESIDUE_SUPPORT","selected_atom_indices":[0]});
        let mut r = row(10, 1);
        r.extend(row(10, 1));
        r[88..92].copy_from_slice(&123.0f32.to_le_bytes());
        let mut audit = RecordedInterventionSupport::from_schema(&schema, 2).unwrap();
        audit.observe(&r).unwrap();
        assert_eq!(audit.observed_atom_rows, 2);
        assert_eq!(audit.observed_nonzero_force_atom_halves, 0);
        r[160..168].copy_from_slice(&1e-300f64.to_le_bytes());
        assert!(audit.observe(&r).is_err());
        let active = json!({"control_alpha":0.005,"support_mode":"SELECTED_STRUCTURAL_ATOM_RESIDUE_SUPPORT","selected_atom_indices":[0]});
        let mut audit = RecordedInterventionSupport::from_schema(&active, 2).unwrap();
        audit.observe(&r).unwrap();
        assert_eq!(audit.observed_nonzero_force_atom_halves, 1);
        r[ROW_BYTES + 112..ROW_BYTES + 120].copy_from_slice(&(-0.01f64).to_le_bytes());
        assert!(audit.observe(&r).is_err());
        assert!(RecordedInterventionSupport::from_schema(&json!({"support_mode":"SELECTED_STRUCTURAL_ATOM_RESIDUE_SUPPORT","selected_atom_indices":[0,0]}),2).is_err());
        assert!(RecordedInterventionSupport::from_schema(&json!({"support_mode":"SELECTED_STRUCTURAL_ATOM_RESIDUE_SUPPORT","selected_atom_indices":[2]}),2).is_err());
        let old = RecordedInterventionSupport::from_schema(&json!({}), 2).unwrap();
        assert!(old.control_alpha.is_none() && old.selected_atom_indices.is_none());
    }
    #[test]
    fn exact_rows_reject_missing_or_stale_kick_epoch() {
        let mut r = row(10, 1);
        validate_rows(&r, 1, 10, 1).unwrap();
        r[24..32].copy_from_slice(&9u64.to_le_bytes());
        assert!(validate_rows(&r, 1, 10, 1).is_err());
        assert!(validate_rows(&r[..319], 1, 10, 1).is_err());
    }
    #[test]
    fn exact_rows_preserve_finite_zero_and_reject_nonfinite_force() {
        let mut r = row(20, 10);
        validate_rows(&r, 1, 20, 10).unwrap();
        r[160..168].copy_from_slice(&f64::NAN.to_le_bytes());
        assert!(validate_rows(&r, 1, 20, 10).is_err());
    }
    #[test]
    fn decoded_measurement_retains_every_signed_field_and_ordered_half() {
        let mut r = row(20, 10);
        for offset in (64..88).step_by(8).chain((112..320).step_by(8)) {
            let value = -(offset as f64) / 8.0;
            r[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
        }
        for offset in (88..112).step_by(4) {
            let value = offset as f32 / 4.0;
            r[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        validate_rows(&r, 1, 20, 10).unwrap();
        let decoded = decode_atom(&r, 7, 12);
        assert_eq!(decoded.physical_epochs, [20; 4]);
        assert_eq!(decoded.decision_epochs, [10; 4]);
        assert_eq!((decoded.atom_index, decoded.analysis_residue_id), (7, 12));
        assert_eq!(decoded.potential_kcal_mol, [-8., -9.]);
        assert_eq!(decoded.control_change_work_kcal_mol, -10.);
        assert_eq!(
            decoded.base_force_kcal_mol_angstrom,
            [[22., 23., 24.], [25., 26., 27.]]
        );
        for (start, values) in [
            (14, decoded.impulse_kcal_ps_mol_angstrom),
            (20, decoded.isolated_asc_force_kcal_mol_angstrom),
            (26, decoded.torque_about_coordinate_origin_kcal_mol),
            (32, decoded.velocity_delta_angstrom_ps),
        ] {
            assert_eq!(
                values,
                std::array::from_fn(|half| std::array::from_fn(
                    |axis| -((start + half * 3 + axis) as f64)
                ))
            );
        }
        assert_eq!(decoded.differential_kick_energy_kcal_mol, [-38., -39.]);
        assert_eq!(
            serde_json::from_slice::<AscAtomMeasurement>(&serde_json::to_vec(&decoded).unwrap())
                .unwrap(),
            decoded
        );
    }

    #[test]
    #[ignore = "requires an explicitly pinned existing native journal, topology and fresh output; advances no MD"]
    fn saved_native_journal_reaches_residue_consumer_without_md() {
        let dir = PathBuf::from(std::env::var("PRISM_ASC_SAVED_JOURNAL").unwrap());
        let top = fs::read(std::env::var("PRISM_ASC_SAVED_TOPOLOGY").unwrap()).unwrap();
        let topology: serde_json::Value = serde_json::from_slice(&top).unwrap();
        let residues: Vec<i32> =
            serde_json::from_value(topology["atom_analysis_residue_ids"].clone()).unwrap();
        let run = std::env::var("PRISM_ASC_SAVED_RUN_ID").unwrap();
        let out = PathBuf::from(std::env::var("PRISM_ASC_CONSUMER_OUTPUT").unwrap());
        let value =
            export_isolated_asc_measurements(&dir, &run, 0, &hash(&top), &residues, &out).unwrap();
        assert!(value["endpoint_index"].as_array().unwrap().len() > 0);
        assert!(visit_isolated_asc_endpoints(
            &dir,
            &run,
            0,
            &"0".repeat(64),
            &residues,
            |_, _| Ok(())
        )
        .is_err());
        assert!(visit_isolated_asc_endpoints(
            &dir,
            &run,
            0,
            &hash(&top),
            &residues[..residues.len() - 1],
            |_, _| Ok(())
        )
        .is_err());
        let mut observed = 0;
        for record in
            BufReader::new(zstd::stream::read::Decoder::new(File::open(&out).unwrap()).unwrap())
                .lines()
        {
            let record: serde_json::Value = serde_json::from_str(&record.unwrap()).unwrap();
            let atoms: Vec<AscAtomMeasurement> =
                serde_json::from_value(record["atom_measurements"].clone()).unwrap();
            assert_eq!(atoms.len(), residues.len());
            for (i, atom) in atoms.iter().enumerate() {
                assert_eq!(atom.atom_index, i);
                assert_eq!(atom.analysis_residue_id, residues[i]);
                assert_eq!(
                    atom.physical_epochs,
                    [record["endpoint"]["completed_physical_step"]
                        .as_u64()
                        .unwrap(); 4]
                );
            }
            observed += 1;
        }
        assert_eq!(observed, value["endpoint_index"].as_array().unwrap().len());
        println!("{}", value);
    }
}
