use anyhow::{Context, Result};
use clap::Parser;
use serde::Deserialize;
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};

const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;
const LE_MARKER: u32 = 0x01020304;

#[derive(Parser, Debug)]
#[command(name = "prism-md-evidence-audit")]
#[command(
    about = "Deep audit PRISM MD-only evidence handoff files for teacher-substrate readiness."
)]
struct Args {
    #[arg(short, long)]
    manifest: PathBuf,

    #[arg(short, long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct Manifest {
    run_id: String,
    target: String,
    topology_input: String,
    stream_count: usize,
    streams_serialized: usize,
    total_spikes_md: u64,
    artifacts: Vec<ManifestArtifact>,
    validation_status: Option<String>,
    required_artifacts_complete: Option<bool>,
    missing_required_artifacts: Option<Vec<String>>,
    v2_was_live: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct ManifestArtifact {
    kind: String,
    stream_id: Option<u32>,
    path: String,
    size_bytes: u64,
    checksum_fnv1a_64_hex: Option<String>,
    record_count: Option<u64>,
    present: bool,
}

#[derive(Debug, Deserialize)]
struct TopologyMin {
    n_atoms: usize,
    n_residues: usize,
    positions: Vec<f32>,
    residue_ids: Vec<i32>,
}

#[derive(Debug)]
struct EnvelopeHeader {
    magic: [u8; 8],
    schema_version: u32,
    endian_marker: u32,
    stream_id: u32,
    run_id: String,
    stem: String,
    record_count: u64,
    byte_stride: u64,
    payload_size: u64,
    header_size: u64,
}

#[derive(Debug, Default)]
struct RunningF64 {
    n: u64,
    sum: f64,
    min: f64,
    max: f64,
}

impl RunningF64 {
    fn observe(&mut self, v: f64) {
        if self.n == 0 {
            self.min = v;
            self.max = v;
        } else {
            self.min = self.min.min(v);
            self.max = self.max.max(v);
        }
        self.n += 1;
        self.sum += v;
    }

    fn mean(&self) -> Option<f64> {
        (self.n > 0).then(|| self.sum / self.n as f64)
    }
}

#[derive(Debug, Default)]
struct RunningI64 {
    n: u64,
    sum: i128,
    min: i64,
    max: i64,
    nonzero: u64,
    negative: u64,
}

impl RunningI64 {
    fn observe(&mut self, v: i64) {
        if self.n == 0 {
            self.min = v;
            self.max = v;
        } else {
            self.min = self.min.min(v);
            self.max = self.max.max(v);
        }
        self.n += 1;
        self.sum += v as i128;
        if v != 0 {
            self.nonzero += 1;
        }
        if v < 0 {
            self.negative += 1;
        }
    }
}

fn read_u32_le<R: Read>(r: &mut R) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64_le<R: Read>(r: &mut R) -> Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_bytes<R: Read>(r: &mut R, n: usize) -> Result<Vec<u8>> {
    let mut v = vec![0u8; n];
    r.read_exact(&mut v)?;
    Ok(v)
}

fn read_envelope_header<R: Read>(r: &mut R) -> Result<EnvelopeHeader> {
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    let schema_version = read_u32_le(r)?;
    let endian_marker = read_u32_le(r)?;
    let stream_id = read_u32_le(r)?;
    let run_id_len = read_u64_le(r)? as usize;
    let run_id = String::from_utf8(read_bytes(r, run_id_len)?).context("run_id utf8")?;
    let stem_len = read_u64_le(r)? as usize;
    let stem = String::from_utf8(read_bytes(r, stem_len)?).context("stem utf8")?;
    let record_count = read_u64_le(r)?;
    let byte_stride = read_u64_le(r)?;
    let payload_size = read_u64_le(r)?;
    let header_size = 8 + 4 + 4 + 4 + 8 + run_id_len as u64 + 8 + stem_len as u64 + 8 + 8 + 8;
    Ok(EnvelopeHeader {
        magic,
        schema_version,
        endian_marker,
        stream_id,
        run_id,
        stem,
        record_count,
        byte_stride,
        payload_size,
        header_size,
    })
}

fn fnv_update(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

fn read_i32_at(buf: &[u8], offset: usize) -> i32 {
    let mut b = [0u8; 4];
    b.copy_from_slice(&buf[offset..offset + 4]);
    i32::from_le_bytes(b)
}

fn read_u32_at(buf: &[u8], offset: usize) -> u32 {
    let mut b = [0u8; 4];
    b.copy_from_slice(&buf[offset..offset + 4]);
    u32::from_le_bytes(b)
}

fn read_f32_at(buf: &[u8], offset: usize) -> f32 {
    let mut b = [0u8; 4];
    b.copy_from_slice(&buf[offset..offset + 4]);
    f32::from_le_bytes(b)
}

fn resolve_artifact_path(manifest_path: &Path, artifact_path: &str) -> PathBuf {
    let p = PathBuf::from(artifact_path);
    if p.is_absolute() {
        p
    } else {
        manifest_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(p)
    }
}

fn load_topology(path: &Path) -> Result<TopologyMin> {
    let s = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let t: TopologyMin =
        serde_json::from_str(&s).with_context(|| format!("parse {}", path.display()))?;
    anyhow::ensure!(
        t.positions.len() == t.n_atoms * 3,
        "topology positions len {} != 3*n_atoms {}",
        t.positions.len(),
        t.n_atoms
    );
    Ok(t)
}

fn count_run_log_warnings(run_dir: &Path) -> serde_json::Value {
    let log_path = run_dir.join("run.log");
    let Ok(text) = std::fs::read_to_string(&log_path) else {
        return json!({"present": false});
    };
    let warn_count = text.matches(" WARN ").count();
    let error_count = text.matches(" ERROR ").count();
    let signal_retry_count = text.matches("download_signal_preservation attempt").count();
    let signal_recovered_count = text
        .matches("download_signal_preservation recovered")
        .count();
    let kcc_retry_count = text.matches("compute_and_download_kcc attempt").count();
    json!({
        "present": true,
        "warn_count": warn_count,
        "error_count": error_count,
        "signal_grid_retry_warnings": signal_retry_count,
        "signal_grid_recovered": signal_recovered_count,
        "kcc_retry_warnings": kcc_retry_count,
    })
}

fn envelope_status_json(
    path: &Path,
    header: &EnvelopeHeader,
    expected_magic: &[u8; 8],
    expected_trailer: &[u8; 8],
    art: &ManifestArtifact,
    manifest: &Manifest,
    checksum: u64,
    stored_checksum: u64,
    trailer: &[u8; 8],
) -> serde_json::Value {
    let disk_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let declared_total_size = header.header_size + header.payload_size + 16;
    let checksum_hex = format!("{:016x}", checksum);
    let manifest_checksum_match = art
        .checksum_fnv1a_64_hex
        .as_ref()
        .map(|h| h.eq_ignore_ascii_case(&checksum_hex));
    json!({
        "path": path,
        "kind": art.kind,
        "stream_id": header.stream_id,
        "manifest_stream_id_match": art.stream_id.map(|id| id == header.stream_id),
        "magic": String::from_utf8_lossy(&header.magic),
        "magic_match": &header.magic == expected_magic,
        "schema_version": header.schema_version,
        "endian_marker_hex": format!("{:08x}", header.endian_marker),
        "endian_ok": header.endian_marker == LE_MARKER,
        "run_id_match": header.run_id == manifest.run_id,
        "stem": header.stem,
        "record_count": header.record_count,
        "manifest_record_count_match": art.record_count.map(|n| n == header.record_count),
        "byte_stride": header.byte_stride,
        "payload_size": header.payload_size,
        "declared_total_size": declared_total_size,
        "disk_size": disk_size,
        "size_match": disk_size == declared_total_size && art.size_bytes == disk_size,
        "payload_checksum_fnv1a_64_hex": checksum_hex,
        "stored_checksum_match": stored_checksum == checksum,
        "manifest_checksum_match": manifest_checksum_match,
        "trailer": String::from_utf8_lossy(trailer),
        "trailer_match": trailer == expected_trailer,
        "valid": &header.magic == expected_magic
            && header.endian_marker == LE_MARKER
            && header.run_id == manifest.run_id
            && disk_size == declared_total_size
            && stored_checksum == checksum
            && trailer == expected_trailer
            && art.record_count.map(|n| n == header.record_count).unwrap_or(true)
            && manifest_checksum_match.unwrap_or(true),
    })
}

fn audit_spikes(
    manifest_path: &Path,
    art: &ManifestArtifact,
    manifest: &Manifest,
    topology_n_residues: usize,
) -> Result<(serde_json::Value, serde_json::Value)> {
    let path = resolve_artifact_path(manifest_path, &art.path);
    let f = File::open(&path).with_context(|| format!("open {}", path.display()))?;
    let mut r = BufReader::with_capacity(1 << 20, f);
    let header = read_envelope_header(&mut r)?;
    // There is exactly ONE spike record layout: the sealed 96 B packet record
    // shared with the CUDA emitter (`PRISM_SPIKE_RECORD_BYTES`). A `>= 96`
    // floor accepted any stride at all above that size, which would have let
    // an unknown layout be audited at guessed offsets. Pin it to the one
    // layout this auditor actually understands.
    anyhow::ensure!(
        header.byte_stride == prism_nhs::fused_engine::PRISM_SPIKE_RECORD_BYTES as u64,
        "spike stride {} is not the sealed packet record stride ({})",
        header.byte_stride,
        prism_nhs::fused_engine::PRISM_SPIKE_RECORD_BYTES
    );
    anyhow::ensure!(
        header.payload_size == header.record_count * header.byte_stride,
        "spike payload size mismatch"
    );

    let mut checksum = FNV_OFFSET;
    let stride = header.byte_stride as usize;
    let records_per_batch = 1 << 15;
    let mut buf = vec![0u8; records_per_batch * stride];
    let mut remaining = header.record_count;

    let mut intensity = RunningF64::default();
    let mut water_density = RunningF64::default();
    let mut vibrational_energy = RunningF64::default();
    let mut wd_change = RunningF64::default();
    let mut timestep_min = i32::MAX;
    let mut timestep_max = i32::MIN;
    let mut voxel_min = i32::MAX;
    let mut voxel_max = i32::MIN;
    let mut bbox_min = [f32::INFINITY; 3];
    let mut bbox_max = [f32::NEG_INFINITY; 3];
    let mut source_counts: BTreeMap<String, u64> = BTreeMap::new();
    let mut wavelength_counts: BTreeMap<String, u64> = BTreeMap::new();
    let mut aromatic_counts: BTreeMap<String, u64> = BTreeMap::new();
    let mut phase_bits = BTreeSet::new();
    let mut nearby_unique = BTreeSet::new();
    let mut aromatic_residue_unique = BTreeSet::new();
    let mut invalid_nearby_residue_refs = 0u64;
    let mut invalid_aromatic_residue_refs = 0u64;
    let mut invalid_n_residues = 0u64;
    let mut finite_failures = 0u64;
    let mut n_nearby_excited = RunningI64::default();

    while remaining > 0 {
        let batch = remaining.min(records_per_batch as u64) as usize;
        let bytes = batch * stride;
        r.read_exact(&mut buf[..bytes])?;
        checksum = fnv_update(checksum, &buf[..bytes]);
        for rec in buf[..bytes].chunks_exact(stride) {
            let timestep = read_i32_at(rec, 0);
            let voxel_idx = read_i32_at(rec, 4);
            let px = read_f32_at(rec, 8);
            let py = read_f32_at(rec, 12);
            let pz = read_f32_at(rec, 16);
            let inten = read_f32_at(rec, 20);
            let n_res = read_i32_at(rec, 56);
            let source = read_i32_at(rec, 60);
            let wavelength = read_f32_at(rec, 64);
            let aromatic_type = read_i32_at(rec, 68);
            let aromatic_residue_id = read_i32_at(rec, 72);
            let wd = read_f32_at(rec, 76);
            let ve = read_f32_at(rec, 80);
            let nne = read_i32_at(rec, 84);
            let wdc = read_f32_at(rec, 88);
            let phase = read_u32_at(rec, 92);

            if !px.is_finite()
                || !py.is_finite()
                || !pz.is_finite()
                || !inten.is_finite()
                || !wavelength.is_finite()
                || !wd.is_finite()
                || !ve.is_finite()
                || !wdc.is_finite()
            {
                finite_failures += 1;
                continue;
            }

            timestep_min = timestep_min.min(timestep);
            timestep_max = timestep_max.max(timestep);
            voxel_min = voxel_min.min(voxel_idx);
            voxel_max = voxel_max.max(voxel_idx);
            bbox_min[0] = bbox_min[0].min(px);
            bbox_min[1] = bbox_min[1].min(py);
            bbox_min[2] = bbox_min[2].min(pz);
            bbox_max[0] = bbox_max[0].max(px);
            bbox_max[1] = bbox_max[1].max(py);
            bbox_max[2] = bbox_max[2].max(pz);
            intensity.observe(inten as f64);
            water_density.observe(wd as f64);
            vibrational_energy.observe(ve as f64);
            wd_change.observe(wdc as f64);
            n_nearby_excited.observe(nne as i64);
            phase_bits.insert(phase);

            let source_key = match source {
                1 => "UV",
                2 => "LIF",
                _ => "OTHER",
            };
            *source_counts.entry(source_key.to_string()).or_default() += 1;
            if wavelength > 0.0 {
                *wavelength_counts
                    .entry(format!("{:.1}", wavelength))
                    .or_default() += 1;
            }
            let arom_key = match aromatic_type {
                0 => "TRP",
                1 => "TYR",
                2 => "PHE",
                3 => "SS",
                -1 => "none",
                _ => "other",
            };
            *aromatic_counts.entry(arom_key.to_string()).or_default() += 1;
            if aromatic_residue_id >= 0 {
                if (aromatic_residue_id as usize) < topology_n_residues {
                    aromatic_residue_unique.insert(aromatic_residue_id);
                } else {
                    invalid_aromatic_residue_refs += 1;
                }
            }

            if !(0..=8).contains(&n_res) {
                invalid_n_residues += 1;
            }
            let n_nearby = n_res.clamp(0, 8) as usize;
            for k in 0..n_nearby {
                let rid = read_i32_at(rec, 24 + k * 4);
                if rid >= 0 && (rid as usize) < topology_n_residues {
                    nearby_unique.insert(rid);
                } else {
                    invalid_nearby_residue_refs += 1;
                }
            }
        }
        remaining -= batch as u64;
    }

    let mut stored = [0u8; 8];
    let mut trailer = [0u8; 8];
    r.read_exact(&mut stored)?;
    r.read_exact(&mut trailer)?;
    let stored_checksum = u64::from_le_bytes(stored);
    let envelope = envelope_status_json(
        &path,
        &header,
        b"PRSPK001",
        b"PRSPKEND",
        art,
        manifest,
        checksum,
        stored_checksum,
        &trailer,
    );

    let summary = json!({
        "stream_id": header.stream_id,
        "records": header.record_count,
        "timestep_min": timestep_min,
        "timestep_max": timestep_max,
        "voxel_idx_min": voxel_min,
        "voxel_idx_max": voxel_max,
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "intensity": {"min": intensity.min, "max": intensity.max, "mean": intensity.mean()},
        "water_density": {"min": water_density.min, "max": water_density.max, "mean": water_density.mean()},
        "vibrational_energy": {"min": vibrational_energy.min, "max": vibrational_energy.max, "mean": vibrational_energy.mean()},
        "wd_change": {"min": wd_change.min, "max": wd_change.max, "mean": wd_change.mean()},
        "source_counts": source_counts,
        "wavelength_nm_counts": wavelength_counts,
        "aromatic_counts": aromatic_counts,
        "phase_bits_unique_count": phase_bits.len(),
        "phase_bits_min": phase_bits.first().copied(),
        "phase_bits_max": phase_bits.last().copied(),
        "nearby_residue_unique_count": nearby_unique.len(),
        "aromatic_residue_unique_count": aromatic_residue_unique.len(),
        "invalid_nearby_residue_refs": invalid_nearby_residue_refs,
        "invalid_aromatic_residue_refs": invalid_aromatic_residue_refs,
        "invalid_n_residues": invalid_n_residues,
        "finite_failures": finite_failures,
        "n_nearby_excited": {
            "min": n_nearby_excited.min,
            "max": n_nearby_excited.max,
            "nonzero": n_nearby_excited.nonzero,
        },
    });
    Ok((envelope, summary))
}

fn read_payload_and_tail(
    manifest_path: &Path,
    art: &ManifestArtifact,
) -> Result<(PathBuf, EnvelopeHeader, Vec<u8>, u64, [u8; 8])> {
    let path = resolve_artifact_path(manifest_path, &art.path);
    let f = File::open(&path).with_context(|| format!("open {}", path.display()))?;
    let mut r = BufReader::with_capacity(1 << 20, f);
    let header = read_envelope_header(&mut r)?;
    let payload = read_bytes(&mut r, header.payload_size as usize)?;
    let mut stored = [0u8; 8];
    let mut trailer = [0u8; 8];
    r.read_exact(&mut stored)?;
    r.read_exact(&mut trailer)?;
    Ok((path, header, payload, u64::from_le_bytes(stored), trailer))
}

fn audit_signal_grid(
    manifest_path: &Path,
    art: &ManifestArtifact,
    manifest: &Manifest,
    topology_n_residues: usize,
) -> Result<(serde_json::Value, serde_json::Value)> {
    let (path, header, payload, stored_checksum, trailer) =
        read_payload_and_tail(manifest_path, art)?;
    let checksum = fnv_update(FNV_OFFSET, &payload);
    anyhow::ensure!(payload.len() >= 16, "signal payload too small");
    let grid_dim = u64::from_le_bytes(payload[0..8].try_into().unwrap());
    let voxel_count = u64::from_le_bytes(payload[8..16].try_into().unwrap());
    let expected_payload = 16 + 4 * voxel_count as usize * 4;
    anyhow::ensure!(
        payload.len() == expected_payload,
        "signal payload layout mismatch"
    );

    let mut hit = RunningI64::default();
    let mut coupled = RunningI64::default();
    let mut primary_count = RunningI64::default();
    let mut valid_primary_voxels = 0u64;
    let mut invalid_primary_residue_ids = 0u64;
    let mut primary_count_without_residue = 0u64;
    let mut unique_primary = BTreeSet::new();

    let base_hit = 16;
    let base_coupled = base_hit + voxel_count as usize * 4;
    let base_primary_id = base_coupled + voxel_count as usize * 4;
    let base_primary_count = base_primary_id + voxel_count as usize * 4;
    for i in 0..voxel_count as usize {
        let h = i32::from_le_bytes(
            payload[base_hit + i * 4..base_hit + i * 4 + 4]
                .try_into()
                .unwrap(),
        );
        let c = i32::from_le_bytes(
            payload[base_coupled + i * 4..base_coupled + i * 4 + 4]
                .try_into()
                .unwrap(),
        );
        let rid = i32::from_le_bytes(
            payload[base_primary_id + i * 4..base_primary_id + i * 4 + 4]
                .try_into()
                .unwrap(),
        );
        let pc = i32::from_le_bytes(
            payload[base_primary_count + i * 4..base_primary_count + i * 4 + 4]
                .try_into()
                .unwrap(),
        );
        hit.observe(h as i64);
        coupled.observe(c as i64);
        primary_count.observe(pc as i64);
        if rid >= 0 {
            if (rid as usize) < topology_n_residues {
                valid_primary_voxels += 1;
                unique_primary.insert(rid);
            } else {
                invalid_primary_residue_ids += 1;
            }
        }
        if pc > 0 && !(rid >= 0 && (rid as usize) < topology_n_residues) {
            primary_count_without_residue += 1;
        }
    }

    let envelope = envelope_status_json(
        &path,
        &header,
        b"PRSGD001",
        b"PRSGDEND",
        art,
        manifest,
        checksum,
        stored_checksum,
        &trailer,
    );
    let summary = json!({
        "stream_id": header.stream_id,
        "grid_dim": grid_dim,
        "voxel_count": voxel_count,
        "grid_dim_cubed_matches_voxel_count": grid_dim * grid_dim * grid_dim == voxel_count,
        "voxel_hit_grid": {"sum": hit.sum.to_string(), "nonzero": hit.nonzero, "max": hit.max, "negative": hit.negative},
        "coupled_spike_grid": {"sum": coupled.sum.to_string(), "nonzero": coupled.nonzero, "max": coupled.max, "negative": coupled.negative},
        "primary_residue_count": {"sum": primary_count.sum.to_string(), "nonzero": primary_count.nonzero, "max": primary_count.max, "negative": primary_count.negative},
        "primary_residue_id": {
            "valid_voxels": valid_primary_voxels,
            "unique_valid_residues": unique_primary.len(),
            "invalid_residue_ids": invalid_primary_residue_ids,
            "primary_count_without_valid_residue": primary_count_without_residue,
        }
    });
    Ok((envelope, summary))
}

fn audit_kcc(
    manifest_path: &Path,
    art: &ManifestArtifact,
    manifest: &Manifest,
    topology_n_residues: usize,
) -> Result<(serde_json::Value, serde_json::Value)> {
    let (path, header, payload, stored_checksum, trailer) =
        read_payload_and_tail(manifest_path, art)?;
    let checksum = fnv_update(FNV_OFFSET, &payload);
    anyhow::ensure!(payload.len() >= 16, "kcc payload too small");
    let mut cursor = 0usize;
    let take_u64 = |payload: &[u8], cursor: &mut usize| -> Result<u64> {
        anyhow::ensure!(*cursor + 8 <= payload.len(), "kcc truncated u64");
        let v = u64::from_le_bytes(payload[*cursor..*cursor + 8].try_into().unwrap());
        *cursor += 8;
        Ok(v)
    };
    let n_residues = take_u64(&payload, &mut cursor)? as usize;
    let field_count = take_u64(&payload, &mut cursor)?;
    let mut fields = BTreeMap::new();
    let mut active_causal_nonzero = 0u64;
    let mut active_causal_sum: u128 = 0;
    let mut active_causal_max = 0u32;
    let mut residue_count_nonzero = 0u64;
    let mut residue_count_sum: u128 = 0;
    let mut finite_failures = 0u64;
    let mut nullable_nonfinite_values = 0u64;
    for _ in 0..field_count {
        let name_len = take_u64(&payload, &mut cursor)? as usize;
        anyhow::ensure!(
            cursor + name_len <= payload.len(),
            "kcc truncated field name"
        );
        let name = String::from_utf8(payload[cursor..cursor + name_len].to_vec())?;
        cursor += name_len;
        anyhow::ensure!(cursor < payload.len(), "kcc truncated dtype");
        let dtype = payload[cursor];
        cursor += 1;
        let section_size = take_u64(&payload, &mut cursor)? as usize;
        anyhow::ensure!(
            cursor + section_size <= payload.len(),
            "kcc truncated section"
        );
        let section = &payload[cursor..cursor + section_size];
        cursor += section_size;
        anyhow::ensure!(
            section_size == n_residues * 4,
            "kcc field {} wrong size",
            name
        );
        if dtype == 1 {
            let mut stats = RunningF64::default();
            let mut nonzero = 0u64;
            let mut null_count = 0u64;
            let nullable_nonfinite = matches!(name.as_str(), "causal_lag" | "lag_corr_peak");
            for k in 0..n_residues {
                let v = f32::from_le_bytes(section[k * 4..k * 4 + 4].try_into().unwrap());
                if v.is_finite() {
                    stats.observe(v as f64);
                    if v != 0.0 {
                        nonzero += 1;
                    }
                } else if nullable_nonfinite {
                    null_count += 1;
                    nullable_nonfinite_values += 1;
                } else {
                    finite_failures += 1;
                }
            }
            fields.insert(
                name,
                json!({
                    "dtype": "f32",
                    "nonzero": nonzero,
                    "null_count": null_count,
                    "nullable_nonfinite": nullable_nonfinite,
                    "min": stats.min,
                    "max": stats.max,
                    "mean": stats.mean(),
                }),
            );
        } else if dtype == 2 {
            let mut stats = RunningI64::default();
            for k in 0..n_residues {
                let v = u32::from_le_bytes(section[k * 4..k * 4 + 4].try_into().unwrap());
                stats.observe(v as i64);
                if name == "active_causal" {
                    if v != 0 {
                        active_causal_nonzero += 1;
                    }
                    active_causal_sum += v as u128;
                    active_causal_max = active_causal_max.max(v);
                }
                if name == "residue_count" {
                    if v != 0 {
                        residue_count_nonzero += 1;
                    }
                    residue_count_sum += v as u128;
                }
            }
            fields.insert(
                name,
                json!({
                    "dtype": "u32",
                    "nonzero": stats.nonzero,
                    "min": stats.min,
                    "max": stats.max,
                    "sum": stats.sum.to_string(),
                }),
            );
        } else {
            fields.insert(
                name,
                json!({"dtype": format!("unknown_{}", dtype), "section_size": section_size}),
            );
        }
    }

    let envelope = envelope_status_json(
        &path,
        &header,
        b"PRKCC001",
        b"PRKCCEND",
        art,
        manifest,
        checksum,
        stored_checksum,
        &trailer,
    );
    let summary = json!({
        "stream_id": header.stream_id,
        "n_residues": n_residues,
        "topology_n_residues_match": n_residues == topology_n_residues,
        "field_count": field_count,
        "finite_failures": finite_failures,
        "nullable_nonfinite_values": nullable_nonfinite_values,
        "active_causal": {
            "nonzero_residues": active_causal_nonzero,
            "sum": active_causal_sum.to_string(),
            "max": active_causal_max,
        },
        "residue_count": {
            "nonzero_residues": residue_count_nonzero,
            "sum": residue_count_sum.to_string(),
        },
        "fields": fields,
    });
    Ok((envelope, summary))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let manifest_text = std::fs::read_to_string(&args.manifest)
        .with_context(|| format!("read manifest {}", args.manifest.display()))?;
    let manifest: Manifest = serde_json::from_str(&manifest_text).context("parse manifest")?;
    let run_dir = args.manifest.parent().unwrap_or_else(|| Path::new("."));
    let topology_path = PathBuf::from(&manifest.topology_input);
    let topology = load_topology(&topology_path)?;

    let mut envelopes = Vec::new();
    let mut spike_streams = Vec::new();
    let mut signal_streams = Vec::new();
    let mut kcc_streams = Vec::new();
    let mut errors = Vec::new();

    for art in manifest.artifacts.iter().filter(|a| a.present) {
        let result = match art.kind.as_str() {
            "spikes" => {
                audit_spikes(&args.manifest, art, &manifest, topology.n_residues).map(|(e, s)| {
                    envelopes.push(e);
                    spike_streams.push(s);
                })
            }
            "signal_grid" => audit_signal_grid(&args.manifest, art, &manifest, topology.n_residues)
                .map(|(e, s)| {
                    envelopes.push(e);
                    signal_streams.push(s);
                }),
            "kcc_v2full" => {
                audit_kcc(&args.manifest, art, &manifest, topology.n_residues).map(|(e, s)| {
                    envelopes.push(e);
                    kcc_streams.push(s);
                })
            }
            _ => Ok(()),
        };
        if let Err(e) = result {
            errors.push(json!({"kind": art.kind, "path": art.path, "error": e.to_string()}));
        }
    }

    spike_streams.sort_by_key(|v| v["stream_id"].as_u64().unwrap_or(u64::MAX));
    signal_streams.sort_by_key(|v| v["stream_id"].as_u64().unwrap_or(u64::MAX));
    kcc_streams.sort_by_key(|v| v["stream_id"].as_u64().unwrap_or(u64::MAX));

    let total_spikes: u64 = spike_streams
        .iter()
        .map(|v| v["records"].as_u64().unwrap_or(0))
        .sum();
    let envelope_invalid = envelopes
        .iter()
        .filter(|v| v["valid"].as_bool() != Some(true))
        .count();

    let mut source_totals: BTreeMap<String, u64> = BTreeMap::new();
    let mut wavelength_totals: BTreeMap<String, u64> = BTreeMap::new();
    let mut aromatic_totals: BTreeMap<String, u64> = BTreeMap::new();
    let mut invalid_spike_refs = 0u64;
    let mut spike_finite_failures = 0u64;
    let mut total_phase_unique_across_streams = BTreeSet::new();
    for s in &spike_streams {
        if let Some(obj) = s["source_counts"].as_object() {
            for (k, v) in obj {
                *source_totals.entry(k.clone()).or_default() += v.as_u64().unwrap_or(0);
            }
        }
        if let Some(obj) = s["wavelength_nm_counts"].as_object() {
            for (k, v) in obj {
                *wavelength_totals.entry(k.clone()).or_default() += v.as_u64().unwrap_or(0);
            }
        }
        if let Some(obj) = s["aromatic_counts"].as_object() {
            for (k, v) in obj {
                *aromatic_totals.entry(k.clone()).or_default() += v.as_u64().unwrap_or(0);
            }
        }
        invalid_spike_refs += s["invalid_nearby_residue_refs"].as_u64().unwrap_or(0)
            + s["invalid_aromatic_residue_refs"].as_u64().unwrap_or(0)
            + s["invalid_n_residues"].as_u64().unwrap_or(0);
        spike_finite_failures += s["finite_failures"].as_u64().unwrap_or(0);
        if let Some(min) = s["phase_bits_min"].as_u64() {
            total_phase_unique_across_streams.insert(min);
        }
        if let Some(max) = s["phase_bits_max"].as_u64() {
            total_phase_unique_across_streams.insert(max);
        }
    }

    let signal_invalid_residue_ids: u64 = signal_streams
        .iter()
        .map(|s| {
            s["primary_residue_id"]["invalid_residue_ids"]
                .as_u64()
                .unwrap_or(0)
        })
        .sum();
    let kcc_finite_failures: u64 = kcc_streams
        .iter()
        .map(|s| s["finite_failures"].as_u64().unwrap_or(0))
        .sum();
    let kcc_topology_mismatches = kcc_streams
        .iter()
        .filter(|s| s["topology_n_residues_match"].as_bool() != Some(true))
        .count();

    let ready = envelope_invalid == 0
        && errors.is_empty()
        && spike_streams.len() == manifest.stream_count
        && signal_streams.len() == manifest.stream_count
        && kcc_streams.len() == manifest.stream_count
        && total_spikes == manifest.total_spikes_md
        && invalid_spike_refs == 0
        && signal_invalid_residue_ids == 0
        && spike_finite_failures == 0
        && kcc_finite_failures == 0
        && kcc_topology_mismatches == 0;

    let mut caveats = Vec::new();
    if manifest.v2_was_live != Some(true) {
        caveats.push("v2_was_live=false: graph/site centroid products are not part of this MD-only substrate");
    }
    if total_phase_unique_across_streams.len() <= 2 {
        caveats.push("phase_bits show little variation in this fast scan; usable as protocol-tagged spike evidence, not as rich phase-manifold training alone");
    }
    let run_log = count_run_log_warnings(run_dir);
    if run_log["signal_grid_retry_warnings"].as_u64().unwrap_or(0) > 0 {
        caveats.push("signal_grid teardown required retry recovery; payload validated, but CUDA status source still needs hardening");
    }

    let report = json!({
        "schema_version": 1,
        "schema_kind": "prism_md_evidence_deep_audit",
        "manifest_path": args.manifest,
        "run_dir": run_dir,
        "run_id": manifest.run_id,
        "target": manifest.target,
        "validation_status": manifest.validation_status,
        "required_artifacts_complete": manifest.required_artifacts_complete,
        "missing_required_artifacts": manifest.missing_required_artifacts,
        "topology": {
            "path": topology_path,
            "n_atoms": topology.n_atoms,
            "n_residues": topology.n_residues,
            "residue_ids_count": topology.residue_ids.len(),
        },
        "manifest_counts": {
            "stream_count": manifest.stream_count,
            "streams_serialized": manifest.streams_serialized,
            "total_spikes_md": manifest.total_spikes_md,
        },
        "envelope_validation": {
            "validated": envelopes.len(),
            "invalid": envelope_invalid,
            "errors": errors,
            "artifacts": envelopes,
        },
        "spike_payload": {
            "streams": spike_streams.len(),
            "total_records": total_spikes,
            "matches_manifest_total": total_spikes == manifest.total_spikes_md,
            "source_totals": source_totals,
            "wavelength_nm_totals": wavelength_totals,
            "aromatic_totals": aromatic_totals,
            "invalid_reference_count": invalid_spike_refs,
            "finite_failures": spike_finite_failures,
            "streams_detail": spike_streams,
        },
        "signal_grid_payload": {
            "streams": signal_streams.len(),
            "invalid_primary_residue_ids": signal_invalid_residue_ids,
            "streams_detail": signal_streams,
        },
        "kcc_v2full_payload": {
            "streams": kcc_streams.len(),
            "finite_failures": kcc_finite_failures,
            "topology_residue_mismatches": kcc_topology_mismatches,
            "streams_detail": kcc_streams,
        },
        "run_log": run_log,
        "teacher_substrate_readiness": {
            "ready_for_next_phase": ready,
            "basis": [
                "all PRSPK/PRSGD/PRKCC envelopes validate against checksum/trailer/header",
                "spike record count matches manifest",
                "spike residue references are in topology range",
                "signal grid primary residue ids are in topology range",
                "KCC residue count matches topology",
                "all parsed numeric payloads are finite"
            ],
            "caveats": caveats,
        }
    });

    let output = args
        .output
        .unwrap_or_else(|| run_dir.join("md_evidence_deep_audit.json"));
    let mut f = File::create(&output).with_context(|| format!("create {}", output.display()))?;
    serde_json::to_writer_pretty(&mut f, &report)?;
    f.write_all(b"\n")?;
    eprintln!(
        "wrote {} ready={} envelopes={} spikes={}",
        output.display(),
        ready,
        envelopes.len(),
        total_spikes
    );
    Ok(())
}
