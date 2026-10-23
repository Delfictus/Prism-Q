//! Stage-B teacher tag materializer for PRISM Twin MD-only evidence.
//!
//! This binary consumes the raw evidence handoff produced by `nhs_rt_full
//! --md-only-evidence`:
//!
//! - PRSPK001 per-stream spike envelopes
//! - PRSGD001 per-stream signal-grid envelopes
//! - PRKCC001 per-stream KCC v2-full envelopes
//! - protocol_state / ghost_time sidecars through the manifest run directory
//!
//! It deliberately does not require `binding_sites.json`, LIGSITE, or legacy
//! site centroids.  The primary label object is a reproducible residue support
//! region.  Centroid/AABB views are derived diagnostics after support is
//! proven, not inputs to the label.

use anyhow::{bail, Context, Result};
use arrow_array::{builder::Float64Builder, builder::Int32Builder, ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use clap::Parser;
use parquet::{
    arrow::ArrowWriter,
    basic::{Compression, ZstdLevel},
    file::properties::WriterProperties,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs::File,
    io::{BufReader, Read},
    path::{Path, PathBuf},
    sync::Arc,
};

const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;
const LE_MARKER: u32 = 0x01020304;
const UV_WAVELENGTHS: [f32; 5] = [280.0, 274.0, 258.0, 254.0, 211.0];

#[derive(Parser, Debug)]
#[command(name = "prism-teacher-tags")]
#[command(about = "Materialize PRISM Twin MD-only evidence into Stage-B teacher tags")]
struct Args {
    #[arg(long)]
    manifest: PathBuf,

    #[arg(long)]
    output_dir: PathBuf,

    #[arg(long, default_value_t = 64)]
    top_regions: usize,

    #[arg(long, default_value_t = 3)]
    min_region_size: usize,

    #[arg(long, default_value_t = 0.20)]
    top_residue_fraction: f64,

    #[arg(long, default_value_t = 8.0)]
    cluster_distance_a: f64,

    #[arg(long)]
    replica_record: Option<PathBuf>,

    #[arg(long, default_value_t = false)]
    no_update_replica_record: bool,
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
    run_config: Option<Value>,
}

#[derive(Debug, Deserialize, Clone)]
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
    #[serde(default)]
    residue_ids: Vec<i32>,
    #[serde(default)]
    ca_indices: Vec<usize>,
}

#[derive(Debug, Deserialize, Default, Clone)]
struct ProtocolStateSidecar {
    #[serde(default)]
    cold_hold_end: i32,
    #[serde(default)]
    ramp_end: i32,
    #[serde(default)]
    warm_hold_end: i32,
    #[serde(default)]
    ramp_down_end: i32,
    #[serde(default)]
    current_step: i32,
    #[serde(default)]
    scan_wavelengths_nm: Vec<f32>,
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

#[derive(Debug, Default, Clone)]
struct ResidueAccum {
    spike_hits: u64,
    total_intensity: f64,
    max_intensity: f64,
    water_sum: f64,
    wd_abs_sum: f64,
    vibrational_sum: f64,
    nearby_excited_sum: f64,
    phase_popcount_sum: u64,
    source_counts: [u64; 6],
    mechanism_counts: [u64; 7],
    wavelength_counts: [u64; 5],
    stream_counts: Vec<u64>,
    group_counts: [u64; 4],
    protocol_phase_counts: [u64; 5],
    phase_bit_counts: [u64; 4],
    aromatic_hits: u64,
    signal_voxel_hits: u64,
    signal_primary_count: u64,
    signal_coupled_count: u64,
}

#[derive(Debug, Default, Clone)]
struct KccMerged {
    temporal_corr: f64,
    direction_score: f64,
    motion_efficiency: f64,
    burst_motion: f64,
    phase_shift: f64,
    causal_lag: f64,
    lag_corr_peak: f64,
    local_cov: f64,
    net_dx: f64,
    net_dy: f64,
    net_dz: f64,
    sum_m: f64,
    residue_count: f64,
    active_causal: f64,
    selected_stream: u32,
}

#[derive(Debug)]
struct KccStream {
    stream_id: u32,
    n_residues: usize,
    fields: HashMap<String, Vec<f64>>,
}

#[derive(Debug, Default, Clone, Serialize)]
struct ScoreComponents {
    spike: f64,
    signal: f64,
    coupled: f64,
    kcc: f64,
    stream_consensus: f64,
    group_consensus: f64,
    uv_lif_concordance: f64,
    phase_entropy: f64,
    desolvation: f64,
    source_diversity: f64,
}

#[derive(Debug, Default, Clone)]
struct ResidueLabels {
    region_id: i32,
    max_phase_manifold_score: f64,
    top1_site_score: f64,
    top1_site_classification: f64,
    top1_site_rank: f64,
    is_in_top1_phase_site: f64,
    is_in_top5_phase_site: f64,
    is_in_top10_phase_site: f64,
    n_sites_containing_residue: f64,
    is_all_region: f64,
    n_sites_as_all_region: f64,
    is_lining: f64,
    n_sites_as_lining: f64,
    is_kcc_driver: f64,
    n_sites_as_kcc_driver: f64,
    is_hot_phase: f64,
    n_sites_as_hot_phase: f64,
    is_cold_phase: f64,
    n_sites_as_cold_phase: f64,
    is_burst_motion: f64,
    n_sites_as_burst_motion: f64,
    cryptic_likelihood_proxy: f64,
}

#[derive(Debug, Clone, Serialize)]
struct Region {
    region_id: i32,
    rank: usize,
    final_phase_manifold_score: f64,
    cryptic_likelihood_proxy: f64,
    support_residues: Vec<i32>,
    lining_or_surface_residues: Vec<i32>,
    kcc_driver_residues: Vec<i32>,
    hot_phase_supported_residues: Vec<i32>,
    cold_phase_supported_residues: Vec<i32>,
    burst_motion_supported_residues: Vec<i32>,
    score_components: ScoreComponents,
    centroid_views: Value,
}

fn main() -> Result<()> {
    let args = Args::parse();
    std::fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("create {}", args.output_dir.display()))?;

    let manifest_text = std::fs::read_to_string(&args.manifest)
        .with_context(|| format!("read manifest {}", args.manifest.display()))?;
    let manifest: Manifest = serde_json::from_str(&manifest_text)
        .with_context(|| format!("parse manifest {}", args.manifest.display()))?;

    if manifest.required_artifacts_complete != Some(true) {
        bail!(
            "manifest required_artifacts_complete is not true: {:?}",
            manifest.required_artifacts_complete
        );
    }
    if manifest.streams_serialized != manifest.stream_count {
        bail!(
            "manifest streams_serialized {} != stream_count {}",
            manifest.streams_serialized,
            manifest.stream_count
        );
    }

    let topology_path = resolve_existing_input_path(&args.manifest, &manifest.topology_input);
    let topology = load_topology(&topology_path)?;
    let ca_xyz = residue_ca_positions(&topology);
    let residue_keys = residue_output_ids(&topology);

    let mut accum = (0..topology.n_residues)
        .map(|_| ResidueAccum {
            stream_counts: vec![0; manifest.stream_count],
            ..ResidueAccum::default()
        })
        .collect::<Vec<_>>();

    let protocols = load_protocol_sidecars(&args.manifest, &manifest);

    let mut kcc_streams = Vec::new();
    for art in artifacts_by_kind(&manifest, "kcc_v2full") {
        kcc_streams.push(read_kcc_stream(&args.manifest, art, &manifest)?);
    }
    let kcc = merge_kcc(&kcc_streams, topology.n_residues);

    for art in artifacts_by_kind(&manifest, "signal_grid") {
        apply_signal_grid(
            &args.manifest,
            art,
            &manifest,
            topology.n_residues,
            &mut accum,
        )?;
    }
    for art in artifacts_by_kind(&manifest, "spikes") {
        apply_spikes(
            &args.manifest,
            art,
            &manifest,
            topology.n_residues,
            &protocols,
            &mut accum,
        )?;
    }

    let scores = score_residues(&accum, &kcc, manifest.stream_count);
    let mut regions = build_regions(&args, &accum, &kcc, &scores, &ca_xyz, manifest.stream_count);
    regions.truncate(args.top_regions);
    let labels = assign_region_labels(topology.n_residues, &regions);

    let parquet_path = args.output_dir.join("teacher_residue_tags.parquet");
    write_residue_parquet(
        &parquet_path,
        &accum,
        &kcc,
        &scores,
        &labels,
        &residue_keys,
        manifest.stream_count,
        max_current_step(&protocols),
    )?;

    let regions_path = args.output_dir.join("phase_manifold_regions.json");
    std::fs::write(&regions_path, serde_json::to_string_pretty(&regions)?)?;

    let replica_record_update = if args.no_update_replica_record {
        json!({"attempted": false, "reason": "disabled_by_flag"})
    } else {
        update_replica_record_if_available(&args, &parquet_path)?
    };

    let summary = json!({
        "schema_version": 1,
        "schema_kind": "prism_teacher_tags_stage_b",
        "computed_by": "prism-teacher-tags",
        "run_id": manifest.run_id,
        "target": manifest.target,
        "source_manifest": args.manifest,
        "validation_status": manifest.validation_status,
        "n_residues": topology.n_residues,
        "n_streams": manifest.stream_count,
        "total_spikes_md": manifest.total_spikes_md,
        "protocol_wavelengths_nm": observed_protocol_wavelengths(&protocols),
        "materializer_policy": {
            "primary_object": "residue_support_region",
            "centroids": "derived_diagnostic_views_only",
            "binding_sites_json_required": false,
            "ligsite_required": false,
            "phase_manifold_ranker_python_required": false,
            "consensus_axis": "replica_id_after_this_stage",
            "stream_axis": "protocol_conditioned_internal_views_not_iid_replicas"
        },
        "outputs": {
            "teacher_residue_tags_parquet": parquet_path,
            "phase_manifold_regions_json": regions_path,
            "ensemble_replica_record_update": replica_record_update
        },
        "regions": {
            "n_regions": regions.len(),
            "top_region_score": regions.first().map(|r| r.final_phase_manifold_score),
            "top_region_support": regions.first().map(|r| r.support_residues.len())
        },
        "feature_contract": {
            "per_residue": "flat numeric columns consumable by prism-v004-ensemble",
            "phase_manifold": [
                "max_phase_manifold_score",
                "is_lining",
                "is_kcc_driver",
                "is_hot_phase",
                "is_cold_phase",
                "is_burst_motion",
                "cryptic_likelihood_proxy"
            ],
            "kcc": ["active_causal_steps", "burst_motion", "causal_lag", "direction_score", "lag_corr_peak", "local_cov", "motion_efficiency"],
            "stream": ["stream_entropy", "effective_n_streams", "scout_observer_contrast"],
            "phase_bit": ["phase_bit_entropy", "ccns_phase_entropy", "phase_popcount_mean"]
        },
        "residue_key_policy": {
            "residue_id": "topology residue id when available, otherwise internal zero-based residue index",
            "teacher_residue_index": "internal zero-based residue index used by PRSPK/PRSGD/PRKCC payloads"
        },
        "mechanism_tag_policy": {
            "UV_AROMATIC_PERTURBATION": "spike_source == 1",
            "LIF_THERMAL_SHAPE": "spike_source == 2 and aromatic_type < 0",
            "LIF_LOCAL_INTENSITY": "spike_source == 2 and aromatic_type >= 0",
            "EFP": "spike_source == 3",
            "LADD": "spike_source == 4",
            "COFIRE": "spike_source == 5",
            "OTHER": "all other spike_source values"
        }
    });
    let summary_path = args.output_dir.join("teacher_tag_summary.json");
    std::fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)?;

    println!("wrote {}", parquet_path.display());
    println!("wrote {}", regions_path.display());
    println!("wrote {}", summary_path.display());
    println!(
        "regions={} residues={} source_spikes={}",
        regions.len(),
        topology.n_residues,
        manifest.total_spikes_md
    );
    Ok(())
}

fn artifacts_by_kind<'a>(manifest: &'a Manifest, kind: &str) -> Vec<&'a ManifestArtifact> {
    let mut out = manifest
        .artifacts
        .iter()
        .filter(|a| a.kind == kind && a.present)
        .collect::<Vec<_>>();
    out.sort_by_key(|a| a.stream_id.unwrap_or(u32::MAX));
    out
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

fn resolve_existing_input_path(manifest_path: &Path, input_path: &str) -> PathBuf {
    let p = PathBuf::from(input_path);
    if p.is_absolute() || p.exists() {
        return p;
    }
    let manifest_relative = manifest_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(input_path);
    if manifest_relative.exists() {
        return manifest_relative;
    }
    if let Ok(cwd) = std::env::current_dir() {
        let cwd_relative = cwd.join(input_path);
        if cwd_relative.exists() {
            return cwd_relative;
        }
    }
    manifest_relative
}

fn load_topology(path: &Path) -> Result<TopologyMin> {
    let s = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let t: TopologyMin =
        serde_json::from_str(&s).with_context(|| format!("parse {}", path.display()))?;
    if t.positions.len() != t.n_atoms * 3 {
        bail!(
            "topology positions len {} != 3*n_atoms {}",
            t.positions.len(),
            t.n_atoms
        );
    }
    Ok(t)
}

fn residue_ca_positions(topology: &TopologyMin) -> Vec<Option<[f64; 3]>> {
    let mut out = vec![None; topology.n_residues];
    for (rid, &atom_idx) in topology.ca_indices.iter().enumerate() {
        if rid >= out.len() {
            break;
        }
        let base = atom_idx * 3;
        if base + 2 < topology.positions.len() {
            out[rid] = Some([
                topology.positions[base] as f64,
                topology.positions[base + 1] as f64,
                topology.positions[base + 2] as f64,
            ]);
        }
    }
    out
}

fn residue_output_ids(topology: &TopologyMin) -> Vec<i32> {
    if topology.residue_ids.len() == topology.n_residues {
        return topology.residue_ids.clone();
    }

    let mut out = (0..topology.n_residues)
        .map(|rid| rid as i32)
        .collect::<Vec<_>>();

    if topology.residue_ids.len() == topology.n_atoms {
        for (rid, &atom_idx) in topology.ca_indices.iter().enumerate() {
            if rid < out.len() && atom_idx < topology.residue_ids.len() {
                out[rid] = topology.residue_ids[atom_idx];
            }
        }
        let mut filled = vec![false; topology.n_residues];
        for (rid, &atom_idx) in topology.ca_indices.iter().enumerate() {
            if rid < filled.len() && atom_idx < topology.residue_ids.len() {
                filled[rid] = true;
            }
        }
        for (atom_idx, &external_id) in topology.residue_ids.iter().enumerate() {
            let internal_id = external_id;
            if internal_id >= 0 {
                let rid = internal_id as usize;
                if rid < out.len() && !filled[rid] && atom_idx * 3 + 2 < topology.positions.len() {
                    out[rid] = external_id;
                    filled[rid] = true;
                }
            }
        }
    }

    out
}

fn load_protocol_sidecars(
    manifest_path: &Path,
    manifest: &Manifest,
) -> BTreeMap<u32, ProtocolStateSidecar> {
    let run_dir = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let mut out = BTreeMap::new();
    for sid in 0..manifest.stream_count {
        let p = run_dir.join(format!(
            "{}_stream{}_protocol_state.json",
            manifest.target, sid
        ));
        if let Ok(text) = std::fs::read_to_string(&p) {
            if let Ok(ps) = serde_json::from_str::<ProtocolStateSidecar>(&text) {
                out.insert(sid as u32, ps);
            }
        }
    }
    out
}

fn max_current_step(protocols: &BTreeMap<u32, ProtocolStateSidecar>) -> f64 {
    protocols
        .values()
        .map(|p| p.current_step as f64)
        .fold(0.0, f64::max)
}

fn observed_protocol_wavelengths(protocols: &BTreeMap<u32, ProtocolStateSidecar>) -> Vec<f32> {
    let mut xs = protocols
        .values()
        .flat_map(|p| p.scan_wavelengths_nm.iter().copied())
        .filter(|v| v.is_finite())
        .map(|v| (v * 10.0).round() as i32)
        .collect::<Vec<_>>();
    xs.sort_unstable();
    xs.dedup();
    xs.into_iter().map(|v| v as f32 / 10.0).collect()
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

fn validate_header(
    path: &Path,
    header: &EnvelopeHeader,
    manifest: &Manifest,
    art: &ManifestArtifact,
    expected_magic: &[u8; 8],
) -> Result<()> {
    if &header.magic != expected_magic {
        bail!(
            "{} magic mismatch: got {:?}, expected {:?}",
            path.display(),
            String::from_utf8_lossy(&header.magic),
            String::from_utf8_lossy(expected_magic)
        );
    }
    if header.endian_marker != LE_MARKER {
        bail!("{} non-little-endian marker", path.display());
    }
    if header.schema_version != 1 {
        bail!(
            "{} unsupported schema version {}",
            path.display(),
            header.schema_version
        );
    }
    if header.stem.is_empty() {
        bail!("{} empty envelope stem", path.display());
    }
    if header.run_id != manifest.run_id {
        bail!(
            "{} run_id mismatch {} != {}",
            path.display(),
            header.run_id,
            manifest.run_id
        );
    }
    if art.stream_id != Some(header.stream_id) {
        bail!("{} stream id mismatch", path.display());
    }
    if let Some(n) = art.record_count {
        if n != header.record_count {
            bail!(
                "{} manifest record_count {} != header {}",
                path.display(),
                n,
                header.record_count
            );
        }
    }
    let declared_total_size = header.header_size + header.payload_size + 16;
    if art.size_bytes != declared_total_size {
        bail!(
            "{} manifest size_bytes {} != envelope-declared {}",
            path.display(),
            art.size_bytes,
            declared_total_size
        );
    }
    Ok(())
}

fn validate_tail(
    path: &Path,
    payload: &[u8],
    stored_checksum: u64,
    trailer: &[u8; 8],
    expected_trailer: &[u8; 8],
    art: &ManifestArtifact,
) -> Result<()> {
    let checksum = fnv_update(FNV_OFFSET, payload);
    if checksum != stored_checksum {
        bail!("{} payload checksum mismatch", path.display());
    }
    if trailer != expected_trailer {
        bail!("{} trailer mismatch", path.display());
    }
    if let Some(hex) = &art.checksum_fnv1a_64_hex {
        let computed = format!("{:016x}", checksum);
        if !hex.eq_ignore_ascii_case(&computed) {
            bail!("{} manifest checksum mismatch", path.display());
        }
    }
    Ok(())
}

fn update_replica_record_if_available(args: &Args, parquet_path: &Path) -> Result<Value> {
    let manifest_dir = args.manifest.parent().unwrap_or_else(|| Path::new("."));
    let record_path = if let Some(path) = &args.replica_record {
        Some(path.clone())
    } else {
        let mut hits = Vec::new();
        for entry in std::fs::read_dir(manifest_dir)
            .with_context(|| format!("read {}", manifest_dir.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if name.starts_with("ensemble_replica_") && name.ends_with(".json") {
                hits.push(path);
            }
        }
        hits.sort();
        match hits.len() {
            0 => None,
            1 => hits.into_iter().next(),
            n => {
                return Ok(json!({
                    "attempted": false,
                    "reason": "multiple_records_require_explicit_replica_record",
                    "record_count": n
                }));
            }
        }
    };

    let Some(record_path) = record_path else {
        return Ok(json!({"attempted": false, "reason": "no_ensemble_replica_record_present"}));
    };
    if !record_path.exists() {
        return Ok(json!({
            "attempted": false,
            "reason": "replica_record_not_found",
            "replica_record": record_path
        }));
    }

    let text = std::fs::read_to_string(&record_path)
        .with_context(|| format!("read {}", record_path.display()))?;
    let mut value: Value =
        serde_json::from_str(&text).with_context(|| format!("parse {}", record_path.display()))?;
    let outputs = value
        .get_mut("replica")
        .and_then(|v| v.get_mut("outputs"))
        .and_then(Value::as_object_mut)
        .ok_or_else(|| anyhow::anyhow!("{} missing replica.outputs", record_path.display()))?;

    let record_dir = record_path.parent().unwrap_or_else(|| Path::new("."));
    let parquet_rel = relative_path_string(record_dir, parquet_path);
    let parquet_sha = sha256_file(parquet_path)?;
    let parquet_bytes = std::fs::metadata(parquet_path)
        .with_context(|| format!("stat {}", parquet_path.display()))?
        .len();
    outputs.insert(
        "feature_parquet_relative".to_string(),
        Value::String(parquet_rel.clone()),
    );
    outputs.insert(
        "feature_parquet_sha256".to_string(),
        Value::String(parquet_sha.clone()),
    );
    outputs.insert(
        "feature_parquet_bytes".to_string(),
        Value::Number(parquet_bytes.into()),
    );
    std::fs::write(&record_path, serde_json::to_vec_pretty(&value)?)
        .with_context(|| format!("write {}", record_path.display()))?;
    Ok(json!({
        "attempted": true,
        "updated": true,
        "replica_record": record_path,
        "feature_parquet_relative": parquet_rel,
        "feature_parquet_sha256": parquet_sha,
        "feature_parquet_bytes": parquet_bytes
    }))
}

fn relative_path_string(base: &Path, path: &Path) -> String {
    path.strip_prefix(base)
        .unwrap_or(path)
        .to_string_lossy()
        .into_owned()
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = f
            .read(&mut buf)
            .with_context(|| format!("read {}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn checked_advance(off: usize, len: usize, path: &Path, what: &str) -> Result<usize> {
    off.checked_add(len).with_context(|| {
        format!(
            "{} overflow while advancing {} by {} bytes",
            path.display(),
            what,
            len
        )
    })
}

fn checked_slice<'a>(
    payload: &'a [u8],
    off: usize,
    len: usize,
    path: &Path,
    what: &str,
) -> Result<&'a [u8]> {
    let end = checked_advance(off, len, path, what)?;
    if end > payload.len() {
        bail!(
            "{} truncated {}: need bytes [{}..{}), payload has {} bytes",
            path.display(),
            what,
            off,
            end,
            payload.len()
        );
    }
    Ok(&payload[off..end])
}

fn read_u64_at(payload: &[u8], off: usize, what: &str) -> Result<u64> {
    let end = off
        .checked_add(8)
        .with_context(|| format!("overflow reading {what}"))?;
    if end > payload.len() {
        bail!(
            "truncated {what}: need bytes [{}..{}), payload has {} bytes",
            off,
            end,
            payload.len()
        );
    }
    Ok(u64::from_le_bytes(
        payload[off..end].try_into().context("u64 decode")?,
    ))
}

fn read_i32_checked(payload: &[u8], off: usize, what: &str) -> Result<i32> {
    let end = off
        .checked_add(4)
        .with_context(|| format!("overflow reading {what}"))?;
    if end > payload.len() {
        bail!(
            "truncated {what}: need bytes [{}..{}), payload has {} bytes",
            off,
            end,
            payload.len()
        );
    }
    Ok(i32::from_le_bytes(
        payload[off..end].try_into().context("i32 decode")?,
    ))
}

fn read_kcc_stream(
    manifest_path: &Path,
    art: &ManifestArtifact,
    manifest: &Manifest,
) -> Result<KccStream> {
    let (path, header, payload, stored, trailer) = read_payload_and_tail(manifest_path, art)?;
    validate_header(&path, &header, manifest, art, b"PRKCC001")?;
    validate_tail(&path, &payload, stored, &trailer, b"PRKCCEND", art)?;
    if payload.len() < 16 {
        bail!("{} KCC payload too small", path.display());
    }
    let n_residues = read_u64_at(&payload, 0, "KCC n_residues")? as usize;
    let field_count = read_u64_at(&payload, 8, "KCC field_count")? as usize;
    let mut off = 16usize;
    let mut fields = HashMap::new();
    for _ in 0..field_count {
        let name_len = read_u64_at(&payload, off, "KCC field name length")? as usize;
        off = checked_advance(off, 8, &path, "KCC field name length")?;
        let name_bytes = checked_slice(&payload, off, name_len, &path, "KCC field name")?;
        let name = String::from_utf8(name_bytes.to_vec())?;
        off = checked_advance(off, name_len, &path, "KCC field name")?;
        let dtype = *checked_slice(&payload, off, 1, &path, "KCC field dtype")?
            .first()
            .context("missing KCC dtype")?;
        off = checked_advance(off, 1, &path, "KCC field dtype")?;
        let section_size = read_u64_at(&payload, off, "KCC section size")? as usize;
        off = checked_advance(off, 8, &path, "KCC section size")?;
        if section_size != n_residues * 4 {
            bail!("{} KCC section {} size mismatch", path.display(), name);
        }
        let section = checked_slice(&payload, off, section_size, &path, "KCC section payload")?;
        let mut vals = Vec::with_capacity(n_residues);
        for i in 0..n_residues {
            let b = &section[i * 4..i * 4 + 4];
            let v = match dtype {
                1 => f32::from_le_bytes(b.try_into().context("KCC f32 decode")?) as f64,
                2 => u32::from_le_bytes(b.try_into().context("KCC u32 decode")?) as f64,
                _ => bail!("{} unknown KCC dtype {}", path.display(), dtype),
            };
            vals.push(v);
        }
        off = checked_advance(off, section_size, &path, "KCC section payload")?;
        fields.insert(name, vals);
    }
    if off != payload.len() {
        bail!(
            "{} KCC payload has {} trailing bytes after declared sections",
            path.display(),
            payload.len() - off
        );
    }
    Ok(KccStream {
        stream_id: header.stream_id,
        n_residues,
        fields,
    })
}

fn merge_kcc(streams: &[KccStream], n_residues: usize) -> Vec<KccMerged> {
    let mut out = vec![KccMerged::default(); n_residues];
    for rid in 0..n_residues {
        let mut best: Option<&KccStream> = None;
        let mut best_active = f64::NEG_INFINITY;
        for s in streams {
            if s.n_residues <= rid {
                continue;
            }
            let active = field_value(s, "active_causal", rid);
            if active > best_active {
                best_active = active;
                best = Some(s);
            }
        }
        if let Some(s) = best {
            out[rid] = KccMerged {
                temporal_corr: field_value(s, "temporal_corr", rid),
                direction_score: field_value(s, "direction_score", rid),
                motion_efficiency: field_value(s, "motion_efficiency", rid),
                burst_motion: field_value(s, "burst_motion", rid),
                phase_shift: field_value(s, "phase_shift", rid),
                causal_lag: field_value(s, "causal_lag", rid),
                lag_corr_peak: field_value(s, "lag_corr_peak", rid),
                local_cov: field_value(s, "local_cov", rid),
                net_dx: field_value(s, "net_dx", rid),
                net_dy: field_value(s, "net_dy", rid),
                net_dz: field_value(s, "net_dz", rid),
                sum_m: field_value(s, "sum_m", rid),
                residue_count: field_value(s, "residue_count", rid),
                active_causal: best_active.max(0.0),
                selected_stream: s.stream_id,
            };
        }
    }
    out
}

fn field_value(s: &KccStream, name: &str, rid: usize) -> f64 {
    s.fields
        .get(name)
        .and_then(|v| v.get(rid))
        .copied()
        .filter(|v| v.is_finite())
        .unwrap_or(0.0)
}

fn apply_signal_grid(
    manifest_path: &Path,
    art: &ManifestArtifact,
    manifest: &Manifest,
    n_residues: usize,
    accum: &mut [ResidueAccum],
) -> Result<()> {
    let (path, header, payload, stored, trailer) = read_payload_and_tail(manifest_path, art)?;
    validate_header(&path, &header, manifest, art, b"PRSGD001")?;
    validate_tail(&path, &payload, stored, &trailer, b"PRSGDEND", art)?;
    if payload.len() < 16 {
        bail!("{} signal payload too small", path.display());
    }
    let voxel_count = read_u64_at(&payload, 8, "signal voxel_count")? as usize;
    let expected = 16 + 4 * voxel_count * 4;
    if payload.len() != expected {
        bail!("{} signal payload layout mismatch", path.display());
    }
    let base_hit = 16;
    let base_coupled = base_hit + voxel_count * 4;
    let base_primary_id = base_coupled + voxel_count * 4;
    let base_primary_count = base_primary_id + voxel_count * 4;
    for i in 0..voxel_count {
        let h = read_i32_checked(&payload, base_hit + i * 4, "signal hit count")?;
        let c = read_i32_checked(&payload, base_coupled + i * 4, "signal coupled count")?;
        let rid = read_i32_checked(
            &payload,
            base_primary_id + i * 4,
            "signal primary residue id",
        )?;
        let pc = read_i32_checked(
            &payload,
            base_primary_count + i * 4,
            "signal primary residue count",
        )?;
        if rid >= 0 && (rid as usize) < n_residues {
            let r = &mut accum[rid as usize];
            r.signal_voxel_hits = r.signal_voxel_hits.saturating_add(h.max(0) as u64);
            r.signal_coupled_count = r.signal_coupled_count.saturating_add(c.max(0) as u64);
            r.signal_primary_count = r.signal_primary_count.saturating_add(pc.max(0) as u64);
        }
    }
    Ok(())
}

fn apply_spikes(
    manifest_path: &Path,
    art: &ManifestArtifact,
    manifest: &Manifest,
    n_residues: usize,
    protocols: &BTreeMap<u32, ProtocolStateSidecar>,
    accum: &mut [ResidueAccum],
) -> Result<()> {
    let path = resolve_artifact_path(manifest_path, &art.path);
    let f = File::open(&path).with_context(|| format!("open {}", path.display()))?;
    let mut r = BufReader::with_capacity(1 << 20, f);
    let header = read_envelope_header(&mut r)?;
    validate_header(&path, &header, manifest, art, b"PRSPK001")?;
    // Exactly one spike record layout exists: the sealed 96 B packet record.
    // A `< 96` floor accepted any wider stride and would have parsed an
    // unknown layout at guessed offsets; every offset read below is fixed.
    if header.byte_stride != prism_nhs::fused_engine::PRISM_SPIKE_RECORD_BYTES as u64 {
        bail!(
            "{} spike byte_stride {} is not the sealed packet record stride ({})",
            path.display(),
            header.byte_stride,
            prism_nhs::fused_engine::PRISM_SPIKE_RECORD_BYTES
        );
    }
    if header.payload_size != header.record_count * header.byte_stride {
        bail!("{} spike payload size mismatch", path.display());
    }

    let stream_id = header.stream_id as usize;
    let group_id = group_id_for_stream(stream_id, manifest);
    let protocol = protocols
        .get(&header.stream_id)
        .cloned()
        .unwrap_or_default();
    let mut checksum = FNV_OFFSET;
    let stride = header.byte_stride as usize;
    let records_per_batch = 1 << 15;
    let mut buf = vec![0u8; records_per_batch * stride];
    let mut remaining = header.record_count;
    while remaining > 0 {
        let batch = remaining.min(records_per_batch as u64) as usize;
        let bytes = batch * stride;
        r.read_exact(&mut buf[..bytes])?;
        checksum = fnv_update(checksum, &buf[..bytes]);
        for rec in buf[..bytes].chunks_exact(stride) {
            let timestep = read_i32_at(rec, 0);
            let intensity = read_f32_at(rec, 20);
            let n_res = read_i32_at(rec, 56);
            let source = read_i32_at(rec, 60);
            let wavelength = read_f32_at(rec, 64);
            let aromatic_type = read_i32_at(rec, 68);
            let aromatic_residue_id = read_i32_at(rec, 72);
            let wd = read_f32_at(rec, 76);
            let ve = read_f32_at(rec, 80);
            let nne = read_i32_at(rec, 84);
            let wdc = read_f32_at(rec, 88);
            let phase_bits = read_u32_at(rec, 92);
            if !intensity.is_finite() || !wd.is_finite() || !ve.is_finite() || !wdc.is_finite() {
                continue;
            }
            let source_idx = source_index(source);
            let mechanism_idx = mechanism_index(source, aromatic_type);
            let wavelength_idx = wavelength_index(wavelength);
            let protocol_phase = protocol_phase_for_step(timestep, &protocol);
            let phase_bin = (phase_bits.count_ones() as usize).min(3);
            let n_valid = n_res.clamp(0, 8) as usize;
            for k in 0..n_valid {
                let rid = read_i32_at(rec, 24 + k * 4);
                if rid < 0 || (rid as usize) >= n_residues {
                    continue;
                }
                let r = &mut accum[rid as usize];
                r.spike_hits += 1;
                r.total_intensity += intensity.max(0.0) as f64;
                r.max_intensity = r.max_intensity.max(intensity.max(0.0) as f64);
                r.water_sum += wd as f64;
                r.wd_abs_sum += (wdc as f64).abs();
                r.vibrational_sum += ve.max(0.0) as f64;
                r.nearby_excited_sum += nne.max(0) as f64;
                r.phase_popcount_sum += phase_bits.count_ones() as u64;
                r.source_counts[source_idx] += 1;
                r.mechanism_counts[mechanism_idx] += 1;
                if let Some(wi) = wavelength_idx {
                    r.wavelength_counts[wi] += 1;
                }
                if stream_id < r.stream_counts.len() {
                    r.stream_counts[stream_id] += 1;
                }
                if group_id < 4 {
                    r.group_counts[group_id] += 1;
                }
                r.protocol_phase_counts[protocol_phase] += 1;
                r.phase_bit_counts[phase_bin] += 1;
            }
            if aromatic_residue_id >= 0 && (aromatic_residue_id as usize) < n_residues {
                accum[aromatic_residue_id as usize].aromatic_hits += 1;
            }
        }
        remaining -= batch as u64;
    }
    let mut stored = [0u8; 8];
    let mut trailer = [0u8; 8];
    r.read_exact(&mut stored)?;
    r.read_exact(&mut trailer)?;
    let stored_checksum = u64::from_le_bytes(stored);
    if checksum != stored_checksum {
        bail!("{} spike payload checksum mismatch", path.display());
    }
    if &trailer != b"PRSPKEND" {
        bail!("{} spike trailer mismatch", path.display());
    }
    if let Some(hex) = &art.checksum_fnv1a_64_hex {
        if !hex.eq_ignore_ascii_case(&format!("{:016x}", checksum)) {
            bail!("{} spike manifest checksum mismatch", path.display());
        }
    }
    Ok(())
}

fn group_id_for_stream(stream_id: usize, manifest: &Manifest) -> usize {
    let multi_diff = manifest
        .run_config
        .as_ref()
        .and_then(|v| v.get("multi_differential"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if !multi_diff || manifest.stream_count < 4 {
        return 0;
    }
    let epg = (manifest.stream_count / 4).max(1);
    (stream_id / epg).min(3)
}

fn source_index(source: i32) -> usize {
    match source {
        1 => 0, // UV
        2 => 1, // LIF
        3 => 2, // EFP
        4 => 3, // LADD
        5 => 4, // COFIRE
        _ => 5,
    }
}

fn mechanism_index(source: i32, aromatic_type: i32) -> usize {
    match source {
        1 => 0,                      // UV_AROMATIC_PERTURBATION
        2 if aromatic_type < 0 => 1, // LIF_THERMAL_SHAPE
        2 => 2,                      // LIF_LOCAL_INTENSITY
        3 => 3,                      // EFP
        4 => 4,                      // LADD
        5 => 5,                      // COFIRE
        _ => 6,                      // OTHER
    }
}

fn wavelength_index(wavelength: f32) -> Option<usize> {
    if wavelength <= 0.0 || !wavelength.is_finite() {
        return None;
    }
    UV_WAVELENGTHS
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            ((*a - wavelength).abs())
                .partial_cmp(&((*b - wavelength).abs()))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .and_then(|(i, w)| ((*w - wavelength).abs() <= 2.0).then_some(i))
}

fn protocol_phase_for_step(timestep: i32, p: &ProtocolStateSidecar) -> usize {
    if p.cold_hold_end > 0 && timestep < p.cold_hold_end {
        0
    } else if p.ramp_end > 0 && timestep < p.ramp_end {
        1
    } else if p.warm_hold_end > 0 && timestep < p.warm_hold_end {
        2
    } else if p.ramp_down_end > 0 && timestep < p.ramp_down_end {
        3
    } else {
        4
    }
}

fn score_residues(
    accum: &[ResidueAccum],
    kcc: &[KccMerged],
    n_streams: usize,
) -> Vec<(f64, ScoreComponents)> {
    let max_log_spikes = accum
        .iter()
        .map(|r| (r.spike_hits as f64).ln_1p())
        .fold(0.0, f64::max)
        .max(1.0);
    let max_signal = accum
        .iter()
        .map(|r| (r.signal_primary_count as f64).ln_1p())
        .fold(0.0, f64::max)
        .max(1.0);
    let max_coupled = accum
        .iter()
        .map(|r| (r.signal_coupled_count as f64).ln_1p())
        .fold(0.0, f64::max)
        .max(1.0);
    let max_active = kcc
        .iter()
        .map(|k| k.active_causal)
        .fold(0.0, f64::max)
        .max(1.0);
    let max_desolv = accum
        .iter()
        .map(|r| mean_or_zero(r.wd_abs_sum, r.spike_hits))
        .fold(0.0, f64::max)
        .max(1.0e-9);

    accum
        .iter()
        .enumerate()
        .map(|(rid, r)| {
            let spike = (r.spike_hits as f64).ln_1p() / max_log_spikes;
            let signal = (r.signal_primary_count as f64).ln_1p() / max_signal;
            let coupled = (r.signal_coupled_count as f64).ln_1p() / max_coupled;
            let k = &kcc[rid];
            let kcc_score = (0.45 * (k.active_causal / max_active)
                + 0.20 * k.local_cov.clamp(0.0, 1.0)
                + 0.15 * k.lag_corr_peak.abs().clamp(0.0, 1.0)
                + 0.10 * k.burst_motion.min(4.0) / 4.0
                + 0.10 * k.direction_score.abs().clamp(0.0, 1.0))
            .clamp(0.0, 1.0);
            let stream_consensus =
                nonzero_count(&r.stream_counts) as f64 / (n_streams.max(1) as f64);
            let group_consensus = r.group_counts.iter().filter(|&&v| v > 0).count() as f64 / 4.0;
            let uv = r.source_counts[0] as f64;
            let lif = r.source_counts[1] as f64;
            let uv_lif = if uv + lif > 0.0 {
                2.0 * uv.min(lif) / (uv + lif)
            } else {
                0.0
            };
            let phase_entropy = normalized_entropy_u64(&r.phase_bit_counts);
            let desolvation =
                (mean_or_zero(r.wd_abs_sum, r.spike_hits) / max_desolv).clamp(0.0, 1.0);
            let source_diversity = r.source_counts.iter().filter(|&&v| v > 0).count() as f64
                / r.source_counts.len() as f64;
            let components = ScoreComponents {
                spike,
                signal,
                coupled,
                kcc: kcc_score,
                stream_consensus,
                group_consensus,
                uv_lif_concordance: uv_lif,
                phase_entropy,
                desolvation,
                source_diversity,
            };
            let score = 0.14 * spike
                + 0.14 * signal
                + 0.10 * coupled
                + 0.18 * kcc_score
                + 0.12 * stream_consensus
                + 0.10 * group_consensus
                + 0.10 * uv_lif
                + 0.08 * phase_entropy
                + 0.08 * desolvation
                + 0.06 * source_diversity;
            (score.clamp(0.0, 1.0), components)
        })
        .collect()
}

fn mean_or_zero(sum: f64, n: u64) -> f64 {
    if n == 0 {
        0.0
    } else {
        sum / n as f64
    }
}

fn nonzero_count(xs: &[u64]) -> usize {
    xs.iter().filter(|&&v| v > 0).count()
}

fn normalized_entropy_u64<const N: usize>(xs: &[u64; N]) -> f64 {
    let total = xs.iter().sum::<u64>() as f64;
    if total <= 0.0 || N <= 1 {
        return 0.0;
    }
    let h = xs
        .iter()
        .filter(|&&v| v > 0)
        .map(|&v| {
            let p = v as f64 / total;
            -p * p.ln()
        })
        .sum::<f64>();
    (h / (N as f64).ln()).clamp(0.0, 1.0)
}

fn normalized_entropy_slice(xs: &[u64]) -> f64 {
    let total = xs.iter().sum::<u64>() as f64;
    if total <= 0.0 || xs.len() <= 1 {
        return 0.0;
    }
    let h = xs
        .iter()
        .filter(|&&v| v > 0)
        .map(|&v| {
            let p = v as f64 / total;
            -p * p.ln()
        })
        .sum::<f64>();
    (h / (xs.len() as f64).ln()).clamp(0.0, 1.0)
}

fn build_regions(
    args: &Args,
    accum: &[ResidueAccum],
    kcc: &[KccMerged],
    scores: &[(f64, ScoreComponents)],
    ca_xyz: &[Option<[f64; 3]>],
    n_streams: usize,
) -> Vec<Region> {
    let n = scores.len();
    let mut order = (0..n).collect::<Vec<_>>();
    order.sort_by(|&a, &b| scores[b].0.total_cmp(&scores[a].0).then(a.cmp(&b)));
    let keep = ((n as f64 * args.top_residue_fraction.clamp(0.01, 1.0)).ceil() as usize)
        .max(args.min_region_size)
        .min(n);
    let candidates = order.into_iter().take(keep).collect::<Vec<_>>();
    let candidate_set = candidates.iter().copied().collect::<HashSet<_>>();

    let mut uf = UnionFind::new(n);
    for (ia, &a) in candidates.iter().enumerate() {
        for &b in candidates.iter().skip(ia + 1) {
            if residues_connected(a, b, ca_xyz, args.cluster_distance_a) {
                uf.union(a, b);
            }
        }
    }
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for &rid in &candidate_set {
        groups.entry(uf.find(rid)).or_default().push(rid);
    }
    let mut regions = groups
        .into_values()
        .filter(|g| g.len() >= args.min_region_size)
        .map(|mut support| {
            support.sort_unstable();
            build_region_from_support(support, accum, kcc, scores, ca_xyz, n_streams)
        })
        .collect::<Vec<_>>();

    if regions.is_empty() && !scores.is_empty() {
        let mut fallback = (0..n).collect::<Vec<_>>();
        fallback.sort_by(|&a, &b| scores[b].0.total_cmp(&scores[a].0).then(a.cmp(&b)));
        fallback.truncate(args.min_region_size.min(fallback.len()));
        regions.push(build_region_from_support(
            fallback, accum, kcc, scores, ca_xyz, n_streams,
        ));
    }

    regions.sort_by(|a, b| {
        b.final_phase_manifold_score
            .total_cmp(&a.final_phase_manifold_score)
            .then(a.region_id.cmp(&b.region_id))
    });
    for (i, r) in regions.iter_mut().enumerate() {
        r.rank = i + 1;
        r.region_id = i as i32 + 1;
    }
    regions
}

fn residues_connected(a: usize, b: usize, ca_xyz: &[Option<[f64; 3]>], max_dist: f64) -> bool {
    if let (Some(pa), Some(pb)) = (
        ca_xyz.get(a).and_then(|x| *x),
        ca_xyz.get(b).and_then(|x| *x),
    ) {
        let d2 = (pa[0] - pb[0]).powi(2) + (pa[1] - pb[1]).powi(2) + (pa[2] - pb[2]).powi(2);
        d2 <= max_dist * max_dist
    } else {
        a.abs_diff(b) <= 4
    }
}

fn build_region_from_support(
    support: Vec<usize>,
    accum: &[ResidueAccum],
    kcc: &[KccMerged],
    scores: &[(f64, ScoreComponents)],
    ca_xyz: &[Option<[f64; 3]>],
    n_streams: usize,
) -> Region {
    let mut support_by_score = support.clone();
    support_by_score.sort_by(|&a, &b| scores[b].0.total_cmp(&scores[a].0).then(a.cmp(&b)));
    let top_score_mean = support_by_score
        .iter()
        .take(8)
        .map(|&rid| scores[rid].0)
        .sum::<f64>()
        / support_by_score.len().min(8).max(1) as f64;
    let region_components = mean_components(&support, scores);
    let stream_support = support
        .iter()
        .flat_map(|&rid| {
            accum[rid]
                .stream_counts
                .iter()
                .enumerate()
                .filter_map(|(sid, &v)| (v > 0).then_some(sid))
        })
        .collect::<HashSet<_>>()
        .len() as f64
        / n_streams.max(1) as f64;
    let score = (0.70 * top_score_mean + 0.30 * stream_support).clamp(0.0, 1.0);

    let kcc_driver_residues = top_by(&support, 0.30, |rid| kcc[rid].active_causal + scores[rid].0);
    let lining_or_surface_residues = top_by(&support, 0.50, |rid| {
        (accum[rid].signal_primary_count as f64).ln_1p() + (accum[rid].spike_hits as f64).ln_1p()
    });
    let burst_motion_supported_residues = top_by(&support, 0.25, |rid| {
        accum[rid].max_intensity + kcc[rid].burst_motion
    });
    let hot_phase_supported_residues = support
        .iter()
        .copied()
        .filter(|&rid| {
            accum[rid].protocol_phase_counts[2] + accum[rid].protocol_phase_counts[3] > 0
        })
        .collect::<Vec<_>>();
    let cold_phase_supported_residues = support
        .iter()
        .copied()
        .filter(|&rid| {
            accum[rid].protocol_phase_counts[0] + accum[rid].protocol_phase_counts[4] > 0
        })
        .collect::<Vec<_>>();

    let phase_modulation = support
        .iter()
        .map(|&rid| normalized_entropy_u64(&accum[rid].phase_bit_counts))
        .sum::<f64>()
        / support.len().max(1) as f64;
    let desolvation = support
        .iter()
        .map(|&rid| mean_or_zero(accum[rid].wd_abs_sum, accum[rid].spike_hits))
        .sum::<f64>()
        / support.len().max(1) as f64;
    let burst = support
        .iter()
        .map(|&rid| kcc[rid].burst_motion.min(4.0) / 4.0)
        .sum::<f64>()
        / support.len().max(1) as f64;
    let cryptic_proxy =
        (0.40 * phase_modulation + 0.25 * burst + 0.20 * desolvation.min(1.0) + 0.15 * score)
            .clamp(0.0, 1.0);

    Region {
        region_id: 0,
        rank: 0,
        final_phase_manifold_score: score,
        cryptic_likelihood_proxy: cryptic_proxy,
        support_residues: support.iter().map(|&r| r as i32).collect(),
        lining_or_surface_residues: lining_or_surface_residues
            .iter()
            .map(|&r| r as i32)
            .collect(),
        kcc_driver_residues: kcc_driver_residues.iter().map(|&r| r as i32).collect(),
        hot_phase_supported_residues: hot_phase_supported_residues
            .iter()
            .map(|&r| r as i32)
            .collect(),
        cold_phase_supported_residues: cold_phase_supported_residues
            .iter()
            .map(|&r| r as i32)
            .collect(),
        burst_motion_supported_residues: burst_motion_supported_residues
            .iter()
            .map(|&r| r as i32)
            .collect(),
        score_components: region_components,
        centroid_views: json!({
            "policy": "derived_diagnostic_only_after_support_region",
            "all_region": centroid_view(&support, ca_xyz),
            "driver": centroid_view(&kcc_driver_residues, ca_xyz),
            "lining": centroid_view(&lining_or_surface_residues, ca_xyz),
            "burst_motion": centroid_view(&burst_motion_supported_residues, ca_xyz),
            "hot_phase": centroid_view(&hot_phase_supported_residues, ca_xyz),
            "cold_phase": centroid_view(&cold_phase_supported_residues, ca_xyz),
        }),
    }
}

fn mean_components(support: &[usize], scores: &[(f64, ScoreComponents)]) -> ScoreComponents {
    let n = support.len().max(1) as f64;
    let mut c = ScoreComponents::default();
    for &rid in support {
        let x = &scores[rid].1;
        c.spike += x.spike;
        c.signal += x.signal;
        c.coupled += x.coupled;
        c.kcc += x.kcc;
        c.stream_consensus += x.stream_consensus;
        c.group_consensus += x.group_consensus;
        c.uv_lif_concordance += x.uv_lif_concordance;
        c.phase_entropy += x.phase_entropy;
        c.desolvation += x.desolvation;
        c.source_diversity += x.source_diversity;
    }
    c.spike /= n;
    c.signal /= n;
    c.coupled /= n;
    c.kcc /= n;
    c.stream_consensus /= n;
    c.group_consensus /= n;
    c.uv_lif_concordance /= n;
    c.phase_entropy /= n;
    c.desolvation /= n;
    c.source_diversity /= n;
    c
}

fn top_by<F>(support: &[usize], frac: f64, f: F) -> Vec<usize>
where
    F: Fn(usize) -> f64,
{
    let mut xs = support.to_vec();
    xs.sort_by(|&a, &b| f(b).total_cmp(&f(a)).then(a.cmp(&b)));
    let n = ((support.len() as f64 * frac).ceil() as usize)
        .max(1)
        .min(support.len());
    xs.truncate(n);
    xs.sort_unstable();
    xs
}

fn centroid_view(support: &[usize], ca_xyz: &[Option<[f64; 3]>]) -> Value {
    let pts = support
        .iter()
        .filter_map(|&rid| ca_xyz.get(rid).and_then(|x| *x))
        .collect::<Vec<_>>();
    if pts.is_empty() {
        return json!({"available": false, "support_count": support.len()});
    }
    let mut c = [0.0; 3];
    let mut mn = [f64::INFINITY; 3];
    let mut mx = [f64::NEG_INFINITY; 3];
    for p in &pts {
        for d in 0..3 {
            c[d] += p[d];
            mn[d] = mn[d].min(p[d]);
            mx[d] = mx[d].max(p[d]);
        }
    }
    for v in &mut c {
        *v /= pts.len() as f64;
    }
    json!({
        "available": true,
        "support_count": support.len(),
        "coordinate_source": "topology.ca_indices",
        "centroid_A": c,
        "aabb_A": [mn[0], mn[1], mn[2], mx[0], mx[1], mx[2]]
    })
}

fn assign_region_labels(n_residues: usize, regions: &[Region]) -> Vec<ResidueLabels> {
    let mut labels = vec![ResidueLabels::default(); n_residues];
    for region in regions {
        let support = region
            .support_residues
            .iter()
            .map(|&r| r as usize)
            .collect::<HashSet<_>>();
        let lining = region
            .lining_or_surface_residues
            .iter()
            .map(|&r| r as usize)
            .collect::<HashSet<_>>();
        let driver = region
            .kcc_driver_residues
            .iter()
            .map(|&r| r as usize)
            .collect::<HashSet<_>>();
        let hot = region
            .hot_phase_supported_residues
            .iter()
            .map(|&r| r as usize)
            .collect::<HashSet<_>>();
        let cold = region
            .cold_phase_supported_residues
            .iter()
            .map(|&r| r as usize)
            .collect::<HashSet<_>>();
        let burst = region
            .burst_motion_supported_residues
            .iter()
            .map(|&r| r as usize)
            .collect::<HashSet<_>>();
        for &rid in &support {
            if rid >= labels.len() {
                continue;
            }
            let l = &mut labels[rid];
            l.n_sites_containing_residue += 1.0;
            l.is_all_region = 1.0;
            l.n_sites_as_all_region += 1.0;
            if l.max_phase_manifold_score < region.final_phase_manifold_score {
                l.max_phase_manifold_score = region.final_phase_manifold_score;
                l.top1_site_score = region.final_phase_manifold_score;
                l.top1_site_rank = region.rank as f64;
                l.region_id = region.region_id;
                l.cryptic_likelihood_proxy = region.cryptic_likelihood_proxy;
                l.top1_site_classification = if region.cryptic_likelihood_proxy >= 0.50 {
                    1.0
                } else {
                    2.0
                };
            }
            if region.rank == 1 {
                l.is_in_top1_phase_site = 1.0;
            }
            if region.rank <= 5 {
                l.is_in_top5_phase_site = 1.0;
            }
            if region.rank <= 10 {
                l.is_in_top10_phase_site = 1.0;
            }
            if lining.contains(&rid) {
                l.is_lining = 1.0;
                l.n_sites_as_lining += 1.0;
            }
            if driver.contains(&rid) {
                l.is_kcc_driver = 1.0;
                l.n_sites_as_kcc_driver += 1.0;
            }
            if hot.contains(&rid) {
                l.is_hot_phase = 1.0;
                l.n_sites_as_hot_phase += 1.0;
            }
            if cold.contains(&rid) {
                l.is_cold_phase = 1.0;
                l.n_sites_as_cold_phase += 1.0;
            }
            if burst.contains(&rid) {
                l.is_burst_motion = 1.0;
                l.n_sites_as_burst_motion += 1.0;
            }
        }
    }
    labels
}

fn write_residue_parquet(
    path: &Path,
    accum: &[ResidueAccum],
    kcc: &[KccMerged],
    scores: &[(f64, ScoreComponents)],
    labels: &[ResidueLabels],
    residue_keys: &[i32],
    n_streams: usize,
    total_steps: f64,
) -> Result<()> {
    let n = accum.len();
    let mut residue_id = Int32Builder::with_capacity(n);
    let mut cols: Vec<(&'static str, Float64Builder)> = feature_names()
        .into_iter()
        .map(|name| (name, Float64Builder::with_capacity(n)))
        .collect();
    for rid in 0..n {
        residue_id.append_value(residue_keys.get(rid).copied().unwrap_or(rid as i32));
        let values = feature_values(
            rid,
            &accum[rid],
            &kcc[rid],
            scores[rid].0,
            &labels[rid],
            n_streams,
            total_steps,
        );
        for ((_, builder), value) in cols.iter_mut().zip(values) {
            builder.append_value(value);
        }
    }
    let mut fields = vec![Field::new("residue_id", DataType::Int32, false)];
    fields.extend(
        cols.iter()
            .map(|(name, _)| Field::new(*name, DataType::Float64, false)),
    );
    let schema = Arc::new(Schema::new(fields));
    let mut arrays: Vec<ArrayRef> = vec![Arc::new(residue_id.finish()) as ArrayRef];
    arrays.extend(
        cols.into_iter()
            .map(|(_, mut b)| Arc::new(b.finish()) as ArrayRef),
    );
    let batch = RecordBatch::try_new(schema.clone(), arrays)?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(9)?))
        .build();
    let f = File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut writer = ArrowWriter::try_new(f, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

fn feature_names() -> Vec<&'static str> {
    vec![
        "teacher_residue_index",
        "teacher_spike_hits",
        "teacher_total_intensity",
        "teacher_mean_intensity",
        "teacher_max_intensity",
        "teacher_uv_fraction",
        "teacher_lif_fraction",
        "teacher_source_diversity",
        "teacher_source_entropy",
        "teacher_wavelength_diversity",
        "teacher_wavelength_entropy",
        "teacher_uv_lif_concordance",
        "teacher_mechanism_uv_aromatic_fraction",
        "teacher_mechanism_lif_thermal_shape_fraction",
        "teacher_mechanism_lif_local_intensity_fraction",
        "teacher_mechanism_efp_fraction",
        "teacher_mechanism_ladd_fraction",
        "teacher_mechanism_cofire_fraction",
        "teacher_mechanism_other_fraction",
        "teacher_mean_abs_wd_change",
        "teacher_mean_water_density",
        "teacher_mean_vibrational_energy",
        "teacher_mean_nearby_excited",
        "teacher_signal_voxel_hits",
        "teacher_signal_primary_count",
        "teacher_signal_coupled_count",
        "teacher_signal_coupled_fraction",
        "teacher_aromatic_hits",
        "active_causal_steps",
        "burst_motion",
        "causal_lag",
        "direction_score",
        "lag_corr_peak",
        "local_cov",
        "motion_efficiency",
        "phase_shift",
        "kcc_selected_stream",
        "sum_motion",
        "total_steps",
        "net_dx_norm",
        "nearest_site_gtck",
        "nearest_kcc_confidence",
        "nearest_temporal_corr",
        "nearest_site_burst_motion",
        "nearest_site_causal_lag",
        "nearest_site_direction_score",
        "nearest_site_lag_corr_peak",
        "nearest_site_local_cov",
        "nearest_site_motion_efficiency",
        "max_phase_manifold_score",
        "top1_site_score",
        "top1_site_classification",
        "top1_site_centroid_view",
        "top1_site_rank",
        "is_in_top1_phase_site",
        "is_in_top5_phase_site",
        "is_in_top10_phase_site",
        "n_sites_containing_residue",
        "is_all_region",
        "n_sites_as_all_region",
        "is_lining",
        "n_sites_as_lining",
        "is_kcc_driver",
        "n_sites_as_kcc_driver",
        "is_hot_phase",
        "n_sites_as_hot_phase",
        "is_cold_phase",
        "n_sites_as_cold_phase",
        "is_burst_motion",
        "n_sites_as_burst_motion",
        "cryptic_likelihood_proxy",
        "stream_entropy",
        "stream_dominant_id",
        "stream_max_fraction",
        "effective_n_streams",
        "scout_mean_spikes",
        "observer_mean_spikes",
        "scout_observer_contrast",
        "phase_bit_entropy",
        "ccns_phase_entropy",
        "ccns_dominant_phase",
        "ccns_max_fraction",
        "phase_popcount_mean",
    ]
}

#[allow(clippy::too_many_arguments)]
fn feature_values(
    rid: usize,
    r: &ResidueAccum,
    k: &KccMerged,
    score: f64,
    l: &ResidueLabels,
    n_streams: usize,
    total_steps: f64,
) -> Vec<f64> {
    let total_sources = r.source_counts.iter().sum::<u64>() as f64;
    let total_mechanisms = r.mechanism_counts.iter().sum::<u64>() as f64;
    let uv = r.source_counts[0] as f64;
    let lif = r.source_counts[1] as f64;
    let uv_frac = if total_sources > 0.0 {
        uv / total_sources
    } else {
        0.0
    };
    let lif_frac = if total_sources > 0.0 {
        lif / total_sources
    } else {
        0.0
    };
    let uv_lif = if uv + lif > 0.0 {
        2.0 * uv.min(lif) / (uv + lif)
    } else {
        0.0
    };
    let stream_total = r.stream_counts.iter().sum::<u64>() as f64;
    let stream_entropy = normalized_entropy_slice(&r.stream_counts);
    let (dominant_stream, stream_max) = r
        .stream_counts
        .iter()
        .enumerate()
        .max_by_key(|(_, v)| *v)
        .map(|(i, &v)| (i as f64, v as f64))
        .unwrap_or((0.0, 0.0));
    let stream_max_fraction = if stream_total > 0.0 {
        stream_max / stream_total
    } else {
        0.0
    };
    let effective_n_streams = if stream_entropy > 0.0 {
        (stream_entropy * (n_streams.max(1) as f64).ln()).exp()
    } else {
        0.0
    };
    let scout = r.group_counts[0] as f64 + r.group_counts[2] as f64;
    let observer = r.group_counts[1] as f64 + r.group_counts[3] as f64;
    let scout_observer_contrast = if scout + observer > 0.0 {
        (scout - observer) / (scout + observer)
    } else {
        0.0
    };
    let (dominant_phase, phase_max) = r
        .protocol_phase_counts
        .iter()
        .enumerate()
        .max_by_key(|(_, v)| *v)
        .map(|(i, &v)| (i as f64, v as f64))
        .unwrap_or((0.0, 0.0));
    let phase_total = r.protocol_phase_counts.iter().sum::<u64>() as f64;
    let ccns_max_fraction = if phase_total > 0.0 {
        phase_max / phase_total
    } else {
        0.0
    };
    let net_dx_norm = (k.net_dx.powi(2) + k.net_dy.powi(2) + k.net_dz.powi(2)).sqrt();
    let mechanism_fraction = |idx: usize| {
        if total_mechanisms > 0.0 {
            r.mechanism_counts[idx] as f64 / total_mechanisms
        } else {
            0.0
        }
    };
    vec![
        rid as f64,
        r.spike_hits as f64,
        r.total_intensity,
        mean_or_zero(r.total_intensity, r.spike_hits),
        r.max_intensity,
        uv_frac,
        lif_frac,
        r.source_counts.iter().filter(|&&v| v > 0).count() as f64,
        normalized_entropy_u64(&r.source_counts),
        r.wavelength_counts.iter().filter(|&&v| v > 0).count() as f64,
        normalized_entropy_u64(&r.wavelength_counts),
        uv_lif,
        mechanism_fraction(0),
        mechanism_fraction(1),
        mechanism_fraction(2),
        mechanism_fraction(3),
        mechanism_fraction(4),
        mechanism_fraction(5),
        mechanism_fraction(6),
        mean_or_zero(r.wd_abs_sum, r.spike_hits),
        mean_or_zero(r.water_sum, r.spike_hits),
        mean_or_zero(r.vibrational_sum, r.spike_hits),
        mean_or_zero(r.nearby_excited_sum, r.spike_hits),
        r.signal_voxel_hits as f64,
        r.signal_primary_count as f64,
        r.signal_coupled_count as f64,
        if r.signal_voxel_hits > 0 {
            r.signal_coupled_count as f64 / r.signal_voxel_hits as f64
        } else {
            0.0
        },
        r.aromatic_hits as f64,
        k.active_causal,
        k.burst_motion,
        k.causal_lag,
        k.direction_score,
        k.lag_corr_peak,
        k.local_cov,
        k.motion_efficiency,
        k.phase_shift,
        k.selected_stream as f64,
        k.sum_m,
        total_steps.max(k.residue_count),
        net_dx_norm,
        l.region_id as f64,
        k.active_causal,
        k.temporal_corr,
        k.burst_motion,
        k.causal_lag,
        k.direction_score,
        k.lag_corr_peak,
        k.local_cov,
        k.motion_efficiency,
        l.max_phase_manifold_score.max(score),
        l.top1_site_score,
        l.top1_site_classification,
        0.0,
        l.top1_site_rank,
        l.is_in_top1_phase_site,
        l.is_in_top5_phase_site,
        l.is_in_top10_phase_site,
        l.n_sites_containing_residue,
        l.is_all_region,
        l.n_sites_as_all_region,
        l.is_lining,
        l.n_sites_as_lining,
        l.is_kcc_driver,
        l.n_sites_as_kcc_driver,
        l.is_hot_phase,
        l.n_sites_as_hot_phase,
        l.is_cold_phase,
        l.n_sites_as_cold_phase,
        l.is_burst_motion,
        l.n_sites_as_burst_motion,
        l.cryptic_likelihood_proxy,
        stream_entropy,
        dominant_stream,
        stream_max_fraction,
        effective_n_streams,
        scout / 2.0,
        observer / 2.0,
        scout_observer_contrast,
        normalized_entropy_u64(&r.phase_bit_counts),
        normalized_entropy_u64(&r.protocol_phase_counts),
        dominant_phase,
        ccns_max_fraction,
        mean_or_zero(r.phase_popcount_sum as f64, r.spike_hits),
    ]
}

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            let root = self.find(self.parent[x]);
            self.parent[x] = root;
        }
        self.parent[x]
    }

    fn union(&mut self, a: usize, b: usize) {
        let mut ra = self.find(a);
        let mut rb = self.find(b);
        if ra == rb {
            return;
        }
        if self.rank[ra] < self.rank[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb] = ra;
        if self.rank[ra] == self.rank[rb] {
            self.rank[ra] += 1;
        }
    }
}
