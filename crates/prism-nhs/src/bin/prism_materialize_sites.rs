// Path B materializer (ad412680+1).
//
// Offline consumer of the MD-only evidence handoff bundle produced by
// `nhs_rt_full --md-only-evidence`. Reads md_evidence_manifest.json,
// validates every per-stream PRSPK001 / PRSGD001 / PRKCC001 binary
// envelope, and materializes honest binding-site candidates from PRISM
// dynamic evidence using spatiotemporal + phase-coherence support.
//
// EMITS:
//
//   materialization_report.json                — per-artifact validation + summaries
//   site_candidates.json                        — voxel-density seed candidates
//   validation_inputs.json                      — paths Path B downstream needs
//   field_completeness_report.pathb.json        — Path B view of what's available
//   binding_sites.materialized.json             — materialized sites with full evidence
//                                                 (only sites meeting minimum criteria;
//                                                  others recorded in
//                                                  `non_materialized_candidates`)
//   site_accuracy_report.json                   — DCC vs. reference ligand if valid;
//                                                 explicit "reference_unavailable" when
//                                                 ground-truth ligand is on a different
//                                                 chain (see ground_truth.json)
//   materialization_field_completeness.json     — per-evidence-source availability matrix
//
// HONEST SCOPE
//
// This is NOT a static-pocket detector. Density peaks are SEED candidates
// only; a site is promoted to "materialized_dynamic_site" only after it
// accumulates spatiotemporal evidence:
//
//   * stream_support       — which of the 8 streams observe it
//   * temporal_support     — timestep extent, recurrence across chunks
//   * phase_support        — phase_bits histogram + Rayleigh phase coherence
//   * residue_support      — nearby_residues from spikes
//   * centroid_manifold    — whole_site / lining_mass / causal_driver /
//                            hot_phase / cold_phase / ligand_adjacent
//
// LIGSITE / static geometry is NOT used. find_neighbors_union DBSCAN
// is NOT used. XGBoost rerank is NOT used.
//
// Warp-matrix / forces / ASC-vectors / adaptive-dt / BOCPD are canonical
// first-class PrismOS surfaces. The materializer hydrates their QNANL/NANL
// packets when raw engine-side files were intentionally removed, parses their
// producer-defined layouts, and emits parsed summaries into the rank protocol.
// `phase_coherence_rayleigh_r` is protocol-phase coherence only. Dense GPU
// surface parsing is tracked separately from unavailable per-region rotations.
//
// Materialization levels:
//
//   candidate_density_only            seeds only; no per-spike evidence
//   candidate_spatiotemporal_partial  has stream + temporal + phase, but
//                                     missing manifold or sub-min support
//   materialized_dynamic_site         all minimum criteria satisfied
//   materialized_dynamic_site_with_validation
//                                     above + DCC against valid reference
//
// `binding_sites.materialized.json` is written only when at least one
// site reaches `materialized_dynamic_site`. Otherwise the file is still
// written with `site_count: 0` and `status: "no_sites_met_criteria"` for
// honest reporting.

#![recursion_limit = "256"]

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use prism_nhs::causome::{
    native_residue_kcc_confidence, NativeKccRegionStream, NativeKccResidueObservation,
};
use prismcode::compression_registry::{CompressionCodecKind, RuntimeCodecRegistry};
use prismcode::qnanl_artifact::{self, QnanlArtifactFamily, QnanlEncodeOptions};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

#[cfg(any(feature = "gpu", feature = "cpu-analysis"))]
#[path = "materialization_measurements/ghost_kcc.rs"]
mod ghost_kcc;

const RANK_POLICY_ID_V2: &str = "PRISMOS_SITE_MATERIALIZATION_LEXICOGRAPHIC_V2";
const RANKING_METHODOLOGY_V2: &str = "prismos_lexicographic_site_materialization_v2";
const RANK_KEY_ORDER_V2: &[&str] = &[
    "promoted desc",
    "materialization_level_rank desc",
    "site_family_preservation_rank desc",
    "pocket_morphology_rank desc",
    "thermodynamic_physiology_rank desc",
    "n_active_non_density_dimensions desc",
    "kcc_driver_factor desc",
    "signal_preservation_factor desc",
    "temporal_persistence_factor desc",
    "centroid_manifold_consistency_factor desc",
    "scale_normalized_dynamic_support desc",
    "final_prism_score desc",
    "n_streams_supporting desc",
    "region_id asc",
];

const RANK_POLICY_ID_NATIVE_KCC: &str = "PRISMOS_SITE_MATERIALIZATION_NATIVE_KCC_V1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum RankingPolicy {
    LegacyV2,
    NativeKcc,
}

impl RankingPolicy {
    fn id(self) -> &'static str {
        match self {
            Self::LegacyV2 => RANK_POLICY_ID_V2,
            Self::NativeKcc => RANK_POLICY_ID_NATIVE_KCC,
        }
    }
    fn key_order(self) -> Vec<&'static str> {
        let mut keys = RANK_KEY_ORDER_V2.to_vec();
        if self == Self::NativeKcc {
            keys.splice(
                2..2,
                [
                    "native_residue_kcc_confidence desc (unavailable last)",
                    "return_inclusive_geometric_consistency desc",
                ],
            );
        }
        keys
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum SupportAccountingPolicy {
    LegacyPeakV2,
    UniqueRecordV1,
}

// ─── CLI ───────────────────────────────────────────────────────────────────

const MIN_SPIKES_PER_SUPPORTING_STREAM: u64 = 50;

#[derive(Parser, Debug)]
#[command(name = "prism-materialize-sites")]
#[command(about = "Path B offline materializer — consumes MD-only evidence handoff bundle.")]
struct Args {
    /// md_evidence_manifest.json path produced by `nhs_rt_full --md-only-evidence`.
    #[arg(short, long)]
    manifest: PathBuf,

    /// Output dir. Defaults to the manifest's parent directory.
    #[arg(short, long)]
    output_dir: Option<PathBuf>,

    /// Voxel size for density-peak candidate generation (Å).
    #[arg(long, default_value = "2.0")]
    voxel_size_a: f32,

    /// Stage A — number of raw density peaks pulled from the voxel grid
    /// before consolidation. Default raised from 50 to 200 in v2 so that
    /// non-N-terminal regions (e.g., catalytic pockets) survive into the
    /// consolidation stage.
    #[arg(long, default_value = "200")]
    top_k_candidates: usize,

    /// Stage B — centroid distance threshold (Å) for consolidating raw
    /// peaks into candidate regions.
    #[arg(long, default_value = "4.0")]
    consolidation_radius_a: f32,

    /// Stage B — minimum Jaccard overlap of top-12 lining-residue sets
    /// at which two peaks merge regardless of centroid distance.
    #[arg(long, default_value = "0.5")]
    consolidation_jaccard: f32,

    /// Final number of materialized sites to retain after PRISM-native
    /// scoring (top-N by `final_prism_score`). Smaller than
    /// `top_k_candidates` because consolidation collapses duplicates.
    #[arg(long, default_value = "20")]
    materialized_top_n: usize,

    /// Minimum qualifying native streams for promotion. Use 1 explicitly for
    /// a single-stream campaign; stream count does not establish independent repeats.
    #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u32).range(1..))]
    min_supporting_streams: u32,

    /// Recount conserved source records once per consolidated region. Region
    /// membership/IDs remain unchanged; legacy totals remain in the receipt.
    #[arg(long, value_enum, default_value = "legacy-peak-v2")]
    support_accounting: SupportAccountingPolicy,

    /// Opt into recorded native residue-confidence ranking before morphology.
    /// Admission thresholds remain unchanged; legacy-v2 retains previous ranking.
    #[arg(long, value_enum, default_value_t = RankingPolicy::LegacyV2)]
    ranking_policy: RankingPolicy,

    /// Sub-chunk bin size (steps) for refined temporal-persistence scoring.
    /// 100 steps gives 100 bins on a 10000-step run, restoring
    /// discrimination that the engine's `chunk_set` (1000-step bins) loses.
    #[arg(long, default_value = "100")]
    persistence_bin_steps: i32,

    /// **M1.2.23 §12** — path to `ghost_zstr_candidate_features.json`
    /// produced by `prism-ghost-evidence-scan`. When provided, an
    /// additional `ghost_zstr_factor` component is folded into the
    /// PRISM-native composite score (clamped [0.7, 1.6]). When absent,
    /// the factor is neutral 1.0 — no fake boost. Per directive Phase
    /// 12: "if absent → ghost_zstr_factor = 1.0; if present but spatial
    /// mapping absent → use non-spatial aggregate features only; if
    /// present with spatial mapping → connect Ghost/ZSTR support to
    /// candidate regions".
    #[arg(long)]
    ghost_features: Option<PathBuf>,

    /// Chem-perturbed DTSG state graph emitted by dstw-dispatch-variants.
    /// When present, edges are linked only to candidate regions carrying one
    /// of the edge residues; absent or unmatched evidence remains neutral.
    #[arg(long)]
    chem_perturbed_dtsg_state_graph: Option<PathBuf>,

    /// Exact native lattice lineage to associate with KCC chronology; no ranking changes.
    #[arg(long)]
    native_ghost_lattice: Option<PathBuf>,

    /// Saved per-stream KCC causal-lead JSONL (repeat for multiple streams).
    #[arg(long)]
    kcc_chronology: Vec<PathBuf>,

    /// Skip the spike-density pass entirely (faster validation-only run).
    #[arg(long, default_value = "false")]
    headers_only: bool,

    /// Verbose logging.
    #[arg(short, long, default_value = "false")]
    verbose: bool,
}

// ─── Manifest schema (read-only mirror of what the engine writes) ──────────

#[derive(Debug, Deserialize)]
struct ManifestArtifact {
    kind: String,
    stream_id: Option<u32>,
    path: String,
    size_bytes: u64,
    schema_version: Option<u32>,
    checksum_fnv1a_64_hex: Option<String>,
    record_count: Option<u64>,
    present: bool,
    note: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Manifest {
    schema_version: u32,
    schema_kind: String,
    run_id: String,
    target: String,
    topology_input: String,
    stream_count: usize,
    streams_serialized: usize,
    total_spikes_md: u64,
    artifacts: Vec<ManifestArtifact>,
    compression_status: String,
    checksum_algorithm: String,
    endian: String,
    v2_was_live: bool,
    path_b_required: bool,
    serialization_failure: Option<String>,
}

// ─── PRISM binary envelope parser ──────────────────────────────────────────

#[derive(Debug, Clone)]
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
}

#[derive(Debug, Serialize)]
struct ArtifactValidation {
    artifact_path: String,
    expected_kind: String,
    magic_observed: String,
    magic_match: bool,
    schema_version: u32,
    endian_marker_hex: String,
    endian_ok_le: bool,
    stream_id: u32,
    stream_id_matches_filename: Option<bool>,
    run_id_in_header: String,
    run_id_matches_manifest: bool,
    stem_in_header: String,
    stem_matches_manifest: bool,
    record_count: u64,
    record_count_matches_manifest: Option<bool>,
    byte_stride: u64,
    payload_size: u64,
    declared_total_size: u64,
    on_disk_size: u64,
    size_consistency: bool,
    payload_checksum_fnv1a_64_hex: String,
    checksum_matches_manifest: Option<bool>,
    trailer_observed: String,
    trailer_match: bool,
    valid: bool,
    failure_reason: Option<String>,
}

const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

fn fnv1a_64_streamed<R: Read>(r: &mut R, total: u64) -> Result<u64> {
    let mut h = FNV_OFFSET;
    let mut buf = vec![0u8; 1 << 20];
    let mut remaining = total;
    while remaining > 0 {
        let to_read = std::cmp::min(remaining as usize, buf.len());
        r.read_exact(&mut buf[..to_read])?;
        for &b in &buf[..to_read] {
            h ^= b as u64;
            h = h.wrapping_mul(FNV_PRIME);
        }
        remaining -= to_read as u64;
    }
    Ok(h)
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

fn read_u64_le_at(bytes: &[u8], cursor: &mut usize, label: &str) -> Result<u64> {
    let end = cursor
        .checked_add(8)
        .with_context(|| format!("cursor overflow reading {label}"))?;
    anyhow::ensure!(end <= bytes.len(), "truncated bytes reading {label}");
    let value = u64::from_le_bytes(bytes[*cursor..end].try_into()?);
    *cursor = end;
    Ok(value)
}

fn untranspose_fixed_stride_payload(
    transposed: &[u8],
    record_count: usize,
    stride: usize,
) -> Result<Vec<u8>> {
    if record_count == 0 {
        return Ok(Vec::new());
    }
    anyhow::ensure!(stride > 0, "spike codec stride cannot be zero");
    anyhow::ensure!(
        transposed.len() == record_count * stride,
        "spike codec transposed length mismatch: transposed={} record_count={} stride={}",
        transposed.len(),
        record_count,
        stride
    );
    let mut out = vec![0u8; transposed.len()];
    let mut read = 0usize;
    for lane in 0..stride {
        for record in 0..record_count {
            out[record * stride + lane] = transposed[read];
            read += 1;
        }
    }
    Ok(out)
}

fn decode_spike_prismnanl_packet(packet_path: &Path) -> Result<Vec<u8>> {
    const MAGIC: &[u8; 8] = b"PRSPCZ01";
    let packet = std::fs::read(packet_path)
        .with_context(|| format!("read spike NANL packet {}", packet_path.display()))?;
    anyhow::ensure!(
        packet.len() >= MAGIC.len() + 4 + 8 * 4 && &packet[0..8] == MAGIC,
        "spike NANL packet magic mismatch"
    );
    let mut cursor = 8usize + 4usize;
    let header_len = read_u64_le_at(&packet, &mut cursor, "spike_packet_header_len")? as usize;
    let prefix_len = read_u64_le_at(&packet, &mut cursor, "spike_packet_prefix_len")? as usize;
    let suffix_len = read_u64_le_at(&packet, &mut cursor, "spike_packet_suffix_len")? as usize;
    let compressed_len =
        read_u64_le_at(&packet, &mut cursor, "spike_packet_compressed_len")? as usize;
    let header_end = cursor
        .checked_add(header_len)
        .context("spike packet header overflow")?;
    let prefix_start = header_end;
    let prefix_end = prefix_start
        .checked_add(prefix_len)
        .context("spike packet prefix overflow")?;
    let suffix_start = prefix_end;
    let suffix_end = suffix_start
        .checked_add(suffix_len)
        .context("spike packet suffix overflow")?;
    let compressed_start = suffix_end;
    let compressed_end = compressed_start
        .checked_add(compressed_len)
        .context("spike packet compressed body overflow")?;
    anyhow::ensure!(
        compressed_end == packet.len(),
        "spike packet length mismatch: decoded end {} != packet len {}",
        compressed_end,
        packet.len()
    );
    let header: serde_json::Value = serde_json::from_slice(&packet[cursor..header_end])
        .context("parse spike NANL packet header")?;
    let record_count = header
        .get("record_count")
        .and_then(serde_json::Value::as_u64)
        .context("spike NANL packet missing record_count")? as usize;
    let stride = header
        .get("payload_stride")
        .and_then(serde_json::Value::as_u64)
        .context("spike NANL packet missing payload_stride")? as usize;
    let registry = RuntimeCodecRegistry::with_builtin_codecs();
    let codec = registry.codec_by_kind(CompressionCodecKind::SectionZstd)?;
    let decoded_transposed = codec
        .decode(&packet[compressed_start..compressed_end])
        .context("decode spike NANL transposed payload")?;
    let decoded_payload =
        untranspose_fixed_stride_payload(&decoded_transposed, record_count, stride)?;
    let mut reconstructed = Vec::with_capacity(prefix_len + decoded_payload.len() + suffix_len);
    reconstructed.extend_from_slice(&packet[prefix_start..prefix_end]);
    reconstructed.extend_from_slice(&decoded_payload);
    reconstructed.extend_from_slice(&packet[suffix_start..suffix_end]);
    if let Some(expected_hash) = header
        .get("raw_hash_blake3")
        .and_then(serde_json::Value::as_str)
    {
        let actual_hash = blake3::hash(&reconstructed).to_hex().to_string();
        anyhow::ensure!(
            actual_hash == expected_hash,
            "spike NANL raw hash mismatch: actual {} expected {}",
            actual_hash,
            expected_hash
        );
    }
    Ok(reconstructed)
}

fn canonical_packet_path_for(artifact_path: &Path, kind: &str) -> Result<PathBuf> {
    if kind == "spikes" {
        let file_name = artifact_path
            .file_name()
            .and_then(|value| value.to_str())
            .context("spike artifact path missing file name")?;
        let packet_name = file_name.trim_end_matches(".bin").to_string() + ".prismnanl";
        return Ok(artifact_path.with_file_name(packet_name));
    }
    Ok(PathBuf::from(format!(
        "{}.prismqnanl",
        artifact_path.display()
    )))
}

fn hydrate_artifact_from_packet(
    artifact_path: &Path,
    kind: &str,
    hydration_dir: &Path,
) -> Result<PathBuf> {
    let file_name = artifact_path
        .file_name()
        .and_then(|value| value.to_str())
        .context("artifact path missing file name")?;
    std::fs::create_dir_all(hydration_dir)
        .with_context(|| format!("create hydration cache {}", hydration_dir.display()))?;
    let hydrated_path = hydration_dir.join(file_name);
    let packet_path = canonical_packet_path_for(artifact_path, kind)?;

    // PrismOS durability rule: canonical QNANL/NANL packets are primary.
    // If both raw and packet are present, decode the packet and consume the
    // decoded canonical payload. Raw files are debug/provenance fallbacks only.
    if packet_path.exists() && hydrated_path.exists() {
        return Ok(hydrated_path);
    }
    if kind == "spikes" && packet_path.exists() {
        let decoded = decode_spike_prismnanl_packet(&packet_path)
            .with_context(|| format!("hydrate spikes from {}", packet_path.display()))?;
        std::fs::write(&hydrated_path, decoded)
            .with_context(|| format!("write hydrated spikes {}", hydrated_path.display()))?;
        return Ok(hydrated_path);
    }
    if packet_path.exists() {
        let decoded = prismcode::qnanl_artifact::decode_file(&packet_path, &hydrated_path)
            .with_context(|| format!("hydrate QNANL artifact from {}", packet_path.display()))?;
        anyhow::ensure!(
            !decoded.is_empty() || artifact_path.metadata().map(|m| m.len()).unwrap_or(1) == 0,
            "hydrated artifact is empty for {}",
            artifact_path.display()
        );
        return Ok(hydrated_path);
    }
    if artifact_path.exists() {
        return Ok(artifact_path.to_path_buf());
    }
    anyhow::bail!(
        "neither canonical packet {} nor raw artifact {} exists",
        packet_path.display(),
        artifact_path.display()
    )
}

/// Parse the PRISM envelope header from a file. Leaves the cursor positioned
/// at the start of the payload bytes. Returns the header.
fn read_envelope_header<R: Read>(r: &mut R) -> Result<EnvelopeHeader> {
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    let schema_version = read_u32_le(r)?;
    let endian_marker = read_u32_le(r)?;
    let stream_id = read_u32_le(r)?;
    let run_id_len = read_u64_le(r)? as usize;
    let run_id_bytes = read_bytes(r, run_id_len)?;
    let run_id = String::from_utf8(run_id_bytes).context("envelope: run_id not utf-8")?;
    let stem_len = read_u64_le(r)? as usize;
    let stem_bytes = read_bytes(r, stem_len)?;
    let stem = String::from_utf8(stem_bytes).context("envelope: stem not utf-8")?;
    let record_count = read_u64_le(r)?;
    let byte_stride = read_u64_le(r)?;
    let payload_size = read_u64_le(r)?;
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
    })
}

/// Validate an artifact, optionally streaming through the payload to compute
/// the FNV-1a checksum. Returns the validation record + computed checksum.
fn validate_artifact(
    artifact_path: &Path,
    expected_magic: &[u8; 8],
    expected_trailer: &[u8; 8],
    expected_kind: &str,
    manifest_record_count: Option<u64>,
    manifest_checksum_hex: Option<&str>,
    manifest_run_id: &str,
    manifest_stem: &str,
    manifest_filename_stream_id: Option<u32>,
    headers_only: bool,
) -> Result<(ArtifactValidation, Option<EnvelopeHeader>)> {
    let on_disk_size = std::fs::metadata(artifact_path)
        .with_context(|| format!("stat {}", artifact_path.display()))?
        .len();
    let f =
        File::open(artifact_path).with_context(|| format!("open {}", artifact_path.display()))?;
    let mut r = BufReader::with_capacity(1 << 20, f);
    let header = read_envelope_header(&mut r)?;
    let header_size_so_far: u64 =
        8 + 4 + 4 + 4 + 8 + header.run_id.len() as u64 + 8 + header.stem.len() as u64 + 8 + 8 + 8;

    let declared_total = header_size_so_far + header.payload_size + 8 /* checksum */ + 8 /* trailer */;
    let size_consistency = declared_total == on_disk_size;

    let magic_match = &header.magic == expected_magic;
    let endian_ok_le = header.endian_marker == 0x01020304u32;

    // Compute payload checksum (streaming) — unless headers_only.
    let computed_checksum: u64 =
        if headers_only || !magic_match || !endian_ok_le || !size_consistency {
            0
        } else {
            fnv1a_64_streamed(&mut r, header.payload_size)?
        };

    // Read trailing 8 bytes checksum + 8 bytes trailer.
    let mut tail_checksum_bytes = [0u8; 8];
    let mut trailer_bytes = [0u8; 8];
    if !headers_only && magic_match && endian_ok_le && size_consistency {
        r.read_exact(&mut tail_checksum_bytes)?;
        r.read_exact(&mut trailer_bytes)?;
    }
    let stored_checksum = if headers_only {
        0
    } else {
        u64::from_le_bytes(tail_checksum_bytes)
    };
    let trailer_match = !headers_only && &trailer_bytes == expected_trailer;

    let checksum_matches_manifest = if headers_only {
        None
    } else {
        manifest_checksum_hex
            .map(|hex| format!("{:016x}", computed_checksum) == hex.to_ascii_lowercase())
    };

    let record_count_matches_manifest = manifest_record_count.map(|n| n == header.record_count);

    let stream_id_matches_filename = manifest_filename_stream_id.map(|id| id == header.stream_id);

    let identity_matches = header.run_id == manifest_run_id
        && header.stem == manifest_stem
        && stream_id_matches_filename.unwrap_or(true);
    let valid = identity_matches
        && magic_match
        && endian_ok_le
        && size_consistency
        && (headers_only || trailer_match)
        && (headers_only || stored_checksum == computed_checksum)
        && record_count_matches_manifest.unwrap_or(true)
        && checksum_matches_manifest.unwrap_or(true);

    let failure_reason: Option<String> = if !identity_matches {
        Some("envelope run/stem/filename stream identity mismatch".to_string())
    } else if !magic_match {
        Some(format!(
            "magic mismatch: expected {:?}, got {:?}",
            std::str::from_utf8(expected_magic).unwrap_or("?"),
            std::str::from_utf8(&header.magic).unwrap_or("?")
        ))
    } else if !endian_ok_le {
        Some(format!(
            "endian marker not LE: 0x{:08x}",
            header.endian_marker
        ))
    } else if !size_consistency {
        Some(format!(
            "size mismatch: declared total {}, on-disk {}",
            declared_total, on_disk_size
        ))
    } else if !headers_only && !trailer_match {
        Some(format!(
            "trailer mismatch: expected {:?}, got {:?}",
            std::str::from_utf8(expected_trailer).unwrap_or("?"),
            std::str::from_utf8(&trailer_bytes).unwrap_or("?")
        ))
    } else if !headers_only && stored_checksum != computed_checksum {
        Some(format!(
            "checksum mismatch: stored 0x{:016x}, computed 0x{:016x}",
            stored_checksum, computed_checksum
        ))
    } else if record_count_matches_manifest == Some(false) {
        Some("record_count mismatch with manifest".to_string())
    } else if checksum_matches_manifest == Some(false) {
        Some("checksum mismatch with manifest".to_string())
    } else {
        None
    };

    let v = ArtifactValidation {
        artifact_path: artifact_path.display().to_string(),
        expected_kind: expected_kind.to_string(),
        magic_observed: String::from_utf8_lossy(&header.magic).to_string(),
        magic_match,
        schema_version: header.schema_version,
        endian_marker_hex: format!("{:08x}", header.endian_marker),
        endian_ok_le,
        stream_id: header.stream_id,
        stream_id_matches_filename,
        run_id_in_header: header.run_id.clone(),
        run_id_matches_manifest: header.run_id == manifest_run_id,
        stem_in_header: header.stem.clone(),
        stem_matches_manifest: header.stem == manifest_stem,
        record_count: header.record_count,
        record_count_matches_manifest,
        byte_stride: header.byte_stride,
        payload_size: header.payload_size,
        declared_total_size: declared_total,
        on_disk_size,
        size_consistency,
        payload_checksum_fnv1a_64_hex: format!("{:016x}", computed_checksum),
        checksum_matches_manifest,
        trailer_observed: String::from_utf8_lossy(&trailer_bytes).to_string(),
        trailer_match,
        valid,
        failure_reason,
    };
    Ok((v, Some(header)))
}

// ─── Spike density pass ───────────────────────────────────────────────────

#[derive(Debug, Default, Serialize)]
struct StreamSpikeSummary {
    stream_id: u32,
    n_spikes: u64,
    bbox_min: [f32; 3],
    bbox_max: [f32; 3],
    intensity_min: f32,
    intensity_max: f32,
    intensity_mean: f64,
    n_uv: u64,
    n_lif: u64,
    n_other_source: u64,
    n_aromatic_trp: u64,
    n_aromatic_tyr: u64,
    n_aromatic_phe: u64,
    n_aromatic_ss: u64,
    n_aromatic_none: u64,
    elapsed_ms: u64,
}

/// Spike record byte offsets (must match `GpuSpikeEvent` in fused_engine.rs).
/// We do NOT mirror the whole struct — we only parse the fields needed for
/// density-peak generation and basic summary stats.
///
/// All offsets here live in bytes 0..95 of the sealed 96 B packet record, so
/// this reader is correct provided it strides by the envelope's declared
/// `byte_stride` (it does) and refuses a stride it does not recognise
/// (see below).
mod spike_offsets {
    pub const POSITION_X: usize = 8; // f32
    pub const POSITION_Y: usize = 12; // f32
    pub const POSITION_Z: usize = 16; // f32
    pub const INTENSITY: usize = 20; // f32
    pub const SPIKE_SOURCE: usize = 60; // i32 — 1=UV, 2=LIF
    pub const AROMATIC_TYPE: usize = 68; // i32 — 0=TRP, 1=TYR, 2=PHE, 3=SS, -1=none
}

fn read_f32_at(buf: &[u8], offset: usize) -> f32 {
    let mut b = [0u8; 4];
    b.copy_from_slice(&buf[offset..offset + 4]);
    f32::from_le_bytes(b)
}

fn read_i32_at(buf: &[u8], offset: usize) -> i32 {
    let mut b = [0u8; 4];
    b.copy_from_slice(&buf[offset..offset + 4]);
    i32::from_le_bytes(b)
}

#[allow(clippy::too_many_arguments)]
fn process_spikes_stream(
    artifact_path: &Path,
    expected_record_count: u64,
    byte_stride: u64,
    voxel_size_a: f32,
    voxel_grid: &mut std::collections::HashMap<(i32, i32, i32), u32>,
) -> Result<StreamSpikeSummary> {
    let start = std::time::Instant::now();
    let f = File::open(artifact_path)
        .with_context(|| format!("open spikes {}", artifact_path.display()))?;
    let mut r = BufReader::with_capacity(1 << 20, f);
    let header = read_envelope_header(&mut r)?;

    let mut summary = StreamSpikeSummary {
        stream_id: header.stream_id,
        n_spikes: 0,
        bbox_min: [f32::INFINITY; 3],
        bbox_max: [f32::NEG_INFINITY; 3],
        intensity_min: f32::INFINITY,
        intensity_max: f32::NEG_INFINITY,
        intensity_mean: 0.0,
        ..Default::default()
    };

    let stride = byte_stride as usize;
    // Pin to the one record layout this reader understands. A bare `>= 72`
    // floor would silently accept an unknown layout and parse it at guessed
    // offsets, producing plausible coordinates with no diagnostic.
    if stride != prism_tokenizer::codec::records::GPU_SPIKE_EVENT_BYTES {
        anyhow::bail!(
            "spikes byte_stride {} is not the sealed packet record stride ({}); \
             refusing to parse an unknown record layout at guessed offsets",
            stride,
            prism_tokenizer::codec::records::GPU_SPIKE_EVENT_BYTES
        );
    }
    let inv_voxel = 1.0f32 / voxel_size_a;

    // Stream in batches of records to avoid loading 1.26 GB into RAM.
    let records_per_batch: u64 = 1 << 16; // 64K records (~6 MB at 96 B/record)
    let mut intensity_sum: f64 = 0.0;
    let mut remaining = header.record_count;
    let mut buf = vec![0u8; (records_per_batch * byte_stride) as usize];
    while remaining > 0 {
        let batch = std::cmp::min(remaining, records_per_batch);
        let bytes = (batch * byte_stride) as usize;
        r.read_exact(&mut buf[..bytes])?;
        for k in 0..(batch as usize) {
            let off = k * stride;
            let rec = &buf[off..off + stride];
            let px = read_f32_at(rec, spike_offsets::POSITION_X);
            let py = read_f32_at(rec, spike_offsets::POSITION_Y);
            let pz = read_f32_at(rec, spike_offsets::POSITION_Z);
            let intensity = read_f32_at(rec, spike_offsets::INTENSITY);
            let source = read_i32_at(rec, spike_offsets::SPIKE_SOURCE);
            let arom = read_i32_at(rec, spike_offsets::AROMATIC_TYPE);
            if px < summary.bbox_min[0] {
                summary.bbox_min[0] = px;
            }
            if py < summary.bbox_min[1] {
                summary.bbox_min[1] = py;
            }
            if pz < summary.bbox_min[2] {
                summary.bbox_min[2] = pz;
            }
            if px > summary.bbox_max[0] {
                summary.bbox_max[0] = px;
            }
            if py > summary.bbox_max[1] {
                summary.bbox_max[1] = py;
            }
            if pz > summary.bbox_max[2] {
                summary.bbox_max[2] = pz;
            }
            if intensity < summary.intensity_min {
                summary.intensity_min = intensity;
            }
            if intensity > summary.intensity_max {
                summary.intensity_max = intensity;
            }
            intensity_sum += intensity as f64;
            match source {
                1 => summary.n_uv += 1,
                2 => summary.n_lif += 1,
                _ => summary.n_other_source += 1,
            }
            match arom {
                0 => summary.n_aromatic_trp += 1,
                1 => summary.n_aromatic_tyr += 1,
                2 => summary.n_aromatic_phe += 1,
                3 => summary.n_aromatic_ss += 1,
                _ => summary.n_aromatic_none += 1,
            }
            // Voxel bin (sparse hashmap).
            let vx = (px * inv_voxel).floor() as i32;
            let vy = (py * inv_voxel).floor() as i32;
            let vz = (pz * inv_voxel).floor() as i32;
            *voxel_grid.entry((vx, vy, vz)).or_insert(0) += 1;
            summary.n_spikes += 1;
        }
        remaining -= batch;
    }
    if header.record_count != expected_record_count {
        anyhow::bail!(
            "spikes record_count mismatch: header {} vs manifest {}",
            header.record_count,
            expected_record_count
        );
    }
    if summary.n_spikes > 0 {
        summary.intensity_mean = intensity_sum / summary.n_spikes as f64;
    }
    summary.elapsed_ms = start.elapsed().as_millis() as u64;
    Ok(summary)
}

// ─── Voxel-density peak extraction (HONEST candidates, no ML rerank) ───────

#[derive(Debug, Serialize)]
struct DensityPeak {
    peak_id: u32,
    voxel_index: [i32; 3],
    centroid_a: [f32; 3],
    spike_count: u32,
    rank: u32,
    kind: &'static str,
}

// Count ties must not inherit randomized HashMap iteration order: they affect
// top-k support and greedy consolidation, not only presentation order.
fn count_desc_identity_asc<K: Ord, N: Ord>(a: &(K, N), b: &(K, N)) -> std::cmp::Ordering {
    b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0))
}

fn extract_top_density_peaks(
    voxel_grid: &std::collections::HashMap<(i32, i32, i32), u32>,
    voxel_size_a: f32,
    top_k: usize,
) -> Vec<DensityPeak> {
    let mut entries: Vec<((i32, i32, i32), u32)> =
        voxel_grid.iter().map(|(k, v)| (*k, *v)).collect();
    entries.sort_unstable_by(count_desc_identity_asc);
    entries
        .into_iter()
        .take(top_k)
        .enumerate()
        .map(|(idx, ((vx, vy, vz), count))| DensityPeak {
            peak_id: idx as u32,
            voxel_index: [vx, vy, vz],
            centroid_a: [
                (vx as f32 + 0.5) * voxel_size_a,
                (vy as f32 + 0.5) * voxel_size_a,
                (vz as f32 + 0.5) * voxel_size_a,
            ],
            spike_count: count,
            rank: idx as u32,
            kind: "voxel_density_peak",
        })
        .collect()
}

// ─── Filename → stream_id parser (sanity check vs header) ──────────────────

fn parse_stream_id_from_filename(p: &Path) -> Option<u32> {
    let name = p.file_name()?.to_string_lossy();
    // patterns: "..._streamN_..." or "..._stream0N_..."
    let idx = name.find("_stream")? + "_stream".len();
    let tail = &name[idx..];
    // collect digits
    let digits: String = tail.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse::<u32>().ok()
}

// ─── Topology loader (minimal — for residue centers + ligand reference) ──

const RESIDUE_ALIGNMENT_POLICY_ID: &str = "PRISMOS_TOPOLOGY_RESIDUE_INDEX_ZERO_BASED_V1";

#[derive(Debug, Default, Deserialize)]
struct TopologyResidueMin {
    #[serde(default)]
    residue_idx: Option<usize>,
    #[serde(default)]
    residue_id: Option<i32>,
    #[serde(default)]
    residue_name: Option<String>,
}

/// Missing legacy topology has no author claim; malformed JSON must not be
/// reclassified as missing metadata and silently admitted.
fn author_contract_present_in_topology(path: &Path) -> Result<bool> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error).with_context(|| format!("read topology {}", path.display())),
    };
    let value: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("malformed topology JSON {}", path.display()))?;
    anyhow::ensure!(value.is_object(), "topology JSON must be an object");
    Ok(value.get("analysis_residue_contract").is_some_and(|v| !v.is_null()))
}

#[derive(Debug, Deserialize)]
struct TopologyMin {
    #[serde(default)]
    analysis_residue_contract: Option<prism_nhs::input::AnalysisResidueContract>,
    n_atoms: usize,
    n_residues: usize,
    positions: Vec<f32>,
    #[serde(default)]
    elements: Vec<String>,
    #[serde(default)]
    residue_ids: Vec<i32>,
    #[serde(default)]
    residue_names: Vec<String>,
    #[serde(default)]
    residues: Vec<TopologyResidueMin>,
    #[serde(default)]
    residue_to_atom_indices: std::collections::HashMap<String, Vec<usize>>,
    #[serde(default)]
    source_pdb: Option<String>,
    ca_indices: Vec<i32>,
}

impl TopologyMin {
    fn load(path: &Path) -> Result<Self> {
        let s = std::fs::read_to_string(path)
            .with_context(|| format!("read topology {}", path.display()))?;
        let t: TopologyMin = serde_json::from_str(&s)
            .with_context(|| format!("parse topology {}", path.display()))?;
        anyhow::ensure!(
            t.positions.len() == 3 * t.n_atoms,
            "topology positions len {} != 3 * n_atoms {}",
            t.positions.len(),
            t.n_atoms
        );
        if let Some(contract) = &t.analysis_residue_contract {
            contract.validate_author_mapping().map_err(anyhow::Error::msg)?;
            anyhow::ensure!(contract.n_atoms == t.n_atoms
                && contract.total_physical_residue_count == t.n_residues,
                "author residue contract does not match topology dimensions");
            anyhow::ensure!(contract.n_analysis_residues == t.n_residues
                && contract.analysis_residues.iter().all(|row| row.analysis_residue_id == row.physical_residue_id),
                "materializer physical/analysis residue axes differ; explicit geometry projection required");
        }
        Ok(t)
    }

    fn author_references(&self, ids: &[i32]) -> Result<serde_json::Value> {
        match &self.analysis_residue_contract {
            Some(contract) => Ok(serde_json::to_value(contract.author_references(ids).map_err(anyhow::Error::msg)?)?),
            None => Ok(serde_json::Value::Null),
        }
    }

    fn residue_names_by_residue(&self) -> Vec<String> {
        let mut names = vec![String::new(); self.n_residues];
        for (fallback_idx, row) in self.residues.iter().enumerate() {
            let rid = row
                .residue_idx
                .or_else(|| row.residue_id.and_then(|id| usize::try_from(id).ok()))
                .unwrap_or(fallback_idx);
            if rid < self.n_residues {
                if let Some(name) = &row.residue_name {
                    names[rid] = name.clone();
                }
            }
        }
        if names.iter().all(|name| name.is_empty()) && self.residue_names.len() == self.n_residues {
            return self.residue_names.clone();
        }
        if self.residue_names.len() == self.n_atoms {
            for rid in 0..self.n_residues {
                if !names[rid].is_empty() {
                    continue;
                }
                let key = rid.to_string();
                if let Some(atom_indices) = self.residue_to_atom_indices.get(&key) {
                    if let Some(atom_idx) = atom_indices.first().copied() {
                        if atom_idx < self.residue_names.len() {
                            names[rid] = self.residue_names[atom_idx].clone();
                        }
                    }
                }
            }
        }
        names
    }

    fn residue_to_atom_stats(&self) -> (usize, usize, usize) {
        let mapped_residues = self
            .residue_to_atom_indices
            .values()
            .filter(|indices| !indices.is_empty())
            .count();
        let empty_residues = self
            .residue_to_atom_indices
            .values()
            .filter(|indices| indices.is_empty())
            .count();
        let mapped_atoms = self
            .residue_to_atom_indices
            .values()
            .map(std::vec::Vec::len)
            .sum();
        (mapped_residues, empty_residues, mapped_atoms)
    }

    fn residue_alignment_policy_json(&self, topology_path: &Path) -> serde_json::Value {
        let (mapped_residues, empty_residues, mapped_atoms) = self.residue_to_atom_stats();
        serde_json::json!({
            "policy_id": RESIDUE_ALIGNMENT_POLICY_ID,
            "topology_path": topology_path.display().to_string(),
            "source_pdb": self.source_pdb.clone(),
            "residue_coordinate_system": "zero_based_topology_residue_idx",
            "residue_name_source_precedence": [
                "topology.residues[].residue_name",
                "topology.residue_names via residue_to_atom_indices",
                "topology.residue_names only when residue-length"
            ],
            "external_numbering_requires_residue_map": true,
            "residue_map_file_hint": "same stem as topology with .residue_map.json",
            "n_atoms": self.n_atoms,
            "n_residues": self.n_residues,
            "residue_names_len": self.residue_names.len(),
            "residues_table_len": self.residues.len(),
            "residue_to_atom_indices_len": self.residue_to_atom_indices.len(),
            "residue_to_atom_mapped_residues": mapped_residues,
            "residue_to_atom_empty_residues": empty_residues,
            "residue_to_atom_mapped_atoms": mapped_atoms,
            "residue_to_atom_status": if mapped_residues == self.n_residues && empty_residues == 0 {
                "complete"
            } else if mapped_residues > 0 {
                "partial"
            } else {
                "missing"
            },
        })
    }

    /// CA position per residue_id. Returns Vec<Option<[f32;3]>> indexed by residue_id.
    fn ca_positions_by_residue(&self) -> Vec<Option<[f32; 3]>> {
        let mut out: Vec<Option<[f32; 3]>> = vec![None; self.n_residues];
        for (rid, &ca_idx) in self.ca_indices.iter().enumerate() {
            if ca_idx < 0 || (ca_idx as usize) >= self.n_atoms {
                continue;
            }
            let i = (ca_idx as usize) * 3;
            out[rid] = Some([
                self.positions[i],
                self.positions[i + 1],
                self.positions[i + 2],
            ]);
        }
        out
    }
}

// ─── Ground truth loader (for accuracy validation) ──────────────────────────

#[derive(Debug, Deserialize)]
struct GroundTruth {
    pdb_id: Option<String>,
    target_chain: Option<String>,
    valid_for_dcc_validation: bool,
    skip_reason: Option<String>,
    ligand_centroid: Option<[f32; 3]>,
    #[serde(default)]
    #[allow(dead_code)]
    ligand: Option<serde_json::Value>,
}

// ─── KCC v2-full reader (PRKCC001) ──────────────────────────────────────────

#[derive(Debug, Clone)]
struct KccPerStream {
    lag_provenance: Vec<prism_nhs::causome::KccLagProvenance>,
    lag_provenance_source: Option<serde_json::Value>,
    lag_support: Vec<prism_nhs::causome::KccLagSupportResidue>,
    lag_support_source: Option<serde_json::Value>,
    stream_id: u32,
    n_residues: usize,
    active_causal: Vec<u32>,
    decoded: prism_tokenizer::codec::kcc::Prkcc001Decoded,
    resolved_source: PathBuf,
}

fn decode_kcc_full(raw: &[u8]) -> Result<KccPerStream> {
    let decoded = prism_tokenizer::codec::kcc::decode_prkcc001(raw)
        .map_err(|e| anyhow::anyhow!("PRKCC001: {e}"))?;
    let h = &decoded.header;
    anyhow::ensure!(
        raw.len() == h.payload_offset + h.payload_size + 16,
        "PRKCC001 unexpected bytes after envelope"
    );
    let field_count =
        u64::from_le_bytes(raw[h.payload_offset + 8..h.payload_offset + 16].try_into()?) as usize;
    anyhow::ensure!(
        field_count == decoded.f32_fields.len() + decoded.u32_fields.len(),
        "PRKCC001 duplicate field names"
    );
    anyhow::ensure!(
        !decoded
            .f32_fields
            .keys()
            .any(|k| decoded.u32_fields.contains_key(k)),
        "PRKCC001 conflicting typed field names"
    );
    for field in ["active_causal", "residue_count"] {
        anyhow::ensure!(
            decoded.u32_fields.get(field).map(Vec::len) == Some(h.record_count),
            "PRKCC001 missing or invalid required u32 field: {}",
            field
        );
    }
    Ok(KccPerStream {
        lag_provenance: Vec::new(),
        lag_provenance_source: None,
        lag_support: Vec::new(),
        lag_support_source: None,
        stream_id: h.stream_id,
        n_residues: h.record_count,
        active_causal: decoded.u32_fields["active_causal"].clone(),
        decoded,
        resolved_source: PathBuf::new(),
    })
}

fn validate_kcc_association(
    k: &KccPerStream,
    run_id: &str,
    stem: &str,
    stream_id: Option<u32>,
) -> Result<()> {
    anyhow::ensure!(
        k.decoded.header.run_id == run_id
            && k.decoded.header.stem == stem
            && stream_id == Some(k.stream_id),
        "KCC manifest run/stem/stream identity mismatch"
    );
    Ok(())
}

fn attach_kcc_lag_support(k: &mut KccPerStream, path: &Path) -> Result<()> {
    let bytes = std::fs::read(path)?;
    let archive: prism_nhs::causome::KccLagSupportArchive = serde_json::from_slice(&bytes)?;
    anyhow::ensure!(
        archive.schema_kind == "prism.kcc_lag_support.v1"
            && archive.schema_version == 1
            && archive.run_id == k.decoded.header.run_id
            && archive.stem == k.decoded.header.stem
            && archive.stream_id == k.stream_id,
        "KCC lag-support sidecar identity mismatch"
    );
    let source_hash = prism_nhs::ensemble::manifest::sha256_file(&k.resolved_source)?;
    anyhow::ensure!(
        archive.kcc_v2full_sha256.as_deref() == Some(source_hash.as_str()),
        "KCC lag-support sidecar payload SHA256 mismatch"
    );
    anyhow::ensure!(
        archive.rows.len() == k.n_residues,
        "KCC lag-support row count mismatch"
    );
    let support_sha256 = prism_nhs::ensemble::manifest::sha256_file(path)?;
    if let Some(expected) = k
        .lag_provenance_source
        .as_ref()
        .and_then(|v| v.get("lag_support_ref"))
        .and_then(|v| v.get("sha256"))
        .and_then(|v| v.as_str())
    {
        anyhow::ensure!(
            expected == support_sha256,
            "KCC lag-support SHA256 disagrees with lag provenance reference"
        );
    }
    const KCC_SUPPORT_RING_STEPS: usize = 64;
    const KCC_SUPPORT_EVENT_BYTES: usize = 20;
    const KCC_SUPPORT_RING_HEX_LEN: usize = KCC_SUPPORT_RING_STEPS * KCC_SUPPORT_EVENT_BYTES * 2;
    for (rid, row) in archive.rows.iter().enumerate() {
        anyhow::ensure!(
            row.residue_index as usize == rid
                && row.source_stream_id == Some(k.stream_id)
                && row.observation_count == k.decoded.u32_fields["residue_count"][rid]
                && row.active_causal_count == k.decoded.u32_fields["active_causal"][rid]
                && row.final_causal_lag_f32_bits
                    == k.decoded
                        .f32_fields
                        .get("causal_lag")
                        .and_then(|v| v.get(rid))
                        .map(|v| v.to_bits())
                        .unwrap_or(f32::NAN.to_bits())
                && row.final_lag_corr_peak_f32_bits
                    == k.decoded
                        .f32_fields
                        .get("lag_corr_peak")
                        .and_then(|v| v.get(rid))
                        .map(|v| v.to_bits())
                        .unwrap_or(f32::NAN.to_bits()),
            "KCC lag-support source measurement mismatch at residue {}",
            rid
        );
        if let Some(hex) = &row.ring_raw_slots_hex {
            anyhow::ensure!(
                hex.len() == KCC_SUPPORT_RING_HEX_LEN,
                "KCC lag-support ring hex length mismatch at residue {}",
                rid
            );
            anyhow::ensure!(
                hex.bytes().all(|b| b.is_ascii_hexdigit()),
                "KCC lag-support ring hex contains non-hex byte at residue {}",
                rid
            );
        }
        if let Some(count) = row.ring_event_count {
            anyhow::ensure!(
                count as usize == row.ordered_events.len(),
                "KCC lag-support ordered event count mismatch at residue {}",
                rid
            );
        }
        for (idx, event) in row.ordered_events.iter().enumerate() {
            anyhow::ensure!(
                event.ordered_index as usize == idx
                    && event.ring_slot < KCC_SUPPORT_RING_STEPS as u32,
                "KCC lag-support ordered event identity mismatch at residue {} event {}",
                rid,
                idx
            );
        }
        anyhow::ensure!(
            row.winning_pair_count as usize == row.winning_pairs.len(),
            "KCC lag-support winning pair count mismatch at residue {}",
            rid
        );
        for (pair_idx, pair) in row.winning_pairs.iter().enumerate() {
            let causal = row
                .ordered_events
                .get(pair.causal_event_ordered_index as usize)
                .with_context(|| {
                    format!(
                        "KCC lag-support causal pair index out of range at residue {} pair {}",
                        rid, pair_idx
                    )
                })?;
            let response = row
                .ordered_events
                .get(pair.response_event_ordered_index as usize)
                .with_context(|| {
                    format!(
                        "KCC lag-support response pair index out of range at residue {} pair {}",
                        rid, pair_idx
                    )
                })?;
            anyhow::ensure!(
                pair.causal_ring_slot == causal.ring_slot
                    && pair.response_ring_slot == response.ring_slot
                    && pair.causal_step == causal.step
                    && pair.response_step == response.step
                    && pair.causal_count == causal.causal_count
                    && pair.causal_flags_raw == causal.flags_raw
                    && pair.response_flags_raw == response.flags_raw
                    && pair.response_dx.ieee754_bits == response.dx.ieee754_bits
                    && pair.response_dy.ieee754_bits == response.dy.ieee754_bits
                    && pair.response_dz.ieee754_bits == response.dz.ieee754_bits
                    && pair.response_motion_magnitude.ieee754_bits
                        == response.motion_magnitude.ieee754_bits,
                "KCC lag-support pair does not match ordered event payload at residue {} pair {}",
                rid,
                pair_idx
            );
            if let Some(lag) = row.gpu_requested_lag_md_steps {
                anyhow::ensure!(
                    pair.requested_gpu_lag_md_steps == Some(lag),
                    "KCC lag-support GPU lag mismatch at residue {} pair {}",
                    rid,
                    pair_idx
                );
            }
            if let Some(offset) = row.host_winning_event_index_offset {
                anyhow::ensure!(
                    pair.host_event_index_offset == Some(offset),
                    "KCC lag-support host offset mismatch at residue {} pair {}",
                    rid,
                    pair_idx
                );
            }
        }
    }
    k.lag_support = archive.rows;
    k.lag_support_source = Some(serde_json::json!({
        "path": path.display().to_string(),
        "sha256": support_sha256,
        "kcc_v2full_sha256": source_hash,
        "schema_kind": "prism.kcc_lag_support.v1",
    }));
    Ok(())
}

fn attach_kcc_lag_provenance(k: &mut KccPerStream, path: &Path) -> Result<()> {
    let bytes = std::fs::read(path)?;
    let value: serde_json::Value = serde_json::from_slice(&bytes)?;
    anyhow::ensure!(
        value["schema_kind"] == "prism.kcc_lag_provenance.v1"
            && value["schema_version"] == 1
            && value["run_id"] == k.decoded.header.run_id
            && value["stem"] == k.decoded.header.stem
            && value["stream_id"].as_u64() == Some(k.stream_id as u64)
            && value["n_residues"].as_u64() == Some(k.n_residues as u64),
        "KCC lag sidecar identity mismatch"
    );
    let source_hash = prism_nhs::ensemble::manifest::sha256_file(&k.resolved_source)?;
    anyhow::ensure!(
        value["kcc_v2full_sha256"].as_str() == Some(source_hash.as_str()),
        "KCC lag sidecar payload SHA256 mismatch"
    );
    let lag_support_ref = value
        .get("lag_support_ref")
        .cloned()
        .filter(|v| !v.is_null());
    if let Some(reference) = &lag_support_ref {
        anyhow::ensure!(
            reference["schema_kind"] == "prism.kcc_lag_support.v1"
                && reference["sha256"].as_str().is_some(),
            "KCC lag sidecar lag_support_ref is incomplete"
        );
        if let (Some(path_str), Some(expected_sha)) =
            (reference["path"].as_str(), reference["sha256"].as_str())
        {
            let reference_path = PathBuf::from(path_str);
            let candidates = [
                if reference_path.is_absolute() {
                    reference_path.clone()
                } else {
                    path.parent()
                        .unwrap_or_else(|| Path::new("."))
                        .join(&reference_path)
                },
                path.with_extension("lag_support.json"),
            ];
            for candidate in candidates.iter().filter(|candidate| candidate.exists()) {
                let observed = prism_nhs::ensemble::manifest::sha256_file(candidate)?;
                anyhow::ensure!(
                    observed == expected_sha,
                    "KCC lag sidecar lag_support_ref SHA256 mismatch for {}",
                    candidate.display()
                );
            }
        }
    }
    let rows: Vec<prism_nhs::causome::KccLagProvenance> =
        serde_json::from_value(value["rows"].clone())?;
    anyhow::ensure!(
        rows.len() == k.n_residues,
        "KCC lag sidecar row count mismatch"
    );
    for (rid, row) in rows.iter().enumerate() {
        anyhow::ensure!(
            row.source_stream_id == Some(k.stream_id)
                && row.observation_count == k.decoded.u32_fields["residue_count"][rid]
                && row.active_causal_count == k.decoded.u32_fields["active_causal"][rid]
                && row.causal_lag_f32_bits
                    == k.decoded
                        .f32_fields
                        .get("causal_lag")
                        .and_then(|v| v.get(rid))
                        .map(|v| v.to_bits())
                && row.lag_corr_peak_f32_bits
                    == k.decoded
                        .f32_fields
                        .get("lag_corr_peak")
                        .and_then(|v| v.get(rid))
                        .map(|v| v.to_bits()),
            "KCC lag sidecar source measurement mismatch at residue {}",
            rid
        );
    }
    k.lag_provenance = rows;
    k.lag_provenance_source = Some(
        serde_json::json!({"path":path.display().to_string(),"sha256":prism_nhs::ensemble::manifest::sha256_file(path)?,"kcc_v2full_sha256":source_hash,"lag_support_ref":lag_support_ref}),
    );
    Ok(())
}

fn read_kcc_v2full(path: &Path) -> Result<KccPerStream> {
    let mut k = decode_kcc_full(
        &std::fs::read(path).with_context(|| format!("open kcc {}", path.display()))?,
    )?;
    k.resolved_source = path.to_path_buf();
    let sidecar = path.with_extension("lag_provenance.json");
    if sidecar.exists() {
        attach_kcc_lag_provenance(&mut k, &sidecar)?;
    }
    let support = path.with_extension("lag_support.json");
    if support.exists() {
        attach_kcc_lag_support(&mut k, &support)?;
    }
    Ok(k)
}

// JSON numbers cannot represent NaN/Inf or preserve NaN payloads. Retain the
// exact IEEE-754 bits alongside a nullable numeric view, never silently zero.
fn kcc_f32_value(value: f32) -> serde_json::Value {
    serde_json::json!({
        "value": if value.is_finite() { Some(value) } else { None },
        "classification": if value.is_nan() { "nan" } else if value == f32::INFINITY {
            "positive_infinity"
        } else if value == f32::NEG_INFINITY { "negative_infinity" } else { "finite" },
        "ieee754_bits_hex": format!("{:08x}", value.to_bits()),
    })
}

fn kcc_descriptor_surface(k: &KccPerStream, source: &str) -> serde_json::Value {
    let f32_fields: serde_json::Map<String, serde_json::Value> = k.decoded.f32_fields.iter()
        .map(|(name, values)| (name.clone(), serde_json::json!({
            "dtype": "f32", "values": values.iter().copied().map(kcc_f32_value).collect::<Vec<_>>()
        }))).collect();
    let u32_fields: serde_json::Map<String, serde_json::Value> = k
        .decoded
        .u32_fields
        .iter()
        .map(|(name, values)| {
            (
                name.clone(),
                serde_json::json!({"dtype": "u32", "values": values}),
            )
        })
        .collect();
    serde_json::json!({
        "stream_id": k.stream_id, "run_id": k.decoded.header.run_id,
        "stem": k.decoded.header.stem, "source": source,
        "source_provenance": surface_source_status(Path::new(source), "kcc_v2full", &k.resolved_source),
        "checksum_fnv1a_64_hex": format!("{:016x}", k.decoded.checksum),
        "integrity_ok": k.decoded.integrity_ok, "n_residues": k.n_residues,
        "residue_identity": "zero_based_native_descriptor_index; topology mapping required for external numbering",
        "f32_fields": f32_fields, "u32_fields": u32_fields,
        "lag_provenance": k.lag_provenance, "lag_provenance_source": k.lag_provenance_source,
        "lag_support_source": k.lag_support_source,
        "lag_support_summary": {
            "rows_present": k.lag_support.len(),
            "rows_with_winning_pairs": k.lag_support.iter().filter(|row| row.winning_pair_count > 0).count(),
            "rows_with_raw_ring_slots": k.lag_support.iter().filter(|row| row.ring_raw_slots_hex.is_some()).count(),
        },
        "sample_count_field": if k.decoded.u32_fields.contains_key("residue_count") { Some("residue_count") } else { None },
        "sample_count_semantics": "number_of_KCC_accumulations_not_MD_step_count",
        "absent_fields": "unavailable; absence never implies zero",
        "missing_expected_fields": (["temporal_corr","direction_score","motion_efficiency","burst_motion",
            "phase_shift","causal_lag","lag_corr_peak","local_cov","net_dx","net_dy","net_dz",
            "sum_m","residue_count","active_causal"].iter().filter(|name|
                !k.decoded.f32_fields.contains_key(**name) && !k.decoded.u32_fields.contains_key(**name)).collect::<Vec<_>>()),
        "direction_score_semantics": "net_displacement_over_path_length; not directed causal influence",
        "temporal_corr_semantics": "signed displacement_magnitude versus UV_LIF_linked_event_count correlation",
        "lag_semantics": {
            "native_unit": "MD_step", "estimator_origin": if k.lag_provenance.is_empty() { "legacy_unknown" } else { "per_residue_hash_bound_sidecar" },
            "estimators_not_interchangeable": true,
            "physical_time_conversion": "not_evaluated; join recorded adaptive_dt timeline before conversion",
            "directed_source_target_identity": "unavailable_in_this_descriptor_schema"
        }
    })
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ThermalSchedule {
    stream_index: u32,
    cold_hold_end: i64,
    ramp_end: i64,
    warm_hold_end: i64,
    ramp_down_end: i64,
    total_steps: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    current_step: Option<i64>,
    steering_flags: u32,
}

impl ThermalSchedule {
    fn validate(&self, expected_stream: u32) -> Result<()> {
        anyhow::ensure!(
            self.stream_index == expected_stream,
            "protocol stream identity mismatch"
        );
        anyhow::ensure!(
            self.total_steps > 0
                && 0 <= self.cold_hold_end
                && self.cold_hold_end <= self.ramp_end
                && self.ramp_end <= self.warm_hold_end
                && self.warm_hold_end <= self.ramp_down_end
                && self.ramp_down_end <= self.total_steps,
            "invalid protocol schedule boundaries"
        );
        anyhow::ensure!(self.current_step.is_none_or(|step| step >= 0), "negative executed protocol step");
        Ok(())
    }

    // Matches protocol_director.cu schedule branches, including 0x10 warm tail.
    // These are prescribed schedule phases, not measured temperatures.
    fn phase(&self, step: i32) -> usize {
        let step = i64::from(step);
        // Native protocol_director holds its terminal phase beyond nominal
        // total_steps. Admit that continuation only within the recorded executed
        // clock; a historical schedule without that clock stays nominal-bounded.
        if step < 0 || step > self.current_step.unwrap_or(self.total_steps) {
            6
        } else if step < self.cold_hold_end {
            0
        } else if step < self.ramp_end {
            1
        } else if step < self.warm_hold_end {
            2
        } else if step < self.ramp_down_end {
            3
        } else if self.steering_flags & 0x10 != 0 {
            5
        } else {
            4
        }
    }
}

const THERMAL_PHASE_NAMES: [&str; 7] = [
    "cold_hold",
    "ramp_up",
    "warm_hold",
    "ramp_down",
    "cold_return",
    "warm_tail",
    "unclassified",
];

#[derive(Debug, Default, Clone, Serialize)]
struct ThermalEvidence {
    counts: [u64; 7],
    position_sums: [[f64; 3]; 7],
}
impl ThermalEvidence {
    fn observe(&mut self, phase: usize, position: [f32; 3]) {
        self.counts[phase] += 1;
        for axis in 0..3 {
            self.position_sums[phase][axis] += position[axis] as f64;
        }
    }
    fn merge(&mut self, other: &Self) {
        for phase in 0..7 {
            self.counts[phase] += other.counts[phase];
            for axis in 0..3 {
                self.position_sums[phase][axis] += other.position_sums[phase][axis];
            }
        }
    }
    fn centroid(&self, phase: usize) -> Option<[f32; 3]> {
        // Preserve existing 50-spike centroid admission threshold.
        (self.counts[phase] >= 50)
            .then(|| vec3_div(self.position_sums[phase], self.counts[phase] as f64))
    }
}

fn parse_thermal_schedule(
    value: serde_json::Value,
    expected_stream: u32,
    run_id: &str,
) -> Result<ThermalSchedule> {
    if let Some(embedded_run) = value.get("run_id") {
        anyhow::ensure!(
            embedded_run.as_str() == Some(run_id),
            "protocol run identity mismatch"
        );
    }
    let schedule: ThermalSchedule = serde_json::from_value(value)?;
    schedule.validate(expected_stream)?;
    Ok(schedule)
}

fn read_thermal_schedule(
    path: &Path,
    expected_stream: u32,
    run_id: &str,
) -> Result<ThermalSchedule> {
    parse_thermal_schedule(
        serde_json::from_reader(File::open(path)?)?,
        expected_stream,
        run_id,
    )
}

fn adaptive_dt_time_axis(
    path: &Path,
    schedule: Option<&ThermalSchedule>,
) -> Result<serde_json::Value> {
    adaptive_dt_time_axis_bytes(&std::fs::read(path)?, schedule)
}

fn adaptive_dt_time_axis_bytes(
    bytes: &[u8],
    schedule: Option<&ThermalSchedule>,
) -> Result<serde_json::Value> {
    anyhow::ensure!(
        !bytes.is_empty() && bytes.len() % 32 == 0,
        "invalid adaptive_dt record length"
    );
    let rows: Vec<(u64, u64, f64)> = bytes
        .chunks_exact(32)
        .map(|r| {
            (
                u64::from_le_bytes(r[0..8].try_into().unwrap()),
                u64::from_le_bytes(r[8..16].try_into().unwrap()),
                f64::from_le_bytes(r[16..24].try_into().unwrap()),
            )
        })
        .collect();
    let valid_dt = rows.iter().all(|r| r.2.is_finite() && r.2 > 0.0);
    let constant_dt = valid_dt && rows.iter().all(|r| r.2.to_bits() == rows[0].2.to_bits());
    let cadence = rows
        .get(1)
        .and_then(|r| r.1.checked_sub(rows[0].1))
        .filter(|s| *s > 0);
    let contiguous = cadence
        .map(|c| {
            rows.windows(2)
                .all(|w| w[1].0 == w[0].0 + 1 && w[1].1.checked_sub(w[0].1) == Some(c))
        })
        .unwrap_or(false);
    let coverage = schedule
        .zip(cadence)
        .map(|(s, c)| {
            rows[0].1 == 0 && rows.last().unwrap().1.checked_add(c) == Some(s.total_steps as u64)
        })
        .unwrap_or(false);
    let validated_fixed = constant_dt && contiguous && coverage;
    Ok(serde_json::json!({
        "record_count": rows.len(), "first_step": rows[0].1, "last_step": rows.last().unwrap().1,
        "recorded_step_cadence": cadence, "valid_positive_dt": valid_dt,
        "all_recorded_dt_equal": constant_dt, "contiguous_chunk_and_step_records": contiguous,
        "covers_declared_schedule": coverage,
        "fixed_dt_ps": if validated_fixed { Some(rows[0].2) } else { None },
        "physical_time_conversion_status": if validated_fixed { "validated_constant_recorded_dt" } else { "unavailable_fixed_dt_mapping" },
        "source_semantics": "chunk_start_MD_step_and_integrator_dt; cadence_is_not_dt",
    }))
}

// ─── First-class PrismOS surface parsers ───────────────────────────────────

const WARP_ENTRY_BYTES: usize = 136;
const WARP_VOXEL_INDEX_OFFSET: usize = 0;
const WARP_ATOM_INDEX_OFFSET: usize = 4;
const WARP_ATOM_WEIGHT_OFFSET: usize = 68;
const WARP_N_ATOMS_OFFSET: usize = 132;
const WARP_MAX_ATOMS: usize = 16;

fn finite_min_max_update(value: f64, min_v: &mut f64, max_v: &mut f64) {
    if value.is_finite() {
        if value < *min_v {
            *min_v = value;
        }
        if value > *max_v {
            *max_v = value;
        }
    }
}

fn stream_id_for_artifact(art: &ManifestArtifact, path: &Path) -> Option<u32> {
    art.stream_id
        .or_else(|| parse_stream_id_from_filename(path))
}

fn surface_source_status(
    artifact_path: &Path,
    kind: &str,
    resolved_path: &Path,
) -> serde_json::Value {
    let packet_path = canonical_packet_path_for(artifact_path, kind).ok();
    let packet_present = packet_path.as_ref().map(|p| p.exists()).unwrap_or(false);
    serde_json::json!({
        "manifest_path": artifact_path.display().to_string(),
        "resolved_path": resolved_path.display().to_string(),
        "canonical_packet_path": packet_path.as_ref().map(|p| p.display().to_string()),
        "canonical_packet_present": packet_present,
        "source_kind": if packet_present && resolved_path != artifact_path {
            "hydrated_from_canonical_packet"
        } else if resolved_path == artifact_path {
            "raw_manifest_path_no_canonical_packet_platform_violation"
        } else {
            "hydrated_from_nonstandard_source_platform_violation"
        },
        "platform_canonicalization_status": if packet_present {
            "canonical_packet_consumed"
        } else {
            "raw_only_platform_violation"
        },
    })
}

fn summarize_signal_grid(path: &Path, stream_id: Option<u32>) -> Result<serde_json::Value> {
    let f = File::open(path).with_context(|| format!("open signal grid {}", path.display()))?;
    let mut r = BufReader::with_capacity(1 << 20, f);
    let h = read_envelope_header(&mut r)?;
    anyhow::ensure!(&h.magic == b"PRSGD001", "signal grid bad magic");
    let grid_dim = read_u64_le(&mut r)?;
    let voxel_count = read_u64_le(&mut r)? as usize;
    anyhow::ensure!(
        voxel_count as u64 == h.record_count,
        "signal grid voxel_count {} != record_count {}",
        voxel_count,
        h.record_count
    );
    let mut sections: Vec<Vec<i32>> = Vec::with_capacity(4);
    for _ in 0..4 {
        let mut values = vec![0i32; voxel_count];
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(values.as_mut_ptr() as *mut u8, voxel_count * 4)
        };
        r.read_exact(bytes)?;
        sections.push(values);
    }
    let mut checksum = [0u8; 8];
    let mut trailer = [0u8; 8];
    r.read_exact(&mut checksum)?;
    r.read_exact(&mut trailer)?;
    anyhow::ensure!(&trailer == b"PRSGDEND", "signal grid bad trailer");

    let voxel_hit = &sections[0];
    let coupled = &sections[1];
    let residue_id = &sections[2];
    let residue_count = &sections[3];
    let nonzero_voxel_hits = voxel_hit.iter().filter(|&&v| v > 0).count();
    let nonzero_coupled = coupled.iter().filter(|&&v| v > 0).count();
    let hit_sum: i64 = voxel_hit.iter().map(|&v| v as i64).sum();
    let coupled_sum: i64 = coupled.iter().map(|&v| v as i64).sum();
    let residue_voxels = residue_id.iter().filter(|&&v| v >= 0).count();
    let residue_support_sum: i64 = residue_count.iter().map(|&v| v.max(0) as i64).sum();
    Ok(serde_json::json!({
        "kind": "signal_grid",
        "status": "materialized_parsed",
        "stream_id": stream_id.or(Some(h.stream_id)),
        "grid_dim": grid_dim,
        "voxel_count": voxel_count,
        "nonzero_voxel_hit_count": nonzero_voxel_hits,
        "nonzero_coupled_spike_count": nonzero_coupled,
        "voxel_hit_sum": hit_sum,
        "coupled_spike_sum": coupled_sum,
        "primary_residue_voxel_count": residue_voxels,
        "primary_residue_support_sum": residue_support_sum,
        "max_voxel_hit": voxel_hit.iter().copied().max().unwrap_or(0),
        "max_coupled_spike": coupled.iter().copied().max().unwrap_or(0),
        "schema": "PRSGD001:grid_dim,voxel_count,voxel_hit_grid,coupled_spike_grid,primary_residue_id,primary_residue_count",
    }))
}

fn summarize_vector_tensor(
    path: &Path,
    kind: &str,
    stream_id: Option<u32>,
) -> Result<serde_json::Value> {
    let bytes =
        std::fs::read(path).with_context(|| format!("read vector tensor {}", path.display()))?;
    anyhow::ensure!(
        !bytes.is_empty() && bytes.len() % 12 == 0,
        "{} vector tensor length {} is not f32[n x 3]",
        path.display(),
        bytes.len()
    );
    let mut n_vectors = 0usize;
    let mut norm_sum = 0.0f64;
    let mut norm_min = f64::INFINITY;
    let mut norm_max = f64::NEG_INFINITY;
    let mut nonfinite_count = 0usize;
    for chunk in bytes.chunks_exact(12) {
        let x = f32::from_le_bytes(chunk[0..4].try_into()?);
        let y = f32::from_le_bytes(chunk[4..8].try_into()?);
        let z = f32::from_le_bytes(chunk[8..12].try_into()?);
        let norm = ((x as f64).mul_add(
            x as f64,
            (y as f64).mul_add(y as f64, (z as f64) * (z as f64)),
        ))
        .sqrt();
        if norm.is_finite() {
            norm_sum += norm;
            finite_min_max_update(norm, &mut norm_min, &mut norm_max);
        } else {
            nonfinite_count += 1;
        }
        n_vectors += 1;
    }
    let finite_vectors = n_vectors.saturating_sub(nonfinite_count);
    Ok(serde_json::json!({
        "kind": kind,
        "status": "materialized_parsed",
        "stream_id": stream_id,
        "layout": "raw f32 AoS n_vectors x 3",
        "size_bytes": bytes.len(),
        "n_vectors": n_vectors,
        "finite_vectors": finite_vectors,
        "nonfinite_vectors": nonfinite_count,
        "norm_mean": if finite_vectors > 0 { serde_json::Value::from(norm_sum / finite_vectors as f64) } else { serde_json::Value::Null },
        "norm_min": if finite_vectors > 0 { serde_json::Value::from(norm_min) } else { serde_json::Value::Null },
        "norm_max": if finite_vectors > 0 { serde_json::Value::from(norm_max) } else { serde_json::Value::Null },
    }))
}

fn summarize_warp_matrix(path: &Path, stream_id: Option<u32>) -> Result<serde_json::Value> {
    let len = std::fs::metadata(path)
        .with_context(|| format!("stat warp matrix {}", path.display()))?
        .len() as usize;
    anyhow::ensure!(
        len > 0 && len % WARP_ENTRY_BYTES == 0,
        "{} is not aligned to {}-byte GpuWarpEntry records",
        path.display(),
        WARP_ENTRY_BYTES
    );
    let mut file =
        File::open(path).with_context(|| format!("open warp matrix {}", path.display()))?;
    let mut record = [0u8; WARP_ENTRY_BYTES];
    let mut record_count = 0usize;
    let mut nonempty_voxels = 0usize;
    let mut voxel_idx_mismatch = 0usize;
    let mut invalid_n_atoms = 0usize;
    let mut invalid_atom_refs = 0usize;
    let mut nonfinite_weights = 0usize;
    let mut weight_abs_sum = 0.0f64;
    let mut weight_n = 0usize;
    for expected_idx in 0..(len / WARP_ENTRY_BYTES) {
        file.read_exact(&mut record)?;
        let voxel_idx = read_i32_at(&record, WARP_VOXEL_INDEX_OFFSET);
        if voxel_idx >= 0 && voxel_idx as usize != expected_idx {
            voxel_idx_mismatch += 1;
        }
        let n_atoms_raw = read_i32_at(&record, WARP_N_ATOMS_OFFSET);
        if n_atoms_raw < 0 || n_atoms_raw as usize > WARP_MAX_ATOMS {
            invalid_n_atoms += 1;
        }
        let n_atoms = n_atoms_raw.clamp(0, WARP_MAX_ATOMS as i32) as usize;
        if n_atoms > 0 {
            nonempty_voxels += 1;
        }
        for slot in 0..n_atoms {
            let atom_idx = read_i32_at(&record, WARP_ATOM_INDEX_OFFSET + slot * 4);
            let weight = read_f32_at(&record, WARP_ATOM_WEIGHT_OFFSET + slot * 4);
            if atom_idx < 0 {
                invalid_atom_refs += 1;
            }
            if weight.is_finite() {
                weight_abs_sum += (weight as f64).abs();
                weight_n += 1;
            } else {
                nonfinite_weights += 1;
            }
        }
        record_count += 1;
    }
    Ok(serde_json::json!({
        "kind": "warp_matrix",
        "status": "materialized_parsed",
        "stream_id": stream_id,
        "layout": "GpuWarpEntry: i32 voxel_idx, i32 atom_indices[16], f32 atom_weights[16], i32 n_atoms; 136 bytes",
        "size_bytes": len,
        "record_count": record_count,
        "nonempty_voxels": nonempty_voxels,
        "empty_voxels": record_count.saturating_sub(nonempty_voxels),
        "voxel_idx_mismatch_count": voxel_idx_mismatch,
        "invalid_n_atoms_count": invalid_n_atoms,
        "invalid_atom_ref_count": invalid_atom_refs,
        "nonfinite_weight_count": nonfinite_weights,
        "mean_abs_weight": if weight_n > 0 { serde_json::Value::from(weight_abs_sum / weight_n as f64) } else { serde_json::Value::Null },
    }))
}

fn summarize_adaptive_dt(path: &Path, stream_id: Option<u32>) -> Result<serde_json::Value> {
    let bytes =
        std::fs::read(path).with_context(|| format!("read adaptive dt {}", path.display()))?;
    anyhow::ensure!(
        bytes.len() % 32 == 0,
        "{} adaptive_dt length {} is not 32-byte records",
        path.display(),
        bytes.len()
    );
    let mut dt_min = f64::INFINITY;
    let mut dt_max = f64::NEG_INFINITY;
    let mut dt_sum = 0.0f64;
    let mut reason_counts = serde_json::Map::new();
    let mut first_chunk: Option<u64> = None;
    let mut last_chunk: Option<u64> = None;
    let mut first_step: Option<u64> = None;
    let mut last_step: Option<u64> = None;
    for rec in bytes.chunks_exact(32) {
        let chunk_idx = u64::from_le_bytes(rec[0..8].try_into()?);
        let step = u64::from_le_bytes(rec[8..16].try_into()?);
        let dt_ps = f64::from_le_bytes(rec[16..24].try_into()?);
        let reason = u32::from_le_bytes(rec[24..28].try_into()?);
        first_chunk.get_or_insert(chunk_idx);
        first_step.get_or_insert(step);
        last_chunk = Some(chunk_idx);
        last_step = Some(step);
        finite_min_max_update(dt_ps, &mut dt_min, &mut dt_max);
        if dt_ps.is_finite() {
            dt_sum += dt_ps;
        }
        let key = reason.to_string();
        let next = reason_counts
            .get(&key)
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
            + 1;
        reason_counts.insert(key, serde_json::Value::from(next));
    }
    let record_count = bytes.len() / 32;
    Ok(serde_json::json!({
        "kind": "adaptive_dt",
        "status": "materialized_parsed",
        "stream_id": stream_id,
        "layout": "chunk_idx:u64, step:u64, dt_ps:f64, reason_code:u32, pad:u32",
        "record_count": record_count,
        "first_chunk_idx": first_chunk,
        "last_chunk_idx": last_chunk,
        "first_step": first_step,
        "last_step": last_step,
        "dt_ps_mean": if record_count > 0 { serde_json::Value::from(dt_sum / record_count as f64) } else { serde_json::Value::Null },
        "dt_ps_min": if record_count > 0 { serde_json::Value::from(dt_min) } else { serde_json::Value::Null },
        "dt_ps_max": if record_count > 0 { serde_json::Value::from(dt_max) } else { serde_json::Value::Null },
        "reason_code_counts": reason_counts,
    }))
}

fn summarize_jsonl_numeric_events(
    path: &Path,
    kind: &str,
    stream_id: Option<u32>,
) -> Result<serde_json::Value> {
    let f = File::open(path).with_context(|| format!("open jsonl {}", path.display()))?;
    let r = BufReader::with_capacity(1 << 20, f);
    let mut n_records = 0u64;
    let mut trigger_counts = serde_json::Map::new();
    let mut condition_counts = serde_json::Map::new();
    let mut phase_transition_counts = serde_json::Map::new();
    let mut bridge_observation_events = 0u64;
    let mut bocpd_probability_max: Option<f64> = None;
    let mut recent_p_max: Option<f64> = None;
    let mut spike_delta_sum = 0u64;
    let mut first_step: Option<u64> = None;
    let mut last_step: Option<u64> = None;
    for line in r.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(&line)
            .with_context(|| format!("parse {} jsonl line {}", kind, n_records + 1))?;
        n_records += 1;
        let step = v
            .get("step")
            .or_else(|| v.get("frame_idx"))
            .and_then(|x| x.as_u64());
        if let Some(step) = step {
            first_step.get_or_insert(step);
            last_step = Some(step);
        }
        if let Some(trigger) = v.get("trigger_kind").and_then(|x| x.as_str()) {
            let next = trigger_counts
                .get(trigger)
                .and_then(|x| x.as_u64())
                .unwrap_or(0)
                + 1;
            trigger_counts.insert(trigger.to_string(), serde_json::Value::from(next));
        }
        if let Some(condition) = v.get("condition_id").and_then(|x| x.as_str()) {
            let next = condition_counts
                .get(condition)
                .and_then(|x| x.as_u64())
                .unwrap_or(0)
                + 1;
            condition_counts.insert(condition.to_string(), serde_json::Value::from(next));
        }
        if let (Some(from), Some(to)) = (
            v.get("phase_from").and_then(|x| x.as_str()),
            v.get("phase_to").and_then(|x| x.as_str()),
        ) {
            let key = format!("{from}->{to}");
            let next = phase_transition_counts
                .get(&key)
                .and_then(|x| x.as_u64())
                .unwrap_or(0)
                + 1;
            phase_transition_counts.insert(key, serde_json::Value::from(next));
        }
        if v.get("bridge_groups_observed")
            .and_then(|x| x.as_array())
            .map(|a| !a.is_empty())
            .unwrap_or(false)
        {
            bridge_observation_events += 1;
        }
        if let Some(p) = v.get("bocpd_probability").and_then(|x| x.as_f64()) {
            if p.is_finite() {
                bocpd_probability_max = Some(bocpd_probability_max.unwrap_or(p).max(p));
            }
        }
        if let Some(p) = v.get("recent_p").and_then(|x| x.as_f64()) {
            if p.is_finite() {
                recent_p_max = Some(recent_p_max.unwrap_or(p).max(p));
            }
        }
        if let Some(delta) = v.get("spike_delta").and_then(|x| x.as_u64()) {
            spike_delta_sum = spike_delta_sum.saturating_add(delta);
        }
    }
    Ok(serde_json::json!({
        "kind": kind,
        "status": "materialized_parsed",
        "stream_id": stream_id,
        "record_count": n_records,
        "first_step": first_step,
        "last_step": last_step,
        "trigger_counts": trigger_counts,
        "condition_counts": condition_counts,
        "phase_transition_counts": phase_transition_counts,
        "bridge_observation_events": bridge_observation_events,
        "bocpd_probability_max": bocpd_probability_max,
        "recent_p_max": recent_p_max,
        "spike_delta_sum": spike_delta_sum,
    }))
}

fn summarize_json_document(
    path: &Path,
    kind: &str,
    stream_id: Option<u32>,
) -> Result<serde_json::Value> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("read json {}", path.display()))?;
    let value: serde_json::Value =
        serde_json::from_str(&text).with_context(|| format!("parse json {}", path.display()))?;
    let object_len = value.as_object().map(|o| o.len()).unwrap_or(0);
    Ok(serde_json::json!({
        "kind": kind,
        "status": "materialized_parsed",
        "stream_id": stream_id,
        "json_object_keys": object_len,
        "size_bytes": text.len(),
        "schema_kind": value.get("schema_kind").and_then(|v| v.as_str()),
    }))
}

fn summarize_first_class_surface(
    art: &ManifestArtifact,
    hydration_dir: &Path,
) -> serde_json::Value {
    let manifest_path = PathBuf::from(&art.path);
    let stream_id = stream_id_for_artifact(art, &manifest_path);
    match hydrate_artifact_from_packet(&manifest_path, &art.kind, hydration_dir) {
        Ok(resolved) => {
            let source = surface_source_status(&manifest_path, &art.kind, &resolved);
            let parsed = match art.kind.as_str() {
                "signal_grid" => summarize_signal_grid(&resolved, stream_id),
                "kcc_v2full" => read_kcc_v2full(&resolved).map(|k| {
                    let active_causal_sum: u64 = k.active_causal.iter().map(|&v| v as u64).sum();
                    let active_causal_nonzero = k.active_causal.iter().filter(|&&v| v > 0).count();
                    serde_json::json!({
                        "kind": "kcc_v2full",
                        "status": "materialized_parsed",
                        "stream_id": stream_id.or(Some(k.stream_id)),
                        "n_residues": k.n_residues,
                        "active_causal_sum": active_causal_sum,
                        "active_causal_nonzero_residues": active_causal_nonzero,
                        "schema": "PRKCC001 full typed field table; all fields retained independently of region selection",
                    })
                }),
                "warp_matrix" => summarize_warp_matrix(&resolved, stream_id),
                "forces_final" | "asc_vectors" | "aromatic_centroids_final" => {
                    summarize_vector_tensor(&resolved, &art.kind, stream_id)
                }
                "adaptive_dt" => summarize_adaptive_dt(&resolved, stream_id),
                "bocpd_log" | "spike_transition_events" => {
                    summarize_jsonl_numeric_events(&resolved, &art.kind, stream_id)
                }
                "protocol_state" | "noise_floor" => {
                    summarize_json_document(&resolved, &art.kind, stream_id)
                }
                _ => Ok(serde_json::json!({
                    "kind": art.kind,
                    "status": "materialized_packet_hydrated",
                    "stream_id": stream_id,
                })),
            };
            match parsed {
                Ok(mut value) => {
                    if let Some(obj) = value.as_object_mut() {
                        obj.insert("source".to_string(), source);
                    }
                    value
                }
                Err(e) => serde_json::json!({
                    "kind": art.kind,
                    "status": "blocked_materialization_parser_error",
                    "stream_id": stream_id,
                    "source": source,
                    "failure": e.to_string(),
                }),
            }
        }
        Err(e) => serde_json::json!({
            "kind": art.kind,
            "status": "blocked_materialization_hydration_error",
            "stream_id": stream_id,
            "manifest_path": art.path,
            "failure": e.to_string(),
        }),
    }
}

/// Observed native spike positions, independent of merged-centroid spread.
/// Counts retain existing peak attribution multiplicity; bounds do not invent
/// a padding radius, cavity surface, or unobserved geometry.
#[derive(Debug, Default, Clone, Serialize)]
struct SpikeSupportGeometry {
    aabb_min: Option<[f32; 3]>,
    aabb_max: Option<[f32; 3]>,
    attributed_record_count: u64,
    nonfinite_position_count: u64,
}
impl SpikeSupportGeometry {
    fn observe(&mut self, position: [f32; 3]) {
        self.attributed_record_count += 1;
        if !position.iter().all(|v| v.is_finite()) {
            self.nonfinite_position_count += 1;
            return;
        }
        match (self.aabb_min.as_mut(), self.aabb_max.as_mut()) {
            (Some(low), Some(high)) => {
                for axis in 0..3 {
                    low[axis] = low[axis].min(position[axis]);
                    high[axis] = high[axis].max(position[axis]);
                }
            }
            _ => {
                self.aabb_min = Some(position);
                self.aabb_max = Some(position);
            }
        }
    }
    fn merge(&mut self, other: &Self) {
        self.attributed_record_count += other.attributed_record_count;
        self.nonfinite_position_count += other.nonfinite_position_count;
        if let (Some(other_low), Some(other_high)) = (other.aabb_min, other.aabb_max) {
            match (self.aabb_min.as_mut(), self.aabb_max.as_mut()) {
                (Some(low), Some(high)) => {
                    for axis in 0..3 {
                        low[axis] = low[axis].min(other_low[axis]);
                        high[axis] = high[axis].max(other_high[axis]);
                    }
                }
                _ => {
                    self.aabb_min = Some(other_low);
                    self.aabb_max = Some(other_high);
                }
            }
        }
    }
    fn surface(&self) -> serde_json::Value {
        serde_json::json!({
            "aabb_min": self.aabb_min, "aabb_max": self.aabb_max,
            "attributed_record_count": self.attributed_record_count,
            "nonfinite_position_count": self.nonfinite_position_count,
            "population_semantics": "actual_attributed_spike_positions; existing_overlapping_raw_peak_multiplicity_preserved; union_over_region_streams",
            "geometry_semantics": "observed_spike_support_AABB; not_merged_centroid_spread_or_cavity_surface",
            "coordinate_unit": "angstrom",
        })
    }
}

// ─── Per-candidate evidence accumulator ─────────────────────────────────────

#[derive(Debug, Default, Clone)]
struct CandidateEvidence {
    seed_peak_id: u32,
    seed_voxel: [i32; 3],
    seed_centroid_a: [f32; 3],
    /// Total spikes attributed to this candidate's region.
    n_spikes: u64,
    intensity_sum: f64,
    /// Weighted position sum for whole_site centroid.
    pos_sum_x: f64,
    pos_sum_y: f64,
    pos_sum_z: f64,
    /// Per-stream spike counts.
    per_stream_spikes: std::collections::HashMap<u32, u64>,
    /// Set of unique voxels touched per stream (for per_stream_voxel_counts).
    per_stream_voxels: std::collections::HashMap<u32, std::collections::HashSet<(i32, i32, i32)>>,
    /// Timestep range.
    timestep_min: i32,
    timestep_max: i32,
    /// Recurrence: number of unique 1000-step chunks observed.
    chunk_set: std::collections::HashSet<i32>,
    /// Phase histogram (10 bins of phase_bits 0-1023).
    phase_hist: [u32; 10],
    /// Rayleigh r-stat accumulators.
    phase_cos_sum: f64,
    phase_sin_sum: f64,
    /// Prescribed warm-hold support; independent of phase_bits.
    hot_pos_sum: [f64; 3],
    n_hot: u64,
    /// Prescribed initial cold-hold support.
    cold_pos_sum: [f64; 3],
    n_cold: u64,
    thermal: ThermalEvidence,
    /// Source / aromatic counts.
    n_uv: u64,
    n_lif: u64,
    n_other_source: u64,
    n_arom: [u64; 5], // [TRP, TYR, PHE, SS, none]
    /// Residue support: residue_id → count.
    residue_support: std::collections::HashMap<i32, u32>,
    spike_support_geometry: SpikeSupportGeometry,
    per_stream_residue_support:
        std::collections::BTreeMap<u32, std::collections::BTreeMap<i32, u64>>,
    /// Refined sub-chunk persistence — bin_id (= timestep / persistence_bin_steps) → spikes.
    bin_counts: std::collections::HashMap<i32, u64>,
}

impl CandidateEvidence {
    fn new(seed_peak_id: u32, seed_voxel: [i32; 3], seed_centroid_a: [f32; 3]) -> Self {
        Self {
            seed_peak_id,
            seed_voxel,
            seed_centroid_a,
            timestep_min: i32::MAX,
            timestep_max: i32::MIN,
            ..Default::default()
        }
    }
}

// ─── Materialized site (output JSON shape) ──────────────────────────────────

#[derive(Debug, Serialize)]
struct CentroidManifold {
    whole_site: Option<[f32; 3]>,
    lining_mass: Option<[f32; 3]>,
    causal_driver: Option<[f32; 3]>,
    hot_phase: Option<[f32; 3]>,
    cold_phase: Option<[f32; 3]>,
    return_phase: Option<[f32; 3]>,
    ligand_adjacent: Option<[f32; 3]>,
}

#[derive(Debug, Serialize)]
struct StreamSupport {
    n_streams_supporting: u32,
    supporting_streams: Vec<u32>,
    per_stream_spike_counts: serde_json::Map<String, serde_json::Value>,
    per_stream_voxel_counts: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct TemporalSupport {
    first_observed_step: Option<i32>,
    last_observed_step: Option<i32>,
    /// "Recurrence" = unique 1000-step chunks observed.
    recurrence_count: u32,
    chunk_support: Vec<i32>,
    n_phase_bins_nonzero: u32,
}

#[derive(Debug, Serialize)]
struct PhaseSupport {
    phase_bits_histogram: [u32; 10],
    rayleigh_r_stat: f64,
    /// Mean phase angle in radians; null if rayleigh_r_stat ≈ 0.
    mean_phase_radians: Option<f64>,
    n_hot: u64,
    n_cold: u64,
    thermal: ThermalEvidence,
}

#[derive(Debug, Serialize)]
struct So3Support {
    /// Protocol-phase Rayleigh statistic; does not measure SO(3) rotation.
    phase_coherence_rayleigh_r: f64,
    /// Per-source-class breakdown: separate r for UV vs LIF if useful.
    /// In this commit, single value across all spikes.
    orientation_coherence_score: Option<f64>,
    /// Parser-required fields for warp_matrix / forces / asc_vectors. These
    /// are blocked from rank consumption until canonical parsers exist.
    local_frame_stability: Option<f64>,
    rotation_dispersion: Option<f64>,
    warp_matrix_support: &'static str,
    asc_vector_support: &'static str,
    force_direction_support: &'static str,
    evidence_files: Vec<String>,
}

#[derive(Debug, Serialize)]
struct FieldCompleteness {
    spikes: &'static str,
    phase_bits: &'static str,
    warp_matrix: &'static str,
    asc_vectors: &'static str,
    forces: &'static str,
    signal_grid: &'static str,
    kcc: &'static str,
}

#[derive(Debug, Serialize)]
struct SpatiotemporalSo3Evidence {
    status: &'static str, // "available" | "partial" | "missing"
    stream_support: StreamSupport,
    temporal_support: TemporalSupport,
    phase_support: PhaseSupport,
    so3_support: So3Support,
    centroid_manifold: CentroidManifold,
    field_completeness: FieldCompleteness,
}

#[derive(Debug, Serialize)]
struct Provenance {
    source_manifest: String,
    run_id: String,
    target: String,
    seed_voxel_index: [i32; 3],
    seed_density_peak_rank: u32,
    spikes_files_consumed: Vec<String>,
    kcc_files_consumed: Vec<String>,
    topology_input: String,
}

#[derive(Debug, Serialize)]
struct MaterializedSite {
    site_id: String,
    rank: u32,
    materialization_status: &'static str, // see MaterializationLevel
    materialization_level: &'static str,
    centroid_xyz: [f32; 3],
    centroid_manifold: CentroidManifold,
    representative_residues: Vec<i32>,
    lining_residues: Vec<i32>,
    driver_residues: Vec<i32>,
    n_spikes: u64,
    intensity_sum: f64,
    spike_support: serde_json::Value,
    stream_support: StreamSupport,
    phase_support: PhaseSupport,
    spatiotemporal_so3_evidence: SpatiotemporalSo3Evidence,
    field_completeness: FieldCompleteness,
    provenance: Provenance,
    limitations: Vec<&'static str>,
    accuracy: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct NonMaterializedCandidate {
    seed_peak_id: u32,
    seed_voxel: [i32; 3],
    seed_centroid_a: [f32; 3],
    n_spikes: u64,
    n_streams_supporting: u32,
    materialization_level: &'static str,
    blocking_reasons: Vec<&'static str>,
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn vec3_div(p: [f64; 3], n: f64) -> [f32; 3] {
    [(p[0] / n) as f32, (p[1] / n) as f32, (p[2] / n) as f32]
}

fn dist3(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Gini coefficient for a vector of non-negative counts (low gini = balanced).
fn gini_coefficient(counts: &[u64]) -> f64 {
    if counts.is_empty() {
        return 0.0;
    }
    let n = counts.len() as f64;
    let total: f64 = counts.iter().map(|&c| c as f64).sum();
    if total <= 0.0 {
        return 0.0;
    }
    let mut sorted: Vec<f64> = counts.iter().map(|&c| c as f64).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut cumsum = 0.0;
    let mut weighted_sum = 0.0;
    for (i, c) in sorted.iter().enumerate() {
        cumsum += *c;
        weighted_sum += cumsum;
    }
    let gini = (n + 1.0 - 2.0 * weighted_sum / total) / n;
    gini.clamp(0.0, 1.0)
}

/// Jaccard overlap between two integer sets.
fn jaccard_overlap(a: &[i32], b: &[i32]) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let sa: std::collections::HashSet<i32> = a.iter().copied().collect();
    let sb: std::collections::HashSet<i32> = b.iter().copied().collect();
    let inter = sa.intersection(&sb).count();
    let union = sa.union(&sb).count();
    if union == 0 {
        return 0.0;
    }
    inter as f32 / union as f32
}

fn residue_shell_atom_count(topology: Option<&TopologyMin>, lining_residues: &[i32]) -> u32 {
    let Some(topology) = topology else {
        return 0;
    };
    let mut atoms = 0_u32;
    for residue_id in lining_residues {
        if *residue_id < 0 {
            continue;
        }
        if let Some(indices) = topology
            .residue_to_atom_indices
            .get(&residue_id.to_string())
        {
            atoms = atoms.saturating_add(indices.len() as u32);
        }
    }
    atoms
}

fn residue_shell_heavy_atom_positions(
    topology: Option<&TopologyMin>,
    lining_residues: &[i32],
) -> Vec<[f32; 3]> {
    let Some(topology) = topology else {
        return Vec::new();
    };
    let mut seen = std::collections::BTreeSet::new();
    let mut out = Vec::new();
    for residue_id in lining_residues {
        if *residue_id < 0 {
            continue;
        }
        let Some(indices) = topology
            .residue_to_atom_indices
            .get(&residue_id.to_string())
        else {
            continue;
        };
        for atom_idx in indices {
            if *atom_idx >= topology.n_atoms || !seen.insert(*atom_idx) {
                continue;
            }
            if topology
                .elements
                .get(*atom_idx)
                .map(|e| e.eq_ignore_ascii_case("H"))
                .unwrap_or(false)
            {
                continue;
            }
            let base = *atom_idx * 3;
            if base + 2 >= topology.positions.len() {
                continue;
            }
            out.push([
                topology.positions[base],
                topology.positions[base + 1],
                topology.positions[base + 2],
            ]);
        }
    }
    out
}

fn coordinate_enclosure_fraction(shell_atoms: &[[f32; 3]], centroid: [f32; 3]) -> Option<f64> {
    if shell_atoms.is_empty() {
        return None;
    }
    let mut covered = 0_u32;
    let mut total = 0_u32;
    for dx in [-1.0_f64, 0.0, 1.0] {
        for dy in [-1.0_f64, 0.0, 1.0] {
            for dz in [-1.0_f64, 0.0, 1.0] {
                if dx == 0.0 && dy == 0.0 && dz == 0.0 {
                    continue;
                }
                let norm = (dx * dx + dy * dy + dz * dz).sqrt();
                let dir = [dx / norm, dy / norm, dz / norm];
                total += 1;
                let mut direction_covered = false;
                for atom in shell_atoms {
                    let v = [
                        atom[0] as f64 - centroid[0] as f64,
                        atom[1] as f64 - centroid[1] as f64,
                        atom[2] as f64 - centroid[2] as f64,
                    ];
                    let d = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                    if !(1.0..=12.0).contains(&d) {
                        continue;
                    }
                    let dot = (v[0] / d) * dir[0] + (v[1] / d) * dir[1] + (v[2] / d) * dir[2];
                    if dot >= 0.76 {
                        direction_covered = true;
                        break;
                    }
                }
                if direction_covered {
                    covered += 1;
                }
            }
        }
    }
    if total == 0 {
        None
    } else {
        Some(covered as f64 / total as f64)
    }
}

fn coordinate_empty_probe_volume_a3(
    shell_atoms: &[[f32; 3]],
    centroid: [f32; 3],
) -> Option<(f64, f64, u32, u32)> {
    if shell_atoms.is_empty() {
        return None;
    }
    let radius = 8.0_f64;
    let spacing = 1.0_f64;
    let cutoff2 = (1.7_f64 + 1.4_f64).powi(2);
    let mut empty = 0_u32;
    let mut total = 0_u32;
    for ix in -8..=8 {
        for iy in -8..=8 {
            for iz in -8..=8 {
                let offset = [
                    ix as f64 * spacing,
                    iy as f64 * spacing,
                    iz as f64 * spacing,
                ];
                let r2 = offset[0] * offset[0] + offset[1] * offset[1] + offset[2] * offset[2];
                if r2 > radius * radius {
                    continue;
                }
                total += 1;
                let p = [
                    centroid[0] as f64 + offset[0],
                    centroid[1] as f64 + offset[1],
                    centroid[2] as f64 + offset[2],
                ];
                let mut occupied = false;
                for atom in shell_atoms {
                    let dx = p[0] - atom[0] as f64;
                    let dy = p[1] - atom[1] as f64;
                    let dz = p[2] - atom[2] as f64;
                    if dx * dx + dy * dy + dz * dz <= cutoff2 {
                        occupied = true;
                        break;
                    }
                }
                if !occupied {
                    empty += 1;
                }
            }
        }
    }
    if total == 0 {
        None
    } else {
        let volume = empty as f64 * spacing.powi(3);
        Some((volume, empty as f64 / total as f64, empty, total))
    }
}

fn mean_inter_bin_gap(bin_counts: &std::collections::HashMap<i32, u64>) -> Option<f64> {
    if bin_counts.len() < 2 {
        return None;
    }
    let mut bins: Vec<i32> = bin_counts.keys().copied().collect();
    bins.sort_unstable();
    let mut gap_sum = 0_i64;
    let mut gap_n = 0_u32;
    for pair in bins.windows(2) {
        gap_sum += (pair[1] - pair[0]) as i64;
        gap_n += 1;
    }
    if gap_n == 0 {
        None
    } else {
        Some(gap_sum as f64 / gap_n as f64)
    }
}

fn seal_json_artifact(
    path: &Path,
    artifact_id: &str,
    artifact_family: QnanlArtifactFamily,
) -> Result<serde_json::Value> {
    let packet_path = PathBuf::from(format!("{}.prismqnanl", path.display()));
    let receipt_path = PathBuf::from(format!("{}.prismqnanl.receipt.json", path.display()));
    let receipt = qnanl_artifact::encode_file(
        path,
        &packet_path,
        &receipt_path,
        QnanlEncodeOptions::runtime_subprocess_orchestrated_by_prismos(
            artifact_id.to_string(),
            artifact_family,
            None,
            "prism_materialize_sites::seal_pathb_runtime_artifact",
        ),
    )
    .with_context(|| format!("QNANL-seal {}", path.display()))?;
    Ok(serde_json::json!({
        "status": receipt.status,
        "artifact_id": receipt.artifact_id,
        "artifact_family": receipt.artifact_family.to_string(),
        "packet_path": receipt.packet_path,
        "receipt_path": receipt.receipt_path,
        "raw_hash_blake3": receipt.raw_hash_blake3,
        "packet_hash_blake3": receipt.packet_hash_blake3,
        "roundtrip_verified": receipt.roundtrip_verified,
        "qnanl_seal_status": receipt.qnanl_seal_status,
    }))
}

// ─── Stage-A raw peak (per top-K voxel, with full evidence accumulator) ────

#[derive(Debug, Clone)]
struct RawPeak {
    raw_peak_id: u32,
    voxel: [i32; 3],
    seed_centroid_a: [f32; 3],
    /// Detailed evidence accumulator from the spike re-pass.
    evidence: CandidateEvidence,
    /// Top-12 lining residues (cached for consolidation Jaccard checks).
    top12_lining_residues: Vec<i32>,
}

// ─── Stage-B consolidated candidate region ─────────────────────────────────

#[derive(Debug)]
#[allow(dead_code)]
struct CandidateRegion {
    region_id: u32,
    representative_raw_peak_id: u32,
    merged_raw_peak_ids: Vec<u32>,
    aggregate_centroid_a: [f32; 3],
    region_radius_a: f32,
    /// Aggregate evidence (sums of merged raw peaks).
    n_spikes: u64,
    intensity_sum: f64,
    pos_sum_x: f64,
    pos_sum_y: f64,
    pos_sum_z: f64,
    per_stream_spikes: std::collections::HashMap<u32, u64>,
    per_stream_voxels: std::collections::HashMap<u32, std::collections::HashSet<(i32, i32, i32)>>,
    timestep_min: i32,
    timestep_max: i32,
    chunk_set: std::collections::HashSet<i32>,
    /// Refined sub-chunk persistence bin set (using args.persistence_bin_steps).
    persistence_bin_set: std::collections::HashSet<i32>,
    /// Per-bin spike counts (for burstiness scoring).
    bin_counts: std::collections::HashMap<i32, u64>,
    phase_hist: [u32; 10],
    phase_cos_sum: f64,
    phase_sin_sum: f64,
    hot_pos_sum: [f64; 3],
    n_hot: u64,
    cold_pos_sum: [f64; 3],
    n_cold: u64,
    thermal: ThermalEvidence,
    n_uv: u64,
    n_lif: u64,
    n_other_source: u64,
    n_arom: [u64; 5],
    residue_support: std::collections::HashMap<i32, u32>,
    spike_support_geometry: SpikeSupportGeometry,
    per_stream_residue_support:
        std::collections::BTreeMap<u32, std::collections::BTreeMap<i32, u64>>,
    merge_reasons: Vec<String>,
    support_accounting: serde_json::Value,
    /// First and last steps observed (for temporal status). When refined
    /// timesteps are not available (e.g., headers_only), this is left
    /// at its sentinel min/max.
    centroid_spread_a: f32,
    residue_shell_overlap_with_rep: f32,
}

// Region attribution uses the existing native spike pass. No cluster/rank IDs
// are reused as residue IDs, and no residue support is pooled across streams.
/// Reuse the exact source support predicates; each call represents one source
/// record, never a peak vote. Invalid/missing protocol phases stay unclassified.
fn observe_unique_record(
    ev: &mut CandidateEvidence,
    rec: &[u8],
    stream: u32,
    phase: usize,
    inv_voxel: f32,
    persistence_bin_steps: i32,
) {
    let position = [
        read_f32_at(rec, spike_offsets::POSITION_X),
        read_f32_at(rec, spike_offsets::POSITION_Y),
        read_f32_at(rec, spike_offsets::POSITION_Z),
    ];
    let intensity = read_f32_at(rec, spike_offsets::INTENSITY);
    let step = read_i32_at(rec, 0);
    let phase_bits = if rec.len() >= 96 {
        u32::from_le_bytes(rec[92..96].try_into().unwrap())
    } else {
        0
    };
    let angle = phase_bits as f64 / 1024.0 * 2.0 * std::f64::consts::PI;
    ev.n_spikes += 1;
    ev.intensity_sum += intensity as f64;
    ev.pos_sum_x += position[0] as f64 * intensity as f64;
    ev.pos_sum_y += position[1] as f64 * intensity as f64;
    ev.pos_sum_z += position[2] as f64 * intensity as f64;
    ev.spike_support_geometry.observe(position);
    *ev.per_stream_spikes.entry(stream).or_default() += 1;
    ev.per_stream_voxels.entry(stream).or_default().insert((
        (position[0] * inv_voxel).floor() as i32,
        (position[1] * inv_voxel).floor() as i32,
        (position[2] * inv_voxel).floor() as i32,
    ));
    ev.timestep_min = ev.timestep_min.min(step);
    ev.timestep_max = ev.timestep_max.max(step);
    ev.chunk_set.insert(step / 1000);
    *ev.bin_counts
        .entry(step / persistence_bin_steps.max(1))
        .or_default() += 1;
    ev.phase_hist[((phase_bits as usize) * 10 / 1024).min(9)] += 1;
    ev.phase_cos_sum += angle.cos();
    ev.phase_sin_sum += angle.sin();
    ev.thermal.observe(phase, position);
    if phase == 2 {
        ev.n_hot += 1;
        for a in 0..3 {
            ev.hot_pos_sum[a] += position[a] as f64;
        }
    }
    if phase == 0 {
        ev.n_cold += 1;
        for a in 0..3 {
            ev.cold_pos_sum[a] += position[a] as f64;
        }
    }
    match read_i32_at(rec, spike_offsets::SPIKE_SOURCE) {
        1 => ev.n_uv += 1,
        2 => ev.n_lif += 1,
        _ => ev.n_other_source += 1,
    }
    let aromatic = read_i32_at(rec, spike_offsets::AROMATIC_TYPE);
    ev.n_arom[if (0..4).contains(&aromatic) {
        aromatic as usize
    } else {
        4
    }] += 1;
    // Preserve original slot semantics; event uniqueness does not silently alter
    // the producer's nearby-residue vote representation.
    for j in 0..read_i32_at(rec, 56).clamp(0, 8) as usize {
        let rid = read_i32_at(rec, 24 + j * 4);
        *ev.residue_support.entry(rid).or_default() += 1;
        *ev.per_stream_residue_support
            .entry(stream)
            .or_default()
            .entry(rid)
            .or_default() += 1;
    }
}

#[derive(Default)]
struct PhaseRecordSupport {
    count: u64,
    weighted_votes: u64,
    first_ordinal: Option<u64>,
    last_ordinal: Option<u64>,
    step_min: Option<i32>,
    step_max: Option<i32>,
    position_sum: [f64; 3],
    residue_support: std::collections::BTreeMap<i32, u64>,
    multiplicities: std::collections::BTreeMap<u32, u64>,
}
impl PhaseRecordSupport {
    fn observe(&mut self, rec: &[u8], ordinal: u64, multiplicity: u32) {
        self.count += 1;
        self.weighted_votes += u64::from(multiplicity);
        *self.multiplicities.entry(multiplicity).or_default() += 1;
        self.first_ordinal.get_or_insert(ordinal);
        self.last_ordinal = Some(ordinal);
        let step = read_i32_at(rec, 0);
        self.step_min = Some(self.step_min.map_or(step, |s| s.min(step)));
        self.step_max = Some(self.step_max.map_or(step, |s| s.max(step)));
        for (a, off) in [
            spike_offsets::POSITION_X,
            spike_offsets::POSITION_Y,
            spike_offsets::POSITION_Z,
        ]
        .into_iter()
        .enumerate()
        {
            self.position_sum[a] += read_f32_at(rec, off) as f64;
        }
        for j in 0..read_i32_at(rec, 56).clamp(0, 8) as usize {
            *self
                .residue_support
                .entry(read_i32_at(rec, 24 + j * 4))
                .or_default() += 1;
        }
    }
}

fn region_attribution_lookup(
    voxel_candidates: &std::collections::HashMap<(i32, i32, i32), Vec<u32>>,
    peak_region: &std::collections::HashMap<u32, usize>,
) -> std::collections::HashMap<(i32, i32, i32), Vec<(usize, u32)>> {
    voxel_candidates
        .iter()
        .map(|(&voxel, candidates)| {
            let mut counts = std::collections::BTreeMap::new();
            for peak in candidates {
                *counts.entry(peak_region[peak]).or_insert(0u32) += 1;
            }
            (voxel, counts.into_iter().collect())
        })
        .collect()
}

#[derive(Debug)]
struct RecordedTransitionWindow {
    stream_id: u32,
    start_step: i64,
    end_step_exclusive: i64,
    anchors: Vec<serde_json::Value>,
    identity: serde_json::Value,
}

fn load_recorded_transition_windows(
    surfaces: &[serde_json::Value],
    schedules: &std::collections::HashMap<u32, ThermalSchedule>,
    run_id: &str,
) -> Result<Vec<RecordedTransitionWindow>> {
    use sha2::Digest;
    let mut result = Vec::new();
    let mut seen_streams = std::collections::BTreeSet::new();
    for surface in surfaces
        .iter()
        .filter(|s| s["kind"] == "spike_transition_events")
    {
        anyhow::ensure!(
            surface["status"] == "materialized_parsed",
            "recorded transition surface was not admitted"
        );
        let stream = u32::try_from(
            surface["stream_id"]
                .as_u64()
                .context("transition surface lacks stream")?,
        )?;
        anyhow::ensure!(
            seen_streams.insert(stream),
            "duplicate transition artifact for stream {}",
            stream
        );
        let path = surface["source"]["resolved_path"]
            .as_str()
            .context("transition surface lacks resolved source")?;
        let bytes = std::fs::read(path)?;
        let source_sha256 = format!("{:x}", sha2::Sha256::digest(&bytes));
        let schedule = schedules
            .get(&stream)
            .context("recorded event windows require exact stream protocol bounds")?;
        let mut groups = std::collections::BTreeMap::<i64, Vec<serde_json::Value>>::new();
        let mut hashes = std::collections::BTreeSet::new();
        for (line_index, line) in std::str::from_utf8(&bytes)?.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let event: serde_json::Value = serde_json::from_str(line)?;
            anyhow::ensure!(
                event["schema_kind"] == "SPIKE_TRANSITION_EVENT"
                    && event["schema_version"] == 1
                    && event["stream_id"].as_u64() == Some(u64::from(stream)),
                "transition event schema/stream mismatch"
            );
            let step = event["step"]
                .as_i64()
                .context("transition event lacks completed step")?;
            anyhow::ensure!(
                step >= 0 && step <= schedule.current_step.unwrap_or(schedule.total_steps),
                "transition event outside recorded protocol"
            );
            let event_hash = event["event_hash_blake3"]
                .as_str()
                .context("transition event lacks native identity")?;
            let mut body = event.clone();
            body.as_object_mut()
                .context("event must be object")?
                .remove("event_hash_blake3");
            anyhow::ensure!(
                blake3::hash(&serde_json::to_vec(&body)?).to_hex().as_str() == event_hash,
                "native transition event BLAKE3 mismatch at line {}",
                line_index + 1
            );
            anyhow::ensure!(
                hashes.insert(event_hash.to_string()),
                "duplicate transition event identity"
            );
            groups.entry(step).or_default().push(serde_json::json!({"source_line":line_index+1,
                "native_event":event,
                "initialization_context":if event["trigger_kind"]=="BOCPD_changepoint" && event["chunk_idx"].as_u64().is_some_and(|c|c<2) {
                    "first_two_observations_of_native_recent_probability_horizon; not_established_site_episode"
                } else {"not_first_two_observations"}}));
        }
        let starts: Vec<_> = groups.keys().copied().collect();
        for (i, start) in starts.iter().copied().enumerate() {
            let end = starts.get(i + 1).copied().unwrap_or(
                schedule
                    .current_step.unwrap_or(schedule.total_steps)
                    .checked_add(1)
                    .context("protocol bound overflow")?,
            );
            let anchors = groups.remove(&start).unwrap();
            let identity = serde_json::json!({"run_id":run_id,"stream_id":stream,"start_completed_step":start,
                "end_step_exclusive":end,"source_event_jsonl_sha256":source_sha256,
                "source_manifest_path":surface["source"]["manifest_path"],
                "native_anchor_event_hashes":anchors.iter().map(|a|a["native_event"]["event_hash_blake3"].clone()).collect::<Vec<_>>(),
                "protocol_schedule":schedule,"window_policy":"recorded_completed_anchor_to_next_distinct_anchor_half_open_v1",
                "final_boundary_semantics":"last interval includes recorded current_step when present, otherwise nominal total_steps; executed bound+1 is exclusive integer bound, not an extra MD step"});
            result.push(RecordedTransitionWindow {
                stream_id: stream,
                start_step: start,
                end_step_exclusive: end,
                anchors,
                identity,
            });
        }
    }
    result.sort_by_key(|w| (w.stream_id, w.start_step));
    Ok(result)
}

fn transition_window_index(
    windows: &[RecordedTransitionWindow],
    stream: u32,
    step: i32,
) -> Option<usize> {
    // Windows are sorted and disjoint within a stream. Binary search keeps the
    // saved 19M-record replay independent of the number of event anchors.
    let end = windows.partition_point(|w| (w.stream_id, w.start_step) <= (stream, i64::from(step)));
    end.checked_sub(1).filter(|&i| {
        windows[i].stream_id == stream && i64::from(step) < windows[i].end_step_exclusive
    })
}

fn transition_support_record(
    identity: serde_json::Value,
    p: Option<&PhaseRecordSupport>,
) -> Result<serde_json::Value> {
    let empty = PhaseRecordSupport::default();
    let p = p.unwrap_or(&empty);
    let mut record = serde_json::json!({"observation_sha256":prism_nhs::chemical_state_fiber::canonical_sha256(&identity)?,
        "identity":identity,"unique_record_count":p.count,"raw_peak_attribution_count":p.weighted_votes,
        "attribution_multiplicity_histogram":p.multiplicities,
        "first_source_record_ordinal":p.first_ordinal,"last_source_record_ordinal":p.last_ordinal,
        "first_observed_step":p.step_min,"last_observed_step":p.step_max,
        "spike_position_centroid_a":if p.count>0 {Some(p.position_sum.map(|v|v/p.count as f64))} else {None},
        "residue_vote_counts":p.residue_support,
        "status":if p.count>0 {"observed_support"} else {"recorded_anchor_window_without_support"},
        "scope":"global_recorded_transition_conditioned_observation; not_region_specific_changepoint_or_chemical_state",
        "episode_identity":null,"chemical_state_identity":null});
    record["support_record_sha256"] =
        serde_json::json!(prism_nhs::chemical_state_fiber::canonical_sha256(&record)?);
    Ok(record)
}

fn recount_unique_region_support(
    regions: &mut [CandidateRegion],
    voxel_candidates: &std::collections::HashMap<(i32, i32, i32), Vec<u32>>,
    artifacts: &[(u32, String, String, u64)],
    schedules: &std::collections::HashMap<u32, ThermalSchedule>,
    voxel_size: f32,
    bin_steps: i32,
    run_id: &str,
    transition_surfaces: &[serde_json::Value],
    output_dir: &Path,
) -> Result<()> {
    let peak_region: std::collections::HashMap<_, _> = regions
        .iter()
        .enumerate()
        .flat_map(|(ri, r)| r.merged_raw_peak_ids.iter().map(move |&peak| (peak, ri)))
        .collect();
    let lookup = region_attribution_lookup(voxel_candidates, &peak_region);
    let mut canonical_lookup: Vec<_> = lookup
        .iter()
        .map(|(voxel, regions)| (*voxel, regions.clone()))
        .collect();
    canonical_lookup.sort_by_key(|row| row.0);
    let lookup_sha256 = prism_nhs::chemical_state_fiber::canonical_sha256(&canonical_lookup)?;
    let mut unique: Vec<_> = regions
        .iter()
        .map(|r| {
            CandidateEvidence::new(r.representative_raw_peak_id, [0; 3], r.aggregate_centroid_a)
        })
        .collect();
    let mut phases: Vec<std::collections::BTreeMap<(usize, u32, usize), PhaseRecordSupport>> = (0
        ..regions.len())
        .map(|_| std::collections::BTreeMap::new())
        .collect();
    let windows = load_recorded_transition_windows(transition_surfaces, schedules, run_id)?;
    let mut event_support: Vec<std::collections::BTreeMap<(usize, usize), PhaseRecordSupport>> = (0
        ..regions.len())
        .map(|_| std::collections::BTreeMap::new())
        .collect();
    let mut global_event_support =
        std::collections::BTreeMap::<(usize, usize), PhaseRecordSupport>::new();
    let mut global_event_with_region = std::collections::BTreeMap::<(usize, usize), u64>::new();
    let mut uncovered_source_records = 0u64;
    let mut sources = Vec::new();
    let inv_voxel = 1.0 / voxel_size;
    for (artifact_index, (stream, manifest_path, path, expected_count)) in
        artifacts.iter().enumerate()
    {
        let mut reader = BufReader::with_capacity(1 << 20, File::open(path)?);
        let header = read_envelope_header(&mut reader)?;
        anyhow::ensure!(
            header.stream_id == *stream
                && header.run_id == run_id
                && header.record_count == *expected_count
                && header.byte_stride >= 84,
            "unique support source identity mismatch"
        );
        let stride = header.byte_stride as usize;
        let mut buffer = vec![0u8; 65536 * stride];
        let mut ordinal = 0u64;
        while ordinal < header.record_count {
            let count = (header.record_count - ordinal).min(65536) as usize;
            reader.read_exact(&mut buffer[..count * stride])?;
            for rec in buffer[..count * stride].chunks_exact(stride) {
                let voxel = (
                    (read_f32_at(rec, spike_offsets::POSITION_X) * inv_voxel).floor() as i32,
                    (read_f32_at(rec, spike_offsets::POSITION_Y) * inv_voxel).floor() as i32,
                    (read_f32_at(rec, spike_offsets::POSITION_Z) * inv_voxel).floor() as i32,
                );
                let event_window = transition_window_index(&windows, *stream, read_i32_at(rec, 0));
                if let Some(wi) = event_window {
                    global_event_support
                        .entry((artifact_index, wi))
                        .or_default()
                        .observe(rec, ordinal, 1);
                    if lookup.contains_key(&voxel) {
                        *global_event_with_region
                            .entry((artifact_index, wi))
                            .or_default() += 1;
                    }
                } else {
                    uncovered_source_records += 1;
                }
                if let Some(attributions) = lookup.get(&voxel) {
                    let phase = schedules
                        .get(stream)
                        .map(|s| s.phase(read_i32_at(rec, 0)))
                        .unwrap_or(6);
                    for &(ri, multiplicity) in attributions {
                        observe_unique_record(
                            &mut unique[ri],
                            rec,
                            *stream,
                            phase,
                            inv_voxel,
                            bin_steps,
                        );
                        phases[ri]
                            .entry((artifact_index, *stream, phase))
                            .or_default()
                            .observe(rec, ordinal, multiplicity);
                        if let Some(wi) = event_window {
                            event_support[ri]
                                .entry((artifact_index, wi))
                                .or_default()
                                .observe(rec, ordinal, multiplicity);
                        }
                    }
                }
                ordinal += 1;
            }
        }
        let checksum = read_u64_le(&mut reader)?;
        sources.push(
            serde_json::json!({"artifact_index":artifact_index,"stream_id":stream,
            "manifest_path":manifest_path,"resolved_path":path,"record_count":header.record_count,
            "payload_checksum_fnv1a64":format!("{:016x}",checksum),"record_stride_bytes":stride}),
        );
    }
    let mut global_records = Vec::new();
    for (artifact_index, source) in sources.iter().enumerate() {
        for (wi, window) in windows
            .iter()
            .enumerate()
            .filter(|(_, w)| Some(u64::from(w.stream_id)) == source["stream_id"].as_u64())
        {
            let identity = serde_json::json!({"event_window":window.identity,"spike_source":{
                "manifest_path":source["manifest_path"],"payload_checksum_fnv1a64":source["payload_checksum_fnv1a64"],
                "record_count":source["record_count"],"record_stride_bytes":source["record_stride_bytes"]}});
            let support = global_event_support.get(&(artifact_index, wi));
            let observed = support.map_or(0, |p| p.count);
            let covered = global_event_with_region
                .get(&(artifact_index, wi))
                .copied()
                .unwrap_or(0);
            global_records.push(serde_json::json!({"support_record":transition_support_record(identity,support)?,
                "records_with_any_region_support":covered,"records_without_any_region_support":observed-covered}));
        }
    }
    let event_ledger_path = output_dir.join("recorded_transition_event_support.prismos_v1.json");
    let event_ledger = serde_json::json!({"schema_kind":"prismos_recorded_transition_event_support","schema_version":1,
        "run_id":run_id,"event_artifact_status":if windows.is_empty() {"not_available"} else {"native_hash_verified"},
        "window_count":windows.len(),"native_anchor_count":windows.iter().map(|w|w.anchors.len()).sum::<usize>(),
        "source_records_without_recorded_event_window":uncovered_source_records,
        "anchors":windows.iter().flat_map(|w|w.anchors.iter()).collect::<Vec<_>>(),
        "global_window_support":global_records,
        "historical_spike_packet_pending_status_preserved":true,
        "region_rank_and_aggregation_changed_by_event_strata":false});
    std::fs::write(
        &event_ledger_path,
        serde_json::to_vec_pretty(&event_ledger)?,
    )?;
    let event_ledger_sha256 = prism_nhs::chemical_state_fiber::canonical_sha256(&event_ledger)?;
    for (((region, ev), strata), event_strata) in regions
        .iter_mut()
        .zip(unique)
        .zip(phases)
        .zip(event_support)
    {
        let legacy_count = region.n_spikes;
        let weighted: u64 = strata.values().map(|s| s.weighted_votes).sum();
        anyhow::ensure!(
            weighted == legacy_count,
            "region {} attribution multiplicity conservation failed",
            region.region_id
        );
        let mut peak_ids = region.merged_raw_peak_ids.clone();
        peak_ids.sort_unstable();
        let mut children = Vec::new();
        for ((artifact_index, stream, phase), p) in strata {
            let identity = serde_json::json!({"run_id":run_id,"source":{
                "manifest_path":sources[artifact_index]["manifest_path"],
                "payload_checksum_fnv1a64":sources[artifact_index]["payload_checksum_fnv1a64"],
                "record_count":sources[artifact_index]["record_count"],
                "record_stride_bytes":sources[artifact_index]["record_stride_bytes"]},
                "raw_peak_ids":peak_ids,"stream_id":stream,"protocol_phase":phase,
                "protocol_schedule":schedules.get(&stream),"support_policy":"unique-record-v1",
                "region_id":region.region_id,"voxel_size_a":voxel_size,"temporal_bin_steps":bin_steps,
                "attribution_lookup_sha256":lookup_sha256,
                "attribution_lookup_contract":"sorted_voxel_xyz_to_sorted_region_index_and_raw_peak_multiplicity; reconstructed_from_preserved_raw_peaks_and_consolidation"});
            let digest = prism_nhs::chemical_state_fiber::canonical_sha256(&identity)?;
            let mut child = serde_json::json!({"observation_sha256":digest,
                "observation_sha256_scope":"phase_stratum_identity_and_replay_lookup_contract; support_record_sha256_binds_all_emitted_measurements",
                "identity":identity,
                "parent_region_id":region.region_id,"protocol_phase_name":THERMAL_PHASE_NAMES[phase],
                "unique_record_count":p.count,"raw_peak_attribution_count":p.weighted_votes,
                "attribution_multiplicity_histogram":p.multiplicities,
                "first_source_record_ordinal":p.first_ordinal,"last_source_record_ordinal":p.last_ordinal,
                "source_record_range_semantics":"bounding_ordinals_not_dense_membership; exact membership is replayable from source and voxel lookup",
                "first_observed_step":p.step_min,"last_observed_step":p.step_max,
                "spike_position_centroid_a":p.position_sum.map(|v|v/p.count as f64),
                "residue_vote_counts":p.residue_support,
                "episode_identity":null,"episode_status":"protocol_phase_stratum_not_dynamical_episode",
                "chemical_state_identity":null,"chemical_state_status":"not_recorded_in_PRSPK",
                "ghost_component_identity":null,"ghost_join_status":"no_shared_event_identifier; independent_native_observation_surface_preserved"});
            let support_record_sha256 = prism_nhs::chemical_state_fiber::canonical_sha256(&child)?;
            child["support_record_sha256"] = serde_json::json!(support_record_sha256);
            children.push(child);
        }
        let mut event_children = Vec::new();
        for (artifact_index, source) in sources.iter().enumerate() {
            for (wi, window) in windows
                .iter()
                .enumerate()
                .filter(|(_, w)| Some(u64::from(w.stream_id)) == source["stream_id"].as_u64())
            {
                let identity = serde_json::json!({"run_id":run_id,"parent_region_id":region.region_id,
                    "raw_peak_ids":peak_ids,"event_window":window.identity,
                    "voxel_size_a":voxel_size,"attribution_lookup_sha256":lookup_sha256,
                    "spike_source":{"manifest_path":source["manifest_path"],
                        "payload_checksum_fnv1a64":source["payload_checksum_fnv1a64"],
                        "record_count":source["record_count"],"record_stride_bytes":source["record_stride_bytes"]}});
                event_children.push(transition_support_record(
                    identity,
                    event_strata.get(&(artifact_index, wi)),
                )?);
            }
        }
        let event_supported_unique: u64 = event_strata.values().map(|p| p.count).sum();
        anyhow::ensure!(
            event_supported_unique <= ev.n_spikes,
            "event windows duplicate within-region records"
        );
        region.support_accounting = serde_json::json!({"policy":"unique-record-v1",
            "unique_record_count":ev.n_spikes,"legacy_raw_peak_attribution_count":legacy_count,
            "duplicate_attributions_within_region":legacy_count-ev.n_spikes,
            "multiplicity_conservation_pass":weighted==legacy_count,
            "cross_region_support":"same_record_may_support_distinct_regions; not_independent_observations",
            "source_identity":"artifact_payload_checksum_stream_and_zero_based_record_ordinal",
            "parent_region_identity":"legacy_consolidation_membership_preserved_before_unique_recount",
            "spatial_parent_consolidation_scope":"distance_or_lining_overlap; separate_phase_observations_not_collapsed",
            "dynamic_episode_consolidation_status":"not_inferred_from_protocol_phase_or_spatial_parent",
            "chemical_state_consolidation_status":"unavailable_without_recorded_state_identity",
            "raw_peak_ids":peak_ids,"phase_observations":children,
            "event_conditioned_observations":event_children,
            "unique_records_without_recorded_event_window":ev.n_spikes-event_supported_unique,
            "recorded_transition_event_ledger":{"path":event_ledger_path,"canonical_json_sha256":event_ledger_sha256}});
        region.n_spikes = ev.n_spikes;
        region.intensity_sum = ev.intensity_sum;
        region.pos_sum_x = ev.pos_sum_x;
        region.pos_sum_y = ev.pos_sum_y;
        region.pos_sum_z = ev.pos_sum_z;
        if ev.intensity_sum > 0.0 {
            region.aggregate_centroid_a =
                vec3_div([ev.pos_sum_x, ev.pos_sum_y, ev.pos_sum_z], ev.intensity_sum);
        }
        region.per_stream_spikes = ev.per_stream_spikes;
        region.per_stream_voxels = ev.per_stream_voxels;
        region.timestep_min = ev.timestep_min;
        region.timestep_max = ev.timestep_max;
        region.chunk_set = ev.chunk_set;
        region.persistence_bin_set = ev.bin_counts.keys().copied().collect();
        region.bin_counts = ev.bin_counts;
        region.phase_hist = ev.phase_hist;
        region.phase_cos_sum = ev.phase_cos_sum;
        region.phase_sin_sum = ev.phase_sin_sum;
        region.hot_pos_sum = ev.hot_pos_sum;
        region.n_hot = ev.n_hot;
        region.cold_pos_sum = ev.cold_pos_sum;
        region.n_cold = ev.n_cold;
        region.thermal = ev.thermal;
        region.n_uv = ev.n_uv;
        region.n_lif = ev.n_lif;
        region.n_other_source = ev.n_other_source;
        region.n_arom = ev.n_arom;
        region.residue_support = ev.residue_support;
        region.spike_support_geometry = ev.spike_support_geometry;
        region.per_stream_residue_support = ev.per_stream_residue_support;
    }
    eprintln!(
        "[materialize] unique source record accounting verified for {} stable parent regions",
        regions.len()
    );
    Ok(())
}

fn region_native_kcc(
    region_id: u32,
    per_stream_residue_support: &std::collections::BTreeMap<
        u32,
        std::collections::BTreeMap<i32, u64>,
    >,
    streams: &[KccPerStream],
) -> Vec<NativeKccRegionStream> {
    per_stream_residue_support.iter().map(|(&stream_id, support)| {
        let matches: Vec<_> = streams.iter().filter(|k| k.stream_id == stream_id).collect();
        let mut result = NativeKccRegionStream {
            stream_id, region_id,
            source_path: String::new(), source_checksum_fnv1a64: String::new(),
            support_semantics: "spike_nearby_residue_votes_within_stream_region; not_signal_grid_primary_residue_or_coupling_weights".into(),
            sample_count_semantics: "KCC_accumulations_not_MD_steps".into(),
            lag_unit: "MD_step".into(),
            lag_estimator_origin: "unknown_legacy_schema_does_not_record_per_residue_origin".into(),
            direction_score_semantics: "net_displacement_over_path_length_not_directed_causal_influence".into(),
            candidates: Vec::new(), selected_residue_index: None,
            native_residue_confidence: None,
            status: "missing_or_ambiguous_stream_KCC".into(),
        };
        if matches.len() != 1 { return result; }
        let k = matches[0];
        result.source_path = k.resolved_source.display().to_string();
        if !k.lag_provenance.is_empty() { result.lag_estimator_origin = "per_residue_hash_bound_sidecar".into(); }
        result.source_checksum_fnv1a64 = format!("{:016x}", k.decoded.checksum);
        if !k.decoded.integrity_ok {
            result.status = "KCC_integrity_failed".into();
            return result;
        }
        let mut candidates: Vec<_> = support.iter()
            .filter(|(rid, _)| **rid >= 0 && (**rid as usize) < k.n_residues)
            .map(|(&rid, &count)| (rid, count)).collect();
        let total_support: u64 = candidates.iter().map(|(_, count)| *count).sum();
        candidates.sort_by(count_desc_identity_asc);
        candidates.truncate(3);
        for (rid, count) in candidates {
            let r = rid as usize;
            let get = |name: &str| k.decoded.f32_fields.get(name)
                .and_then(|values| values.get(r)).copied().filter(|v| v.is_finite());
            let observations = k.decoded.u32_fields["residue_count"][r];
            let active = k.decoded.u32_fields["active_causal"][r];
            let support_fraction = count as f32 / total_support.max(1) as f32;
            let confidence = match (get("motion_efficiency"), get("burst_motion"), get("local_cov")) {
                (Some(me), Some(bm), Some(lc)) if observations > 0 && active <= observations => {
                    let v = native_residue_kcc_confidence(me, active, observations, bm, lc, support_fraction);
                    v.is_finite().then_some(v)
                }
                _ => None,
            };
            result.candidates.push(NativeKccResidueObservation {
                residue_index: rid as u32, nearby_residue_support_count: count,
                lag_provenance: Some(k.lag_provenance.get(r).cloned().unwrap_or_else(|| prism_nhs::causome::KccLagProvenance::legacy_unknown(observations, active))),
                support_fraction, observation_count: observations, active_causal: active,
                temporal_corr: get("temporal_corr"), direction_score: get("direction_score"),
                motion_efficiency: get("motion_efficiency"), burst_motion: get("burst_motion"),
                phase_shift: get("phase_shift"), causal_lag: get("causal_lag"),
                lag_corr_peak: get("lag_corr_peak"), local_cov: get("local_cov"),
                net_dx: get("net_dx"), net_dy: get("net_dy"), net_dz: get("net_dz"), sum_m: get("sum_m"),
                native_f32_bits: k.decoded.f32_fields.iter().filter_map(|(name, values)|
                    values.get(r).map(|value| (name.clone(), format!("{:08x}", value.to_bits())))).collect(),
                native_residue_confidence: confidence,
            });
        }
        if let Some(best) = result.candidates.iter().filter(|c| c.native_residue_confidence.is_some())
            .max_by(|a, b| a.native_residue_confidence.unwrap().total_cmp(&b.native_residue_confidence.unwrap())
                .then_with(|| a.local_cov.unwrap().total_cmp(&b.local_cov.unwrap()))
                .then_with(|| b.residue_index.cmp(&a.residue_index))) {
            result.selected_residue_index = Some(best.residue_index);
            result.native_residue_confidence = best.native_residue_confidence;
            result.status = "available".into();
        } else { result.status = "no_candidate_with_finite_required_measurements_and_valid_sample_counts".into(); }
        result
    }).collect()
}

fn return_inclusive_geometry(manifold: &CentroidManifold) -> serde_json::Value {
    let views: Vec<_> = [
        manifold.whole_site,
        manifold.lining_mass,
        manifold.causal_driver,
        manifold.hot_phase,
        manifold.cold_phase,
        manifold.return_phase,
    ]
    .into_iter()
    .flatten()
    .collect();
    let dispersion = if views.len() >= 2 {
        let mut max_distance = 0.0f32;
        for i in 0..views.len() {
            for j in i + 1..views.len() {
                max_distance = max_distance.max(dist3(&views[i], &views[j]));
            }
        }
        Some(max_distance)
    } else {
        None
    };
    let factor = dispersion
        .map(|d| (-(d as f64) / 4.0).exp() * (views.len() as f64 / 6.0).min(1.0).max(0.2));
    serde_json::json!({
        "formula_id": "native_centroid_max_pair_distance_return_inclusive_v1",
        "formula": "exp(-max_pairwise_centroid_distance_A/4.0) * clamp(n_present_views/6.0,0.2,1.0)",
        "views": ["whole_site", "lining_mass", "causal_driver", "warm_hold", "initial_cold_hold", "cold_return"],
        "n_views_present": views.len(), "manifold_dispersion_a": dispersion,
        "consistency_factor": factor,
        "cold_to_return_distance_a": manifold.cold_phase.zip(manifold.return_phase).map(|(a,b)| dist3(&a,&b)),
        "scope": "geometric_centroid_consistency; not_thermodynamic_hysteresis_or_causal_reversibility",
        "admission_use": "none; original_five_view_manifold_preserved",
    })
}

// ─── PRISM-native score components (every factor is emitted) ───────────────

#[derive(Debug, Default, Serialize, Clone)]
struct PrismScoreComponents {
    density_score: f64,
    scale_normalized_dynamic_support_factor: f64,
    stream_balance_factor: f64,
    temporal_persistence_factor: f64,
    residue_shell_plausibility_factor: f64,
    centroid_manifold_consistency_factor: f64,
    kcc_driver_factor: f64,
    signal_preservation_factor: f64,
    pocket_morphology_rank: f64,
    thermodynamic_physiology_rank: f64,
    site_family_preservation_rank: f64,
    field_completeness_factor: f64,
    duplicate_region_factor: f64,
    tail_penalty_factor: f64,
    /// M1.2.23 §12 — Ghost/ZSTR candidate-features factor. Neutral 1.0
    /// when --ghost-features is absent. Range [0.7, 1.6] when present
    /// (mirror of kcc_driver_factor scale). Status string emitted
    /// alongside in `ghost_zstr_status`.
    ghost_zstr_factor: f64,
    /// Free-form status: "absent_neutral" | "present_non_spatial" |
    /// "present_spatial_linked".
    ghost_zstr_status: &'static str,
    /// Free-form per-site reasoning string. Empty when neutral.
    ghost_zstr_reasoning: Vec<String>,
    /// Structured source-component and spatial-link provenance for this region.
    ghost_zstr_links: Vec<GhostRegionLink>,
    /// Chem-perturbed DTSG contribution. Neutral when the state graph is absent
    /// or has no edge linked to this region's residue support.
    chem_dtsg_factor: f64,
    chem_dtsg_status: &'static str,
    chem_dtsg_links: Vec<ChemDtsgRegionLink>,
    final_prism_score: f64,
    score_explanation: Vec<String>,
}

#[derive(Debug, Default, Serialize, Clone)]
struct LexicographicSiteRankKey {
    native_residue_kcc_confidence: Option<f64>,
    return_inclusive_geometric_consistency: Option<f64>,
    rank_policy_id: &'static str,
    key_order: Vec<&'static str>,
    promoted: bool,
    materialization_level_rank: u8,
    site_family_preservation_rank: f64,
    pocket_morphology_rank: f64,
    thermodynamic_physiology_rank: f64,
    n_active_non_density_dimensions: u32,
    kcc_driver_factor: f64,
    signal_preservation_factor: f64,
    temporal_persistence_factor: f64,
    centroid_manifold_consistency_factor: f64,
    scale_normalized_dynamic_support: f64,
    final_prism_score: f64,
    n_streams_supporting: u32,
    region_id: u32,
}

fn materialization_level_rank(level: &str) -> u8 {
    match level {
        "materialized_dynamic_site_with_validation" => 4,
        "materialized_dynamic_site" => 3,
        "candidate_spatiotemporal_partial" => 2,
        "candidate_density_only" => 1,
        _ => 0,
    }
}

fn cmp_f64_desc(a: f64, b: f64) -> std::cmp::Ordering {
    b.partial_cmp(&a).unwrap_or(std::cmp::Ordering::Equal)
}

fn compare_available_desc(a: Option<f64>, b: Option<f64>) -> std::cmp::Ordering {
    match (a, b) {
        (Some(a), Some(b)) => cmp_f64_desc(a, b),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => std::cmp::Ordering::Equal,
    }
}

fn compare_lexicographic_site_rank(
    a: &LexicographicSiteRankKey,
    b: &LexicographicSiteRankKey,
) -> std::cmp::Ordering {
    b.promoted
        .cmp(&a.promoted)
        .then_with(|| {
            b.materialization_level_rank
                .cmp(&a.materialization_level_rank)
        })
        .then_with(|| {
            if a.rank_policy_id == RANK_POLICY_ID_NATIVE_KCC
                && b.rank_policy_id == RANK_POLICY_ID_NATIVE_KCC
            {
                compare_available_desc(
                    a.native_residue_kcc_confidence,
                    b.native_residue_kcc_confidence,
                )
                .then_with(|| {
                    compare_available_desc(
                        a.return_inclusive_geometric_consistency,
                        b.return_inclusive_geometric_consistency,
                    )
                })
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .then_with(|| {
            cmp_f64_desc(
                a.site_family_preservation_rank,
                b.site_family_preservation_rank,
            )
        })
        .then_with(|| cmp_f64_desc(a.pocket_morphology_rank, b.pocket_morphology_rank))
        .then_with(|| {
            cmp_f64_desc(
                a.thermodynamic_physiology_rank,
                b.thermodynamic_physiology_rank,
            )
        })
        .then_with(|| {
            b.n_active_non_density_dimensions
                .cmp(&a.n_active_non_density_dimensions)
        })
        .then_with(|| cmp_f64_desc(a.kcc_driver_factor, b.kcc_driver_factor))
        .then_with(|| cmp_f64_desc(a.signal_preservation_factor, b.signal_preservation_factor))
        .then_with(|| cmp_f64_desc(a.temporal_persistence_factor, b.temporal_persistence_factor))
        .then_with(|| {
            cmp_f64_desc(
                a.centroid_manifold_consistency_factor,
                b.centroid_manifold_consistency_factor,
            )
        })
        .then_with(|| {
            cmp_f64_desc(
                a.scale_normalized_dynamic_support,
                b.scale_normalized_dynamic_support,
            )
        })
        .then_with(|| cmp_f64_desc(a.final_prism_score, b.final_prism_score))
        .then_with(|| b.n_streams_supporting.cmp(&a.n_streams_supporting))
        .then_with(|| a.region_id.cmp(&b.region_id))
}

// ─── Residue-shell plausibility helpers ────────────────────────────────────

#[derive(Debug, Default, Serialize, Clone)]
struct ResidueShellPlausibility {
    residue_shell_size: u32,
    residue_diversity_score: f64,
    /// Mean pairwise distance between supporting residue CAs (Å); lower = more compact.
    compactness_a: Option<f64>,
    /// Heuristic enclosure proxy (function of compactness × residue count).
    enclosure_proxy: Option<f64>,
    tail_penalty: f64,
    terminal_residue_fraction: f64,
    backbone_exposure_proxy: Option<f64>,
    aromatic_count: u32,
    hydrophobic_count: u32,
    polar_count: u32,
    catalytic_like_pocket_support: f64,
    residue_shell_reasoning: Vec<String>,
}

#[derive(Debug, Default, Serialize, Clone)]
struct GeometricPocketMorphology {
    surface_id: &'static str,
    volume_a3: f64,
    legacy_region_sphere_volume_a3: Option<f64>,
    coordinate_empty_probe_volume_a3: Option<f64>,
    coordinate_empty_probe_fraction: Option<f64>,
    coordinate_enclosure_fraction: Option<f64>,
    coordinate_probe_empty_points: Option<u32>,
    coordinate_probe_total_points: Option<u32>,
    coordinate_geometry_status: Option<&'static str>,
    volume_source: &'static str,
    shell_volume_normalized_a3: f64,
    enclosure_wall_blocking_score: f64,
    cavity_support_score: f64,
    groove_support_score: f64,
    compactness_score: f64,
    sphericity_proxy: f64,
    mouth_accessibility_proxy: f64,
    hydrophobic_lining_fraction: f64,
    aromatic_lining_fraction: f64,
    polar_lining_fraction: f64,
    catalytic_like_lining_support: f64,
    residue_shell_atom_count: u32,
    residue_shell_size: u32,
    mean_lining_distance_a: Option<f64>,
    morphology_rank: f64,
    morphology_reasoning: Vec<String>,
}

#[derive(Debug, Default, Serialize, Clone)]
struct BindingSitePhysiology {
    surface_id: &'static str,
    signal_preservation_factor: f64,
    scale_normalized_dynamic_support: f64,
    prism_therm_rank_g_geometry: f64,
    prism_therm_rank_t_temporal: f64,
    prism_therm_rank_c_coupling: f64,
    prism_therm_rank_k_kcc: f64,
    prism_therm_rank_l_localization: f64,
    gtck_rank: f64,
    tide_te_out_proxy: f64,
    tide_te_in_proxy: f64,
    tide_delta_hc_proxy: f64,
    hysteresis_asymmetry: f64,
    relative_asymmetry: f64,
    ccns_tau_proxy: f64,
    cold_phase_fraction: f64,
    hot_phase_fraction: f64,
    cryptic_spike_bin_count: u32,
    cryptic_spike_max_bin_count: u64,
    cryptic_inter_spike_bin_gap_mean: Option<f64>,
    physiology_rank: f64,
    physiology_reasoning: Vec<String>,
}

const HYDROPHOBIC: &[&str] = &[
    "ALA", "VAL", "LEU", "ILE", "PRO", "MET", "TRP", "PHE", "GLY",
];
const POLAR: &[&str] = &[
    "SER", "THR", "ASN", "GLN", "TYR", "CYS", "HIS", "ASP", "GLU", "LYS", "ARG",
];
const AROMATIC: &[&str] = &["PHE", "TYR", "TRP", "HIS"];

// ─── Manifold consistency block ────────────────────────────────────────────

#[derive(Debug, Default, Serialize, Clone)]
struct ManifoldConsistency {
    d_whole_lining_a: Option<f32>,
    d_whole_driver_a: Option<f32>,
    d_hot_cold_a: Option<f32>,
    manifold_dispersion_a: Option<f32>,
    manifold_consistency_score: f64,
    n_views_present: u32,
}

// ─── Refined temporal support ──────────────────────────────────────────────

#[derive(Debug, Default, Serialize, Clone)]
struct RefinedTemporalSupport {
    persistence_bin_size_steps: i32,
    persistence_bins_observed: u32,
    persistence_bins_total_possible: u32,
    persistence_fraction: f64,
    burstiness_score: f64,
    temporal_entropy: f64,
    /// `coarse_non_discriminative` when timesteps are unavailable or all
    /// candidates saturate the same bins; otherwise `available`.
    temporal_support_status: &'static str,
}

// ─── KCC driver block (active in scoring) ──────────────────────────────────

#[derive(Debug, Default, Serialize, Clone)]
struct KccDriverBlock {
    driver_residue_count: u32,
    driver_residue_fraction_of_shell: f64,
    driver_centroid_present: bool,
    driver_centroid_distance_to_whole_site_a: Option<f32>,
    active_causal_support_sum: u64,
    kcc_field_completeness: &'static str,
    kcc_driver_score: f64,
    kcc_driver_reasoning: Vec<String>,
}

// ─── Stream balance block ──────────────────────────────────────────────────

#[derive(Debug, Default, Serialize, Clone)]
struct StreamBalance {
    n_streams_supporting: u32,
    per_stream_spike_counts: serde_json::Map<String, serde_json::Value>,
    gini_coefficient: f64,
    stream_balance_score: f64,
    per_stream_support_balance_status: &'static str,
}

// ─── Duplicate-cluster report record ───────────────────────────────────────

#[derive(Debug, Serialize)]
struct DuplicateClusterRecord {
    duplicate_cluster_id: u32,
    kept_region_id: u32,
    kept_centroid_a: [f32; 3],
    suppressed_raw_peak_ids: Vec<u32>,
    n_suppressed: u32,
    centroid_spread_a: f32,
    residue_shell_overlap_with_rep: f32,
    merge_reasons: Vec<String>,
}

// ─── Main ──────────────────────────────────────────────────────────────────

// ─── M1.2.23 §12 — Ghost/ZSTR features integration ────────────────────────

#[derive(Debug, Deserialize, Clone)]
struct GhostMappedComponent {
    component_id: String,
    stream_id: u32,
    site_id: u32,
    centroid_xyz: [f32; 3],
    aabb: Option<[f32; 6]>,
    #[allow(dead_code)]
    residue_support: Vec<i32>,
    coordinate_frame: String,
    mapping_method: String,
    mapping_source: String,
    ghost_event_count: u64,
    ghost_tile_score_mean: f64,
    #[allow(dead_code)]
    ghost_tile_score_max: f64,
    zstr_bytes: u64,
    #[serde(default)]
    mapping_scope: Option<String>,
    #[serde(default)]
    exact_component_adapter: bool,
    #[serde(default)]
    component_step_min: Option<u64>,
    #[serde(default)]
    component_step_max: Option<u64>,
    #[serde(default)]
    exact_component_provenance: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Clone, Default)]
struct GhostRegionLink {
    component_id: String,
    stream_id: u32,
    site_id: u32,
    centroid_xyz: [f32; 3],
    coordinate_frame: String,
    mapping_method: String,
    mapping_source: String,
    distance_a: f64,
    match_basis: String,
    ghost_event_count: u64,
    zstr_bytes: u64,
    evidence_score: f64,
    contribution_factor: f64,
    mapping_scope: Option<String>,
    exact_component_adapter: bool,
    component_step_min: Option<u64>,
    component_step_max: Option<u64>,
    exact_component_provenance: Option<serde_json::Value>,
    native_region_filter: Option<serde_json::Value>,
}

#[derive(Debug, Default)]
struct GhostFeaturesState {
    loaded: bool,
    path: Option<String>,
    total_ghost_events: u64,
    spatial_mapped: bool,
    spatial_status_text: String,
    mapped_components: Vec<GhostMappedComponent>,
    exact_component_adapter_receipt: Option<serde_json::Value>,
}

fn load_ghost_features(path: &Path) -> Result<GhostFeaturesState> {
    let s = std::fs::read_to_string(path)
        .with_context(|| format!("read ghost features {}", path.display()))?;
    let v: serde_json::Value = serde_json::from_str(&s)
        .with_context(|| format!("parse ghost features {}", path.display()))?;
    if v.get("schema_kind").and_then(|value| value.as_str())
        != Some("pathb_ghost_zstr_candidate_features")
    {
        anyhow::bail!("ghost features {} has wrong schema_kind", path.display());
    }
    let total_ghost_events = v
        .get("per_stream_ghost_support")
        .and_then(|x| x.as_object())
        .map(|m| {
            m.values()
                .filter_map(|s| s.get("events").and_then(|y| y.as_u64()))
                .sum::<u64>()
        })
        .unwrap_or(0);
    let spatial_status_text = v
        .get("spatial_status")
        .and_then(|x| x.as_str())
        .unwrap_or("unknown")
        .to_string();
    let mapped_components: Vec<GhostMappedComponent> = serde_json::from_value(
        v.get("mapped_components")
            .cloned()
            .unwrap_or_else(|| serde_json::Value::Array(Vec::new())),
    )
    .with_context(|| format!("parse mapped_components from {}", path.display()))?;
    let exact_component_adapter_receipt = v
        .get("exact_component_adapter")
        .and_then(|adapter| adapter.get("receipt"))
        .cloned()
        .or_else(|| v.get("exact_component_adapter_receipt").cloned());
    let spatial_mapped =
        spatial_status_text.starts_with("mapped_") && !mapped_components.is_empty();
    if spatial_status_text.starts_with("mapped_") && mapped_components.is_empty() {
        anyhow::bail!(
            "ghost features {} claims mapped spatial evidence but has no mapped_components",
            path.display()
        );
    }
    if spatial_status_text == "mapped_components_from_exact_native_component_join_v1" {
        let receipt = exact_component_adapter_receipt.as_ref().with_context(|| {
            format!(
                "ghost features {} selected exact native component adapter but omitted receipt",
                path.display()
            )
        })?;
        anyhow::ensure!(
            receipt.get("status").and_then(|value| value.as_str())
                == Some("IMPLEMENTATION_VERIFIED"),
            "ghost features {} exact adapter receipt is not IMPLEMENTATION_VERIFIED",
            path.display()
        );
        anyhow::ensure!(
            receipt
                .get("mapped_component_count")
                .and_then(|value| value.as_u64())
                == Some(mapped_components.len() as u64),
            "ghost features {} exact adapter mapped count disagrees with mapped_components",
            path.display()
        );
        for component in &mapped_components {
            anyhow::ensure!(
                component.exact_component_adapter,
                "ghost features {} exact adapter output contains legacy mapped component {}",
                path.display(),
                component.component_id
            );
            anyhow::ensure!(
                component.mapping_method == "exact_native_component_join.component_aabb_midpoint",
                "ghost features {} exact adapter component {} has unexpected mapping_method={}",
                path.display(),
                component.component_id,
                component.mapping_method
            );
            anyhow::ensure!(
                component.aabb.is_some(),
                "ghost features {} exact adapter component {} lacks component AABB",
                path.display(),
                component.component_id
            );
            let (Some(step_min), Some(step_max)) =
                (component.component_step_min, component.component_step_max)
            else {
                anyhow::bail!(
                    "ghost features {} exact adapter component {} lacks component step bounds",
                    path.display(),
                    component.component_id
                );
            };
            anyhow::ensure!(
                step_min <= step_max,
                "ghost features {} exact adapter component {} has inverted step bounds",
                path.display(),
                component.component_id
            );
            anyhow::ensure!(
                component.exact_component_provenance.is_some(),
                "ghost features {} exact adapter component {} lacks exact provenance",
                path.display(),
                component.component_id
            );
        }
    }
    Ok(GhostFeaturesState {
        loaded: true,
        path: Some(path.display().to_string()),
        total_ghost_events,
        spatial_mapped,
        spatial_status_text,
        mapped_components,
        exact_component_adapter_receipt,
    })
}

// Verify common files once; then bind each adapted component to the authenticated graph.
fn validate_exact_ghost_adapter_input(
    path: &Path,
    run_id: &str,
    target: &str,
    state: &GhostFeaturesState,
) -> Result<serde_json::Value> {
    use prism_nhs::ensemble::manifest::sha256_file;
    let features: serde_json::Value = serde_json::from_reader(File::open(path)?)?;
    anyhow::ensure!(
        features["run_id"] == run_id && features["target"] == target,
        "exact Ghost adapter features belong to another run/target"
    );
    let adapter = state
        .exact_component_adapter_receipt
        .as_ref()
        .context("missing exact adapter receipt")?;
    let namespace = adapter["selected_namespace"]
        .as_str()
        .context("missing exact graph namespace")?;
    let graph_path = Path::new(
        adapter["source_graph_path"]
            .as_str()
            .context("missing graph path")?,
    );
    let graph_sha = sha256_file(graph_path)?;
    anyhow::ensure!(
        adapter["source_graph_sha256"] == graph_sha,
        "exact adapter graph hash mismatch"
    );
    let graph: serde_json::Value = serde_json::from_reader(File::open(graph_path)?)?;
    anyhow::ensure!(
        graph["run_id"] == run_id && graph["target"] == target,
        "exact adapter graph run/target mismatch"
    );
    let joins = features["exact_native_component_joins"]
        .as_array()
        .context("missing exact component joins")?;
    let selected: Vec<_> = joins
        .iter()
        .filter(|j| j["namespace"] == namespace)
        .collect();
    anyhow::ensure!(selected.len() == 1, "ambiguous exact graph namespace");
    let join = selected[0];
    anyhow::ensure!(
        join["source_graph_sha256"] == graph_sha
            && join["all_nodes_exactly_once"] == true
            && join["all_events_exactly_once"] == true,
        "exact component join source/conservation mismatch"
    );
    let lineage = &graph["ghost_record_byte_lineage_artifact"];
    let lineage_sha = sha256_file(Path::new(
        lineage["path"].as_str().context("missing graph lineage")?,
    ))?;
    anyhow::ensure!(
        lineage["sha256"] == lineage_sha
            && adapter["byte_lineage_artifact_sha256"] == lineage_sha
            && join["byte_lineage_artifact_sha256"] == lineage_sha,
        "exact graph lineage hash mismatch"
    );
    let mut source_bindings = Vec::new();
    for (path_key, hash_key) in [
        ("event_join_path", "event_join_sha256"),
        (
            "component_observations_path",
            "component_observations_sha256",
        ),
    ] {
        let input = Path::new(
            join[path_key]
                .as_str()
                .context("missing exact adapter source path")?,
        );
        let digest = sha256_file(input)?;
        anyhow::ensure!(
            join[hash_key] == digest && adapter[path_key] == join[path_key],
            "exact adapter source path/hash mismatch: {path_key}"
        );
        source_bindings.push(serde_json::json!({"path":input,"sha256":digest}));
    }
    let graph_components = graph["components"]
        .as_array()
        .context("missing graph components")?;
    let mut index = std::collections::BTreeMap::new();
    for component in graph_components {
        let id = component["component_id"]
            .as_u64()
            .context("missing component identity")?;
        anyhow::ensure!(
            index.insert(id, component).is_none(),
            "duplicate native component identity"
        );
    }
    let mut seen = std::collections::BTreeSet::new();
    let mut mapped_events = 0u64;
    for component in &state.mapped_components {
        let provenance = component
            .exact_component_provenance
            .as_ref()
            .context("missing native component provenance")?;
        let id = provenance["source_component_id"]
            .as_u64()
            .context("missing source component ID")?;
        anyhow::ensure!(seen.insert(id), "duplicate adapted native component");
        let original = index
            .get(&id)
            .context("adapted component absent from graph")?;
        anyhow::ensure!(
            component.component_id == format!("{namespace}:{id}")
                && provenance["namespace"] == namespace
                && provenance["source_graph_sha256"] == graph_sha
                && provenance["byte_lineage_artifact_sha256"] == lineage_sha,
            "adapted component source identity mismatch"
        );
        let lo: [f32; 3] = serde_json::from_value(original["component_aabb_min"].clone())?;
        let hi: [f32; 3] = serde_json::from_value(original["component_aabb_max"].clone())?;
        anyhow::ensure!(
            (0..3).all(|i| lo[i].is_finite() && hi[i].is_finite() && lo[i] <= hi[i]),
            "native component AABB invalid"
        );
        anyhow::ensure!(
            component.aabb == Some([lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]])
                && component.centroid_xyz
                    == [
                        0.5 * (lo[0] + hi[0]),
                        0.5 * (lo[1] + hi[1]),
                        0.5 * (lo[2] + hi[2])
                    ],
            "adapted AABB/centroid differs from native component"
        );
        let nodes = original["node_lineage"]
            .as_array()
            .context("missing native node membership")?;
        anyhow::ensure!(
            component.ghost_event_count == nodes.len() as u64
                && !nodes.is_empty()
                && original["n_nodes"].as_u64() == Some(component.ghost_event_count),
            "adapted component count differs from native membership"
        );
        let mut step_min = u64::MAX;
        let mut step_max = 0u64;
        for node in nodes {
            anyhow::ensure!(
                node["stream_id"].as_u64() == Some(u64::from(component.stream_id))
                    && node["site_id"].as_u64() == Some(u64::from(component.site_id)),
                "adapted component is not the declared single stream/site"
            );
            let step = node["step_idx"].as_u64().context("missing native step")?;
            step_min = step_min.min(step);
            step_max = step_max.max(step);
        }
        anyhow::ensure!(
            component.component_step_min == Some(step_min)
                && component.component_step_max == Some(step_max),
            "adapted component time support differs from native membership"
        );
        anyhow::ensure!(
            component.ghost_tile_score_mean.is_finite()
                && component.ghost_tile_score_max.is_finite()
                && (0.0..=1.0).contains(&component.ghost_tile_score_mean)
                && component.ghost_tile_score_mean <= component.ghost_tile_score_max
                && component.ghost_tile_score_max <= 1.0,
            "invalid existing native tile score range"
        );
        mapped_events += component.ghost_event_count;
    }
    Ok(
        serde_json::json!({"status":"IMPLEMENTATION_VERIFIED","features_path":path,
        "features_sha256":sha256_file(path)?,"run_id":run_id,"target":target,
        "source_graph_path":graph_path,"source_graph_sha256":graph_sha,
        "byte_lineage_artifact_sha256":lineage_sha,"checked_component_count":seen.len(),
        "checked_event_membership_count":mapped_events,"source_bindings":source_bindings,
        "validation":"common source hashes once; per-component graph identity, AABB, exact stream/site membership and min/max step bounds"}),
    )
}

#[derive(Debug, Default)]
struct GhostZstrRegionSummary {
    factor: f64,
    status: &'static str,
    reasoning: Vec<String>,
    links: Vec<GhostRegionLink>,
}

fn point_aabb_distance(point: [f32; 3], aabb: [f32; 6]) -> f64 {
    let mut squared = 0.0_f64;
    for axis in 0..3 {
        let value = f64::from(point[axis]);
        let lo = f64::from(aabb[axis]);
        let hi = f64::from(aabb[axis + 3]);
        let delta = if value < lo {
            lo - value
        } else if value > hi {
            value - hi
        } else {
            0.0
        };
        squared += delta * delta;
    }
    squared.sqrt()
}

fn exact_component_adapter_region_filter_status(
    component: &GhostMappedComponent,
    region_streams: &std::collections::HashMap<u32, u64>,
    region_timestep_min: i32,
    region_timestep_max: i32,
) -> (bool, serde_json::Value) {
    let stream_pass = region_streams.contains_key(&component.stream_id);
    let component_ordered = component
        .component_step_min
        .zip(component.component_step_max)
        .map(|(lo, hi)| lo <= hi)
        .unwrap_or(false);
    let region_step_min = if region_timestep_min == i32::MAX {
        None
    } else {
        u64::try_from(region_timestep_min).ok()
    };
    let region_step_max = if region_timestep_max == i32::MIN {
        None
    } else {
        u64::try_from(region_timestep_max).ok()
    };
    let region_ordered = region_step_min
        .zip(region_step_max)
        .map(|(lo, hi)| lo <= hi)
        .unwrap_or(false);
    let step_overlap = match (
        component.component_step_min,
        component.component_step_max,
        region_step_min,
        region_step_max,
    ) {
        (Some(component_min), Some(component_max), Some(region_min), Some(region_max))
            if component_min <= component_max && region_min <= region_max =>
        {
            component_min <= region_max && component_max >= region_min
        }
        _ => false,
    };
    let pass = stream_pass && step_overlap;
    let status = if pass {
        "pass"
    } else if !stream_pass {
        "fail_stream_absent_from_region_per_stream_spikes"
    } else if !component_ordered || !region_ordered {
        "fail_missing_or_inverted_step_bounds"
    } else {
        "fail_no_inclusive_step_window_overlap"
    };
    let mut region_stream_ids = region_streams.keys().copied().collect::<Vec<_>>();
    region_stream_ids.sort_unstable();
    (
        pass,
        serde_json::json!({
            "schema_kind": "prism_exact_component_region_filter",
            "schema_version": 1,
            "predicate": "region.per_stream_spikes.contains_key(component.stream_id) && component_step_min <= region.timestep_max && component_step_max >= region.timestep_min",
            "predicate_semantics": "broad recorded step-window overlap only; not exact coincident causal event ownership",
            "component_id": component.component_id.as_str(),
            "component_stream_id": component.stream_id,
            "component_site_id": component.site_id,
            "component_step_min": component.component_step_min,
            "component_step_max": component.component_step_max,
            "region_stream_ids": region_stream_ids,
            "region_timestep_min": region_step_min,
            "region_timestep_max": region_step_max,
            "stream_pass": stream_pass,
            "step_overlap_inclusive": step_overlap,
            "status": status,
        }),
    )
}

fn ghost_zstr_summary_for_geometry_impl(
    state: &GhostFeaturesState,
    region: Option<&CandidateRegion>,
    region_id: u32,
    centroid: [f32; 3],
    radius_a: f32,
) -> GhostZstrRegionSummary {
    if !state.loaded {
        return GhostZstrRegionSummary {
            factor: 1.0,
            status: "absent_neutral",
            ..GhostZstrRegionSummary::default()
        };
    }
    if !state.spatial_mapped {
        return GhostZstrRegionSummary {
            factor: 1.0,
            status: "present_non_spatial",
            reasoning: vec![format!(
                "ghost_features.spatial_status={} → no per-region linking; factor neutral",
                state.spatial_status_text
            )],
            links: Vec::new(),
        };
    }
    let admission_radius = f64::from(radius_a.max(2.0) + 2.0);
    let mut links = Vec::new();
    let mut exact_adapter_filter_reject_count = 0usize;
    for component in &state.mapped_components {
        let mut native_region_filter = None;
        if component.exact_component_adapter {
            let Some(region) = region else {
                exact_adapter_filter_reject_count += 1;
                continue;
            };
            let (filter_pass, receipt) = exact_component_adapter_region_filter_status(
                component,
                &region.per_stream_spikes,
                region.timestep_min,
                region.timestep_max,
            );
            native_region_filter = Some(receipt);
            if !filter_pass {
                exact_adapter_filter_reject_count += 1;
                continue;
            }
        }
        let dx = f64::from(component.centroid_xyz[0] - centroid[0]);
        let dy = f64::from(component.centroid_xyz[1] - centroid[1]);
        let dz = f64::from(component.centroid_xyz[2] - centroid[2]);
        let centroid_distance = (dx * dx + dy * dy + dz * dz).sqrt();
        let (distance, match_basis) = if let Some(aabb) = component.aabb {
            let distance = point_aabb_distance(centroid, aabb);
            (distance, "region_centroid_to_component_aabb")
        } else {
            (centroid_distance, "region_centroid_to_component_centroid")
        };
        if distance > admission_radius {
            continue;
        }
        let event_support = (component.ghost_event_count as f64).ln_1p() / 101.0_f64.ln();
        let zstr_support = if component.zstr_bytes > 0 { 1.0 } else { 0.0 };
        let evidence_score = (0.75 * component.ghost_tile_score_mean.clamp(0.0, 1.0)
            + 0.15 * event_support.clamp(0.0, 1.0)
            + 0.10 * zstr_support)
            .clamp(0.0, 1.0);
        let contribution_factor = (0.7 + 0.9 * evidence_score).clamp(0.7, 1.6);
        links.push(GhostRegionLink {
            component_id: component.component_id.clone(),
            stream_id: component.stream_id,
            site_id: component.site_id,
            centroid_xyz: component.centroid_xyz,
            coordinate_frame: component.coordinate_frame.clone(),
            mapping_method: component.mapping_method.clone(),
            mapping_source: component.mapping_source.clone(),
            distance_a: distance,
            match_basis: match_basis.to_string(),
            ghost_event_count: component.ghost_event_count,
            zstr_bytes: component.zstr_bytes,
            evidence_score,
            contribution_factor,
            mapping_scope: component.mapping_scope.clone(),
            exact_component_adapter: component.exact_component_adapter,
            component_step_min: component.component_step_min,
            component_step_max: component.component_step_max,
            exact_component_provenance: component.exact_component_provenance.clone(),
            native_region_filter,
        });
    }
    if links.is_empty() {
        let mut reasoning = vec![format!(
            "region_id={region_id} has no mapped Ghost/ZSTR component within {admission_radius:.3} A; factor neutral"
        )];
        if exact_adapter_filter_reject_count > 0 {
            reasoning.push(format!(
                "exact_component_adapter_region_filter_rejected={exact_adapter_filter_reject_count}; predicate=stream_id_present_and_inclusive_recorded_step_window_overlap"
            ));
        }
        return GhostZstrRegionSummary {
            factor: 1.0,
            status: "present_spatial_unmatched_neutral",
            reasoning,
            links,
        };
    }
    let weight_sum = links
        .iter()
        .map(|link| link.ghost_event_count.max(1) as f64)
        .sum::<f64>();
    let factor = links
        .iter()
        .map(|link| link.contribution_factor * link.ghost_event_count.max(1) as f64)
        .sum::<f64>()
        / weight_sum;
    GhostZstrRegionSummary {
        factor: factor.clamp(0.7, 1.6),
        status: "present_spatial_linked",
        reasoning: {
            let mut reasoning = vec![format!(
                "region_id={region_id} linked to {} Ghost/ZSTR components from {}",
                links.len(),
                state.path.as_deref().unwrap_or("unknown")
            )];
            if exact_adapter_filter_reject_count > 0 {
                reasoning.push(format!(
                    "exact_component_adapter_region_filter_rejected={exact_adapter_filter_reject_count}; predicate=stream_id_present_and_inclusive_recorded_step_window_overlap"
                ));
            }
            reasoning
        },
        links,
    }
}

fn ghost_zstr_summary_for_geometry(
    state: &GhostFeaturesState,
    region_id: u32,
    centroid: [f32; 3],
    radius_a: f32,
) -> GhostZstrRegionSummary {
    ghost_zstr_summary_for_geometry_impl(state, None, region_id, centroid, radius_a)
}

fn ghost_zstr_summary_compute(
    state: &GhostFeaturesState,
    region: &CandidateRegion,
) -> GhostZstrRegionSummary {
    ghost_zstr_summary_for_geometry_impl(
        state,
        Some(region),
        region.region_id,
        region.aggregate_centroid_a,
        region.region_radius_a,
    )
}

fn ghost_features_consumption_receipt(state: &GhostFeaturesState) -> serde_json::Value {
    serde_json::json!({
        "schema_kind": "prism_materializer_ghost_zstr_features_consumption",
        "schema_version": 1,
        "loaded": state.loaded,
        "path": state.path.as_deref(),
        "total_ghost_events": state.total_ghost_events,
        "spatial_mapped": state.spatial_mapped,
        "spatial_status": state.spatial_status_text.as_str(),
        "mapped_component_count": state.mapped_components.len(),
        "exact_component_adapter_receipt": state.exact_component_adapter_receipt.as_ref(),
        "exact_component_adapter_region_predicate": "region.per_stream_spikes.contains_key(component.stream_id) && component_step_min <= region.timestep_max && component_step_max >= region.timestep_min",
        "exact_component_adapter_region_predicate_semantics": "broad recorded step-window overlap only; not exact coincident causal event ownership",
        "factor_policy": "existing ghost_zstr_summary_for_geometry score formula unchanged; no new weights or thresholds",
    })
}

#[derive(Debug, Deserialize)]
struct ChemDtsgStateGraph {
    schema: String,
    status: String,
    fallback_used: bool,
    input_sha256: String,
    source_state_graph_sha256: String,
    parquet_sha256: String,
    edges: Vec<ChemDtsgStateEdge>,
}

#[derive(Debug, Deserialize)]
struct ChemDtsgStateEdge {
    from_residue: i32,
    to_residue: i32,
    baseline: ChemDtsgBaselineMetrics,
    perturbed: ChemDtsgPerturbedMetrics,
}

#[derive(Debug, Deserialize)]
struct ChemDtsgBaselineMetrics {
    te_out: f64,
    te_in: f64,
    delta_hc: f64,
}

#[derive(Debug, Deserialize)]
struct ChemDtsgPerturbedMetrics {
    te_out_mean: f64,
    te_in_mean: f64,
    delta_hc_mean: f64,
    u_pose_te: f64,
    u_pose_hc: f64,
}

#[derive(Debug, Default)]
struct ChemDtsgState {
    loaded: bool,
    path: Option<String>,
    graph: Option<ChemDtsgStateGraph>,
}

#[derive(Debug, Serialize, Clone, Default)]
struct ChemDtsgRegionLink {
    edge_index: usize,
    from_residue: i32,
    to_residue: i32,
    baseline_magnitude: f64,
    perturbed_magnitude: f64,
    pose_uncertainty: f64,
    relative_effect: f64,
    contribution_factor: f64,
}

#[derive(Debug, Default)]
struct ChemDtsgRegionSummary {
    factor: f64,
    status: &'static str,
    links: Vec<ChemDtsgRegionLink>,
}

fn load_chem_dtsg_state(path: &Path) -> Result<ChemDtsgState> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("read chem-DTSG state graph {}", path.display()))?;
    let graph: ChemDtsgStateGraph = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse chem-DTSG state graph {}", path.display()))?;
    if graph.schema != "prismos.chem_perturbed_dtsg_state_graph.v1"
        || graph.status != "PASS"
        || graph.fallback_used
        || graph.edges.is_empty()
    {
        anyhow::bail!(
            "chem-DTSG state graph {} failed schema/status/no-fallback/nonempty gate",
            path.display()
        );
    }
    for digest in [
        &graph.input_sha256,
        &graph.source_state_graph_sha256,
        &graph.parquet_sha256,
    ] {
        if digest.len() != 64 || !digest.as_bytes().iter().all(u8::is_ascii_hexdigit) {
            anyhow::bail!(
                "chem-DTSG state graph {} has invalid digest",
                path.display()
            );
        }
    }
    Ok(ChemDtsgState {
        loaded: true,
        path: Some(path.display().to_string()),
        graph: Some(graph),
    })
}

fn chem_dtsg_summary_compute(
    state: &ChemDtsgState,
    region: &CandidateRegion,
) -> ChemDtsgRegionSummary {
    let Some(graph) = state.graph.as_ref() else {
        return ChemDtsgRegionSummary {
            factor: 1.0,
            status: "absent_neutral",
            ..ChemDtsgRegionSummary::default()
        };
    };
    let mut links = Vec::new();
    for (edge_index, edge) in graph.edges.iter().enumerate() {
        if !region.residue_support.contains_key(&edge.from_residue)
            && !region.residue_support.contains_key(&edge.to_residue)
        {
            continue;
        }
        let baseline_magnitude =
            edge.baseline.te_out.abs() + edge.baseline.te_in.abs() + edge.baseline.delta_hc.abs();
        let perturbed_magnitude = edge.perturbed.te_out_mean.abs()
            + edge.perturbed.te_in_mean.abs()
            + edge.perturbed.delta_hc_mean.abs();
        let pose_uncertainty = edge.perturbed.u_pose_te + edge.perturbed.u_pose_hc;
        let relative_effect =
            (perturbed_magnitude - baseline_magnitude) / baseline_magnitude.max(1.0e-12);
        let confidence = 1.0 / (1.0 + pose_uncertainty.max(0.0));
        let contribution_factor =
            (1.0 + 0.15 * (relative_effect * confidence).tanh()).clamp(0.8, 1.2);
        links.push(ChemDtsgRegionLink {
            edge_index,
            from_residue: edge.from_residue,
            to_residue: edge.to_residue,
            baseline_magnitude,
            perturbed_magnitude,
            pose_uncertainty,
            relative_effect,
            contribution_factor,
        });
    }
    if links.is_empty() {
        return ChemDtsgRegionSummary {
            factor: 1.0,
            status: "present_unmatched_neutral",
            links,
        };
    }
    let factor = links
        .iter()
        .map(|link| link.contribution_factor)
        .sum::<f64>()
        / links.len() as f64;
    ChemDtsgRegionSummary {
        factor: factor.clamp(0.8, 1.2),
        status: "present_region_linked",
        links,
    }
}

fn main() -> Result<()> {
    if std::env::args().nth(1).as_deref() == Some("--recover-manifest") {
        return recover_manifest(RecoveryArgs::parse_from(std::iter::once(std::env::args().next().unwrap()).chain(std::env::args().skip(2))));
    }
    let args = Args::parse();
    let rank_policy_id = args.ranking_policy.id();
    let rank_key_order = args.ranking_policy.key_order();
    let ranking_methodology = if args.ranking_policy == RankingPolicy::NativeKcc {
        "prismos_native_kcc_then_lexicographic_site_materialization_v1"
    } else {
        RANKING_METHODOLOGY_V2
    };
    let promotion_policy = serde_json::json!({
        "policy_id": "PRISMOS_CONFIGURABLE_STREAM_SUPPORT_V1",
        "min_supporting_streams": args.min_supporting_streams,
        "min_spikes_per_supporting_stream": MIN_SPIKES_PER_SUPPORTING_STREAM,
        "configuration": "--min-supporting-streams; backward_compatible_default=3",
        "materialized_top_n": args.materialized_top_n,
        "support_semantics": "qualifying_native_stream_ids; independent_replication_not_established_by_stream_count"
    });
    let log_level = if args.verbose { "info" } else { "warn" };
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", log_level);
    }
    env_logger::init();

    eprintln!("prism-materialize-sites — Path B materializer");
    eprintln!("manifest: {}", args.manifest.display());

    let manifest_text = std::fs::read_to_string(&args.manifest)
        .with_context(|| format!("read manifest {}", args.manifest.display()))?;
    let manifest: Manifest = serde_json::from_str(&manifest_text)
        .with_context(|| format!("parse manifest {}", args.manifest.display()))?;

    if manifest.schema_kind != "md_evidence_manifest" {
        anyhow::bail!(
            "manifest schema_kind = {:?}, expected \"md_evidence_manifest\"",
            manifest.schema_kind
        );
    }
    if manifest.schema_version != 1 {
        anyhow::bail!(
            "manifest schema_version = {}, this binary supports 1",
            manifest.schema_version
        );
    }

    let output_dir = args
        .output_dir
        .clone()
        .or_else(|| args.manifest.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."));
    std::fs::create_dir_all(&output_dir)
        .with_context(|| format!("create output dir {}", output_dir.display()))?;
    let hydration_dir = output_dir.join("PRISMOS_QNANL_HYDRATED_INPUTS");
    let mut hydration_records: Vec<serde_json::Value> = Vec::new();

    eprintln!("output dir: {}", output_dir.display());
    eprintln!(
        "run_id: {}  target: {}  streams: {}/{}",
        manifest.run_id, manifest.target, manifest.streams_serialized, manifest.stream_count
    );
    eprintln!("topology: {}", manifest.topology_input);
    eprintln!(
        "compression={}  checksum={}  endian={}",
        manifest.compression_status, manifest.checksum_algorithm, manifest.endian
    );
    if manifest.serialization_failure.is_some() {
        eprintln!(
            "WARNING: manifest reports serialization_failure: {:?}",
            manifest.serialization_failure
        );
    }

    // ─── Validate every present binary artifact ────────────────────────────
    let mut validations: Vec<ArtifactValidation> = Vec::new();
    let mut spike_artifacts: Vec<(u32, String, String, u64)> = Vec::new(); // stream_id, manifest path, resolved path, record_count

    for art in manifest.artifacts.iter() {
        if !art.present {
            continue;
        }
        let p = PathBuf::from(&art.path);
        let (magic, trailer): (Option<[u8; 8]>, Option<[u8; 8]>) = match art.kind.as_str() {
            "spikes" => (Some(*b"PRSPK001"), Some(*b"PRSPKEND")),
            "signal_grid" => (Some(*b"PRSGD001"), Some(*b"PRSGDEND")),
            "kcc_v2full" => (Some(*b"PRKCC001"), Some(*b"PRKCCEND")),
            // rayon_scope_emit (warp_matrix, forces, etc.) — no PRISM envelope.
            _ => (None, None),
        };
        let (Some(magic), Some(trailer)) = (magic, trailer) else {
            continue;
        };
        let canonical_packet_path = canonical_packet_path_for(&p, &art.kind)
            .unwrap_or_else(|_| PathBuf::from(format!("{}.prismqnanl", p.display())));
        let canonical_packet_present = canonical_packet_path.exists();
        let resolved_p = match hydrate_artifact_from_packet(&p, &art.kind, &hydration_dir) {
            Ok(resolved) => {
                let source_kind = if canonical_packet_present && resolved != p {
                    "hydrated_from_canonical_packet"
                } else if resolved == p {
                    "raw_manifest_path_no_canonical_packet_platform_violation"
                } else {
                    "hydrated_from_nonstandard_source_platform_violation"
                };
                hydration_records.push(serde_json::json!({
                    "artifact_kind": art.kind,
                    "manifest_path": art.path,
                    "canonical_packet_path": canonical_packet_path.display().to_string(),
                    "canonical_packet_present": canonical_packet_present,
                    "resolved_path": resolved.display().to_string(),
                    "source_kind": source_kind,
                    "platform_canonicalization_status": if canonical_packet_present { "canonical_packet_consumed" } else { "raw_only_platform_violation" },
                }));
                resolved
            }
            Err(e) => {
                eprintln!("  ✗ {} kind={} HYDRATE-ERROR: {}", p.display(), art.kind, e);
                hydration_records.push(serde_json::json!({
                    "artifact_kind": art.kind,
                    "manifest_path": art.path,
                    "canonical_packet_path": canonical_packet_path.display().to_string(),
                    "canonical_packet_present": canonical_packet_present,
                    "resolved_path": serde_json::Value::Null,
                    "source_kind": "missing_or_invalid_canonical_packet",
                    "platform_canonicalization_status": "canonical_packet_required_hydration_failed",
                    "failure": e.to_string(),
                }));
                validations.push(ArtifactValidation {
                    artifact_path: p.display().to_string(),
                    expected_kind: art.kind.clone(),
                    magic_observed: String::new(),
                    magic_match: false,
                    schema_version: 0,
                    endian_marker_hex: String::new(),
                    endian_ok_le: false,
                    stream_id: 0,
                    stream_id_matches_filename: None,
                    run_id_in_header: String::new(),
                    run_id_matches_manifest: false,
                    stem_in_header: String::new(),
                    stem_matches_manifest: false,
                    record_count: 0,
                    record_count_matches_manifest: None,
                    byte_stride: 0,
                    payload_size: 0,
                    declared_total_size: 0,
                    on_disk_size: 0,
                    size_consistency: false,
                    payload_checksum_fnv1a_64_hex: String::new(),
                    checksum_matches_manifest: None,
                    trailer_observed: String::new(),
                    trailer_match: false,
                    valid: false,
                    failure_reason: Some(format!("hydrate error: {}", e)),
                });
                continue;
            }
        };
        let filename_stream_id = parse_stream_id_from_filename(&p);
        match validate_artifact(
            &resolved_p,
            &magic,
            &trailer,
            &art.kind,
            art.record_count,
            art.checksum_fnv1a_64_hex.as_deref(),
            &manifest.run_id,
            &manifest.target,
            filename_stream_id,
            args.headers_only,
        ) {
            Ok((mut v, _hdr)) => {
                if art.stream_id != Some(v.stream_id) {
                    v.valid = false;
                    v.failure_reason = Some(
                        "envelope stream identity disagrees with manifest artifact stream_id"
                            .into(),
                    );
                }
                if v.valid {
                    eprintln!(
                        "  ✓ {} kind={} stream={} records={} stride={}",
                        p.display(),
                        art.kind,
                        v.stream_id,
                        v.record_count,
                        v.byte_stride
                    );
                    if art.kind == "spikes" {
                        spike_artifacts.push((
                            v.stream_id,
                            art.path.clone(),
                            resolved_p.display().to_string(),
                            v.record_count,
                        ));
                    }
                } else {
                    eprintln!(
                        "  ✗ {} kind={} INVALID: {}",
                        resolved_p.display(),
                        art.kind,
                        v.failure_reason.as_deref().unwrap_or("(unknown)")
                    );
                }
                validations.push(v);
            }
            Err(e) => {
                eprintln!(
                    "  ✗ {} kind={} READ-ERROR: {}",
                    resolved_p.display(),
                    art.kind,
                    e
                );
                validations.push(ArtifactValidation {
                    artifact_path: resolved_p.display().to_string(),
                    expected_kind: art.kind.clone(),
                    magic_observed: String::new(),
                    magic_match: false,
                    schema_version: 0,
                    endian_marker_hex: String::new(),
                    endian_ok_le: false,
                    stream_id: 0,
                    stream_id_matches_filename: None,
                    run_id_in_header: String::new(),
                    run_id_matches_manifest: false,
                    stem_in_header: String::new(),
                    stem_matches_manifest: false,
                    record_count: 0,
                    record_count_matches_manifest: None,
                    byte_stride: 0,
                    payload_size: 0,
                    declared_total_size: 0,
                    on_disk_size: 0,
                    size_consistency: false,
                    payload_checksum_fnv1a_64_hex: String::new(),
                    checksum_matches_manifest: None,
                    trailer_observed: String::new(),
                    trailer_match: false,
                    valid: false,
                    failure_reason: Some(format!("read error: {}", e)),
                });
            }
        }
    }

    let mut input_failures: Vec<String> = validations
        .iter()
        .filter(|v| !v.valid)
        .map(|v| {
            format!(
                "{}: {}",
                v.artifact_path,
                v.failure_reason.as_deref().unwrap_or("invalid artifact")
            )
        })
        .collect();
    if let Some(reason) = &manifest.serialization_failure {
        input_failures.push(format!("manifest serialization failure: {}", reason));
    }
    if manifest.streams_serialized != manifest.stream_count {
        input_failures.push(format!(
            "manifest serialized {} of {} required streams",
            manifest.streams_serialized, manifest.stream_count
        ));
    }
    for row in &hydration_records {
        if row
            .get("platform_canonicalization_status")
            .and_then(|v| v.as_str())
            != Some("canonical_packet_consumed")
        {
            input_failures.push(format!("canonical input admission failed: {}", row));
        }
    }
    let distinct_spike_streams: std::collections::HashSet<u32> =
        spike_artifacts.iter().map(|row| row.0).collect();
    if distinct_spike_streams.len() != spike_artifacts.len()
        || distinct_spike_streams.len() != manifest.streams_serialized
    {
        input_failures
            .push("validated spike stream identity/cardinality differs from manifest".to_string());
    }

    // ─── Density-peak pass over spikes ─────────────────────────────────────
    let mut voxel_grid: std::collections::HashMap<(i32, i32, i32), u32> =
        std::collections::HashMap::new();
    let mut stream_summaries: Vec<StreamSpikeSummary> = Vec::new();
    let mut total_processed: u64 = 0;
    let density_pass_skipped = args.headers_only;
    if !args.headers_only {
        for (stream_id, manifest_path, resolved_path, record_count) in spike_artifacts.iter() {
            eprintln!(
                "  [density] stream {} processing {} resolved_from={} ({} records)…",
                stream_id, resolved_path, manifest_path, record_count
            );
            let p = PathBuf::from(resolved_path);
            // Read byte_stride from the header to feed process_spikes_stream
            // — we already validated the file above; reread just the header.
            let f = File::open(&p)?;
            let mut r = BufReader::new(f);
            let h = read_envelope_header(&mut r)?;
            drop(r);
            match process_spikes_stream(
                &p,
                *record_count,
                h.byte_stride,
                args.voxel_size_a,
                &mut voxel_grid,
            ) {
                Ok(s) => {
                    eprintln!(
                        "    ✓ stream {}: {} spikes, bbox=[{:.1},{:.1},{:.1}]→[{:.1},{:.1},{:.1}], elapsed={} ms",
                        s.stream_id, s.n_spikes,
                        s.bbox_min[0], s.bbox_min[1], s.bbox_min[2],
                        s.bbox_max[0], s.bbox_max[1], s.bbox_max[2],
                        s.elapsed_ms
                    );
                    total_processed += s.n_spikes;
                    stream_summaries.push(s);
                }
                Err(e) => {
                    eprintln!("    ✗ stream {}: {}", stream_id, e);
                    input_failures.push(format!(
                        "spike stream {} density decoding failed: {}",
                        stream_id, e
                    ));
                }
            }
        }
    }

    let candidates: Vec<DensityPeak> = if !args.headers_only && !voxel_grid.is_empty() {
        extract_top_density_peaks(&voxel_grid, args.voxel_size_a, args.top_k_candidates)
    } else {
        Vec::new()
    };

    let first_class_materialization_kinds: std::collections::HashSet<&'static str> = [
        "signal_grid",
        "kcc_v2full",
        "warp_matrix",
        "forces_final",
        "aromatic_centroids_final",
        "adaptive_dt",
        "bocpd_log",
        "spike_transition_events",
        "protocol_state",
        "noise_floor",
        "asc_vectors",
    ]
    .into_iter()
    .collect();
    let mut tensor_surface_summaries: Vec<serde_json::Value> = Vec::new();
    for art in manifest.artifacts.iter() {
        if !art.present || !first_class_materialization_kinds.contains(art.kind.as_str()) {
            continue;
        }
        tensor_surface_summaries.push(summarize_first_class_surface(art, &hydration_dir));
    }
    let mut tensor_surface_counts_by_kind: std::collections::BTreeMap<String, u64> =
        std::collections::BTreeMap::new();
    let mut tensor_surface_blockers: Vec<serde_json::Value> = Vec::new();
    for summary in tensor_surface_summaries.iter() {
        let kind = summary
            .get("kind")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        *tensor_surface_counts_by_kind.entry(kind).or_insert(0) += 1;
        let status = summary
            .get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let canonical_violation = summary
            .get("source")
            .and_then(|s| s.get("platform_canonicalization_status"))
            .and_then(|v| v.as_str())
            == Some("raw_only_platform_violation");
        if status.starts_with("blocked_") || canonical_violation {
            tensor_surface_blockers.push(summary.clone());
        }
    }
    let materialized_surface_count = |kind: &str| -> usize {
        tensor_surface_summaries
            .iter()
            .filter(|s| {
                s.get("kind").and_then(|v| v.as_str()) == Some(kind)
                    && s.get("status")
                        .and_then(|v| v.as_str())
                        .map(|status| status == "materialized_parsed")
                        .unwrap_or(false)
            })
            .count()
    };
    let manifest_surface_count = |kind: &str| -> usize {
        manifest
            .artifacts
            .iter()
            .filter(|a| a.present && a.kind == kind)
            .count()
    };
    let surface_status_for_kind = |kind: &str| -> &'static str {
        if materialized_surface_count(kind) > 0 {
            "materialized_parsed"
        } else if manifest_surface_count(kind) > 0 {
            "blocked_materialization_parser_error"
        } else {
            "missing"
        }
    };
    let signal_grid_surface_status = surface_status_for_kind("signal_grid");
    let kcc_surface_status = surface_status_for_kind("kcc_v2full");
    let warp_matrix_surface_status = surface_status_for_kind("warp_matrix");
    let forces_surface_status = surface_status_for_kind("forces_final");
    let asc_vectors_surface_status = surface_status_for_kind("asc_vectors");
    let adaptive_dt_surface_status = surface_status_for_kind("adaptive_dt");
    let bocpd_log_surface_status = surface_status_for_kind("bocpd_log");
    let protocol_state_surface_status = surface_status_for_kind("protocol_state");
    let spike_transition_surface_status = surface_status_for_kind("spike_transition_events");
    let dense_so3_surfaces_parsed = warp_matrix_surface_status == "materialized_parsed"
        && forces_surface_status == "materialized_parsed"
        && asc_vectors_surface_status == "materialized_parsed";

    if !args.headers_only && total_processed != manifest.total_spikes_md {
        input_failures.push(format!(
            "decoded {} spikes but manifest declares {}",
            total_processed, manifest.total_spikes_md
        ));
    }
    for failure in &tensor_surface_blockers {
        input_failures.push(format!(
            "declared tensor surface failed materialization: {}",
            failure
        ));
    }

    // ─── Emit materialization_report.json ──────────────────────────────────
    let report_path = output_dir.join("materialization_report.json");
    let report = serde_json::json!({
        "schema_version": 1,
        "schema_kind":    "pathb_materialization_report",
        "promotion_policy": promotion_policy,
        "input_admission_status": if input_failures.is_empty() { "VALIDATED" } else { "FAILED_INPUT_VALIDATION" },
        "input_failures": input_failures,
        "completion_scope": if args.headers_only { "HEADERS_ONLY" } else { "FULL_MATERIALIZATION_REQUESTED" },
        "tool":           "prism-materialize-sites",
        "tool_version":   env!("CARGO_PKG_VERSION"),
        "manifest_path":  args.manifest.display().to_string(),
        "manifest_run_id": manifest.run_id,
        "manifest_target": manifest.target,
        "manifest_stream_count":     manifest.stream_count,
        "manifest_streams_serialized": manifest.streams_serialized,
        "manifest_total_spikes_md":  manifest.total_spikes_md,
        "manifest_v2_was_live":      manifest.v2_was_live,
        "manifest_serialization_failure": manifest.serialization_failure,
        "validations":   validations,
        "stream_summaries": stream_summaries,
        "voxel_grid_unique_voxels": voxel_grid.len() as u64,
        "voxel_size_a":  args.voxel_size_a,
        "spike_density_pass_skipped": density_pass_skipped,
        "total_spikes_processed":     total_processed,
        "matches_manifest_total":     total_processed == manifest.total_spikes_md,
        "first_class_tensor_surface_materialization": {
            "status": if tensor_surface_blockers.is_empty() { "materialized_parsed" } else { "blocked_platform_surface_materialization" },
            "canonical_storage_policy": "PrismOS/QNANL packets are primary; raw files are debug/provenance fallback only",
            "counts_by_kind": tensor_surface_counts_by_kind.clone(),
            "surface_count": tensor_surface_summaries.len(),
            "blocker_count": tensor_surface_blockers.len(),
            "blockers": tensor_surface_blockers.clone(),
            "surfaces": tensor_surface_summaries.clone(),
        },
        "ml_rerank_status":           "not_run",
        "ligsite_status":             "not_run",
        "ph_refinement_status":       "not_run",
        "find_neighbors_union_status":"not_run",
        "binding_sites_materialization_status": "emitted_later_in_this_run",
        "qnanl_hydration": {
            "status": "enabled",
            "cache_dir": hydration_dir.display().to_string(),
            "records": hydration_records,
        },
        "path_b_required":            true,
    });
    {
        let f = File::create(&report_path)
            .with_context(|| format!("create {}", report_path.display()))?;
        let mut bw = std::io::BufWriter::new(f);
        serde_json::to_writer_pretty(&mut bw, &report)?;
        use std::io::Write as _;
        bw.flush()?;
    }
    eprintln!("✓ wrote {}", report_path.display());

    // ─── Emit site_candidates.json (HONEST density peaks, NOT binding sites) ──
    let candidates_path = output_dir.join("site_candidates.json");
    let candidates_json = serde_json::json!({
        "schema_version": 1,
        "schema_kind":    "pathb_site_candidates",
        "kind":           "voxel_density_peak",
        "kind_note":      "These are coarse spike-density voxel peaks. They are NOT materialized binding sites. \
                           This same tool subsequently performs PRISM dynamic materialization and lexicographic ranking.",
        "manifest_run_id": manifest.run_id,
        "manifest_target": manifest.target,
        "voxel_size_a":   args.voxel_size_a,
        "top_k":          args.top_k_candidates,
        "n_candidates":   candidates.len(),
        "n_spikes_basis": total_processed,
        "candidates":     candidates,
        "ml_reranked":    false,
        "ligsite_applied":false,
        "ph_refined":     false,
        "binding_sites_materialized": false,
    });
    {
        let f = File::create(&candidates_path)
            .with_context(|| format!("create {}", candidates_path.display()))?;
        let mut bw = std::io::BufWriter::new(f);
        serde_json::to_writer_pretty(&mut bw, &candidates_json)?;
        use std::io::Write as _;
        bw.flush()?;
    }
    eprintln!("✓ wrote {}", candidates_path.display());

    // ─── Emit validation_inputs.json (paths Path B downstream needs) ──────
    let validation_path = output_dir.join("validation_inputs.json");
    let validation_json = serde_json::json!({
        "schema_version": 1,
        "schema_kind":    "pathb_validation_inputs",
        "manifest_run_id": manifest.run_id,
        "manifest_target": manifest.target,
        "topology_input": manifest.topology_input,
        "topology_input_present": Path::new(&manifest.topology_input).exists(),
        "candidates_path": candidates_path.display().to_string(),
        "report_path":    report_path.display().to_string(),
        "materialization_protocol": rank_policy_id,
        "not_required_by_protocol": [
            "rt_clustering_offline",
            "dynamic_ligsite",
            "cubical_ph_refinement",
            "xgb_rerank"
        ],
        "ground_truth_input": serde_json::Value::Null,
        "ground_truth_status": "not_supplied_for_posthoc_validation",
    });
    {
        let f = File::create(&validation_path)
            .with_context(|| format!("create {}", validation_path.display()))?;
        let mut bw = std::io::BufWriter::new(f);
        serde_json::to_writer_pretty(&mut bw, &validation_json)?;
        use std::io::Write as _;
        bw.flush()?;
    }
    eprintln!("✓ wrote {}", validation_path.display());

    // ─── Emit field_completeness_report.pathb.json ────────────────────────
    let fcr_path = output_dir.join("field_completeness_report.pathb.json");
    let fcr = serde_json::json!({
        "schema_version": 1,
        "schema_kind":    "pathb_field_completeness_report",
        "manifest_run_id": manifest.run_id,
        "materializer_version": env!("CARGO_PKG_VERSION"),
        "fields": {
            "spikes_per_stream":            { "status": "available", "source": "PRSPK001 binaries on disk" },
            "signal_grid_per_stream":       { "status": signal_grid_surface_status, "source": "PRSGD001 payload hydrated from canonical QNANL when packetized", "streams_present": materialized_surface_count("signal_grid") },
            "kcc_v2full_per_stream":        { "status": kcc_surface_status, "source": "PRKCC001 payload hydrated from canonical QNANL when packetized", "streams_present": materialized_surface_count("kcc_v2full") },
            "warp_matrix_per_stream":       { "status": warp_matrix_surface_status, "source": "DenseGpuField QNANL payload; GpuWarpEntry[136]", "streams_present": materialized_surface_count("warp_matrix") },
            "forces_final_per_stream":      { "status": forces_surface_status, "source": "NumericArray QNANL payload; f32[n_atoms x 3]", "streams_present": materialized_surface_count("forces_final") },
            "asc_vectors_per_stream":       { "status": asc_vectors_surface_status, "source": "NumericArray QNANL payload; f32[n_atoms x 3]", "streams_present": materialized_surface_count("asc_vectors") },
            "adaptive_dt_per_stream":       { "status": adaptive_dt_surface_status, "source": "NumericArray QNANL payload; 32-byte adaptive dt records", "streams_present": materialized_surface_count("adaptive_dt") },
            "bocpd_log_per_stream":         { "status": bocpd_log_surface_status, "source": "PRISM_THERM_TEMPORAL_SURFACE QNANL/JSONL", "streams_present": materialized_surface_count("bocpd_log") },
            "spike_transition_events_per_stream": { "status": spike_transition_surface_status, "source": "EVENT_STREAM QNANL/JSONL", "streams_present": materialized_surface_count("spike_transition_events") },
            "voxel_density_candidates":     { "status": "available", "source": "prism-materialize-sites voxel-binned spike density" },
            "all_stream_spikes_concat":     { "status": "implicit_streamed", "source": "spikes were streamed in-memory per file; not concatenated to disk" },
            "merged_kcc":                   { "status": "per_stream_materialized_consumed", "source": "Path B consumes per-stream PRKCC001; merged tensor is not required for lexicographic site materialization" },
            "merged_signal_grid":           { "status": "per_stream_validated", "source": "Path B validates per-stream PRSGD001; merged tensor is not required for lexicographic site materialization" },
            "rt_clustering_sites":          { "status": "not_required_by_prismos_lexicographic_materializer", "source": "region consolidation is performed by this materializer" },
            "ligsite_pockets":              { "status": "intentionally_excluded_by_protocol", "source": "static LIGSITE geometry is excluded from PRISM dynamic materialization rank" },
            "cubical_ph_centroids":         { "status": "not_required_by_prismos_lexicographic_materializer", "source": "phase/coherence support is derived from PRISM spike phase bits and centroid manifold views" },
            "xgb_reranked_sites":           { "status": "not_required_by_prismos_lexicographic_materializer", "source": format!("final decision layer is {}", rank_policy_id) },
            "pocket_morphology_surface":    { "status": "emitted_by_prismos_materializer", "source": "PRISMOS_GEOMETRIC_POCKET_MORPHOLOGY_V1" },
            "binding_site_physiology_surface": { "status": "emitted_by_prismos_materializer", "source": "PRISMOS_BINDING_SITE_PHYSIOLOGY_V1" },
            "binding_sites_materialized":   { "status": "emitted_by_prismos_materializer", "source": "binding_sites.materialized.json" },
            "ground_truth_validation":      { "status": "requires_explicit_ground_truth_input", "source": "post-hoc validation boundary" },
            "transform_dag": {
                "status": if manifest.artifacts.iter().any(|a| a.kind == "transform_dag" && a.present) { "declared_in_manifest_not_parsed_by_this_materializer" } else { "not_declared_in_manifest" },
                "manifest_v2_was_live": manifest.v2_was_live,
                "reason": "availability follows explicit manifest artifact records; V2 liveness alone is not a transform-DAG artifact"
            }
        },
        "canonical_storage_policy": "PrismOS/QNANL packets are primary; raw files are debug/provenance fallback only",
        "first_class_tensor_surface_materialization": {
            "status": if tensor_surface_blockers.is_empty() { "materialized_parsed" } else { "blocked_platform_surface_materialization" },
            "counts_by_kind": tensor_surface_counts_by_kind.clone(),
            "surface_count": tensor_surface_summaries.len(),
            "blocker_count": tensor_surface_blockers.len(),
            "blockers": tensor_surface_blockers.clone(),
        },
        "path_b_required": true,
        "binding_sites_materialized": false,
    });
    {
        let f =
            File::create(&fcr_path).with_context(|| format!("create {}", fcr_path.display()))?;
        let mut bw = std::io::BufWriter::new(f);
        serde_json::to_writer_pretty(&mut bw, &fcr)?;
        use std::io::Write as _;
        bw.flush()?;
    }
    eprintln!("✓ wrote {}", fcr_path.display());

    // ═══════════════════════════════════════════════════════════════════════
    // MATERIALIZATION PASS — spatiotemporal SO(3) evidence per candidate
    // ═══════════════════════════════════════════════════════════════════════
    //
    // For each top-K density peak, scan spikes again and accumulate:
    //   * stream_support, temporal_support, phase_support
    //   * residue_support, source/aromatic histograms
    //   * hot/cold-phase position centroids
    // Then load per-stream KCC and identify driver residues. Then load
    // topology and compute residue-position-weighted lining_mass +
    // causal_driver centroids. Then load ground_truth.json and decide
    // whether DCC validation is appropriate.

    // Manifest-bound schedules: JSON sidecars need explicit stream identity;
    // absent embedded run_id is reported as manifest association, not attestation.
    let mut thermal_schedules = std::collections::HashMap::<u32, ThermalSchedule>::new();
    let mut thermal_schedule_records = Vec::<serde_json::Value>::new();
    let mut ambiguous_schedule_streams = std::collections::HashSet::<u32>::new();
    for art in manifest
        .artifacts
        .iter()
        .filter(|a| a.present && a.kind == "protocol_state")
    {
        let parsed = (|| -> Result<(u32, ThermalSchedule, PathBuf)> {
            let stream = art
                .stream_id
                .context("protocol manifest stream_id absent")?;
            anyhow::ensure!(
                spike_artifacts.iter().any(|s| s.0 == stream),
                "protocol stream has no validated spike envelope"
            );
            if let Some(filename_stream) = parse_stream_id_from_filename(Path::new(&art.path)) {
                anyhow::ensure!(
                    filename_stream == stream,
                    "protocol filename stream mismatch"
                );
            }
            let path =
                hydrate_artifact_from_packet(Path::new(&art.path), &art.kind, &hydration_dir)?;
            let schedule = read_thermal_schedule(&path, stream, &manifest.run_id)?;
            Ok((stream, schedule, path))
        })();
        match parsed {
            Ok((stream, schedule, path)) => {
                if thermal_schedules.contains_key(&stream)
                    || ambiguous_schedule_streams.contains(&stream)
                {
                    thermal_schedules.remove(&stream);
                    ambiguous_schedule_streams.insert(stream);
                    thermal_schedule_records.push(serde_json::json!({"stream_id": stream, "status": "unclassified_duplicate_schedule"}));
                } else {
                    thermal_schedule_records.push(serde_json::json!({
                        "stream_id": stream, "status": "schedule_parsed", "source": art.path,
                        "resolved_source": path, "schedule": schedule,
                        "run_association": "manifest_declared_sidecar_with_validated_same_stream_spike_envelope",
                        "measurement_scope": "prescribed_schedule_not_measured_temperature",
                        "physical_time_axis": "MD_step; adaptive_dt_source_required_for_ps_conversion",
                        "event_clock_upper_bound": schedule.current_step.unwrap_or(schedule.total_steps),
                        "event_clock_bound_source": if schedule.current_step.is_some() { "recorded_protocol_current_step" } else { "nominal_horizon_legacy_bound" },
                        "terminal_continuation": "native_terminal_hold_without_thermal_wrap; bounded_by_recorded_execution"
                    }));
                    thermal_schedules.insert(stream, schedule);
                }
            }
            Err(error) => {
                if let Some(stream) = art.stream_id {
                    thermal_schedules.remove(&stream);
                    ambiguous_schedule_streams.insert(stream);
                }
                thermal_schedule_records.push(serde_json::json!({
                    "stream_id": art.stream_id, "source": art.path, "status": "unclassified_invalid_schedule", "reason": error.to_string()
                }));
            }
        }
    }
    // ─── Pass 3 — load per-stream KCC, identify driver residues ─────────────
    let mut kcc_per_stream: Vec<KccPerStream> = Vec::new();
    let mut kcc_files_consumed: Vec<String> = Vec::new();
    for art in manifest.artifacts.iter() {
        if !art.present || art.kind != "kcc_v2full" {
            continue;
        }
        let p = PathBuf::from(&art.path);
        let resolved_p = match hydrate_artifact_from_packet(&p, &art.kind, &hydration_dir) {
            Ok(resolved) => resolved,
            Err(e) => {
                input_failures.push(format!("KCC hydration failed {}: {}", art.path, e));
                eprintln!(
                    "[materialize] kcc hydration failed for {}: {}",
                    p.display(),
                    e
                );
                continue;
            }
        };
        match read_kcc_v2full(&resolved_p) {
            Ok(mut k) => {
                if validate_kcc_association(&k, &manifest.run_id, &manifest.target, art.stream_id)
                    .is_err()
                {
                    input_failures.push(format!("KCC manifest association mismatch: {}", art.path));
                    continue;
                }
                let companions: Vec<_> = manifest
                    .artifacts
                    .iter()
                    .filter(|a| {
                        a.present && a.kind == "kcc_lag_provenance" && a.stream_id == art.stream_id
                    })
                    .collect();
                anyhow::ensure!(
                    companions.len() <= 1,
                    "ambiguous KCC lag sidecar association"
                );
                if let Some(companion) = companions.first() {
                    let companion_path = hydrate_artifact_from_packet(
                        Path::new(&companion.path),
                        &companion.kind,
                        &hydration_dir,
                    )?;
                    let companion_bytes = std::fs::read(&companion_path)?;
                    anyhow::ensure!(
                        companion_bytes.len() as u64 == companion.size_bytes,
                        "KCC lag sidecar manifest size mismatch"
                    );
                    let companion_checksum = format!("{:016x}", fnv1a_64(&companion_bytes));
                    anyhow::ensure!(
                        companion.checksum_fnv1a_64_hex.as_deref()
                            == Some(companion_checksum.as_str()),
                        "KCC lag sidecar manifest checksum mismatch"
                    );
                    attach_kcc_lag_provenance(&mut k, &companion_path).with_context(|| {
                        format!("required registered KCC lag sidecar {}", companion.path)
                    })?;
                }
                let support_companions: Vec<_> = manifest
                    .artifacts
                    .iter()
                    .filter(|a| {
                        a.present && a.kind == "kcc_lag_support" && a.stream_id == art.stream_id
                    })
                    .collect();
                anyhow::ensure!(
                    support_companions.len() <= 1,
                    "ambiguous KCC lag-support sidecar association"
                );
                if let Some(companion) = support_companions.first() {
                    let companion_path = hydrate_artifact_from_packet(
                        Path::new(&companion.path),
                        &companion.kind,
                        &hydration_dir,
                    )?;
                    let companion_bytes = std::fs::read(&companion_path)?;
                    anyhow::ensure!(
                        companion_bytes.len() as u64 == companion.size_bytes,
                        "KCC lag-support sidecar manifest size mismatch"
                    );
                    let companion_checksum = format!("{:016x}", fnv1a_64(&companion_bytes));
                    anyhow::ensure!(
                        companion.checksum_fnv1a_64_hex.as_deref()
                            == Some(companion_checksum.as_str()),
                        "KCC lag-support sidecar manifest checksum mismatch"
                    );
                    attach_kcc_lag_support(&mut k, &companion_path).with_context(|| {
                        format!(
                            "required registered KCC lag-support sidecar {}",
                            companion.path
                        )
                    })?;
                }
                kcc_per_stream.push(k);
                kcc_files_consumed.push(art.path.clone());
            }
            Err(e) => {
                input_failures.push(format!("KCC decode failed {}: {}", art.path, e));
                eprintln!(
                    "[materialize] kcc read failed for {} resolved_from={}: {}",
                    resolved_p.display(),
                    p.display(),
                    e
                );
            }
        }
    }
    for (stream, _, _, _) in &spike_artifacts {
        if !thermal_schedules.contains_key(stream) {
            thermal_schedule_records.push(serde_json::json!({"stream_id": stream,
                "status": "unclassified_no_unambiguous_valid_schedule"}));
        }
    }
    let adaptive_dt_axes: Vec<serde_json::Value> = manifest.artifacts.iter()
        .filter(|a| a.present && a.kind == "adaptive_dt").map(|art| {
            let axis = (|| -> Result<serde_json::Value> {
                let stream = art.stream_id.context("adaptive_dt stream_id absent")?;
                let path = hydrate_artifact_from_packet(Path::new(&art.path), &art.kind, &hydration_dir)?;
                adaptive_dt_time_axis(&path, thermal_schedules.get(&stream))
            })();
            serde_json::json!({"stream_id": art.stream_id, "source": art.path,
                "layout": "32_bytes:chunk_idx_u64,step_u64,dt_ps_f64,reason_u32,pad_u32",
                "time_axis": match axis { Ok(value) => value, Err(error) => serde_json::json!({
                    "physical_time_conversion_status": "unavailable", "reason": error.to_string()}) }})
        }).collect();
    let kcc_full_path = output_dir.join("kcc_descriptors.full_residue.prismos_v1.json");
    let kcc_full_surface = serde_json::json!({
        "schema_kind": "prismos_full_residue_kcc_descriptors", "schema_version": 1,
        "run_id": manifest.run_id, "target": manifest.target,
        "population_scope": "all_decoded_residues_all_available_streams_independent_of_density_selection_and_promotion",
        "streams": kcc_per_stream.iter().zip(kcc_files_consumed.iter())
            .map(|(k, p)| kcc_descriptor_surface(k, p)).collect::<Vec<_>>(),
        "protocol_schedules": thermal_schedule_records,
        "thermal_phase_order": THERMAL_PHASE_NAMES,
        "adaptive_dt_sources": adaptive_dt_axes,
        "input_admission_status": if input_failures.is_empty() { "VALIDATED" } else { "FAILED_INPUT_VALIDATION" },
        "input_failures": input_failures,
    });
    {
        let mut writer = std::io::BufWriter::new(File::create(&kcc_full_path)?);
        serde_json::to_writer_pretty(&mut writer, &kcc_full_surface)?;
        use std::io::Write as _;
        writer.flush()?;
    }
    let kcc_full_qnanl = seal_json_artifact(
        &kcc_full_path,
        "PRISMOS_FULL_RESIDUE_KCC_DESCRIPTORS_V1",
        QnanlArtifactFamily::KccMaterialization,
    )?;
    if !input_failures.is_empty() {
        anyhow::bail!("MATERIALIZATION_INPUT_VALIDATION_FAILED: full descriptor evidence preserved at {}; {} failures",
            kcc_full_path.display(), input_failures.len());
    }

    let native_observation_surface = match (
        &args.native_ghost_lattice,
        args.kcc_chronology.is_empty(),
    ) {
        (None, true) => None,
        (Some(lattice), false) => {
            #[cfg(any(feature = "gpu", feature = "cpu-analysis"))]
            {
                let path = ghost_kcc::emit(
                    lattice,
                    &args.kcc_chronology,
                    &manifest,
                    args.manifest
                        .parent()
                        .context("manifest has no evidence root")?,
                    &output_dir,
                    &thermal_schedules,
                )?;
                let seal = seal_json_artifact(
                    &path,
                    "PRISMOS_GHOST_KCC_OBSERVATION_MANIFEST_V1",
                    QnanlArtifactFamily::KccMaterialization,
                )?;
                Some(serde_json::json!({"manifest": path, "qnanl": seal,
                "scope": "exact_native_observation_association_independent_of_density_and_promotion"}))
            }
            #[cfg(not(any(feature = "gpu", feature = "cpu-analysis")))]
            {
                let _ = lattice;
                anyhow::bail!("native Ghost/KCC association requires cpu-analysis or gpu feature");
            }
        }
        _ => anyhow::bail!(
            "--native-ghost-lattice and at least one --kcc-chronology must be supplied together"
        ),
    };

    if args.headers_only || candidates.is_empty() {
        eprintln!("[materialize] skipped (headers_only or no candidates)");
        // Emit empty materialized stubs honestly.
        let materialized_path = output_dir.join("binding_sites.materialized.json");
        let stub = serde_json::json!({
            "schema_version": 1,
            "status": if !input_failures.is_empty() { "failed_input_validation" } else if args.headers_only { "headers_only" } else { "valid_no_sites" },
            "input_admission_status": if input_failures.is_empty() { "VALIDATED" } else { "FAILED_INPUT_VALIDATION" },
            "input_failures": input_failures,
            "target":         manifest.target,
            "run_id":         manifest.run_id,
            "source_manifest": args.manifest.display().to_string(),
            "promotion_policy": promotion_policy,
            "site_count":     0,
            "binding_sites":  Vec::<serde_json::Value>::new(),
            "non_materialized_candidates": Vec::<serde_json::Value>::new(),
            "native_ghost_kcc_observation_surface": native_observation_surface,
            "missing_fields": if args.headers_only { vec!["spikes_pass_skipped"] } else { Vec::<&str>::new() },
            "path_b_required": true,
        });
        let f = File::create(&materialized_path)
            .with_context(|| format!("create {}", materialized_path.display()))?;
        let mut bw = std::io::BufWriter::new(f);
        serde_json::to_writer_pretty(&mut bw, &stub)?;
        use std::io::Write as _;
        bw.flush()?;
        eprintln!(
            "✓ wrote {} (empty evidence artifact)",
            materialized_path.display()
        );
        if !input_failures.is_empty() {
            anyhow::bail!("MATERIALIZATION_INPUT_VALIDATION_FAILED: {} failures; emitted artifacts preserved; see {}", input_failures.len(), report_path.display());
        }
        return Ok(());
    }

    // ─── Build voxel→candidate-id lookup (radius 4 Å = 2 voxels at 2.0 Å) ─
    let radius_voxels: i32 = (4.0_f32 / args.voxel_size_a).ceil() as i32;
    let mut voxel_to_cands: std::collections::HashMap<(i32, i32, i32), Vec<u32>> =
        std::collections::HashMap::new();
    for (cand_idx, peak) in candidates.iter().enumerate() {
        for dx in -radius_voxels..=radius_voxels {
            for dy in -radius_voxels..=radius_voxels {
                for dz in -radius_voxels..=radius_voxels {
                    let key = (
                        peak.voxel_index[0] + dx,
                        peak.voxel_index[1] + dy,
                        peak.voxel_index[2] + dz,
                    );
                    voxel_to_cands.entry(key).or_default().push(cand_idx as u32);
                }
            }
        }
    }
    eprintln!(
        "[materialize] voxel→candidate lookup built: {} voxel keys (radius={} voxels)",
        voxel_to_cands.len(),
        radius_voxels
    );

    // ─── Pass 2 — accumulate per-candidate evidence ─────────────────────────
    let mut evidence: Vec<CandidateEvidence> = candidates
        .iter()
        .map(|p| CandidateEvidence::new(p.peak_id, p.voxel_index, p.centroid_a))
        .collect();
    let inv_voxel = 1.0_f32 / args.voxel_size_a;

    for (stream_id, _manifest_path, resolved_path, record_count) in spike_artifacts.iter() {
        let p = PathBuf::from(resolved_path);
        let f = File::open(&p)?;
        let mut r = BufReader::with_capacity(1 << 20, f);
        let h = read_envelope_header(&mut r)?;
        let stride = h.byte_stride as usize;
        anyhow::ensure!(stride >= 84, "spikes stride {} too small", stride);
        let mut remaining = h.record_count;
        let records_per_batch: u64 = 1 << 16;
        let mut buf = vec![0u8; (records_per_batch * h.byte_stride) as usize];
        while remaining > 0 {
            let batch = std::cmp::min(remaining, records_per_batch);
            let bytes = (batch * h.byte_stride) as usize;
            r.read_exact(&mut buf[..bytes])?;
            for k in 0..(batch as usize) {
                let off = k * stride;
                let rec = &buf[off..off + stride];
                let px = read_f32_at(rec, spike_offsets::POSITION_X);
                let py = read_f32_at(rec, spike_offsets::POSITION_Y);
                let pz = read_f32_at(rec, spike_offsets::POSITION_Z);
                let vx = (px * inv_voxel).floor() as i32;
                let vy = (py * inv_voxel).floor() as i32;
                let vz = (pz * inv_voxel).floor() as i32;
                let Some(cand_ids) = voxel_to_cands.get(&(vx, vy, vz)) else {
                    continue;
                };
                let intensity = read_f32_at(rec, spike_offsets::INTENSITY);
                let timestep = read_i32_at(rec, 0);
                let source = read_i32_at(rec, spike_offsets::SPIKE_SOURCE);
                let arom = read_i32_at(rec, spike_offsets::AROMATIC_TYPE);
                // phase_bits at offset 92 (last u32 in 96-byte struct)
                let phase_bits: u32 = if stride >= 96 {
                    let mut b = [0u8; 4];
                    b.copy_from_slice(&rec[92..96]);
                    u32::from_le_bytes(b)
                } else {
                    0
                };
                // nearby_residues[8] starts at offset 24 (after timestep+voxel_idx+pos[3]+intensity)
                let mut nearby_res = [0i32; 8];
                let n_res = read_i32_at(rec, 56).max(0).min(8) as usize;
                for j in 0..n_res {
                    let mut b = [0u8; 4];
                    b.copy_from_slice(&rec[24 + j * 4..24 + j * 4 + 4]);
                    nearby_res[j] = i32::from_le_bytes(b);
                }
                let phase_angle = (phase_bits as f64 / 1024.0) * 2.0 * std::f64::consts::PI;
                let cs = phase_angle.cos();
                let sn = phase_angle.sin();
                let phase_bin = ((phase_bits as usize) * 10 / 1024).min(9);
                let chunk_id = timestep / 1000;
                let persistence_bin_id = timestep / args.persistence_bin_steps.max(1);
                for &ci in cand_ids.iter() {
                    let ev = &mut evidence[ci as usize];
                    ev.n_spikes += 1;
                    ev.spike_support_geometry.observe([px, py, pz]);
                    ev.intensity_sum += intensity as f64;
                    ev.pos_sum_x += px as f64 * intensity as f64;
                    ev.pos_sum_y += py as f64 * intensity as f64;
                    ev.pos_sum_z += pz as f64 * intensity as f64;
                    *ev.per_stream_spikes.entry(*stream_id).or_insert(0) += 1;
                    ev.per_stream_voxels
                        .entry(*stream_id)
                        .or_default()
                        .insert((vx, vy, vz));
                    if timestep < ev.timestep_min {
                        ev.timestep_min = timestep;
                    }
                    if timestep > ev.timestep_max {
                        ev.timestep_max = timestep;
                    }
                    ev.chunk_set.insert(chunk_id);
                    *ev.bin_counts.entry(persistence_bin_id).or_insert(0) += 1;
                    ev.phase_hist[phase_bin] += 1;
                    ev.phase_cos_sum += cs;
                    ev.phase_sin_sum += sn;
                    let thermal_phase = thermal_schedules
                        .get(stream_id)
                        .map(|schedule| schedule.phase(timestep))
                        .unwrap_or(6);
                    ev.thermal.observe(thermal_phase, [px, py, pz]);
                    if thermal_phase == 2 {
                        for axis in 0..3 {
                            ev.hot_pos_sum[axis] += [px, py, pz][axis] as f64;
                        }
                        ev.n_hot += 1;
                    } else if thermal_phase == 0 {
                        for axis in 0..3 {
                            ev.cold_pos_sum[axis] += [px, py, pz][axis] as f64;
                        }
                        ev.n_cold += 1;
                    }
                    match source {
                        1 => ev.n_uv += 1,
                        2 => ev.n_lif += 1,
                        _ => ev.n_other_source += 1,
                    }
                    let arom_bin = match arom {
                        0 => 0,
                        1 => 1,
                        2 => 2,
                        3 => 3,
                        _ => 4,
                    };
                    ev.n_arom[arom_bin] += 1;
                    for j in 0..n_res {
                        *ev.residue_support.entry(nearby_res[j]).or_insert(0) += 1;
                        *ev.per_stream_residue_support
                            .entry(*stream_id)
                            .or_default()
                            .entry(nearby_res[j])
                            .or_default() += 1;
                    }
                }
            }
            remaining -= batch;
        }
        let _ = record_count;
        let total_attributed: u64 = evidence
            .iter()
            .map(|e| e.per_stream_spikes.get(stream_id).copied().unwrap_or(0))
            .sum();
        eprintln!(
            "[materialize]  stream {} contributed {} attributed spikes",
            stream_id, total_attributed
        );
    }

    // Driver residues = residues with sum(active_causal) > 0 across streams.
    let mut driver_active_causal_sum: std::collections::HashMap<usize, u64> =
        std::collections::HashMap::new();
    for k in kcc_per_stream.iter() {
        for (rid, &count) in k.active_causal.iter().enumerate() {
            if count > 0 {
                *driver_active_causal_sum.entry(rid).or_insert(0) += count as u64;
            }
        }
    }
    eprintln!(
        "[materialize] {} streams with KCC; {} driver residues (active_causal>0)",
        kcc_per_stream.len(),
        driver_active_causal_sum.len()
    );

    // ─── Pass 4 — load topology for residue centers (CA-based) ──────────────
    let topo_path = PathBuf::from(&manifest.topology_input);
    let author_contract_present = author_contract_present_in_topology(&topo_path)?;
    let (topology, residue_centers): (Option<TopologyMin>, Vec<Option<[f32; 3]>>) =
        match TopologyMin::load(&topo_path) {
            Ok(t) => {
                let centers = t.ca_positions_by_residue();
                eprintln!(
                    "[materialize] topology loaded: {} atoms, {} residues, {} CA centers resolved",
                    t.n_atoms,
                    t.n_residues,
                    centers.iter().filter(|c| c.is_some()).count()
                );
                (Some(t), centers)
            }
            Err(e) => {
                if author_contract_present { return Err(e.context("author residue contract rejected")); }
                eprintln!(
                    "[materialize] topology load FAILED: {} (residue-based centroids skipped)",
                    e
                );
                (None, Vec::new())
            }
        };
    let author_residue_mapping = match topology.as_ref().and_then(|t| t.analysis_residue_contract.as_ref()) {
        Some(contract) => contract.author_mapping_surface().map_err(anyhow::Error::msg)?,
        None => serde_json::json!({"status": "unavailable", "reason": "authoritative_analysis_residue_contract_absent", "records": null}),
    };
    let residue_alignment_policy = topology
        .as_ref()
        .map(|t| t.residue_alignment_policy_json(&topo_path))
        .unwrap_or_else(|| {
            serde_json::json!({
                "policy_id": RESIDUE_ALIGNMENT_POLICY_ID,
                "topology_path": topo_path.display().to_string(),
                "residue_coordinate_system": "zero_based_topology_residue_idx",
                "status": "topology_missing",
                "external_numbering_requires_residue_map": true,
            })
        });

    // ─── Pass 5 — load ground truth and decide validation status ────────────
    let gt_path = output_dir.join(format!("{}_ground_truth.json", manifest.target));
    let (ground_truth, gt_status, gt_skip_reason): (
        Option<GroundTruth>,
        &'static str,
        Option<String>,
    ) = match std::fs::read_to_string(&gt_path) {
        Ok(s) => match serde_json::from_str::<GroundTruth>(&s) {
            Ok(gt) => {
                let valid = gt.valid_for_dcc_validation && gt.ligand_centroid.is_some();
                let status = if valid {
                    "available"
                } else {
                    "reference_unavailable"
                };
                let reason = gt.skip_reason.clone();
                (Some(gt), status, reason)
            }
            Err(e) => {
                eprintln!("[materialize] ground_truth.json parse failed: {}", e);
                (
                    None,
                    "reference_unavailable",
                    Some(format!("parse_error: {}", e)),
                )
            }
        },
        Err(_) => (
            None,
            "reference_unavailable",
            Some("ground_truth_file_missing".to_string()),
        ),
    };
    eprintln!("[materialize] ground_truth status: {}", gt_status);

    // materialization_report.json is emitted before post-hoc ground truth is
    // loaded. Rewrite only provenance/status fields here so downstream
    // operational ledgers can bind the validation sidecar without changing
    // site scoring, DCC computation, or ranking.
    let ground_truth_input = match std::fs::metadata(&gt_path) {
        Ok(meta) => {
            let checksum = std::fs::read(&gt_path)
                .map(|bytes| format!("{:016x}", fnv1a_64(&bytes)))
                .unwrap_or_else(|_| "unavailable".to_string());
            serde_json::json!({
                "path": gt_path.display().to_string(),
                "exists": true,
                "size_bytes": meta.len(),
                "checksum_fnv1a_64_hex": checksum,
                "validation_role": "post_hoc_site_accuracy_reference",
            })
        }
        Err(_) => serde_json::json!({
            "path": gt_path.display().to_string(),
            "exists": false,
            "validation_role": "post_hoc_site_accuracy_reference",
        }),
    };
    if let Ok(report_text) = std::fs::read_to_string(&report_path) {
        if let Ok(mut report_json) = serde_json::from_str::<serde_json::Value>(&report_text) {
            if let Some(obj) = report_json.as_object_mut() {
                obj.insert("ground_truth_input".to_string(), ground_truth_input.clone());
                obj.insert(
                    "ground_truth_status".to_string(),
                    serde_json::json!(gt_status),
                );
                obj.insert(
                    "ground_truth_skip_reason".to_string(),
                    serde_json::json!(gt_skip_reason.clone()),
                );
            }
            let f = File::create(&report_path)
                .with_context(|| format!("rewrite {}", report_path.display()))?;
            let mut bw = std::io::BufWriter::new(f);
            serde_json::to_writer_pretty(&mut bw, &report_json)?;
            use std::io::Write as _;
            bw.flush()?;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // PRISM-NATIVE PIPELINE v2 (commit dc175006+1):
    //   Stage A — raw peaks (per top-K voxel)
    //   Stage B — consolidation by centroid distance + residue-shell Jaccard
    //   Pass C — per-region: residue-shell plausibility, manifold consistency,
    //            KCC driver, refined temporal, stream balance
    //   Pass D — composite PRISM-native score
    //   Pass E — promotion (no automatic promotion of all candidates)
    //   Pass F — post-hoc accuracy (Chain C reference is NOT used in ranking)
    // ═══════════════════════════════════════════════════════════════════════

    let min_spikes_per_stream = MIN_SPIKES_PER_SUPPORTING_STREAM;
    let min_supporting_streams = args.min_supporting_streams;

    let mut spikes_files_consumed: Vec<String> = manifest
        .artifacts
        .iter()
        .filter(|a| a.present && a.kind == "spikes")
        .map(|a| a.path.clone())
        .collect();
    spikes_files_consumed.sort();

    let n_total_kcc_streams = kcc_per_stream.len();
    let signal_grid_present_any: bool = manifest
        .artifacts
        .iter()
        .any(|a| a.present && a.kind == "signal_grid");

    // ─── Stage A — convert per-peak evidence into RawPeak records ────────
    let mut raw_peaks: Vec<RawPeak> = evidence
        .into_iter()
        .enumerate()
        .map(|(idx, ev)| {
            let mut lining_pairs: Vec<(i32, u32)> =
                ev.residue_support.iter().map(|(r, c)| (*r, *c)).collect();
            lining_pairs.sort_by(count_desc_identity_asc);
            let top12: Vec<i32> = lining_pairs.iter().take(12).map(|(r, _)| *r).collect();
            RawPeak {
                raw_peak_id: idx as u32,
                voxel: ev.seed_voxel,
                seed_centroid_a: ev.seed_centroid_a,
                evidence: ev,
                top12_lining_residues: top12,
            }
        })
        .collect();
    // Sort by raw n_spikes descending so the highest-density peak seeds each region.
    raw_peaks.sort_by(|a, b| {
        b.evidence
            .n_spikes
            .cmp(&a.evidence.n_spikes)
            .then_with(|| a.voxel.cmp(&b.voxel))
            .then_with(|| a.raw_peak_id.cmp(&b.raw_peak_id))
    });
    eprintln!(
        "[v2] Stage A — {} raw peaks (sorted by n_spikes)",
        raw_peaks.len()
    );

    // ─── Stage B — greedy consolidation into candidate regions ───────────
    fn aggregate_centroid_of_evidence(ev: &CandidateEvidence) -> [f32; 3] {
        if ev.intensity_sum > 0.0 {
            vec3_div([ev.pos_sum_x, ev.pos_sum_y, ev.pos_sum_z], ev.intensity_sum)
        } else {
            ev.seed_centroid_a
        }
    }
    let mut regions: Vec<CandidateRegion> = Vec::new();
    let mut duplicate_records: Vec<DuplicateClusterRecord> = Vec::new();

    for raw in raw_peaks.iter() {
        let raw_centroid = aggregate_centroid_of_evidence(&raw.evidence);
        // Find first region this peak should merge into.
        let mut merge_into: Option<usize> = None;
        let mut merge_reasons: Vec<String> = Vec::new();
        let mut residue_overlap_observed: f32 = 0.0;
        for (ri, region) in regions.iter().enumerate() {
            let d = dist3(&raw_centroid, &region.aggregate_centroid_a);
            let mut rep_top12: Vec<i32> = region
                .residue_support
                .iter()
                .map(|(r, c)| (*r, *c))
                .collect::<Vec<_>>()
                .into_iter()
                .map(|(r, c)| (r, c))
                .collect::<Vec<(i32, u32)>>()
                .iter()
                .take(12)
                .map(|(r, _)| *r)
                .collect();
            // The above pattern is ugly; rebuild more cleanly:
            let mut tmp: Vec<(i32, u32)> = region
                .residue_support
                .iter()
                .map(|(r, c)| (*r, *c))
                .collect();
            tmp.sort_by(count_desc_identity_asc);
            rep_top12 = tmp.into_iter().take(12).map(|(r, _)| r).collect();
            let jacc = jaccard_overlap(&raw.top12_lining_residues, &rep_top12);
            if d <= args.consolidation_radius_a {
                merge_into = Some(ri);
                merge_reasons.push(format!(
                    "centroid_distance_{:.2}A_le_{:.1}",
                    d, args.consolidation_radius_a
                ));
                residue_overlap_observed = jacc;
                break;
            }
            if jacc >= args.consolidation_jaccard {
                merge_into = Some(ri);
                merge_reasons.push(format!(
                    "residue_jaccard_{:.2}_ge_{:.2}_centroid_dist_{:.2}A",
                    jacc, args.consolidation_jaccard, d
                ));
                residue_overlap_observed = jacc;
                break;
            }
        }
        if let Some(ri) = merge_into {
            // Aggregate raw evidence into the existing region.
            let region = &mut regions[ri];
            region.merged_raw_peak_ids.push(raw.raw_peak_id);
            region.merge_reasons.extend(merge_reasons);
            region.n_spikes += raw.evidence.n_spikes;
            region
                .spike_support_geometry
                .merge(&raw.evidence.spike_support_geometry);
            region.intensity_sum += raw.evidence.intensity_sum;
            region.pos_sum_x += raw.evidence.pos_sum_x;
            region.pos_sum_y += raw.evidence.pos_sum_y;
            region.pos_sum_z += raw.evidence.pos_sum_z;
            for (sid, cnt) in raw.evidence.per_stream_spikes.iter() {
                *region.per_stream_spikes.entry(*sid).or_insert(0) += cnt;
            }
            for (sid, vox) in raw.evidence.per_stream_voxels.iter() {
                let entry = region.per_stream_voxels.entry(*sid).or_default();
                for v in vox.iter() {
                    entry.insert(*v);
                }
            }
            if raw.evidence.timestep_min < region.timestep_min {
                region.timestep_min = raw.evidence.timestep_min;
            }
            if raw.evidence.timestep_max > region.timestep_max {
                region.timestep_max = raw.evidence.timestep_max;
            }
            for c in raw.evidence.chunk_set.iter() {
                region.chunk_set.insert(*c);
            }
            for (b, c) in raw.evidence.bin_counts.iter() {
                *region.bin_counts.entry(*b).or_insert(0) += c;
                region.persistence_bin_set.insert(*b);
            }
            for k in 0..10 {
                region.phase_hist[k] += raw.evidence.phase_hist[k];
            }
            region.phase_cos_sum += raw.evidence.phase_cos_sum;
            region.phase_sin_sum += raw.evidence.phase_sin_sum;
            for k in 0..3 {
                region.hot_pos_sum[k] += raw.evidence.hot_pos_sum[k];
                region.cold_pos_sum[k] += raw.evidence.cold_pos_sum[k];
            }
            region.n_hot += raw.evidence.n_hot;
            region.n_cold += raw.evidence.n_cold;
            region.thermal.merge(&raw.evidence.thermal);
            region.n_uv += raw.evidence.n_uv;
            region.n_lif += raw.evidence.n_lif;
            region.n_other_source += raw.evidence.n_other_source;
            for k in 0..5 {
                region.n_arom[k] += raw.evidence.n_arom[k];
            }
            for (rid, cnt) in raw.evidence.residue_support.iter() {
                *region.residue_support.entry(*rid).or_insert(0) += cnt;
            }
            for (stream, residues) in &raw.evidence.per_stream_residue_support {
                for (residue, count) in residues {
                    *region
                        .per_stream_residue_support
                        .entry(*stream)
                        .or_default()
                        .entry(*residue)
                        .or_default() += count;
                }
            }
            // Recompute aggregate centroid from updated sums.
            region.aggregate_centroid_a = if region.intensity_sum > 0.0 {
                vec3_div(
                    [region.pos_sum_x, region.pos_sum_y, region.pos_sum_z],
                    region.intensity_sum,
                )
            } else {
                region.aggregate_centroid_a
            };
            region.region_radius_a = region
                .region_radius_a
                .max(dist3(&raw_centroid, &region.aggregate_centroid_a));
            region.centroid_spread_a = region.region_radius_a;
            region.residue_shell_overlap_with_rep = region
                .residue_shell_overlap_with_rep
                .max(residue_overlap_observed);
        } else {
            // New region seeded from this raw peak.
            let region_id = regions.len() as u32;
            regions.push(CandidateRegion {
                region_id,
                representative_raw_peak_id: raw.raw_peak_id,
                merged_raw_peak_ids: vec![raw.raw_peak_id],
                aggregate_centroid_a: raw_centroid,
                region_radius_a: 0.0,
                n_spikes: raw.evidence.n_spikes,
                intensity_sum: raw.evidence.intensity_sum,
                pos_sum_x: raw.evidence.pos_sum_x,
                pos_sum_y: raw.evidence.pos_sum_y,
                pos_sum_z: raw.evidence.pos_sum_z,
                per_stream_spikes: raw.evidence.per_stream_spikes.clone(),
                per_stream_voxels: raw.evidence.per_stream_voxels.clone(),
                timestep_min: raw.evidence.timestep_min,
                timestep_max: raw.evidence.timestep_max,
                chunk_set: raw.evidence.chunk_set.clone(),
                persistence_bin_set: raw.evidence.bin_counts.keys().copied().collect(),
                bin_counts: raw.evidence.bin_counts.clone(),
                phase_hist: raw.evidence.phase_hist,
                phase_cos_sum: raw.evidence.phase_cos_sum,
                phase_sin_sum: raw.evidence.phase_sin_sum,
                hot_pos_sum: raw.evidence.hot_pos_sum,
                n_hot: raw.evidence.n_hot,
                cold_pos_sum: raw.evidence.cold_pos_sum,
                n_cold: raw.evidence.n_cold,
                thermal: raw.evidence.thermal.clone(),
                n_uv: raw.evidence.n_uv,
                n_lif: raw.evidence.n_lif,
                n_other_source: raw.evidence.n_other_source,
                n_arom: raw.evidence.n_arom,
                residue_support: raw.evidence.residue_support.clone(),
                spike_support_geometry: raw.evidence.spike_support_geometry.clone(),
                per_stream_residue_support: raw.evidence.per_stream_residue_support.clone(),
                merge_reasons: vec!["region_seed".to_string()],
                support_accounting: serde_json::json!({"policy": "legacy-peak-v2", "unique_record_count": null, "status": "not_recounted"}),
                centroid_spread_a: 0.0,
                residue_shell_overlap_with_rep: 1.0,
            });
        }
    }
    eprintln!(
        "[v2] Stage B — {} raw peaks consolidated to {} regions (radius={} Å, jaccard>={})",
        raw_peaks.len(),
        regions.len(),
        args.consolidation_radius_a,
        args.consolidation_jaccard
    );

    if args.support_accounting == SupportAccountingPolicy::UniqueRecordV1 {
        recount_unique_region_support(
            &mut regions,
            &voxel_to_cands,
            &spike_artifacts,
            &thermal_schedules,
            args.voxel_size_a,
            args.persistence_bin_steps,
            &manifest.run_id,
            &tensor_surface_summaries,
            &output_dir,
        )?;
    }

    // Build duplicate-cluster records (only for regions that absorbed peaks).
    for (ri, region) in regions.iter().enumerate() {
        if region.merged_raw_peak_ids.len() <= 1 {
            continue;
        }
        let suppressed: Vec<u32> = region
            .merged_raw_peak_ids
            .iter()
            .filter(|&&id| id != region.representative_raw_peak_id)
            .copied()
            .collect();
        duplicate_records.push(DuplicateClusterRecord {
            duplicate_cluster_id: ri as u32,
            kept_region_id: region.region_id,
            kept_centroid_a: region.aggregate_centroid_a,
            n_suppressed: suppressed.len() as u32,
            suppressed_raw_peak_ids: suppressed,
            centroid_spread_a: region.centroid_spread_a,
            residue_shell_overlap_with_rep: region.residue_shell_overlap_with_rep,
            merge_reasons: region.merge_reasons.clone(),
        });
    }

    // ─── Per-region computation: residue-shell plausibility, manifold,
    // KCC driver, refined temporal, stream balance, composite PRISM score ──
    let topology_residue_names: Vec<String> = topology
        .as_ref()
        .map(TopologyMin::residue_names_by_residue)
        .unwrap_or_default();

    let mut materialized_v2: Vec<serde_json::Value> = Vec::new();
    let mut density_only_candidates: Vec<serde_json::Value> = Vec::new();
    let mut spatiotemporal_partial_candidates: Vec<serde_json::Value> = Vec::new();
    let max_n_spikes_for_density_normalize: f64 = regions
        .iter()
        .map(|r| r.n_spikes as f64)
        .fold(0.0, f64::max)
        .max(1.0);

    // ── M1.2.23 §12 — load ghost features once ──────────────────────────────
    let ghost_features_state: GhostFeaturesState = match args.ghost_features.as_ref() {
        Some(p) => {
            let mut s = load_ghost_features(p)?;
            if s.exact_component_adapter_receipt.is_some() {
                let binding =
                    validate_exact_ghost_adapter_input(p, &manifest.run_id, &manifest.target, &s)?;
                s.exact_component_adapter_receipt.as_mut().unwrap()["consumer_input_binding"] =
                    binding;
            }
            eprintln!(
                "[ghost-features] {} → loaded={} spatial_mapped={} total_events={} status={}",
                p.display(),
                s.loaded,
                s.spatial_mapped,
                s.total_ghost_events,
                s.spatial_status_text
            );
            s
        }
        None => {
            eprintln!(
                "[ghost-features] not supplied → ghost_zstr_factor = 1.0 (neutral) for all regions"
            );
            GhostFeaturesState::default()
        }
    };
    let chem_dtsg_state = match args.chem_perturbed_dtsg_state_graph.as_ref() {
        Some(path) => {
            let state = load_chem_dtsg_state(path)?;
            eprintln!(
                "[chem-dtsg] {} -> loaded={} edges={}",
                path.display(),
                state.loaded,
                state
                    .graph
                    .as_ref()
                    .map(|graph| graph.edges.len())
                    .unwrap_or(0)
            );
            state
        }
        None => {
            eprintln!("[chem-dtsg] not supplied -> factor = 1.0 (neutral) for all regions");
            ChemDtsgState::default()
        }
    };

    // Pre-compute density-only ranking (for ablation report) — by raw n_spikes.
    let mut density_only_rank_by_region: std::collections::HashMap<u32, u32> =
        std::collections::HashMap::new();
    {
        let mut ord: Vec<(u32, u64)> = regions.iter().map(|r| (r.region_id, r.n_spikes)).collect();
        ord.sort_by(count_desc_identity_asc);
        for (i, (rid, _)) in ord.iter().enumerate() {
            density_only_rank_by_region.insert(*rid, i as u32);
        }
    }

    struct ScoredRegion {
        region_id: u32,
        old_density_rank: u32,
        lexicographic_rank_key: LexicographicSiteRankKey,
        score_components: PrismScoreComponents,
        manifest_payload: serde_json::Value,
        materialization_level: &'static str,
        materialization_status: &'static str,
        promoted: bool,
        lining_residues: Vec<i32>,
        pocket_morphology_rank: f64,
        thermodynamic_physiology_rank: f64,
        scale_normalized_dynamic_support: f64,
        site_family_preservation_rank: f64,
    }
    let mut scored: Vec<ScoredRegion> = Vec::new();

    let n_residues_topo: usize = topology.as_ref().map(|t| t.n_residues).unwrap_or(306);
    let terminal_threshold: i32 = 5; // residues 0..5 or last-5 are "terminal"

    for region in regions.iter() {
        let region_id = region.region_id;
        // ── stream support / balance ────────────────────────────────────────
        let supporting_streams: Vec<u32> = region
            .per_stream_spikes
            .iter()
            .filter_map(|(s, c)| {
                if *c >= min_spikes_per_stream {
                    Some(*s)
                } else {
                    None
                }
            })
            .collect();
        let mut supporting_sorted = supporting_streams.clone();
        supporting_sorted.sort_unstable();
        let mut per_stream_counts_json = serde_json::Map::new();
        for (s, c) in region.per_stream_spikes.iter() {
            per_stream_counts_json.insert(s.to_string(), serde_json::Value::from(*c));
        }
        let stream_counts_vec: Vec<u64> = region.per_stream_spikes.values().copied().collect();
        let gini = gini_coefficient(&stream_counts_vec);
        let stream_balance_score = (1.0 - gini).max(0.0);
        let stream_balance = StreamBalance {
            n_streams_supporting: supporting_sorted.len() as u32,
            per_stream_spike_counts: per_stream_counts_json.clone(),
            gini_coefficient: gini,
            stream_balance_score,
            per_stream_support_balance_status: if stream_counts_vec.is_empty() {
                "no_streams"
            } else if gini < 0.1 {
                "balanced"
            } else if gini < 0.3 {
                "moderately_balanced"
            } else {
                "imbalanced"
            },
        };

        // ── refined temporal support ────────────────────────────────────────
        // Reduce in chronological bin order so floating-point variance and
        // entropy do not inherit randomized HashMap iteration order.
        let mut ordered_bins: Vec<_> = region.bin_counts.iter().collect();
        ordered_bins.sort_unstable_by_key(|(step, _)| **step);
        let counts: Vec<u64> = ordered_bins.into_iter().map(|(_, count)| *count).collect();
        let total_bins: i32 = if region.timestep_max > region.timestep_min {
            ((region.timestep_max - region.timestep_min) / args.persistence_bin_steps.max(1)) + 1
        } else {
            1
        };
        let persistence_fraction = if total_bins > 0 {
            region.persistence_bin_set.len() as f64 / total_bins as f64
        } else {
            0.0
        };
        let mean_count = if !counts.is_empty() {
            counts.iter().sum::<u64>() as f64 / counts.len() as f64
        } else {
            0.0
        };
        let var = if counts.len() > 1 {
            let m = mean_count;
            counts.iter().map(|c| (*c as f64 - m).powi(2)).sum::<f64>() / counts.len() as f64
        } else {
            0.0
        };
        let burstiness_score = if mean_count > 0.0 {
            (var.sqrt() / mean_count).min(2.0) / 2.0
        } else {
            0.0
        };
        let temporal_entropy = if !counts.is_empty() {
            let total: f64 = counts.iter().map(|&c| c as f64).sum();
            if total > 0.0 {
                let mut ent = 0.0;
                for c in counts.iter() {
                    let p = (*c as f64) / total;
                    if p > 0.0 {
                        ent -= p * p.ln();
                    }
                }
                ent
            } else {
                0.0
            }
        } else {
            0.0
        };
        let temporal_status: &'static str = if region.bin_counts.is_empty() {
            "coarse_non_discriminative"
        } else {
            "available"
        };
        let refined_temporal = RefinedTemporalSupport {
            persistence_bin_size_steps: args.persistence_bin_steps,
            persistence_bins_observed: region.persistence_bin_set.len() as u32,
            persistence_bins_total_possible: total_bins.max(0) as u32,
            persistence_fraction,
            burstiness_score,
            temporal_entropy,
            temporal_support_status: temporal_status,
        };
        let temporal_persistence_factor: f64 = if temporal_status == "available" {
            // Reward high persistence, penalize bursty.
            (persistence_fraction.powf(0.5)) * (1.0 - 0.5 * burstiness_score).max(0.0)
        } else {
            1.0
        };

        // ── lining residues / driver residues / residue shell features ─────
        let mut lining_pairs: Vec<(i32, u32)> = region
            .residue_support
            .iter()
            .map(|(r, c)| (*r, *c))
            .collect();
        lining_pairs.sort_by(count_desc_identity_asc);
        let lining_residues: Vec<i32> = lining_pairs.iter().take(12).map(|x| x.0).collect();
        let representative_residues: Vec<i32> = lining_residues.iter().take(5).cloned().collect();

        // Tail penalty + terminal fraction.
        let mut n_terminal: u32 = 0;
        for r in lining_residues.iter() {
            let r = *r;
            if r >= 0
                && (r < terminal_threshold || r >= (n_residues_topo as i32 - terminal_threshold))
            {
                n_terminal += 1;
            }
        }
        let terminal_residue_fraction = if !lining_residues.is_empty() {
            n_terminal as f64 / lining_residues.len() as f64
        } else {
            0.0
        };
        let tail_penalty = terminal_residue_fraction;
        let tail_penalty_factor = (1.0 - tail_penalty).max(0.05);

        // Residue diversity = unique residue NAMES / shell_size.
        let mut residue_diversity_score = 1.0_f64;
        let mut aromatic_count: u32 = 0;
        let mut hydrophobic_count: u32 = 0;
        let mut polar_count: u32 = 0;
        if !topology_residue_names.is_empty() && !lining_residues.is_empty() {
            let mut name_set: std::collections::HashSet<&str> = std::collections::HashSet::new();
            for r in lining_residues.iter() {
                if (*r as usize) < topology_residue_names.len() {
                    let name = topology_residue_names[*r as usize].as_str();
                    name_set.insert(name);
                    if AROMATIC.contains(&name) {
                        aromatic_count += 1;
                    }
                    if HYDROPHOBIC.contains(&name) {
                        hydrophobic_count += 1;
                    }
                    if POLAR.contains(&name) {
                        polar_count += 1;
                    }
                }
            }
            residue_diversity_score =
                (name_set.len() as f64 / lining_residues.len() as f64).clamp(0.0, 1.0);
        }
        // Compactness: mean pairwise CA distance among lining residues.
        let mut compactness_a: Option<f64> = None;
        let mut enclosure_proxy: Option<f64> = None;
        if !residue_centers.is_empty() && lining_residues.len() >= 2 {
            let mut sum_d = 0.0_f64;
            let mut n = 0_u32;
            for i in 0..lining_residues.len() {
                for j in (i + 1)..lining_residues.len() {
                    let ri = lining_residues[i] as usize;
                    let rj = lining_residues[j] as usize;
                    if ri < residue_centers.len() && rj < residue_centers.len() {
                        if let (Some(a), Some(b)) = (residue_centers[ri], residue_centers[rj]) {
                            sum_d += dist3(&a, &b) as f64;
                            n += 1;
                        }
                    }
                }
            }
            if n > 0 {
                let mean_d = sum_d / n as f64;
                compactness_a = Some(mean_d);
                // Pocket-like = compact (mean d <~12 Å) AND multi-residue.
                enclosure_proxy = Some(
                    (1.0 / (1.0 + (mean_d - 8.0).max(0.0) / 4.0))
                        * (lining_residues.len() as f64 / 12.0).min(1.0),
                );
            }
        }
        // Catalytic-like proxy = aromatic+polar mix without ligand-truth use.
        let catalytic_like = if !lining_residues.is_empty() {
            let mix = (aromatic_count.min(polar_count)) as f64 / lining_residues.len() as f64;
            (mix * (1.0 - tail_penalty)).min(1.0)
        } else {
            0.0
        };
        let mut shell_reasoning: Vec<String> = Vec::new();
        if tail_penalty > 0.5 {
            shell_reasoning.push(format!(
                "dominated_by_terminal_residues ({:.0}%)",
                tail_penalty * 100.0
            ));
        }
        if residue_diversity_score < 0.5 {
            shell_reasoning.push("low_residue_name_diversity".to_string());
        }
        if let Some(c) = compactness_a {
            if c > 14.0 {
                shell_reasoning.push(format!("residue_shell_loose_mean_d={:.1}A", c));
            }
        }
        if catalytic_like > 0.15 {
            shell_reasoning.push("aromatic_polar_mix_present".to_string());
        }
        let residue_shell = ResidueShellPlausibility {
            residue_shell_size: lining_residues.len() as u32,
            residue_diversity_score,
            compactness_a,
            enclosure_proxy,
            tail_penalty,
            terminal_residue_fraction,
            backbone_exposure_proxy: None,
            aromatic_count,
            hydrophobic_count,
            polar_count,
            catalytic_like_pocket_support: catalytic_like,
            residue_shell_reasoning: shell_reasoning,
        };
        let residue_shell_plausibility_factor: f64 = {
            let div = residue_diversity_score.max(0.1);
            let comp = enclosure_proxy.unwrap_or(0.5);
            (div * comp).clamp(0.05, 1.5)
        };

        // ── KCC driver block ────────────────────────────────────────────────
        let mut driver_pairs: Vec<(i32, u64)> = lining_residues
            .iter()
            .filter_map(|r| {
                driver_active_causal_sum
                    .get(&(*r as usize))
                    .map(|ac| (*r, *ac))
            })
            .collect();
        driver_pairs.sort_by(count_desc_identity_asc);
        let driver_residues: Vec<i32> = driver_pairs.iter().take(8).map(|x| x.0).collect();
        let driver_residue_count = driver_pairs.len() as u32;
        let driver_residue_fraction_of_shell = if !lining_residues.is_empty() {
            driver_residue_count as f64 / lining_residues.len() as f64
        } else {
            0.0
        };
        let active_causal_support_sum: u64 = driver_pairs.iter().map(|(_, ac)| *ac).sum();
        let causal_driver_centroid: Option<[f32; 3]> =
            if !residue_centers.is_empty() && !driver_pairs.is_empty() {
                let mut sum = [0.0f64; 3];
                let mut wsum = 0.0f64;
                for (rid, ac) in driver_pairs.iter() {
                    let r = *rid as usize;
                    if r < residue_centers.len() {
                        if let Some(c) = residue_centers[r] {
                            let w = *ac as f64;
                            sum[0] += c[0] as f64 * w;
                            sum[1] += c[1] as f64 * w;
                            sum[2] += c[2] as f64 * w;
                            wsum += w;
                        }
                    }
                }
                if wsum > 0.0 {
                    Some(vec3_div(sum, wsum))
                } else {
                    None
                }
            } else {
                None
            };
        let kcc_field_completeness: &'static str = if n_total_kcc_streams >= 4 {
            "partial_strong"
        } else if n_total_kcc_streams > 0 {
            "partial_weak"
        } else {
            "missing"
        };
        let kcc_driver_factor: f64 = if n_total_kcc_streams == 0 {
            1.0 // neutral
        } else {
            (1.0 + 0.5 * driver_residue_fraction_of_shell).clamp(0.7, 1.6)
        };
        let kcc_driver_score = if n_total_kcc_streams > 0 {
            driver_residue_fraction_of_shell * (active_causal_support_sum as f64).ln().max(0.0)
        } else {
            0.0
        };
        let mut kcc_reasoning: Vec<String> = Vec::new();
        if n_total_kcc_streams == 0 {
            kcc_reasoning.push("kcc_unavailable_neutral_factor".to_string());
        }
        if driver_residue_fraction_of_shell > 0.5 {
            kcc_reasoning.push(format!(
                "driver_dominant_shell_fraction={:.2}",
                driver_residue_fraction_of_shell
            ));
        }
        let kcc_driver_block = KccDriverBlock {
            driver_residue_count,
            driver_residue_fraction_of_shell,
            driver_centroid_present: causal_driver_centroid.is_some(),
            driver_centroid_distance_to_whole_site_a: None, // computed below
            active_causal_support_sum,
            kcc_field_completeness,
            kcc_driver_score,
            kcc_driver_reasoning: kcc_reasoning,
        };

        // ── Centroid manifold ───────────────────────────────────────────────
        let whole_site = if region.intensity_sum > 0.0 {
            Some(vec3_div(
                [region.pos_sum_x, region.pos_sum_y, region.pos_sum_z],
                region.intensity_sum,
            ))
        } else {
            None
        };
        let hot_phase = if region.n_hot >= 50 {
            Some(vec3_div(region.hot_pos_sum, region.n_hot as f64))
        } else {
            None
        };
        let cold_phase = if region.n_cold >= 50 {
            Some(vec3_div(region.cold_pos_sum, region.n_cold as f64))
        } else {
            None
        };
        let lining_mass: Option<[f32; 3]> = if !residue_centers.is_empty() {
            let mut sum = [0.0f64; 3];
            let mut wsum = 0.0f64;
            for (rid, count) in lining_pairs.iter().take(20) {
                let r = *rid as usize;
                if r < residue_centers.len() {
                    if let Some(c) = residue_centers[r] {
                        let w = *count as f64;
                        sum[0] += c[0] as f64 * w;
                        sum[1] += c[1] as f64 * w;
                        sum[2] += c[2] as f64 * w;
                        wsum += w;
                    }
                }
            }
            if wsum > 0.0 {
                Some(vec3_div(sum, wsum))
            } else {
                None
            }
        } else {
            None
        };
        let causal_driver = causal_driver_centroid;
        let ligand_adjacent: Option<[f32; 3]> = ground_truth.as_ref().and_then(|gt| {
            if gt.valid_for_dcc_validation {
                gt.ligand_centroid
            } else {
                None
            }
        });
        let manifold = CentroidManifold {
            whole_site,
            lining_mass,
            causal_driver,
            hot_phase,
            cold_phase,
            return_phase: region.thermal.centroid(4),
            ligand_adjacent,
        };

        // Manifold consistency.
        let d_whole_lining = whole_site.zip(lining_mass).map(|(a, b)| dist3(&a, &b));
        let d_whole_driver = whole_site.zip(causal_driver).map(|(a, b)| dist3(&a, &b));
        let d_hot_cold = hot_phase.zip(cold_phase).map(|(a, b)| dist3(&a, &b));
        let mut manifold_dispersion: Option<f32> = None;
        let n_views_present = [
            whole_site,
            lining_mass,
            causal_driver,
            hot_phase,
            cold_phase,
        ]
        .iter()
        .filter(|v| v.is_some())
        .count() as u32;
        let views_vec: Vec<[f32; 3]> = [
            whole_site,
            lining_mass,
            causal_driver,
            hot_phase,
            cold_phase,
        ]
        .iter()
        .flatten()
        .copied()
        .collect();
        if views_vec.len() >= 2 {
            let mut max_d: f32 = 0.0;
            for i in 0..views_vec.len() {
                for j in (i + 1)..views_vec.len() {
                    let d = dist3(&views_vec[i], &views_vec[j]);
                    if d > max_d {
                        max_d = d;
                    }
                }
            }
            manifold_dispersion = Some(max_d);
        }
        // Manifold consistency: exp(-dispersion / 4Å).
        let manifold_consistency_score: f64 = if let Some(d) = manifold_dispersion {
            ((-(d as f64) / 4.0).exp()).clamp(0.0, 1.0)
        } else {
            0.5
        };
        let centroid_manifold_consistency_factor: f64 =
            manifold_consistency_score * (n_views_present as f64 / 5.0).min(1.0).max(0.2);
        let manifold_block = ManifoldConsistency {
            d_whole_lining_a: d_whole_lining,
            d_whole_driver_a: d_whole_driver,
            d_hot_cold_a: d_hot_cold,
            manifold_dispersion_a: manifold_dispersion,
            manifold_consistency_score,
            n_views_present,
        };
        let return_geometry = return_inclusive_geometry(&manifold);
        // Patch driver-distance into KccDriverBlock (we already constructed it above).
        let kcc_driver_block_with_dist = KccDriverBlock {
            driver_centroid_distance_to_whole_site_a: d_whole_driver,
            ..kcc_driver_block
        };

        // ── Stream support struct (legacy shape kept for back-compat) ───────
        let stream_support = StreamSupport {
            n_streams_supporting: supporting_sorted.len() as u32,
            supporting_streams: supporting_sorted.clone(),
            per_stream_spike_counts: per_stream_counts_json.clone(),
            per_stream_voxel_counts: {
                let mut m = serde_json::Map::new();
                for (s, vox) in region.per_stream_voxels.iter() {
                    m.insert(s.to_string(), serde_json::Value::from(vox.len()));
                }
                m
            },
        };

        // ── Phase support / Rayleigh r ──────────────────────────────────────
        let n = region.n_spikes as f64;
        let r_stat = if n > 0.0 {
            (region.phase_cos_sum.powi(2) + region.phase_sin_sum.powi(2)).sqrt() / n
        } else {
            0.0
        };
        let mean_phase = if r_stat > 1e-3 {
            Some(region.phase_sin_sum.atan2(region.phase_cos_sum))
        } else {
            None
        };
        let phase_support = PhaseSupport {
            phase_bits_histogram: region.phase_hist,
            rayleigh_r_stat: r_stat,
            mean_phase_radians: mean_phase,
            n_hot: region.n_hot,
            n_cold: region.n_cold,
            thermal: region.thermal.clone(),
        };

        // ── Field completeness factor ───────────────────────────────────────
        let mut sources_present: f64 = 0.0;
        let mut sources_total: f64 = 0.0;
        for (status, weight) in [
            ("present", 1.0), // spikes
            ("present", 1.0), // phase_bits
            (signal_grid_surface_status, 1.0),
            (kcc_surface_status, 1.0),
            (warp_matrix_surface_status, 0.6),
            (forces_surface_status, 0.6),
            (asc_vectors_surface_status, 0.6),
            (adaptive_dt_surface_status, 0.4),
            (bocpd_log_surface_status, 0.4),
            (protocol_state_surface_status, 0.4),
            (spike_transition_surface_status, 0.4),
        ] {
            sources_total += weight;
            sources_present += match status {
                "present" | "materialized_parsed" => 1.0 * weight,
                "partial" => 0.5 * weight,
                _ => 0.0,
            };
        }
        let field_completeness_factor: f64 = (sources_present / sources_total).clamp(0.5, 1.0);

        // ── M1.2.23 §12 — Ghost/ZSTR candidate-features factor ──────────────
        // Set per-region by main()-level ghost_zstr_summary; default neutral.
        // (Per-region linking is explicit when a spatial map exists.)
        let ghost_zstr_summary = ghost_zstr_summary_compute(&ghost_features_state, region);
        let ghost_zstr_factor = ghost_zstr_summary.factor;
        let ghost_zstr_status = ghost_zstr_summary.status;
        let chem_dtsg_summary = chem_dtsg_summary_compute(&chem_dtsg_state, region);
        let chem_dtsg_factor = chem_dtsg_summary.factor;
        let chem_dtsg_status = chem_dtsg_summary.status;
        let field_completeness = FieldCompleteness {
            spikes: "present",
            phase_bits: "present",
            warp_matrix: warp_matrix_surface_status,
            asc_vectors: asc_vectors_surface_status,
            forces: forces_surface_status,
            signal_grid: signal_grid_surface_status,
            kcc: kcc_surface_status,
        };

        // ── Density score (log-normalized) ──────────────────────────────────
        let density_score =
            (region.n_spikes as f64 + 1.0).ln() / (max_n_spikes_for_density_normalize + 1.0).ln();
        let stream_opportunity_spikes =
            (manifest.total_spikes_md as f64 / manifest.stream_count.max(1) as f64).max(1.0);
        let stream_coverage = supporting_sorted.len() as f64 / manifest.stream_count.max(1) as f64;
        let scale_normalized_dynamic_support = (((region.n_spikes as f64 + 1.0).ln()
            / (stream_opportunity_spikes + 1.0).ln())
        .clamp(0.0, 1.25)
            * stream_coverage.sqrt()
            * temporal_persistence_factor.clamp(0.05, 1.0))
        .clamp(0.0, 1.25);

        // ── PRISMOS_GEOMETRIC_POCKET_MORPHOLOGY_V1 ──────────────────────────
        let shell_size = lining_residues.len() as f64;
        let shell_fraction = (shell_size / 12.0).clamp(0.0, 1.0);
        let mean_lining_distance_a = compactness_a;
        let compactness_score = compactness_a
            .map(|d| (1.0 / (1.0 + (d - 8.0).max(0.0) / 4.0)).clamp(0.0, 1.0))
            .unwrap_or(0.5);
        let enclosure_wall_blocking_score = enclosure_proxy.unwrap_or(0.5).clamp(0.0, 1.0);
        let legacy_region_sphere_volume_a3 = (4.0_f64 / 3.0_f64)
            * std::f64::consts::PI
            * (region.region_radius_a.max(0.1) as f64).powi(3);
        let shell_heavy_atom_positions =
            residue_shell_heavy_atom_positions(topology.as_ref(), &lining_residues);
        let geometry_centroid = whole_site
            .or(lining_mass)
            .or(causal_driver)
            .unwrap_or([0.0, 0.0, 0.0]);
        let coordinate_enclosure_fraction =
            coordinate_enclosure_fraction(&shell_heavy_atom_positions, geometry_centroid);
        let coordinate_probe =
            coordinate_empty_probe_volume_a3(&shell_heavy_atom_positions, geometry_centroid);
        let coordinate_empty_probe_volume_a3 = coordinate_probe.map(|x| x.0);
        let coordinate_empty_probe_fraction = coordinate_probe.map(|x| x.1);
        let coordinate_probe_empty_points = coordinate_probe.map(|x| x.2);
        let coordinate_probe_total_points = coordinate_probe.map(|x| x.3);
        // A shell-empty-space count is not by itself pocket geometry. Keep it
        // as an auxiliary diagnostic, but do not promote it into `volume_a3`
        // unless a connected-component pocket gate is available.
        let coordinate_geometry_status = match coordinate_empty_probe_volume_a3 {
            Some(v) if v >= 150.0 => {
                Some("coordinate_shell_empty_space_requires_component_validation")
            }
            Some(v) if v >= 75.0 => Some("coordinate_shell_empty_space_borderline_recheck"),
            Some(_) => Some("coordinate_shell_empty_space_weak_or_surface_like"),
            _ => None,
        };
        let volume_a3 = legacy_region_sphere_volume_a3;
        let volume_source = "legacy_region_radius_sphere_component_validation_required";
        let residue_shell_atom_count =
            residue_shell_atom_count(topology.as_ref(), &lining_residues);
        let shell_volume_normalized_a3 = if residue_shell_atom_count > 0 {
            volume_a3 / residue_shell_atom_count as f64
        } else {
            volume_a3
        };
        let hydrophobic_lining_fraction = if shell_size > 0.0 {
            hydrophobic_count as f64 / shell_size
        } else {
            0.0
        };
        let aromatic_lining_fraction = if shell_size > 0.0 {
            aromatic_count as f64 / shell_size
        } else {
            0.0
        };
        let polar_lining_fraction = if shell_size > 0.0 {
            polar_count as f64 / shell_size
        } else {
            0.0
        };
        let cavity_support_score = (enclosure_wall_blocking_score
            * compactness_score
            * shell_fraction
            * (1.0 - terminal_residue_fraction).clamp(0.0, 1.0))
        .clamp(0.0, 1.0);
        let groove_support_score = ((1.0 - compactness_score) * shell_fraction
            + (region.centroid_spread_a as f64 / 12.0).clamp(0.0, 1.0) * 0.25)
            .clamp(0.0, 1.0);
        let sphericity_proxy = (compactness_score
            * (1.0 - (region.centroid_spread_a as f64 / 18.0).clamp(0.0, 0.8)))
        .clamp(0.0, 1.0);
        let mouth_accessibility_proxy = (1.0 - 0.5 * enclosure_wall_blocking_score
            + 0.25 * groove_support_score)
            .clamp(0.0, 1.0);
        let pocket_morphology_rank = 1000.0
            * (0.24 * enclosure_wall_blocking_score
                + 0.18 * cavity_support_score
                + 0.12 * groove_support_score
                + 0.16 * compactness_score
                + 0.10 * sphericity_proxy
                + 0.08 * (hydrophobic_lining_fraction + aromatic_lining_fraction).clamp(0.0, 1.0)
                + 0.08 * catalytic_like
                + 0.04 * shell_fraction)
                .clamp(0.0, 1.0);
        let mut morphology_reasoning = Vec::new();
        if enclosure_wall_blocking_score >= 0.75 {
            morphology_reasoning.push("enclosed_residue_shell".to_string());
        }
        if cavity_support_score >= 0.6 {
            morphology_reasoning.push("cavity_support_from_compact_multiresidue_shell".to_string());
        }
        if groove_support_score >= 0.45 {
            morphology_reasoning.push("groove_support_from_extended_shell".to_string());
        }
        if catalytic_like > 0.15 {
            morphology_reasoning.push("aromatic_polar_lining_mix".to_string());
        }
        if let Some(status) = coordinate_geometry_status {
            morphology_reasoning.push(status.to_string());
        }
        if coordinate_empty_probe_volume_a3.unwrap_or(0.0) >= 150.0 {
            morphology_reasoning.push(
                "coordinate_probe_volume_not_promoted_without_v2_component_geometry".to_string(),
            );
        }
        let pocket_morphology = GeometricPocketMorphology {
            surface_id: "PRISMOS_GEOMETRIC_POCKET_MORPHOLOGY_V1",
            volume_a3,
            legacy_region_sphere_volume_a3: Some(legacy_region_sphere_volume_a3),
            coordinate_empty_probe_volume_a3,
            coordinate_empty_probe_fraction,
            coordinate_enclosure_fraction,
            coordinate_probe_empty_points,
            coordinate_probe_total_points,
            coordinate_geometry_status,
            volume_source,
            shell_volume_normalized_a3,
            enclosure_wall_blocking_score,
            cavity_support_score,
            groove_support_score,
            compactness_score,
            sphericity_proxy,
            mouth_accessibility_proxy,
            hydrophobic_lining_fraction,
            aromatic_lining_fraction,
            polar_lining_fraction,
            catalytic_like_lining_support: catalytic_like,
            residue_shell_atom_count,
            residue_shell_size: lining_residues.len() as u32,
            mean_lining_distance_a,
            morphology_rank: pocket_morphology_rank,
            morphology_reasoning,
        };

        // ── PRISMOS_BINDING_SITE_PHYSIOLOGY_V1 ───────────────────────────────
        let phase_total = (region.n_hot + region.n_cold).max(1) as f64;
        let hot_phase_fraction = region.n_hot as f64 / phase_total;
        let cold_phase_fraction = region.n_cold as f64 / phase_total;
        let hysteresis_asymmetry = (hot_phase_fraction - cold_phase_fraction).clamp(-1.0, 1.0);
        let relative_asymmetry = hysteresis_asymmetry.abs();
        let cryptic_spike_max_bin_count = region.bin_counts.values().copied().max().unwrap_or(0);
        let cryptic_inter_spike_bin_gap_mean = mean_inter_bin_gap(&region.bin_counts);
        let signal_preservation_factor = (manifold_consistency_score
            * stream_balance_score.clamp(0.0, 1.0)
            * field_completeness_factor)
            .clamp(0.0, 1.0);
        let rank_g = (pocket_morphology_rank / 1000.0).clamp(0.0, 1.0);
        let rank_t = (temporal_persistence_factor.clamp(0.0, 1.0)
            * (1.0 - 0.25 * burstiness_score).clamp(0.0, 1.0))
        .clamp(0.0, 1.0);
        let rank_c = signal_preservation_factor;
        let rank_k = ((kcc_driver_factor / 1.6).clamp(0.0, 1.0)
            * (1.0 + kcc_driver_score.max(0.0).ln_1p() / 12.0).clamp(1.0, 1.25))
        .clamp(0.0, 1.0);
        let rank_l = residue_shell_plausibility_factor.clamp(0.0, 1.0);
        let gtck_rank =
            (0.25 * rank_g + 0.20 * rank_t + 0.20 * rank_c + 0.20 * rank_k + 0.15 * rank_l)
                .clamp(0.0, 1.0);
        let tide_te_out_proxy = density_score.max(0.0)
            * scale_normalized_dynamic_support
            * stream_balance_score.max(0.0)
            * kcc_driver_factor.max(1.0);
        let tide_te_in_proxy = rank_l * rank_g * stream_balance_score.max(0.0);
        let tide_delta_hc_proxy = r_stat.max(0.0) * rank_c;
        let ccns_tau_proxy = if temporal_entropy > 0.0 && !counts.is_empty() {
            (temporal_entropy / (counts.len() as f64).ln_1p()).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let thermodynamic_physiology_rank = (1000.0
            * (0.32 * gtck_rank
                + 0.22 * signal_preservation_factor
                + 0.16 * scale_normalized_dynamic_support.clamp(0.0, 1.0)
                + 0.12 * (1.0 - relative_asymmetry).clamp(0.0, 1.0)
                + 0.10 * ccns_tau_proxy
                + 0.08 * r_stat.clamp(0.0, 1.0)))
        .clamp(0.0, 1000.0);
        let mut physiology_reasoning = Vec::new();
        if signal_preservation_factor >= 0.2 {
            physiology_reasoning.push("signal_preservation_present".to_string());
        }
        if scale_normalized_dynamic_support >= 0.5 {
            physiology_reasoning.push("scale_normalized_dynamic_support_present".to_string());
        }
        if kcc_driver_factor > 1.0 {
            physiology_reasoning.push("kcc_driver_support_present".to_string());
        }
        let binding_site_physiology = BindingSitePhysiology {
            surface_id: "PRISMOS_BINDING_SITE_PHYSIOLOGY_V1",
            signal_preservation_factor,
            scale_normalized_dynamic_support,
            prism_therm_rank_g_geometry: rank_g,
            prism_therm_rank_t_temporal: rank_t,
            prism_therm_rank_c_coupling: rank_c,
            prism_therm_rank_k_kcc: rank_k,
            prism_therm_rank_l_localization: rank_l,
            gtck_rank,
            tide_te_out_proxy,
            tide_te_in_proxy,
            tide_delta_hc_proxy,
            hysteresis_asymmetry,
            relative_asymmetry,
            ccns_tau_proxy,
            cold_phase_fraction,
            hot_phase_fraction,
            cryptic_spike_bin_count: region.bin_counts.len() as u32,
            cryptic_spike_max_bin_count,
            cryptic_inter_spike_bin_gap_mean,
            physiology_rank: thermodynamic_physiology_rank,
            physiology_reasoning,
        };
        // ── Duplicate region factor: representative regions have factor 1.0.
        let duplicate_region_factor: f64 = 1.0;

        // ── Composite PRISM score ───────────────────────────────────────────
        let final_prism_score = density_score
            * stream_balance_score.clamp(0.05, 1.0)
            * temporal_persistence_factor.clamp(0.05, 1.0)
            * residue_shell_plausibility_factor
            * centroid_manifold_consistency_factor.clamp(0.05, 1.0)
            * kcc_driver_factor
            * scale_normalized_dynamic_support.clamp(0.05, 1.25)
            * field_completeness_factor
            * duplicate_region_factor
            * tail_penalty_factor
            * ghost_zstr_factor
            * chem_dtsg_factor;

        let mut score_explanation: Vec<String> = Vec::new();
        score_explanation.push(format!("density_score={:.3}", density_score));
        score_explanation.push(format!(
            "scale_normalized_dynamic_support={:.3}",
            scale_normalized_dynamic_support
        ));
        score_explanation.push(format!(
            "stream_balance={:.3} (gini={:.2})",
            stream_balance_score, gini
        ));
        score_explanation.push(format!(
            "temporal_persistence={:.3} ({})",
            temporal_persistence_factor, temporal_status
        ));
        score_explanation.push(format!(
            "residue_shell={:.3} (div={:.2}, encl={:?})",
            residue_shell_plausibility_factor, residue_diversity_score, enclosure_proxy
        ));
        score_explanation.push(format!(
            "manifold={:.3} ({} views)",
            centroid_manifold_consistency_factor, n_views_present
        ));
        score_explanation.push(format!(
            "kcc_driver={:.3} (drv_frac={:.2})",
            kcc_driver_factor, driver_residue_fraction_of_shell
        ));
        score_explanation.push(format!(
            "signal_preservation={:.3}",
            signal_preservation_factor
        ));
        score_explanation.push(format!(
            "pocket_morphology_rank={:.1}",
            pocket_morphology_rank
        ));
        score_explanation.push(format!(
            "thermodynamic_physiology_rank={:.1}",
            thermodynamic_physiology_rank
        ));
        score_explanation.push(format!(
            "field_completeness={:.3}",
            field_completeness_factor
        ));
        score_explanation.push(format!(
            "tail_penalty_factor={:.3} (term_frac={:.2})",
            tail_penalty_factor, terminal_residue_fraction
        ));
        score_explanation.push(format!(
            "ghost_zstr={:.3} ({})",
            ghost_zstr_factor, ghost_zstr_status
        ));
        score_explanation.push(format!(
            "chem_dtsg={:.3} ({})",
            chem_dtsg_factor, chem_dtsg_status
        ));
        score_explanation.push(format!("FINAL={:.6}", final_prism_score));

        let mut score_components = PrismScoreComponents {
            density_score,
            scale_normalized_dynamic_support_factor: scale_normalized_dynamic_support,
            stream_balance_factor: stream_balance_score.clamp(0.05, 1.0),
            temporal_persistence_factor: temporal_persistence_factor.clamp(0.05, 1.0),
            residue_shell_plausibility_factor,
            centroid_manifold_consistency_factor: centroid_manifold_consistency_factor
                .clamp(0.05, 1.0),
            kcc_driver_factor,
            signal_preservation_factor,
            pocket_morphology_rank,
            thermodynamic_physiology_rank,
            site_family_preservation_rank: 0.0,
            field_completeness_factor,
            duplicate_region_factor,
            tail_penalty_factor,
            ghost_zstr_factor,
            ghost_zstr_status,
            ghost_zstr_reasoning: ghost_zstr_summary.reasoning,
            ghost_zstr_links: ghost_zstr_summary.links,
            chem_dtsg_factor,
            chem_dtsg_status,
            chem_dtsg_links: chem_dtsg_summary.links,
            final_prism_score,
            score_explanation,
        };
        // Mutable by construction so spatial-link materializers can extend
        // ghost_zstr_reasoning when a canonical region link is present.
        let _ = &mut score_components;

        // ── Promotion criteria (tightened per directive Part 8) ─────────────
        let mut blocking: Vec<&'static str> = Vec::new();
        if whole_site.is_none() {
            blocking.push("missing_whole_site_centroid");
        }
        if (supporting_sorted.len() as u32) < min_supporting_streams {
            blocking.push("insufficient_stream_support");
        }
        if region.n_spikes < 1000 {
            blocking.push("insufficient_spike_support");
        }
        if region.chunk_set.is_empty() {
            blocking.push("no_temporal_support");
        }
        // At least one non-density PRISM evidence dimension active:
        let kcc_active = n_total_kcc_streams > 0 && driver_residue_count > 0;
        let manifold_active = n_views_present >= 2 && manifold_consistency_score > 0.4;
        let temporal_active = temporal_status == "available" && persistence_fraction > 0.3;
        let shell_active =
            !lining_residues.is_empty() && tail_penalty < 0.8 && residue_diversity_score > 0.4;
        let signal_grid_active = signal_grid_present_any;
        let n_active_dims = [
            kcc_active,
            manifold_active,
            temporal_active,
            shell_active,
            signal_grid_active,
        ]
        .iter()
        .filter(|x| **x)
        .count();
        if n_active_dims == 0 {
            blocking.push("no_non_density_evidence_dimension_active");
        }

        let materialization_level: &'static str = if blocking.is_empty() {
            if gt_status == "available" {
                "materialized_dynamic_site_with_validation"
            } else {
                "materialized_dynamic_site"
            }
        } else if !blocking.contains(&"missing_whole_site_centroid")
            && !blocking.contains(&"no_temporal_support")
        {
            "candidate_spatiotemporal_partial"
        } else {
            "candidate_density_only"
        };
        let materialization_status: &'static str = if blocking.is_empty() {
            "materialized"
        } else {
            "non_materialized_candidate"
        };
        let promoted = blocking.is_empty();
        let native_kcc_streams = region_native_kcc(
            region.region_id,
            &region.per_stream_residue_support,
            &kcc_per_stream,
        );
        let selected_native_kcc = native_kcc_streams
            .iter()
            .filter_map(|stream| {
                stream
                    .native_residue_confidence
                    .map(|score| (stream, score))
            })
            .max_by(|(a, av), (b, bv)| {
                av.total_cmp(bv).then_with(|| b.stream_id.cmp(&a.stream_id))
            });
        let native_residue_kcc_confidence = selected_native_kcc.map(|(_, score)| score as f64);
        let confidence_branches: Vec<_> = native_kcc_streams.iter().flat_map(|stream| {
            stream.candidates.iter().map(move |candidate| {
                let persistent = candidate.motion_efficiency.filter(|_| candidate.observation_count > 0)
                    .map(|m| m * (candidate.active_causal as f32 / candidate.observation_count.max(1) as f32));
                let transient = candidate.burst_motion.zip(candidate.local_cov)
                    .map(|(b,c)| b.max(0.0).min(5.0) / 5.0 * c.max(0.0).min(1.0));
                serde_json::json!({"stream_id": stream.stream_id, "residue_index": candidate.residue_index,
                    "persistent_contribution": persistent, "transient_contribution": transient,
                    "support_multiplier": 1.0_f32 + 0.1_f32 * candidate.support_fraction,
                    "selected_branch": persistent.zip(transient).map(|(p,t)| if p >= t { "persistent" } else { "transient" }),
                    "native_residue_confidence": candidate.native_residue_confidence})
            })
        }).collect();
        let kcc_observation_availability = serde_json::json!({
            "legacy_kcc_field_completeness_semantics": "stream_count_category_only; >=4 partial_strong, 1..3 partial_weak, 0 missing",
            "observed_kcc_stream_count": n_total_kcc_streams,
            "independent_replicates": null,
            "replication_status": "not_established_by_stream_count",
            "streams": kcc_per_stream.iter().map(|k| serde_json::json!({
                "stream_id": k.stream_id, "integrity_ok": k.decoded.integrity_ok,
                "f32_field_count": k.decoded.f32_fields.len(), "u32_field_count": k.decoded.u32_fields.len(),
                "finite_f32_values": k.decoded.f32_fields.values().flatten().filter(|v| v.is_finite()).count(),
                "nonfinite_f32_values": k.decoded.f32_fields.values().flatten().filter(|v| !v.is_finite()).count(),
                "scope": "full_recorded_residue_population; not_candidate_top_three"
            })).collect::<Vec<_>>()
        });
        let native_causome = serde_json::json!({
            "schema_kind": "native_region_causome_observations", "schema_version": 1,
            "region_id": region_id, "residue_identity": "zero_based_topology_residue_index",
            "streams": native_kcc_streams,
            "selected_stream_id": selected_native_kcc.map(|(stream, _)| stream.stream_id),
            "selected_residue_index": selected_native_kcc.and_then(|(stream, _)| stream.selected_residue_index),
            "native_residue_kcc_confidence": native_residue_kcc_confidence,
            "confidence_formula": "max(motion_efficiency * active_causal / max(observation_count,1), clamp(burst_motion,0,5)/5 * clamp(local_cov,0,1)) * (1 + 0.1 * support_fraction)",
            "confidence_implementation": "prism_nhs::causome::native_residue_kcc_confidence",
            "confidence_branch_contributions": confidence_branches,
            "selection": "top_3_residues_by_stream_region_nearby_residue_support; max_native_confidence; local_cov_then_residue_id_tiebreak; max_stream_confidence_then_stream_id",
            "support_difference_from_live_cluster": "offline_spike_nearby_residue_votes; live_cluster_uses_signal_grid_primary_residue_votes",
            "rank_use": if args.ranking_policy == RankingPolicy::NativeKcc { "before_family_and_morphology" } else { "emitted_legacy_ranking_unchanged" },
            "lag_rank_use": "not_used_by_existing_native_confidence_formula; no_cross_origin_aggregation",
            "directed_causal_influence": "not_established_by_direction_score_or_lag_alone",
        });
        let lexicographic_rank_key = LexicographicSiteRankKey {
            native_residue_kcc_confidence,
            return_inclusive_geometric_consistency: return_geometry["consistency_factor"].as_f64(),
            rank_policy_id: rank_policy_id,
            key_order: rank_key_order.clone(),
            promoted,
            materialization_level_rank: materialization_level_rank(materialization_level),
            site_family_preservation_rank: 0.0,
            pocket_morphology_rank,
            thermodynamic_physiology_rank,
            n_active_non_density_dimensions: n_active_dims as u32,
            kcc_driver_factor,
            signal_preservation_factor,
            temporal_persistence_factor: temporal_persistence_factor.clamp(0.05, 1.0),
            centroid_manifold_consistency_factor: centroid_manifold_consistency_factor
                .clamp(0.05, 1.0),
            scale_normalized_dynamic_support,
            final_prism_score,
            n_streams_supporting: supporting_sorted.len() as u32,
            region_id,
        };

        // ── Build manifest payload (one JSON value per region) ──────────────
        let so3_block = serde_json::json!({
            "status": if n_views_present >= 3 && r_stat > 0.05 { "available" }
                      else if n_views_present >= 2 { "partial" }
                      else { "missing" },
            "stream_support": stream_support,
            "promotion_policy": promotion_policy,
            "temporal_support": {
                "first_observed_step": if region.timestep_min == i32::MAX { None } else { Some(region.timestep_min) },
                "last_observed_step":  if region.timestep_max == i32::MIN { None } else { Some(region.timestep_max) },
                "recurrence_count": region.chunk_set.len() as u32,
                "chunk_support": ({ let mut chunks = region.chunk_set.iter().copied().collect::<Vec<i32>>(); chunks.sort_unstable(); chunks }),
                "n_phase_bins_nonzero": region.phase_hist.iter().filter(|c| **c > 0).count() as u32,
            },
            "phase_support": phase_support,
            "so3_support": {
                "phase_coherence_rayleigh_r": r_stat,
                "orientation_coherence_score": serde_json::Value::Null,
                "local_frame_stability": serde_json::Value::Null,
                "rotation_dispersion": serde_json::Value::Null,
                "warp_matrix_support": warp_matrix_surface_status,
                "asc_vector_support": asc_vectors_surface_status,
                "force_direction_support": forces_surface_status,
                "evidence_files": spikes_files_consumed,
                "so3_score_status": "rotation_measurements_unavailable_native_tensor_join_required",
                "phase_coherence_semantics": "protocol_phase_Rayleigh_statistic_not_SO3_rotation",
                "dense_tensor_parse_status": if dense_so3_surfaces_parsed { "materialized_parsed" } else { "incomplete" },
                "rotation_evaluation_status": "not_evaluated",
                "dense_gpu_surfaces": {
                    "warp_matrix": warp_matrix_surface_status,
                    "asc_vectors": asc_vectors_surface_status,
                    "forces_final": forces_surface_status,
                },
            },
            "thermal_phase_semantics": {
                "counts_order": THERMAL_PHASE_NAMES,
                "hot_phase": "warm_hold", "cold_phase": "initial_cold_hold", "return_phase": "cold_return",
                "classification": "manifest_bound_protocol_schedule; missing_or_invalid_is_unclassified",
                "centroid_min_spikes": 50,
                "return_centroid_evaluation": "included_in_separate_return_inclusive_geometric_consistency; legacy_admission_unchanged",
                "return_inclusive_geometry": return_geometry,
            },
            "native_ghost_kcc_observation_surface": native_observation_surface,
            "support_accounting": region.support_accounting,
            "kcc_observation_availability": kcc_observation_availability,
            "kcc_full_residue_descriptors": {
                "surface": kcc_full_path,
                "qnanl": kcc_full_qnanl,
                "stream_ids": ({ let mut ids = region.per_stream_spikes.keys().copied().collect::<Vec<_>>(); ids.sort_unstable(); ids }),
                "region_residue_indices": ({ let mut ids = region.residue_support.keys().copied().collect::<Vec<_>>(); ids.sort_unstable(); ids }),
                "reference_semantics": "native_descriptor_index; includes_all_region_supported_residues_not_only_lining_top_k",
                "scoring_semantics": "native_confidence_consumes_motion_efficiency_active_fraction_burst_local_cov_support; signed_temporal_and_lag_fields_preserved_without_invented_weights",
                "native_causome": native_causome,
            },
            "centroid_manifold": manifold,
            "field_completeness": field_completeness,
        });
        let accuracy: serde_json::Value = if let Some(lig) = ligand_adjacent {
            // Ligand-adjacent comes from valid ground_truth only — used purely
            // as POST-HOC validation, never in score (per directive Part 9).
            let dcc_whole = whole_site.map(|c| dist3(&c, &lig));
            let dcc_lining = lining_mass.map(|c| dist3(&c, &lig));
            let dcc_driver = causal_driver.map(|c| dist3(&c, &lig));
            let dcc_hot = hot_phase.map(|c| dist3(&c, &lig));
            let dcc_cold = cold_phase.map(|c| dist3(&c, &lig));
            let best = [dcc_whole, dcc_lining, dcc_driver, dcc_hot, dcc_cold]
                .into_iter()
                .flatten()
                .fold(f32::INFINITY, f32::min);
            let grade = if best <= 2.0 {
                "EXCELLENT"
            } else if best <= 3.0 {
                "ACCEPTABLE"
            } else if best <= 5.0 {
                "USEFUL"
            } else if best <= 8.0 {
                "LENIENT"
            } else {
                "MISS"
            };
            serde_json::json!({
                "validation_status": "computed",
                "ligand_centroid": lig,
                "dcc_whole_site_a": dcc_whole,
                "dcc_lining_mass_a": dcc_lining,
                "dcc_causal_driver_a": dcc_driver,
                "dcc_hot_phase_a": dcc_hot,
                "dcc_cold_phase_a": dcc_cold,
                "best_dcc_a": best,
                "grade": grade,
            })
        } else {
            serde_json::json!({
                "validation_status": gt_status,
                "skip_reason": gt_skip_reason,
            })
        };
        let mut limitations: Vec<&'static str> = Vec::new();
        if warp_matrix_surface_status != "materialized_parsed" {
            limitations.push("warp_matrix_surface_not_materialized_parsed");
        }
        if asc_vectors_surface_status != "materialized_parsed" {
            limitations.push("asc_vector_surface_not_materialized_parsed");
        }
        if forces_surface_status != "materialized_parsed" {
            limitations.push("force_direction_surface_not_materialized_parsed");
        }
        let provenance = serde_json::json!({
            "source_manifest": args.manifest.display().to_string(),
            "promotion_policy": promotion_policy,
            "run_id": manifest.run_id,
            "target": manifest.target,
            "region_id": region_id,
            "representative_raw_peak_id": region.representative_raw_peak_id,
            "merged_raw_peak_ids": region.merged_raw_peak_ids,
            "n_raw_peaks_merged": region.merged_raw_peak_ids.len() as u32,
            "spikes_files_consumed": spikes_files_consumed,
            "kcc_files_consumed": kcc_files_consumed,
            "topology_input": manifest.topology_input,
            "residue_alignment_policy_id": RESIDUE_ALIGNMENT_POLICY_ID,
        });

        let payload = serde_json::json!({
            "site_id": format!("region_{:03}", region_id),
            "rank": null,
            "old_density_rank": density_only_rank_by_region.get(&region_id).copied().unwrap_or(0),
            "materialization_status": materialization_status,
            "materialization_level": materialization_level,
            "centroid_xyz": whole_site.unwrap_or(region.aggregate_centroid_a),
            "thermal_phase_semantics": {
                "counts_order": THERMAL_PHASE_NAMES,
                "hot_phase": "warm_hold", "cold_phase": "initial_cold_hold", "return_phase": "cold_return",
                "classification": "manifest_bound_protocol_schedule; missing_or_invalid_is_unclassified",
                "centroid_min_spikes": 50,
                "return_centroid_evaluation": "included_in_separate_return_inclusive_geometric_consistency; legacy_admission_unchanged",
                "return_inclusive_geometry": return_geometry,
            },
            "native_ghost_kcc_observation_surface": native_observation_surface,
            "support_accounting": region.support_accounting,
            "kcc_observation_availability": kcc_observation_availability,
            "kcc_full_residue_descriptors": {
                "surface": kcc_full_path,
                "qnanl": kcc_full_qnanl,
                "stream_ids": ({ let mut ids = region.per_stream_spikes.keys().copied().collect::<Vec<_>>(); ids.sort_unstable(); ids }),
                "region_residue_indices": ({ let mut ids = region.residue_support.keys().copied().collect::<Vec<_>>(); ids.sort_unstable(); ids }),
                "reference_semantics": "native_descriptor_index; includes_all_region_supported_residues_not_only_lining_top_k",
                "scoring_semantics": "native_confidence_consumes_motion_efficiency_active_fraction_burst_local_cov_support; signed_temporal_and_lag_fields_preserved_without_invented_weights",
                "native_causome": native_causome,
            },
            "centroid_manifold": manifold,
            "representative_residues": representative_residues,
            "lining_residues": lining_residues,
            "driver_residues": driver_residues,
            "author_residue_mapping_status": author_residue_mapping["status"],
            "representative_author_residues": topology.as_ref().map(|t| t.author_references(&representative_residues)).transpose()?.unwrap_or(serde_json::Value::Null),
            "lining_author_residues": topology.as_ref().map(|t| t.author_references(&lining_residues)).transpose()?.unwrap_or(serde_json::Value::Null),
            "driver_author_residues": topology.as_ref().map(|t| t.author_references(&driver_residues)).transpose()?.unwrap_or(serde_json::Value::Null),
            "n_spikes": region.n_spikes,
            "intensity_sum": region.intensity_sum,
            "n_raw_peaks_merged": region.merged_raw_peak_ids.len() as u32,
            "region_radius_a": region.region_radius_a,
            "centroid_spread_a": region.centroid_spread_a,
            "spike_support": {
                "n_uv": region.n_uv,
                "n_lif": region.n_lif,
                "n_other_source": region.n_other_source,
                "n_aromatic": {
                    "TRP": region.n_arom[0],
                    "TYR": region.n_arom[1],
                    "PHE": region.n_arom[2],
                    "SS":  region.n_arom[3],
                    "none": region.n_arom[4],
                },
            },
            "stream_support": stream_support,
            "promotion_policy": promotion_policy,
            "stream_balance": stream_balance,
            "phase_support": phase_support,
            "refined_temporal_support": refined_temporal,
            "residue_shell_plausibility": residue_shell,
            "pocket_morphology_surface": pocket_morphology,
            "binding_site_physiology_surface": binding_site_physiology,
            "manifold_consistency": manifold_block,
            "spike_support_geometry": region.spike_support_geometry.surface(),
            "return_inclusive_geometry": return_geometry,
            "native_causome": native_causome,
            "kcc_driver": kcc_driver_block_with_dist,
            "spatiotemporal_so3_evidence": so3_block,
            "field_completeness": field_completeness,
            "score_components": score_components,
            "lexicographic_rank_key": lexicographic_rank_key.clone(),
            "blocking_reasons": blocking,
            "n_active_non_density_dimensions": n_active_dims as u32,
            "promoted": promoted,
            "provenance": provenance,
            "limitations": limitations,
            "accuracy": accuracy,
        });

        scored.push(ScoredRegion {
            region_id,
            old_density_rank: density_only_rank_by_region
                .get(&region_id)
                .copied()
                .unwrap_or(0),
            lexicographic_rank_key,
            score_components,
            manifest_payload: payload,
            materialization_level,
            materialization_status,
            promoted,
            lining_residues,
            pocket_morphology_rank,
            thermodynamic_physiology_rank,
            scale_normalized_dynamic_support,
            site_family_preservation_rank: 0.0,
        });
    }

    // ── Site-family preservation/admission lane (pre-sort) ──────────────────
    let mut family_members: Vec<Vec<usize>> = Vec::new();
    let mut family_lining_union: Vec<Vec<i32>> = Vec::new();
    for idx in 0..scored.len() {
        let lining = scored[idx].lining_residues.clone();
        let mut placed = false;
        for (fi, prev) in family_lining_union.iter_mut().enumerate() {
            if jaccard_overlap(&lining, prev) >= 0.4 {
                family_members[fi].push(idx);
                let mut union: std::collections::HashSet<i32> = prev.iter().copied().collect();
                for residue_id in &lining {
                    union.insert(*residue_id);
                }
                let mut next_union: Vec<i32> = union.into_iter().collect();
                next_union.sort_unstable();
                *prev = next_union;
                placed = true;
                break;
            }
        }
        if !placed {
            family_members.push(vec![idx]);
            let mut unique = lining;
            unique.sort_unstable();
            unique.dedup();
            family_lining_union.push(unique);
        }
    }

    let mut site_family_records: Vec<serde_json::Value> = Vec::new();
    for family_id in 0..family_members.len() {
        let members = family_members[family_id].clone();
        let union_lining = family_lining_union[family_id].clone();
        let shared_lining: Vec<i32> = if members.len() <= 1 {
            union_lining.clone()
        } else {
            let mut shared: std::collections::HashSet<i32> =
                scored[members[0]].lining_residues.iter().copied().collect();
            for idx in members.iter().skip(1) {
                let current: std::collections::HashSet<i32> =
                    scored[*idx].lining_residues.iter().copied().collect();
                shared = shared.intersection(&current).copied().collect();
            }
            let mut shared_vec: Vec<i32> = shared.into_iter().collect();
            shared_vec.sort_unstable();
            shared_vec
        };
        let family_span = union_lining
            .iter()
            .min()
            .zip(union_lining.iter().max())
            .map(|(lo, hi)| (*hi - *lo).max(0))
            .unwrap_or(0);
        let topology_span_fraction = family_span as f64 / (n_residues_topo.max(1) as f64);
        let mean_morphology_rank = members
            .iter()
            .map(|idx| scored[*idx].pocket_morphology_rank)
            .sum::<f64>()
            / members.len().max(1) as f64;
        let mean_physiology_rank = members
            .iter()
            .map(|idx| scored[*idx].thermodynamic_physiology_rank)
            .sum::<f64>()
            / members.len().max(1) as f64;
        let mean_scale_dynamic = members
            .iter()
            .map(|idx| scored[*idx].scale_normalized_dynamic_support)
            .sum::<f64>()
            / members.len().max(1) as f64;
        let multi_member_score = if members.len() >= 2 { 1.0 } else { 0.0 };
        let shared_shell_score = (shared_lining.len() as f64 / 12.0).clamp(0.0, 1.0);
        let union_shell_score = (union_lining.len() as f64 / 16.0).clamp(0.0, 1.0);
        let discontinuous_topology_score =
            if topology_span_fraction >= 0.30 && shared_lining.len() >= 6 {
                topology_span_fraction.clamp(0.0, 1.0)
            } else {
                0.0
            };
        let preservation_rank = 1000.0
            * (0.22 * multi_member_score
                + 0.18 * shared_shell_score
                + 0.12 * union_shell_score
                + 0.18 * discontinuous_topology_score
                + 0.15 * (mean_morphology_rank / 1000.0).clamp(0.0, 1.0)
                + 0.10 * (mean_physiology_rank / 1000.0).clamp(0.0, 1.0)
                + 0.05 * mean_scale_dynamic.clamp(0.0, 1.0))
            .clamp(0.0, 1.0);
        let family_status = if preservation_rank >= 650.0 {
            "preserve_high_value_family"
        } else if preservation_rank >= 400.0 {
            "preserve_supported_family"
        } else {
            "ordinary_family"
        };
        let member_region_ids: Vec<u32> =
            members.iter().map(|idx| scored[*idx].region_id).collect();
        let family_record = serde_json::json!({
            "family_id": family_id as u32,
            "n_members": members.len(),
            "member_region_ids": member_region_ids,
            "shared_lining": shared_lining,
            "family_lining_union": union_lining,
            "topology_span_residues": family_span,
            "topology_span_fraction": topology_span_fraction,
            "mean_pocket_morphology_rank": mean_morphology_rank,
            "mean_thermodynamic_physiology_rank": mean_physiology_rank,
            "mean_scale_normalized_dynamic_support": mean_scale_dynamic,
            "site_family_preservation_rank": preservation_rank,
            "family_preservation_status": family_status,
            "rank_lane": "PRISMOS_SITE_FAMILY_PRESERVATION_ADMISSION_V1",
            "identity_scope": "spatial_parent_family; not_a_shared_temporal_episode_or_chemical_state",
            "phase_observation_identity_policy": "preserve_each_member_region_stream_phase_source_observation; never_pool_child_counts_through_family",
            "temporal_or_chemical_identity_used_for_family_grouping": false,
        });
        for idx in members {
            scored[idx].site_family_preservation_rank = preservation_rank;
            scored[idx].score_components.site_family_preservation_rank = preservation_rank;
            scored[idx]
                .lexicographic_rank_key
                .site_family_preservation_rank = preservation_rank;
            scored[idx].manifest_payload["site_family_preservation"] = serde_json::json!({
                "surface_id": "PRISMOS_SITE_FAMILY_PRESERVATION_ADMISSION_V1",
                "family_id": family_id as u32,
                "site_family_preservation_rank": preservation_rank,
                "family_preservation_status": family_status,
                "topology_span_fraction": topology_span_fraction,
            });
            scored[idx].manifest_payload["score_components"]["site_family_preservation_rank"] =
                serde_json::Value::from(preservation_rank);
            scored[idx].manifest_payload["lexicographic_rank_key"]
                ["site_family_preservation_rank"] = serde_json::Value::from(preservation_rank);
        }
        site_family_records.push(family_record);
    }

    // ── Final rank: deterministic PRISMOS lexicographic site-rank key ──────
    scored.sort_by(|a, b| {
        compare_lexicographic_site_rank(&a.lexicographic_rank_key, &b.lexicographic_rank_key)
            .then_with(|| a.region_id.cmp(&b.region_id))
    });
    for (i, sr) in scored.iter_mut().enumerate() {
        sr.manifest_payload["rank"] = serde_json::Value::from(i as u32);
        sr.manifest_payload["new_prism_rank"] = serde_json::Value::from(i as u32);
        sr.manifest_payload["site_id"] = serde_json::Value::from(format!("site_{:03}", i));
    }
    // Promotion: top materialized_top_n among those with `promoted=true`.
    for (i, sr) in scored.iter_mut().enumerate() {
        if !sr.promoted {
            // Bucket non-promoted into spatiotemporal_partial vs density_only.
            match sr.materialization_level {
                "candidate_spatiotemporal_partial" => {
                    spatiotemporal_partial_candidates.push(sr.manifest_payload.clone())
                }
                _ => density_only_candidates.push(sr.manifest_payload.clone()),
            }
            continue;
        }
        if materialized_v2.len() < args.materialized_top_n {
            materialized_v2.push(sr.manifest_payload.clone());
        } else {
            // Beyond top-N, retain as spatiotemporal_partial for reporting.
            spatiotemporal_partial_candidates.push(sr.manifest_payload.clone());
        }
        let _ = i;
    }
    eprintln!(
        "[v2] scored {} regions; {} promoted (top-{} materialized); {} spatiotemporal_partial; {} density_only",
        scored.len(), materialized_v2.len(), args.materialized_top_n,
        spatiotemporal_partial_candidates.len(), density_only_candidates.len()
    );

    // ─── Emit canonical physiology/morphology surfaces ───────────────────────
    let pocket_morphology_surface_path =
        output_dir.join("pocket_morphology_surface.prismos_v1.json");
    let pocket_morphology_surface_json = serde_json::json!({
        "schema_version": 1,
        "schema_kind": "prismos_geometric_pocket_morphology_surface",
        "surface_id": "PRISMOS_GEOMETRIC_POCKET_MORPHOLOGY_V1",
        "rank_policy_id": rank_policy_id,
        "target": manifest.target,
        "run_id": manifest.run_id,
        "source_manifest": args.manifest.display().to_string(),
            "promotion_policy": promotion_policy,
        "residue_alignment_policy": residue_alignment_policy.clone(),
        "author_residue_mapping": author_residue_mapping.clone(),
        "n_regions": scored.len(),
        "n_materialized_top_set": materialized_v2.len(),
        "site_family_preservation": site_family_records.clone(),
        "records": scored.iter().map(|sr| serde_json::json!({
            "site_id": sr.manifest_payload["site_id"],
            "rank": sr.manifest_payload["rank"],
            "region_id": sr.region_id,
            "old_density_rank": sr.old_density_rank,
            "materialization_status": sr.materialization_status,
            "materialization_level": sr.materialization_level,
            "promoted": sr.promoted,
            "materialized_top_set": sr.promoted && sr.manifest_payload["rank"].as_u64().unwrap_or(u64::MAX) < args.materialized_top_n as u64,
            "centroid_xyz": sr.manifest_payload["centroid_xyz"],
            "lining_residues": sr.manifest_payload["lining_residues"],
            "author_residue_mapping_status": sr.manifest_payload["author_residue_mapping_status"],
            "representative_author_residues": sr.manifest_payload["representative_author_residues"],
            "lining_author_residues": sr.manifest_payload["lining_author_residues"],
            "driver_author_residues": sr.manifest_payload["driver_author_residues"],
            "site_family_preservation": sr.manifest_payload["site_family_preservation"],
            "pocket_morphology_surface": sr.manifest_payload["pocket_morphology_surface"],
            "lexicographic_rank_key": sr.manifest_payload["lexicographic_rank_key"],
        })).collect::<Vec<_>>(),
    });
    {
        let f = File::create(&pocket_morphology_surface_path)?;
        let mut bw = std::io::BufWriter::new(f);
        serde_json::to_writer_pretty(&mut bw, &pocket_morphology_surface_json)?;
        use std::io::Write as _;
        bw.flush()?;
    }
    let pocket_morphology_qnanl = seal_json_artifact(
        &pocket_morphology_surface_path,
        "PRISMOS_GEOMETRIC_POCKET_MORPHOLOGY_V1",
        QnanlArtifactFamily::GeometricPocketMorphology,
    )?;
    eprintln!("✓ wrote {}", pocket_morphology_surface_path.display());

    let binding_site_physiology_surface_path =
        output_dir.join("binding_site_physiology_surface.prismos_v1.json");
    let binding_site_physiology_surface_json = serde_json::json!({
        "schema_version": 1,
        "schema_kind": "prismos_binding_site_physiology_surface",
        "surface_id": "PRISMOS_BINDING_SITE_PHYSIOLOGY_V1",
        "rank_policy_id": rank_policy_id,
        "target": manifest.target,
        "run_id": manifest.run_id,
        "source_manifest": args.manifest.display().to_string(),
            "promotion_policy": promotion_policy,
        "residue_alignment_policy": residue_alignment_policy.clone(),
        "author_residue_mapping": author_residue_mapping.clone(),
        "n_regions": scored.len(),
        "n_materialized_top_set": materialized_v2.len(),
        "records": scored.iter().map(|sr| serde_json::json!({
            "site_id": sr.manifest_payload["site_id"],
            "rank": sr.manifest_payload["rank"],
            "region_id": sr.region_id,
            "old_density_rank": sr.old_density_rank,
            "materialization_status": sr.materialization_status,
            "materialization_level": sr.materialization_level,
            "promoted": sr.promoted,
            "materialized_top_set": sr.promoted && sr.manifest_payload["rank"].as_u64().unwrap_or(u64::MAX) < args.materialized_top_n as u64,
            "centroid_xyz": sr.manifest_payload["centroid_xyz"],
            "lining_residues": sr.manifest_payload["lining_residues"],
            "author_residue_mapping_status": sr.manifest_payload["author_residue_mapping_status"],
            "representative_author_residues": sr.manifest_payload["representative_author_residues"],
            "lining_author_residues": sr.manifest_payload["lining_author_residues"],
            "driver_author_residues": sr.manifest_payload["driver_author_residues"],
            "driver_residues": sr.manifest_payload["driver_residues"],
            "site_family_preservation": sr.manifest_payload["site_family_preservation"],
            "binding_site_physiology_surface": sr.manifest_payload["binding_site_physiology_surface"],
            "score_components": sr.manifest_payload["score_components"],
            "lexicographic_rank_key": sr.manifest_payload["lexicographic_rank_key"],
        })).collect::<Vec<_>>(),
    });
    {
        let f = File::create(&binding_site_physiology_surface_path)?;
        let mut bw = std::io::BufWriter::new(f);
        serde_json::to_writer_pretty(&mut bw, &binding_site_physiology_surface_json)?;
        use std::io::Write as _;
        bw.flush()?;
    }
    let binding_site_physiology_qnanl = seal_json_artifact(
        &binding_site_physiology_surface_path,
        "PRISMOS_BINDING_SITE_PHYSIOLOGY_V1",
        QnanlArtifactFamily::BindingSitePhysiology,
    )?;
    eprintln!("✓ wrote {}", binding_site_physiology_surface_path.display());

    // Backward-compat shims used by later code paths and emit blocks.
    let materialized: Vec<serde_json::Value> = materialized_v2.clone();
    let non_materialized: Vec<serde_json::Value> = density_only_candidates.clone();

    // ═══════════════════════════════════════════════════════════════════════
    // EMIT v2 OUTPUTS
    // ═══════════════════════════════════════════════════════════════════════

    // ─── Emit binding_sites.materialized.json (v2 schema) ───────────────────
    let materialized_path = output_dir.join("binding_sites.materialized.json");
    let materialized_status: &str = if materialized_v2.is_empty() {
        "no_sites_met_criteria"
    } else if gt_status == "available" {
        "materialized_from_md_evidence_with_validation"
    } else {
        "materialized_from_md_evidence"
    };
    let materialized_json = serde_json::json!({
        "schema_version":   2,
        "schema_kind":      "pathb_binding_sites_materialized",
        "ranking_methodology": ranking_methodology,
        "rank_policy": {
            "rank_policy_id": rank_policy_id,
            "final_decision_layer": "lexicographic",
            "score_role": "final_prism_score remains a late key; explicit key_order is authoritative",
            "native_confidence_support_semantics": "per_stream_spike_nearby_residue_votes; no_cross_stream_lag_aggregation",
            "admission_policy": "legacy_five_view_manifold_and_existing_promotion_predicate_unchanged",
            "return_geometry_scope": "centroid_geometric_consistency_not_thermodynamic_hysteresis",
            "key_order": rank_key_order,
            "successor_to": "PRISMOS_SITE_MATERIALIZATION_LEXICOGRAPHIC_V1"
        },
        "canonical_surfaces": {
            "pocket_morphology_surface": {
                "surface_id": "PRISMOS_GEOMETRIC_POCKET_MORPHOLOGY_V1",
                "path": pocket_morphology_surface_path.display().to_string(),
                "qnanl": pocket_morphology_qnanl,
            },
            "binding_site_physiology_surface": {
                "surface_id": "PRISMOS_BINDING_SITE_PHYSIOLOGY_V1",
                "path": binding_site_physiology_surface_path.display().to_string(),
                "qnanl": binding_site_physiology_qnanl,
            },
            "site_family_preservation": {
                "surface_id": "PRISMOS_SITE_FAMILY_PRESERVATION_ADMISSION_V1",
                "record_count": site_family_records.len(),
            }
        },
        "ghost_zstr_candidate_features_consumption": ghost_features_consumption_receipt(&ghost_features_state),
        "status":           materialized_status,
        "input_admission_status": if input_failures.is_empty() { "VALIDATED" } else { "FAILED_INPUT_VALIDATION" },
        "input_failures": input_failures,
        "target":           manifest.target,
        "run_id":           manifest.run_id,
        "source_manifest":  args.manifest.display().to_string(),
        "promotion_policy": promotion_policy,
        "residue_alignment_policy": residue_alignment_policy.clone(),
        "author_residue_mapping": author_residue_mapping.clone(),
        "site_count":       materialized_v2.len(),
        "n_raw_peaks":      raw_peaks.len(),
        "n_consolidated_regions": regions.len(),
        "consolidation_radius_a": args.consolidation_radius_a,
        "consolidation_jaccard":  args.consolidation_jaccard,
        "binding_sites":    materialized_v2,
        "spatiotemporal_partial_candidates": spatiotemporal_partial_candidates,
        "density_only_candidates": density_only_candidates,
        "missing_fields":   {
            "warp_matrix_so3":         warp_matrix_surface_status,
            "asc_vector_orientation":  asc_vectors_surface_status,
            "force_direction":         forces_surface_status,
            "ligsite_static_geometry": "intentionally_excluded_per_directive",
        },
        "ground_truth_status":     gt_status,
        "ground_truth_skip_reason": gt_skip_reason,
        "path_b_required":          gt_status != "available",
    });
    {
        let f = File::create(&materialized_path)?;
        let mut bw = std::io::BufWriter::new(f);
        serde_json::to_writer_pretty(&mut bw, &materialized_json)?;
        use std::io::Write as _;
        bw.flush()?;
    }
    eprintln!(
        "✓ wrote {} (sites={}, st_partial={}, density_only={})",
        materialized_path.display(),
        materialized_v2.len(),
        spatiotemporal_partial_candidates.len(),
        density_only_candidates.len()
    );

    // ─── Emit site_accuracy_report.json (post-hoc validation only) ─────────
    let accuracy_path = output_dir.join("site_accuracy_report.json");
    // Aggregate grade summary if validation was computed.
    let (mut n_excellent, mut n_acceptable, mut n_useful, mut n_lenient, mut n_miss) =
        (0u32, 0u32, 0u32, 0u32, 0u32);
    let mut best_dcc_overall: Option<f32> = None;
    let mut best_grade_overall: Option<String> = None;
    let mut rank_first_lenient: Option<u32> = None;
    if gt_status == "available" {
        for s in materialized_v2.iter() {
            let acc = &s["accuracy"];
            if let Some(best) = acc["best_dcc_a"].as_f64() {
                let best_f = best as f32;
                if best <= 2.0 {
                    n_excellent += 1;
                } else if best <= 3.0 {
                    n_acceptable += 1;
                } else if best <= 5.0 {
                    n_useful += 1;
                } else if best <= 8.0 {
                    n_lenient += 1;
                } else {
                    n_miss += 1;
                }
                if best_dcc_overall.is_none() || best_f < best_dcc_overall.unwrap() {
                    best_dcc_overall = Some(best_f);
                    best_grade_overall = acc["grade"].as_str().map(|x| x.to_string());
                }
                if best_f <= 8.0 && rank_first_lenient.is_none() {
                    rank_first_lenient = s["rank"].as_u64().map(|x| x as u32);
                }
            }
        }
    }
    let accuracy_json = serde_json::json!({
        "schema_version": 2,
        "schema_kind":    "pathb_site_accuracy_report",
        "target":         manifest.target,
        "run_id":         manifest.run_id,
        "ground_truth_status":     gt_status,
        "ground_truth_skip_reason": gt_skip_reason,
        "ligand_centroid_used":    ground_truth.as_ref().and_then(|gt| gt.ligand_centroid),
        "n_materialized_sites":    materialized_v2.len(),
        "validation_computable":   gt_status == "available",
        "validation_independent_of_ranking": true,
        "per_site_accuracy":       if gt_status == "available" {
            serde_json::Value::Array(materialized_v2.iter().map(|s| serde_json::json!({
                "site_id":      s["site_id"],
                "rank":         s["rank"],
                "centroid_xyz": s["centroid_xyz"],
                "accuracy":     s["accuracy"],
            })).collect())
        } else { serde_json::Value::Null },
        "summary": {
            "n_excellent_le2A":   if gt_status == "available" { serde_json::Value::from(n_excellent) } else { serde_json::Value::Null },
            "n_acceptable_le3A":  if gt_status == "available" { serde_json::Value::from(n_acceptable) } else { serde_json::Value::Null },
            "n_useful_le5A":      if gt_status == "available" { serde_json::Value::from(n_useful) } else { serde_json::Value::Null },
            "n_lenient_le8A":     if gt_status == "available" { serde_json::Value::from(n_lenient) } else { serde_json::Value::Null },
            "n_miss_gt8A":        if gt_status == "available" { serde_json::Value::from(n_miss) } else { serde_json::Value::Null },
            "best_dcc_a_overall": best_dcc_overall,
            "best_grade_overall": best_grade_overall,
            "rank_first_le8A":    rank_first_lenient,
            "grade_note":         if gt_status == "available" { "validation aggregated post-hoc; ranking did NOT use ligand reference"} else { "validation not computable: reference unavailable per ground_truth.json" },
        },
        "limitations": [
            if warp_matrix_surface_status == "materialized_parsed" { "warp_matrix_parsed_rotation_measurement_not_evaluated" } else { "warp_matrix_surface_not_materialized_parsed" },
            if asc_vectors_surface_status == "materialized_parsed" { "asc_vector_parsed_orientation_measurement_not_evaluated" } else { "asc_vector_surface_not_materialized_parsed" },
            if forces_surface_status == "materialized_parsed" { "forces_parsed_direction_measurement_not_evaluated" } else { "force_direction_surface_not_materialized_parsed" },
            "ligsite_static_geometry_intentionally_excluded",
        ],
    });
    {
        let f = File::create(&accuracy_path)?;
        let mut bw = std::io::BufWriter::new(f);
        serde_json::to_writer_pretty(&mut bw, &accuracy_json)?;
        use std::io::Write as _;
        bw.flush()?;
    }
    eprintln!("✓ wrote {}", accuracy_path.display());

    // ─── Emit materialization_field_completeness.json ──────────────────────
    let mfc_path = output_dir.join("materialization_field_completeness.json");
    let mfc_json = serde_json::json!({
        "schema_version": 2,
        "schema_kind":    "pathb_materialization_field_completeness",
        "evidence_sources": {
            "spikes":         { "status": "available", "format": "PRSPK001", "streams_present": spikes_files_consumed.len() },
            "kcc_v2full":     { "status": kcc_surface_status, "format": "PRKCC001", "streams_present": materialized_surface_count("kcc_v2full") },
            "signal_grid":    { "status": signal_grid_surface_status, "format": "PRSGD001", "streams_present": materialized_surface_count("signal_grid") },
            "warp_matrix":    { "status": warp_matrix_surface_status, "format": "GpuWarpEntry[136]", "streams_present": materialized_surface_count("warp_matrix") },
            "forces_final":   { "status": forces_surface_status, "format": "f32[n_atoms x 3]", "streams_present": materialized_surface_count("forces_final") },
            "asc_vectors":    { "status": asc_vectors_surface_status, "format": "f32[n_atoms x 3]", "streams_present": materialized_surface_count("asc_vectors") },
            "adaptive_dt":    { "status": adaptive_dt_surface_status, "format": "adaptive_dt_32_byte_records", "streams_present": materialized_surface_count("adaptive_dt") },
            "protocol_state": { "status": protocol_state_surface_status, "format": "json", "streams_present": materialized_surface_count("protocol_state") },
            "bocpd_log":      { "status": bocpd_log_surface_status, "format": "jsonl", "streams_present": materialized_surface_count("bocpd_log") },
            "spike_transition_events": { "status": spike_transition_surface_status, "format": "jsonl", "streams_present": materialized_surface_count("spike_transition_events") },
            "first_class_tensor_surface_materialization": {
                "status": if tensor_surface_blockers.is_empty() { "materialized_parsed" } else { "blocked_platform_surface_materialization" },
                "canonical_storage_policy": "PrismOS/QNANL packets are primary; raw files are debug/provenance fallback only",
                "surface_count": tensor_surface_summaries.len(),
                "blocker_count": tensor_surface_blockers.len(),
                "counts_by_kind": tensor_surface_counts_by_kind.clone(),
                "blockers": tensor_surface_blockers.clone(),
            },
            "topology":       { "status": if topology.is_some() { "available" } else { "missing" }, "format": "json", "path": manifest.topology_input },
            "residue_alignment": residue_alignment_policy.clone(),
            "ground_truth":   { "status": gt_status, "skip_reason": gt_skip_reason },
        },
        "n_raw_peaks":                     raw_peaks.len(),
        "n_consolidated_regions":          regions.len(),
        "n_promoted_materialized":         materialized_v2.len(),
        "n_spatiotemporal_partial":        spatiotemporal_partial_candidates.len(),
        "n_density_only":                  density_only_candidates.len(),
        "promotion_summary": {
            "materialized_dynamic_site_with_validation":  materialized_v2.iter().filter(|s| s["materialization_level"].as_str() == Some("materialized_dynamic_site_with_validation")).count(),
            "materialized_dynamic_site":                  materialized_v2.iter().filter(|s| s["materialization_level"].as_str() == Some("materialized_dynamic_site")).count(),
            "candidate_spatiotemporal_partial":           spatiotemporal_partial_candidates.len(),
            "candidate_density_only":                     density_only_candidates.len(),
        },
        "ranking_methodology":             ranking_methodology,
        "rank_policy_id":                  rank_policy_id,
        "canonical_surface_materialization": {
            "pocket_morphology_surface": "PRISMOS_GEOMETRIC_POCKET_MORPHOLOGY_V1",
            "binding_site_physiology_surface": "PRISMOS_BINDING_SITE_PHYSIOLOGY_V1",
            "site_family_preservation": "PRISMOS_SITE_FAMILY_PRESERVATION_ADMISSION_V1",
        },
        "ligsite_role":                    "intentionally_excluded_per_directive",
    });
    {
        let f = File::create(&mfc_path)?;
        let mut bw = std::io::BufWriter::new(f);
        serde_json::to_writer_pretty(&mut bw, &mfc_json)?;
        use std::io::Write as _;
        bw.flush()?;
    }
    eprintln!("✓ wrote {}", mfc_path.display());

    // ─── Emit duplicate_cluster_report.json ─────────────────────────────────
    let dup_path = output_dir.join("duplicate_cluster_report.json");
    let dup_json = serde_json::json!({
        "schema_version": 1,
        "schema_kind":    "pathb_duplicate_cluster_report",
        "consolidation_radius_a":  args.consolidation_radius_a,
        "consolidation_jaccard":   args.consolidation_jaccard,
        "n_raw_peaks":             raw_peaks.len(),
        "n_regions":               regions.len(),
        "n_clusters_with_suppression": duplicate_records.len(),
        "duplicate_clusters":      duplicate_records,
    });
    {
        let f = File::create(&dup_path)?;
        let mut bw = std::io::BufWriter::new(f);
        serde_json::to_writer_pretty(&mut bw, &dup_json)?;
        use std::io::Write as _;
        bw.flush()?;
    }
    eprintln!("✓ wrote {}", dup_path.display());

    // ─── Emit ranking_ablation_report.json ──────────────────────────────────
    // Re-rank the SAME consolidated regions under different scoring schemes to
    // show which component moved a region. Each scheme returns the top-10
    // region_ids in rank order.
    fn top_k_by<F: FnMut(&ScoredRegion) -> f64>(
        scored: &[ScoredRegion],
        k: usize,
        mut score_fn: F,
    ) -> Vec<serde_json::Value> {
        let mut s: Vec<(u32, f64, [f32; 3], Vec<i32>)> = scored
            .iter()
            .map(|r| {
                (
                    r.region_id,
                    score_fn(r),
                    {
                        let v = &r.manifest_payload["centroid_xyz"];
                        [
                            v[0].as_f64().unwrap_or(0.0) as f32,
                            v[1].as_f64().unwrap_or(0.0) as f32,
                            v[2].as_f64().unwrap_or(0.0) as f32,
                        ]
                    },
                    r.manifest_payload["lining_residues"]
                        .as_array()
                        .map(|a| {
                            a.iter()
                                .filter_map(|x| x.as_i64().map(|n| n as i32))
                                .take(5)
                                .collect()
                        })
                        .unwrap_or_default(),
                )
            })
            .collect();
        s.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        s.into_iter()
            .take(k)
            .enumerate()
            .map(|(i, (rid, score, c, lining))| {
                serde_json::json!({
                    "rank": i,
                    "region_id": rid,
                    "score": score,
                    "centroid_xyz": c,
                    "lining_residues_top5": lining,
                })
            })
            .collect()
    }
    let abl_path = output_dir.join("ranking_ablation_report.json");
    let abl_json = serde_json::json!({
        "schema_version": 1,
        "schema_kind":    "pathb_ranking_ablation_report",
        "n_regions": scored.len(),
        "schemes": {
            "density_only":         top_k_by(&scored, 10, |r| r.score_components.density_score),
            "density_plus_nms":     top_k_by(&scored, 10, |r| r.score_components.density_score * r.score_components.duplicate_region_factor),
            "nms_plus_residue_shell": top_k_by(&scored, 10, |r| r.score_components.density_score * r.score_components.residue_shell_plausibility_factor * r.score_components.tail_penalty_factor),
            "nms_plus_manifold":    top_k_by(&scored, 10, |r| r.score_components.density_score * r.score_components.centroid_manifold_consistency_factor),
            "nms_plus_kcc":         top_k_by(&scored, 10, |r| r.score_components.density_score * r.score_components.kcc_driver_factor),
            "final_prism_score":    top_k_by(&scored, 10, |r| r.score_components.final_prism_score),
        },
    });
    {
        let f = File::create(&abl_path)?;
        let mut bw = std::io::BufWriter::new(f);
        serde_json::to_writer_pretty(&mut bw, &abl_json)?;
        use std::io::Write as _;
        bw.flush()?;
    }
    eprintln!("✓ wrote {}", abl_path.display());

    // ─── Emit site_family_report.json ───────────────────────────────────────
    // Uses the pre-sort family preservation lane rather than top-N-only
    // materialized sites, so quiet but high-value families remain visible.
    let family_path = output_dir.join("site_family_report.json");
    let family_json = serde_json::json!({
        "schema_version": 2,
        "schema_kind":    "pathb_site_family_report",
        "rank_lane":      "PRISMOS_SITE_FAMILY_PRESERVATION_ADMISSION_V1",
        "rank_policy_id": rank_policy_id,
        "family_scope":   "all_scored_regions_pre_top_n_truncation",
        "n_families":     site_family_records.len(),
        "families":       site_family_records.clone(),
    });
    {
        let f = File::create(&family_path)?;
        let mut bw = std::io::BufWriter::new(f);
        serde_json::to_writer_pretty(&mut bw, &family_json)?;
        use std::io::Write as _;
        bw.flush()?;
    }
    eprintln!("✓ wrote {}", family_path.display());

    // ─── Honest summary ────────────────────────────────────────────────────
    let n_valid: usize = validations.iter().filter(|v| v.valid).count();
    let n_total: usize = validations.len();
    eprintln!(
        "\nDone. {} / {} envelope artifacts validated. {} spikes processed across {} streams. {} unique voxels at {:.1} Å. Top {} density peaks emitted as honest candidates (NOT materialized sites).",
        n_valid,
        n_total,
        total_processed,
        stream_summaries.len(),
        voxel_grid.len(),
        args.voxel_size_a,
        candidates.len(),
    );
    if n_total > 0 && n_valid < n_total {
        eprintln!(
            "WARNING: {} artifacts failed validation — see materialization_report.json",
            n_total - n_valid
        );
    }
    eprintln!(
        "Materialization summary: {} sites materialized, {} non_materialized candidates, ground_truth={}",
        materialized.len(),
        non_materialized.len(),
        gt_status,
    );

    if !input_failures.is_empty() {
        anyhow::bail!("MATERIALIZATION_INPUT_VALIDATION_FAILED: {} failures; emitted artifacts preserved; see {}", input_failures.len(), report_path.display());
    }
    Ok(())
}

#[cfg(test)]
mod operationalization_tests {
    use super::*;

    fn component(id: &str, x: f32, score: f64) -> GhostMappedComponent {
        GhostMappedComponent {
            component_id: id.to_string(),
            stream_id: 0,
            site_id: if x < 50.0 { 1 } else { 2 },
            centroid_xyz: [x, 0.0, 0.0],
            aabb: None,
            residue_support: vec![10],
            coordinate_frame: "topology_angstrom".to_string(),
            mapping_method: "ghost_site_map.centroid_xyz".to_string(),
            mapping_source: "unit-test".to_string(),
            ghost_event_count: 10,
            ghost_tile_score_mean: score,
            ghost_tile_score_max: score,
            zstr_bytes: 4096,
            mapping_scope: None,
            exact_component_adapter: false,
            component_step_min: None,
            component_step_max: None,
            exact_component_provenance: None,
        }
    }

    fn exact_component(
        id: &str,
        stream_id: u32,
        step_min: u64,
        step_max: u64,
    ) -> GhostMappedComponent {
        let mut component = component(id, 0.0, 0.9);
        component.stream_id = stream_id;
        component.aabb = Some([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]);
        component.mapping_method =
            "exact_native_component_join.component_aabb_midpoint".to_string();
        component.mapping_scope =
            Some("exact_native_component_join_single_stream_site_aabb".to_string());
        component.exact_component_adapter = true;
        component.component_step_min = Some(step_min);
        component.component_step_max = Some(step_max);
        component.exact_component_provenance = Some(serde_json::json!({
            "schema_kind": "prism_exact_component_mapped_adapter_provenance",
            "schema_version": 1,
            "test": true,
        }));
        component
    }

    #[test]
    fn absent_ghost_evidence_is_neutral() {
        let summary = ghost_zstr_summary_for_geometry(
            &GhostFeaturesState::default(),
            0,
            [0.0, 0.0, 0.0],
            2.0,
        );
        assert_eq!(summary.factor, 1.0);
        assert_eq!(summary.status, "absent_neutral");
        assert!(summary.links.is_empty());
    }

    #[test]
    fn mapped_ghost_change_affects_only_linked_region() {
        let mut state = GhostFeaturesState {
            loaded: true,
            path: Some("features.json".to_string()),
            total_ghost_events: 20,
            spatial_mapped: true,
            spatial_status_text: "mapped_components_from_ghost_site_map_v1".to_string(),
            mapped_components: vec![
                component("near-a", 0.0, 0.9),
                component("near-b", 100.0, 0.4),
            ],
            exact_component_adapter_receipt: None,
        };
        let before_a = ghost_zstr_summary_for_geometry(&state, 0, [0.0, 0.0, 0.0], 2.0);
        let before_b = ghost_zstr_summary_for_geometry(&state, 1, [100.0, 0.0, 0.0], 2.0);
        assert_eq!(before_a.links.len(), 1);
        assert_eq!(before_b.links.len(), 1);
        state.mapped_components[0].ghost_tile_score_mean = 0.1;
        let after_a = ghost_zstr_summary_for_geometry(&state, 0, [0.0, 0.0, 0.0], 2.0);
        let after_b = ghost_zstr_summary_for_geometry(&state, 1, [100.0, 0.0, 0.0], 2.0);
        assert_ne!(before_a.factor, after_a.factor);
        assert_eq!(before_b.factor, after_b.factor);
        assert_eq!(after_a.links[0].component_id, "near-a");
        assert_eq!(after_b.links[0].component_id, "near-b");
    }

    #[test]
    fn exact_component_adapter_filter_uses_stream_and_inclusive_step_overlap() {
        let mut region_streams = std::collections::HashMap::new();
        region_streams.insert(7, 100);
        let component = exact_component("floor_aware_graph:4", 7, 10, 20);

        let (passes_at_boundary, receipt) =
            exact_component_adapter_region_filter_status(&component, &region_streams, 20, 30);
        assert!(passes_at_boundary);
        assert_eq!(receipt["status"].as_str(), Some("pass"));
        assert_eq!(
            receipt["predicate_semantics"].as_str(),
            Some("broad recorded step-window overlap only; not exact coincident causal event ownership")
        );

        let (fails_after_window, receipt) =
            exact_component_adapter_region_filter_status(&component, &region_streams, 21, 30);
        assert!(!fails_after_window);
        assert_eq!(
            receipt["status"].as_str(),
            Some("fail_no_inclusive_step_window_overlap")
        );

        let component = exact_component("floor_aware_graph:5", 8, 10, 20);
        let (fails_stream, receipt) =
            exact_component_adapter_region_filter_status(&component, &region_streams, 10, 20);
        assert!(!fails_stream);
        assert_eq!(
            receipt["status"].as_str(),
            Some("fail_stream_absent_from_region_per_stream_spikes")
        );
    }
}

#[cfg(test)]
mod measurement_semantics_tests {
    use super::*;

    #[test]
    fn observed_spike_bounds_survive_seed_merge_and_do_not_use_centroid_radius() {
        let mut first = SpikeSupportGeometry::default();
        first.observe([-4.0, 2.0, 9.0]);
        first.observe([1.0, 6.0, -3.0]);
        let mut second = SpikeSupportGeometry::default();
        second.observe([-8.0, 3.0, 2.0]);
        second.observe([7.0, -5.0, 11.0]);
        second.observe([f32::NAN, 0.0, 0.0]);
        first.merge(&second);
        assert_eq!(first.aabb_min, Some([-8.0, -5.0, -3.0]));
        assert_eq!(first.aabb_max, Some([7.0, 6.0, 11.0]));
        assert_eq!(first.attributed_record_count, 5);
        assert_eq!(first.nonfinite_position_count, 1);
        first.merge(&SpikeSupportGeometry::default());
        assert_eq!(first.attributed_record_count, 5);
        assert!(SpikeSupportGeometry::default().surface()["aabb_min"].is_null());
    }

    #[test]
    fn native_confidence_reuses_live_arithmetic_in_both_regimes() {
        for me in [0.0f32, 0.5, 2.0] {
            for bm in [-1.0f32, 0.0, 2.5, 8.0] {
                for lc in [-1.0f32, 0.25, 1.5] {
                    let causal_frac = 13.0f32 / 100.0;
                    let old = (me * causal_frac)
                        .max(bm.max(0.0).min(5.0) / 5.0 * lc.max(0.0).min(1.0))
                        * (1.0 + 0.1 * 0.25);
                    assert_eq!(
                        native_residue_kcc_confidence(me, 13, 100, bm, lc, 0.25).to_bits(),
                        old.to_bits()
                    );
                }
            }
        }
    }

    #[test]
    fn native_rank_precedes_morphology_and_preserves_legacy_order() {
        let mut a = LexicographicSiteRankKey {
            rank_policy_id: RANK_POLICY_ID_NATIVE_KCC,
            promoted: true,
            materialization_level_rank: 3,
            native_residue_kcc_confidence: Some(0.8),
            pocket_morphology_rank: 0.1,
            ..Default::default()
        };
        let mut b = a.clone();
        b.native_residue_kcc_confidence = Some(0.2);
        b.pocket_morphology_rank = 0.9;
        assert_eq!(
            compare_lexicographic_site_rank(&a, &b),
            std::cmp::Ordering::Less
        );
        a.rank_policy_id = RANK_POLICY_ID_V2;
        b.rank_policy_id = RANK_POLICY_ID_V2;
        assert_eq!(
            compare_lexicographic_site_rank(&a, &b),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            compare_available_desc(Some(0.0), None),
            std::cmp::Ordering::Less
        );
        let args = Args::try_parse_from([
            "materializer",
            "--manifest",
            "x",
            "--ranking-policy",
            "native-kcc",
        ])
        .unwrap();
        assert_eq!(args.ranking_policy, RankingPolicy::NativeKcc);
        assert_eq!(
            RankingPolicy::NativeKcc.key_order()[2],
            "native_residue_kcc_confidence desc (unavailable last)"
        );
    }

    #[test]
    fn return_geometry_changes_with_return_without_relabeling_hysteresis() {
        let mut manifold = CentroidManifold {
            whole_site: Some([0.0; 3]),
            lining_mass: Some([0.0; 3]),
            causal_driver: Some([0.0; 3]),
            hot_phase: Some([0.0; 3]),
            cold_phase: Some([0.0; 3]),
            return_phase: Some([0.0; 3]),
            ligand_adjacent: None,
        };
        assert_eq!(
            return_inclusive_geometry(&manifold)["consistency_factor"],
            1.0
        );
        manifold.return_phase = Some([4.0, 0.0, 0.0]);
        let observed = return_inclusive_geometry(&manifold);
        assert_eq!(observed["cold_to_return_distance_a"], 4.0);
        assert!((observed["consistency_factor"].as_f64().unwrap() - (-1.0f64).exp()).abs() < 1e-12);
        assert_eq!(
            observed["admission_use"],
            "none; original_five_view_manifold_preserved"
        );
    }

    #[test]
    fn native_region_join_is_stream_specific_and_keeps_unavailable_lag() {
        let mut k = decode_kcc_full(&wire_kcc()).unwrap();
        for (name, value) in [
            ("motion_efficiency", 0.4),
            ("burst_motion", 1.0),
            ("local_cov", 0.5),
        ] {
            k.decoded.f32_fields.insert(name.into(), vec![value, value]);
        }
        k.decoded
            .f32_fields
            .insert("causal_lag".into(), vec![f32::NAN, -40.0]);
        let support = [
            (0u32, [(0i32, 100u64)].into_iter().collect()),
            (1u32, [(1i32, 200u64)].into_iter().collect()),
        ]
        .into_iter()
        .collect();
        let joined = region_native_kcc(23, &support, &[k.clone()]);
        assert_eq!(joined[0].region_id, 23);
        assert_eq!(joined[0].selected_residue_index, Some(0));
        assert_eq!(joined[0].candidates[0].observation_count, 5500);
        assert!(joined[0].candidates[0].causal_lag.is_none());
        assert_eq!(
            joined[0].candidates[0].native_f32_bits["causal_lag"],
            "7fc00000"
        );
        assert!(joined[1].native_residue_confidence.is_none());
        assert!(joined[1].candidates.is_empty());
        k.decoded.f32_fields.get_mut("motion_efficiency").unwrap()[0] = f32::NAN;
        assert!(region_native_kcc(23, &support, &[k.clone()])[0]
            .native_residue_confidence
            .is_none());
        assert!(region_native_kcc(23, &support, &[k.clone(), k])[0]
            .candidates
            .is_empty());
    }

    #[test]
    fn stream_support_policy_is_explicit_positive_and_backward_compatible() {
        let default = Args::try_parse_from(["materializer", "--manifest", "input.json"]).unwrap();
        assert_eq!(default.min_supporting_streams, 3);
        let single = Args::try_parse_from([
            "materializer",
            "--manifest",
            "input.json",
            "--min-supporting-streams",
            "1",
        ])
        .unwrap();
        assert_eq!(single.min_supporting_streams, 1);
        assert_eq!(single.materialized_top_n, default.materialized_top_n);
        assert!(Args::try_parse_from([
            "materializer",
            "--manifest",
            "input.json",
            "--min-supporting-streams",
            "0",
        ])
        .is_err());
    }

    #[test]
    fn tied_density_and_residue_support_do_not_depend_on_insertion_order() {
        for rotation in 0..16 {
            let mut voxels = std::collections::HashMap::new();
            let mut residues = std::collections::HashMap::new();
            for offset in 0..16 {
                let id = ((rotation + offset) % 16) as i32;
                voxels.insert((id, 0, 0), 100u32);
                residues.insert(id, 100u32);
            }
            let peaks = extract_top_density_peaks(&voxels, 2.0, 3);
            assert_eq!(
                peaks.iter().map(|p| p.voxel_index).collect::<Vec<_>>(),
                vec![[0, 0, 0], [1, 0, 0], [2, 0, 0]]
            );
            let mut shell: Vec<_> = residues.into_iter().collect();
            shell.sort_by(count_desc_identity_asc);
            assert_eq!(
                shell.into_iter().take(12).map(|p| p.0).collect::<Vec<_>>(),
                (0..12).collect::<Vec<_>>()
            );
        }
    }

    fn schedule() -> ThermalSchedule {
        ThermalSchedule {
            stream_index: 0,
            cold_hold_end: 14000,
            ramp_end: 20000,
            warm_hold_end: 35000,
            ramp_down_end: 41000,
            total_steps: 55000,
            current_step: Some(60000),
            steering_flags: 0,
        }
    }

    #[test]
    fn protocol_boundaries_do_not_split_warm_hold_at_half_cycle() {
        let s = schedule();
        s.validate(0).unwrap();
        for (step, phase) in [
            (0, 0),
            (13999, 0),
            (14000, 1),
            (19999, 1),
            (20000, 2),
            (27499, 2),
            (27500, 2),
            (34999, 2),
            (35000, 3),
            (40999, 3),
            (41000, 4),
            (55000, 4),
            (-1, 6),
            (55001, 4),
            (60000, 4),
            (60001, 6),
        ] {
            assert_eq!(s.phase(step), phase, "step={step}");
        }
    }

    #[test]
    fn warm_tail_flag_does_not_fabricate_cold_return() {
        let mut s = schedule();
        s.steering_flags = 0x10;
        assert_eq!(s.phase(41000), 5);
        assert_eq!(s.phase(55000), 5);
        assert_eq!(s.phase(35000), 3);
    }

    #[test]
    fn terminal_hold_matches_native_director_and_never_accepts_future_events() {
        let mut s = schedule();
        // Independent source oracle: protocol_director.cu has a final else
        // selecting start_temp (or end_temp with flag 0x10), not a total-step
        // cutoff or modulo reset. Sidecar current_step proves execution extent.
        assert_eq!(s.phase(55001), 4);
        s.steering_flags = 0x10;
        assert_eq!(s.phase(55001), 5);
        assert_eq!(s.phase(60000), 5);
        assert_eq!(s.phase(60001), 6);
        s.current_step = Some(10112);
        assert_eq!(s.phase(10112), 0);
        assert_eq!(s.phase(10113), 6);
        s.current_step = None;
        assert_eq!(s.phase(55001), 6);
        s.current_step = Some(-1);
        assert!(s.validate(0).is_err());
    }

    #[test]
    fn invalid_missing_and_cross_run_schedules_are_rejected() {
        assert!(parse_thermal_schedule(serde_json::json!({}), 0, "run").is_err());
        let mut s = schedule();
        s.ramp_end = 13999;
        assert!(s.validate(0).is_err());
        assert!(schedule().validate(1).is_err());
        let mut value = serde_json::to_value(schedule()).unwrap();
        value["run_id"] = serde_json::json!("other_run");
        assert!(parse_thermal_schedule(value, 0, "run").is_err());
    }

    #[test]
    fn thermal_merge_preserves_all_phase_counts_and_return_centroid() {
        let mut a = ThermalEvidence::default();
        let mut b = ThermalEvidence::default();
        for _ in 0..25 {
            a.observe(4, [1.0, 2.0, 3.0]);
            b.observe(4, [3.0, 4.0, 5.0]);
        }
        b.observe(6, [9.0, 9.0, 9.0]);
        assert!(a.centroid(4).is_none());
        a.merge(&b);
        assert_eq!(a.centroid(4), Some([2.0, 3.0, 4.0]));
        assert_eq!(a.counts.iter().sum::<u64>(), 51);
        assert_eq!(a.counts[6], 1);
        assert_eq!(a.counts[2], 0);
    }

    // Real PRKCC001 wire layout, including checksum/trailer. Exercises the
    // production decoder and typed field preservation rather than a JSON mock.
    fn wire_kcc() -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(&2u64.to_le_bytes());
        payload.extend_from_slice(&3u64.to_le_bytes());
        for (name, dtype, values) in [
            (
                "temporal_corr",
                1u8,
                vec![(-0.031_f32).to_bits(), 0x7fc00123],
            ),
            ("residue_count", 2u8, vec![5500, 0]),
            ("active_causal", 2u8, vec![42, 0]),
        ] {
            payload.extend_from_slice(&(name.len() as u64).to_le_bytes());
            payload.extend_from_slice(name.as_bytes());
            payload.push(dtype);
            payload.extend_from_slice(&8u64.to_le_bytes());
            for v in values {
                payload.extend_from_slice(&v.to_le_bytes());
            }
        }
        let mut raw = b"PRKCC001".to_vec();
        raw.extend_from_slice(&1u32.to_le_bytes());
        raw.extend_from_slice(&0x01020304u32.to_le_bytes());
        raw.extend_from_slice(&0u32.to_le_bytes());
        for s in ["run", "target"] {
            raw.extend_from_slice(&(s.len() as u64).to_le_bytes());
            raw.extend_from_slice(s.as_bytes());
        }
        raw.extend_from_slice(&2u64.to_le_bytes());
        raw.extend_from_slice(&4u64.to_le_bytes());
        raw.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        raw.extend_from_slice(&payload);
        raw.extend_from_slice(&fnv1a_64(&payload).to_le_bytes());
        raw.extend_from_slice(b"PRKCCEND");
        raw
    }

    #[test]
    fn lag_sidecar_binds_source_identity_counts_and_forwards_descriptors() {
        let dir = std::env::temp_dir().join(format!("prism_kcc_lag_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let source = dir.join("source.bin");
        std::fs::write(&source, wire_kcc()).unwrap();
        let mut k = read_kcc_v2full(&source).unwrap();
        assert!(k.lag_provenance.is_empty());
        let source_sha = prism_nhs::ensemble::manifest::sha256_file(&source).unwrap();
        let rows: Vec<_> = (0..k.n_residues)
            .map(|r| {
                let mut p = prism_nhs::causome::KccLagProvenance::legacy_unknown(
                    k.decoded.u32_fields["residue_count"][r],
                    k.active_causal[r],
                );
                p.source_stream_id = Some(k.stream_id);
                p
            })
            .collect();
        let mut v = serde_json::json!({"schema_kind":"prism.kcc_lag_provenance.v1","schema_version":1,"run_id":"run","stem":"target","stream_id":0,"n_residues":2,
            "kcc_v2full_sha256":source_sha,"rows":rows});
        let sidecar = dir.join("sidecar.json");
        std::fs::write(&sidecar, serde_json::to_vec(&v).unwrap()).unwrap();
        attach_kcc_lag_provenance(&mut k, &sidecar).unwrap();
        assert_eq!(
            kcc_descriptor_surface(&k, "source")["lag_provenance"][0]["observation_count"],
            5500
        );
        let mut support_rows: Vec<_> = (0..k.n_residues)
            .map(|r| {
                let mut row = prism_nhs::causome::KccLagSupportResidue::legacy_unknown(
                    r as u32,
                    k.decoded.u32_fields["residue_count"][r],
                    k.active_causal[r],
                    f32::NAN.to_bits(),
                    f32::NAN.to_bits(),
                );
                row.source_stream_id = Some(k.stream_id);
                row.ring_head = Some(0);
                row.ring_event_count = Some(0);
                row.ring_raw_slots_hex = Some("00".repeat(64 * 20));
                row
            })
            .collect();
        let support = dir.join("support.json");
        let support_archive = prism_nhs::causome::KccLagSupportArchive::new(
            "run",
            "target",
            0,
            None,
            "test",
            Some(prism_nhs::ensemble::manifest::sha256_file(&source).unwrap()),
            support_rows.clone(),
        );
        std::fs::write(&support, serde_json::to_vec(&support_archive).unwrap()).unwrap();
        attach_kcc_lag_support(&mut k, &support).unwrap();
        let surface = kcc_descriptor_surface(&k, "source");
        assert_eq!(surface["lag_support_summary"]["rows_present"], 2);
        v["lag_support_ref"] = serde_json::json!({
            "schema_kind":"prism.kcc_lag_support.v1",
            "path":support.display().to_string(),
            "sha256":prism_nhs::ensemble::manifest::sha256_file(&support).unwrap(),
        });
        std::fs::write(&sidecar, serde_json::to_vec(&v).unwrap()).unwrap();
        attach_kcc_lag_provenance(&mut k, &sidecar).unwrap();
        k.lag_provenance_source = None;
        support_rows[0].ring_raw_slots_hex = Some("00".repeat(64 * 20 - 1));
        let bad_archive = prism_nhs::causome::KccLagSupportArchive::new(
            "run",
            "target",
            0,
            None,
            "test",
            Some(prism_nhs::ensemble::manifest::sha256_file(&source).unwrap()),
            support_rows,
        );
        std::fs::write(&support, serde_json::to_vec(&bad_archive).unwrap()).unwrap();
        assert!(attach_kcc_lag_support(&mut k, &support).is_err());
        v["rows"][0]["observation_count"] = serde_json::json!(5499);
        std::fs::write(&sidecar, serde_json::to_vec(&v).unwrap()).unwrap();
        assert!(attach_kcc_lag_provenance(&mut k, &sidecar).is_err());
        v["rows"][0]["observation_count"] = serde_json::json!(5500);
        v["kcc_v2full_sha256"] = serde_json::json!("wrong");
        std::fs::write(&sidecar, serde_json::to_vec(&v).unwrap()).unwrap();
        assert!(attach_kcc_lag_provenance(&mut k, &sidecar).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn full_kcc_decoder_preserves_signed_nonfinite_bits_and_sample_counts() {
        let k = decode_kcc_full(&wire_kcc()).unwrap();
        validate_kcc_association(&k, "run", "target", Some(0)).unwrap();
        assert!(validate_kcc_association(&k, "other", "target", Some(0)).is_err());
        assert!(validate_kcc_association(&k, "run", "other", Some(0)).is_err());
        assert!(validate_kcc_association(&k, "run", "target", Some(1)).is_err());
        assert_eq!(k.decoded.u32_fields["residue_count"], vec![5500, 0]);
        let v = kcc_descriptor_surface(&k, "source");
        assert!(
            v["f32_fields"]["temporal_corr"]["values"][0]["value"]
                .as_f64()
                .unwrap()
                < 0.0
        );
        assert_eq!(
            v["f32_fields"]["temporal_corr"]["values"][1]["classification"],
            "nan"
        );
        assert_eq!(
            v["f32_fields"]["temporal_corr"]["values"][1]["ieee754_bits_hex"],
            "7fc00123"
        );
        assert!(v["f32_fields"].get("lag").is_none());
        assert_eq!(v["sample_count_field"], "residue_count");
    }

    #[test]
    fn full_kcc_decoder_rejects_corruption_truncation_and_trailing_bytes() {
        let raw = wire_kcc();
        assert!(decode_kcc_full(&raw[..raw.len() - 1]).is_err());
        let mut corrupted = raw.clone();
        let n = corrupted.len();
        corrupted[n - 17] ^= 1;
        assert!(decode_kcc_full(&corrupted).is_err());
        let mut trailing = raw;
        trailing.push(0);
        assert!(decode_kcc_full(&trailing).is_err());
        assert_eq!(
            kcc_f32_value(f32::INFINITY)["classification"],
            "positive_infinity"
        );
        assert_eq!(
            kcc_f32_value(f32::NEG_INFINITY)["classification"],
            "negative_infinity"
        );
        assert_eq!(kcc_f32_value(-0.0)["ieee754_bits_hex"], "80000000");
    }

    fn dt_records(rows: &[(u64, u64, f64)]) -> Vec<u8> {
        let mut raw = Vec::new();
        for (chunk, step, dt) in rows {
            raw.extend_from_slice(&chunk.to_le_bytes());
            raw.extend_from_slice(&step.to_le_bytes());
            raw.extend_from_slice(&dt.to_le_bytes());
            raw.extend_from_slice(&[0u8; 8]);
        }
        raw
    }

    #[test]
    fn physical_time_requires_complete_contiguous_positive_constant_dt() {
        let mut s = schedule();
        s.cold_hold_end = 0;
        s.ramp_end = 10;
        s.warm_hold_end = 20;
        s.ramp_down_end = 30;
        s.total_steps = 30;
        s.validate(0).unwrap();
        let v = adaptive_dt_time_axis_bytes(
            &dt_records(&[(0, 0, 0.002), (1, 10, 0.002), (2, 20, 0.002)]),
            Some(&s),
        )
        .unwrap();
        assert_eq!(v["fixed_dt_ps"], 0.002);
        assert_eq!(v["recorded_step_cadence"], 10);
        for rows in [
            vec![(0, 0, 0.002), (2, 20, 0.002)],
            vec![(0, 10, 0.002), (1, 20, 0.002)],
            vec![(0, 0, 0.002), (1, 10, 0.001), (2, 20, 0.002)],
            vec![(0, 0, 0.002), (1, 10, f64::NAN), (2, 20, 0.002)],
        ] {
            assert!(
                adaptive_dt_time_axis_bytes(&dt_records(&rows), Some(&s)).unwrap()["fixed_dt_ps"]
                    .is_null()
            );
        }
        assert!(adaptive_dt_time_axis_bytes(&[], Some(&s)).is_err());
        assert!(adaptive_dt_time_axis_bytes(&[0u8; 31], Some(&s)).is_err());
    }
    #[test]
    fn required_kcc_counters_cannot_be_silently_absent() {
        let raw = wire_kcc();
        let decoded = prism_tokenizer::codec::kcc::decode_prkcc001(&raw).unwrap();
        for field in ["active_causal", "residue_count"] {
            let mut changed = raw.clone();
            let offset = changed
                .windows(field.len())
                .position(|s| s == field.as_bytes())
                .unwrap();
            changed[offset] = b'X';
            let start = decoded.header.payload_offset;
            let end = start + decoded.header.payload_size;
            let checksum = fnv1a_64(&changed[start..end]);
            changed[end..end + 8].copy_from_slice(&checksum.to_le_bytes());
            assert!(decode_kcc_full(&changed).is_err(), "missing {field}");
        }
        let k = decode_kcc_full(&raw).unwrap();
        let surface = kcc_descriptor_surface(&k, "source");
        assert_eq!(
            surface["missing_expected_fields"].as_array().unwrap().len(),
            11
        );
    }

    #[test]
    fn generic_envelope_identity_is_an_admission_requirement() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("target_stream0_kcc.bin");
        std::fs::write(&path, wire_kcc()).unwrap();
        for (run, stem, sid, expected) in [
            ("run", "target", Some(0), true),
            ("other", "target", Some(0), false),
            ("run", "other", Some(0), false),
            ("run", "target", Some(1), false),
        ] {
            let (v, _) = validate_artifact(
                &path,
                b"PRKCC001",
                b"PRKCCEND",
                "kcc_v2full",
                Some(2),
                None,
                run,
                stem,
                sid,
                false,
            )
            .unwrap();
            assert_eq!(v.valid, expected);
        }
    }
}

#[cfg(test)]
mod unique_record_support_tests {
    use super::*;
    #[test]
    fn overlapping_peaks_become_one_region_update_with_exact_multiplicity() {
        let input =
            std::collections::HashMap::from([((0, 0, 0), vec![0, 1, 2]), ((1, 0, 0), vec![1])]);
        let peak_region = std::collections::HashMap::from([(0, 0usize), (1, 0usize), (2, 1usize)]);
        let actual = region_attribution_lookup(&input, &peak_region);
        assert_eq!(actual[&(0, 0, 0)], vec![(0, 2), (1, 1)]);
        assert_eq!(actual[&(1, 0, 0)], vec![(0, 1)]);
    }
    fn record(step: i32) -> Vec<u8> {
        let mut r = vec![0u8; 96];
        r[0..4].copy_from_slice(&step.to_le_bytes());
        r[spike_offsets::INTENSITY..spike_offsets::INTENSITY + 4]
            .copy_from_slice(&2.0f32.to_le_bytes());
        r[24..28].copy_from_slice(&0i32.to_le_bytes());
        r[28..32].copy_from_slice(&9i32.to_le_bytes());
        r[56..60].copy_from_slice(&2i32.to_le_bytes());
        r
    }
    #[test]
    fn unique_record_counts_and_phase_votes_remain_separate() {
        let r = record(14000);
        let mut unique = CandidateEvidence::new(0, [0; 3], [0.0; 3]);
        observe_unique_record(&mut unique, &r, 0, 1, 0.5, 100);
        let mut phase = PhaseRecordSupport::default();
        phase.observe(&r, 31, 3);
        assert_eq!(unique.n_spikes, 1);
        assert_eq!(unique.residue_support[&0], 1);
        assert_eq!(unique.residue_support[&9], 1);
        assert_eq!(unique.thermal.counts[1], 1);
        assert_eq!(phase.count, 1);
        assert_eq!(phase.weighted_votes, 3);
        assert_eq!(phase.multiplicities[&3], 1);
        assert_eq!(phase.first_ordinal, Some(31));
    }
    #[test]
    fn phase_strata_preserve_native_boundary_and_cold_return() {
        let schedule = ThermalSchedule {
            stream_index: 0,
            cold_hold_end: 14000,
            ramp_end: 20000,
            warm_hold_end: 35000,
            ramp_down_end: 41000,
            total_steps: 55000,
            current_step: None,
            steering_flags: 0,
        };
        assert_eq!(schedule.phase(13999), 0);
        assert_eq!(schedule.phase(14000), 1);
        assert_eq!(schedule.phase(35000), 3);
        assert_eq!(schedule.phase(41000), 4);
        assert_eq!(schedule.phase(55000), 4);
        assert_eq!(schedule.phase(55001), 6);
    }
}

#[cfg(test)]
mod recorded_transition_support_tests {
    use super::*;
    #[test]
    fn recorded_transition_uses_executed_clock_and_rejects_future() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("events.jsonl");
        let mut event = serde_json::json!({"schema_kind":"SPIKE_TRANSITION_EVENT","schema_version":1,"stream_id":0,"step":40,"trigger_kind":"THERMAL_PHASE"});
        let hash = blake3::hash(&serde_json::to_vec(&event).unwrap()).to_hex().to_string();
        event["event_hash_blake3"] = hash.into();
        std::fs::write(&path, serde_json::to_vec(&event).unwrap()).unwrap();
        let surfaces = vec![serde_json::json!({"kind":"spike_transition_events","status":"materialized_parsed","stream_id":0,"source":{"resolved_path":path,"manifest_path":"native"}})];
        let mut schedule = ThermalSchedule { stream_index:0,cold_hold_end:2,ramp_end:4,warm_hold_end:6,ramp_down_end:8,total_steps:35,current_step:Some(40),steering_flags:0 };
        let mut schedules = std::collections::HashMap::new();
        schedules.insert(0,schedule.clone());
        let windows = load_recorded_transition_windows(&surfaces,&schedules,"native").unwrap();
        assert_eq!(windows[0].end_step_exclusive,41);
        assert_eq!(transition_window_index(&windows,0,40),Some(0));
        assert_eq!(transition_window_index(&windows,0,41),None);
        schedule.current_step=Some(39);schedules.insert(0,schedule.clone());
        assert!(load_recorded_transition_windows(&surfaces,&schedules,"native").is_err());
        schedule.current_step=None;schedules.insert(0,schedule);
        assert!(load_recorded_transition_windows(&surfaces,&schedules,"native").is_err());
    }

    fn window(start: i64, end: i64, stream: u32) -> RecordedTransitionWindow {
        RecordedTransitionWindow {
            stream_id: stream,
            start_step: start,
            end_step_exclusive: end,
            anchors: vec![],
            identity: serde_json::json!({"step":start}),
        }
    }
    #[test]
    fn completed_event_boundaries_are_half_open_and_stream_specific() {
        let windows = vec![window(10, 20, 0), window(20, 55001, 0), window(0, 101, 1)];
        assert_eq!(transition_window_index(&windows, 0, 9), None);
        assert_eq!(transition_window_index(&windows, 0, 10), Some(0));
        assert_eq!(transition_window_index(&windows, 0, 20), Some(1));
        assert_eq!(transition_window_index(&windows, 0, 55000), Some(1));
        assert_eq!(transition_window_index(&windows, 0, 55001), None);
        assert_eq!(transition_window_index(&windows, 1, 10), Some(2));
    }
    #[test]
    fn zero_support_window_is_preserved_with_null_measurements_and_bound_hash() {
        let record =
            transition_support_record(serde_json::json!({"anchor_hashes":["a","b"]}), None)
                .unwrap();
        assert_eq!(record["unique_record_count"], 0);
        assert!(record["spike_position_centroid_a"].is_null());
        let mut body = record.clone();
        let hash = body
            .as_object_mut()
            .unwrap()
            .remove("support_record_sha256")
            .unwrap();
        assert_eq!(
            hash,
            serde_json::json!(prism_nhs::chemical_state_fiber::canonical_sha256(&body).unwrap())
        );
    }
}

#[cfg(test)]
mod author_topology_availability_tests {
    use super::*;

    #[test]
    fn missing_metadata_is_distinct_from_malformed_topology() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("topology.json");
        assert!(!author_contract_present_in_topology(&path).unwrap());
        for text in ["{}", "{\"analysis_residue_contract\":null}"] {
            std::fs::write(&path, text).unwrap();
            assert!(!author_contract_present_in_topology(&path).unwrap());
        }
        std::fs::write(&path, "{\"analysis_residue_contract\":{}}").unwrap();
        assert!(author_contract_present_in_topology(&path).unwrap());
        for text in ["{\"analysis_residue_contract\":", "not json", "null", "[]"] {
            std::fs::write(&path, text).unwrap();
            assert!(author_contract_present_in_topology(&path).is_err());
        }
    }
}

// Offline recovery is a separate native operation, never a replayed runtime epoch.
#[derive(Parser, Debug)]
struct RecoveryArgs {
    #[arg(long)] run_dir: PathBuf,
    #[arg(long)] topology: PathBuf,
    #[arg(long)] execution_receipt: PathBuf,
    #[arg(long)] manifest: PathBuf,
    #[arg(long, default_value_t = 8)] expected_streams: u32,
}

fn recovery_json(path: &Path) -> Result<serde_json::Value> {
    Ok(serde_json::from_reader(File::open(path)?)?)
}
fn recovery_regular(path: &Path) -> Result<()> {
    let m = std::fs::symlink_metadata(path)?;
    anyhow::ensure!(m.is_file() && !m.file_type().is_symlink(), "recovery requires regular file: {}", path.display());
    Ok(())
}
fn recovery_hash_check(bytes: &[u8], receipt: &serde_json::Value, key: &str) -> Result<()> {
    let expected = receipt[key].as_str().with_context(|| format!("receipt missing {key}"))?;
    anyhow::ensure!(blake3::hash(bytes).to_hex().as_str() == expected, "recovery {key} mismatch");
    Ok(())
}
fn recovery_identity(header: &EnvelopeHeader, run: &str, target: &str, stream: u32) -> Result<()> {
    anyhow::ensure!(header.run_id == run && header.stem == target && header.stream_id == stream,
        "recovery envelope run/target/stream mismatch");
    anyhow::ensure!(header.schema_version == 1 && (header.magic != *b"PRSPK001" || header.record_count.checked_mul(header.byte_stride) == Some(header.payload_size)),
        "recovery envelope schema/record dimensions mismatch");
    Ok(())
}
fn recovery_kind(name: &str) -> &'static str {
    // Same filename contracts consumed by the production materializer. Unknown
    // retained surfaces remain inventoried, without inventing scientific meaning.
    const SUFFIXES: &[(&str, &str)] = &[
        ("_spikes.bin", "spikes"), ("_signal_grid.bin", "signal_grid"),
        ("_kcc_v2full.bin", "kcc_v2full"), ("_warp_matrix.bin", "warp_matrix"),
        ("_forces_final.bin", "forces_final"), ("_aromatic_centroids_final.bin", "aromatic_centroids_final"),
        ("_adaptive_dt.bin", "adaptive_dt"), ("_bocpd.jsonl", "bocpd_log"),
        ("_spike_transition_events.jsonl", "spike_transition_events"),
        ("_protocol_state.json", "protocol_state"), ("_asc_vectors.bin", "asc_vectors"),
        ("_noise_floor.json", "noise_floor"), (".frames.bin", "trajectory_frames"),
        ("_physical_time.jsonl", "physical_time_journal"),
        ("_physical_energy_balance.jsonl", "physical_energy_balance_journal"),
        ("_kcc_v2full.lag_support.json", "kcc_lag_support"),
        ("_kcc_v2full.lag_provenance.json", "kcc_lag_provenance"),
        (".transform_dag.json", "transform_dag"), ("_binding_sites_lattice.json", "ghost_phase_lattice_components"),
        ("ghost_site_map.json", "ghost_site_map"), ("ghost_time_map.json", "ghost_time_map"),
        ("thermodynamic_phase_2x2x2x2_condition_ledger.json", "thermodynamic_phase_2x2x2x2_condition_ledger"),
        ("TWIN16_INTERFEROMETRIC_TRANSITION_VIEW_RESIDUALS.json", "twin16_interferometric_transition_view_residuals"),
        ("bounded_thermo_temporal_ccns_sdst_summary.json", "bounded_thermo_temporal_ccns_sdst_summary"),
    ];
    SUFFIXES.iter().find(|(suffix, _)| name.ends_with(suffix)).map(|(_,kind)| *kind).unwrap_or("retained_native_artifact")
}
fn recovery_complete_protocol(value: serde_json::Value, stream: u32, run: &str) -> Result<()> {
    anyhow::ensure!(value["status_code"].as_u64() == Some(0), "recovery protocol failed status");
    let schedule = parse_thermal_schedule(value, stream, run)?;
    anyhow::ensure!(schedule.current_step.is_some_and(|step| step >= schedule.total_steps),
        "recovery protocol has not completed its recorded horizon");
    Ok(())
}
fn recover_manifest(args: RecoveryArgs) -> Result<()> {
    use std::io::Write;
    use std::collections::BTreeSet;
    anyhow::ensure!(args.expected_streams > 0, "expected stream count must be positive");
    anyhow::ensure!(!args.manifest.exists(), "refusing to replace existing manifest");
    let root = std::fs::canonicalize(&args.run_dir)?;
    anyhow::ensure!(std::fs::canonicalize(args.manifest.parent().context("manifest parent")?)? == root,
        "recovered manifest must be in original run directory");
    recovery_regular(&args.execution_receipt)?;
    recovery_regular(&args.topology)?;
    let execution = recovery_json(&args.execution_receipt)?;
    anyhow::ensure!(execution["status"] == "FAILED" && execution["exit_status"].as_i64().is_some_and(|rc| rc != 0),
        "recovery requires original failed execution receipt");
    let target = execution["target"].as_str().context("execution target missing")?.to_string();
    let topology = std::fs::canonicalize(&args.topology)?;
    let recorded_topology = execution["topology"].as_str().context("execution topology missing")?;
    anyhow::ensure!(std::fs::canonicalize(recorded_topology)? == topology, "topology source mismatch");
    let topology_hash = prism_nhs::ensemble::manifest::sha256_file(&topology)?;
    anyhow::ensure!(execution["topology_sha256"].as_str() == Some(topology_hash.as_str()), "topology hash mismatch");
    TopologyMin::load(&topology)?;
    let steps = execution["steps"].as_array().context("execution steps missing")?;
    let engine = steps.iter().find(|s| s["stage"] == "engine").context("engine execution entry missing")?;
    let command = engine["command"].as_array().context("engine command missing")?;
    let output_arg = command.windows(2).find(|v| v[0] == "-o").and_then(|v| v[1].as_str()).context("engine output argument missing")?;
    anyhow::ensure!(std::fs::canonicalize(output_arg)? == root, "execution output path mismatch");
    let expected_arg = command.windows(2).find(|v| v[0] == "--multi-stream").and_then(|v| v[1].as_str()).context("engine stream count missing")?;
    anyhow::ensure!(expected_arg.parse::<u32>()? == args.expected_streams, "requested stream count differs from execution");
    let hydration = root.join("PRISMOS_QNANL_HYDRATED_INPUTS");
    std::fs::create_dir_all(&hydration)?;
    let mut packets = Vec::new();
    for entry in std::fs::read_dir(&root)? {
        let path = entry?.path();
        let name = path.file_name().and_then(|v|v.to_str()).context("nonUTF8 run filename")?;
        if name.ends_with(".prismnanl") || name.ends_with(".prismqnanl") {packets.push(path);}
    }
    packets.sort();
    anyhow::ensure!(!packets.is_empty(), "no retained canonical packets");
    let mut artifacts = Vec::new();
    let mut bindings = Vec::new();
    let mut raw_names = BTreeSet::new();
    let mut stream_kinds = BTreeSet::new();
    let mut run_id: Option<String> = None;
    let mut total = 0u64;
    // Derive the run identity from retained spike envelope prefix, not logs or a new ID.
    for packet in &packets {
        if packet.extension().and_then(|v|v.to_str()) == Some("prismnanl") {
            let raw = decode_spike_prismnanl_packet(packet)?;
            let h = read_envelope_header(&mut std::io::Cursor::new(&raw))?;
            run_id = Some(h.run_id); break;
        }
    }
    let run_id = run_id.context("no spike packet run identity")?;
    for packet in &packets {
        recovery_regular(packet)?;
        let receipt_path = PathBuf::from(format!("{}.receipt.json",packet.display()));
        recovery_regular(&receipt_path)?;
        let receipt = recovery_json(&receipt_path)?;
        anyhow::ensure!(receipt["status"] == "PASS" && receipt["roundtrip_verified"] == true, "unverified packet receipt");
        let packet_bytes = std::fs::read(packet)?;
        recovery_hash_check(&packet_bytes,&receipt,"packet_hash_blake3")?;
        anyhow::ensure!(receipt["packet_bytes"].as_u64() == Some(packet_bytes.len() as u64), "packet byte count mismatch");
        let recorded_packet = receipt["packet_path"].as_str().context("packet receipt path missing")?;
        anyhow::ensure!(std::fs::canonicalize(recorded_packet)? == *packet, "receipt packet path mismatch");
        let spike = packet.extension().and_then(|v|v.to_str()) == Some("prismnanl");
        let original = PathBuf::from(receipt[if spike {"raw_path"} else {"source_path"}].as_str().context("receipt source path missing")?);
        anyhow::ensure!(original.parent() == Some(root.as_path()), "receipt source outside original run root");
        let name = original.file_name().and_then(|v|v.to_str()).context("source filename missing")?;
        anyhow::ensure!(raw_names.insert(name.to_string()), "duplicate recovered source identity");
        let kind = recovery_kind(name);
        anyhow::ensure!(spike == (kind == "spikes") && canonical_packet_path_for(&original,kind)? == *packet,
            "packet filename/family/source mismatch");
        let decoded = if spike {decode_spike_prismnanl_packet(packet)?} else {
            // Existing decoder validates the canonical packet seal and transforms.
            prismcode::qnanl_artifact::decode_file(packet,&hydration.join(name))?
        };
        recovery_hash_check(&decoded,&receipt,"raw_hash_blake3")?;
        anyhow::ensure!(receipt["raw_bytes"].as_u64() == Some(decoded.len() as u64), "raw byte count mismatch");
        let hydrated = hydration.join(name);
        // Always replace cache bytes from freshly verified canonical packets.
        std::fs::write(&hydrated,&decoded)?;
        let stream = parse_stream_id_from_filename(&original);
        if let Some(s) = stream {anyhow::ensure!(s < args.expected_streams,"unexpected stream identity");}
        let mut count = None;
        let mut checksum = if ["kcc_lag_provenance", "kcc_lag_support"].contains(&kind) {
            Some(format!("{:016x}", fnv1a_64(&decoded)))
        } else { None };
        let mut version = None;
        if ["spikes","signal_grid","kcc_v2full"].contains(&kind) {
            let h = read_envelope_header(&mut std::io::Cursor::new(&decoded))?;
            let s = stream.context("envelope filename stream missing")?;
            recovery_identity(&h,&run_id,&target,s)?;
            let (magic,trailer) = match kind {"spikes" => (*b"PRSPK001",*b"PRSPKEND"),"signal_grid" => (*b"PRSGD001",*b"PRSGDEND"),_ => (*b"PRKCC001",*b"PRKCCEND")};
            let (validation,_) = validate_artifact(&hydrated,&magic,&trailer,kind,Some(h.record_count),None,&run_id,&target,Some(s),false)?;
            anyhow::ensure!(validation.valid,"recovered envelope validation failed: {:?}",validation.failure_reason);
            if kind == "spikes" {
                anyhow::ensure!(h.byte_stride == 96 && receipt["record_count"].as_u64() == Some(h.record_count),"spike dimensions/receipt count mismatch");
                total = total.checked_add(h.record_count).context("total count overflow")?;
            }
            anyhow::ensure!(stream_kinds.insert((s,kind.to_string())),"duplicate stream/kind");
            count=Some(h.record_count);checksum=Some(validation.payload_checksum_fnv1a_64_hex);version=Some(h.schema_version);
        }
        if kind == "protocol_state" {
            let s = stream.context("protocol stream missing")?;
            recovery_complete_protocol(serde_json::from_slice(&decoded)?,s,&run_id)?;
            anyhow::ensure!(stream_kinds.insert((s,kind.to_string())),"duplicate protocol stream");
        }
        artifacts.push(serde_json::json!({"kind":kind,"stream_id":stream,"path":original,"size_bytes":decoded.len(),"schema_version":version,"checksum_fnv1a_64_hex":checksum,"record_count":count,"present":true,"note":"native offline recovery from freshly verified canonical packet"}));
        bindings.push(serde_json::json!({"source":original,"packet":packet,"receipt":receipt_path,"raw_hash_blake3":receipt["raw_hash_blake3"],"packet_hash_blake3":receipt["packet_hash_blake3"],"receipt_sha256":prism_nhs::ensemble::manifest::sha256_file(&receipt_path)?}));
    }
    for s in 0..args.expected_streams {
        for kind in ["spikes","signal_grid","kcc_v2full","protocol_state"] {
            anyhow::ensure!(stream_kinds.contains(&(s,kind.to_string())),"missing required stream {s} artifact {kind}");
        }
    }
    let lattice_path=root.join(format!("{target}_binding_sites_lattice.json"));
    recovery_regular(&lattice_path)?;
    let lattice=recovery_json(&lattice_path)?;
    anyhow::ensure!(lattice["run_id"]==run_id && lattice["target"]==target && lattice["backend"]=="ghost_phase_lattice_4d","native lattice association mismatch");
    anyhow::ensure!(lattice["topology_input"].as_str().is_some_and(|p|std::fs::canonicalize(p).ok().as_ref()==Some(&topology)),"lattice topology mismatch");
    let ignition_path=root.join(format!("{target}.v2_ignition_summary.json"));
    recovery_regular(&ignition_path)?;
    let ignition=recovery_json(&ignition_path)?;
    anyhow::ensure!(ignition["v2_ignition"]==true && ignition["n_streams"].as_u64()==Some(args.expected_streams as u64),"V2 ignition witness mismatch");
    // Preserve unpacketized surfaces in the inventory as observed, without synthesizing missing files.
    for entry in std::fs::read_dir(&root)? {
        let path=entry?.path();let name=path.file_name().and_then(|v|v.to_str()).context("filename")?;
        if !path.is_file() || name.contains(".prismnanl") || name.contains(".prismqnanl") || raw_names.contains(name) {continue;}
        recovery_regular(&path)?;
        artifacts.push(serde_json::json!({"kind":recovery_kind(name),"stream_id":parse_stream_id_from_filename(&path),"path":path,"size_bytes":path.metadata()?.len(),"schema_version":null,"checksum_fnv1a_64_hex":null,"record_count":null,"present":true,"note":"retained unpacketized native surface; SHA256 in recovered binding","sha256":prism_nhs::ensemble::manifest::sha256_file(&path)?}));
    }
    let manifest=serde_json::json!({"schema_version":1,"schema_kind":"md_evidence_manifest","run_id":run_id,"target":target,"topology_input":topology,"topology_sha256":topology_hash,"stream_count":args.expected_streams,"streams_serialized":args.expected_streams,"total_spikes_md":total,"artifacts":artifacts,"compression_status":"prismcode_nanl_spike_codec_verified","checksum_algorithm":"fnv1a_64","endian":"little","v2_was_live":true,"path_b_required":true,"serialization_failure":null,"completion_status":"recovered_md_evidence_postmd_materialization_required","post_md_aggregation_status":"materialization_required_path_b","recovery":{"schema_kind":"prism.native_offline_manifest_recovery.v1","runtime_origin":"POST_RUN_NATIVE_RECOVERY","runtime_inline_epoch_closed":false,"original_execution_status":execution["status"],"original_exit_status":execution["exit_status"],"original_execution_receipt":args.execution_receipt,"original_execution_receipt_sha256":prism_nhs::ensemble::manifest::sha256_file(&args.execution_receipt)?,"original_closeout":"FAILED_RETAINED; no retrospective runtime chronology closure","native_recovery_executable_sha256":prism_nhs::ensemble::manifest::sha256_file(&std::env::current_exe()?)?,"packet_bindings":bindings,"v2_ignition_witness_sha256":prism_nhs::ensemble::manifest::sha256_file(&ignition_path)?,"native_lattice_sha256":prism_nhs::ensemble::manifest::sha256_file(&lattice_path)?,"scope":"Verified durable packet populations and completed protocol horizons. Original terminal runtime handoff remained interrupted; fresh materialization required."}});
    // Create-new commit: malformed evidence never commits a manifest; preserve pre-existing output.
    let mut file=std::fs::OpenOptions::new().write(true).create_new(true).open(&args.manifest)?;
    file.write_all(&serde_json::to_vec_pretty(&manifest)?)?;file.write_all(b"\n")?;file.sync_all()?;
    println!("{}",serde_json::json!({"status":"RECOVERED_NATIVE_HANDOFF","manifest":args.manifest,"run_id":manifest["run_id"],"total_spikes_md":total,"original_runtime_closeout":"FAILED_RETAINED"}));
    Ok(())
}

#[cfg(test)]
mod native_recovery_tests {
    use super::*;
    fn protocol() -> serde_json::Value {serde_json::json!({"stream_index":0,"run_id":"retained","status_code":0,"cold_hold_end":2,"ramp_end":4,"warm_hold_end":6,"ramp_down_end":8,"total_steps":10,"current_step":10,"steering_flags":0})}
    #[test] fn recovery_completed_protocol_accepts_terminal_and_continuation() {
        recovery_complete_protocol(protocol(),0,"retained").unwrap();
        let mut p=protocol();p["current_step"]=11.into();recovery_complete_protocol(p,0,"retained").unwrap();
    }
    #[test] fn recovery_protocol_rejects_incomplete_missing_failed_and_foreign() {
        for (key,value) in [("current_step",serde_json::json!(9)),("current_step",serde_json::Value::Null),("status_code",serde_json::json!(1)),("run_id",serde_json::json!("foreign")),("stream_index",serde_json::json!(1))] {
            let mut p=protocol();p[key]=value;assert!(recovery_complete_protocol(p,0,"retained").is_err());
        }
    }
    #[test] fn recovery_hash_rejects_corruption_and_missing_receipt_field() {
        let r=serde_json::json!({"raw_hash_blake3":blake3::hash(b"native bytes").to_hex().to_string()});
        recovery_hash_check(b"native bytes",&r,"raw_hash_blake3").unwrap();
        assert!(recovery_hash_check(b"changed bytes",&r,"raw_hash_blake3").is_err());
        assert!(recovery_hash_check(b"native bytes",&r,"missing").is_err());
    }
    #[test] fn recovery_envelope_identity_and_dimensions_are_bound() {
        let mut h=EnvelopeHeader{magic:*b"PRSPK001",schema_version:1,endian_marker:0x01020304,stream_id:0,run_id:"r".into(),stem:"t".into(),record_count:2,byte_stride:96,payload_size:192};
        recovery_identity(&h,"r","t",0).unwrap();
        assert!(recovery_identity(&h,"other","t",0).is_err());
        assert!(recovery_identity(&h,"r","other",0).is_err());
        assert!(recovery_identity(&h,"r","t",1).is_err());
        h.payload_size=193;assert!(recovery_identity(&h,"r","t",0).is_err());
    }
    #[test] fn recovery_surface_names_preserve_native_roles() {
        assert_eq!(recovery_kind("X_stream0_spikes.bin"),"spikes");
        assert_eq!(recovery_kind("X_stream07_physical_time.jsonl"),"physical_time_journal");
        assert_eq!(recovery_kind("X_stream0_kcc_v2full.lag_support.json"),"kcc_lag_support");
        assert_eq!(recovery_kind("unrecognized.bin"),"retained_native_artifact");
    }
    #[test] fn recovery_never_overwrites_existing_manifest() {
        let d=tempfile::tempdir().unwrap();let m=d.path().join("md_evidence_manifest.json");std::fs::write(&m,b"retained").unwrap();
        let a=RecoveryArgs{run_dir:d.path().into(),topology:d.path().join("absent"),execution_receipt:d.path().join("absent"),manifest:m.clone(),expected_streams:8};
        assert!(recover_manifest(a).is_err());assert_eq!(std::fs::read(m).unwrap(),b"retained");
    }
    #[test] fn recovery_spike_decode_uses_existing_codec_and_integrity_check() {
        let d=tempfile::tempdir().unwrap();let p=d.path().join("spikes.prismnanl");
        let payload=vec![3u8;96];let prefix=b"native-prefix";let suffix=b"native-suffix";
        let raw=[prefix.as_slice(),payload.as_slice(),suffix.as_slice()].concat();
        let encoded=RuntimeCodecRegistry::with_builtin_codecs().codec_by_kind(CompressionCodecKind::SectionZstd).unwrap().encode(&payload).unwrap();
        let mut header=serde_json::json!({"record_count":1,"payload_stride":96,"raw_hash_blake3":blake3::hash(&raw).to_hex().to_string()});
        for corrupted in [false,true] {
            if corrupted {header["raw_hash_blake3"]="wrong".into();}
            let h=serde_json::to_vec(&header).unwrap();let mut bytes=b"PRSPCZ01".to_vec();bytes.extend(1u32.to_le_bytes());
            for n in [h.len(),prefix.len(),suffix.len(),encoded.len()] {bytes.extend((n as u64).to_le_bytes());}
            bytes.extend(h);bytes.extend(prefix);bytes.extend(suffix);bytes.extend(&encoded);std::fs::write(&p,bytes).unwrap();
            let result=decode_spike_prismnanl_packet(&p);if corrupted {assert!(result.is_err());} else {assert_eq!(result.unwrap(),raw);}
        }
    }
}
