// PRISM Ghost/ZSTR Evidence Scanner — Path B feature extractor.
//
// CORRECT ROLE — read the directive carefully:
//   * This binary is an OFFLINE evidence reader, scorer, and provenance
//     emitter. It is NOT the binding-site materializer.
//   * It does NOT emit binding_sites.materialized.json.
//   * It does NOT use LIGSITE.
//   * It does NOT fake values, schemas, or bilateral support.
//   * It DOES emit feature/provenance JSONs that prism-materialize-sites
//     can consume in a follow-up integration.
//
// Inputs: md_evidence_manifest.json (manifest-first discovery; no
// hardcoded run paths). All Ghost/ZSTR/V2 files referenced from the
// manifest's run directory.
//
// Outputs (always emitted for honest reporting; status fields explicit):
//   tile_field_completeness.json    — per-file schema validation matrix
//   ghost_tile_evidence.json        — ghost-tile-stream summaries
//   zstr_event_summary.json         — zstr-stream summaries
//   ranked_tile_events.json         — events ranked by verified components
//   ghost_zstr_candidate_features.json — handoff for materializer
//   ghost_zstr_scan_report.json     — overall scan report
//
// SCHEMA SOURCES (cited; do NOT modify these without re-verifying):
//   GhostTileFrame      crates/prism-nhs/src/ghost_tile.rs:80-213
//                       repr(C, align(4096)), 4096 B total, const-asserted offsets:
//                         frame_idx        u64    @ 0
//                         site_id          u32    @ 8
//                         chain_id         u8     @ 12
//                         adjudication_code u8    @ 13
//                         telemetry_flags  u16    @ 14
//                         kl_divergence    f32    @ 16
//                         power_spectrum   [f32;24] @ 20  (4 planes × 6 bands)
//                         thermo_flux      [f32;2] @ 116  ([wd_change, vib_energy])
//                         causal_lead_residue u32 @ 124
//                         _reserved_payload [u32;32] @ 128
//                         _slack           [u8;3840] @ 256
//   File layout:        crates/prism-nhs/src/ghost_tile.rs:225-237
//                         offset 0..4   u32 n_frames_written
//                         offset 4..4096 pad  (counter sector)
//                         offset 4096..end GhostTileFrame × n_frames_written
//
//   ZstrFrameHeader     crates/prism-nhs/src/zstr.rs (ZstrFrameHeader)
//                       repr(C, align(4096)), 4096 B total:
//                         frame_idx        u64    @ 0     (authoritative device step)
//                         dt               f32    @ 8     (picoseconds at capture)
//                         adjudication_code u32   @ 12
//                         completion_fence u32    @ 16
//                         n_atoms          u32    @ 20
//                         gear_id          u32    @ 24
//                         force_norm       f32    @ 28
//                         external_work    f64    @ 32
//                         potential_energy f64    @ 40
//                         stream_id        u32    @ 48    (WP4)
//                         schema_version   u32    @ 52    (WP4; 1 = stamped header)
//                         _padding         [u32;1010] @ 56
//                       WP4 note: offsets 0/8/12/20/24/48/52 are stamped on
//                       device by prism_zstr_device_slot_update_kernel and
//                       zstr_stage_audit_kernel. Pre-WP4 output left the whole
//                       header at zero, so n_atoms read 0, the derived slot
//                       stride was 0, and this scanner reported
//                       parse_status="first_header_only".
//   Per-slot file:      crates/prism-nhs/src/zstr.rs:154-159
//                         [header 4096 B][positions n_atoms*12][forces n_atoms*12][pad→4096]
//   File name pattern:  crates/prism-nhs/src/bin/nhs_rt_full.rs:7638,7788
//                         prism_zstr_<pid>_<stream>.bin
//
//   prism_v2 trajectory crates/prism-nhs/src/bin/nhs_rt_full.rs:6260-6301
//                         offset 0..8   u64 n_frames
//                         per frame: u64 step, u32 n_floats, f32×n_floats
//
//   intensity_packed    crates/prism-nhs/src/interferometric_adjudicator.rs:712-735
//                         QI_SHIFT = 30
//                         INTENSITY_PAYLOAD_MASK = 0x3FFF_FFFF
//                         UV-code → λ:
//                           0 → 260 nm (PHE-dominant)
//                           1 → 280 nm (TRP primary)
//                           2 → 305 nm (TRP gaussian tail)
//                           3 → 320 nm (TRP control / baseline)
//                       NOTE: intensity_packed lives on `RichSpike`, NOT on
//                       GhostTileFrame. Ghost tiles do NOT carry the perturbation
//                       channel directly; the perturbation_channel_evidence block
//                       on a ghost-tile event is therefore status="not_in_ghost_tile_schema"
//                       and would require cross-referencing with PRSPK001 spikes
//                       (which carry per-spike `wavelength_nm` directly).
//
// HARD INVARIANT NOTES (per directive):
//   I1 perturbation_channel:  Ghost tiles lack intensity_packed. Status
//                             "not_in_ghost_tile_schema". Ranking neutral.
//   I2 gear-aware timing:     ZstrFrameHeader.gear_id IS schema-verified (offset 24)
//                             but Wave A producer leaves it at 0. Status
//                             "schema_verified_but_producer_wave_a_writes_zero".
//                             dt IS schema-verified (offset 8); semantics are
//                             "picoseconds at capture". Apply only when present.
//   I3 ASC work integral Γw:  asc_vectors.bin format has NO schema header (raw blob).
//                             Without verified format, Γw is NOT computed.
//                             gamma_w_status = "asc_vector_format_unverified_no_compute".

use anyhow::{bail, Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// **GEAR-PROVENANCE-002** — the producer's "no gear measurement" sentinel.
///
/// Mirrors `crate::zstr::ZSTR_GEAR_ID_UNMEASURED`,
/// `PRISM_ZSTR_GEAR_ID_UNMEASURED` (zstr_kernels.cuh) and
/// `GHOST_V2_GEAR_ID_UNMEASURED` (ghost_tile_kernel.cuh). Declared locally
/// because this scanner reads headers by raw byte offset and must stay
/// buildable without the `gpu` feature that gates `prism_nhs::zstr`.
///
/// The documented gear encoding is 0 = 0.5 fs, 1 = 2.0 fs, 2 = 4.0 fs (HMR),
/// 3 = abort-trap; every legal gear is a small integer, so 0 cannot double as
/// "unknown". This value is the only in-band way to say a gear was not
/// measured.
const ZSTR_GEAR_ID_UNMEASURED: u32 = 0xFFFF_FFFF;

// ─── CLI ───────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(name = "prism-ghost-evidence-scan")]
#[command(
    about = "Path B Ghost/ZSTR evidence scanner — emits features for prism-materialize-sites."
)]
struct Args {
    /// md_evidence_manifest.json path produced by `nhs_rt_full --md-only-evidence`.
    #[arg(short = 'm', long)]
    manifest: PathBuf,

    /// Output dir. Defaults to the manifest's parent directory.
    #[arg(short = 'o', long)]
    out_dir: Option<PathBuf>,

    /// Z-score sigma threshold for tile-event scoring. Default 12.0.
    /// Only applied when noise_floor.json is present and shapes match.
    #[arg(long, default_value = "12.0")]
    threshold_sigma: f32,

    /// Max events to emit (0 = no cap).
    #[arg(long, default_value = "0")]
    max_events: usize,

    /// Allow missing Ghost/ZSTR files; emit honest "missing" status and
    /// exit 0 instead of nonzero.
    #[arg(long, default_value = "false")]
    allow_missing: bool,

    /// Fail if any present binary's schema is unknown / unverifiable.
    #[arg(long, default_value = "false")]
    strict_schema: bool,

    /// Explicit Ghost v2 spatial map compiled from this run's native tile records.
    #[arg(long)]
    site_map: Option<PathBuf>,

    /// Require a nonempty, independently verified native-v2 site map. This
    /// rejects the engine's legacy empty sidecar and any non-v2 map.
    #[arg(long, default_value = "false", requires = "site_map")]
    require_native_v2_site_map: bool,

    /// Emit per-event NDJSON (one event per line) in addition to the
    /// summary JSON.
    #[arg(long, default_value = "false")]
    emit_ndjson: bool,

    /// Emit ALL parsed events into ranked_tile_events.json (large output).
    /// Default: emit top 200 by tile_score.
    #[arg(long, default_value = "false")]
    emit_all_events: bool,

    /// Reconstruct native GPU lattice from all verified segments using this recorded lattice config.
    /// Original graph is preserved; output is a separately identified floor-aware graph.
    #[arg(long)]
    replay_lattice_from: Option<PathBuf>,

    /// Select one exact native component graph to expose through the existing
    /// materializer mapped-component contract. Values are graph namespaces
    /// emitted by this scanner, e.g. `original_graph` or `floor_aware_graph`.
    /// No namespace is selected by default so legacy behavior is unchanged.
    #[arg(long)]
    exact_component_adapter_namespace: Option<String>,

    /// Verbose logging.
    #[arg(short, long, default_value = "false")]
    verbose: bool,
}

// ─── Manifest schema (read-only mirror) ────────────────────────────────────

#[derive(Debug, Deserialize)]
struct ManifestArtifact {
    kind: String,
    stream_id: Option<u32>,
    path: String,
    #[allow(dead_code)]
    size_bytes: u64,
    present: bool,
    #[allow(dead_code)]
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
    artifacts: Vec<ManifestArtifact>,
}

// ─── Schema constants — cited from producer source ────────────────────────

mod ghost_schema {
    /// `GhostTileFrame` per-record byte size (cited: ghost_tile.rs:217).
    pub const RECORD_BYTES: u64 = 4096;
    /// File-leading counter sector size (cited: ghost_tile.rs:225-237).
    pub const COUNTER_SECTOR_BYTES: u64 = 4096;
    /// Magic offsets within a 4096-byte record (cited: ghost_tile.rs:198-213
    /// const-context offset assertions).
    pub const OFF_FRAME_IDX: usize = 0;
    pub const OFF_SITE_ID: usize = 8;
    pub const OFF_CHAIN_ID: usize = 12;
    pub const OFF_ADJ_CODE: usize = 13;
    pub const OFF_TELEMETRY_FLAGS: usize = 14;
    pub const OFF_KL_DIV: usize = 16;
    pub const OFF_POWER_SPECTRUM: usize = 20;
    pub const OFF_THERMO_FLUX: usize = 116;
    pub const OFF_CAUSAL_LEAD: usize = 124;
    /// Bit 0 — CLASS_TAINTED (cited: ghost_tile.rs:172). Set when the kernel
    /// substituted NaN/Inf in thermo_flux with 0.0.
    pub const FLAG_CLASS_TAINTED: u16 = 0x0001;
}

mod zstr_schema {
    /// `ZstrFrameHeader` byte size (cited: zstr.rs:67 + size_of comment line 162).
    pub const HEADER_BYTES: u64 = 4096;
    /// Field offsets within the 4096 B header (cited: zstr.rs:68-114 inline
    /// comments after each field).
    pub const OFF_FRAME_IDX: usize = 0;
    pub const OFF_DT: usize = 8;
    pub const OFF_ADJ_CODE: usize = 12;
    pub const OFF_COMPLETION_FENCE: usize = 16;
    pub const OFF_N_ATOMS: usize = 20;
    pub const OFF_GEAR_ID: usize = 24;
    pub const OFF_FORCE_NORM: usize = 28;
    pub const OFF_EXTERNAL_WORK: usize = 32;
    pub const OFF_POTENTIAL_ENERGY: usize = 40;
}

mod uv_qi_schema {
    /// QI bit shift (cited: interferometric_adjudicator.rs:714).
    pub const QI_SHIFT: u32 = 30;
    /// UV-code → wavelength_nm map (cited: interferometric_adjudicator.rs:722-728).
    pub const UV_NM: [u32; 4] = [260, 280, 305, 320];
    pub const UV_LABELS: [&str; 4] = [
        "260nm_PHE_dominant",
        "280nm_TRP_primary",
        "305nm_TRP_gaussian_tail",
        "320nm_TRP_control",
    ];
}

// ─── Discovered file record ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct DiscoveredFile {
    path: PathBuf,
    stream_id: Option<u32>,
    file_kind: FileKind,
    file_size: u64,
}

#[derive(Debug, Clone, PartialEq)]
enum FileKind {
    GhostTiles,
    Zstr,
    PrismV2Trajectory,
}

fn parse_stream_id_after_underscore(name: &str, marker: &str) -> Option<u32> {
    let idx = name.find(marker)? + marker.len();
    let tail = &name[idx..];
    let digits: String = tail.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse::<u32>().ok()
}

fn parse_stream_id_from_filename(p: &Path) -> Option<u32> {
    let name = p.file_name()?.to_string_lossy().to_string();
    if let Some(id) = parse_stream_id_after_underscore(&name, "_stream") {
        return Some(id);
    }
    // prism_zstr_<pid>_<stream>.bin and prism_v2_<pid>_<stream>.bin
    // — last underscore-separated digit run before the extension.
    let stem = name.strip_suffix(".bin").unwrap_or(&name);
    let last = stem.rsplit('_').next()?;
    last.parse::<u32>().ok()
}

fn discover_files(run_dir: &Path) -> Result<Vec<DiscoveredFile>> {
    let mut out: Vec<DiscoveredFile> = Vec::new();
    let entries = std::fs::read_dir(run_dir)
        .with_context(|| format!("read run dir {}", run_dir.display()))?;
    for e in entries.flatten() {
        let p = e.path();
        let Some(name) = p.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        let kind = if name.ends_with("_ghost_tiles.bin") {
            FileKind::GhostTiles
        } else if name.starts_with("prism_zstr_") && name.ends_with(".bin") {
            FileKind::Zstr
        } else if name.starts_with("prism_v2_") && name.ends_with(".bin") {
            FileKind::PrismV2Trajectory
        } else {
            continue;
        };
        let file_size = p.metadata().map(|m| m.len()).unwrap_or(0);
        let stream_id = parse_stream_id_from_filename(&p);
        out.push(DiscoveredFile {
            path: p,
            stream_id,
            file_kind: kind,
            file_size,
        });
    }
    out.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(out)
}

// ─── Schema validation per file ────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
struct FileSchemaValidation {
    file_path: String,
    file_kind: &'static str,
    stream_id: Option<u32>,
    file_size: u64,
    format_guess: &'static str,
    schema_status: &'static str,
    parse_status: &'static str,
    record_count: Option<u64>,
    byte_stride: Option<u64>,
    parser_used: &'static str,
    missing_fields: Vec<&'static str>,
    unsafe_assumptions: Vec<&'static str>,
    notes: Vec<String>,
}

fn validate_ghost_tiles_schema(f: &DiscoveredFile) -> FileSchemaValidation {
    let mut notes = Vec::new();
    let mut record_count: Option<u64> = None;
    let mut parse_status: &'static str = "header_only";
    let mut schema_status: &'static str = "verified_const_offsets_in_ghost_tile_rs_198_213";

    if f.file_size < ghost_schema::COUNTER_SECTOR_BYTES {
        notes.push(format!(
            "file_size {} < counter_sector {} — file too small to contain even the leading u32 n_frames_written counter",
            f.file_size, ghost_schema::COUNTER_SECTOR_BYTES
        ));
        return FileSchemaValidation {
            file_path: f.path.display().to_string(),
            file_kind: "ghost_tiles",
            stream_id: f.stream_id,
            file_size: f.file_size,
            format_guess: "ghost_tiles_v9d_sector_locked",
            schema_status,
            parse_status: "file_too_small",
            record_count: None,
            byte_stride: Some(ghost_schema::RECORD_BYTES),
            parser_used: "ghost_tiles_envelope_validator",
            missing_fields: vec![],
            unsafe_assumptions: vec![],
            notes,
        };
    }
    // Read u32 n_frames_written from offset 0.
    if let Ok(file) = File::open(&f.path) {
        let mut br = BufReader::new(file);
        let mut buf = [0u8; 4];
        if br.read_exact(&mut buf).is_ok() {
            let n = u32::from_le_bytes(buf);
            record_count = Some(n as u64);
            // Verify file size = counter_sector + n * 4096.
            let expected =
                ghost_schema::COUNTER_SECTOR_BYTES + (n as u64) * ghost_schema::RECORD_BYTES;
            if f.file_size == expected {
                parse_status = "envelope_consistent";
            } else if f.file_size >= expected {
                parse_status = "envelope_consistent_with_trailing_bytes";
                notes.push(format!(
                    "trailing {} bytes after expected payload",
                    f.file_size - expected
                ));
            } else {
                parse_status = "envelope_size_mismatch";
                schema_status = "verified_offsets_but_file_truncated";
                notes.push(format!(
                    "file_size {} < counter_sector + n*4096 = {}",
                    f.file_size, expected
                ));
            }
        }
    }
    FileSchemaValidation {
        file_path: f.path.display().to_string(),
        file_kind: "ghost_tiles",
        stream_id: f.stream_id,
        file_size: f.file_size,
        format_guess: "ghost_tiles_v9d_sector_locked",
        schema_status,
        parse_status,
        record_count,
        byte_stride: Some(ghost_schema::RECORD_BYTES),
        parser_used: "ghost_tiles_envelope_validator",
        missing_fields: vec![],
        unsafe_assumptions: vec![],
        notes,
    }
}

fn validate_zstr_schema(f: &DiscoveredFile) -> FileSchemaValidation {
    let mut notes = Vec::new();
    let mut record_count: Option<u64> = None;
    let mut byte_stride: Option<u64> = None;
    let mut parse_status: &'static str = "header_only";
    let schema_status: &'static str =
        "header_offsets_verified_in_zstr_rs_67_114_per_slot_size_unknown_until_n_atoms_read";

    if f.file_size < zstr_schema::HEADER_BYTES {
        notes.push(format!(
            "file_size {} < header_bytes {} — cannot contain even one ZstrFrameHeader",
            f.file_size,
            zstr_schema::HEADER_BYTES
        ));
        return FileSchemaValidation {
            file_path: f.path.display().to_string(),
            file_kind: "prism_zstr",
            stream_id: f.stream_id,
            file_size: f.file_size,
            format_guess: "prism_zstr_5slot_4096_padded",
            schema_status,
            parse_status: "file_too_small",
            record_count: None,
            byte_stride: None,
            parser_used: "zstr_header_validator",
            missing_fields: vec![],
            unsafe_assumptions: vec![],
            notes,
        };
    }
    // Read first header to recover n_atoms, then derive per-slot stride.
    if let Ok(file) = File::open(&f.path) {
        let mut br = BufReader::new(file);
        let mut hdr = [0u8; 4096];
        if br.read_exact(&mut hdr).is_ok() {
            let n_atoms = u32::from_le_bytes(
                hdr[zstr_schema::OFF_N_ATOMS..zstr_schema::OFF_N_ATOMS + 4]
                    .try_into()
                    .unwrap(),
            );
            let pos_bytes_raw = (n_atoms as u64) * 12;
            let pos_bytes_aligned = (pos_bytes_raw + 15) & !15;
            let force_bytes = (n_atoms as u64) * 12;
            let raw_frame = zstr_schema::HEADER_BYTES + pos_bytes_aligned + force_bytes;
            let frame_size = (raw_frame + 4095) & !4095;
            byte_stride = Some(frame_size);
            if frame_size == 0 {
                notes.push("derived frame_size = 0 — n_atoms in first header likely zero (file may be only the first header padded)".to_string());
                record_count = Some(if f.file_size >= zstr_schema::HEADER_BYTES {
                    1
                } else {
                    0
                });
                parse_status = "first_header_only";
            } else {
                let n = f.file_size / frame_size;
                let trailing = f.file_size - n * frame_size;
                record_count = Some(n);
                if trailing == 0 {
                    parse_status = "envelope_consistent";
                } else {
                    parse_status = "envelope_consistent_with_trailing_bytes";
                    notes.push(format!(
                        "trailing {} bytes after {} full slots of {} bytes each",
                        trailing, n, frame_size
                    ));
                }
            }
        }
    }
    FileSchemaValidation {
        file_path: f.path.display().to_string(),
        file_kind: "prism_zstr",
        stream_id: f.stream_id,
        file_size: f.file_size,
        format_guess: "prism_zstr_5slot_4096_padded",
        schema_status,
        parse_status,
        record_count,
        byte_stride,
        parser_used: "zstr_header_validator",
        missing_fields: vec![],
        unsafe_assumptions: vec![],
        notes,
    }
}

fn validate_prism_v2_schema(f: &DiscoveredFile) -> FileSchemaValidation {
    let mut notes = Vec::new();
    let mut record_count: Option<u64> = None;
    let mut parse_status: &'static str = "header_only";
    if f.file_size < 8 {
        notes.push("file_size < 8 — no n_frames header".to_string());
        return FileSchemaValidation {
            file_path: f.path.display().to_string(),
            file_kind: "prism_v2_trajectory",
            stream_id: f.stream_id,
            file_size: f.file_size,
            format_guess: "prism_v2_trajectory_writer",
            schema_status: "verified_writer_format_in_nhs_rt_full_rs_6291_6301",
            parse_status: "file_too_small",
            record_count: None,
            byte_stride: None,
            parser_used: "prism_v2_header_validator",
            missing_fields: vec![],
            unsafe_assumptions: vec![],
            notes,
        };
    }
    if let Ok(file) = File::open(&f.path) {
        let mut br = BufReader::new(file);
        let mut buf = [0u8; 8];
        if br.read_exact(&mut buf).is_ok() {
            let n = u64::from_le_bytes(buf);
            record_count = Some(n);
            if n == 0 && f.file_size == 8 {
                parse_status = "empty_trajectory_n_frames_zero";
                notes.push(
                    "V2 was not live this run; only the 8-byte header was written".to_string(),
                );
            } else {
                parse_status = "header_present_payload_present";
            }
        }
    }
    FileSchemaValidation {
        file_path: f.path.display().to_string(),
        file_kind: "prism_v2_trajectory",
        stream_id: f.stream_id,
        file_size: f.file_size,
        format_guess: "prism_v2_trajectory_writer",
        schema_status: "verified_writer_format_in_nhs_rt_full_rs_6291_6301",
        parse_status,
        record_count,
        byte_stride: None,
        parser_used: "prism_v2_header_validator",
        missing_fields: vec![],
        unsafe_assumptions: vec![],
        notes,
    }
}

// ─── Ghost tile event parsing ───────────────────────────────────────────────

#[derive(Debug, Serialize, Clone)]
struct GhostTileEvent {
    event_id: String,
    stream_id: u32,
    frame_idx: u64,
    step_idx: Option<u64>,
    site_id: u32,
    chain_id: Option<char>,
    adj_code: u8,
    telemetry_flags: u16,
    class_tainted: bool,
    /// M1.2.23 §5 — v1/v2 schema overlay detection.
    schema_version: u32,
    is_v2: bool,
    /// M1.2.23 §4 — Transparent MAR observation/discovery gates,
    /// populated only when is_v2.
    v2_observation_pass: bool,
    v2_discovery_pass: bool,
    v2_field_completeness_flags: u16,
    kl_divergence: f32,
    /// 4 planes × 6 bands. Plane 0 = geometry; planes 1-3 may be 0.0
    /// per producer note (ghost_tile.rs:128-132 — kernel doesn't yet
    /// populate planes 1-3).
    power_spectrum: [f32; 24],
    plane_evidence: prism_nhs::ghost_tile::GhostPlaneEvidence,
    /// [wd_change, vib_energy] — both NaN until upstream wires them
    /// (ghost_tile.rs:136-141).
    thermo_flux: [f32; 2],
    causal_lead_residue: u32,
    /// I1 — Ghost tile schema does NOT carry intensity_packed.
    perturbation_channel_evidence: PerturbationChannelEvidence,
    /// I2 — Ghost tile schema does NOT carry gear_id (it's in _reserved_payload).
    gear_normalized_timing: GearTimingEvidence,
    /// I3 — Γw not computable without verified asc_vectors format.
    asc_work_evidence: AscWorkEvidence,
    /// Flux coupling η = wd_change / max(|vib_energy|, ε)
    flux_coupling: FluxCoupling,
    field_completeness: GhostEventFieldCompleteness,
    provenance: EventProvenance,
    source_segment_sha256: Option<String>,
    source_record_sha256: String,
    stream_record_ordinal: u64,
}

#[derive(Debug, Serialize, Clone, Default)]
struct PerturbationChannelEvidence {
    qi_bits: Option<u32>,
    uv_wavelength_nm: Option<u32>,
    uv_label: Option<&'static str>,
    intensity_packed_raw: Option<u32>,
    status: &'static str,
}

#[derive(Debug, Serialize, Clone, Default)]
struct GearTimingEvidence {
    gear_id: Option<u32>,
    dt_fs: Option<f32>,
    physical_time_fs: Option<f64>,
    nominal_header_step_time_fs: Option<f64>,
    physical_delta_fs: Option<f64>,
    status: &'static str,
}

#[derive(Debug, Serialize, Clone, Default)]
struct AscWorkEvidence {
    gamma_w: Option<f64>,
    gamma_w_status: &'static str,
    gamma_w_units: &'static str,
    formula: &'static str,
    is_proxy: bool,
    source_files: Vec<String>,
}

#[derive(Debug, Serialize, Clone, Default)]
struct FluxCoupling {
    eta: Option<f32>,
    eta_status: &'static str,
    wd_change_finite: bool,
    vib_energy_finite: bool,
}

#[derive(Debug, Serialize, Clone, Default)]
struct GhostEventFieldCompleteness {
    frame_idx: bool,
    site_id: bool,
    chain_id: bool,
    adjudication_code: bool,
    kl_divergence: bool,
    power_spectrum_geo: bool,
    power_spectrum_planes_1_3_populated: bool,
    thermo_flux_finite: bool,
    causal_lead_residue_resolved: bool,
}

#[derive(Debug, Serialize, Clone)]
struct EventProvenance {
    file: String,
    byte_offset: u64,
    schema_status: &'static str,
}

fn decode_ghost_chain_id(value: u8) -> Option<char> {
    match value {
        // The CUDA producer and coupled runtime use compact numeric chain
        // roles 0/1. Early schema prose described ASCII A/B, and historical
        // artifacts may contain those bytes.
        0x00 | 0x41 => Some('A'),
        0x01 | 0x42 => Some('B'),
        _ => None,
    }
}

#[cfg(test)]
mod ghost_chain_encoding_tests {
    use super::decode_ghost_chain_id;

    #[test]
    fn accepts_runtime_numeric_and_historical_ascii_encodings() {
        assert_eq!(decode_ghost_chain_id(0), Some('A'));
        assert_eq!(decode_ghost_chain_id(1), Some('B'));
        assert_eq!(decode_ghost_chain_id(b'A'), Some('A'));
        assert_eq!(decode_ghost_chain_id(b'B'), Some('B'));
        assert_eq!(decode_ghost_chain_id(0xFF), None);
    }
}

fn parse_ghost_tiles_file(
    path: &Path,
    stream_id: u32,
    n_records: u64,
    max_events: usize,
) -> Result<Vec<GhostTileEvent>> {
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(ghost_schema::COUNTER_SECTOR_BYTES))?;
    parse_ghost_records(BufReader::with_capacity(1 << 20,file),path,stream_id,n_records,max_events,ghost_schema::COUNTER_SECTOR_BYTES,0,None)
}

fn parse_ghost_records<R:Read>(mut br:R,path:&Path,stream_id:u32,n_records:u64,max_events:usize,
    payload_offset:u64,first_ordinal:u64,segment_file_sha256:Option<&str>) -> Result<Vec<GhostTileEvent>> {
    let mut out=Vec::new();
    let mut buf = [0u8; 4096];
    let cap = if max_events == 0 {
        n_records
    } else {
        (max_events as u64).min(n_records)
    };
    for k in 0..cap {
        br.read_exact(&mut buf).with_context(||format!("missing complete Ghost record {k} in {}",path.display()))?;
        let frame_idx = u64::from_le_bytes(buf[0..8].try_into().unwrap());
        let site_id = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        let chain_id_byte = buf[12];
        let adj_code = buf[13];
        let telemetry_flags = u16::from_le_bytes(buf[14..16].try_into().unwrap());
        let kl_divergence = f32::from_le_bytes(buf[16..20].try_into().unwrap());
        let mut power_spectrum = [0f32; 24];
        for i in 0..24 {
            let off = 20 + i * 4;
            power_spectrum[i] = f32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
        }
        let thermo_flux: [f32; 2] = [
            f32::from_le_bytes(buf[116..120].try_into().unwrap()),
            f32::from_le_bytes(buf[120..124].try_into().unwrap()),
        ];
        let causal_lead_residue = u32::from_le_bytes(buf[124..128].try_into().unwrap());

        // ─── M1.2.23 §5 — v1/v2 schema detection ─────────────────────────
        // Read schema_version u32 at offset 128. 0=v1 (legacy zero), 2=v2.
        let schema_version_v2: u32 = u32::from_le_bytes(buf[128..132].try_into().unwrap());
        let is_v2 = matches!(schema_version_v2, 2 | 3);
        let (
            v2_obs_pass,
            v2_disc_pass,
            v2_perturb_chan,
            v2_uv_nm,
            v2_field_complete_flags,
            v2_gear_id,
            v2_dt_fs,
            v2_step_idx,
        ) = if is_v2 {
            (
                buf[132] != 0,
                buf[133] != 0,
                buf[134],
                u16::from_le_bytes(buf[136..138].try_into().unwrap()),
                u16::from_le_bytes(buf[138..140].try_into().unwrap()),
                u32::from_le_bytes(buf[140..144].try_into().unwrap()),
                f32::from_le_bytes(buf[144..148].try_into().unwrap()),
                // M1.2.24 fix: step_idx moved from 148 → 152 (148 was
                // misaligned for STG.E.64). Bytes 148..152 are reserved
                // padding (zero-initialized by the v2 kernel).
                u64::from_le_bytes(buf[152..160].try_into().unwrap()),
            )
        } else {
            (false, false, 0xFFu8, 0u16, 0u16, 0u32, 0.0f32, 0u64)
        };

        let chain_id = decode_ghost_chain_id(chain_id_byte);
        let class_tainted = (telemetry_flags & ghost_schema::FLAG_CLASS_TAINTED) != 0;
        let wd = thermo_flux[0];
        let vib = thermo_flux[1];
        let wd_finite = wd.is_finite();
        let vib_finite = vib.is_finite();
        let (eta, eta_status) = if wd_finite && vib_finite && !class_tainted {
            let denom = vib.abs().max(1e-9);
            (Some(wd / denom), "computed")
        } else if class_tainted {
            (None, "skipped_class_tainted")
        } else {
            (None, "skipped_thermo_flux_nonfinite_or_unwired")
        };

        // Geo plane = power_spectrum[0..6].
        let geo_band_sum: f32 = power_spectrum[0..6].iter().sum();
        // P4(B) — planes 1-3 are KL_EPS-padded (1e-7), never 0.0, so testing the
        // spectrum against zero always reports them populated. Read the
        // producer's exact sum_w-derived mask when present; fall back to the old
        // heuristic only for records written before the mask existed, so
        // archived scans keep their historical (wrong but unchanged) behaviour
        // rather than silently shifting.
        let plane_evidence = prism_nhs::ghost_tile::ghost_plane_evidence(&buf[..])
            .context("Ghost record lacks complete spectral plane bytes")?;
        let other_planes_nonzero = plane_evidence.cosine_evidence_mask & 0b1110 != 0;

        let mut field_completeness = GhostEventFieldCompleteness::default();
        field_completeness.frame_idx = true;
        field_completeness.site_id = true;
        field_completeness.chain_id = chain_id.is_some();
        field_completeness.adjudication_code = true;
        field_completeness.kl_divergence = kl_divergence.is_finite();
        field_completeness.power_spectrum_geo = geo_band_sum.is_finite();
        field_completeness.power_spectrum_planes_1_3_populated = other_planes_nonzero;
        field_completeness.thermo_flux_finite = wd_finite && vib_finite && !class_tainted;
        field_completeness.causal_lead_residue_resolved = causal_lead_residue != u32::MAX;

        out.push(GhostTileEvent {
            event_id: format!("ghost_s{}_step{}_t{}_record{}", stream_id, if is_v2 {v2_step_idx}else{frame_idx}, site_id, first_ordinal+k),
            stream_id,
            frame_idx,
            step_idx: is_v2.then_some(v2_step_idx),
            site_id,
            chain_id,
            adj_code,
            telemetry_flags,
            class_tainted,
            schema_version: schema_version_v2,
            is_v2,
            v2_observation_pass: v2_obs_pass,
            v2_discovery_pass: v2_disc_pass,
            v2_field_completeness_flags: v2_field_complete_flags,
            kl_divergence,
            power_spectrum,
            plane_evidence,
            thermo_flux,
            causal_lead_residue,
            perturbation_channel_evidence: if is_v2 && v2_perturb_chan != 0xFF {
                PerturbationChannelEvidence {
                    qi_bits: Some(v2_perturb_chan as u32),
                    uv_wavelength_nm: if v2_uv_nm > 0 {
                        Some(v2_uv_nm as u32)
                    } else {
                        None
                    },
                    uv_label: match v2_perturb_chan {
                        0 => Some("260nm_PHE_dominant"),
                        1 => Some("280nm_TRP_primary"),
                        2 => Some("305nm_TRP_gaussian_tail"),
                        3 => Some("320nm_TRP_control"),
                        _ => None,
                    },
                    intensity_packed_raw: None,
                    status: "v2_schema_perturbation_channel_resolved",
                }
            } else if is_v2 {
                PerturbationChannelEvidence {
                    qi_bits: None,
                    uv_wavelength_nm: None,
                    uv_label: None,
                    intensity_packed_raw: None,
                    status: "v2_schema_perturbation_channel_unknown_kernel_left_sentinel",
                }
            } else {
                PerturbationChannelEvidence {
                    qi_bits: None,
                    uv_wavelength_nm: None,
                    uv_label: None,
                    intensity_packed_raw: None,
                    status: "v1_schema_no_perturbation_channel_in_record",
                }
            },
            gear_normalized_timing: if is_v2 {
                let dt_ok = v2_dt_fs > 0.0 && v2_dt_fs.is_finite();
                let phys = if dt_ok {
                    Some((v2_step_idx as f64) * (v2_dt_fs as f64))
                } else {
                    None
                };
                GearTimingEvidence {
                    gear_id: Some(v2_gear_id),
                    dt_fs: if dt_ok { Some(v2_dt_fs) } else { None },
                    physical_time_fs: None,
                    nominal_header_step_time_fs: phys,
                    physical_delta_fs: None,
                    // GEAR-PROVENANCE-002 — the sentinel must not be read as
                    // "wave B active"; it is a producer-declared absence.
                    status: if v2_gear_id == ZSTR_GEAR_ID_UNMEASURED {
                        "v2_schema_gear_unmeasured_sentinel_use_dt_fs_only"
                    } else if v2_gear_id <= 3 {
                        "v2_schema_gear_in_documented_encoding"
                    } else {
                        "v2_schema_gear_out_of_known_range"
                    },
                }
            } else {
                GearTimingEvidence {
                    gear_id: None,
                    dt_fs: None,
                    physical_time_fs: None,
                    nominal_header_step_time_fs: None,
                    physical_delta_fs: None,
                    status: "v1_schema_no_gear_id_in_record",
                }
            },
            asc_work_evidence: AscWorkEvidence {
                gamma_w: None,
                gamma_w_status: "asc_vector_format_unverified_no_compute",
                gamma_w_units: "n/a",
                formula: "Γw = Σ_t F_ASC(t) · Δr(t)  (NOT computed in this commit)",
                is_proxy: true,
                source_files: vec![],
            },
            flux_coupling: FluxCoupling {
                eta,
                eta_status,
                wd_change_finite: wd_finite,
                vib_energy_finite: vib_finite,
            },
            field_completeness,
            source_segment_sha256: segment_file_sha256.map(str::to_owned),
            source_record_sha256: format!("{:x}",sha2::Sha256::digest(&buf)),
            stream_record_ordinal: first_ordinal+k,
            provenance: EventProvenance {
                file: path.display().to_string(),
                byte_offset: payload_offset + k * ghost_schema::RECORD_BYTES,
                schema_status: if segment_file_sha256.is_some(){"verified_hash_chained_segment_exact_payload_record"}else{"verified_ghost_tile_rs_198_213"},
            },
        });
    }
    Ok(out)
}

// ─── ZSTR header parsing — file-level summary only ─────────────────────────

#[derive(Debug, Serialize)]
struct ZstrStreamSummary {
    stream_id: Option<u32>,
    file: String,
    file_size: u64,
    first_header: Option<ZstrFirstHeader>,
    decode_status: &'static str,
    notes: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ZstrFirstHeader {
    frame_idx: u64,
    dt_fs: f64,
    adjudication_code: u32,
    completion_fence: u32,
    n_atoms: u32,
    gear_id: u32,
    force_norm: f32,
    external_work: f64,
    potential_energy: f64,
    gear_id_status: &'static str,
    dt_status: &'static str,
}

fn summarize_zstr_file(f: &DiscoveredFile) -> Result<ZstrStreamSummary> {
    let mut notes: Vec<String> = Vec::new();
    if f.file_size < zstr_schema::HEADER_BYTES {
        return Ok(ZstrStreamSummary {
            stream_id: f.stream_id,
            file: f.path.display().to_string(),
            file_size: f.file_size,
            first_header: None,
            decode_status: "file_too_small_for_header",
            notes,
        });
    }
    let file = File::open(&f.path)?;
    let mut br = BufReader::new(file);
    let mut hdr = [0u8; 4096];
    br.read_exact(&mut hdr)?;
    let frame_idx = u64::from_le_bytes(hdr[0..8].try_into().unwrap());
    let dt_ps = f32::from_le_bytes(hdr[8..12].try_into().unwrap());
    let adjudication_code = u32::from_le_bytes(hdr[12..16].try_into().unwrap());
    let completion_fence = u32::from_le_bytes(hdr[16..20].try_into().unwrap());
    let n_atoms = u32::from_le_bytes(hdr[20..24].try_into().unwrap());
    let gear_id = u32::from_le_bytes(hdr[24..28].try_into().unwrap());
    let force_norm = f32::from_le_bytes(hdr[28..32].try_into().unwrap());
    let external_work = f64::from_le_bytes(hdr[32..40].try_into().unwrap());
    let potential_energy = f64::from_le_bytes(hdr[40..48].try_into().unwrap());

    let dt_fs = (dt_ps as f64) * 1000.0;
    let dt_status: &'static str = if dt_fs > 0.0 && dt_fs.is_finite() {
        "available_picoseconds_to_femtoseconds"
    } else {
        "missing_or_zero"
    };
    // GEAR-PROVENANCE-002 — the producer now stamps
    // PRISM_ZSTR_GEAR_ID_UNMEASURED (0xFFFFFFFF) whenever no measured gear
    // exists, because a build-time literal cannot witness a per-replay device
    // decision and 0 is a LEGAL gear. Without an explicit arm the sentinel
    // fell into "out_of_known_range", which reads as corruption rather than
    // as a declared absence.
    let gear_id_status: &'static str = match gear_id {
        ZSTR_GEAR_ID_UNMEASURED => "unmeasured_sentinel_producer_declared_no_gear_measurement",
        0..=3 => "schema_verified_value_in_known_range",
        _ => "schema_verified_value_out_of_known_range",
    };
    if gear_id == ZSTR_GEAR_ID_UNMEASURED {
        notes.push(
            "gear_id=0xFFFFFFFF — producer declared NO gear measurement for this frame \
             (GEAR-PROVENANCE-002). Gear-aware temporal normalization must use the header \
             dt field and must NOT assume a gear."
                .to_string(),
        );
    } else if gear_id == 0 {
        notes.push(
            "gear_id=0 — a MEASURED 0.5 fs gear under the documented encoding. Legacy files \
             produced before GEAR-PROVENANCE-002 may instead carry an unmeasured placeholder \
             here; cross-check the header dt field before trusting it."
                .to_string(),
        );
    }
    Ok(ZstrStreamSummary {
        stream_id: f.stream_id,
        file: f.path.display().to_string(),
        file_size: f.file_size,
        first_header: Some(ZstrFirstHeader {
            frame_idx,
            dt_fs,
            adjudication_code,
            completion_fence,
            n_atoms,
            gear_id,
            force_norm,
            external_work,
            potential_energy,
            gear_id_status,
            dt_status,
        }),
        decode_status: "first_header_decoded",
        notes,
    })
}

// ─── Noise floor ────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct NoiseFloor {
    #[serde(default)]
    mu: Option<f64>,
    #[serde(default)]
    sigma: Option<f64>,
    #[serde(default)]
    notes: Option<String>,
}

fn try_load_noise_floor(run_dir: &Path, stem: &str, stream_id: u32) -> Option<NoiseFloor> {
    let p = run_dir.join(format!("{}_stream{:02}_noise_floor.json", stem, stream_id));
    let s = std::fs::read_to_string(&p).ok()?;
    serde_json::from_str(&s).ok()
}

// ─── Tile event ranking (only on verified components) ─────────────────────

#[derive(Debug, Serialize)]
struct RankedTileEvent {
    event_id: String,
    rank: u32,
    tile_score: f64,
    score_components: serde_json::Value,
    event: serde_json::Value,
    limitations: Vec<&'static str>,
}

fn score_ghost_tile_event(ev: &GhostTileEvent) -> (f64, serde_json::Value) {
    // All components in [0, 1]. Components from unverified fields → neutral 1.0.
    // tile_score = product.

    let kl_factor: f64 = if ev.field_completeness.kl_divergence {
        let v = ev.kl_divergence as f64;
        if v.is_finite() && v > 0.0 {
            (v / (v + 1.0)).clamp(0.0, 1.0)
        } else {
            0.5
        }
    } else {
        0.5
    };
    let geo_band_sum: f64 = ev.power_spectrum[0..6]
        .iter()
        .map(|x| (*x as f64).max(0.0))
        .sum();
    let geo_factor: f64 = if ev.field_completeness.power_spectrum_geo {
        (geo_band_sum / (geo_band_sum + 1.0)).clamp(0.0, 1.0)
    } else {
        0.5
    };
    let flux_factor: f64 = match (ev.flux_coupling.eta_status, ev.flux_coupling.eta) {
        ("computed", Some(eta)) => {
            let v = eta as f64;
            // Bound η to [0, 1] via squashed transform; sign explicit.
            (v.abs() / (v.abs() + 1.0)).clamp(0.0, 1.0)
        }
        _ => 0.5,
    };
    let causal_factor: f64 = if ev.field_completeness.causal_lead_residue_resolved {
        1.0
    } else {
        0.5
    };
    let adj_factor: f64 = match ev.adj_code {
        1 => 1.0, // Construct
        2 => 0.6, // Violation
        _ => 0.5,
    };
    let class_factor: f64 = if ev.class_tainted { 0.3 } else { 1.0 };
    // Status fields force NEUTRAL on unverified evidence.
    let perturbation_factor: f64 = match ev.perturbation_channel_evidence.status {
        s if s.starts_with("not_in_ghost_tile_schema") => 1.0,
        _ => 1.0,
    };
    let gear_factor: f64 = match ev.gear_normalized_timing.status {
        s if s.starts_with("ghost_tile_has_no_gear_id_field") => 1.0,
        _ => 1.0,
    };
    let asc_factor: f64 = match ev.asc_work_evidence.gamma_w_status {
        "asc_vector_format_unverified_no_compute" => 1.0,
        _ => 1.0,
    };
    let tile_score = kl_factor
        * geo_factor
        * flux_factor
        * causal_factor
        * adj_factor
        * class_factor
        * perturbation_factor
        * gear_factor
        * asc_factor;

    let components = serde_json::json!({
        "kl_divergence_factor":          kl_factor,
        "geo_band_sum_factor":           geo_factor,
        "flux_coupling_factor":          flux_factor,
        "flux_coupling_eta":             ev.flux_coupling.eta,
        "flux_coupling_eta_status":      ev.flux_coupling.eta_status,
        "causal_lead_resolved_factor":   causal_factor,
        "adjudication_code_factor":      adj_factor,
        "class_tainted_factor":          class_factor,
        "perturbation_channel_factor":   perturbation_factor,
        "perturbation_channel_status":   ev.perturbation_channel_evidence.status,
        "gear_timing_factor":            gear_factor,
        "gear_timing_status":            ev.gear_normalized_timing.status,
        "asc_work_factor":               asc_factor,
        "asc_work_status":               ev.asc_work_evidence.gamma_w_status,
        "tile_score":                    tile_score,
    });
    (tile_score, components)
}

// ─── M1.2.23 §6 + §7 sidecar readers ───────────────────────────────────────

#[derive(Debug, Deserialize, Default, Clone)]
struct GhostTimeMapStream {
    #[allow(dead_code)]
    stream_id: u32,
    #[serde(default)]
    gear_id: u32,
    #[serde(default)]
    dt_fs: f32,
    #[serde(default)]
    physical_time_fs: f64,
    #[serde(default)]
    dt_source: Option<String>,
    #[serde(default)]
    gear_id_status: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct GhostTimeMap {
    #[serde(default)]
    schema_version: u32,
    #[serde(default)]
    streams: Vec<GhostTimeMapStream>,
}

#[derive(Debug, Deserialize, Serialize, Default, Clone)]
#[allow(dead_code)]
struct GhostSiteMapEntry {
    component_id: Option<String>,
    stream_id: Option<u32>,
    frame_idx: Option<u64>,
    site_id: Option<u32>,
    aabb: Option<[f32; 6]>,
    centroid_xyz: Option<[f32; 3]>,
    voxel_xyz: Option<[i32; 3]>,
    residue_support: Option<Vec<i32>>,
    source: Option<String>,
    record_provenance: Option<GhostRecordProvenance>,
}

#[derive(Debug, Deserialize, Serialize, Default, Clone)]
struct GhostRecordProvenance {
    file: String,
    file_sha256: String,
    file_bytes: u64,
    record_index: u64,
    byte_offset: u64,
    record_bytes: u64,
    schema_version: u32,
    field_completeness_flags: u16,
    spatial_native_bit: u16,
}

#[derive(Debug, Deserialize, Default)]
struct GhostSiteMapManifestBinding {
    path: String,
    sha256: String,
    bytes: u64,
}

#[derive(Debug, Deserialize, Default)]
struct GhostSiteMap {
    #[serde(default)]
    schema: Option<String>,
    #[serde(default)]
    schema_version: u32,
    #[serde(default)]
    schema_kind: Option<String>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    coordinate_frame: Option<String>,
    #[serde(default)]
    run_id: Option<String>,
    #[serde(default)]
    native_v2_event_count: Option<u64>,
    #[serde(default)]
    source_manifest: Option<GhostSiteMapManifestBinding>,
    #[serde(default)]
    source_lattice_manifest: Option<String>,
    #[serde(default)]
    entries: Vec<GhostSiteMapEntry>,
}

#[derive(Debug, Serialize, Clone)]
struct MappedGhostComponent {
    component_id: String,
    stream_id: u32,
    site_id: u32,
    centroid_xyz: [f32; 3],
    aabb: Option<[f32; 6]>,
    residue_support: Vec<i32>,
    coordinate_frame: String,
    mapping_method: &'static str,
    mapping_source: String,
    ghost_event_count: u64,
    ghost_tile_score_mean: f64,
    ghost_tile_score_max: f64,
    zstr_bytes: u64,
    native_record_provenance: Option<GhostRecordProvenance>,
    mapping_scope: &'static str,
    exact_component_adapter: bool,
    component_step_min: Option<u64>,
    component_step_max: Option<u64>,
    exact_component_provenance: Option<serde_json::Value>,
}

fn sha256_path(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .with_context(|| format!("read {}", path.display()))?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn validate_native_v2_site_map(
    site_map: &GhostSiteMap,
    site_map_path: &Path,
    manifest: &Manifest,
    manifest_path: &Path,
    run_dir: &Path,
) -> Result<()> {
    anyhow::ensure!(
        site_map.schema.as_deref() == Some("prismos.ghost_v2_native_site_map.v1"),
        "native Ghost map {} has wrong schema",
        site_map_path.display()
    );
    anyhow::ensure!(
        site_map.schema_version == 2,
        "native Ghost map schema_version must be 2"
    );
    anyhow::ensure!(
        site_map.schema_kind.as_deref() == Some("pathb_ghost_v2_native_site_map"),
        "native Ghost map has wrong schema_kind"
    );
    anyhow::ensure!(
        site_map.status.as_deref() == Some("available_native_v2"),
        "native Ghost map is not available_native_v2"
    );
    anyhow::ensure!(
        site_map.coordinate_frame.as_deref() == Some("topology_cartesian_angstrom"),
        "native Ghost map has an unsupported coordinate frame"
    );
    anyhow::ensure!(
        site_map.run_id.as_deref() == Some(manifest.run_id.as_str()),
        "native Ghost map run_id mismatch"
    );
    anyhow::ensure!(
        !site_map.entries.is_empty(),
        "native Ghost map has zero entries"
    );
    anyhow::ensure!(
        site_map.native_v2_event_count == Some(site_map.entries.len() as u64),
        "native Ghost map event count does not equal entry count"
    );
    let manifest_binding = site_map
        .source_manifest
        .as_ref()
        .context("native Ghost map lacks source_manifest binding")?;
    anyhow::ensure!(
        Path::new(&manifest_binding.path) == manifest_path,
        "native Ghost map source manifest path mismatch"
    );
    let manifest_metadata = std::fs::metadata(manifest_path)?;
    anyhow::ensure!(
        manifest_binding.bytes == manifest_metadata.len(),
        "native Ghost map source manifest size mismatch"
    );
    anyhow::ensure!(
        manifest_binding.sha256 == sha256_path(manifest_path)?,
        "native Ghost map source manifest hash mismatch"
    );

    let canonical_run_dir = std::fs::canonicalize(run_dir)?;
    let mut seen_components = HashSet::new();
    let mut seen_events = HashSet::new();
    let mut source_hashes: HashMap<PathBuf, String> = HashMap::new();
    for entry in &site_map.entries {
        let component_id = entry
            .component_id
            .as_deref()
            .context("native Ghost map entry lacks component_id")?;
        anyhow::ensure!(
            seen_components.insert(component_id.to_string()),
            "duplicate native Ghost component_id {component_id}"
        );
        let stream_id = entry
            .stream_id
            .context("native Ghost map entry lacks stream_id")?;
        let frame_idx = entry
            .frame_idx
            .context("native Ghost map entry lacks frame_idx")?;
        let site_id = entry
            .site_id
            .context("native Ghost map entry lacks site_id")?;
        anyhow::ensure!(
            seen_events.insert((stream_id, frame_idx, site_id)),
            "duplicate native Ghost event identity"
        );
        let aabb = entry.aabb.context("native Ghost map entry lacks AABB")?;
        let centroid = entry
            .centroid_xyz
            .context("native Ghost map entry lacks centroid")?;
        anyhow::ensure!(
            aabb.iter()
                .chain(centroid.iter())
                .all(|value| value.is_finite()),
            "native Ghost map has non-finite spatial values"
        );
        for axis in 0..3 {
            anyhow::ensure!(
                aabb[axis + 3] > aabb[axis],
                "native Ghost map has non-positive AABB extent"
            );
            anyhow::ensure!(
                centroid[axis] >= aabb[axis] && centroid[axis] <= aabb[axis + 3],
                "native Ghost centroid falls outside AABB"
            );
            let midpoint = 0.5 * (aabb[axis] + aabb[axis + 3]);
            let tolerance = 1.0e-5_f32.max(midpoint.abs() * 1.0e-5);
            anyhow::ensure!(
                (centroid[axis] - midpoint).abs() <= tolerance,
                "native Ghost centroid does not match AABB midpoint"
            );
        }
        let provenance = entry
            .record_provenance
            .as_ref()
            .context("native Ghost map entry lacks record provenance")?;
        anyhow::ensure!(
            provenance.schema_version == 2,
            "native Ghost provenance schema is not v2"
        );
        anyhow::ensure!(
            provenance.spatial_native_bit == 0x0010,
            "native Ghost provenance spatial bit contract mismatch"
        );
        anyhow::ensure!(
            provenance.field_completeness_flags & 0x0010 != 0,
            "native Ghost provenance lacks spatial bit"
        );
        anyhow::ensure!(
            provenance.record_bytes == ghost_schema::RECORD_BYTES,
            "native Ghost record size mismatch"
        );
        anyhow::ensure!(
            provenance.byte_offset
                == ghost_schema::COUNTER_SECTOR_BYTES
                    + provenance.record_index * ghost_schema::RECORD_BYTES,
            "native Ghost byte offset mismatch"
        );
        let source = PathBuf::from(&provenance.file);
        let source_metadata = std::fs::symlink_metadata(&source)
            .with_context(|| format!("stat native Ghost source {}", source.display()))?;
        anyhow::ensure!(
            !source_metadata.file_type().is_symlink() && source_metadata.is_file(),
            "native Ghost source must be a regular non-symlink file"
        );
        let canonical_source = std::fs::canonicalize(&source)?;
        anyhow::ensure!(
            canonical_source.parent() == Some(canonical_run_dir.as_path()),
            "native Ghost source escapes the manifest run directory"
        );
        anyhow::ensure!(
            source_metadata.len() == provenance.file_bytes,
            "native Ghost source size mismatch"
        );
        let observed_hash = if let Some(value) = source_hashes.get(&canonical_source) {
            value.clone()
        } else {
            let value = sha256_path(&canonical_source)?;
            source_hashes.insert(canonical_source.clone(), value.clone());
            value
        };
        anyhow::ensure!(
            observed_hash == provenance.file_sha256,
            "native Ghost source hash mismatch"
        );
        let mut source_file = File::open(&canonical_source)?;
        source_file.seek(SeekFrom::Start(provenance.byte_offset))?;
        let mut record = [0u8; 4096];
        source_file.read_exact(&mut record)?;
        anyhow::ensure!(
            u64::from_le_bytes(record[0..8].try_into().unwrap()) == frame_idx,
            "native Ghost frame identity mismatch"
        );
        anyhow::ensure!(
            u32::from_le_bytes(record[8..12].try_into().unwrap()) == site_id,
            "native Ghost site identity mismatch"
        );
        anyhow::ensure!(
            u32::from_le_bytes(record[128..132].try_into().unwrap()) == 2,
            "native Ghost source record is not schema v2"
        );
        anyhow::ensure!(
            u16::from_le_bytes(record[138..140].try_into().unwrap()) & 0x0010 != 0,
            "native Ghost source record lacks spatial bit"
        );
        for axis in 0..3 {
            let lo = f32::from_le_bytes(record[160 + axis * 4..164 + axis * 4].try_into().unwrap());
            let hi = f32::from_le_bytes(record[172 + axis * 4..176 + axis * 4].try_into().unwrap());
            let center =
                f32::from_le_bytes(record[184 + axis * 4..188 + axis * 4].try_into().unwrap());
            anyhow::ensure!(
                lo == aabb[axis] && hi == aabb[axis + 3] && center == centroid[axis],
                "native Ghost map bytes do not match source record"
            );
        }
    }
    Ok(())
}

fn build_mapped_ghost_components(
    site_map: Option<&GhostSiteMap>,
    events: &[GhostTileEvent],
    zstr: &[ZstrStreamSummary],
) -> Vec<MappedGhostComponent> {
    let Some(site_map) = site_map else {
        return Vec::new();
    };
    let coordinate_frame = site_map
        .coordinate_frame
        .clone()
        .unwrap_or_else(|| "unknown".to_string());
    let mut components = Vec::new();
    for (entry_index, entry) in site_map.entries.iter().enumerate() {
        let (Some(stream_id), Some(site_id)) = (entry.stream_id, entry.site_id) else {
            continue;
        };
        let (centroid_xyz, mapping_method) = if let Some(centroid) = entry.centroid_xyz {
            (centroid, "ghost_site_map.centroid_xyz")
        } else if let Some(aabb) = entry.aabb {
            (
                [
                    0.5 * (aabb[0] + aabb[3]),
                    0.5 * (aabb[1] + aabb[4]),
                    0.5 * (aabb[2] + aabb[5]),
                ],
                "ghost_site_map.aabb_midpoint",
            )
        } else {
            // voxel_xyz lacks an Angstrom transform in the sidecar. Preserve it
            // in provenance but do not mislabel voxel indices as spatial Å.
            continue;
        };
        if centroid_xyz.iter().any(|value| !value.is_finite()) {
            continue;
        }
        let matching = events
            .iter()
            .filter(|event| {
                event.stream_id == stream_id
                    && event.site_id == site_id
                    && entry
                        .frame_idx
                        .map_or(true, |frame_idx| event.frame_idx == frame_idx)
            })
            .collect::<Vec<_>>();
        let mut score_sum = 0.0_f64;
        let mut score_max = 0.0_f64;
        for event in &matching {
            let (score, _) = score_ghost_tile_event(event);
            score_sum += score;
            score_max = score_max.max(score);
        }
        let score_mean = if matching.is_empty() {
            0.0
        } else {
            score_sum / matching.len() as f64
        };
        let zstr_bytes = zstr
            .iter()
            .filter(|summary| summary.stream_id == Some(stream_id))
            .map(|summary| summary.file_size)
            .sum();
        components.push(MappedGhostComponent {
            component_id: entry
                .component_id
                .clone()
                .unwrap_or_else(|| format!("ghost-site-{stream_id}-{site_id}-{entry_index}")),
            stream_id,
            site_id,
            centroid_xyz,
            aabb: entry.aabb,
            residue_support: entry.residue_support.clone().unwrap_or_default(),
            coordinate_frame: coordinate_frame.clone(),
            mapping_method: if entry.record_provenance.is_some() {
                "native_ghost_v2_record.aabb_centroid"
            } else {
                mapping_method
            },
            mapping_source: entry.record_provenance.as_ref().map_or_else(
                || {
                    entry
                        .source
                        .clone()
                        .unwrap_or_else(|| "ghost_site_map.json".to_string())
                },
                |provenance| format!("{}#byte_offset={}", provenance.file, provenance.byte_offset),
            ),
            ghost_event_count: matching.len() as u64,
            ghost_tile_score_mean: score_mean,
            ghost_tile_score_max: score_max,
            zstr_bytes,
            native_record_provenance: entry.record_provenance.clone(),
            mapping_scope: "legacy_site_map_spatial_component",
            exact_component_adapter: false,
            component_step_min: None,
            component_step_max: None,
            exact_component_provenance: None,
        });
    }
    components
}

/// Read path_a_completion.json from the run dir; extract branch-trace fields
/// per the follow-up invariant. Best-effort; missing fields → zeros + status.
fn read_branch_trace(run_dir: &Path, target: &str) -> serde_json::Value {
    let p = run_dir.join(format!("{}_path_a_completion.json", target));
    let s = match std::fs::read_to_string(&p) {
        Ok(s) => s,
        Err(_) => {
            return serde_json::json!({
                "status": "path_a_completion_json_absent",
                "branch_decision_status": "no_completion_json_to_read",
            })
        }
    };
    let v: serde_json::Value = match serde_json::from_str(&s) {
        Ok(v) => v,
        Err(e) => {
            return serde_json::json!({
                "status": "parse_error",
                "branch_decision_status": "completion_json_parse_failed",
                "error": e.to_string(),
            })
        }
    };
    // The watchdog's emit_minimal_completion_json carries branch_trace_by_stream
    // when CHUNK13 trace was registered. For graceful runs the field may be
    // absent — emit explicit "default_branch_only" only if the predicate
    // counts say so; otherwise honestly mark absent.
    let trace = v
        .get("branch_trace_by_stream")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let monomer_passthrough = v.get("monomer_passthrough").and_then(|x| x.as_bool());
    let bilateral_status = v
        .get("bilateral_status")
        .and_then(|x| x.as_str())
        .map(|x| x.to_string());
    serde_json::json!({
        "status":                  if trace.is_null() { "branch_trace_absent_in_completion_json" } else { "available" },
        "branch_trace_by_stream":  trace,
        "monomer_passthrough":     monomer_passthrough,
        "bilateral_status":        bilateral_status,
        "branch_decision_status":  if trace.is_null() {
            "default_branch_only_or_unrecorded"
        } else {
            "see_branch_trace_by_stream"
        },
        "predicate_liveness_status": "infer_from_branch_trace_field_diversity_when_available",
        "threshold_sigma_observation": 3.0,
        "threshold_sigma_discovery":   12.0,
        "threshold_source":            "directive_M1_2_23_section_4_default",
    })
}

// ─── Main ──────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = Args::parse();
    if args.require_native_v2_site_map && args.max_events != 0 {
        anyhow::bail!(
            "--require-native-v2-site-map requires --max-events=0 so every native record is joined"
        );
    }
    let log_level = if args.verbose { "info" } else { "warn" };
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", log_level);
    }
    env_logger::init();

    eprintln!("prism-ghost-evidence-scan — Path B feature scanner");
    eprintln!("manifest: {}", args.manifest.display());

    // ─── Phase 0 — manifest discovery ───────────────────────────────────────
    let manifest_text = std::fs::read_to_string(&args.manifest)
        .with_context(|| format!("read manifest {}", args.manifest.display()))?;
    let manifest: Manifest = serde_json::from_str(&manifest_text)
        .with_context(|| format!("parse manifest {}", args.manifest.display()))?;
    if manifest.schema_kind != "md_evidence_manifest" {
        eprintln!(
            "ERROR: manifest schema_kind = {:?}, expected \"md_evidence_manifest\"",
            manifest.schema_kind
        );
        std::process::exit(1);
    }
    if manifest.schema_version != 1 {
        eprintln!(
            "ERROR: manifest schema_version = {} (this scanner supports 1)",
            manifest.schema_version
        );
        std::process::exit(1);
    }
    let run_dir = args
        .manifest
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    let out_dir = args.out_dir.clone().unwrap_or(run_dir.clone());
    std::fs::create_dir_all(&out_dir)
        .with_context(|| format!("create out dir {}", out_dir.display()))?;
    let stem = manifest.target.clone();
    eprintln!(
        "run_id={} target={} stream_count={} run_dir={}",
        manifest.run_id,
        stem,
        manifest.stream_count,
        run_dir.display()
    );
    eprintln!("topology: {}", manifest.topology_input);

    // ─── Topology chain map (for bilateral logic) ────────────────────────────
    let mut chain_set: std::collections::HashSet<String> = std::collections::HashSet::new();
    if let Ok(t) = std::fs::read_to_string(&manifest.topology_input) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&t) {
            if let Some(arr) = v.get("chain_ids").and_then(|x| x.as_array()) {
                for c in arr.iter() {
                    if let Some(s) = c.as_str() {
                        chain_set.insert(s.to_string());
                    }
                }
            }
        }
    }
    let bilateral_status: &'static str = if chain_set.len() <= 1 {
        "not_applicable_monomer"
    } else {
        "deferred_real_partner_matching_required_no_implementation_in_this_commit"
    };
    eprintln!(
        "topology chains observed: {:?}  bilateral_status: {}",
        chain_set, bilateral_status
    );

    // ─── Phase 1 — file discovery + schema validation ──────────────────────
    let discovered = discover_files(&run_dir)?;
    let segment_prefix=format!("{}_stream",stem);
    let has_segments=std::fs::read_dir(&run_dir)?.filter_map(Result::ok).any(|e| e.file_type().is_ok_and(|t|t.is_dir()) && e.file_name().to_string_lossy().starts_with(&segment_prefix) && e.file_name().to_string_lossy().ends_with("_ghost_segments"));
    let verified_segments=if has_segments {prism_nhs::ghost_segment::discover_verified_ghost_streams(&run_dir,&stem)?} else {Vec::new()};
    for stream in &verified_segments {
        anyhow::ensure!(stream.manifest.completed && stream.manifest.produced_equals_persisted && stream.manifest.context.run_id==manifest.run_id,"Ghost segment run identity/conservation mismatch");
    }
    if !verified_segments.is_empty() && discovered.iter().any(|f|f.file_kind==FileKind::GhostTiles) {bail!("both flat and versioned Ghost populations present; explicit source reconciliation required to prevent duplicates");}
    let n_ghost = discovered
        .iter()
        .filter(|f| f.file_kind == FileKind::GhostTiles)
        .count() + verified_segments.iter().map(|s|s.segments.len()).sum::<usize>();
    let n_zstr = discovered
        .iter()
        .filter(|f| f.file_kind == FileKind::Zstr)
        .count();
    let n_v2 = discovered
        .iter()
        .filter(|f| f.file_kind == FileKind::PrismV2Trajectory)
        .count();
    eprintln!(
        "discovered files: ghost_tiles={} zstr={} prism_v2_trajectory={} (run_dir scan)",
        n_ghost, n_zstr, n_v2
    );

    // ─── M1.2.23 §6 + §7 — sidecar ingestion ───────────────────────────────
    let gtm: Option<GhostTimeMap> = std::fs::read_to_string(run_dir.join("ghost_time_map.json"))
        .ok()
        .and_then(|s| serde_json::from_str::<GhostTimeMap>(&s).ok());
    let site_map_path = args
        .site_map
        .clone()
        .unwrap_or_else(|| run_dir.join("ghost_site_map.json"));
    let gsm: Option<GhostSiteMap> = std::fs::read_to_string(&site_map_path)
        .ok()
        .and_then(|s| serde_json::from_str::<GhostSiteMap>(&s).ok());
    if args.require_native_v2_site_map {
        let native_map = gsm.as_ref().with_context(|| {
            format!(
                "required native Ghost v2 site map is missing or invalid JSON: {}",
                site_map_path.display()
            )
        })?;
        validate_native_v2_site_map(
            native_map,
            &site_map_path,
            &manifest,
            &args.manifest,
            &run_dir,
        )?;
    }
    let branch_trace = read_branch_trace(&run_dir, &stem);
    let gtm_status: &'static str = match gtm.as_ref() {
        Some(g) if !g.streams.is_empty() => "available",
        Some(_) => "present_but_empty_streams",
        None => "missing",
    };
    let gsm_status: &'static str = match gsm.as_ref() {
        Some(g)
            if g.schema.as_deref() == Some("prismos.ghost_v2_native_site_map.v1")
                && !g.entries.is_empty() =>
        {
            "available_native_v2_with_entries"
        }
        Some(g) if !g.entries.is_empty() => "available_with_entries",
        Some(_) => "present_but_empty_entries",
        None => "missing",
    };
    eprintln!(
        "sidecars: ghost_time_map={} ghost_site_map={} branch_trace={}",
        gtm_status,
        gsm_status,
        branch_trace
            .get("status")
            .and_then(|x| x.as_str())
            .unwrap_or("?")
    );

    if n_ghost == 0 && n_zstr == 0 {
        eprintln!(
            "WARNING: no Ghost or ZSTR files in run_dir. V2 monolithic was likely not live this run."
        );
        if !args.allow_missing && args.strict_schema {
            // strict mode: still error
        } else if !args.allow_missing {
            // default: still continue but will exit with code 2
        }
    }

    let mut validations: Vec<FileSchemaValidation> = Vec::new();
    for f in discovered.iter() {
        let v = match f.file_kind {
            FileKind::GhostTiles => validate_ghost_tiles_schema(f),
            FileKind::Zstr => validate_zstr_schema(f),
            FileKind::PrismV2Trajectory => validate_prism_v2_schema(f),
        };
        eprintln!(
            "  {} kind={} status={} parse={} records={:?} stride={:?}",
            v.file_path,
            v.file_kind,
            v.schema_status,
            v.parse_status,
            v.record_count,
            v.byte_stride
        );
        validations.push(v);
    }

    // ─── Phase 2 — ghost tile parsing (only files with consistent envelope) ─
    let mut all_ghost_events: Vec<GhostTileEvent> = Vec::new();
    let mut ghost_streams_summary: Vec<serde_json::Value> = Vec::new();
    for (f, val) in discovered.iter().zip(validations.iter()) {
        if f.file_kind != FileKind::GhostTiles {
            continue;
        }
        let n = val.record_count.unwrap_or(0);
        if n == 0 {
            ghost_streams_summary.push(serde_json::json!({
                "stream_id": f.stream_id,
                "file":      f.path.display().to_string(),
                "record_count": 0,
                "events_parsed": 0,
                "parse_status": val.parse_status,
            }));
            continue;
        }
        let events =
            parse_ghost_tiles_file(&f.path, f.stream_id.unwrap_or(u32::MAX), n, args.max_events)?;
        ghost_streams_summary.push(serde_json::json!({
            "stream_id":     f.stream_id,
            "file":          f.path.display().to_string(),
            "record_count":  n,
            "events_parsed": events.len(),
            "parse_status":  val.parse_status,
            "n_class_tainted": events.iter().filter(|e| e.class_tainted).count(),
            "n_with_finite_thermo_flux": events.iter().filter(|e| e.flux_coupling.eta_status == "computed").count(),
            "n_chain_a": events.iter().filter(|e| e.chain_id == Some('A')).count(),
            "n_chain_b": events.iter().filter(|e| e.chain_id == Some('B')).count(),
            "n_chain_unknown": events.iter().filter(|e| e.chain_id.is_none()).count(),
        }));
        all_ghost_events.extend(events);
    }
    for stream in &verified_segments {
        for segment in &stream.segments {
            let entry=&segment.entry;
            let path=prism_nhs::ghost_segment::segment_root(&run_dir,&stem,stream.manifest.stream_id).join(&entry.file);
            let events=parse_ghost_records(std::io::Cursor::new(&segment.payload),&path,u32::from(stream.manifest.stream_id),entry.record_count,args.max_events,entry.payload_offset,entry.first_record_ordinal,Some(&entry.file_sha256))?;
            ghost_streams_summary.push(serde_json::json!({"stream_id":stream.manifest.stream_id,"file":path,"segment_ordinal":entry.segment_ordinal,"first_record_ordinal":entry.first_record_ordinal,"manifest_sha256":stream.manifest_sha256,"file_sha256":entry.file_sha256,"payload_sha256":entry.payload_sha256,"record_count":entry.record_count,"events_parsed":events.len(),"parse_status":"native_segment_chain_verified"}));
            validations.push(FileSchemaValidation{file_path:path.display().to_string(),file_kind:"ghost_segment",stream_id:Some(u32::from(stream.manifest.stream_id)),file_size:entry.payload_offset+entry.payload_bytes,format_guess:"native_versioned_ghost_segment",schema_status:"schema_and_hash_chain_verified",parse_status:"complete_records_verified",record_count:Some(entry.record_count),byte_stride:Some(4096),parser_used:"ghost_segment::discover_verified_ghost_streams",missing_fields:vec![],unsafe_assumptions:vec![],notes:vec![format!("manifest_sha256={}",stream.manifest_sha256)]});
            all_ghost_events.extend(events);
        }
    }
    if args.max_events==0 && !verified_segments.is_empty() {
        let expected:u64=verified_segments.iter().map(|s|s.manifest.persisted_records).sum();
        anyhow::ensure!(all_ghost_events.len() as u64==expected,"Ghost complete population mismatch");
    }
    let mut event_ids=std::collections::HashSet::new();
    anyhow::ensure!(all_ghost_events.iter().all(|e|event_ids.insert(&e.event_id)),"duplicate Ghost event identities");
    if let Some(source)=&args.replay_lattice_from {
        replay_native_lattice(source,&run_dir,&stem,&out_dir,&manifest)?;
    }
    if args.require_native_v2_site_map {
        anyhow::ensure!(
            !all_ghost_events.is_empty(),
            "native Ghost v2 site-map mode observed zero Ghost events"
        );
        anyhow::ensure!(
            all_ghost_events
                .iter()
                .all(|event| matches!(event.schema_version,2|3)
                    && event.v2_field_completeness_flags & 0x0010 != 0),
            "native Ghost v2 site-map mode observed a non-v2 or non-spatial event"
        );
    }

    // ─── Phase 3 — ZSTR summary ─────────────────────────────────────────────
    let mut zstr_summaries: Vec<ZstrStreamSummary> = Vec::new();
    for f in discovered.iter() {
        if f.file_kind == FileKind::Zstr {
            zstr_summaries.push(summarize_zstr_file(f)?);
        }
    }

    // ─── Phase 4 — noise floor (per-stream) ────────────────────────────────
    let mut noise_floor_present = false;
    let mut noise_floor_records: Vec<serde_json::Value> = Vec::new();
    for stream_id in 0..(manifest.stream_count as u32) {
        if let Some(nf) = try_load_noise_floor(&run_dir, &stem, stream_id) {
            noise_floor_present = true;
            noise_floor_records.push(serde_json::json!({
                "stream_id": stream_id,
                "mu":        nf.mu,
                "sigma":     nf.sigma,
                "notes":     nf.notes,
            }));
        }
    }
    let _ = args.threshold_sigma;

    // ─── Phase 6 — bilateral status (already computed above) ───────────────

    // ─── Phase 7 — rank tile events ─────────────────────────────────────────
    let mut ranked: Vec<RankedTileEvent> = Vec::new();
    for ev in all_ghost_events.iter() {
        let (score, comps) = score_ghost_tile_event(ev);
        let event_value = serde_json::to_value(ev).unwrap_or(serde_json::Value::Null);
        ranked.push(RankedTileEvent {
            event_id: ev.event_id.clone(),
            rank: 0,
            tile_score: score,
            score_components: comps,
            event: event_value,
            limitations: vec![
                "perturbation_channel_unavailable_in_ghost_tile_schema",
                "gear_id_unavailable_in_ghost_tile_schema",
                "asc_vector_format_unverified_no_gamma_w_compute",
                "ranking_components_neutral_when_unverified",
            ],
        });
    }
    ranked.sort_by(|a, b| {
        b.tile_score
            .partial_cmp(&a.tile_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (i, r) in ranked.iter_mut().enumerate() {
        r.rank = i as u32;
    }
    let n_emit = if args.emit_all_events {
        ranked.len()
    } else {
        ranked.len().min(200)
    };
    let ranked_to_emit: Vec<&RankedTileEvent> = ranked.iter().take(n_emit).collect();

    // ─── Phase 8 — candidate features for materializer ─────────────────────
    let mut per_stream_ghost_support: serde_json::Map<String, serde_json::Value> =
        serde_json::Map::new();
    for ev in all_ghost_events.iter() {
        let key = ev.stream_id.to_string();
        let entry = per_stream_ghost_support
            .entry(key)
            .or_insert_with(|| serde_json::json!({"events":0,"with_finite_eta":0,"site_ids":[]}));
        let m = entry.as_object_mut().unwrap();
        let prev = m.get("events").and_then(|x| x.as_u64()).unwrap_or(0);
        m.insert("events".into(), serde_json::Value::from(prev + 1));
        if ev.flux_coupling.eta_status == "computed" {
            let p = m
                .get("with_finite_eta")
                .and_then(|x| x.as_u64())
                .unwrap_or(0);
            m.insert("with_finite_eta".into(), serde_json::Value::from(p + 1));
        }
    }

    // ─── Output: tile_field_completeness.json ──────────────────────────────
    let any_intensity_packed = false; // Ghost tiles never carry intensity_packed.
    let any_qi_extracted = false;
    // GEAR-PROVENANCE-002 — "present" means a gear inside the documented
    // 0..=3 encoding. The unmeasured sentinel is non-zero, so the previous
    // `!= 0` test would have counted a declared ABSENCE as a present gear.
    let any_gear_id_present = zstr_summaries.iter().any(|s| {
        s.first_header
            .as_ref()
            .map(|h| h.gear_id <= 3)
            .unwrap_or(false)
    });
    let any_gear_id_unmeasured_sentinel = zstr_summaries.iter().any(|s| {
        s.first_header
            .as_ref()
            .map(|h| h.gear_id == ZSTR_GEAR_ID_UNMEASURED)
            .unwrap_or(false)
    });
    let any_dt_resolved = zstr_summaries.iter().any(|s| {
        s.first_header
            .as_ref()
            .map(|h| h.dt_fs > 0.0)
            .unwrap_or(false)
    });
    let asc_vectors_present = manifest
        .artifacts
        .iter()
        .any(|a| a.present && a.path.ends_with("_asc_vectors.bin"))
        || std::fs::read_dir(&run_dir)
            .map(|d| {
                d.flatten().any(|e| {
                    e.path()
                        .file_name()
                        .and_then(|s| s.to_str())
                        .map(|n| n.ends_with("_asc_vectors.bin"))
                        .unwrap_or(false)
                })
            })
            .unwrap_or(false);

    // P7a — ASC artifact schema, verified from the PRODUCER's own sidecar.
    //
    // This scan previously hardcoded `asc_schema_verified: false` with the
    // reason "format must be verified from producer code in a follow-up".
    // nhs_rt_full now writes a `*_asc_trajectory.schema.json` next to the
    // artifact at file-creation time, so the declaration cannot drift from
    // what was emitted. Absence of the sidecar keeps the old honest `false`
    // rather than assuming a format.
    let asc_schema: Option<serde_json::Value> = std::fs::read_dir(&run_dir)
        .ok()
        .and_then(|d| {
            d.flatten()
                .map(|e| e.path())
                .find(|p| {
                    p.file_name()
                        .and_then(|s| s.to_str())
                        .map(|n| n.ends_with("_asc_trajectory.schema.json"))
                        .unwrap_or(false)
                })
                .and_then(|p| std::fs::read_to_string(p).ok())
        })
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok());
    let asc_schema_verified = asc_schema
        .as_ref()
        .and_then(|v| v.get("schema_kind"))
        .and_then(|v| v.as_str())
        .map(|k| k == "prism_asc_trajectory_v1")
        .unwrap_or(false);
    let asc_content_identity: String = asc_schema
        .as_ref()
        .and_then(|v| v.get("content_identity"))
        .and_then(|v| v.as_str())
        .unwrap_or("UNDECLARED_NO_PRODUCER_SIDECAR")
        .to_string();
    let asc_schema_reason: String = if asc_schema_verified {
        format!(
            "verified from producer sidecar *_asc_trajectory.schema.json \
             (prism_asc_trajectory_v1); content_identity={asc_content_identity}. \
             NOTE: the payload is the TOTAL AMBER force buffer, not an isolated ASC \
             steering vector — the ASC term is not recoverable from it by subtraction."
        )
    } else {
        "no *_asc_trajectory.schema.json sidecar in this run directory; the artifact \
         carries no in-band schema header, so its format remains unverified"
            .to_string()
    };
    let tile_field_completeness = serde_json::json!({
        "schema_version": 1,
        "schema_kind":    "pathb_ghost_zstr_tile_field_completeness",
        "run_id":         manifest.run_id,
        "target":         stem,
        "files_validated": validations,
        "intensity_packed_present":          any_intensity_packed,
        "intensity_packed_present_reason":   "ghost_tile_schema_has_no_intensity_packed_field_see_richspike_in_interferometric_adjudicator_rs",
        "qi_bits_extracted":                 any_qi_extracted,
        "wavelength_mapping_verified":       true,
        "wavelength_mapping_source":         "interferometric_adjudicator.rs:712-735",
        "uv_mapping": {
            "0": "260nm_PHE_dominant",
            "1": "280nm_TRP_primary",
            "2": "305nm_TRP_gaussian_tail",
            "3": "320nm_TRP_control",
        },
        "gear_id_present":                   any_gear_id_present,
        "gear_id_schema_source":             "zstr.rs:88_gear_id_u32_offset_24",
        // GEAR-PROVENANCE-002 — replaces the unconditional
        // "gear_id_wave_a_default_zero": true assertion, which claimed a
        // producer behavior the scan never checked.
        "gear_id_unmeasured_sentinel_observed": any_gear_id_unmeasured_sentinel,
        "gear_id_unmeasured_sentinel_value":  ZSTR_GEAR_ID_UNMEASURED,
        "dt_resolved":                       any_dt_resolved,
        "dt_schema_source":                  "zstr.rs:71_dt_f32_offset_8_picoseconds",
        "physical_time_axis_available":      any_dt_resolved,
        "asc_vectors_present":               asc_vectors_present,
        "asc_schema_verified":               asc_schema_verified,
        "asc_schema_verified_reason":        asc_schema_reason,
        "asc_content_identity":              asc_content_identity,
        "asc_tile_mapping_available":        false,
        "gamma_w_computable":                false,
        "gamma_w_used_in_score":             false,
        "noise_floor_present":               noise_floor_present,
        "n_noise_floor_streams":             noise_floor_records.len(),
        "bilateral_status":                  bilateral_status,
        "topology_chains":                   chain_set.iter().collect::<Vec<_>>(),
        // M1.2.23 §5/§6/§7 + branch-trace follow-up — sidecar + trace status.
        "ghost_time_map_status":             gtm_status,
        "ghost_site_map_status":             gsm_status,
        "ghost_site_map_status_field":       gsm.as_ref().and_then(|g| g.status.clone()),
        "ghost_site_map_coordinate_frame":   gsm.as_ref().and_then(|g| g.coordinate_frame.clone()),
        "branch_trace_status":               branch_trace.get("status").cloned().unwrap_or(serde_json::Value::Null),
        "branch_decision_status":            branch_trace.get("branch_decision_status").cloned().unwrap_or(serde_json::Value::Null),
        "threshold_sigma_observation":       3.0,
        "threshold_sigma_discovery":         12.0,
        "v1_v2_schema_overlay_supported":    true,
        "v2_schema_constants_source":        "ghost_tile.rs:GHOST_FRAME_SCHEMA_V2_constant_offsets_at_GHOST_V2_OFFSET_*",
    });
    let tfc_path = out_dir.join("tile_field_completeness.json");
    write_json(&tfc_path, &tile_field_completeness)?;
    eprintln!("✓ wrote {}", tfc_path.display());

    // ─── Output: ghost_tile_evidence.json ──────────────────────────────────
    let n_ghost_events = all_ghost_events.len();
    let any_ghost_files = n_ghost > 0;
    // M1.2.25 — data-driven plane-1/2/3 status (was a static limitation string).
    // Per-plane nonzero counts so the report can distinguish unwired from no-signal.
    let mut plane_nonzero_counts: [u64; 4] = [0; 4];
    for ev in all_ghost_events.iter() {
        for plane in 0..4 {
            if ev.plane_evidence.cosine_evidence_mask & (1 << plane) != 0 {
                plane_nonzero_counts[plane] += 1;
            }
        }
    }
    let planes_1_3_any_populated =
        plane_nonzero_counts[1] > 0 || plane_nonzero_counts[2] > 0 || plane_nonzero_counts[3] > 0;
    let planes_status_string: String = if planes_1_3_any_populated {
        format!(
            "observed_cosine_eligible_counts_geo_{}_caus_{}_therm_{}_chem_{}",
            plane_nonzero_counts[0],
            plane_nonzero_counts[1],
            plane_nonzero_counts[2],
            plane_nonzero_counts[3],
        )
    } else {
        format!(
            "no_above_floor_evidence_geo_{}_caus_0_therm_0_chem_0_producer_occupancy_separately_reported",
            plane_nonzero_counts[0]
        )
    };
    let ghost_status: &str = if !any_ghost_files {
        "missing"
    } else if n_ghost_events > 0 {
        "available"
    } else {
        "partial_no_records"
    };
    let ghost_evidence = serde_json::json!({
        "schema_version":   1,
        "schema_kind":      "pathb_ghost_tile_evidence",
        "source_manifest":  args.manifest.display().to_string(),
        "run_id":           manifest.run_id,
        "target":           stem,
        "status":           ghost_status,
        "parser": {
            "format":            "GhostTileFrame_v9d_sector_locked_4096B",
            "schema_source":     "ghost_tile.rs:80-213_const_assert_offsets",
            "layout_verified":   true,
            "unsafe_assumptions": Vec::<&str>::new(),
        },
        "n_ghost_files":      n_ghost,
        "n_events_parsed":    n_ghost_events,
        "spectral_plane_interpretation": "producer occupancy and measured above-floor content separate; exact 1e-7 padding excluded from cosine",
        "producer_mask_absent_events": all_ghost_events.iter().filter(|e|e.plane_evidence.producer_populated_mask.is_none()).count(),
        "exact_floor_only_counts": (0..4).map(|p|all_ghost_events.iter().filter(|e|e.plane_evidence.exact_floor_only_mask&(1<<p)!=0).count()).collect::<Vec<_>>(),
        "streams":            ghost_streams_summary,
        "field_completeness": {
            "frame_idx":                          true,
            "site_id":                            true,
            "chain_id":                           "u8_at_offset_12_accepts_numeric_0_1_and_historical_ascii_A_B",
            "adjudication_code":                  "u8_at_offset_13",
            "kl_divergence":                      "f32_at_offset_16",
            "power_spectrum_geo_plane_0":         "f32x6_at_offset_20",
            "power_spectrum_planes_1_3":          planes_status_string.clone(),
            "thermo_flux_wd_change_vib_energy":   "f32x2_at_offset_116_NaN_until_upstream_wires_them_per_ghost_tile_rs_136_141",
            "causal_lead_residue":                "u32_at_offset_124_sentinel_uMAX_means_unresolved",
            "intensity_packed":                   "absent_in_ghost_tile_schema",
            "gear_id":                            "absent_in_ghost_tile_schema_reserved_payload_unwired",
            "asc_vectors":                        "format_unverified",
        },
    });
    let ge_path = out_dir.join("ghost_tile_evidence.json");
    write_json(&ge_path, &ghost_evidence)?;
    eprintln!("✓ wrote {}", ge_path.display());

    // ─── Output: zstr_event_summary.json ───────────────────────────────────
    let zstr_status: &str = if n_zstr == 0 {
        "missing"
    } else {
        "header_only_decoded"
    };
    let zstr_summary_json = serde_json::json!({
        "schema_version": 1,
        "schema_kind":    "pathb_zstr_event_summary",
        "run_id":         manifest.run_id,
        "target":         stem,
        "status":         zstr_status,
        "parser": {
            "format":          "ZstrFrameHeader_4096B_per_slot_5slot_ring",
            "schema_source":   "zstr.rs:67-114",
            "layout_verified": true,
        },
        "n_zstr_files":   n_zstr,
        "streams":        zstr_summaries,
        "decode_status":  if n_zstr == 0 { "no_zstr_files" } else { "first_header_only_payload_not_decoded_this_commit" },
        "notes": [
            "ZstrFrameHeader fields verified by offset; per-slot positions/forces payload not decoded in this commit.",
            "On-disk file pattern: prism_zstr_<pid>_<stream>.bin (cited: nhs_rt_full.rs:7638).",
        ],
    });
    let zs_path = out_dir.join("zstr_event_summary.json");
    write_json(&zs_path, &zstr_summary_json)?;
    eprintln!("✓ wrote {}", zs_path.display());

    // ─── Output: ranked_tile_events.json ───────────────────────────────────
    let mut ranked_limitations: Vec<&'static str> = vec![
        "ghost_tile_schema_has_no_perturbation_channel",
        "ghost_tile_schema_has_no_gear_id",
        "asc_work_integral_not_computed_format_unverified",
        "thermo_flux_NaN_until_upstream_wires_them_per_producer_note",
    ];
    if !planes_1_3_any_populated {
        ranked_limitations.push("power_spectrum_planes_1_3_have_no_cosine_eligible_observations; producer_occupancy_and_padding_are_separately_reported");
    }
    let ranked_json = serde_json::json!({
        "schema_version":     1,
        "schema_kind":        "pathb_ranked_tile_events",
        "run_id":             manifest.run_id,
        "target":             stem,
        "n_events_total":     all_ghost_events.len(),
        "n_events_emitted":   ranked_to_emit.len(),
        "emit_all_events":    args.emit_all_events,
        "ranking_methodology":"product_of_factors_unverified_components_neutral_1_0",
        "ranked_events":      ranked_to_emit,
        "limitations":        ranked_limitations,
    });
    let r_path = out_dir.join("ranked_tile_events.json");
    write_json(&r_path, &ranked_json)?;
    eprintln!("✓ wrote {}", r_path.display());

    // ─── Output: mar_plane_completeness.json (M1.2.25 Invariant 2) ─────────
    // Per-plane statistics (mean/max/p95/nonzero count) so downstream audit
    // can verify Pillar 2 (causality plane nonzero when geometry plane nonzero).
    let plane_names = ["geometry", "causality", "thermodynamics", "chemistry"];
    let producer_sources = [
        "so3_project.cu:518_geo_power_spectrum",
        "so3_project.cu:522_caus_power_spectrum",
        "so3_project.cu:526_therm_power_spectrum",
        "so3_project.cu:530_chem_power_spectrum",
    ];
    let total_events = all_ghost_events.len() as u64;
    let mut plane_records = Vec::with_capacity(4);
    for plane in 0..4usize {
        let mut sums: f64 = 0.0;
        let mut maxv: f32 = 0.0;
        let mut all_vals: Vec<f32> = Vec::with_capacity(all_ghost_events.len() * 6);
        for ev in all_ghost_events.iter() {
            for l in 0..6 {
                let v = ev.power_spectrum[plane * 6 + l];
                if v.is_finite() {
                    sums += v as f64;
                    if v > maxv {
                        maxv = v;
                    }
                    all_vals.push(v);
                }
            }
        }
        let n_total = all_vals.len();
        let mean = if n_total > 0 {
            sums / (n_total as f64)
        } else {
            0.0
        };
        all_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p95 = if n_total > 0 {
            let idx = ((n_total as f64) * 0.95) as usize;
            all_vals[idx.min(n_total - 1)]
        } else {
            0.0
        };
        let nonzero_count = plane_nonzero_counts[plane];
        let status = if total_events == 0 {
            "no_events"
        } else if nonzero_count == 0 {
            "no_cosine_eligible_observations"
        } else if (nonzero_count as f64) / (total_events as f64) >= 0.5 {
            "majority_cosine_eligible_observations"
        } else {
            "minority_cosine_eligible_observations"
        };
        plane_records.push(serde_json::json!({
            "plane_id":         plane,
            "plane_name":       plane_names[plane],
            "nonzero_count": all_ghost_events.iter().filter(|ev|ev.power_spectrum[plane*6..plane*6+6].iter().any(|v|*v!=0.0)).count(),
            "cosine_eligible_count": nonzero_count,
            "exact_floor_only_count": all_ghost_events.iter().filter(|ev|ev.plane_evidence.exact_floor_only_mask&(1<<plane)!=0).count(),
            "producer_occupancy_unknown_count": all_ghost_events.iter().filter(|ev|ev.plane_evidence.producer_populated_mask.is_none()).count(),
            "total_count":      total_events,
            "mean":             mean,
            "max":              maxv,
            "p95":              p95,
            "status":           status,
            "producer_source":  producer_sources[plane],
            "scanner_source":   "prism_ghost_evidence_scan.rs::mar_plane_completeness_emit_M1_2_25",
            "missing_reason":   if nonzero_count == 0 && total_events > 0 {
                "no_eligible_spectral_content; exact_floor_padding_does_not_prove_producer_occupancy_or_measured_coupling"
            } else { "none" },
            "eligible_for_interferometric_contrast": plane > 0 && nonzero_count > 0,
        }));
    }
    let geo_nz = plane_nonzero_counts[0];
    let caus_nz = plane_nonzero_counts[1];
    let therm_nz = plane_nonzero_counts[2];
    let chem_nz = plane_nonzero_counts[3];
    let pillar_2_violation = geo_nz > 0 && caus_nz == 0;
    let full_4plane = geo_nz > 0 && caus_nz > 0 && (therm_nz > 0 || chem_nz > 0);
    let c_total_status = if full_4plane {
        "computable_4plane"
    } else if geo_nz > 0 && caus_nz > 0 {
        "computable_2plane_geo_plus_causality"
    } else if geo_nz > 0 {
        "partial_geometry_only"
    } else {
        "no_signal"
    };
    let mar_plane_completeness = serde_json::json!({
        "schema_version":           1,
        "schema_kind":              "pathb_mar_plane_completeness",
        "run_id":                   manifest.run_id,
        "target":                   stem,
        "n_events_total":           total_events,
        "geometry_nonzero":         geo_nz,
        "causality_nonzero":        caus_nz,
        "thermodynamics_nonzero":   therm_nz,
        "chemistry_nonzero":        chem_nz,
        "full_4plane_available":    full_4plane,
        "c_total_status":           c_total_status,
        "pillar_2_driver_violation": pillar_2_violation,
        "pillar_2_violation_reason": if pillar_2_violation {
            "geometry_present_but_kcc_causality_plane_zero"
        } else { "none" },
        "planes":                   plane_records,
    });
    let mpc_path = out_dir.join("mar_plane_completeness.json");
    write_json(&mpc_path, &mar_plane_completeness)?;
    eprintln!("✓ wrote {}", mpc_path.display());

    if args.emit_ndjson {
        let nd_path = out_dir.join("ghost_tile_events.ndjson");
        let f = File::create(&nd_path)?;
        let mut bw = std::io::BufWriter::new(f);
        for ev in all_ghost_events.iter() {
            use std::io::Write as _;
            let s = serde_json::to_string(ev)?;
            bw.write_all(s.as_bytes())?;
            bw.write_all(b"\n")?;
        }
        use std::io::Write as _;
        bw.flush()?;
        eprintln!("✓ wrote {}", nd_path.display());
    }

    // ─── Output: ghost_zstr_candidate_features.json ────────────────────────
    let adapter_namespace = args.exact_component_adapter_namespace.as_deref();
    let mut exact_native_component_joins=Vec::new();
    let mut exact_component_adapter_receipt: Option<serde_json::Value> = None;
    let mut exact_component_adapter_source_namespace: Option<String> = None;
    let mut exact_adapter_mapped_components: Vec<MappedGhostComponent> = Vec::new();
    if let Some(source)=gsm.as_ref().and_then(|map|map.source_lattice_manifest.as_deref()) {
        let product = emit_exact_component_event_join(
            Path::new(source),
            &all_ghost_events,
            &out_dir,
            &manifest,
            "original_graph",
            adapter_namespace,
            &zstr_summaries,
        )?;
        if let Some(receipt) = product.adapter_receipt {
            exact_component_adapter_source_namespace = Some("original_graph".to_string());
            exact_adapter_mapped_components = product.mapped_components;
            exact_component_adapter_receipt = Some(receipt);
        }
        exact_native_component_joins.push(product.receipt);
    }
    if args.replay_lattice_from.is_some() {
        let product = emit_exact_component_event_join(
            &out_dir.join("native_floor_aware_lattice/binding_sites_lattice.json"),
            &all_ghost_events,
            &out_dir,
            &manifest,
            "floor_aware_graph",
            adapter_namespace,
            &zstr_summaries,
        )?;
        if let Some(receipt) = product.adapter_receipt {
            exact_component_adapter_source_namespace = Some("floor_aware_graph".to_string());
            exact_adapter_mapped_components = product.mapped_components;
            exact_component_adapter_receipt = Some(receipt);
        }
        exact_native_component_joins.push(product.receipt);
    }
    if let Some(namespace) = adapter_namespace {
        anyhow::ensure!(
            exact_component_adapter_receipt.is_some(),
            "exact component adapter namespace `{namespace}` was requested but no emitted graph namespace matched it"
        );
    }
    let legacy_mapped_components =
        build_mapped_ghost_components(gsm.as_ref(), &all_ghost_events, &zstr_summaries);
    let legacy_mapped_component_count = legacy_mapped_components.len();
    let exact_component_adapter_selected = exact_component_adapter_receipt.is_some();
    let mapped_components = if exact_component_adapter_selected {
        exact_adapter_mapped_components
    } else {
        legacy_mapped_components
    };
    if args.require_native_v2_site_map && !exact_component_adapter_selected {
        let expected_entries = gsm.as_ref().map(|value| value.entries.len()).unwrap_or(0);
        anyhow::ensure!(
            mapped_components.len() == expected_entries,
            "native Ghost map/component join mismatch: entries={} mapped={}",
            expected_entries,
            mapped_components.len()
        );
        anyhow::ensure!(
            mapped_components
                .iter()
                .all(|component| component.ghost_event_count == 1),
            "native Ghost map did not join one-to-one to exact frame events"
        );
    }
    let spatial_status = if exact_component_adapter_selected && !mapped_components.is_empty() {
        "mapped_components_from_exact_native_component_join_v1"
    } else if exact_component_adapter_selected {
        "exact_native_component_join_selected_but_no_single_stream_site_aabb_components"
    } else if args.require_native_v2_site_map && !mapped_components.is_empty() {
        "mapped_components_from_native_ghost_v2_site_map"
    } else if !mapped_components.is_empty() {
        "mapped_components_from_ghost_site_map_v1"
    } else if gsm.as_ref().is_some_and(|map| !map.entries.is_empty()) {
        "unmapped_ghost_site_map_entries_lack_centroid_or_aabb"
    } else {
        "unmapped_ghost_tile_records_have_no_usable_ghost_site_map"
    };
    let candidate_features = serde_json::json!({
        "schema_version":   1,
        "schema_kind":      "pathb_ghost_zstr_candidate_features",
        "exact_native_component_joins": exact_native_component_joins,
        "exact_component_adapter": {
            "requested_namespace": args.exact_component_adapter_namespace.as_deref(),
            "selected_namespace": exact_component_adapter_source_namespace.as_deref(),
            "selected": exact_component_adapter_selected,
            "receipt": exact_component_adapter_receipt.as_ref(),
            "legacy_mapped_component_count": legacy_mapped_component_count,
            "effective_mapped_component_count": mapped_components.len(),
            "region_filter_semantics": if exact_component_adapter_selected {
                "materializer filters exact adapter components by stream_id in region.per_stream_spikes and inclusive recorded-step interval overlap; broad window overlap only, not exact coincident causal ownership"
            } else {
                "legacy mapped components retain prior geometry-only materializer association"
            }
        },
        "run_id":           manifest.run_id,
        "target":           stem,
        "spatial_status":   spatial_status,
        "mapped_components": &mapped_components,
        "per_stream_ghost_support":  per_stream_ghost_support,
        "noise_floor_records":       noise_floor_records,
        "feature_handoff_notes": [
            if args.require_native_v2_site_map {
                "legacy_mapped_components_use_stream_id_frame_idx_and_site_id"
            } else {
                "legacy_mapped_components_use_stream_id_and_site_id"
            },
            "exact_native_component_joins_use_stream_id_stream_record_ordinal_and_verified_record_segment_hashes_byte_offsets_and_physical_steps",
            "exact_component_adapter_selected_only_by_explicit_namespace_and_preserves_original_graph_identity",
            "exact_component_adapter_region_filter_is_stream_identity_plus_inclusive_recorded_step_window_overlap_not_rank_match",
            "centroid_xyz_or_aabb_midpoint_is_required_for_spatial_admission",
            "voxel_xyz_without_an_explicit_angstrom_transform_is_not_promoted_to_spatial_evidence",
        ],
        "site_map_input": {
            "path": site_map_path.display().to_string(),
            "sha256": sha256_path(&site_map_path).ok(),
            "required_native_v2": args.require_native_v2_site_map,
            "schema": gsm.as_ref().and_then(|value| value.schema.clone()),
        },
        "asc_work_evidence_summary": {
            "gamma_w_computable_from_disk_in_this_commit": false,
            "blocking_reasons": [
                "asc_vectors.bin has no schema header",
                "force/displacement vector layout not verified from producer source",
                "tile->voxel mapping not available",
            ],
            "follow_up": "verify asc_vectors layout from engine emit code at nhs_rt_full.rs ~10150",
        },
        "perturbation_channel_evidence_summary": {
            "qi_bits_in_ghost_tile":      false,
            "qi_bits_in_richspike":       true,
            "uv_mapping_source":          "interferometric_adjudicator.rs:712-735",
            "follow_up": "to attach perturbation channel to a tile event, cross-ref with PRSPK001 spikes in same time window",
        },
        "gear_normalized_timing_summary": {
            "gear_id_field_in_zstr_schema": true,
            // GEAR-PROVENANCE-002 — was `"gear_id_wave_a_default": 0`, which
            // published a producer literal as if the scan had observed it.
            "gear_id_unmeasured_sentinel":  ZSTR_GEAR_ID_UNMEASURED,
            "gear_id_unmeasured_sentinel_observed": any_gear_id_unmeasured_sentinel,
            "dt_field_in_zstr_schema":      true,
            "dt_units":                     "picoseconds",
            "ghost_tile_has_gear":          false,
        },
        "bilateral_status":  bilateral_status,
    });
    let cf_path = out_dir.join("ghost_zstr_candidate_features.json");
    write_json(&cf_path, &candidate_features)?;
    eprintln!("✓ wrote {}", cf_path.display());

    // ─── Output: ghost_zstr_scan_report.json ───────────────────────────────
    let any_files = n_ghost > 0 || n_zstr > 0;
    let materializer_ready: bool = n_ghost_events > 0 && !mapped_components.is_empty();
    let report = serde_json::json!({
        "schema_version":   1,
        "schema_kind":      "pathb_ghost_zstr_scan_report",
        "run_id":           manifest.run_id,
        "target":           stem,
        "scanned_files":    discovered.iter().map(|f| f.path.display().to_string()).collect::<Vec<_>>(),
        "n_ghost_files":    n_ghost,
        "n_zstr_files":     n_zstr,
        "n_v2_files":       n_v2,
        "n_events":         all_ghost_events.len(),
        "n_ranked_events":  ranked.len(),
        "files_validated":  validations.len(),
        "materializer_ready": materializer_ready,
        "mapped_component_count": mapped_components.len(),
        "spatial_status": spatial_status,
        "exact_component_adapter_requested_namespace": adapter_namespace,
        "exact_component_adapter_selected_namespace": exact_component_adapter_source_namespace,
        "exact_component_adapter_selected": exact_component_adapter_selected,
        "limitations": [
            "perturbation_channel_not_in_ghost_tile_schema",
            "gear_id_not_in_ghost_tile_schema",
            "asc_vectors_format_unverified",
            "spatial_link_requires_ghost_site_map_centroid_or_aabb",
            "bilateral_logic_disabled_for_monomer_or_deferred_for_dimer",
        ],
        "recommended_next_materializer_integration_step": if !any_files {
            "rerun_engine_with_v2_live_to_produce_ghost_tile_files_then_re_scan"
        } else if exact_component_adapter_selected && !materializer_ready {
            "inspect_exact_component_adapter_receipt_unmappable_counts_or_missing_component_aabb"
        } else if !materializer_ready {
            "emit_ghost_site_map_entries_with_centroid_xyz_or_aabb_and_matching_stream_site_ids"
        } else {
            "consume_ghost_zstr_candidate_features_in_prism-materialize-sites_v3"
        },
        "honest_assessment": if any_files {
            "ghost_zstr_evidence_extracted_for_downstream_materializer"
        } else {
            "no_ghost_or_zstr_files_present_in_run_dir_v2_was_not_live"
        },
        // M1.2.23 §6/§7 + branch-trace follow-up consumption summary.
        "sidecars": {
            "ghost_time_map_status": gtm_status,
            "ghost_site_map_status": gsm_status,
            "ghost_site_map_path": site_map_path.display().to_string(),
            "ghost_site_map_required_native_v2": args.require_native_v2_site_map,
        },
        "branch_trace": branch_trace,
    });
    let rep_path = out_dir.join("ghost_zstr_scan_report.json");
    write_json(&rep_path, &report)?;
    eprintln!("✓ wrote {}", rep_path.display());

    // ─── Exit policy ────────────────────────────────────────────────────────
    if !any_files {
        if args.allow_missing {
            eprintln!("\nNo Ghost/ZSTR files in run_dir; --allow-missing → exit 0 with honest 'missing' status.");
            return Ok(());
        } else {
            eprintln!("\nNo Ghost/ZSTR files in run_dir; --allow-missing not set → exit 2.");
            std::process::exit(2);
        }
    }
    if args.strict_schema {
        let any_unverified = validations.iter().any(|v| {
            v.schema_status
                .starts_with("verified_offsets_but_file_truncated")
                || v.parse_status == "envelope_size_mismatch"
        });
        if any_unverified {
            eprintln!(
                "\n--strict-schema set and one or more files failed envelope validation → exit 3."
            );
            std::process::exit(3);
        }
    }
    eprintln!(
        "\nDone. {} ghost files / {} zstr files / {} events / {} ranked.",
        n_ghost,
        n_zstr,
        all_ghost_events.len(),
        ranked.len()
    );
    Ok(())
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let f = File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut bw = std::io::BufWriter::new(f);
    serde_json::to_writer_pretty(&mut bw, value)?;
    use std::io::Write as _;
    bw.flush()?;
    Ok(())
}

#[cfg(not(feature="gpu"))]
fn replay_native_lattice(_source:&Path,_run:&Path,_stem:&str,_out:&Path,_manifest:&Manifest)->Result<()> {
    bail!("--replay-lattice-from requires the GPU build of this scanner; no CPU substitute")
}
#[cfg(feature="gpu")]
fn replay_native_lattice(source:&Path,run:&Path,stem:&str,out:&Path,manifest:&Manifest)->Result<()> {
    use prism_nhs::ghost_phase_lattice::{load_verified_ghost_segments_from_dir,GhostPhaseLatticeConfig,GhostPhaseLattice4D,GhostLatticeAcceptedEdgeArtifact};
    fn digest(path:&Path)->Result<String>{
        let mut f=File::open(path)?;let mut h=Sha256::new();let mut b=vec![0u8;1<<20];
        loop{let n=f.read(&mut b)?;if n==0{break;}h.update(&b[..n]);}Ok(format!("{:x}",h.finalize()))
    }
    fn durable<T:Serialize>(path:&Path,value:&T)->Result<serde_json::Value>{
        use std::io::Write;
        anyhow::ensure!(!path.exists(),"native replay output already exists");
        let partial=path.with_extension("json.partial");
        let file=std::fs::OpenOptions::new().write(true).create_new(true).open(&partial)?;
        let mut writer=std::io::BufWriter::new(file);serde_json::to_writer_pretty(&mut writer,value)?;writer.write_all(b"\n")?;writer.flush()?;writer.get_ref().sync_all()?;
        std::fs::rename(&partial,path)?;
        Ok(serde_json::json!({"path":path,"sha256":digest(path)?,"bytes":path.metadata()?.len()}))
    }
    let original:serde_json::Value=serde_json::from_slice(&std::fs::read(source)?)?;
    anyhow::ensure!(original["run_id"]==manifest.run_id && original["target"]==manifest.target,"recorded lattice run/target mismatch");
    let cfg:GhostPhaseLatticeConfig=serde_json::from_value(original["config"].clone())?;
    let loaded=load_verified_ghost_segments_from_dir(run,stem,cfg.step_bucket_size)?;
    anyhow::ensure!(loaded.equality_pass && loaded.phase_protocols_bound_by_stream && loaded.nodes.len() as u64==loaded.persisted_records,"native lattice source conservation failed");
    let root=out.join("native_floor_aware_lattice");std::fs::create_dir_all(&root)?;
    let context=cudarc::driver::CudaContext::new(0)?;let stream=context.default_stream();
    let mut backend=GhostPhaseLattice4D::new(context,stream,cfg);
    let outcome=backend.cluster(&loaded.nodes)?;
    let lineage=serde_json::json!({"schema_kind":"prism_ghost_record_byte_lineage","schema_version":1,"run_id":manifest.run_id,"target":manifest.target,"row_count":loaded.record_lineage.len(),"produced_equals_persisted_equals_decoded_equals_lattice_nodes":true,"records":loaded.record_lineage});
    let mut lineage_receipt=durable(&root.join("ghost_record_byte_lineage.json"),&lineage)?;
    lineage_receipt["row_count"]=loaded.record_lineage.len().into();lineage_receipt["schema_kind"]="prism_ghost_record_byte_lineage".into();lineage_receipt["schema_version"]=1.into();
    let edges=GhostLatticeAcceptedEdgeArtifact::from_outcome(&manifest.run_id,&manifest.target,lineage_receipt["sha256"].as_str().unwrap(),&loaded.nodes,&outcome)?;
    let edge_path=root.join("ghost_lattice_accepted_edges.json");
    let mut edge_receipt=durable(&edge_path,&edges)?;
    let checked=prism_nhs::ghost_phase_lattice::load_verified_ghost_lattice_edges(&edge_path,&loaded.nodes,edge_receipt["sha256"].as_str().unwrap(),edge_receipt["bytes"].as_u64().unwrap(),lineage_receipt["sha256"].as_str().unwrap())?;
    anyhow::ensure!(checked.edges==edges.edges,"persisted native edge readback differs");
    edge_receipt["row_count"]=edges.edges.len().into();edge_receipt["schema_kind"]="prism_ghost_lattice_accepted_edges".into();edge_receipt["schema_version"]=edges.schema_version.into();edge_receipt["node_lineage_artifact_sha256"]=lineage_receipt["sha256"].clone();
    edge_receipt["counted_equals_produced_equals_persisted_equals_decoded_equals_indexed"]=true.into();
    let blocks=prism_nhs::ghost_phase_materializer::materialize_components(&loaded.nodes,&outcome);
    let result=serde_json::json!({"schema_kind":"prismos_ghost_phase_lattice_components","schema_version":4,"backend":"ghost_phase_lattice_4d","run_id":manifest.run_id,"target":manifest.target,"topology_input":manifest.topology_input,"config":outcome.config,
        "replay_provenance":{"original_lattice_path":source,"original_lattice_sha256":digest(source)?,"original_graph_unchanged":true,"scanner_binary_sha256":digest(&std::env::current_exe()?)?,"spectral_plane_interpretation":"producer mask retained; absent producer mask remains unknown; exact epsilon-only spectra excluded from cosine evidence; no threshold changed","MD_rerun":false},
        "stats":{"n_nodes":outcome.stats.n_nodes,"n_lattice_cells":outcome.stats.n_lattice_cells,"n_pairs_evaluated":outcome.stats.n_pairs_evaluated,"n_pairs_phase_legal":outcome.stats.n_pairs_phase_legal,"n_pairs_aabb_overlap":outcome.stats.n_pairs_aabb_overlap,"n_directed_edges":outcome.stats.n_directed_edges,"kernel_elapsed_ms":outcome.stats.kernel_elapsed_ms,"host_elapsed_ms":outcome.stats.host_elapsed_ms,"min_step_idx":outcome.stats.min_step_idx,"max_step_idx":outcome.stats.max_step_idx},
        "ghost_segment_lineage":{"schema_kind":"prism_ghost_segment_lattice_lineage","schema_version":1,"produced_records":loaded.produced_records,"persisted_records":loaded.persisted_records,"decoded_records":loaded.decoded_records,"lattice_node_lineage_count":loaded.lattice_node_lineage_count,"phase_protocols_bound_by_stream":loaded.phase_protocols_bound_by_stream,"equality_pass":loaded.equality_pass,"streams":loaded.streams},
        "ghost_record_byte_lineage_artifact":lineage_receipt,"ghost_lattice_accepted_edge_artifact":edge_receipt,"edge_conservation":outcome.edge_conservation,
        "components":blocks.iter().enumerate().map(|(i,b)|serde_json::json!({"component_id":i,"n_nodes":b.n_nodes,"component_aabb_min":b.component_aabb_min,"component_aabb_max":b.component_aabb_max,"node_lineage":b.node_lineage,"driver_residues":b.driver_residues,"ghost_phase_lattice":b.ghost_phase_lattice,"phase_manifold":b.phase_manifold,"therm_ccns_lifecycle":b.therm_ccns_lifecycle,"so3_manifold":b.so3_manifold})).collect::<Vec<_>>()});
    durable(&root.join("binding_sites_lattice.json"),&result)?;
    File::open(&root)?.sync_all()?;
    eprintln!("native floor-aware lattice replay: {} nodes, {} components, {} exact accepted edges",loaded.nodes.len(),blocks.len(),edges.edges.len());Ok(())
}


struct ExactComponentJoinProduct {
    receipt: serde_json::Value,
    mapped_components: Vec<MappedGhostComponent>,
    adapter_receipt: Option<serde_json::Value>,
}

fn json_array3_f32(value: &serde_json::Value) -> Option<[f32; 3]> {
    let arr = value.as_array()?;
    if arr.len() != 3 {
        return None;
    }
    let mut out = [0.0_f32; 3];
    for (idx, slot) in out.iter_mut().enumerate() {
        let v = arr[idx].as_f64()? as f32;
        if !v.is_finite() {
            return None;
        }
        *slot = v;
    }
    Some(out)
}

fn component_aabb6(component: &serde_json::Value) -> Option<[f32; 6]> {
    let lo = json_array3_f32(&component["component_aabb_min"])?;
    let hi = json_array3_f32(&component["component_aabb_max"])?;
    if (0..3).any(|axis| lo[axis] > hi[axis]) {
        return None;
    }
    Some([lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]])
}

fn aabb_midpoint(aabb: [f32; 6]) -> [f32; 3] {
    [
        0.5 * (aabb[0] + aabb[3]),
        0.5 * (aabb[1] + aabb[4]),
        0.5 * (aabb[2] + aabb[5]),
    ]
}

fn driver_residue_support(component: &serde_json::Value) -> Vec<i32> {
    component["driver_residues"]
        .as_array()
        .map(|rows| {
            rows.iter()
                .filter_map(|value| value.as_i64().and_then(|v| i32::try_from(v).ok()))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn zstr_bytes_for_stream(zstr: &[ZstrStreamSummary], stream_id: u32) -> u64 {
    zstr.iter()
        .filter(|summary| summary.stream_id == Some(stream_id))
        .map(|summary| summary.file_size)
        .sum()
}

/// Complete event membership of one native graph, preserving graph identity.
/// This deliberately does not coerce a many-stream component into the legacy
/// singular-ID/scalar candidate contract.
fn emit_exact_component_event_join(graph_path:&Path, events:&[GhostTileEvent], out:&Path,
    manifest:&Manifest, namespace:&str, adapter_namespace:Option<&str>,
    zstr:&[ZstrStreamSummary])->Result<ExactComponentJoinProduct> {
    use std::io::Write;
    let graph_bytes=std::fs::read(graph_path)?;
    let graph_sha=format!("{:x}",Sha256::digest(&graph_bytes));
    let graph:serde_json::Value=serde_json::from_slice(&graph_bytes)?;
    anyhow::ensure!(graph["run_id"]==manifest.run_id && graph["target"]==manifest.target,"component graph run identity mismatch");
    let source=&graph["ghost_record_byte_lineage_artifact"];
    let lineage_path=PathBuf::from(source["path"].as_str().context("graph lacks byte-lineage path")?);
    let lineage_bytes=std::fs::read(&lineage_path)?;
    let lineage_sha=format!("{:x}",Sha256::digest(&lineage_bytes));
    anyhow::ensure!(source["sha256"]==lineage_sha,"component byte-lineage artifact SHA mismatch");
    let lineage:serde_json::Value=serde_json::from_slice(&lineage_bytes)?;
    anyhow::ensure!(lineage["run_id"]==manifest.run_id && lineage["target"]==manifest.target,"byte-lineage run identity mismatch");
    let rows:Vec<prism_nhs::ghost_phase_lattice::GhostRecordByteLineage>=serde_json::from_value(lineage["records"].clone())?;
    anyhow::ensure!(lineage["row_count"].as_u64()==Some(rows.len() as u64) && graph["stats"]["n_nodes"].as_u64()==Some(rows.len() as u64),"component source node census mismatch");
    let mut event_index=std::collections::HashMap::new();
    for event in events {anyhow::ensure!(event_index.insert((event.stream_id,event.stream_record_ordinal),event).is_none(),"ambiguous stream record identity");}
    let components=graph["components"].as_array().context("native graph lacks components")?;
    let partial=out.join(format!("{namespace}_component_events.jsonl.partial"));
    let target=out.join(format!("{namespace}_component_events.jsonl"));
    anyhow::ensure!(!target.exists(),"exact component join target already exists");
    let mut writer=std::io::BufWriter::new(std::fs::OpenOptions::new().write(true).create_new(true).open(&partial)?);
    let mut visited=vec![false;rows.len()];let mut total=0usize;let mut summaries=Vec::new();
    let adapter_enabled = adapter_namespace == Some(namespace);
    let mut mapped_components = Vec::new();
    let mut unmappable_multi_stream_or_site = 0u64;
    let mut unmappable_missing_aabb = 0u64;
    let mut unmappable_zero_events = 0u64;
    for component in components {
        let component_id=component["component_id"].as_u64().context("missing native component ID")?;
        let nodes=component["node_lineage"].as_array().context("native component lacks exact node membership")?;
        anyhow::ensure!(component["n_nodes"].as_u64()==Some(nodes.len() as u64),"component node population mismatch");
        let mut partitions=std::collections::BTreeMap::<(u32,u32),u64>::new();
        let mut plane_counts=[0u64;4];let mut floor_counts=[0u64;4];let mut unknown=0u64;
        let mut score_sum=0.0_f64;let mut score_max=0.0_f64;
        let mut first_event_identity:Option<serde_json::Value>=None;let mut last_event_identity:Option<serde_json::Value>=None;
        let mut component_step_min:Option<u64>=None;let mut component_step_max:Option<u64>=None;
        let mut exact_component_adapter_status:&'static str="not_selected";
        for node in nodes {
            let index=node["tile_index"].as_u64().context("node lacks global lattice index")? as usize;
            let row=rows.get(index).context("component node index beyond source lineage")?;
            anyhow::ensure!(!visited[index] && row.global_lattice_index==index as u64,"duplicate/misaligned graph node membership");
            visited[index]=true;
            anyhow::ensure!(node["stream_id"].as_u64()==Some(u64::from(row.stream_id)) && node["site_id"].as_u64()==Some(u64::from(row.site_id)) && node["step_idx"].as_u64()==Some(row.step_idx) && node["frame_idx"].as_u64()==Some(row.frame_idx) && node["protocol_phase"].as_u64()==Some(u64::from(row.protocol_phase)),"component node identity differs from authenticated byte lineage");
            let event=event_index.get(&(u32::from(row.stream_id),row.stream_record_ordinal)).context("component source record absent from decoded population")?;
            anyhow::ensure!(event.source_record_sha256==row.record_sha256 && event.source_segment_sha256.as_deref()==Some(row.segment_file_sha256.as_str()) && event.provenance.byte_offset==row.payload_byte_offset && event.step_idx==Some(row.step_idx) && event.site_id==row.site_id && event.frame_idx==row.frame_idx,"component/event exact source binding differs");
            *partitions.entry((event.stream_id,event.site_id)).or_default()+=1;
            component_step_min=Some(component_step_min.map_or(row.step_idx,|v|v.min(row.step_idx)));
            component_step_max=Some(component_step_max.map_or(row.step_idx,|v|v.max(row.step_idx)));
            let (tile_score, _)=score_ghost_tile_event(event);
            score_sum+=tile_score;score_max=score_max.max(tile_score);
            let event_identity=serde_json::json!({"stream_id":event.stream_id,"stream_record_ordinal":event.stream_record_ordinal,
                "site_id":event.site_id,"step_idx":event.step_idx,"source_record_sha256":event.source_record_sha256,
                "source_segment_sha256":event.source_segment_sha256,"source_byte_offset":event.provenance.byte_offset});
            if first_event_identity.is_none(){first_event_identity=Some(event_identity.clone());}
            last_event_identity=Some(event_identity);
            for p in 0..4 {if event.plane_evidence.cosine_evidence_mask&(1<<p)!=0{plane_counts[p]+=1;}if event.plane_evidence.exact_floor_only_mask&(1<<p)!=0{floor_counts[p]+=1;}}
            if event.plane_evidence.producer_populated_mask.is_none(){unknown+=1;}
            let record=serde_json::json!({"schema_kind":"prism_exact_native_component_event_join","schema_version":1,"graph_sha256":graph_sha,"component_id":component_id,"global_lattice_index":index,"byte_lineage_artifact_sha256":lineage_sha,"stream_id":event.stream_id,"stream_record_ordinal":event.stream_record_ordinal,"site_id":event.site_id,"step_idx":event.step_idx,"raw_frame_idx":event.frame_idx,"native_protocol_phase":row.protocol_phase,"event_id":event.event_id,"source_record_sha256":event.source_record_sha256,"source_segment_sha256":event.source_segment_sha256,"source_byte_offset":event.provenance.byte_offset,"plane_evidence":event.plane_evidence});
            serde_json::to_writer(&mut writer,&record)?;writer.write_all(b"\n")?;total+=1;
        }
        if adapter_enabled {
            if nodes.is_empty() {
                exact_component_adapter_status="unmappable_zero_events";
                unmappable_zero_events+=1;
            } else if partitions.len()!=1 {
                exact_component_adapter_status="unmappable_multi_stream_or_site";
                unmappable_multi_stream_or_site+=1;
            } else if let Some(aabb)=component_aabb6(component) {
                exact_component_adapter_status="mapped_single_stream_site_aabb";
                let ((stream_id,site_id),event_count)=partitions.iter().next().map(|(k,v)|(*k,*v)).unwrap();
                let score_mean=score_sum/(nodes.len() as f64);
                mapped_components.push(MappedGhostComponent {
                    component_id: format!("{namespace}:{component_id}"),
                    stream_id,
                    site_id,
                    centroid_xyz: aabb_midpoint(aabb),
                    aabb: Some(aabb),
                    residue_support: driver_residue_support(component),
                    coordinate_frame: "native_ghost_lattice_component_aabb_angstrom".to_string(),
                    mapping_method: "exact_native_component_join.component_aabb_midpoint",
                    mapping_source: format!("{}#component_id={component_id};graph_sha256={graph_sha};byte_lineage_sha256={lineage_sha}", graph_path.display()),
                    ghost_event_count: event_count,
                    ghost_tile_score_mean: score_mean,
                    ghost_tile_score_max: score_max,
                    zstr_bytes: zstr_bytes_for_stream(zstr, stream_id),
                    native_record_provenance: None,
                    mapping_scope: "exact_native_component_join_single_stream_site_aabb",
                    exact_component_adapter: true,
                    component_step_min,
                    component_step_max,
                    exact_component_provenance: Some(serde_json::json!({
                        "schema_kind":"prism_exact_component_mapped_adapter_provenance",
                        "schema_version":1,
                        "namespace":namespace,
                        "source_component_id":component_id,
                        "source_graph_path":graph_path,
                        "source_graph_sha256":graph_sha,
                        "byte_lineage_artifact_path":lineage_path,
                        "byte_lineage_artifact_sha256":lineage_sha,
                        "component_event_join_path":target,
                        "component_observations_path":out.join(format!("{namespace}_component_observations.json")),
                        "exact_event_membership":{
                            "event_count":event_count,
                            "stream_id":stream_id,
                            "site_id":site_id,
                            "component_step_min":component_step_min,
                            "component_step_max":component_step_max,
                            "first_event":first_event_identity,
                            "last_event":last_event_identity
                        },
                        "region_association_semantics":"downstream materializer applies existing region-centroid/component-AABB admission; no rank matching; no causal claim",
                        "score_semantics":"ghost_tile_score_mean/max from existing score_ghost_tile_event over exact component event membership; no new weights"
                    })),
                });
            } else {
                exact_component_adapter_status="unmappable_missing_component_aabb";
                unmappable_missing_aabb+=1;
            }
        }
        summaries.push(serde_json::json!({"graph_sha256":graph_sha,"component_id":component_id,"native_component_record":component,"exact_joined_event_count":nodes.len(),"stream_site_partitions":partitions.into_iter().map(|((stream_id,site_id),count)|serde_json::json!({"stream_id":stream_id,"site_id":site_id,"event_count":count})).collect::<Vec<_>>(),"cosine_eligible_event_counts_by_plane":plane_counts,"exact_padding_only_event_counts_by_plane":floor_counts,"producer_occupancy_unknown_event_count":unknown,"join_semantics":"exact native node index to source byte lineage to stream-record ordinal and matching SHA/offset/step/site; no Cartesian range or centroid association","legacy_scalar_region_binding_status":"legacy_scalar_adapter_unavailable_without_exact_region_event_membership","exact_component_adapter_status":exact_component_adapter_status}));
    }
    anyhow::ensure!(total==rows.len() && visited.iter().all(|seen|*seen) && total==events.len(),"full native component/event population conservation failed");
    writer.flush()?;writer.get_ref().sync_all()?;std::fs::rename(&partial,&target)?;
    let support_path=out.join(format!("{namespace}_component_observations.json"));
    write_json(&support_path,&serde_json::json!({"schema_kind":"prism_exact_native_component_observations","schema_version":1,"graph_path":graph_path,"graph_sha256":graph_sha,"byte_lineage_sha256":lineage_sha,"component_count":components.len(),"event_count":total,"components":summaries}))?;
    let adapter_receipt=if adapter_enabled {Some(serde_json::json!({"status":"IMPLEMENTATION_VERIFIED",
        "adapter_kind":"exact_native_component_join_to_mapped_component_v1",
        "selected_namespace":namespace,
        "mapped_component_count":mapped_components.len(),
        "unmappable_multi_stream_or_site_count":unmappable_multi_stream_or_site,
        "unmappable_missing_aabb_count":unmappable_missing_aabb,
        "unmappable_zero_event_count":unmappable_zero_events,
        "source_graph_path":graph_path,"source_graph_sha256":graph_sha,
        "byte_lineage_artifact_sha256":lineage_sha,
        "event_join_path":target,"component_observations_path":support_path,
        "factor_policy":"downstream existing ghost_zstr_summary_for_geometry; no new weights or thresholds",
        "membership_semantics":"exact component event membership; region association remains downstream geometry admission, not rank match"}))}else{None};
    let receipt=serde_json::json!({"status":"IMPLEMENTATION_VERIFIED","namespace":namespace,"source_graph_path":graph_path,"source_graph_sha256":graph_sha,"byte_lineage_artifact_sha256":lineage_sha,"component_count":components.len(),"event_count":total,"source_node_count":rows.len(),"all_nodes_exactly_once":true,"all_events_exactly_once":true,"event_join_path":target,"event_join_sha256":sha256_path(&target)?,"component_observations_path":support_path,"component_observations_sha256":sha256_path(&support_path)?,"graph_identity_preserved":true,"legacy_region_adapter_status":"legacy_scalar_adapter_unavailable_without_exact_region_event_membership","effective_region_adapter_status":if adapter_receipt.is_some(){"exact_component_adapter_selected"}else{"not_selected"},"exact_component_adapter_receipt":adapter_receipt.clone()});
    write_json(&out.join(format!("{namespace}_component_join_receipt.json")),&receipt)?;
    Ok(ExactComponentJoinProduct{receipt,mapped_components,adapter_receipt})
}
