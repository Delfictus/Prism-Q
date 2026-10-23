//! PRISM ZSTR offline replay — the decoder `zstr.rs` documents but that has
//! never existed in this tree.
//!
//! ZSTR is the engine's highest-fidelity coordinate stream: on the monolithic
//! captured-graph path it stages full-system positions **and** post-ASC forces
//! at the recorded graph observation cadence, plus potential energy, external work, |F|2 and
//! the adjudication code. Until now nothing could read the payload back — the
//! only parser in the tree (`prism_ghost_evidence_scan`) decodes the first
//! 4096-byte header and stops, and says so: `first_header_only`.
//!
//! This binary closes that gap. It decodes a `prism_zstr_<pid>_<stream>.bin`
//! into formats the tree already has readers for:
//!   * `PRISM4D\0` `.frames.bin` (positions) — `crates/prism-nhs/src/gpu.rs`
//!     documents the layout; a Python loader exists in the appliance scripts.
//!   * multi-MODEL PDB (positions) — `trajectory.rs::load_ensemble_pdb` and the
//!     `nhs_analyze*` binaries read it. Requires a PRISM-PREP topology for atom
//!     and residue naming.
//!
//! ## Refusal discipline
//!
//! This decoder does not invent a time axis.
//!
//! * `frame_idx` is read from EVERY record. Record *k* is not step *k*: the
//!   consumer drains the lowest ready slot, and a device-side backpressure
//!   refusal drops frames without shifting byte offsets. Gaps are reported,
//!   never interpolated.
//! * The header `dt` is a GRAPH-BUILD-TIME CONSTANT. Under `--adaptive-dt` or
//!   the G26 gearbox the real timestep varies (0.5 / 2.0 / 4.0 fs), so
//!   `step x dt` is not elapsed physics. This tool therefore emits step indices
//!   and refuses to emit a physical-time column. Join
//!   `*_governor_decisions.jsonl` (`cumulative_simulated_time_ps`) if you need
//!   one.
//! * `gear_id == 0xFFFFFFFF` is the declared UNMEASURED sentinel, not a gear.
//! * `schema_version == 0` marks a pre-WP4 file whose header was never stamped.
//!   Such files carry `n_atoms == 0`, so `--n-atoms` is REQUIRED, and because
//!   `frame_idx` is also unstamped their chronological order is file-offset
//!   order and is NOT provable from the file. Those runs are marked
//!   `ORDER_UNPROVEN` in the manifest rather than silently presented as a
//!   time series.

use anyhow::{bail, Context, Result};
use clap::Parser;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::PathBuf;

const HEADER_BYTES: usize = 4096;
const ZSTR_SCHEMA_WP4: u32 = 1;
const GEAR_ID_UNMEASURED: u32 = 0xFFFF_FFFF;

// Header field offsets. Authoritative source: the `offset_of!` assertions in
// crates/prism-nhs/src/zstr.rs, mirrored in src/cuda/zstr_kernels.cuh.
const OFF_FRAME_IDX: usize = 0;
const OFF_DT: usize = 8;
const OFF_ADJ_CODE: usize = 12;
const OFF_N_ATOMS: usize = 20;
const OFF_GEAR_ID: usize = 24;
const OFF_FORCE_NORM: usize = 28;
const OFF_EXTERNAL_WORK: usize = 32;
const OFF_POTENTIAL_ENERGY: usize = 40;
const OFF_STREAM_ID: usize = 48;
const OFF_SCHEMA_VERSION: usize = 52;

const FRAMES_BIN_MAGIC: [u8; 8] = *b"PRISM4D\0";
const FRAMES_BIN_VERSION: u32 = 1;

#[derive(Parser, Debug)]
#[command(
    name = "prism-zstr-replay",
    about = "Decode a ZSTR coordinate/force stream into readable trajectory formats"
)]
struct Args {
    /// ZSTR binary to decode (prism_zstr_<pid>_<stream>.bin).
    #[arg(short, long)]
    input: PathBuf,

    /// Output directory. Created if absent.
    #[arg(short, long)]
    output: PathBuf,

    /// Atom count override. REQUIRED for pre-WP4 files whose header was never
    /// stamped (schema_version == 0, n_atoms == 0). Refused when the file
    /// stamps a conflicting value, rather than silently preferring one.
    #[arg(long)]
    n_atoms: Option<u32>,

    /// PRISM-PREP topology JSON. Required to emit PDB (atom/residue naming).
    #[arg(long)]
    topology: Option<PathBuf>,

    /// Emit `<stem>.frames.bin` in PRISM4D format. Default true.
    #[arg(long, default_value = "true")]
    emit_frames_bin: bool,

    /// Emit `<stem>.pdb` as multi-MODEL. Requires --topology.
    #[arg(long, default_value = "false")]
    emit_pdb: bool,

    /// Source-established expected observation cadence; never substitute observed GCD.
    #[arg(long, requires = "cadence_source")]
    expected_step_cadence: Option<u64>,
    /// Exact source/receipt reference establishing the explicit cadence.
    #[arg(long, requires = "expected_step_cadence")]
    cadence_source: Option<String>,
    /// Independently verified capture-order receipt bound to this ZSTR source SHA.
    #[arg(long)]
    work_capture_receipt: Option<PathBuf>,
    /// Source-bound fixed-clock receipt using the native exporter guard schema.
    #[arg(long)]
    physical_clock_receipt: Option<PathBuf>,

    /// Decode at most N frames (0 = all). Useful for inspection.
    #[arg(long, default_value = "0")]
    max_frames: usize,
}

fn step_gcd(mut a:u64,mut b:u64)->u64 {
    while b!=0 {let r=a%b;a=b;b=r;} a
}

fn align16(x: usize) -> usize {
    (x + 15) & !15
}
fn align4096(x: usize) -> usize {
    (x + 4095) & !4095
}

/// Slot stride, reproducing `ZstrRing::allocate` exactly.
fn frame_size_for(n_atoms: usize) -> Option<usize> {
    let vector_bytes = n_atoms.checked_mul(3)?.checked_mul(4)?;
    let pos_aligned = vector_bytes.checked_add(15)? & !15;
    HEADER_BYTES.checked_add(pos_aligned)?.checked_add(vector_bytes)?.checked_add(4095).map(|x| x & !4095)
}

fn rd_u32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}
fn rd_u64(b: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(b[off..off + 8].try_into().unwrap())
}
fn rd_f32(b: &[u8], off: usize) -> f32 {
    f32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}
fn rd_f64(b: &[u8], off: usize) -> f64 {
    f64::from_le_bytes(b[off..off + 8].try_into().unwrap())
}

struct FrameHeader {
    frame_idx: u64,
    dt_ps: f32,
    adj_code: u32,
    n_atoms: u32,
    gear_id: u32,
    force_norm: f32,
    external_work: f64,
    potential_energy: f64,
    stream_id: u32,
    schema_version: u32,
}

fn parse_header(b: &[u8]) -> FrameHeader {
    FrameHeader {
        frame_idx: rd_u64(b, OFF_FRAME_IDX),
        dt_ps: rd_f32(b, OFF_DT),
        adj_code: rd_u32(b, OFF_ADJ_CODE),
        n_atoms: rd_u32(b, OFF_N_ATOMS),
        gear_id: rd_u32(b, OFF_GEAR_ID),
        force_norm: rd_f32(b, OFF_FORCE_NORM),
        external_work: rd_f64(b, OFF_EXTERNAL_WORK),
        potential_energy: rd_f64(b, OFF_POTENTIAL_ENERGY),
        stream_id: rd_u32(b, OFF_STREAM_ID),
        schema_version: rd_u32(b, OFF_SCHEMA_VERSION),
    }
}

/// Validate each record before treating its positions as a trajectory row.
/// Legacy zero headers remain explicit; unsupported or mixed schemas do not
/// inherit the first record's authority.
fn validate_record_header(h: &FrameHeader, first: &FrameHeader, n_atoms: u32) -> std::result::Result<(), String> {
    if h.schema_version > ZSTR_SCHEMA_WP4 { return Err(format!("unsupported schema {}", h.schema_version)); }
    if h.schema_version != first.schema_version { return Err("mixed schema versions".into()); }
    if h.n_atoms != first.n_atoms { return Err("mixed atom-count headers".into()); }
    if h.stream_id != first.stream_id { return Err("mixed source-stream identities".into()); }
    if h.schema_version == ZSTR_SCHEMA_WP4 {
        if h.n_atoms == 0 || h.n_atoms != n_atoms { return Err("stamped atom population mismatch".into()); }
        if !h.dt_ps.is_finite() || h.dt_ps <= 0.0 { return Err("invalid nominal header dt".into()); }
        if h.adj_code > 2 { return Err("invalid adjudication code".into()); }
        if h.gear_id > 3 && h.gear_id != GEAR_ID_UNMEASURED { return Err("invalid gear code".into()); }
    } else if h.n_atoms != 0 {
        return Err("legacy unstamped schema carries a contradictory atom count".into());
    }
    Ok(())
}

fn validate_positions(bytes: &[u8]) -> std::result::Result<(), String> {
    if bytes.len() % 12 != 0 { return Err("incomplete float3 position payload".into()); }
    for (i, part) in bytes.chunks_exact(4).enumerate() {
        if !f32::from_le_bytes(part.try_into().unwrap()).is_finite() {
            return Err(format!("nonfinite coordinate at component {}", i));
        }
    }
    Ok(())
}

/// Minimal topology view: only what PDB emission needs.
struct TopologyView {
    atom_names: Vec<String>,
    residue_names: Vec<String>,
    residue_ids: Vec<i64>,
    chain_ids: Vec<String>,
    elements: Vec<String>,
}

fn load_topology(p: &PathBuf) -> Result<TopologyView> {
    let raw = std::fs::read_to_string(p).with_context(|| format!("read topology {}", p.display()))?;
    let v: serde_json::Value = serde_json::from_str(&raw).context("parse topology JSON")?;
    let arr = |k: &str| -> Vec<String> {
        v.get(k)
            .and_then(|x| x.as_array())
            .map(|a| {
                a.iter()
                    .map(|e| e.as_str().unwrap_or("").to_string())
                    .collect()
            })
            .unwrap_or_default()
    };
    let ids = v
        .get("residue_ids")
        .and_then(|x| x.as_array())
        .map(|a| a.iter().map(|e| e.as_i64().unwrap_or(0)).collect())
        .unwrap_or_default();
    Ok(TopologyView {
        atom_names: arr("atom_names"),
        residue_names: arr("residue_names"),
        residue_ids: ids,
        chain_ids: arr("chain_ids"),
        elements: arr("elements"),
    })
}

fn main() -> Result<()> {
    let args = Args::parse();
    std::fs::create_dir_all(&args.output)
        .with_context(|| format!("create output dir {}", args.output.display()))?;

    let file_len = std::fs::metadata(&args.input)
        .with_context(|| format!("stat {}", args.input.display()))?
        .len();
    let file_len = usize::try_from(file_len).context("ZSTR size exceeds host address range")?;
    if file_len < HEADER_BYTES {
        bail!(
            "{} is {} bytes: shorter than one ZSTR header ({})",
            args.input.display(),
            file_len,
            HEADER_BYTES
        );
    }

    let mut f = File::open(&args.input)
        .with_context(|| format!("open {}", args.input.display()))?;
    let mut hdr0 = vec![0u8; HEADER_BYTES];
    f.read_exact(&mut hdr0).context("read first header")?;
    let h0 = parse_header(&hdr0);

    // ── stride resolution, fail-closed ────────────────────────────────────
    let stamped = h0.n_atoms;
    let n_atoms = match (stamped, args.n_atoms) {
        (0, None) => bail!(
            "header stamps n_atoms = 0 (schema_version = {}), so the slot stride cannot be \
             derived from the file. This is a pre-WP4 unstamped stream. Supply --n-atoms \
             from the run's topology or a co-located .frames.bin header.",
            h0.schema_version
        ),
        (0, Some(n)) => n,
        (s, None) => s,
        (s, Some(n)) if s == n => s,
        (s, Some(n)) => bail!(
            "--n-atoms {} conflicts with the value stamped in the file ({}). Refusing to guess \
             which is authoritative.",
            n,
            s
        ),
    };
    if n_atoms == 0 {
        bail!("resolved n_atoms = 0; nothing to decode");
    }

    let n_atoms_us = n_atoms as usize;
    let stride = frame_size_for(n_atoms_us).context("ZSTR frame stride overflow")?;
    let pos_off = HEADER_BYTES;
    let force_off = HEADER_BYTES + align16(n_atoms_us * 3 * 4);
    let n_records = file_len / stride;
    let trailing = file_len % stride;

    let order_provable = h0.schema_version == ZSTR_SCHEMA_WP4;
    let stem = args
        .input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("zstr")
        .to_string();

    eprintln!("[zstr-replay] {}", args.input.display());
    eprintln!(
        "  n_atoms={} (stamped={}) stride={} records={} trailing={}",
        n_atoms, stamped, stride, n_records, trailing
    );
    eprintln!(
        "  schema_version={} {}",
        h0.schema_version,
        if order_provable {
            "(WP4 stamped: frame_idx authoritative)"
        } else {
            "(PRE-WP4 UNSTAMPED: chronological order is file-offset order and is NOT provable)"
        }
    );
    if trailing != 0 { bail!("truncated ZSTR payload: {} trailing bytes after {} complete records", trailing, n_records); }
    if n_records == 0 { bail!("ZSTR contains no complete record for the declared atom count"); }
    validate_record_header(&h0, &h0, n_atoms).map_err(anyhow::Error::msg)?;
    if h0.schema_version == 0 && args.n_atoms.is_none() {
        bail!("legacy ZSTR requires an explicit topology-derived atom-count override");
    }
    let mut slot = Vec::new();
    slot.try_reserve_exact(stride).context("allocate ZSTR record buffer")?;
    slot.resize(stride, 0);
    // Preflight every source record, including records outside --max-frames.
    // No output is opened until the complete source population is admitted.
    let mut source_digest = Sha256::new();
    let mut observed_step_deltas=std::collections::BTreeMap::<u64,u64>::new();
    let mut preflight_previous_step=None;
    let mut observed_cadence=0u64;
    f.seek(SeekFrom::Start(0))?;
    for rec in 0..n_records {
        f.read_exact(&mut slot).with_context(|| format!("preflight ZSTR record {}", rec))?;
        source_digest.update(&slot);
        let header=parse_header(&slot);
        if order_provable {
            if let Some(previous)=preflight_previous_step {
                if header.frame_idx>previous {
                    let delta=header.frame_idx-previous;
                    *observed_step_deltas.entry(delta).or_default()+=1;
                    observed_cadence=step_gcd(observed_cadence,delta);
                }
            }
            preflight_previous_step=Some(header.frame_idx);
        }
        validate_record_header(&header, &h0, n_atoms)
            .map_err(anyhow::Error::msg).with_context(|| format!("record {} header", rec))?;
        validate_positions(&slot[pos_off..pos_off+n_atoms_us*12])
            .map_err(anyhow::Error::msg).with_context(|| format!("record {} positions", rec))?;
    }
    let source_sha256 = format!("{:x}", source_digest.finalize());
    if args.expected_step_cadence == Some(0) { bail!("expected step cadence must be positive"); }
    let work_capture_binding = if let Some(path) = &args.work_capture_receipt {
        let bytes=std::fs::read(path)?;
        let receipt:serde_json::Value=serde_json::from_slice(&bytes)?;
        if receipt["source_zstr_sha256"].as_str()!=Some(source_sha256.as_str()) {
            bail!("work capture receipt does not bind this ZSTR source");
        }
        Some(serde_json::json!({"path":path,"sha256":format!("{:x}",Sha256::digest(&bytes)),"receipt":receipt,"authority":"explicit source-bound supplied receipt; not inferred from zero work"}))
    } else {None};
    let (verified_dt_ps, physical_clock_binding) = if let Some(path)=&args.physical_clock_receipt {
        let bytes=std::fs::read(path)?;let receipt:serde_json::Value=serde_json::from_slice(&bytes)?;
        if receipt["source_zstr_sha256"].as_str()!=Some(source_sha256.as_str()) {bail!("physical clock receipt does not bind this ZSTR source");}
        let clock=&receipt["physical_clock"];
        if clock["physical_time_available"]!=true || clock["runtime_fixed_step_guard_witness"]!=true || clock["manifest_adaptive_dt"]!=false || clock["origin"]!="MD step zero" || clock["mapping"]!="signed_md_step_delta_times_verified_fixed_dt" {bail!("physical clock receipt lacks native fixed-step guards");}
        let dt=clock["dt_ps"].as_f64().filter(|v|v.is_finite()&&*v>0.0).context("invalid physical clock dt")?;
        if dt!=h0.dt_ps as f64 {bail!("physical clock dt differs from nominal ZSTR source header");}
        let log=&clock["runtime_log"];let log_path=log["path"].as_str().context("missing clock runtime-log path")?;
        let log_bytes=std::fs::read(log_path)?;
        if log["sha256"].as_str()!=Some(format!("{:x}",Sha256::digest(&log_bytes)).as_str()) {bail!("clock runtime-log SHA mismatch");}
        let log_text=std::str::from_utf8(&log_bytes)?;
        if !log_text.lines().any(|line|line.contains(&format!("[PHYSGOV stream {}] DISARMED (--gearbox off)",h0.stream_id))) || !log_text.lines().any(|line|line.contains(&format!("PHYSICAL_GUARD_CHRONOMETRY source={} ",h0.stream_id))&&line.contains("director_adaptive_dt=0 host_legacy_dt_authorized=false")) {bail!("runtime log lacks source-stream fixed-clock witness");}
        (Some(dt),Some(serde_json::json!({"path":path,"sha256":format!("{:x}",Sha256::digest(&bytes)),"receipt":receipt})))
    } else {(None,None)};
    let topology_sha256 = args.topology.as_ref().map(|path| -> Result<String> {
        Ok(format!("{:x}", Sha256::digest(std::fs::read(path)?)))
    }).transpose()?;

    let want = if args.max_frames == 0 {
        n_records
    } else {
        n_records.min(args.max_frames)
    };

    let topo = match (&args.topology, args.emit_pdb) {
        (Some(p), _) => Some(load_topology(p)?),
        (None, true) => bail!("--emit-pdb requires --topology for atom and residue naming"),
        (None, false) => None,
    };
    if let Some(t) = &topo {
        if t.atom_names.len() != n_atoms_us {
            bail!(
                "topology has {} atoms but the ZSTR stream carries {}",
                t.atom_names.len(),
                n_atoms_us
            );
        }
    }

    let mut frames_bin = if args.emit_frames_bin {
        let p = args.output.join(format!("{stem}.frames.bin"));
        let mut w = BufWriter::new(File::create(&p).with_context(|| format!("create {}", p.display()))?);
        w.write_all(&FRAMES_BIN_MAGIC)?;
        w.write_all(&FRAMES_BIN_VERSION.to_le_bytes())?;
        w.write_all(&n_atoms.to_le_bytes())?;
        // Observed positive step-delta GCD, not an invented one-step cadence.
        // Zero explicitly means unavailable; exact per-record steps remain authoritative.
        w.write_all(&u32::try_from(observed_cadence).unwrap_or(0).to_le_bytes())?;
        w.write_all(&h0.dt_ps.to_le_bytes())?;
        Some((w, p))
    } else {
        None
    };

    let mut pdb = if args.emit_pdb {
        let p = args.output.join(format!("{stem}.pdb"));
        Some((
            BufWriter::new(File::create(&p).with_context(|| format!("create {}", p.display()))?),
            p,
        ))
    } else {
        None
    };

    let mut replay_digest = Sha256::new();
    let mut per_frame: Vec<serde_json::Value> = Vec::with_capacity(want);
    let mut prev_idx: Option<u64> = None;
    let mut gaps: u64 = 0;
    let mut missing_expected_observations = 0u64;
    let mut cadence_mismatches = 0u64;
    let mut nonmono = 0u64;
    let mut nonfinite_force_norm = 0u64;
    let mut decoded = 0usize;

    for rec in 0..n_records {
        f.seek(SeekFrom::Start((rec * stride) as u64))?;
        f.read_exact(&mut slot)
            .with_context(|| format!("read slot {rec}"))?;
        replay_digest.update(&slot);
        let h = parse_header(&slot);
        validate_record_header(&h, &h0, n_atoms).map_err(anyhow::Error::msg)?;
        validate_positions(&slot[pos_off..pos_off+n_atoms_us*12]).map_err(anyhow::Error::msg)?;
        if rec >= want { continue; }

        if order_provable {
            if let Some(p) = prev_idx {
                if h.frame_idx <= p {
                    nonmono += 1;
                } else {
                    let delta=h.frame_idx-p;
                    gaps += delta - 1;
                    if let Some(cadence)=args.expected_step_cadence {
                        if delta%cadence!=0 {cadence_mismatches+=1;}
                        else {missing_expected_observations+=delta/cadence-1;}
                    }
                }
            }
            prev_idx = Some(h.frame_idx);
        }
        if !h.force_norm.is_finite() {
            nonfinite_force_norm += 1;
        }

        let pos_bytes = &slot[pos_off..pos_off + n_atoms_us * 3 * 4];
        let mut pos = vec![0f32; n_atoms_us * 3];
        for (i, c) in pos.iter_mut().enumerate() {
            *c = rd_f32(pos_bytes, i * 4);
        }

        if let Some((w, _)) = frames_bin.as_mut() {
            for c in &pos {
                w.write_all(&c.to_le_bytes())?;
            }
        }
        if let Some((w, _)) = pdb.as_mut() {
            let t = topo.as_ref().unwrap();
            writeln!(w, "MODEL     {:>4}", decoded + 1)?;
            writeln!(w, "REMARK   ZSTR_FRAME_IDX {}", h.frame_idx)?;
            writeln!(w, "REMARK   POTENTIAL_ENERGY_KCAL_MOL {:.6}", h.potential_energy)?;
            writeln!(w, "REMARK   FORCE_NORM {:.6}", h.force_norm)?;
            writeln!(w, "REMARK   ADJUDICATION_CODE {}", h.adj_code)?;
            for i in 0..n_atoms_us {
                let name = t.atom_names.get(i).map(String::as_str).unwrap_or("X");
                let res = t.residue_names.get(i).map(String::as_str).unwrap_or("UNK");
                // HID/HIE/HIP are AMBER protonation variants; PDB consumers and
                // DSSP expect HIS. Mapped on output only; the topology is untouched.
                let res = match res {
                    "HID" | "HIE" | "HIP" => "HIS",
                    other => other,
                };
                let rid = t.residue_ids.get(i).copied().unwrap_or(0);
                let ch = t
                    .chain_ids
                    .get(i)
                    .and_then(|c| c.chars().next())
                    .unwrap_or('A');
                let el = t.elements.get(i).map(String::as_str).unwrap_or("");
                writeln!(
                    w,
                    "ATOM  {:>5} {:<4}{:>4} {}{:>4}    {:>8.3}{:>8.3}{:>8.3}  1.00  0.00          {:>2}",
                    (i + 1) % 100000,
                    name,
                    res,
                    ch,
                    rid,
                    pos[i * 3],
                    pos[i * 3 + 1],
                    pos[i * 3 + 2],
                    el
                )?;
            }
            writeln!(w, "ENDMDL")?;
        }

        per_frame.push(serde_json::json!({
            "record_index": rec,
            "source_byte_offset": rec * stride,
            "source_byte_length": stride,
            "source_record_sha256": format!("{:x}", Sha256::digest(&slot)),
            "frame_idx": h.frame_idx,
            "physical_time_ps":if order_provable {verified_dt_ps.map(|dt|h.frame_idx as f64*dt)}else{None},
            "stream_id": h.stream_id,
            "adjudication_code": h.adj_code,
            "potential_energy_kcal_mol": h.potential_energy,
            "external_work_kcal_mol": h.external_work,
            "force_norm": if h.force_norm.is_finite() { serde_json::json!(h.force_norm) } else { serde_json::Value::Null },
            "gear_id": if h.gear_id == GEAR_ID_UNMEASURED { serde_json::Value::Null } else { serde_json::json!(h.gear_id) },
            "force_offset_bytes": force_off,
        }));
        decoded += 1;
    }

    let replay_sha256 = format!("{:x}", replay_digest.finalize());
    if replay_sha256 != source_sha256 || f.metadata()?.len() != file_len as u64 {
        bail!("ZSTR source changed during replay; emitted partial outputs are unsealed");
    }
    if let Some((mut w, p)) = frames_bin {
        w.flush()?;
        eprintln!("  wrote {}", p.display());
    }
    if let Some((mut w, p)) = pdb {
        w.flush()?;
        eprintln!("  wrote {}", p.display());
    }

    let manifest = serde_json::json!({
        "schema_kind": "prismos_zstr_replay_manifest",
        "schema_version": 1,
        "source": args.input.display().to_string(),
        "source_bytes": file_len,
        "source_sha256": source_sha256,
        "topology_sha256": topology_sha256,
        "record_validation": { "status": "ALL_SOURCE_RECORDS_VALIDATED", "records_checked": n_records, "coordinate_finite": true },
        "geometry": {
            "n_atoms": n_atoms,
            "n_atoms_source": if stamped != 0 { "file_header" } else { "operator_override" },
            "frame_size_bytes": stride,
            "positions_offset": pos_off,
            "forces_offset": force_off,
        },
        "records": { "present": n_records, "decoded": decoded, "trailing_bytes": trailing },
        "chronology": {
            "schema_version": h0.schema_version,
            "order_status": if order_provable { "FRAME_IDX_AUTHORITATIVE" } else { "ORDER_UNPROVEN_PRE_WP4_UNSTAMPED" },
            "frame_idx_gaps": if order_provable { serde_json::json!(gaps) } else { serde_json::Value::Null },
            "observed_positive_step_delta_counts":observed_step_deltas,
            "expected_step_cadence":args.expected_step_cadence,
            "cadence_source":args.cadence_source,
            "missing_expected_observations":if order_provable && args.expected_step_cadence.is_some(){Some(missing_expected_observations)}else{None},
            "cadence_mismatches":if order_provable && args.expected_step_cadence.is_some(){Some(cadence_mismatches)}else{None},
            "record_gap_assessment":if !order_provable{"ORDER_UNPROVEN"}else if args.expected_step_cadence.is_none(){"EXPECTED_CADENCE_NOT_SUPPLIED"}else if cadence_mismatches>0{"SOURCE_CADENCE_MISMATCH"}else if missing_expected_observations>0{"MISSING_EXPECTED_OBSERVATIONS"}else{"NO_INTERNAL_GAPS_AT_SOURCE_DECLARED_CADENCE"},
            "observed_step_delta_gcd":if observed_cadence>0{Some(observed_cadence)}else{None},
            "frames_bin_interval_semantics":"observed positive step-delta GCD; not evidence every interval was captured",
            "frame_idx_gaps_semantics":"unobserved physical MD step indices, not proof of dropped ZSTR records",
            "non_monotonic_records": if order_provable { serde_json::json!(nonmono) } else { serde_json::Value::Null },
        },
        "energy_measurement_scope": {
            "potential_energy":"native reduced potential-energy plane; excludes kinetic energy",
            "external_work":"raw saved observation-window scalar; capture order and reset lineage must be verified against run source",
            "external_work_capture_status":if work_capture_binding.is_some(){"SOURCE_BOUND_RECEIPT_SUPPLIED"}else{"NOT_ESTABLISHED_FROM_ZSTR_HEADER"},
            "external_work_capture_receipt":work_capture_binding,
            "zero_external_work_proves_zero_intervention":false,
            "kinetic_energy_available":false,
            "complete_first_law_residual_available":false,
            "reason":"ZSTR v1 does not carry velocities/kinetic energy or independently labeled heat and intervention components",
        },
        "physical_time": {
            "status": if verified_dt_ps.is_some() && order_provable {"EMITTED_FROM_SOURCE_BOUND_FIXED_CLOCK"}else{"NOT_EMITTED"},
            "verified_dt_ps":verified_dt_ps,
            "source_bound_clock_receipt":physical_clock_binding,
            "unverified_header_scope": "The ZSTR header dt is a graph-build-time constant. Under --adaptive-dt or the G26 gearbox the real timestep varies (0.5/2.0/4.0 fs), so step x dt is not elapsed physics.",
            "header_dt_ps_nominal": h0.dt_ps,
            "authoritative_source": if verified_dt_ps.is_some() && order_provable {"source_bound_clock_receipt: verified fixed dt and native completed MD step"} else {"join *_governor_decisions.jsonl on global_md_step and use cumulative_simulated_time_ps"},
        },
        "gear": { "status": "UNMEASURED_SENTINEL_HONOURED", "sentinel": "0xFFFFFFFF" },
        "forces": {
            "present_in_stream": true,
            "decoded_to_output": false,
            "caveat": "Staged forces are captured downstream of the ASC steering node and are therefore total (MM + steering) on Construct frames, not conservative -grad(V).",
            "nonfinite_force_norm_records": nonfinite_force_norm,
        },
        "outputs": {
            "frames_bin": args.emit_frames_bin,
            "pdb": args.emit_pdb,
        },
        "per_frame": per_frame,
    });
    let mp = args.output.join(format!("{stem}.zstr_replay_manifest.json"));
    std::fs::write(&mp, serde_json::to_string_pretty(&manifest)? + "\n")
        .with_context(|| format!("write {}", mp.display()))?;
    eprintln!("  wrote {}", mp.display());
    eprintln!(
        "  decoded {} records; gaps={} non_monotonic={}",
        decoded,
        if order_provable { gaps.to_string() } else { "n/a".into() },
        if order_provable { nonmono.to_string() } else { "n/a".into() }
    );
    Ok(())
}

#[cfg(test)]
mod chronology_tests {
    #[test]
    fn cadence_uses_physical_step_differences() {
        assert_eq!([10,10,20,10].into_iter().fold(0,super::step_gcd),10);
        assert_eq!([2,4,6].into_iter().fold(0,super::step_gcd),2);
        assert_eq!(super::step_gcd(0,0),0);
    }
}
