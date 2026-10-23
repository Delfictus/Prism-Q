//! tok_export_corpus — Directive v3 Phase 1: deterministic tokenized training-corpus exporter.
//!
//! Reads the verified TWIN16 `--multi-differential` closure fixture (default
//! `output/5VEX_multidiff_a3`), DECODES QNANL before tokenization (never tokenizes compressed
//! bytes, never trusts filenames), routes the evidence into the packed-`u64` token stream + lossless
//! sidecar, and emits the canonical token corpus (8 files). Deterministic: strided + boundary-row
//! spike sampling, sorted output, no timestamps in content → rerun hashes match.
//!
//! ```bash
//! cargo run -p prism-tokenizer --bin tok_export_corpus -- \
//!   --fixture-root output/5VEX_multidiff_a3 \
//!   --out data/tokenized_training_corpus/twin16_a3 \
//!   --max-spike-records-per-cell 4096 --top-edge-k 64
//! ```

use prism_tokenizer::codec::{decode_qnanl, GpuSpikeEvent};
use prism_tokenizer::sidecar::ExactValue;
use prism_tokenizer::{
    route_object, route_twin16_residual, ArtifactFamily, Encoder, TokenStream, TokenizerBackend,
};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const SEAL: [u8; 32] = [0u8; 32]; // export seal placeholder; real per-chunk hashes live in provenance

struct Args {
    fixture_root: PathBuf,
    out: PathBuf,
    max_spike_records_per_cell: usize,
    top_edge_k: usize,
}

fn parse_args() -> Args {
    let mut a = Args {
        fixture_root: PathBuf::from("output/5VEX_multidiff_a3"),
        out: PathBuf::from("data/tokenized_training_corpus/twin16_a3"),
        max_spike_records_per_cell: 4096,
        top_edge_k: 64,
    };
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--fixture-root" => {
                a.fixture_root = PathBuf::from(&argv[i + 1]);
                i += 2;
            }
            "--out" => {
                a.out = PathBuf::from(&argv[i + 1]);
                i += 2;
            }
            "--max-spike-records-per-cell" => {
                a.max_spike_records_per_cell = argv[i + 1].parse().unwrap_or(4096);
                i += 2;
            }
            "--top-edge-k" => {
                a.top_edge_k = argv[i + 1].parse().unwrap_or(64);
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }
    a
}

fn read_json(p: &Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(p).unwrap_or_default()).unwrap_or(serde_json::Value::Null)
}

/// Deterministic sample of `max_k` record indices from `0..n`: always the boundary rows (0, n-1)
/// plus an even stride. No RNG — reproducible across runs.
fn sample_indices(n: usize, max_k: usize) -> Vec<usize> {
    if n == 0 {
        return vec![];
    }
    if n <= max_k {
        return (0..n).collect();
    }
    let mut idx: Vec<usize> = Vec::with_capacity(max_k);
    for j in 0..max_k {
        let pos = ((j as u128 * (n as u128 - 1)) / (max_k as u128 - 1)) as usize;
        idx.push(pos);
    }
    idx.dedup();
    idx
}

/// Resolve the record stride of a raw spike chunk payload.
///
/// Prefers the declared record count (exact division), then falls back to the
/// known record stride dividing the payload evenly. Returns `None` when the
/// payload matches no known record layout — an unknown layout must fail loudly
/// rather than be parsed at a guessed stride.
fn resolve_spike_chunk_stride(payload_len: usize, record_count: u64) -> Option<usize> {
    use prism_tokenizer::codec::GPU_SPIKE_EVENT_BYTES;
    if record_count > 0 {
        let rc = record_count as usize;
        if rc.checked_mul(GPU_SPIKE_EVENT_BYTES) == Some(payload_len) {
            return Some(GPU_SPIKE_EVENT_BYTES);
        }
    }
    if payload_len > 0 && payload_len % GPU_SPIKE_EVENT_BYTES == 0 {
        return Some(GPU_SPIKE_EVENT_BYTES);
    }
    None
}

fn spike_to_json(s: &GpuSpikeEvent) -> serde_json::Value {
    serde_json::json!({
        "timestep": s.timestep,
        "voxel_idx": s.voxel_idx,
        "position": [s.position[0], s.position[1], s.position[2]],
        "intensity": s.intensity,
        "spike_source": s.spike_source,
        "wavelength_nm": s.wavelength_nm,
        "aromatic_type": s.aromatic_type,
        "aromatic_residue_id": s.aromatic_residue_id,
        "water_density": s.water_density,
        "n_nearby_excited": s.n_nearby_excited,
        "phase_bits": s.phase_bits,
    })
}

fn exact_to_json(e: &ExactValue) -> serde_json::Value {
    match e {
        ExactValue::Scalar(x) => serde_json::json!({"kind": "scalar", "value": x}),
        ExactValue::Vec3(v) => serde_json::json!({"kind": "vec3", "value": v}),
        ExactValue::Text(t) => serde_json::json!({"kind": "text", "value": t}),
    }
}

fn write_jsonl(path: &Path, lines: &[serde_json::Value]) {
    let mut s = String::new();
    for v in lines {
        s.push_str(&serde_json::to_string(v).unwrap_or_default());
        s.push('\n');
    }
    std::fs::write(path, s).unwrap();
}

fn main() {
    let args = parse_args();
    let root = &args.fixture_root;
    std::fs::create_dir_all(&args.out).unwrap();
    let mut enc = Encoder::new();
    let mut stream = TokenStream::new();
    // Directive v3 GPU requirement: prefer the GPU tokenizer when compiled with --features
    // gpu-tokenizer (device + bit-exact parity proven in TOKENIZER_GPU_PARITY_LEDGER.json).
    #[cfg(feature = "gpu-tokenizer")]
    let (vq_backend, backend_name) = (TokenizerBackend::Gpu, "gpu-tokenizer");
    #[cfg(not(feature = "gpu-tokenizer"))]
    let (vq_backend, backend_name) = (TokenizerBackend::Cpu, "cpu-parity-fallback");

    // ── 1. Condition edges (TwinConditionEdge) — the FULL 256-pair TWIN16 factorial ────────────
    // A1's per_cellpair_condition_edges.jsonl is the 240 off-diagonal DIRECTED edges (locked). The
    // canonical 16×16 = 256 factorial is the TWIN16 artifact's pairwise_residuals; we enrich each
    // pair to the TwinConditionEdge schema (axis/hamming/material via the same bit logic as A1) so
    // condition_edges.jsonl carries all 256 (240 directed + 16 diagonal self-pairs).
    let twin16 = read_json(&root.join("TWIN16_INTERFEROMETRIC_TRANSITION_VIEW_RESIDUALS.json"));
    let axes = ["thermal", "uv", "ramp", "return"];
    let twin16_idx = |c: &str| -> Option<u8> {
        let bits = c.strip_prefix("TWIN16_")?;
        if bits.len() == 4 && bits.chars().all(|ch| ch == '0' || ch == '1') {
            u8::from_str_radix(bits, 2).ok()
        } else {
            None
        }
    };
    let pairwise = twin16
        .get("pairwise_residuals")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let mut condition_edges: Vec<serde_json::Value> = Vec::new();
    for p in &pairwise {
        let ca = p.get("cell_a").and_then(|v| v.as_str()).unwrap_or("");
        let cb = p.get("cell_b").and_then(|v| v.as_str()).unwrap_or("");
        let residual = p
            .get("transition_view_l1_residual")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let (ham, axis) = match (twin16_idx(ca), twin16_idx(cb)) {
            (Some(a), Some(b)) => {
                let xor = a ^ b;
                let h = xor.count_ones();
                let ax = if h == 1 {
                    serde_json::Value::String(axes[xor.trailing_zeros() as usize % 4].to_string())
                } else {
                    serde_json::Value::Null
                };
                (h, ax)
            }
            _ => (0, serde_json::Value::Null),
        };
        let mut rec = serde_json::json!({
            "cell_a": ca, "cell_b": cb, "hamming": ham,
            "transition_view_l1_residual": residual, "material": residual > 0.05,
        });
        if !axis.is_null() {
            rec.as_object_mut()
                .unwrap()
                .insert("axis".to_string(), axis);
        }
        let obj = rec.as_object().unwrap().clone();
        let seg = stream.begin_record(ArtifactFamily::TwinConditionEdge, condition_edges.len());
        route_object(
            &enc,
            &mut stream,
            ArtifactFamily::TwinConditionEdge,
            &obj,
            true,
            SEAL,
            true,
        );
        stream.end_record(seg);
        condition_edges.push(rec);
    }
    // top-k edges by |residual| (deterministic) — recorded in the card for the bridge stage.
    let mut ranked: Vec<(f64, usize)> = condition_edges
        .iter()
        .enumerate()
        .map(|(i, e)| {
            (
                e.get("transition_view_l1_residual")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0),
                i,
            )
        })
        .collect();
    ranked.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    let top_edge_idx: Vec<usize> = ranked
        .iter()
        .take(args.top_edge_k)
        .map(|(_, i)| *i)
        .collect();

    // ── 2. TWIN16 cells — route + per-cell summaries (reuses `twin16` read above) ──────────────
    route_twin16_residual(&enc, &mut stream, &twin16, true, SEAL);
    let cells = twin16
        .get("cells")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    // ── 3. Spike chunks — decode QNANL, bounded per-cell sampling + summaries ──────────────────
    let manifest = read_json(&root.join("SPIKE_EVENT_STREAM/MANIFEST.qnanl.json"));
    let rows = manifest
        .get("rows")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let mut total_raw_spikes: u64 = 0;
    let mut sampled_spikes: u64 = 0;
    let mut cell_summaries: Vec<serde_json::Value> = Vec::new();
    let mut sampled: Vec<(u64, GpuSpikeEvent)> = Vec::new();
    let mut positions: Vec<[f64; 3]> = Vec::new();
    for row in &rows {
        let chunk_id = row.get("chunk_id").and_then(|v| v.as_u64()).unwrap_or(0);
        let cell = row
            .get("twin16_cell")
            .and_then(|v| v.as_str())
            .unwrap_or("UNKNOWN")
            .to_string();
        let stream_id = row.get("stream_id").and_then(|v| v.as_u64()).unwrap_or(0);
        let rec_count = row
            .get("record_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        total_raw_spikes += rec_count;
        let chunkp = root.join(format!(
            "SPIKE_EVENT_STREAM/chunks/spike_chunk_{:06}.prsqev",
            chunk_id
        ));
        let payload = decode_qnanl(&std::fs::read(&chunkp).unwrap())
            .ok()
            .and_then(|d| d.payload)
            .unwrap_or_default();
        // Chunk records are raw GpuSpikeEvent bytes. The stride is resolved from
        // the payload rather than assumed: a payload that does not divide by the
        // record width would otherwise be read at the wrong offset and emit
        // plausible-looking physics with no diagnostic.
        let stride = resolve_spike_chunk_stride(payload.len(), rec_count).unwrap_or_else(|| {
            panic!(
                "spike chunk {} payload of {} bytes with record_count {} does not match the \
                 GpuSpikeEvent record stride ({} B)",
                chunk_id,
                payload.len(),
                rec_count,
                prism_tokenizer::codec::GPU_SPIKE_EVENT_BYTES,
            )
        });
        let n = payload.len() / stride;
        let idxs = sample_indices(n, args.max_spike_records_per_cell);
        // per-cell aggregate over ALL records (not just sampled) for honest summaries.
        let (mut sum_i, mut n_uv, mut sum_w) = (0.0f64, 0u64, 0.0f64);
        for k in 0..n {
            if let Some(ev) =
                GpuSpikeEvent::from_le_bytes_with_stride(&payload[k * stride..], stride)
            {
                sum_i += ev.intensity as f64;
                sum_w += ev.water_density as f64;
                if ev.spike_source == 1 {
                    n_uv += 1;
                }
            }
        }
        // collect the sampled spikes + positions; routing happens after the GPU/CPU VQ fit below.
        for &k in &idxs {
            if let Some(ev) =
                GpuSpikeEvent::from_le_bytes_with_stride(&payload[k * stride..], stride)
            {
                positions.push([
                    ev.position[0] as f64,
                    ev.position[1] as f64,
                    ev.position[2] as f64,
                ]);
                sampled.push((chunk_id, ev));
            }
        }
        let denom = n.max(1) as f64;
        let cell_meta = cells
            .iter()
            .find(|c| c.get("condition_id").and_then(|v| v.as_str()) == Some(&cell));
        cell_summaries.push(serde_json::json!({
            "twin16_cell": cell,
            "stream_id": stream_id,
            "chunk_id": chunk_id,
            "raw_spike_records": rec_count,
            "sampled_records": idxs.len(),
            "mean_intensity": sum_i / denom,
            "mean_water_density": sum_w / denom,
            "uv_fraction": n_uv as f64 / denom,
            "protocol_name": cell_meta.and_then(|c| c.get("protocol_name").cloned()).unwrap_or(serde_json::Value::Null),
            "bocpd_count": cell_meta.and_then(|c| c.get("bocpd_count").cloned()).unwrap_or(serde_json::Value::Null),
        }));
    }

    // GPU/CPU VQ fit of the sampled spike positions (the GPU tokenizer path; with --features
    // gpu-tokenizer + a device this fits the Cα/position codebook on GPU — bit-exact vs CPU per
    // TOKENIZER_GPU_PARITY_LEDGER.json). Then route the spikes with the fitted encoder.
    let vq_fit_ran = enc.fit_coords_with_backend(
        ArtifactFamily::SpikeEvent,
        "position",
        &positions,
        vq_backend,
    );
    let mut cur_chunk = u64::MAX;
    let mut seg = 0usize;
    for (chunk_id, ev) in &sampled {
        if *chunk_id != cur_chunk {
            if cur_chunk != u64::MAX {
                stream.end_record(seg);
            }
            seg = stream.begin_record(ArtifactFamily::SpikeEvent, *chunk_id as usize);
            cur_chunk = *chunk_id;
        }
        if let Some(obj) = spike_to_json(ev).as_object() {
            route_object(
                &enc,
                &mut stream,
                ArtifactFamily::SpikeEvent,
                obj,
                true,
                SEAL,
                true,
            );
            sampled_spikes += 1;
        }
    }
    if cur_chunk != u64::MAX {
        stream.end_record(seg);
    }

    // ── Exports (deterministic) ───────────────────────────────────────────────────────────────
    let tokens: Vec<serde_json::Value> = stream
        .tokens
        .iter()
        .enumerate()
        .map(|(i, t)| serde_json::json!({"pos": i, "token_u64": t.0}))
        .collect();
    write_jsonl(&args.out.join("tokens.u64.jsonl"), &tokens);

    let segments: Vec<serde_json::Value> = stream
        .segments
        .iter()
        .map(|s| serde_json::json!({"start": s.start, "len": s.len, "record_index": s.record_index, "family": format!("{:?}", s.family)}))
        .collect();
    write_jsonl(&args.out.join("segments.jsonl"), &segments);

    let sidecar: Vec<serde_json::Value> = stream
        .sidecar
        .iter()
        .map(|e| serde_json::json!({"position": e.position, "field": e.field, "exact": exact_to_json(&e.exact)}))
        .collect();
    write_jsonl(&args.out.join("sidecar.jsonl"), &sidecar);

    // provenance: receipt ops + manifest links (no token may be unprovenanced for the receipts).
    let spike_receipt = read_json(&root.join("SPIKE_EVENT_STREAM/RECEIPT.prismcode.json"));
    let edge_receipt = read_json(&root.join("CONDITION_EDGE_READOUT/RECEIPT.prismcode.json"));
    let mut provenance: Vec<serde_json::Value> = Vec::new();
    provenance.push(serde_json::json!({
        "kind": "spike_stream_manifest",
        "path": "SPIKE_EVENT_STREAM/MANIFEST.qnanl.json",
        "n_chunks": manifest.get("n_chunks").cloned().unwrap_or(serde_json::Value::Null),
        "all_chunks_roundtrip_verified": manifest.get("all_chunks_roundtrip_verified").cloned().unwrap_or(serde_json::Value::Null),
    }));
    if let Some(ops) = spike_receipt.get("ops").and_then(|v| v.as_array()) {
        for op in ops {
            provenance.push(serde_json::json!({"kind": "spike_receipt_op", "op": op.clone()}));
        }
    }
    if let Some(ops) = edge_receipt.get("ops").and_then(|v| v.as_array()) {
        for op in ops {
            provenance
                .push(serde_json::json!({"kind": "condition_edge_receipt_op", "op": op.clone()}));
        }
    }
    write_jsonl(&args.out.join("provenance.jsonl"), &provenance);

    write_jsonl(&args.out.join("cell_summaries.jsonl"), &cell_summaries);
    write_jsonl(&args.out.join("condition_edges.jsonl"), &condition_edges);

    let card = serde_json::json!({
        "schema_kind": "tokenized_training_corpus_card",
        "fixture_root": root.display().to_string(),
        "lane_a_status": "IMPLEMENTATION_VERIFIED",
        "substrate": "twin16_multi_differential_captured_graph",
        "coupled_ab_excluded": true,
        "tokenizer_backend": backend_name,
        "vq_fit_ran": vq_fit_ran,
        "gpu_parity_ledger": "OPERATOR_CONTROL/TOKENIZER_GPU_PARITY_LEDGER.json",
        "gpu_parity_note": "CPU/GPU bit-exact parity proven (vq_assign + vq_fit codes_equal/distortion_bitwise + coord_encode exact). With --features gpu-tokenizer the position VQ fit runs on GPU; CPU build records cpu-parity-fallback (bit-identical).",
        "total_raw_spike_records": total_raw_spikes,
        "sampled_spike_records": sampled_spikes,
        "max_spike_records_per_cell": args.max_spike_records_per_cell,
        "top_edge_k": args.top_edge_k,
        "top_edge_indices": top_edge_idx,
        "n_condition_edges": condition_edges.len(),
        "n_twin16_cells": cell_summaries.len(),
        "n_tokens": stream.tokens.len(),
        "n_segments": stream.segments.len(),
        "n_sidecar_entries": stream.sidecar.len(),
        "spike_stream_manifest_verified": manifest.get("all_chunks_roundtrip_verified").cloned().unwrap_or(serde_json::Value::Null),
        "spike_receipt_ops": spike_receipt.get("n_ops").cloned().unwrap_or(serde_json::Value::Null),
        "condition_edge_receipt_op": edge_receipt.get("ops").and_then(|o| o.get(0)).and_then(|o| o.get("op")).cloned().unwrap_or(serde_json::Value::Null),
        "outputs": ["tokens.u64.jsonl", "segments.jsonl", "sidecar.jsonl", "provenance.jsonl", "cell_summaries.jsonl", "condition_edges.jsonl", "corpus_card.json", "sha256sums.txt"],
    });
    std::fs::write(
        args.out.join("corpus_card.json"),
        serde_json::to_string_pretty(&card).unwrap(),
    )
    .unwrap();

    // sha256sums.txt over every emitted file (sorted for determinism).
    let mut sums: BTreeMap<String, String> = BTreeMap::new();
    for f in [
        "tokens.u64.jsonl",
        "segments.jsonl",
        "sidecar.jsonl",
        "provenance.jsonl",
        "cell_summaries.jsonl",
        "condition_edges.jsonl",
        "corpus_card.json",
    ] {
        let bytes = std::fs::read(args.out.join(f)).unwrap_or_default();
        let mut h = Sha256::new();
        h.update(&bytes);
        sums.insert(f.to_string(), format!("{:x}", h.finalize()));
    }
    let sums_txt: String = sums
        .iter()
        .map(|(f, h)| format!("{}  {}\n", h, f))
        .collect();
    std::fs::write(args.out.join("sha256sums.txt"), sums_txt).unwrap();

    eprintln!(
        "tok_export_corpus: {} tokens, {} segments, {} sidecar, {} edges, {} cells, {} raw spikes ({} sampled) → {}",
        stream.tokens.len(),
        stream.segments.len(),
        stream.sidecar.len(),
        condition_edges.len(),
        cell_summaries.len(),
        total_raw_spikes,
        sampled_spikes,
        args.out.display()
    );
}
