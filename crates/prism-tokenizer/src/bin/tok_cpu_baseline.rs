//! Phase 1 — CPU baseline performance harness for the tokenizer-wide GPU acceleration directive.
//!
//! Measures per-stage wall timings + peak RSS on REAL PRISM artifacts (PRSPK001 spikes, frozen
//! KCC fixture, sealed QNANL, bocpd.jsonl) and on deterministic synthetic large corpora (coords +
//! scalars). For binary surfaces without a structured decoder (warp_matrix.bin) it measures raw
//! read + decode-attempt and marks CODEC_BLOCKED. Emits:
//!   OPERATOR_CONTROL/TOKENIZER_CPU_BASELINE_PERFORMANCE_LEDGER.json
//!
//! CPU-only; buildable and runnable without any GPU. Determinism: synthetic corpora are fixed-seed.

use std::collections::HashMap;
use std::time::Instant;

use prism_tokenizer::alloc::measure_corpus;
use prism_tokenizer::bench::{
    bocpd_fixture, find_real_spikes_bin, kcc_fixture, load_kcc_scalars, peak_rss_kib,
    qnanl_fixture, read_prefix, repo_root, synth_coords, synth_scalars, timed, warp_matrix_fixture,
    write_ledger,
};
use prism_tokenizer::codec::{decode_prspk001_head, decode_qnanl};
use prism_tokenizer::quant::Vq3dEma;
use prism_tokenizer::stream::TokenStream;
use prism_tokenizer::types::ArtifactFamily;
use prism_tokenizer::Encoder;
use serde_json::json;

fn now_iso() -> String {
    // no chrono dep here; epoch seconds is enough for a provenance stamp.
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("epoch_{secs}")
}

fn main() {
    let mut workloads: Vec<serde_json::Value> = Vec::new();

    // -- 1. real PRSPK001 spikes: read prefix → decode → fit coords → batch encode --------------
    workloads.push(bench_real_spikes());

    // -- 2. frozen KCC fixture: read → JSON parse → route → scalar corpus fit (measure) ---------
    workloads.push(bench_kcc_fixture());

    // -- 3. sealed QNANL: read → seal decode → JSON parse → route --------------------------------
    workloads.push(bench_qnanl());

    // -- 4. bocpd.jsonl: read → per-line JSON parse → route -------------------------------------
    workloads.push(bench_bocpd());

    // -- 5. synthetic large coordinate corpus: VQ fit + batch encode (the hot loop) -------------
    for &n in &[100_000usize, 1_000_000] {
        workloads.push(bench_synth_coords(n, 4096));
    }

    // -- 6. synthetic large scalar corpus: histogram/entropy/variance (alloc measure) -----------
    for &n in &[1_000_000usize, 8_000_000] {
        workloads.push(bench_synth_scalars(n));
    }

    // -- 7. binary surfaces without a structured decoder: CODEC_BLOCKED -------------------------
    workloads.push(bench_codec_blocked_warp_matrix());

    let ledger = json!({
        "directive": "tokenizer-wide GPU acceleration — Phase 1 CPU baseline",
        "generated": now_iso(),
        "host": {
            "gpu": "RTX 5080 16GB (present; not used in this CPU baseline)",
            "cpu_reference": true,
            "peak_rss_kib_final": peak_rss_kib(),
        },
        "stage_legend": [
            "artifact_read", "codec_decode", "json_parse", "binary_decode", "routing",
            "scalar_quant_fit", "coord_vq_fit", "encode", "token_packing", "sidecar_write",
            "total_wall_s", "peak_rss_kib",
        ],
        "workloads": workloads,
        "notes": [
            "All real artifacts decoded through the crate's own authoritative codecs (INV-1).",
            "Synthetic corpora are fixed-seed deterministic (no rand dep).",
            "coord_vq_fit is the confirmed hot loop (Vq3dEma::fit, O(epochs*N*K)).",
            "Binary surfaces with no structured decoder are CODEC_BLOCKED, never raw-parsed.",
        ],
    });

    match write_ledger(
        "OPERATOR_CONTROL/TOKENIZER_CPU_BASELINE_PERFORMANCE_LEDGER.json",
        &ledger,
    ) {
        Ok(p) => println!("wrote {}", p.display()),
        Err(e) => eprintln!("ledger write failed: {e}"),
    }
}

fn bench_real_spikes() -> serde_json::Value {
    let Some(path) = find_real_spikes_bin() else {
        return json!({"workload": "real_prspk001_spikes", "status": "FIXTURE_ABSENT"});
    };
    let max_records = 200_000usize;
    let prefix_bytes = 4096 + max_records * 96 + 4096;
    let rss0 = peak_rss_kib();

    let (buf, t_read) = timed(|| read_prefix(&path, prefix_bytes));
    let Some(buf) = buf else {
        return json!({"workload": "real_prspk001_spikes", "status": "READ_FAILED"});
    };

    let (decoded, t_decode) = timed(|| decode_prspk001_head(&buf, max_records));
    let Ok(decoded) = decoded else {
        return json!({"workload": "real_prspk001_spikes", "status": "DECODE_FAILED", "path": path.display().to_string()});
    };

    let coords: Vec<[f64; 3]> = decoded
        .events
        .iter()
        .map(|e| {
            [
                e.position[0] as f64,
                e.position[1] as f64,
                e.position[2] as f64,
            ]
        })
        .collect();

    // coord VQ fit (the hot loop) on the real spike coordinates.
    let (vq, t_fit) = timed(|| {
        let mut q = Vq3dEma::new(12);
        q.fit(&coords);
        q
    });

    // batch encode all real coords.
    let (codes, t_encode) = timed(|| vq.encode_batch_cpu(&coords));

    // full route → token packing + sidecar write.
    let mut enc = Encoder::new();
    enc.fit_coords(ArtifactFamily::SpikeEvent, "position", &coords);
    let (stream, t_route) = timed(|| {
        let mut s = TokenStream::new();
        for ev in &decoded.events {
            enc.route_spike(&mut s, ev, true, [0u8; 32]);
        }
        s
    });

    let total = t_read + t_decode + t_fit + t_encode + t_route;
    json!({
        "workload": "real_prspk001_spikes",
        "status": "MEASURED",
        "path": path.display().to_string(),
        "declared_record_count": decoded.record_count,
        "records_decoded": decoded.events.len(),
        "coords_fit": coords.len(),
        "codes_emitted": codes.0.len(),
        "tokens": stream.len(),
        "stages_s": {
            "artifact_read": t_read,
            "codec_decode": t_decode,
            "coord_vq_fit": t_fit,
            "encode": t_encode,
            "routing_pack_sidecar": t_route,
        },
        "total_wall_s": total,
        "peak_rss_kib": peak_rss_kib().max(rss0),
        "hot_stage": "coord_vq_fit",
    })
}

fn bench_kcc_fixture() -> serde_json::Value {
    let Some(path) = kcc_fixture() else {
        return json!({"workload": "frozen_kcc_fixture", "status": "FIXTURE_ABSENT"});
    };
    let (bytes, t_read) = timed(|| std::fs::read(&path).ok());
    let Some(bytes) = bytes else {
        return json!({"workload": "frozen_kcc_fixture", "status": "READ_FAILED"});
    };
    let (json_val, t_parse) = timed(|| serde_json::from_slice::<serde_json::Value>(&bytes).ok());
    let Some(_json_val) = json_val else {
        return json!({"workload": "frozen_kcc_fixture", "status": "PARSE_FAILED"});
    };
    let Some(samples) = load_kcc_scalars(&path) else {
        return json!({"workload": "frozen_kcc_fixture", "status": "NO_RESIDUES"});
    };
    let n_residues = samples.values().map(|v| v.len()).max().unwrap_or(0);

    let enc = Encoder::new();
    let (infos, t_measure) =
        timed(|| measure_corpus(&enc, ArtifactFamily::KccDescriptor, &samples));

    let total = t_read + t_parse + t_measure;
    json!({
        "workload": "frozen_kcc_fixture",
        "status": "MEASURED",
        "path": path.display().to_string(),
        "n_residues": n_residues,
        "scalar_fields_measured": infos.len(),
        "stages_s": {
            "artifact_read": t_read,
            "json_parse": t_parse,
            "scalar_quant_fit_measure": t_measure,
        },
        "total_wall_s": total,
        "peak_rss_kib": peak_rss_kib(),
        "note": "small artifact (≈396 residues × 13 fields) — CPU-bound parse + measure; not a GPU target",
    })
}

fn bench_qnanl() -> serde_json::Value {
    let Some(path) = qnanl_fixture() else {
        return json!({"workload": "sealed_qnanl", "status": "FIXTURE_ABSENT"});
    };
    let (bytes, t_read) = timed(|| std::fs::read(&path).ok());
    let Some(bytes) = bytes else {
        return json!({"workload": "sealed_qnanl", "status": "READ_FAILED"});
    };
    let (decoded, t_decode) = timed(|| decode_qnanl(&bytes).ok());
    let Some(decoded) = decoded else {
        return json!({"workload": "sealed_qnanl", "status": "SEAL_DECODE_FAILED"});
    };
    let payload = decoded.payload.unwrap_or_default();
    let (parsed, t_parse) = timed(|| serde_json::from_slice::<serde_json::Value>(&payload).ok());

    let total = t_read + t_decode + t_parse;
    json!({
        "workload": "sealed_qnanl",
        "status": "MEASURED",
        "path": path.display().to_string(),
        "bytes": bytes.len(),
        "seal_verified": decoded.verified,
        "payload_bytes": payload.len(),
        "json_parsed": parsed.is_some(),
        "stages_s": {
            "artifact_read": t_read,
            "codec_decode": t_decode,
            "json_parse": t_parse,
        },
        "total_wall_s": total,
        "peak_rss_kib": peak_rss_kib(),
        "note": "seal-codec + JSON parse: serial, CPU-bound — NOT_ACCELERATION_TARGET",
    })
}

fn bench_bocpd() -> serde_json::Value {
    let Some(path) = bocpd_fixture() else {
        return json!({"workload": "bocpd_jsonl", "status": "FIXTURE_ABSENT"});
    };
    let (bytes, t_read) = timed(|| std::fs::read(&path).ok());
    let Some(bytes) = bytes else {
        return json!({"workload": "bocpd_jsonl", "status": "READ_FAILED"});
    };
    let text = String::from_utf8_lossy(&bytes).into_owned();
    let t = Instant::now();
    let mut parsed_lines = 0usize;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if serde_json::from_str::<serde_json::Value>(line).is_ok() {
            parsed_lines += 1;
        }
    }
    let t_parse = t.elapsed().as_secs_f64();
    let total = t_read + t_parse;
    json!({
        "workload": "bocpd_jsonl",
        "status": "MEASURED",
        "path": path.display().to_string(),
        "bytes": bytes.len(),
        "lines_parsed": parsed_lines,
        "stages_s": { "artifact_read": t_read, "json_parse": t_parse },
        "total_wall_s": total,
        "peak_rss_kib": peak_rss_kib(),
        "note": "per-line JSON parse + enum/string routing — CPU_BETTER (irregular, serial)",
    })
}

fn bench_synth_coords(n: usize, codebook: usize) -> serde_json::Value {
    let coords = synth_coords(n, 12, 0xC00Du64);
    let bits = (codebook as f64).log2().round() as u8;

    let (vq, t_fit) = timed(|| {
        let mut q = Vq3dEma::new(bits);
        q.fit(&coords);
        q
    });
    let (codes, t_encode) = timed(|| vq.encode_batch_cpu(&coords));

    json!({
        "workload": format!("synth_coords_{n}"),
        "status": "MEASURED",
        "n_points": n,
        "codebook_size": 1usize << bits,
        "codebook_bits": bits,
        "stages_s": { "coord_vq_fit": t_fit, "encode": t_encode },
        "total_wall_s": t_fit + t_encode,
        "peak_rss_kib": peak_rss_kib(),
        "codes": codes.0.len(),
        "hot_stage": "coord_vq_fit",
        "note": "the confirmed hot loop; primary GPU candidate (O(epochs*min(N,20k)*K))",
    })
}

fn bench_synth_scalars(n: usize) -> serde_json::Value {
    let s = synth_scalars(n, 1e-3, 1e3, 0x5CA1u64);

    // CPU scalar reductions: min/max/mean/variance + a 4096-bin histogram + Shannon entropy.
    let t = Instant::now();
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let mut sum = 0.0;
    for &x in &s {
        lo = lo.min(x);
        hi = hi.max(x);
        sum += x;
    }
    let mean = sum / n as f64;
    let mut var = 0.0;
    for &x in &s {
        var += (x - mean) * (x - mean);
    }
    var /= n as f64;
    const BINS: usize = 4096;
    let mut hist = vec![0u64; BINS];
    let span = (hi - lo).max(1e-12);
    for &x in &s {
        let b = (((x - lo) / span) * BINS as f64).floor() as i64;
        let b = b.clamp(0, BINS as i64 - 1) as usize;
        hist[b] += 1;
    }
    let mut entropy = 0.0;
    let nf = n as f64;
    for &c in &hist {
        if c > 0 {
            let p = c as f64 / nf;
            entropy -= p * p.log2();
        }
    }
    let t_reduce = t.elapsed().as_secs_f64();

    json!({
        "workload": format!("synth_scalars_{n}"),
        "status": "MEASURED",
        "n": n,
        "min": lo, "max": hi, "mean": mean, "variance": var, "entropy_bits": entropy,
        "stages_s": { "scalar_reduce_histogram_entropy": t_reduce },
        "total_wall_s": t_reduce,
        "peak_rss_kib": peak_rss_kib(),
        "note": "scalar histogram/entropy/range reduction; GPU candidate only at very large N",
    })
}

fn bench_codec_blocked_warp_matrix() -> serde_json::Value {
    let Some(path) = warp_matrix_fixture() else {
        return json!({"workload": "warp_matrix_bin", "status": "FIXTURE_ABSENT", "classification": "CODEC_BLOCKED"});
    };
    // measure raw read of a prefix + a decode ATTEMPT (no structured decoder exists → blocked).
    let (buf, t_read) = timed(|| read_prefix(&path, 1 << 20)); // 1 MiB prefix
    let buf = buf.unwrap_or_default();
    // decode-attempt: there is no PRSPK/QNANL magic; we do NOT guess the layout (no ad-hoc parse).
    let magic_ok = buf.len() >= 8
        && (&buf[0..8] == b"PRSPK001" || &buf[0..8] == b"QNANLv01" || &buf[0..8] == b"PRSPCZ01");
    json!({
        "workload": "warp_matrix_bin",
        "status": "CODEC_BLOCKED",
        "classification": "CODEC_BLOCKED",
        "path": path.display().to_string(),
        "prefix_bytes_read": buf.len(),
        "stages_s": { "artifact_read": t_read },
        "structured_decoder_present": magic_ok,
        "note": "no structured decoder for *_warp_matrix.bin in the crate; per directive Phase 4(4), GPU reducers operate ONLY on DECODED numeric arrays — raw layout is NEVER guessed. Blocked until a codec exists.",
    })
}

// silence the unused import on builds where repo_root isn't otherwise referenced.
#[allow(dead_code)]
fn _keep_repo_root() -> std::path::PathBuf {
    repo_root()
}

// silence unused HashMap import warning in case the compiler can't see the use through serde.
#[allow(dead_code)]
type _Keep = HashMap<&'static str, Vec<f64>>;
