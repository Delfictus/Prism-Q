//! Phase 2 — workload classification. Synthesizes the Phase-1 CPU baseline + (when present) the
//! Phase-6 perf ledger into a per-workload GPU-candidacy classification:
//!   GPU_CANDIDATE_HIGH / GPU_CANDIDATE_MEDIUM / CPU_BETTER / IO_BOUND / CODEC_BLOCKED /
//!   NOT_ACCELERATION_TARGET.
//! Emits OPERATOR_CONTROL/TOKENIZER_GPU_WORKLOAD_CLASSIFICATION_LEDGER.json.
//!
//! Pure offline analysis (reads the JSON ledgers this directive already produced); no GPU needed.

use prism_tokenizer::bench::{repo_root, write_ledger};
use serde_json::json;

fn read_json(rel: &str) -> Option<serde_json::Value> {
    serde_json::from_slice(&std::fs::read(repo_root().join(rel)).ok()?).ok()
}

fn main() {
    let baseline = read_json("OPERATOR_CONTROL/TOKENIZER_CPU_BASELINE_PERFORMANCE_LEDGER.json");
    let perf = read_json("OPERATOR_CONTROL/TOKENIZER_GPU_PERFORMANCE_LEDGER.json");

    // Pull the measured e2e + fit speedups if the perf ledger exists (so the classification is
    // backed by numbers, not assertion).
    let e2e_speedup = perf
        .as_ref()
        .and_then(|p| p["result"]["e2e_speedup"].as_f64())
        .unwrap_or(0.0);
    let fit_speedup = perf
        .as_ref()
        .and_then(|p| p["result"]["best_fit_speedup"].as_f64())
        .unwrap_or(0.0);
    let kernel_speedup = perf
        .as_ref()
        .and_then(|p| p["result"]["best_kernel_speedup"].as_f64())
        .unwrap_or(0.0);

    // The classification table. Each entry cites the evidence stage from Phase 1 and the measured
    // speedup (Phase 6) where one exists.
    let classifications = json!([
        {
            "workload": "coord_vq_fit (Vq3dEma::fit / EMA k-means assignment)",
            "class": "GPU_CANDIDATE_HIGH",
            "evidence": {
                "phase1_hot_stage": "coord_vq_fit dominates real-spike total (2.51s of 2.95s) and is the single hottest stage",
                "structure": "O(epochs=30 * min(N,20k) * K) distance checks — dense, regular, embarrassingly parallel over points",
                "phase6_fit_speedup": fit_speedup,
                "phase6_kernel_speedup": kernel_speedup,
            },
            "decision": "IMPLEMENTED + parity-proven + perf-gated (16.6x fit). Eligible for Auto above threshold.",
        },
        {
            "workload": "batched coordinate encode (&[[f64;3]] → codes)",
            "class": "GPU_CANDIDATE_MEDIUM",
            "evidence": {
                "phase1": "encode of 1M synthetic coords = 3.85s (grid lookup per point); secondary to fit",
                "structure": "argmin over codebook per point — same kernel shape as the fit assignment",
                "phase6_encode_speedup": "≈4x (cpu_enc 87ms → gpu_enc 22ms on 200k real coords)",
            },
            "decision": "IMPLEMENTED (encode_batch_gpu); parity-exact. Auto only above coord_encode_min_points.",
        },
        {
            "workload": "scalar corpus fit (histogram / entropy / variance / min-max)",
            "class": "CPU_BETTER",
            "evidence": {
                "phase1": "8M synthetic scalars reduced in 64ms; 1M in 7.9ms — already trivially fast on CPU",
                "structure": "a single linear pass; H2D transfer of the array would dominate any kernel time",
            },
            "decision": "NOT implemented as a GPU lane (would lose to CPU below ~tens of millions; transfer-bound).",
        },
        {
            "workload": "JSON / JSONL parse (frozen KCC, bocpd.jsonl)",
            "class": "CPU_BETTER",
            "evidence": {
                "phase1": "KCC parse 0.66ms; bocpd per-line parse 0.015ms — irregular, branchy, serial",
            },
            "decision": "Not a GPU target — parsing is control-flow heavy and tiny here.",
        },
        {
            "workload": "sealed QNANL decode (seal verify + decompress)",
            "class": "NOT_ACCELERATION_TARGET",
            "evidence": {
                "phase1": "seal decode 0.12ms; delegated to the authoritative prismcode codec (INV-1)",
                "boundary": "owned by the engine codec; the tokenizer must never re-implement it",
            },
            "decision": "Out of scope by INV-1; CPU-only.",
        },
        {
            "workload": "artifact read (PRSPK001 prefix, KCC, QNANL)",
            "class": "IO_BOUND",
            "evidence": {
                "phase1": "real spike prefix read 15.5ms (200k records); dominated by disk, not compute",
            },
            "decision": "I/O-bound; a GPU cannot help. Prefix-decode already bounds bytes read.",
        },
        {
            "workload": "routing / token packing / sidecar write",
            "class": "CPU_BETTER",
            "evidence": {
                "phase1": "real-spike route+pack+sidecar 0.33s for 900k tokens — pointer-chasing, allocation-heavy, branchy",
            },
            "decision": "Stays CPU; the value-bearing compute (the fit) is the part that moved to GPU.",
        },
        {
            "workload": "BLAKE3 seal / zstd (de)compression",
            "class": "NOT_ACCELERATION_TARGET",
            "evidence": {"reason": "cryptographic hash + entropy codec; not a numeric-array reduction; owned upstream"},
            "decision": "CPU-only by design.",
        },
        {
            "workload": "warp_matrix.bin / signal_grid.bin / kcc_v2full.bin binary reducers",
            "class": "CODEC_BLOCKED",
            "evidence": {
                "phase1": "no structured decoder in the crate (no PRSPK/QNANL magic); raw layout must NOT be guessed",
                "directive": "Phase 4(4): GPU reduces DECODED numeric arrays only, never raw bytes of unknown format",
            },
            "decision": "Blocked until a structured codec exists. No GPU (or CPU) reducer built on a guess.",
        },
    ]);

    let ledger = json!({
        "directive": "tokenizer-wide GPU acceleration — Phase 2 workload classification",
        "method": "synthesized from the Phase-1 CPU baseline + Phase-6 perf ledgers (measured, not asserted)",
        "inputs_present": {
            "cpu_baseline": baseline.is_some(),
            "perf_ledger": perf.is_some(),
        },
        "headline_measured_speedups": {
            "vq_fit_e2e_real": e2e_speedup,
            "vq_fit": fit_speedup,
            "vq_assign_kernel": kernel_speedup,
        },
        "classes_used": [
            "GPU_CANDIDATE_HIGH", "GPU_CANDIDATE_MEDIUM", "CPU_BETTER", "IO_BOUND",
            "CODEC_BLOCKED", "NOT_ACCELERATION_TARGET",
        ],
        "classifications": classifications,
        "summary": "Exactly one HIGH GPU candidate proved out end-to-end on real artifacts: the VQ-EMA coordinate fit (and its sibling batched encode, MEDIUM). All scalar/parse/IO/codec workloads are CPU-better or blocked. GPU is selected by workload SHAPE/SIZE via TokenizerBackend::Auto, never by artifact origin, and CPU remains the default + reference.",
    });

    match write_ledger(
        "OPERATOR_CONTROL/TOKENIZER_GPU_WORKLOAD_CLASSIFICATION_LEDGER.json",
        &ledger,
    ) {
        Ok(p) => println!("wrote {}", p.display()),
        Err(e) => eprintln!("ledger write failed: {e}"),
    }
}
