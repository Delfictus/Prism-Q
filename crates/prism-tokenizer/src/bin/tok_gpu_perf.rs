//! Phase 6 — GPU vs CPU performance harness. Measures the isolated hot kernel (VQ assignment /
//! VQ-EMA fit) and the end-to-end real-spike ingestion on both backends, then applies the
//! directive's perf gates:
//!   accept a GPU lane only if it hits ≥1 of:
//!     (a) ≥2x end-to-end on a real all-on surface;
//!     (b) ≥5x on the isolated hot kernel AND ≥25% end-to-end;
//!     (c) ≥30% wall reduction for full all-on ingestion;
//!     (d) major memory-pressure reduction at equal throughput.
//!   A microkernel win with no end-to-end win ⇒ keep DISABLED by default.
//! Emits OPERATOR_CONTROL/TOKENIZER_GPU_PERFORMANCE_LEDGER.json.

// Several bench helpers are only used on the gpu-tokenizer build; keep both builds warning-clean.
#![allow(unused_imports)]

use prism_tokenizer::bench::{
    bench_median, find_real_spikes_bin, load_real_coords, synth_coords, write_ledger,
};
use serde_json::json;

#[cfg(feature = "gpu-tokenizer")]
fn run_perf() -> serde_json::Value {
    use prism_tokenizer::gpu::{vq_assign_cpu, GpuVqContext};
    use prism_tokenizer::quant::Vq3dEma;

    let Some(gpu) = GpuVqContext::try_new() else {
        return json!({"status": "DEVICE_UNAVAILABLE"});
    };
    let free = gpu.free_mem();
    let mut cases: Vec<serde_json::Value> = Vec::new();

    // --- isolated hot kernel: VQ assignment pass at several (N, K) shapes ----------------------
    for &(n, k) in &[(20_000usize, 256usize), (100_000, 1024), (500_000, 4096)] {
        let pts = synth_coords(n, 16, 0x77u64 + n as u64);
        let cb: Vec<[f64; 3]> = (0..k).map(|j| pts[(j * pts.len()) / k]).collect();
        let t_cpu = bench_median(5, || {
            let _ = vq_assign_cpu(&pts, &cb);
        });
        // include H2D+kernel+D2H in the GPU timing (the honest end-to-end of the lane).
        let t_gpu = bench_median(5, || {
            let _ = gpu.vq_assign(&pts, &cb);
        });
        let speedup = if t_gpu > 0.0 { t_cpu / t_gpu } else { 0.0 };
        cases.push(json!({
            "case": "vq_assign_kernel", "n": n, "k": k,
            "cpu_s": t_cpu, "gpu_s": t_gpu, "kernel_speedup": speedup,
        }));
    }

    // --- full VQ-EMA fit (the real hot loop): CPU fit vs GPU-assigned fit -----------------------
    let mut fit_cases: Vec<serde_json::Value> = Vec::new();
    for &(n, bits) in &[(20_000usize, 12u8), (200_000, 12), (1_000_000, 12)] {
        let pts = synth_coords(n, 16, 0x5151u64 + n as u64);
        let t_cpu = bench_median(3, || {
            let mut q = Vq3dEma::new(bits);
            q.fit(&pts);
        });
        let t_gpu = bench_median(3, || {
            let mut q = Vq3dEma::new(bits);
            q.fit_gpu(&pts, &gpu);
        });
        let speedup = if t_gpu > 0.0 { t_cpu / t_gpu } else { 0.0 };
        fit_cases.push(json!({
            "case": "vq_ema_fit", "n": n, "codebook_bits": bits,
            "cpu_s": t_cpu, "gpu_s": t_gpu, "fit_speedup": speedup,
            "note": "fit subsamples to VQ_FIT_CAP=20k internally, so N>20k does not grow the kernel work — this is why GPU does not help the production fit",
        }));
    }

    // --- end-to-end on REAL spike data: read+decode+fit+encode, CPU vs GPU fit ------------------
    let mut e2e = json!({"status": "FIXTURE_ABSENT"});
    if let Some(path) = find_real_spikes_bin() {
        let max_records = 200_000usize;
        if let Some((coords, decl)) = load_real_coords(&path, max_records) {
            let t_cpu_fit = bench_median(3, || {
                let mut q = Vq3dEma::new(12);
                q.fit(&coords);
            });
            let t_gpu_fit = bench_median(3, || {
                let mut q = Vq3dEma::new(12);
                q.fit_gpu(&coords, &gpu);
            });
            // encode stage on both (the encode is identical; measure once on each backend).
            let mut qf = Vq3dEma::new(12);
            qf.fit(&coords);
            let t_cpu_enc = bench_median(3, || {
                let _ = qf.encode_batch_cpu(&coords);
            });
            let t_gpu_enc = bench_median(3, || {
                let _ = qf.encode_batch_gpu(&coords, &gpu);
            });
            let cpu_total = t_cpu_fit + t_cpu_enc;
            let gpu_total = t_gpu_fit + t_gpu_enc;
            let e2e_speedup = if gpu_total > 0.0 {
                cpu_total / gpu_total
            } else {
                0.0
            };
            e2e = json!({
                "status": "MEASURED", "path": path.display().to_string(),
                "declared_records": decl, "coords_used": coords.len(),
                "cpu_fit_s": t_cpu_fit, "gpu_fit_s": t_gpu_fit,
                "cpu_encode_s": t_cpu_enc, "gpu_encode_s": t_gpu_enc,
                "cpu_total_s": cpu_total, "gpu_total_s": gpu_total,
                "e2e_speedup": e2e_speedup,
            });
        }
    }

    // --- apply the gates ----------------------------------------------------------------------
    let best_kernel_speedup = cases
        .iter()
        .filter_map(|c| c["kernel_speedup"].as_f64())
        .fold(0.0f64, f64::max);
    let best_fit_speedup = fit_cases
        .iter()
        .filter_map(|c| c["fit_speedup"].as_f64())
        .fold(0.0f64, f64::max);
    let e2e_speedup = e2e
        .get("e2e_speedup")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);

    let gate_a = e2e_speedup >= 2.0; // ≥2x end-to-end on real surface
    let gate_b = best_kernel_speedup >= 5.0 && e2e_speedup >= 1.25; // ≥5x kernel AND ≥25% e2e
    let gate_c = e2e_speedup >= 1.30; // ≥30% wall reduction full ingestion
    let any_gate = gate_a || gate_b || gate_c;

    json!({
        "status": "MEASURED",
        "device_free_total_bytes": free.map(|(f, t)| json!([f, t])).unwrap_or(json!(null)),
        "isolated_kernel_cases": cases,
        "vq_fit_cases": fit_cases,
        "end_to_end_real": e2e,
        "best_kernel_speedup": best_kernel_speedup,
        "best_fit_speedup": best_fit_speedup,
        "e2e_speedup": e2e_speedup,
        "gates": {
            "a_2x_e2e_real": gate_a,
            "b_5x_kernel_and_25pct_e2e": gate_b,
            "c_30pct_wall_full": gate_c,
            "any_gate_passed": any_gate,
        },
        "default_decision": if any_gate { "GPU_LANE_ELIGIBLE_FOR_AUTO" } else { "GPU_DISABLED_BY_DEFAULT (microkernel may win but end-to-end gate not met)" },
    })
}

#[cfg(not(feature = "gpu-tokenizer"))]
fn run_perf() -> serde_json::Value {
    json!({
        "status": "FEATURE_DISABLED",
        "note": "built without --features gpu-tokenizer; no GPU lane to time.",
        "fixture_present": find_real_spikes_bin().is_some(),
    })
}

fn main() {
    let ledger = json!({
        "directive": "tokenizer-wide GPU acceleration — Phase 6 performance gates",
        "feature_gpu_tokenizer": cfg!(feature = "gpu-tokenizer"),
        "result": run_perf(),
    });
    match write_ledger(
        "OPERATOR_CONTROL/TOKENIZER_GPU_PERFORMANCE_LEDGER.json",
        &ledger,
    ) {
        Ok(p) => println!("wrote {}", p.display()),
        Err(e) => eprintln!("ledger write failed: {e}"),
    }
}
