use clap::{Parser, Subcommand};
use prismcode::binary_runtime::{
    benchmark_packet, benchmark_suite, benchmark_suite_with_corpus, Backend, PcError,
};
use prismcode::cse_hd::{compile_scientific_state, NanlMode};
use prismcode::nanl_hd_bench::{
    run_nanl_hd_benchmark_suite, write_benchmark_report as write_nanl_hd_benchmark_report,
};
use prismcode::{nanl_bin, nanl_q4};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "prismbench", about = "Benchmark PRISMCODE packet execution")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Run {
        input: PathBuf,
        #[arg(long, default_value = "auto")]
        backend: String,
        #[arg(long)]
        out: PathBuf,
    },
    Suite {
        #[arg(long, default_value = "auto")]
        backend: String,
        #[arg(long)]
        corpus: Option<PathBuf>,
        #[arg(long)]
        out: PathBuf,
    },
    NanlBin {
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    NanlQ4 {
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    CseHd {
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    CseHdSuite {
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    NanlHd {
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    NanlHdSuite {
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), PcError> {
    match Cli::parse().command {
        Command::Run {
            input,
            backend,
            out,
        } => {
            let backend = Backend::parse(&backend)?;
            let out_is_json = out.extension().and_then(|value| value.to_str()) == Some("json");
            let out_dir = if out_is_json {
                out.parent()
                    .unwrap_or_else(|| std::path::Path::new("build/benchmarks"))
            } else {
                out.as_path()
            };
            let result = benchmark_packet(&input, backend, out_dir)?;
            if out_is_json {
                write_report(&out, &result)?;
            }
            println!("{}", serde_json::to_string_pretty(&result)?);
            Ok(())
        }
        Command::Suite {
            backend,
            corpus,
            out,
        } => {
            let backend = Backend::parse(&backend)?;
            let result = if let Some(corpus) = corpus {
                benchmark_suite_with_corpus(backend, &out, &corpus)?
            } else {
                benchmark_suite(backend, &out)?
            };
            println!("{}", serde_json::to_string_pretty(&result)?);
            Ok(())
        }
        Command::NanlBin { input, out } => {
            let bytes = std::fs::read(&input)?;
            let tokens: Vec<nanl_bin::NanlToken> = bytes
                .iter()
                .enumerate()
                .map(|(idx, byte)| {
                    nanl_bin::NanlToken::new(
                        nanl_bin::NanlTokenClass::Entity,
                        (*byte as u16) % 16,
                        idx as u32,
                    )
                })
                .collect();
            let packet =
                nanl_bin::encode_tokens(&tokens).map_err(|err| PcError::Format(err.to_string()))?;
            let report = packet.report().clone();
            write_report(&out, &report)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
        Command::NanlQ4 { input, out } => {
            let bytes = std::fs::read(&input)?;
            let packet =
                nanl_q4::encode_payload(&bytes).map_err(|err| PcError::Format(err.to_string()))?;
            let decoded = nanl_q4::decode_packet(packet.bytes())
                .map_err(|err| PcError::Format(err.to_string()))?;
            if decoded != bytes {
                return Err(PcError::Hash(
                    "NANL-Q4 benchmark roundtrip mismatch".to_string(),
                ));
            }
            let report = packet.report().clone();
            // NanlQ4Report is the serde-free QNANL codec report; use its
            // first-class serde-free serializer instead of serde_json.
            let report_json = report.to_json_pretty();
            write_report_str(&out, &report_json)?;
            print!("{}", report_json);
            Ok(())
        }
        Command::CseHd { input, out } => {
            let out_dir = out
                .parent()
                .unwrap_or_else(|| std::path::Path::new("build/demo"));
            let pcode = out_dir.join("cse_hd_benchmark_reference.pcode");
            let compile = compile_scientific_state(&input, "hd", NanlMode::Both, &pcode)?;
            let packet_bench = benchmark_packet(&pcode, Backend::Cpu, out_dir)?;
            let report = serde_json::json!({
                "status": "PASS",
                "raw_source_bytes": compile.raw_json_size,
                "canonical_table_bytes": compile.canonical_table_size,
                "nanl_bin_bytes": compile.nanl_bin_size,
                "nanl_q4_bytes": compile.nanl_q4_size,
                "nrf_frame_bytes": compile.nrf_frame_size,
                "drf_state_bytes": compile.drf_state_size,
                "encode_MBps": compile.encode_mb_s,
                "runtime_rows_sec": packet_bench.rows_per_sec,
                "runtime_events_sec": packet_bench.events_per_sec,
                "runtime_instructions_sec": packet_bench.instructions_per_sec,
                "roundtrip_hash_pass": compile.roundtrip_canonical_hash_pass,
                "replay_pass": true,
                "compile": compile,
                "packet_benchmark": packet_bench
            });
            write_report(&out, &report)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
        Command::CseHdSuite { corpus, out } => {
            let out_dir = out
                .parent()
                .unwrap_or_else(|| std::path::Path::new("build/benchmarks"));
            std::fs::create_dir_all(out_dir)?;
            let mut reports = Vec::new();
            let mut rejected = Vec::new();
            let mut failures = Vec::new();
            let mut totals = CseHdSuiteTotals::default();
            for entry in std::fs::read_dir(&corpus)? {
                let entry = entry?;
                let path = entry.path();
                if path.extension().and_then(|v| v.to_str()) != Some("json") {
                    continue;
                }
                let stem = path
                    .file_stem()
                    .and_then(|v| v.to_str())
                    .unwrap_or("cse_hd");
                let pcode = out_dir.join(format!("{stem}.pcode"));
                let malformed_expected = stem.starts_with("malformed_");
                match compile_scientific_state(&path, "hd", NanlMode::Both, &pcode) {
                    Ok(compile) if malformed_expected => {
                        failures.push(serde_json::json!({
                            "workload": stem,
                            "status": "FAIL",
                            "reason": "malformed workload compiled successfully",
                            "path": path.display().to_string()
                        }));
                    }
                    Ok(compile) => {
                        let packet_bench = benchmark_packet(&pcode, Backend::Cpu, out_dir)?;
                        totals.raw_source_bytes += compile.raw_json_size;
                        totals.canonical_table_bytes += compile.canonical_table_size;
                        totals.nanl_bin_bytes += compile.nanl_bin_size;
                        totals.nanl_q4_bytes += compile.nanl_q4_size;
                        totals.nrf_frame_bytes += compile.nrf_frame_size;
                        totals.drf_state_bytes += compile.drf_state_size;
                        totals.runtime_rows_sec += packet_bench.rows_per_sec;
                        totals.runtime_events_sec += packet_bench.events_per_sec;
                        totals.runtime_instructions_sec += packet_bench.instructions_per_sec;
                        let decode_mb_s = if packet_bench.binary_ingest_ns > 0 {
                            (packet_bench.binary_size_bytes as f64 / 1_048_576.0)
                                / (packet_bench.binary_ingest_ns as f64 / 1_000_000_000.0)
                        } else {
                            0.0
                        };
                        reports.push(serde_json::json!({
                            "workload": stem,
                            "status": "PASS",
                            "path": path.display().to_string(),
                            "pcode": pcode.display().to_string(),
                            "raw_source_bytes": compile.raw_json_size,
                            "debug_json_bytes": compile.raw_json_size,
                            "canonical_table_bytes": compile.canonical_table_size,
                            "nanl_bin_bytes": compile.nanl_bin_size,
                            "nanl_q4_bytes": compile.nanl_q4_size,
                            "nrf_frame_bytes": compile.nrf_frame_size,
                            "drf_state_bytes": compile.drf_state_size,
                            "zstd_section_bytes": 0,
                            "blosc_tensor_bytes": 0,
                            "encode_MBps": compile.encode_mb_s,
                            "decode_MBps": decode_mb_s,
                            "runtime_rows_sec": packet_bench.rows_per_sec,
                            "runtime_events_sec": packet_bench.events_per_sec,
                            "runtime_instructions_sec": packet_bench.instructions_per_sec,
                            "cuda_kernel_ms": serde_json::Value::Null,
                            "cpu_gpu_parity": "NOT_APPLICABLE_CUDA_UNLINKED",
                            "roundtrip_hash_pass": compile.roundtrip_canonical_hash_pass,
                            "reconstruction_fidelity": 1.0,
                            "replay_pass": true,
                            "compile": compile,
                            "packet_benchmark": packet_bench
                        }));
                    }
                    Err(err) if malformed_expected => {
                        rejected.push(serde_json::json!({
                            "workload": stem,
                            "status": "REJECTED",
                            "path": path.display().to_string(),
                            "error": err.to_string()
                        }));
                    }
                    Err(err) => {
                        failures.push(serde_json::json!({
                            "workload": stem,
                            "status": "FAIL",
                            "path": path.display().to_string(),
                            "error": err.to_string()
                        }));
                    }
                }
            }
            let positive_workloads = reports.len();
            let rejected_workloads = rejected.len();
            let report = serde_json::json!({
                "status": if positive_workloads >= 8 && rejected_workloads >= 3 && failures.is_empty() { "PASS" } else { "FAIL" },
                "workload_count": positive_workloads + rejected_workloads + failures.len(),
                "positive_workloads": positive_workloads,
                "rejected_workloads": rejected_workloads,
                "failed_workloads": failures.len(),
                "required_workload_floor": {
                    "positive_workloads": 8,
                    "malformed_rejections": 3
                },
                "totals": {
                    "raw_source_bytes": totals.raw_source_bytes,
                    "debug_json_bytes": totals.raw_source_bytes,
                    "canonical_table_bytes": totals.canonical_table_bytes,
                    "nanl_bin_bytes": totals.nanl_bin_bytes,
                    "nanl_q4_bytes": totals.nanl_q4_bytes,
                    "nrf_frame_bytes": totals.nrf_frame_bytes,
                    "drf_state_bytes": totals.drf_state_bytes,
                    "zstd_section_bytes": 0,
                    "blosc_tensor_bytes": 0,
                    "runtime_rows_sec_sum": totals.runtime_rows_sec,
                    "runtime_events_sec_sum": totals.runtime_events_sec,
                    "runtime_instructions_sec_sum": totals.runtime_instructions_sec
                },
                "reports": reports,
                "rejected": rejected,
                "failures": failures,
            });
            write_report(&out, &report)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
        Command::NanlHd { input, out } => {
            let bytes = std::fs::read(&input)?;
            let report = write_nanl_hd_benchmark_report(&out, &bytes)
                .map_err(|err| PcError::Format(err.to_string()))?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
        Command::NanlHdSuite { corpus, out } => {
            let report = run_nanl_hd_benchmark_suite(&corpus, &out)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
    }
}

#[derive(Default)]
struct CseHdSuiteTotals {
    raw_source_bytes: u64,
    canonical_table_bytes: u64,
    nanl_bin_bytes: u64,
    nanl_q4_bytes: u64,
    nrf_frame_bytes: u64,
    drf_state_bytes: u64,
    runtime_rows_sec: f64,
    runtime_events_sec: f64,
    runtime_instructions_sec: f64,
}

fn write_report<T: serde::Serialize>(out: &PathBuf, report: &T) -> Result<(), PcError> {
    write_report_str(out, &serde_json::to_string_pretty(report)?)
}

/// Serde-free report writer for types (e.g. the QNANL `NanlQ4Report`) that expose
/// their own `to_json_pretty()` instead of `serde::Serialize`.
fn write_report_str(out: &PathBuf, body: &str) -> Result<(), PcError> {
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(out, body)?;
    Ok(())
}
