use clap::Parser;
use prismcode::binary_runtime::{ui_report, PcError};
use serde_json::Value;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

const DEFAULT_STATUS_PATH: &str = "build/status/prismcode_status.json";
const GATE_LANES: &[(&str, &str)] = &[
    ("L3 validation", "L3_VALIDATION"),
    ("L5 demo envelope", "L5_DEMO_ENVELOPE"),
    ("Vault provider", "VAULT_PROVIDER"),
    ("Release signature", "RELEASE_SIGNATURE"),
    ("CUDA linkage", "CUDA_LINKAGE"),
    ("CUDA kernel trust", "CUDA_KERNEL_TRUST"),
    ("Model runtime", "MODEL_RUNTIME"),
    ("Warnings", "WARNING_POLICY"),
    ("NANL-HD", "NANL_HD"),
    ("NAS guard", "NAS_GUARD"),
    ("CSE/HD OS boot", "CSE_HD_FIRMWARE_OS_BOOT"),
    ("CSE/HD full audit", "CSE_HD_FULL_IMPLEMENTATION_AUDIT"),
];
const RELEASE_MODES: &[&str] = &[
    "CORE_ISA",
    "AI_ASSISTED_OS",
    "GPU_ACCELERATED_CORE",
    "SECURE_PRODUCTION",
];

#[derive(Debug, Parser)]
#[command(name = "prismui", about = "PRISMCODE terminal operator/science view")]
struct Cli {
    input: Option<PathBuf>,
    #[arg(long)]
    report: Option<PathBuf>,
    #[arg(long, default_value = DEFAULT_STATUS_PATH)]
    status: PathBuf,
    #[arg(long = "open-panel")]
    open_panel: Option<String>,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), PcError> {
    let args = Cli::parse();
    let input = args.input.unwrap_or_else(|| {
        if args.open_panel.as_deref() == Some("nanl-hd") {
            first_existing_path(&[
                "release/cse_hd_reference.cpu.pcout",
                "build/demo/nanl_hd_reference.cpu.pcout",
                "build/demo/cse_hd_reference.cpu.pcout",
            ])
        } else {
            first_existing_path(&[
                "release/cse_hd_reference.cpu.pcout",
                "build/demo/cse_hd_reference.cpu.pcout",
            ])
        }
    });
    let text = render_cli_report(
        ui_report(&input, None)?,
        &args.status,
        args.open_panel.as_deref(),
    );
    if let Some(path) = args.report.as_deref() {
        write_report(path, &text)?;
    }
    print!("{text}");
    Ok(())
}

fn first_existing_path(candidates: &[&str]) -> PathBuf {
    candidates
        .iter()
        .map(PathBuf::from)
        .find(|path| path.exists())
        .unwrap_or_else(|| PathBuf::from(candidates[0]))
}

fn write_report(path: &Path, text: &str) -> Result<(), PcError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, text)?;
    Ok(())
}

fn render_cli_report(mut base: String, status_path: &Path, open_panel: Option<&str>) -> String {
    if !base.ends_with('\n') {
        base.push('\n');
    }
    base.push('\n');
    base.push_str(&dashboard_panel(status_path));
    base.push_str("\n\n");
    base.push_str(&scientific_firmware_panel_index(status_path));
    base.push_str("\n\n");
    base.push_str(&nanl_hd_panel(open_panel));
    if !base.ends_with('\n') {
        base.push('\n');
    }
    base
}

fn nanl_hd_panel(open_panel: Option<&str>) -> String {
    let audit = read_json_first(&[
        "build/audit/nanl_hd_full_implementation_matrix.json",
        "release/nanl_hd_full_implementation_matrix.json",
    ]);
    let benchmark = read_json_first(&[
        "build/benchmarks/nanl_high_density_codec_report.json",
        "release/nanl_high_density_codec_report.json",
    ]);
    let overclaim = read_json_first(&[
        "build/audit/nanl_hd_overclaim_report.json",
        "release/nanl_hd_overclaim_report.json",
    ]);
    let status = if audit
        .as_ref()
        .and_then(|value| value.get("status"))
        .and_then(Value::as_str)
        == Some("PASS")
        && benchmark
            .as_ref()
            .and_then(|value| value.get("status"))
            .and_then(Value::as_str)
            == Some("PASS")
        && overclaim
            .as_ref()
            .and_then(|value| value.get("status"))
            .and_then(Value::as_str)
            == Some("PASS")
    {
        "PASS"
    } else {
        "BLOCKED"
    };
    let mut lines = vec![
        "NANL-HD Stack".to_string(),
        format!("Panel requested: {}", open_panel == Some("nanl-hd")),
        format!("NANL-HD overall: {status}"),
        format!(
            "  DICT: {}",
            nanl_capability_status(audit.as_ref(), "NANL_DICT_PRISM_NATIVE")
        ),
        format!(
            "  MOTIF: {}",
            nanl_capability_status(audit.as_ref(), "NANL_MOTIF_PRISM_NATIVE")
        ),
        format!(
            "  DELTA: {}",
            nanl_capability_status(audit.as_ref(), "NANL_DELTA_FRAME_OF_REFERENCE_LOSSLESS")
        ),
        format!(
            "  VAR: {}",
            nanl_capability_status(audit.as_ref(), "NANL_VAR_VARIABLE_WIDTH")
        ),
        format!(
            "  ANS: {}",
            nanl_capability_status(audit.as_ref(), "NANL_ANS_LOSSLESS_ENTROPY_CORE")
        ),
        format!(
            "  BIN: {}",
            nanl_capability_status(audit.as_ref(), "NANL_BIN_EXECUTION")
        ),
        format!(
            "  Q4: {}",
            nanl_capability_status(audit.as_ref(), "NANL_Q4_NONTRIVIAL_CONSTRAINED_CODEC")
        ),
        format!(
            "  H16: {}",
            nanl_capability_status(audit.as_ref(), "NANL_H16_INTERNAL_CODEC")
        ),
        format!(
            "  DNA simulator: {}",
            nanl_capability_status(audit.as_ref(), "NANL_DNA_SIMULATOR_ONLY")
        ),
        format!(
            "  NRF sparse: {}",
            nanl_capability_status(audit.as_ref(), "NRF_SPARSE_EVENT_COMPRESSION")
        ),
        format!(
            "  DRF active state: {}",
            nanl_capability_status(audit.as_ref(), "DRF_ACTIVE_STATE_BOUNDARY")
        ),
        format!(
            "  compression_ratio_vs_raw: {}",
            benchmark
                .as_ref()
                .and_then(|value| {
                    value
                        .get("compression_ratio_vs_raw")
                        .or_else(|| value.pointer("/aggregate/compression_ratio_vs_raw"))
                })
                .map(Value::to_string)
                .unwrap_or_else(|| "EVIDENCE_NOT_FOUND".to_string())
        ),
        format!(
            "  roundtrip hash: {}",
            benchmark
                .as_ref()
                .and_then(|value| {
                    value
                        .get("roundtrip_hash_pass")
                        .or_else(|| value.pointer("/reports/0/roundtrip_hash_pass"))
                })
                .map(Value::to_string)
                .unwrap_or_else(|| "EVIDENCE_NOT_FOUND".to_string())
        ),
        format!(
            "  replay: {}",
            benchmark
                .as_ref()
                .and_then(|value| {
                    value
                        .get("replay_pass")
                        .or_else(|| value.pointer("/reports/0/replay_pass"))
                })
                .map(Value::to_string)
                .unwrap_or_else(|| "EVIDENCE_NOT_FOUND".to_string())
        ),
        format!(
            "  overclaim status: {}",
            overclaim
                .as_ref()
                .and_then(|value| value.get("status"))
                .and_then(Value::as_str)
                .unwrap_or("EVIDENCE_NOT_FOUND")
        ),
    ];
    if status != "PASS" {
        lines.push(
            "No green NANL-HD state is emitted while Q4/ANS/benchmark/overclaim evidence is missing."
                .to_string(),
        );
    }
    lines.join("\n")
}

fn nanl_capability_status(audit: Option<&Value>, capability_id: &str) -> String {
    audit
        .and_then(|value| value.get("rows"))
        .and_then(Value::as_array)
        .and_then(|rows| {
            rows.iter()
                .find(|row| row.get("capability_id").and_then(Value::as_str) == Some(capability_id))
        })
        .and_then(|row| row.get("status"))
        .and_then(Value::as_str)
        .unwrap_or("EVIDENCE_NOT_FOUND")
        .to_string()
}

fn scientific_firmware_panel_index(status_path: &Path) -> String {
    let status = fs::read(status_path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok());
    let lane_status = |key: &str| -> String {
        status
            .as_ref()
            .and_then(|value| find_lane(value, key))
            .and_then(|lane| lane.get("status"))
            .and_then(Value::as_str)
            .or_else(|| direct_gate_status(key))
            .unwrap_or("EVIDENCE_NOT_FOUND")
            .to_string()
    };
    let panels = [
        ("Firmware Profile", lane_status("CSE_HD_FIRMWARE_OS_BOOT")),
        (
            "Implementation Audit",
            lane_status("CSE_HD_FULL_IMPLEMENTATION_AUDIT"),
        ),
        ("Scientific Encoding Audit", lane_status("L3_VALIDATION")),
        ("CSE/HD Tables", "VISIBLE".to_string()),
        ("CSE Dictionaries", "VISIBLE".to_string()),
        ("Molecule Encoding", "VISIBLE".to_string()),
        ("Protein Encoding", "VISIBLE".to_string()),
        ("Water Encoding", "VISIBLE".to_string()),
        ("Timebase Encoding", "VISIBLE".to_string()),
        ("Variant Encoding", "VISIBLE".to_string()),
        ("Event Encoding", "VISIBLE".to_string()),
        ("Uncertainty Encoding", "VISIBLE".to_string()),
        ("Thermodynamics Encoding", "VISIBLE".to_string()),
        ("Teacher Output Encoding", lane_status("MODEL_AIRLOCK")),
        ("GFlowNet Trace Encoding", "VISIBLE".to_string()),
        ("NANL-BIN", "VISIBLE".to_string()),
        ("NANL-Q4", "VISIBLE".to_string()),
        ("NANL-DNA Simulator", "SIMULATOR_ONLY".to_string()),
        ("NRF Frames", "VISIBLE".to_string()),
        ("DRF Fabric", "VISIBLE".to_string()),
        ("Compression Benchmarks", lane_status("COMPRESSION_RUNTIME")),
        ("Replay", "VISIBLE".to_string()),
        (
            "Market Readiness",
            if status
                .as_ref()
                .and_then(|value| value.pointer("/release_modes/CORE_ISA/blockers"))
                .and_then(Value::as_array)
                .map(|blockers| blockers.is_empty())
                .unwrap_or(false)
            {
                "PASS".to_string()
            } else {
                "BLOCKED".to_string()
            },
        ),
    ];
    let mut lines = vec![
        "Scientific/firmware gates:".to_string(),
        "No green overall state is emitted while market-ready gates are blocked.".to_string(),
    ];
    for (panel, panel_status) in panels {
        lines.push(format!("  {panel}: {panel_status}"));
    }
    lines.join("\n")
}

fn direct_gate_status(key: &str) -> Option<&'static str> {
    let path = match key {
        "CSE_HD_FIRMWARE_OS_BOOT" => "build/os/prism_os_boot_report.json",
        "CSE_HD_FULL_IMPLEMENTATION_AUDIT" => "build/audit/cse_hd_full_implementation_matrix.json",
        "L3_VALIDATION" => "build/demo/cse_hd_reference_l3.json",
        "MODEL_AIRLOCK" => "build/demo/model_airlock_report.json",
        "COMPRESSION_RUNTIME" => "build/audit/cse_hd_benchmark_coverage_report.json",
        _ => return None,
    };
    let bytes = fs::read(path).ok()?;
    let value = serde_json::from_slice::<Value>(&bytes).ok()?;
    match value.get("status").and_then(Value::as_str) {
        Some("PASS") => Some("PASS"),
        Some("SIMULATOR_ONLY") => Some("SIMULATOR_ONLY"),
        Some("FAIL_CLOSED") => Some("FAIL_CLOSED"),
        Some("FAIL") => Some("FAIL"),
        Some("BLOCKED") => Some("BLOCKED"),
        Some(_) => Some("UNKNOWN"),
        None => None,
    }
}

fn dashboard_panel(status_path: &Path) -> String {
    let bytes = match fs::read(status_path) {
        Ok(bytes) => bytes,
        Err(err) => {
            return missing_or_invalid_status_panel(
                status_path,
                &format!("EVIDENCE_NOT_FOUND ({err})"),
            )
        }
    };
    let status: Value = match serde_json::from_slice(&bytes) {
        Ok(status) => status,
        Err(err) => {
            return missing_or_invalid_status_panel(status_path, &format!("INVALID_JSON ({err})"))
        }
    };

    let schema = status
        .get("status_schema_version")
        .and_then(Value::as_str)
        .unwrap_or("UNKNOWN_SCHEMA");
    let blocking_lanes = blocking_lanes(&status);
    let blockers = collect_blockers(&status);
    let overall = if blocking_lanes.is_empty() && blockers.is_empty() {
        "PASS"
    } else {
        "BLOCKED"
    };

    let mut lines = vec![
        "Operational Dashboard".to_string(),
        format!("Status source: {} ({schema})", status_path.display()),
        format!("Overall readiness: {overall}"),
    ];
    if !blocking_lanes.is_empty() {
        lines.push(format!("Blocked lanes: {}", blocking_lanes.join(", ")));
    }

    lines.push("Validation/runtime gates:".to_string());
    for (label, key) in GATE_LANES {
        lines.push(format!("  {}", lane_line(&status, label, key)));
    }
    lines.push("Security/vault gates:".to_string());
    lines.extend(security_vault_lines());
    lines.push("Warning gates:".to_string());
    lines.extend(warning_policy_lines());

    lines.push("Market blockers:".to_string());
    for mode in RELEASE_MODES {
        lines.push(format!("  {}", release_mode_line(&status, mode)));
    }

    let commands = remediation_commands(&status, &blockers);
    if !commands.is_empty() {
        lines.push("Remediation commands:".to_string());
        for command in commands {
            lines.push(format!("  {command}"));
        }
    }

    let external = external_blockers(&blockers);
    if !external.is_empty() {
        lines.push(format!(
            "External blockers without repo-local remediation command: {}",
            external.join(", ")
        ));
    }

    lines.join("\n")
}

fn warning_policy_lines() -> Vec<String> {
    let report = read_json("build/release/warning_policy_report.json");
    vec![
        format!(
            "  Warnings: {}",
            json_str(report.as_ref(), "/status").unwrap_or("EVIDENCE_NOT_FOUND")
        ),
        format!(
            "  Shipped crates warnings: {}",
            if json_bool(report.as_ref(), "/shipped_crates_warnings_zero") {
                "ZERO"
            } else {
                "PRESENT_OR_UNKNOWN"
            }
        ),
        format!(
            "  xtask warnings: {}",
            if json_bool(report.as_ref(), "/xtask_warnings_zero_or_formally_waived") {
                "ZERO_OR_WAIVED_NON_SHIPPED"
            } else {
                "UNWAIVED_OR_UNKNOWN"
            }
        ),
        format!(
            "  waiver count: {}",
            json_u64(report.as_ref(), "/waiver_count")
                .map(|value| value.to_string())
                .unwrap_or_else(|| "EVIDENCE_NOT_FOUND".to_string())
        ),
        "  waiver expiry conditions: required by policy".to_string(),
        "  market impact: shipped crates must be warning-free".to_string(),
        format!(
            "  Blocker 05: {}",
            if json_str(report.as_ref(), "/status") == Some("PASS")
                && json_bool(report.as_ref(), "/shipped_crates_warnings_zero")
            {
                "PASS"
            } else {
                "BLOCKED"
            }
        ),
    ]
}

fn security_vault_lines() -> Vec<String> {
    let provider = read_json("build/security/key_provider_report.json");
    let wrap = read_json("build/security/vault_wrap_test_report.json");
    let rotation = read_json("build/security/vault_rotation_report.json");
    let sign = read_json("build/security/vault_sign_test_report.json");
    let cuda_sig = read_json("build/security/cuda_manifest_signature_report.json");
    let cuda_verify = read_json("build/security/cuda_manifest_verify_report.json");
    vec![
        format!(
            "  Vault provider state: {}",
            json_str(provider.as_ref(), "/provider_state").unwrap_or("EVIDENCE_NOT_FOUND")
        ),
        format!(
            "  Release key policy status: {}",
            json_pass_fail_text(provider.as_ref(), "/release_key_policy_bound")
        ),
        format!(
            "  Raw key exported: {}",
            json_bool_text(provider.as_ref(), "/raw_key_exported")
        ),
        format!(
            "  Wrap/unwrap status: {}/{}",
            json_str(wrap.as_ref(), "/key_wrap").unwrap_or("EVIDENCE_NOT_FOUND"),
            json_str(wrap.as_ref(), "/key_unwrap").unwrap_or("EVIDENCE_NOT_FOUND")
        ),
        format!(
            "  Sign/verify status: {}",
            json_str(sign.as_ref(), "/sign_verify").unwrap_or("EVIDENCE_NOT_FOUND")
        ),
        format!(
            "  Rotation status: {}",
            json_str(rotation.as_ref(), "/rotation_test").unwrap_or("EVIDENCE_NOT_FOUND")
        ),
        format!(
            "  CUDA manifest signature status: {}/{}",
            json_str(cuda_sig.as_ref(), "/signature_status").unwrap_or("EVIDENCE_NOT_FOUND"),
            json_str(cuda_verify.as_ref(), "/signature_status").unwrap_or("EVIDENCE_NOT_FOUND")
        ),
        format!(
            "  FILE_KEYRING_DEV market allowed: {}",
            json_bool_text(provider.as_ref(), "/file_keyring_dev_for_market")
        ),
        format!(
            "  Blocker 02 status: {}",
            read_json("build/security/vault_blocker_02_report.json")
                .as_ref()
                .and_then(|value| value.get("status"))
                .and_then(Value::as_str)
                .unwrap_or("EVIDENCE_NOT_FOUND")
        ),
    ]
}

fn read_json(path: &str) -> Option<Value> {
    fs::read(path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
}

fn read_json_first(paths: &[&str]) -> Option<Value> {
    paths.iter().find_map(|path| read_json(path))
}

fn json_str<'a>(value: Option<&'a Value>, pointer: &str) -> Option<&'a str> {
    value?.pointer(pointer)?.as_str()
}

fn json_bool(value: Option<&Value>, pointer: &str) -> bool {
    value
        .and_then(|value| value.pointer(pointer))
        .and_then(Value::as_bool)
        == Some(true)
}

fn json_u64(value: Option<&Value>, pointer: &str) -> Option<u64> {
    value?.pointer(pointer)?.as_u64()
}

fn json_bool_text(value: Option<&Value>, pointer: &str) -> &'static str {
    match value
        .and_then(|value| value.pointer(pointer))
        .and_then(Value::as_bool)
    {
        Some(true) => "true",
        Some(false) => "false",
        None => "EVIDENCE_NOT_FOUND",
    }
}

fn json_value_text(value: Option<&Value>, pointer: &str) -> String {
    match value.and_then(|value| value.pointer(pointer)) {
        Some(Value::Bool(true)) => "true".to_string(),
        Some(Value::Bool(false)) => "false".to_string(),
        Some(Value::String(text)) => text.clone(),
        Some(other) => other.to_string(),
        None => "EVIDENCE_NOT_FOUND".to_string(),
    }
}

fn json_pass_fail_text(value: Option<&Value>, pointer: &str) -> String {
    match value
        .and_then(|value| value.pointer(pointer))
        .and_then(Value::as_bool)
    {
        Some(true) => "PASS".to_string(),
        Some(false) => "FAIL".to_string(),
        None => json_value_text(value, pointer),
    }
}

fn missing_or_invalid_status_panel(status_path: &Path, source_status: &str) -> String {
    [
        "Operational Dashboard".to_string(),
        format!("Status source: {} {source_status}", status_path.display()),
        "Overall readiness: BLOCKED".to_string(),
        "Blocked lanes: STATUS_SOURCE".to_string(),
        "Remediation commands:".to_string(),
        "  cargo xtask status".to_string(),
    ]
    .join("\n")
}

fn blocking_lanes(status: &Value) -> Vec<String> {
    let mut blocking = status
        .get("lanes")
        .and_then(Value::as_array)
        .map(|lanes| {
            lanes
                .iter()
                .filter_map(|lane| {
                    let key = lane.get("key").and_then(Value::as_str)?;
                    let lane_status = lane
                        .get("status")
                        .and_then(Value::as_str)
                        .unwrap_or("BLOCKED");
                    let blockers = blockers_from_value(lane);
                    if lane_status == "PASS" && blockers.is_empty() {
                        None
                    } else {
                        Some(format!("{key}={lane_status}"))
                    }
                })
                .collect()
        })
        .unwrap_or_else(|| vec!["STATUS_LANES_MISSING=BLOCKED".to_string()]);
    for (_, key) in GATE_LANES {
        if find_lane(status, key).is_none() {
            push_unique(&mut blocking, &format!("{key}=MISSING"));
        }
    }
    if status
        .get("release_modes")
        .and_then(Value::as_object)
        .is_none()
    {
        push_unique(&mut blocking, "RELEASE_MODES=MISSING");
    }
    blocking
}

fn lane_line(status: &Value, label: &str, key: &str) -> String {
    let Some(lane) = find_lane(status, key) else {
        return format!(
            "{label}: BLOCKED blockers=STATUS_LANE_MISSING evidence=EVIDENCE_NOT_FOUND"
        );
    };
    let lane_status = lane
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("BLOCKED");
    let evidence = lane
        .get("evidence")
        .and_then(Value::as_str)
        .unwrap_or("EVIDENCE_NOT_FOUND");
    format!(
        "{label}: {lane_status} blockers={} evidence={evidence}",
        blocker_text(&blockers_from_value(lane))
    )
}

fn release_mode_line(status: &Value, mode: &str) -> String {
    let blockers = status
        .get("release_modes")
        .and_then(|modes| modes.get(mode))
        .map(blockers_from_value)
        .unwrap_or_else(|| vec!["RELEASE_MODE_STATUS_MISSING".to_string()]);
    format!("{mode}: {}", blocker_text(&blockers))
}

fn find_lane<'a>(status: &'a Value, key: &str) -> Option<&'a Value> {
    status
        .get("lanes")
        .and_then(Value::as_array)?
        .iter()
        .find(|lane| lane.get("key").and_then(Value::as_str) == Some(key))
}

fn blockers_from_value(value: &Value) -> Vec<String> {
    value
        .get("blockers")
        .and_then(Value::as_array)
        .map(|blockers| {
            blockers
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn blocker_text(blockers: &[String]) -> String {
    if blockers.is_empty() {
        "none".to_string()
    } else {
        blockers.join(", ")
    }
}

fn collect_blockers(status: &Value) -> BTreeSet<String> {
    let mut blockers = BTreeSet::new();
    if let Some(lanes) = status.get("lanes").and_then(Value::as_array) {
        for lane in lanes {
            blockers.extend(blockers_from_value(lane));
        }
    } else {
        blockers.insert("STATUS_LANES_MISSING".to_string());
    }
    if let Some(modes) = status.get("release_modes").and_then(Value::as_object) {
        for mode in RELEASE_MODES {
            if !modes.contains_key(*mode) {
                blockers.insert(format!("{mode}_STATUS_MISSING"));
            }
        }
        for mode in modes.values() {
            blockers.extend(blockers_from_value(mode));
        }
    } else {
        blockers.insert("RELEASE_MODES_MISSING".to_string());
    }
    blockers
}

fn remediation_commands(status: &Value, blockers: &BTreeSet<String>) -> Vec<String> {
    let mut commands = Vec::new();
    if !blockers.is_empty() {
        push_unique(&mut commands, "cargo xtask status");
    }
    for (blocker, command) in [
        ("L3_PACKET_MISSING", "cargo xtask demo"),
        ("L5_DEMO_ENVELOPE_NOT_VERIFIED", "cargo xtask secure-release-check"),
        (
            "CUDA_RUNTIME_LINKED",
            "cargo build -p prismcode --features cuda --bin prismhw --bin prismrun",
        ),
        (
            "CUDA_TENSOR_COPY_PARITY",
            "cargo test -p prismcode --test cuda_backend",
        ),
        (
            "MODEL_RUNTIME_ACTIVE",
            "cargo run -p prismcode --bin prismsec -- model-runtime-smoke --out build/demo/model_runtime_smoke_report.json",
        ),
        (
            "MODEL_RUNTIME_SMOKE_PASSED",
            "cargo run -p prismcode --bin prismsec -- model-runtime-smoke --out build/demo/model_runtime_smoke_report.json",
        ),
        (
            "MODEL_RUNTIME_HASH_BOUND",
            "cargo run -p prismcode --bin prismsec -- model-inspect --out build/demo/model_inspection_report.json",
        ),
        (
            "MODEL_RUNTIME_SMOKE_REF_BOUND",
            "cargo run -p prismcode --bin prismsec -- model-inspect --out build/demo/model_inspection_report.json",
        ),
        (
            "MODEL_ARTIFACT_VERIFIED",
            "cargo run -p prismcode --bin prismsec -- model-inspect --out build/demo/model_inspection_report.json",
        ),
        (
            "MODEL_AIRLOCK_VERIFIED",
            "cargo run -p prismcode --bin prismsec -- model-airlock-verify --out build/demo/model_airlock_report.json",
        ),
        (
            "COMPRESSION_RUNTIME_CODECS",
            "cargo test -p prismcode --test compression_abi",
        ),
        (
            "BENCHMARK_CORPUS_EXPANDED",
            "cargo run -p prismcode --bin prismbench -- suite --backend auto --out build/demo/benchmark_suite/benchmark_suite_report.json",
        ),
        (
            "PRISMUI_TRUTH_SOURCE",
            "cargo run -p prismcode --bin prismui -- build/demo/reference.cpu.pcout --report build/demo/human_report.md",
        ),
        ("NAS_GUARD_NOT_VERIFIED", "cargo xtask nas-guard"),
        (
            "CSE_HD_OS_BOOT_NOT_VERIFIED",
            "cargo xtask prism-os-boot",
        ),
        (
            "CSE_HD_SCIENTIFIC_PANEL_NOT_VISIBLE",
            "cargo xtask prism-os-boot",
        ),
        (
            "CSE_HD_FULL_IMPLEMENTATION_AUDIT_NOT_VERIFIED",
            "cargo xtask cse-hd-full-implementation-audit",
        ),
    ] {
        if blockers.contains(blocker) {
            push_unique(&mut commands, command);
        }
    }
    for mode in RELEASE_MODES {
        let mode_blocked = status
            .get("release_modes")
            .and_then(|modes| modes.get(*mode))
            .map(|mode_status| !blockers_from_value(mode_status).is_empty())
            .unwrap_or(false);
        if mode_blocked {
            push_unique(
                &mut commands,
                &format!("cargo xtask market-ready-check --mode {mode}"),
            );
        }
    }
    commands
}

fn push_unique(commands: &mut Vec<String>, command: &str) {
    if !commands.iter().any(|existing| existing == command) {
        commands.push(command.to_string());
    }
}

fn external_blockers(blockers: &BTreeSet<String>) -> Vec<String> {
    blockers
        .iter()
        .filter(|blocker| {
            matches!(
                blocker.as_str(),
                "OPENBAO_OR_VAULT_BOUND"
                    | "TRUSTED_ATTESTATION"
                    | "MARKET_READY_TRUE"
                    | "CUDA_KERNEL_TRUSTED"
                    | "SBOM_BOUND"
                    | "SLSA_PROVENANCE_BOUND"
            )
        })
        .cloned()
        .collect()
}
