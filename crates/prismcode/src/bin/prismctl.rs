use clap::{Parser, Subcommand};
use prismcode::binary_runtime::{PcError, PcResult};
use prismcode::manifest_signature::verify_artifact_report;
use prismcode::trust_root;
use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Parser)]
#[command(
    name = "prismctl",
    about = "Installed PRISM OS control and release verification"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Status {
        #[arg(long, default_value = ".")]
        bundle: PathBuf,
    },
    Audit {
        #[arg(long, default_value = ".")]
        bundle: PathBuf,
    },
    MarketReady {
        #[arg(long, default_value = ".")]
        bundle: PathBuf,
        #[arg(long, default_value = "FULL_PRISM_OS")]
        mode: String,
    },
    VerifySbom {
        #[arg(long, default_value = ".")]
        bundle: PathBuf,
    },
    VerifyProvenance {
        #[arg(long, default_value = ".")]
        bundle: PathBuf,
    },
    VerifyEvidenceLedger {
        #[arg(long, default_value = ".")]
        bundle: PathBuf,
    },
    VerifyWarnings {
        #[arg(long, default_value = ".")]
        bundle: PathBuf,
    },
    VerifyNanlHd {
        #[arg(long, default_value = ".")]
        bundle: PathBuf,
    },
    PinnedRoot,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(2);
    }
}

fn run() -> PcResult<()> {
    match Cli::parse().command {
        Command::Status { bundle } => print_report(installed_status(&bundle)?),
        Command::Audit { bundle } => print_report(installed_audit(&bundle)?),
        Command::MarketReady { bundle, mode } => {
            let report = market_ready(&bundle, &mode)?;
            print_report(report)
        }
        Command::VerifySbom { bundle } => print_report(verify_sbom(&bundle)?),
        Command::VerifyProvenance { bundle } => print_report(verify_provenance(&bundle)?),
        Command::VerifyEvidenceLedger { bundle } => print_report(verify_evidence_ledger(&bundle)?),
        Command::VerifyWarnings { bundle } => print_report(verify_warnings(&bundle)?),
        Command::VerifyNanlHd { bundle } => print_report(verify_nanl_hd(&bundle)?),
        Command::PinnedRoot => print_report(trust_root::report()),
    }
}

fn print_report(report: Value) -> PcResult<()> {
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn installed_status(bundle: &Path) -> PcResult<Value> {
    let root = fs::canonicalize(bundle)?;
    let manifest = read_json(&root.join("release/release_manifest.json"))?;
    Ok(json!({
        "status": "PASS",
        "bundle_root": root.display().to_string(),
        "release_manifest_hash_blake3": file_hash_hex(&root.join("release/release_manifest.json"))?,
        "manifest_status": manifest.get("status").cloned().unwrap_or_else(|| json!("UNKNOWN")),
    }))
}

fn installed_audit(bundle: &Path) -> PcResult<Value> {
    let root = fs::canonicalize(bundle)?;
    let sbom = verify_sbom(&root)?;
    let provenance = verify_provenance(&root)?;
    let evidence = verify_evidence_ledger(&root)?;
    let status = if all_pass([&sbom, &provenance, &evidence]) {
        "PASS"
    } else {
        "FAIL"
    };
    Ok(json!({
        "status": status,
        "bundle_root": root.display().to_string(),
        "sbom": sbom,
        "provenance": provenance,
        "evidence_ledger": evidence,
    }))
}

fn verify_sbom(bundle: &Path) -> PcResult<Value> {
    trust_root::require_release_pinned_root()?;
    let root = fs::canonicalize(bundle)?;
    let manifest = read_json(&root.join("release/release_manifest.json"))?;
    let sbom = root.join("release/sbom.json");
    require_nonempty(&sbom)?;
    verify_manifest_hash(&manifest, "sbom_hash_blake3", &sbom)?;
    Ok(json!({
        "status": "PASS",
        "check_id": "SBOM_BOUND",
        "bundle_root": root.display().to_string(),
        "sbom": sbom.display().to_string(),
        "sbom_hash_blake3": file_hash_hex(&sbom)?,
    }))
}

fn verify_provenance(bundle: &Path) -> PcResult<Value> {
    trust_root::require_release_pinned_root()?;
    let root = fs::canonicalize(bundle)?;
    let manifest_path = root.join("release/release_manifest.json");
    let manifest = read_json(&manifest_path)?;
    let provenance = root.join("release/release_provenance.json");
    require_nonempty(&provenance)?;
    verify_manifest_hash(&manifest, "provenance_hash_blake3", &provenance)?;
    verify_manifest_hash(
        &manifest,
        "cuda_manifest_signature_hash_blake3",
        &root.join("release/cuda_manifest.sig"),
    )?;
    let artifacts = manifest
        .get("artifacts")
        .and_then(Value::as_array)
        .ok_or_else(|| PcError::Format("release manifest missing artifacts array".to_string()))?;
    let mut checked = Vec::new();
    for artifact in artifacts {
        let rel = artifact
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| PcError::Format("artifact missing path".to_string()))?;
        let expected = artifact
            .get("hash_blake3")
            .and_then(Value::as_str)
            .ok_or_else(|| PcError::Format("artifact missing hash_blake3".to_string()))?;
        let path = root.join(rel);
        require_nonempty(&path)?;
        let actual = file_hash_hex(&path)?;
        if actual != expected {
            return Err(PcError::Hash(format!(
                "artifact hash mismatch for {rel}: expected {expected} actual {actual}"
            )));
        }
        checked.push(json!({"path": rel, "hash_blake3": actual, "status": "PASS"}));
    }
    Ok(json!({
        "status": "PASS",
        "check_id": "PROVENANCE_BOUND",
        "bundle_root": root.display().to_string(),
        "release_manifest": manifest_path.display().to_string(),
        "provenance_hash_blake3": file_hash_hex(&provenance)?,
        "artifact_count": checked.len(),
        "artifacts": checked,
    }))
}

fn verify_evidence_ledger(bundle: &Path) -> PcResult<Value> {
    trust_root::require_release_pinned_root()?;
    let root = fs::canonicalize(bundle)?;
    let manifest = read_json(&root.join("release/release_manifest.json"))?;
    let ledger = root.join("release/evidence_ledger.json");
    let sig = root.join("release/evidence_ledger.sig");
    require_nonempty(&ledger)?;
    require_nonempty(&sig)?;
    verify_manifest_hash(&manifest, "evidence_ledger_hash_blake3", &ledger)?;
    let signature = verify_artifact_report(&ledger, &sig)?;
    Ok(json!({
        "status": "PASS",
        "check_id": "EVIDENCE_LEDGER_SIGNED",
        "bundle_root": root.display().to_string(),
        "evidence_ledger_hash_blake3": file_hash_hex(&ledger)?,
        "signature": signature,
    }))
}

fn verify_warnings(bundle: &Path) -> PcResult<Value> {
    trust_root::require_release_pinned_root()?;
    let root = fs::canonicalize(bundle)?;
    let manifest = read_json(&root.join("release/release_manifest.json"))?;
    let report_path = root.join("release/warning_policy_report.json");
    let policy_path = root.join("configs/release/warning_policy.toml");
    require_nonempty(&report_path)?;
    require_nonempty(&policy_path)?;
    verify_manifest_hash(&manifest, "warning_policy_report_hash_blake3", &report_path)?;
    verify_manifest_hash(&manifest, "warning_policy_hash_blake3", &policy_path)?;
    verify_manifest_artifact_hash(
        &manifest,
        "release/warning_policy_report.json",
        &report_path,
    )?;
    verify_manifest_artifact_hash(
        &manifest,
        "configs/release/warning_policy.toml",
        &policy_path,
    )?;
    let report = read_json(&report_path)?;
    let policy_hash = file_hash_hex(&policy_path)?;
    let blockers = warning_policy_blockers(&report, &policy_hash);
    let out = json!({
        "status": if blockers.is_empty() { "PASS" } else { "FAIL" },
        "check_id": "WARNING_POLICY_BOUND",
        "bundle_root": root.display().to_string(),
        "warning_policy_report": report_path.display().to_string(),
        "warning_policy_report_hash_blake3": file_hash_hex(&report_path)?,
        "warning_policy_hash_blake3": policy_hash,
        "blockers": blockers,
    });
    if out.get("status").and_then(Value::as_str) != Some("PASS") {
        return Err(PcError::Security(format!(
            "warning policy verification failed: {}",
            out["blockers"]
        )));
    }
    Ok(out)
}

fn verify_nanl_hd(bundle: &Path) -> PcResult<Value> {
    trust_root::require_release_pinned_root()?;
    let root = fs::canonicalize(bundle)?;
    let manifest = read_json(&root.join("release/release_manifest.json"))?;
    let required = [
        "release/nanl_hd_full_implementation_matrix.json",
        "release/nanl_hd_negative_validation_report.json",
        "release/nanl_hd_overclaim_report.json",
        "release/nanl_hd_evidence_report.json",
        "release/nanl_high_density_codec_report.json",
        "release/nanl_high_density_codec_report.md",
        "release/nanl_codec_stack.json",
        "release/nanl_dict.nanld",
        "release/nanl_motif.nanlm",
        "release/nanl_delta.nanldelta",
        "release/nanl_var.nanlvar",
        "release/nanl_ans.nanlans",
        "release/nanl_h16.nanlh16",
        "release/nanl_bin.nanlb",
        "release/nanl_q4.nanlq4",
        "release/nanl_dna_sim.json",
        "release/nrf_sparse.nrf",
        "release/drf_active_state.drf",
        "release/nanl_hd_prismui_report.md",
    ];
    let mut checked = Vec::new();
    for rel in required {
        let path = root.join(rel);
        require_nonempty(&path)?;
        verify_manifest_artifact_hash(&manifest, rel, &path)?;
        checked.push(json!({
            "path": rel,
            "hash_blake3": file_hash_hex(&path)?,
            "status": "PASS",
        }));
    }
    let matrix = read_json(&root.join("release/nanl_hd_full_implementation_matrix.json"))?;
    let benchmark = read_json(&root.join("release/nanl_high_density_codec_report.json"))?;
    let overclaim = read_json(&root.join("release/nanl_hd_overclaim_report.json"))?;
    let matrix_pass = matrix.get("status").and_then(Value::as_str) == Some("PASS")
        && matrix
            .get("NANL_HD_FULL_IMPLEMENTATION")
            .and_then(Value::as_str)
            == Some("PASS");
    let benchmark_pass = benchmark.get("status").and_then(Value::as_str) == Some("PASS")
        && benchmark
            .pointer("/sample_policy/all_performance_claims_have_minimum_samples")
            .and_then(Value::as_bool)
            == Some(true)
        && benchmark
            .pointer("/baseline_coverage/missing_baselines")
            .and_then(Value::as_array)
            .map(Vec::is_empty)
            == Some(true)
        && benchmark
            .get("malformed_rejections")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            >= 5;
    let overclaim_pass = overclaim.get("status").and_then(Value::as_str) == Some("PASS");
    let first_roundtrip = benchmark
        .pointer("/reports/0/roundtrip_hash_pass")
        .and_then(Value::as_bool)
        == Some(true);
    let first_replay = benchmark
        .pointer("/reports/0/replay_pass")
        .and_then(Value::as_bool)
        == Some(true);
    let q4_row = matrix
        .get("rows")
        .and_then(Value::as_array)
        .and_then(|rows| {
            rows.iter().find(|row| {
                row.get("capability_id").and_then(Value::as_str)
                    == Some("NANL_Q4_NONTRIVIAL_CONSTRAINED_CODEC")
            })
        })
        .and_then(|row| row.get("status"))
        .and_then(Value::as_str)
        == Some("PASS");
    let status = if matrix_pass
        && benchmark_pass
        && overclaim_pass
        && first_roundtrip
        && first_replay
        && q4_row
    {
        "PASS"
    } else {
        "FAIL"
    };
    let report = json!({
        "status": status,
        "check_id": "NANL_HD_INSTALLED_BUNDLE_VERIFY",
        "bundle_root": root.display().to_string(),
        "matrix_pass": matrix_pass,
        "q4_nontrivial_pass": q4_row,
        "benchmark_pass": benchmark_pass,
        "overclaim_guard_pass": overclaim_pass,
        "roundtrip_hash_pass": first_roundtrip,
        "replay_pass": first_replay,
        "artifact_count": checked.len(),
        "artifacts": checked,
    });
    if status != "PASS" {
        return Err(PcError::Security(format!(
            "NANL-HD installed verification failed: {report}"
        )));
    }
    Ok(report)
}

fn market_ready(bundle: &Path, mode: &str) -> PcResult<Value> {
    trust_root::require_release_pinned_root()?;
    let root = fs::canonicalize(bundle)?;
    let mut blockers = Vec::new();
    push_if_err(&mut blockers, "SBOM_BOUND", verify_sbom(&root));
    push_if_err(&mut blockers, "PROVENANCE_BOUND", verify_provenance(&root));
    push_if_err(
        &mut blockers,
        "EVIDENCE_LEDGER_SIGNED",
        verify_evidence_ledger(&root),
    );
    push_if_err(
        &mut blockers,
        "WARNING_POLICY_BOUND",
        verify_warnings(&root),
    );
    push_if_err(
        &mut blockers,
        "FIRMWARE_PROFILE_SIGNATURE",
        verify_firmware_hash(&root),
    );
    if let Ok(existing_market_report) = read_json(&root.join("release/market_ready_report.json")) {
        if existing_market_report
            .get("MARKET_READY_TRUE")
            .and_then(Value::as_bool)
            == Some(true)
        {
            blockers.push("MANUAL_MARKET_READY_TRUE_REJECTED".to_string());
        }
    }
    let key_provider = read_json(&root.join("release/key_provider_report.json"))?;
    if key_provider.get("provider_state").and_then(Value::as_str)
        != Some("LOCAL_RELEASE_VAULT_BOUND")
    {
        blockers.push("VAULT_PROVIDER_NOT_LOCAL_RELEASE_BOUND".to_string());
    }
    if key_provider
        .get("raw_key_exported")
        .and_then(Value::as_bool)
        != Some(false)
    {
        blockers.push("RAW_KEY_EXPORTED_TRUE".to_string());
    }
    if mode == "FULL_PRISM_OS" {
        if let Some((smoke, smoke_json)) = model_runtime_smoke_evidence(&root)? {
            let smoke_blockers = model_runtime_smoke_blockers(&smoke_json);
            if smoke_blockers.is_empty() {
                // Fresh executable acceptance smoke overrides the immutable placeholder report.
            } else {
                blockers.extend(
                    smoke_blockers
                        .into_iter()
                        .map(|blocker| format!("{blocker}@{}", smoke.display())),
                );
            }
        } else {
            blockers.push("MODEL_RUNTIME_SMOKE_MISSING".to_string());
        }
    }
    let report = json!({
        "status": if blockers.is_empty() { "PASS" } else { "FAIL" },
        "market_ready_computed": blockers.is_empty(),
        "mode": mode,
        "bundle_root": root.display().to_string(),
        "blockers": blockers,
    });
    if report.get("status").and_then(Value::as_str) != Some("PASS") {
        return Err(PcError::Security(format!(
            "market-ready blocked: {}",
            report["blockers"]
        )));
    }
    Ok(report)
}

fn model_runtime_smoke_evidence(root: &Path) -> PcResult<Option<(PathBuf, Value)>> {
    let release_smoke = root.join("release/model_runtime_smoke_report.json");
    if release_smoke.exists() {
        let value = read_json(&release_smoke)?;
        if model_runtime_smoke_blockers(&value).is_empty() {
            return Ok(Some((release_smoke, value)));
        }
    }
    let acceptance = root.join("var/acceptance");
    let mut candidates = Vec::new();
    if acceptance.is_dir() {
        for entry in fs::read_dir(acceptance)? {
            let entry = entry?;
            let path = entry.path().join("model_runtime_smoke_report.json");
            if path.exists() {
                candidates.push(path);
            }
        }
    }
    candidates.sort();
    for path in candidates.into_iter().rev() {
        let value = read_json(&path)?;
        if model_runtime_smoke_blockers(&value).is_empty() {
            return Ok(Some((path, value)));
        }
    }
    if release_smoke.exists() {
        return Ok(Some((release_smoke.clone(), read_json(&release_smoke)?)));
    }
    Ok(None)
}

fn model_runtime_smoke_blockers(smoke: &Value) -> Vec<String> {
    let mut blockers = Vec::new();
    if smoke.get("status").and_then(Value::as_str) != Some("PASS") {
        blockers.push("MODEL_RUNTIME_SMOKE_NOT_PASS".to_string());
    }
    if smoke.get("model_runtime_active").and_then(Value::as_bool) != Some(true) {
        blockers.push("MODEL_RUNTIME_ACTIVE_FALSE".to_string());
    }
    if smoke
        .get("runtime_binary_hash")
        .and_then(Value::as_str)
        .map(|value| value.trim().is_empty())
        .unwrap_or(true)
    {
        blockers.push("MODEL_RUNTIME_BINARY_HASH_MISSING".to_string());
    }
    if smoke.get("runtime_bind_address").and_then(Value::as_str) != Some("127.0.0.1") {
        blockers.push("MODEL_RUNTIME_BIND_NOT_LOCALHOST".to_string());
    }
    if smoke
        .get("model_runtime_no_external_egress")
        .and_then(Value::as_bool)
        != Some(true)
    {
        blockers.push("MODEL_RUNTIME_EXTERNAL_EGRESS_NOT_BLOCKED".to_string());
    }
    if smoke
        .get("model_runtime_no_secret_access")
        .and_then(Value::as_bool)
        != Some(true)
    {
        blockers.push("MODEL_RUNTIME_SECRET_ACCESS_NOT_BLOCKED".to_string());
    }
    if smoke
        .get("model_runtime_no_canonical_mutation")
        .and_then(Value::as_bool)
        != Some(true)
        || smoke
            .get("canonical_mutation_allowed")
            .and_then(Value::as_bool)
            != Some(false)
    {
        blockers.push("MODEL_RUNTIME_CANONICAL_MUTATION_NOT_FORBIDDEN".to_string());
    }
    if smoke.get("output_classification").and_then(Value::as_str)
        != Some("MODEL_GENERATED_UNVERIFIED")
    {
        blockers.push("MODEL_RUNTIME_OUTPUT_CLASSIFICATION_INVALID".to_string());
    }
    if missing_string(smoke, "/prompt_hash") {
        blockers.push("MODEL_RUNTIME_PROMPT_HASH_MISSING".to_string());
    }
    if missing_string(smoke, "/output_hash") {
        blockers.push("MODEL_RUNTIME_OUTPUT_HASH_MISSING".to_string());
    }
    if missing_string(smoke, "/loaded_model_digest") {
        blockers.push("MODEL_RUNTIME_LOADED_MODEL_DIGEST_MISSING".to_string());
    }
    if smoke
        .pointer("/model_artifact_proof/all_model_hashes_verified")
        .and_then(Value::as_bool)
        != Some(true)
    {
        blockers.push("MODEL_RUNTIME_MODEL_ARTIFACT_HASHES_NOT_VERIFIED".to_string());
    }
    match smoke.get("model_artifact_source").and_then(Value::as_str) {
        Some("BUNDLED_MODEL_PACK") | Some("INSTALLED_PORTABLE_MODEL_PACK") => {}
        Some(other) => blockers.push(format!("MODEL_ARTIFACT_SOURCE_NOT_PORTABLE:{other}")),
        None => blockers.push("MODEL_ARTIFACT_SOURCE_MISSING".to_string()),
    }
    if smoke
        .get("model_artifact_not_nas_dependent")
        .and_then(Value::as_bool)
        != Some(true)
    {
        blockers.push("MODEL_ARTIFACT_NOT_NAS_DEPENDENT_FAIL".to_string());
    }
    for (key, blocker) in [
        (
            "model_artifact_no_host_cache_path",
            "MODEL_RUNTIME_HOST_CACHE_PATH_USED",
        ),
        (
            "model_artifact_no_hf_cache_path",
            "MODEL_RUNTIME_HF_CACHE_PATH_USED",
        ),
        (
            "model_artifact_no_ollama_path",
            "MODEL_RUNTIME_OLLAMA_PATH_USED",
        ),
        (
            "model_artifact_no_source_tree_path",
            "MODEL_RUNTIME_SOURCE_TREE_PATH_USED",
        ),
        (
            "model_artifact_no_cargo_path",
            "MODEL_RUNTIME_CARGO_PATH_USED",
        ),
        (
            "model_load_timeout_enforced",
            "MODEL_LOAD_TIMEOUT_NOT_ENFORCED",
        ),
        (
            "model_smoke_timeout_enforced",
            "MODEL_SMOKE_TIMEOUT_NOT_ENFORCED",
        ),
        ("hung_process_reaped", "MODEL_HUNG_PROCESS_NOT_REAPED"),
    ] {
        if smoke.get(key).and_then(Value::as_bool) != Some(true) {
            blockers.push(blocker.to_string());
        }
    }
    blockers
}

fn missing_string(value: &Value, pointer: &str) -> bool {
    value
        .pointer(pointer)
        .and_then(Value::as_str)
        .map(|text| text.trim().is_empty())
        .unwrap_or(true)
}

fn warning_policy_blockers(report: &Value, policy_hash: &str) -> Vec<String> {
    let mut blockers = Vec::new();
    if report.get("status").and_then(Value::as_str) != Some("PASS") {
        blockers.push("WARNING_POLICY_REPORT_NOT_PASS".to_string());
    }
    if report.get("policy_hash_blake3").and_then(Value::as_str) != Some(policy_hash) {
        blockers.push("WARNING_POLICY_HASH_MISMATCH".to_string());
    }
    if report
        .get("shipped_crates_warnings_zero")
        .and_then(Value::as_bool)
        != Some(true)
    {
        blockers.push("SHIPPED_CRATE_WARNINGS_PRESENT".to_string());
    }
    if report
        .get("xtask_warnings_zero_or_formally_waived")
        .and_then(Value::as_bool)
        != Some(true)
    {
        blockers.push("XTASK_WARNINGS_UNWAIVED".to_string());
    }
    if report
        .get("blanket_suppression_findings")
        .and_then(Value::as_array)
        .map(|findings| !findings.is_empty())
        .unwrap_or(true)
    {
        blockers.push("BLANKET_WARNING_SUPPRESSION_FOUND".to_string());
    }
    blockers
}

fn verify_firmware_hash(root: &Path) -> PcResult<()> {
    let profile = root.join("firmware/active.prismfw.toml");
    let hash_path = root.join("firmware/active.prismfw.hash");
    let sig = root.join("firmware/active.prismfw.sig");
    require_nonempty(&profile)?;
    require_nonempty(&hash_path)?;
    require_nonempty(&sig)?;
    let expected = fs::read_to_string(hash_path)?.trim().to_string();
    let actual = file_hash_hex(&profile)?;
    if expected != actual {
        return Err(PcError::Hash("firmware profile hash mismatch".to_string()));
    }
    verify_artifact_report(&profile, &sig)?;
    Ok(())
}

fn push_if_err<T>(blockers: &mut Vec<String>, id: &str, result: PcResult<T>) {
    if let Err(err) = result {
        blockers.push(format!("{id}:{err}"));
    }
}

fn all_pass<'a>(values: impl IntoIterator<Item = &'a Value>) -> bool {
    values
        .into_iter()
        .all(|value| value.get("status").and_then(Value::as_str) == Some("PASS"))
}

fn verify_manifest_hash(manifest: &Value, key: &str, path: &Path) -> PcResult<()> {
    let expected = manifest
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| PcError::Format(format!("release manifest missing {key}")))?;
    let actual = file_hash_hex(path)?;
    if actual != expected {
        return Err(PcError::Hash(format!(
            "{key} mismatch: expected {expected} actual {actual}"
        )));
    }
    Ok(())
}

fn verify_manifest_artifact_hash(manifest: &Value, rel: &str, path: &Path) -> PcResult<()> {
    let artifacts = manifest
        .get("artifacts")
        .and_then(Value::as_array)
        .ok_or_else(|| PcError::Format("release manifest missing artifacts array".to_string()))?;
    let expected = artifacts
        .iter()
        .find(|artifact| artifact.get("path").and_then(Value::as_str) == Some(rel))
        .and_then(|artifact| artifact.get("hash_blake3"))
        .and_then(Value::as_str)
        .ok_or_else(|| PcError::Format(format!("release manifest missing artifact {rel}")))?;
    let actual = file_hash_hex(path)?;
    if expected != actual {
        return Err(PcError::Hash(format!(
            "artifact hash mismatch for {rel}: expected {expected} actual {actual}"
        )));
    }
    Ok(())
}

fn require_nonempty(path: &Path) -> PcResult<()> {
    let meta = fs::metadata(path).map_err(|err| {
        PcError::Io(format!(
            "required artifact missing {}: {err}",
            path.display()
        ))
    })?;
    if meta.len() == 0 {
        return Err(PcError::Format(format!(
            "required artifact is empty: {}",
            path.display()
        )));
    }
    Ok(())
}

fn read_json(path: &Path) -> PcResult<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn file_hash_hex(path: &Path) -> PcResult<String> {
    Ok(blake3::hash(&fs::read(path)?).to_hex().to_string())
}
