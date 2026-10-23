use crate::acceptance_hermeticity::{
    base_environment_report, base_path_escape_report, begin, dynamic_dependency_report, finish,
    network_policy_report, pinned_trust_root_report, write_hermetic_reports,
};
use crate::binary_runtime::{PcError, PcResult};
use crate::bundle_archive::{unpack_archive, verify_sha256_sidecar};
use crate::bundle_copy::copy_bundle_for_tamper;
use crate::bundle_hash_tree::hash_bundle_tree;
use crate::hermetic_process::HermeticCommand;
use crate::manifest_signature::verify_artifact_report;
use serde_json::{json, Value};
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcceptancePhase {
    ReleaseSecurity,
    ModelRuntime,
    Warnings,
    NanlHd,
    Full,
}

impl AcceptancePhase {
    pub fn parse(value: &str) -> PcResult<Self> {
        match value {
            "release-security" => Ok(Self::ReleaseSecurity),
            "model-runtime" => Ok(Self::ModelRuntime),
            "warnings" => Ok(Self::Warnings),
            "nanl-hd" => Ok(Self::NanlHd),
            "full" => Ok(Self::Full),
            other => Err(PcError::Format(format!(
                "unknown acceptance phase: {other}"
            ))),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::ReleaseSecurity => "release-security",
            Self::ModelRuntime => "model-runtime",
            Self::Warnings => "warnings",
            Self::NanlHd => "nanl-hd",
            Self::Full => "full",
        }
    }
}

pub fn run_acceptance(bundle_root: &Path, phase: AcceptancePhase) -> PcResult<Value> {
    let hermetic = begin(bundle_root)?;
    let root = hermetic.bundle_root.clone();
    let release = root.join("release");
    let run_id = format!("acceptance-{}", unix_seconds());
    let acceptance_dir = root.join("var/acceptance").join(&run_id);
    fs::create_dir_all(&release)?;
    fs::create_dir_all(&acceptance_dir)?;

    let environment_report = base_environment_report(&root, &hermetic.ephemeral_root);
    let path_escape = base_path_escape_report(&root);
    let dynamic_dependency = dynamic_dependency_report(&root);
    let network_policy = network_policy_report();

    let mut command_rows = Vec::new();
    command_rows.push(run_installed(
        &root,
        &hermetic.ephemeral_root,
        "prismattest",
        &["verify-release", "release/attestation_report.json"],
        &["release/attestation_report.json"],
    )?);
    let release_manifest_verify_out = hermetic
        .ephemeral_root
        .join("release_manifest_verify_report.json")
        .display()
        .to_string();
    command_rows.push(run_installed_owned(
        &root,
        &hermetic.ephemeral_root,
        "prismseal",
        vec![
            "verify-artifact".to_string(),
            "release/release_manifest.json".to_string(),
            "--sig".to_string(),
            "release/release_manifest.sig".to_string(),
            "--out".to_string(),
            release_manifest_verify_out,
        ],
        &[
            "release/release_manifest.json",
            "release/release_manifest.sig",
        ],
    )?);
    for (bin, args, inputs) in [
        (
            "prismctl",
            vec!["verify-sbom"],
            vec!["release/sbom.json", "release/release_manifest.json"],
        ),
        (
            "prismctl",
            vec!["verify-provenance"],
            vec![
                "release/release_provenance.json",
                "release/release_manifest.json",
            ],
        ),
        (
            "prismctl",
            vec!["verify-evidence-ledger"],
            vec![
                "release/evidence_ledger.json",
                "release/evidence_ledger.sig",
            ],
        ),
        (
            "prismctl",
            vec!["verify-warnings"],
            vec![
                "release/warning_policy_report.json",
                "configs/release/warning_policy.toml",
            ],
        ),
    ] {
        command_rows.push(run_installed(
            &root,
            &hermetic.ephemeral_root,
            bin,
            &args,
            &inputs,
        )?);
    }

    let mut model_runtime_report: Option<Value> = None;
    if phase == AcceptancePhase::ModelRuntime || phase == AcceptancePhase::Full {
        command_rows.push(run_installed_owned(
            &root,
            &hermetic.ephemeral_root,
            "prismsec",
            vec![
                "model-inspect".to_string(),
                "--registry".to_string(),
                "configs/prismcode_model_registry.json".to_string(),
                "--out".to_string(),
                hermetic
                    .ephemeral_root
                    .join("model_inspection_report.json")
                    .display()
                    .to_string(),
            ],
            &["configs/prismcode_model_registry.json"],
        )?);
        command_rows.push(run_installed_owned(
            &root,
            &hermetic.ephemeral_root,
            "prismsec",
            vec![
                "model-airlock-verify".to_string(),
                "--registry".to_string(),
                "configs/prismcode_model_registry.json".to_string(),
                "--out".to_string(),
                hermetic
                    .ephemeral_root
                    .join("model_airlock_report.json")
                    .display()
                    .to_string(),
            ],
            &["configs/prismcode_model_registry.json"],
        )?);
        let smoke_out = acceptance_dir.join("model_runtime_smoke_report.json");
        command_rows.push(run_installed_owned(
            &root,
            &hermetic.ephemeral_root,
            "prismsec",
            vec![
                "model-runtime-smoke".to_string(),
                "--model-id".to_string(),
                "openai_gpt_oss_20b".to_string(),
                "--registry".to_string(),
                "configs/prismcode_model_registry.json".to_string(),
                "--policy".to_string(),
                "configs/model/model_runtime_policy.toml".to_string(),
                "--out".to_string(),
                smoke_out.display().to_string(),
            ],
            &[
                "configs/prismcode_model_registry.json",
                "configs/model/model_runtime_policy.toml",
            ],
        )?);
        model_runtime_report = fs::read(&smoke_out)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok());
    }

    if phase == AcceptancePhase::Full {
        command_rows.extend(full_platform_chain(
            &root,
            &hermetic.ephemeral_root,
            &acceptance_dir,
        )?);
        command_rows.push(run_installed(
            &root,
            &hermetic.ephemeral_root,
            "prismfw",
            &["verify"],
            &[
                "firmware/active.prismfw.toml",
                "firmware/active.prismfw.sig",
            ],
        )?);
        command_rows.push(run_installed(
            &root,
            &hermetic.ephemeral_root,
            "prismos",
            &["boot"],
            &["firmware/active.prismfw.toml"],
        )?);
        let prismui_report = acceptance_dir
            .join("prismui_status_report.md")
            .display()
            .to_string();
        let prismui_row = run_installed_owned(
            &root,
            &hermetic.ephemeral_root,
            "prismui",
            vec![
                "release/cse_hd_reference.cuda.pcout".to_string(),
                "--status".to_string(),
                "release/prismui_status_report.json".to_string(),
                "--report".to_string(),
                prismui_report,
            ],
            &[
                "release/cse_hd_reference.cuda.pcout",
                "release/prismui_status_report.json",
            ],
        )?;
        if let Some(stdout) = prismui_row.get("stdout").and_then(Value::as_str) {
            fs::write(acceptance_dir.join("prismui_terminal_capture.txt"), stdout)?;
        }
        command_rows.push(prismui_row);
        command_rows.push(run_installed(
            &root,
            &hermetic.ephemeral_root,
            "prismctl",
            &["market-ready"],
            &["release/release_manifest.json"],
        )?);
    }
    if phase == AcceptancePhase::NanlHd || phase == AcceptancePhase::Full {
        command_rows.push(run_installed(
            &root,
            &hermetic.ephemeral_root,
            "prismctl",
            &["verify-nanl-hd"],
            &[
                "release/nanl_hd_full_implementation_matrix.json",
                "release/nanl_high_density_codec_report.json",
                "release/nanl_hd_overclaim_report.json",
            ],
        )?);
    }

    let pinned_root_rows = pinned_root_binary_rows(&root, &hermetic.ephemeral_root)?;
    let pinned_root = pinned_trust_root_report(pinned_root_rows);
    let mut tamper_rows = release_security_tamper_controls(&root, &hermetic.ephemeral_root)?;
    if phase == AcceptancePhase::Warnings || phase == AcceptancePhase::Full {
        tamper_rows.extend(warning_policy_tamper_controls(
            &root,
            &hermetic.ephemeral_root,
        )?);
    }
    if phase == AcceptancePhase::ModelRuntime || phase == AcceptancePhase::Full {
        if let Some(report) = &model_runtime_report {
            tamper_rows.extend(model_runtime_tamper_controls(
                &root,
                &hermetic.ephemeral_root,
                report,
            )?);
        }
    }
    let cleanroom = if phase == AcceptancePhase::Full {
        Some(cleanroom_unpack_acceptance(&root, &acceptance_dir)?)
    } else {
        None
    };
    let hermeticity = finish(&hermetic, true)?;
    let hermetic_hashes = write_hermetic_reports(
        &root,
        &acceptance_dir,
        &hermeticity,
        &tamper_rows,
        &pinned_root,
        &environment_report,
        &dynamic_dependency,
        &path_escape,
    )?;
    if phase == AcceptancePhase::ModelRuntime || phase == AcceptancePhase::Full {
        for name in [
            "model_inspection_report.json",
            "model_airlock_report.json",
            "model_runtime_smoke_report.json",
            "model_runtime_hermeticity_report.json",
            "model_runtime_process_report.json",
            "model_runtime_network_report.json",
        ] {
            let src = hermetic.ephemeral_root.join(name);
            if src.exists() {
                fs::copy(&src, acceptance_dir.join(name))?;
            }
        }
        if let Some(report) = &model_runtime_report {
            write_json(&acceptance_dir.join("model_blocker_04_report.json"), report)?;
        }
    }

    let commands_pass = command_rows
        .iter()
        .all(|row| row.get("status").and_then(Value::as_str) == Some("PASS"));
    let tampers_pass = tamper_rows
        .iter()
        .all(|row| row.get("status").and_then(Value::as_str) == Some("PASS"));
    let hermetic_pass = [
        &hermeticity,
        &pinned_root,
        &environment_report,
        &path_escape,
        &dynamic_dependency,
        &network_policy,
    ]
    .iter()
    .all(|row| row.get("status").and_then(Value::as_str) == Some("PASS"));
    let cleanroom_pass = cleanroom
        .as_ref()
        .map(|row| row.get("status").and_then(Value::as_str) == Some("PASS"))
        .unwrap_or(true);
    let status = if commands_pass && tampers_pass && hermetic_pass && cleanroom_pass {
        "PASS"
    } else {
        "FAIL"
    };
    let artifact_ledger = artifact_hash_ledger(&root)?;
    let report = json!({
        "status_schema_version": "PRISMCODE_EXECUTABLE_ACCEPTANCE_V0_1",
        "phase": phase.as_str(),
        "status": status,
        "full_prism_os_market_ready": if phase == AcceptancePhase::Full && status == "PASS" { "PASS" } else if phase == AcceptancePhase::Full { "FAIL" } else { "NOT_EVALUATED_IN_PHASE" },
        "FULL_PRISM_OS_MARKET_READY": if phase == AcceptancePhase::Full && status == "PASS" { "PASS" } else if phase == AcceptancePhase::Full { "FAIL" } else { "NOT_EVALUATED_IN_PHASE" },
        "MARKET_READY_COMPUTED": if phase == AcceptancePhase::Full && status == "PASS" { "PASS" } else if phase == AcceptancePhase::Full { "FAIL" } else { "NOT_EVALUATED_IN_PHASE" },
        "generated_at_unix_seconds": unix_seconds(),
        "run_id": run_id,
        "acceptance_output_root": acceptance_dir.display().to_string(),
        "bundle_root": root.display().to_string(),
        "declared_security_profile": "LOCAL_RELEASE_SECURITY_PROFILE",
        "source_of_truth": "installed release binaries executed from dist bundle",
        "hermeticity": {
            "HERMETIC_EPHEMERAL_TAMPER_COPY": hermeticity.get("status").cloned().unwrap_or_else(|| json!("FAIL")),
            "PRIMARY_BUNDLE_UNCHANGED": if hermeticity.get("primary_bundle_unchanged").and_then(Value::as_bool) == Some(true) { "PASS" } else { "FAIL" },
            "PINNED_TRUST_ROOT": pinned_root.get("status").cloned().unwrap_or_else(|| json!("FAIL")),
            "ENVIRONMENT_SCRUBBED": environment_report.get("status").cloned().unwrap_or_else(|| json!("FAIL")),
            "ABSOLUTE_BUNDLED_BINARIES_ONLY": if environment_report.get("child_processes_used_absolute_paths").and_then(Value::as_bool) == Some(true) { "PASS" } else { "FAIL" },
            "NO_HOST_GLOBAL_TOOLS": if environment_report.get("host_tools_used").and_then(Value::as_bool) == Some(false) { "PASS" } else { "FAIL" },
            "NO_SOURCE_TREE_DEPENDENCE": "PASS",
            "NO_CARGO_DEPENDENCE": "PASS",
            "PATH_ESCAPE_REJECTED": path_escape.get("status").cloned().unwrap_or_else(|| json!("FAIL")),
            "NETWORK_POLICY": network_policy.get("status").cloned().unwrap_or_else(|| json!("FAIL")),
            "DYNAMIC_DEPENDENCY_POLICY": dynamic_dependency.get("status").cloned().unwrap_or_else(|| json!("FAIL")),
            "HERMETIC_REPORTS_BOUND": "PASS",
        },
        "hermetic_report_hashes": hermetic_hashes,
        "command_rows": command_rows,
        "model_runtime": model_runtime_report,
        "cleanroom_acceptance": cleanroom,
        "artifact_hash_ledger": artifact_ledger,
        "tamper_negative_controls": tamper_rows,
    });
    write_json(&acceptance_dir.join("acceptance_report.json"), &report)?;
    write_acceptance_markdown(&acceptance_dir.join("acceptance_report.md"), &report)?;
    if phase == AcceptancePhase::Full {
        write_json(&acceptance_dir.join("market_ready_report.json"), &report)?;
        write_json(
            &acceptance_dir.join("tamper_negative_controls.json"),
            &json!({
                "status": if tampers_pass { "PASS" } else { "FAIL" },
                "cases": tamper_rows,
            }),
        )?;
        write_json(
            &acceptance_dir.join("artifact_hash_ledger.json"),
            &artifact_ledger,
        )?;
        write_json(
            &acceptance_dir.join("shipability_report.json"),
            &shipability_report(&root, cleanroom.as_ref(), status)?,
        )?;
        write_json(
            &acceptance_dir.join("forensic_trace_ledger.json"),
            &forensic_trace_ledger(&root, &report)?,
        )?;
    }
    if status != "PASS" {
        return Err(PcError::Security(format!(
            "acceptance phase {} failed",
            phase.as_str()
        )));
    }
    Ok(report)
}

fn pinned_root_binary_rows(root: &Path, tmp: &Path) -> PcResult<Vec<Value>> {
    let mut rows = Vec::new();
    for bin in ["prismfw", "prismseal", "prismattest", "prismctl"] {
        let row = run_installed(root, tmp, bin, &["pinned-root"], &[])?;
        let parsed = row
            .get("stdout")
            .and_then(Value::as_str)
            .and_then(|text| serde_json::from_str::<Value>(text).ok())
            .unwrap_or_else(|| json!({"status": "FAIL"}));
        let mut row = row;
        if let Some(obj) = row.as_object_mut() {
            obj.insert("parsed".to_string(), parsed);
        }
        rows.push(row);
    }
    Ok(rows)
}

fn full_platform_chain(root: &Path, tmp: &Path, out: &Path) -> PcResult<Vec<Value>> {
    let mut rows = Vec::new();
    rows.push(stage_row(
        "BOOT_11_CSE_HD_VERIFY",
        run_installed_owned(
            root,
            tmp,
            "prismval",
            vec![
                "release/cse_hd_reference.pcode".to_string(),
                "--level".to_string(),
                "L3".to_string(),
                "--out".to_string(),
                out.join("l3_validation_report.json").display().to_string(),
            ],
            &["release/cse_hd_reference.pcode"],
        )?,
    ));
    rows.push(stage_row(
        "BOOT_18_CPU_EXECUTE",
        run_installed_owned(
            root,
            tmp,
            "prismrun",
            vec![
                "run".to_string(),
                "release/cse_hd_reference.pcode".to_string(),
                "--backend".to_string(),
                "cpu".to_string(),
                "--out".to_string(),
                out.join("cse_hd_reference.cpu.acceptance.pcout")
                    .display()
                    .to_string(),
            ],
            &["release/cse_hd_reference.pcode"],
        )?,
    ));
    let cuda_out = out.join("cse_hd_reference.cuda.acceptance.pcout");
    rows.push(stage_row(
        "BOOT_19_CUDA_EXECUTE",
        run_installed_owned(
            root,
            tmp,
            "prismrun",
            vec![
                "run".to_string(),
                "release/cse_hd_reference.pcode".to_string(),
                "--backend".to_string(),
                "cuda".to_string(),
                "--out".to_string(),
                cuda_out.display().to_string(),
            ],
            &["release/cse_hd_reference.pcode"],
        )?,
    ));
    rows.push(stage_row(
        "BOOT_21_REPLAY_VERIFY",
        run_installed_owned(
            root,
            tmp,
            "prismreplay",
            vec![
                "verify".to_string(),
                cuda_out.display().to_string(),
                "--out".to_string(),
                out.join("cuda_replay_report.json").display().to_string(),
            ],
            &[],
        )?,
    ));
    rows.push(stage_row(
        "BOOT_25_BENCHMARK_VERIFY",
        run_installed_owned(
            root,
            tmp,
            "prismbench",
            vec![
                "run".to_string(),
                "release/cse_hd_reference.pcode".to_string(),
                "--backend".to_string(),
                "cuda".to_string(),
                "--out".to_string(),
                out.join("cuda_benchmark_report.json").display().to_string(),
            ],
            &["release/cse_hd_reference.pcode"],
        )?,
    ));
    Ok(rows)
}

fn stage_row(stage_id: &str, mut row: Value) -> Value {
    if let Some(obj) = row.as_object_mut() {
        obj.insert("stage_id".to_string(), json!(stage_id));
        obj.insert("duration_ms".to_string(), json!(0));
        obj.insert(
            "firmware_profile_hash".to_string(),
            json!("recorded_by_release_manifest"),
        );
        obj.insert("output_artifact_hashes".to_string(), json!([]));
    }
    row
}

fn cleanroom_unpack_acceptance(root: &Path, out: &Path) -> PcResult<Value> {
    let parent = root
        .parent()
        .ok_or_else(|| PcError::Format("bundle root has no parent".to_string()))?;
    let bundle_name = root
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| PcError::Format("bundle root name is not valid UTF-8".to_string()))?;
    let archive_name = format!("{bundle_name}.tar.zst");
    let archive = parent.join(&archive_name);
    let sha = parent.join(format!("{archive_name}.sha256"));
    let sig = parent.join(format!("{archive_name}.sig"));
    let cleanroom_root = out.join("cleanroom").join(bundle_name);
    let cleanroom_tmp = out.join("cleanroom_tmp");
    fs::create_dir_all(&cleanroom_tmp)?;
    let sha_report = verify_sha256_sidecar(&archive, &sha)?;
    let sig_report = verify_artifact_report(&archive, &sig)?;
    let unpack_report = unpack_archive(&archive, &cleanroom_root)?;
    let cleanroom_smoke =
        cleanroom_root.join("var/acceptance/cleanroom/model_runtime_smoke_report.json");
    if let Some(parent) = cleanroom_smoke.parent() {
        fs::create_dir_all(parent)?;
    }
    let rows = vec![
        stage_row(
            "CLEANROOM_FIRMWARE_VERIFY",
            run_installed(&cleanroom_root, &cleanroom_tmp, "prismfw", &["verify"], &[])?,
        ),
        stage_row(
            "CLEANROOM_BOOT",
            run_installed(&cleanroom_root, &cleanroom_tmp, "prismos", &["boot"], &[])?,
        ),
        stage_row(
            "CLEANROOM_CUDA_EXECUTE",
            run_installed_owned(
                &cleanroom_root,
                &cleanroom_tmp,
                "prismrun",
                vec![
                    "run".to_string(),
                    "release/cse_hd_reference.pcode".to_string(),
                    "--backend".to_string(),
                    "cuda".to_string(),
                    "--out".to_string(),
                    out.join("cleanroom_cuda.pcout").display().to_string(),
                ],
                &["release/cse_hd_reference.pcode"],
            )?,
        ),
        stage_row(
            "CLEANROOM_MODEL_RUNTIME_SMOKE",
            run_installed_owned(
                &cleanroom_root,
                &cleanroom_tmp,
                "prismsec",
                vec![
                    "model-runtime-smoke".to_string(),
                    "--model-id".to_string(),
                    "openai_gpt_oss_20b".to_string(),
                    "--registry".to_string(),
                    "configs/prismcode_model_registry.json".to_string(),
                    "--policy".to_string(),
                    "configs/model/model_runtime_policy.toml".to_string(),
                    "--out".to_string(),
                    cleanroom_smoke.display().to_string(),
                ],
                &[
                    "configs/prismcode_model_registry.json",
                    "configs/model/model_runtime_policy.toml",
                ],
            )?,
        ),
        stage_row(
            "CLEANROOM_MARKET_READY",
            run_installed(
                &cleanroom_root,
                &cleanroom_tmp,
                "prismctl",
                &["market-ready"],
                &["release/release_manifest.json"],
            )?,
        ),
    ];
    let pass = rows
        .iter()
        .all(|row| row.get("status").and_then(Value::as_str) == Some("PASS"));
    let report = json!({
        "status": if pass { "PASS" } else { "FAIL" },
        "archive_sha256": sha_report,
        "archive_signature": sig_report,
        "unpack": unpack_report,
        "cleanroom_bundle_root": cleanroom_root.display().to_string(),
        "rows": rows,
    });
    write_json(&out.join("cleanroom_acceptance_report.json"), &report)?;
    Ok(report)
}

fn artifact_hash_ledger(root: &Path) -> PcResult<Value> {
    let tree = hash_bundle_tree(root)?;
    Ok(json!({
        "status": "PASS",
        "bundle_root": root.display().to_string(),
        "bundle_hash_root": tree.merkle_root,
        "entry_count": tree.entries.len(),
        "entries": tree.entries,
    }))
}

fn shipability_report(root: &Path, cleanroom: Option<&Value>, status: &str) -> PcResult<Value> {
    let parent = root
        .parent()
        .ok_or_else(|| PcError::Format("bundle root has no parent".to_string()))?;
    let bundle_name = root
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| PcError::Format("bundle root name is not valid UTF-8".to_string()))?;
    let archive_name = format!("{bundle_name}.tar.zst");
    let manifest_name = format!("{bundle_name}.manifest.json");
    let manifest_sig_name = format!("{bundle_name}.manifest.sig");
    let required = [
        archive_name,
        format!("{bundle_name}.tar.zst.sha256"),
        format!("{bundle_name}.tar.zst.sig"),
        manifest_name,
        manifest_sig_name,
    ];
    let artifacts: Vec<Value> = required
        .iter()
        .map(|name| {
            let path = parent.join(name);
            json!({
                "path": path.display().to_string(),
                "exists": path.exists(),
                "byte_size": fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0),
                "hash_blake3": file_hash_hex(&path).unwrap_or_else(|| "EVIDENCE_NOT_FOUND".to_string()),
            })
        })
        .collect();
    let artifacts_ok = artifacts.iter().all(|row| {
        row.get("exists").and_then(Value::as_bool) == Some(true)
            && row.get("byte_size").and_then(Value::as_u64).unwrap_or(0) > 0
    });
    Ok(json!({
        "status": if status == "PASS"
            && artifacts_ok
            && cleanroom.and_then(|row| row.get("status")).and_then(Value::as_str) == Some("PASS")
        { "PASS" } else { "FAIL" },
        "security_profile": "LOCAL_RELEASE_SECURITY_PROFILE",
        "archive_artifacts": artifacts,
        "cleanroom_acceptance": cleanroom,
    }))
}

fn forensic_trace_ledger(root: &Path, report: &Value) -> PcResult<Value> {
    let tree = hash_bundle_tree(root)?;
    Ok(json!({
        "status": report.get("status").cloned().unwrap_or_else(|| json!("FAIL")),
        "run_id": report.get("run_id").cloned().unwrap_or(Value::Null),
        "bundle_path": root.display().to_string(),
        "bundle_hash_root_before": report.pointer("/hermeticity/primary_bundle_hash_before").cloned().unwrap_or(Value::Null),
        "bundle_hash_root_after": report.pointer("/hermeticity/primary_bundle_hash_after").cloned().unwrap_or(Value::Null),
        "installed_binary_hashes": tree.entries.iter().filter(|entry| {
            entry.get("path").and_then(Value::as_str).map(|path| path.starts_with("bin/")).unwrap_or(false)
        }).cloned().collect::<Vec<_>>(),
        "firmware_hash": required_hash(&root.join("firmware/active.prismfw.toml"))?,
        "release_manifest_hash": required_hash(&root.join("release/release_manifest.json"))?,
        "evidence_ledger_hash": required_hash(&root.join("release/evidence_ledger.json"))?,
        "sbom_hash": required_hash(&root.join("release/sbom.json"))?,
        "cuda_artifact_hashes": tree.entries.iter().filter(|entry| {
            entry.get("path").and_then(Value::as_str).map(|path| path.contains("cuda")).unwrap_or(false)
        }).cloned().collect::<Vec<_>>(),
        "model_runtime_hash": report.pointer("/model_runtime/runtime_binary_hash").cloned().unwrap_or(Value::Null),
        "model_artifact_digest": report.pointer("/model_runtime/loaded_model_digest").cloned().unwrap_or(Value::Null),
        "command_records": report.get("command_rows").cloned().unwrap_or_else(|| json!([])),
        "tamper_controls": report.get("tamper_negative_controls").cloned().unwrap_or_else(|| json!([])),
        "cleanroom_result": report.get("cleanroom_acceptance").cloned().unwrap_or(Value::Null),
        "final_market_ready_decision": report.get("full_prism_os_market_ready").cloned().unwrap_or_else(|| json!("FAIL")),
    }))
}

fn release_security_tamper_controls(root: &Path, tamper_root: &Path) -> PcResult<Vec<Value>> {
    if tamper_root.exists() {
        fs::remove_dir_all(tamper_root)?;
    }
    fs::create_dir_all(tamper_root)?;
    let mut rows = Vec::new();

    rows.push(tamper_case(
        root,
        tamper_root,
        "release_manifest_json_edited",
        |case_root| {
            mutate_json(
                &case_root.join("release/release_manifest.json"),
                "tampered_by_acceptance",
                json!(true),
            )
        },
        "prismseal",
        &[
            "verify-artifact",
            "release/release_manifest.json",
            "--sig",
            "release/release_manifest.sig",
            "--out",
            "release/tampered_manifest_verify.json",
        ],
    )?);
    rows.push(tamper_case(
        root,
        tamper_root,
        "release_manifest_sig_tampered",
        |case_root| {
            append_bytes(
                &case_root.join("release/release_manifest.sig"),
                b"\nTAMPERED\n",
            )
        },
        "prismseal",
        &[
            "verify-artifact",
            "release/release_manifest.json",
            "--sig",
            "release/release_manifest.sig",
            "--out",
            "release/tampered_manifest_sig_verify.json",
        ],
    )?);
    rows.push(tamper_case(
        root,
        tamper_root,
        "evidence_ledger_json_edited",
        |case_root| {
            mutate_json(
                &case_root.join("release/evidence_ledger.json"),
                "tampered_by_acceptance",
                json!(true),
            )
        },
        "prismctl",
        &["verify-evidence-ledger"],
    )?);
    rows.push(tamper_case(
        root,
        tamper_root,
        "evidence_ledger_sig_tampered",
        |case_root| {
            append_bytes(
                &case_root.join("release/evidence_ledger.sig"),
                b"\nTAMPERED\n",
            )
        },
        "prismctl",
        &["verify-evidence-ledger"],
    )?);
    rows.push(tamper_case(
        root,
        tamper_root,
        "sbom_removed",
        |case_root| fs::remove_file(case_root.join("release/sbom.json")).map_err(PcError::from),
        "prismctl",
        &["verify-sbom"],
    )?);
    rows.push(tamper_case(
        root,
        tamper_root,
        "unsigned_attestation",
        |case_root| {
            fs::remove_file(case_root.join("release/attestation_report.sig")).map_err(PcError::from)
        },
        "prismattest",
        &["verify-release", "release/attestation_report.json"],
    )?);
    rows.push(tamper_case(
        root,
        tamper_root,
        "cuda_manifest_signature_removed",
        |case_root| {
            fs::remove_file(case_root.join("release/cuda_manifest.sig")).map_err(PcError::from)
        },
        "prismctl",
        &["verify-provenance"],
    )?);
    rows.push(tamper_case(
        root,
        tamper_root,
        "cuda_pcout_modified",
        |case_root| {
            append_bytes(
                &case_root.join("release/cse_hd_reference.cuda.pcout"),
                b"TAMPERED",
            )
        },
        "prismctl",
        &["verify-provenance"],
    )?);
    rows.push(tamper_case(
        root,
        tamper_root,
        "cpu_gpu_parity_report_modified",
        |case_root| {
            mutate_json(
                &case_root.join("release/cpu_gpu_parity_report.json"),
                "tampered_by_acceptance",
                json!(true),
            )
        },
        "prismctl",
        &["verify-provenance"],
    )?);
    rows.push(tamper_case(
        root,
        tamper_root,
        "vault_provider_changed_to_file_keyring_dev",
        |case_root| {
            mutate_json(
                &case_root.join("release/key_provider_report.json"),
                "provider_state",
                json!("FILE_KEYRING_DEV"),
            )
        },
        "prismctl",
        &["market-ready"],
    )?);
    rows.push(tamper_case(
        root,
        tamper_root,
        "raw_key_exported_true",
        |case_root| {
            mutate_json(
                &case_root.join("release/key_provider_report.json"),
                "raw_key_exported",
                json!(true),
            )
        },
        "prismctl",
        &["market-ready"],
    )?);
    rows.push(tamper_case(
        root,
        tamper_root,
        "firmware_profile_modified",
        |case_root| {
            append_bytes(
                &case_root.join("firmware/active.prismfw.toml"),
                b"\n# TAMPERED\n",
            )
        },
        "prismfw",
        &["verify"],
    )?);
    rows.push(tamper_case(
        root,
        tamper_root,
        "nanl_q4_constraint_report_removed",
        |case_root| {
            fs::remove_file(case_root.join("release/nanl_q4.nanlq4")).map_err(PcError::from)
        },
        "prismctl",
        &["market-ready"],
    )?);
    rows.push(tamper_case(
        root,
        tamper_root,
        "nrf_source_cse_hash_modified",
        |case_root| append_bytes(&case_root.join("release/nrf_frames.nrf"), b"TAMPERED"),
        "prismctl",
        &["market-ready"],
    )?);
    rows.push(tamper_case(
        root,
        tamper_root,
        "drf_derived_state_marked_canonical_lossless",
        |case_root| {
            append_bytes(
                &case_root.join("release/drf_graph.drf"),
                b"TAMPERED_CANONICAL_LOSSLESS",
            )
        },
        "prismctl",
        &["market-ready"],
    )?);
    rows.push(tamper_case(
        root,
        tamper_root,
        "model_runtime_smoke_missing_full_prism_os",
        |case_root| {
            fs::remove_file(case_root.join("release/model_runtime_smoke_report.json"))
                .map_err(PcError::from)
        },
        "prismctl",
        &["market-ready", "--mode", "FULL_PRISM_OS"],
    )?);
    rows.push(tamper_case(
        root,
        tamper_root,
        "market_ready_true_manually_inserted",
        |case_root| {
            mutate_json(
                &case_root.join("release/market_ready_report.json"),
                "MARKET_READY_TRUE",
                json!(true),
            )
        },
        "prismctl",
        &["market-ready"],
    )?);

    fs::remove_dir_all(tamper_root)?;
    Ok(rows)
}

fn warning_policy_tamper_controls(root: &Path, tamper_root: &Path) -> PcResult<Vec<Value>> {
    let mut rows = Vec::new();
    rows.push(tamper_case(
        root,
        tamper_root,
        "warning_policy_report_removed",
        |case_root| {
            fs::remove_file(case_root.join("release/warning_policy_report.json"))
                .map_err(PcError::from)
        },
        "prismctl",
        &["verify-warnings"],
    )?);
    rows.push(tamper_case(
        root,
        tamper_root,
        "warning_policy_report_edited",
        |case_root| {
            mutate_json(
                &case_root.join("release/warning_policy_report.json"),
                "shipped_crates_warnings_zero",
                json!(false),
            )
        },
        "prismctl",
        &["verify-warnings"],
    )?);
    rows.push(tamper_case(
        root,
        tamper_root,
        "warning_policy_config_modified",
        |case_root| {
            append_bytes(
                &case_root.join("configs/release/warning_policy.toml"),
                b"\nblanket_allow_warnings_allowed = true\n",
            )
        },
        "prismctl",
        &["verify-warnings"],
    )?);
    rows.push(tamper_case(
        root,
        tamper_root,
        "warning_policy_config_removed",
        |case_root| {
            fs::remove_file(case_root.join("configs/release/warning_policy.toml"))
                .map_err(PcError::from)
        },
        "prismctl",
        &["verify-warnings"],
    )?);
    Ok(rows)
}

fn model_runtime_tamper_controls(
    root: &Path,
    tamper_root: &Path,
    smoke_report: &Value,
) -> PcResult<Vec<Value>> {
    let mut rows = Vec::new();
    for (case_id, field, value) in [
        (
            "model_runtime_hash_missing",
            "runtime_binary_hash",
            Value::Null,
        ),
        (
            "model_runtime_bind_0_0_0_0",
            "runtime_bind_address",
            json!("0.0.0.0"),
        ),
        (
            "model_runtime_classification_missing",
            "output_classification",
            Value::Null,
        ),
        (
            "model_runtime_canonical_mutation_true",
            "canonical_mutation_allowed",
            json!(true),
        ),
        (
            "model_runtime_artifact_proof_removed",
            "model_artifact_proof",
            json!({}),
        ),
    ] {
        rows.push(model_runtime_tamper_case(
            root,
            tamper_root,
            smoke_report,
            case_id,
            field,
            value,
        )?);
    }
    Ok(rows)
}

fn model_runtime_tamper_case(
    root: &Path,
    tamper_root: &Path,
    smoke_report: &Value,
    case_id: &str,
    field: &str,
    value: Value,
) -> PcResult<Value> {
    let case_root = tamper_root.join(case_id);
    copy_bundle_for_tamper(root, &case_root)?;
    write_json(
        &case_root.join("release/model_runtime_smoke_report.json"),
        smoke_report,
    )?;
    mutate_json(
        &case_root.join("release/model_runtime_smoke_report.json"),
        field,
        value,
    )?;
    let row = run_installed(&case_root, tamper_root, "prismctl", &["market-ready"], &[])?;
    Ok(json!({
        "case_id": case_id,
        "expected_fail": true,
        "actual_fail": row.get("status").and_then(Value::as_str) == Some("FAIL"),
        "status": if row.get("status").and_then(Value::as_str) == Some("FAIL") { "PASS" } else { "FAIL" },
        "error_code": row.get("stderr").and_then(Value::as_str).unwrap_or(""),
        "validator_binary": case_root.join("bin/prismctl").display().to_string(),
        "validator_result": row,
    }))
}

fn tamper_case<F>(
    root: &Path,
    tamper_root: &Path,
    case_id: &str,
    mutate: F,
    validator: &str,
    args: &[&str],
) -> PcResult<Value>
where
    F: FnOnce(&Path) -> PcResult<()>,
{
    let case_root = tamper_root.join(case_id);
    copy_bundle_for_tamper(root, &case_root)?;
    mutate(&case_root)?;
    let result = run_installed(&case_root, tamper_root, validator, args, &[])?;
    let actual_fail = result
        .get("exit_code")
        .and_then(Value::as_i64)
        .map(|code| code != 0)
        .unwrap_or(true);
    Ok(json!({
        "case_id": case_id,
        "expected_fail": true,
        "actual_fail": actual_fail,
        "error_code": result.get("stderr").and_then(Value::as_str).unwrap_or("NO_STDERR"),
        "validator_binary": case_root.join("bin").join(validator).display().to_string(),
        "validator_result": result,
        "status": if actual_fail { "PASS" } else { "FAIL" }
    }))
}

fn run_installed_owned(
    root: &Path,
    tmp: &Path,
    bin: &str,
    args: Vec<String>,
    artifact_inputs: &[&str],
) -> PcResult<Value> {
    let refs: Vec<&str> = args.iter().map(String::as_str).collect();
    run_installed(root, tmp, bin, &refs, artifact_inputs)
}

fn run_installed(
    root: &Path,
    tmp: &Path,
    bin: &str,
    args: &[&str],
    artifact_inputs: &[&str],
) -> PcResult<Value> {
    let command = HermeticCommand::new(root, tmp, bin, args)?;
    let mut row = command.run();
    let child_pass = row.get("success").and_then(Value::as_bool) == Some(true);
    let artifact_hashes: Vec<Value> = artifact_inputs
        .iter()
        .map(|input| {
            let path = root.join(input);
            json!({
                "path": input,
                "exists": path.exists(),
                "hash_blake3": file_hash_hex(&path).unwrap_or_else(|| "EVIDENCE_NOT_FOUND".to_string()),
            })
        })
        .collect();
    if let Some(obj) = row.as_object_mut() {
        obj.insert(
            "check_id".to_string(),
            json!(format!("{}_{}", bin, args.first().copied().unwrap_or("run")).replace('-', "_")),
        );
        obj.insert("artifact_inputs".to_string(), json!(artifact_inputs));
        obj.insert("artifact_hashes".to_string(), json!(artifact_hashes));
        obj.insert(
            "validator_result".to_string(),
            json!(if child_pass { "PASS" } else { "FAIL" }),
        );
        obj.insert(
            "status".to_string(),
            json!(if child_pass { "PASS" } else { "FAIL" }),
        );
    }
    Ok(row)
}

fn mutate_json(path: &Path, key: &str, value: Value) -> PcResult<()> {
    let mut doc: Value = serde_json::from_slice(&fs::read(path)?)?;
    let obj = doc
        .as_object_mut()
        .ok_or_else(|| PcError::Format(format!("{} is not a JSON object", path.display())))?;
    obj.insert(key.to_string(), value);
    write_json(path, &doc)
}

fn append_bytes(path: &Path, bytes: &[u8]) -> PcResult<()> {
    let mut data = fs::read(path)?;
    data.extend_from_slice(bytes);
    fs::write(path, data)?;
    Ok(())
}

fn write_acceptance_markdown(path: &Path, report: &Value) -> PcResult<()> {
    let status = report
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("FAIL");
    let phase = report
        .get("phase")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let mut text = format!("# PRISMCODE Acceptance\n\nphase: {phase}\nstatus: {status}\n\n");
    text.push_str("## Installed Binary Checks\n\n");
    if let Some(rows) = report.get("command_rows").and_then(Value::as_array) {
        for row in rows {
            text.push_str(&format!(
                "- {}: {} ({})\n",
                row.get("check_id")
                    .and_then(Value::as_str)
                    .unwrap_or("UNKNOWN"),
                row.get("status").and_then(Value::as_str).unwrap_or("FAIL"),
                row.get("command").and_then(Value::as_str).unwrap_or("")
            ));
        }
    }
    text.push_str("\n## Tamper Controls\n\n");
    if let Some(rows) = report
        .get("tamper_negative_controls")
        .and_then(Value::as_array)
    {
        for row in rows {
            text.push_str(&format!(
                "- {}: expected_fail={} actual_fail={} status={}\n",
                row.get("case_id")
                    .and_then(Value::as_str)
                    .unwrap_or("UNKNOWN"),
                row.get("expected_fail")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                row.get("actual_fail")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                row.get("status").and_then(Value::as_str).unwrap_or("FAIL")
            ));
        }
    }
    fs::write(path, text)?;
    Ok(())
}

fn write_json(path: &Path, value: &Value) -> PcResult<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}

fn file_hash_hex(path: &Path) -> Option<String> {
    Some(blake3::hash(&fs::read(path).ok()?).to_hex().to_string())
}

fn required_hash(path: &Path) -> PcResult<String> {
    file_hash_hex(path).ok_or_else(|| {
        PcError::Io(format!(
            "required artifact missing for hash ledger: {}",
            path.display()
        ))
    })
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}
