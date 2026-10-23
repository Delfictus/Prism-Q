use clap::{Parser, Subcommand};
use prismcode::binary_runtime::{seal_sidecar, security_inspect, PcError};
use prismcode::key_provider;
use prismcode::key_provider_local_release_vault::{
    build_blocker_02_report, report_artifact, write_json,
};
use prismcode::manifest_signature::{sign_artifact_report, verify_artifact_report};
use prismcode::trust_root;
use prismcode::{attestation, release_manifest, signature, vault_provider};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "prismseal", about = "PRISM-SEAL dev sealing and verification")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Seal {
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    Verify {
        input: PathBuf,
    },
    SignArtifact {
        artifact: PathBuf,
        #[arg(long)]
        policy: PathBuf,
        #[arg(long)]
        sig_out: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    VerifyArtifact {
        artifact: PathBuf,
        #[arg(long)]
        sig: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    PinnedRoot,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), PcError> {
    match Cli::parse().command {
        Command::Seal { input, out } => {
            let mut report = seal_sidecar(&input, &out)?;
            signature::apply_unsigned_dev_signature_state(&mut report);
            let mut inspection = report["inspection"].clone();
            inspection = attestation::mark_artifact_attestation_unavailable(inspection);
            release_manifest::apply_release_candidate_policy(&mut inspection);
            if let Some(obj) = report.as_object_mut() {
                obj.insert("inspection".to_string(), inspection);
            }
            release_manifest::apply_release_candidate_policy(&mut report);
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::Verify { input } => {
            let mut report = security_inspect(&input)?;
            apply_provider_state(&mut report);
            report = attestation::mark_artifact_attestation_unavailable(report);
            release_manifest::apply_release_candidate_policy(&mut report);
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::SignArtifact {
            artifact,
            policy,
            sig_out,
            out,
        } => {
            let mut report = sign_artifact_report(&artifact, &policy, &sig_out)?;
            attach_report_record(
                &mut report,
                &out,
                "cargo run -p prismcode --bin prismseal -- sign-artifact",
            );
            write_report_with_record(
                &out,
                &mut report,
                "cargo run -p prismcode --bin prismseal -- sign-artifact",
            )?;
            maybe_write_blocker_report()?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::VerifyArtifact { artifact, sig, out } => {
            trust_root::require_release_pinned_root()?;
            let mut report = verify_artifact_report(&artifact, &sig)?;
            attach_report_record(
                &mut report,
                &out,
                "cargo run -p prismcode --bin prismseal -- verify-artifact",
            );
            write_report_with_record(
                &out,
                &mut report,
                "cargo run -p prismcode --bin prismseal -- verify-artifact",
            )?;
            maybe_write_blocker_report()?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::PinnedRoot => {
            println!("{}", serde_json::to_string_pretty(&trust_root::report())?);
        }
    }
    Ok(())
}

fn apply_provider_state(report: &mut serde_json::Value) {
    match report
        .get("vault_status")
        .and_then(serde_json::Value::as_str)
    {
        Some("FILE_KEYRING_DEV_BOUND") => {
            if let Some(obj) = report.as_object_mut() {
                obj.insert(
                    "key_provider".to_string(),
                    serde_json::json!(key_provider::FILE_KEYRING_DEV),
                );
            }
            key_provider::apply_file_keyring_dev_overlay(report);
            vault_provider::apply_file_keyring_dev_binding(report);
            signature::apply_file_keyring_dev_signature_state(report);
        }
        _ => {
            vault_provider::apply_unbound_vault_binding(report);
            signature::apply_unsigned_dev_signature_state(report);
        }
    }
}

fn attach_report_record(report: &mut serde_json::Value, out: &std::path::Path, producer: &str) {
    if let Some(obj) = report.as_object_mut() {
        obj.insert(
            "path".to_string(),
            serde_json::json!(out.display().to_string()),
        );
        let realpath = std::fs::canonicalize(out)
            .map(|path| path.display().to_string())
            .unwrap_or_else(|_| "PENDING_WRITE".to_string());
        obj.insert("realpath".to_string(), serde_json::json!(realpath));
        obj.insert("byte_size".to_string(), serde_json::json!(0));
        obj.insert("hash_algorithm".to_string(), serde_json::json!("BLAKE3"));
        obj.insert("hash".to_string(), serde_json::json!("PENDING_WRITE"));
        obj.insert("producer_command".to_string(), serde_json::json!(producer));
    }
}

fn write_report_with_record(
    out: &std::path::Path,
    report: &mut serde_json::Value,
    producer: &str,
) -> Result<(), PcError> {
    write_json(out, report)?;
    let policy_hash = report
        .get("policy_hash")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("EVIDENCE_NOT_FOUND");
    let record = report_artifact(out, producer, "PASS", policy_hash);
    if let (Some(obj), Some(record_obj)) = (report.as_object_mut(), record.as_object()) {
        for (key, value) in record_obj {
            obj.insert(key.clone(), value.clone());
        }
    }
    write_json(out, report)?;
    Ok(())
}

fn maybe_write_blocker_report() -> Result<(), PcError> {
    let reports = [
        (
            "cargo run -p prismcode --bin prismkey -- provider-inspect",
            std::path::Path::new("build/security/key_provider_report.json"),
        ),
        (
            "cargo run -p prismcode --bin prismkey -- wrap-test",
            std::path::Path::new("build/security/vault_wrap_test_report.json"),
        ),
        (
            "cargo run -p prismcode --bin prismkey -- rotate-test",
            std::path::Path::new("build/security/vault_rotation_report.json"),
        ),
        (
            "cargo run -p prismcode --bin prismkey -- sign-test",
            std::path::Path::new("build/security/vault_sign_test_report.json"),
        ),
        (
            "cargo run -p prismcode --bin prismseal -- sign-artifact",
            std::path::Path::new("build/security/cuda_manifest_signature_report.json"),
        ),
        (
            "cargo run -p prismcode --bin prismseal -- verify-artifact",
            std::path::Path::new("build/security/cuda_manifest_verify_report.json"),
        ),
    ];
    if reports.iter().all(|(_, path)| path.exists()) {
        let policy_hash = std::fs::read_to_string("configs/security/release_key_policy.toml")
            .map(|text| blake3::hash(text.as_bytes()).to_hex().to_string())
            .unwrap_or_else(|_| "EVIDENCE_NOT_FOUND".to_string());
        let report = build_blocker_02_report(&reports, &policy_hash);
        write_json(
            std::path::Path::new("build/security/vault_blocker_02_report.json"),
            &report,
        )?;
    }
    Ok(())
}
