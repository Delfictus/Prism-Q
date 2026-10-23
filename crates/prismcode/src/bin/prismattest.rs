use clap::{Parser, Subcommand};
use prismcode::binary_runtime::{runtime_attestation_report, security_inspect, PcError};
use prismcode::manifest_signature::verify_artifact_report;
use prismcode::trust_root;
use prismcode::{attestation, release_manifest};
use serde_json::{json, Value};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "prismattest", about = "PRISM runtime and artifact attestation")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Runtime {
        #[arg(long)]
        out: Option<PathBuf>,
    },
    Verify {
        input: PathBuf,
    },
    VerifyRelease {
        input: PathBuf,
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
        Command::Runtime { out } => {
            let report = runtime_attestation_report(None)?;
            let mut report = attestation::mark_runtime_attestation_unavailable(report);
            release_manifest::apply_release_candidate_policy(&mut report);
            if let Some(path) = out {
                if let Some(parent) = path.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::write(path, serde_json::to_vec_pretty(&report)?)?;
            }
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::Verify { input } => {
            let report = security_inspect(&input)?;
            let mut report = attestation::mark_artifact_attestation_unavailable(report);
            release_manifest::apply_release_candidate_policy(&mut report);
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::VerifyRelease { input } => {
            trust_root::require_release_pinned_root()?;
            let sig = input.with_file_name("attestation_report.sig");
            let signature = verify_artifact_report(&input, &sig)?;
            let report: Value = serde_json::from_slice(&fs::read(&input)?)?;
            let status = report
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or("FAIL");
            let attestation = report
                .get("attestation_status")
                .and_then(Value::as_str)
                .unwrap_or("ATTESTATION_UNAVAILABLE");
            let signature_status = report
                .get("signature_status")
                .and_then(Value::as_str)
                .unwrap_or("UNSIGNED");
            let signature_verified =
                report.get("signature_verified").and_then(Value::as_bool) == Some(true);
            let raw_key_exported =
                report.get("raw_key_exported").and_then(Value::as_bool) == Some(true);
            let pass = status == "PASS"
                && matches!(
                    attestation,
                    "SIGNED_RELEASE_ATTESTED" | "TRUSTED_ATTESTED" | "SIGNED_ONLY_ACCEPTED"
                )
                && signature_status == "SIGNED_BY_RELEASE_VAULT"
                && signature_verified
                && !raw_key_exported
                && report
                    .get("release_manifest_hash_blake3")
                    .and_then(Value::as_str)
                    .is_some();
            let out = json!({
                "status": if pass { "PASS" } else { "FAIL" },
                "check_id": "RELEASE_ATTESTATION_VERIFIED",
                "input": input.display().to_string(),
                "attestation_status": attestation,
                "signature_status": signature_status,
                "signature_verified": signature_verified,
                "raw_key_exported": raw_key_exported,
                "pinned_root_hash": trust_root::pinned_root_hash(),
                "signature": signature,
            });
            println!("{}", serde_json::to_string_pretty(&out)?);
            if !pass {
                return Err(PcError::Security(
                    "release attestation verification failed".to_string(),
                ));
            }
        }
        Command::PinnedRoot => {
            println!("{}", serde_json::to_string_pretty(&trust_root::report())?);
        }
    }
    Ok(())
}
