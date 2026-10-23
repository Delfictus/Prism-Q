use clap::{Parser, Subcommand};
use prismcode::binary_runtime::{security_inspect, validate_packet, PcError, ValidationLevel};
use prismcode::key_provider;
use prismcode::{attestation, release_manifest, signature, vault_provider};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "prismval", about = "Validate PRISMCODE binary packets")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
    input: Option<PathBuf>,
    #[arg(long, default_value = "L3")]
    level: String,
    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Debug, Subcommand)]
enum Command {
    Validate {
        input: PathBuf,
        #[arg(long, default_value = "L3")]
        level: String,
        #[arg(long)]
        out: Option<PathBuf>,
    },
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), PcError> {
    let cli = Cli::parse();
    match cli.command {
        Some(Command::Validate { input, level, out }) => {
            let level = ValidationLevel::parse(&level)?;
            let is_l5 = matches!(level, ValidationLevel::L5);
            let result = validate_packet(&input, level)?;
            let mut report = serde_json::to_value(&result)?;
            decorate_validation_report(&mut report, &input, is_l5)?;
            if let Some(out) = out {
                write_report(&out, &report)?;
            }
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
        None => {
            let input = cli.input.ok_or_else(|| {
                PcError::Format("missing input path; use prismval <input> --level L3".to_string())
            })?;
            let level = ValidationLevel::parse(&cli.level)?;
            let is_l5 = matches!(level, ValidationLevel::L5);
            let result = validate_packet(&input, level)?;
            let mut report = serde_json::to_value(&result)?;
            decorate_validation_report(&mut report, &input, is_l5)?;
            if let Some(out) = cli.out {
                write_report(&out, &report)?;
            }
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
    }
}

fn write_report(out: &PathBuf, report: &serde_json::Value) -> Result<(), PcError> {
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(out, serde_json::to_vec_pretty(report)?)?;
    Ok(())
}

fn decorate_validation_report(
    report: &mut serde_json::Value,
    input: &PathBuf,
    is_l5: bool,
) -> Result<(), PcError> {
    if !is_l5 {
        return Ok(());
    }

    let mut inspection = security_inspect(input)?;
    match inspection
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
            if let Some(obj) = inspection.as_object_mut() {
                obj.insert(
                    "key_provider".to_string(),
                    serde_json::json!(key_provider::FILE_KEYRING_DEV),
                );
            }
        }
        _ => {
            vault_provider::apply_unbound_vault_binding(report);
            signature::apply_unsigned_dev_signature_state(report);
        }
    }
    inspection = attestation::mark_artifact_attestation_unavailable(inspection);
    release_manifest::apply_release_candidate_policy(report);
    if let Some(obj) = report.as_object_mut() {
        obj.insert(
            "attestation".to_string(),
            attestation::unavailable_attestation_block(),
        );
        obj.insert("security_inspection".to_string(), inspection);
    }
    Ok(())
}
