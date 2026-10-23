use clap::{Parser, Subcommand};
use prismcode::binary_runtime::{init_file_keyring, security_inspect, PcError};
use prismcode::key_provider;
use prismcode::key_provider_local_release_vault::{
    build_blocker_02_report, report_artifact, write_json, LocalReleaseVault,
};
use prismcode::release_key_policy::ReleaseKeyPolicy;
use prismcode::vault_provider;
use serde_json::json;
use std::path::{Path, PathBuf};

#[derive(Debug, Parser)]
#[command(name = "prismkey", about = "PRISM-VAULT key abstraction status")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    List,
    ProviderInspect {
        #[arg(long)]
        policy: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    WrapTest {
        #[arg(long)]
        policy: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    RotateTest {
        #[arg(long)]
        policy: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    SignTest {
        #[arg(long)]
        artifact: PathBuf,
        #[arg(long)]
        policy: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    InitFileKeyring {
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        public_out: PathBuf,
    },
    Rotate {
        #[arg(long)]
        scope: String,
        #[arg(long)]
        packet: PathBuf,
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
        Command::List => {
            println!(
                "{}",
                serde_json::to_string_pretty(&key_provider::provider_matrix())?
            );
        }
        Command::ProviderInspect { policy, out } => {
            let policy = ReleaseKeyPolicy::load(&policy)?;
            let vault = LocalReleaseVault::open_or_create(policy.clone())?;
            let mut report = vault.provider_report()?;
            attach_report_record(
                &mut report,
                &out,
                "cargo run -p prismcode --bin prismkey -- provider-inspect",
                &policy.policy_hash_blake3,
            );
            write_and_print(
                &out,
                &mut report,
                "cargo run -p prismcode --bin prismkey -- provider-inspect",
                &policy.policy_hash_blake3,
            )?;
        }
        Command::WrapTest { policy, out } => {
            let policy = ReleaseKeyPolicy::load(&policy)?;
            let vault = LocalReleaseVault::open_or_create(policy.clone())?;
            let mut report = vault.wrap_test_report()?;
            attach_report_record(
                &mut report,
                &out,
                "cargo run -p prismcode --bin prismkey -- wrap-test",
                &policy.policy_hash_blake3,
            );
            write_and_print(
                &out,
                &mut report,
                "cargo run -p prismcode --bin prismkey -- wrap-test",
                &policy.policy_hash_blake3,
            )?;
        }
        Command::RotateTest { policy, out } => {
            let policy = ReleaseKeyPolicy::load(&policy)?;
            let vault = LocalReleaseVault::open_or_create(policy.clone())?;
            let mut report = vault.rotate_test_report()?;
            attach_report_record(
                &mut report,
                &out,
                "cargo run -p prismcode --bin prismkey -- rotate-test",
                &policy.policy_hash_blake3,
            );
            write_and_print(
                &out,
                &mut report,
                "cargo run -p prismcode --bin prismkey -- rotate-test",
                &policy.policy_hash_blake3,
            )?;
        }
        Command::SignTest {
            artifact,
            policy,
            out,
        } => {
            let policy = ReleaseKeyPolicy::load(&policy)?;
            let vault = LocalReleaseVault::open_or_create(policy.clone())?;
            let mut report = vault.sign_test_report(&artifact)?;
            attach_report_record(
                &mut report,
                &out,
                "cargo run -p prismcode --bin prismkey -- sign-test",
                &policy.policy_hash_blake3,
            );
            write_and_print(
                &out,
                &mut report,
                "cargo run -p prismcode --bin prismkey -- sign-test",
                &policy.policy_hash_blake3,
            )?;
        }
        Command::InitFileKeyring { out, public_out } => {
            let mut report = init_file_keyring(&out, &public_out)?;
            key_provider::apply_file_keyring_dev_overlay(&mut report);
            vault_provider::apply_file_keyring_dev_binding(&mut report);
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::Rotate { scope, packet } => {
            let inspect = security_inspect(&packet)?;
            let mut report = json!({
                "status": "ROTATION_REQUEST_RECORDED",
                "scope": scope,
                "packet": packet.display().to_string(),
                "packet_hash_blake3": inspect["artifact_hash_blake3"],
                "raw_keys_exposed": false,
                "raw_keys_printed": false,
                "applied_to_ciphertext": false,
                "reason": "PRISM_VAULT provider is unbound; no raw key was created or emitted.",
            });
            vault_provider::apply_unbound_vault_binding(&mut report);
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
    }
    Ok(())
}

fn write_and_print(
    out: &Path,
    report: &mut serde_json::Value,
    producer: &str,
    policy_hash: &str,
) -> Result<(), PcError> {
    write_json(out, report)?;
    let record = report_artifact(out, producer, "PASS", policy_hash);
    if let (Some(obj), Some(record_obj)) = (report.as_object_mut(), record.as_object()) {
        for (key, value) in record_obj {
            obj.insert(key.clone(), value.clone());
        }
    }
    write_json(out, report)?;
    maybe_write_blocker_report()?;
    println!("{}", serde_json::to_string_pretty(report)?);
    Ok(())
}

fn attach_report_record(
    report: &mut serde_json::Value,
    out: &Path,
    producer: &str,
    policy_hash: &str,
) {
    if let Some(obj) = report.as_object_mut() {
        obj.insert("path".to_string(), json!(out.display().to_string()));
        let realpath = std::fs::canonicalize(out)
            .map(|path| path.display().to_string())
            .unwrap_or_else(|_| "PENDING_WRITE".to_string());
        obj.insert("realpath".to_string(), json!(realpath));
        obj.insert("byte_size".to_string(), json!(0));
        obj.insert("hash_algorithm".to_string(), json!("BLAKE3"));
        obj.insert("hash".to_string(), json!("PENDING_WRITE"));
        obj.insert("producer_command".to_string(), json!(producer));
        obj.insert("policy_hash".to_string(), json!(policy_hash));
    }
}

fn maybe_write_blocker_report() -> Result<(), PcError> {
    let reports = [
        (
            "cargo run -p prismcode --bin prismkey -- provider-inspect",
            Path::new("build/security/key_provider_report.json"),
        ),
        (
            "cargo run -p prismcode --bin prismkey -- wrap-test",
            Path::new("build/security/vault_wrap_test_report.json"),
        ),
        (
            "cargo run -p prismcode --bin prismkey -- rotate-test",
            Path::new("build/security/vault_rotation_report.json"),
        ),
        (
            "cargo run -p prismcode --bin prismkey -- sign-test",
            Path::new("build/security/vault_sign_test_report.json"),
        ),
        (
            "cargo run -p prismcode --bin prismseal -- sign-artifact",
            Path::new("build/security/cuda_manifest_signature_report.json"),
        ),
        (
            "cargo run -p prismcode --bin prismseal -- verify-artifact",
            Path::new("build/security/cuda_manifest_verify_report.json"),
        ),
    ];
    if reports.iter().all(|(_, path)| path.exists()) {
        let policy_hash = std::fs::read_to_string("configs/security/release_key_policy.toml")
            .map(|text| blake3::hash(text.as_bytes()).to_hex().to_string())
            .unwrap_or_else(|_| "EVIDENCE_NOT_FOUND".to_string());
        let report = build_blocker_02_report(&reports, &policy_hash);
        write_json(
            Path::new("build/security/vault_blocker_02_report.json"),
            &report,
        )?;
    }
    Ok(())
}
