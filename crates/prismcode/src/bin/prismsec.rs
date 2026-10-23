use clap::{Parser, Subcommand};
use prismcode::binary_runtime::{
    write_dev_encrypted_packet, write_prod_encrypted_packet, write_security_inspection, PcError,
};
use prismcode::model_governance::{
    inspect_model_registry_with_options, verify_model_airlock_with_options, ModelInspectOptions,
};
use prismcode::model_runtime_smoke::{run_model_runtime_smoke, ModelRuntimeSmokeRequest};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "prismsec", about = "PRISM Shield Kernel inspection")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Inspect {
        input: PathBuf,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    Audit {
        input: PathBuf,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    Classify {
        input: PathBuf,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    EncryptDev {
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    EncryptProd {
        input: PathBuf,
        #[arg(long)]
        keyring: PathBuf,
        #[arg(long)]
        public_key: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    ModelInspect {
        #[arg(long, default_value = "configs/prismcode_model_registry.json")]
        registry: PathBuf,
        #[arg(long, help = "Force full live BLAKE3 rehash of registered model files")]
        live_hash: bool,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    ModelAirlockVerify {
        artifact: Option<PathBuf>,
        #[arg(long, default_value = "configs/prismcode_model_registry.json")]
        registry: PathBuf,
        #[arg(long, help = "Force full live BLAKE3 rehash of registered model files")]
        live_hash: bool,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    ModelRuntimeSmoke {
        #[arg(long, default_value = "openai_gpt_oss_20b")]
        model_id: String,
        #[arg(long, default_value = "configs/prismcode_model_registry.json")]
        registry: PathBuf,
        #[arg(long, default_value = "configs/model/model_runtime_policy.toml")]
        policy: PathBuf,
        #[arg(long, help = "Force full live BLAKE3 rehash of registered model files")]
        _live_hash: bool,
        #[arg(long, default_value = "build/demo/model_runtime_smoke_report.json")]
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
        Command::Inspect { input, out }
        | Command::Audit { input, out }
        | Command::Classify { input, out } => {
            let report = write_security_inspection(&input, out.as_deref())?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::EncryptDev { input, out } => {
            let report = write_dev_encrypted_packet(&input, &out)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::EncryptProd {
            input,
            keyring,
            public_key,
            out,
        } => {
            let report = write_prod_encrypted_packet(&input, &out, &keyring, &public_key)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::ModelInspect {
            registry,
            live_hash,
            out,
        } => {
            let report = inspect_model_registry_with_options(
                &registry,
                ModelInspectOptions {
                    force_live_hash: live_hash,
                },
            )?;
            if let Some(path) = out {
                if let Some(parent) = path.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::write(&path, serde_json::to_vec_pretty(&report)?)?;
            }
            println!("{}", serde_json::to_string_pretty(&report)?);
            if report.status != "PASS" {
                return Err(PcError::Security(
                    "model registry inspection failed".to_string(),
                ));
            }
        }
        Command::ModelAirlockVerify {
            artifact,
            registry,
            live_hash,
            out,
        } => {
            let report = verify_model_airlock_with_options(
                &registry,
                artifact.as_deref(),
                ModelInspectOptions {
                    force_live_hash: live_hash,
                },
            )?;
            if let Some(path) = out {
                if let Some(parent) = path.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::write(&path, serde_json::to_vec_pretty(&report)?)?;
            }
            println!("{}", serde_json::to_string_pretty(&report)?);
            if report.status != "PASS" {
                return Err(PcError::Security(
                    "model airlock verification failed".to_string(),
                ));
            }
        }
        Command::ModelRuntimeSmoke {
            model_id,
            registry,
            policy,
            _live_hash,
            out,
        } => {
            let report = run_model_runtime_smoke(&ModelRuntimeSmokeRequest {
                registry,
                policy,
                model_id,
                out,
            })?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
    }
    Ok(())
}
