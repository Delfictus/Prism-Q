use clap::{Parser, Subcommand};
use prismcode::binary_runtime::{write_hardware_report, PcError};
use prismcode::cuda_backend::cuda_selftest_report;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "prismhw", about = "Inspect PRISMCODE runtime hardware targets")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Inspect {
        #[arg(long)]
        cuda: bool,
        #[arg(long = "json")]
        json_path: Option<PathBuf>,
    },
    CudaSelftest {
        #[arg(long = "json")]
        json_path: Option<PathBuf>,
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
        Command::Inspect {
            cuda: _cuda,
            json_path,
        } => {
            let value = write_hardware_report(json_path.as_deref())?;
            println!("{}", serde_json::to_string_pretty(&value)?);
            Ok(())
        }
        Command::CudaSelftest { json_path } => {
            let value = cuda_selftest_report();
            if let Some(path) = json_path {
                if let Some(parent) = path.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::write(&path, serde_json::to_vec_pretty(&value)?)?;
            }
            println!("{}", serde_json::to_string_pretty(&value)?);
            require_successful_selftest(&value)
        }
    }
}

fn require_successful_selftest(value: &serde_json::Value) -> Result<(), PcError> {
    let status = value["cuda_execution_status"].as_str().unwrap_or("MISSING_SELFTEST_STATUS");
    if matches!(status, "LINKED_SIGNED_SELFTEST_READY" | "LINKED_DEV_HASH_BOUND_SELFTEST_READY")
        && value["cuda_device_observation_performed"] == true
    {
        Ok(())
    } else if status == "FAIL_CLOSED_CUDA_OPERATION" {
        Err(PcError::Gpu(format!("CUDA operation failed: {}",
            value["cuda_operation_error"].as_str().unwrap_or("operation failed"))))
    } else {
        Err(PcError::Gpu(format!("CUDA diagnostic did not establish a trusted passing self-test: {status}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selftest_exit_status_requires_observed_trusted_success() {
        for status in ["FAIL_CLOSED_CUDA_OPERATION", "FAIL_CLOSED_UNLINKED",
            "NOT_OBSERVED_PASSIVE_METADATA", "FAIL_CLOSED_SELFTEST_FAILED"] {
            assert!(require_successful_selftest(&serde_json::json!({"cuda_execution_status":status,
                "cuda_device_observation_performed":false})).is_err());
        }
        for status in ["LINKED_SIGNED_SELFTEST_READY", "LINKED_DEV_HASH_BOUND_SELFTEST_READY"] {
            assert!(require_successful_selftest(&serde_json::json!({"cuda_execution_status":status,
                "cuda_device_observation_performed":true})).is_ok());
            assert!(require_successful_selftest(&serde_json::json!({"cuda_execution_status":status,
                "cuda_device_observation_performed":false})).is_err());
        }
    }
}
