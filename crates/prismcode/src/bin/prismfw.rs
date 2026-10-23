use clap::{Parser, Subcommand};
use prismcode::binary_runtime::{PcError, PcResult};
use prismcode::manifest_signature::verify_artifact_report;
use prismcode::trust_root;
use serde_json::json;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Parser)]
#[command(name = "prismfw", about = "Installed PRISM firmware profile verifier")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Verify {
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
        Command::Verify { bundle } => {
            trust_root::require_release_pinned_root()?;
            let root = fs::canonicalize(bundle)?;
            let profile = root.join("firmware/active.prismfw.toml");
            let hash_path = root.join("firmware/active.prismfw.hash");
            let sig = root.join("firmware/active.prismfw.sig");
            require_nonempty(&profile)?;
            require_nonempty(&hash_path)?;
            require_nonempty(&sig)?;
            let expected = fs::read_to_string(&hash_path)?.trim().to_string();
            let actual = blake3::hash(&fs::read(&profile)?).to_hex().to_string();
            if expected != actual {
                return Err(PcError::Hash("firmware profile hash mismatch".to_string()));
            }
            let signature = verify_artifact_report(&profile, &sig)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "status": "PASS",
                    "check_id": "FIRMWARE_PROFILE_SIGNATURE",
                    "firmware_profile": profile.display().to_string(),
                    "firmware_profile_hash_blake3": actual,
                    "signature": signature,
                }))?
            );
        }
        Command::PinnedRoot => {
            println!("{}", serde_json::to_string_pretty(&trust_root::report())?);
        }
    }
    Ok(())
}

fn require_nonempty(path: &Path) -> PcResult<()> {
    let meta = fs::metadata(path)?;
    if meta.len() == 0 {
        return Err(PcError::Format(format!(
            "empty firmware artifact: {}",
            path.display()
        )));
    }
    Ok(())
}
