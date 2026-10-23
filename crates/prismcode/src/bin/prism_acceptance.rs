use clap::{Parser, Subcommand};
use prismcode::acceptance::{run_acceptance, AcceptancePhase};
use prismcode::binary_runtime::PcError;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(
    name = "prism-acceptance",
    about = "Installed PRISM OS/Firmware acceptance harness"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Run {
        #[arg(long, default_value = "full")]
        phase: String,
        #[arg(long, default_value = ".")]
        bundle: PathBuf,
        #[arg(long)]
        explain_hermeticity: bool,
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
        Command::Run {
            phase,
            bundle,
            explain_hermeticity,
        } => {
            let phase = AcceptancePhase::parse(&phase)?;
            let report = run_acceptance(&bundle, phase)?;
            if explain_hermeticity {
                eprintln!(
                    "hermeticity: child validators run by absolute bundled paths, environment is scrubbed, tamper tests use NAS ephemeral copies, and pinned roots are compiled into verifier binaries"
                );
            }
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
    }
    Ok(())
}
