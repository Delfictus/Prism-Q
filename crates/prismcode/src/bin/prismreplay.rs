use clap::{Parser, Subcommand};
use prismcode::binary_runtime::{verify_replay, PcError};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(
    name = "prismreplay",
    about = "Verify deterministic PRISMCODE replay outputs"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Verify {
        input: PathBuf,
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
    match Cli::parse().command {
        Command::Verify { input, out } => {
            let result = verify_replay(&input, out.as_deref())?;
            println!("{}", serde_json::to_string_pretty(&result)?);
            Ok(())
        }
    }
}
