use clap::{Parser, Subcommand};
use prismcode::binary_runtime::{redact_export, PcError};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "prismredact", about = "PRISM redacted export gate")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Export {
        input: PathBuf,
        #[arg(long)]
        classification: String,
        #[arg(long)]
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
        Command::Export {
            input,
            classification,
            out,
        } => {
            let text = redact_export(&input, &classification, &out)?;
            print!("{text}");
        }
    }
    Ok(())
}
