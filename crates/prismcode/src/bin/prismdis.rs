use clap::Parser;
use prismcode::binary_runtime::{disassemble_packet, PcError};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "prismdis", about = "Disassemble a PRISMCODE .pcode packet")]
struct Cli {
    input: PathBuf,
    #[arg(long)]
    out: Option<PathBuf>,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), PcError> {
    let args = Cli::parse();
    let text = disassemble_packet(&args.input, args.out.as_deref())?;
    if args.out.is_none() {
        print!("{text}");
    }
    Ok(())
}
