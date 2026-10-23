//! Canonical deterministic PrismCode MD-policy normalizer.
//!
//! The tracked packet bytes are the source artifact.  This tool preserves all
//! non-policy section bytes and non-policy instructions, inserts the typed
//! pre-run MD launch policy, and refuses in-place writes.

use clap::Parser;
use prismcode::binary_runtime::normalize_md_engine_policy_packet;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "prismcode-packet-policy")]
struct Args {
    /// Existing hash-valid/L3-valid PrismCode packet.
    #[arg(long)]
    input: PathBuf,
    /// Distinct output path for the normalized packet.
    #[arg(long)]
    output: PathBuf,
    /// Machine-readable normalization receipt.
    #[arg(long)]
    report: PathBuf,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.input == args.output || args.input == args.report || args.output == args.report {
        return Err("input, output, and report paths must be distinct".into());
    }
    if args.output.exists() || args.report.exists() {
        return Err("output and report must not already exist".into());
    }
    let report = normalize_md_engine_policy_packet(&args.input, &args.output)?;
    if let Some(parent) = args.report.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut report_file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.report)?;
    report_file.write_all(&serde_json::to_vec_pretty(&report)?)?;
    report_file.sync_all()?;
    if let Some(parent) = args.report.parent() {
        fs::File::open(parent)?.sync_all()?;
    }
    println!("{}", serde_json::to_string(&report)?);
    Ok(())
}
