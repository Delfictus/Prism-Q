//! Gate G1 — CLI driver for the multi-seed Arrow merger.
//!
//! Combines N per-seed `<prefix>.topology.spike_events.arrow` files
//! into a single multi-seed Arrow IPC stream and emits a sidecar
//! merge-stats JSON for the dossier script's `replica_axis_active`
//! check.
//!
//! Usage:
//!     prism-multi-seed-merge \
//!         --output <merged.arrow> \
//!         <input1.arrow> <input2.arrow> [<input3.arrow>...]
//!
//! Exits 0 on success, non-zero on any merge error.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use prism_nhs::multi_seed_merger::{merge_arrow_files, MergeStats};

#[derive(Parser, Debug)]
#[command(
    name = "prism-multi-seed-merge",
    about = "Gate G1 — concatenate multiple per-seed spike_events.arrow files into a single multi-seed stream"
)]
struct Args {
    /// Output Arrow IPC file path. Will be created (overwritten if it exists).
    #[arg(short, long)]
    output: PathBuf,

    /// Optional sidecar JSON path for merge stats. If omitted, defaults to
    /// `<output>.merge_stats.json` next to the merged Arrow.
    #[arg(long)]
    stats_json: Option<PathBuf>,

    /// One or more input Arrow files. Order is preserved in the output.
    #[arg(required = true)]
    inputs: Vec<PathBuf>,
}

fn write_stats_json(
    path: &PathBuf,
    stats: &MergeStats,
    inputs: &[PathBuf],
    output: &PathBuf,
) -> std::io::Result<()> {
    use std::io::Write;
    let mut buf = String::new();
    buf.push_str("{\n");
    buf.push_str("  \"schema_version\": \"1.0.0\",\n");
    buf.push_str("  \"generator\": \"prism-multi-seed-merge\",\n");
    buf.push_str(&format!("  \"output\": \"{}\",\n", output.display()));
    buf.push_str("  \"inputs\": [\n");
    for (i, p) in inputs.iter().enumerate() {
        let comma = if i + 1 == inputs.len() { "" } else { "," };
        buf.push_str(&format!("    \"{}\"{}\n", p.display(), comma));
    }
    buf.push_str("  ],\n");
    buf.push_str(&format!("  \"total_rows\": {},\n", stats.total_rows));
    buf.push_str(&format!("  \"total_batches\": {},\n", stats.total_batches));
    buf.push_str(&format!(
        "  \"replica_axis_active\": {},\n",
        stats.replica_axis_active()
    ));
    buf.push_str("  \"distinct_seeds\": [");
    for (i, s) in stats.distinct_seeds.iter().enumerate() {
        let comma = if i + 1 == stats.distinct_seeds.len() {
            ""
        } else {
            ", "
        };
        buf.push_str(&format!("{}{}", s, comma));
    }
    buf.push_str("],\n");
    buf.push_str("  \"per_seed_rows\": {");
    let n = stats.per_seed_rows.len();
    for (i, (k, v)) in stats.per_seed_rows.iter().enumerate() {
        let comma = if i + 1 == n { "" } else { ", " };
        buf.push_str(&format!("\"{}\": {}{}", k, v, comma));
    }
    buf.push_str("}\n");
    buf.push_str("}\n");

    let mut f = std::fs::File::create(path)?;
    f.write_all(buf.as_bytes())?;
    Ok(())
}

fn main() -> ExitCode {
    let args = Args::parse();

    if args.inputs.is_empty() {
        eprintln!("error: at least one input Arrow file is required");
        return ExitCode::from(2);
    }

    let stats_path = args
        .stats_json
        .clone()
        .unwrap_or_else(|| args.output.with_extension("merge_stats.json"));

    let stats = match merge_arrow_files(&args.inputs, &args.output) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("merge failed: {e}");
            return ExitCode::from(1);
        }
    };

    if let Err(e) = write_stats_json(&stats_path, &stats, &args.inputs, &args.output) {
        eprintln!(
            "warn: failed to write stats JSON {}: {}",
            stats_path.display(),
            e
        );
        // Not a fatal error — the merged Arrow file was produced.
    }

    println!("Merge OK:");
    println!("  inputs               : {}", args.inputs.len());
    println!("  output               : {}", args.output.display());
    println!("  stats_json           : {}", stats_path.display());
    println!("  total_rows           : {}", stats.total_rows);
    println!("  total_batches        : {}", stats.total_batches);
    println!("  distinct_seeds       : {:?}", stats.distinct_seeds);
    println!("  replica_axis_active  : {}", stats.replica_axis_active());
    for (seed, rows) in &stats.per_seed_rows {
        println!("    seed={:>5}  rows={}", seed, rows);
    }

    ExitCode::SUCCESS
}
