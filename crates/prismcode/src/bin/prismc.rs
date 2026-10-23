use clap::{Parser, Subcommand};
use prismcode::binary_runtime::{
    compile_reference_json, compression_audit, export_debug_json, migrate_packet, PcError,
};
use prismcode::cse_hd::{compile_scientific_state, write_cse_hd_sidecars, NanlMode};
use prismcode::nanl_dna::write_nanl_demo_artifacts;
use prismcode::nanl_hd_bench::write_nanl_hd_artifacts;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "prismc", about = "PRISMCODE ISA v0.1 compiler / assembler")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Compile {
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    Migrate {
        input: PathBuf,
        #[arg(long = "to-isa")]
        to_isa: String,
        #[arg(long)]
        out: PathBuf,
    },
    ExportDebug {
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    CompressionAudit {
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    CompileScientificState {
        input: PathBuf,
        #[arg(long = "cse-mode", default_value = "hd")]
        cse_mode: String,
        #[arg(long = "nanl-mode", default_value = "bin")]
        nanl_mode: String,
        #[arg(long)]
        out: PathBuf,
    },
    CseHdSidecars {
        input: PathBuf,
        #[arg(long = "out-dir")]
        out_dir: PathBuf,
    },
    NanlDemo {
        #[arg(long, default_value = "build/demo/nanl")]
        out_dir: PathBuf,
    },
    NanlHdDemo {
        #[arg(long, default_value = "build/nanl_hd")]
        out_dir: PathBuf,
        #[arg(long)]
        input: Option<PathBuf>,
    },
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(exit_code(&err));
    }
}

fn run() -> Result<(), PcError> {
    match Cli::parse().command {
        Command::Compile { input, out } => compile_reference_json(&input, &out),
        Command::Migrate { input, to_isa, out } => migrate_packet(&input, &out, &to_isa),
        Command::ExportDebug { input, out } => export_debug_json(&input, &out),
        Command::CompressionAudit { input, out } => {
            let report = compression_audit(&input, Some(&out))?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
        Command::CompileScientificState {
            input,
            cse_mode,
            nanl_mode,
            out,
        } => {
            let nanl_mode = NanlMode::parse(&nanl_mode)?;
            let report = compile_scientific_state(&input, &cse_mode, nanl_mode, &out)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
        Command::CseHdSidecars { input, out_dir } => {
            let report = write_cse_hd_sidecars(&input, &out_dir)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
        Command::NanlDemo { out_dir } => {
            let report = write_nanl_demo_artifacts(&out_dir)
                .map_err(|err| PcError::Format(err.to_string()))?;
            // NanlDemoReport embeds the serde-free QNANL codec report; use its
            // first-class serde-free serializer instead of serde_json.
            print!("{}", report.to_json_pretty());
            Ok(())
        }
        Command::NanlHdDemo { out_dir, input } => {
            let payload = if let Some(input) = input {
                std::fs::read(input)?
            } else {
                b"PRISM NANL-HD canonical payload: molecule protein water time event uncertainty thermo teacher gflownet nrf drf".to_vec()
            };
            let report = write_nanl_hd_artifacts(&out_dir, &payload)
                .map_err(|err| PcError::Format(err.to_string()))?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
    }
}

fn exit_code(err: &PcError) -> i32 {
    match err {
        PcError::Gpu(_) => 12,
        PcError::Hash(_) => 13,
        PcError::Replay(_) => 14,
        _ => 2,
    }
}
