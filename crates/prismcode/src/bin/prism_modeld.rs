use clap::{Parser, Subcommand};
use prismcode::binary_runtime::PcError;
use prismcode::model_runtime::modeld_serve_once;
use prismcode::model_runtime_champion::{champion_query, ChampionEndpoint};
use serde_json::json;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "prism-modeld", about = "Bounded PRISM advisory model runtime")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Governed attestation handshake (registry + policy + nonce). Advisory only.
    ServeOnce {
        #[arg(long)]
        registry: PathBuf,
        #[arg(long)]
        policy: PathBuf,
        #[arg(long)]
        model_id: String,
        #[arg(long)]
        nonce: String,
        #[arg(long)]
        ready: PathBuf,
        #[arg(long)]
        report: PathBuf,
    },
    /// Governed one-shot query against the champion physics Oracle (attach to the
    /// already-running devbox server; advisory envelope, no canonical mutation).
    /// Prints the advisory-envelope JSON to stdout for the PrismOS UI adapter.
    Query {
        #[arg(long)]
        model_id: String,
        /// `predict` (physics reconstruction) or `generate`.
        #[arg(long, default_value = "predict")]
        endpoint: String,
        /// Canonical SMILES to score.
        #[arg(long)]
        smiles: Option<String>,
        /// Comma-separated raw u64 PRISM context tokens.
        #[arg(long)]
        u64_context: Option<String>,
        #[arg(long, default_value_t = 5)]
        top_k: u32,
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
        Command::ServeOnce {
            registry,
            policy,
            model_id,
            nonce,
            ready,
            report,
        } => modeld_serve_once(&registry, &policy, &model_id, &nonce, &ready, &report),
        Command::Query {
            model_id,
            endpoint,
            smiles,
            u64_context,
            top_k,
        } => {
            let ep = match endpoint.as_str() {
                "generate" => ChampionEndpoint::Generate,
                _ => ChampionEndpoint::Predict,
            };
            let request = if let Some(s) = smiles {
                json!({ "smiles": s, "top_k": top_k })
            } else if let Some(ctx) = u64_context {
                let toks: Vec<i64> = ctx
                    .split(',')
                    .filter_map(|x| x.trim().parse::<i64>().ok())
                    .collect();
                json!({ "u64_context": toks, "top_k": top_k })
            } else {
                return Err(PcError::Security(
                    "query requires --smiles or --u64-context".to_string(),
                ));
            };
            let out = champion_query(&model_id, ep, &request)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&out).unwrap_or_else(|_| out.to_string())
            );
            Ok(())
        }
    }
}
