//! PRISM-Zero v3.1 Training Binary
//!
//! This binary implements the training pipeline that automatically calibrates
//! molecular dynamics parameters using reinforcement learning.

use anyhow::{Context, Result};
use clap::Parser;
use log::{error, info};
use std::path::Path;

#[cfg(feature = "rl")]
use prism_learning::{PrismTrainer, TrainingConfig};

/// Command line arguments for PRISM training
#[derive(Parser)]
#[command(name = "prism-train")]
#[command(about = "PRISM-Zero v3.1: Self-calibrating molecular dynamics engine")]
#[command(version = env!("CARGO_PKG_VERSION"))]
struct Args {
    /// Path to calibration manifest JSON file
    #[arg(short, long)]
    manifest: String,

    /// Maximum episodes per target
    #[arg(long, default_value = "1000")]
    max_episodes: usize,

    /// Output directory for checkpoints and results
    #[arg(short, long, default_value = "training_output")]
    output: String,

    /// CUDA device ID to use
    #[arg(long, default_value = "0")]
    device: usize,

    /// Checkpoint interval (episodes)
    #[arg(long, default_value = "100")]
    checkpoint_interval: usize,

    /// Early stopping patience (episodes)
    #[arg(long, default_value = "50")]
    patience: usize,

    /// Target reward for early stopping
    #[arg(long)]
    target_reward: Option<f32>,

    /// Enable Hyper-Q parallel training (multi-stream CUDA)
    #[arg(long)]
    parallel: bool,

    /// Number of parallel CUDA streams (Hyper-Q jobs)
    #[arg(long, default_value = "4")]
    parallel_jobs: usize,

    /// Enable macro-step training (chunked episodes)
    #[arg(long)]
    macro_steps: bool,

    /// Verbose logging
    #[arg(short, long)]
    verbose: bool,
}

#[cfg(feature = "rl")]
fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging
    let log_level = if args.verbose { "debug" } else { "info" };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(log_level))
        .format_timestamp_secs()
        .format_module_path(false)
        .init();

    // Print header
    println!(
        "🧬 PRISM-Zero v{} Training Engine",
        prism_learning::PRISM_ZERO_VERSION
    );
    println!("🚀 Self-calibrating molecular dynamics with reinforcement learning");
    println!("{}", "=".repeat(70));

    // Validate inputs
    validate_inputs(&args)?;

    // Create training configuration
    let config = TrainingConfig {
        max_episodes: args.max_episodes,
        checkpoint_interval: args.checkpoint_interval,
        early_stopping_patience: args.patience,
        target_reward: args.target_reward,
        output_dir: args.output.clone(),
        parallel_targets: args.parallel,
        parallel_jobs: args.parallel_jobs,
        use_macro_steps: args.macro_steps,
    };

    info!("Configuration:");
    info!("  Manifest: {}", args.manifest);
    info!("  Max episodes per target: {}", config.max_episodes);
    info!("  Output directory: {}", config.output_dir);
    info!("  CUDA device: {}", args.device);
    info!("  Macro-step training: {}", config.use_macro_steps);
    if let Some(reward) = config.target_reward {
        info!("  Target reward: {:.3}", reward);
    }
    if config.parallel_targets {
        info!(
            "  ⚡ HYPER-Q PARALLEL: {} CUDA streams",
            config.parallel_jobs
        );
    }
    println!();

    // Initialize trainer
    info!("Initializing PRISM trainer...");
    let mut trainer = PrismTrainer::new(&args.manifest, config, args.device)
        .context("Failed to initialize PRISM trainer")?;

    // Run training
    info!("🏃 Starting training pipeline...");
    let training_start = std::time::Instant::now();

    match trainer.train_all_targets() {
        Ok(()) => {
            let training_time = training_start.elapsed();
            let stats = trainer.get_stats();

            println!("\n✅ Training completed successfully!");
            println!("📊 Final Statistics:");
            println!("   Targets completed: {}", stats["targets_completed"]);
            println!("   Total episodes: {}", stats["total_episodes"]);
            println!(
                "   Total simulation steps: {}",
                stats["total_simulation_steps"]
            );
            println!("   Total transitions: {}", stats["total_transitions"]);
            println!(
                "   Average best reward: {:.3}",
                stats["average_best_reward"]
            );
            println!(
                "   Best overall reward: {:.3}",
                stats["best_overall_reward"]
            );
            println!(
                "   Training time: {:.1} minutes",
                training_time.as_secs_f32() / 60.0
            );
            println!(
                "\n📁 Results saved to: {}/training_results.json",
                args.output
            );

            Ok(())
        }
        Err(e) => {
            error!("Training failed: {}", e);
            error!("Check logs for detailed error information");
            std::process::exit(1);
        }
    }
}

#[cfg(not(feature = "rl"))]
fn main() -> Result<()> {
    eprintln!("Error: prism-train requires the 'rl' feature to be enabled");
    eprintln!("Rebuild with: cargo build --features rl");
    std::process::exit(1);
}

fn validate_inputs(args: &Args) -> Result<()> {
    // Check manifest file exists
    if !Path::new(&args.manifest).exists() {
        anyhow::bail!("Manifest file not found: {}", args.manifest);
    }

    // Validate manifest is valid JSON
    let manifest_content =
        std::fs::read_to_string(&args.manifest).context("Failed to read manifest file")?;

    serde_json::from_str::<serde_json::Value>(&manifest_content)
        .context("Manifest file contains invalid JSON")?;

    // Validate parameter ranges
    if args.max_episodes == 0 {
        anyhow::bail!("max_episodes must be greater than 0");
    }

    if args.checkpoint_interval > args.max_episodes {
        anyhow::bail!("checkpoint_interval cannot be larger than max_episodes");
    }

    if let Some(reward) = args.target_reward {
        if !reward.is_finite() {
            anyhow::bail!("target_reward must be a finite number");
        }
    }

    // Create output directory if it doesn't exist
    std::fs::create_dir_all(&args.output)
        .with_context(|| format!("Failed to create output directory: {}", args.output))?;

    Ok(())
}
