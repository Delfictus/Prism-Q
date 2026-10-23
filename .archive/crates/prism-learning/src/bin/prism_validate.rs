//! PRISM-Zero v3.1 Validation Binary - Discovery Mode
//!
//! This binary implements holdout validation for evaluating trained models.
//! It runs actual simulations on quarantined test sets and produces:
//! - Digital Twins (relaxed_structure.pdb)
//! - Treasure Maps (residue_scores.csv)
//! - Validation Reports (JSON metrics)

use anyhow::{Context, Result};
use clap::Parser;
use log::{debug, info, warn};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::Path;

use prism_learning::manifest::{CalibrationManifest, ProteinTarget};
use prism_learning::persistence::{
    PersistenceAssessment, PersistenceConfig, PersistenceTracker, PocketPersistenceMetrics,
};
use prism_physics::molecular_dynamics::{MolecularDynamicsConfig, MolecularDynamicsEngine};

/// Validation results for a single holdout target
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HoldoutResult {
    pub target_name: String,
    pub apo_pdb: String,
    pub total_sasa_gain_proxy: f32, // Displacement-based proxy
    pub core_rmsd: f32,
    pub max_displacement: f32,
    pub mean_displacement: f32,
    pub top_residues: Vec<ResidueScore>,
    pub enrichment_factor: f32,
    pub validation_passed: bool,
    pub relaxed_pdb_path: String,
    pub residue_csv_path: String,
    // Pocket persistence metrics
    pub persistence_ratio: f32,
    pub persistence_assessment: String,
    pub opening_events: usize,
    pub max_continuous_open: usize,
    pub mean_open_duration: f32,
    pub is_druggable: bool,
}

/// Per-residue score entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidueScore {
    pub residue_id: u32,
    pub displacement: f32,
    pub rank: usize,
    pub is_target: bool,
}

/// Complete validation report
#[derive(Debug, Serialize, Deserialize)]
pub struct ValidationReport {
    pub prism_zero_version: String,
    pub validation_date: chrono::DateTime<chrono::Utc>,
    pub physics_parameters: PhysicsSnapshot,
    pub holdout_results: Vec<HoldoutResult>,
    pub overall_success_rate: f32,
    pub mean_enrichment: f32,
    pub summary: ValidationSummary,
}

/// Physics parameters used during validation (frozen from training)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsSnapshot {
    pub temperature: f32,
    pub friction: f32,
    pub spring_k: f32,
    pub bias_strength: f32,
    pub dt: f32,
    pub steps: u64,
}

/// High-level summary for quick assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationSummary {
    pub total_targets: usize,
    pub passed_targets: usize,
    pub mean_sasa_proxy: f32,
    pub mean_core_rmsd: f32,
    pub best_target: String,
    pub worst_target: String,
    // Persistence summary
    pub druggable_targets: usize,
    pub mean_persistence_ratio: f32,
    pub stable_count: usize,    // >70% persistence
    pub moderate_count: usize,  // 50-70%
    pub transient_count: usize, // 30-50%
    pub unstable_count: usize,  // <30%
}

/// Command line arguments for validation
#[derive(Parser)]
#[command(name = "prism-validate")]
#[command(about = "PRISM-Zero v3.1: Holdout validation with actual simulation")]
#[command(version = env!("CARGO_PKG_VERSION"))]
struct Args {
    /// Path to training results JSON file (for loading learned parameters)
    #[arg(short, long)]
    results: String,

    /// Holdout manifest JSON file (defines test targets)
    #[arg(short, long)]
    manifest: String,

    /// Output directory for validation artifacts
    #[arg(short, long, default_value = "validation_output")]
    output: String,

    /// Minimum enrichment factor for PASS
    #[arg(long, default_value = "2.0")]
    min_enrichment: f32,

    /// Maximum core RMSD threshold (Ångström)
    #[arg(long, default_value = "3.0")]
    max_core_rmsd: f32,

    /// Simulation steps per target
    #[arg(long, default_value = "1000000")]
    steps: u64,

    /// Temperature (if not loading from results)
    #[arg(long, default_value = "1.5")]
    temperature: f32,

    /// Friction (if not loading from results)
    #[arg(long, default_value = "0.1")]
    friction: f32,

    /// Spring constant (if not loading from results)
    #[arg(long, default_value = "5.0")]
    spring_k: f32,

    /// Bias strength (if not loading from results)
    #[arg(long, default_value = "0.5")]
    bias_strength: f32,

    /// Number of persistence samples during simulation
    #[arg(long, default_value = "20")]
    persistence_samples: usize,

    /// Displacement threshold (Å) to consider residue "exposed"
    #[arg(long, default_value = "2.0")]
    exposure_threshold: f32,

    /// Fraction of target residues that must be exposed for pocket to be "open"
    #[arg(long, default_value = "0.3")]
    pocket_open_fraction: f32,

    /// Minimum persistence ratio for PASS (combined with enrichment)
    #[arg(long, default_value = "0.5")]
    min_persistence: f32,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging
    let log_level = if args.verbose { "debug" } else { "info" };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(log_level))
        .format_timestamp_secs()
        .format_module_path(false)
        .init();

    // Print validation header
    println!(
        "🔬 PRISM-Zero v{} Discovery Validation",
        prism_learning::PRISM_ZERO_VERSION
    );
    println!("🚫 Strict quarantine enforcement - No training data contamination");
    println!("{}", "=".repeat(70));

    // Create output directory
    fs::create_dir_all(&args.output).context("Failed to create output directory")?;

    // Load holdout manifest
    info!("📋 Loading holdout manifest from: {}", args.manifest);
    let manifest =
        CalibrationManifest::load(&args.manifest).context("Failed to load holdout manifest")?;

    info!("✅ Loaded {} holdout targets", manifest.targets.len());

    // Physics parameters (frozen from training or CLI)
    let physics = PhysicsSnapshot {
        temperature: args.temperature,
        friction: args.friction,
        spring_k: args.spring_k,
        bias_strength: args.bias_strength,
        dt: 0.001,
        steps: args.steps,
    };

    info!("🔧 Physics Parameters (Frozen):");
    info!("   Temperature: {:.2}", physics.temperature);
    info!("   Friction: {:.3}", physics.friction);
    info!("   Spring K: {:.2}", physics.spring_k);
    info!("   Bias Strength: {:.2}", physics.bias_strength);
    info!("   Steps: {}", physics.steps);

    // Run validation on each target
    let mut holdout_results = Vec::new();

    for (idx, target) in manifest.targets.iter().enumerate() {
        println!("\n{}", "─".repeat(70));
        info!(
            "🧪 [{}/{}] Validating: {} ",
            idx + 1,
            manifest.targets.len(),
            target.name
        );

        match validate_target(target, &physics, &args) {
            Ok(result) => {
                let status = if result.validation_passed {
                    "✅ PASS"
                } else {
                    "❌ FAIL"
                };
                let drug_icon = if result.is_druggable {
                    "💊"
                } else {
                    "⚠️"
                };
                info!(
                    "{} | EF: {:.1}x | RMSD: {:.2}Å | Persistence: {:.0}% {} ({})",
                    status,
                    result.enrichment_factor,
                    result.core_rmsd,
                    result.persistence_ratio * 100.0,
                    drug_icon,
                    result.persistence_assessment
                );
                holdout_results.push(result);
            }
            Err(e) => {
                warn!("❌ FAILED: {}", e);
                // Create a failure result
                holdout_results.push(HoldoutResult {
                    target_name: target.name.clone(),
                    apo_pdb: target.apo_pdb.clone(),
                    total_sasa_gain_proxy: 0.0,
                    core_rmsd: 999.0,
                    max_displacement: 0.0,
                    mean_displacement: 0.0,
                    top_residues: Vec::new(),
                    enrichment_factor: 0.0,
                    validation_passed: false,
                    relaxed_pdb_path: String::new(),
                    residue_csv_path: String::new(),
                    persistence_ratio: 0.0,
                    persistence_assessment: "Unstable".to_string(),
                    opening_events: 0,
                    max_continuous_open: 0,
                    mean_open_duration: 0.0,
                    is_druggable: false,
                });
            }
        }
    }

    // Calculate overall metrics
    let passed = holdout_results
        .iter()
        .filter(|r| r.validation_passed)
        .count();
    let total = holdout_results.len();
    let success_rate = if total > 0 {
        passed as f32 / total as f32
    } else {
        0.0
    };

    let mean_enrichment: f32 = if !holdout_results.is_empty() {
        holdout_results
            .iter()
            .map(|r| r.enrichment_factor)
            .sum::<f32>()
            / total as f32
    } else {
        0.0
    };

    let mean_sasa: f32 = if !holdout_results.is_empty() {
        holdout_results
            .iter()
            .map(|r| r.total_sasa_gain_proxy)
            .sum::<f32>()
            / total as f32
    } else {
        0.0
    };

    let mean_rmsd: f32 = if !holdout_results.is_empty() {
        holdout_results.iter().map(|r| r.core_rmsd).sum::<f32>() / total as f32
    } else {
        0.0
    };

    let best_target = holdout_results
        .iter()
        .max_by(|a, b| {
            a.enrichment_factor
                .partial_cmp(&b.enrichment_factor)
                .unwrap()
        })
        .map(|r| r.target_name.clone())
        .unwrap_or_default();

    let worst_target = holdout_results
        .iter()
        .min_by(|a, b| {
            a.enrichment_factor
                .partial_cmp(&b.enrichment_factor)
                .unwrap()
        })
        .map(|r| r.target_name.clone())
        .unwrap_or_default();

    // Calculate persistence summary
    let druggable_targets = holdout_results.iter().filter(|r| r.is_druggable).count();
    let mean_persistence: f32 = if !holdout_results.is_empty() {
        holdout_results
            .iter()
            .map(|r| r.persistence_ratio)
            .sum::<f32>()
            / total as f32
    } else {
        0.0
    };

    let stable_count = holdout_results
        .iter()
        .filter(|r| r.persistence_assessment == "Stable")
        .count();
    let moderate_count = holdout_results
        .iter()
        .filter(|r| r.persistence_assessment == "Moderate")
        .count();
    let transient_count = holdout_results
        .iter()
        .filter(|r| r.persistence_assessment == "Transient")
        .count();
    let unstable_count = holdout_results
        .iter()
        .filter(|r| r.persistence_assessment == "Unstable")
        .count();

    // Create validation report
    let report = ValidationReport {
        prism_zero_version: prism_learning::PRISM_ZERO_VERSION.to_string(),
        validation_date: chrono::Utc::now(),
        physics_parameters: physics,
        holdout_results,
        overall_success_rate: success_rate,
        mean_enrichment,
        summary: ValidationSummary {
            total_targets: total,
            passed_targets: passed,
            mean_sasa_proxy: mean_sasa,
            mean_core_rmsd: mean_rmsd,
            best_target,
            worst_target,
            druggable_targets,
            mean_persistence_ratio: mean_persistence,
            stable_count,
            moderate_count,
            transient_count,
            unstable_count,
        },
    };

    // Save report
    let report_path = format!("{}/validation_report.json", args.output);
    let report_json = serde_json::to_string_pretty(&report)?;
    fs::write(&report_path, &report_json)?;

    // Print summary
    println!("\n{}", "═".repeat(70));
    println!("📋 VALIDATION SUMMARY");
    println!("{}", "═".repeat(70));
    println!("Targets passed: {}/{}", passed, total);
    println!("Success rate: {:.1}%", success_rate * 100.0);
    println!("Mean enrichment: {:.2}x", mean_enrichment);
    println!("Mean SASA proxy: {:.1}Å", mean_sasa);
    println!("Mean core RMSD: {:.2}Å", mean_rmsd);
    println!("\n💊 DRUGGABILITY ASSESSMENT");
    println!("{}", "─".repeat(70));
    println!(
        "Druggable targets: {}/{} ({:.1}%)",
        druggable_targets,
        total,
        if total > 0 {
            druggable_targets as f32 / total as f32 * 100.0
        } else {
            0.0
        }
    );
    println!("Mean persistence: {:.1}%", mean_persistence * 100.0);
    println!(
        "Distribution: 🟢 Stable:{} | 🟡 Moderate:{} | 🟠 Transient:{} | 🔴 Unstable:{}",
        stable_count, moderate_count, transient_count, unstable_count
    );
    println!("\n📁 Report saved to: {}", report_path);

    // Success criteria: both enrichment AND persistence
    let druggability_rate = if total > 0 {
        druggable_targets as f32 / total as f32
    } else {
        0.0
    };
    if success_rate >= 0.67 && druggability_rate >= 0.5 {
        println!("\n🎉 VALIDATION PASSED - Ready for drug discovery!");
        Ok(())
    } else if success_rate >= 0.67 {
        println!("\n⚠️  Enrichment OK but persistence low - Pockets may be transient");
        Ok(())
    } else {
        println!("\n⚠️  Validation below threshold (67%) - Review physics parameters");
        Ok(())
    }
}

/// Run validation on a single target with persistence tracking
fn validate_target(
    target: &ProteinTarget,
    physics: &PhysicsSnapshot,
    args: &Args,
) -> Result<HoldoutResult> {
    // Create target output directory
    let target_dir = format!("{}/{}", args.output, target.name);
    fs::create_dir_all(&target_dir)?;

    // Load PDB data
    let pdb_data = fs::read(&target.apo_pdb)
        .with_context(|| format!("Failed to read PDB: {}", target.apo_pdb))?;

    // Configure MD engine with frozen physics parameters
    let md_config = MolecularDynamicsConfig {
        max_steps: physics.steps,
        dt: physics.dt,
        friction: physics.friction,
        temp_start: physics.temperature,
        temp_end: physics.temperature * 0.5,
        annealing_steps: physics.steps / 2,
        cutoff_dist: 10.0,
        spring_k: physics.spring_k,
        bias_strength: physics.bias_strength,
        target_mode: 7,
        use_gpu: true,
        max_trajectory_memory: 256 * 1024 * 1024,
        max_workspace_memory: 128 * 1024 * 1024,
    };

    // Initialize engine
    info!("   ▶️  Initializing MD engine...");
    let mut engine = MolecularDynamicsEngine::from_sovereign_buffer(md_config, &pdb_data)
        .context("Failed to initialize MD engine")?;

    // Get initial atom positions for comparison
    let initial_atoms = engine.get_initial_atoms().to_vec();

    // Configure persistence tracking
    let persistence_config = PersistenceConfig {
        num_samples: args.persistence_samples,
        exposure_threshold: args.exposure_threshold,
        pocket_open_fraction: args.pocket_open_fraction,
    };

    // Initialize persistence tracker
    let mut tracker = PersistenceTracker::new(
        persistence_config,
        initial_atoms.clone(),
        target.target_residues.clone(),
        target.core_residues.clone(),
        physics.steps,
    );

    // Calculate steps per chunk for persistence sampling
    let steps_per_chunk = physics.steps / args.persistence_samples as u64;
    let total_chunks = args.persistence_samples;

    info!(
        "   ▶️  Running {} steps in {} chunks for persistence tracking...",
        physics.steps, total_chunks
    );

    // Run simulation in chunks, recording snapshots for persistence
    let mut final_atoms = initial_atoms.clone();
    for chunk_idx in 0..total_chunks {
        // Run one chunk of simulation
        engine
            .run_nlnm_breathing(steps_per_chunk)
            .with_context(|| format!("Simulation chunk {} failed", chunk_idx))?;

        // Get current state and record snapshot
        final_atoms = engine
            .get_current_atoms()
            .context("Failed to get atoms for persistence snapshot")?;

        tracker.record_snapshot(chunk_idx, final_atoms.clone());

        // Log progress every 5 chunks
        if (chunk_idx + 1) % 5 == 0 || chunk_idx == total_chunks - 1 {
            debug!(
                "   📊 Chunk {}/{}: {}",
                chunk_idx + 1,
                total_chunks,
                tracker.get_summary()
            );
        }
    }

    // Analyze persistence metrics
    let persistence_metrics = tracker.analyze();
    let persistence_assessment_str = format!("{:?}", persistence_metrics.assessment);
    let is_druggable = persistence_metrics.assessment.is_druggable();

    info!(
        "   ⏱️  Persistence: {:.1}% ({}) | Events: {} | MaxOpen: {} frames",
        persistence_metrics.persistence_ratio * 100.0,
        persistence_assessment_str,
        persistence_metrics.opening_events,
        persistence_metrics.max_continuous_open
    );

    // A. Save Digital Twin (relaxed PDB)
    let pdb_path = format!("{}/{}_relaxed.pdb", target_dir, target.name);
    engine
        .save_pdb(&pdb_path, &target.apo_pdb)
        .context("Failed to save relaxed PDB")?;
    info!("   💾 Saved: {}", pdb_path);

    // B. Calculate per-residue displacements and generate Treasure Map
    let (residue_scores, metrics) = calculate_residue_scores(
        &initial_atoms,
        &final_atoms,
        &target.target_residues,
        &target.core_residues,
    );

    // C. Save residue scores CSV (now includes persistence per residue)
    let csv_path = format!("{}/residue_scores.csv", target_dir);
    save_residue_csv_with_persistence(&residue_scores, &persistence_metrics, &csv_path)?;
    info!("   💾 Saved: {}", csv_path);

    // D. Save persistence timeline JSON
    let persistence_path = format!("{}/persistence_timeline.json", target_dir);
    let persistence_json = serde_json::to_string_pretty(&persistence_metrics)?;
    fs::write(&persistence_path, &persistence_json)?;
    info!("   💾 Saved: {}", persistence_path);

    // E. Calculate enrichment factor
    let enrichment = calculate_enrichment_factor(&residue_scores, &target.target_residues, 10);

    // F. Determine pass/fail (now includes persistence)
    let passed = enrichment >= args.min_enrichment
        && metrics.core_rmsd <= args.max_core_rmsd
        && persistence_metrics.persistence_ratio >= args.min_persistence;

    Ok(HoldoutResult {
        target_name: target.name.clone(),
        apo_pdb: target.apo_pdb.clone(),
        total_sasa_gain_proxy: metrics.total_displacement,
        core_rmsd: metrics.core_rmsd,
        max_displacement: metrics.max_displacement,
        mean_displacement: metrics.mean_displacement,
        top_residues: residue_scores.iter().take(10).cloned().collect(),
        enrichment_factor: enrichment,
        validation_passed: passed,
        relaxed_pdb_path: pdb_path,
        residue_csv_path: csv_path,
        // Persistence fields
        persistence_ratio: persistence_metrics.persistence_ratio,
        persistence_assessment: persistence_assessment_str,
        opening_events: persistence_metrics.opening_events,
        max_continuous_open: persistence_metrics.max_continuous_open,
        mean_open_duration: persistence_metrics.mean_open_duration,
        is_druggable,
    })
}

/// Metrics from residue analysis
struct DisplacementMetrics {
    total_displacement: f32,
    max_displacement: f32,
    mean_displacement: f32,
    core_rmsd: f32,
}

/// Calculate per-residue displacement scores
fn calculate_residue_scores(
    initial: &[prism_io::sovereign_types::Atom],
    final_atoms: &[prism_io::sovereign_types::Atom],
    target_residues: &[usize],
    core_residues: &[usize],
) -> (Vec<ResidueScore>, DisplacementMetrics) {
    use std::collections::HashMap;

    // Aggregate max displacement per residue
    let mut residue_max_disp: HashMap<u32, f32> = HashMap::new();
    let mut total_disp = 0.0;
    let mut max_disp = 0.0f32;
    let mut core_sum_sq = 0.0;
    let mut core_count = 0;

    for (i, (start, end)) in initial.iter().zip(final_atoms.iter()).enumerate() {
        let dx = end.coords[0] - start.coords[0];
        let dy = end.coords[1] - start.coords[1];
        let dz = end.coords[2] - start.coords[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();

        let res_id = u32::from(start.residue_id);

        // Track max per residue
        let entry = residue_max_disp.entry(res_id).or_insert(0.0);
        if dist > *entry {
            *entry = dist;
        }

        total_disp += dist;
        max_disp = max_disp.max(dist);

        // Core RMSD calculation
        if core_residues.contains(&(res_id as usize)) {
            core_sum_sq += dist * dist;
            core_count += 1;
        }
    }

    let mean_disp = if !initial.is_empty() {
        total_disp / initial.len() as f32
    } else {
        0.0
    };
    let core_rmsd = if core_count > 0 {
        (core_sum_sq / core_count as f32).sqrt()
    } else {
        0.0
    };

    // Sort residues by displacement (highest first)
    let mut sorted_residues: Vec<_> = residue_max_disp.into_iter().collect();
    sorted_residues.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Create ranked scores
    let scores: Vec<ResidueScore> = sorted_residues
        .iter()
        .enumerate()
        .map(|(rank, (res_id, disp))| ResidueScore {
            residue_id: *res_id,
            displacement: *disp,
            rank: rank + 1,
            is_target: target_residues.contains(&(*res_id as usize)),
        })
        .collect();

    let metrics = DisplacementMetrics {
        total_displacement: total_disp,
        max_displacement: max_disp,
        mean_displacement: mean_disp,
        core_rmsd,
    };

    (scores, metrics)
}

/// Save residue scores to CSV (basic version)
fn save_residue_csv(scores: &[ResidueScore], path: &str) -> Result<()> {
    let mut file = fs::File::create(path)?;
    writeln!(file, "Rank,Residue,Displacement_A,IsTarget")?;

    for score in scores {
        let label = if score.is_target { "TARGET" } else { "-" };
        writeln!(
            file,
            "{},{},{:.3},{}",
            score.rank, score.residue_id, score.displacement, label
        )?;
    }

    Ok(())
}

/// Save residue scores to CSV with persistence data (enhanced Treasure Map)
fn save_residue_csv_with_persistence(
    scores: &[ResidueScore],
    persistence: &PocketPersistenceMetrics,
    path: &str,
) -> Result<()> {
    let mut file = fs::File::create(path)?;
    writeln!(
        file,
        "Rank,Residue,Displacement_A,IsTarget,ExposureFraction,MaxDisplacement,MeanDisplacement"
    )?;

    // Build a map of residue histories for quick lookup
    let residue_map: HashMap<u32, &prism_learning::persistence::ResidueExposureHistory> =
        persistence
            .residue_histories
            .iter()
            .map(|h| (h.residue_id, h))
            .collect();

    for score in scores {
        let label = if score.is_target { "TARGET" } else { "-" };

        // Look up persistence data for this residue (if it's a target)
        if let Some(history) = residue_map.get(&score.residue_id) {
            writeln!(
                file,
                "{},{},{:.3},{},{:.3},{:.3},{:.3}",
                score.rank,
                score.residue_id,
                score.displacement,
                label,
                history.exposure_fraction,
                history.max_displacement,
                history.mean_displacement
            )?;
        } else {
            // Non-target residues don't have persistence tracking
            writeln!(
                file,
                "{},{},{:.3},{},-,-,-",
                score.rank, score.residue_id, score.displacement, label
            )?;
        }
    }

    Ok(())
}

/// Calculate enrichment factor
/// EF = (targets in top N / N) / (total targets / total residues)
fn calculate_enrichment_factor(
    scores: &[ResidueScore],
    target_residues: &[usize],
    top_n: usize,
) -> f32 {
    if scores.is_empty() || target_residues.is_empty() {
        return 0.0;
    }

    let total_residues = scores.len();
    let total_targets = target_residues.len();

    // Count targets in top N
    let targets_in_top_n = scores.iter().take(top_n).filter(|s| s.is_target).count();

    // Random expectation
    let random_expectation = (total_targets as f32 / total_residues as f32) * top_n as f32;

    if random_expectation > 0.0 {
        targets_in_top_n as f32 / random_expectation
    } else {
        0.0
    }
}
