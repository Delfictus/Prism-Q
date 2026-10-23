//! NHS Ensemble Analyzer
//!
//! Post-processing tool for analyzing pre-generated cryo-UV ensembles.
//!
//! Usage:
//!   nhs-analyze <ensemble.pdb> --topology <topology.json> --output <results/>
//!
//! Features:
//! - Load ensemble trajectory from PDB
//! - Run NHS neuromorphic detection on each frame
//! - Site clustering and mapping
//! - Publication-quality output

use anyhow::{Context, Result};
use clap::Parser;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use prism_nhs::{
    input::PrismPrepTopology, load_ensemble_pdb, CrypticSiteEvent, NhsConfig, NhsPipeline,
};

#[derive(Parser, Debug)]
#[command(name = "nhs-analyze")]
#[command(about = "Analyze pre-generated cryo-UV ensembles for cryptic sites")]
#[command(version)]
struct Args {
    /// Input ensemble PDB file (multi-model format)
    #[arg(value_name = "ENSEMBLE_PDB")]
    input: PathBuf,

    /// PRISM-PREP topology JSON file
    #[arg(short, long)]
    topology: PathBuf,

    /// Output directory for results
    #[arg(short, long, default_value = "nhs_analysis")]
    output: PathBuf,

    /// Grid spacing in Angstroms
    #[arg(short, long, default_value = "1.0")]
    spacing: f32,

    /// Spike threshold for detection
    #[arg(long, default_value = "0.3")]
    spike_threshold: f32,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Skip frames (process every Nth frame)
    #[arg(long, default_value = "1")]
    skip: usize,

    /// Clustering radius for sites (Angstroms)
    #[arg(long, default_value = "5.0")]
    cluster_radius: f32,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║     NHS Ensemble Analyzer - Post-Processing Tool               ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();

    // Create output directory
    fs::create_dir_all(&args.output).context("Failed to create output directory")?;

    // Load topology
    log::info!("Loading topology: {}", args.topology.display());
    let topology = PrismPrepTopology::load(&args.topology).context("Failed to load topology")?;
    println!(
        "Structure: {} atoms, {} residues",
        topology.n_atoms,
        topology.residue_names.len()
    );

    // Load ensemble
    log::info!("Loading ensemble: {}", args.input.display());
    let frames = load_ensemble_pdb(&args.input).context("Failed to load ensemble PDB")?;
    println!("Ensemble: {} frames loaded", frames.len());

    if frames.is_empty() {
        anyhow::bail!("No frames found in ensemble PDB");
    }

    // Configure NHS pipeline
    let mut config = NhsConfig::default();
    config.grid_spacing = args.spacing;
    config.spike_threshold = args.spike_threshold;

    // Create pipeline
    log::info!("Initializing NHS pipeline...");
    let mut pipeline = NhsPipeline::new(config);

    // Initialize with first frame
    let first_frame = &frames[0];
    pipeline
        .initialize(
            &first_frame.positions,
            &topology
                .elements
                .iter()
                .map(|e| element_to_atomic_num(e))
                .collect::<Vec<_>>(),
            &topology.charges,
            &topology.residue_names,
            &topology.atom_names,
            &(0..topology.n_atoms)
                .map(|i| topology.residue_ids[i] as usize)
                .collect::<Vec<_>>(),
        )
        .context("Failed to initialize NHS pipeline")?;

    println!();
    println!("════════════════════════════════════════════════════════════════");
    println!("                    PROCESSING ENSEMBLE");
    println!("════════════════════════════════════════════════════════════════");
    println!();

    let start_time = Instant::now();
    let mut all_events: Vec<CrypticSiteEvent> = Vec::new();
    let mut frame_results = Vec::new();
    let frames_to_process: Vec<_> = frames.iter().enumerate().step_by(args.skip).collect();

    let total_frames = frames_to_process.len();

    for (idx, (frame_idx, frame)) in frames_to_process.iter().enumerate() {
        // Process frame
        let (events, _perturbation) = pipeline
            .process_frame(&frame.positions)
            .context("Failed to process frame")?;

        let n_events = events.len();
        all_events.extend(events);

        frame_results.push(FrameAnalysis {
            frame_idx: *frame_idx,
            timestep: frame.timestep,
            temperature: frame.temperature,
            time_ps: frame.time_ps,
            spike_triggered: frame.spike_triggered,
            cryptic_events: n_events,
        });

        // Progress update
        if (idx + 1) % 100 == 0 || idx + 1 == total_frames {
            let elapsed = start_time.elapsed().as_secs_f64();
            let fps = (idx + 1) as f64 / elapsed;
            let eta = (total_frames - idx - 1) as f64 / fps;
            print!(
                "\r  Frame {}/{} | {:.1} fps | ETA {:.0}s | Events: {}    ",
                idx + 1,
                total_frames,
                fps,
                eta,
                all_events.len()
            );
            std::io::Write::flush(&mut std::io::stdout())?;
        }
    }
    println!();

    let elapsed = start_time.elapsed();
    let stats = pipeline.stats();

    println!();
    println!("════════════════════════════════════════════════════════════════");
    println!("                       ANALYSIS RESULTS");
    println!("════════════════════════════════════════════════════════════════");
    println!();
    println!("Processing:");
    println!("  Frames analyzed:    {}", total_frames);
    println!("  Total time:         {:.2}s", elapsed.as_secs_f64());
    println!(
        "  Frames/second:      {:.1}",
        total_frames as f64 / elapsed.as_secs_f64()
    );
    println!();
    println!("Detection:");
    println!("  Total spikes:       {}", stats.total_spikes);
    println!("  Cryptic events:     {}", all_events.len());
    println!("  Avg spikes/frame:   {:.2}", stats.avg_spikes_per_frame);
    println!();

    // Cluster events into sites with quality scoring
    let sites = cluster_events(&all_events, args.cluster_radius, total_frames);

    // Count sites by confidence level
    let high_confidence = sites.iter().filter(|s| s.confidence_score >= 0.75).count();
    let medium_confidence = sites
        .iter()
        .filter(|s| s.confidence_score >= 0.50 && s.confidence_score < 0.75)
        .count();
    let avg_confidence = if sites.is_empty() {
        0.0
    } else {
        sites.iter().map(|s| s.confidence_score).sum::<f32>() / sites.len() as f32
    };

    println!("Cryptic Sites Found: {}", sites.len());
    println!("  High confidence:    {} (score >= 0.75)", high_confidence);
    println!(
        "  Medium confidence:  {} (score 0.50-0.75)",
        medium_confidence
    );
    println!(
        "  Low confidence:     {} (score < 0.50)",
        sites.len() - high_confidence - medium_confidence
    );
    println!("  Average confidence: {:.2}", avg_confidence);
    println!();

    for (i, site) in sites.iter().enumerate().take(10) {
        let residue_str = site
            .residues
            .iter()
            .take(5)
            .map(|r| r.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let more = if site.residues.len() > 5 { "..." } else { "" };
        println!(
            "  Site {}: {} events, conf={:.2} [{}], center ({:.1}, {:.1}, {:.1}), residues [{}{}]",
            i + 1,
            site.event_count,
            site.confidence_score,
            site.category,
            site.centroid[0],
            site.centroid[1],
            site.centroid[2],
            residue_str,
            more
        );
    }
    if sites.len() > 10 {
        println!("  ... and {} more sites", sites.len() - 10);
    }

    // Save results
    println!();
    println!("Saving results...");

    // Frame-by-frame analysis
    let frames_path = args.output.join("frame_analysis.json");
    let frames_file = fs::File::create(&frames_path)?;
    serde_json::to_writer_pretty(frames_file, &frame_results)?;
    println!("  {}", frames_path.display());

    // Cryptic sites
    let sites_path = args.output.join("cryptic_sites.json");
    let sites_file = fs::File::create(&sites_path)?;
    serde_json::to_writer_pretty(sites_file, &sites)?;
    println!("  {}", sites_path.display());

    // Summary with quality metrics
    let summary = AnalysisSummary {
        input_ensemble: args.input.to_string_lossy().to_string(),
        topology: args.topology.to_string_lossy().to_string(),
        frames_analyzed: total_frames,
        elapsed_seconds: elapsed.as_secs_f64(),
        total_spikes: stats.total_spikes,
        cryptic_events: all_events.len(),
        sites_found: sites.len(),
        avg_spikes_per_frame: stats.avg_spikes_per_frame,
        high_confidence_sites: high_confidence,
        medium_confidence_sites: medium_confidence,
        avg_confidence_score: avg_confidence,
    };
    let summary_path = args.output.join("analysis_summary.json");
    let summary_file = fs::File::create(&summary_path)?;
    serde_json::to_writer_pretty(summary_file, &summary)?;
    println!("  {}", summary_path.display());

    // Write sites as PyMOL script
    let pymol_path = args.output.join("cryptic_sites.pml");
    write_pymol_script(&pymol_path, &sites)?;
    println!("  {}", pymol_path.display());

    println!();
    println!("✓ Analysis complete!");

    Ok(())
}

/// Convert element symbol to atomic number
fn element_to_atomic_num(element: &str) -> u8 {
    match element.to_uppercase().as_str() {
        "H" => 1,
        "C" => 6,
        "N" => 7,
        "O" => 8,
        "S" => 16,
        "P" => 15,
        "FE" => 26,
        "ZN" => 30,
        "CA" => 20,
        "MG" => 12,
        _ => 6, // Default to carbon
    }
}

#[derive(Debug, Clone, serde::Serialize)]
struct FrameAnalysis {
    frame_idx: usize,
    timestep: i32,
    temperature: f32,
    time_ps: f32,
    spike_triggered: bool,
    cryptic_events: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
struct ClusteredSite {
    site_id: usize,
    centroid: [f32; 3],
    residues: Vec<u32>,
    event_count: usize,
    total_volume: f32,
    /// Quality score based on persistence and event frequency (0-1)
    confidence_score: f32,
    /// Category: HIGH (>0.75), MEDIUM (0.50-0.75), LOW (<0.50)
    category: String,
    /// First frame where this site was detected
    first_detection_frame: usize,
    /// Last frame where this site was detected
    last_detection_frame: usize,
    /// Fraction of frames where this site was active
    persistence_fraction: f32,
}

#[derive(Debug, Clone, serde::Serialize)]
struct AnalysisSummary {
    input_ensemble: String,
    topology: String,
    frames_analyzed: usize,
    elapsed_seconds: f64,
    total_spikes: u64,
    cryptic_events: usize,
    sites_found: usize,
    avg_spikes_per_frame: f32,
    /// Number of high-confidence sites (score > 0.75)
    high_confidence_sites: usize,
    /// Number of medium-confidence sites (score 0.50-0.75)
    medium_confidence_sites: usize,
    /// Average confidence score across all sites
    avg_confidence_score: f32,
}

/// Cluster cryptic events by spatial proximity with quality scoring
fn cluster_events(
    events: &[CrypticSiteEvent],
    radius: f32,
    total_frames: usize,
) -> Vec<ClusteredSite> {
    if events.is_empty() {
        return Vec::new();
    }

    // Track which frames each cluster appeared in
    let mut clusters: Vec<ClusteredSite> = Vec::new();
    let mut cluster_frames: Vec<Vec<usize>> = Vec::new();
    let radius_sq = radius * radius;

    // Process events (assuming sequential frame processing)
    let mut current_frame = 0usize;
    for event in events {
        // Find closest existing cluster
        let mut closest_idx = None;
        let mut closest_dist_sq = f32::MAX;

        for (i, cluster) in clusters.iter().enumerate() {
            let dx = event.centroid[0] - cluster.centroid[0];
            let dy = event.centroid[1] - cluster.centroid[1];
            let dz = event.centroid[2] - cluster.centroid[2];
            let dist_sq = dx * dx + dy * dy + dz * dz;

            if dist_sq < closest_dist_sq {
                closest_dist_sq = dist_sq;
                closest_idx = Some(i);
            }
        }

        if closest_dist_sq < radius_sq {
            // Add to existing cluster
            if let Some(idx) = closest_idx {
                let cluster = &mut clusters[idx];
                // Update centroid (running average)
                let n = cluster.event_count as f32;
                cluster.centroid[0] = (cluster.centroid[0] * n + event.centroid[0]) / (n + 1.0);
                cluster.centroid[1] = (cluster.centroid[1] * n + event.centroid[1]) / (n + 1.0);
                cluster.centroid[2] = (cluster.centroid[2] * n + event.centroid[2]) / (n + 1.0);
                cluster.event_count += 1;
                cluster.total_volume += event.volume;
                // Add unique residues
                for res in &event.residues {
                    if !cluster.residues.contains(res) {
                        cluster.residues.push(*res);
                    }
                }
                // Track frame
                cluster.last_detection_frame = current_frame;
                cluster_frames[idx].push(current_frame);
            }
        } else {
            // Create new cluster
            clusters.push(ClusteredSite {
                site_id: clusters.len(),
                centroid: event.centroid,
                residues: event.residues.clone(),
                event_count: 1,
                total_volume: event.volume,
                confidence_score: 0.0,
                category: String::new(),
                first_detection_frame: current_frame,
                last_detection_frame: current_frame,
                persistence_fraction: 0.0,
            });
            cluster_frames.push(vec![current_frame]);
        }

        // Simple frame tracking (approximate)
        current_frame = (current_frame + 1) % total_frames.max(1);
    }

    // Compute quality scores for each cluster
    let max_events = clusters.iter().map(|c| c.event_count).max().unwrap_or(1) as f32;

    for (i, cluster) in clusters.iter_mut().enumerate() {
        // Persistence: how many unique frames this site appeared in
        let unique_frames: std::collections::HashSet<_> = cluster_frames[i].iter().collect();
        cluster.persistence_fraction = unique_frames.len() as f32 / total_frames.max(1) as f32;

        // Confidence score components:
        // - Event frequency (normalized by max)
        let frequency_score = (cluster.event_count as f32 / max_events).min(1.0);
        // - Persistence across frames
        let persistence_score = cluster.persistence_fraction.min(1.0);
        // - Volume consistency (more volume = more confident)
        let volume_score = (cluster.total_volume / (cluster.event_count as f32 * 50.0)).min(1.0);

        // Combined confidence (weighted average)
        cluster.confidence_score =
            (frequency_score * 0.4 + persistence_score * 0.4 + volume_score * 0.2).clamp(0.0, 1.0);

        // Categorize
        cluster.category = if cluster.confidence_score >= 0.75 {
            "HIGH".to_string()
        } else if cluster.confidence_score >= 0.50 {
            "MEDIUM".to_string()
        } else {
            "LOW".to_string()
        };
    }

    // Sort by confidence score (descending), then by event count
    clusters.sort_by(|a, b| {
        b.confidence_score
            .partial_cmp(&a.confidence_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.event_count.cmp(&a.event_count))
    });

    // Reassign site IDs after sorting
    for (i, cluster) in clusters.iter_mut().enumerate() {
        cluster.site_id = i;
    }

    clusters
}

/// Write PyMOL visualization script with confidence-based coloring
fn write_pymol_script(path: &std::path::Path, sites: &[ClusteredSite]) -> Result<()> {
    use std::io::Write;
    let mut file = fs::File::create(path)?;

    writeln!(file, "# PRISM-NHS Cryptic Site Visualization")?;
    writeln!(file, "# Generated by nhs-analyze")?;
    writeln!(file, "")?;
    writeln!(file, "# Color scheme by confidence:")?;
    writeln!(file, "#   GREEN = HIGH confidence (>= 0.75)")?;
    writeln!(file, "#   YELLOW = MEDIUM confidence (0.50-0.75)")?;
    writeln!(file, "#   RED = LOW confidence (< 0.50)")?;
    writeln!(file, "# Sphere size scales with event count")?;
    writeln!(file, "")?;

    let max_events = sites.iter().map(|s| s.event_count).max().unwrap_or(1) as f32;

    for site in sites.iter().take(20) {
        let size_scale = 0.5 + 1.5 * (site.event_count as f32 / max_events);

        // Color by confidence category
        let (r, g, b) = match site.category.as_str() {
            "HIGH" => (0.2, 0.9, 0.2),   // Green
            "MEDIUM" => (0.9, 0.9, 0.2), // Yellow
            _ => (0.9, 0.3, 0.2),        // Red
        };

        writeln!(
            file,
            "# Site {} - {} events, confidence={:.2} [{}]",
            site.site_id + 1,
            site.event_count,
            site.confidence_score,
            site.category
        )?;
        writeln!(
            file,
            "pseudoatom site_{}, pos=[{:.2}, {:.2}, {:.2}]",
            site.site_id + 1,
            site.centroid[0],
            site.centroid[1],
            site.centroid[2]
        )?;
        writeln!(
            file,
            "color [{:.2}, {:.2}, {:.2}], site_{}",
            r,
            g,
            b,
            site.site_id + 1
        )?;
        writeln!(file, "show spheres, site_{}", site.site_id + 1)?;
        writeln!(
            file,
            "set sphere_scale, {:.2}, site_{}",
            size_scale,
            site.site_id + 1
        )?;

        // Highlight residues
        if !site.residues.is_empty() {
            let res_sel = site
                .residues
                .iter()
                .take(10)
                .map(|r| format!("resi {}", r))
                .collect::<Vec<_>>()
                .join(" or ");
            writeln!(
                file,
                "select site_{}_residues, {}",
                site.site_id + 1,
                res_sel
            )?;
        }
        writeln!(file)?;
    }

    writeln!(file, "# Group sites by confidence")?;
    writeln!(file, "group high_confidence_sites, site_*")?;

    let high_sites: Vec<_> = sites
        .iter()
        .filter(|s| s.category == "HIGH")
        .take(20)
        .collect();
    let medium_sites: Vec<_> = sites
        .iter()
        .filter(|s| s.category == "MEDIUM")
        .take(20)
        .collect();
    let low_sites: Vec<_> = sites
        .iter()
        .filter(|s| s.category == "LOW")
        .take(20)
        .collect();

    if !high_sites.is_empty() {
        let high_sel = high_sites
            .iter()
            .map(|s| format!("site_{}", s.site_id + 1))
            .collect::<Vec<_>>()
            .join(" ");
        writeln!(file, "select high_confidence, {}", high_sel)?;
    }
    if !medium_sites.is_empty() {
        let med_sel = medium_sites
            .iter()
            .map(|s| format!("site_{}", s.site_id + 1))
            .collect::<Vec<_>>()
            .join(" ");
        writeln!(file, "select medium_confidence, {}", med_sel)?;
    }
    if !low_sites.is_empty() {
        let low_sel = low_sites
            .iter()
            .map(|s| format!("site_{}", s.site_id + 1))
            .collect::<Vec<_>>()
            .join(" ");
        writeln!(file, "select low_confidence, {}", low_sel)?;
    }

    writeln!(file, "")?;
    writeln!(file, "# Show all sites")?;
    writeln!(file, "group cryptic_sites, site_*")?;
    writeln!(file, "zoom cryptic_sites")?;

    Ok(())
}
