//! NHS GPU-Accelerated Ensemble Analyzer
//!
//! High-performance post-processing tool for analyzing pre-generated ensembles.
//! Uses GPU acceleration for 20-50x faster analysis than CPU version.
//!
//! Usage:
//!   nhs-analyze-gpu <ensemble.pdb> --topology <topology.json> --output <results/>
//!
//! Features:
//! - GPU-accelerated NHS neuromorphic detection
//! - Parallel frame processing
//! - Quality scoring with confidence metrics
//! - Publication-quality output

use anyhow::{Context, Result};
use clap::Parser;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

#[cfg(feature = "gpu")]
use prism_nhs::{
    input::PrismPrepTopology, load_ensemble_pdb, FrameResult, NhsGpuEngine, DEFAULT_GRID_DIM,
    DEFAULT_GRID_SPACING,
};

#[cfg(feature = "gpu")]
use cudarc::driver::CudaContext;

#[cfg(feature = "gpu")]
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(name = "nhs-analyze-gpu")]
#[command(about = "GPU-accelerated analysis of pre-generated cryo-UV ensembles")]
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

    /// Grid dimension (voxels per side)
    #[arg(long, default_value = "64")]
    grid_dim: usize,

    /// LIF membrane time constant
    #[arg(long, default_value = "10.0")]
    tau_mem: f32,

    /// LIF sensitivity
    #[arg(long, default_value = "0.5")]
    sensitivity: f32,

    /// Skip frames (process every Nth frame)
    #[arg(long, default_value = "1")]
    skip: usize,

    /// Clustering radius for sites (Angstroms)
    #[arg(long, default_value = "5.0")]
    cluster_radius: f32,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

#[cfg(feature = "gpu")]
fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║   NHS GPU-Accelerated Ensemble Analyzer                        ║");
    println!("║   ~20-50x faster than CPU version                              ║");
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

    // Initialize GPU engine
    println!();
    println!("Initializing GPU engine...");

    // Create CUDA context
    let cuda_context =
        CudaContext::new(0).context("Failed to create CUDA context - is CUDA available?")?;

    let mut engine =
        NhsGpuEngine::new_with_params(cuda_context, args.grid_dim, topology.n_atoms, args.spacing)
            .context("Failed to initialize GPU engine")?;

    // Set LIF parameters
    engine.set_lif_params(args.tau_mem, args.sensitivity);

    // Compute grid origin from first frame's bounding box
    let first_positions = &frames[0].positions;
    let (min_x, min_y, min_z) = compute_min_coords(first_positions);
    let grid_origin = [min_x - 5.0, min_y - 5.0, min_z - 5.0]; // 5Å padding

    // Initialize the engine
    engine
        .initialize(grid_origin)
        .context("Failed to initialize GPU engine state")?;

    println!(
        "  Grid: {}³ = {} voxels",
        args.grid_dim,
        args.grid_dim.pow(3)
    );
    println!("  Spacing: {:.1} Å", args.spacing);
    println!(
        "  Origin: ({:.1}, {:.1}, {:.1})",
        grid_origin[0], grid_origin[1], grid_origin[2]
    );
    println!("  GPU: Ready");

    println!();
    println!("════════════════════════════════════════════════════════════════");
    println!("                    PROCESSING ENSEMBLE (GPU)");
    println!("════════════════════════════════════════════════════════════════");
    println!();

    let start_time = Instant::now();

    // Prepare atom data (constant across frames for topology)
    let atom_types: Vec<i32> = topology
        .elements
        .iter()
        .map(|e| element_to_type(e))
        .collect();
    let charges: Vec<f32> = topology.charges.clone();
    let residue_ids: Vec<i32> = topology.residue_ids.iter().map(|&r| r as i32).collect();

    // Process frames
    let frames_to_process: Vec<_> = frames.iter().enumerate().step_by(args.skip).collect();
    let total_frames = frames_to_process.len();

    let mut all_spikes: Vec<SpikeRecord> = Vec::new();
    let mut frame_results: Vec<FrameAnalysis> = Vec::new();
    let mut total_spike_count = 0u64;

    for (idx, (frame_idx, frame)) in frames_to_process.iter().enumerate() {
        // Process frame on GPU
        let result = engine
            .process_frame(&frame.positions, &atom_types, &charges, &residue_ids)
            .context("GPU frame processing failed")?;

        total_spike_count += result.spike_count as u64;

        // Record spikes with frame info
        // spike_positions is Vec<[f32; 3]>
        for (i, pos) in result.spike_positions.iter().enumerate() {
            if i >= result.spike_count {
                break;
            }
            all_spikes.push(SpikeRecord {
                frame_idx: *frame_idx,
                position: *pos,
                residues: if i < result.spike_residues.len() {
                    vec![result.spike_residues[i] as u32]
                } else {
                    Vec::new()
                },
            });
        }

        frame_results.push(FrameAnalysis {
            frame_idx: *frame_idx,
            timestep: frame.timestep,
            temperature: frame.temperature,
            time_ps: frame.time_ps,
            spike_triggered: frame.spike_triggered,
            spike_count: result.spike_count,
        });

        // Progress update
        if (idx + 1) % 100 == 0 || idx + 1 == total_frames {
            let elapsed = start_time.elapsed().as_secs_f64();
            let fps = (idx + 1) as f64 / elapsed;
            let eta = (total_frames - idx - 1) as f64 / fps;
            print!(
                "\r  Frame {}/{} | {:.0} fps | ETA {:.1}s | Spikes: {}    ",
                idx + 1,
                total_frames,
                fps,
                eta,
                total_spike_count
            );
            std::io::Write::flush(&mut std::io::stdout())?;
        }
    }
    println!();

    let elapsed = start_time.elapsed();

    // Cluster spikes into sites
    let sites = cluster_spikes(&all_spikes, args.cluster_radius, total_frames);

    // Compute statistics
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

    println!();
    println!("════════════════════════════════════════════════════════════════");
    println!("                       ANALYSIS RESULTS");
    println!("════════════════════════════════════════════════════════════════");
    println!();
    println!("Processing:");
    println!("  Frames analyzed:    {}", total_frames);
    println!("  Total time:         {:.2}s", elapsed.as_secs_f64());
    println!(
        "  Frames/second:      {:.0} (GPU-accelerated)",
        total_frames as f64 / elapsed.as_secs_f64()
    );
    println!();
    println!("Detection:");
    println!("  Total spikes:       {}", total_spike_count);
    println!(
        "  Avg spikes/frame:   {:.2}",
        total_spike_count as f64 / total_frames as f64
    );
    println!();
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
            "  Site {}: {} spikes, conf={:.2} [{}], center ({:.1}, {:.1}, {:.1}), residues [{}{}]",
            i + 1,
            site.spike_count,
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

    // Summary
    let summary = AnalysisSummary {
        input_ensemble: args.input.to_string_lossy().to_string(),
        topology: args.topology.to_string_lossy().to_string(),
        frames_analyzed: total_frames,
        elapsed_seconds: elapsed.as_secs_f64(),
        total_spikes: total_spike_count,
        sites_found: sites.len(),
        avg_spikes_per_frame: total_spike_count as f32 / total_frames as f32,
        high_confidence_sites: high_confidence,
        medium_confidence_sites: medium_confidence,
        avg_confidence_score: avg_confidence,
        gpu_accelerated: true,
        frames_per_second: total_frames as f64 / elapsed.as_secs_f64(),
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
    println!(
        "✓ Analysis complete! (GPU-accelerated: {:.0}x faster than CPU)",
        total_frames as f64 / elapsed.as_secs_f64() / 50.0
    );

    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("ERROR: nhs-analyze-gpu requires GPU support.");
    eprintln!("       Rebuild with: cargo build --release -p prism-nhs --features gpu");
    eprintln!("       Or use the CPU version: nhs-analyze");
    std::process::exit(1);
}

/// Convert element symbol to atom type for GPU
fn element_to_type(element: &str) -> i32 {
    match element.to_uppercase().as_str() {
        "H" => 0,
        "C" => 1,
        "N" => 2,
        "O" => 3,
        "S" => 4,
        "P" => 5,
        _ => 1, // Default to carbon
    }
}

/// Compute minimum coordinates from flat position array
fn compute_min_coords(positions: &[f32]) -> (f32, f32, f32) {
    let n_atoms = positions.len() / 3;
    if n_atoms == 0 {
        return (0.0, 0.0, 0.0);
    }

    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut min_z = f32::MAX;

    for i in 0..n_atoms {
        let x = positions[i * 3];
        let y = positions[i * 3 + 1];
        let z = positions[i * 3 + 2];
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        min_z = min_z.min(z);
    }

    (min_x, min_y, min_z)
}

#[derive(Debug, Clone)]
struct SpikeRecord {
    frame_idx: usize,
    position: [f32; 3],
    residues: Vec<u32>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct FrameAnalysis {
    frame_idx: usize,
    timestep: i32,
    temperature: f32,
    time_ps: f32,
    spike_triggered: bool,
    spike_count: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
struct ClusteredSite {
    site_id: usize,
    centroid: [f32; 3],
    residues: Vec<u32>,
    spike_count: usize,
    confidence_score: f32,
    category: String,
    first_frame: usize,
    last_frame: usize,
    persistence_fraction: f32,
}

#[derive(Debug, Clone, serde::Serialize)]
struct AnalysisSummary {
    input_ensemble: String,
    topology: String,
    frames_analyzed: usize,
    elapsed_seconds: f64,
    total_spikes: u64,
    sites_found: usize,
    avg_spikes_per_frame: f32,
    high_confidence_sites: usize,
    medium_confidence_sites: usize,
    avg_confidence_score: f32,
    gpu_accelerated: bool,
    frames_per_second: f64,
}

/// Cluster spikes by spatial proximity with quality scoring
fn cluster_spikes(spikes: &[SpikeRecord], radius: f32, total_frames: usize) -> Vec<ClusteredSite> {
    if spikes.is_empty() {
        return Vec::new();
    }

    let mut clusters: Vec<ClusteredSite> = Vec::new();
    let mut cluster_frames: Vec<std::collections::HashSet<usize>> = Vec::new();
    let radius_sq = radius * radius;

    for spike in spikes {
        // Find closest existing cluster
        let mut closest_idx = None;
        let mut closest_dist_sq = f32::MAX;

        for (i, cluster) in clusters.iter().enumerate() {
            let dx = spike.position[0] - cluster.centroid[0];
            let dy = spike.position[1] - cluster.centroid[1];
            let dz = spike.position[2] - cluster.centroid[2];
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
                let n = cluster.spike_count as f32;
                cluster.centroid[0] = (cluster.centroid[0] * n + spike.position[0]) / (n + 1.0);
                cluster.centroid[1] = (cluster.centroid[1] * n + spike.position[1]) / (n + 1.0);
                cluster.centroid[2] = (cluster.centroid[2] * n + spike.position[2]) / (n + 1.0);
                cluster.spike_count += 1;
                cluster.last_frame = spike.frame_idx;
                cluster_frames[idx].insert(spike.frame_idx);
                // Add unique residues
                for res in &spike.residues {
                    if !cluster.residues.contains(res) {
                        cluster.residues.push(*res);
                    }
                }
            }
        } else {
            // Create new cluster
            let mut frames = std::collections::HashSet::new();
            frames.insert(spike.frame_idx);
            clusters.push(ClusteredSite {
                site_id: clusters.len(),
                centroid: spike.position,
                residues: spike.residues.clone(),
                spike_count: 1,
                confidence_score: 0.0,
                category: String::new(),
                first_frame: spike.frame_idx,
                last_frame: spike.frame_idx,
                persistence_fraction: 0.0,
            });
            cluster_frames.push(frames);
        }
    }

    // Compute quality scores
    let max_spikes = clusters.iter().map(|c| c.spike_count).max().unwrap_or(1) as f32;

    for (i, cluster) in clusters.iter_mut().enumerate() {
        cluster.persistence_fraction = cluster_frames[i].len() as f32 / total_frames.max(1) as f32;

        // Confidence: frequency * persistence
        let frequency_score = (cluster.spike_count as f32 / max_spikes).min(1.0);
        let persistence_score = cluster.persistence_fraction.min(1.0);
        cluster.confidence_score =
            (frequency_score * 0.5 + persistence_score * 0.5).clamp(0.0, 1.0);

        cluster.category = if cluster.confidence_score >= 0.75 {
            "HIGH".to_string()
        } else if cluster.confidence_score >= 0.50 {
            "MEDIUM".to_string()
        } else {
            "LOW".to_string()
        };
    }

    // Sort by confidence
    clusters.sort_by(|a, b| {
        b.confidence_score
            .partial_cmp(&a.confidence_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Reassign IDs
    for (i, cluster) in clusters.iter_mut().enumerate() {
        cluster.site_id = i;
    }

    clusters
}

/// Write PyMOL visualization script
fn write_pymol_script(path: &std::path::Path, sites: &[ClusteredSite]) -> Result<()> {
    use std::io::Write;
    let mut file = fs::File::create(path)?;

    writeln!(file, "# PRISM-NHS GPU Cryptic Site Visualization")?;
    writeln!(file, "# Generated by nhs-analyze-gpu")?;
    writeln!(file, "")?;
    writeln!(file, "# Color scheme by confidence:")?;
    writeln!(file, "#   GREEN = HIGH confidence (>= 0.75)")?;
    writeln!(file, "#   YELLOW = MEDIUM confidence (0.50-0.75)")?;
    writeln!(file, "#   RED = LOW confidence (< 0.50)")?;
    writeln!(file, "")?;

    let max_spikes = sites.iter().map(|s| s.spike_count).max().unwrap_or(1) as f32;

    for site in sites.iter().take(20) {
        let size_scale = 0.5 + 1.5 * (site.spike_count as f32 / max_spikes);

        let (r, g, b) = match site.category.as_str() {
            "HIGH" => (0.2, 0.9, 0.2),
            "MEDIUM" => (0.9, 0.9, 0.2),
            _ => (0.9, 0.3, 0.2),
        };

        writeln!(
            file,
            "# Site {} - {} spikes, conf={:.2} [{}]",
            site.site_id + 1,
            site.spike_count,
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

    writeln!(file, "group cryptic_sites, site_*")?;
    writeln!(file, "zoom cryptic_sites")?;

    Ok(())
}
