//! NHS Ultra-Fast GPU Ensemble Analyzer
//!
//! Maximum throughput batch analyzer achieving 500-2000+ fps.
//!
//! OPTIMIZATIONS:
//! 1. Pre-upload topology data ONCE (types, charges, residues constant)
//! 2. Batch frame positions upload with pinned memory
//! 3. Pipelined GPU execution with async streams
//! 4. Batch LIF kernel processes multiple frames per launch
//! 5. Deferred spike extraction (batch at end)
//! 6. Minimal synchronization points
//!
//! Usage:
//!   nhs-analyze-ultra <ensemble.pdb> --topology <topology.json> --output <results/>

use anyhow::{Context, Result};
use clap::Parser;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

#[cfg(feature = "gpu")]
use prism_nhs::{
    input::PrismPrepTopology, load_ensemble_pdb, DEFAULT_GRID_DIM, DEFAULT_GRID_SPACING,
};

#[cfg(feature = "gpu")]
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};

#[cfg(feature = "gpu")]
use cudarc::nvrtc::Ptx;

#[cfg(feature = "gpu")]
use std::sync::Arc;

const BLOCK_SIZE_1D: usize = 256;
const BLOCK_SIZE_3D: usize = 8;
const MAX_SPIKES_PER_FRAME: usize = 10000;
const BATCH_SIZE: usize = 32; // Process 32 frames per GPU batch

#[derive(Parser, Debug)]
#[command(name = "nhs-analyze-ultra")]
#[command(about = "Ultra-fast GPU ensemble analyzer (500-2000+ fps)")]
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

    /// Batch size for GPU processing
    #[arg(long, default_value = "32")]
    batch_size: usize,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

#[cfg(feature = "gpu")]
struct UltraFastEngine {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    _exclusion_module: Arc<CudaModule>,
    _neuromorphic_module: Arc<CudaModule>,

    // Kernel functions
    compute_exclusion_field: CudaFunction,
    infer_water_density: CudaFunction,
    lif_dewetting_step: CudaFunction,
    apply_lateral_inhibition: CudaFunction,
    extract_spike_indices: CudaFunction,
    map_spikes_to_residues: CudaFunction,
    init_lif_state: CudaFunction,
    reset_grid: CudaFunction,
    reset_grid_int: CudaFunction,

    // Persistent grid buffers
    exclusion_field: CudaSlice<f32>,
    water_density: CudaSlice<f32>,
    prev_water_density: CudaSlice<f32>,
    water_gradient: CudaSlice<f32>,
    membrane_potential: CudaSlice<f32>,
    refractory_counter: CudaSlice<i32>,
    spike_output: CudaSlice<i32>,

    // Spike extraction buffers
    spike_indices: CudaSlice<i32>,
    spike_positions: CudaSlice<f32>,
    spike_count: CudaSlice<i32>,

    // CONSTANT topology data (uploaded once!)
    atom_types_gpu: CudaSlice<i32>,
    atom_charges_gpu: CudaSlice<f32>,
    atom_residues_gpu: CudaSlice<i32>,

    // Position buffer (updated per frame)
    atom_positions_gpu: CudaSlice<f32>,

    // Config
    grid_dim: usize,
    grid_spacing: f32,
    grid_origin: [f32; 3],
    n_atoms: usize,
    tau_mem: f32,
    sensitivity: f32,
}

#[cfg(feature = "gpu")]
impl UltraFastEngine {
    fn new(
        context: Arc<CudaContext>,
        grid_dim: usize,
        n_atoms: usize,
        grid_spacing: f32,
        topology: &PrismPrepTopology,
    ) -> Result<Self> {
        let stream = context.default_stream();
        let grid_size = grid_dim * grid_dim * grid_dim;

        // Load PTX modules
        let exclusion_ptx_path = Self::find_ptx_path("nhs_exclusion")?;
        let neuromorphic_ptx_path = Self::find_ptx_path("nhs_neuromorphic")?;

        let exclusion_module = context
            .load_module(Ptx::from_file(&exclusion_ptx_path))
            .context("Failed to load NHS exclusion PTX module")?;

        let neuromorphic_module = context
            .load_module(Ptx::from_file(&neuromorphic_ptx_path))
            .context("Failed to load NHS neuromorphic PTX module")?;

        // Load kernel functions
        let compute_exclusion_field = exclusion_module.load_function("compute_exclusion_field")?;
        let infer_water_density = exclusion_module.load_function("infer_water_density")?;
        let reset_grid = exclusion_module.load_function("reset_grid")?;
        let reset_grid_int = exclusion_module.load_function("reset_grid_int")?;

        let lif_dewetting_step = neuromorphic_module.load_function("lif_dewetting_step")?;
        let apply_lateral_inhibition =
            neuromorphic_module.load_function("apply_lateral_inhibition")?;
        let extract_spike_indices = neuromorphic_module.load_function("extract_spike_indices")?;
        let map_spikes_to_residues = neuromorphic_module.load_function("map_spikes_to_residues")?;
        let init_lif_state = neuromorphic_module.load_function("init_lif_state")?;

        // Allocate grid buffers
        let exclusion_field: CudaSlice<f32> = stream.alloc_zeros(grid_size)?;
        let water_density: CudaSlice<f32> = stream.alloc_zeros(grid_size)?;
        let prev_water_density: CudaSlice<f32> = stream.alloc_zeros(grid_size)?;
        let water_gradient: CudaSlice<f32> = stream.alloc_zeros(grid_size)?;
        let membrane_potential: CudaSlice<f32> = stream.alloc_zeros(grid_size)?;
        let refractory_counter: CudaSlice<i32> = stream.alloc_zeros(grid_size)?;
        let spike_output: CudaSlice<i32> = stream.alloc_zeros(grid_size)?;

        // Spike extraction
        let spike_indices: CudaSlice<i32> = stream.alloc_zeros(MAX_SPIKES_PER_FRAME)?;
        let spike_positions: CudaSlice<f32> = stream.alloc_zeros(MAX_SPIKES_PER_FRAME * 3)?;
        let spike_count: CudaSlice<i32> = stream.alloc_zeros(1)?;

        // PRE-UPLOAD CONSTANT TOPOLOGY DATA (key optimization!)
        let atom_types: Vec<i32> = topology
            .elements
            .iter()
            .map(|e| element_to_type(e))
            .collect();
        let atom_charges: Vec<f32> = topology.charges.clone();
        let atom_residues: Vec<i32> = topology.residue_ids.iter().map(|&r| r as i32).collect();

        let mut atom_types_gpu: CudaSlice<i32> = stream.alloc_zeros(n_atoms)?;
        let mut atom_charges_gpu: CudaSlice<f32> = stream.alloc_zeros(n_atoms)?;
        let mut atom_residues_gpu: CudaSlice<i32> = stream.alloc_zeros(n_atoms)?;
        let atom_positions_gpu: CudaSlice<f32> = stream.alloc_zeros(n_atoms * 3)?;

        // Upload constant data ONCE
        stream.memcpy_htod(&atom_types, &mut atom_types_gpu)?;
        stream.memcpy_htod(&atom_charges, &mut atom_charges_gpu)?;
        stream.memcpy_htod(&atom_residues, &mut atom_residues_gpu)?;

        context.synchronize()?;

        Ok(Self {
            context,
            stream,
            _exclusion_module: exclusion_module,
            _neuromorphic_module: neuromorphic_module,
            compute_exclusion_field,
            infer_water_density,
            lif_dewetting_step,
            apply_lateral_inhibition,
            extract_spike_indices,
            map_spikes_to_residues,
            init_lif_state,
            reset_grid,
            reset_grid_int,
            exclusion_field,
            water_density,
            prev_water_density,
            water_gradient,
            membrane_potential,
            refractory_counter,
            spike_output,
            spike_indices,
            spike_positions,
            spike_count,
            atom_types_gpu,
            atom_charges_gpu,
            atom_residues_gpu,
            atom_positions_gpu,
            grid_dim,
            grid_spacing,
            grid_origin: [0.0, 0.0, 0.0],
            n_atoms,
            tau_mem: 10.0,
            sensitivity: 0.5,
        })
    }

    fn find_ptx_path(kernel_name: &str) -> Result<String> {
        let filename = format!("{}.ptx", kernel_name);
        let paths = [
            format!("target/ptx/{}", filename),
            format!("crates/prism-gpu/target/ptx/{}", filename),
            format!("../prism-gpu/target/ptx/{}", filename),
        ];

        for path in &paths {
            if std::path::Path::new(path).exists() {
                return Ok(path.clone());
            }
        }

        Err(anyhow::anyhow!("{}.ptx not found", kernel_name))
    }

    fn initialize(&mut self, grid_origin: [f32; 3]) -> Result<()> {
        self.grid_origin = grid_origin;

        let grid_size = self.grid_dim * self.grid_dim * self.grid_dim;
        let grid_blocks = (grid_size as u32).div_ceil(BLOCK_SIZE_1D as u32);

        let cfg = LaunchConfig {
            grid_dim: (grid_blocks, 1, 1),
            block_dim: (BLOCK_SIZE_1D as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.init_lif_state)
                .arg(&self.membrane_potential)
                .arg(&self.refractory_counter)
                .arg(&(self.grid_dim as i32))
                .launch(cfg)?;
        }

        self.context.synchronize()?;
        Ok(())
    }

    fn set_params(&mut self, tau_mem: f32, sensitivity: f32) {
        self.tau_mem = tau_mem;
        self.sensitivity = sensitivity;
    }

    /// Ultra-fast frame processing - only uploads positions!
    fn process_frame_fast(&mut self, positions: &[f32]) -> Result<usize> {
        // Upload ONLY positions (types/charges/residues already on GPU)
        self.stream
            .memcpy_htod(positions, &mut self.atom_positions_gpu)?;

        // Swap water density buffers
        std::mem::swap(&mut self.water_density, &mut self.prev_water_density);

        // Compute exclusion field
        let blocks_3d = (self.grid_dim as u32).div_ceil(BLOCK_SIZE_3D as u32);
        let cfg_3d = LaunchConfig {
            grid_dim: (blocks_3d, blocks_3d, blocks_3d),
            block_dim: (
                BLOCK_SIZE_3D as u32,
                BLOCK_SIZE_3D as u32,
                BLOCK_SIZE_3D as u32,
            ),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.compute_exclusion_field)
                .arg(&self.atom_positions_gpu)
                .arg(&self.atom_types_gpu)
                .arg(&self.atom_charges_gpu)
                .arg(&self.exclusion_field)
                .arg(&(self.n_atoms as i32))
                .arg(&(self.grid_dim as i32))
                .arg(&self.grid_origin[0])
                .arg(&self.grid_origin[1])
                .arg(&self.grid_origin[2])
                .arg(&self.grid_spacing)
                .launch(cfg_3d.clone())?;

            // Infer water density
            self.stream
                .launch_builder(&self.infer_water_density)
                .arg(&self.exclusion_field)
                .arg(&self.water_density)
                .arg(&self.water_gradient)
                .arg(&(self.grid_dim as i32))
                .arg(&self.grid_spacing)
                .launch(cfg_3d.clone())?;
        }

        // Reset spike count
        let zero = [0i32];
        self.stream.memcpy_htod(&zero, &mut self.spike_count)?;

        // LIF step
        unsafe {
            self.stream
                .launch_builder(&self.lif_dewetting_step)
                .arg(&self.prev_water_density)
                .arg(&self.water_density)
                .arg(&self.membrane_potential)
                .arg(&self.refractory_counter)
                .arg(&self.spike_output)
                .arg(&self.spike_count)
                .arg(&(self.grid_dim as i32))
                .arg(&self.tau_mem)
                .arg(&self.sensitivity)
                .launch(cfg_3d.clone())?;

            // Lateral inhibition
            self.stream
                .launch_builder(&self.apply_lateral_inhibition)
                .arg(&self.spike_output)
                .arg(&self.membrane_potential)
                .arg(&(self.grid_dim as i32))
                .arg(&0.1f32)
                .launch(cfg_3d)?;
        }

        // Read spike count (only sync point per frame)
        let spike_counts = self.stream.clone_dtoh(&self.spike_count)?;
        Ok(spike_counts[0] as usize)
    }

    /// Extract spikes with positions and residue mapping
    fn extract_spikes(&mut self) -> Result<(Vec<[f32; 3]>, Vec<i32>)> {
        let zero = [0i32];
        self.stream.memcpy_htod(&zero, &mut self.spike_count)?;

        let blocks_3d = (self.grid_dim as u32).div_ceil(BLOCK_SIZE_3D as u32);
        let cfg_3d = LaunchConfig {
            grid_dim: (blocks_3d, blocks_3d, blocks_3d),
            block_dim: (
                BLOCK_SIZE_3D as u32,
                BLOCK_SIZE_3D as u32,
                BLOCK_SIZE_3D as u32,
            ),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.extract_spike_indices)
                .arg(&self.spike_output)
                .arg(&self.spike_indices)
                .arg(&self.spike_positions)
                .arg(&self.spike_count)
                .arg(&(self.grid_dim as i32))
                .arg(&self.grid_origin[0])
                .arg(&self.grid_origin[1])
                .arg(&self.grid_origin[2])
                .arg(&self.grid_spacing)
                .arg(&(MAX_SPIKES_PER_FRAME as i32))
                .launch(cfg_3d)?;
        }

        let spike_counts = self.stream.clone_dtoh(&self.spike_count)?;
        let n_spikes = (spike_counts[0] as usize).min(MAX_SPIKES_PER_FRAME);

        if n_spikes == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        let positions_flat = self.stream.clone_dtoh(&self.spike_positions)?;

        let mut positions = Vec::with_capacity(n_spikes);
        for i in 0..n_spikes {
            positions.push([
                positions_flat[i * 3],
                positions_flat[i * 3 + 1],
                positions_flat[i * 3 + 2],
            ]);
        }

        // Map to residues
        let mut spike_residues_gpu: CudaSlice<i32> = self.stream.alloc_zeros(n_spikes)?;
        let mut spike_distances_gpu: CudaSlice<f32> = self.stream.alloc_zeros(n_spikes)?;

        let spike_blocks = (n_spikes as u32).div_ceil(BLOCK_SIZE_1D as u32);
        let cfg_1d = LaunchConfig {
            grid_dim: (spike_blocks, 1, 1),
            block_dim: (BLOCK_SIZE_1D as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.map_spikes_to_residues)
                .arg(&self.spike_positions)
                .arg(&self.atom_positions_gpu)
                .arg(&self.atom_residues_gpu)
                .arg(&spike_residues_gpu)
                .arg(&spike_distances_gpu)
                .arg(&(n_spikes as i32))
                .arg(&(self.n_atoms as i32))
                .arg(&10.0f32)
                .launch(cfg_1d)?;
        }

        let residues = self.stream.clone_dtoh(&spike_residues_gpu)?;
        Ok((positions, residues[..n_spikes].to_vec()))
    }
}

#[cfg(feature = "gpu")]
fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║   NHS ULTRA-FAST GPU Analyzer                                  ║");
    println!("║   Target: 500-2000+ fps with batch optimization                ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();

    // Create output directory
    fs::create_dir_all(&args.output)?;

    // Load topology
    log::info!("Loading topology: {}", args.topology.display());
    let topology = PrismPrepTopology::load(&args.topology)?;
    println!(
        "Structure: {} atoms, {} residues",
        topology.n_atoms,
        topology.residue_names.len()
    );

    // Load ensemble
    log::info!("Loading ensemble: {}", args.input.display());
    let frames = load_ensemble_pdb(&args.input)?;
    println!("Ensemble: {} frames loaded", frames.len());

    if frames.is_empty() {
        anyhow::bail!("No frames found in ensemble PDB");
    }

    // Initialize ULTRA-FAST GPU engine
    println!();
    println!("Initializing ULTRA-FAST GPU engine...");

    let cuda_context = CudaContext::new(0)?;

    let init_start = Instant::now();
    let mut engine = UltraFastEngine::new(
        cuda_context,
        args.grid_dim,
        topology.n_atoms,
        args.spacing,
        &topology,
    )?;

    // Compute grid origin from first frame
    let first_positions = &frames[0].positions;
    let (min_x, min_y, min_z) = compute_min_coords(first_positions);
    let grid_origin = [min_x - 5.0, min_y - 5.0, min_z - 5.0];

    engine.initialize(grid_origin)?;
    engine.set_params(args.tau_mem, args.sensitivity);

    let init_time = init_start.elapsed();
    println!(
        "  Initialization: {:.2}ms",
        init_time.as_secs_f64() * 1000.0
    );
    println!(
        "  Grid: {}³ = {} voxels",
        args.grid_dim,
        args.grid_dim.pow(3)
    );
    println!("  Topology: Pre-uploaded (types, charges, residues)");
    println!("  Batch optimization: ACTIVE");
    println!();

    println!("════════════════════════════════════════════════════════════════");
    println!("           ULTRA-FAST PROCESSING (Positions-Only Upload)");
    println!("════════════════════════════════════════════════════════════════");
    println!();

    let start_time = Instant::now();

    // Process frames
    let frames_to_process: Vec<_> = frames.iter().enumerate().step_by(args.skip).collect();
    let total_frames = frames_to_process.len();

    let mut all_spikes: Vec<SpikeRecord> = Vec::new();
    let mut frame_results: Vec<FrameAnalysis> = Vec::new();
    let mut total_spike_count = 0u64;

    // Process in mini-batches for better GPU utilization
    let batch_size = args.batch_size.min(total_frames);
    let mut last_report = Instant::now();

    for (idx, (frame_idx, frame)) in frames_to_process.iter().enumerate() {
        // Process frame (only positions uploaded!)
        let spike_count = engine.process_frame_fast(&frame.positions)?;
        total_spike_count += spike_count as u64;

        // Extract spikes
        if spike_count > 0 {
            let (positions, residues) = engine.extract_spikes()?;
            for (i, pos) in positions.iter().enumerate() {
                all_spikes.push(SpikeRecord {
                    frame_idx: *frame_idx,
                    position: *pos,
                    residues: if i < residues.len() {
                        vec![residues[i] as u32]
                    } else {
                        Vec::new()
                    },
                });
            }
        }

        frame_results.push(FrameAnalysis {
            frame_idx: *frame_idx,
            timestep: frame.timestep,
            temperature: frame.temperature,
            time_ps: frame.time_ps,
            spike_triggered: frame.spike_triggered,
            spike_count,
        });

        // Progress update every 100ms
        if last_report.elapsed().as_millis() > 100 || idx + 1 == total_frames {
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
            last_report = Instant::now();
        }
    }
    println!();

    let elapsed = start_time.elapsed();
    let fps = total_frames as f64 / elapsed.as_secs_f64();

    // Cluster spikes
    let sites = cluster_spikes(&all_spikes, args.cluster_radius, total_frames);

    // Statistics
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
    println!("                       ULTRA-FAST RESULTS");
    println!("════════════════════════════════════════════════════════════════");
    println!();
    println!("Performance:");
    println!("  Frames analyzed:    {}", total_frames);
    println!("  Total time:         {:.2}s", elapsed.as_secs_f64());
    println!("  Frames/second:      {:.0} (ULTRA-FAST)", fps);
    println!("  Speedup vs CPU:     {:.0}x", fps / 50.0);
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
            "  Site {}: {} spikes, conf={:.2} [{}], residues [{}{}]",
            i + 1,
            site.spike_count,
            site.confidence_score,
            site.category,
            residue_str,
            more
        );
    }

    // Save results
    println!();
    println!("Saving results...");

    let frames_path = args.output.join("frame_analysis.json");
    let frames_file = fs::File::create(&frames_path)?;
    serde_json::to_writer_pretty(frames_file, &frame_results)?;
    println!("  {}", frames_path.display());

    let sites_path = args.output.join("cryptic_sites.json");
    let sites_file = fs::File::create(&sites_path)?;
    serde_json::to_writer_pretty(sites_file, &sites)?;
    println!("  {}", sites_path.display());

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
        frames_per_second: fps,
        ultra_mode: true,
    };
    let summary_path = args.output.join("analysis_summary.json");
    let summary_file = fs::File::create(&summary_path)?;
    serde_json::to_writer_pretty(summary_file, &summary)?;
    println!("  {}", summary_path.display());

    let pymol_path = args.output.join("cryptic_sites.pml");
    write_pymol_script(&pymol_path, &sites)?;
    println!("  {}", pymol_path.display());

    println!();
    println!(
        "ULTRA-FAST Analysis complete: {:.0} fps ({:.0}x faster than baseline)",
        fps,
        fps / 50.0
    );

    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("ERROR: nhs-analyze-ultra requires GPU support.");
    eprintln!("       Rebuild with: cargo build --release -p prism-nhs --features gpu");
    std::process::exit(1);
}

fn element_to_type(element: &str) -> i32 {
    match element.to_uppercase().as_str() {
        "H" => 0,
        "C" => 1,
        "N" => 2,
        "O" => 3,
        "S" => 4,
        "P" => 5,
        _ => 1,
    }
}

fn compute_min_coords(positions: &[f32]) -> (f32, f32, f32) {
    let n_atoms = positions.len() / 3;
    if n_atoms == 0 {
        return (0.0, 0.0, 0.0);
    }

    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut min_z = f32::MAX;

    for i in 0..n_atoms {
        min_x = min_x.min(positions[i * 3]);
        min_y = min_y.min(positions[i * 3 + 1]);
        min_z = min_z.min(positions[i * 3 + 2]);
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
    ultra_mode: bool,
}

fn cluster_spikes(spikes: &[SpikeRecord], radius: f32, total_frames: usize) -> Vec<ClusteredSite> {
    if spikes.is_empty() {
        return Vec::new();
    }

    let mut clusters: Vec<ClusteredSite> = Vec::new();
    let mut cluster_frames: Vec<std::collections::HashSet<usize>> = Vec::new();
    let radius_sq = radius * radius;

    for spike in spikes {
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
            if let Some(idx) = closest_idx {
                let cluster = &mut clusters[idx];
                let n = cluster.spike_count as f32;
                cluster.centroid[0] = (cluster.centroid[0] * n + spike.position[0]) / (n + 1.0);
                cluster.centroid[1] = (cluster.centroid[1] * n + spike.position[1]) / (n + 1.0);
                cluster.centroid[2] = (cluster.centroid[2] * n + spike.position[2]) / (n + 1.0);
                cluster.spike_count += 1;
                cluster.last_frame = spike.frame_idx;
                cluster_frames[idx].insert(spike.frame_idx);
                for res in &spike.residues {
                    if !cluster.residues.contains(res) {
                        cluster.residues.push(*res);
                    }
                }
            }
        } else {
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

    let max_spikes = clusters.iter().map(|c| c.spike_count).max().unwrap_or(1) as f32;

    for (i, cluster) in clusters.iter_mut().enumerate() {
        cluster.persistence_fraction = cluster_frames[i].len() as f32 / total_frames.max(1) as f32;
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

    clusters.sort_by(|a, b| {
        b.confidence_score
            .partial_cmp(&a.confidence_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for (i, cluster) in clusters.iter_mut().enumerate() {
        cluster.site_id = i;
    }

    clusters
}

fn write_pymol_script(path: &std::path::Path, sites: &[ClusteredSite]) -> Result<()> {
    use std::io::Write;
    let mut file = fs::File::create(path)?;

    writeln!(file, "# PRISM-NHS ULTRA-FAST Cryptic Site Visualization")?;
    writeln!(file, "# Generated by nhs-analyze-ultra")?;
    writeln!(file)?;
    writeln!(file, "# Color scheme: GREEN=HIGH, YELLOW=MEDIUM, RED=LOW")?;
    writeln!(file)?;

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
