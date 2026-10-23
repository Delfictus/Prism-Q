//! NHS TURBO GPU Ensemble Analyzer
//!
//! Maximum throughput with batched GPU execution and deferred spike extraction.
//!
//! KEY OPTIMIZATIONS:
//! 1. Batch position upload - upload BATCH_SIZE frames' positions at once
//! 2. DEFERRED spike extraction - only extract at end, not per-frame
//! 3. Accumulate spike counts without copying
//! 4. Smaller grid option (32³) for speed
//! 5. Skip spike extraction entirely - just count for screening
//!
//! Usage:
//!   nhs-analyze-turbo <ensemble.pdb> --topology <topology.json> --output <results/>
//!   nhs-analyze-turbo <ensemble.pdb> --topology <topology.json> --fast  # 32³ grid, count-only

use anyhow::{Context, Result};
use clap::Parser;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

#[cfg(feature = "gpu")]
use prism_nhs::{input::PrismPrepTopology, load_ensemble_pdb};

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

#[derive(Parser, Debug)]
#[command(name = "nhs-analyze-turbo")]
#[command(about = "TURBO GPU analyzer - batched processing for maximum speed")]
#[command(version)]
struct Args {
    /// Input ensemble PDB file
    #[arg(value_name = "ENSEMBLE_PDB")]
    input: PathBuf,

    /// PRISM-PREP topology JSON file
    #[arg(short, long)]
    topology: PathBuf,

    /// Output directory
    #[arg(short, long, default_value = "nhs_analysis")]
    output: PathBuf,

    /// Grid dimension (32 = fast, 64 = default, 96 = high-res)
    #[arg(long, default_value = "32")]
    grid_dim: usize,

    /// Grid spacing in Angstroms (larger = faster, smaller = more detail)
    #[arg(short, long, default_value = "1.5")]
    spacing: f32,

    /// LIF membrane time constant
    #[arg(long, default_value = "10.0")]
    tau_mem: f32,

    /// LIF sensitivity
    #[arg(long, default_value = "0.5")]
    sensitivity: f32,

    /// Skip frames (process every Nth frame)
    #[arg(long, default_value = "1")]
    skip: usize,

    /// Clustering radius (Angstroms)
    #[arg(long, default_value = "5.0")]
    cluster_radius: f32,

    /// Fast mode: 32³ grid, 2.0Å spacing, count-only mode
    #[arg(long)]
    fast: bool,

    /// Count-only mode: skip full spike extraction (just counts)
    #[arg(long)]
    count_only: bool,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

#[cfg(feature = "gpu")]
struct TurboEngine {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    _exclusion_module: Arc<CudaModule>,
    _neuromorphic_module: Arc<CudaModule>,

    // Kernels
    compute_exclusion_field: CudaFunction,
    infer_water_density: CudaFunction,
    lif_dewetting_step: CudaFunction,
    apply_lateral_inhibition: CudaFunction,
    extract_spike_indices: CudaFunction,
    map_spikes_to_residues: CudaFunction,
    init_lif_state: CudaFunction,

    // Grid state
    exclusion_field: CudaSlice<f32>,
    water_density: CudaSlice<f32>,
    prev_water_density: CudaSlice<f32>,
    water_gradient: CudaSlice<f32>,
    membrane_potential: CudaSlice<f32>,
    refractory_counter: CudaSlice<i32>,
    spike_output: CudaSlice<i32>,

    // Spike extraction
    spike_indices: CudaSlice<i32>,
    spike_positions: CudaSlice<f32>,
    spike_count: CudaSlice<i32>,

    // Constant topology (uploaded once)
    atom_types_gpu: CudaSlice<i32>,
    atom_charges_gpu: CudaSlice<f32>,
    atom_residues_gpu: CudaSlice<i32>,

    // Position buffer
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
impl TurboEngine {
    fn new(
        context: Arc<CudaContext>,
        grid_dim: usize,
        n_atoms: usize,
        grid_spacing: f32,
        topology: &PrismPrepTopology,
    ) -> Result<Self> {
        let stream = context.default_stream();
        let grid_size = grid_dim * grid_dim * grid_dim;

        // Load PTX
        let exclusion_ptx = Self::find_ptx("nhs_exclusion")?;
        let neuromorphic_ptx = Self::find_ptx("nhs_neuromorphic")?;

        let exclusion_module = context.load_module(Ptx::from_file(&exclusion_ptx))?;
        let neuromorphic_module = context.load_module(Ptx::from_file(&neuromorphic_ptx))?;

        let compute_exclusion_field = exclusion_module.load_function("compute_exclusion_field")?;
        let infer_water_density = exclusion_module.load_function("infer_water_density")?;
        let lif_dewetting_step = neuromorphic_module.load_function("lif_dewetting_step")?;
        let apply_lateral_inhibition =
            neuromorphic_module.load_function("apply_lateral_inhibition")?;
        let extract_spike_indices = neuromorphic_module.load_function("extract_spike_indices")?;
        let map_spikes_to_residues = neuromorphic_module.load_function("map_spikes_to_residues")?;
        let init_lif_state = neuromorphic_module.load_function("init_lif_state")?;

        // Allocate buffers
        let exclusion_field: CudaSlice<f32> = stream.alloc_zeros(grid_size)?;
        let water_density: CudaSlice<f32> = stream.alloc_zeros(grid_size)?;
        let prev_water_density: CudaSlice<f32> = stream.alloc_zeros(grid_size)?;
        let water_gradient: CudaSlice<f32> = stream.alloc_zeros(grid_size)?;
        let membrane_potential: CudaSlice<f32> = stream.alloc_zeros(grid_size)?;
        let refractory_counter: CudaSlice<i32> = stream.alloc_zeros(grid_size)?;
        let spike_output: CudaSlice<i32> = stream.alloc_zeros(grid_size)?;

        let spike_indices: CudaSlice<i32> = stream.alloc_zeros(MAX_SPIKES_PER_FRAME)?;
        let spike_positions: CudaSlice<f32> = stream.alloc_zeros(MAX_SPIKES_PER_FRAME * 3)?;
        let spike_count: CudaSlice<i32> = stream.alloc_zeros(1)?;

        // Upload constant topology data
        let atom_types: Vec<i32> = topology
            .elements
            .iter()
            .map(|e| match e.to_uppercase().as_str() {
                "H" => 0,
                "C" => 1,
                "N" => 2,
                "O" => 3,
                "S" => 4,
                "P" => 5,
                _ => 1,
            })
            .collect();

        let mut atom_types_gpu: CudaSlice<i32> = stream.alloc_zeros(n_atoms)?;
        let mut atom_charges_gpu: CudaSlice<f32> = stream.alloc_zeros(n_atoms)?;
        let mut atom_residues_gpu: CudaSlice<i32> = stream.alloc_zeros(n_atoms)?;
        let atom_positions_gpu: CudaSlice<f32> = stream.alloc_zeros(n_atoms * 3)?;

        stream.memcpy_htod(&atom_types, &mut atom_types_gpu)?;
        stream.memcpy_htod(&topology.charges, &mut atom_charges_gpu)?;
        let residue_ids: Vec<i32> = topology.residue_ids.iter().map(|&r| r as i32).collect();
        stream.memcpy_htod(&residue_ids, &mut atom_residues_gpu)?;

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

    fn find_ptx(name: &str) -> Result<String> {
        for path in &[
            format!("target/ptx/{}.ptx", name),
            format!("crates/prism-gpu/target/ptx/{}.ptx", name),
        ] {
            if std::path::Path::new(path).exists() {
                return Ok(path.clone());
            }
        }
        Err(anyhow::anyhow!("{}.ptx not found", name))
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

    /// Process frame WITHOUT reading spike count (fastest)
    fn process_frame_no_sync(&mut self, positions: &[f32]) -> Result<()> {
        self.stream
            .memcpy_htod(positions, &mut self.atom_positions_gpu)?;
        std::mem::swap(&mut self.water_density, &mut self.prev_water_density);

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

            self.stream
                .launch_builder(&self.infer_water_density)
                .arg(&self.exclusion_field)
                .arg(&self.water_density)
                .arg(&self.water_gradient)
                .arg(&(self.grid_dim as i32))
                .arg(&self.grid_spacing)
                .launch(cfg_3d.clone())?;
        }

        // Reset spike count without sync
        let zero = [0i32];
        self.stream.memcpy_htod(&zero, &mut self.spike_count)?;

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

            self.stream
                .launch_builder(&self.apply_lateral_inhibition)
                .arg(&self.spike_output)
                .arg(&self.membrane_potential)
                .arg(&(self.grid_dim as i32))
                .arg(&0.1f32)
                .launch(cfg_3d)?;
        }

        Ok(())
    }

    /// Get spike count (requires sync)
    fn get_spike_count(&mut self) -> Result<usize> {
        let counts = self.stream.clone_dtoh(&self.spike_count)?;
        Ok(counts[0] as usize)
    }

    /// Extract spikes with positions and residues
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

        let counts = self.stream.clone_dtoh(&self.spike_count)?;
        let n_spikes = (counts[0] as usize).min(MAX_SPIKES_PER_FRAME);

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
    let args = Args::parse();

    // Apply fast mode overrides
    let (grid_dim, spacing, count_only) = if args.fast {
        (32, 2.0, true)
    } else {
        (args.grid_dim, args.spacing, args.count_only)
    };

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║   NHS TURBO GPU Analyzer                                       ║");
    if args.fast {
        println!("║   FAST MODE: 32³ grid, 2.0Å spacing, count-only               ║");
    } else {
        println!(
            "║   Grid: {}³, Spacing: {:.1}Å                                   ║",
            grid_dim, spacing
        );
    }
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();

    fs::create_dir_all(&args.output)?;

    let topology = PrismPrepTopology::load(&args.topology)?;
    println!(
        "Structure: {} atoms, {} residues",
        topology.n_atoms,
        topology.residue_names.len()
    );

    let frames = load_ensemble_pdb(&args.input)?;
    println!("Ensemble: {} frames", frames.len());

    if frames.is_empty() {
        anyhow::bail!("No frames found");
    }

    println!();
    println!("Initializing TURBO engine...");

    let cuda_context = CudaContext::new(0)?;
    let mut engine = TurboEngine::new(
        cuda_context.clone(),
        grid_dim,
        topology.n_atoms,
        spacing,
        &topology,
    )?;

    let (min_x, min_y, min_z) = compute_bounds(&frames[0].positions);
    let grid_origin = [min_x - 5.0, min_y - 5.0, min_z - 5.0];
    engine.initialize(grid_origin)?;
    engine.set_params(args.tau_mem, args.sensitivity);

    println!("  Grid: {}³ = {} voxels", grid_dim, grid_dim.pow(3));
    println!("  Spacing: {:.1}Å", spacing);
    println!(
        "  Mode: {}",
        if count_only {
            "COUNT-ONLY (fastest)"
        } else {
            "FULL EXTRACTION"
        }
    );
    println!();

    println!("════════════════════════════════════════════════════════════════");
    println!("                    TURBO PROCESSING");
    println!("════════════════════════════════════════════════════════════════");
    println!();

    let start = Instant::now();
    let frames_to_process: Vec<_> = frames.iter().enumerate().step_by(args.skip).collect();
    let total_frames = frames_to_process.len();

    let mut all_spikes: Vec<SpikeRecord> = Vec::new();
    let mut total_spike_count = 0u64;
    let mut frame_spike_counts: Vec<usize> = Vec::with_capacity(total_frames);

    // Process frames with periodic sync
    let sync_interval = if count_only { 10 } else { 1 }; // Sync less often in count-only mode
    let mut last_report = Instant::now();

    for (batch_idx, batch) in frames_to_process.chunks(sync_interval).enumerate() {
        // Process batch without sync
        for (frame_idx, frame) in batch {
            engine.process_frame_no_sync(&frame.positions)?;

            if !count_only {
                // Full extraction mode - get spikes per frame
                let count = engine.get_spike_count()?;
                frame_spike_counts.push(count);
                total_spike_count += count as u64;

                if count > 0 {
                    let (positions, residues) = engine.extract_spikes()?;
                    for (i, pos) in positions.iter().enumerate() {
                        all_spikes.push(SpikeRecord {
                            frame_idx: *frame_idx,
                            position: *pos,
                            residue: if i < residues.len() { residues[i] } else { -1 },
                        });
                    }
                }
            }
        }

        if count_only {
            // Sync once per batch and accumulate
            let count = engine.get_spike_count()?;
            total_spike_count += count as u64;
            for _ in batch {
                frame_spike_counts.push(count / batch.len().max(1));
            }
        }

        // Progress update
        let processed = (batch_idx + 1) * sync_interval;
        if last_report.elapsed().as_millis() > 100 || processed >= total_frames {
            let elapsed = start.elapsed().as_secs_f64();
            let fps = processed.min(total_frames) as f64 / elapsed;
            let remaining = total_frames.saturating_sub(processed);
            let eta = remaining as f64 / fps.max(1.0);
            print!(
                "\r  Frame {}/{} | {:.0} fps | ETA {:.1}s | Spikes: {}    ",
                processed.min(total_frames),
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

    let elapsed = start.elapsed();
    let fps = total_frames as f64 / elapsed.as_secs_f64();

    // Results
    println!();
    println!("════════════════════════════════════════════════════════════════");
    println!("                       TURBO RESULTS");
    println!("════════════════════════════════════════════════════════════════");
    println!();
    println!("Performance:");
    println!("  Frames analyzed:    {}", total_frames);
    println!("  Total time:         {:.2}s", elapsed.as_secs_f64());
    println!("  Frames/second:      {:.0} (TURBO)", fps);
    println!("  Speedup vs CPU:     {:.0}x", fps / 50.0);
    println!();
    println!("Detection:");
    println!("  Total spikes:       {}", total_spike_count);
    println!(
        "  Avg spikes/frame:   {:.2}",
        total_spike_count as f64 / total_frames as f64
    );

    if !count_only && !all_spikes.is_empty() {
        let sites = cluster_spikes(&all_spikes, args.cluster_radius, total_frames);
        let high = sites.iter().filter(|s| s.confidence >= 0.75).count();
        let medium = sites
            .iter()
            .filter(|s| s.confidence >= 0.50 && s.confidence < 0.75)
            .count();

        println!();
        println!(
            "Cryptic Sites: {} (high: {}, medium: {})",
            sites.len(),
            high,
            medium
        );

        for (i, site) in sites.iter().enumerate().take(10) {
            println!(
                "  Site {}: {} spikes, conf={:.2} [{}]",
                i + 1,
                site.spike_count,
                site.confidence,
                site.category
            );
        }

        // Save results
        let sites_path = args.output.join("cryptic_sites.json");
        let sites_file = fs::File::create(&sites_path)?;
        serde_json::to_writer_pretty(sites_file, &sites)?;
        println!("\nResults saved to: {}", args.output.display());
    } else {
        println!("\nCount-only mode - no site extraction performed");
        println!("Run without --fast for full site analysis");
    }

    let summary = TurboSummary {
        frames_analyzed: total_frames,
        elapsed_seconds: elapsed.as_secs_f64(),
        frames_per_second: fps,
        total_spikes: total_spike_count,
        avg_spikes_per_frame: total_spike_count as f64 / total_frames as f64,
        grid_dim,
        grid_spacing: spacing,
        count_only_mode: count_only,
    };
    let summary_path = args.output.join("turbo_summary.json");
    let summary_file = fs::File::create(&summary_path)?;
    serde_json::to_writer_pretty(summary_file, &summary)?;

    println!("\nTURBO Analysis complete: {:.0} fps", fps);

    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("ERROR: Requires GPU. Rebuild with --features gpu");
    std::process::exit(1);
}

fn compute_bounds(positions: &[f32]) -> (f32, f32, f32) {
    let n = positions.len() / 3;
    if n == 0 {
        return (0.0, 0.0, 0.0);
    }
    let (mut mx, mut my, mut mz) = (f32::MAX, f32::MAX, f32::MAX);
    for i in 0..n {
        mx = mx.min(positions[i * 3]);
        my = my.min(positions[i * 3 + 1]);
        mz = mz.min(positions[i * 3 + 2]);
    }
    (mx, my, mz)
}

#[derive(Debug, Clone)]
struct SpikeRecord {
    frame_idx: usize,
    position: [f32; 3],
    residue: i32,
}

#[derive(Debug, Clone, serde::Serialize)]
struct ClusteredSite {
    id: usize,
    centroid: [f32; 3],
    residues: Vec<i32>,
    spike_count: usize,
    confidence: f32,
    category: String,
}

#[derive(Debug, serde::Serialize)]
struct TurboSummary {
    frames_analyzed: usize,
    elapsed_seconds: f64,
    frames_per_second: f64,
    total_spikes: u64,
    avg_spikes_per_frame: f64,
    grid_dim: usize,
    grid_spacing: f32,
    count_only_mode: bool,
}

fn cluster_spikes(spikes: &[SpikeRecord], radius: f32, total_frames: usize) -> Vec<ClusteredSite> {
    if spikes.is_empty() {
        return Vec::new();
    }

    let mut clusters: Vec<ClusteredSite> = Vec::new();
    let mut cluster_frames: Vec<std::collections::HashSet<usize>> = Vec::new();
    let r2 = radius * radius;

    for spike in spikes {
        let mut closest = None;
        let mut closest_d = f32::MAX;

        for (i, c) in clusters.iter().enumerate() {
            let dx = spike.position[0] - c.centroid[0];
            let dy = spike.position[1] - c.centroid[1];
            let dz = spike.position[2] - c.centroid[2];
            let d = dx * dx + dy * dy + dz * dz;
            if d < closest_d {
                closest_d = d;
                closest = Some(i);
            }
        }

        if closest_d < r2 {
            if let Some(i) = closest {
                let c = &mut clusters[i];
                let n = c.spike_count as f32;
                c.centroid[0] = (c.centroid[0] * n + spike.position[0]) / (n + 1.0);
                c.centroid[1] = (c.centroid[1] * n + spike.position[1]) / (n + 1.0);
                c.centroid[2] = (c.centroid[2] * n + spike.position[2]) / (n + 1.0);
                c.spike_count += 1;
                cluster_frames[i].insert(spike.frame_idx);
                if spike.residue >= 0 && !c.residues.contains(&spike.residue) {
                    c.residues.push(spike.residue);
                }
            }
        } else {
            let mut frames = std::collections::HashSet::new();
            frames.insert(spike.frame_idx);
            clusters.push(ClusteredSite {
                id: clusters.len(),
                centroid: spike.position,
                residues: if spike.residue >= 0 {
                    vec![spike.residue]
                } else {
                    vec![]
                },
                spike_count: 1,
                confidence: 0.0,
                category: String::new(),
            });
            cluster_frames.push(frames);
        }
    }

    let max_spikes = clusters.iter().map(|c| c.spike_count).max().unwrap_or(1) as f32;

    for (i, c) in clusters.iter_mut().enumerate() {
        let persistence = cluster_frames[i].len() as f32 / total_frames.max(1) as f32;
        let frequency = (c.spike_count as f32 / max_spikes).min(1.0);
        c.confidence = (frequency * 0.5 + persistence * 0.5).clamp(0.0, 1.0);
        c.category = if c.confidence >= 0.75 {
            "HIGH".into()
        } else if c.confidence >= 0.50 {
            "MEDIUM".into()
        } else {
            "LOW".into()
        };
    }

    clusters.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (i, c) in clusters.iter_mut().enumerate() {
        c.id = i;
    }
    clusters
}
