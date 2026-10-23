//! NHS PRO GPU Analyzer - Publication-Quality Results
//!
//! Key improvements over TURBO:
//! 1. PROPER atom classification (hydrophobic/polar/charged/aromatic based on chemistry)
//! 2. Tuned LIF parameters for better sensitivity
//! 3. Multi-factor confidence scoring (frequency, persistence, coherence, volume)
//! 4. Higher resolution grid support (96³)
//! 5. Site quality metrics (druggability, accessibility)

use anyhow::{Context, Result};
use clap::Parser;
use std::collections::{HashMap, HashSet};
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
const MAX_SPIKES_PER_FRAME: usize = 20000;

// Atom type constants matching the CUDA kernel
const ATOM_HYDROPHOBIC: i32 = 0;
const ATOM_POLAR: i32 = 1;
const ATOM_CHARGED_POS: i32 = 2;
const ATOM_CHARGED_NEG: i32 = 3;
const ATOM_AROMATIC: i32 = 4;
const ATOM_BACKBONE: i32 = 5;

#[derive(Parser, Debug)]
#[command(name = "nhs-analyze-pro")]
#[command(about = "PRO analyzer - Publication-quality cryptic site detection")]
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

    /// Grid dimension (64=standard, 96=high-res, 128=ultra)
    #[arg(long, default_value = "64")]
    grid_dim: usize,

    /// Grid spacing (0.5=high-res, 0.75=balanced, 1.0=fast)
    #[arg(short, long, default_value = "0.75")]
    spacing: f32,

    /// LIF membrane time constant (lower=faster response)
    #[arg(long, default_value = "8.0")]
    tau_mem: f32,

    /// LIF sensitivity (higher=more sensitive)
    #[arg(long, default_value = "1.5")]
    sensitivity: f32,

    /// Clustering radius (Angstroms)
    #[arg(long, default_value = "4.0")]
    cluster_radius: f32,

    /// Minimum spikes for a valid site
    #[arg(long, default_value = "50")]
    min_spikes: usize,

    /// Skip frames
    #[arg(long, default_value = "1")]
    skip: usize,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Frames JSON file with wavelength data (from nhs-cryo-probe --spectroscopy)
    /// Enables wavelength-aware scoring for improved confidence
    #[arg(long)]
    frames_json: Option<PathBuf>,
}

/// Classify atom based on residue name, atom name, and element
fn classify_atom(residue_name: &str, atom_name: &str, element: &str, charge: f32) -> i32 {
    let res = residue_name.trim().to_uppercase();
    let atm = atom_name.trim().to_uppercase();

    // Backbone atoms
    if atm == "CA" || atm == "C" || atm == "N" || atm == "O" {
        return ATOM_BACKBONE;
    }

    // Charged residues
    match res.as_str() {
        "ARG" | "LYS"
            if !atm.starts_with("C")
                || atm == "CZ"
                || atm == "NZ"
                || atm == "NH1"
                || atm == "NH2"
                || atm == "NE" =>
        {
            return ATOM_CHARGED_POS;
        }
        "ASP" | "GLU" if atm == "OD1" || atm == "OD2" || atm == "OE1" || atm == "OE2" => {
            return ATOM_CHARGED_NEG;
        }
        "HIS" if atm == "ND1" || atm == "NE2" => {
            // Histidine can be charged or neutral
            if charge.abs() > 0.3 {
                return ATOM_CHARGED_POS;
            }
            return ATOM_POLAR;
        }
        _ => {}
    }

    // Aromatic residues
    match res.as_str() {
        "PHE" if atm.starts_with("C") && atm != "CA" && atm != "C" && atm != "CB" => {
            return ATOM_AROMATIC;
        }
        "TYR"
            if (atm.starts_with("C") && atm != "CA" && atm != "C" && atm != "CB")
                || atm == "OH" =>
        {
            if atm == "OH" {
                return ATOM_POLAR;
            }
            return ATOM_AROMATIC;
        }
        "TRP" if atm.starts_with("C") && atm != "CA" && atm != "C" && atm != "CB" => {
            return ATOM_AROMATIC;
        }
        _ => {}
    }

    // Hydrophobic residues/atoms
    match res.as_str() {
        "ALA" | "VAL" | "LEU" | "ILE" | "MET" | "PRO" => {
            if element == "C" && !atm.starts_with("C") {
                return ATOM_HYDROPHOBIC;
            }
            if atm == "CB"
                || atm == "CG"
                || atm == "CG1"
                || atm == "CG2"
                || atm == "CD"
                || atm == "CD1"
                || atm == "CD2"
                || atm == "CE"
                || atm == "SD"
            {
                return ATOM_HYDROPHOBIC;
            }
        }
        _ => {}
    }

    // Polar atoms by element/charge
    if element == "O" || element == "N" {
        if charge.abs() > 0.4 {
            if charge > 0.0 {
                return ATOM_CHARGED_POS;
            } else {
                return ATOM_CHARGED_NEG;
            }
        }
        return ATOM_POLAR;
    }

    if element == "S" {
        // Cysteine SG is somewhat polar, Met SD is hydrophobic
        if res == "CYS" {
            return ATOM_POLAR;
        }
        return ATOM_HYDROPHOBIC;
    }

    // Default: carbon and hydrogen are hydrophobic
    if element == "C" || element == "H" {
        return ATOM_HYDROPHOBIC;
    }

    ATOM_POLAR // Default fallback
}

#[cfg(feature = "gpu")]
struct ProEngine {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    _exclusion_module: Arc<CudaModule>,
    _neuromorphic_module: Arc<CudaModule>,

    compute_exclusion_field: CudaFunction,
    infer_water_density: CudaFunction,
    lif_dewetting_step: CudaFunction,
    apply_lateral_inhibition: CudaFunction,
    extract_spike_indices: CudaFunction,
    map_spikes_to_residues: CudaFunction,
    init_lif_state: CudaFunction,

    exclusion_field: CudaSlice<f32>,
    water_density: CudaSlice<f32>,
    prev_water_density: CudaSlice<f32>,
    water_gradient: CudaSlice<f32>,
    membrane_potential: CudaSlice<f32>,
    refractory_counter: CudaSlice<i32>,
    spike_output: CudaSlice<i32>,

    spike_indices: CudaSlice<i32>,
    spike_positions: CudaSlice<f32>,
    spike_count: CudaSlice<i32>,

    atom_types_gpu: CudaSlice<i32>,
    atom_charges_gpu: CudaSlice<f32>,
    atom_residues_gpu: CudaSlice<i32>,
    atom_positions_gpu: CudaSlice<f32>,

    grid_dim: usize,
    grid_spacing: f32,
    grid_origin: [f32; 3],
    n_atoms: usize,
    tau_mem: f32,
    sensitivity: f32,
}

#[cfg(feature = "gpu")]
impl ProEngine {
    fn new(
        context: Arc<CudaContext>,
        grid_dim: usize,
        n_atoms: usize,
        grid_spacing: f32,
        topology: &PrismPrepTopology,
    ) -> Result<Self> {
        let stream = context.default_stream();
        let grid_size = grid_dim * grid_dim * grid_dim;

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

        // PROPER atom classification
        let mut atom_types: Vec<i32> = Vec::with_capacity(n_atoms);
        for i in 0..n_atoms {
            let res_name = &topology.residue_names[i];
            let atom_name = &topology.atom_names[i];
            let element = &topology.elements[i];
            let charge = topology.charges[i];
            atom_types.push(classify_atom(res_name, atom_name, element, charge));
        }

        // Log classification stats
        let mut type_counts = [0usize; 6];
        for &t in &atom_types {
            if t >= 0 && t < 6 {
                type_counts[t as usize] += 1;
            }
        }
        log::info!("Atom classification: Hydrophobic={}, Polar={}, Charged+={}, Charged-={}, Aromatic={}, Backbone={}",
            type_counts[0], type_counts[1], type_counts[2], type_counts[3], type_counts[4], type_counts[5]);

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
            tau_mem: 8.0,
            sensitivity: 1.5,
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

    fn process_frame(&mut self, positions: &[f32]) -> Result<usize> {
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
                .arg(&0.05f32) // Reduced inhibition for more sensitivity
                .launch(cfg_3d)?;
        }

        let counts = self.stream.clone_dtoh(&self.spike_count)?;
        Ok(counts[0] as usize)
    }

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
                .arg(&8.0f32) // Closer proximity required
                .launch(cfg_1d)?;
        }

        let residues = self.stream.clone_dtoh(&spike_residues_gpu)?;
        Ok((positions, residues[..n_spikes].to_vec()))
    }
}

#[derive(Debug, Clone)]
struct SpikeRecord {
    frame_idx: usize,
    position: [f32; 3],
    residue: i32,
    temperature: f32,
    wavelength_nm: Option<f32>,
}

// ============================================================================
// TIER 2 OUTPUT: Temporal dynamics and residue attribution
// ============================================================================

/// Per-frame spike data for a site - enables temporal dynamics analysis
#[derive(Debug, Clone, serde::Serialize)]
struct FrameTimeseries {
    frame: usize,
    spikes: usize,
    temperature: f32,
    wavelengths: HashMap<String, usize>,
}

/// Per-residue contribution to a site - enables VASIL cross-reference
#[derive(Debug, Clone, serde::Serialize)]
struct ResidueContribution {
    residue_id: i32,
    frames_active: Vec<usize>,
    total_spikes: usize,
    spikes_per_wavelength: HashMap<String, usize>,
}

/// Tier 2 detailed output for mechanistic analysis
#[derive(Debug, Clone, serde::Serialize)]
struct Tier2Output {
    /// Frame-by-frame spike timeseries for this site
    frame_timeseries: Vec<FrameTimeseries>,
    /// Per-residue contribution breakdown (enables VASIL cross-reference)
    residue_contributions: HashMap<String, ResidueContribution>,
}

// ============================================================================
// COMPREHENSIVE REPORT STRUCTURES
// ============================================================================

/// Confidence improvement tracking for edge case analysis
#[derive(Debug, Clone, serde::Serialize)]
struct ConfidenceImprovement {
    site_id: usize,
    residues: Vec<i32>,
    original_confidence: f32,
    enhanced_confidence: f32,
    promoted_to_high: bool,
    boost_breakdown: BoostBreakdown,
}

#[derive(Debug, Clone, serde::Serialize)]
struct BoostBreakdown {
    chromophore_boost: f32,
    burst_boost: f32,
    selectivity_boost: f32,
    total_boost: f32,
}

/// Edge case analysis report (Report #5)
#[derive(Debug, Clone, serde::Serialize)]
struct EdgeCaseReport {
    phase2_triggered: bool,
    trigger_reason: String,
    sites_evaluated: usize,
    sites_enhanced: usize,
    confidence_improvements: Vec<ConfidenceImprovement>,
    burst_analysis: BurstAnalysis,
}

#[derive(Debug, Clone, serde::Serialize)]
struct BurstAnalysis {
    burst_events_detected: usize,
    max_burst_intensity: usize,
    burst_wavelength_distribution: HashMap<String, usize>,
    sites_with_burst_enhancement: Vec<usize>,
}

/// Chromophore selectivity report (Report #6)
#[derive(Debug, Clone, serde::Serialize)]
struct ChromophoreSelectivityReport {
    dominant_chromophores: ChromophoreCounts,
    selectivity_scores: SelectivityDistribution,
    chromophore_weighting_applied: bool,
    weighting_impact: WeightingImpact,
}

#[derive(Debug, Clone, serde::Serialize)]
struct ChromophoreCounts {
    trp_280nm_sites: usize,
    ss_250nm_sites: usize,
    phe_258nm_sites: usize,
    tyr_274nm_sites: usize,
    thermal_290nm_sites: usize,
    other_sites: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
struct SelectivityDistribution {
    high_selectivity: usize,   // entropy < 0.5
    medium_selectivity: usize, // entropy 0.5-0.8
    low_selectivity: usize,    // entropy > 0.8
}

#[derive(Debug, Clone, serde::Serialize)]
struct WeightingImpact {
    sites_promoted: usize,
    sites_in_promotion_zone: usize,
    average_boost: f32,
}

/// Validation readiness report (Report #8)
#[derive(Debug, Clone, serde::Serialize)]
struct ValidationReadinessReport {
    high_confidence_residues: Vec<i32>,
    top_residue_profiles: Vec<ResidueProfile>,
    benchmark_ready_features: BenchmarkFeatures,
}

#[derive(Debug, Clone, serde::Serialize)]
struct ResidueProfile {
    residue_id: i32,
    frames_active: usize,
    total_spikes: usize,
    dominant_wavelength: String,
    sites_contributing_to: Vec<usize>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct BenchmarkFeatures {
    cryptic_adjacency_scores: bool,
    wavelength_entropy_values: bool,
    burst_intensity_metrics: bool,
    tier2_residue_contributions: bool,
}

/// Performance & quality control report (Report #9)
#[derive(Debug, Clone, serde::Serialize)]
struct PerformanceQCReport {
    detection_quality: DetectionQuality,
    computational_efficiency: ComputationalEfficiency,
}

#[derive(Debug, Clone, serde::Serialize)]
struct DetectionQuality {
    thermal_noise_ratio: f32,            // 290nm / total
    aromatic_signal_ratio: f32,          // (250+258+274+280) / total
    high_entropy_high_confidence: usize, // Should be rare - potential FP
    single_residue_sites: usize,         // Potentially suspicious
    low_spike_high_confidence: usize,    // Potentially suspicious
}

#[derive(Debug, Clone, serde::Serialize)]
struct ComputationalEfficiency {
    total_time_ms: u64,
    frames_per_second: f32,
    sites_per_second: f32,
}

/// Cross-target compatibility report (Report #10)
#[derive(Debug, Clone, serde::Serialize)]
struct CrossTargetReport {
    chromophore_inventory: ChromophoreInventory,
    detection_parameters: DetectionParameters,
    generalization_notes: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct ChromophoreInventory {
    trp_residues_total: usize,
    tyr_residues_total: usize,
    phe_residues_total: usize,
    cys_residues_total: usize, // Potential disulfides
    total_aromatic_residues: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
struct DetectionParameters {
    burst_threshold_spikes: usize,
    chromophore_weighting_enabled: bool,
    edge_case_detection_enabled: bool,
    wavelength_channels: Vec<String>,
}

/// Master comprehensive report containing all sub-reports
#[derive(Debug, Clone, serde::Serialize)]
struct ComprehensiveReport {
    // Metadata
    pdb_id: String,
    analysis_timestamp: String,
    prism_version: String,

    // Core results
    summary: AnalysisSummary,

    // Sub-reports
    edge_case_analysis: EdgeCaseReport,
    chromophore_selectivity: ChromophoreSelectivityReport,
    validation_readiness: ValidationReadinessReport,
    performance_qc: PerformanceQCReport,
    cross_target: CrossTargetReport,
}

#[derive(Debug, Clone, serde::Serialize)]
struct AnalysisSummary {
    frames_analyzed: usize,
    elapsed_seconds: f64,
    frames_per_second: f64,
    total_spikes: usize,
    sites_found: usize,
    high_confidence: usize,
    medium_confidence: usize,
    low_confidence: usize,
    grid_dim: usize,
    grid_spacing: f32,
    tau_mem: f32,
    sensitivity: f32,
}

#[derive(Debug, Clone, serde::Serialize)]
struct CrypticSite {
    id: usize,
    centroid: [f32; 3],
    residues: Vec<i32>,
    spike_count: usize,

    // Multi-factor confidence
    frequency_score: f32,
    persistence_score: f32,
    coherence_score: f32,
    thermal_stability: f32,
    overall_confidence: f32,

    // Wavelength-aware factors (spectroscopy mode)
    /// Spikes correlate with chromophore absorption profiles (0-1)
    chromophore_specificity: f32,
    /// Site shows activity at multiple wavelengths (0-1)
    wavelength_diversity: f32,
    /// Distinct wavelengths that contributed to this site
    wavelengths_observed: Vec<f32>,
    /// Spike counts per wavelength channel (the key fix!)
    spikes_per_wavelength: HashMap<String, usize>,
    /// Wavelength with most spikes (indicates chromophore type)
    dominant_wavelength: Option<f32>,
    /// Entropy of wavelength distribution (low=selective, high=uniform)
    wavelength_entropy: f32,

    category: String,
    first_frame: usize,
    last_frame: usize,
    volume_estimate: f32,

    // Quality metrics
    avg_distance_to_protein: f32,
    unique_frames: usize,

    /// Maximum spikes detected in any single frame (for burst detection)
    max_single_frame_spikes: usize,
    /// Frame index where max spike burst occurred
    peak_frame: usize,

    /// Tier 2 output: temporal dynamics and residue attribution
    /// Enables VASIL cross-reference without separate query system
    #[serde(skip_serializing_if = "Option::is_none")]
    tier2: Option<Tier2Output>,
}

/// Advanced clustering with multi-factor scoring
/// Cryo-aware: only counts persistence during cold phase (T < 250K)
fn cluster_spikes_pro(
    spikes: &[SpikeRecord],
    radius: f32,
    total_frames: usize,
    min_spikes: usize,
) -> Vec<CrypticSite> {
    if spikes.is_empty() {
        return Vec::new();
    }

    // Count cryo-phase frames (T < 250K) for proper persistence normalization
    let cold_threshold = 250.0;
    let cold_frame_set: HashSet<usize> = spikes
        .iter()
        .filter(|s| s.temperature < cold_threshold)
        .map(|s| s.frame_idx)
        .collect();
    let cold_frames = cold_frame_set.len().max(1);

    let mut clusters: Vec<CrypticSite> = Vec::new();
    let mut cluster_frames: Vec<HashSet<usize>> = Vec::new();
    let mut cluster_cold_frames: Vec<HashSet<usize>> = Vec::new();
    let mut cluster_temps: Vec<Vec<f32>> = Vec::new();
    let mut cluster_positions: Vec<Vec<[f32; 3]>> = Vec::new();
    let mut cluster_wavelength_counts: Vec<HashMap<String, usize>> = Vec::new(); // Track spike counts per wavelength

    // TIER 2: Per-frame and per-residue tracking for mechanistic analysis
    // Per-frame: (spike_count, temperature, wavelength_counts)
    let mut cluster_frame_data: Vec<HashMap<usize, (usize, f32, HashMap<String, usize>)>> =
        Vec::new();
    // Per-residue: (total_spikes, frames_active, wavelength_counts)
    let mut cluster_residue_data: Vec<HashMap<i32, (usize, Vec<usize>, HashMap<String, usize>)>> =
        Vec::new();

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

        let is_cold = spike.temperature < cold_threshold;

        if closest_d < r2 {
            if let Some(i) = closest {
                let c = &mut clusters[i];
                let n = c.spike_count as f32;
                c.centroid[0] = (c.centroid[0] * n + spike.position[0]) / (n + 1.0);
                c.centroid[1] = (c.centroid[1] * n + spike.position[1]) / (n + 1.0);
                c.centroid[2] = (c.centroid[2] * n + spike.position[2]) / (n + 1.0);
                c.spike_count += 1;
                c.last_frame = spike.frame_idx;
                cluster_frames[i].insert(spike.frame_idx);
                if is_cold {
                    cluster_cold_frames[i].insert(spike.frame_idx);
                }
                cluster_temps[i].push(spike.temperature);
                cluster_positions[i].push(spike.position);
                if spike.residue >= 0 && !c.residues.contains(&spike.residue) {
                    c.residues.push(spike.residue);
                }

                // Get wavelength key for tracking
                let wkey = spike.wavelength_nm.map(|w| format!("{:.0}", w));

                // Track wavelengths for spectroscopy mode - now with COUNTS
                if let Some(w) = spike.wavelength_nm {
                    // Track unique wavelengths (for backwards compat)
                    if !c
                        .wavelengths_observed
                        .iter()
                        .any(|&existing| (existing - w).abs() < 1.0)
                    {
                        c.wavelengths_observed.push(w);
                    }
                    // INCREMENT spike count for this wavelength
                    let wk = wkey.as_ref().unwrap();
                    *cluster_wavelength_counts[i].entry(wk.clone()).or_insert(0) += 1;
                }

                // TIER 2: Per-frame tracking
                let frame_entry = cluster_frame_data[i]
                    .entry(spike.frame_idx)
                    .or_insert_with(|| (0, spike.temperature, HashMap::new()));
                frame_entry.0 += 1; // increment spike count
                if let Some(wk) = &wkey {
                    *frame_entry.2.entry(wk.clone()).or_insert(0) += 1;
                }

                // TIER 2: Per-residue tracking
                if spike.residue >= 0 {
                    let res_entry = cluster_residue_data[i]
                        .entry(spike.residue)
                        .or_insert_with(|| (0, Vec::new(), HashMap::new()));
                    res_entry.0 += 1; // increment total spikes
                    if !res_entry.1.contains(&spike.frame_idx) {
                        res_entry.1.push(spike.frame_idx); // add to active frames
                    }
                    if let Some(wk) = &wkey {
                        *res_entry.2.entry(wk.clone()).or_insert(0) += 1;
                    }
                }
            }
        } else {
            let mut frames = HashSet::new();
            frames.insert(spike.frame_idx);
            let mut cold_set = HashSet::new();
            if is_cold {
                cold_set.insert(spike.frame_idx);
            }
            // Initialize wavelength counts for new cluster
            let mut init_wavelength_counts = HashMap::new();
            if let Some(w) = spike.wavelength_nm {
                let wkey = format!("{:.0}", w);
                init_wavelength_counts.insert(wkey, 1);
            }

            clusters.push(CrypticSite {
                id: clusters.len(),
                centroid: spike.position,
                residues: if spike.residue >= 0 {
                    vec![spike.residue]
                } else {
                    vec![]
                },
                spike_count: 1,
                frequency_score: 0.0,
                persistence_score: 0.0,
                coherence_score: 0.0,
                thermal_stability: 0.0,
                overall_confidence: 0.0,
                chromophore_specificity: 0.0,
                wavelength_diversity: 0.0,
                wavelengths_observed: spike.wavelength_nm.map(|w| vec![w]).unwrap_or_default(),
                spikes_per_wavelength: HashMap::new(), // Will be populated later
                dominant_wavelength: None,
                wavelength_entropy: 0.0,
                category: String::new(),
                first_frame: spike.frame_idx,
                last_frame: spike.frame_idx,
                volume_estimate: 0.0,
                avg_distance_to_protein: 0.0,
                unique_frames: 0,
                max_single_frame_spikes: 0, // Will be calculated from frame_data
                peak_frame: spike.frame_idx,
                tier2: None, // Will be populated after clustering
            });
            cluster_frames.push(frames);
            cluster_cold_frames.push(cold_set);
            cluster_temps.push(vec![spike.temperature]);
            cluster_positions.push(vec![spike.position]);
            cluster_wavelength_counts.push(init_wavelength_counts.clone());

            // TIER 2: Initialize per-frame and per-residue tracking for new cluster
            let mut init_frame_data = HashMap::new();
            init_frame_data.insert(
                spike.frame_idx,
                (1, spike.temperature, init_wavelength_counts),
            );

            let mut init_residue_data: HashMap<i32, (usize, Vec<usize>, HashMap<String, usize>)> =
                HashMap::new();
            if spike.residue >= 0 {
                let mut res_wavelengths = HashMap::new();
                if let Some(w) = spike.wavelength_nm {
                    let wkey = format!("{:.0}", w);
                    res_wavelengths.insert(wkey, 1);
                }
                init_residue_data
                    .insert(spike.residue, (1, vec![spike.frame_idx], res_wavelengths));
            }

            cluster_frame_data.push(init_frame_data);
            cluster_residue_data.push(init_residue_data);
        }
    }

    // Filter by minimum spikes - include Tier 2 tracking data
    type ClusterData = (
        CrypticSite,
        HashSet<usize>,                                            // frames
        HashSet<usize>,                                            // cold_frames
        Vec<f32>,                                                  // temps
        Vec<[f32; 3]>,                                             // positions
        HashMap<String, usize>,                                    // wavelength_counts
        HashMap<usize, (usize, f32, HashMap<String, usize>)>,      // TIER 2: frame_data
        HashMap<i32, (usize, Vec<usize>, HashMap<String, usize>)>, // TIER 2: residue_data
    );

    let mut filtered: Vec<ClusterData> = clusters
        .into_iter()
        .zip(cluster_frames.into_iter())
        .zip(cluster_cold_frames.into_iter())
        .zip(cluster_temps.into_iter())
        .zip(cluster_positions.into_iter())
        .zip(cluster_wavelength_counts.into_iter())
        .zip(cluster_frame_data.into_iter())
        .zip(cluster_residue_data.into_iter())
        .filter(|(((((((c, _), _), _), _), _), _), _)| c.spike_count >= min_spikes)
        .map(|(((((((c, f), cf), t), p), wc), fd), rd)| (c, f, cf, t, p, wc, fd, rd))
        .collect();

    if filtered.is_empty() {
        return Vec::new();
    }

    let max_spikes = filtered
        .iter()
        .map(|(c, _, _, _, _, _, _, _)| c.spike_count)
        .max()
        .unwrap_or(1) as f32;

    for (
        c,
        frames,
        cold_frames_set,
        temps,
        positions,
        wavelength_counts,
        frame_data,
        residue_data,
    ) in &mut filtered
    {
        c.unique_frames = frames.len();

        // 1. Frequency score (normalized spike count)
        c.frequency_score = (c.spike_count as f32 / max_spikes).min(1.0);

        // 2. Persistence score - BURST-AWARE: high-intensity single-frame events
        //    compensate for low temporal spread (fixes residue 469-type cases)
        let frame_persistence = cold_frames_set.len() as f32 / cold_frames.max(1) as f32;

        // Find max single-frame spike intensity from Tier 2 frame data
        let (max_frame_spikes, peak_frame_idx) = frame_data
            .iter()
            .max_by_key(|(_, (spikes, _, _))| *spikes)
            .map(|(&frame, (spikes, _, _))| (*spikes, frame))
            .unwrap_or((0, 0));

        c.max_single_frame_spikes = max_frame_spikes;
        c.peak_frame = peak_frame_idx;

        // Burst significance: 200+ spikes in one frame is notable (P95 threshold)
        // This allows high-intensity transient events (like TRP exposure) to score well
        // Below 200: partial credit; 200+: full burst credit
        let burst_significance = if max_frame_spikes >= 200 {
            1.0 // Top 5% - full burst credit
        } else if max_frame_spikes >= 150 {
            0.75 // Top 25% - partial credit
        } else if max_frame_spikes >= 100 {
            0.5 // Above median - minor credit
        } else {
            0.0 // Below median - no burst credit
        };

        // Persistence = max of temporal spread OR burst intensity (scaled to 0.75)
        // The 0.75 factor prevents pure bursts from dominating sustained signals
        c.persistence_score = frame_persistence.max(burst_significance * 0.75).min(1.0);

        // 3. Spatial coherence (how tightly clustered are the spikes?)
        if positions.len() > 1 {
            let mut total_dist = 0.0;
            for pos in positions.iter() {
                let dx = pos[0] - c.centroid[0];
                let dy = pos[1] - c.centroid[1];
                let dz = pos[2] - c.centroid[2];
                total_dist += (dx * dx + dy * dy + dz * dz).sqrt();
            }
            let avg_dist = total_dist / positions.len() as f32;
            // Tighter clusters get higher scores (inverse of spread)
            c.coherence_score = (1.0 / (1.0 + avg_dist / 2.0)).min(1.0);
        } else {
            c.coherence_score = 0.5;
        }

        // 4. Thermal stability (consistent across temperature range)
        if temps.len() > 1 {
            let mean_temp: f32 = temps.iter().sum::<f32>() / temps.len() as f32;
            let variance: f32 =
                temps.iter().map(|t| (t - mean_temp).powi(2)).sum::<f32>() / temps.len() as f32;
            // Lower variance = more stable = higher score
            c.thermal_stability = (1.0 / (1.0 + variance / 1000.0)).min(1.0);
        } else {
            c.thermal_stability = 0.5;
        }

        // 5. Volume estimate (based on spread of positions)
        if positions.len() > 3 {
            let mut max_dist = 0.0f32;
            for (i, p1) in positions.iter().enumerate() {
                for p2 in positions.iter().skip(i + 1) {
                    let d = ((p1[0] - p2[0]).powi(2)
                        + (p1[1] - p2[1]).powi(2)
                        + (p1[2] - p2[2]).powi(2))
                    .sqrt();
                    max_dist = max_dist.max(d);
                }
            }
            // Approximate as sphere
            c.volume_estimate = (4.0 / 3.0) * std::f32::consts::PI * (max_dist / 2.0).powi(3);
        }

        // 6. Wavelength-aware factors (spectroscopy mode only) - NOW WITH SPIKE COUNTS
        let has_spectroscopy_data = !wavelength_counts.is_empty();
        if has_spectroscopy_data {
            // Copy wavelength counts to the site
            c.spikes_per_wavelength = wavelength_counts.clone();

            // Find dominant wavelength (one with most spikes)
            if let Some((dominant_wl, _count)) =
                wavelength_counts.iter().max_by_key(|(_, &count)| count)
            {
                c.dominant_wavelength = dominant_wl.parse::<f32>().ok();
            }

            // Calculate wavelength entropy (Shannon entropy)
            // Low entropy = selective (spikes concentrated in few channels)
            // High entropy = uniform (spikes spread across all channels)
            let total: usize = wavelength_counts.values().sum();
            if total > 0 {
                let mut entropy = 0.0f32;
                for &count in wavelength_counts.values() {
                    if count > 0 {
                        let p = count as f32 / total as f32;
                        entropy -= p * p.ln();
                    }
                }
                // Normalize by max entropy (ln(6) for 6 wavelengths)
                let max_entropy = (6.0f32).ln();
                c.wavelength_entropy = entropy / max_entropy;
            }

            // Wavelength diversity: number of channels observed
            let n_wavelengths = wavelength_counts.len() as f32;
            c.wavelength_diversity = (n_wavelengths / 6.0).min(1.0);

            // IMPROVED Chromophore specificity:
            // High score if dominant wavelength matches a chromophore AND entropy is low
            // Chromophore wavelengths: TRP=280nm, TYR=274nm, PHE=258nm, S-S=250nm
            let chromophore_wavelengths = ["280", "274", "258", "250"];

            // Check if dominant wavelength is a chromophore channel
            let dominant_is_chromophore = c
                .dominant_wavelength
                .map(|w| {
                    chromophore_wavelengths
                        .iter()
                        .any(|&cw| (w - cw.parse::<f32>().unwrap_or(0.0)).abs() < 5.0)
                })
                .unwrap_or(false);

            // Calculate selectivity: fraction of spikes in dominant channel
            let dominant_fraction = if let Some(dom_wl) = &c.dominant_wavelength {
                let dom_key = format!("{:.0}", dom_wl);
                wavelength_counts.get(&dom_key).copied().unwrap_or(0) as f32 / total.max(1) as f32
            } else {
                0.0
            };

            // Specificity = chromophore match * selectivity * (1 - entropy)
            // This rewards: correct wavelength + concentrated spikes + low entropy
            c.chromophore_specificity = if dominant_is_chromophore {
                (dominant_fraction * (1.0 - c.wavelength_entropy * 0.5)).clamp(0.0, 1.0)
            } else {
                // Non-chromophore dominant wavelength gets lower score
                (dominant_fraction * 0.3 * (1.0 - c.wavelength_entropy)).clamp(0.0, 1.0)
            };
        }

        // Overall confidence: weighted combination
        // With spectroscopy: add wavelength-aware factors (rebalance weights)
        c.overall_confidence = if has_spectroscopy_data {
            // Enhanced scoring with spectroscopy data
            (c.frequency_score * 0.25
                + c.persistence_score * 0.25
                + c.coherence_score * 0.20
                + c.thermal_stability * 0.10
                + c.chromophore_specificity * 0.10
                + c.wavelength_diversity * 0.10)
                .clamp(0.0, 1.0)
        } else {
            // Standard scoring without spectroscopy
            (c.frequency_score * 0.30
                + c.persistence_score * 0.30
                + c.coherence_score * 0.25
                + c.thermal_stability * 0.15)
                .clamp(0.0, 1.0)
        };

        c.category = if c.overall_confidence >= 0.70 {
            "HIGH".into()
        } else if c.overall_confidence >= 0.50 {
            "MEDIUM".into()
        } else {
            "LOW".into()
        };

        // TIER 2: Build detailed output for mechanistic analysis
        // Convert frame_data HashMap to sorted FrameTimeseries Vec
        let mut frame_timeseries: Vec<FrameTimeseries> = frame_data
            .iter()
            .map(|(&frame, (spikes, temp, wl_counts))| FrameTimeseries {
                frame,
                spikes: *spikes,
                temperature: *temp,
                wavelengths: wl_counts.clone(),
            })
            .collect();
        frame_timeseries.sort_by_key(|ft| ft.frame);

        // Convert residue_data HashMap to ResidueContribution map
        let residue_contributions: HashMap<String, ResidueContribution> = residue_data
            .iter()
            .map(|(&res_id, (total_spikes, frames_active, wl_counts))| {
                let mut sorted_frames = frames_active.clone();
                sorted_frames.sort();
                (
                    res_id.to_string(),
                    ResidueContribution {
                        residue_id: res_id,
                        frames_active: sorted_frames,
                        total_spikes: *total_spikes,
                        spikes_per_wavelength: wl_counts.clone(),
                    },
                )
            })
            .collect();

        c.tier2 = Some(Tier2Output {
            frame_timeseries,
            residue_contributions,
        });
    }

    let mut result: Vec<CrypticSite> = filtered
        .into_iter()
        .map(|(c, _, _, _, _, _, _, _)| c)
        .collect();

    // =========================================================================
    // EDGE CASE DETECTION & CHROMOPHORE-WEIGHTED RE-SCORING
    // =========================================================================
    // Check if edge case enhancement is warranted based on detection patterns
    let (edge_case_triggered, trigger_reason) = check_edge_case_triggers(&result);

    let improvements = if edge_case_triggered {
        eprintln!(
            "  [Edge Case] Chromophore-weighted re-scoring triggered: {}",
            trigger_reason
        );
        apply_chromophore_weighted_rescoring(&mut result)
    } else {
        Vec::new()
    };

    result.sort_by(|a, b| {
        b.overall_confidence
            .partial_cmp(&a.overall_confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (i, c) in result.iter_mut().enumerate() {
        c.id = i;
    }

    // Build and store edge case report (accessible via thread-local or return value)
    let edge_case_report =
        build_edge_case_report(&result, edge_case_triggered, &trigger_reason, improvements);

    // Store in thread-local for access by report generator
    EDGE_CASE_REPORT.with(|r| *r.borrow_mut() = Some(edge_case_report));

    result
}

// Thread-local storage for edge case report (accessible from report generator)
thread_local! {
    static EDGE_CASE_REPORT: std::cell::RefCell<Option<EdgeCaseReport>> = std::cell::RefCell::new(None);
}

/// Check edge case triggers and return (triggered, reason)
fn check_edge_case_triggers(sites: &[CrypticSite]) -> (bool, String) {
    // Condition 1: High-intensity aromatic bursts near threshold
    let high_intensity_aromatics = sites
        .iter()
        .filter(|s| {
            let is_aromatic = s
                .dominant_wavelength
                .map(|w| (w - 280.0).abs() < 5.0 || (w - 274.0).abs() < 5.0)
                .unwrap_or(false);
            let is_burst = s.max_single_frame_spikes >= 150;
            let is_near_threshold = s.overall_confidence >= 0.65 && s.overall_confidence < 0.72;
            is_aromatic && is_burst && is_near_threshold
        })
        .count();

    // Condition 2: Wavelength-selective sites that missed HIGH threshold
    let selective_low_confidence = sites
        .iter()
        .filter(|s| {
            s.wavelength_entropy < 0.7
                && s.overall_confidence >= 0.65
                && s.overall_confidence < 0.70
        })
        .count();

    if high_intensity_aromatics >= 1 {
        (
            true,
            format!("high_intensity_aromatics ({})", high_intensity_aromatics),
        )
    } else if selective_low_confidence >= 2 {
        (
            true,
            format!("selective_low_confidence ({})", selective_low_confidence),
        )
    } else {
        (false, "none".to_string())
    }
}

/// Build EdgeCaseReport from detection results
fn build_edge_case_report(
    sites: &[CrypticSite],
    triggered: bool,
    reason: &str,
    improvements: Vec<ConfidenceImprovement>,
) -> EdgeCaseReport {
    // Build burst analysis
    let mut burst_wavelength_dist: HashMap<String, usize> = HashMap::new();
    let mut sites_with_burst: Vec<usize> = Vec::new();
    let mut max_burst = 0usize;

    for site in sites {
        if site.max_single_frame_spikes >= 150 {
            sites_with_burst.push(site.id);
            max_burst = max_burst.max(site.max_single_frame_spikes);
            if let Some(w) = site.dominant_wavelength {
                let key = format!("{}nm", w as i32);
                *burst_wavelength_dist.entry(key).or_insert(0) += 1;
            }
        }
    }

    EdgeCaseReport {
        phase2_triggered: triggered,
        trigger_reason: reason.to_string(),
        sites_evaluated: sites
            .iter()
            .filter(|s| s.overall_confidence >= 0.65 && s.overall_confidence < 0.72)
            .count(),
        sites_enhanced: improvements.len(),
        confidence_improvements: improvements,
        burst_analysis: BurstAnalysis {
            burst_events_detected: sites_with_burst.len(),
            max_burst_intensity: max_burst,
            burst_wavelength_distribution: burst_wavelength_dist,
            sites_with_burst_enhancement: sites_with_burst,
        },
    }
}

/// Apply chromophore-weighted re-scoring to boost biologically significant sites
/// Weights: TRP(280) > S-S(250) > TYR(274) > PHE(258) > non-specific
/// Returns tracking data for the EdgeCaseReport
fn apply_chromophore_weighted_rescoring(sites: &mut [CrypticSite]) -> Vec<ConfidenceImprovement> {
    let mut improvements = Vec::new();

    for site in sites.iter_mut() {
        // Only re-score sites in the "promotion zone" (0.65-0.72)
        if site.overall_confidence < 0.65 || site.overall_confidence >= 0.72 {
            continue;
        }

        // Calculate chromophore weight based on dominant wavelength
        let chromophore_boost: f32 = match site.dominant_wavelength {
            Some(w) if (w - 280.0).abs() < 5.0 => 0.04, // TRP: critical for pocket detection
            Some(w) if (w - 250.0).abs() < 5.0 => 0.03, // S-S: disulfide exposure is diagnostic
            Some(w) if (w - 274.0).abs() < 5.0 => 0.025, // TYR: moderate importance
            Some(w) if (w - 258.0).abs() < 5.0 => 0.02, // PHE: lower importance
            _ => 0.0,                                   // Non-specific: no boost
        };

        // Additional boost for high-intensity bursts (transient exposure events)
        let burst_boost: f32 = if site.max_single_frame_spikes >= 200 {
            0.03 // Top 5% burst
        } else if site.max_single_frame_spikes >= 150 {
            0.02 // Top 25% burst
        } else if site.max_single_frame_spikes >= 100 {
            0.01 // Above median burst
        } else {
            0.0
        };

        // Additional boost for wavelength-selective sites (low entropy)
        let selectivity_boost: f32 = if site.wavelength_entropy < 0.65 {
            0.02
        } else if site.wavelength_entropy < 0.75 {
            0.01
        } else {
            0.0
        };

        // Apply combined boost (capped to prevent over-promotion)
        let total_boost: f32 = (chromophore_boost + burst_boost + selectivity_boost).min(0.08);
        let old_confidence = site.overall_confidence;
        site.overall_confidence = (site.overall_confidence + total_boost).min(0.85);
        let promoted = site.overall_confidence >= 0.70 && old_confidence < 0.70;

        // Update category if promoted
        if promoted {
            site.category = "HIGH".to_string();
            eprintln!(
                "    [Promoted] Site {} ({:?}nm): {:.2} → {:.2} [{}]",
                site.id,
                site.dominant_wavelength,
                old_confidence,
                site.overall_confidence,
                site.category
            );
        }

        // Track improvement
        if total_boost > 0.0 {
            improvements.push(ConfidenceImprovement {
                site_id: site.id,
                residues: site.residues.clone(),
                original_confidence: old_confidence,
                enhanced_confidence: site.overall_confidence,
                promoted_to_high: promoted,
                boost_breakdown: BoostBreakdown {
                    chromophore_boost,
                    burst_boost,
                    selectivity_boost,
                    total_boost,
                },
            });
        }
    }

    improvements
}

/// Generate comprehensive report containing all sub-reports
fn generate_comprehensive_report(
    sites: &[CrypticSite],
    pdb_id: &str,
    total_spikes: usize,
    frames_analyzed: usize,
    elapsed_seconds: f64,
    topology: &PrismPrepTopology,
    grid_dim: usize,
    grid_spacing: f32,
    tau_mem: f32,
    sensitivity: f32,
) -> ComprehensiveReport {
    // Get edge case report from thread-local
    let edge_case_report = EDGE_CASE_REPORT.with(|r| {
        r.borrow().clone().unwrap_or_else(|| EdgeCaseReport {
            phase2_triggered: false,
            trigger_reason: "none".to_string(),
            sites_evaluated: 0,
            sites_enhanced: 0,
            confidence_improvements: Vec::new(),
            burst_analysis: BurstAnalysis {
                burst_events_detected: 0,
                max_burst_intensity: 0,
                burst_wavelength_distribution: HashMap::new(),
                sites_with_burst_enhancement: Vec::new(),
            },
        })
    });

    // Build chromophore selectivity report
    let chromophore_selectivity = build_chromophore_selectivity_report(sites, &edge_case_report);

    // Build validation readiness report
    let validation_readiness = build_validation_readiness_report(sites);

    // Build performance QC report
    let performance_qc =
        build_performance_qc_report(sites, total_spikes, frames_analyzed, elapsed_seconds);

    // Build cross-target report
    let cross_target = build_cross_target_report(topology, grid_dim, grid_spacing);

    // Build analysis summary
    let high_count = sites.iter().filter(|s| s.category == "HIGH").count();
    let medium_count = sites.iter().filter(|s| s.category == "MEDIUM").count();
    let low_count = sites.iter().filter(|s| s.category == "LOW").count();

    ComprehensiveReport {
        pdb_id: pdb_id.to_string(),
        analysis_timestamp: chrono::Utc::now().to_rfc3339(),
        prism_version: env!("CARGO_PKG_VERSION").to_string(),

        summary: AnalysisSummary {
            frames_analyzed,
            elapsed_seconds,
            frames_per_second: frames_analyzed as f64 / elapsed_seconds,
            total_spikes,
            sites_found: sites.len(),
            high_confidence: high_count,
            medium_confidence: medium_count,
            low_confidence: low_count,
            grid_dim,
            grid_spacing,
            tau_mem,
            sensitivity,
        },

        edge_case_analysis: edge_case_report,
        chromophore_selectivity,
        validation_readiness,
        performance_qc,
        cross_target,
    }
}

fn build_chromophore_selectivity_report(
    sites: &[CrypticSite],
    edge_report: &EdgeCaseReport,
) -> ChromophoreSelectivityReport {
    let mut counts = ChromophoreCounts {
        trp_280nm_sites: 0,
        ss_250nm_sites: 0,
        phe_258nm_sites: 0,
        tyr_274nm_sites: 0,
        thermal_290nm_sites: 0,
        other_sites: 0,
    };

    for site in sites {
        match site.dominant_wavelength {
            Some(w) if (w - 280.0).abs() < 5.0 => counts.trp_280nm_sites += 1,
            Some(w) if (w - 250.0).abs() < 5.0 => counts.ss_250nm_sites += 1,
            Some(w) if (w - 258.0).abs() < 5.0 => counts.phe_258nm_sites += 1,
            Some(w) if (w - 274.0).abs() < 5.0 => counts.tyr_274nm_sites += 1,
            Some(w) if (w - 290.0).abs() < 5.0 => counts.thermal_290nm_sites += 1,
            _ => counts.other_sites += 1,
        }
    }

    let selectivity = SelectivityDistribution {
        high_selectivity: sites.iter().filter(|s| s.wavelength_entropy < 0.5).count(),
        medium_selectivity: sites
            .iter()
            .filter(|s| s.wavelength_entropy >= 0.5 && s.wavelength_entropy < 0.8)
            .count(),
        low_selectivity: sites.iter().filter(|s| s.wavelength_entropy >= 0.8).count(),
    };

    let promoted_count = edge_report
        .confidence_improvements
        .iter()
        .filter(|c| c.promoted_to_high)
        .count();
    let avg_boost = if edge_report.confidence_improvements.is_empty() {
        0.0
    } else {
        edge_report
            .confidence_improvements
            .iter()
            .map(|c| c.boost_breakdown.total_boost)
            .sum::<f32>()
            / edge_report.confidence_improvements.len() as f32
    };

    ChromophoreSelectivityReport {
        dominant_chromophores: counts,
        selectivity_scores: selectivity,
        chromophore_weighting_applied: edge_report.phase2_triggered,
        weighting_impact: WeightingImpact {
            sites_promoted: promoted_count,
            sites_in_promotion_zone: edge_report.sites_evaluated,
            average_boost: avg_boost,
        },
    }
}

fn build_validation_readiness_report(sites: &[CrypticSite]) -> ValidationReadinessReport {
    // Collect high-confidence residues from tier2 data
    let mut residue_spikes: HashMap<i32, (usize, usize, String, Vec<usize>)> = HashMap::new(); // (total_spikes, frames, dominant_wl, site_ids)

    for site in sites
        .iter()
        .filter(|s| s.category == "HIGH" || s.category == "MEDIUM")
    {
        if let Some(tier2) = &site.tier2 {
            for (res_str, contrib) in &tier2.residue_contributions {
                if let Ok(res_id) = res_str.parse::<i32>() {
                    let entry = residue_spikes.entry(res_id).or_insert_with(|| {
                        // Find dominant wavelength for this residue
                        let dom_wl = contrib
                            .spikes_per_wavelength
                            .iter()
                            .max_by_key(|(_, &v)| v)
                            .map(|(k, _)| format!("{}nm", k))
                            .unwrap_or_else(|| "mixed".to_string());
                        (0, 0, dom_wl, Vec::new())
                    });
                    entry.0 += contrib.total_spikes;
                    entry.1 += contrib.frames_active.len();
                    entry.3.push(site.id);
                }
            }
        }
    }

    // Sort by total spikes and take top 20
    let mut sorted_residues: Vec<_> = residue_spikes.into_iter().collect();
    sorted_residues.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));

    let top_profiles: Vec<ResidueProfile> = sorted_residues
        .iter()
        .take(20)
        .map(|(res_id, (spikes, frames, wl, site_ids))| ResidueProfile {
            residue_id: *res_id,
            frames_active: *frames,
            total_spikes: *spikes,
            dominant_wavelength: wl.clone(),
            sites_contributing_to: site_ids.clone(),
        })
        .collect();

    let high_conf_residues: Vec<i32> = sorted_residues
        .iter()
        .take(10)
        .map(|(res_id, _)| *res_id)
        .collect();

    ValidationReadinessReport {
        high_confidence_residues: high_conf_residues,
        top_residue_profiles: top_profiles,
        benchmark_ready_features: BenchmarkFeatures {
            cryptic_adjacency_scores: true,
            wavelength_entropy_values: true,
            burst_intensity_metrics: true,
            tier2_residue_contributions: true,
        },
    }
}

fn build_performance_qc_report(
    sites: &[CrypticSite],
    total_spikes: usize,
    frames_analyzed: usize,
    elapsed_seconds: f64,
) -> PerformanceQCReport {
    // Calculate wavelength distribution across all sites
    let mut total_290nm = 0usize;
    let mut total_aromatic = 0usize;
    let mut site_spike_sum = 0usize;

    for site in sites {
        for (wl_str, &count) in &site.spikes_per_wavelength {
            site_spike_sum += count;
            if let Ok(wl) = wl_str.parse::<i32>() {
                if wl == 290 {
                    total_290nm += count;
                } else if wl == 250 || wl == 258 || wl == 274 || wl == 280 {
                    total_aromatic += count;
                }
            }
        }
    }

    let thermal_noise_ratio = if site_spike_sum > 0 {
        total_290nm as f32 / site_spike_sum as f32
    } else {
        0.0
    };

    let aromatic_signal_ratio = if site_spike_sum > 0 {
        total_aromatic as f32 / site_spike_sum as f32
    } else {
        0.0
    };

    // Quality indicators
    let high_entropy_high_conf = sites
        .iter()
        .filter(|s| s.wavelength_entropy > 0.9 && s.overall_confidence >= 0.70)
        .count();

    let single_residue_sites = sites
        .iter()
        .filter(|s| s.residues.len() == 1 && s.category == "HIGH")
        .count();

    let low_spike_high_conf = sites
        .iter()
        .filter(|s| s.spike_count < 100 && s.overall_confidence >= 0.70)
        .count();

    PerformanceQCReport {
        detection_quality: DetectionQuality {
            thermal_noise_ratio,
            aromatic_signal_ratio,
            high_entropy_high_confidence: high_entropy_high_conf,
            single_residue_sites,
            low_spike_high_confidence: low_spike_high_conf,
        },
        computational_efficiency: ComputationalEfficiency {
            total_time_ms: (elapsed_seconds * 1000.0) as u64,
            frames_per_second: frames_analyzed as f32 / elapsed_seconds as f32,
            sites_per_second: sites.len() as f32 / elapsed_seconds as f32,
        },
    }
}

fn build_cross_target_report(
    topology: &PrismPrepTopology,
    grid_dim: usize,
    grid_spacing: f32,
) -> CrossTargetReport {
    // Count chromophore residues from topology
    let mut trp_count = 0;
    let mut tyr_count = 0;
    let mut phe_count = 0;
    let mut cys_count = 0;

    for res_name in &topology.residue_names {
        match res_name.as_str() {
            "TRP" => trp_count += 1,
            "TYR" => tyr_count += 1,
            "PHE" => phe_count += 1,
            "CYS" => cys_count += 1,
            _ => {}
        }
    }

    CrossTargetReport {
        chromophore_inventory: ChromophoreInventory {
            trp_residues_total: trp_count,
            tyr_residues_total: tyr_count,
            phe_residues_total: phe_count,
            cys_residues_total: cys_count,
            total_aromatic_residues: trp_count + tyr_count + phe_count,
        },
        detection_parameters: DetectionParameters {
            burst_threshold_spikes: 200,
            chromophore_weighting_enabled: true,
            edge_case_detection_enabled: true,
            wavelength_channels: vec![
                "250nm".into(),
                "258nm".into(),
                "265nm".into(),
                "274nm".into(),
                "280nm".into(),
                "290nm".into(),
            ],
        },
        generalization_notes: vec![
            format!("Grid resolution: {}³ @ {}Å spacing", grid_dim, grid_spacing),
            format!(
                "Aromatic coverage: {} TRP, {} TYR, {} PHE",
                trp_count, tyr_count, phe_count
            ),
            format!("Potential disulfides: {} CYS residues", cys_count),
        ],
    }
}

#[cfg(feature = "gpu")]
fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║   NHS PRO Analyzer - Publication Quality                       ║");
    println!("║   Proper atom classification + Multi-factor confidence         ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();

    fs::create_dir_all(&args.output)?;

    let topology = PrismPrepTopology::load(&args.topology).context("Failed to load topology")?;
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

    // Load wavelength data from frames JSON if provided (spectroscopy mode)
    let wavelength_map: HashMap<usize, f32> = if let Some(ref json_path) = args.frames_json {
        println!("Loading wavelength data from: {}", json_path.display());
        let json_str = fs::read_to_string(json_path).context("Failed to read frames JSON")?;

        // Parse as array of TrajectoryFrame-like objects
        #[derive(serde::Deserialize)]
        struct FrameWavelength {
            frame_idx: usize,
            wavelength_nm: Option<f32>,
        }

        let frame_data: Vec<FrameWavelength> =
            serde_json::from_str(&json_str).context("Failed to parse frames JSON")?;

        let mut map = HashMap::new();
        let mut wavelengths_found = 0;
        for f in frame_data {
            if let Some(w) = f.wavelength_nm {
                map.insert(f.frame_idx, w);
                wavelengths_found += 1;
            }
        }

        if wavelengths_found > 0 {
            // Collect unique wavelengths
            let unique: HashSet<_> = map.values().map(|w| (*w * 10.0) as i32).collect();
            let unique_wavelengths: Vec<f32> = unique.iter().map(|&w| w as f32 / 10.0).collect();
            println!("  Loaded {} frames with wavelength data", wavelengths_found);
            println!("  Wavelengths: {:?} nm", unique_wavelengths);
            println!("  Wavelength-aware scoring: ENABLED");
        } else {
            println!("  No wavelength data found in frames JSON");
        }
        map
    } else {
        HashMap::new()
    };

    println!();
    println!("Initializing PRO engine...");

    let cuda_context = CudaContext::new(0)?;
    let mut engine = ProEngine::new(
        cuda_context.clone(),
        args.grid_dim,
        topology.n_atoms,
        args.spacing,
        &topology,
    )?;

    let (min_x, min_y, min_z) = compute_bounds(&frames[0].positions);
    let grid_origin = [min_x - 5.0, min_y - 5.0, min_z - 5.0];
    engine.initialize(grid_origin)?;
    engine.set_params(args.tau_mem, args.sensitivity);

    println!(
        "  Grid: {}³ = {} voxels",
        args.grid_dim,
        args.grid_dim.pow(3)
    );
    println!("  Spacing: {:.2}Å", args.spacing);
    println!(
        "  LIF: tau={:.1}, sensitivity={:.1}",
        args.tau_mem, args.sensitivity
    );
    println!();

    println!("════════════════════════════════════════════════════════════════");
    println!("                    PRO PROCESSING");
    println!("════════════════════════════════════════════════════════════════");
    println!();

    let start = Instant::now();
    let frames_to_process: Vec<_> = frames.iter().enumerate().step_by(args.skip).collect();
    let total_frames = frames_to_process.len();

    let mut all_spikes: Vec<SpikeRecord> = Vec::new();
    let mut total_spike_count = 0u64;
    let mut last_report = Instant::now();

    for (idx, (frame_idx, frame)) in frames_to_process.iter().enumerate() {
        let count = engine.process_frame(&frame.positions)?;
        total_spike_count += count as u64;

        if count > 0 {
            let (positions, residues) = engine.extract_spikes()?;
            // Get wavelength from JSON map if available, otherwise from frame (which is usually None for PDB)
            let wavelength = wavelength_map
                .get(frame_idx)
                .copied()
                .or(frame.wavelength_nm);
            for (i, pos) in positions.iter().enumerate() {
                all_spikes.push(SpikeRecord {
                    frame_idx: *frame_idx,
                    position: *pos,
                    residue: if i < residues.len() { residues[i] } else { -1 },
                    temperature: frame.temperature,
                    wavelength_nm: wavelength,
                });
            }
        }

        if last_report.elapsed().as_millis() > 100 || idx + 1 == total_frames {
            let elapsed = start.elapsed().as_secs_f64();
            let fps = (idx + 1) as f64 / elapsed;
            let eta = (total_frames - idx - 1) as f64 / fps.max(1.0);
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

    let elapsed = start.elapsed();
    let fps = total_frames as f64 / elapsed.as_secs_f64();

    // Advanced clustering
    let sites = cluster_spikes_pro(
        &all_spikes,
        args.cluster_radius,
        total_frames,
        args.min_spikes,
    );
    let high = sites.iter().filter(|s| s.category == "HIGH").count();
    let medium = sites.iter().filter(|s| s.category == "MEDIUM").count();

    println!();
    println!("════════════════════════════════════════════════════════════════");
    println!("                       PRO RESULTS");
    println!("════════════════════════════════════════════════════════════════");
    println!();
    println!("Performance:");
    println!("  Frames analyzed:    {}", total_frames);
    println!("  Total time:         {:.2}s", elapsed.as_secs_f64());
    println!("  Frames/second:      {:.0}", fps);
    println!();
    println!("Detection:");
    println!("  Total spikes:       {}", total_spike_count);
    println!(
        "  Avg spikes/frame:   {:.2}",
        total_spike_count as f64 / total_frames as f64
    );
    println!();
    println!(
        "Cryptic Sites: {} (HIGH: {}, MEDIUM: {})",
        sites.len(),
        high,
        medium
    );
    println!();

    for (i, site) in sites.iter().enumerate().take(15) {
        let res_str: String = site
            .residues
            .iter()
            .take(5)
            .map(|r| r.to_string())
            .collect::<Vec<_>>()
            .join(",");
        println!(
            "  Site {}: {} spikes | conf={:.2} [{}]",
            i + 1,
            site.spike_count,
            site.overall_confidence,
            site.category
        );
        println!(
            "          freq={:.2} pers={:.2} coher={:.2} therm={:.2}",
            site.frequency_score,
            site.persistence_score,
            site.coherence_score,
            site.thermal_stability
        );
        println!(
            "          residues: [{}{}]",
            res_str,
            if site.residues.len() > 5 { "..." } else { "" }
        );
    }

    // Save results
    let sites_path = args.output.join("cryptic_sites.json");
    let sites_file = fs::File::create(&sites_path)?;
    serde_json::to_writer_pretty(sites_file, &sites)?;

    // PyMOL script
    let pymol_path = args.output.join("cryptic_sites.pml");
    write_pymol_script(&pymol_path, &sites)?;

    #[derive(serde::Serialize)]
    struct ProSummary {
        frames_analyzed: usize,
        elapsed_seconds: f64,
        frames_per_second: f64,
        total_spikes: u64,
        sites_found: usize,
        high_confidence: usize,
        medium_confidence: usize,
        grid_dim: usize,
        grid_spacing: f32,
        tau_mem: f32,
        sensitivity: f32,
    }

    let summary = ProSummary {
        frames_analyzed: total_frames,
        elapsed_seconds: elapsed.as_secs_f64(),
        frames_per_second: fps,
        total_spikes: total_spike_count,
        sites_found: sites.len(),
        high_confidence: high,
        medium_confidence: medium,
        grid_dim: args.grid_dim,
        grid_spacing: args.spacing,
        tau_mem: args.tau_mem,
        sensitivity: args.sensitivity,
    };

    let summary_path = args.output.join("analysis_summary.json");
    let summary_file = fs::File::create(&summary_path)?;
    serde_json::to_writer_pretty(summary_file, &summary)?;

    // Generate and save comprehensive report
    let pdb_id = args
        .input
        .file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.split('_').next().unwrap_or(s).to_string())
        .unwrap_or_else(|| "unknown".to_string());

    let comprehensive_report = generate_comprehensive_report(
        &sites,
        &pdb_id,
        total_spike_count as usize,
        total_frames,
        elapsed.as_secs_f64(),
        &topology,
        args.grid_dim,
        args.spacing,
        args.tau_mem,
        args.sensitivity,
    );

    let report_path = args.output.join("comprehensive_report.json");
    let report_file = fs::File::create(&report_path)?;
    serde_json::to_writer_pretty(report_file, &comprehensive_report)?;

    println!();
    println!("Results saved to: {}", args.output.display());
    println!("  - cryptic_sites.json");
    println!("  - cryptic_sites.pml (PyMOL visualization)");
    println!("  - analysis_summary.json");
    println!("  - comprehensive_report.json (all sub-reports)");

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

fn write_pymol_script(path: &std::path::Path, sites: &[CrypticSite]) -> Result<()> {
    use std::io::Write;
    let mut file = fs::File::create(path)?;

    writeln!(file, "# PRISM-NHS PRO Cryptic Site Visualization")?;
    writeln!(file, "# Multi-factor confidence scoring")?;
    writeln!(file)?;
    writeln!(file, "# GREEN = HIGH (conf >= 0.70)")?;
    writeln!(file, "# YELLOW = MEDIUM (conf 0.50-0.70)")?;
    writeln!(file, "# RED = LOW (conf < 0.50)")?;
    writeln!(file)?;

    let max_spikes = sites.iter().map(|s| s.spike_count).max().unwrap_or(1) as f32;

    for site in sites.iter().take(25) {
        let size = 0.5 + 2.0 * (site.spike_count as f32 / max_spikes);
        let (r, g, b) = match site.category.as_str() {
            "HIGH" => (0.2, 0.9, 0.2),
            "MEDIUM" => (0.9, 0.9, 0.2),
            _ => (0.9, 0.3, 0.2),
        };

        writeln!(
            file,
            "# Site {} - {} spikes, conf={:.2} [{}]",
            site.id + 1,
            site.spike_count,
            site.overall_confidence,
            site.category
        )?;
        writeln!(
            file,
            "pseudoatom site_{}, pos=[{:.2}, {:.2}, {:.2}]",
            site.id + 1,
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
            site.id + 1
        )?;
        writeln!(file, "show spheres, site_{}", site.id + 1)?;
        writeln!(file, "set sphere_scale, {:.2}, site_{}", size, site.id + 1)?;

        if !site.residues.is_empty() {
            let res_sel = site
                .residues
                .iter()
                .take(10)
                .map(|r| format!("resi {}", r))
                .collect::<Vec<_>>()
                .join(" or ");
            writeln!(file, "select site_{}_residues, {}", site.id + 1, res_sel)?;
        }
        writeln!(file)?;
    }

    writeln!(file, "group cryptic_sites, site_*")?;
    writeln!(file, "zoom cryptic_sites")?;

    Ok(())
}
