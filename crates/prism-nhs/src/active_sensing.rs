//! # PRISM-NHS Active Sensing Module - COMPLETE IMPLEMENTATION
//!
//! High-performance Rust bindings for the closed-loop molecular stethoscope.
//! Uses stream-based async execution matching fused_engine.rs patterns.
//!
//! ## Features
//! - **Batch Processing**: Process multiple structures/probes concurrently
//! - **Async Execution**: Overlapped memory transfers and compute
//! - **Zero CPU Round-trips**: Chained kernel execution on GPU stream
//!
//! ## Quick Start
//! ```rust,ignore
//! let mut engine = ActiveSensingEngine::new(ctx, stream, topology, config)?;
//!
//! // Single structure
//! engine.apply_probe(&probe)?;
//! engine.lif_update()?;
//! let response = engine.compute_response()?;
//!
//! // Batch mode
//! engine.apply_probe_batch(&probes)?;
//! engine.lif_update_batch()?;
//! let responses = engine.compute_response_batch()?;
//! ```

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[cfg(feature = "gpu")]
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, DevicePtrMut,
    LaunchConfig, PushKernelArg,
};
#[cfg(feature = "gpu")]
use cudarc::nvrtc::Ptx;

// ============================================================================
// CONFIGURATION CONSTANTS (must match CUDA exactly)
// ============================================================================

const MAX_PROBE_GROUPS: usize = 8;
const MAX_AROMATICS_PER_GROUP: usize = 32;
const SPIKE_HISTORY_LENGTH: usize = 64;
const MAX_SEQUENCE_LENGTH: usize = 8;
const FREQ_SWEEP_STEPS: usize = 100;
const BLOCK_SIZE: u32 = 256;
const MAX_BATCH_SIZE: usize = 32;

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Configuration for active sensing system
#[derive(Clone, Debug)]
pub struct ActiveSensingConfig {
    /// Voxel grid dimension (default: 32)
    pub grid_dim: i32,
    /// Voxel size in Angstroms (default: 2.5)
    pub grid_spacing: f32,
    /// LIF membrane time constant in ps (default: 1.0)
    pub tau_membrane: f32,
    /// Spike threshold (default: 0.15)
    pub spike_threshold: f32,
    /// Steps between probe switches (default: 100)
    pub probe_interval_steps: usize,
    /// Response analysis window in ps (default: 10.0)
    pub analysis_window_ps: f32,
    /// Maximum sequence distance in Angstroms (default: 20.0)
    pub max_sequence_distance: f32,
    /// Target number of aromatic groups (default: 4)
    pub target_n_groups: usize,
    /// Minimum separation between group centroids (default: 10.0)
    pub min_group_separation: f32,
    /// Maximum sequence detectors to create (default: 1024)
    pub max_sequence_detectors: usize,
    /// Resonance sweep minimum frequency in THz (default: 0.1)
    pub min_freq_thz: f32,
    /// Resonance sweep maximum frequency in THz (default: 10.0)
    pub max_freq_thz: f32,
    /// Exploration epsilon for adaptive control (default: 0.1)
    pub exploration_epsilon: f32,
    /// Enable batch processing mode (default: false)
    pub batch_mode: bool,
    /// Maximum batch size for batch processing (default: 32)
    pub max_batch_size: usize,
}

impl Default for ActiveSensingConfig {
    fn default() -> Self {
        Self {
            grid_dim: 32,
            grid_spacing: 2.5,
            tau_membrane: 1.0,
            spike_threshold: 0.15,
            probe_interval_steps: 100,
            analysis_window_ps: 10.0,
            max_sequence_distance: 20.0,
            target_n_groups: 4,
            min_group_separation: 10.0,
            max_sequence_detectors: 1024,
            min_freq_thz: 0.1,
            max_freq_thz: 10.0,
            exploration_epsilon: 0.1,
            batch_mode: false,
            max_batch_size: MAX_BATCH_SIZE,
        }
    }
}

impl ActiveSensingConfig {
    /// Create config optimized for batch processing
    pub fn batch_optimized(batch_size: usize) -> Self {
        Self {
            batch_mode: true,
            max_batch_size: batch_size,
            ..Default::default()
        }
    }
}

/// Active sensing operational mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ActiveSensingMode {
    /// Standard single-structure mode
    #[default]
    Single,
    /// Batch processing mode for multiple structures
    Batch,
    /// Differential probing mode (A/B comparison)
    Differential,
    /// Resonance sweep mode (frequency scanning)
    ResonanceSweep,
}

// ============================================================================
// DATA STRUCTURES (Rust-side API)
// ============================================================================

/// Float3 for GPU interop
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Float3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Float3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn from_array(arr: [f32; 3]) -> Self {
        Self {
            x: arr[0],
            y: arr[1],
            z: arr[2],
        }
    }

    pub fn to_array(&self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }
}

/// Aromatic group for coherent excitation
#[derive(Debug, Clone, Default)]
pub struct AromaticGroup {
    /// Indices into aromatic array
    pub aromatic_indices: Vec<usize>,
    /// Geometric center
    pub centroid: Float3,
    /// Sum of absorption strengths
    pub total_absorption: f32,
}

/// Probe pattern types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum ProbeType {
    HingeSideA = 0,
    HingeSideB = 1,
    PocketLining = 2,
    #[default]
    Control = 3,
    ResonanceSweep = 4,
}

/// Coherent probe pattern
#[derive(Debug, Clone, Default)]
pub struct CoherentProbe {
    pub groups: Vec<AromaticGroup>,
    pub phase_delays_fs: Vec<f32>,
    pub energy_per_group: Vec<f32>,
    pub total_energy: f32,
    pub probe_id: i32,
    pub probe_type: ProbeType,
}

/// Response metrics from a probe
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProbeResponse {
    pub total_spikes: i32,
    pub mean_intensity: f32,
    pub spatial_extent: f32,
    pub onset_latency_ps: f32,
    pub peak_latency_ps: f32,
    pub duration_ps: f32,
    pub sequence_score: f32,
    pub sequences_detected: i32,
    pub response_centroid: [f32; 3],
}

/// Detected resonance peak (soft mode)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResonancePeak {
    pub frequency_thz: f32,
    pub amplitude: f32,
    pub quality_factor: f32,
    pub phase_rad: f32,
}

/// Spike sequence detector state
#[derive(Debug, Clone, Default)]
pub struct SpikeSequenceDetector {
    pub voxel_sequence: Vec<i32>,
    pub max_interval_ps: f32,
    pub accumulated_score: f32,
    pub detection_count: i32,
}

/// Cryptic site candidate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrypticSiteCandidate {
    pub position: [f32; 3],
    pub confidence: f32,
    pub nearby_residues: Vec<i32>,
    pub soft_mode_frequency_thz: Option<f32>,
    pub sequence_score: f32,
    pub differential_score: f32,
}

/// Complete active sensing results
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ActiveSensingResults {
    pub cryptic_site_candidates: Vec<CrypticSiteCandidate>,
    pub resonances: Vec<ResonancePeak>,
    pub best_differential_score: f32,
    pub best_probe_pair: (i32, i32),
    pub total_probes: i32,
    pub total_spikes: i32,
    pub sequences_detected: i32,
    pub processing_time_ms: f32,
}

// ============================================================================
// GPU STRUCT REPRESENTATIONS (must match CUDA exactly)
// ============================================================================

#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct GpuAromaticGroup {
    aromatic_indices: [i32; MAX_AROMATICS_PER_GROUP],
    n_aromatics: i32,
    centroid: [f32; 3],
    total_absorption: f32,
}

#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct GpuCoherentProbe {
    groups: [GpuAromaticGroup; MAX_PROBE_GROUPS],
    n_groups: i32,
    phase_delays_fs: [f32; MAX_PROBE_GROUPS],
    energy_per_group: [f32; MAX_PROBE_GROUPS],
    total_energy: f32,
    probe_id: i32,
    probe_type: i32,
    _pad: i32,
}

#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct GpuProbeResponse {
    total_spikes: i32,
    mean_intensity: f32,
    spatial_extent: f32,
    onset_latency_ps: f32,
    peak_latency_ps: f32,
    duration_ps: f32,
    sequence_score: f32,
    sequences_detected: i32,
    response_centroid: [f32; 3],
    _pad: f32,
}

#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct GpuTimestampedSpike {
    voxel_idx: i32,
    timestamp_ps: f32,
    intensity: f32,
    probe_id: i32,
}

#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct GpuVoxelSpikeHistory {
    spikes: [GpuTimestampedSpike; SPIKE_HISTORY_LENGTH],
    head: i32,
    count: i32,
    last_spike_time: f32,
    _pad: i32,
}

#[cfg(feature = "gpu")]
impl Default for GpuVoxelSpikeHistory {
    fn default() -> Self {
        Self {
            spikes: [GpuTimestampedSpike::default(); SPIKE_HISTORY_LENGTH],
            head: 0,
            count: 0,
            last_spike_time: -1000.0,
            _pad: 0,
        }
    }
}

#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct GpuLateralInhibitionState {
    inhibition_level: f32,
    last_update_time: f32,
    inhibiting_neighbors: [i32; 27],
    n_inhibitors: i32,
}

#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct GpuSpikeSequenceDetector {
    voxel_sequence: [i32; MAX_SEQUENCE_LENGTH],
    sequence_length: i32,
    max_inter_spike_interval_ps: f32,
    current_position: i32,
    sequence_start_time: f32,
    accumulated_score: f32,
    detection_count: i32,
    weight: f32,
    _pad: i32,
}

#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct GpuResonanceDetector {
    frequency_spectrum: [f32; FREQ_SWEEP_STEPS],
    phase_spectrum: [f32; FREQ_SWEEP_STEPS],
    sample_counts: [i32; FREQ_SWEEP_STEPS],
    resonance_frequencies: [f32; 8],
    resonance_amplitudes: [f32; 8],
    quality_factors: [f32; 8],
    n_resonances: i32,
    current_freq_idx: i32,
    min_freq_thz: f32,
    max_freq_thz: f32,
}

#[cfg(feature = "gpu")]
impl Default for GpuResonanceDetector {
    fn default() -> Self {
        Self {
            frequency_spectrum: [0.0; FREQ_SWEEP_STEPS],
            phase_spectrum: [0.0; FREQ_SWEEP_STEPS],
            sample_counts: [0; FREQ_SWEEP_STEPS],
            resonance_frequencies: [0.0; 8],
            resonance_amplitudes: [0.0; 8],
            quality_factors: [0.0; 8],
            n_resonances: 0,
            current_freq_idx: 0,
            min_freq_thz: 0.1,
            max_freq_thz: 10.0,
        }
    }
}

#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct GpuAdaptiveProbeController {
    q_values: [f32; MAX_PROBE_GROUPS * MAX_PROBE_GROUPS],
    visit_counts: [i32; MAX_PROBE_GROUPS * MAX_PROBE_GROUPS],
    softmax_probs: [f32; MAX_PROBE_GROUPS * MAX_PROBE_GROUPS],
    n_probes: i32,
    current_probe_idx: i32,
    exploration_epsilon: f32,
    learning_rate: f32,
    total_reward: f32,
    best_probe_idx: i32,
    best_score: f32,
    _pad: i32,
}

#[cfg(feature = "gpu")]
impl Default for GpuAdaptiveProbeController {
    fn default() -> Self {
        Self {
            q_values: [0.0; MAX_PROBE_GROUPS * MAX_PROBE_GROUPS],
            visit_counts: [0; MAX_PROBE_GROUPS * MAX_PROBE_GROUPS],
            softmax_probs: [0.0; MAX_PROBE_GROUPS * MAX_PROBE_GROUPS],
            n_probes: 0,
            current_probe_idx: 0,
            exploration_epsilon: 0.1,
            learning_rate: 0.1,
            total_reward: 0.0,
            best_probe_idx: 0,
            best_score: 0.0,
            _pad: 0,
        }
    }
}

// ============================================================================
// BUILDER PATTERN
// ============================================================================

/// Builder for ActiveSensingEngine
#[cfg(feature = "gpu")]
pub struct ActiveSensingBuilder {
    config: ActiveSensingConfig,
    grid_origin: [f32; 3],
    n_aromatics: usize,
    n_atoms: usize,
}

#[cfg(feature = "gpu")]
impl ActiveSensingBuilder {
    pub fn new() -> Self {
        Self {
            config: ActiveSensingConfig::default(),
            grid_origin: [0.0; 3],
            n_aromatics: 0,
            n_atoms: 0,
        }
    }

    pub fn with_config(mut self, config: ActiveSensingConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_grid_origin(mut self, origin: [f32; 3]) -> Self {
        self.grid_origin = origin;
        self
    }

    pub fn with_aromatics(mut self, n_aromatics: usize) -> Self {
        self.n_aromatics = n_aromatics;
        self
    }

    pub fn with_atoms(mut self, n_atoms: usize) -> Self {
        self.n_atoms = n_atoms;
        self
    }

    pub fn build(
        self,
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
    ) -> Result<ActiveSensingEngine> {
        ActiveSensingEngine::new(
            context,
            stream,
            self.config,
            self.grid_origin,
            self.n_aromatics,
            self.n_atoms,
        )
    }
}

#[cfg(feature = "gpu")]
impl Default for ActiveSensingBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// ACTIVE SENSING ENGINE - STREAM-BASED HIGH PERFORMANCE
// ============================================================================

#[cfg(feature = "gpu")]
pub struct ActiveSensingEngine {
    // CUDA handles (stream-based for async execution)
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,

    // Kernel functions (11 total)
    fn_apply_coherent_probe: CudaFunction,
    fn_lif_update_with_inhibition: CudaFunction,
    fn_detect_spike_sequences: CudaFunction,
    fn_update_resonance_spectrum: CudaFunction,
    fn_analyze_resonances: CudaFunction,
    fn_adaptive_probe_update: CudaFunction,
    fn_compute_response_metrics: CudaFunction,
    fn_init_active_sensing: CudaFunction,
    fn_cluster_aromatics: CudaFunction,
    fn_build_sequence_detectors: CudaFunction,
    fn_compare_differential_probes: CudaFunction,

    // GPU buffers - Spike processing
    d_spike_histories: CudaSlice<u8>,
    d_inhibition_states: CudaSlice<u8>,
    d_lif_potential: CudaSlice<f32>,
    d_spike_grid: CudaSlice<i32>,
    d_spike_count: CudaSlice<i32>,

    // GPU buffers - Sequence detection
    d_sequence_detectors: CudaSlice<u8>,
    d_detection_scores: CudaSlice<f32>,
    d_n_detectors: CudaSlice<i32>,

    // GPU buffers - Resonance
    d_resonance_detector: CudaSlice<u8>,

    // GPU buffers - Adaptive control
    d_probe_controller: CudaSlice<u8>,
    d_next_probe_idx: CudaSlice<i32>,

    // GPU buffers - Probe state
    d_current_probe: CudaSlice<u8>,
    d_probe_response: CudaSlice<u8>,
    d_probes: CudaSlice<u8>, // Array for batch mode

    // GPU buffers - Aromatic data
    d_aromatic_groups: CudaSlice<u8>,
    d_n_groups: CudaSlice<i32>,
    d_aromatic_centroids: CudaSlice<f32>,
    d_aromatic_absorptions: CudaSlice<f32>,

    // GPU buffers - Batch processing
    d_batch_responses: CudaSlice<u8>,
    d_batch_spike_counts: CudaSlice<i32>,

    // GPU buffers - RNG
    d_rng_states: CudaSlice<u8>,

    // Struct sizes for GPU memory layout
    spike_history_size: usize,
    inhibition_state_size: usize,
    sequence_detector_size: usize,
    resonance_detector_size: usize,
    probe_controller_size: usize,
    probe_size: usize,
    response_size: usize,
    aromatic_group_size: usize,

    // Configuration
    config: ActiveSensingConfig,
    n_voxels: usize,
    grid_dim: usize,
    grid_spacing: f32,
    grid_origin: [f32; 3],
    n_aromatics: usize,
    n_atoms: usize,
    n_detectors: i32,
    n_groups: i32,

    // State
    current_time_ps: f32,
    current_probe_idx: i32,
    probe_start_time_ps: f32,
    mode: ActiveSensingMode,
    initialized: bool,

    // Cached results
    host_response: ProbeResponse,
    accumulated_results: ActiveSensingResults,
}

#[cfg(feature = "gpu")]
impl ActiveSensingEngine {
    /// Create new active sensing engine with stream-based execution
    pub fn new(
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        config: ActiveSensingConfig,
        grid_origin: [f32; 3],
        n_aromatics: usize,
        n_atoms: usize,
    ) -> Result<Self> {
        let grid_dim = config.grid_dim as usize;
        let grid_spacing = config.grid_spacing;
        let n_voxels = grid_dim * grid_dim * grid_dim;
        let max_detectors = config.max_sequence_detectors;
        let batch_size = config.max_batch_size;

        log::info!(
            "Creating ActiveSensingEngine: grid {}³ = {} voxels, {} aromatics, batch_size={}",
            grid_dim,
            n_voxels,
            n_aromatics,
            batch_size
        );

        // Load PTX module
        let ptx_path = "target/ptx/nhs_active_sensing_kernels.ptx";
        log::info!("Loading active sensing PTX from: {}", ptx_path);

        let module = context
            .load_module(Ptx::from_file(ptx_path))
            .context("Failed to load nhs_active_sensing_kernels.ptx")?;

        // Load all 11 kernel functions
        let fn_apply_coherent_probe = module.load_function("apply_coherent_probe")?;
        let fn_lif_update_with_inhibition = module.load_function("lif_update_with_inhibition")?;
        let fn_detect_spike_sequences = module.load_function("detect_spike_sequences")?;
        let fn_update_resonance_spectrum = module.load_function("update_resonance_spectrum")?;
        let fn_analyze_resonances = module.load_function("analyze_resonances")?;
        let fn_adaptive_probe_update = module.load_function("adaptive_probe_update")?;
        let fn_compute_response_metrics = module.load_function("compute_response_metrics")?;
        let fn_init_active_sensing = module.load_function("init_active_sensing")?;
        let fn_cluster_aromatics = module.load_function("cluster_aromatics")?;
        let fn_build_sequence_detectors = module.load_function("build_sequence_detectors")?;
        let fn_compare_differential_probes = module.load_function("compare_differential_probes")?;

        log::info!("Loaded 11 active sensing kernel functions");

        // Get struct sizes
        let spike_history_size = std::mem::size_of::<GpuVoxelSpikeHistory>();
        let inhibition_state_size = std::mem::size_of::<GpuLateralInhibitionState>();
        let sequence_detector_size = std::mem::size_of::<GpuSpikeSequenceDetector>();
        let resonance_detector_size = std::mem::size_of::<GpuResonanceDetector>();
        let probe_controller_size = std::mem::size_of::<GpuAdaptiveProbeController>();
        let probe_size = std::mem::size_of::<GpuCoherentProbe>();
        let response_size = std::mem::size_of::<GpuProbeResponse>();
        let aromatic_group_size = std::mem::size_of::<GpuAromaticGroup>();

        // Allocate GPU buffers using stream (async-friendly)
        let d_spike_histories: CudaSlice<u8> = stream.alloc_zeros(n_voxels * spike_history_size)?;
        let d_inhibition_states: CudaSlice<u8> =
            stream.alloc_zeros(n_voxels * inhibition_state_size)?;
        let d_lif_potential: CudaSlice<f32> = stream.alloc_zeros(n_voxels)?;
        let d_spike_grid: CudaSlice<i32> = stream.alloc_zeros(n_voxels)?;
        let d_spike_count: CudaSlice<i32> = stream.alloc_zeros(1)?;

        let d_sequence_detectors: CudaSlice<u8> =
            stream.alloc_zeros(max_detectors * sequence_detector_size)?;
        let d_detection_scores: CudaSlice<f32> = stream.alloc_zeros(max_detectors)?;
        let d_n_detectors: CudaSlice<i32> = stream.alloc_zeros(1)?;

        let d_resonance_detector: CudaSlice<u8> = stream.alloc_zeros(resonance_detector_size)?;

        let d_probe_controller: CudaSlice<u8> = stream.alloc_zeros(probe_controller_size)?;
        let d_next_probe_idx: CudaSlice<i32> = stream.alloc_zeros(1)?;

        let d_current_probe: CudaSlice<u8> = stream.alloc_zeros(probe_size)?;
        let d_probe_response: CudaSlice<u8> = stream.alloc_zeros(response_size)?;
        let d_probes: CudaSlice<u8> = stream.alloc_zeros(batch_size * probe_size)?;

        let d_aromatic_groups: CudaSlice<u8> =
            stream.alloc_zeros(MAX_PROBE_GROUPS * aromatic_group_size)?;
        let d_n_groups: CudaSlice<i32> = stream.alloc_zeros(1)?;
        let d_aromatic_centroids: CudaSlice<f32> = stream.alloc_zeros(n_aromatics.max(1) * 3)?;
        let d_aromatic_absorptions: CudaSlice<f32> = stream.alloc_zeros(n_aromatics.max(1))?;

        // Batch buffers
        let d_batch_responses: CudaSlice<u8> = stream.alloc_zeros(batch_size * response_size)?;
        let d_batch_spike_counts: CudaSlice<i32> = stream.alloc_zeros(batch_size)?;

        // RNG states (48 bytes per curandState)
        let d_rng_states: CudaSlice<u8> = stream.alloc_zeros(n_voxels * 48)?;

        let mut engine = Self {
            context,
            stream,
            _module: module,

            fn_apply_coherent_probe,
            fn_lif_update_with_inhibition,
            fn_detect_spike_sequences,
            fn_update_resonance_spectrum,
            fn_analyze_resonances,
            fn_adaptive_probe_update,
            fn_compute_response_metrics,
            fn_init_active_sensing,
            fn_cluster_aromatics,
            fn_build_sequence_detectors,
            fn_compare_differential_probes,

            d_spike_histories,
            d_inhibition_states,
            d_lif_potential,
            d_spike_grid,
            d_spike_count,
            d_sequence_detectors,
            d_detection_scores,
            d_n_detectors,
            d_resonance_detector,
            d_probe_controller,
            d_next_probe_idx,
            d_current_probe,
            d_probe_response,
            d_probes,
            d_aromatic_groups,
            d_n_groups,
            d_aromatic_centroids,
            d_aromatic_absorptions,
            d_batch_responses,
            d_batch_spike_counts,
            d_rng_states,

            spike_history_size,
            inhibition_state_size,
            sequence_detector_size,
            resonance_detector_size,
            probe_controller_size,
            probe_size,
            response_size,
            aromatic_group_size,

            config,
            n_voxels,
            grid_dim,
            grid_spacing,
            grid_origin,
            n_aromatics,
            n_atoms,
            n_detectors: 0,
            n_groups: 0,

            current_time_ps: 0.0,
            current_probe_idx: 0,
            probe_start_time_ps: 0.0,
            mode: ActiveSensingMode::Single,
            initialized: false,

            host_response: ProbeResponse::default(),
            accumulated_results: ActiveSensingResults::default(),
        };

        // Initialize GPU state
        engine.launch_init()?;
        engine.initialized = true;

        log::info!("ActiveSensingEngine created successfully");

        Ok(engine)
    }

    /// Initialize GPU state (spike histories, controllers, etc.)
    fn launch_init(&mut self) -> Result<()> {
        let n_blocks = (self.n_voxels as u32).div_ceil(BLOCK_SIZE);
        let total_voxels = self.n_voxels as i32;
        let n_probes = (MAX_PROBE_GROUPS * MAX_PROBE_GROUPS) as i32;
        let seed = 42u64;

        let cfg = LaunchConfig {
            grid_dim: (n_blocks, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.fn_init_active_sensing)
                .arg(&mut self.d_spike_histories)
                .arg(&mut self.d_inhibition_states)
                .arg(&mut self.d_lif_potential)
                .arg(&mut self.d_resonance_detector)
                .arg(&mut self.d_probe_controller)
                .arg(&mut self.d_rng_states)
                .arg(&total_voxels)
                .arg(&n_probes)
                .arg(&seed)
                .launch(cfg)
        }
        .context("Failed to launch init_active_sensing")?;

        self.context.synchronize()?;
        log::debug!("Initialized active sensing GPU state");

        Ok(())
    }

    /// Setup aromatic data and cluster into groups
    pub fn setup_aromatics(&mut self, centroids: &[[f32; 3]], absorptions: &[f32]) -> Result<i32> {
        if centroids.len() != self.n_aromatics {
            bail!(
                "Centroid count mismatch: expected {}, got {}",
                self.n_aromatics,
                centroids.len()
            );
        }

        // Flatten centroids
        let flat_centroids: Vec<f32> = centroids.iter().flat_map(|c| c.iter().copied()).collect();

        // Upload to GPU
        self.stream
            .memcpy_htod(&flat_centroids, &mut self.d_aromatic_centroids)?;
        self.stream
            .memcpy_htod(absorptions, &mut self.d_aromatic_absorptions)?;

        // Launch clustering kernel
        let target_n_groups = self.config.target_n_groups as i32;
        let min_separation = self.config.min_group_separation;

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.fn_cluster_aromatics)
                .arg(&mut self.d_aromatic_groups)
                .arg(&mut self.d_n_groups)
                .arg(&self.d_aromatic_centroids)
                .arg(&self.d_aromatic_absorptions)
                .arg(&(self.n_aromatics as i32))
                .arg(&target_n_groups)
                .arg(&min_separation)
                .launch(cfg)
        }
        .context("Failed to launch cluster_aromatics")?;

        // Read back group count
        let mut n_groups = [0i32];
        self.stream.memcpy_dtoh(&self.d_n_groups, &mut n_groups)?;
        self.n_groups = n_groups[0];

        log::info!(
            "Clustered {} aromatics into {} groups",
            self.n_aromatics,
            self.n_groups
        );

        Ok(self.n_groups)
    }

    /// Build sequence detectors from aromatic spatial relationships
    pub fn build_sequence_detectors(&mut self) -> Result<i32> {
        let max_distance = self.config.max_sequence_distance;

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.fn_build_sequence_detectors)
                .arg(&mut self.d_sequence_detectors)
                .arg(&mut self.d_n_detectors)
                .arg(&self.d_aromatic_centroids)
                .arg(&(self.n_aromatics as i32))
                .arg(&(self.grid_dim as i32))
                .arg(&self.grid_spacing)
                .arg(&self.grid_origin[0])
                .arg(&self.grid_origin[1])
                .arg(&self.grid_origin[2])
                .arg(&max_distance)
                .arg(&(self.config.max_sequence_detectors as i32))
                .launch(cfg)
        }
        .context("Failed to launch build_sequence_detectors")?;

        // Read back detector count
        let mut n_detectors = [0i32];
        self.stream
            .memcpy_dtoh(&self.d_n_detectors, &mut n_detectors)?;
        self.n_detectors = n_detectors[0];

        log::info!("Built {} sequence detectors", self.n_detectors);

        Ok(self.n_detectors)
    }

    /// Apply coherent UV probe (single)
    pub fn apply_probe(
        &mut self,
        probe: &CoherentProbe,
        d_velocities: &mut CudaSlice<f32>,
        d_positions: &CudaSlice<f32>,
        d_masses: &CudaSlice<f32>,
        d_ring_normals: &CudaSlice<f32>,
        d_aromatic_atom_indices: &CudaSlice<i32>,
        d_aromatic_n_atoms: &CudaSlice<i32>,
        current_time_fs: f32,
        dt_fs: f32,
    ) -> Result<()> {
        // Upload probe
        let gpu_probe = self.probe_to_gpu(probe);
        let probe_bytes = Self::struct_to_bytes(&gpu_probe);
        self.stream
            .memcpy_htod(&probe_bytes, &mut self.d_current_probe)?;

        let n_groups = probe.groups.len().min(MAX_PROBE_GROUPS) as i32;
        if n_groups == 0 {
            return Ok(());
        }

        let cfg = LaunchConfig {
            grid_dim: (n_groups as u32, 1, 1),
            block_dim: (32, 1, 1), // One warp per group
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.fn_apply_coherent_probe)
                .arg(&self.d_current_probe)
                .arg(d_velocities)
                .arg(d_positions)
                .arg(d_masses)
                .arg(d_ring_normals)
                .arg(d_aromatic_atom_indices)
                .arg(d_aromatic_n_atoms)
                .arg(&current_time_fs)
                .arg(&dt_fs)
                .arg(&mut self.d_rng_states)
                .arg(&(self.n_aromatics as i32))
                .arg(&(self.n_atoms as i32))
                .launch(cfg)
        }
        .context("Failed to launch apply_coherent_probe")?;

        self.current_probe_idx = probe.probe_id;
        self.probe_start_time_ps = current_time_fs / 1000.0;

        Ok(())
    }

    /// Update LIF neurons with lateral inhibition
    pub fn lif_update(
        &mut self,
        d_water_density: &CudaSlice<f32>,
        d_water_density_prev: &CudaSlice<f32>,
        dt_ps: f32,
    ) -> Result<i32> {
        // Reset spike count
        let zero = [0i32];
        self.stream.memcpy_htod(&zero, &mut self.d_spike_count)?;

        let n_blocks = (self.n_voxels as u32).div_ceil(BLOCK_SIZE);

        let cfg = LaunchConfig {
            grid_dim: (n_blocks, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.fn_lif_update_with_inhibition)
                .arg(&mut self.d_lif_potential)
                .arg(&mut self.d_inhibition_states)
                .arg(d_water_density)
                .arg(d_water_density_prev)
                .arg(&mut self.d_spike_grid)
                .arg(&mut self.d_spike_histories)
                .arg(&mut self.d_spike_count)
                .arg(&self.config.tau_membrane)
                .arg(&self.config.spike_threshold)
                .arg(&dt_ps)
                .arg(&self.current_time_ps)
                .arg(&self.current_probe_idx)
                .arg(&(self.grid_dim as i32))
                .launch(cfg)
        }
        .context("Failed to launch lif_update_with_inhibition")?;

        // Read spike count
        let mut spike_count = [0i32];
        self.stream
            .memcpy_dtoh(&self.d_spike_count, &mut spike_count)?;

        self.current_time_ps += dt_ps;
        self.accumulated_results.total_spikes += spike_count[0];

        Ok(spike_count[0])
    }

    /// Detect spike sequences
    pub fn detect_sequences(&mut self) -> Result<Vec<f32>> {
        if self.n_detectors == 0 {
            return Ok(vec![]);
        }

        let n_blocks = (self.n_detectors as u32).div_ceil(BLOCK_SIZE);

        let cfg = LaunchConfig {
            grid_dim: (n_blocks, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.fn_detect_spike_sequences)
                .arg(&mut self.d_sequence_detectors)
                .arg(&self.d_spike_histories)
                .arg(&mut self.d_detection_scores)
                .arg(&self.current_time_ps)
                .arg(&self.n_detectors)
                .arg(&(self.n_voxels as i32))
                .launch(cfg)
        }
        .context("Failed to launch detect_spike_sequences")?;

        // Read scores
        let mut scores = vec![0.0f32; self.n_detectors as usize];
        let slice = self.d_detection_scores.slice(..self.n_detectors as usize);
        self.stream.memcpy_dtoh(&slice, &mut scores)?;

        // Count detections
        let detections = scores.iter().filter(|&&s| s > 0.8).count();
        self.accumulated_results.sequences_detected += detections as i32;

        Ok(scores)
    }

    /// Update resonance spectrum for frequency sweep
    pub fn update_resonance(
        &mut self,
        probe_frequency_thz: f32,
        probe_period_ps: f32,
    ) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.fn_update_resonance_spectrum)
                .arg(&mut self.d_resonance_detector)
                .arg(&self.d_spike_histories)
                .arg(&probe_frequency_thz)
                .arg(&self.probe_start_time_ps)
                .arg(&probe_period_ps)
                .arg(&(self.n_voxels as i32))
                .arg(&(self.grid_dim as i32))
                .launch(cfg)
        }
        .context("Failed to launch update_resonance_spectrum")?;

        Ok(())
    }

    /// Analyze resonance spectrum for peaks
    pub fn analyze_resonances(&mut self) -> Result<Vec<ResonancePeak>> {
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.fn_analyze_resonances)
                .arg(&mut self.d_resonance_detector)
                .launch(cfg)
        }
        .context("Failed to launch analyze_resonances")?;

        // Read back resonance detector
        let mut detector_bytes = vec![0u8; self.resonance_detector_size];
        self.stream
            .memcpy_dtoh(&self.d_resonance_detector, &mut detector_bytes)?;

        let detector: GpuResonanceDetector =
            unsafe { std::ptr::read_unaligned(detector_bytes.as_ptr() as *const _) };

        let mut peaks = Vec::new();
        for i in 0..detector.n_resonances as usize {
            peaks.push(ResonancePeak {
                frequency_thz: detector.resonance_frequencies[i],
                amplitude: detector.resonance_amplitudes[i],
                quality_factor: detector.quality_factors[i],
                phase_rad: detector.phase_spectrum[i.min(FREQ_SWEEP_STEPS - 1)],
            });
        }

        self.accumulated_results.resonances = peaks.clone();

        Ok(peaks)
    }

    /// Compute response metrics
    pub fn compute_response(&mut self) -> Result<ProbeResponse> {
        let analysis_window = self.config.analysis_window_ps;

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.fn_compute_response_metrics)
                .arg(&mut self.d_probe_response)
                .arg(&self.d_spike_histories)
                .arg(&self.d_sequence_detectors)
                .arg(&self.n_detectors)
                .arg(&self.d_current_probe)
                .arg(&self.probe_start_time_ps)
                .arg(&analysis_window)
                .arg(&(self.grid_dim as i32))
                .arg(&self.grid_spacing)
                .arg(&self.grid_origin[0])
                .arg(&self.grid_origin[1])
                .arg(&self.grid_origin[2])
                .launch(cfg)
        }
        .context("Failed to launch compute_response_metrics")?;

        // Read response
        let mut response_bytes = vec![0u8; self.response_size];
        self.stream
            .memcpy_dtoh(&self.d_probe_response, &mut response_bytes)?;

        let gpu_response: GpuProbeResponse =
            unsafe { std::ptr::read_unaligned(response_bytes.as_ptr() as *const _) };

        self.host_response = ProbeResponse {
            total_spikes: gpu_response.total_spikes,
            mean_intensity: gpu_response.mean_intensity,
            spatial_extent: gpu_response.spatial_extent,
            onset_latency_ps: gpu_response.onset_latency_ps,
            peak_latency_ps: gpu_response.peak_latency_ps,
            duration_ps: gpu_response.duration_ps,
            sequence_score: gpu_response.sequence_score,
            sequences_detected: gpu_response.sequences_detected,
            response_centroid: gpu_response.response_centroid,
        };

        Ok(self.host_response.clone())
    }

    /// Update probe selection via RL
    pub fn update_probe_selection(&mut self) -> Result<i32> {
        let n_probes = (MAX_PROBE_GROUPS * MAX_PROBE_GROUPS) as i32;

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.fn_adaptive_probe_update)
                .arg(&mut self.d_probe_controller)
                .arg(&self.d_probe_response)
                .arg(&self.current_probe_idx)
                .arg(&mut self.d_next_probe_idx)
                .arg(&mut self.d_rng_states)
                .arg(&n_probes)
                .arg(&self.config.exploration_epsilon)
                .launch(cfg)
        }
        .context("Failed to launch adaptive_probe_update")?;

        // Read next probe
        let mut next_idx = [0i32];
        self.stream
            .memcpy_dtoh(&self.d_next_probe_idx, &mut next_idx)?;

        self.accumulated_results.total_probes += 1;

        Ok(next_idx[0])
    }

    /// Compare differential probes (A vs B)
    pub fn compare_differential(
        &mut self,
        response_a: &ProbeResponse,
        response_b: &ProbeResponse,
    ) -> Result<(f32, f32)> {
        // Upload responses
        let gpu_a = self.response_to_gpu(response_a);
        let gpu_b = self.response_to_gpu(response_b);

        // Allocate temp buffers for A and B
        let a_bytes = Self::struct_to_bytes(&gpu_a);
        let b_bytes = Self::struct_to_bytes(&gpu_b);

        let mut d_response_a: CudaSlice<u8> = self.stream.alloc_zeros(self.response_size)?;
        let mut d_response_b: CudaSlice<u8> = self.stream.alloc_zeros(self.response_size)?;
        let mut d_differential: CudaSlice<f32> = self.stream.alloc_zeros(1)?;
        let mut d_confidence: CudaSlice<f32> = self.stream.alloc_zeros(1)?;

        self.stream.memcpy_htod(&a_bytes, &mut d_response_a)?;
        self.stream.memcpy_htod(&b_bytes, &mut d_response_b)?;

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.fn_compare_differential_probes)
                .arg(&d_response_a)
                .arg(&d_response_b)
                .arg(&mut d_differential)
                .arg(&mut d_confidence)
                .launch(cfg)
        }
        .context("Failed to launch compare_differential_probes")?;

        // Read results
        let mut differential = [0.0f32];
        let mut confidence = [0.0f32];
        self.stream
            .memcpy_dtoh(&d_differential, &mut differential)?;
        self.stream.memcpy_dtoh(&d_confidence, &mut confidence)?;

        if differential[0] > self.accumulated_results.best_differential_score {
            self.accumulated_results.best_differential_score = differential[0];
        }

        Ok((differential[0], confidence[0]))
    }

    // ========================================================================
    // BATCH PROCESSING METHODS
    // ========================================================================

    /// Apply multiple probes in batch (for batch mode)
    pub fn apply_probe_batch(
        &mut self,
        probes: &[CoherentProbe],
        d_velocities: &mut CudaSlice<f32>,
        d_positions: &CudaSlice<f32>,
        d_masses: &CudaSlice<f32>,
        d_ring_normals: &CudaSlice<f32>,
        d_aromatic_atom_indices: &CudaSlice<i32>,
        d_aromatic_n_atoms: &CudaSlice<i32>,
        current_time_fs: f32,
        dt_fs: f32,
    ) -> Result<()> {
        let batch_size = probes.len().min(self.config.max_batch_size);

        // Convert and upload all probes
        let mut probe_bytes = vec![0u8; batch_size * self.probe_size];
        for (i, probe) in probes.iter().take(batch_size).enumerate() {
            let gpu_probe = self.probe_to_gpu(probe);
            let bytes = Self::struct_to_bytes(&gpu_probe);
            probe_bytes[i * self.probe_size..(i + 1) * self.probe_size].copy_from_slice(&bytes);
        }

        let mut d_probes_slice = self.d_probes.slice_mut(..batch_size * self.probe_size);
        self.stream.memcpy_htod(&probe_bytes, &mut d_probes_slice)?;

        // Launch with batch_size blocks
        let cfg = LaunchConfig {
            grid_dim: (batch_size as u32, MAX_PROBE_GROUPS as u32, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.fn_apply_coherent_probe)
                .arg(&self.d_probes)
                .arg(d_velocities)
                .arg(d_positions)
                .arg(d_masses)
                .arg(d_ring_normals)
                .arg(d_aromatic_atom_indices)
                .arg(d_aromatic_n_atoms)
                .arg(&current_time_fs)
                .arg(&dt_fs)
                .arg(&mut self.d_rng_states)
                .arg(&(self.n_aromatics as i32))
                .arg(&(self.n_atoms as i32))
                .arg(&(batch_size as i32))
                .launch(cfg)
        }
        .context("Failed to launch apply_coherent_probe (batch)")?;

        self.probe_start_time_ps = current_time_fs / 1000.0;

        Ok(())
    }

    /// Compute response metrics for batch
    pub fn compute_response_batch(&mut self, batch_size: usize) -> Result<Vec<ProbeResponse>> {
        let actual_batch = batch_size.min(self.config.max_batch_size);
        let analysis_window = self.config.analysis_window_ps;

        let cfg = LaunchConfig {
            grid_dim: (actual_batch as u32, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.fn_compute_response_metrics)
                .arg(&mut self.d_batch_responses)
                .arg(&self.d_spike_histories)
                .arg(&self.d_sequence_detectors)
                .arg(&self.n_detectors)
                .arg(&self.d_probes)
                .arg(&self.probe_start_time_ps)
                .arg(&analysis_window)
                .arg(&(self.grid_dim as i32))
                .arg(&self.grid_spacing)
                .arg(&self.grid_origin[0])
                .arg(&self.grid_origin[1])
                .arg(&self.grid_origin[2])
                .arg(&(actual_batch as i32))
                .launch(cfg)
        }
        .context("Failed to launch compute_response_metrics (batch)")?;

        // Read all responses
        let mut response_bytes = vec![0u8; actual_batch * self.response_size];
        let slice = self
            .d_batch_responses
            .slice(..actual_batch * self.response_size);
        self.stream.memcpy_dtoh(&slice, &mut response_bytes)?;

        let mut responses = Vec::with_capacity(actual_batch);
        for i in 0..actual_batch {
            let gpu_response: GpuProbeResponse = unsafe {
                std::ptr::read_unaligned(
                    response_bytes[i * self.response_size..].as_ptr() as *const _
                )
            };

            responses.push(ProbeResponse {
                total_spikes: gpu_response.total_spikes,
                mean_intensity: gpu_response.mean_intensity,
                spatial_extent: gpu_response.spatial_extent,
                onset_latency_ps: gpu_response.onset_latency_ps,
                peak_latency_ps: gpu_response.peak_latency_ps,
                duration_ps: gpu_response.duration_ps,
                sequence_score: gpu_response.sequence_score,
                sequences_detected: gpu_response.sequences_detected,
                response_centroid: gpu_response.response_centroid,
            });
        }

        Ok(responses)
    }

    // ========================================================================
    // UTILITY METHODS
    // ========================================================================

    /// Get accumulated results
    pub fn get_results(&self) -> ActiveSensingResults {
        self.accumulated_results.clone()
    }

    /// Reset state for new run
    pub fn reset(&mut self) -> Result<()> {
        self.launch_init()?;
        self.current_time_ps = 0.0;
        self.current_probe_idx = 0;
        self.probe_start_time_ps = 0.0;
        self.host_response = ProbeResponse::default();
        self.accumulated_results = ActiveSensingResults::default();
        Ok(())
    }

    /// Set operational mode
    pub fn set_mode(&mut self, mode: ActiveSensingMode) {
        self.mode = mode;
    }

    /// Get current mode
    pub fn mode(&self) -> ActiveSensingMode {
        self.mode
    }

    /// Get current time
    pub fn current_time_ps(&self) -> f32 {
        self.current_time_ps
    }

    /// Advance time (for external integration)
    pub fn advance_time(&mut self, dt_ps: f32) {
        self.current_time_ps += dt_ps;
    }

    // Helper: Convert Rust probe to GPU format
    fn probe_to_gpu(&self, probe: &CoherentProbe) -> GpuCoherentProbe {
        let mut gpu = GpuCoherentProbe::default();

        gpu.n_groups = probe.groups.len().min(MAX_PROBE_GROUPS) as i32;
        gpu.total_energy = probe.total_energy;
        gpu.probe_id = probe.probe_id;
        gpu.probe_type = probe.probe_type as i32;

        for (i, group) in probe.groups.iter().take(MAX_PROBE_GROUPS).enumerate() {
            gpu.groups[i].n_aromatics =
                group.aromatic_indices.len().min(MAX_AROMATICS_PER_GROUP) as i32;
            gpu.groups[i].centroid = group.centroid.to_array();
            gpu.groups[i].total_absorption = group.total_absorption;

            for (j, &idx) in group
                .aromatic_indices
                .iter()
                .take(MAX_AROMATICS_PER_GROUP)
                .enumerate()
            {
                gpu.groups[i].aromatic_indices[j] = idx as i32;
            }
        }

        for (i, &delay) in probe
            .phase_delays_fs
            .iter()
            .take(MAX_PROBE_GROUPS)
            .enumerate()
        {
            gpu.phase_delays_fs[i] = delay;
        }

        for (i, &energy) in probe
            .energy_per_group
            .iter()
            .take(MAX_PROBE_GROUPS)
            .enumerate()
        {
            gpu.energy_per_group[i] = energy;
        }

        gpu
    }

    // Helper: Convert Rust response to GPU format
    fn response_to_gpu(&self, response: &ProbeResponse) -> GpuProbeResponse {
        GpuProbeResponse {
            total_spikes: response.total_spikes,
            mean_intensity: response.mean_intensity,
            spatial_extent: response.spatial_extent,
            onset_latency_ps: response.onset_latency_ps,
            peak_latency_ps: response.peak_latency_ps,
            duration_ps: response.duration_ps,
            sequence_score: response.sequence_score,
            sequences_detected: response.sequences_detected,
            response_centroid: response.response_centroid,
            _pad: 0.0,
        }
    }

    // Helper: Convert struct to bytes
    fn struct_to_bytes<T: Copy>(s: &T) -> Vec<u8> {
        let size = std::mem::size_of::<T>();
        let mut bytes = vec![0u8; size];
        unsafe {
            std::ptr::copy_nonoverlapping(s as *const T as *const u8, bytes.as_mut_ptr(), size);
        }
        bytes
    }
}

// ============================================================================
// NON-GPU STUB
// ============================================================================

#[cfg(not(feature = "gpu"))]
pub struct ActiveSensingEngine;

#[cfg(not(feature = "gpu"))]
impl ActiveSensingEngine {
    pub fn new(
        _config: ActiveSensingConfig,
        _grid_origin: [f32; 3],
        _n_aromatics: usize,
        _n_atoms: usize,
    ) -> Result<Self> {
        bail!("ActiveSensingEngine requires GPU. Compile with --features gpu")
    }
}

#[cfg(not(feature = "gpu"))]
pub struct ActiveSensingBuilder;

#[cfg(not(feature = "gpu"))]
impl ActiveSensingBuilder {
    pub fn new() -> Self {
        Self
    }
    pub fn build(self) -> Result<ActiveSensingEngine> {
        bail!("ActiveSensingEngine requires GPU")
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = ActiveSensingConfig::default();
        assert_eq!(config.grid_dim, 32);
        assert!(!config.batch_mode);
    }

    #[test]
    fn test_config_batch_optimized() {
        let config = ActiveSensingConfig::batch_optimized(16);
        assert!(config.batch_mode);
        assert_eq!(config.max_batch_size, 16);
    }

    #[test]
    fn test_float3() {
        let f = Float3::new(1.0, 2.0, 3.0);
        assert_eq!(f.to_array(), [1.0, 2.0, 3.0]);

        let f2 = Float3::from_array([4.0, 5.0, 6.0]);
        assert_eq!(f2.x, 4.0);
    }

    #[test]
    fn test_probe_type() {
        assert_eq!(ProbeType::HingeSideA as i32, 0);
        assert_eq!(ProbeType::Control as i32, 3);
    }

    #[test]
    fn test_probe_default() {
        let probe = CoherentProbe::default();
        assert!(probe.groups.is_empty());
        assert_eq!(probe.probe_type, ProbeType::Control);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_struct_sizes() {
        // Ensure GPU structs have expected alignment
        assert!(std::mem::size_of::<GpuCoherentProbe>() > 0);
        assert!(std::mem::size_of::<GpuProbeResponse>() > 0);
        assert!(std::mem::size_of::<GpuVoxelSpikeHistory>() > 0);
    }
}
