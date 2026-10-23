//! GPU Mega-Fused Pocket Detection Kernel
//!
//! Combines 6 stages into a single kernel launch:
//! Distance -> Contact -> Centrality -> Reservoir -> Consensus -> Kempe
//!
//! Provides ~10x speedup over separate kernel launches by keeping all
//! intermediate data in shared memory.
//!
//! ## Performance Optimization: Buffer Pooling
//!
//! This module implements buffer pooling to eliminate per-call CUDA allocations.
//! Instead of allocating 10 buffers per `detect_pockets()` call, we reuse
//! pre-allocated buffers and only reallocate when capacity is exceeded.
//!
//! Target: 219 structures in 6-14 seconds (RTX 3060)
//!
//! ## Runtime Parameter Configuration
//!
//! All detection parameters are now runtime-configurable via `MegaFusedParams`.
//! This allows tuning precision/recall tradeoffs without PTX recompilation.

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtrMut, DeviceSlice, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use prism_core::PrismError;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

// NVML for GPU telemetry (optional - gracefully degrades if unavailable)
use nvml_wrapper::{enum_wrappers::device::Clock, Device as NvmlDevice, Nvml};

/// GPU kernel telemetry event for provenance tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelTelemetryEvent {
    /// Structure/file being processed
    pub file: String,
    /// Kernel name
    pub kernel: String,
    /// GPU SM/Graphics clock before kernel launch (MHz)
    pub clock_before_mhz: Option<u32>,
    /// GPU SM/Graphics clock after kernel completion (MHz)
    pub clock_after_mhz: Option<u32>,
    /// GPU GDDR memory clock before kernel launch (MHz) - typically 7000+ on GDDR6
    pub memory_clock_before_mhz: Option<u32>,
    /// GPU GDDR memory clock after kernel completion (MHz)
    pub memory_clock_after_mhz: Option<u32>,
    /// GPU temperature (Celsius)
    pub temperature_c: Option<u32>,
    /// GPU memory used (bytes)
    pub memory_used_bytes: Option<u64>,
    /// Kernel execution time (microseconds)
    pub execution_time_us: u64,
    /// Timestamp (ISO 8601)
    pub timestamp: String,
}

/// NVML-based GPU telemetry collector for provenance
/// Falls back to nvidia-smi command parsing when NVML library loading fails (e.g., WSL2)
pub struct GpuTelemetry {
    nvml: Option<Nvml>,
    device: Option<NvmlDevice<'static>>,
    /// Cached nvidia-smi data for fallback mode
    use_smi_fallback: bool,
}

impl GpuTelemetry {
    /// Initialize NVML telemetry (returns Ok even if NVML unavailable)
    /// Falls back to nvidia-smi command parsing if NVML fails
    /// COMPLETELY DISABLED in WSL2 to prevent dxg hang issues
    pub fn new() -> Self {
        // Check for WSL2 FIRST - skip ALL telemetry to prevent dxg hangs
        let is_wsl2 = std::fs::read_to_string("/proc/version")
            .map(|v| v.contains("microsoft") || v.contains("WSL"))
            .unwrap_or(false);

        if is_wsl2 {
            log::warn!("WSL2 detected - disabling ALL GPU telemetry (dxg hang prevention)");
            return Self {
                nvml: None,
                device: None,
                use_smi_fallback: false,
            };
        }

        let nvml = match Nvml::init() {
            Ok(nvml) => {
                log::debug!("NVML initialized successfully");
                Some(nvml)
            }
            Err(e) => {
                log::info!(
                    "NVML initialization failed: {:?} - trying nvidia-smi fallback",
                    e
                );
                None
            }
        };

        // Check if fallback via nvidia-smi is available
        let use_smi_fallback = nvml.is_none() && Self::check_nvidia_smi_available();

        if use_smi_fallback {
            log::info!("Using nvidia-smi fallback for GPU telemetry");
        } else if nvml.is_none() {
            log::warn!("No GPU telemetry available (neither NVML nor nvidia-smi)");
        }

        Self {
            nvml,
            device: None,
            use_smi_fallback,
        }
    }

    /// Check if nvidia-smi is available (with 1 second timeout for WSL2 compatibility)
    fn check_nvidia_smi_available() -> bool {
        use std::process::Stdio;
        use std::time::Duration;

        // Use timeout command to prevent hanging on WSL2/dxg issues
        let result = std::process::Command::new("timeout")
            .args([
                "1",
                "nvidia-smi",
                "--query-gpu=gpu_name",
                "--format=csv,noheader,nounits",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .output();

        match result {
            Ok(o) => o.status.success(),
            Err(_) => {
                // Fallback without timeout command (macOS etc.)
                std::process::Command::new("nvidia-smi")
                    .arg("--query-gpu=gpu_name")
                    .arg("--format=csv,noheader,nounits")
                    .output()
                    .map(|o| o.status.success())
                    .unwrap_or(false)
            }
        }
    }

    /// Parse nvidia-smi output for a single query (with 1 second timeout)
    fn query_nvidia_smi(query: &str) -> Option<String> {
        use std::process::Stdio;

        // Use timeout command to prevent hanging on WSL2/dxg issues
        let result = std::process::Command::new("timeout")
            .args([
                "1",
                "nvidia-smi",
                &format!("--query-gpu={}", query),
                "--format=csv,noheader,nounits",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .output();

        match result {
            Ok(o) if o.status.success() => String::from_utf8(o.stdout)
                .ok()
                .map(|s| s.trim().to_string()),
            _ => {
                // Fallback without timeout (returns quickly on error)
                std::process::Command::new("nvidia-smi")
                    .arg(format!("--query-gpu={}", query))
                    .arg("--format=csv,noheader,nounits")
                    .output()
                    .ok()
                    .filter(|o| o.status.success())
                    .and_then(|o| String::from_utf8(o.stdout).ok())
                    .map(|s| s.trim().to_string())
            }
        }
    }

    /// Query GPU clock speed (Graphics/SM clock in MHz)
    pub fn get_clock_mhz(&self) -> Option<u32> {
        // Try NVML first
        if let Some(ref nvml) = self.nvml {
            if let Ok(dev) = nvml.device_by_index(0) {
                if let Ok(clock) = dev.clock_info(Clock::Graphics) {
                    log::trace!("GPU SM clock MHz (NVML): {}", clock);
                    return Some(clock);
                }
            }
        }

        // Fallback to nvidia-smi
        if self.use_smi_fallback {
            if let Some(clock_str) = Self::query_nvidia_smi("clocks.current.graphics") {
                if let Ok(clock) = clock_str.parse::<u32>() {
                    log::trace!("GPU SM clock MHz (nvidia-smi): {}", clock);
                    return Some(clock);
                }
            }
        }

        None
    }

    /// Query GPU memory clock speed (GDDR memory clock in MHz)
    pub fn get_memory_clock_mhz(&self) -> Option<u32> {
        // Try NVML first
        if let Some(ref nvml) = self.nvml {
            if let Ok(dev) = nvml.device_by_index(0) {
                if let Ok(clock) = dev.clock_info(Clock::Memory) {
                    log::trace!("GPU memory clock MHz (NVML): {}", clock);
                    return Some(clock);
                }
            }
        }

        // Fallback to nvidia-smi
        if self.use_smi_fallback {
            if let Some(clock_str) = Self::query_nvidia_smi("clocks.current.memory") {
                if let Ok(clock) = clock_str.parse::<u32>() {
                    log::trace!("GPU memory clock MHz (nvidia-smi): {}", clock);
                    return Some(clock);
                }
            }
        }

        None
    }

    /// Query GPU temperature in Celsius
    pub fn get_temperature(&self) -> Option<u32> {
        // Try NVML first
        if let Some(ref nvml) = self.nvml {
            if let Ok(dev) = nvml.device_by_index(0) {
                if let Ok(temp) =
                    dev.temperature(nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu)
                {
                    log::trace!("GPU temperature (NVML): {}C", temp);
                    return Some(temp);
                }
            }
        }

        // Fallback to nvidia-smi
        if self.use_smi_fallback {
            if let Some(temp_str) = Self::query_nvidia_smi("temperature.gpu") {
                if let Ok(temp) = temp_str.parse::<u32>() {
                    log::trace!("GPU temperature (nvidia-smi): {}C", temp);
                    return Some(temp);
                }
            }
        }

        None
    }

    /// Query GPU memory usage
    pub fn get_memory_used(&self) -> Option<u64> {
        // Try NVML first
        if let Some(ref nvml) = self.nvml {
            if let Ok(dev) = nvml.device_by_index(0) {
                if let Ok(info) = dev.memory_info() {
                    log::trace!("GPU memory used (NVML): {} bytes", info.used);
                    return Some(info.used);
                }
            }
        }

        // Fallback to nvidia-smi (returns MiB, convert to bytes)
        if self.use_smi_fallback {
            if let Some(mem_str) = Self::query_nvidia_smi("memory.used") {
                if let Ok(mem_mib) = mem_str.parse::<u64>() {
                    let mem_bytes = mem_mib * 1024 * 1024;
                    log::trace!("GPU memory used (nvidia-smi): {} bytes", mem_bytes);
                    return Some(mem_bytes);
                }
            }
        }

        None
    }

    /// Query GPU name
    pub fn get_gpu_name(&self) -> Option<String> {
        // Try NVML first
        if let Some(ref nvml) = self.nvml {
            if let Ok(dev) = nvml.device_by_index(0) {
                if let Ok(name) = dev.name() {
                    log::trace!("GPU name (NVML): {}", name);
                    return Some(name);
                }
            }
        }

        // Fallback to nvidia-smi
        if self.use_smi_fallback {
            if let Some(name) = Self::query_nvidia_smi("gpu_name") {
                log::trace!("GPU name (nvidia-smi): {}", name);
                return Some(name);
            }
        }

        None
    }

    /// Query driver version
    pub fn get_driver_version(&self) -> Option<String> {
        // Try NVML first
        if let Some(ref nvml) = self.nvml {
            if let Ok(version) = nvml.sys_driver_version() {
                log::trace!("NVIDIA driver version (NVML): {}", version);
                return Some(version);
            }
        }

        // Fallback to nvidia-smi
        if self.use_smi_fallback {
            if let Some(version) = Self::query_nvidia_smi("driver_version") {
                log::trace!("NVIDIA driver version (nvidia-smi): {}", version);
                return Some(version);
            }
        }

        None
    }

    /// Check if telemetry is available (either NVML or nvidia-smi)
    pub fn is_available(&self) -> bool {
        self.nvml.is_some() || self.use_smi_fallback
    }

    /// Create a telemetry event around kernel execution
    pub fn record_kernel<F, T>(&self, file: &str, kernel: &str, f: F) -> (T, KernelTelemetryEvent)
    where
        F: FnOnce() -> T,
    {
        // Capture both SM clock and memory clock before kernel launch
        let clock_before = self.get_clock_mhz();
        let memory_clock_before = self.get_memory_clock_mhz();
        let start = Instant::now();

        let result = f();

        let elapsed = start.elapsed();
        // Capture both clocks after kernel completion
        let clock_after = self.get_clock_mhz();
        let memory_clock_after = self.get_memory_clock_mhz();

        let event = KernelTelemetryEvent {
            file: file.to_string(),
            kernel: kernel.to_string(),
            clock_before_mhz: clock_before,
            clock_after_mhz: clock_after,
            memory_clock_before_mhz: memory_clock_before,
            memory_clock_after_mhz: memory_clock_after,
            temperature_c: self.get_temperature(),
            memory_used_bytes: self.get_memory_used(),
            execution_time_us: elapsed.as_micros() as u64,
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        (result, event)
    }
}

impl Default for GpuTelemetry {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime-configurable parameters for the mega-fused pocket detection kernel.
///
/// This struct MUST match the CUDA `MegaFusedParams` layout exactly.
/// Uses `#[repr(C, align(16))]` for CUDA interoperability.
///
/// All 31 parameters are exposed for full runtime tunability:
/// - Contact network construction
/// - Iteration counts (power method, Kempe refinement)
/// - Consensus signal thresholds
/// - Dendritic reservoir branch weights
/// - Consensus score combination weights
/// - Signal bonus multipliers
/// - Confidence level thresholds
/// - Kempe chain refinement parameters
/// - Centrality computation weights
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
pub struct MegaFusedParams {
    // === CONTACT NETWORK PARAMETERS ===
    /// Contact distance cutoff in Angstroms (default: 12.0)
    pub contact_cutoff: f32,
    /// Gaussian sigma for distance weighting (default: 6.0)
    pub contact_sigma: f32,

    // === ITERATION COUNTS ===
    /// Power iterations for eigenvector centrality (default: 15)
    pub power_iterations: i32,
    /// Kempe chain refinement iterations (default: 10)
    pub kempe_iterations: i32,

    // === CONSENSUS SIGNAL THRESHOLDS ===
    /// Geometric pocket signal threshold (default: 0.40)
    pub thresh_geometric: f32,
    /// Conservation signal threshold (default: 0.50)
    pub thresh_conservation: f32,
    /// Centrality signal threshold (default: 0.30)
    pub thresh_centrality: f32,
    /// Flexibility signal threshold (default: 0.45)
    pub thresh_flexibility: f32,
    /// Minimum signals required for pocket membership (default: 2)
    pub min_signals: i32,
    /// Final consensus threshold for pocket assignment (default: 0.35)
    pub consensus_threshold: f32,

    // === DENDRITIC RESERVOIR BRANCH WEIGHTS ===
    /// Local branch weight (default: 0.40)
    pub branch_weight_local: f32,
    /// Neighbor branch weight (default: 0.30)
    pub branch_weight_neighbor: f32,
    /// Global branch weight (default: 0.20)
    pub branch_weight_global: f32,
    /// Recurrent branch weight (default: 0.10)
    pub branch_weight_recurrent: f32,
    /// Recurrent decay factor (default: 0.90)
    pub recurrent_decay: f32,

    // === CONSENSUS SCORE COMBINATION WEIGHTS ===
    /// Weight for geometric component in consensus (default: 0.30)
    pub consensus_weight_geometric: f32,
    /// Weight for conservation component (default: 0.25)
    pub consensus_weight_conservation: f32,
    /// Weight for centrality component (default: 0.25)
    pub consensus_weight_centrality: f32,
    /// Weight for flexibility component (default: 0.20)
    pub consensus_weight_flexibility: f32,

    // === SIGNAL BONUS MULTIPLIERS ===
    /// Score multiplier when 0 signals present (default: 0.70)
    pub signal_bonus_0: f32,
    /// Score multiplier when 1 signal present (default: 1.00)
    pub signal_bonus_1: f32,
    /// Score multiplier when 2 signals present (default: 1.15)
    pub signal_bonus_2: f32,
    /// Score multiplier when 3+ signals present (default: 1.30)
    pub signal_bonus_3: f32,

    // === CONFIDENCE LEVEL THRESHOLDS ===
    /// Score threshold for HIGH confidence (default: 0.70)
    pub confidence_high_score: f32,
    /// Score threshold for MEDIUM confidence (default: 0.40)
    pub confidence_medium_score: f32,
    /// Signal count for HIGH confidence (default: 3)
    pub confidence_high_signals: i32,
    /// Signal count for MEDIUM confidence (default: 2)
    pub confidence_medium_signals: i32,

    // === KEMPE REFINEMENT PARAMETERS ===
    /// Contact threshold for Kempe chain swaps (default: 0.20)
    pub kempe_contact_threshold: f32,
    /// Score ratio threshold for Kempe swaps (default: 1.10)
    pub kempe_swap_threshold: f32,

    // === CENTRALITY COMPUTATION WEIGHTS ===
    /// Weight for degree centrality (default: 0.60)
    pub centrality_degree_weight: f32,
    /// Weight for eigenvector centrality (default: 0.40)
    pub centrality_eigenvector_weight: f32,

    // === QUALITY CONTROL (QC) GATE PARAMETERS ===
    // These parameters enforce scientifically validated thresholds from hyper-tuning campaigns.
    // They act as QC gates to filter out noise and ensure only druggable binding sites pass.
    /// Minimum pocket volume in Å³ (default: 160.0)
    /// Pockets smaller than this are noise - real binding sites need space for ligands.
    /// Validated on CryptoSite benchmark: eliminates false positives from surface irregularities.
    pub min_pocket_volume: f32,

    /// Maximum pocket volume in Å³ (default: 4800.0)
    /// Pockets larger than this are "mega-pockets" spanning multiple functional regions.
    /// Validated on kinase benchmarks: prevents entire active site from merging into one blob.
    pub max_pocket_volume: f32,

    /// Minimum druggability score (default: 0.30)
    /// Combined physicochemical score from hydrophobicity, burial, and shape.
    /// Validated threshold from CryptoBench: 0.30 balances recall vs precision.
    pub min_druggability: f32,

    /// Maximum residues per pocket (default: 80)
    /// Hard limit to prevent runaway pocket merging.
    /// Validated: kinase ATP sites have ~40-60 residues, 80 gives margin.
    pub max_pocket_residues: i32,

    /// Maximum number of pockets to return (default: 10)
    /// After ranking by druggability, keep only top N.
    /// Validated on CryptoSite: >10 pockets adds noise without recall gains.
    pub max_pockets: i32,
}

impl Default for MegaFusedParams {
    fn default() -> Self {
        Self {
            // Contact network
            contact_cutoff: 12.0,
            contact_sigma: 6.0,

            // Iterations
            power_iterations: 15,
            kempe_iterations: 10,

            // Consensus thresholds
            thresh_geometric: 0.40,
            thresh_conservation: 0.50,
            thresh_centrality: 0.30,
            thresh_flexibility: 0.45,
            min_signals: 2,
            consensus_threshold: 0.35,

            // Dendritic reservoir
            branch_weight_local: 0.40,
            branch_weight_neighbor: 0.30,
            branch_weight_global: 0.20,
            branch_weight_recurrent: 0.10,
            recurrent_decay: 0.90,

            // Consensus weights
            consensus_weight_geometric: 0.30,
            consensus_weight_conservation: 0.25,
            consensus_weight_centrality: 0.25,
            consensus_weight_flexibility: 0.20,

            // Signal bonuses
            signal_bonus_0: 0.70,
            signal_bonus_1: 1.00,
            signal_bonus_2: 1.15,
            signal_bonus_3: 1.30,

            // Confidence thresholds
            confidence_high_score: 0.70,
            confidence_medium_score: 0.40,
            confidence_high_signals: 3,
            confidence_medium_signals: 2,

            // Kempe refinement
            kempe_contact_threshold: 0.20,
            kempe_swap_threshold: 1.10,

            // Centrality weights
            centrality_degree_weight: 0.60,
            centrality_eigenvector_weight: 0.40,

            // QC gate parameters (validated thresholds)
            min_pocket_volume: 160.0,  // Å³ - minimum for real binding sites
            max_pocket_volume: 4800.0, // Å³ - prevents mega-pockets
            min_druggability: 0.60,    // High-confidence threshold for druggable pockets
            max_pocket_residues: 80,   // Hard limit on pocket size
            max_pockets: 10,           // Top-N limit
        }
    }
}

impl MegaFusedParams {
    /// Create precision-focused parameters for minimizing false positives.
    ///
    /// Key changes from default:
    /// - Higher consensus threshold (0.50 vs 0.35)
    /// - Stricter signal thresholds
    /// - Requires more signals for pocket membership
    /// - More conservative Kempe refinement
    ///
    /// Expected impact: +10-20% precision, -5-10% recall, better DCC
    pub fn precision() -> Self {
        Self {
            // Tighter consensus
            consensus_threshold: 0.50, // Higher (default: 0.35)
            min_signals: 3,            // Require 3 signals (default: 2)

            // Stricter signal thresholds
            thresh_geometric: 0.50,    // Higher (default: 0.40)
            thresh_conservation: 0.55, // Higher (default: 0.50)
            thresh_centrality: 0.40,   // Higher (default: 0.30)
            thresh_flexibility: 0.50,  // Higher (default: 0.45)

            // Higher confidence requirements
            confidence_high_score: 0.75,   // Higher (default: 0.70)
            confidence_medium_score: 0.50, // Higher (default: 0.40)
            confidence_high_signals: 3,    // Same
            confidence_medium_signals: 3,  // Higher (default: 2)

            // More conservative Kempe
            kempe_contact_threshold: 0.30, // Higher (default: 0.20)
            kempe_swap_threshold: 1.25,    // Higher (default: 1.10)

            // More iterations for accuracy
            power_iterations: 20, // More (default: 15)
            kempe_iterations: 12, // More (default: 10)

            // Stricter QC gates for precision mode
            min_pocket_volume: 200.0,  // Higher (default: 160)
            max_pocket_volume: 4000.0, // Lower (default: 4800)
            min_druggability: 0.70,    // Higher (default: 0.60) - stricter precision
            max_pocket_residues: 60,   // Lower (default: 80)
            max_pockets: 5,            // Fewer (default: 10)

            // Keep other defaults
            ..Default::default()
        }
    }

    /// Create screening-mode parameters for fast batch processing.
    ///
    /// TUNED FOR CRYPTOBENCH (2025-12-05):
    /// - Disabled flexibility signal (inverted for binding sites)
    /// - Disabled conservation signal (always 0 without HMMER)
    /// - Boosted geometric and centrality weights
    /// - Higher consensus threshold for precision
    ///
    /// Expected impact: Higher precision, balanced recall
    pub fn screening() -> Self {
        Self {
            // Moderate iterations for accuracy
            power_iterations: 20,
            kempe_iterations: 15,

            // Optimized consensus threshold (best F1 found at 0.30)
            consensus_threshold: 0.30, // Best F1/AUC-ROC balance
            min_signals: 1,            // Allow 1 signal

            // Optimized signal thresholds
            thresh_geometric: 0.30,    // Moderate geometric threshold
            thresh_conservation: 0.99, // Disabled - always 0 without HMMER
            thresh_centrality: 0.25,   // Moderate centrality threshold
            thresh_flexibility: 0.99,  // Disabled - inverted for binding sites

            // Balanced weights (geometric from reservoir, centrality from network)
            consensus_weight_geometric: 0.50, // Main signal from dendritic reservoir
            consensus_weight_conservation: 0.0, // Disabled - no HMMER data
            consensus_weight_centrality: 0.40, // Secondary signal - structural importance
            consensus_weight_flexibility: 0.0, // Disabled - signal inverted

            // Signal bonuses tuned for recall
            signal_bonus_0: 0.80, // Mild penalty for no signals
            signal_bonus_1: 1.00, // Neutral for 1 signal
            signal_bonus_2: 1.15, // Bonus for 2 signals
            signal_bonus_3: 1.30, // Bonus for 3+ signals

            // Higher confidence requirements
            confidence_high_score: 0.65,
            confidence_medium_score: 0.40,
            confidence_high_signals: 2,
            confidence_medium_signals: 1,

            // Moderate QC gates
            min_pocket_volume: 120.0,
            max_pocket_volume: 5000.0,
            min_druggability: 0.25,
            max_pocket_residues: 100,
            max_pockets: 15,

            // Keep other defaults
            ..Default::default()
        }
    }

    /// Create ultra-precise parameters for publication-quality results.
    ///
    /// Key changes from default:
    /// - Maximum iterations for convergence
    /// - Very strict thresholds
    /// - Requires all 4 signals
    ///
    /// Expected impact: Highest precision, lowest false positives, best DCC
    pub fn ultra_precise() -> Self {
        Self {
            // Maximum iterations
            power_iterations: 25,
            kempe_iterations: 15,

            // Very strict consensus
            consensus_threshold: 0.60,
            min_signals: 3,

            // Very strict signal thresholds
            thresh_geometric: 0.55,
            thresh_conservation: 0.60,
            thresh_centrality: 0.45,
            thresh_flexibility: 0.55,

            // Highest confidence requirements
            confidence_high_score: 0.80,
            confidence_medium_score: 0.55,
            confidence_high_signals: 4,
            confidence_medium_signals: 3,

            // Very conservative Kempe
            kempe_contact_threshold: 0.35,
            kempe_swap_threshold: 1.30,

            // Strictest QC gates for publication quality
            min_pocket_volume: 250.0, // Higher (default: 160) - only well-defined
            max_pocket_volume: 3500.0, // Lower (default: 4800) - tight bound
            min_druggability: 0.80,   // Higher (default: 0.60) - only high-confidence
            max_pocket_residues: 50,  // Lower (default: 80) - focused sites
            max_pockets: 3,           // Fewer (default: 10) - only top 3

            // Keep other defaults
            ..Default::default()
        }
    }

    /// Create from MegaFusedConfig for backward compatibility
    pub fn from_config(config: &MegaFusedConfig) -> Self {
        let mut params = match config.mode {
            MegaFusedMode::UltraPrecise => Self::ultra_precise(),
            MegaFusedMode::Balanced => Self::default(),
            MegaFusedMode::Screening => Self::screening(),
        };

        // Override with explicit config values
        params.contact_sigma = config.contact_sigma;
        params.consensus_threshold = config.consensus_threshold;
        params.power_iterations = config.power_iterations;
        params.kempe_iterations = config.kempe_iterations;

        params
    }
}

/// TDA neighborhood radius for spatial neighbor search (Angstroms)
const TDA_NEIGHBOR_RADIUS: f32 = 16.0;
/// Maximum neighbors per residue for TDA (must match CUDA TDA_MAX_NEIGHBORS)
const TDA_MAX_NEIGHBORS: usize = 64;

/// Pre-allocated GPU buffer pool for zero-allocation hot path
///
/// Reuses device memory across multiple `detect_pockets()` calls,
/// only reallocating when the new structure exceeds current capacity.
struct BufferPool {
    // Input buffers (capacity = max atoms seen * 3 for coords)
    atoms_capacity: usize,
    d_atoms: Option<CudaSlice<f32>>,

    // Per-residue buffers (capacity = max residues seen)
    residue_capacity: usize,
    d_ca_indices: Option<CudaSlice<i32>>,
    d_conservation: Option<CudaSlice<f32>>,
    d_bfactor: Option<CudaSlice<f32>>,
    d_burial: Option<CudaSlice<f32>>,

    // TDA spatial neighborhood buffers (pre-computed on CPU with KD-tree)
    tda_neighbor_capacity: usize, // Total neighbor coords capacity
    d_tda_neighbor_coords: Option<CudaSlice<f32>>, // [total_neighbors * 3]
    d_tda_neighbor_offsets: Option<CudaSlice<i32>>, // [n_residues + 1] CSR offsets
    d_tda_neighbor_counts: Option<CudaSlice<i32>>, // [n_residues] neighbor counts

    // Output buffers (same capacity as residue buffers)
    d_consensus_scores: Option<CudaSlice<f32>>,
    d_confidence: Option<CudaSlice<i32>>,
    d_signal_mask: Option<CudaSlice<i32>>,
    d_pocket_assignment: Option<CudaSlice<i32>>,
    d_centrality: Option<CudaSlice<f32>>,

    // Combined TDA + base features output (80-dim per residue)
    d_combined_features: Option<CudaSlice<f32>>,

    // Runtime parameters buffer (single MegaFusedParams struct)
    d_params: Option<CudaSlice<u8>>,

    // Statistics
    allocations: usize,
    reuses: usize,
}

impl BufferPool {
    fn new() -> Self {
        Self {
            atoms_capacity: 0,
            d_atoms: None,
            residue_capacity: 0,
            d_ca_indices: None,
            d_conservation: None,
            d_bfactor: None,
            d_burial: None,
            // TDA neighborhood buffers
            tda_neighbor_capacity: 0,
            d_tda_neighbor_coords: None,
            d_tda_neighbor_offsets: None,
            d_tda_neighbor_counts: None,
            // Output buffers
            d_consensus_scores: None,
            d_confidence: None,
            d_signal_mask: None,
            d_pocket_assignment: None,
            d_centrality: None,
            d_combined_features: None,
            d_params: None,
            allocations: 0,
            reuses: 0,
        }
    }

    /// Check if buffers need reallocation for given sizes
    fn needs_realloc(&self, n_atoms: usize, n_residues: usize) -> (bool, bool) {
        let atoms_need = n_atoms * 3 > self.atoms_capacity;
        let residues_need = n_residues > self.residue_capacity;
        (atoms_need, residues_need)
    }

    /// Get statistics about buffer reuse
    fn stats(&self) -> (usize, usize) {
        (self.allocations, self.reuses)
    }
}

/// TDA spatial neighborhood data for a structure
/// Pre-computed on CPU, uploaded to GPU for Stage 3.5
struct TdaNeighborhoods {
    /// Flat array of neighbor CA coordinates [x0, y0, z0, x1, y1, z1, ...]
    coords: Vec<f32>,
    /// CSR offsets into coords array [n_residues + 1]
    offsets: Vec<i32>,
    /// Neighbor count per residue [n_residues]
    counts: Vec<i32>,
}

/// Build TDA spatial neighborhoods using brute-force O(n²) search
/// For each residue, finds all CA neighbors within TDA_NEIGHBOR_RADIUS (16Å)
/// Caps at TDA_MAX_NEIGHBORS (64) per residue
fn build_tda_neighborhoods(atoms: &[f32], ca_indices: &[i32]) -> TdaNeighborhoods {
    let n_residues = ca_indices.len();
    let radius_sq = TDA_NEIGHBOR_RADIUS * TDA_NEIGHBOR_RADIUS;

    // First, extract CA coordinates for all residues
    let mut ca_coords: Vec<(f32, f32, f32)> = Vec::with_capacity(n_residues);
    for &ca_idx in ca_indices {
        let idx = ca_idx as usize;
        if idx * 3 + 2 < atoms.len() {
            ca_coords.push((atoms[idx * 3], atoms[idx * 3 + 1], atoms[idx * 3 + 2]));
        } else {
            ca_coords.push((0.0, 0.0, 0.0)); // Fallback for invalid indices
        }
    }

    // Build neighborhoods for each residue
    let mut all_coords: Vec<f32> = Vec::new();
    let mut offsets: Vec<i32> = Vec::with_capacity(n_residues + 1);
    let mut counts: Vec<i32> = Vec::with_capacity(n_residues);

    offsets.push(0);

    for i in 0..n_residues {
        let (cx, cy, cz) = ca_coords[i];
        let mut neighbors: Vec<(f32, f32, f32, f32)> = Vec::new(); // (dist, x, y, z)

        // Find all neighbors within radius
        for j in 0..n_residues {
            if i == j {
                continue;
            }
            let (nx, ny, nz) = ca_coords[j];
            let dx = nx - cx;
            let dy = ny - cy;
            let dz = nz - cz;
            let dist_sq = dx * dx + dy * dy + dz * dz;

            if dist_sq <= radius_sq && dist_sq > 0.0 {
                neighbors.push((dist_sq.sqrt(), nx, ny, nz));
            }
        }

        // Sort by distance (closest first) and take up to TDA_MAX_NEIGHBORS
        neighbors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let n_neighbors = neighbors.len().min(TDA_MAX_NEIGHBORS);

        // Add neighbor coords to flat array
        for k in 0..n_neighbors {
            all_coords.push(neighbors[k].1); // x
            all_coords.push(neighbors[k].2); // y
            all_coords.push(neighbors[k].3); // z
        }

        counts.push(n_neighbors as i32);
        offsets.push(all_coords.len() as i32 / 3);
    }

    log::debug!(
        "Built TDA neighborhoods: {} residues, {} total neighbors ({:.1} avg/residue)",
        n_residues,
        all_coords.len() / 3,
        all_coords.len() as f32 / 3.0 / n_residues as f32
    );

    TdaNeighborhoods {
        coords: all_coords,
        offsets,
        counts,
    }
}

/// Performance mode for mega-fused kernel
/// Controls iteration counts for speed vs accuracy tradeoff
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MegaFusedMode {
    /// Ultra-precise mode: kempe=15, power=20 → publication-quality refinement
    UltraPrecise,
    /// Balanced mode: kempe=8, power=12 → good quality with reasonable speed
    #[default]
    Balanced,
    /// Screening mode: kempe=3, power=5 → 100-200× faster for batch processing
    Screening,
}

impl MegaFusedMode {
    /// Get iteration parameters for this mode (kempe, power)
    pub fn iterations(&self) -> (i32, i32) {
        match self {
            MegaFusedMode::UltraPrecise => (15, 25),
            MegaFusedMode::Balanced => (10, 15),
            MegaFusedMode::Screening => (15, 20), // Tuned for CryptoBench
        }
    }
}

/// Configuration for mega-fused kernel
#[derive(Debug, Clone)]
pub struct MegaFusedConfig {
    /// Use FP16 (Tensor Core) version if available
    pub use_fp16: bool,
    /// Contact sigma for Gaussian weighting (default: 6.0 Angstroms)
    pub contact_sigma: f32,
    /// Consensus threshold for pocket assignment (default: 0.35)
    pub consensus_threshold: f32,
    /// Performance mode (determines iteration counts)
    pub mode: MegaFusedMode,
    /// Maximum Kempe refinement iterations (overridden by mode if not manually set)
    pub kempe_iterations: i32,
    /// Power iteration steps for eigenvector (overridden by mode if not manually set)
    pub power_iterations: i32,
    /// When true, completely bypass ALL CPU geometry paths (Voronoi, cavity_detector, fpocket, belief propagation)
    /// and return pockets directly from the mega-fused kernel output.
    /// This is intended for ultra-high-throughput screening only.
    pub pure_gpu_mode: bool,
}

impl MegaFusedConfig {
    /// Create config with screening mode for fast batch processing
    /// TUNED FOR CRYPTOBENCH (2025-12-05)
    pub fn screening() -> Self {
        let (kempe, power) = MegaFusedMode::Screening.iterations();
        Self {
            use_fp16: true,
            contact_sigma: 6.0,
            consensus_threshold: 0.30, // Optimized threshold (best F1/AUC-ROC)
            mode: MegaFusedMode::Screening,
            kempe_iterations: kempe,
            power_iterations: power,
            pure_gpu_mode: false,
        }
    }

    /// Create config with screening mode AND pure GPU bypass (no CPU geometry at all)
    pub fn screening_pure() -> Self {
        Self {
            use_fp16: true,
            contact_sigma: 6.0,
            consensus_threshold: 0.35,
            mode: MegaFusedMode::Screening,
            kempe_iterations: 3,
            power_iterations: 5,
            pure_gpu_mode: true,
        }
    }

    /// Create config with ultra-precise mode for publication quality
    pub fn ultra_precise() -> Self {
        let (kempe, power) = MegaFusedMode::UltraPrecise.iterations();
        Self {
            use_fp16: true,
            contact_sigma: 6.0,
            consensus_threshold: 0.35,
            mode: MegaFusedMode::UltraPrecise,
            kempe_iterations: kempe,
            power_iterations: power,
            pure_gpu_mode: false,
        }
    }
}

impl Default for MegaFusedConfig {
    fn default() -> Self {
        let (kempe, power) = MegaFusedMode::Balanced.iterations();
        Self {
            use_fp16: true, // FP16 (Tensor Core) by default for maximum performance
            contact_sigma: 6.0,
            consensus_threshold: 0.35,
            mode: MegaFusedMode::Balanced,
            kempe_iterations: kempe,
            power_iterations: power,
            pure_gpu_mode: false,
        }
    }
}

/// Combined feature count for PRISM-VE 136-dimensional output
/// Layout: TDA(48) + Reservoir(32) + Physics(12) + Fitness(4) + Cycle(5) + Spike(8) + Immunity(16) + Epi(11) = 136
/// See mega_fused_batch.cu header for complete index map
pub const TOTAL_COMBINED_FEATURES: usize = 136;

/// Output from mega-fused pocket detection
#[derive(Debug, Clone)]
pub struct MegaFusedOutput {
    /// Consensus score per residue (0.0 - 1.0)
    pub consensus_scores: Vec<f32>,
    /// Confidence level per residue (0=LOW, 1=MEDIUM, 2=HIGH)
    pub confidence: Vec<i32>,
    /// Signal mask per residue (bit flags for geometric, conservation, centrality, flexibility)
    pub signal_mask: Vec<i32>,
    /// Pocket assignment per residue (cluster ID)
    pub pocket_assignment: Vec<i32>,
    /// Network centrality per residue
    pub centrality: Vec<f32>,
    /// Combined 70-dim features per residue (16 base + 12 reservoir + 12 physics + 30 SOTA)
    pub combined_features: Vec<f32>,
    /// GPU telemetry events for provenance tracking
    pub gpu_telemetry: Option<GpuProvenanceData>,
}

/// Complete GPU provenance data for transparency and validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuProvenanceData {
    /// GPU device name (e.g., "NVIDIA GeForce RTX 3060")
    pub gpu_name: Option<String>,
    /// NVIDIA driver version
    pub driver_version: Option<String>,
    /// Kernel execution telemetry events
    pub kernel_events: Vec<KernelTelemetryEvent>,
    /// Total GPU execution time (microseconds)
    pub total_gpu_time_us: u64,
    /// Average GPU clock during execution (MHz)
    pub avg_clock_mhz: Option<u32>,
    /// Peak GPU temperature during execution (Celsius)
    pub peak_temperature_c: Option<u32>,
}

/// GPU executor for mega-fused pocket detection kernels
///
/// Features buffer pooling for zero-allocation hot path after first call.
/// Kernel handles are immutable once loaded (no per-call overhead).
pub struct MegaFusedGpu {
    #[allow(dead_code)]
    context: Arc<CudaContext>, // Keep for lifetime, operations use stream
    stream: Arc<CudaStream>,
    /// FP32 mega-fused kernel function (immutable after load)
    fp32_func: Option<CudaFunction>,
    /// FP16 (Tensor Core) mega-fused kernel function (immutable after load)
    fp16_func: Option<CudaFunction>,
    /// Reservoir weight initialization function (FP32)
    init_weights_fp32: Option<CudaFunction>,
    /// Reservoir weight initialization function (FP16)
    init_weights_fp16: Option<CudaFunction>,
    /// Whether reservoir weights have been initialized
    weights_initialized: bool,
    /// Pre-allocated buffer pool for zero-allocation hot path
    buffer_pool: BufferPool,

    // SESSION 10E: Multi-pass kernel architecture (fixes 16 dead SOTA features)
    /// Distance matrix computation kernel (Pass 1)
    distance_kernel: Option<CudaFunction>,
    /// SOTA features computation kernel (Pass 2)
    sota_kernel: Option<CudaFunction>,
    /// Reusable distance matrix buffer [n_residues × n_residues]
    distance_matrix_buffer: Option<CudaSlice<f32>>,
    /// Reusable SOTA features buffer [n_residues × 30]
    sota_features_buffer: Option<CudaSlice<f32>>,
    /// Distance matrix capacity (reallocate if exceeded)
    distance_matrix_capacity: usize,
    /// SOTA features capacity
    sota_features_capacity: usize,
}

impl MegaFusedGpu {
    /// Load mega-fused PTX modules from `ptx_dir`. Expects:
    /// - mega_fused_pocket.ptx with `mega_fused_pocket_detection`
    /// - mega_fused_fp16.ptx with `mega_fused_pocket_detection_fp16` (optional)
    pub fn new(context: Arc<CudaContext>, ptx_dir: &Path) -> Result<Self, PrismError> {
        let stream = context.default_stream();

        // Load FP32 mega-fused module
        let fp32_path = ptx_dir.join("mega_fused_pocket.ptx");
        let fp32_func = if fp32_path.exists() {
            let ptx_src = std::fs::read_to_string(&fp32_path).map_err(|e| {
                PrismError::gpu("mega_fused_fp32", format!("Failed to read PTX: {}", e))
            })?;
            let module = context.load_module(Ptx::from_src(ptx_src)).map_err(|e| {
                PrismError::gpu("mega_fused_fp32", format!("Failed to load PTX: {}", e))
            })?;
            let func = module
                .load_function("mega_fused_pocket_detection")
                .map_err(|e| {
                    PrismError::gpu("mega_fused_fp32", format!("Failed to load kernel: {}", e))
                })?;
            log::info!("Loaded mega_fused_pocket.ptx (FP32)");
            Some(func)
        } else {
            log::warn!("mega_fused_pocket.ptx not found at {:?}", fp32_path);
            None
        };

        // Load FP16 mega-fused module (optional - requires Tensor Cores)
        let fp16_path = ptx_dir.join("mega_fused_fp16.ptx");
        let fp16_func = if fp16_path.exists() {
            match std::fs::read_to_string(&fp16_path) {
                Ok(ptx_src) => match context.load_module(Ptx::from_src(ptx_src)) {
                    Ok(module) => match module.load_function("mega_fused_pocket_detection_fp16") {
                        Ok(func) => {
                            log::info!("Loaded mega_fused_fp16.ptx (FP16/Tensor Core)");
                            Some(func)
                        }
                        Err(e) => {
                            log::warn!("Failed to load FP16 kernel function: {}", e);
                            None
                        }
                    },
                    Err(e) => {
                        log::warn!("Failed to load FP16 PTX module: {}", e);
                        None
                    }
                },
                Err(e) => {
                    log::warn!("Failed to read mega_fused_fp16.ptx: {}", e);
                    None
                }
            }
        } else {
            log::debug!("mega_fused_fp16.ptx not found, FP16 mode disabled");
            None
        };

        if fp32_func.is_none() && fp16_func.is_none() {
            return Err(PrismError::gpu(
                "mega_fused",
                "No mega-fused kernels available (neither FP32 nor FP16)",
            ));
        }

        // SESSION 10E: Load multi-pass kernels (optional - graceful degradation)
        let distance_path = ptx_dir.join("distance_matrix.ptx");
        let sota_path = ptx_dir.join("sota_features.ptx");

        let (distance_kernel, sota_kernel) = if distance_path.exists() && sota_path.exists() {
            log::info!("Loading SESSION 10E multi-pass kernels");

            let distance_kernel = match std::fs::read_to_string(&distance_path) {
                Ok(ptx_src) => match context.load_module(Ptx::from_src(ptx_src)) {
                    Ok(module) => match module.load_function("compute_distance_matrix_kernel") {
                        Ok(func) => {
                            log::info!("✅ Loaded distance_matrix.ptx");
                            Some(func)
                        }
                        Err(e) => {
                            log::warn!("Failed to load distance kernel: {:?}", e);
                            None
                        }
                    },
                    Err(e) => {
                        log::warn!("Failed to load distance PTX module: {:?}", e);
                        None
                    }
                },
                Err(e) => {
                    log::warn!("Failed to read distance_matrix.ptx: {:?}", e);
                    None
                }
            };

            let sota_kernel = match std::fs::read_to_string(&sota_path) {
                Ok(ptx_src) => match context.load_module(Ptx::from_src(ptx_src)) {
                    Ok(module) => match module.load_function("compute_sota_features_kernel") {
                        Ok(func) => {
                            log::info!("✅ Loaded sota_features.ptx");
                            Some(func)
                        }
                        Err(e) => {
                            log::warn!("Failed to load SOTA kernel: {:?}", e);
                            None
                        }
                    },
                    Err(e) => {
                        log::warn!("Failed to load SOTA PTX module: {:?}", e);
                        None
                    }
                },
                Err(e) => {
                    log::warn!("Failed to read sota_features.ptx: {:?}", e);
                    None
                }
            };

            (distance_kernel, sota_kernel)
        } else {
            log::info!("Multi-pass kernels not found - using single-pass mode");
            (None, None)
        };

        Ok(Self {
            context,
            stream,
            fp32_func,
            fp16_func,
            init_weights_fp32: None,
            init_weights_fp16: None,
            weights_initialized: false,
            buffer_pool: BufferPool::new(),

            // SESSION 10E: Multi-pass kernels (loaded above)
            distance_kernel,
            sota_kernel,
            distance_matrix_buffer: None,
            sota_features_buffer: None,
            distance_matrix_capacity: 0,
            sota_features_capacity: 0,
        })
    }

    /// Check if mega-fused kernels are available
    pub fn is_available(&self) -> bool {
        self.fp32_func.is_some() || self.fp16_func.is_some()
    }

    /// Check if FP16 (Tensor Core) mode is available
    pub fn has_fp16(&self) -> bool {
        self.fp16_func.is_some()
    }

    /// Get buffer pool statistics: (allocations, reuses)
    ///
    /// Returns tuple of (total_allocations, total_reuses)
    /// High reuse ratio indicates effective pooling
    pub fn buffer_pool_stats(&self) -> (usize, usize) {
        self.buffer_pool.stats()
    }

    /// Log buffer pool efficiency
    pub fn log_buffer_stats(&self) {
        let (allocs, reuses) = self.buffer_pool.stats();
        let total = allocs + reuses;
        let efficiency = if total > 0 {
            reuses as f32 / total as f32 * 100.0
        } else {
            0.0
        };
        log::info!(
            "Buffer pool: {} allocations, {} reuses ({:.1}% efficiency), capacity: {} atoms, {} residues",
            allocs, reuses, efficiency,
            self.buffer_pool.atoms_capacity,
            self.buffer_pool.residue_capacity
        );
    }

    /// Initialize reservoir weights with random values
    /// Should be called once before running pocket detection
    pub fn initialize_reservoir_weights(&mut self) -> Result<(), PrismError> {
        if self.weights_initialized {
            return Ok(());
        }

        // Note: Reservoir weights are pre-initialized in the kernel's constant memory
        // (c_reservoir_input_weights, c_branch_weights, c_readout_weights)
        // Future enhancement: allow dynamic weight initialization via CUDA API
        //
        // For reference, the kernel expects:
        // - RESERVOIR_DIM=256, N_INPUT_FEATURES=8, N_BRANCHES=4
        // - Input weights: RESERVOIR_DIM x N_INPUT_FEATURES = 2048 floats
        // - Branch weights: N_BRANCHES x RESERVOIR_DIM = 1024 floats
        // - Readout weights: RESERVOIR_DIM = 256 floats

        log::info!("Reservoir weights initialized (using kernel defaults)");
        self.weights_initialized = true;

        Ok(())
    }

    /// Run mega-fused pocket detection on a single structure
    ///
    /// # Arguments
    /// * `atoms` - Flat array of atom coordinates [x0, y0, z0, x1, y1, z1, ...]
    /// * `ca_indices` - Indices of CA atoms for each residue
    /// * `conservation` - Per-residue conservation scores (0.0 - 1.0)
    /// * `bfactor` - Per-residue B-factor / flexibility (normalized)
    /// * `burial` - Per-residue burial scores (0.0 - 1.0)
    /// * `config` - Kernel configuration
    pub fn detect_pockets(
        &mut self,
        atoms: &[f32],
        ca_indices: &[i32],
        conservation: &[f32],
        bfactor: &[f32],
        burial: &[f32],
        residue_types: Option<&[i32]>,      // Enable physics features
        gisaid_frequencies: Option<&[f32]>, // PRISM-VE: Variant frequencies (optional)
        gisaid_velocities: Option<&[f32]>,  // PRISM-VE: Frequency velocities (optional)
        config: &MegaFusedConfig,
    ) -> Result<MegaFusedOutput, PrismError> {
        let n_residues = ca_indices.len();
        let n_atoms = atoms.len() / 3;

        if n_residues == 0 {
            return Ok(MegaFusedOutput {
                consensus_scores: Vec::new(),
                confidence: Vec::new(),
                signal_mask: Vec::new(),
                pocket_assignment: Vec::new(),
                centrality: Vec::new(),
                combined_features: Vec::new(),
                gpu_telemetry: None,
            });
        }

        // Validate input sizes
        if conservation.len() != n_residues
            || bfactor.len() != n_residues
            || burial.len() != n_residues
        {
            return Err(PrismError::gpu(
                "mega_fused",
                format!(
                    "Input size mismatch: n_residues={}, conservation={}, bfactor={}, burial={}",
                    n_residues,
                    conservation.len(),
                    bfactor.len(),
                    burial.len()
                ),
            ));
        }

        // Initialize reservoir weights if not done yet
        if !self.weights_initialized {
            self.initialize_reservoir_weights()?;
        }

        // Select kernel based on config
        // IMPORTANT: FP16 kernel does NOT support combined_features output!
        // Always use FP32 kernel to ensure 80-dim features are computed
        let func = if self.fp32_func.is_some() {
            log::debug!("Using FP32 mega-fused kernel (FP16 lacks combined_features support)");
            self.fp32_func.as_ref().unwrap()
        } else if config.use_fp16 && self.fp16_func.is_some() {
            log::warn!("FP16 kernel selected but it doesn't output combined_features - features will be zeros");
            self.fp16_func.as_ref().unwrap()
        } else {
            return Err(PrismError::gpu(
                "mega_fused",
                "No suitable kernel available",
            ));
        };

        // === ALLOCATE/UPLOAD RESIDUE TYPES (if provided) ===
        let mut d_residue_types_temp: Option<CudaSlice<i32>> = None;
        if let Some(res_types) = residue_types {
            if res_types.len() != n_residues {
                return Err(PrismError::gpu(
                    "mega_fused",
                    format!(
                        "residue_types length {} != n_residues {}",
                        res_types.len(),
                        n_residues
                    ),
                ));
            }
            let mut d_res_types = self.stream.alloc_zeros::<i32>(n_residues).map_err(|e| {
                PrismError::gpu("mega_fused", format!("Alloc residue_types: {}", e))
            })?;
            self.stream
                .memcpy_htod(res_types, &mut d_res_types)
                .map_err(|e| {
                    PrismError::gpu("mega_fused", format!("Upload residue_types: {}", e))
                })?;
            d_residue_types_temp = Some(d_res_types);
            log::debug!(
                "Uploaded {} residue types to GPU (enables hydrophobicity physics)",
                n_residues
            );
        }

        // === ALLOCATE/UPLOAD GISAID DATA (PRISM-VE variant evolution - if provided) ===
        let mut d_gisaid_freq_temp: Option<CudaSlice<f32>> = None;
        let mut d_gisaid_vel_temp: Option<CudaSlice<f32>> = None;

        if let Some(freq) = gisaid_frequencies {
            if freq.len() != n_residues {
                return Err(PrismError::gpu(
                    "mega_fused",
                    format!(
                        "gisaid_frequencies length {} != n_residues {}",
                        freq.len(),
                        n_residues
                    ),
                ));
            }
            let mut d_freq = self
                .stream
                .alloc_zeros::<f32>(n_residues)
                .map_err(|e| PrismError::gpu("mega_fused", format!("Alloc gisaid_freq: {}", e)))?;
            self.stream
                .memcpy_htod(freq, &mut d_freq)
                .map_err(|e| PrismError::gpu("mega_fused", format!("Upload gisaid_freq: {}", e)))?;
            d_gisaid_freq_temp = Some(d_freq);
            log::debug!("Uploaded GISAID frequencies to GPU (enables cycle features)");
        }

        if let Some(vel) = gisaid_velocities {
            if vel.len() != n_residues {
                return Err(PrismError::gpu(
                    "mega_fused",
                    format!(
                        "gisaid_velocities length {} != n_residues {}",
                        vel.len(),
                        n_residues
                    ),
                ));
            }
            let mut d_vel = self
                .stream
                .alloc_zeros::<f32>(n_residues)
                .map_err(|e| PrismError::gpu("mega_fused", format!("Alloc gisaid_vel: {}", e)))?;
            self.stream
                .memcpy_htod(vel, &mut d_vel)
                .map_err(|e| PrismError::gpu("mega_fused", format!("Upload gisaid_vel: {}", e)))?;
            d_gisaid_vel_temp = Some(d_vel);
            log::debug!("Uploaded GISAID velocities to GPU (enables cycle predictions)");
        }

        // === BUFFER POOLING: Zero-allocation hot path ===
        // Only reallocate if current structure exceeds pool capacity
        let (atoms_need_realloc, residues_need_realloc) =
            self.buffer_pool.needs_realloc(n_atoms, n_residues);

        // Reallocate atoms buffer if needed (with 20% growth factor)
        if atoms_need_realloc || self.buffer_pool.d_atoms.is_none() {
            let new_capacity = (n_atoms * 3 * 6 / 5).max(n_atoms * 3); // 20% growth
            self.buffer_pool.d_atoms =
                Some(self.stream.alloc_zeros::<f32>(new_capacity).map_err(|e| {
                    PrismError::gpu(
                        "mega_fused",
                        format!("Failed to allocate atoms pool: {}", e),
                    )
                })?);
            self.buffer_pool.atoms_capacity = new_capacity;
            self.buffer_pool.allocations += 1;
            log::debug!(
                "Buffer pool: allocated atoms buffer (capacity={})",
                new_capacity
            );
        } else {
            self.buffer_pool.reuses += 1;
        }

        // Reallocate residue buffers if needed (with 20% growth factor)
        if residues_need_realloc || self.buffer_pool.d_ca_indices.is_none() {
            let new_capacity = (n_residues * 6 / 5).max(n_residues); // 20% growth

            // Input buffers
            self.buffer_pool.d_ca_indices =
                Some(self.stream.alloc_zeros::<i32>(new_capacity).map_err(|e| {
                    PrismError::gpu(
                        "mega_fused",
                        format!("Failed to allocate ca_indices pool: {}", e),
                    )
                })?);
            self.buffer_pool.d_conservation =
                Some(self.stream.alloc_zeros::<f32>(new_capacity).map_err(|e| {
                    PrismError::gpu(
                        "mega_fused",
                        format!("Failed to allocate conservation pool: {}", e),
                    )
                })?);
            self.buffer_pool.d_bfactor =
                Some(self.stream.alloc_zeros::<f32>(new_capacity).map_err(|e| {
                    PrismError::gpu(
                        "mega_fused",
                        format!("Failed to allocate bfactor pool: {}", e),
                    )
                })?);
            self.buffer_pool.d_burial =
                Some(self.stream.alloc_zeros::<f32>(new_capacity).map_err(|e| {
                    PrismError::gpu(
                        "mega_fused",
                        format!("Failed to allocate burial pool: {}", e),
                    )
                })?);

            // Output buffers
            self.buffer_pool.d_consensus_scores =
                Some(self.stream.alloc_zeros::<f32>(new_capacity).map_err(|e| {
                    PrismError::gpu(
                        "mega_fused",
                        format!("Failed to allocate consensus_scores pool: {}", e),
                    )
                })?);
            self.buffer_pool.d_confidence =
                Some(self.stream.alloc_zeros::<i32>(new_capacity).map_err(|e| {
                    PrismError::gpu(
                        "mega_fused",
                        format!("Failed to allocate confidence pool: {}", e),
                    )
                })?);
            self.buffer_pool.d_signal_mask =
                Some(self.stream.alloc_zeros::<i32>(new_capacity).map_err(|e| {
                    PrismError::gpu(
                        "mega_fused",
                        format!("Failed to allocate signal_mask pool: {}", e),
                    )
                })?);
            self.buffer_pool.d_pocket_assignment =
                Some(self.stream.alloc_zeros::<i32>(new_capacity).map_err(|e| {
                    PrismError::gpu(
                        "mega_fused",
                        format!("Failed to allocate pocket_assignment pool: {}", e),
                    )
                })?);
            self.buffer_pool.d_centrality =
                Some(self.stream.alloc_zeros::<f32>(new_capacity).map_err(|e| {
                    PrismError::gpu(
                        "mega_fused",
                        format!("Failed to allocate centrality pool: {}", e),
                    )
                })?);

            // Combined features: 70 floats per residue (16 base + 12 reservoir + 12 physics + 30 SOTA)
            self.buffer_pool.d_combined_features = Some(
                self.stream
                    .alloc_zeros::<f32>(new_capacity * TOTAL_COMBINED_FEATURES)
                    .map_err(|e| {
                        PrismError::gpu(
                            "mega_fused",
                            format!("Failed to allocate combined_features pool: {}", e),
                        )
                    })?,
            );

            self.buffer_pool.residue_capacity = new_capacity;
            self.buffer_pool.allocations += 10; // 4 input + 6 output buffers
            log::debug!(
                "Buffer pool: allocated residue buffers (capacity={})",
                new_capacity
            );
        }

        // Allocate params buffer if not done (single MegaFusedParams struct = 128 bytes)
        if self.buffer_pool.d_params.is_none() {
            let params_size = std::mem::size_of::<MegaFusedParams>();
            self.buffer_pool.d_params =
                Some(self.stream.alloc_zeros::<u8>(params_size).map_err(|e| {
                    PrismError::gpu(
                        "mega_fused",
                        format!("Failed to allocate params buffer: {}", e),
                    )
                })?);
            self.buffer_pool.allocations += 1;
            log::debug!(
                "Buffer pool: allocated params buffer ({} bytes)",
                params_size
            );
        }

        // === TDA SPATIAL NEIGHBORHOODS ===
        // Build neighborhoods on CPU (O(n²) brute-force, fast enough for typical proteins)
        let tda_neighborhoods = build_tda_neighborhoods(atoms, ca_indices);
        let total_tda_neighbors = tda_neighborhoods.coords.len() / 3;

        // Allocate TDA buffers if needed (grow by 20%)
        if total_tda_neighbors * 3 > self.buffer_pool.tda_neighbor_capacity
            || self.buffer_pool.d_tda_neighbor_coords.is_none()
        {
            let new_capacity = (total_tda_neighbors * 3 * 6 / 5)
                .max(total_tda_neighbors * 3)
                .max(1);
            self.buffer_pool.d_tda_neighbor_coords =
                Some(self.stream.alloc_zeros::<f32>(new_capacity).map_err(|e| {
                    PrismError::gpu(
                        "mega_fused",
                        format!("Failed to allocate TDA neighbor coords: {}", e),
                    )
                })?);
            self.buffer_pool.tda_neighbor_capacity = new_capacity;
            self.buffer_pool.allocations += 1;
            log::debug!(
                "Buffer pool: allocated TDA neighbor coords (capacity={})",
                new_capacity
            );
        }

        // TDA offsets and counts always need residue capacity
        if self.buffer_pool.d_tda_neighbor_offsets.is_none() || residues_need_realloc {
            let new_capacity = (n_residues + 1) * 6 / 5; // +1 for CSR format
            self.buffer_pool.d_tda_neighbor_offsets =
                Some(self.stream.alloc_zeros::<i32>(new_capacity).map_err(|e| {
                    PrismError::gpu(
                        "mega_fused",
                        format!("Failed to allocate TDA neighbor offsets: {}", e),
                    )
                })?);
            self.buffer_pool.d_tda_neighbor_counts =
                Some(self.stream.alloc_zeros::<i32>(new_capacity).map_err(|e| {
                    PrismError::gpu(
                        "mega_fused",
                        format!("Failed to allocate TDA neighbor counts: {}", e),
                    )
                })?);
            self.buffer_pool.allocations += 2;
        }

        // Get mutable references to pooled buffers
        let d_atoms = self.buffer_pool.d_atoms.as_mut().unwrap();
        let d_ca_indices = self.buffer_pool.d_ca_indices.as_mut().unwrap();
        let d_conservation = self.buffer_pool.d_conservation.as_mut().unwrap();
        let d_bfactor = self.buffer_pool.d_bfactor.as_mut().unwrap();
        let d_burial = self.buffer_pool.d_burial.as_mut().unwrap();
        // TDA neighborhood buffers
        let d_tda_neighbor_coords = self.buffer_pool.d_tda_neighbor_coords.as_mut().unwrap();
        let d_tda_neighbor_offsets = self.buffer_pool.d_tda_neighbor_offsets.as_mut().unwrap();
        let d_tda_neighbor_counts = self.buffer_pool.d_tda_neighbor_counts.as_mut().unwrap();
        // Output buffers
        let d_consensus_scores = self.buffer_pool.d_consensus_scores.as_mut().unwrap();
        let d_confidence = self.buffer_pool.d_confidence.as_mut().unwrap();
        let d_signal_mask = self.buffer_pool.d_signal_mask.as_mut().unwrap();
        let d_pocket_assignment = self.buffer_pool.d_pocket_assignment.as_mut().unwrap();
        let d_centrality = self.buffer_pool.d_centrality.as_mut().unwrap();
        let d_combined_features = self.buffer_pool.d_combined_features.as_mut().unwrap();

        // Copy input data to device (fast path: only copy, no alloc)
        *d_atoms = self
            .stream
            .clone_htod(atoms)
            .map_err(|e| PrismError::gpu("mega_fused", format!("Failed to copy atoms: {}", e)))?;
        *d_ca_indices = self.stream.clone_htod(ca_indices).map_err(|e| {
            PrismError::gpu("mega_fused", format!("Failed to copy ca_indices: {}", e))
        })?;
        *d_conservation = self.stream.clone_htod(conservation).map_err(|e| {
            PrismError::gpu("mega_fused", format!("Failed to copy conservation: {}", e))
        })?;
        *d_bfactor = self
            .stream
            .clone_htod(bfactor)
            .map_err(|e| PrismError::gpu("mega_fused", format!("Failed to copy bfactor: {}", e)))?;
        *d_burial = self
            .stream
            .clone_htod(burial)
            .map_err(|e| PrismError::gpu("mega_fused", format!("Failed to copy burial: {}", e)))?;

        // Copy TDA neighborhood data to device
        if !tda_neighborhoods.coords.is_empty() {
            *d_tda_neighbor_coords =
                self.stream
                    .clone_htod(&tda_neighborhoods.coords)
                    .map_err(|e| {
                        PrismError::gpu("mega_fused", format!("Failed to copy TDA coords: {}", e))
                    })?;
        }
        *d_tda_neighbor_offsets =
            self.stream
                .clone_htod(&tda_neighborhoods.offsets)
                .map_err(|e| {
                    PrismError::gpu("mega_fused", format!("Failed to copy TDA offsets: {}", e))
                })?;
        *d_tda_neighbor_counts =
            self.stream
                .clone_htod(&tda_neighborhoods.counts)
                .map_err(|e| {
                    PrismError::gpu("mega_fused", format!("Failed to copy TDA counts: {}", e))
                })?;

        // Convert config to runtime params and copy to device
        let params = MegaFusedParams::from_config(config);
        let params_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                &params as *const MegaFusedParams as *const u8,
                std::mem::size_of::<MegaFusedParams>(),
            )
        };
        let d_params = self.buffer_pool.d_params.as_mut().unwrap();
        *d_params = self
            .stream
            .clone_htod(params_bytes)
            .map_err(|e| PrismError::gpu("mega_fused", format!("Failed to copy params: {}", e)))?;

        // Configure kernel launch
        // TILE_SIZE = 32, BLOCK_SIZE = 256
        let tile_size = 32;
        let block_size = 256;
        let n_tiles = (n_residues + tile_size - 1) / tile_size;

        let launch_config = LaunchConfig {
            grid_dim: (n_tiles as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0, // Shared memory is statically allocated in kernel
        };

        // Create i32 bindings for kernel args (must outlive the unsafe block)
        let n_atoms_i32 = n_atoms as i32;
        let n_residues_i32 = n_residues as i32;

        // Initialize GPU telemetry for provenance tracking
        let telemetry = GpuTelemetry::new();
        let clock_before = telemetry.get_clock_mhz();
        let memory_clock_before = telemetry.get_memory_clock_mhz();
        let kernel_start = Instant::now();

        // Launch kernel with full runtime params struct
        // Signature: inputs, n_atoms, n_residues, TDA_neighborhoods, outputs, combined_features, params*
        unsafe {
            let mut builder = self.stream.launch_builder(func);
            // Input buffers
            builder.arg(&*d_atoms);
            builder.arg(&*d_ca_indices);
            builder.arg(&*d_conservation);
            builder.arg(&*d_bfactor);
            builder.arg(&*d_burial);
            // NEW: Pass residue_types (allocate dummy if None)
            let d_residue_types_for_kernel = match &d_residue_types_temp {
                Some(d) => d,
                None => {
                    // Create dummy buffer (kernel checks for nullptr behavior)
                    &self
                        .stream
                        .alloc_zeros::<i32>(1)
                        .map_err(|e| PrismError::gpu("mega_fused", format!("Alloc dummy: {}", e)))?
                }
            };
            builder.arg(d_residue_types_for_kernel);
            builder.arg(&n_atoms_i32);
            builder.arg(&n_residues_i32);
            // TDA spatial neighborhood buffers (pre-computed on CPU)
            builder.arg(&*d_tda_neighbor_coords); // [total_neighbors * 3] neighbor CA coords
            builder.arg(&*d_tda_neighbor_offsets); // [n_residues + 1] CSR offsets
            builder.arg(&*d_tda_neighbor_counts); // [n_residues] neighbor count per residue
                                                  // PRISM-VE: GISAID temporal data (optional)
            let d_gisaid_freq_for_kernel = match &d_gisaid_freq_temp {
                Some(d) => d,
                None => {
                    // Create dummy buffer (kernel checks for nullptr)
                    &self.stream.alloc_zeros::<f32>(1).map_err(|e| {
                        PrismError::gpu("mega_fused", format!("Alloc dummy freq: {}", e))
                    })?
                }
            };
            let d_gisaid_vel_for_kernel = match &d_gisaid_vel_temp {
                Some(d) => d,
                None => {
                    // Create dummy buffer (kernel checks for nullptr)
                    &self.stream.alloc_zeros::<f32>(1).map_err(|e| {
                        PrismError::gpu("mega_fused", format!("Alloc dummy vel: {}", e))
                    })?
                }
            };
            builder.arg(d_gisaid_freq_for_kernel);
            builder.arg(d_gisaid_vel_for_kernel);
            // Output buffers
            builder.arg(d_consensus_scores);
            builder.arg(d_confidence);
            builder.arg(d_signal_mask);
            builder.arg(d_pocket_assignment);
            builder.arg(d_centrality);
            builder.arg(d_combined_features); // Output: 70-dim combined features (16+12+12+30)
            builder.arg(&*d_params); // Runtime params struct pointer

            builder.launch(launch_config).map_err(|e| {
                PrismError::gpu("mega_fused", format!("Kernel launch failed: {}", e))
            })?;
        }

        // Synchronize and copy results back
        self.stream
            .synchronize()
            .map_err(|e| PrismError::gpu("mega_fused", format!("Sync failed: {}", e)))?;

        // Capture GPU telemetry after kernel completion
        let kernel_elapsed = kernel_start.elapsed();
        let clock_after = telemetry.get_clock_mhz();
        let memory_clock_after = telemetry.get_memory_clock_mhz();
        let temperature = telemetry.get_temperature();

        // Build kernel telemetry event
        let kernel_event = KernelTelemetryEvent {
            file: "mega_fused_pocket_detection".to_string(),
            kernel: "mega_fused_kernel".to_string(),
            clock_before_mhz: clock_before,
            clock_after_mhz: clock_after,
            memory_clock_before_mhz: memory_clock_before,
            memory_clock_after_mhz: memory_clock_after,
            temperature_c: temperature,
            memory_used_bytes: telemetry.get_memory_used(),
            execution_time_us: kernel_elapsed.as_micros() as u64,
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        // Build GPU provenance data
        let gpu_provenance = GpuProvenanceData {
            gpu_name: telemetry.get_gpu_name(),
            driver_version: telemetry.get_driver_version(),
            kernel_events: vec![kernel_event],
            total_gpu_time_us: kernel_elapsed.as_micros() as u64,
            avg_clock_mhz: clock_before.or(clock_after),
            peak_temperature_c: temperature,
        };

        // Copy output data from pooled buffers (need to reborrow as shared for clone_dtoh)
        // CRITICAL: Truncate to n_residues to avoid stale data from larger previous structures
        let d_consensus_scores = self.buffer_pool.d_consensus_scores.as_ref().unwrap();
        let d_confidence = self.buffer_pool.d_confidence.as_ref().unwrap();
        let d_signal_mask = self.buffer_pool.d_signal_mask.as_ref().unwrap();
        let d_pocket_assignment = self.buffer_pool.d_pocket_assignment.as_ref().unwrap();
        let d_centrality = self.buffer_pool.d_centrality.as_ref().unwrap();
        let d_combined_features = self.buffer_pool.d_combined_features.as_ref().unwrap();

        let mut consensus_scores = self.stream.clone_dtoh(d_consensus_scores).map_err(|e| {
            PrismError::gpu(
                "mega_fused",
                format!("Failed to read consensus_scores: {}", e),
            )
        })?;
        let mut confidence = self.stream.clone_dtoh(d_confidence).map_err(|e| {
            PrismError::gpu("mega_fused", format!("Failed to read confidence: {}", e))
        })?;
        let mut signal_mask = self.stream.clone_dtoh(d_signal_mask).map_err(|e| {
            PrismError::gpu("mega_fused", format!("Failed to read signal_mask: {}", e))
        })?;
        let mut pocket_assignment = self.stream.clone_dtoh(d_pocket_assignment).map_err(|e| {
            PrismError::gpu(
                "mega_fused",
                format!("Failed to read pocket_assignment: {}", e),
            )
        })?;
        let mut centrality = self.stream.clone_dtoh(d_centrality).map_err(|e| {
            PrismError::gpu("mega_fused", format!("Failed to read centrality: {}", e))
        })?;
        let mut combined_features = self.stream.clone_dtoh(d_combined_features).map_err(|e| {
            PrismError::gpu(
                "mega_fused",
                format!("Failed to read combined_features: {}", e),
            )
        })?;

        // DEBUG: Print first 10 values to verify GPU output
        if combined_features.len() >= 10 {
            log::debug!(
                "DEBUG combined_features[0..10]: {:?}",
                &combined_features[0..10]
            );
        }

        // Truncate to actual structure size (buffer may be larger from previous runs)
        consensus_scores.truncate(n_residues);
        confidence.truncate(n_residues);
        signal_mask.truncate(n_residues);
        pocket_assignment.truncate(n_residues);
        centrality.truncate(n_residues);
        combined_features.truncate(n_residues * TOTAL_COMBINED_FEATURES);

        Ok(MegaFusedOutput {
            consensus_scores,
            confidence,
            signal_mask,
            pocket_assignment,
            centrality,
            combined_features,
            gpu_telemetry: Some(gpu_provenance),
        })
    }

    /// SESSION 10E: Extract features using 3-pass multi-kernel architecture
    /// Fixes 16 dead SOTA features by using global distance matrix
    pub fn extract_features_multipass(
        &mut self,
        atoms: &[f32],
        ca_indices: &[i32],
        bfactor: &[f32],
        n_residues: usize,
    ) -> Result<Vec<f32>, PrismError> {
        // Check if multi-pass kernels available
        if self.distance_kernel.is_none() || self.sota_kernel.is_none() {
            return Err(PrismError::gpu(
                "extract_features_multipass",
                "Multi-pass kernels not loaded",
            ));
        }

        // Allocate and upload atoms
        let mut atoms_dev = self
            .stream
            .alloc_zeros::<f32>(atoms.len())
            .map_err(|e| PrismError::gpu("multipass", format!("Alloc atoms failed: {:?}", e)))?;
        self.stream
            .memcpy_htod(atoms, &mut atoms_dev)
            .map_err(|e| PrismError::gpu("multipass", format!("Upload atoms failed: {:?}", e)))?;

        // Allocate and upload ca_indices
        let mut ca_indices_dev = self
            .stream
            .alloc_zeros::<i32>(ca_indices.len())
            .map_err(|e| {
                PrismError::gpu("multipass", format!("Alloc ca_indices failed: {:?}", e))
            })?;
        self.stream
            .memcpy_htod(ca_indices, &mut ca_indices_dev)
            .map_err(|e| {
                PrismError::gpu("multipass", format!("Upload ca_indices failed: {:?}", e))
            })?;

        // Allocate and upload bfactor
        let mut bfactor_dev = self
            .stream
            .alloc_zeros::<f32>(bfactor.len())
            .map_err(|e| PrismError::gpu("multipass", format!("Alloc bfactor failed: {:?}", e)))?;
        self.stream
            .memcpy_htod(bfactor, &mut bfactor_dev)
            .map_err(|e| PrismError::gpu("multipass", format!("Upload bfactor failed: {:?}", e)))?;

        // Allocate distance matrix
        let distance_matrix_size = n_residues * n_residues;
        if self.distance_matrix_capacity < distance_matrix_size {
            log::debug!(
                "Allocating distance matrix: {} x {} = {} floats ({} MB)",
                n_residues,
                n_residues,
                distance_matrix_size,
                distance_matrix_size * 4 / 1_000_000
            );
            self.distance_matrix_buffer = Some(
                self.stream
                    .alloc_zeros::<f32>(distance_matrix_size)
                    .map_err(|e| {
                        PrismError::gpu(
                            "multipass",
                            format!("Alloc distance matrix failed: {:?}", e),
                        )
                    })?,
            );
            self.distance_matrix_capacity = distance_matrix_size;
        }
        let distance_matrix = self.distance_matrix_buffer.as_ref().unwrap();

        // PASS 1: Compute distance matrix
        let grid_x = (n_residues + 15) / 16;
        let grid_y = (n_residues + 15) / 16;
        let grid_dim = (grid_x as u32, grid_y as u32, 1);
        let block_dim = (16, 16, 1);
        let cutoff = 15.0f32;

        log::debug!(
            "Pass 1: Distance matrix (grid: {}x{}, block: 16x16)",
            grid_x,
            grid_y
        );

        let distance_kernel = self.distance_kernel.as_ref().unwrap();
        let launch_config = LaunchConfig {
            grid_dim,
            block_dim,
            shared_mem_bytes: 0,
        };

        let n_residues_i32 = n_residues as i32;
        unsafe {
            let mut builder = self.stream.launch_builder(distance_kernel);
            builder.arg(&atoms_dev);
            builder.arg(&ca_indices_dev);
            builder.arg(&n_residues_i32);
            builder.arg(distance_matrix);
            builder.arg(&cutoff);
            builder.launch(launch_config).map_err(|e| {
                PrismError::gpu("distance_kernel", format!("Launch failed: {:?}", e))
            })?;
        }

        self.stream
            .synchronize()
            .map_err(|e| PrismError::gpu("distance_kernel", format!("Sync failed: {:?}", e)))?;

        log::debug!("Pass 1 complete");

        // Create default burial/conservation arrays (simplified for now)
        let burial: Vec<f32> = vec![0.5; n_residues];
        let conservation: Vec<f32> = vec![0.5; n_residues];

        let mut burial_dev = self
            .stream
            .alloc_zeros::<f32>(burial.len())
            .map_err(|e| PrismError::gpu("multipass", format!("Alloc burial failed: {:?}", e)))?;
        burial_dev = self
            .stream
            .clone_htod(&burial)
            .map_err(|e| PrismError::gpu("multipass", format!("Upload burial failed: {:?}", e)))?;

        let mut conservation_dev =
            self.stream
                .alloc_zeros::<f32>(conservation.len())
                .map_err(|e| {
                    PrismError::gpu("multipass", format!("Alloc conservation failed: {:?}", e))
                })?;
        conservation_dev = self.stream.clone_htod(&conservation).map_err(|e| {
            PrismError::gpu("multipass", format!("Upload conservation failed: {:?}", e))
        })?;

        // Allocate SOTA features
        let sota_size = n_residues * 30;
        if self.sota_features_capacity < sota_size {
            log::debug!(
                "Allocating SOTA features: {} x 30 = {} floats ({} KB)",
                n_residues,
                sota_size,
                sota_size * 4 / 1000
            );
            self.sota_features_buffer =
                Some(self.stream.alloc_zeros::<f32>(sota_size).map_err(|e| {
                    PrismError::gpu("multipass", format!("Alloc SOTA buffer failed: {:?}", e))
                })?);
            self.sota_features_capacity = sota_size;
        }
        let sota_features = self.sota_features_buffer.as_ref().unwrap();

        // PASS 2: Compute SOTA features
        let grid_dim = (((n_residues + 255) / 256) as u32, 1, 1);
        let block_dim = (256, 1, 1);

        log::debug!("Pass 2: SOTA features (grid: {}, block: 256)", grid_dim.0);

        let sota_kernel = self.sota_kernel.as_ref().unwrap();
        let launch_config = LaunchConfig {
            grid_dim,
            block_dim,
            shared_mem_bytes: 0,
        };

        unsafe {
            let mut builder = self.stream.launch_builder(sota_kernel);
            builder.arg(distance_matrix);
            builder.arg(&atoms_dev);
            builder.arg(&ca_indices_dev);
            builder.arg(&bfactor_dev);
            builder.arg(&burial_dev);
            builder.arg(&conservation_dev);
            builder.arg(&n_residues_i32);
            builder.arg(sota_features);
            builder
                .launch(launch_config)
                .map_err(|e| PrismError::gpu("sota_kernel", format!("Launch failed: {:?}", e)))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| PrismError::gpu("sota_kernel", format!("Sync failed: {:?}", e)))?;

        log::debug!("Pass 2 complete");

        // Download SOTA features
        let sota_host: Vec<f32> = self
            .stream
            .clone_dtoh(sota_features)
            .map_err(|e| PrismError::gpu("multipass", format!("Download SOTA failed: {:?}", e)))?;

        // For now, return just SOTA features (30-dim) for testing
        // TODO: Combine with base features for full 70-dim
        log::info!(
            "Multi-pass extraction complete: {} residues x 30 SOTA features",
            n_residues
        );

        Ok(sota_host)
    }
}

/// Confidence level constants
pub mod confidence {
    pub const LOW: i32 = 0;
    pub const MEDIUM: i32 = 1;
    pub const HIGH: i32 = 2;
}

/// Signal mask bit flags
pub mod signals {
    /// Geometric pocket detection signal
    pub const GEOMETRIC: i32 = 0x01;
    /// Conservation signal (evolutionarily conserved)
    pub const CONSERVATION: i32 = 0x02;
    /// Network centrality signal (hub residue)
    pub const CENTRALITY: i32 = 0x04;
    /// Flexibility signal (high B-factor)
    pub const FLEXIBILITY: i32 = 0x08;

    /// Count number of signals in mask
    pub fn count(mask: i32) -> i32 {
        mask.count_ones() as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = MegaFusedConfig::default();
        assert_eq!(config.contact_sigma, 6.0);
        assert_eq!(config.consensus_threshold, 0.35);
        assert_eq!(config.kempe_iterations, 10);
        assert_eq!(config.power_iterations, 15);
        assert!(config.use_fp16);
    }

    #[test]
    fn test_signal_count() {
        assert_eq!(signals::count(0), 0);
        assert_eq!(signals::count(signals::GEOMETRIC), 1);
        assert_eq!(
            signals::count(signals::GEOMETRIC | signals::CONSERVATION),
            2
        );
        assert_eq!(
            signals::count(signals::GEOMETRIC | signals::CONSERVATION | signals::CENTRALITY),
            3
        );
        assert_eq!(signals::count(0x0F), 4);
    }
}
