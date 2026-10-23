//! RuntimeConfig - GPU-transferable configuration struct
//!
//! Provides FFI-safe, #[repr(C)] configuration structures that can be transferred to GPU kernels.
//! All fields use FFI-safe types (primitives, fixed-size arrays) for CUDA compatibility.

use serde::{Deserialize, Serialize};

// Flag bit positions for packed flags field
const FLAG_QUANTUM_ENABLED: i32 = 1 << 0;
const FLAG_TPTP_ENABLED: i32 = 1 << 1;
const FLAG_DENDRITIC_ENABLED: i32 = 1 << 2;
const FLAG_PARALLEL_TEMPERING_ENABLED: i32 = 1 << 3;
const FLAG_ACTIVE_INFERENCE_ENABLED: i32 = 1 << 4;
const FLAG_MULTIGRID_ENABLED: i32 = 1 << 5;
const FLAG_ADAPTIVE_LEARNING: i32 = 1 << 6;
const FLAG_WARMSTART_MODE: i32 = 1 << 7;

/// RuntimeConfig for GPU kernel - must be #[repr(C)] for FFI compatibility
///
/// This structure is designed to be transferred to GPU memory and used by CUDA kernels.
/// Total size: 256 bytes for efficient GPU memory alignment.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct RuntimeConfig {
    // ========== WHCR Parameters ==========
    /// Weight for stress-based coloring heuristic
    pub stress_weight: f32,
    /// Weight for color persistence across iterations
    pub persistence_weight: f32,
    /// Weight for belief propagation signals
    pub belief_weight: f32,
    /// Multiplier for conflict hotspot detection
    pub hotspot_multiplier: f32,

    // ========== Dendritic Reservoir (8-branch) ==========
    /// Time decay constants for each dendritic branch
    pub tau_decay: [f32; 8],
    /// Weights for combining 8 dendritic branches
    pub branch_weights: [f32; 8],
    /// Leak rate for reservoir state updates
    pub reservoir_leak_rate: f32,
    /// Spectral radius for reservoir stability
    pub spectral_radius: f32,
    /// Input scaling factor for reservoir
    pub input_scaling: f32,
    /// Sparsity of reservoir connectivity
    pub reservoir_sparsity: f32,

    // ========== W-Cycle Multigrid ==========
    /// Number of multigrid levels
    pub num_levels: i32,
    /// Ratio for coarsening between levels
    pub coarsening_ratio: f32,
    /// Weight for restriction operator
    pub restriction_weight: f32,
    /// Weight for prolongation operator
    pub prolongation_weight: f32,
    /// Number of pre-smoothing iterations
    pub pre_smooth_iterations: i32,
    /// Number of post-smoothing iterations
    pub post_smooth_iterations: i32,

    // ========== Quantum Tunneling ==========
    /// Base probability for quantum tunneling events
    pub tunneling_prob_base: f32,
    /// Boost factor for tunneling probability
    pub tunneling_prob_boost: f32,
    /// Chemical potential for quantum system
    pub chemical_potential: f32,
    /// Transverse field strength
    pub transverse_field: f32,
    /// Decay rate for quantum interference
    pub interference_decay: f32,
    /// Number of quantum states to track
    pub num_quantum_states: i32,

    // ========== Parallel Tempering ==========
    /// Temperature schedule for parallel replicas (reduced to 8 for size constraint)
    pub temperatures: [f32; 8],
    /// Number of parallel replicas
    pub num_replicas: i32,
    /// Iterations between replica swap attempts
    pub swap_interval: i32,
    /// Probability of accepting replica swaps
    pub swap_probability: f32,

    // ========== TPTP (Topological Phase Transition Predictor) ==========
    /// Threshold for 0-dimensional Betti number
    pub betti_0_threshold: f32,
    /// Threshold for 1-dimensional Betti number
    pub betti_1_threshold: f32,
    /// Threshold for 2-dimensional Betti number
    pub betti_2_threshold: f32,
    /// Threshold for persistence diagram features
    pub persistence_threshold: f32,
    /// Window size for stability detection
    pub stability_window: i32,
    /// Sensitivity for detecting phase transitions
    pub transition_sensitivity: f32,

    // ========== Active Inference ==========
    /// Threshold for free energy convergence
    pub free_energy_threshold: f32,
    /// Rate for updating belief states
    pub belief_update_rate: f32,
    /// Weight for precision in belief updates
    pub precision_weight: f32,
    /// Temperature for policy selection
    pub policy_temperature: f32,

    // ========== Meta/Control ==========
    /// Current iteration counter
    pub iteration: i32,
    /// Current phase identifier (0-6)
    pub phase_id: i32,
    /// Global temperature for simulated annealing
    pub global_temperature: f32,
    /// Learning rate for adaptive algorithms
    pub learning_rate: f32,
    /// Exploration rate for RL/search
    pub exploration_rate: f32,

    // ========== Flags ==========
    /// Packed feature flags (see FLAG_* constants)
    pub flags: i32,

    // ========== Padding ==========
    /// Padding to align to 256 bytes (252 base + 4 padding = 256)
    pub _padding: f32,
}

impl RuntimeConfig {
    /// Production-grade default configuration
    ///
    /// Optimized for DIMACS benchmarks with proven parameter values.
    pub fn production() -> Self {
        Self {
            // WHCR Parameters
            stress_weight: 0.45,
            persistence_weight: 0.35,
            belief_weight: 0.20,
            hotspot_multiplier: 2.5,

            // Dendritic Reservoir (8-branch)
            tau_decay: [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60],
            branch_weights: [0.20, 0.18, 0.16, 0.14, 0.12, 0.10, 0.06, 0.04],
            reservoir_leak_rate: 0.1,
            spectral_radius: 0.95,
            input_scaling: 1.0,
            reservoir_sparsity: 0.15,

            // W-Cycle Multigrid
            num_levels: 4,
            coarsening_ratio: 0.5,
            restriction_weight: 0.7,
            prolongation_weight: 0.8,
            pre_smooth_iterations: 2,
            post_smooth_iterations: 2,

            // Quantum Tunneling
            tunneling_prob_base: 0.01,
            tunneling_prob_boost: 1.5,
            chemical_potential: 1.0,
            transverse_field: 0.5,
            interference_decay: 0.95,
            num_quantum_states: 64,

            // Parallel Tempering
            temperatures: [1.0, 1.5, 2.0, 3.0, 4.5, 6.5, 9.5, 14.0],
            num_replicas: 8,
            swap_interval: 10,
            swap_probability: 0.3,

            // TPTP
            betti_0_threshold: 0.5,
            betti_1_threshold: 0.3,
            betti_2_threshold: 0.2,
            persistence_threshold: 0.1,
            stability_window: 20,
            transition_sensitivity: 0.15,

            // Active Inference
            free_energy_threshold: 0.01,
            belief_update_rate: 0.3,
            precision_weight: 1.5,
            policy_temperature: 1.0,

            // Meta/Control
            iteration: 0,
            phase_id: 0,
            global_temperature: 1.0,
            learning_rate: 0.001,
            exploration_rate: 0.1,

            // All features enabled by default
            flags: FLAG_QUANTUM_ENABLED
                | FLAG_TPTP_ENABLED
                | FLAG_DENDRITIC_ENABLED
                | FLAG_PARALLEL_TEMPERING_ENABLED
                | FLAG_ACTIVE_INFERENCE_ENABLED
                | FLAG_MULTIGRID_ENABLED
                | FLAG_ADAPTIVE_LEARNING,

            _padding: 0.0,
        }
    }

    /// Check if quantum tunneling is enabled
    pub fn quantum_enabled(&self) -> bool {
        (self.flags & FLAG_QUANTUM_ENABLED) != 0
    }

    /// Check if TPTP (Topological Phase Transition Predictor) is enabled
    pub fn tptp_enabled(&self) -> bool {
        (self.flags & FLAG_TPTP_ENABLED) != 0
    }

    /// Check if dendritic reservoir is enabled
    pub fn dendritic_enabled(&self) -> bool {
        (self.flags & FLAG_DENDRITIC_ENABLED) != 0
    }

    /// Check if parallel tempering is enabled
    pub fn parallel_tempering_enabled(&self) -> bool {
        (self.flags & FLAG_PARALLEL_TEMPERING_ENABLED) != 0
    }

    /// Check if active inference is enabled
    pub fn active_inference_enabled(&self) -> bool {
        (self.flags & FLAG_ACTIVE_INFERENCE_ENABLED) != 0
    }

    /// Check if multigrid is enabled
    pub fn multigrid_enabled(&self) -> bool {
        (self.flags & FLAG_MULTIGRID_ENABLED) != 0
    }

    /// Check if adaptive learning is enabled
    pub fn adaptive_learning(&self) -> bool {
        (self.flags & FLAG_ADAPTIVE_LEARNING) != 0
    }

    /// Check if warmstart mode is enabled
    pub fn warmstart_mode(&self) -> bool {
        (self.flags & FLAG_WARMSTART_MODE) != 0
    }

    /// Enable quantum tunneling
    pub fn enable_quantum(&mut self) {
        self.flags |= FLAG_QUANTUM_ENABLED;
    }

    /// Disable quantum tunneling
    pub fn disable_quantum(&mut self) {
        self.flags &= !FLAG_QUANTUM_ENABLED;
    }

    /// Enable TPTP
    pub fn enable_tptp(&mut self) {
        self.flags |= FLAG_TPTP_ENABLED;
    }

    /// Disable TPTP
    pub fn disable_tptp(&mut self) {
        self.flags &= !FLAG_TPTP_ENABLED;
    }

    /// Enable dendritic reservoir
    pub fn enable_dendritic(&mut self) {
        self.flags |= FLAG_DENDRITIC_ENABLED;
    }

    /// Disable dendritic reservoir
    pub fn disable_dendritic(&mut self) {
        self.flags &= !FLAG_DENDRITIC_ENABLED;
    }

    /// Enable warmstart mode
    pub fn enable_warmstart(&mut self) {
        self.flags |= FLAG_WARMSTART_MODE;
    }

    /// Disable warmstart mode
    pub fn disable_warmstart(&mut self) {
        self.flags &= !FLAG_WARMSTART_MODE;
    }
}

/// KernelTelemetry - GPU kernel execution metrics
///
/// This structure is populated by GPU kernels and returned to the host.
/// Total size: 64 bytes for efficient GPU memory alignment.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct KernelTelemetry {
    /// Number of graph coloring conflicts detected
    pub conflicts: i32,
    /// Number of colors currently used
    pub colors_used: i32,
    /// Number of moves/color changes applied
    pub moves_applied: i32,
    /// Number of quantum tunneling events
    pub tunneling_events: i32,
    /// Number of topological phase transitions detected
    pub phase_transitions: i32,
    /// Betti numbers [B0, B1, B2] for topological features
    pub betti_numbers: [f32; 3],
    /// Average reservoir activity level
    pub reservoir_activity: f32,
    /// Current free energy (active inference)
    pub free_energy: f32,
    /// Best performing replica index (parallel tempering)
    pub best_replica: i32,
    /// Kernel iteration time in microseconds
    pub iteration_time_us: i32,
    /// Padding to 64 bytes (48 base + 16 padding = 64)
    pub _padding: [f32; 4],
}

impl KernelTelemetry {
    /// Create a new empty telemetry instance
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if the solution is conflict-free
    pub fn is_valid(&self) -> bool {
        self.conflicts == 0
    }

    /// Get the chromatic number (number of colors used)
    pub fn chromatic_number(&self) -> usize {
        self.colors_used as usize
    }

    /// Get total quantum events
    pub fn total_quantum_events(&self) -> usize {
        self.tunneling_events as usize
    }

    /// Get total topological events
    pub fn total_topological_events(&self) -> usize {
        self.phase_transitions as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_config_size() {
        // Ensure RuntimeConfig is exactly 256 bytes for GPU alignment
        assert_eq!(std::mem::size_of::<RuntimeConfig>(), 256);
    }

    #[test]
    fn test_kernel_telemetry_size() {
        // Ensure KernelTelemetry is exactly 64 bytes for GPU alignment
        assert_eq!(std::mem::size_of::<KernelTelemetry>(), 64);
    }

    #[test]
    fn test_production_config() {
        let config = RuntimeConfig::production();
        assert!(config.quantum_enabled());
        assert!(config.tptp_enabled());
        assert!(config.dendritic_enabled());
        assert!(config.parallel_tempering_enabled());
        assert_eq!(config.num_replicas, 8);
        assert_eq!(config.num_levels, 4);
    }

    #[test]
    fn test_flag_accessors() {
        let mut config = RuntimeConfig::default();
        assert!(!config.quantum_enabled());

        config.enable_quantum();
        assert!(config.quantum_enabled());

        config.disable_quantum();
        assert!(!config.quantum_enabled());
    }

    #[test]
    fn test_telemetry_validity() {
        let mut telem = KernelTelemetry::new();
        telem.conflicts = 0;
        telem.colors_used = 42;
        assert!(telem.is_valid());
        assert_eq!(telem.chromatic_number(), 42);

        telem.conflicts = 5;
        assert!(!telem.is_valid());
    }
}

// ════════════════════════════════════════════════════════════════════════════
// GPU FFI Traits (cudarc 0.18.1 compatibility)
// ════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "cuda")]
mod cuda_impls {
    use super::*;

    // SAFETY: RuntimeConfig is #[repr(C)] with only primitive types and fixed-size arrays.
    // All fields are valid when zero-initialized.
    unsafe impl cudarc::driver::ValidAsZeroBits for RuntimeConfig {}
    unsafe impl cudarc::driver::DeviceRepr for RuntimeConfig {}

    // SAFETY: KernelTelemetry is #[repr(C)] with only primitive types and fixed-size arrays.
    // All fields are valid when zero-initialized.
    unsafe impl cudarc::driver::ValidAsZeroBits for KernelTelemetry {}
    unsafe impl cudarc::driver::DeviceRepr for KernelTelemetry {}
}
