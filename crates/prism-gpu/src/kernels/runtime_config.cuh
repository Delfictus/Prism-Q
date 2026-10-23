#ifndef RUNTIME_CONFIG_CUH
#define RUNTIME_CONFIG_CUH

/**
 * RuntimeConfig - GPU-side mirror of Rust RuntimeConfig
 * MUST be kept in sync with crates/prism-core/src/runtime_config.rs
 * Total size: 256 bytes (cache-aligned)
 */
struct RuntimeConfig {
    // WHCR Parameters
    float stress_weight;
    float persistence_weight;
    float belief_weight;
    float hotspot_multiplier;

    // Dendritic Reservoir (8-branch)
    float tau_decay[8];
    float branch_weights[8];
    float reservoir_leak_rate;
    float spectral_radius;
    float input_scaling;
    float reservoir_sparsity;

    // W-Cycle Multigrid
    int num_levels;
    float coarsening_ratio;
    float restriction_weight;
    float prolongation_weight;
    int pre_smooth_iterations;
    int post_smooth_iterations;

    // Quantum Tunneling
    float tunneling_prob_base;
    float tunneling_prob_boost;
    float chemical_potential;
    float transverse_field;
    float interference_decay;
    int num_quantum_states;

    // Parallel Tempering (reduced to 8 replicas for 256-byte alignment)
    float temperatures[8];
    int num_replicas;
    int swap_interval;
    float swap_probability;

    // TPTP
    float betti_0_threshold;
    float betti_1_threshold;
    float betti_2_threshold;
    float persistence_threshold;
    int stability_window;
    float transition_sensitivity;

    // Active Inference
    float free_energy_threshold;
    float belief_update_rate;
    float precision_weight;
    float policy_temperature;

    // Meta/Control
    int iteration;
    int phase_id;
    float global_temperature;
    float learning_rate;
    float exploration_rate;

    // Flags (packed bits)
    int flags;

    // Padding to 256 bytes
    float _padding;
};

// Flag bit accessors
__device__ __forceinline__ bool quantum_enabled(const RuntimeConfig* cfg) {
    return (cfg->flags & (1 << 0)) != 0;
}

__device__ __forceinline__ bool tptp_enabled(const RuntimeConfig* cfg) {
    return (cfg->flags & (1 << 1)) != 0;
}

__device__ __forceinline__ bool dendritic_enabled(const RuntimeConfig* cfg) {
    return (cfg->flags & (1 << 2)) != 0;
}

__device__ __forceinline__ bool active_inference_enabled(const RuntimeConfig* cfg) {
    return (cfg->flags & (1 << 3)) != 0;
}

__device__ __forceinline__ bool tempering_enabled(const RuntimeConfig* cfg) {
    return (cfg->flags & (1 << 4)) != 0;
}

__device__ __forceinline__ bool multigrid_enabled(const RuntimeConfig* cfg) {
    return (cfg->flags & (1 << 5)) != 0;
}

__device__ __forceinline__ bool f64_precision_enabled(const RuntimeConfig* cfg) {
    return (cfg->flags & (1 << 6)) != 0;
}

__device__ __forceinline__ bool online_learning_enabled(const RuntimeConfig* cfg) {
    return (cfg->flags & (1 << 7)) != 0;
}

/**
 * KernelTelemetry - Output from GPU kernels
 * Total size: 64 bytes (cache-line aligned)
 */
struct KernelTelemetry {
    int conflicts;
    int colors_used;
    int moves_applied;
    int tunneling_events;
    int phase_transitions;
    float betti_numbers[3];
    float reservoir_activity;
    float free_energy;
    int best_replica;
    int iteration_time_us;
    float _padding[4];
};

#endif // RUNTIME_CONFIG_CUH
