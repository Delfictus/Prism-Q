//! PRISM-NHS Ultra-Sensitive Neuromorphic Detector
//!
//! Enhanced LIF neurons for detecting subtle thermal signatures
//! from UV probe absorption at binding sites.
//!
//! KEY INSIGHT:
//! In cryo conditions, structural motion is frozen, but THERMAL
//! signatures from UV absorption are still detectable:
//!
//! 1. Temperature spikes at absorption sites
//! 2. Thermal gradients as heat spreads
//! 3. Phase transitions (ice melting)
//! 4. Correlated responses at pocket aromatics
//!
//! The neuromorphic detector acts like a "thermal camera" that
//! converts these subtle heat patterns into spike trains.

#ifndef PRISM_SENSITIVE_DETECTOR_CUH
#define PRISM_SENSITIVE_DETECTOR_CUH

#include <cuda_runtime.h>

// ============================================================================
// SENSITIVITY CONFIGURATION
// ============================================================================

// Detection modes (can be combined)
#define DETECT_TEMP_SPIKE       0x01    // Direct temperature increase
#define DETECT_TEMP_GRADIENT    0x02    // Heat flow direction
#define DETECT_MELT_WAVE        0x04    // Ice→water transition
#define DETECT_CORRELATION      0x08    // Multi-aromatic pocket signature
#define DETECT_ALL              0x0F    // All modes

// Sensitivity levels
#define SENSITIVITY_ULTRA       0       // Detect EVERYTHING (noisy but complete)
#define SENSITIVITY_HIGH        1       // Detect weak signals
#define SENSITIVITY_NORMAL      2       // Balanced
#define SENSITIVITY_LOW         3       // Only strong signals

// Threshold scaling by sensitivity level
__constant__ float SENSITIVITY_SCALES[4] = {
    0.1f,   // ULTRA: 10x more sensitive
    0.3f,   // HIGH: 3x more sensitive
    1.0f,   // NORMAL: baseline
    3.0f    // LOW: 3x less sensitive
};

// ============================================================================
// MULTI-MODAL DETECTION STATE
// ============================================================================

// Each voxel has multiple detection channels
struct MultiModalVoxelState {
    // Thermal channel
    float thermal_potential;        // Integrates temperature changes
    float thermal_baseline;         // Reference temperature
    int thermal_spike_count;        // Spikes from this channel
    
    // Gradient channel
    float gradient_potential;       // Integrates heat flow
    float3 last_gradient_dir;       // Direction of last detected flow
    int gradient_spike_count;
    
    // Melt channel
    float melt_potential;           // Integrates melting events
    float ice_fraction;             // Current ice content
    int melt_spike_count;
    
    // Combined state
    float combined_potential;       // Weighted sum of channels
    int last_spike_time;
    bool in_refractory;
    
    // Signal quality
    float signal_to_noise;          // Current SNR estimate
    float confidence;               // Detection confidence
};

// Global detector configuration
struct SensitiveDetectorConfig {
    // Sensitivity
    int sensitivity_level;          // ULTRA, HIGH, NORMAL, LOW
    int detection_modes;            // Bitmask of active modes
    
    // Channel weights
    float thermal_weight;           // Weight for temp spike channel
    float gradient_weight;          // Weight for gradient channel
    float melt_weight;              // Weight for melt channel
    float correlation_weight;       // Weight for multi-aromatic
    
    // Thresholds (will be scaled by sensitivity)
    float base_thermal_threshold;   // K
    float base_gradient_threshold;  // K/Å
    float base_melt_threshold;      // fraction/step
    float base_correlation_threshold; // unitless
    
    // Time constants
    float tau_fast;                 // Fast integration (ps)
    float tau_slow;                 // Slow integration (ps)
    float refractory_period;        // ps
    
    // Noise rejection
    float noise_floor;              // Minimum signal level
    float adaptation_rate;          // Background adaptation
};

// ============================================================================
// DEVICE FUNCTIONS: SENSITIVE DETECTION
// ============================================================================

// Get effective threshold based on sensitivity level
__device__ __forceinline__ float get_effective_threshold(
    float base_threshold,
    int sensitivity_level
) {
    return base_threshold * SENSITIVITY_SCALES[sensitivity_level];
}

// Update thermal detection channel
__device__ __forceinline__ float update_thermal_channel(
    MultiModalVoxelState* state,
    float current_temp,
    float dt,
    float tau,
    float threshold,
    int sensitivity_level
) {
    // Calculate temperature change from baseline
    float delta_T = current_temp - state->thermal_baseline;
    
    // Slowly adapt baseline (to reject DC offset)
    state->thermal_baseline += (current_temp - state->thermal_baseline) * 0.001f * dt;
    
    // Get effective threshold
    float eff_threshold = get_effective_threshold(threshold, sensitivity_level);
    
    // Normalize input
    float input = fabsf(delta_T) / eff_threshold;
    
    // Leaky integration
    float decay = expf(-dt / tau);
    state->thermal_potential = state->thermal_potential * decay + input * (1.0f - decay);
    
    return state->thermal_potential;
}

// Update gradient detection channel
__device__ __forceinline__ float update_gradient_channel(
    MultiModalVoxelState* state,
    float3 gradient,
    float dt,
    float tau,
    float threshold,
    int sensitivity_level
) {
    float grad_mag = sqrtf(gradient.x*gradient.x + gradient.y*gradient.y + gradient.z*gradient.z);
    
    // Get effective threshold
    float eff_threshold = get_effective_threshold(threshold, sensitivity_level);
    
    // Normalize input
    float input = grad_mag / eff_threshold;
    
    // Leaky integration
    float decay = expf(-dt / tau);
    state->gradient_potential = state->gradient_potential * decay + input * (1.0f - decay);
    
    // Store gradient direction for tracking heat flow
    if (grad_mag > 1e-10f) {
        state->last_gradient_dir.x = gradient.x / grad_mag;
        state->last_gradient_dir.y = gradient.y / grad_mag;
        state->last_gradient_dir.z = gradient.z / grad_mag;
    }
    
    return state->gradient_potential;
}

// Update melt detection channel
__device__ __forceinline__ float update_melt_channel(
    MultiModalVoxelState* state,
    float current_ice,
    float melt_rate,
    float dt,
    float tau,
    float threshold,
    int sensitivity_level
) {
    // Get effective threshold
    float eff_threshold = get_effective_threshold(threshold, sensitivity_level);
    
    // Melt rate as input
    float input = melt_rate / eff_threshold;
    
    // Also respond to ice fraction change
    float ice_change = fabsf(state->ice_fraction - current_ice);
    input += ice_change / eff_threshold;
    
    // Update ice fraction
    state->ice_fraction = current_ice;
    
    // Leaky integration
    float decay = expf(-dt / tau);
    state->melt_potential = state->melt_potential * decay + input * (1.0f - decay);
    
    return state->melt_potential;
}

// Combine channels and check for spike
__device__ __forceinline__ bool check_multimodal_spike(
    MultiModalVoxelState* state,
    const SensitiveDetectorConfig* config,
    float dt,
    int current_time
) {
    // Check refractory period
    if (state->in_refractory) {
        if (current_time - state->last_spike_time > config->refractory_period) {
            state->in_refractory = false;
        } else {
            return false;
        }
    }
    
    // Combine potentials with weights
    state->combined_potential = 0.0f;
    
    if (config->detection_modes & DETECT_TEMP_SPIKE) {
        state->combined_potential += config->thermal_weight * state->thermal_potential;
    }
    
    if (config->detection_modes & DETECT_TEMP_GRADIENT) {
        state->combined_potential += config->gradient_weight * state->gradient_potential;
    }
    
    if (config->detection_modes & DETECT_MELT_WAVE) {
        state->combined_potential += config->melt_weight * state->melt_potential;
    }
    
    // Noise floor rejection
    if (state->combined_potential < config->noise_floor) {
        return false;
    }
    
    // Calculate SNR
    state->signal_to_noise = state->combined_potential / (config->noise_floor + 1e-10f);
    
    // Spike threshold is 1.0 (normalized)
    if (state->combined_potential >= 1.0f) {
        // Calculate confidence based on how much over threshold
        state->confidence = fminf(1.0f, state->combined_potential);
        
        // Reset
        state->thermal_potential = 0.0f;
        state->gradient_potential = 0.0f;
        state->melt_potential = 0.0f;
        state->combined_potential = 0.0f;
        
        state->last_spike_time = current_time;
        state->in_refractory = true;
        
        // Count spikes per channel
        if (state->thermal_potential > 0.3f) state->thermal_spike_count++;
        if (state->gradient_potential > 0.3f) state->gradient_spike_count++;
        if (state->melt_potential > 0.3f) state->melt_spike_count++;
        
        return true;
    }
    
    return false;
}

// ============================================================================
// KERNEL: ULTRA-SENSITIVE THERMAL DETECTION
// ============================================================================

__global__ void sensitive_thermal_detection(
    // Thermal field
    const float* __restrict__ temperatures,
    const float3* __restrict__ temp_gradients,
    const float* __restrict__ ice_fractions,
    const float* __restrict__ melt_rates,
    
    // Detector state (one per voxel)
    MultiModalVoxelState* detector_states,
    int n_voxels,
    
    // Configuration
    SensitiveDetectorConfig config,
    
    // Timing
    float dt,
    int current_time,
    
    // Output
    int* spike_indices,
    float* spike_confidences,
    int* n_spikes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_voxels) return;
    
    MultiModalVoxelState* state = &detector_states[idx];
    
    // Update each channel
    update_thermal_channel(
        state,
        temperatures[idx],
        dt,
        config.tau_fast,
        config.base_thermal_threshold,
        config.sensitivity_level
    );
    
    update_gradient_channel(
        state,
        temp_gradients[idx],
        dt,
        config.tau_fast,
        config.base_gradient_threshold,
        config.sensitivity_level
    );
    
    update_melt_channel(
        state,
        ice_fractions[idx],
        melt_rates[idx],
        dt,
        config.tau_slow,  // Melting is slower process
        config.base_melt_threshold,
        config.sensitivity_level
    );
    
    // Check for spike
    bool spiked = check_multimodal_spike(state, &config, dt, current_time);
    
    if (spiked) {
        int spike_idx = atomicAdd(n_spikes, 1);
        if (spike_idx < n_voxels) {  // Bounds check
            spike_indices[spike_idx] = idx;
            spike_confidences[spike_idx] = state->confidence;
        }
    }
}

// ============================================================================
// KERNEL: POCKET CORRELATION DETECTION
// ============================================================================

__global__ void detect_correlated_response(
    // Aromatic thermal responses
    const float* __restrict__ aromatic_temp_rises,
    const float3* __restrict__ aromatic_positions,
    const int* __restrict__ aromatic_types,  // TRP, TYR, PHE
    int n_aromatics,
    
    // Timing of responses
    const int* __restrict__ response_times,
    int current_time,
    int time_window,  // ps - window for correlation
    
    // Configuration
    SensitiveDetectorConfig config,
    
    // Output
    float3* pocket_centers,
    float* pocket_confidences,
    int* n_pockets
) {
    // Single-threaded for now (could parallelize)
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    float eff_threshold = get_effective_threshold(
        config.base_correlation_threshold,
        config.sensitivity_level
    );
    
    // For each aromatic, look for nearby responding aromatics
    for (int i = 0; i < n_aromatics; i++) {
        // Skip if didn't respond recently
        if (current_time - response_times[i] > time_window) continue;
        if (aromatic_temp_rises[i] < 0.01f) continue;  // Minimal response
        
        // Find cluster of responding aromatics
        float3 center = aromatic_positions[i];
        float total_weight = aromatic_temp_rises[i];
        int cluster_size = 1;
        
        float responses[8];
        responses[0] = aromatic_temp_rises[i];
        
        for (int j = i + 1; j < n_aromatics && cluster_size < 8; j++) {
            // Check timing
            if (abs(response_times[j] - response_times[i]) > time_window) continue;
            if (aromatic_temp_rises[j] < 0.005f) continue;
            
            // Check distance
            float3 diff = make_float3(
                aromatic_positions[j].x - center.x,
                aromatic_positions[j].y - center.y,
                aromatic_positions[j].z - center.z
            );
            float dist = sqrtf(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);
            
            if (dist < 15.0f) {  // Within pocket-like distance
                // Add to cluster
                float w = aromatic_temp_rises[j];
                center.x = (center.x * total_weight + aromatic_positions[j].x * w) / (total_weight + w);
                center.y = (center.y * total_weight + aromatic_positions[j].y * w) / (total_weight + w);
                center.z = (center.z * total_weight + aromatic_positions[j].z * w) / (total_weight + w);
                total_weight += w;
                
                responses[cluster_size] = aromatic_temp_rises[j];
                cluster_size++;
            }
        }
        
        // Need at least 2 aromatics for pocket
        if (cluster_size >= 2) {
            // Calculate correlation
            float mean = 0.0f;
            for (int k = 0; k < cluster_size; k++) {
                mean += responses[k];
            }
            mean /= cluster_size;
            
            float variance = 0.0f;
            for (int k = 0; k < cluster_size; k++) {
                float diff = responses[k] - mean;
                variance += diff * diff;
            }
            variance /= cluster_size;
            
            // Correlation metric: low variance = high correlation
            float correlation = 1.0f / (1.0f + variance / (mean * mean + 1e-10f));
            
            // Boost for more aromatics
            float size_bonus = fminf(1.0f, (float)cluster_size / 4.0f);
            
            float confidence = correlation * size_bonus;
            
            if (confidence >= eff_threshold) {
                int pocket_idx = atomicAdd(n_pockets, 1);
                if (pocket_idx < 32) {  // Max pockets
                    pocket_centers[pocket_idx] = center;
                    pocket_confidences[pocket_idx] = confidence;
                }
            }
        }
    }
}

// ============================================================================
// HOST: DEFAULT CONFIGURATION
// ============================================================================

__host__ SensitiveDetectorConfig get_cryo_config() {
    SensitiveDetectorConfig config;
    
    // ULTRA sensitive for cryo - we want to detect EVERYTHING
    config.sensitivity_level = SENSITIVITY_ULTRA;
    config.detection_modes = DETECT_ALL;
    
    // Equal weights for all channels in cryo
    config.thermal_weight = 1.0f;
    config.gradient_weight = 0.8f;
    config.melt_weight = 1.5f;  // Melting is key signal
    config.correlation_weight = 2.0f;  // Pocket signature is most important
    
    // Base thresholds (will be scaled by sensitivity)
    config.base_thermal_threshold = 0.1f;   // K
    config.base_gradient_threshold = 0.01f; // K/Å
    config.base_melt_threshold = 0.001f;    // fraction/step
    config.base_correlation_threshold = 0.3f;
    
    // Time constants
    config.tau_fast = 5.0f;     // ps - fast for thermal
    config.tau_slow = 20.0f;    // ps - slower for melting
    config.refractory_period = 2.0f;  // ps - short refractory
    
    // Noise rejection (low floor in cryo - less thermal noise)
    config.noise_floor = 0.01f;
    config.adaptation_rate = 0.001f;
    
    return config;
}

__host__ SensitiveDetectorConfig get_thermal_ramp_config() {
    SensitiveDetectorConfig config = get_cryo_config();
    
    // HIGH sensitivity during ramp
    config.sensitivity_level = SENSITIVITY_HIGH;
    
    // Thermal becomes more important as temp rises
    config.thermal_weight = 1.5f;
    config.melt_weight = 0.5f;  // Less melting signal as ice disappears
    
    // Higher noise floor (more thermal noise)
    config.noise_floor = 0.05f;
    
    return config;
}

__host__ SensitiveDetectorConfig get_dig_config() {
    SensitiveDetectorConfig config;
    
    // NORMAL sensitivity for dig - focused detection
    config.sensitivity_level = SENSITIVITY_NORMAL;
    config.detection_modes = DETECT_TEMP_SPIKE | DETECT_CORRELATION;
    
    // Focus on thermal and correlation
    config.thermal_weight = 1.0f;
    config.gradient_weight = 0.3f;
    config.melt_weight = 0.0f;  // No ice at 300K
    config.correlation_weight = 2.0f;
    
    config.base_thermal_threshold = 0.5f;
    config.base_gradient_threshold = 0.1f;
    config.base_melt_threshold = 0.0f;
    config.base_correlation_threshold = 0.4f;
    
    config.tau_fast = 10.0f;
    config.tau_slow = 50.0f;
    config.refractory_period = 5.0f;
    
    config.noise_floor = 0.1f;
    config.adaptation_rate = 0.01f;
    
    return config;
}

#endif // PRISM_SENSITIVE_DETECTOR_CUH
