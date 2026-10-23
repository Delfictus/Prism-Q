/**
 * @file nhs_active_sensing.cuh
 * @brief PRISM-NHS Active Sensing Module
 * 
 * Implements closed-loop molecular stethoscope:
 * 1. Coherent multi-site UV excitation (vibrational interferometry)
 * 2. Spike sequence detection (STDP-like temporal patterns)
 * 3. Lateral inhibition (contrast enhancement)
 * 4. Resonance detection (soft mode identification)
 * 5. Adaptive probe selection (feedback control)
 *
 * THEORY:
 * Cryptic binding sites are metastable - small perturbations trigger large
 * conformational changes. UV excitation of aromatics provides spatially-targeted
 * energy injection. Neuromorphic readout detects the response patterns.
 *
 * The key innovation: vibrational interferometry on protein conformational modes.
 * By exciting aromatics with controlled phase relationships, we can:
 * - Amplify signals along opening pathways (constructive interference)
 * - Cancel noise from irrelevant regions (destructive interference)
 * - Identify soft modes via resonance sweeps
 *
 * REFERENCES:
 * - Pump-probe spectroscopy (Zewail, femtochemistry)
 * - Spike-timing dependent plasticity (Bi & Poo, 1998)
 * - Lateral inhibition in retinal processing (Hartline, 1949)
 */

#ifndef NHS_ACTIVE_SENSING_CUH
#define NHS_ACTIVE_SENSING_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>

// ============================================================================
// ACTIVE SENSING CONFIGURATION
// ============================================================================

// Coherent probing
#define MAX_PROBE_GROUPS 8           // Max aromatic groups for coherent excitation
#define MAX_AROMATICS_PER_GROUP 32   // Max aromatics per group
#define MAX_PHASE_DELAY_FS 1000.0f   // Max phase delay in femtoseconds

// Spike sequence detection
#define SPIKE_HISTORY_LENGTH 64      // Ring buffer size per voxel
#define MAX_SEQUENCE_LENGTH 8        // Max spikes in detected sequence
#define CAUSALITY_WINDOW_PS 10.0f    // Time window for causal sequences

// Lateral inhibition
#define INHIBITION_RADIUS 2          // Voxels (in each direction)
#define INHIBITION_STRENGTH 0.3f     // Fraction of potential suppressed
#define INHIBITION_DECAY_TAU 0.5f    // ps - decay time constant

// Resonance detection
#define MIN_PROBE_FREQ_THZ 0.1f      // Minimum probe frequency
#define MAX_PROBE_FREQ_THZ 10.0f     // Maximum probe frequency
#define FREQ_SWEEP_STEPS 100         // Resolution of frequency sweep
#define RESONANCE_QUALITY_THRESHOLD 2.0f  // Q factor for resonance detection

// Feedback control
#define ADAPTATION_RATE 0.1f         // Learning rate for probe selection
#define EXPLORATION_EPSILON 0.1f     // Random exploration probability

// ============================================================================
// DATA STRUCTURES: COHERENT PROBING
// ============================================================================

/**
 * @brief Group of aromatics for coherent excitation
 * 
 * Aromatics in a group are excited together (or with phase delays)
 * to create vibrational interference patterns.
 */
struct AromaticGroup {
    int aromatic_indices[MAX_AROMATICS_PER_GROUP];  // Indices into UV target array
    int n_aromatics;                                 // Number in this group
    float3 centroid;                                 // Geometric center
    float total_absorption;                          // Sum of absorption strengths
};

/**
 * @brief Coherent probe pattern for vibrational interferometry
 *
 * Defines which aromatic groups to excite and with what timing.
 * Phase delays create interference patterns that amplify or cancel
 * depending on protein mechanical pathways.
 */
struct CoherentProbe {
    AromaticGroup groups[MAX_PROBE_GROUPS];  // Aromatic groups
    int n_groups;                             // Active groups
    float phase_delays_fs[MAX_PROBE_GROUPS];  // Relative timing (femtoseconds)
    float energy_per_group[MAX_PROBE_GROUPS]; // Energy allocation
    float total_energy;                       // Total burst energy (kcal/mol)
    
    // Probe metadata
    int probe_id;                             // Unique identifier
    int probe_type;                           // 0=hinge_A, 1=hinge_B, 2=pocket_lining, 3=control
    float expected_response;                  // Predicted LIF response (for learning)
};

/**
 * @brief Differential probe pair for comparison
 *
 * Two probes designed to excite opposite sides of a suspected hinge.
 * Large differential response indicates mechanical asymmetry → cryptic site.
 */
struct DifferentialProbePair {
    CoherentProbe probe_A;     // One side of hinge
    CoherentProbe probe_B;     // Opposite side
    float differential_score;  // |response_A - response_B| / (response_A + response_B)
    float confidence;          // Statistical confidence
    int n_trials;              // Number of comparison trials
};

// ============================================================================
// DATA STRUCTURES: NEUROMORPHIC PROCESSING
// ============================================================================

/**
 * @brief Spike event with precise timing
 */
struct TimestampedSpike {
    int voxel_idx;           // Which voxel spiked
    float timestamp_ps;      // When (picoseconds since start)
    float intensity;         // Spike magnitude
    int probe_id;            // Which probe was active (for correlation)
};

/**
 * @brief Spike history ring buffer for a voxel
 */
struct VoxelSpikeHistory {
    TimestampedSpike spikes[SPIKE_HISTORY_LENGTH];
    int head;                // Current write position
    int count;               // Number of valid entries
    float last_spike_time;   // For refractory period
};

/**
 * @brief Spike sequence detector (STDP-like)
 *
 * Detects causal spike sequences across voxels.
 * True cryptic site opening produces ordered sequences (water leaves in order).
 * Random fluctuations produce unordered spikes.
 */
struct SpikeSequenceDetector {
    int voxel_sequence[MAX_SEQUENCE_LENGTH];  // Expected voxel order
    int sequence_length;                       // Length of sequence
    float max_inter_spike_interval_ps;         // Causality window
    
    // Detection state
    int current_position;                      // How far through sequence
    float sequence_start_time;                 // When first spike occurred
    float accumulated_score;                   // Evidence for this sequence
    int detection_count;                       // Times fully detected
    
    // Learning
    float weight;                              // Importance weight (learned)
};

/**
 * @brief Lateral inhibition state for a voxel
 */
struct LateralInhibitionState {
    float inhibition_level;      // Current inhibition (0-1)
    float last_update_time;      // For decay calculation
    int inhibiting_neighbors[27]; // Which neighbors are inhibiting this voxel
    int n_inhibitors;
};

/**
 * @brief Resonance detector for soft mode identification
 *
 * Soft modes (low-frequency collective motions) indicate regions
 * primed for conformational change. We detect them by sweeping
 * probe frequency and measuring response amplitude.
 */
struct ResonanceDetector {
    float probe_frequency_thz;    // Current probe frequency
    float response_amplitude;     // Measured response at this frequency
    float response_phase;         // Phase of response relative to probe
    
    // Accumulated spectrum
    float frequency_spectrum[FREQ_SWEEP_STEPS];   // Response vs frequency
    float phase_spectrum[FREQ_SWEEP_STEPS];       // Phase vs frequency
    int sample_counts[FREQ_SWEEP_STEPS];          // Samples per frequency
    
    // Detected resonances
    float resonance_frequencies[8];  // Identified soft mode frequencies
    float resonance_amplitudes[8];   // Peak amplitudes
    float quality_factors[8];        // Q = f0 / Δf (sharpness)
    int n_resonances;
};

/**
 * @brief Feedback controller for adaptive probe selection
 */
struct AdaptiveProbeController {
    // Probe effectiveness scores (learned)
    float probe_scores[MAX_PROBE_GROUPS * MAX_PROBE_GROUPS];  // Pairwise combinations
    int probe_trial_counts[MAX_PROBE_GROUPS * MAX_PROBE_GROUPS];
    
    // Current state
    int current_probe_idx;
    float exploration_temperature;   // For softmax selection
    
    // Performance tracking
    float cumulative_reward;
    int total_trials;
    float best_score;
    int best_probe_idx;
};

// ============================================================================
// DEVICE FUNCTIONS: COHERENT PROBING
// ============================================================================

/**
 * @brief Compute centroid of aromatic group
 */
__device__ void compute_group_centroid(
    AromaticGroup* group,
    const float3* positions,
    const int* aromatic_atom_indices,  // [n_aromatics * 16]
    const int* aromatic_n_atoms,       // [n_aromatics]
    int n_aromatics
) {
    float3 centroid = make_float3(0.0f, 0.0f, 0.0f);
    int total_atoms = 0;
    
    for (int g = 0; g < group->n_aromatics; g++) {
        int arom_idx = group->aromatic_indices[g];
        if (arom_idx < 0 || arom_idx >= n_aromatics) continue;
        
        int n_atoms = aromatic_n_atoms[arom_idx];
        for (int a = 0; a < n_atoms && a < 16; a++) {
            int atom_idx = aromatic_atom_indices[arom_idx * 16 + a];
            if (atom_idx >= 0) {
                centroid.x += positions[atom_idx].x;
                centroid.y += positions[atom_idx].y;
                centroid.z += positions[atom_idx].z;
                total_atoms++;
            }
        }
    }
    
    if (total_atoms > 0) {
        centroid.x /= total_atoms;
        centroid.y /= total_atoms;
        centroid.z /= total_atoms;
    }
    
    group->centroid = centroid;
}

/**
 * @brief Apply coherent UV excitation to aromatic groups
 *
 * Excites groups with specified phase delays to create vibrational
 * interference patterns. Energy is deposited as velocity kicks
 * in the direction of ring normals.
 */
__device__ void apply_coherent_excitation(
    const CoherentProbe* probe,
    float3* velocities,
    const float* masses,
    const float3* ring_normals,
    const int* aromatic_atom_indices,
    const int* aromatic_n_atoms,
    float current_time_fs,
    float dt_fs,
    curandState* rng_state,
    int n_aromatics,
    int n_atoms
) {
    // Check each group for excitation
    for (int g = 0; g < probe->n_groups; g++) {
        const AromaticGroup* group = &probe->groups[g];
        float phase_delay = probe->phase_delays_fs[g];
        
        // Check if this group should be excited at current time
        float group_time = current_time_fs - phase_delay;
        if (group_time < 0.0f || group_time >= dt_fs) continue;
        
        float energy = probe->energy_per_group[g];
        
        // Excite all aromatics in group
        for (int a = 0; a < group->n_aromatics; a++) {
            int arom_idx = group->aromatic_indices[a];
            if (arom_idx < 0 || arom_idx >= n_aromatics) continue;
            
            float3 normal = ring_normals[arom_idx];
            int n_ring_atoms = aromatic_n_atoms[arom_idx];
            
            // Distribute energy among ring atoms
            float energy_per_atom = energy / fmaxf(1.0f, (float)n_ring_atoms);
            
            for (int r = 0; r < n_ring_atoms && r < 16; r++) {
                int atom_idx = aromatic_atom_indices[arom_idx * 16 + r];
                if (atom_idx < 0 || atom_idx >= n_atoms) continue;
                
                float mass = masses[atom_idx];
                if (mass <= 0.0f) continue;
                
                // v = sqrt(2E/m) in direction of ring normal
                // Add small random component for realistic energy distribution
                float v_mag = sqrtf(2.0f * energy_per_atom / mass);
                float rand_factor = 0.8f + 0.4f * curand_uniform(rng_state);
                
                // Kick perpendicular to ring (along normal)
                atomicAdd(&velocities[atom_idx].x, v_mag * normal.x * rand_factor);
                atomicAdd(&velocities[atom_idx].y, v_mag * normal.y * rand_factor);
                atomicAdd(&velocities[atom_idx].z, v_mag * normal.z * rand_factor);
            }
        }
    }
}

// ============================================================================
// DEVICE FUNCTIONS: SPIKE SEQUENCE DETECTION
// ============================================================================

/**
 * @brief Record spike in voxel history
 */
__device__ void record_spike(
    VoxelSpikeHistory* history,
    int voxel_idx,
    float timestamp_ps,
    float intensity,
    int probe_id
) {
    // Refractory period check
    if (timestamp_ps - history->last_spike_time < 0.1f) return;  // 0.1 ps refractory
    
    int slot = history->head;
    history->spikes[slot].voxel_idx = voxel_idx;
    history->spikes[slot].timestamp_ps = timestamp_ps;
    history->spikes[slot].intensity = intensity;
    history->spikes[slot].probe_id = probe_id;
    
    history->head = (history->head + 1) % SPIKE_HISTORY_LENGTH;
    history->count = min(history->count + 1, SPIKE_HISTORY_LENGTH);
    history->last_spike_time = timestamp_ps;
}

/**
 * @brief Check if spike sequence is detected
 *
 * Returns score (0-1) indicating how well recent spikes match expected sequence.
 * Higher scores indicate true cryptic site opening events.
 */
__device__ float check_sequence_detection(
    SpikeSequenceDetector* detector,
    const VoxelSpikeHistory* histories,  // [n_voxels]
    float current_time_ps,
    int n_voxels
) {
    if (detector->sequence_length < 2) return 0.0f;
    
    // Look for sequence start (first voxel spiking)
    int first_voxel = detector->voxel_sequence[0];
    if (first_voxel < 0 || first_voxel >= n_voxels) return 0.0f;
    
    const VoxelSpikeHistory* first_hist = &histories[first_voxel];
    
    // Find most recent spike from first voxel within window
    float best_score = 0.0f;
    
    for (int s = 0; s < first_hist->count; s++) {
        int idx = (first_hist->head - 1 - s + SPIKE_HISTORY_LENGTH) % SPIKE_HISTORY_LENGTH;
        const TimestampedSpike* start_spike = &first_hist->spikes[idx];
        
        // Check if within temporal window
        if (current_time_ps - start_spike->timestamp_ps > 
            detector->max_inter_spike_interval_ps * detector->sequence_length) {
            continue;
        }
        
        // Try to match rest of sequence
        float last_time = start_spike->timestamp_ps;
        float sequence_score = 1.0f;
        int matched = 1;
        
        for (int seq_pos = 1; seq_pos < detector->sequence_length; seq_pos++) {
            int expected_voxel = detector->voxel_sequence[seq_pos];
            if (expected_voxel < 0 || expected_voxel >= n_voxels) break;
            
            const VoxelSpikeHistory* hist = &histories[expected_voxel];
            bool found = false;
            
            // Look for spike from this voxel after last_time
            for (int h = 0; h < hist->count; h++) {
                int hidx = (hist->head - 1 - h + SPIKE_HISTORY_LENGTH) % SPIKE_HISTORY_LENGTH;
                const TimestampedSpike* spike = &hist->spikes[hidx];
                
                float dt = spike->timestamp_ps - last_time;
                
                // Must be after previous spike, within causality window
                if (dt > 0.0f && dt < detector->max_inter_spike_interval_ps) {
                    // Score based on timing precision
                    float timing_score = expf(-dt / (detector->max_inter_spike_interval_ps * 0.5f));
                    sequence_score *= timing_score;
                    last_time = spike->timestamp_ps;
                    matched++;
                    found = true;
                    break;
                }
            }
            
            if (!found) {
                sequence_score *= 0.1f;  // Penalty for missing spike
            }
        }
        
        // Final score weighted by completeness
        float completeness = (float)matched / detector->sequence_length;
        float final_score = sequence_score * completeness * completeness;
        
        if (final_score > best_score) {
            best_score = final_score;
        }
    }
    
    return best_score;
}

// ============================================================================
// DEVICE FUNCTIONS: LATERAL INHIBITION
// ============================================================================

/**
 * @brief Apply lateral inhibition from spiking voxel
 *
 * When a voxel spikes, it inhibits its neighbors, enhancing contrast
 * at the boundaries of dewetting regions (like retinal ganglion cells).
 */
__device__ void apply_lateral_inhibition(
    float* lif_potentials,
    LateralInhibitionState* inhibition_states,
    int spiked_voxel,
    int grid_dim,
    float current_time_ps
) {
    int total_voxels = grid_dim * grid_dim * grid_dim;
    
    // Decode voxel position
    int vz = spiked_voxel / (grid_dim * grid_dim);
    int vy = (spiked_voxel / grid_dim) % grid_dim;
    int vx = spiked_voxel % grid_dim;
    
    // Inhibit neighbors within radius
    for (int dz = -INHIBITION_RADIUS; dz <= INHIBITION_RADIUS; dz++) {
        int nz = vz + dz;
        if (nz < 0 || nz >= grid_dim) continue;
        
        for (int dy = -INHIBITION_RADIUS; dy <= INHIBITION_RADIUS; dy++) {
            int ny = vy + dy;
            if (ny < 0 || ny >= grid_dim) continue;
            
            for (int dx = -INHIBITION_RADIUS; dx <= INHIBITION_RADIUS; dx++) {
                int nx = vx + dx;
                if (nx < 0 || nx >= grid_dim) continue;
                
                // Skip self
                if (dx == 0 && dy == 0 && dz == 0) continue;
                
                int neighbor_idx = nx + ny * grid_dim + nz * grid_dim * grid_dim;
                
                // Distance-weighted inhibition
                float dist = sqrtf((float)(dx*dx + dy*dy + dz*dz));
                float inhibition = INHIBITION_STRENGTH / dist;
                
                // Apply inhibition
                atomicAdd(&inhibition_states[neighbor_idx].inhibition_level, inhibition);
                
                // Suppress potential
                float suppression = lif_potentials[neighbor_idx] * inhibition;
                atomicAdd(&lif_potentials[neighbor_idx], -suppression);
            }
        }
    }
}

/**
 * @brief Decay lateral inhibition over time
 */
__device__ void decay_inhibition(
    LateralInhibitionState* state,
    float current_time_ps,
    float dt_ps
) {
    float decay = expf(-dt_ps / INHIBITION_DECAY_TAU);
    state->inhibition_level *= decay;
    state->last_update_time = current_time_ps;
}

// ============================================================================
// DEVICE FUNCTIONS: RESONANCE DETECTION
// ============================================================================

/**
 * @brief Update resonance detector with current response
 */
__device__ void update_resonance_detector(
    ResonanceDetector* detector,
    float probe_frequency_thz,
    float response_amplitude,
    float response_phase
) {
    detector->probe_frequency_thz = probe_frequency_thz;
    detector->response_amplitude = response_amplitude;
    detector->response_phase = response_phase;
    
    // Map frequency to spectrum bin
    float freq_range = MAX_PROBE_FREQ_THZ - MIN_PROBE_FREQ_THZ;
    int bin = (int)((probe_frequency_thz - MIN_PROBE_FREQ_THZ) / freq_range * FREQ_SWEEP_STEPS);
    bin = max(0, min(bin, FREQ_SWEEP_STEPS - 1));
    
    // Running average update
    int n = detector->sample_counts[bin];
    detector->frequency_spectrum[bin] = 
        (detector->frequency_spectrum[bin] * n + response_amplitude) / (n + 1);
    detector->phase_spectrum[bin] = 
        (detector->phase_spectrum[bin] * n + response_phase) / (n + 1);
    detector->sample_counts[bin]++;
}

/**
 * @brief Find resonance peaks in accumulated spectrum
 */
__device__ void find_resonances(
    ResonanceDetector* detector
) {
    detector->n_resonances = 0;
    
    float freq_range = MAX_PROBE_FREQ_THZ - MIN_PROBE_FREQ_THZ;
    float freq_step = freq_range / FREQ_SWEEP_STEPS;
    
    // Simple peak detection
    for (int i = 2; i < FREQ_SWEEP_STEPS - 2; i++) {
        float amp = detector->frequency_spectrum[i];
        
        // Check if local maximum
        if (amp > detector->frequency_spectrum[i-1] &&
            amp > detector->frequency_spectrum[i-2] &&
            amp > detector->frequency_spectrum[i+1] &&
            amp > detector->frequency_spectrum[i+2]) {
            
            // Estimate quality factor (Q = f0 / FWHM)
            float half_max = amp * 0.5f;
            int left = i, right = i;
            while (left > 0 && detector->frequency_spectrum[left] > half_max) left--;
            while (right < FREQ_SWEEP_STEPS-1 && detector->frequency_spectrum[right] > half_max) right++;
            
            float fwhm = (right - left) * freq_step;
            float f0 = MIN_PROBE_FREQ_THZ + i * freq_step;
            float Q = (fwhm > 0.001f) ? f0 / fwhm : 0.0f;
            
            // Store if significant resonance
            if (Q > RESONANCE_QUALITY_THRESHOLD && detector->n_resonances < 8) {
                int r = detector->n_resonances;
                detector->resonance_frequencies[r] = f0;
                detector->resonance_amplitudes[r] = amp;
                detector->quality_factors[r] = Q;
                detector->n_resonances++;
            }
        }
    }
}

// ============================================================================
// DEVICE FUNCTIONS: ADAPTIVE CONTROL
// ============================================================================

/**
 * @brief Update probe scores based on response
 */
__device__ void update_probe_score(
    AdaptiveProbeController* controller,
    int probe_idx,
    float response_score
) {
    if (probe_idx < 0 || probe_idx >= MAX_PROBE_GROUPS * MAX_PROBE_GROUPS) return;
    
    int n = controller->probe_trial_counts[probe_idx];
    controller->probe_scores[probe_idx] = 
        (controller->probe_scores[probe_idx] * n + response_score) / (n + 1);
    controller->probe_trial_counts[probe_idx]++;
    
    controller->cumulative_reward += response_score;
    controller->total_trials++;
    
    if (response_score > controller->best_score) {
        controller->best_score = response_score;
        controller->best_probe_idx = probe_idx;
    }
}

/**
 * @brief Select next probe using softmax with exploration
 */
__device__ int select_next_probe(
    AdaptiveProbeController* controller,
    curandState* rng_state,
    int n_probes
) {
    // Exploration: random probe
    if (curand_uniform(rng_state) < EXPLORATION_EPSILON) {
        return curand(rng_state) % n_probes;
    }
    
    // Exploitation: softmax selection based on scores
    float max_score = -1e30f;
    for (int i = 0; i < n_probes; i++) {
        if (controller->probe_scores[i] > max_score) {
            max_score = controller->probe_scores[i];
        }
    }
    
    float sum_exp = 0.0f;
    float exp_scores[64];  // Assume max 64 probes
    for (int i = 0; i < n_probes && i < 64; i++) {
        exp_scores[i] = expf((controller->probe_scores[i] - max_score) / 
                             controller->exploration_temperature);
        sum_exp += exp_scores[i];
    }
    
    float rand_val = curand_uniform(rng_state) * sum_exp;
    float cumsum = 0.0f;
    for (int i = 0; i < n_probes && i < 64; i++) {
        cumsum += exp_scores[i];
        if (rand_val <= cumsum) {
            return i;
        }
    }
    
    return n_probes - 1;
}

// ============================================================================
// AGGREGATE RESPONSE METRICS
// ============================================================================

/**
 * @brief Comprehensive response to a probe
 */
struct ProbeResponse {
    // Basic metrics
    int total_spikes;                    // Raw spike count
    float mean_intensity;                // Average spike intensity
    float spatial_extent;                // How spread out were spikes (voxels)
    
    // Temporal metrics
    float onset_latency_ps;              // Time to first spike
    float peak_latency_ps;               // Time to maximum activity
    float duration_ps;                   // Duration of response
    
    // Sequence metrics
    float sequence_score;                // Best sequence detection score
    int sequences_detected;              // Number of complete sequences
    
    // Spatial metrics
    float3 response_centroid;            // Center of spike activity
    float response_anisotropy;           // Directional bias (0=isotropic, 1=linear)
    
    // Correlation with probe
    float probe_correlation;             // Response aligned with probe geometry
};

/**
 * @brief Compute comprehensive response metrics
 */
__device__ void compute_probe_response(
    ProbeResponse* response,
    const VoxelSpikeHistory* histories,
    const SpikeSequenceDetector* detectors,
    int n_detectors,
    const CoherentProbe* probe,
    float probe_start_time_ps,
    float analysis_window_ps,
    int grid_dim,
    float grid_spacing,
    float3 grid_origin
) {
    int total_voxels = grid_dim * grid_dim * grid_dim;
    
    // Initialize
    response->total_spikes = 0;
    response->mean_intensity = 0.0f;
    response->spatial_extent = 0.0f;
    response->onset_latency_ps = analysis_window_ps;
    response->peak_latency_ps = 0.0f;
    response->duration_ps = 0.0f;
    response->sequence_score = 0.0f;
    response->sequences_detected = 0;
    response->response_centroid = make_float3(0.0f, 0.0f, 0.0f);
    response->response_anisotropy = 0.0f;
    response->probe_correlation = 0.0f;
    
    float first_spike_time = probe_start_time_ps + analysis_window_ps;
    float last_spike_time = probe_start_time_ps;
    float max_intensity_time = probe_start_time_ps;
    float max_intensity = 0.0f;
    
    // Scan all voxel histories
    for (int v = 0; v < total_voxels; v++) {
        const VoxelSpikeHistory* hist = &histories[v];
        
        // Decode voxel position
        int vz = v / (grid_dim * grid_dim);
        int vy = (v / grid_dim) % grid_dim;
        int vx = v % grid_dim;
        float3 voxel_pos = make_float3(
            grid_origin.x + (vx + 0.5f) * grid_spacing,
            grid_origin.y + (vy + 0.5f) * grid_spacing,
            grid_origin.z + (vz + 0.5f) * grid_spacing
        );
        
        for (int s = 0; s < hist->count; s++) {
            int idx = (hist->head - 1 - s + SPIKE_HISTORY_LENGTH) % SPIKE_HISTORY_LENGTH;
            const TimestampedSpike* spike = &hist->spikes[idx];
            
            // Check if within analysis window
            float rel_time = spike->timestamp_ps - probe_start_time_ps;
            if (rel_time < 0.0f || rel_time > analysis_window_ps) continue;
            
            // Accumulate metrics
            response->total_spikes++;
            response->mean_intensity += spike->intensity;
            response->response_centroid.x += voxel_pos.x * spike->intensity;
            response->response_centroid.y += voxel_pos.y * spike->intensity;
            response->response_centroid.z += voxel_pos.z * spike->intensity;
            
            if (spike->timestamp_ps < first_spike_time) {
                first_spike_time = spike->timestamp_ps;
            }
            if (spike->timestamp_ps > last_spike_time) {
                last_spike_time = spike->timestamp_ps;
            }
            if (spike->intensity > max_intensity) {
                max_intensity = spike->intensity;
                max_intensity_time = spike->timestamp_ps;
            }
        }
    }
    
    // Finalize metrics
    if (response->total_spikes > 0) {
        response->mean_intensity /= response->total_spikes;
        response->response_centroid.x /= (response->mean_intensity * response->total_spikes);
        response->response_centroid.y /= (response->mean_intensity * response->total_spikes);
        response->response_centroid.z /= (response->mean_intensity * response->total_spikes);
        response->onset_latency_ps = first_spike_time - probe_start_time_ps;
        response->peak_latency_ps = max_intensity_time - probe_start_time_ps;
        response->duration_ps = last_spike_time - first_spike_time;
    }
    
    // Check sequence detectors
    for (int d = 0; d < n_detectors; d++) {
        // Note: check_sequence_detection would need to be called here
        // This is a simplified placeholder
        if (detectors[d].detection_count > 0) {
            response->sequences_detected++;
            if (detectors[d].accumulated_score > response->sequence_score) {
                response->sequence_score = detectors[d].accumulated_score;
            }
        }
    }
}

#endif // NHS_ACTIVE_SENSING_CUH
