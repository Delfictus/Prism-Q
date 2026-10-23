/**
 * @file nhs_active_sensing_kernels.cu
 * @brief PRISM-NHS Active Sensing Kernel Implementations
 *
 * GPU kernels for closed-loop molecular stethoscope:
 * - Coherent multi-site UV excitation
 * - Neuromorphic spike processing with STDP-like detection
 * - Resonance sweeping for soft mode identification
 * - Adaptive probe selection
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "nhs_active_sensing.cuh"

// ============================================================================
// KERNEL: COHERENT UV EXCITATION
// ============================================================================

/**
 * @brief Apply coherent UV excitation pattern
 *
 * Excites aromatic groups with controlled phase delays for
 * vibrational interferometry.
 */
extern "C" __global__ void apply_coherent_probe(
    // Probe specification
    const CoherentProbe* probe,
    
    // Atom state
    float3* velocities,
    const float3* positions,
    const float* masses,
    
    // Aromatic topology
    const float3* ring_normals,
    const int* aromatic_atom_indices,    // [n_aromatics * 16]
    const int* aromatic_n_atoms,         // [n_aromatics]
    
    // Timing
    float current_time_fs,
    float dt_fs,
    
    // RNG
    curandState* rng_states,
    
    // Dimensions
    int n_aromatics,
    int n_atoms
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread handles one aromatic group
    if (tid >= probe->n_groups) return;
    
    const AromaticGroup* group = &probe->groups[tid];
    float phase_delay = probe->phase_delays_fs[tid];
    
    // Check if this group should be excited now
    float group_time = current_time_fs - phase_delay;
    
    // Excitation window: group activates when its phase delay is reached
    bool should_excite = (group_time >= 0.0f && group_time < dt_fs);
    if (!should_excite) return;
    
    float energy = probe->energy_per_group[tid];
    curandState local_rng = rng_states[tid];
    
    // Excite all aromatics in this group
    for (int a = 0; a < group->n_aromatics; a++) {
        int arom_idx = group->aromatic_indices[a];
        if (arom_idx < 0 || arom_idx >= n_aromatics) continue;
        
        float3 normal = ring_normals[arom_idx];
        int n_ring_atoms = aromatic_n_atoms[arom_idx];
        
        // Energy per atom
        float energy_per_atom = energy / fmaxf(1.0f, (float)(n_ring_atoms * group->n_aromatics));
        
        for (int r = 0; r < n_ring_atoms && r < 16; r++) {
            int atom_idx = aromatic_atom_indices[arom_idx * 16 + r];
            if (atom_idx < 0 || atom_idx >= n_atoms) continue;
            
            float mass = masses[atom_idx];
            if (mass <= 0.0f) continue;
            
            // Velocity kick: v = sqrt(2E/m)
            float v_mag = sqrtf(2.0f * energy_per_atom / mass);
            
            // Add stochastic component for realistic distribution
            float rand_x = 0.8f + 0.4f * curand_uniform(&local_rng);
            float rand_y = 0.8f + 0.4f * curand_uniform(&local_rng);
            float rand_z = 0.8f + 0.4f * curand_uniform(&local_rng);
            
            // Apply kick along ring normal (perpendicular to aromatic plane)
            atomicAdd(&velocities[atom_idx].x, v_mag * normal.x * rand_x);
            atomicAdd(&velocities[atom_idx].y, v_mag * normal.y * rand_y);
            atomicAdd(&velocities[atom_idx].z, v_mag * normal.z * rand_z);
        }
    }
    
    // Save RNG state
    rng_states[tid] = local_rng;
}

// ============================================================================
// KERNEL: ENHANCED LIF UPDATE WITH LATERAL INHIBITION
// ============================================================================

/**
 * @brief Enhanced LIF neuron update with lateral inhibition and spike recording
 */
extern "C" __global__ void lif_update_with_inhibition(
    // LIF state
    float* lif_potential,
    LateralInhibitionState* inhibition_states,
    
    // Input
    const float* water_density,
    const float* water_density_prev,
    
    // Spike output
    int* spike_grid,
    VoxelSpikeHistory* spike_histories,
    int* spike_count,
    
    // Parameters
    float tau_membrane,
    float threshold,
    float dt_ps,
    float current_time_ps,
    int current_probe_id,
    
    // Grid dimensions
    int grid_dim
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_voxels = grid_dim * grid_dim * grid_dim;
    
    if (tid >= total_voxels) return;
    
    // Decay inhibition
    decay_inhibition(&inhibition_states[tid], current_time_ps, dt_ps);
    
    // Compute dewetting signal (decrease in water density)
    float dewetting = fmaxf(0.0f, water_density_prev[tid] - water_density[tid]);
    
    // Apply inhibition to input
    float inhibited_input = dewetting * (1.0f - inhibition_states[tid].inhibition_level);
    
    // Leaky integration
    float decay = expf(-dt_ps / tau_membrane);
    float new_potential = lif_potential[tid] * decay + inhibited_input;
    
    // Spike check
    bool spiked = new_potential >= threshold;
    spike_grid[tid] = spiked ? 1 : 0;
    
    if (spiked) {
        // Record spike with timestamp
        float intensity = new_potential;
        record_spike(&spike_histories[tid], tid, current_time_ps, intensity, current_probe_id);
        
        // Reset potential
        new_potential = 0.0f;
        
        // Apply lateral inhibition to neighbors
        apply_lateral_inhibition(lif_potential, inhibition_states, tid, grid_dim, current_time_ps);
        
        // Increment global spike count
        atomicAdd(spike_count, 1);
    }
    
    lif_potential[tid] = new_potential;
}

// ============================================================================
// KERNEL: SPIKE SEQUENCE DETECTION
// ============================================================================

/**
 * @brief Detect causal spike sequences (STDP-like)
 *
 * Each thread handles one sequence detector.
 */
extern "C" __global__ void detect_spike_sequences(
    SpikeSequenceDetector* detectors,
    const VoxelSpikeHistory* spike_histories,
    float* detection_scores,              // Output: score per detector
    float current_time_ps,
    int n_detectors,
    int n_voxels
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_detectors) return;
    
    SpikeSequenceDetector* detector = &detectors[tid];
    
    float score = check_sequence_detection(detector, spike_histories, current_time_ps, n_voxels);
    
    // Update detector state
    detector->accumulated_score = fmaxf(detector->accumulated_score, score);
    
    // Check for full detection
    if (score > 0.8f) {  // High confidence threshold
        detector->detection_count++;
        detector->accumulated_score = 0.0f;  // Reset for next detection
    }
    
    detection_scores[tid] = score;
}

// ============================================================================
// KERNEL: RESONANCE SWEEP
// ============================================================================

/**
 * @brief Accumulate response for resonance detection
 *
 * Called after each probe cycle to update the frequency spectrum.
 */
extern "C" __global__ void update_resonance_spectrum(
    ResonanceDetector* detector,
    const VoxelSpikeHistory* spike_histories,
    float probe_frequency_thz,
    float probe_start_time_ps,
    float probe_period_ps,
    int n_voxels,
    int grid_dim
) {
    // Single thread for aggregation
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    int total_voxels = grid_dim * grid_dim * grid_dim;
    
    // Compute response amplitude: total spike activity in probe period
    float total_intensity = 0.0f;
    int spike_count = 0;
    float weighted_phase = 0.0f;
    
    for (int v = 0; v < total_voxels; v++) {
        const VoxelSpikeHistory* hist = &spike_histories[v];
        
        for (int s = 0; s < hist->count; s++) {
            int idx = (hist->head - 1 - s + SPIKE_HISTORY_LENGTH) % SPIKE_HISTORY_LENGTH;
            const TimestampedSpike* spike = &hist->spikes[idx];
            
            // Check if within probe period
            float rel_time = spike->timestamp_ps - probe_start_time_ps;
            if (rel_time >= 0.0f && rel_time < probe_period_ps) {
                total_intensity += spike->intensity;
                spike_count++;
                
                // Compute phase relative to probe cycle
                float phase = fmodf(rel_time, 1.0f / probe_frequency_thz) * probe_frequency_thz * 2.0f * 3.14159f;
                weighted_phase += phase * spike->intensity;
            }
        }
    }
    
    float response_amplitude = total_intensity;
    float response_phase = (total_intensity > 0.0f) ? weighted_phase / total_intensity : 0.0f;
    
    update_resonance_detector(detector, probe_frequency_thz, response_amplitude, response_phase);
}

/**
 * @brief Analyze resonance spectrum to find peaks
 */
extern "C" __global__ void analyze_resonances(
    ResonanceDetector* detector
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    find_resonances(detector);
}

// ============================================================================
// KERNEL: ADAPTIVE PROBE SELECTION
// ============================================================================

/**
 * @brief Update probe scores and select next probe
 */
extern "C" __global__ void adaptive_probe_update(
    AdaptiveProbeController* controller,
    const ProbeResponse* last_response,
    int last_probe_idx,
    int* next_probe_idx,
    curandState* rng_state,
    int n_probes
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    // Compute reward from response
    // Higher reward for: more spikes, higher sequence score, good spatial localization
    float reward = 0.0f;
    
    reward += last_response->total_spikes * 0.1f;                    // Spike count
    reward += last_response->sequence_score * 5.0f;                  // Sequence detection (most important)
    reward += last_response->mean_intensity * 2.0f;                  // Intensity
    reward -= last_response->spatial_extent * 0.5f;                  // Penalize diffuse responses
    
    // Bonus for fast onset (responsive site)
    if (last_response->onset_latency_ps < 1.0f) {
        reward += 2.0f;
    }
    
    // Update score for last probe
    update_probe_score(controller, last_probe_idx, reward);
    
    // Select next probe
    curandState local_rng = *rng_state;
    *next_probe_idx = select_next_probe(controller, &local_rng, n_probes);
    *rng_state = local_rng;
    
    // Decay exploration temperature
    controller->exploration_temperature *= 0.999f;
    controller->exploration_temperature = fmaxf(0.1f, controller->exploration_temperature);
}

// ============================================================================
// KERNEL: DIFFERENTIAL PROBE COMPARISON
// ============================================================================

/**
 * @brief Compare responses to differential probe pair
 *
 * Large differential indicates asymmetric mechanical pathway → cryptic site.
 */
extern "C" __global__ void compare_differential_probes(
    DifferentialProbePair* pairs,
    const ProbeResponse* response_A,
    const ProbeResponse* response_B,
    int pair_idx
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    DifferentialProbePair* pair = &pairs[pair_idx];
    
    // Compute differential score
    float sum = response_A->total_spikes + response_B->total_spikes + 0.01f;  // Avoid div by zero
    float diff = fabsf((float)response_A->total_spikes - (float)response_B->total_spikes);
    
    float intensity_sum = response_A->mean_intensity + response_B->mean_intensity + 0.01f;
    float intensity_diff = fabsf(response_A->mean_intensity - response_B->mean_intensity);
    
    float seq_sum = response_A->sequence_score + response_B->sequence_score + 0.01f;
    float seq_diff = fabsf(response_A->sequence_score - response_B->sequence_score);
    
    // Weighted differential score
    float differential = 0.3f * (diff / sum) + 
                        0.3f * (intensity_diff / intensity_sum) +
                        0.4f * (seq_diff / seq_sum);  // Sequence score most important
    
    // Running average update
    int n = pair->n_trials;
    pair->differential_score = (pair->differential_score * n + differential) / (n + 1);
    pair->n_trials++;
    
    // Update confidence based on consistency
    // High differential + low variance = high confidence
    float variance_estimate = fabsf(differential - pair->differential_score);
    pair->confidence = pair->differential_score / (variance_estimate + 0.1f);
    pair->confidence = fminf(1.0f, pair->confidence / 10.0f);  // Normalize to [0,1]
}

// ============================================================================
// KERNEL: COMPUTE PROBE RESPONSE
// ============================================================================

/**
 * @brief Aggregate response metrics from spike histories
 */
extern "C" __global__ void compute_response_metrics(
    ProbeResponse* response,
    const VoxelSpikeHistory* spike_histories,
    const SpikeSequenceDetector* sequence_detectors,
    int n_detectors,
    const CoherentProbe* probe,
    float probe_start_time_ps,
    float analysis_window_ps,
    int grid_dim,
    float grid_spacing,
    float3 grid_origin
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    compute_probe_response(
        response,
        spike_histories,
        sequence_detectors,
        n_detectors,
        probe,
        probe_start_time_ps,
        analysis_window_ps,
        grid_dim,
        grid_spacing,
        grid_origin
    );
}

// ============================================================================
// KERNEL: BUILD AUTOMATIC SEQUENCE DETECTORS
// ============================================================================

/**
 * @brief Automatically build sequence detectors from spatial adjacency
 *
 * Creates detectors for all plausible spike propagation paths
 * based on voxel connectivity.
 */
extern "C" __global__ void build_sequence_detectors(
    SpikeSequenceDetector* detectors,
    int* n_detectors_out,
    const float3* aromatic_centroids,  // Centers of aromatic groups
    int n_aromatics,
    int grid_dim,
    float grid_spacing,
    float3 grid_origin,
    float max_sequence_distance       // Max distance for sequence (Angstroms)
) {
    // Single thread builds all detectors
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    int n_detectors = 0;
    int max_detectors = 1024;  // Limit
    
    // For each pair of aromatics, build detector along path
    for (int a = 0; a < n_aromatics && n_detectors < max_detectors; a++) {
        for (int b = a + 1; b < n_aromatics && n_detectors < max_detectors; b++) {
            float3 start = aromatic_centroids[a];
            float3 end = aromatic_centroids[b];
            
            // Distance between aromatics
            float dx = end.x - start.x;
            float dy = end.y - start.y;
            float dz = end.z - start.z;
            float dist = sqrtf(dx*dx + dy*dy + dz*dz);
            
            if (dist > max_sequence_distance || dist < grid_spacing * 2) continue;
            
            // Build voxel sequence along path
            SpikeSequenceDetector* det = &detectors[n_detectors];
            det->sequence_length = 0;
            det->max_inter_spike_interval_ps = CAUSALITY_WINDOW_PS;
            det->current_position = 0;
            det->sequence_start_time = 0.0f;
            det->accumulated_score = 0.0f;
            det->detection_count = 0;
            det->weight = 1.0f;
            
            int n_steps = (int)(dist / grid_spacing) + 1;
            n_steps = min(n_steps, MAX_SEQUENCE_LENGTH);
            
            for (int s = 0; s < n_steps; s++) {
                float t = (float)s / (float)(n_steps - 1);
                float px = start.x + t * dx;
                float py = start.y + t * dy;
                float pz = start.z + t * dz;
                
                // Convert to voxel index
                int vx = (int)((px - grid_origin.x) / grid_spacing);
                int vy = (int)((py - grid_origin.y) / grid_spacing);
                int vz = (int)((pz - grid_origin.z) / grid_spacing);
                
                vx = max(0, min(vx, grid_dim - 1));
                vy = max(0, min(vy, grid_dim - 1));
                vz = max(0, min(vz, grid_dim - 1));
                
                int voxel_idx = vx + vy * grid_dim + vz * grid_dim * grid_dim;
                det->voxel_sequence[det->sequence_length++] = voxel_idx;
            }
            
            n_detectors++;
        }
    }
    
    *n_detectors_out = n_detectors;
}

// ============================================================================
// KERNEL: INITIALIZE ACTIVE SENSING STATE
// ============================================================================

/**
 * @brief Initialize all active sensing buffers
 */
extern "C" __global__ void init_active_sensing(
    VoxelSpikeHistory* spike_histories,
    LateralInhibitionState* inhibition_states,
    ResonanceDetector* resonance_detector,
    AdaptiveProbeController* probe_controller,
    int total_voxels,
    int n_probes
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize spike histories
    if (tid < total_voxels) {
        spike_histories[tid].head = 0;
        spike_histories[tid].count = 0;
        spike_histories[tid].last_spike_time = -1000.0f;
        
        inhibition_states[tid].inhibition_level = 0.0f;
        inhibition_states[tid].last_update_time = 0.0f;
        inhibition_states[tid].n_inhibitors = 0;
    }
    
    // Initialize resonance detector (single thread)
    if (tid == 0) {
        for (int i = 0; i < FREQ_SWEEP_STEPS; i++) {
            resonance_detector->frequency_spectrum[i] = 0.0f;
            resonance_detector->phase_spectrum[i] = 0.0f;
            resonance_detector->sample_counts[i] = 0;
        }
        resonance_detector->n_resonances = 0;
        
        // Initialize probe controller
        for (int i = 0; i < n_probes * n_probes; i++) {
            probe_controller->probe_scores[i] = 0.0f;
            probe_controller->probe_trial_counts[i] = 0;
        }
        probe_controller->current_probe_idx = 0;
        probe_controller->exploration_temperature = 1.0f;
        probe_controller->cumulative_reward = 0.0f;
        probe_controller->total_trials = 0;
        probe_controller->best_score = 0.0f;
        probe_controller->best_probe_idx = 0;
    }
}

// ============================================================================
// KERNEL: GROUP AROMATICS BY SPATIAL CLUSTERING
// ============================================================================

/**
 * @brief Automatically group aromatics for coherent probing
 *
 * Uses k-means-like clustering to group aromatics by position.
 * Groups on same side of protein will form one cluster.
 */
extern "C" __global__ void cluster_aromatics(
    AromaticGroup* groups,
    int* n_groups_out,
    const float3* aromatic_centroids,
    const float* aromatic_absorptions,
    int n_aromatics,
    int target_n_groups,        // Desired number of groups
    float min_group_separation  // Minimum distance between group centroids
) {
    // Single thread k-means
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    // Limit groups
    int n_groups = min(target_n_groups, MAX_PROBE_GROUPS);
    n_groups = min(n_groups, n_aromatics);
    
    // Initialize group centroids (spread across aromatics)
    float3 centroids[MAX_PROBE_GROUPS];
    for (int g = 0; g < n_groups; g++) {
        int idx = (g * n_aromatics) / n_groups;
        centroids[g] = aromatic_centroids[idx];
        groups[g].n_aromatics = 0;
        groups[g].total_absorption = 0.0f;
    }
    
    // K-means iterations
    for (int iter = 0; iter < 10; iter++) {
        // Clear assignments
        for (int g = 0; g < n_groups; g++) {
            groups[g].n_aromatics = 0;
            groups[g].centroid = make_float3(0, 0, 0);
            groups[g].total_absorption = 0.0f;
        }
        
        // Assign aromatics to nearest centroid
        for (int a = 0; a < n_aromatics; a++) {
            float3 pos = aromatic_centroids[a];
            float best_dist = 1e30f;
            int best_group = 0;
            
            for (int g = 0; g < n_groups; g++) {
                float dx = pos.x - centroids[g].x;
                float dy = pos.y - centroids[g].y;
                float dz = pos.z - centroids[g].z;
                float dist = dx*dx + dy*dy + dz*dz;
                
                if (dist < best_dist) {
                    best_dist = dist;
                    best_group = g;
                }
            }
            
            // Add to group
            AromaticGroup* grp = &groups[best_group];
            if (grp->n_aromatics < MAX_AROMATICS_PER_GROUP) {
                grp->aromatic_indices[grp->n_aromatics] = a;
                grp->n_aromatics++;
                grp->centroid.x += pos.x;
                grp->centroid.y += pos.y;
                grp->centroid.z += pos.z;
                grp->total_absorption += aromatic_absorptions[a];
            }
        }
        
        // Update centroids
        for (int g = 0; g < n_groups; g++) {
            if (groups[g].n_aromatics > 0) {
                centroids[g].x = groups[g].centroid.x / groups[g].n_aromatics;
                centroids[g].y = groups[g].centroid.y / groups[g].n_aromatics;
                centroids[g].z = groups[g].centroid.z / groups[g].n_aromatics;
                groups[g].centroid = centroids[g];
            }
        }
    }
    
    *n_groups_out = n_groups;
}
