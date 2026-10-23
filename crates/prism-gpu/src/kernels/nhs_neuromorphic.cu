/**
 * NHS Neuromorphic Dewetting Network CUDA Kernel
 *
 * Implements a Leaky Integrate-and-Fire (LIF) neural network that spikes
 * when local water density drops (dewetting events = cryptic pocket opening).
 *
 * ARCHITECTURE:
 *   One LIF neuron per grid voxel that monitors local water density.
 *   Neuron spikes when water density DECREASES (dewetting).
 *   Spike clustering (done on host) identifies cooperative dewetting = cryptic pockets.
 *
 * LIF DYNAMICS (adapted for dewetting):
 *   dV/dt = -V/tau_mem + I_dewetting
 *   I_dewetting = max(0, prev_density - curr_density) * sensitivity
 *   spike = (V >= V_thresh)
 *
 * OUTPUT:
 *   Spike events (voxel index, frame) for avalanche detection
 *
 * PERFORMANCE TARGET:
 *   <0.3ms for 100³ grid on RTX 3060
 *
 * REFERENCE: NHS Architecture - Neuromorphic Dewetting Detection
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

// ============================================================================
// LIF NEURON PARAMETERS (OPTIMIZED FOR DEWETTING DETECTION)
// ============================================================================

// Membrane dynamics
#define TAU_MEM_DEFAULT     10.0f       // Membrane time constant (frames)
#define V_REST              0.0f        // Resting potential
#define V_THRESH            0.5f        // Spike threshold
#define V_RESET             0.0f        // Reset after spike

// Dewetting sensitivity
#define DEWETTING_GAIN      5.0f        // Amplification of density drops
#define MIN_DEWETTING       0.05f       // Minimum density drop to register

// Refractory period
#define REFRACTORY_FRAMES   3           // Frames after spike before can spike again

// Lateral inhibition (prevents runaway spiking)
#define LATERAL_INHIBITION  0.1f        // Inhibition from neighboring spikes

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Convert 3D grid coordinates to linear index
 */
__device__ __forceinline__ int grid_to_linear(int x, int y, int z, int dim) {
    return z * dim * dim + y * dim + x;
}

/**
 * Extract 3D coordinates from linear index
 */
__device__ __forceinline__ void linear_to_grid(int idx, int dim, int* x, int* y, int* z) {
    *z = idx / (dim * dim);
    int remainder = idx % (dim * dim);
    *y = remainder / dim;
    *x = remainder % dim;
}

// ============================================================================
// MAIN LIF STEP KERNEL
// ============================================================================

/**
 * Single timestep of LIF network for dewetting detection
 *
 * PARALLELISM: One thread per grid voxel (= one neuron)
 *
 * @param prev_water_density    Previous frame water density [grid_dim³]
 * @param curr_water_density    Current frame water density [grid_dim³]
 * @param membrane_potential    In/Out: Membrane voltages [grid_dim³]
 * @param refractory_counter    In/Out: Refractory period countdown [grid_dim³]
 * @param spike_output          OUTPUT: 1 if spike, 0 otherwise [grid_dim³]
 * @param spike_count           OUTPUT: Atomic counter for total spikes
 * @param grid_dim              Grid dimension
 * @param tau_mem               Membrane time constant
 * @param sensitivity           Dewetting sensitivity multiplier
 */
extern "C" __global__ void lif_dewetting_step(
    const float* __restrict__ prev_water_density,
    const float* __restrict__ curr_water_density,
    float* __restrict__ membrane_potential,
    int* __restrict__ refractory_counter,
    int* __restrict__ spike_output,
    int* __restrict__ spike_count,
    const int grid_dim,
    const float tau_mem,
    const float sensitivity
) {
    const int gx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gy = blockIdx.y * blockDim.y + threadIdx.y;
    const int gz = blockIdx.z * blockDim.z + threadIdx.z;

    if (gx >= grid_dim || gy >= grid_dim || gz >= grid_dim) return;

    const int idx = grid_to_linear(gx, gy, gz, grid_dim);

    // Load state
    float V = membrane_potential[idx];
    int refrac = refractory_counter[idx];
    const float prev_density = prev_water_density[idx];
    const float curr_density = curr_water_density[idx];

    // Compute dewetting input current
    // Positive current when water LEAVES (density decreases)
    float density_drop = prev_density - curr_density;
    float I_dewetting = 0.0f;

    if (density_drop > MIN_DEWETTING) {
        I_dewetting = density_drop * sensitivity * DEWETTING_GAIN;
    }

    // Check refractory period
    int spike = 0;
    if (refrac > 0) {
        // In refractory period: just decay, no integration
        V = V * (1.0f - 1.0f / tau_mem);
        refrac--;
    } else {
        // LIF dynamics: leaky integration
        float leak = 1.0f / tau_mem;
        V = V * (1.0f - leak) + I_dewetting;

        // Spike check
        if (V >= V_THRESH) {
            spike = 1;
            V = V_RESET;
            refrac = REFRACTORY_FRAMES;

            // Atomic increment spike counter
            atomicAdd(spike_count, 1);
        }
    }

    // Clamp membrane potential
    V = fmaxf(fminf(V, 5.0f), -1.0f);

    // Write back state
    membrane_potential[idx] = V;
    refractory_counter[idx] = refrac;
    spike_output[idx] = spike;
}

// ============================================================================
// BATCH LIF STEP (PROCESS MULTIPLE FRAMES)
// ============================================================================

/**
 * Process multiple frames for efficiency
 *
 * @param water_density_batch   Water density for N frames [N * grid_dim³]
 * @param membrane_potential    In/Out: Membrane voltages [grid_dim³]
 * @param refractory_counter    In/Out: Refractory counters [grid_dim³]
 * @param spike_output_batch    OUTPUT: Spikes for N frames [N * grid_dim³]
 * @param spike_counts          OUTPUT: Spike count per frame [N]
 * @param n_frames              Number of frames in batch
 * @param grid_dim              Grid dimension
 * @param tau_mem               Membrane time constant
 * @param sensitivity           Dewetting sensitivity
 */
extern "C" __global__ void lif_dewetting_batch(
    const float* __restrict__ water_density_batch,
    float* __restrict__ membrane_potential,
    int* __restrict__ refractory_counter,
    int* __restrict__ spike_output_batch,
    int* __restrict__ spike_counts,
    const int n_frames,
    const int grid_dim,
    const float tau_mem,
    const float sensitivity
) {
    const int gx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gy = blockIdx.y * blockDim.y + threadIdx.y;
    const int gz = blockIdx.z * blockDim.z + threadIdx.z;

    if (gx >= grid_dim || gy >= grid_dim || gz >= grid_dim) return;

    const int idx = grid_to_linear(gx, gy, gz, grid_dim);
    const int grid_size = grid_dim * grid_dim * grid_dim;

    // Load state
    float V = membrane_potential[idx];
    int refrac = refractory_counter[idx];
    float prev_density = water_density_batch[idx];  // First frame as "previous"

    // Process each frame
    for (int frame = 1; frame < n_frames; frame++) {
        const float curr_density = water_density_batch[frame * grid_size + idx];

        // Compute dewetting input
        float density_drop = prev_density - curr_density;
        float I_dewetting = (density_drop > MIN_DEWETTING) ?
                            density_drop * sensitivity * DEWETTING_GAIN : 0.0f;

        // LIF dynamics
        int spike = 0;
        if (refrac > 0) {
            V = V * (1.0f - 1.0f / tau_mem);
            refrac--;
        } else {
            V = V * (1.0f - 1.0f / tau_mem) + I_dewetting;

            if (V >= V_THRESH) {
                spike = 1;
                V = V_RESET;
                refrac = REFRACTORY_FRAMES;
                atomicAdd(&spike_counts[frame], 1);
            }
        }

        V = fmaxf(fminf(V, 5.0f), -1.0f);
        spike_output_batch[frame * grid_size + idx] = spike;
        prev_density = curr_density;
    }

    // Write final state
    membrane_potential[idx] = V;
    refractory_counter[idx] = refrac;
}

// ============================================================================
// LATERAL INHIBITION KERNEL
// ============================================================================

/**
 * Apply lateral inhibition from neighboring spikes
 *
 * Prevents runaway spiking by having active neighbors suppress a neuron.
 *
 * @param spike_output          Current spike pattern [grid_dim³]
 * @param membrane_potential    In/Out: Apply inhibition [grid_dim³]
 * @param grid_dim              Grid dimension
 * @param inhibition_strength   Strength of lateral inhibition
 */
extern "C" __global__ void apply_lateral_inhibition(
    const int* __restrict__ spike_output,
    float* __restrict__ membrane_potential,
    const int grid_dim,
    const float inhibition_strength
) {
    const int gx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gy = blockIdx.y * blockDim.y + threadIdx.y;
    const int gz = blockIdx.z * blockDim.z + threadIdx.z;

    if (gx >= grid_dim || gy >= grid_dim || gz >= grid_dim) return;

    const int idx = grid_to_linear(gx, gy, gz, grid_dim);

    // Only apply inhibition if this neuron didn't spike
    if (spike_output[idx] == 1) return;

    // Count neighboring spikes (6-connected neighborhood)
    int neighbor_spikes = 0;

    if (gx > 0 && spike_output[grid_to_linear(gx-1, gy, gz, grid_dim)]) neighbor_spikes++;
    if (gx < grid_dim-1 && spike_output[grid_to_linear(gx+1, gy, gz, grid_dim)]) neighbor_spikes++;
    if (gy > 0 && spike_output[grid_to_linear(gx, gy-1, gz, grid_dim)]) neighbor_spikes++;
    if (gy < grid_dim-1 && spike_output[grid_to_linear(gx, gy+1, gz, grid_dim)]) neighbor_spikes++;
    if (gz > 0 && spike_output[grid_to_linear(gx, gy, gz-1, grid_dim)]) neighbor_spikes++;
    if (gz < grid_dim-1 && spike_output[grid_to_linear(gx, gy, gz+1, grid_dim)]) neighbor_spikes++;

    // Apply inhibition proportional to neighbor activity
    if (neighbor_spikes > 0) {
        float inhibition = neighbor_spikes * inhibition_strength;
        membrane_potential[idx] -= inhibition;
        membrane_potential[idx] = fmaxf(membrane_potential[idx], -1.0f);
    }
}

// ============================================================================
// SPIKE EXTRACTION KERNEL
// ============================================================================

/**
 * Compact spike indices for host processing
 *
 * Uses parallel stream compaction to extract spike locations.
 *
 * @param spike_output          Binary spike pattern [grid_dim³]
 * @param spike_indices         OUTPUT: Compacted spike voxel indices [max_spikes]
 * @param spike_positions       OUTPUT: Spike positions in world coords [max_spikes * 3]
 * @param spike_count           In/Out: Number of spikes (atomic counter)
 * @param grid_dim              Grid dimension
 * @param grid_origin_x/y/z     Grid origin coordinates
 * @param grid_spacing          Grid spacing
 * @param max_spikes            Maximum spikes to output
 */
extern "C" __global__ void extract_spike_indices(
    const int* __restrict__ spike_output,
    int* __restrict__ spike_indices,
    float* __restrict__ spike_positions,
    int* __restrict__ spike_count,
    const int grid_dim,
    const float grid_origin_x,
    const float grid_origin_y,
    const float grid_origin_z,
    const float grid_spacing,
    const int max_spikes
) {
    const int gx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gy = blockIdx.y * blockDim.y + threadIdx.y;
    const int gz = blockIdx.z * blockDim.z + threadIdx.z;

    if (gx >= grid_dim || gy >= grid_dim || gz >= grid_dim) return;

    const int idx = grid_to_linear(gx, gy, gz, grid_dim);

    if (spike_output[idx] == 1) {
        // Atomic increment to get output index
        int output_idx = atomicAdd(spike_count, 1);

        if (output_idx < max_spikes) {
            spike_indices[output_idx] = idx;

            // Compute world position of spike
            spike_positions[output_idx * 3 + 0] = grid_origin_x + (gx + 0.5f) * grid_spacing;
            spike_positions[output_idx * 3 + 1] = grid_origin_y + (gy + 0.5f) * grid_spacing;
            spike_positions[output_idx * 3 + 2] = grid_origin_z + (gz + 0.5f) * grid_spacing;
        }
    }
}

// ============================================================================
// SPIKE-TO-RESIDUE MAPPING KERNEL
// ============================================================================

/**
 * Map spikes to nearest residues for biological interpretation
 *
 * @param spike_positions       Spike world positions [n_spikes * 3]
 * @param atom_positions        Atom world positions [n_atoms * 3]
 * @param atom_residues         Residue index per atom [n_atoms]
 * @param spike_residues        OUTPUT: Nearest residue for each spike [n_spikes]
 * @param spike_distances       OUTPUT: Distance to nearest atom [n_spikes]
 * @param n_spikes              Number of spikes
 * @param n_atoms               Number of atoms
 * @param max_distance          Maximum distance to consider (Å)
 */
extern "C" __global__ void map_spikes_to_residues(
    const float* __restrict__ spike_positions,
    const float* __restrict__ atom_positions,
    const int* __restrict__ atom_residues,
    int* __restrict__ spike_residues,
    float* __restrict__ spike_distances,
    const int n_spikes,
    const int n_atoms,
    const float max_distance
) {
    const int spike_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (spike_idx >= n_spikes) return;

    const float sx = spike_positions[spike_idx * 3 + 0];
    const float sy = spike_positions[spike_idx * 3 + 1];
    const float sz = spike_positions[spike_idx * 3 + 2];

    float min_dist = max_distance;
    int nearest_residue = -1;

    // Find nearest atom
    for (int a = 0; a < n_atoms; a++) {
        const float ax = atom_positions[a * 3 + 0];
        const float ay = atom_positions[a * 3 + 1];
        const float az = atom_positions[a * 3 + 2];

        const float dx = sx - ax;
        const float dy = sy - ay;
        const float dz = sz - az;
        const float dist = sqrtf(dx * dx + dy * dy + dz * dz);

        if (dist < min_dist) {
            min_dist = dist;
            nearest_residue = atom_residues[a];
        }
    }

    spike_residues[spike_idx] = nearest_residue;
    spike_distances[spike_idx] = min_dist;
}

// ============================================================================
// STATE MANAGEMENT KERNELS
// ============================================================================

/**
 * Initialize LIF network state
 */
extern "C" __global__ void init_lif_state(
    float* __restrict__ membrane_potential,
    int* __restrict__ refractory_counter,
    const int grid_dim
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = grid_dim * grid_dim * grid_dim;

    if (idx < total) {
        membrane_potential[idx] = V_REST;
        refractory_counter[idx] = 0;
    }
}

/**
 * Reset LIF network state (preserves adaptation)
 */
extern "C" __global__ void reset_lif_state(
    float* __restrict__ membrane_potential,
    int* __restrict__ refractory_counter,
    const int grid_dim,
    const float reset_strength  // 0.0 = full reset, 1.0 = no reset
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = grid_dim * grid_dim * grid_dim;

    if (idx < total) {
        membrane_potential[idx] = membrane_potential[idx] * reset_strength;
        refractory_counter[idx] = 0;
    }
}

// ============================================================================
// ADAPTIVE THRESHOLD KERNEL
// ============================================================================

/**
 * Update adaptive thresholds based on local activity
 *
 * High-activity regions get higher thresholds (habituation)
 * Low-activity regions get lower thresholds (sensitization)
 *
 * @param spike_history         Recent spike counts per voxel [grid_dim³]
 * @param adaptive_threshold    In/Out: Per-voxel threshold [grid_dim³]
 * @param grid_dim              Grid dimension
 * @param adaptation_rate       Learning rate for threshold adaptation
 * @param target_rate           Target spike rate (spikes per N frames)
 */
extern "C" __global__ void update_adaptive_threshold(
    const int* __restrict__ spike_history,
    float* __restrict__ adaptive_threshold,
    const int grid_dim,
    const float adaptation_rate,
    const float target_rate
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = grid_dim * grid_dim * grid_dim;

    if (idx < total) {
        const float activity = (float)spike_history[idx];
        const float error = activity - target_rate;

        // Homeostatic plasticity: adjust threshold to maintain target rate
        float threshold = adaptive_threshold[idx];
        threshold += adaptation_rate * error;

        // Clamp to reasonable range
        threshold = fmaxf(fminf(threshold, 2.0f), 0.1f);
        adaptive_threshold[idx] = threshold;
    }
}
