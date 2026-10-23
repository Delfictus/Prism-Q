/**
 * Dendritic Spiking Neural Network Reservoir for RL Agent
 *
 * This kernel implements a Leaky Integrate-and-Fire (LIF) reservoir
 * for processing 40-dimensional RL feature vectors (with bio-chemistry) and outputting
 * high-dimensional filtered firing rates for linear readout.
 *
 * ARCHITECTURE:
 *   40 features → Input weights (40→N) → LIF neurons (N) → Filtered rates (N)
 *   where N = RESERVOIR_SIZE (default 512)
 *
 * LIF DYNAMICS:
 *   dV/dt = -V/tau_mem + I_input + I_recurrent
 *   spike = (V >= V_thresh)
 *   if spike: V = V_reset
 *
 * FILTERED RATES (for linear separability):
 *   r_new = (1 - alpha) * r_old + alpha * spike
 *   where alpha = dt / tau_syn (exponential filter)
 *
 * MEMORY LAYOUT (all persistent in VRAM):
 *   - membrane_potential[N]: Current membrane voltage
 *   - filtered_rates[N]: Exponentially filtered spike rates
 *   - input_weights[N * INPUT_DIM]: Sparse input connections
 *   - recurrent_weights[N * N]: Sparse recurrent connections (CSR)
 *
 * REFERENCE: PRISM Dendritic Agent Architecture
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <math.h>

// ============================================================================
// FEATURE ADAPTER PROTOCOL - Enhanced for Cryptic Site Detection
// ============================================================================

// Architecture constants (updated for bio-chemistry features)
#define RAW_INPUT_DIM 40        // Raw feature vector size (with bio-chemistry: hydrophobic, anisotropy, frustration)
#define EXPANDED_INPUT_DIM 80   // With velocity features (40 raw + 40 delta)
#define RESERVOIR_SIZE 512      // Number of LIF neurons
#define MAX_RESERVOIR 2048      // Maximum supported

// Sparse topology: Each input connects to ~10% of neurons
// But structured: hash(input_idx) determines which neurons receive it
#define SPARSITY 0.1f           // 10% connectivity
#define SPARSE_FAN_OUT 51       // ~10% of 512 neurons per input

// ============================================================================
// EXCITATORY / INHIBITORY BALANCE (Critical for Stability)
// ============================================================================
// Mimics cortical microcircuits: ~80% excitatory, ~20% inhibitory
// Inhibitory neurons have NEGATIVE output weights to suppress runaway activity
#define EXCITATORY_FRACTION 0.8f  // 80% excitatory neurons
#define INHIBITORY_GAIN 2.0f      // Inhibitory neurons are stronger (GABAergic)

// LIF neuron parameters (biologically inspired, optimized for RL)
#define TAU_MEM 20.0f           // Membrane time constant (ms) - default
#define TAU_SYN 5.0f            // Synaptic filter time constant (ms) - default
#define V_REST 0.0f             // Resting potential
#define V_THRESH 1.0f           // Spike threshold
#define V_RESET 0.0f            // Reset potential after spike
#define DT 1.0f                 // Timestep (ms)

// Adaptive time constants (can be overridden per-neuron)
#define TAU_MEM_MIN 5.0f        // Fast neurons (local vibrations)
#define TAU_MEM_MAX 50.0f       // Slow neurons (domain motions)

// Derived constants
#define LEAK_RATE (DT / TAU_MEM)           // Membrane leak per timestep
#define FILTER_ALPHA (DT / TAU_SYN)        // Firing rate filter coefficient
#define INPUT_SCALE 0.1f                   // Scale input currents
#define RECURRENT_SCALE 0.05f              // Scale recurrent currents

// Tanh normalization gain (squashes inputs to [-1, 1])
#define TANH_GAIN_RAW 1.0f      // Standard gain for raw features
#define TANH_GAIN_DELTA 5.0f    // High gain for velocity/delta features (more sensitive)

/**
 * Simple hash function for structured sparse topology
 * Ensures deterministic input→neuron mapping
 */
__device__ __forceinline__ unsigned int sparse_hash(unsigned int input_idx, unsigned int neuron_idx) {
    // FNV-1a inspired hash for determining connectivity
    unsigned int hash = 2166136261u;
    hash ^= input_idx;
    hash *= 16777619u;
    hash ^= neuron_idx;
    hash *= 16777619u;
    return hash;
}

/**
 * Check if input_idx should connect to neuron_idx using structured sparse topology
 * Each input connects to exactly ~10% of neurons, deterministically
 */
__device__ __forceinline__ bool is_connected(int input_idx, int neuron_idx, int reservoir_size) {
    unsigned int hash = sparse_hash((unsigned int)input_idx, (unsigned int)neuron_idx);
    // Map hash to [0, reservoir_size) and check if in first 10%
    return (hash % 100) < 10;
}

/**
 * Determine if neuron is excitatory or inhibitory
 * First 80% are excitatory, last 20% are inhibitory
 */
__device__ __forceinline__ bool is_excitatory(int neuron_idx, int reservoir_size) {
    return neuron_idx < (int)(reservoir_size * EXCITATORY_FRACTION);
}

/**
 * Initialize reservoir weights with E/I BALANCED sparse connectivity
 *
 * "Flashbulb Reservoir" Implementation:
 * - 80% Excitatory neurons (positive recurrent output)
 * - 20% Inhibitory neurons (negative recurrent output, 2× stronger)
 * - Structured sparse input topology (hash-based)
 * - Adaptive time constants per-neuron
 * - Velocity features get boosted weights
 *
 * @param input_weights    Output: Input weight matrix [N * EXPANDED_INPUT_DIM]
 * @param recurrent_weights Output: Recurrent weight matrix [N * N]
 * @param membrane         Output: Initial membrane potentials [N]
 * @param filtered_rates   Output: Initial filtered rates [N]
 * @param tau_mem_array    Output: Per-neuron membrane time constants [N]
 * @param neuron_signs     Output: +1 for excitatory, -1 for inhibitory [N]
 * @param reservoir_size   Number of reservoir neurons
 * @param seed             Random seed for reproducibility
 */
extern "C" __global__ void init_snn_reservoir(
    float* __restrict__ input_weights,
    float* __restrict__ recurrent_weights,
    float* __restrict__ membrane,
    float* __restrict__ filtered_rates,
    float* __restrict__ tau_mem_array,
    float* __restrict__ neuron_signs,
    const int reservoir_size,
    const unsigned long long seed
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_input_weights = reservoir_size * EXPANDED_INPUT_DIM;
    const int total_recurrent_weights = reservoir_size * reservoir_size;

    // Initialize cuRAND state
    curandState_t rand_state;
    curand_init(seed, idx, 0, &rand_state);

    // Initialize input weights (STRUCTURED sparse topology)
    if (idx < total_input_weights) {
        int neuron_idx = idx / EXPANDED_INPUT_DIM;
        int input_idx = idx % EXPANDED_INPUT_DIM;

        // Check structured sparse connectivity
        if (is_connected(input_idx, neuron_idx, reservoir_size)) {
            // Non-zero weight with Gaussian distribution
            float weight = curand_normal(&rand_state) * INPUT_SCALE;

            // Velocity features (40-79) get 2.5× higher magnitude (your gain=5.0 suggestion)
            if (input_idx >= RAW_INPUT_DIM) {
                weight *= 2.5f;  // Higher sensitivity to temporal changes
            }

            input_weights[idx] = weight;
        } else {
            input_weights[idx] = 0.0f;
        }
    }

    // Initialize recurrent weights with E/I BALANCE
    if (idx < total_recurrent_weights) {
        int target_idx = idx / reservoir_size;  // Postsynaptic neuron
        int source_idx = idx % reservoir_size;  // Presynaptic neuron

        // No self-connections
        if (target_idx == source_idx) {
            recurrent_weights[idx] = 0.0f;
        } else {
            float rand_val = curand_uniform(&rand_state);
            if (rand_val < SPARSITY) {
                // Base weight magnitude
                float weight_mag = fabsf(curand_normal(&rand_state)) * RECURRENT_SCALE;

                // SOURCE neuron determines sign (Dale's Law)
                // Excitatory neurons have positive outputs
                // Inhibitory neurons have negative outputs (and are stronger)
                if (is_excitatory(source_idx, reservoir_size)) {
                    recurrent_weights[idx] = weight_mag;  // Positive (excitatory)
                } else {
                    recurrent_weights[idx] = -weight_mag * INHIBITORY_GAIN;  // Negative & stronger
                }
            } else {
                recurrent_weights[idx] = 0.0f;
            }
        }
    }

    // Initialize membrane potential, filtered rates, adaptive τ, and E/I signs
    if (idx < reservoir_size) {
        membrane[idx] = V_REST + curand_uniform(&rand_state) * 0.1f;
        filtered_rates[idx] = 0.0f;

        // Store neuron type for downstream use
        neuron_signs[idx] = is_excitatory(idx, reservoir_size) ? 1.0f : -1.0f;

        // Adaptive time constants: gradient from fast to slow
        // BUT: Inhibitory neurons are FAST (interneurons in biology)
        float neuron_frac = (float)idx / (float)reservoir_size;
        float tau;

        if (!is_excitatory(idx, reservoir_size)) {
            // Inhibitory neurons: fast (like PV+ interneurons)
            tau = TAU_MEM_MIN + curand_uniform(&rand_state) * 5.0f;  // 5-10ms
        } else if (neuron_frac < 0.25f) {
            tau = TAU_MEM_MIN + (neuron_frac / 0.25f) * 10.0f;  // 5-15ms
        } else if (neuron_frac < 0.6f) {
            tau = 15.0f + ((neuron_frac - 0.25f) / 0.35f) * 20.0f;  // 15-35ms
        } else {
            tau = 35.0f + ((neuron_frac - 0.6f) / 0.2f) * 15.0f;  // 35-50ms
        }
        tau_mem_array[idx] = tau;
    }
}

/**
 * Single LIF simulation step with filtered rate output (LEGACY)
 *
 * NOTE: This kernel is kept for compatibility. The main inference path
 * uses lif_multistep which implements the full Feature Adapter Protocol.
 *
 * Computes one timestep of the reservoir dynamics:
 * 1. Compute input current from features
 * 2. Compute recurrent current from previous spikes
 * 3. Update membrane potential with LIF dynamics
 * 4. Generate spikes where V >= threshold
 * 5. Update filtered firing rates (exponential moving average)
 *
 * @param features         Input feature vector [RAW_INPUT_DIM = 40]
 * @param membrane         In/Out: Membrane potentials [N]
 * @param filtered_rates   In/Out: Filtered firing rates [N]
 * @param input_weights    Input weight matrix [N * RAW_INPUT_DIM]
 * @param recurrent_weights Recurrent weight matrix [N * N]
 * @param spikes           Output: Binary spike vector [N]
 * @param reservoir_size   Number of reservoir neurons
 */
extern "C" __global__ void lif_step(
    const float* __restrict__ features,
    float* __restrict__ membrane,
    float* __restrict__ filtered_rates,
    const float* __restrict__ input_weights,
    const float* __restrict__ recurrent_weights,
    int* __restrict__ spikes,
    const int reservoir_size
) {
    const int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (neuron_idx >= reservoir_size) return;

    // Step 1: Compute input current from features (raw 40-dim)
    float I_input = 0.0f;
    for (int f = 0; f < RAW_INPUT_DIM; f++) {
        int weight_idx = neuron_idx * RAW_INPUT_DIM + f;
        I_input += input_weights[weight_idx] * features[f];
    }

    // Step 2: Compute recurrent current from previous filtered rates
    // Using filtered rates instead of raw spikes for smoothness
    float I_recurrent = 0.0f;
    for (int j = 0; j < reservoir_size; j++) {
        int weight_idx = neuron_idx * reservoir_size + j;
        I_recurrent += recurrent_weights[weight_idx] * filtered_rates[j];
    }

    // Step 3: LIF membrane dynamics
    float V_old = membrane[neuron_idx];
    float V_new = V_old * (1.0f - LEAK_RATE) + I_input + I_recurrent;

    // Step 4: Spike generation
    int spike = (V_new >= V_THRESH) ? 1 : 0;
    spikes[neuron_idx] = spike;

    // Apply reset if spiked
    if (spike) {
        V_new = V_RESET;
    }

    // Clamp membrane potential to prevent explosion
    V_new = fmaxf(fminf(V_new, 10.0f), -10.0f);
    membrane[neuron_idx] = V_new;

    // Step 5: Update filtered firing rate (exponential moving average)
    float r_old = filtered_rates[neuron_idx];
    float r_new = r_old * (1.0f - FILTER_ALPHA) + FILTER_ALPHA * (float)spike;
    filtered_rates[neuron_idx] = r_new;
}

/**
 * Multi-step LIF simulation with Feature Adapter Protocol
 *
 * ENHANCED with:
 * - 80-dimensional input (40 raw + 40 velocity features)
 * - Tanh normalization for input scaling
 * - Adaptive per-neuron time constants
 * - Persistent state (no implicit reset)
 *
 * @param features         Input feature vector [EXPANDED_INPUT_DIM = 80]
 * @param membrane         In/Out: Membrane potentials [N]
 * @param filtered_rates   In/Out: Filtered firing rates [N]
 * @param input_weights    Input weight matrix [N * EXPANDED_INPUT_DIM]
 * @param recurrent_weights Recurrent weight matrix [N * N]
 * @param tau_mem_array    Per-neuron membrane time constants [N]
 * @param reservoir_size   Number of reservoir neurons
 * @param num_steps        Number of timesteps to simulate
 */
extern "C" __global__ void lif_multistep(
    const float* __restrict__ features,
    float* __restrict__ membrane,
    float* __restrict__ filtered_rates,
    const float* __restrict__ input_weights,
    const float* __restrict__ recurrent_weights,
    const float* __restrict__ tau_mem_array,
    const int reservoir_size,
    const int num_steps
) {
    const int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (neuron_idx >= reservoir_size) return;

    // Load adaptive time constant for this neuron
    const float tau_mem = tau_mem_array[neuron_idx];
    const float leak_rate = DT / tau_mem;

    // Precompute input current with DIFFERENTIAL TANH NORMALIZATION
    // Raw features: standard gain (preserve magnitude)
    // Delta features: high gain (amplify small changes)
    float I_input = 0.0f;
    for (int f = 0; f < EXPANDED_INPUT_DIM; f++) {
        int weight_idx = neuron_idx * EXPANDED_INPUT_DIM + f;
        float weight = input_weights[weight_idx];

        // Apply tanh normalization with differential gain
        // Raw features (0-39): standard gain preserves magnitude info
        // Delta features (40-79): high gain amplifies small temporal changes
        float gain = (f < RAW_INPUT_DIM) ? TANH_GAIN_RAW : TANH_GAIN_DELTA;
        float normalized_input = tanhf(features[f] * gain);
        I_input += weight * normalized_input;
    }

    // Local copies for fast access
    float V = membrane[neuron_idx];
    float r = filtered_rates[neuron_idx];

    // Shared memory for filtered rates (for recurrent computation)
    __shared__ float shared_rates[RESERVOIR_SIZE];
    shared_rates[neuron_idx] = r;
    __syncthreads();

    // Run multiple timesteps
    for (int step = 0; step < num_steps; step++) {
        // Compute recurrent current
        float I_recurrent = 0.0f;
        for (int j = 0; j < reservoir_size; j++) {
            int weight_idx = neuron_idx * reservoir_size + j;
            I_recurrent += recurrent_weights[weight_idx] * shared_rates[j];
        }

        // LIF dynamics with ADAPTIVE time constant
        V = V * (1.0f - leak_rate) + I_input + I_recurrent;

        // Spike generation
        int spike = (V >= V_THRESH) ? 1 : 0;
        if (spike) {
            V = V_RESET;
        }

        // Clamp (wider range for slow neurons)
        V = fmaxf(fminf(V, 10.0f), -10.0f);

        // Update filtered rate
        r = r * (1.0f - FILTER_ALPHA) + FILTER_ALPHA * (float)spike;

        // Sync for next iteration's recurrent computation
        __syncthreads();
        shared_rates[neuron_idx] = r;
        __syncthreads();
    }

    // Write final state back to global memory (PERSISTENT - not reset)
    membrane[neuron_idx] = V;
    filtered_rates[neuron_idx] = r;
}

/**
 * Copy filtered rates to output buffer
 *
 * Simple kernel to extract the reservoir state for readout layer.
 *
 * @param filtered_rates   Input: Filtered firing rates [N]
 * @param output           Output: State vector for readout [N]
 * @param reservoir_size   Number of reservoir neurons
 */
extern "C" __global__ void extract_state(
    const float* __restrict__ filtered_rates,
    float* __restrict__ output,
    const int reservoir_size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < reservoir_size) {
        output[idx] = filtered_rates[idx];
    }
}

/**
 * Reset reservoir state to initial conditions
 *
 * Called at the start of each episode to reset temporal memory.
 *
 * @param membrane         Output: Reset membrane potentials [N]
 * @param filtered_rates   Output: Reset filtered rates [N]
 * @param reservoir_size   Number of reservoir neurons
 * @param seed             Random seed for slight randomization
 */
extern "C" __global__ void reset_state(
    float* __restrict__ membrane,
    float* __restrict__ filtered_rates,
    const int reservoir_size,
    const unsigned long long seed
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < reservoir_size) {
        // Initialize with slight random variation for symmetry breaking
        curandState_t rand_state;
        curand_init(seed, idx, 0, &rand_state);

        membrane[idx] = V_REST + curand_uniform(&rand_state) * 0.05f;
        filtered_rates[idx] = 0.0f;
    }
}

/**
 * Compute action values from reservoir state (matrix-vector multiply)
 *
 * Q(s,a) = W_out @ filtered_rates
 * where W_out is [num_actions x reservoir_size]
 *
 * This allows computing Q-values directly on GPU if desired.
 *
 * @param filtered_rates   Input: Reservoir state [N]
 * @param output_weights   Input: Readout weights [num_actions * N]
 * @param q_values         Output: Action values [num_actions]
 * @param reservoir_size   Number of reservoir neurons
 * @param num_actions      Number of output actions
 */
extern "C" __global__ void compute_q_values(
    const float* __restrict__ filtered_rates,
    const float* __restrict__ output_weights,
    float* __restrict__ q_values,
    const int reservoir_size,
    const int num_actions
) {
    const int action_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (action_idx >= num_actions) return;

    float q_value = 0.0f;
    for (int j = 0; j < reservoir_size; j++) {
        int weight_idx = action_idx * reservoir_size + j;
        q_value += output_weights[weight_idx] * filtered_rates[j];
    }

    q_values[action_idx] = q_value;
}
