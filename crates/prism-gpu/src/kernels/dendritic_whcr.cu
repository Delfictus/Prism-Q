/**
 * PRISM Dendritic Reservoir for WHCR (DR-WHCR) - CUDA Kernels
 * 
 * Live neuromorphic co-processor that evolves WITH the repair process,
 * providing real-time adaptive guidance for conflict resolution.
 * 
 * Key innovations:
 * 1. Dynamic reservoir topology mapping to current conflict structure
 * 2. Multi-compartment dendritic processing with varied time constants
 * 3. Conflict momentum tracking to identify stubborn vs transient conflicts
 * 4. Cascade prediction via reservoir echo state dynamics
 * 5. Temporal memory preventing oscillation and cycling
 * 
 * Integration: DR updates AFTER each WHCR iteration, outputs MODIFY
 * wavelet priorities for next iteration, creating adaptive feedback loop.
 * 
 * Patent-relevant: "Neuromorphic co-processor for combinatorial optimization"
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// ═══════════════════════════════════════════════════════════════════════════
// CONSTANTS AND CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// Dendritic compartment configuration
#define NUM_COMPARTMENTS 4          // Per-vertex dendritic compartments
#define COMPARTMENT_PROXIMAL 0      // Fast dynamics (tau ~ 1 iteration)
#define COMPARTMENT_DISTAL_1 1      // Medium dynamics (tau ~ 5 iterations)
#define COMPARTMENT_DISTAL_2 2      // Slow dynamics (tau ~ 20 iterations)
#define COMPARTMENT_SPINE 3         // Very slow (tau ~ 50 iterations) - long-term memory

// Time constants for each compartment (as decay factors)
__constant__ float TAU_DECAY[NUM_COMPARTMENTS] = {
    0.1f,   // Proximal: rapid response, 90% decay per step
    0.5f,   // Distal 1: medium memory
    0.85f,  // Distal 2: slow integration
    0.95f   // Spine: long-term conflict memory
};

// Reservoir hyperparameters
#define SPECTRAL_RADIUS 0.9f        // Echo state property threshold
#define INPUT_SCALING 0.3f          // Conflict signal scaling
#define LEAK_RATE 0.3f              // Reservoir state leaking
#define SPARSITY 0.1f               // Recurrent connection sparsity

// Output signal types
#define OUTPUT_PRIORITY_MOD 0       // Priority modulation for WHCR
#define OUTPUT_CASCADE_PRED 1       // Cascade potential prediction
#define OUTPUT_RECEPTIVITY 2        // Repair receptivity score
#define OUTPUT_MOMENTUM 3           // Conflict momentum indicator
#define NUM_OUTPUTS 4

// ═══════════════════════════════════════════════════════════════════════════
// DATA STRUCTURES
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Dendritic compartment state for a single vertex
 * Each vertex has multiple compartments with different time constants
 */
struct DendriticCompartment {
    float activation;           // Current activation level
    float calcium;              // Calcium-like accumulator (for LTP/LTD)
    float threshold;            // Adaptive firing threshold
    float refractory;           // Refractory period remaining
};

/**
 * Per-vertex dendritic state
 */
struct VertexDendriticState {
    DendriticCompartment compartments[NUM_COMPARTMENTS];
    float soma_potential;       // Integrated soma potential
    float spike_history;        // Recent spiking activity
    float conflict_memory;      // Long-term conflict exposure
    int last_repair_iteration;  // When was this vertex last repaired
};

/**
 * Reservoir recurrent connection (sparse)
 */
struct ReservoirConnection {
    int source_vertex;
    int target_vertex;
    int source_compartment;
    int target_compartment;
    float weight;               // Connection strength
    float delay;                // Synaptic delay (in iterations)
};

/**
 * Full reservoir state
 */
struct DendriticReservoir {
    // Vertex states
    VertexDendriticState* vertex_states;    // [num_vertices]
    int num_vertices;
    
    // Sparse recurrent connections (CSR format)
    int* connection_row_ptr;                 // [num_vertices + 1]
    ReservoirConnection* connections;        // [num_connections]
    int num_connections;
    
    // Input projection weights
    float* input_weights;                    // [num_vertices * NUM_COMPARTMENTS]
    
    // Output projection weights
    float* output_weights;                   // [num_vertices * NUM_OUTPUTS]
    
    // Reservoir outputs (computed each iteration)
    float* outputs;                          // [num_vertices * NUM_OUTPUTS]
    
    // Conflict history buffer (ring buffer)
    float* conflict_history;                 // [num_vertices * history_length]
    int history_length;
    int history_index;                       // Current write position
    
    // Global reservoir state
    float global_activity;                   // Average reservoir activity
    float conflict_trend;                    // Global conflict direction
    int iteration_count;                     // Total iterations processed
};

/**
 * Input signals from WHCR iteration
 */
struct WHCRIterationState {
    float* conflict_counts;                  // [num_vertices] current conflicts
    float* conflict_deltas;                  // [num_vertices] change from last iteration
    int* colors;                             // [num_vertices] current coloring
    int* moves_applied;                      // [num_vertices] 1 if moved this iteration
    float* wavelet_details;                  // [num_vertices] from WHCR
    int num_colors;
    int iteration;
};

// ═══════════════════════════════════════════════════════════════════════════
// INITIALIZATION KERNELS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Initialize reservoir topology based on graph adjacency
 * Creates sparse recurrent connections that mirror conflict structure
 */
__global__ void init_reservoir_topology(
    const int* __restrict__ graph_row_ptr,
    const int* __restrict__ graph_col_idx,
    const float* __restrict__ initial_conflicts,
    int* __restrict__ connection_row_ptr,
    ReservoirConnection* __restrict__ connections,
    int num_vertices,
    float sparsity,
    unsigned long long seed
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_vertices) {
        // Initialize RNG for this thread
        curandState state;
        curand_init(seed, tid, 0, &state);
        
        int graph_start = graph_row_ptr[tid];
        int graph_end = graph_row_ptr[tid + 1];
        int degree = graph_end - graph_start;
        
        // Connection count for this vertex based on degree and sparsity
        int target_connections = max(1, (int)(degree * sparsity * NUM_COMPARTMENTS));
        
        int conn_idx = connection_row_ptr[tid];
        int conn_count = 0;
        
        // Create connections to neighbors with conflict-weighted probability
        for (int i = graph_start; i < graph_end && conn_count < target_connections; i++) {
            int neighbor = graph_col_idx[i];
            
            // Higher conflict neighbors get more connections
            float conflict_weight = initial_conflicts[neighbor] + 0.1f;
            float connection_prob = sparsity * conflict_weight;
            
            if (curand_uniform(&state) < connection_prob) {
                // Select compartments
                int src_comp = curand(&state) % NUM_COMPARTMENTS;
                int tgt_comp = curand(&state) % NUM_COMPARTMENTS;
                
                // Weight scaled by spectral radius constraint
                float weight = (curand_uniform(&state) * 2.0f - 1.0f) * SPECTRAL_RADIUS / sqrtf((float)degree);
                
                connections[conn_idx + conn_count] = {
                    .source_vertex = tid,
                    .target_vertex = neighbor,
                    .source_compartment = src_comp,
                    .target_compartment = tgt_comp,
                    .weight = weight,
                    .delay = curand_uniform(&state) * 3.0f  // 0-3 iteration delay
                };
                conn_count++;
            }
        }
        
        // Update actual connection count
        // Note: In practice, would need atomic or two-pass approach
    }
}

/**
 * Initialize vertex dendritic states
 */
__global__ void init_vertex_states(
    VertexDendriticState* __restrict__ states,
    const float* __restrict__ initial_conflicts,
    int num_vertices,
    unsigned long long seed
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_vertices) {
        curandState rng;
        curand_init(seed + 1000, tid, 0, &rng);
        
        VertexDendriticState state;
        
        // Initialize compartments with small random activations
        for (int c = 0; c < NUM_COMPARTMENTS; c++) {
            state.compartments[c].activation = curand_uniform(&rng) * 0.1f;
            state.compartments[c].calcium = 0.0f;
            state.compartments[c].threshold = 0.5f + curand_uniform(&rng) * 0.2f;
            state.compartments[c].refractory = 0.0f;
        }
        
        state.soma_potential = 0.0f;
        state.spike_history = 0.0f;
        state.conflict_memory = initial_conflicts[tid];
        state.last_repair_iteration = -100;  // Never repaired
        
        states[tid] = state;
    }
}

/**
 * Initialize input projection weights
 * Maps conflict signals to dendritic compartments
 */
__global__ void init_input_weights(
    float* __restrict__ input_weights,
    int num_vertices,
    unsigned long long seed
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_vertices * NUM_COMPARTMENTS) {
        curandState rng;
        curand_init(seed + 2000, tid, 0, &rng);
        
        int vertex = tid / NUM_COMPARTMENTS;
        int compartment = tid % NUM_COMPARTMENTS;
        
        // Different compartments respond differently to input
        float base_weight = INPUT_SCALING;
        
        switch (compartment) {
            case COMPARTMENT_PROXIMAL:
                // Proximal: strong direct response to current conflicts
                base_weight *= 1.5f;
                break;
            case COMPARTMENT_DISTAL_1:
                // Distal 1: moderate response, integrates over time
                base_weight *= 1.0f;
                break;
            case COMPARTMENT_DISTAL_2:
                // Distal 2: weak direct response, mostly recurrent
                base_weight *= 0.5f;
                break;
            case COMPARTMENT_SPINE:
                // Spine: very weak direct, long-term memory
                base_weight *= 0.2f;
                break;
        }
        
        // Add small random variation
        input_weights[tid] = base_weight * (0.8f + curand_uniform(&rng) * 0.4f);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// DENDRITIC COMPARTMENT DYNAMICS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Process input through dendritic compartments
 * Each compartment has different temporal dynamics
 */
__global__ void process_dendritic_input(
    VertexDendriticState* __restrict__ states,
    const float* __restrict__ input_weights,
    const WHCRIterationState input,
    int num_vertices
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_vertices) {
        VertexDendriticState state = states[tid];
        
        // Compute input signal: combination of conflict count and delta
        float conflict_signal = input.conflict_counts[tid];
        float delta_signal = input.conflict_deltas[tid];
        float wavelet_signal = input.wavelet_details ? input.wavelet_details[tid] : 0.0f;
        
        // Was this vertex just repaired?
        bool just_repaired = input.moves_applied[tid] != 0;
        if (just_repaired) {
            state.last_repair_iteration = input.iteration;
        }
        
        // Time since last repair (affects receptivity)
        int repair_age = input.iteration - state.last_repair_iteration;
        float repair_decay = expf(-0.1f * repair_age);
        
        // Process each compartment
        for (int c = 0; c < NUM_COMPARTMENTS; c++) {
            DendriticCompartment* comp = &state.compartments[c];
            float tau = TAU_DECAY[c];
            float w = input_weights[tid * NUM_COMPARTMENTS + c];
            
            // Decay existing activation
            comp->activation *= tau;
            
            // Add new input (different compartments weight inputs differently)
            float input_contribution = 0.0f;
            
            switch (c) {
                case COMPARTMENT_PROXIMAL:
                    // Responds to current conflict level
                    input_contribution = w * conflict_signal;
                    break;
                case COMPARTMENT_DISTAL_1:
                    // Responds to conflict changes (derivative)
                    input_contribution = w * (conflict_signal + delta_signal * 2.0f);
                    break;
                case COMPARTMENT_DISTAL_2:
                    // Integrates wavelet detail (structural importance)
                    input_contribution = w * (conflict_signal + fabsf(wavelet_signal));
                    break;
                case COMPARTMENT_SPINE:
                    // Long-term conflict exposure memory
                    input_contribution = w * conflict_signal * 0.3f;
                    // Add to calcium for LTP-like memory
                    comp->calcium += conflict_signal * 0.01f;
                    comp->calcium = fminf(comp->calcium, 1.0f);
                    break;
            }
            
            // Apply input if not in refractory period
            if (comp->refractory <= 0.0f) {
                comp->activation += input_contribution;
            } else {
                comp->refractory -= 1.0f;
            }
            
            // Homeostatic threshold adaptation
            comp->threshold += 0.001f * (comp->activation - comp->threshold);
            comp->threshold = fmaxf(0.3f, fminf(0.9f, comp->threshold));
        }
        
        // Update conflict memory (exponential moving average)
        state.conflict_memory = 0.9f * state.conflict_memory + 0.1f * conflict_signal;
        
        states[tid] = state;
    }
}

/**
 * Process recurrent connections between compartments
 * This is where reservoir computing magic happens
 */
__global__ void process_recurrent_connections(
    VertexDendriticState* __restrict__ states,
    const int* __restrict__ connection_row_ptr,
    const ReservoirConnection* __restrict__ connections,
    float* __restrict__ delayed_buffer,  // For synaptic delays
    int num_vertices,
    int iteration
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_vertices) {
        int start = connection_row_ptr[tid];
        int end = connection_row_ptr[tid + 1];
        
        // Accumulate input from recurrent connections
        float recurrent_input[NUM_COMPARTMENTS] = {0.0f, 0.0f, 0.0f, 0.0f};
        
        for (int i = start; i < end; i++) {
            ReservoirConnection conn = connections[i];
            
            // Check if delay has elapsed
            int delay_steps = (int)conn.delay;
            // In full implementation, would use delay buffer
            
            // Get source activation
            float source_act = states[conn.source_vertex].compartments[conn.source_compartment].activation;
            
            // Apply nonlinearity (tanh) for echo state property
            float transmitted = tanhf(source_act) * conn.weight;
            
            recurrent_input[conn.target_compartment] += transmitted;
        }
        
        // Apply recurrent input with leak rate
        VertexDendriticState state = states[tid];
        for (int c = 0; c < NUM_COMPARTMENTS; c++) {
            state.compartments[c].activation = 
                (1.0f - LEAK_RATE) * state.compartments[c].activation +
                LEAK_RATE * tanhf(recurrent_input[c]);
        }
        states[tid] = state;
    }
}

/**
 * Compute soma integration and spike generation
 * Integrates all compartments into final vertex activity
 */
__global__ void compute_soma_integration(
    VertexDendriticState* __restrict__ states,
    int num_vertices
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_vertices) {
        VertexDendriticState state = states[tid];
        
        // Weight compartments by distance from soma
        // Proximal has strongest influence
        float soma_input = 0.0f;
        float weights[NUM_COMPARTMENTS] = {1.0f, 0.6f, 0.3f, 0.1f};
        
        for (int c = 0; c < NUM_COMPARTMENTS; c++) {
            soma_input += weights[c] * state.compartments[c].activation;
        }
        
        // Leaky integrate
        state.soma_potential = 0.8f * state.soma_potential + 0.2f * soma_input;
        
        // Check for spike
        float threshold = 0.5f;
        if (state.soma_potential > threshold) {
            // Spike!
            state.spike_history = 0.9f * state.spike_history + 0.1f;
            state.soma_potential = 0.0f;  // Reset
            
            // Put proximal compartment in refractory
            state.compartments[COMPARTMENT_PROXIMAL].refractory = 2.0f;
        } else {
            state.spike_history *= 0.95f;  // Decay spike history
        }
        
        states[tid] = state;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// OUTPUT COMPUTATION KERNELS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Compute reservoir outputs for WHCR guidance
 * These outputs modulate the next WHCR iteration
 */
__global__ void compute_reservoir_outputs(
    const VertexDendriticState* __restrict__ states,
    const float* __restrict__ output_weights,
    const float* __restrict__ conflict_history,
    int history_length,
    int history_index,
    float* __restrict__ outputs,
    int num_vertices,
    int iteration
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_vertices) {
        VertexDendriticState state = states[tid];
        
        // ═══════════════════════════════════════════════════════════════════
        // OUTPUT 0: Priority Modulation
        // How much should WHCR prioritize this vertex?
        // ═══════════════════════════════════════════════════════════════════
        
        // High proximal activation = acute conflict needing attention
        // High distal activation = persistent structural issue
        // High spine calcium = long-term problematic vertex
        
        float priority_mod = 
            state.compartments[COMPARTMENT_PROXIMAL].activation * 2.0f +
            state.compartments[COMPARTMENT_DISTAL_2].activation * 1.5f +
            state.compartments[COMPARTMENT_SPINE].calcium * 3.0f;
        
        // Boost if hasn't been repaired recently
        int repair_age = iteration - state.last_repair_iteration;
        if (repair_age > 10) {
            priority_mod *= 1.2f;
        }
        
        outputs[tid * NUM_OUTPUTS + OUTPUT_PRIORITY_MOD] = priority_mod;
        
        // ═══════════════════════════════════════════════════════════════════
        // OUTPUT 1: Cascade Prediction
        // Will repairing this vertex help neighbors?
        // ═══════════════════════════════════════════════════════════════════
        
        // High soma potential with low spike history = building pressure
        // that would be released by a repair
        float cascade_potential = state.soma_potential * (1.0f - state.spike_history);
        
        // Distal compartments indicate connectivity to other problem areas
        cascade_potential += state.compartments[COMPARTMENT_DISTAL_1].activation * 0.5f;
        
        outputs[tid * NUM_OUTPUTS + OUTPUT_CASCADE_PRED] = cascade_potential;
        
        // ═══════════════════════════════════════════════════════════════════
        // OUTPUT 2: Repair Receptivity
        // How likely is a repair attempt to succeed?
        // ═══════════════════════════════════════════════════════════════════
        
        // Low recent activity = stable, good candidate
        // High activity = in flux, might fail
        float receptivity = 1.0f - state.spike_history;
        
        // Recently repaired vertices are less receptive (avoid oscillation)
        if (repair_age < 5) {
            receptivity *= 0.3f;
        }
        
        // Check conflict memory trend
        // If conflict_memory decreasing, more receptive
        if (state.conflict_memory < state.compartments[COMPARTMENT_SPINE].calcium) {
            receptivity *= 1.2f;
        }
        
        outputs[tid * NUM_OUTPUTS + OUTPUT_RECEPTIVITY] = fmaxf(0.0f, fminf(1.0f, receptivity));
        
        // ═══════════════════════════════════════════════════════════════════
        // OUTPUT 3: Conflict Momentum
        // Is conflict at this vertex increasing or decreasing?
        // ═══════════════════════════════════════════════════════════════════
        
        // Compare fast vs slow compartments
        // If proximal > distal_2, conflicts are increasing
        float momentum = state.compartments[COMPARTMENT_PROXIMAL].activation - 
                        state.compartments[COMPARTMENT_DISTAL_2].activation;
        
        // Positive = getting worse, negative = improving
        outputs[tid * NUM_OUTPUTS + OUTPUT_MOMENTUM] = momentum;
    }
}

/**
 * Combine reservoir outputs with WHCR wavelet priorities
 * This is the key integration point
 */
__global__ void modulate_whcr_priorities(
    const float* __restrict__ reservoir_outputs,
    const float* __restrict__ wavelet_priorities,
    float* __restrict__ final_priorities,
    int num_vertices,
    float reservoir_influence  // 0.0 to 1.0, how much DR affects priorities
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_vertices) {
        float wavelet_p = wavelet_priorities[tid];
        
        // Skip non-conflicting vertices
        if (wavelet_p < 0.0f) {
            final_priorities[tid] = wavelet_p;
            return;
        }
        
        // Get reservoir outputs for this vertex
        float priority_mod = reservoir_outputs[tid * NUM_OUTPUTS + OUTPUT_PRIORITY_MOD];
        float cascade_pred = reservoir_outputs[tid * NUM_OUTPUTS + OUTPUT_CASCADE_PRED];
        float receptivity = reservoir_outputs[tid * NUM_OUTPUTS + OUTPUT_RECEPTIVITY];
        float momentum = reservoir_outputs[tid * NUM_OUTPUTS + OUTPUT_MOMENTUM];
        
        // Compute reservoir-based priority adjustment
        float reservoir_score = 
            priority_mod * 1.0f +           // Base priority from reservoir
            cascade_pred * 1.5f +            // Bonus for cascade potential
            receptivity * 0.5f +             // Bonus for repair likelihood
            momentum * 0.3f;                 // Penalty if getting worse (focus elsewhere)
        
        // Blend wavelet and reservoir priorities
        float blended = (1.0f - reservoir_influence) * wavelet_p + 
                       reservoir_influence * (wavelet_p + reservoir_score);
        
        final_priorities[tid] = blended;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CONFLICT HISTORY AND PATTERN DETECTION
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Update conflict history ring buffer
 */
__global__ void update_conflict_history(
    float* __restrict__ conflict_history,
    const float* __restrict__ current_conflicts,
    int num_vertices,
    int history_length,
    int history_index
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_vertices) {
        conflict_history[history_index * num_vertices + tid] = current_conflicts[tid];
    }
}

/**
 * Detect oscillating vertices (conflict cycling)
 * These need special handling to break cycles
 */
__global__ void detect_oscillations(
    const float* __restrict__ conflict_history,
    float* __restrict__ oscillation_scores,
    int num_vertices,
    int history_length,
    int history_index
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_vertices) {
        // Look for sign changes in conflict delta
        int sign_changes = 0;
        float prev_delta = 0.0f;
        
        for (int i = 1; i < history_length; i++) {
            int idx_curr = ((history_index - i + history_length) % history_length) * num_vertices + tid;
            int idx_prev = ((history_index - i - 1 + history_length) % history_length) * num_vertices + tid;
            
            float curr_delta = conflict_history[idx_curr] - conflict_history[idx_prev];
            
            if (i > 1 && prev_delta * curr_delta < 0.0f) {
                sign_changes++;
            }
            prev_delta = curr_delta;
        }
        
        // High sign changes = oscillating
        oscillation_scores[tid] = (float)sign_changes / (float)(history_length - 2);
    }
}

/**
 * Detect stubborn conflicts (persistent across many iterations)
 */
__global__ void detect_stubborn_conflicts(
    const float* __restrict__ conflict_history,
    const VertexDendriticState* __restrict__ states,
    float* __restrict__ stubborn_scores,
    int num_vertices,
    int history_length,
    int history_index,
    float threshold
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_vertices) {
        // Count how many history slots had conflicts
        int conflict_count = 0;
        
        for (int i = 0; i < history_length; i++) {
            int idx = i * num_vertices + tid;
            if (conflict_history[idx] > threshold) {
                conflict_count++;
            }
        }
        
        float persistence = (float)conflict_count / (float)history_length;
        
        // Combine with spine compartment calcium (long-term memory)
        float calcium = states[tid].compartments[COMPARTMENT_SPINE].calcium;
        
        stubborn_scores[tid] = 0.6f * persistence + 0.4f * calcium;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GLOBAL RESERVOIR STATE COMPUTATION
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Compute global reservoir statistics
 * Used for adaptive control of reservoir influence
 */
__global__ void compute_global_state(
    const VertexDendriticState* __restrict__ states,
    const float* __restrict__ current_conflicts,
    float* __restrict__ global_activity,
    float* __restrict__ conflict_trend,
    float* __restrict__ prev_total_conflicts,
    int num_vertices
) {
    // Use parallel reduction
    __shared__ float s_activity[BLOCK_SIZE];
    __shared__ float s_conflicts[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load and accumulate local values
    float local_activity = 0.0f;
    float local_conflicts = 0.0f;
    
    if (gid < num_vertices) {
        // Activity = average soma potential
        local_activity = states[gid].soma_potential;
        local_conflicts = current_conflicts[gid];
    }
    
    s_activity[tid] = local_activity;
    s_conflicts[tid] = local_conflicts;
    __syncthreads();
    
    // Parallel reduction
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_activity[tid] += s_activity[tid + s];
            s_conflicts[tid] += s_conflicts[tid + s];
        }
        __syncthreads();
    }
    
    // Block 0 thread 0 writes final result
    if (tid == 0) {
        atomicAdd(global_activity, s_activity[0]);
        atomicAdd(conflict_trend, s_conflicts[0]);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ADAPTIVE LEARNING KERNELS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Hebbian-like weight update for reservoir connections
 * Strengthens connections that correlate with successful repairs
 */
__global__ void update_reservoir_weights(
    ReservoirConnection* __restrict__ connections,
    const VertexDendriticState* __restrict__ states,
    const int* __restrict__ successful_repairs,  // Vertices that reduced conflicts
    int num_connections,
    float learning_rate
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_connections) {
        ReservoirConnection conn = connections[tid];
        
        // Hebbian: strengthen if source predicted successful target repair
        float source_act = states[conn.source_vertex].compartments[conn.source_compartment].activation;
        float target_act = states[conn.target_vertex].compartments[conn.target_compartment].activation;
        bool target_success = successful_repairs[conn.target_vertex] != 0;
        
        if (target_success) {
            // LTP: strengthen connection
            float delta = learning_rate * source_act * target_act;
            conn.weight += delta;
        } else if (source_act > 0.5f && target_act > 0.5f) {
            // LTD: weaken if both active but no success
            conn.weight -= learning_rate * 0.1f;
        }
        
        // Enforce weight bounds
        conn.weight = fmaxf(-1.0f, fminf(1.0f, conn.weight));
        
        connections[tid] = conn;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// HOST-SIDE FFI FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

extern "C" {

/**
 * Initialize the dendritic reservoir
 */
void launch_init_reservoir(
    DendriticReservoir* reservoir,
    const int* graph_row_ptr,
    const int* graph_col_idx,
    const float* initial_conflicts,
    int num_vertices,
    unsigned long long seed,
    cudaStream_t stream
) {
    int blocks = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    init_vertex_states<<<blocks, BLOCK_SIZE, 0, stream>>>(
        reservoir->vertex_states,
        initial_conflicts,
        num_vertices,
        seed
    );
    
    init_input_weights<<<(num_vertices * NUM_COMPARTMENTS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
        reservoir->input_weights,
        num_vertices,
        seed
    );
}

/**
 * Single reservoir update step (call after each WHCR iteration)
 */
void launch_reservoir_step(
    DendriticReservoir* reservoir,
    const WHCRIterationState* whcr_state,
    cudaStream_t stream
) {
    int blocks = (reservoir->num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // 1. Process input through dendritic compartments
    process_dendritic_input<<<blocks, BLOCK_SIZE, 0, stream>>>(
        reservoir->vertex_states,
        reservoir->input_weights,
        *whcr_state,
        reservoir->num_vertices
    );
    
    // 2. Process recurrent connections
    process_recurrent_connections<<<blocks, BLOCK_SIZE, 0, stream>>>(
        reservoir->vertex_states,
        reservoir->connection_row_ptr,
        reservoir->connections,
        nullptr,  // delayed_buffer - simplified for now
        reservoir->num_vertices,
        whcr_state->iteration
    );
    
    // 3. Compute soma integration
    compute_soma_integration<<<blocks, BLOCK_SIZE, 0, stream>>>(
        reservoir->vertex_states,
        reservoir->num_vertices
    );
    
    // 4. Compute outputs for WHCR
    compute_reservoir_outputs<<<blocks, BLOCK_SIZE, 0, stream>>>(
        reservoir->vertex_states,
        reservoir->output_weights,
        reservoir->conflict_history,
        reservoir->history_length,
        reservoir->history_index,
        reservoir->outputs,
        reservoir->num_vertices,
        whcr_state->iteration
    );
    
    // 5. Update conflict history
    update_conflict_history<<<blocks, BLOCK_SIZE, 0, stream>>>(
        reservoir->conflict_history,
        whcr_state->conflict_counts,
        reservoir->num_vertices,
        reservoir->history_length,
        reservoir->history_index
    );
    
    reservoir->history_index = (reservoir->history_index + 1) % reservoir->history_length;
    reservoir->iteration_count++;
}

/**
 * Get modulated priorities for WHCR
 */
void launch_get_modulated_priorities(
    const DendriticReservoir* reservoir,
    const float* wavelet_priorities,
    float* final_priorities,
    float reservoir_influence,
    cudaStream_t stream
) {
    int blocks = (reservoir->num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    modulate_whcr_priorities<<<blocks, BLOCK_SIZE, 0, stream>>>(
        reservoir->outputs,
        wavelet_priorities,
        final_priorities,
        reservoir->num_vertices,
        reservoir_influence
    );
}

/**
 * Detect problematic patterns (oscillations, stubborn conflicts)
 */
void launch_pattern_detection(
    const DendriticReservoir* reservoir,
    float* oscillation_scores,
    float* stubborn_scores,
    float stubborn_threshold,
    cudaStream_t stream
) {
    int blocks = (reservoir->num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    detect_oscillations<<<blocks, BLOCK_SIZE, 0, stream>>>(
        reservoir->conflict_history,
        oscillation_scores,
        reservoir->num_vertices,
        reservoir->history_length,
        reservoir->history_index
    );
    
    detect_stubborn_conflicts<<<blocks, BLOCK_SIZE, 0, stream>>>(
        reservoir->conflict_history,
        reservoir->vertex_states,
        stubborn_scores,
        reservoir->num_vertices,
        reservoir->history_length,
        reservoir->history_index,
        stubborn_threshold
    );
}

/**
 * Online learning update (call periodically)
 */
void launch_weight_update(
    DendriticReservoir* reservoir,
    const int* successful_repairs,
    float learning_rate,
    cudaStream_t stream
) {
    int blocks = (reservoir->num_connections + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    update_reservoir_weights<<<blocks, BLOCK_SIZE, 0, stream>>>(
        reservoir->connections,
        reservoir->vertex_states,
        successful_repairs,
        reservoir->num_connections,
        learning_rate
    );
}

} // extern "C"
