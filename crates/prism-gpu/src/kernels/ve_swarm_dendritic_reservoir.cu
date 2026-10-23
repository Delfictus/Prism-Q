//=============================================================================
// PRISM VE-SWARM: Dendritic Residue Graph Reservoir
//
// Revolutionary approach to viral variant prediction that preserves the FULL
// 125-dimensional feature tensor across residues by propagating through a
// dendritic reservoir on the protein contact graph.
//
// Key Innovation: Instead of averaging features across residues (destroying
// signal), we use multi-branch neuromorphic computation on the residue graph
// to create EMERGENT features that encode both local chemistry AND topology.
//
// Architecture:
// - Each residue = 1 reservoir node with 4 dendritic branches
// - Input: 125-dim features per residue
// - Graph: Contact map (CA-CA distance < 8A)
// - Output: 32-dim reservoir state per residue
//
// GPU Layout:
// - 1 warp (32 threads) per residue
// - Each thread computes 1 of 32 reservoir neurons
// - Shared memory for neighbor feature aggregation
//
// Target: Process ~200 residue RBD structure in < 10ms
//=============================================================================

#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

//=============================================================================
// CONFIGURATION
//=============================================================================

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define MAX_RESIDUES 512
#define MAX_NEIGHBORS 64
#define INPUT_FEATURES 125
#define RESERVOIR_DIM 32
#define N_BRANCHES 4
#define N_COMPARTMENTS 4

// Dendritic branch types
#define BRANCH_LOCAL 0      // Self features (proximal dendrite)
#define BRANCH_NEIGHBOR 1   // 1-hop neighbor features (basal dendrite)
#define BRANCH_GLOBAL 2     // Eigenvector centrality weighted (apical dendrite)
#define BRANCH_RECURRENT 3  // Previous state (spine)

// Multi-compartment time constants
#define TAU_PROXIMAL 0.1f    // Fast response to current features
#define TAU_DISTAL_1 0.5f    // Medium-term neighbor integration
#define TAU_DISTAL_2 0.85f   // Slow global context
#define TAU_SPINE 0.95f      // Long-term memory

// Leak-integrate-fire parameters
#define LIF_THRESHOLD 0.8f
#define LIF_RESET 0.2f
#define LIF_REFRACTORY 2

// Feature importance weights (from DMS data analysis)
// TDA features (0-47): topology
// Reservoir features (48-79): base structural
// Physics features (80-91): electrostatics/hydrophobicity
// Fitness features (92-95): ddG, expression
// Cycle features (96-100): dynamics
// Spike features (101-108): LIF outputs

//=============================================================================
// CONSTANT MEMORY
//=============================================================================

// Compartment decay rates
__constant__ float c_tau_decay[N_COMPARTMENTS] = {
    TAU_PROXIMAL, TAU_DISTAL_1, TAU_DISTAL_2, TAU_SPINE
};

// Branch integration weights
__constant__ float c_branch_weights[N_BRANCHES] = {
    0.40f,  // Local (strongest)
    0.30f,  // Neighbor
    0.20f,  // Global
    0.10f   // Recurrent
};

// Feature group weights (which features are most important for binding prediction)
__constant__ float c_feature_group_weights[5] = {
    0.15f,  // TDA (0-47): moderate importance
    0.25f,  // Reservoir (48-79): high importance
    0.25f,  // Physics (80-91): high importance
    0.20f,  // Fitness (92-95): important
    0.15f   // Cycle/Spike (96-108): dynamics
};

// ACE2 interface residue indices (from 6M0J structure analysis)
// These residues are critical for binding and should have higher attention
__constant__ int c_ace2_interface[25] = {
    417, 446, 449, 453, 455, 456, 475, 476, 484, 486,
    487, 489, 490, 493, 494, 496, 498, 500, 501, 502,
    503, 505, 506, 520, 521
};

// Epitope class centers (RBD residue positions for 10 antibody classes)
__constant__ int c_epitope_centers[10] = {
    484, 501, 417, 346, 440, 498, 373, 456, 384, 527
};

//=============================================================================
// DEVICE HELPER FUNCTIONS
//=============================================================================

__device__ __forceinline__ float fast_tanh(float x) {
    float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

__device__ __forceinline__ float fast_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Deterministic reservoir weight generation using sin/cos basis
__device__ __forceinline__ float get_reservoir_weight(int neuron, int feature) {
    float phase = (float)(neuron * 13 + feature * 7) * 0.1f;
    float base = sinf(phase) * 0.3f + cosf(phase * 0.7f) * 0.2f;

    // Emphasize important feature ranges
    if (feature >= 80 && feature < 92) {
        // Physics features (electrostatics, hydrophobicity) - CRITICAL
        base *= 1.5f;
    }
    if (feature >= 92 && feature < 96) {
        // Fitness features (ddG_bind, ddG_stab) - CRITICAL
        base *= 1.8f;
    }
    if (feature >= 96 && feature < 101) {
        // Cycle features (dynamics) - IMPORTANT
        base *= 1.3f;
    }

    return base;
}

// Check if residue is at ACE2 interface
__device__ __forceinline__ bool is_interface_residue(int residue_idx) {
    #pragma unroll
    for (int i = 0; i < 25; i++) {
        if (residue_idx == c_ace2_interface[i] - 331) {  // Convert to 0-indexed RBD
            return true;
        }
    }
    return false;
}

// Get epitope proximity score for residue
__device__ __forceinline__ float get_epitope_proximity(int residue_idx) {
    float min_dist = 1000.0f;
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        float dist = fabsf((float)(residue_idx - (c_epitope_centers[i] - 331)));
        min_dist = fminf(min_dist, dist);
    }
    // Convert to proximity score (closer = higher)
    return expf(-min_dist / 10.0f);
}

//=============================================================================
// SHARED MEMORY STRUCTURE
//=============================================================================

struct __align__(16) DendriticSharedMem {
    // Per-residue features (loaded in tiles)
    float features[32][INPUT_FEATURES + 1];  // +1 for bank conflict avoidance

    // Neighbor aggregation buffers
    float neighbor_sum[32][RESERVOIR_DIM];
    int neighbor_count[32];

    // Compartment states
    float compartment[N_COMPARTMENTS][32];

    // Spike state for LIF dynamics
    int refractory_counter[32];
    float membrane_potential[32];

    // Global context (eigenvector centrality weighted)
    float global_context[RESERVOIR_DIM];

    // Temporary reduction buffer
    float reduction_buf[WARP_SIZE];
};

//=============================================================================
// KERNEL: DENDRITIC RESERVOIR INITIALIZATION
//=============================================================================

/**
 * Initialize reservoir state with random weights per residue.
 * Uses cuRAND-like deterministic initialization for reproducibility.
 */
extern "C" __global__ void ve_swarm_init_reservoir(
    float* __restrict__ reservoir_state,    // [N_residues x RESERVOIR_DIM]
    const int N_residues,
    const unsigned long long seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N_residues * RESERVOIR_DIM) {
        int residue = idx / RESERVOIR_DIM;
        int neuron = idx % RESERVOIR_DIM;

        // Deterministic pseudo-random initialization
        unsigned long long state = seed ^ (idx * 0x5DEECE66DULL);
        state = state * 0x5DEECE66DULL + 0xBULL;
        float rand_val = ((float)(state >> 33) / (float)(1ULL << 31)) - 0.5f;

        // Scale by interface proximity (interface residues start more active)
        float interface_boost = is_interface_residue(residue) ? 1.5f : 1.0f;

        reservoir_state[idx] = rand_val * 0.1f * interface_boost;
    }
}

//=============================================================================
// KERNEL: DENDRITIC RESERVOIR PROPAGATION
//=============================================================================

/**
 * Main dendritic reservoir propagation kernel.
 *
 * Processes the full 125-dim feature tensor through a multi-branch
 * neuromorphic reservoir on the protein contact graph.
 *
 * Each residue has 4 dendritic branches:
 * 1. Local: Process self features
 * 2. Neighbor: Aggregate 1-hop neighbor features
 * 3. Global: Eigenvector centrality weighted context
 * 4. Recurrent: Previous reservoir state
 *
 * Output is 32-dim reservoir state that encodes BOTH features AND topology.
 */
extern "C" __global__ void __launch_bounds__(BLOCK_SIZE, 4)
ve_swarm_dendritic_reservoir(
    const float* __restrict__ features,      // [N_residues x INPUT_FEATURES]
    const int* __restrict__ csr_row,         // [N_residues + 1]
    const int* __restrict__ csr_col,         // [N_edges]
    const float* __restrict__ csr_weight,    // [N_edges] contact strengths
    const float* __restrict__ eigenvector,   // [N_residues] centrality
    float* __restrict__ reservoir_state,     // [N_residues x RESERVOIR_DIM]
    float* __restrict__ reservoir_prev,      // [N_residues x RESERVOIR_DIM]
    const int N_residues,
    const int iteration
) {
    extern __shared__ DendriticSharedMem smem[];

    int residue = blockIdx.x;
    int neuron = threadIdx.x % RESERVOIR_DIM;
    int local_id = threadIdx.x;

    if (residue >= N_residues) return;

    // =========================================================================
    // STAGE 1: Load self features into shared memory
    // =========================================================================

    // Coalesced load of features for this residue
    int feature_offset = residue * INPUT_FEATURES;

    // Each thread loads multiple features
    for (int f = local_id; f < INPUT_FEATURES; f += blockDim.x) {
        smem->features[0][f] = features[feature_offset + f];
    }
    __syncthreads();

    // =========================================================================
    // STAGE 2: BRANCH 0 - Local dendrite (self features)
    // =========================================================================

    float local_activation = 0.0f;

    // Each neuron computes weighted sum of all features
    #pragma unroll 4
    for (int f = 0; f < INPUT_FEATURES; f++) {
        float w = get_reservoir_weight(neuron, f);
        local_activation += w * smem->features[0][f];
    }

    // Apply non-linearity
    local_activation = fast_tanh(local_activation);

    // Interface boost for binding-critical residues
    if (is_interface_residue(residue)) {
        local_activation *= 1.3f;
    }

    // =========================================================================
    // STAGE 3: BRANCH 1 - Neighbor dendrite (1-hop aggregation)
    // =========================================================================

    float neighbor_activation = 0.0f;

    int neighbor_start = csr_row[residue];
    int neighbor_end = csr_row[residue + 1];
    int degree = neighbor_end - neighbor_start;

    if (degree > 0) {
        // Aggregate neighbor reservoir states weighted by contact strength
        for (int e = neighbor_start; e < neighbor_end; e++) {
            int neighbor = csr_col[e];
            float weight = csr_weight[e];

            // Get neighbor's previous reservoir state for this neuron
            float neighbor_state = reservoir_prev[neighbor * RESERVOIR_DIM + neuron];
            neighbor_activation += weight * neighbor_state;
        }
        neighbor_activation /= (float)degree;

        // Higher degree = more connected = potentially more important
        float degree_boost = 1.0f + 0.1f * fminf((float)degree, 10.0f);
        neighbor_activation *= degree_boost;
    }

    neighbor_activation = fast_tanh(neighbor_activation);

    // =========================================================================
    // STAGE 4: BRANCH 2 - Global dendrite (eigenvector centrality weighted)
    // =========================================================================

    float global_activation = 0.0f;

    // Weight by eigenvector centrality (how central is this residue?)
    float centrality = eigenvector[residue];

    // Global context: Mean of all reservoir states weighted by centrality
    // This is approximated using a subset for efficiency
    if (local_id == 0) {
        float global_sum = 0.0f;
        float weight_sum = 0.0f;

        // Sample uniformly across residues
        for (int r = 0; r < N_residues; r += max(1, N_residues / 32)) {
            float r_centrality = eigenvector[r];
            float r_state = reservoir_prev[r * RESERVOIR_DIM + neuron];
            global_sum += r_centrality * r_state;
            weight_sum += r_centrality;
        }

        smem->global_context[neuron] = (weight_sum > 0.0f) ?
            global_sum / weight_sum : 0.0f;
    }
    __syncthreads();

    global_activation = centrality * smem->global_context[neuron];
    global_activation = fast_tanh(global_activation);

    // =========================================================================
    // STAGE 5: BRANCH 3 - Recurrent dendrite (previous state)
    // =========================================================================

    float recurrent_activation = reservoir_prev[residue * RESERVOIR_DIM + neuron];

    // Decay based on iteration (older states fade)
    float decay = 0.95f - 0.01f * fminf((float)iteration, 20.0f);
    recurrent_activation *= decay;

    // =========================================================================
    // STAGE 6: Multi-compartment integration
    // =========================================================================

    // Proximal compartment (fast, local)
    float proximal = c_tau_decay[0] * local_activation +
                     (1.0f - c_tau_decay[0]) * 0.0f;

    // Distal-1 compartment (medium, neighbor)
    float distal1 = c_tau_decay[1] * neighbor_activation +
                    (1.0f - c_tau_decay[1]) * proximal;

    // Distal-2 compartment (slow, global)
    float distal2 = c_tau_decay[2] * global_activation +
                    (1.0f - c_tau_decay[2]) * distal1;

    // Spine compartment (very slow, recurrent)
    float spine = c_tau_decay[3] * recurrent_activation +
                  (1.0f - c_tau_decay[3]) * distal2;

    // =========================================================================
    // STAGE 7: Branch integration (soma)
    // =========================================================================

    float soma = c_branch_weights[0] * local_activation +
                 c_branch_weights[1] * neighbor_activation +
                 c_branch_weights[2] * global_activation +
                 c_branch_weights[3] * recurrent_activation;

    // Add compartment contributions
    soma += 0.2f * (proximal + distal1 + distal2 + spine);

    // Final non-linearity
    soma = fast_tanh(soma);

    // =========================================================================
    // STAGE 8: Epitope proximity modulation
    // =========================================================================

    // Residues near epitope centers get activation boost
    float epitope_weight = get_epitope_proximity(residue);
    soma *= (1.0f + 0.3f * epitope_weight);

    // =========================================================================
    // STAGE 9: Write output
    // =========================================================================

    reservoir_state[residue * RESERVOIR_DIM + neuron] = soma;
}

//=============================================================================
// KERNEL: COMPUTE ATTENTION WEIGHTS
//=============================================================================

/**
 * Compute attention weights over residues based on:
 * 1. Reservoir state magnitude (active residues)
 * 2. ACE2 interface proximity
 * 3. Epitope class proximity
 * 4. Degree centrality
 */
extern "C" __global__ void ve_swarm_compute_attention(
    const float* __restrict__ reservoir_state,  // [N_residues x RESERVOIR_DIM]
    const float* __restrict__ eigenvector,      // [N_residues]
    const int* __restrict__ csr_row,            // [N_residues + 1]
    float* __restrict__ attention_weights,       // [N_residues]
    float* __restrict__ attended_features,       // [INPUT_FEATURES]
    const float* __restrict__ features,          // [N_residues x INPUT_FEATURES]
    const int N_residues,
    const float temperature
) {
    extern __shared__ float smem_attention[];

    int residue = blockIdx.x * blockDim.x + threadIdx.x;

    if (residue >= N_residues) return;

    // =========================================================================
    // Compute raw attention score
    // =========================================================================

    // 1. Reservoir state magnitude
    float state_magnitude = 0.0f;
    for (int n = 0; n < RESERVOIR_DIM; n++) {
        float s = reservoir_state[residue * RESERVOIR_DIM + n];
        state_magnitude += s * s;
    }
    state_magnitude = sqrtf(state_magnitude);

    // 2. Interface proximity
    float interface_score = is_interface_residue(residue) ? 2.0f : 0.0f;

    // 3. Epitope proximity
    float epitope_score = get_epitope_proximity(residue);

    // 4. Degree centrality
    int degree = csr_row[residue + 1] - csr_row[residue];
    float degree_score = fminf((float)degree / 10.0f, 1.0f);

    // 5. Eigenvector centrality
    float centrality = eigenvector[residue];

    // Combine scores
    float raw_attention =
        0.25f * state_magnitude +
        0.30f * interface_score +
        0.20f * epitope_score +
        0.15f * degree_score +
        0.10f * centrality;

    // Temperature-scaled attention
    smem_attention[threadIdx.x] = expf(raw_attention / temperature);
    __syncthreads();

    // =========================================================================
    // Softmax normalization (block-level reduction)
    // =========================================================================

    // Compute sum for softmax
    float sum = 0.0f;
    for (int i = 0; i < min(blockDim.x, N_residues - blockIdx.x * blockDim.x); i++) {
        sum += smem_attention[i];
    }

    // Global reduction needed for full softmax (simplified here)
    float attention = smem_attention[threadIdx.x] / fmaxf(sum, 1e-6f);

    // Write attention weight
    attention_weights[residue] = attention;
}

//=============================================================================
// KERNEL: AGGREGATE ATTENDED FEATURES
//=============================================================================

/**
 * Compute attention-weighted feature aggregation.
 *
 * Output: 125-dim attended feature vector (weighted sum across residues)
 */
extern "C" __global__ void ve_swarm_aggregate_features(
    const float* __restrict__ features,          // [N_residues x INPUT_FEATURES]
    const float* __restrict__ attention_weights, // [N_residues]
    float* __restrict__ attended_features,       // [INPUT_FEATURES]
    const int N_residues
) {
    int feature = blockIdx.x * blockDim.x + threadIdx.x;

    if (feature >= INPUT_FEATURES) return;

    float weighted_sum = 0.0f;
    float weight_sum = 0.0f;

    for (int r = 0; r < N_residues; r++) {
        float w = attention_weights[r];
        weighted_sum += w * features[r * INPUT_FEATURES + feature];
        weight_sum += w;
    }

    attended_features[feature] = weighted_sum / fmaxf(weight_sum, 1e-6f);
}

//=============================================================================
// KERNEL: RESERVOIR OUTPUT READOUT
//=============================================================================

/**
 * Compute final reservoir output for each residue.
 *
 * Combines reservoir state with original features via learned readout weights.
 * Output can be used for downstream tasks (classification, regression).
 */
extern "C" __global__ void ve_swarm_reservoir_readout(
    const float* __restrict__ reservoir_state,  // [N_residues x RESERVOIR_DIM]
    const float* __restrict__ features,         // [N_residues x INPUT_FEATURES]
    const float* __restrict__ readout_weights,  // [RESERVOIR_DIM + INPUT_FEATURES]
    float* __restrict__ output,                 // [N_residues]
    const int N_residues
) {
    int residue = blockIdx.x * blockDim.x + threadIdx.x;

    if (residue >= N_residues) return;

    float score = 0.0f;

    // Reservoir state contribution
    for (int n = 0; n < RESERVOIR_DIM; n++) {
        float w = readout_weights[n];
        float s = reservoir_state[residue * RESERVOIR_DIM + n];
        score += w * s;
    }

    // Feature contribution (key features only)
    // ddG_bind (feature 92), ddG_stab (93), expression (94), velocity (100)
    int key_features[] = {92, 93, 94, 100};
    for (int i = 0; i < 4; i++) {
        int f = key_features[i];
        float w = readout_weights[RESERVOIR_DIM + f];
        float v = features[residue * INPUT_FEATURES + f];
        score += w * v;
    }

    // Sigmoid for probability output
    output[residue] = fast_sigmoid(score);
}

//=============================================================================
// KERNEL: BATCH DENDRITIC PROCESSING
//=============================================================================

/**
 * Process multiple structures in a single kernel launch.
 * Each thread block handles one structure.
 */
extern "C" __global__ void ve_swarm_batch_dendritic(
    const float* __restrict__ features_packed,     // All structures packed
    const int* __restrict__ structure_offsets,     // [N_structures + 1]
    const int* __restrict__ csr_row_packed,        // All CSR rows packed
    const int* __restrict__ csr_col_packed,        // All CSR cols packed
    const float* __restrict__ csr_weight_packed,   // All weights packed
    const int* __restrict__ csr_offsets,           // [N_structures + 1]
    const float* __restrict__ eigenvector_packed,  // All centrality packed
    float* __restrict__ reservoir_out,             // [Total_residues x RESERVOIR_DIM]
    const int N_structures,
    const int iterations
) {
    int structure_id = blockIdx.x;

    if (structure_id >= N_structures) return;

    // Get structure-specific offsets
    int residue_start = structure_offsets[structure_id];
    int residue_end = structure_offsets[structure_id + 1];
    int N_residues = residue_end - residue_start;

    int csr_start = csr_offsets[structure_id];

    // Process this structure's residues
    for (int r = threadIdx.x; r < N_residues; r += blockDim.x) {
        int global_r = residue_start + r;

        // Simplified single-pass computation
        float activation = 0.0f;

        // Local features
        int feature_offset = global_r * INPUT_FEATURES;
        for (int f = 0; f < INPUT_FEATURES; f++) {
            float w = get_reservoir_weight(r % RESERVOIR_DIM, f);
            activation += w * features_packed[feature_offset + f];
        }

        // Neighbor influence (simplified)
        int local_csr_row = r;  // Would need proper offset

        // Write output
        for (int n = 0; n < RESERVOIR_DIM; n++) {
            reservoir_out[global_r * RESERVOIR_DIM + n] = fast_tanh(activation * 0.1f);
        }
    }
}

//=============================================================================
// UTILITY KERNELS
//=============================================================================

/**
 * Copy reservoir state for ping-pong buffering
 */
extern "C" __global__ void ve_swarm_copy_state(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = src[idx];
    }
}

/**
 * Compute eigenvector centrality using power iteration
 */
extern "C" __global__ void ve_swarm_power_iteration(
    const int* __restrict__ csr_row,
    const int* __restrict__ csr_col,
    const float* __restrict__ csr_weight,
    float* __restrict__ eigenvector,
    float* __restrict__ eigenvector_new,
    const int N_residues
) {
    int residue = blockIdx.x * blockDim.x + threadIdx.x;

    if (residue >= N_residues) return;

    int start = csr_row[residue];
    int end = csr_row[residue + 1];

    float sum = 0.0f;
    for (int e = start; e < end; e++) {
        int neighbor = csr_col[e];
        float weight = csr_weight[e];
        sum += weight * eigenvector[neighbor];
    }

    eigenvector_new[residue] = sum;
}

/**
 * Normalize eigenvector to unit length
 */
extern "C" __global__ void ve_swarm_normalize_eigenvector(
    float* __restrict__ eigenvector,
    const int N_residues
) {
    extern __shared__ float smem_norm[];

    int idx = threadIdx.x;

    // Compute local sum of squares
    float local_sum = 0.0f;
    for (int r = idx; r < N_residues; r += blockDim.x) {
        float v = eigenvector[r];
        local_sum += v * v;
    }

    smem_norm[idx] = local_sum;
    __syncthreads();

    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (idx < stride) {
            smem_norm[idx] += smem_norm[idx + stride];
        }
        __syncthreads();
    }

    float norm = sqrtf(smem_norm[0]);

    // Normalize
    for (int r = idx; r < N_residues; r += blockDim.x) {
        eigenvector[r] /= fmaxf(norm, 1e-6f);
    }
}
