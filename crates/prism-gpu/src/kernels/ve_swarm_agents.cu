//=============================================================================
// PRISM VE-SWARM: Swarm Agent Intelligence Kernel
//
// 32 GPU-accelerated agents compete and cooperate to discover the best
// feature combinations for predicting viral variant RISE vs FALL.
//
// Key Innovations:
// 1. Each agent has a binary feature mask (which of 125 features to use)
// 2. Agents make independent predictions
// 3. Fitness-based evolution: successful agents reproduce, failing agents die
// 4. Pheromone trail: successful feature combinations leave "scent"
// 5. Swarm consensus: weighted voting across all agents
//
// Architecture:
// - 32 agents (one per warp lane)
// - Each agent: 125-bit feature mask + 125 learned weights
// - Pheromone: 125-dim shared signal indicating important features
// - Evolution: Every N predictions, top agents reproduce
//
// GPU Layout:
// - 1 warp (32 threads) = 32 agents
// - Warp-level voting for consensus
// - Shared memory for pheromone trail
//
// Target: < 1ms for single prediction, < 100ms for swarm evolution
//=============================================================================

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <curand_kernel.h>

namespace cg = cooperative_groups;

//=============================================================================
// CONFIGURATION
//=============================================================================

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define N_AGENTS 32
#define N_FEATURES 125
#define RESERVOIR_DIM 32
#define TEMPORAL_DIM 64

// Swarm evolution parameters
#define EVOLUTION_INTERVAL 100     // Evolve every N predictions
#define MUTATION_RATE 0.05f        // Probability of flipping a feature
#define CROSSOVER_RATE 0.3f        // Probability of crossover vs mutation
#define ELITE_COUNT 8              // Top N agents survive unchanged
#define DEATH_COUNT 8              // Bottom N agents are replaced

// Pheromone parameters
#define PHEROMONE_DECAY 0.95f      // Decay rate per prediction
#define PHEROMONE_DEPOSIT 0.1f     // Amount deposited on success
#define PHEROMONE_MIN 0.01f        // Minimum pheromone level
#define PHEROMONE_MAX 1.0f         // Maximum pheromone level

// Prediction parameters
#define TEMPERATURE 1.0f           // Softmax temperature
#define CONFIDENCE_THRESHOLD 0.7f  // High confidence predictions

//=============================================================================
// DATA STRUCTURES
//=============================================================================

/**
 * Per-agent state (packed for coalesced access)
 */
struct __align__(16) AgentState {
    // Feature mask: 4 x 32-bit = 128 bits (only use first 125)
    unsigned int feature_mask[4];

    // Learned weights (one per feature)
    float weights[N_FEATURES];

    // Agent statistics
    float fitness;           // Historical accuracy
    int correct_count;       // Total correct predictions
    int total_count;         // Total predictions made
    int age;                 // Generations since birth

    // Specialization: Which feature groups this agent focuses on
    float group_focus[5];    // TDA, Reservoir, Physics, Fitness, Cycle
};

/**
 * Swarm-level shared state
 */
struct __align__(16) SwarmState {
    // Pheromone trail (125 features)
    float pheromone[N_FEATURES];

    // Global statistics
    int prediction_count;
    float swarm_accuracy;
    int generation;

    // Best agent history
    int best_agent_id;
    float best_fitness;
};

//=============================================================================
// CONSTANT MEMORY
//=============================================================================

// Feature group boundaries (TDA, Reservoir, Physics, Fitness, Cycle, Spike, Immunity)
__constant__ int c_feature_groups[8] = {0, 48, 80, 92, 96, 101, 109, 125};

// Feature names for logging (indices of most important features)
__constant__ int c_critical_features[20] = {
    80, 81, 82, 83,  // Physics: electrostatics
    84, 85, 86, 87,  // Physics: hydrophobicity
    92, 93, 94, 95,  // Fitness: ddG, expression
    96, 97, 98, 99, 100,  // Cycle: dynamics
    101, 102, 103    // Spike: LIF outputs
};

// Initial pheromone levels (prior knowledge from DMS analysis)
__constant__ float c_initial_pheromone[N_FEATURES] = {
    // TDA features (0-47): Low initial pheromone (found to be less discriminative)
    0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f,
    0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f,
    0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f,
    0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f,
    0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f,
    0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f,
    // Reservoir features (48-79): Medium pheromone
    0.3f, 0.3f, 0.3f, 0.3f, 0.3f, 0.3f, 0.3f, 0.3f,
    0.3f, 0.3f, 0.3f, 0.3f, 0.3f, 0.3f, 0.3f, 0.3f,
    0.3f, 0.3f, 0.3f, 0.3f, 0.3f, 0.3f, 0.3f, 0.3f,
    0.3f, 0.3f, 0.3f, 0.3f, 0.3f, 0.3f, 0.3f, 0.3f,
    // Physics features (80-91): HIGH pheromone (critical for prediction)
    0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f,
    0.8f, 0.8f, 0.8f, 0.8f,
    // Fitness features (92-95): HIGHEST pheromone (direct predictors)
    0.9f, 0.9f, 0.9f, 0.9f,
    // Cycle features (96-100): HIGH pheromone (dynamics matter)
    0.7f, 0.7f, 0.7f, 0.7f, 0.7f,
    // Spike features (101-108): Medium-high
    0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
    // Immunity features (109-124): HIGH pheromone (critical for VASIL gamma)
    0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f,
    0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f
};

//=============================================================================
// DEVICE HELPER FUNCTIONS
//=============================================================================

__device__ __forceinline__ float fast_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ bool get_feature_bit(const unsigned int* mask, int feature) {
    int word = feature / 32;
    int bit = feature % 32;
    return (mask[word] >> bit) & 1;
}

__device__ __forceinline__ void set_feature_bit(unsigned int* mask, int feature, bool value) {
    int word = feature / 32;
    int bit = feature % 32;
    if (value) {
        mask[word] |= (1u << bit);
    } else {
        mask[word] &= ~(1u << bit);
    }
}

__device__ __forceinline__ void flip_feature_bit(unsigned int* mask, int feature) {
    int word = feature / 32;
    int bit = feature % 32;
    mask[word] ^= (1u << bit);
}

__device__ __forceinline__ int count_active_features(const unsigned int* mask) {
    return __popc(mask[0]) + __popc(mask[1]) + __popc(mask[2]) + __popc(mask[3] & 0x1FFF);
}

// Warp-level sorting for agent ranking
__device__ void warp_sort_agents_by_fitness(
    float& fitness,
    int& agent_id,
    cg::thread_block_tile<32>& warp
) {
    // Bitonic sort within warp
    for (int k = 2; k <= WARP_SIZE; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int lane = warp.thread_rank();
            int ixj = lane ^ j;

            float other_fitness = warp.shfl(fitness, ixj);
            int other_id = warp.shfl(agent_id, ixj);

            bool ascending = ((lane & k) == 0);
            bool swap = ascending ? (fitness < other_fitness) : (fitness > other_fitness);

            if (ixj > lane && swap) {
                fitness = other_fitness;
                agent_id = other_id;
            }
        }
    }
}

//=============================================================================
// KERNEL: INITIALIZE SWARM
//=============================================================================

/**
 * Initialize all 32 agents with diverse feature masks and random weights.
 * Uses pheromone trail as prior for feature selection.
 */
extern "C" __global__ void ve_swarm_init_agents(
    AgentState* __restrict__ agents,
    SwarmState* __restrict__ swarm,
    const unsigned long long seed
) {
    int agent_id = threadIdx.x;

    if (agent_id >= N_AGENTS) return;

    // Initialize RNG
    curandState_t rng;
    curand_init(seed, agent_id, 0, &rng);

    // Initialize feature mask based on pheromone + randomness
    for (int f = 0; f < N_FEATURES; f++) {
        float pheromone = c_initial_pheromone[f];
        float rand_val = curand_uniform(&rng);

        // Higher pheromone = higher chance of including feature
        // Add diversity by varying threshold per agent
        float agent_threshold = 0.5f - 0.3f * ((float)agent_id / N_AGENTS - 0.5f);
        bool include = (pheromone * rand_val) > agent_threshold;

        set_feature_bit(agents[agent_id].feature_mask, f, include);
    }

    // Ensure at least some features are active
    int active = count_active_features(agents[agent_id].feature_mask);
    if (active < 10) {
        // Force-include critical features
        for (int i = 0; i < 10; i++) {
            set_feature_bit(agents[agent_id].feature_mask, c_critical_features[i], true);
        }
    }

    // Initialize weights randomly
    for (int f = 0; f < N_FEATURES; f++) {
        agents[agent_id].weights[f] = (curand_uniform(&rng) - 0.5f) * 0.2f;
    }

    // Initialize statistics
    agents[agent_id].fitness = 0.5f;  // Prior: 50% accuracy
    agents[agent_id].correct_count = 0;
    agents[agent_id].total_count = 0;
    agents[agent_id].age = 0;

    // Specialize: Each agent focuses on different feature groups
    for (int g = 0; g < 5; g++) {
        agents[agent_id].group_focus[g] = curand_uniform(&rng);
    }
    // Normalize
    float sum = 0.0f;
    for (int g = 0; g < 5; g++) sum += agents[agent_id].group_focus[g];
    for (int g = 0; g < 5; g++) agents[agent_id].group_focus[g] /= sum;

    // Initialize swarm state (first thread only)
    if (agent_id == 0) {
        for (int f = 0; f < N_FEATURES; f++) {
            swarm->pheromone[f] = c_initial_pheromone[f];
        }
        swarm->prediction_count = 0;
        swarm->swarm_accuracy = 0.5f;
        swarm->generation = 0;
        swarm->best_agent_id = 0;
        swarm->best_fitness = 0.5f;
    }
}

//=============================================================================
// KERNEL: AGENT PREDICTION
//=============================================================================

/**
 * Each agent makes a prediction based on its feature mask and weights.
 * Predictions are combined with temporal features for final output.
 *
 * Input:
 *   - attended_features[125]: From structural attention
 *   - temporal_embedding[64]: From temporal convolution
 *   - reservoir_state[32]: From dendritic reservoir (aggregated)
 *
 * Output:
 *   - agent_predictions[32]: Each agent's RISE probability
 *   - agent_confidences[32]: Prediction confidence
 */
extern "C" __global__ void ve_swarm_agent_predict(
    const AgentState* __restrict__ agents,
    const float* __restrict__ attended_features,    // [125]
    const float* __restrict__ temporal_embedding,   // [64]
    const float* __restrict__ reservoir_summary,    // [32]
    float* __restrict__ agent_predictions,          // [32]
    float* __restrict__ agent_confidences,          // [32]
    const SwarmState* __restrict__ swarm
) {
    int agent_id = threadIdx.x;

    if (agent_id >= N_AGENTS) return;

    const AgentState& agent = agents[agent_id];

    // =========================================================================
    // Compute prediction based on masked features
    // =========================================================================

    float score = 0.0f;
    float active_weight_sum = 0.0f;

    // Feature contribution
    for (int f = 0; f < N_FEATURES; f++) {
        if (get_feature_bit(agent.feature_mask, f)) {
            float feature_val = attended_features[f];
            float weight = agent.weights[f];
            float pheromone = swarm->pheromone[f];

            // Pheromone-weighted contribution
            score += feature_val * weight * (1.0f + pheromone);
            active_weight_sum += fabsf(weight);
        }
    }

    // Normalize by number of active features
    int active = count_active_features(agent.feature_mask);
    if (active > 0) {
        score /= sqrtf((float)active);
    }

    // =========================================================================
    // Add temporal embedding contribution
    // =========================================================================

    // Temporal features: Recent velocity trend (most important)
    float velocity_trend = 0.0f;
    for (int t = 0; t < 16; t++) {
        velocity_trend += temporal_embedding[t] * (1.0f - (float)t / 16.0f);
    }

    // Frequency trajectory
    float freq_trajectory = 0.0f;
    for (int t = 16; t < 32; t++) {
        freq_trajectory += temporal_embedding[t];
    }

    // Emergence signal
    float emergence = 0.0f;
    for (int t = 32; t < 48; t++) {
        emergence += temporal_embedding[t];
    }

    // Combine temporal with structural
    score += 0.3f * velocity_trend + 0.2f * freq_trajectory + 0.1f * emergence;

    // =========================================================================
    // Add reservoir state contribution
    // =========================================================================

    float reservoir_energy = 0.0f;
    for (int n = 0; n < RESERVOIR_DIM; n++) {
        reservoir_energy += reservoir_summary[n] * reservoir_summary[n];
    }
    reservoir_energy = sqrtf(reservoir_energy);

    score += 0.1f * reservoir_energy;

    // =========================================================================
    // Final prediction
    // =========================================================================

    // Sigmoid for probability
    float prediction = fast_sigmoid(score);

    // Confidence based on how far from 0.5
    float confidence = fabsf(prediction - 0.5f) * 2.0f;

    // Agent fitness bonus: More confident if historically accurate
    confidence *= (0.5f + 0.5f * agent.fitness);

    agent_predictions[agent_id] = prediction;
    agent_confidences[agent_id] = confidence;
}

//=============================================================================
// KERNEL: SWARM CONSENSUS
//=============================================================================

/**
 * Aggregate agent predictions into final swarm prediction.
 * Uses fitness-weighted voting with confidence scaling.
 */
extern "C" __global__ void ve_swarm_consensus(
    const float* __restrict__ agent_predictions,
    const float* __restrict__ agent_confidences,
    const AgentState* __restrict__ agents,
    float* __restrict__ final_prediction,
    float* __restrict__ final_confidence,
    const float current_frequency,
    const float current_velocity
) {
    // Single warp computes consensus
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(cg::this_thread_block());

    int lane = warp.thread_rank();

    float pred = agent_predictions[lane];
    float conf = agent_confidences[lane];
    float fitness = agents[lane].fitness;

    // Fitness-weighted prediction
    float weight = fitness * conf;
    float weighted_pred = pred * weight;

    // Warp-level reduction
    float sum_weighted_pred = weighted_pred;
    float sum_weight = weight;

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum_weighted_pred += warp.shfl_down(sum_weighted_pred, offset);
        sum_weight += warp.shfl_down(sum_weight, offset);
    }

    // Thread 0 computes final result
    if (lane == 0) {
        float consensus = sum_weighted_pred / fmaxf(sum_weight, 1e-6f);

        // =====================================================================
        // Physics-informed constraints
        // =====================================================================

        // Constraint 1: High frequency variants tend to FALL (saturation)
        if (current_frequency > 0.7f) {
            // Bias toward FALL
            consensus = consensus * 0.8f + 0.0f * 0.2f;
        }

        // Constraint 2: INVERTED velocity interpretation
        // High velocity = variant is at PEAK, about to FALL
        if (current_velocity > 0.1f && current_frequency > 0.3f) {
            // High velocity + moderate frequency = likely FALLING soon
            consensus = consensus * 0.7f + 0.3f * 0.3f;
        }

        // Constraint 3: Low frequency + positive velocity = likely RISING
        if (current_frequency < 0.1f && current_velocity > 0.0f) {
            consensus = consensus * 0.8f + 0.7f * 0.2f;
        }

        // Constraint 4: Very low frequency = probably not rising
        if (current_frequency < 0.01f) {
            consensus *= 0.5f;  // Dampen overconfident RISE predictions
        }

        // Final confidence
        float final_conf = 0.0f;
        for (int i = 0; i < N_AGENTS; i++) {
            final_conf += agent_confidences[i] * agents[i].fitness;
        }
        final_conf /= N_AGENTS;

        *final_prediction = consensus;
        *final_confidence = final_conf;
    }
}

//=============================================================================
// KERNEL: UPDATE AGENT STATS AND PHEROMONE
//=============================================================================

/**
 * Update agent statistics after observing true label.
 * Deposit pheromone on features used by successful agents.
 */
extern "C" __global__ void ve_swarm_update_stats(
    AgentState* __restrict__ agents,
    SwarmState* __restrict__ swarm,
    const float* __restrict__ agent_predictions,
    const int true_label,  // 1 = RISE, 0 = FALL
    const float prediction_threshold
) {
    int agent_id = threadIdx.x;

    if (agent_id >= N_AGENTS) return;

    float pred = agent_predictions[agent_id];
    int predicted_label = (pred > prediction_threshold) ? 1 : 0;
    bool correct = (predicted_label == true_label);

    // Update agent statistics
    agents[agent_id].total_count++;
    if (correct) {
        agents[agent_id].correct_count++;
    }

    // Update running fitness (exponential moving average)
    float alpha = 0.1f;
    float new_fitness = correct ? 1.0f : 0.0f;
    agents[agent_id].fitness = (1.0f - alpha) * agents[agent_id].fitness + alpha * new_fitness;

    // =========================================================================
    // Pheromone update (thread 0 only)
    // =========================================================================

    if (agent_id == 0) {
        // Decay all pheromones
        for (int f = 0; f < N_FEATURES; f++) {
            swarm->pheromone[f] *= PHEROMONE_DECAY;
            swarm->pheromone[f] = fmaxf(swarm->pheromone[f], PHEROMONE_MIN);
        }
    }
    __syncthreads();

    // Successful agents deposit pheromone on their active features
    if (correct) {
        float deposit = PHEROMONE_DEPOSIT * agents[agent_id].fitness;

        for (int f = 0; f < N_FEATURES; f++) {
            if (get_feature_bit(agents[agent_id].feature_mask, f)) {
                atomicAdd(&swarm->pheromone[f], deposit);
            }
        }
    }

    // Clamp pheromone
    __syncthreads();
    if (agent_id == 0) {
        for (int f = 0; f < N_FEATURES; f++) {
            swarm->pheromone[f] = fminf(swarm->pheromone[f], PHEROMONE_MAX);
        }

        // Update swarm statistics
        swarm->prediction_count++;

        // Update swarm accuracy
        int total_correct = 0;
        for (int a = 0; a < N_AGENTS; a++) {
            total_correct += agents[a].correct_count;
        }
        int total = 0;
        for (int a = 0; a < N_AGENTS; a++) {
            total += agents[a].total_count;
        }
        swarm->swarm_accuracy = (total > 0) ? (float)total_correct / total : 0.5f;

        // Track best agent
        for (int a = 0; a < N_AGENTS; a++) {
            if (agents[a].fitness > swarm->best_fitness) {
                swarm->best_fitness = agents[a].fitness;
                swarm->best_agent_id = a;
            }
        }
    }
}

//=============================================================================
// KERNEL: SWARM EVOLUTION
//=============================================================================

/**
 * Evolve the swarm: top agents reproduce, bottom agents die.
 * Uses genetic algorithm with crossover and mutation.
 */
extern "C" __global__ void ve_swarm_evolve(
    AgentState* __restrict__ agents,
    SwarmState* __restrict__ swarm,
    const unsigned long long seed
) {
    // Single warp handles evolution
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(cg::this_thread_block());

    int lane = warp.thread_rank();

    // Initialize RNG
    curandState_t rng;
    curand_init(seed + swarm->generation, lane, 0, &rng);

    // =========================================================================
    // Sort agents by fitness
    // =========================================================================

    float my_fitness = agents[lane].fitness;
    int my_rank = lane;

    warp_sort_agents_by_fitness(my_fitness, my_rank, warp);

    // Now my_rank contains sorted order (highest fitness first)

    // =========================================================================
    // Evolution: Replace bottom agents with offspring of top agents
    // =========================================================================

    if (lane >= N_AGENTS - DEATH_COUNT) {
        // I'm in the bottom tier - I get replaced

        // Select two parents from elite (top ELITE_COUNT)
        int parent1 = lane % ELITE_COUNT;
        int parent2 = (lane + 1) % ELITE_COUNT;

        AgentState& child = agents[lane];
        const AgentState& p1 = agents[parent1];
        const AgentState& p2 = agents[parent2];

        // Crossover feature masks
        for (int w = 0; w < 4; w++) {
            if (curand_uniform(&rng) < CROSSOVER_RATE) {
                // Two-point crossover
                int point1 = curand(&rng) % 32;
                int point2 = curand(&rng) % 32;
                if (point1 > point2) { int tmp = point1; point1 = point2; point2 = tmp; }

                unsigned int mask = ((1u << (point2 - point1)) - 1) << point1;
                child.feature_mask[w] = (p1.feature_mask[w] & mask) | (p2.feature_mask[w] & ~mask);
            } else {
                // Copy from random parent
                child.feature_mask[w] = (curand_uniform(&rng) < 0.5f) ?
                    p1.feature_mask[w] : p2.feature_mask[w];
            }
        }

        // Mutate feature mask
        for (int f = 0; f < N_FEATURES; f++) {
            if (curand_uniform(&rng) < MUTATION_RATE) {
                flip_feature_bit(child.feature_mask, f);
            }
        }

        // Ensure minimum features
        if (count_active_features(child.feature_mask) < 5) {
            for (int i = 0; i < 5; i++) {
                set_feature_bit(child.feature_mask, c_critical_features[i], true);
            }
        }

        // Crossover weights (arithmetic crossover)
        for (int f = 0; f < N_FEATURES; f++) {
            float alpha = curand_uniform(&rng);
            child.weights[f] = alpha * p1.weights[f] + (1.0f - alpha) * p2.weights[f];

            // Mutate weights
            if (curand_uniform(&rng) < MUTATION_RATE * 2.0f) {
                child.weights[f] += (curand_uniform(&rng) - 0.5f) * 0.1f;
            }
        }

        // Reset child statistics
        child.fitness = (p1.fitness + p2.fitness) / 2.0f * 0.9f;  // Slight discount
        child.correct_count = 0;
        child.total_count = 0;
        child.age = 0;

        // Inherit group focus with mutation
        for (int g = 0; g < 5; g++) {
            child.group_focus[g] = (p1.group_focus[g] + p2.group_focus[g]) / 2.0f;
            child.group_focus[g] += (curand_uniform(&rng) - 0.5f) * 0.1f;
            child.group_focus[g] = fmaxf(0.0f, child.group_focus[g]);
        }
    }

    // Age all agents
    agents[lane].age++;

    // Update swarm generation
    if (lane == 0) {
        swarm->generation++;
    }
}

//=============================================================================
// KERNEL: COMPUTE FEATURE IMPORTANCE
//=============================================================================

/**
 * Compute importance of each feature based on pheromone and agent usage.
 * Used for analysis and visualization.
 */
extern "C" __global__ void ve_swarm_feature_importance(
    const AgentState* __restrict__ agents,
    const SwarmState* __restrict__ swarm,
    float* __restrict__ importance,  // [125]
    float* __restrict__ usage        // [125]
) {
    int feature = blockIdx.x * blockDim.x + threadIdx.x;

    if (feature >= N_FEATURES) return;

    // Count how many agents use this feature, weighted by fitness
    float weighted_usage = 0.0f;
    float fitness_sum = 0.0f;

    for (int a = 0; a < N_AGENTS; a++) {
        if (get_feature_bit(agents[a].feature_mask, feature)) {
            weighted_usage += agents[a].fitness;
        }
        fitness_sum += agents[a].fitness;
    }

    // Normalize usage
    float normalized_usage = weighted_usage / fmaxf(fitness_sum, 1e-6f);

    // Importance = pheromone * normalized_usage
    float pheromone = swarm->pheromone[feature];
    float feature_importance = pheromone * normalized_usage;

    importance[feature] = feature_importance;
    usage[feature] = normalized_usage;
}

//=============================================================================
// KERNEL: BATCH PREDICTION
//=============================================================================

/**
 * Predict multiple variants in parallel.
 * Each thread block handles one variant, warp handles 32 agents.
 */
extern "C" __global__ void ve_swarm_batch_predict(
    const AgentState* __restrict__ agents,
    const SwarmState* __restrict__ swarm,
    const float* __restrict__ features_batch,        // [N_variants x 125]
    const float* __restrict__ temporal_batch,        // [N_variants x 64]
    const float* __restrict__ reservoir_batch,       // [N_variants x 32]
    const float* __restrict__ frequencies,           // [N_variants]
    const float* __restrict__ velocities,            // [N_variants]
    float* __restrict__ predictions,                 // [N_variants]
    float* __restrict__ confidences,                 // [N_variants]
    const int N_variants
) {
    int variant = blockIdx.x;
    int agent_id = threadIdx.x % N_AGENTS;

    if (variant >= N_variants) return;

    extern __shared__ float smem[];
    float* s_predictions = smem;                     // [32]
    float* s_confidences = smem + N_AGENTS;          // [32]

    // Load features for this variant
    const float* features = features_batch + variant * N_FEATURES;
    const float* temporal = temporal_batch + variant * TEMPORAL_DIM;
    const float* reservoir = reservoir_batch + variant * RESERVOIR_DIM;

    // Agent prediction (same as single prediction kernel)
    float score = 0.0f;
    const AgentState& agent = agents[agent_id];

    for (int f = 0; f < N_FEATURES; f++) {
        if (get_feature_bit(agent.feature_mask, f)) {
            score += features[f] * agent.weights[f] * (1.0f + swarm->pheromone[f]);
        }
    }

    int active = count_active_features(agent.feature_mask);
    if (active > 0) score /= sqrtf((float)active);

    // Temporal contribution
    float velocity_trend = 0.0f;
    for (int t = 0; t < 16; t++) velocity_trend += temporal[t];
    score += 0.3f * velocity_trend;

    float pred = fast_sigmoid(score);
    float conf = fabsf(pred - 0.5f) * 2.0f * agent.fitness;

    s_predictions[agent_id] = pred;
    s_confidences[agent_id] = conf;
    __syncthreads();

    // Consensus (thread 0 only)
    if (threadIdx.x == 0) {
        float sum_wp = 0.0f, sum_w = 0.0f;

        for (int a = 0; a < N_AGENTS; a++) {
            float w = agents[a].fitness * s_confidences[a];
            sum_wp += s_predictions[a] * w;
            sum_w += w;
        }

        float consensus = sum_wp / fmaxf(sum_w, 1e-6f);

        // Physics constraints
        float freq = frequencies[variant];
        float vel = velocities[variant];

        if (freq > 0.7f) consensus *= 0.8f;
        if (vel > 0.1f && freq > 0.3f) consensus = consensus * 0.7f + 0.15f;
        if (freq < 0.01f) consensus *= 0.5f;

        predictions[variant] = consensus;
        confidences[variant] = sum_w / N_AGENTS;
    }
}
