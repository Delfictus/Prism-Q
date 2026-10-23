// Copyright (c) 2024 PRISM Research Team | Delfictus I/O Inc.
// GPU-Accelerated FluxNet Reward Computation for PRISM-LBS
//
// Implements batch reward computation and gradient estimation for
// FluxNet RL weight optimization of druggability scoring.

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

// Prediction features (per pocket):
// [0] center_distance - Distance from pocket center to ligand center (Å)
// [1] volume_overlap - Volume overlap score
// [2] coverage - Ligand coverage fraction
// [3] druggability - Druggability score
// [4] depth - Pocket depth
// [5] enclosure - Pocket enclosure score
// [6] conservation - Conservation score
// [7] topology - Topology score

// Ground truth (per ligand):
// [0-2] center_x, center_y, center_z - Ligand center coordinates
// [3] radius - Ligand bounding sphere radius

/// Compute batch rewards from predictions and ground truth
///
/// Each reward combines:
/// - DCC reward: 1.0 if distance < 4Å, linear decay otherwise
/// - Coverage reward: fraction of ligand atoms within threshold
/// - Druggability reward: predicted druggability score
///
/// Weights: 0.5 * dcc + 0.3 * coverage + 0.2 * druggability
extern "C" __global__ void compute_batch_rewards(
    const float* __restrict__ predictions,  // [batch, 8] prediction features
    const float* __restrict__ ground_truth, // [batch, 4] ligand centers + radius
    float* __restrict__ rewards,            // [batch] output rewards
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    // Extract prediction features for this sample
    const float center_dist = predictions[idx * 8 + 0];
    const float volume_overlap = predictions[idx * 8 + 1];
    const float coverage = predictions[idx * 8 + 2];
    const float druggability = predictions[idx * 8 + 3];
    const float depth = predictions[idx * 8 + 4];
    const float enclosure = predictions[idx * 8 + 5];
    const float conservation = predictions[idx * 8 + 6];
    const float topology = predictions[idx * 8 + 7];

    // DCC (Distance to Center of Cavity) reward
    // Success threshold: 4.0 Å (PDBBind standard)
    // Linear decay: 1.0 at 0Å, 0.0 at 4Å, negative beyond
    const float dcc_threshold = 4.0f;
    float dcc_reward = fmaxf(0.0f, 1.0f - center_dist / dcc_threshold);

    // Coverage reward: fraction of ligand atoms within pocket
    float coverage_reward = coverage;

    // Druggability reward: predicted druggability score (already normalized)
    float druggability_reward = druggability;

    // Multi-objective reward combination
    // Prioritize DCC success, then coverage, then druggability
    float total_reward = 0.5f * dcc_reward
                       + 0.3f * coverage_reward
                       + 0.2f * druggability_reward;

    // Add bonus for high-quality predictions (all metrics strong)
    if (dcc_reward > 0.8f && coverage > 0.7f && druggability > 0.6f) {
        total_reward += 0.1f;  // Quality bonus
    }

    // Clamp to [0, 1] range
    rewards[idx] = fminf(1.0f, fmaxf(0.0f, total_reward));
}

/// Compute gradients for weight updates
///
/// Uses finite difference approximation:
/// gradient[i] = feature[i] * (reward - baseline) * temperature
///
/// Temperature implements simulated annealing:
/// - High temperature (1.0): Exploratory, large updates
/// - Low temperature (0.1): Exploitative, small updates
extern "C" __global__ void compute_gradients(
    const float* __restrict__ features,     // [batch, 8] feature values
    const float* __restrict__ rewards,      // [batch] computed rewards
    float* __restrict__ gradients,          // [batch, 8] output gradients
    float baseline,                         // Baseline reward (moving average)
    float temperature,                      // Annealing temperature
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    float reward = rewards[idx];

    // Advantage = reward - baseline
    float advantage = reward - baseline;

    // Sign of improvement (positive = good, negative = bad)
    float sign = advantage * temperature;

    // Compute gradient for each feature
    // gradient = feature_value * sign
    // Intuition: if reward > baseline, increase weights on active features
    for (int f = 0; f < 8; f++) {
        float feature_val = features[idx * 8 + f];
        gradients[idx * 8 + f] = feature_val * sign;
    }
}

/// Aggregate gradients across batch (reduction)
///
/// Computes mean gradient for each feature across all batch samples
/// Uses shared memory for efficient reduction
extern "C" __global__ void aggregate_gradients(
    const float* __restrict__ batch_gradients,  // [batch, 8] per-sample gradients
    float* __restrict__ aggregated,             // [8] output mean gradients
    int batch_size
) {
    __shared__ float shared_grads[256 * 8];  // Shared memory for reduction

    int tid = threadIdx.x;
    int feature_idx = blockIdx.x;  // One block per feature

    if (feature_idx >= 8) return;

    // Load gradients into shared memory
    float sum = 0.0f;
    for (int i = tid; i < batch_size; i += blockDim.x) {
        sum += batch_gradients[i * 8 + feature_idx];
    }
    shared_grads[tid] = sum;
    __syncthreads();

    // Parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_grads[tid] += shared_grads[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 writes result
    if (tid == 0) {
        aggregated[feature_idx] = shared_grads[0] / (float)batch_size;
    }
}

/// Compute batch statistics for monitoring
///
/// Computes mean, std, min, max for rewards and distances
extern "C" __global__ void compute_batch_statistics(
    const float* __restrict__ rewards,      // [batch] rewards
    const float* __restrict__ distances,    // [batch] center distances
    float* __restrict__ stats,              // [8] output: [mean_r, std_r, min_r, max_r, mean_d, std_d, min_d, max_d]
    int batch_size
) {
    __shared__ float shared_rewards[256];
    __shared__ float shared_distances[256];

    int tid = threadIdx.x;

    // Load data into shared memory
    float r = 0.0f, d = 0.0f;
    if (tid < batch_size) {
        r = rewards[tid];
        d = distances[tid];
    }
    shared_rewards[tid] = r;
    shared_distances[tid] = d;
    __syncthreads();

    // Compute mean
    if (tid == 0) {
        float sum_r = 0.0f, sum_d = 0.0f;
        float min_r = 1e10f, max_r = -1e10f;
        float min_d = 1e10f, max_d = -1e10f;

        for (int i = 0; i < batch_size; i++) {
            sum_r += shared_rewards[i];
            sum_d += shared_distances[i];
            min_r = fminf(min_r, shared_rewards[i]);
            max_r = fmaxf(max_r, shared_rewards[i]);
            min_d = fminf(min_d, shared_distances[i]);
            max_d = fmaxf(max_d, shared_distances[i]);
        }

        float mean_r = sum_r / batch_size;
        float mean_d = sum_d / batch_size;

        // Compute std
        float var_r = 0.0f, var_d = 0.0f;
        for (int i = 0; i < batch_size; i++) {
            float diff_r = shared_rewards[i] - mean_r;
            float diff_d = shared_distances[i] - mean_d;
            var_r += diff_r * diff_r;
            var_d += diff_d * diff_d;
        }

        stats[0] = mean_r;
        stats[1] = sqrtf(var_r / batch_size);
        stats[2] = min_r;
        stats[3] = max_r;
        stats[4] = mean_d;
        stats[5] = sqrtf(var_d / batch_size);
        stats[6] = min_d;
        stats[7] = max_d;
    }
}

/// Update weights using momentum SGD
///
/// weights[i] = weights[i] + momentum * velocity[i] + lr * gradient[i]
/// velocity[i] = momentum_decay * velocity[i] + lr * gradient[i]
extern "C" __global__ void update_weights_momentum(
    float* __restrict__ weights,            // [8] current weights (in/out)
    float* __restrict__ velocity,           // [8] velocity for momentum (in/out)
    const float* __restrict__ gradients,    // [8] computed gradients
    float learning_rate,
    float momentum_decay,
    float weight_min,
    float weight_max
) {
    int idx = threadIdx.x;
    if (idx >= 8) return;

    // Momentum SGD update
    float grad = gradients[idx];
    float vel = momentum_decay * velocity[idx] + learning_rate * grad;
    velocity[idx] = vel;

    // Update weight
    float new_weight = weights[idx] + vel;

    // Clamp to valid range
    new_weight = fminf(weight_max, fmaxf(weight_min, new_weight));

    weights[idx] = new_weight;
}

/// Normalize weights to sum to 1.0
extern "C" __global__ void normalize_weights(
    float* __restrict__ weights,  // [8] weights to normalize
    int num_weights
) {
    __shared__ float sum;

    if (threadIdx.x == 0) {
        sum = 0.0f;
        for (int i = 0; i < num_weights; i++) {
            sum += weights[i];
        }
    }
    __syncthreads();

    int idx = threadIdx.x;
    if (idx < num_weights && sum > 0.0f) {
        weights[idx] /= sum;
    }
}
