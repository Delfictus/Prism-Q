/**
 * Bitmask Classification Kernel
 *
 * Final GPU-side processing that converts DQN Q-values into compact bitmasks.
 * This eliminates the need to transfer large feature vectors to the CPU,
 * reducing PCIe traffic by >1000x.
 *
 * Only high-confidence cryptic sites and epitopes are flagged in the bitmask.
 *
 * Copyright (c) 2024 PRISM Research Team | Delfictus I/O Inc.
 */

#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define WARP_SIZE 32
#define BLOCK_SIZE 256

// FluxNet-DQN action indices
#define ACTION_PREDICT_CRYPTIC 0
#define ACTION_PREDICT_EXPOSED 1
#define ACTION_PREDICT_EPITOPE 2
#define ACTION_SKIP 3

/**
 * Main bitmask classification kernel
 *
 * Converts DQN Q-values to compact bitmasks for cryptic sites and epitopes.
 * Each bit represents one residue: 1=predicted positive, 0=negative.
 *
 * @param dqn_output        [total_residues, 4] Q-values for 4 actions
 * @param cryptic_bitmask   [total_residues/32] Output cryptic site bitmask
 * @param epitope_bitmask   [total_residues/32] Output epitope bitmask
 * @param confidence_scores [total_residues] Optional confidence output
 * @param total_residues    Total number of residues to process
 * @param cryptic_threshold Confidence threshold for cryptic sites (0.7)
 * @param epitope_threshold Confidence threshold for epitopes (0.6)
 * @param output_confidence Whether to write confidence scores
 */
extern "C" __global__ void __launch_bounds__(256, 4)
classify_to_bitmask(
    const float* __restrict__ dqn_output,        // [total_residues, 4] Q-values
    unsigned int* __restrict__ cryptic_bitmask,  // [total_residues/32] cryptic bits
    unsigned int* __restrict__ epitope_bitmask,  // [total_residues/32] epitope bits
    float* __restrict__ confidence_scores,       // [total_residues] optional confidence
    const int total_residues,
    const float cryptic_threshold,
    const float epitope_threshold,
    const bool output_confidence
) {
    const int res_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (res_idx >= total_residues) return;

    // Load Q-values for this residue
    const float* q_values = dqn_output + res_idx * 4;
    float q_cryptic = q_values[ACTION_PREDICT_CRYPTIC];
    float q_exposed = q_values[ACTION_PREDICT_EXPOSED];
    float q_epitope = q_values[ACTION_PREDICT_EPITOPE];
    float q_skip = q_values[ACTION_SKIP];

    // Find maximum Q-value and apply softmax for confidence
    float q_max = fmaxf(fmaxf(q_cryptic, q_exposed), fmaxf(q_epitope, q_skip));

    // Softmax probabilities
    float exp_cryptic = expf(q_cryptic - q_max);
    float exp_exposed = expf(q_exposed - q_max);
    float exp_epitope = expf(q_epitope - q_max);
    float exp_skip = expf(q_skip - q_max);

    float sum_exp = exp_cryptic + exp_exposed + exp_epitope + exp_skip;

    float p_cryptic = exp_cryptic / sum_exp;
    float p_epitope = exp_epitope / sum_exp;

    // Determine bitmask positions
    int word_idx = res_idx / 32;
    int bit_idx = res_idx % 32;
    unsigned int bit_mask = 1U << bit_idx;

    // Set cryptic bit if confidence exceeds threshold
    if (p_cryptic > cryptic_threshold) {
        atomicOr(&cryptic_bitmask[word_idx], bit_mask);
    }

    // Set epitope bit if confidence exceeds threshold
    if (p_epitope > epitope_threshold) {
        atomicOr(&epitope_bitmask[word_idx], bit_mask);
    }

    // Optional: output confidence scores for debugging
    if (output_confidence) {
        confidence_scores[res_idx] = fmaxf(p_cryptic, p_epitope);
    }
}

/**
 * Enhanced classification with glycan masking integration
 *
 * Excludes glycan-shielded residues from positive predictions.
 *
 * @param dqn_output        [total_residues, 4] Q-values for 4 actions
 * @param glycan_bitmask    [total_residues/32] Glycan shielding mask
 * @param cryptic_bitmask   [total_residues/32] Output cryptic site bitmask
 * @param epitope_bitmask   [total_residues/32] Output epitope bitmask
 * @param total_residues    Total number of residues to process
 * @param cryptic_threshold Confidence threshold for cryptic sites
 * @param epitope_threshold Confidence threshold for epitopes
 */
extern "C" __global__ void __launch_bounds__(256, 4)
classify_with_glycan_masking(
    const float* __restrict__ dqn_output,
    const unsigned int* __restrict__ glycan_bitmask,
    unsigned int* __restrict__ cryptic_bitmask,
    unsigned int* __restrict__ epitope_bitmask,
    const int total_residues,
    const float cryptic_threshold,
    const float epitope_threshold
) {
    const int res_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (res_idx >= total_residues) return;

    // Check if this residue is glycan-shielded
    int word_idx = res_idx / 32;
    int bit_idx = res_idx % 32;
    bool is_glycan_shielded = (glycan_bitmask[word_idx] >> bit_idx) & 1;

    // Skip glycan-shielded residues for cryptic site prediction
    if (is_glycan_shielded) {
        return;
    }

    // Load Q-values and compute probabilities
    const float* q_values = dqn_output + res_idx * 4;
    float q_cryptic = q_values[ACTION_PREDICT_CRYPTIC];
    float q_exposed = q_values[ACTION_PREDICT_EXPOSED];
    float q_epitope = q_values[ACTION_PREDICT_EPITOPE];
    float q_skip = q_values[ACTION_SKIP];

    // Softmax normalization
    float q_max = fmaxf(fmaxf(q_cryptic, q_exposed), fmaxf(q_epitope, q_skip));
    float exp_cryptic = expf(q_cryptic - q_max);
    float exp_exposed = expf(q_exposed - q_max);
    float exp_epitope = expf(q_epitope - q_max);
    float exp_skip = expf(q_skip - q_max);

    float sum_exp = exp_cryptic + exp_exposed + exp_epitope + exp_skip;
    float p_cryptic = exp_cryptic / sum_exp;
    float p_epitope = exp_epitope / sum_exp;

    unsigned int bit_mask = 1U << bit_idx;

    // Set bits based on confidence thresholds
    if (p_cryptic > cryptic_threshold) {
        atomicOr(&cryptic_bitmask[word_idx], bit_mask);
    }

    if (p_epitope > epitope_threshold) {
        atomicOr(&epitope_bitmask[word_idx], bit_mask);
    }
}

/**
 * Batch processing version for multiple structures
 *
 * @param dqn_output            [total_residues, 4] Q-values
 * @param structure_offsets     [batch_size + 1] Cumulative residue offsets
 * @param residue_counts        [batch_size] Residues per structure
 * @param cryptic_bitmasks      [batch_size, max_residues/32] Per-structure cryptic masks
 * @param epitope_bitmasks      [batch_size, max_residues/32] Per-structure epitope masks
 * @param batch_size            Number of structures
 * @param max_residues          Maximum residues per structure
 * @param cryptic_threshold     Confidence threshold for cryptic sites
 * @param epitope_threshold     Confidence threshold for epitopes
 */
extern "C" __global__ void __launch_bounds__(256, 4)
classify_batch_to_bitmasks(
    const float* __restrict__ dqn_output,
    const int* __restrict__ structure_offsets,
    const int* __restrict__ residue_counts,
    unsigned int* __restrict__ cryptic_bitmasks,
    unsigned int* __restrict__ epitope_bitmasks,
    const int batch_size,
    const int max_residues,
    const float cryptic_threshold,
    const float epitope_threshold
) {
    const int structure_idx = blockIdx.x;
    if (structure_idx >= batch_size) return;

    const int tid = threadIdx.x;
    const int start_residue = structure_offsets[structure_idx];
    const int n_residues = residue_counts[structure_idx];

    // Calculate bitmask dimensions
    const int words_per_structure = (max_residues + 31) / 32;

    unsigned int* cryptic_mask = cryptic_bitmasks + structure_idx * words_per_structure;
    unsigned int* epitope_mask = epitope_bitmasks + structure_idx * words_per_structure;

    // Process residues for this structure
    for (int local_res = tid; local_res < n_residues; local_res += blockDim.x) {
        int global_res = start_residue + local_res;

        // Load Q-values
        const float* q_values = dqn_output + global_res * 4;
        float q_cryptic = q_values[ACTION_PREDICT_CRYPTIC];
        float q_exposed = q_values[ACTION_PREDICT_EXPOSED];
        float q_epitope = q_values[ACTION_PREDICT_EPITOPE];
        float q_skip = q_values[ACTION_SKIP];

        // Softmax probabilities
        float q_max = fmaxf(fmaxf(q_cryptic, q_exposed), fmaxf(q_epitope, q_skip));
        float exp_cryptic = expf(q_cryptic - q_max);
        float exp_exposed = expf(q_exposed - q_max);
        float exp_epitope = expf(q_epitope - q_max);
        float exp_skip = expf(q_skip - q_max);

        float sum_exp = exp_cryptic + exp_exposed + exp_epitope + exp_skip;
        float p_cryptic = exp_cryptic / sum_exp;
        float p_epitope = exp_epitope / sum_exp;

        // Set bits in structure-local bitmask
        int word_idx = local_res / 32;
        int bit_idx = local_res % 32;
        unsigned int bit_mask = 1U << bit_idx;

        if (p_cryptic > cryptic_threshold) {
            atomicOr(&cryptic_mask[word_idx], bit_mask);
        }

        if (p_epitope > epitope_threshold) {
            atomicOr(&epitope_mask[word_idx], bit_mask);
        }
    }
}

/**
 * Count positive predictions in bitmasks
 *
 * @param bitmasks      [batch_size, max_residues/32] Input bitmasks
 * @param counts        [batch_size] Output counts
 * @param residue_counts [batch_size] Actual residues per structure
 * @param batch_size    Number of structures
 * @param max_residues  Maximum residues per structure
 */
extern "C" __global__ void
count_bitmask_positives(
    const unsigned int* __restrict__ bitmasks,
    int* __restrict__ counts,
    const int* __restrict__ residue_counts,
    const int batch_size,
    const int max_residues
) {
    const int structure_idx = blockIdx.x;
    if (structure_idx >= batch_size) return;

    const int tid = threadIdx.x;
    const int n_residues = residue_counts[structure_idx];
    const int words_per_structure = (max_residues + 31) / 32;

    __shared__ int local_count;
    if (tid == 0) local_count = 0;
    __syncthreads();

    const unsigned int* mask = bitmasks + structure_idx * words_per_structure;

    // Count set bits in parallel
    for (int res_idx = tid; res_idx < n_residues; res_idx += blockDim.x) {
        int word_idx = res_idx / 32;
        int bit_idx = res_idx % 32;
        if ((mask[word_idx] >> bit_idx) & 1) {
            atomicAdd(&local_count, 1);
        }
    }
    __syncthreads();

    if (tid == 0) {
        counts[structure_idx] = local_count;
    }
}

/**
 * Extract positive residue indices from bitmask
 *
 * @param bitmask           [max_residues/32] Input bitmask
 * @param positive_indices  [max_positives] Output residue indices
 * @param n_residues        Actual number of residues
 * @param max_positives     Maximum positive predictions to store
 * @return                  Number of positives found
 */
extern "C" __global__ void
extract_positive_indices(
    const unsigned int* __restrict__ bitmask,
    int* __restrict__ positive_indices,
    int* __restrict__ num_positives,
    const int n_residues,
    const int max_positives
) {
    const int tid = threadIdx.x;
    __shared__ int local_count;

    if (tid == 0) local_count = 0;
    __syncthreads();

    // Scan residues in parallel
    for (int res_idx = tid; res_idx < n_residues; res_idx += blockDim.x) {
        int word_idx = res_idx / 32;
        int bit_idx = res_idx % 32;

        if ((bitmask[word_idx] >> bit_idx) & 1) {
            int idx = atomicAdd(&local_count, 1);
            if (idx < max_positives) {
                positive_indices[idx] = res_idx;
            }
        }
    }
    __syncthreads();

    if (tid == 0) {
        *num_positives = min(local_count, max_positives);
    }
}