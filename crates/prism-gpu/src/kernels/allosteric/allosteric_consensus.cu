/**
 * Allosteric Consensus and Gap Detection CUDA Kernels
 *
 * GPU-accelerated consensus scoring and backtrack gap detection:
 * - Conservation scoring (Shannon entropy)
 * - Coverage gap detection
 * - Multi-signal fusion
 * - Confidence calculation
 *
 * Copyright (c) 2024 PRISM Research Team | Delfictus I/O Inc.
 */

#include <cuda_runtime.h>
#include <cmath>

// ============================================================================
// Constants
// ============================================================================

constexpr int BLOCK_SIZE = 256;
constexpr int N_AMINO_ACIDS = 20;
constexpr float MAX_ENTROPY = 2.9957f;  // ln(20)

// ============================================================================
// Conservation Scoring (MSA Analysis)
// ============================================================================

/**
 * Count amino acid frequencies at each position
 * Input: sequences as integer indices (0-19 for amino acids, 20 for gap)
 */
extern "C" __global__ void count_aa_frequencies(
    const int* __restrict__ msa,           // [n_seqs, alignment_length]
    float* __restrict__ frequencies,        // [alignment_length, 21] (20 AA + gap)
    int n_seqs,
    int alignment_length
) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= alignment_length) return;

    // Initialize counts
    float counts[21];
    for (int i = 0; i < 21; i++) {
        counts[i] = 1.0f;  // Pseudocount
    }

    // Count amino acids at this position
    for (int seq = 0; seq < n_seqs; seq++) {
        int aa = msa[seq * alignment_length + pos];
        if (aa >= 0 && aa <= 20) {
            counts[aa] += 1.0f;
        }
    }

    // Store frequencies
    float total = 0.0f;
    for (int i = 0; i < 21; i++) {
        total += counts[i];
    }

    for (int i = 0; i < 21; i++) {
        frequencies[pos * 21 + i] = counts[i] / total;
    }
}

/**
 * Calculate Shannon entropy and conservation score
 */
extern "C" __global__ void calculate_conservation(
    const float* __restrict__ frequencies,   // [alignment_length, 21]
    float* __restrict__ conservation,        // [alignment_length]
    float* __restrict__ entropy,             // [alignment_length]
    float* __restrict__ gap_fraction,        // [alignment_length]
    int alignment_length
) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= alignment_length) return;

    // Calculate entropy from first 20 amino acids (excluding gap)
    float ent = 0.0f;
    float aa_total = 0.0f;

    for (int aa = 0; aa < N_AMINO_ACIDS; aa++) {
        float p = frequencies[pos * 21 + aa];
        if (p > 1e-10f) {
            ent -= p * logf(p);
            aa_total += p;
        }
    }

    // Gap fraction
    float gap = frequencies[pos * 21 + 20];
    gap_fraction[pos] = gap;

    // Normalize entropy to [0,1]
    float norm_entropy = ent / MAX_ENTROPY;

    // Conservation = 1 - normalized entropy, with gap penalty
    float cons = (1.0f - norm_entropy) * (1.0f - 0.5f * gap);

    entropy[pos] = ent;
    conservation[pos] = fmaxf(0.0f, fminf(1.0f, cons));
}

/**
 * Map conservation scores to structure residues
 */
extern "C" __global__ void map_conservation_to_structure(
    const float* __restrict__ conservation,
    const int* __restrict__ alignment_to_structure,  // Maps alignment pos -> residue seq
    float* __restrict__ residue_conservation,        // Output: conservation per residue
    int alignment_length,
    int n_residues
) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= alignment_length) return;

    int residue = alignment_to_structure[pos];
    if (residue >= 0 && residue < n_residues) {
        residue_conservation[residue] = conservation[pos];
    }
}

// ============================================================================
// Coverage Gap Detection
// ============================================================================

/**
 * Detect conserved residues not covered by pockets
 */
extern "C" __global__ void detect_conserved_gaps(
    const float* __restrict__ conservation,
    const int* __restrict__ pocket_mask,    // 1 if residue is in a pocket, 0 otherwise
    int* __restrict__ gap_residues,         // Output: residues with gaps
    int* __restrict__ n_gaps,
    float threshold,
    int n_residues
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_residues) return;

    float cons = conservation[idx];
    int in_pocket = pocket_mask[idx];

    // Conserved but not in pocket
    if (cons >= threshold && in_pocket == 0) {
        int pos = atomicAdd(n_gaps, 1);
        gap_residues[pos] = idx;
    }
}

/**
 * Detect high-centrality residues not covered
 */
extern "C" __global__ void detect_centrality_gaps(
    const float* __restrict__ centrality,
    const int* __restrict__ pocket_mask,
    int* __restrict__ gap_residues,
    int* __restrict__ n_gaps,
    float threshold,
    int n_residues
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_residues) return;

    float cent = centrality[idx];
    int in_pocket = pocket_mask[idx];

    // High centrality but not in pocket
    if (cent >= threshold && in_pocket == 0) {
        int pos = atomicAdd(n_gaps, 1);
        gap_residues[pos] = idx;
    }
}

/**
 * Calculate gap priority scores
 */
extern "C" __global__ void calculate_gap_priority(
    const int* __restrict__ gap_residues,
    const float* __restrict__ conservation,
    const float* __restrict__ centrality,
    const float* __restrict__ bfactors,
    float* __restrict__ priorities,
    int n_gaps,
    int n_residues
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_gaps) return;

    int residue = gap_residues[idx];
    if (residue < 0 || residue >= n_residues) {
        priorities[idx] = 0.0f;
        return;
    }

    // Weighted combination of signals
    float cons = (residue < n_residues) ? conservation[residue] : 0.0f;
    float cent = (residue < n_residues) ? centrality[residue] : 0.0f;
    float bf = (residue < n_residues && bfactors != nullptr) ? bfactors[residue] : 0.5f;

    // Priority = weighted combination
    float priority = 0.4f * cons + 0.4f * cent + 0.2f * (1.0f - bf);
    priorities[idx] = fmaxf(0.0f, fminf(1.0f, priority));
}

// ============================================================================
// Multi-Signal Fusion
// ============================================================================

/**
 * Fuse multiple detection signals for each residue
 */
extern "C" __global__ void fuse_detection_signals(
    const float* __restrict__ geometric_scores,
    const float* __restrict__ flexibility_scores,
    const float* __restrict__ conservation_scores,
    const float* __restrict__ centrality_scores,
    float* __restrict__ fused_scores,
    float* __restrict__ n_signals,          // Count of non-zero signals per residue
    float w_geo,
    float w_flex,
    float w_cons,
    float w_cent,
    int n_residues
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_residues) return;

    float score = 0.0f;
    float signals = 0.0f;
    float total_weight = 0.0f;

    // Geometric
    float geo = (geometric_scores != nullptr) ? geometric_scores[idx] : 0.0f;
    if (geo > 0.01f) {
        score += w_geo * geo;
        total_weight += w_geo;
        signals += 1.0f;
    }

    // Flexibility
    float flex = (flexibility_scores != nullptr) ? flexibility_scores[idx] : 0.0f;
    if (flex > 0.01f) {
        score += w_flex * flex;
        total_weight += w_flex;
        signals += 1.0f;
    }

    // Conservation
    float cons = (conservation_scores != nullptr) ? conservation_scores[idx] : 0.0f;
    if (cons > 0.01f) {
        score += w_cons * cons;
        total_weight += w_cons;
        signals += 1.0f;
    }

    // Centrality
    float cent = (centrality_scores != nullptr) ? centrality_scores[idx] : 0.0f;
    if (cent > 0.01f) {
        score += w_cent * cent;
        total_weight += w_cent;
        signals += 1.0f;
    }

    // Normalize by total weight
    if (total_weight > 0.0f) {
        score /= total_weight;
    }

    // Consensus bonus
    if (signals >= 3.0f) {
        score += 0.1f;
    } else if (signals >= 2.0f) {
        score += 0.05f;
    }

    fused_scores[idx] = fmaxf(0.0f, fminf(1.0f, score));
    n_signals[idx] = signals;
}

// ============================================================================
// Confidence Calculation
// ============================================================================

/**
 * Calculate confidence score for a pocket
 * Pocket is defined by a mask indicating which residues belong to it
 */
extern "C" __global__ void calculate_pocket_confidence(
    const int* __restrict__ pocket_residues,
    int n_pocket_residues,
    const float* __restrict__ geometric_scores,
    const float* __restrict__ flexibility_scores,
    const float* __restrict__ conservation_scores,
    const float* __restrict__ centrality_scores,
    float* __restrict__ confidence_out,      // Single output value
    float* __restrict__ evidence_counts,     // [4] counts of positive evidence
    int n_residues
) {
    __shared__ float shared_geo[BLOCK_SIZE];
    __shared__ float shared_flex[BLOCK_SIZE];
    __shared__ float shared_cons[BLOCK_SIZE];
    __shared__ float shared_cent[BLOCK_SIZE];
    __shared__ int shared_counts[4];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize
    if (tid < 4) {
        shared_counts[tid] = 0;
    }
    __syncthreads();

    float local_geo = 0.0f;
    float local_flex = 0.0f;
    float local_cons = 0.0f;
    float local_cent = 0.0f;

    if (idx < n_pocket_residues) {
        int residue = pocket_residues[idx];
        if (residue >= 0 && residue < n_residues) {
            local_geo = (geometric_scores != nullptr) ? geometric_scores[residue] : 0.0f;
            local_flex = (flexibility_scores != nullptr) ? flexibility_scores[residue] : 0.0f;
            local_cons = (conservation_scores != nullptr) ? conservation_scores[residue] : 0.0f;
            local_cent = (centrality_scores != nullptr) ? centrality_scores[residue] : 0.0f;

            // Count positive evidence
            if (local_geo > 0.3f) atomicAdd(&shared_counts[0], 1);
            if (local_flex > 0.3f) atomicAdd(&shared_counts[1], 1);
            if (local_cons > 0.5f) atomicAdd(&shared_counts[2], 1);
            if (local_cent > 0.3f) atomicAdd(&shared_counts[3], 1);
        }
    }

    shared_geo[tid] = local_geo;
    shared_flex[tid] = local_flex;
    shared_cons[tid] = local_cons;
    shared_cent[tid] = local_cent;
    __syncthreads();

    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_geo[tid] += shared_geo[tid + stride];
            shared_flex[tid] += shared_flex[tid + stride];
            shared_cons[tid] += shared_cons[tid + stride];
            shared_cent[tid] += shared_cent[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        // Average scores
        float avg_geo = shared_geo[0] / fmaxf(1.0f, (float)n_pocket_residues);
        float avg_flex = shared_flex[0] / fmaxf(1.0f, (float)n_pocket_residues);
        float avg_cons = shared_cons[0] / fmaxf(1.0f, (float)n_pocket_residues);
        float avg_cent = shared_cent[0] / fmaxf(1.0f, (float)n_pocket_residues);

        // Count types of positive evidence
        int n_positive = 0;
        if (avg_geo > 0.3f) n_positive++;
        if (avg_flex > 0.3f) n_positive++;
        if (avg_cons > 0.5f) n_positive++;
        if (avg_cent > 0.3f) n_positive++;

        // Calculate confidence
        float confidence = 0.0f;

        // Geometric contribution (0.25 max)
        if (avg_geo > 0.01f) confidence += 0.25f * avg_geo;

        // Flexibility contribution (0.15 max)
        if (avg_flex > 0.01f) confidence += 0.15f * avg_flex;

        // Conservation contribution (0.25 max)
        if (avg_cons > 0.01f) confidence += 0.25f * avg_cons;

        // Centrality contribution (0.15 max)
        if (avg_cent > 0.01f) confidence += 0.15f * avg_cent;

        // Consensus bonus (0.2 max)
        if (n_positive >= 3) confidence += 0.2f;
        else if (n_positive >= 2) confidence += 0.1f;

        confidence_out[0] = fmaxf(0.0f, fminf(1.0f, confidence));

        evidence_counts[0] = (float)shared_counts[0];
        evidence_counts[1] = (float)shared_counts[1];
        evidence_counts[2] = (float)shared_counts[2];
        evidence_counts[3] = (float)shared_counts[3];
    }
}

// ============================================================================
// Spatial Clustering (DBSCAN-like)
// ============================================================================

/**
 * Build distance matrix for gap residues (for clustering)
 */
extern "C" __global__ void build_gap_distance_matrix(
    const float* __restrict__ coords,       // [n_residues, 3]
    const int* __restrict__ gap_residues,
    float* __restrict__ dist_matrix,        // [n_gaps, n_gaps]
    int n_gaps,
    int n_residues
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_gaps || j >= n_gaps) return;

    int res_i = gap_residues[i];
    int res_j = gap_residues[j];

    float dist;
    if (i == j) {
        dist = 0.0f;
    } else if (res_i >= 0 && res_i < n_residues && res_j >= 0 && res_j < n_residues) {
        float dx = coords[res_i * 3 + 0] - coords[res_j * 3 + 0];
        float dy = coords[res_i * 3 + 1] - coords[res_j * 3 + 1];
        float dz = coords[res_i * 3 + 2] - coords[res_j * 3 + 2];
        dist = sqrtf(dx * dx + dy * dy + dz * dz);
    } else {
        dist = 1e30f;
    }

    dist_matrix[i * n_gaps + j] = dist;
}

/**
 * DBSCAN-like clustering of gap residues
 */
extern "C" __global__ void cluster_gap_residues(
    const float* __restrict__ dist_matrix,
    int* __restrict__ cluster_assignments,
    float eps,
    int min_pts,
    int n_gaps
) {
    // Simple single-linkage approach for GPU
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_gaps) return;

    // Initialize: each point in its own cluster (cluster ID = point ID)
    cluster_assignments[idx] = idx;
    __syncthreads();

    // Find minimum cluster ID among neighbors
    int min_cluster = cluster_assignments[idx];

    for (int j = 0; j < n_gaps; j++) {
        if (dist_matrix[idx * n_gaps + j] < eps) {
            int neighbor_cluster = cluster_assignments[j];
            if (neighbor_cluster < min_cluster) {
                min_cluster = neighbor_cluster;
            }
        }
    }

    // Update cluster assignment
    cluster_assignments[idx] = min_cluster;
}
