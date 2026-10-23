/**
 * PRISM Cryptic Site Detection - GPU Hessian & Distance Matrix Kernel
 *
 * Builds the Anisotropic Network Model (ANM) Hessian matrix and computes
 * pairwise distances for contact order analysis in a single fused pass.
 *
 * Copyright (c) 2024 PRISM Research Team | Delfictus I/O Inc.
 */

#include <cuda_runtime.h>
#include <cmath>

// Constants
#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define ANM_CUTOFF 13.0f
#define ANM_CUTOFF_SQ (ANM_CUTOFF * ANM_CUTOFF)
#define CONTACT_CUTOFF 8.0f
#define CONTACT_CUTOFF_SQ (CONTACT_CUTOFF * CONTACT_CUTOFF)
#define MIN_SEQ_SEP 3

/**
 * Fused kernel: Build ANM Hessian off-diagonal blocks + Distance Matrix
 *
 * For N residues, computes:
 * - 3Nx3N Hessian matrix (sparse, stored as dense for simplicity)
 * - NxN distance matrix for contact order
 *
 * Each thread block handles a tile of residue pairs.
 */
extern "C" __global__ void build_hessian_and_distances(
    const float* __restrict__ coords,      // [N, 3] CA coordinates
    const int* __restrict__ residue_seq,   // [N] residue sequence numbers
    float* __restrict__ hessian,           // [3N, 3N] Hessian matrix
    float* __restrict__ distances,         // [N, N] distance matrix
    int* __restrict__ contact_counts,      // [N] contact count per residue
    float* __restrict__ contact_sep_sum,   // [N] sum of sequence separations
    const int n_residues,
    const float spring_constant
) {
    // Shared memory for coordinate tiles
    __shared__ float s_coords_i[BLOCK_SIZE][3];
    __shared__ float s_coords_j[BLOCK_SIZE][3];
    __shared__ int s_seq_i[BLOCK_SIZE];
    __shared__ int s_seq_j[BLOCK_SIZE];

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    // Calculate which residue pair tile this block handles
    // Using triangular indexing for upper triangle
    const int total_pairs = (n_residues * (n_residues - 1)) / 2;
    const int pairs_per_block = (total_pairs + gridDim.x - 1) / gridDim.x;
    const int start_pair = bid * pairs_per_block;
    const int end_pair = min(start_pair + pairs_per_block, total_pairs);

    // Process pairs assigned to this block
    for (int pair_idx = start_pair + tid; pair_idx < end_pair; pair_idx += BLOCK_SIZE) {
        // Convert linear pair index to (i, j) using triangular formula
        // pair_idx = i * (i - 1) / 2 + j for j < i
        // Solve for i: i = floor((1 + sqrt(1 + 8*pair_idx)) / 2)
        int i = (int)((1.0f + sqrtf(1.0f + 8.0f * (float)pair_idx)) * 0.5f);
        int j = pair_idx - (i * (i - 1)) / 2;

        // Ensure valid indices
        if (i >= n_residues) i = n_residues - 1;
        if (j >= i) j = i - 1;
        if (j < 0) continue;

        // Load coordinates
        float xi = coords[i * 3 + 0];
        float yi = coords[i * 3 + 1];
        float zi = coords[i * 3 + 2];
        float xj = coords[j * 3 + 0];
        float yj = coords[j * 3 + 1];
        float zj = coords[j * 3 + 2];

        // Compute displacement vector
        float dx = xj - xi;
        float dy = yj - yi;
        float dz = zj - zi;
        float dist_sq = dx * dx + dy * dy + dz * dz;
        float dist = sqrtf(dist_sq);

        // Store distance (symmetric)
        distances[i * n_residues + j] = dist;
        distances[j * n_residues + i] = dist;

        // Sequence separation for contact order
        int seq_i = residue_seq[i];
        int seq_j = residue_seq[j];
        int seq_sep = abs(seq_i - seq_j);

        // Contact order contribution (if within cutoff and sufficient separation)
        if (dist_sq < CONTACT_CUTOFF_SQ && seq_sep >= MIN_SEQ_SEP) {
            atomicAdd(&contact_counts[i], 1);
            atomicAdd(&contact_counts[j], 1);
            atomicAdd(&contact_sep_sum[i], (float)seq_sep);
            atomicAdd(&contact_sep_sum[j], (float)seq_sep);
        }

        // ANM Hessian contribution (if within ANM cutoff)
        if (dist_sq < ANM_CUTOFF_SQ && dist_sq > 0.01f) {
            float k = spring_constant / dist_sq;

            // Compute 3x3 off-diagonal block: -k * (r_ij ⊗ r_ij) / |r_ij|²
            float dxyz[3] = {dx, dy, dz};

            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++) {
                    float val = -k * dxyz[a] * dxyz[b] / dist_sq;

                    // Off-diagonal blocks (i,j) and (j,i)
                    int idx_ij = (3 * i + a) * (3 * n_residues) + (3 * j + b);
                    int idx_ji = (3 * j + b) * (3 * n_residues) + (3 * i + a);

                    atomicAdd(&hessian[idx_ij], val);
                    atomicAdd(&hessian[idx_ji], val);

                    // Diagonal blocks contribution: subtract from (i,i) and (j,j)
                    int idx_ii = (3 * i + a) * (3 * n_residues) + (3 * i + b);
                    int idx_jj = (3 * j + a) * (3 * n_residues) + (3 * j + b);

                    atomicAdd(&hessian[idx_ii], -val);
                    atomicAdd(&hessian[idx_jj], -val);
                }
            }
        }
    }
}

/**
 * Compute local contact order per residue using a sliding window
 */
extern "C" __global__ void compute_local_contact_order(
    const int* __restrict__ contact_counts,
    const float* __restrict__ contact_sep_sum,
    float* __restrict__ local_contact_order,
    float* __restrict__ flexibility_score,
    const int n_residues,
    const int window_size,
    const float global_rco
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_residues) return;

    const int half_window = window_size / 2;
    const int window_start = max(0, idx - half_window);
    const int window_end = min(n_residues, idx + half_window + 1);

    // Sum contact order in window
    int total_contacts = 0;
    float total_sep = 0.0f;

    for (int w = window_start; w < window_end; w++) {
        total_contacts += contact_counts[w];
        total_sep += contact_sep_sum[w];
    }

    // Compute local contact order
    float local_co = 0.0f;
    if (total_contacts > 0) {
        local_co = (total_sep / (float)total_contacts) / (float)n_residues;
    }

    local_contact_order[idx] = local_co;

    // Flexibility score: inverted contact order (low CO = high flexibility)
    if (local_co > 0.0f && global_rco > 0.0f) {
        flexibility_score[idx] = 1.0f - fminf(local_co / global_rco, 1.0f);
    } else {
        flexibility_score[idx] = 0.5f;  // Default
    }
}

/**
 * Compute conservation scores based on residue type propensities
 * (GPU-accelerated lookup and relative conservation calculation)
 */
extern "C" __global__ void compute_conservation_scores(
    const int* __restrict__ residue_types,  // 0-19 for 20 amino acids
    float* __restrict__ conservation_scores,
    float* __restrict__ relative_conservation,
    const int n_residues,
    const int window_size
) {
    // Conservation propensity lookup table (BLOSUM62-derived)
    __shared__ float propensity[20];
    if (threadIdx.x < 20) {
        const float props[20] = {
            0.45f, 0.70f, 0.55f, 0.65f, 0.80f,  // A, R, N, D, C
            0.50f, 0.65f, 0.75f, 0.70f, 0.40f,  // Q, E, G, H, I
            0.35f, 0.55f, 0.45f, 0.50f, 0.70f,  // L, K, M, F, P
            0.45f, 0.50f, 0.75f, 0.60f, 0.40f   // S, T, W, Y, V
        };
        propensity[threadIdx.x] = props[threadIdx.x];
    }
    __syncthreads();

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_residues) return;

    // Base conservation from residue type
    int res_type = residue_types[idx];
    float base_cons = (res_type >= 0 && res_type < 20) ? propensity[res_type] : 0.5f;

    // Relative conservation (vs local neighborhood)
    const int half_window = window_size / 2;
    const int window_start = max(0, idx - half_window);
    const int window_end = min(n_residues, idx + half_window + 1);

    float window_sum = 0.0f;
    int window_count = 0;

    for (int w = window_start; w < window_end; w++) {
        int wtype = residue_types[w];
        if (wtype >= 0 && wtype < 20) {
            window_sum += propensity[wtype];
            window_count++;
        }
    }

    float window_mean = (window_count > 0) ? window_sum / (float)window_count : 0.5f;
    float rel_cons = (window_mean > 0.0f) ? fminf(base_cons / window_mean, 2.0f) / 2.0f : 0.5f;

    conservation_scores[idx] = base_cons;
    relative_conservation[idx] = 0.6f * base_cons + 0.4f * rel_cons;
}
