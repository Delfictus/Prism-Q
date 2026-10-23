/**
 * PRISM Cryptic Site Detection - GPU Signal Fusion Kernel
 *
 * Combines all cryptic site detection signals into per-residue scores
 * and performs spatial clustering to identify candidate sites.
 *
 * Signals fused:
 * - B-factor flexibility (from structure)
 * - NMA mobility (from eigenmode analysis)
 * - Contact order flexibility (from distance analysis)
 * - Conservation score (from sequence analysis)
 * - Probe binding score (from FTMap-style analysis)
 *
 * Copyright (c) 2024 PRISM Research Team | Delfictus I/O Inc.
 */

#include <cuda_runtime.h>
#include <cmath>

#define WARP_SIZE 32
#define BLOCK_SIZE 256

// Default signal weights (sum to ~1.0)
#define WEIGHT_BFACTOR 0.15f
#define WEIGHT_PACKING 0.15f
#define WEIGHT_HYDROPHOBICITY 0.10f
#define WEIGHT_NMA 0.20f
#define WEIGHT_CONTACT_ORDER 0.12f
#define WEIGHT_CONSERVATION 0.13f
#define WEIGHT_PROBE 0.15f

// Thresholds
#define BFACTOR_MIN_THRESHOLD -0.5f
#define BFACTOR_THRESHOLD 0.8f
#define PACKING_DEFICIT_THRESHOLD 0.85f
#define HYDROPHOBICITY_THRESHOLD 0.6f
#define NMA_MIN_THRESHOLD 0.3f
#define CONSERVATION_THRESHOLD 0.6f
#define PROBE_MIN_THRESHOLD 0.3f

/**
 * Compute B-factor z-scores for all residues
 */
extern "C" __global__ void compute_bfactor_zscores(
    const float* __restrict__ residue_bfactors,  // [n_residues] mean B-factor per residue
    float* __restrict__ zscores,                  // [n_residues] output z-scores
    const int n_residues,
    const float global_mean,
    const float global_std
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_residues) return;

    float bf = residue_bfactors[idx];
    float std_safe = fmaxf(global_std, 0.1f);
    zscores[idx] = (bf - global_mean) / std_safe;
}

/**
 * Compute packing density for all residues
 * Uses pre-computed neighbor counts from spatial hashing
 */
extern "C" __global__ void compute_packing_density(
    const int* __restrict__ neighbor_counts,      // [n_residues] atoms within radius
    float* __restrict__ packing_density,          // [n_residues] output density ratio
    const int n_residues,
    const float sphere_volume,
    const float expected_density
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_residues) return;

    float actual_density = (float)neighbor_counts[idx] / sphere_volume;
    packing_density[idx] = actual_density / expected_density;
}

/**
 * Main signal fusion kernel - combines all signals per residue
 *
 * Outputs:
 * - combined_score: weighted sum of all signals
 * - qualification_flags: bitmask indicating which signals qualified
 */
extern "C" __global__ void fuse_cryptic_signals(
    // Input signals
    const float* __restrict__ bfactor_zscores,
    const float* __restrict__ packing_density,
    const float* __restrict__ hydrophobicity,
    const float* __restrict__ nma_mobility,
    const float* __restrict__ contact_order_flex,
    const float* __restrict__ conservation,
    const float* __restrict__ probe_scores,

    // Signal weights (can be dynamically configured)
    const float weight_bfactor,
    const float weight_packing,
    const float weight_hydro,
    const float weight_nma,
    const float weight_co,
    const float weight_cons,
    const float weight_probe,

    // Outputs
    float* __restrict__ combined_scores,
    int* __restrict__ qualification_flags,
    int* __restrict__ qualified_count,

    const int n_residues
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_residues) return;

    // Load all signals
    float bf = bfactor_zscores[idx];
    float pack = packing_density[idx];
    float hydro = hydrophobicity[idx];
    float nma = nma_mobility[idx];
    float co = contact_order_flex[idx];
    float cons = conservation[idx];
    float probe = probe_scores[idx];

    // Normalize B-factor to [0,1]
    float bf_norm = fminf(fmaxf(bf / 3.0f, 0.0f), 1.0f);

    // Packing deficit (1 - density, clamped)
    float pack_deficit = fminf(fmaxf(1.0f - pack, 0.0f), 1.0f);

    // Compute weighted combined score
    float score = weight_bfactor * bf_norm
                + weight_packing * pack_deficit
                + weight_hydro * hydro
                + weight_nma * nma
                + weight_co * co
                + weight_cons * cons
                + weight_probe * probe;

    combined_scores[idx] = score;

    // Determine qualification (which signals are strong enough)
    int flags = 0;

    // Classic qualification: B-factor + (packing OR hydrophobicity)
    int classic_qualifies = (bf > BFACTOR_THRESHOLD) &&
                           (pack < PACKING_DEFICIT_THRESHOLD || hydro > HYDROPHOBICITY_THRESHOLD);
    if (classic_qualifies) flags |= 0x01;

    // NMA qualification
    if (nma > NMA_MIN_THRESHOLD) flags |= 0x02;

    // Contact order qualification
    if (co > 0.5f) flags |= 0x04;

    // Conservation + probe qualification
    if (cons > CONSERVATION_THRESHOLD && probe > PROBE_MIN_THRESHOLD) flags |= 0x08;

    // Must have some flexibility signal
    int has_flex = (bf > BFACTOR_MIN_THRESHOLD) || (nma > 0.3f) || (co > 0.4f);

    // Final qualification: any strong signal + flexibility
    int qualifies = (flags != 0) && has_flex;

    qualification_flags[idx] = qualifies ? flags : 0;

    if (qualifies) {
        atomicAdd(qualified_count, 1);
    }
}

/**
 * Spatial clustering of qualified residues using DBSCAN-style approach
 */
extern "C" __global__ void cluster_qualified_residues(
    const float* __restrict__ centroids,          // [n_residues, 3]
    const int* __restrict__ qualification_flags,
    int* __restrict__ cluster_labels,             // [n_residues] output cluster ID
    const int n_residues,
    const float cluster_distance
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_residues) return;

    // Only process qualified residues
    if (qualification_flags[idx] == 0) {
        cluster_labels[idx] = -1;
        return;
    }

    float cx = centroids[idx * 3 + 0];
    float cy = centroids[idx * 3 + 1];
    float cz = centroids[idx * 3 + 2];
    float dist_sq = cluster_distance * cluster_distance;

    // Find minimum labeled neighbor
    int min_label = idx;  // Self-label initially

    for (int j = 0; j < idx; j++) {
        if (qualification_flags[j] == 0) continue;

        float dx = cx - centroids[j * 3 + 0];
        float dy = cy - centroids[j * 3 + 1];
        float dz = cz - centroids[j * 3 + 2];

        if (dx * dx + dy * dy + dz * dz < dist_sq) {
            int j_label = cluster_labels[j];
            min_label = min(min_label, j_label);
        }
    }

    cluster_labels[idx] = min_label;
}

/**
 * Label propagation for cluster merging (iterative)
 */
extern "C" __global__ void propagate_cluster_labels_residues(
    const float* __restrict__ centroids,
    const int* __restrict__ qualification_flags,
    int* __restrict__ cluster_labels,
    int* __restrict__ changed,
    const int n_residues,
    const float cluster_distance
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_residues) return;
    if (qualification_flags[idx] == 0) return;

    float cx = centroids[idx * 3 + 0];
    float cy = centroids[idx * 3 + 1];
    float cz = centroids[idx * 3 + 2];
    float dist_sq = cluster_distance * cluster_distance;

    int my_label = cluster_labels[idx];

    for (int j = 0; j < n_residues; j++) {
        if (j == idx || qualification_flags[j] == 0) continue;

        float dx = cx - centroids[j * 3 + 0];
        float dy = cy - centroids[j * 3 + 1];
        float dz = cz - centroids[j * 3 + 2];

        if (dx * dx + dy * dy + dz * dz < dist_sq) {
            int j_label = cluster_labels[j];
            if (j_label < my_label) {
                atomicMin(&cluster_labels[idx], j_label);
                *changed = 1;
            }
        }
    }
}

/**
 * Score candidate clusters - compute aggregate metrics
 */
extern "C" __global__ void score_candidate_clusters(
    const int* __restrict__ cluster_labels,
    const float* __restrict__ combined_scores,
    const int* __restrict__ qualification_flags,
    const float* __restrict__ centroids,

    // Per-cluster outputs (indexed by cluster ID)
    float* __restrict__ cluster_scores,
    float* __restrict__ cluster_centroids,        // [max_clusters, 3]
    int* __restrict__ cluster_sizes,
    int* __restrict__ cluster_valid,

    const int n_residues,
    const int max_clusters,
    const int min_cluster_size,
    const int max_cluster_size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_residues) return;

    int label = cluster_labels[idx];
    if (label < 0 || label >= max_clusters) return;
    if (qualification_flags[idx] == 0) return;

    // Accumulate to cluster
    atomicAdd(&cluster_sizes[label], 1);
    atomicAdd(&cluster_scores[label], combined_scores[idx]);
    atomicAdd(&cluster_centroids[label * 3 + 0], centroids[idx * 3 + 0]);
    atomicAdd(&cluster_centroids[label * 3 + 1], centroids[idx * 3 + 1]);
    atomicAdd(&cluster_centroids[label * 3 + 2], centroids[idx * 3 + 2]);
}

/**
 * Finalize cluster scores and filter by size
 */
extern "C" __global__ void finalize_clusters(
    float* __restrict__ cluster_scores,
    float* __restrict__ cluster_centroids,
    int* __restrict__ cluster_sizes,
    int* __restrict__ cluster_valid,
    int* __restrict__ valid_cluster_count,
    const int max_clusters,
    const int min_cluster_size,
    const int max_cluster_size,
    const float min_score
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= max_clusters) return;

    int size = cluster_sizes[idx];

    if (size >= min_cluster_size && size <= max_cluster_size) {
        // Normalize score and centroid
        float avg_score = cluster_scores[idx] / (float)size;
        cluster_scores[idx] = avg_score;

        cluster_centroids[idx * 3 + 0] /= (float)size;
        cluster_centroids[idx * 3 + 1] /= (float)size;
        cluster_centroids[idx * 3 + 2] /= (float)size;

        if (avg_score >= min_score) {
            cluster_valid[idx] = 1;
            atomicAdd(valid_cluster_count, 1);
        } else {
            cluster_valid[idx] = 0;
        }
    } else {
        cluster_valid[idx] = 0;
        cluster_scores[idx] = 0.0f;
    }
}

/**
 * Compute per-cluster druggability prediction
 */
extern "C" __global__ void compute_cluster_druggability(
    const int* __restrict__ cluster_labels,
    const float* __restrict__ hydrophobicity,
    const float* __restrict__ packing_density,
    const float* __restrict__ nma_mobility,
    const float* __restrict__ probe_scores,
    const int* __restrict__ cluster_sizes,
    const int* __restrict__ cluster_valid,

    float* __restrict__ cluster_druggability,

    const int n_residues,
    const int max_clusters,
    const float optimal_volume
) {
    __shared__ float s_hydro[BLOCK_SIZE];
    __shared__ float s_pack[BLOCK_SIZE];
    __shared__ float s_nma[BLOCK_SIZE];
    __shared__ float s_probe[BLOCK_SIZE];

    const int cluster_id = blockIdx.x;
    if (cluster_id >= max_clusters || !cluster_valid[cluster_id]) return;

    // Reduce over residues in this cluster
    float sum_hydro = 0.0f, sum_pack = 0.0f, sum_nma = 0.0f, sum_probe = 0.0f;
    int count = 0;

    for (int i = threadIdx.x; i < n_residues; i += blockDim.x) {
        if (cluster_labels[i] == cluster_id) {
            sum_hydro += hydrophobicity[i];
            sum_pack += 1.0f - packing_density[i];  // Deficit
            sum_nma += nma_mobility[i];
            sum_probe += probe_scores[i];
            count++;
        }
    }

    s_hydro[threadIdx.x] = sum_hydro;
    s_pack[threadIdx.x] = sum_pack;
    s_nma[threadIdx.x] = sum_nma;
    s_probe[threadIdx.x] = sum_probe;
    __syncthreads();

    // Reduce in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_hydro[threadIdx.x] += s_hydro[threadIdx.x + s];
            s_pack[threadIdx.x] += s_pack[threadIdx.x + s];
            s_nma[threadIdx.x] += s_nma[threadIdx.x + s];
            s_probe[threadIdx.x] += s_probe[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        int size = cluster_sizes[cluster_id];
        if (size > 0) {
            float avg_hydro = s_hydro[0] / size;
            float avg_pack = s_pack[0] / size;
            float avg_nma = s_nma[0] / size;
            float avg_probe = s_probe[0] / size;

            // Estimate volume (residues * ~150 Å³)
            float est_volume = size * 150.0f;
            float volume_factor = fminf(est_volume / optimal_volume, 1.0f);

            // Druggability: weighted combination
            float druggability = 0.25f * avg_hydro
                               + 0.25f * volume_factor
                               + 0.15f * avg_pack
                               + 0.10f * avg_nma
                               + 0.10f * avg_probe
                               + 0.15f * fminf((float)size / 10.0f, 1.0f);

            cluster_druggability[cluster_id] = fminf(fmaxf(druggability, 0.0f), 1.0f);
        }
    }
}
