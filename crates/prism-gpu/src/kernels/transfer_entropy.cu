// Transfer Entropy kernel using KSG estimator for causal discovery
//
// ASSUMPTIONS:
// - Time series data stored as contiguous f32 arrays
// - MAX_SERIES_LENGTH = 10000 (time points limit)
// - MAX_VARIABLES = 256 (number of variables for causal graph)
// - k-nearest neighbors: k = 4 (KSG standard)
// - Precision: f32 for distance calculations
// - Block size: 256 threads (optimal for RTX 3060)
// - Grid size: ceil(num_pairs / threads_per_block)
// - Requires: sm_70+ for efficient sorting
// REFERENCE: PRISM Spec Section 5.3 "Causal Discovery via Transfer Entropy"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <float.h>

// Configuration constants
constexpr int MAX_SERIES_LENGTH = 10000;
constexpr int MAX_VARIABLES = 256;
constexpr int THREADS_PER_BLOCK = 256;
constexpr int K_NEIGHBORS = 4;  // KSG k-nearest neighbors
constexpr float EPSILON = 1e-10f;

// Transfer entropy parameters
struct TEParams {
    int num_variables;     // Number of time series variables
    int series_length;     // Length of each time series
    int history_length;    // Past values to consider (tau)
    int prediction_lag;    // Future prediction lag
    float noise_level;     // Small noise for numerical stability
    int bootstrap_samples; // Number of bootstrap samples for significance
};

// Device function: Compute Euclidean distance between embedded vectors
__device__ float compute_embedding_distance(
    const float* __restrict__ x,
    const float* __restrict__ y,
    int dim
) {
    float dist = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float diff = x[i] - y[i];
        dist += diff * diff;
    }
    return sqrtf(dist + EPSILON);
}

// Device function: Find k-nearest neighbors using partial sort
__device__ void find_k_nearest_neighbors(
    const float* __restrict__ distances,
    int* __restrict__ indices,
    int n,
    int k,
    int exclude_idx
) {
    // Initialize indices
    for (int i = 0; i < k; ++i) {
        indices[i] = -1;
    }

    float max_dist = FLT_MAX;
    int max_idx = 0;

    // Find k smallest distances (excluding self)
    for (int i = 0; i < n; ++i) {
        if (i == exclude_idx) continue;

        float dist = distances[i];

        // Check if this distance should be in top-k
        if (dist < max_dist) {
            // Replace the maximum distance in current k-set
            indices[max_idx] = i;

            // Find new maximum in k-set
            max_dist = 0.0f;
            for (int j = 0; j < k; ++j) {
                if (indices[j] >= 0 && distances[indices[j]] > max_dist) {
                    max_dist = distances[indices[j]];
                    max_idx = j;
                }
            }
        }
    }
}

// Device function: Digamma function approximation
__device__ float digamma(float x) {
    // Asymptotic expansion for digamma function
    if (x < 6.0f) {
        // Recursion to reach asymptotic region
        return digamma(x + 1.0f) - 1.0f / x;
    }

    // Asymptotic series
    float result = logf(x) - 0.5f / x;
    float x2 = x * x;
    result -= 1.0f / (12.0f * x2);
    result += 1.0f / (120.0f * x2 * x2);
    result -= 1.0f / (252.0f * x2 * x2 * x2);
    return result;
}

// Main Transfer Entropy computation kernel (KSG estimator)
extern "C" __global__ void transfer_entropy_ksg_kernel(
    const float* __restrict__ time_series,    // [num_variables][series_length]
    float* __restrict__ te_matrix,            // [num_variables][num_variables] output
    float* __restrict__ significance,         // [num_variables][num_variables] p-values
    int* __restrict__ neighbor_counts,        // Workspace for neighbor counting
    TEParams params
) {
    // Each thread computes TE for one variable pair
    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = params.num_variables * params.num_variables;
    if (pair_idx >= total_pairs) return;

    int source_var = pair_idx / params.num_variables;
    int target_var = pair_idx % params.num_variables;

    // Skip self-connections
    if (source_var == target_var) {
        te_matrix[pair_idx] = 0.0f;
        significance[pair_idx] = 1.0f;
        return;
    }

    // Embedding dimensions
    int dim_x = params.history_length;      // Past of source
    int dim_y = params.history_length;      // Past of target
    int dim_z = 1;                          // Future of target

    int joint_dim = dim_x + dim_y + dim_z;
    int marginal_dim = dim_y + dim_z;

    // Allocate shared memory for distance calculations
    extern __shared__ float shared_mem[];
    float* distances = &shared_mem[threadIdx.x * params.series_length];

    float te_sum = 0.0f;
    int valid_points = 0;

    // Main loop over time points
    for (int t = params.history_length;
         t < params.series_length - params.prediction_lag;
         ++t) {

        // Construct joint embedding [X_past, Y_past, Y_future]
        float joint_embedding[32]; // Max embedding dimension
        int embed_idx = 0;

        // X past values
        for (int tau = 0; tau < params.history_length; ++tau) {
            joint_embedding[embed_idx++] =
                time_series[source_var * params.series_length + t - tau];
        }

        // Y past values
        for (int tau = 0; tau < params.history_length; ++tau) {
            joint_embedding[embed_idx++] =
                time_series[target_var * params.series_length + t - tau];
        }

        // Y future value
        joint_embedding[embed_idx++] =
            time_series[target_var * params.series_length + t + params.prediction_lag];

        // Compute distances to all other time points (joint space)
        for (int t2 = params.history_length;
             t2 < params.series_length - params.prediction_lag;
             ++t2) {

            float other_embedding[32];
            int idx = 0;

            // Build comparison embedding
            for (int tau = 0; tau < params.history_length; ++tau) {
                other_embedding[idx++] =
                    time_series[source_var * params.series_length + t2 - tau];
            }
            for (int tau = 0; tau < params.history_length; ++tau) {
                other_embedding[idx++] =
                    time_series[target_var * params.series_length + t2 - tau];
            }
            other_embedding[idx++] =
                time_series[target_var * params.series_length + t2 + params.prediction_lag];

            distances[t2] = compute_embedding_distance(
                joint_embedding, other_embedding, joint_dim
            );
        }

        // Find k-nearest neighbors in joint space
        int k_neighbors_joint[K_NEIGHBORS];
        find_k_nearest_neighbors(
            distances,
            k_neighbors_joint,
            params.series_length - params.history_length - params.prediction_lag,
            K_NEIGHBORS,
            t
        );

        // Get distance to k-th neighbor
        float eps_joint = 0.0f;
        for (int k = 0; k < K_NEIGHBORS; ++k) {
            if (k_neighbors_joint[k] >= 0) {
                eps_joint = fmaxf(eps_joint, distances[k_neighbors_joint[k]]);
            }
        }
        eps_joint += params.noise_level; // Numerical stability

        // Count neighbors in marginal spaces
        int n_yz = 0;  // Neighbors in (Y_past, Y_future) space
        int n_y = 0;   // Neighbors in Y_past space
        int n_xyz = K_NEIGHBORS; // Already have this

        for (int t2 = params.history_length;
             t2 < params.series_length - params.prediction_lag;
             ++t2) {
            if (t2 == t) continue;

            // Check marginal distance (Y_past, Y_future)
            float dist_yz = 0.0f;
            for (int tau = 0; tau < params.history_length; ++tau) {
                float diff = time_series[target_var * params.series_length + t - tau] -
                            time_series[target_var * params.series_length + t2 - tau];
                dist_yz += diff * diff;
            }
            float diff_future = time_series[target_var * params.series_length + t + params.prediction_lag] -
                               time_series[target_var * params.series_length + t2 + params.prediction_lag];
            dist_yz += diff_future * diff_future;
            dist_yz = sqrtf(dist_yz + EPSILON);

            if (dist_yz < eps_joint) {
                n_yz++;
            }

            // Check marginal distance (Y_past only)
            float dist_y = 0.0f;
            for (int tau = 0; tau < params.history_length; ++tau) {
                float diff = time_series[target_var * params.series_length + t - tau] -
                            time_series[target_var * params.series_length + t2 - tau];
                dist_y += diff * diff;
            }
            dist_y = sqrtf(dist_y + EPSILON);

            if (dist_y < eps_joint) {
                n_y++;
            }
        }

        // KSG estimator formula
        // TE(X->Y) = ψ(k) - ψ(n_yz+1) - ψ(n_y+1) + ψ(n_xyz+1)
        float te_point = digamma((float)K_NEIGHBORS) +
                        digamma((float)(n_y + 1)) -
                        digamma((float)(n_yz + 1)) -
                        digamma((float)(n_xyz + 1));

        te_sum += te_point;
        valid_points++;
    }

    // Average transfer entropy
    float te_value = (valid_points > 0) ? (te_sum / valid_points) : 0.0f;

    // Ensure non-negative (KSG can sometimes give small negative values)
    te_value = fmaxf(0.0f, te_value);

    // Store result
    te_matrix[pair_idx] = te_value;

    // Bootstrap significance test would go here
    // For now, use threshold-based significance
    significance[pair_idx] = (te_value > 0.01f) ? 0.05f : 1.0f;
}

// Conditional Transfer Entropy kernel (accounts for confounders)
extern "C" __global__ void conditional_te_kernel(
    const float* __restrict__ time_series,
    const int* __restrict__ conditioning_vars, // Which variables to condition on
    float* __restrict__ cte_matrix,
    TEParams params,
    int num_conditioning
) {
    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_idx >= params.num_variables * params.num_variables) return;

    int source = pair_idx / params.num_variables;
    int target = pair_idx % params.num_variables;

    if (source == target) {
        cte_matrix[pair_idx] = 0.0f;
        return;
    }

    // Extended embedding includes conditioning variables
    int total_dim = params.history_length * (2 + num_conditioning) + 1;

    // Simplified CTE calculation
    // Full implementation would follow similar pattern to main TE kernel
    // but with larger embedding space including conditioning variables

    float cte_value = 0.0f;

    // Placeholder: Would compute full CTE here
    // For now, approximate as TE with penalty for conditioning
    float base_te = 0.1f; // Would get from te_matrix
    cte_value = base_te * expf(-0.1f * num_conditioning);

    cte_matrix[pair_idx] = cte_value;
}

// Multivariate Transfer Entropy kernel (multiple sources to one target)
extern "C" __global__ void multivariate_te_kernel(
    const float* __restrict__ time_series,
    const int* __restrict__ source_indices,
    int num_sources,
    int target_idx,
    float* __restrict__ mte_result,
    TEParams params
) {
    // Cooperative computation across thread block
    extern __shared__ float block_shared[];

    int tid = threadIdx.x;
    int t = blockIdx.x * blockDim.x + tid;

    if (t >= params.series_length - params.history_length - params.prediction_lag) {
        if (tid == 0 && blockIdx.x == 0) {
            *mte_result = 0.0f;
        }
        return;
    }

    // Build high-dimensional embedding from all sources
    float joint_contrib = 0.0f;

    // Simplified multivariate TE
    // Full implementation would use KSG on joint distribution

    // Store partial results in shared memory
    block_shared[tid] = joint_contrib;
    __syncthreads();

    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            block_shared[tid] += block_shared[tid + stride];
        }
        __syncthreads();
    }

    // Final result
    if (tid == 0) {
        atomicAdd(mte_result, block_shared[0]);
    }
}

// Sliding window TE for dynamic causal discovery
extern "C" __global__ void sliding_window_te_kernel(
    const float* __restrict__ time_series,
    float* __restrict__ te_timeseries, // TE values over time
    int source_idx,
    int target_idx,
    int window_size,
    int window_stride,
    TEParams params
) {
    int window_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_windows = (params.series_length - window_size) / window_stride + 1;

    if (window_idx >= num_windows) return;

    int window_start = window_idx * window_stride;
    int window_end = window_start + window_size;

    // Compute TE for this window
    float window_te = 0.0f;
    int valid_points = 0;

    for (int t = window_start + params.history_length;
         t < window_end - params.prediction_lag && t < params.series_length;
         ++t) {

        // Simplified TE computation for window
        // Would use full KSG here in production

        float local_te = 0.0f;

        // Placeholder calculation
        float source_val = time_series[source_idx * params.series_length + t];
        float target_future = time_series[target_idx * params.series_length +
                                         t + params.prediction_lag];
        local_te = fabsf(source_val * target_future) * 0.01f;

        window_te += local_te;
        valid_points++;
    }

    te_timeseries[window_idx] = (valid_points > 0) ?
                                (window_te / valid_points) : 0.0f;
}

// Ensemble TE kernel for robustness (multiple parameter settings)
extern "C" __global__ void ensemble_te_kernel(
    const float* __restrict__ time_series,
    float* __restrict__ ensemble_te_matrix,
    TEParams* __restrict__ param_variants, // Array of different parameters
    int num_variants
) {
    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_pairs = param_variants[0].num_variables * param_variants[0].num_variables;

    if (pair_idx >= num_pairs) return;

    float te_sum = 0.0f;
    float te_sum_sq = 0.0f;

    // Compute TE for each parameter variant
    for (int v = 0; v < num_variants; ++v) {
        // Would call main TE computation here
        float te_variant = 0.1f * (v + 1); // Placeholder

        te_sum += te_variant;
        te_sum_sq += te_variant * te_variant;
    }

    // Compute mean and variance
    float te_mean = te_sum / num_variants;
    float te_var = (te_sum_sq / num_variants) - (te_mean * te_mean);
    float te_std = sqrtf(te_var + EPSILON);

    // Store robust estimate (mean with confidence)
    ensemble_te_matrix[pair_idx * 2] = te_mean;
    ensemble_te_matrix[pair_idx * 2 + 1] = te_std;
}

// Performance metrics kernel
extern "C" __global__ void te_performance_metrics(
    const float* __restrict__ te_matrix,
    float* __restrict__ metrics, // [sparsity, strength, modularity]
    int num_variables
) {
    // Single thread computes aggregate metrics
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int num_edges = 0;
        float total_strength = 0.0f;
        float max_te = 0.0f;

        for (int i = 0; i < num_variables * num_variables; ++i) {
            float te = te_matrix[i];
            if (te > 0.01f) { // Threshold for edge existence
                num_edges++;
                total_strength += te;
                max_te = fmaxf(max_te, te);
            }
        }

        float sparsity = 1.0f - (float)num_edges /
                        (num_variables * (num_variables - 1));
        float avg_strength = (num_edges > 0) ?
                            (total_strength / num_edges) : 0.0f;

        metrics[0] = sparsity;      // Graph sparsity
        metrics[1] = avg_strength;  // Average edge strength
        metrics[2] = max_te;        // Maximum TE value
    }
}