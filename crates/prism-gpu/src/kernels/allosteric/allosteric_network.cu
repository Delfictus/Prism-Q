/**
 * Allosteric Network Analysis CUDA Kernels
 *
 * GPU-accelerated network algorithms:
 * - Parallel Floyd-Warshall all-pairs shortest paths
 * - GPU Brandes betweenness centrality
 * - Communication pathway analysis
 *
 * Copyright (c) 2024 PRISM Research Team | Delfictus I/O Inc.
 */

#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>

// ============================================================================
// Constants
// ============================================================================

constexpr int BLOCK_SIZE = 256;
constexpr int TILE_SIZE = 16;
constexpr float INF = 1e30f;

// ============================================================================
// Floyd-Warshall All-Pairs Shortest Paths
// ============================================================================

/**
 * Initialize distance matrix from adjacency
 * Distance = 1/weight for weighted graphs
 */
extern "C" __global__ void floyd_warshall_init(
    const float* __restrict__ adjacency,
    float* __restrict__ dist,
    int* __restrict__ next,
    int n
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n || j >= n) return;

    int idx = i * n + j;

    if (i == j) {
        dist[idx] = 0.0f;
        next[idx] = i;
    } else {
        float weight = adjacency[idx];
        if (weight > 1e-10f) {
            dist[idx] = 1.0f / weight;  // Inverse weight as distance
            next[idx] = j;
        } else {
            dist[idx] = INF;
            next[idx] = -1;
        }
    }
}

/**
 * Floyd-Warshall iteration for intermediate vertex k
 * Uses blocked algorithm for better cache utilization
 */
extern "C" __global__ void floyd_warshall_iteration(
    float* __restrict__ dist,
    int* __restrict__ next,
    int k,
    int n
) {
    __shared__ float s_dist_ik[TILE_SIZE][TILE_SIZE];
    __shared__ float s_dist_kj[TILE_SIZE][TILE_SIZE];

    int i = blockIdx.y * TILE_SIZE + threadIdx.y;
    int j = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Load dist[i][k] and dist[k][j] into shared memory
    if (i < n && k < n) {
        s_dist_ik[threadIdx.y][threadIdx.x % TILE_SIZE] = dist[i * n + k];
    } else {
        s_dist_ik[threadIdx.y][threadIdx.x % TILE_SIZE] = INF;
    }

    if (k < n && j < n) {
        s_dist_kj[threadIdx.y % TILE_SIZE][threadIdx.x] = dist[k * n + j];
    } else {
        s_dist_kj[threadIdx.y % TILE_SIZE][threadIdx.x] = INF;
    }

    __syncthreads();

    if (i < n && j < n) {
        int idx = i * n + j;
        float d_ik = s_dist_ik[threadIdx.y][0];
        float d_kj = s_dist_kj[0][threadIdx.x];
        float through_k = d_ik + d_kj;

        if (through_k < dist[idx]) {
            dist[idx] = through_k;
            // Update predecessor: next[i][j] = next[i][k]
            next[idx] = (k < n) ? next[i * n + k] : -1;
        }
    }
}

/**
 * Optimized blocked Floyd-Warshall for phase-dependent updates
 */
extern "C" __global__ void floyd_warshall_blocked(
    float* __restrict__ dist,
    int* __restrict__ next,
    int k_start,
    int k_end,
    int n
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n || j >= n) return;

    int idx = i * n + j;
    float current_dist = dist[idx];
    int current_next = next[idx];

    for (int k = k_start; k < k_end && k < n; k++) {
        float d_ik = dist[i * n + k];
        float d_kj = dist[k * n + j];
        float through_k = d_ik + d_kj;

        if (through_k < current_dist) {
            current_dist = through_k;
            current_next = next[i * n + k];
        }
    }

    dist[idx] = current_dist;
    next[idx] = current_next;
}

// ============================================================================
// Brandes Betweenness Centrality
// ============================================================================

/**
 * BFS from single source for Brandes algorithm
 * Computes sigma (shortest path counts) and distances
 */
extern "C" __global__ void brandes_bfs_step(
    const float* __restrict__ adjacency,
    int* __restrict__ dist,
    float* __restrict__ sigma,
    int* __restrict__ frontier,
    int* __restrict__ next_frontier,
    int* __restrict__ frontier_size,
    int* __restrict__ next_frontier_size,
    int current_dist,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= *frontier_size) return;

    int v = frontier[tid];

    // Explore neighbors of v
    for (int w = 0; w < n; w++) {
        float edge = adjacency[v * n + w];
        if (edge <= 1e-10f) continue;

        // First visit to w
        int old_dist = atomicCAS(&dist[w], -1, current_dist + 1);
        if (old_dist == -1) {
            // Add to next frontier
            int pos = atomicAdd(next_frontier_size, 1);
            next_frontier[pos] = w;
        }

        // If w is at distance current_dist + 1, update sigma
        if (dist[w] == current_dist + 1) {
            atomicAdd(&sigma[w], sigma[v]);
        }
    }
}

/**
 * Back-propagation step for Brandes algorithm
 */
extern "C" __global__ void brandes_backprop(
    const float* __restrict__ adjacency,
    const int* __restrict__ dist,
    const float* __restrict__ sigma,
    float* __restrict__ delta,
    float* __restrict__ centrality,
    int* __restrict__ frontier,
    int frontier_size,
    int source,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= frontier_size) return;

    int w = frontier[tid];
    int w_dist = dist[w];

    float delta_w = delta[w];

    // Find predecessors
    for (int v = 0; v < n; v++) {
        float edge = adjacency[v * n + w];
        if (edge <= 1e-10f) continue;

        // v is predecessor if dist[v] == dist[w] - 1
        if (dist[v] == w_dist - 1) {
            float contribution = (sigma[v] / sigma[w]) * (1.0f + delta_w);
            atomicAdd(&delta[v], contribution);
        }
    }

    if (w != source) {
        atomicAdd(&centrality[w], delta_w);
    }
}

/**
 * Parallel Brandes initialization
 */
extern "C" __global__ void brandes_init(
    int* __restrict__ dist,
    float* __restrict__ sigma,
    float* __restrict__ delta,
    int source,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    if (idx == source) {
        dist[idx] = 0;
        sigma[idx] = 1.0f;
    } else {
        dist[idx] = -1;
        sigma[idx] = 0.0f;
    }
    delta[idx] = 0.0f;
}

/**
 * Normalize betweenness centrality
 */
extern "C" __global__ void brandes_normalize(
    float* __restrict__ centrality,
    float norm_factor,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        centrality[idx] *= norm_factor;
    }
}

// ============================================================================
// Communication Pathway Analysis
// ============================================================================

/**
 * Find shortest paths between source and target regions
 */
extern "C" __global__ void find_communication_paths(
    const float* __restrict__ dist,
    const int* __restrict__ next,
    const int* __restrict__ sources,
    const int* __restrict__ targets,
    float* __restrict__ path_lengths,
    int* __restrict__ path_first_hop,
    int n_sources,
    int n_targets,
    int n
) {
    int s_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (s_idx >= n_sources || t_idx >= n_targets) return;

    int source = sources[s_idx];
    int target = targets[t_idx];
    int output_idx = s_idx * n_targets + t_idx;

    if (source < n && target < n) {
        path_lengths[output_idx] = dist[source * n + target];
        path_first_hop[output_idx] = next[source * n + target];
    } else {
        path_lengths[output_idx] = INF;
        path_first_hop[output_idx] = -1;
    }
}

/**
 * Calculate coupling strength between regions
 * Coupling = mean(1 / (1 + path_length))
 */
extern "C" __global__ void calculate_coupling_strength(
    const float* __restrict__ path_lengths,
    float* __restrict__ coupling,
    int n_paths
) {
    __shared__ float shared_sum[BLOCK_SIZE];
    __shared__ int shared_count[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float local_sum = 0.0f;
    int local_count = 0;

    if (idx < n_paths) {
        float length = path_lengths[idx];
        if (length < INF) {
            local_sum = 1.0f / (1.0f + length);
            local_count = 1;
        }
    }

    shared_sum[tid] = local_sum;
    shared_count[tid] = local_count;
    __syncthreads();

    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
            shared_count[tid] += shared_count[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&coupling[0], shared_sum[0]);
        atomicAdd((int*)&coupling[1], shared_count[0]);  // Store count in second element
    }
}

// ============================================================================
// Closeness Centrality
// ============================================================================

/**
 * Calculate closeness centrality from distance matrix
 */
extern "C" __global__ void calculate_closeness_centrality(
    const float* __restrict__ dist,
    float* __restrict__ closeness,
    int n
) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;

    if (node >= n) return;

    float total_dist = 0.0f;
    int reachable = 0;

    for (int j = 0; j < n; j++) {
        float d = dist[node * n + j];
        if (d < INF && d > 0.0f) {
            total_dist += d;
            reachable++;
        }
    }

    if (total_dist > 1e-10f && reachable > 0) {
        closeness[node] = (float)reachable / total_dist;
    } else {
        closeness[node] = 0.0f;
    }
}

// ============================================================================
// Degree Centrality
// ============================================================================

/**
 * Calculate degree centrality
 */
extern "C" __global__ void calculate_degree_centrality(
    const float* __restrict__ adjacency,
    float* __restrict__ degree,
    int n
) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;

    if (node >= n) return;

    int count = 0;
    for (int j = 0; j < n; j++) {
        if (adjacency[node * n + j] > 1e-10f) {
            count++;
        }
    }

    degree[node] = (n > 1) ? (float)count / (float)(n - 1) : 0.0f;
}

// ============================================================================
// Eigenvector Centrality (Power Iteration)
// ============================================================================

/**
 * Single iteration of power method for eigenvector centrality
 */
extern "C" __global__ void eigenvector_centrality_iteration(
    const float* __restrict__ adjacency,
    const float* __restrict__ current,
    float* __restrict__ next,
    int n
) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;

    if (node >= n) return;

    float sum = 0.0f;
    for (int j = 0; j < n; j++) {
        sum += adjacency[node * n + j] * current[j];
    }

    next[node] = sum;
}

/**
 * Normalize eigenvector centrality
 */
extern "C" __global__ void normalize_eigenvector_centrality(
    float* __restrict__ centrality,
    float norm,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n && norm > 1e-10f) {
        centrality[idx] /= norm;
    }
}

// ============================================================================
// Fused Network Analysis Kernel
// ============================================================================

/**
 * Fused kernel to build network and compute initial centrality metrics
 */
extern "C" __global__ void fused_network_metrics(
    const float* __restrict__ coords,
    float* __restrict__ adjacency,
    float* __restrict__ degrees,
    float* __restrict__ closeness_init,
    int n,
    float contact_cutoff_sq,
    float sigma_sq_2
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n || col >= n) return;

    // Build adjacency
    float weight = 0.0f;
    if (row != col) {
        float dx = coords[row * 3 + 0] - coords[col * 3 + 0];
        float dy = coords[row * 3 + 1] - coords[col * 3 + 1];
        float dz = coords[row * 3 + 2] - coords[col * 3 + 2];
        float dist_sq = dx * dx + dy * dy + dz * dz;

        if (dist_sq < contact_cutoff_sq) {
            weight = expf(-dist_sq / sigma_sq_2);
        }
    }
    adjacency[row * n + col] = weight;
    __syncthreads();

    // Compute degree (row sum)
    if (col == 0) {
        float degree = 0.0f;
        for (int j = 0; j < n; j++) {
            degree += adjacency[row * n + j];
        }
        degrees[row] = degree;

        // Initial closeness estimate (inverse degree-weighted)
        closeness_init[row] = (degree > 1e-10f) ? degree : 0.0f;
    }
}
