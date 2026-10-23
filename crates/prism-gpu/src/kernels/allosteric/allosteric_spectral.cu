/**
 * Allosteric Spectral Clustering CUDA Kernel
 *
 * GPU-accelerated spectral clustering for domain decomposition:
 * - Contact matrix construction with Gaussian weights
 * - Normalized Laplacian computation
 * - Power iteration eigendecomposition
 * - K-means clustering on eigenvector embedding
 *
 * Copyright (c) 2024 PRISM Research Team | Delfictus I/O Inc.
 */

#include <cuda_runtime.h>
#include <cmath>

// ============================================================================
// Constants
// ============================================================================

constexpr int BLOCK_SIZE = 256;
constexpr int TILE_SIZE = 16;
constexpr float CONTACT_CUTOFF_SQ = 100.0f;  // 10 Å squared
constexpr float GAUSSIAN_SIGMA_SQ_2 = 72.0f; // 2 * 6² for Gaussian decay

// ============================================================================
// Contact Matrix Construction
// ============================================================================

/**
 * Build contact matrix from Cα coordinates with Gaussian weighting
 * Each block handles a tile of the matrix
 */
extern "C" __global__ void build_contact_matrix(
    const float* __restrict__ coords,    // [n_residues, 3] Cα coordinates
    float* __restrict__ contact_matrix,   // [n_residues, n_residues] output
    int n_residues
) {
    __shared__ float tile_coords_i[TILE_SIZE][3];
    __shared__ float tile_coords_j[TILE_SIZE][3];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float value = 0.0f;

    // Process tiles
    int n_tiles = (n_residues + TILE_SIZE - 1) / TILE_SIZE;

    // Load coordinates into shared memory
    if (threadIdx.x < 3 && row < n_residues) {
        tile_coords_i[threadIdx.y][threadIdx.x] = coords[row * 3 + threadIdx.x];
    }
    if (threadIdx.x < 3 && col < n_residues) {
        tile_coords_j[threadIdx.y][threadIdx.x] = coords[col * 3 + threadIdx.x];
    }
    __syncthreads();

    if (row < n_residues && col < n_residues && row != col) {
        // Calculate squared distance
        float dx = tile_coords_i[threadIdx.y][0] - tile_coords_j[threadIdx.x][0];
        float dy = tile_coords_i[threadIdx.y][1] - tile_coords_j[threadIdx.x][1];
        float dz = tile_coords_i[threadIdx.y][2] - tile_coords_j[threadIdx.x][2];
        float dist_sq = dx * dx + dy * dy + dz * dz;

        // Gaussian contact weight if within cutoff
        if (dist_sq < CONTACT_CUTOFF_SQ) {
            value = expf(-dist_sq / GAUSSIAN_SIGMA_SQ_2);
        }
    }

    if (row < n_residues && col < n_residues) {
        contact_matrix[row * n_residues + col] = value;
    }
}

// ============================================================================
// Degree Computation
// ============================================================================

/**
 * Compute degree (row sum) for each node
 */
extern "C" __global__ void compute_degrees(
    const float* __restrict__ adjacency,
    float* __restrict__ degrees,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            sum += adjacency[idx * n + j];
        }
        degrees[idx] = sum;
    }
}

// ============================================================================
// Normalized Laplacian
// ============================================================================

/**
 * Compute normalized Laplacian: L = I - D^(-1/2) * A * D^(-1/2)
 */
extern "C" __global__ void compute_normalized_laplacian(
    const float* __restrict__ adjacency,
    const float* __restrict__ degrees,
    float* __restrict__ laplacian,
    int n
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float value;

        if (row == col) {
            value = 1.0f;  // Diagonal is 1
        } else {
            float d_i = degrees[row];
            float d_j = degrees[col];
            float adj = adjacency[row * n + col];

            if (d_i > 1e-10f && d_j > 1e-10f && adj > 0.0f) {
                float inv_sqrt_di = rsqrtf(d_i);
                float inv_sqrt_dj = rsqrtf(d_j);
                value = -adj * inv_sqrt_di * inv_sqrt_dj;
            } else {
                value = 0.0f;
            }
        }

        laplacian[row * n + col] = value;
    }
}

// ============================================================================
// Power Iteration Eigendecomposition
// ============================================================================

/**
 * Matrix-vector multiplication: y = A * x
 */
extern "C" __global__ void matvec_multiply(
    const float* __restrict__ matrix,
    const float* __restrict__ x,
    float* __restrict__ y,
    int n
) {
    __shared__ float shared_x[BLOCK_SIZE];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= n) return;

    float sum = 0.0f;

    // Process in chunks
    for (int chunk = 0; chunk < n; chunk += BLOCK_SIZE) {
        int load_idx = chunk + tid;
        if (load_idx < n) {
            shared_x[tid] = x[load_idx];
        } else {
            shared_x[tid] = 0.0f;
        }
        __syncthreads();

        int end = min(BLOCK_SIZE, n - chunk);
        for (int j = 0; j < end; j++) {
            sum += matrix[row * n + chunk + j] * shared_x[j];
        }
        __syncthreads();
    }

    if (tid == 0) {
        y[row] = sum;
    }
}

/**
 * Compute vector norm (squared)
 */
extern "C" __global__ void vector_norm_sq(
    const float* __restrict__ v,
    float* __restrict__ norm_sq,
    int n
) {
    __shared__ float shared_sum[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float val = (idx < n) ? v[idx] * v[idx] : 0.0f;
    shared_sum[tid] = val;
    __syncthreads();

    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(norm_sq, shared_sum[0]);
    }
}

/**
 * Normalize vector and subtract projection
 */
extern "C" __global__ void normalize_and_deflate(
    float* __restrict__ v,
    const float* __restrict__ prev_eigenvector,
    float norm,
    float projection,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // Normalize
        v[idx] /= norm;

        // Deflate (remove component along previous eigenvector)
        if (prev_eigenvector != nullptr) {
            v[idx] -= projection * prev_eigenvector[idx];
        }
    }
}

// ============================================================================
// K-Means Clustering
// ============================================================================

/**
 * Assign points to nearest centroid
 */
extern "C" __global__ void kmeans_assign(
    const float* __restrict__ embedding,  // [n, k] eigenvector embedding
    const float* __restrict__ centroids,  // [n_clusters, k]
    int* __restrict__ assignments,
    int n,
    int k,
    int n_clusters
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    float min_dist = 1e30f;
    int best_cluster = 0;

    for (int c = 0; c < n_clusters; c++) {
        float dist = 0.0f;
        for (int d = 0; d < k; d++) {
            float diff = embedding[idx * k + d] - centroids[c * k + d];
            dist += diff * diff;
        }

        if (dist < min_dist) {
            min_dist = dist;
            best_cluster = c;
        }
    }

    assignments[idx] = best_cluster;
}

/**
 * Update centroids from assignments
 */
extern "C" __global__ void kmeans_update_centroids(
    const float* __restrict__ embedding,
    const int* __restrict__ assignments,
    float* __restrict__ centroids,
    int* __restrict__ counts,
    int n,
    int k,
    int n_clusters
) {
    int c = blockIdx.x;  // Cluster index
    int d = threadIdx.x; // Dimension index

    if (c >= n_clusters || d >= k) return;

    float sum = 0.0f;
    int count = 0;

    for (int i = 0; i < n; i++) {
        if (assignments[i] == c) {
            sum += embedding[i * k + d];
            if (d == 0) count++;
        }
    }

    if (d == 0) {
        counts[c] = count;
    }

    if (count > 0) {
        centroids[c * k + d] = sum / count;
    }
}

// ============================================================================
// Fused Spectral Clustering Kernel
// ============================================================================

/**
 * Fused kernel for full spectral clustering pipeline
 * Combines contact matrix, Laplacian, and initial eigenvector setup
 */
extern "C" __global__ void fused_spectral_init(
    const float* __restrict__ coords,
    float* __restrict__ contact_matrix,
    float* __restrict__ degrees,
    float* __restrict__ laplacian,
    float* __restrict__ eigenvector,
    int n
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n || col >= n) return;

    // Step 1: Build contact matrix element
    float contact_value = 0.0f;
    if (row != col) {
        float dx = coords[row * 3 + 0] - coords[col * 3 + 0];
        float dy = coords[row * 3 + 1] - coords[col * 3 + 1];
        float dz = coords[row * 3 + 2] - coords[col * 3 + 2];
        float dist_sq = dx * dx + dy * dy + dz * dz;

        if (dist_sq < CONTACT_CUTOFF_SQ) {
            contact_value = expf(-dist_sq / GAUSSIAN_SIGMA_SQ_2);
        }
    }
    contact_matrix[row * n + col] = contact_value;
    __syncthreads();

    // Step 2: Compute degree for this row
    if (col == 0) {
        float degree = 0.0f;
        for (int j = 0; j < n; j++) {
            degree += contact_matrix[row * n + j];
        }
        degrees[row] = degree;
    }
    __syncthreads();

    // Step 3: Compute Laplacian element
    float lap_value;
    if (row == col) {
        lap_value = 1.0f;
    } else {
        float d_i = degrees[row];
        float d_j = degrees[col];
        if (d_i > 1e-10f && d_j > 1e-10f && contact_value > 0.0f) {
            lap_value = -contact_value * rsqrtf(d_i) * rsqrtf(d_j);
        } else {
            lap_value = 0.0f;
        }
    }
    laplacian[row * n + col] = lap_value;

    // Step 4: Initialize eigenvector (uniform)
    if (col == 0) {
        eigenvector[row] = 1.0f / sqrtf((float)n);
    }
}
