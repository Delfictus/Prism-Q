/**
 * Hybrid TDA Ultimate CUDA Kernel
 *
 * Warp-cooperative computation of TDA features from spatial neighborhoods.
 * Uses parallel union-find with __shfl_sync for efficient Betti-0 computation.
 *
 * Features extracted per radius (16 total):
 * - Betti-0 at 4 scales (connected components)
 * - Betti-1 at 4 scales (loops/holes)
 * - Persistence features (total, max, entropy, significant)
 * - Directional features (+x, +y, +z hemisphere densities, anisotropy)
 *
 * Architecture: One warp (32 threads) per residue
 * Memory: Uses registers + warp shuffles for minimal shared memory
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>

// Feature counts
#define FEATURES_PER_RADIUS 16
#define NUM_RADII 3
#define TDA_FEATURE_COUNT (NUM_RADII * FEATURES_PER_RADIUS)
#define MAX_NEIGHBORS 64
#define WARP_SIZE 32

// Feature indices within each radius block
#define BETTI0_SCALE0 0
#define BETTI0_SCALE1 1
#define BETTI0_SCALE2 2
#define BETTI0_SCALE3 3
#define BETTI1_SCALE0 4
#define BETTI1_SCALE1 5
#define BETTI1_SCALE2 6
#define BETTI1_SCALE3 7
#define TOTAL_PERSISTENCE 8
#define MAX_PERSISTENCE 9
#define PERSISTENCE_ENTROPY 10
#define SIGNIFICANT_FEATURES 11
#define DIR_PLUS_X 12
#define DIR_PLUS_Y 13
#define DIR_PLUS_Z 14
#define ANISOTROPY 15

// F16 conversion
__device__ __forceinline__ float f16_to_f32(unsigned short bits) {
    unsigned int sign = (bits >> 15) & 1;
    unsigned int exp = (bits >> 10) & 0x1F;
    unsigned int mant = bits & 0x3FF;

    if (exp == 0) {
        if (mant == 0) {
            return sign ? -0.0f : 0.0f;
        }
        // Subnormal
        float f = ldexpf((float)mant, -24);
        return sign ? -f : f;
    }
    if (exp == 31) {
        if (mant == 0) {
            return sign ? -CUDART_INF_F : CUDART_INF_F;
        }
        return CUDART_NAN_F;
    }

    // Normal
    float f = ldexpf((float)(mant | 0x400), exp - 25);
    return sign ? -f : f;
}

// Warp-level parallel union-find for Betti-0
__device__ __forceinline__ int warp_find(int* parent, int x, unsigned int mask) {
    int p = parent[x];
    while (p != x) {
        x = p;
        p = parent[x];
    }
    return x;
}

__device__ __forceinline__ void warp_union(int* parent, int* rank, int x, int y, unsigned int mask) {
    int rx = warp_find(parent, x, mask);
    int ry = warp_find(parent, y, mask);

    if (rx != ry) {
        if (rank[rx] < rank[ry]) {
            parent[rx] = ry;
        } else if (rank[rx] > rank[ry]) {
            parent[ry] = rx;
        } else {
            parent[ry] = rx;
            rank[rx]++;
        }
    }
}

// Count connected components (Betti-0)
__device__ int count_components(int n, int* parent) {
    int count = 0;
    for (int i = 0; i < n; i++) {
        if (parent[i] == i) {
            count++;
        }
    }
    return count;
}

// Shared memory workspace for union-find
struct TdaWorkspace {
    int parent[MAX_NEIGHBORS];
    int rank[MAX_NEIGHBORS];
    float neighbor_coords[MAX_NEIGHBORS * 3];
    float distances[MAX_NEIGHBORS];
};

/**
 * Hybrid TDA Kernel
 *
 * @param offsets       [n_residues * n_radii + 1] Offsets into neighbor arrays
 * @param neighbor_idx  Packed neighbor indices
 * @param neighbor_dist Packed neighbor distances (F16)
 * @param neighbor_coords Packed neighbor coordinates (xyz)
 * @param center_coords [n_residues * 3] Center coordinates
 * @param features      [n_residues * TDA_FEATURE_COUNT] Output features
 * @param n_residues    Number of residues
 * @param n_radii       Number of radii
 * @param scale0-3      Persistence scales in Angstroms
 */
extern "C" __global__ void hybrid_tda_kernel(
    const unsigned int* __restrict__ offsets,
    const unsigned int* __restrict__ neighbor_idx,
    const unsigned short* __restrict__ neighbor_dist,
    const float* __restrict__ neighbor_coords,
    const float* __restrict__ center_coords,
    float* __restrict__ features,
    unsigned int n_residues,
    unsigned int n_radii,
    float scale0,
    float scale1,
    float scale2,
    float scale3
) {
    // One warp per residue
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id >= n_residues) return;

    const float scales[4] = {scale0, scale1, scale2, scale3};

    // Shared memory workspace (one per warp in the block)
    __shared__ TdaWorkspace workspace[8]; // 256 threads = 8 warps
    const int warp_in_block = (threadIdx.x / WARP_SIZE);
    TdaWorkspace& ws = workspace[warp_in_block];

    // Get center coordinates
    const float cx = center_coords[warp_id * 3 + 0];
    const float cy = center_coords[warp_id * 3 + 1];
    const float cz = center_coords[warp_id * 3 + 2];

    // Output pointer
    float* out = features + warp_id * TDA_FEATURE_COUNT;

    // Process each radius
    for (int r = 0; r < n_radii; r++) {
        const int offset_idx = warp_id * n_radii + r;
        const unsigned int start = offsets[offset_idx];
        const unsigned int end = offsets[offset_idx + 1];
        const int n_neighbors = min((int)(end - start), MAX_NEIGHBORS);

        if (n_neighbors == 0) {
            // No neighbors - zero features
            for (int f = 0; f < FEATURES_PER_RADIUS; f++) {
                out[r * FEATURES_PER_RADIUS + f] = 0.0f;
            }
            continue;
        }

        // Load neighbor data (parallel across warp)
        for (int i = lane_id; i < n_neighbors; i += WARP_SIZE) {
            const unsigned int idx = start + i;
            ws.neighbor_coords[i * 3 + 0] = neighbor_coords[idx * 3 + 0];
            ws.neighbor_coords[i * 3 + 1] = neighbor_coords[idx * 3 + 1];
            ws.neighbor_coords[i * 3 + 2] = neighbor_coords[idx * 3 + 2];
            ws.distances[i] = f16_to_f32(neighbor_dist[idx]);
        }
        __syncwarp();

        // Compute features at each scale
        float betti0[4] = {0, 0, 0, 0};
        float betti1[4] = {0, 0, 0, 0};

        for (int s = 0; s < 4; s++) {
            const float threshold = scales[s];

            // Initialize union-find (parallel)
            for (int i = lane_id; i < n_neighbors; i += WARP_SIZE) {
                ws.parent[i] = i;
                ws.rank[i] = 0;
            }
            __syncwarp();

            // Build edges and union at this scale
            int edge_count = 0;
            int triangle_count = 0;

            // Each lane handles a portion of edge pairs
            for (int i = lane_id; i < n_neighbors; i += WARP_SIZE) {
                const float x1 = ws.neighbor_coords[i * 3 + 0];
                const float y1 = ws.neighbor_coords[i * 3 + 1];
                const float z1 = ws.neighbor_coords[i * 3 + 2];

                for (int j = i + 1; j < n_neighbors; j++) {
                    const float x2 = ws.neighbor_coords[j * 3 + 0];
                    const float y2 = ws.neighbor_coords[j * 3 + 1];
                    const float z2 = ws.neighbor_coords[j * 3 + 2];

                    const float dx = x1 - x2;
                    const float dy = y1 - y2;
                    const float dz = z1 - z2;
                    const float dist = sqrtf(dx*dx + dy*dy + dz*dz);

                    if (dist <= threshold) {
                        // Union the two nodes
                        warp_union(ws.parent, ws.rank, i, j, 0xFFFFFFFF);
                        atomicAdd(&edge_count, 1);
                    }
                }
            }
            __syncwarp();

            // Compress paths (all lanes)
            for (int i = lane_id; i < n_neighbors; i += WARP_SIZE) {
                ws.parent[i] = warp_find(ws.parent, i, 0xFFFFFFFF);
            }
            __syncwarp();

            // Count components (single thread)
            if (lane_id == 0) {
                betti0[s] = (float)count_components(n_neighbors, ws.parent);

                // Estimate Betti-1 using Euler characteristic approximation
                // χ = V - E + F, β0 - β1 + β2 = χ
                // Assuming β2 ≈ 0: β1 ≈ β0 - χ = β0 - V + E - F
                // Rough estimate without triangles: β1 ≈ max(0, E - V + β0)
                const int v = n_neighbors;
                const int e = edge_count;
                betti1[s] = fmaxf(0.0f, (float)(e - v) + betti0[s]);
            }
            __syncwarp();

            // Broadcast from lane 0
            betti0[s] = __shfl_sync(0xFFFFFFFF, betti0[s], 0);
            betti1[s] = __shfl_sync(0xFFFFFFFF, betti1[s], 0);
        }

        // Store Betti numbers
        if (lane_id == 0) {
            out[r * FEATURES_PER_RADIUS + BETTI0_SCALE0] = betti0[0];
            out[r * FEATURES_PER_RADIUS + BETTI0_SCALE1] = betti0[1];
            out[r * FEATURES_PER_RADIUS + BETTI0_SCALE2] = betti0[2];
            out[r * FEATURES_PER_RADIUS + BETTI0_SCALE3] = betti0[3];
            out[r * FEATURES_PER_RADIUS + BETTI1_SCALE0] = betti1[0];
            out[r * FEATURES_PER_RADIUS + BETTI1_SCALE1] = betti1[1];
            out[r * FEATURES_PER_RADIUS + BETTI1_SCALE2] = betti1[2];
            out[r * FEATURES_PER_RADIUS + BETTI1_SCALE3] = betti1[3];
        }
        __syncwarp();

        // Compute persistence features
        float total_persistence = 0.0f;
        float max_persistence = 0.0f;
        float significant_count = 0.0f;

        // Persistence from Betti-0 transitions
        for (int s = 0; s < 3; s++) {
            const float died = betti0[s] - betti0[s + 1];
            if (died > 0) {
                const float persistence = scales[s + 1] - scales[s];
                total_persistence += died * persistence;
                max_persistence = fmaxf(max_persistence, persistence);
                if (persistence > 1.0f) {
                    significant_count += died;
                }
            }
        }

        // Entropy calculation
        float entropy = 0.0f;
        if (total_persistence > 0.0f) {
            for (int s = 0; s < 3; s++) {
                const float died = betti0[s] - betti0[s + 1];
                if (died > 0) {
                    const float persistence = scales[s + 1] - scales[s];
                    const float p = (died * persistence) / total_persistence;
                    if (p > 0.0f) {
                        entropy -= p * logf(p);
                    }
                }
            }
        }

        if (lane_id == 0) {
            out[r * FEATURES_PER_RADIUS + TOTAL_PERSISTENCE] = total_persistence;
            out[r * FEATURES_PER_RADIUS + MAX_PERSISTENCE] = max_persistence;
            out[r * FEATURES_PER_RADIUS + PERSISTENCE_ENTROPY] = entropy;
            out[r * FEATURES_PER_RADIUS + SIGNIFICANT_FEATURES] = significant_count;
        }

        // Directional features (parallel reduction)
        int plus_x = 0, plus_y = 0, plus_z = 0;

        for (int i = lane_id; i < n_neighbors; i += WARP_SIZE) {
            if (ws.neighbor_coords[i * 3 + 0] > cx) plus_x++;
            if (ws.neighbor_coords[i * 3 + 1] > cy) plus_y++;
            if (ws.neighbor_coords[i * 3 + 2] > cz) plus_z++;
        }

        // Warp reduce
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            plus_x += __shfl_down_sync(0xFFFFFFFF, plus_x, offset);
            plus_y += __shfl_down_sync(0xFFFFFFFF, plus_y, offset);
            plus_z += __shfl_down_sync(0xFFFFFFFF, plus_z, offset);
        }

        if (lane_id == 0) {
            const float n_f = (float)n_neighbors;
            const float dx = (float)plus_x / n_f;
            const float dy = (float)plus_y / n_f;
            const float dz = (float)plus_z / n_f;

            out[r * FEATURES_PER_RADIUS + DIR_PLUS_X] = dx;
            out[r * FEATURES_PER_RADIUS + DIR_PLUS_Y] = dy;
            out[r * FEATURES_PER_RADIUS + DIR_PLUS_Z] = dz;

            // Anisotropy (standard deviation of directional densities)
            const float mean = (dx + dy + dz) / 3.0f;
            const float var = ((dx - mean) * (dx - mean) +
                              (dy - mean) * (dy - mean) +
                              (dz - mean) * (dz - mean)) / 3.0f;
            out[r * FEATURES_PER_RADIUS + ANISOTROPY] = sqrtf(var);
        }

        __syncwarp();
    }
}

/**
 * Simple TDA kernel for small neighborhoods (single thread per residue)
 * Used as fallback for very small structures or debugging
 */
extern "C" __global__ void simple_tda_kernel(
    const unsigned int* __restrict__ offsets,
    const unsigned int* __restrict__ neighbor_idx,
    const unsigned short* __restrict__ neighbor_dist,
    const float* __restrict__ neighbor_coords,
    const float* __restrict__ center_coords,
    float* __restrict__ features,
    unsigned int n_residues,
    unsigned int n_radii,
    float scale0,
    float scale1,
    float scale2,
    float scale3
) {
    const int residue_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (residue_id >= n_residues) return;

    const float scales[4] = {scale0, scale1, scale2, scale3};

    const float cx = center_coords[residue_id * 3 + 0];
    const float cy = center_coords[residue_id * 3 + 1];
    const float cz = center_coords[residue_id * 3 + 2];

    float* out = features + residue_id * TDA_FEATURE_COUNT;

    // Process each radius
    for (int r = 0; r < n_radii; r++) {
        const int offset_idx = residue_id * n_radii + r;
        const unsigned int start = offsets[offset_idx];
        const unsigned int end = offsets[offset_idx + 1];
        const int n_neighbors = min((int)(end - start), MAX_NEIGHBORS);

        if (n_neighbors == 0) {
            for (int f = 0; f < FEATURES_PER_RADIUS; f++) {
                out[r * FEATURES_PER_RADIUS + f] = 0.0f;
            }
            continue;
        }

        // Load neighbor coordinates into local arrays
        float local_coords[MAX_NEIGHBORS * 3];
        for (int i = 0; i < n_neighbors; i++) {
            const unsigned int idx = start + i;
            local_coords[i * 3 + 0] = neighbor_coords[idx * 3 + 0];
            local_coords[i * 3 + 1] = neighbor_coords[idx * 3 + 1];
            local_coords[i * 3 + 2] = neighbor_coords[idx * 3 + 2];
        }

        // Simple union-find
        int parent[MAX_NEIGHBORS];
        int rank_arr[MAX_NEIGHBORS];

        for (int s = 0; s < 4; s++) {
            const float threshold = scales[s];

            // Initialize
            for (int i = 0; i < n_neighbors; i++) {
                parent[i] = i;
                rank_arr[i] = 0;
            }

            // Build edges
            for (int i = 0; i < n_neighbors; i++) {
                for (int j = i + 1; j < n_neighbors; j++) {
                    const float dx = local_coords[i*3+0] - local_coords[j*3+0];
                    const float dy = local_coords[i*3+1] - local_coords[j*3+1];
                    const float dz = local_coords[i*3+2] - local_coords[j*3+2];
                    const float dist = sqrtf(dx*dx + dy*dy + dz*dz);

                    if (dist <= threshold) {
                        // Find roots
                        int ri = i, rj = j;
                        while (parent[ri] != ri) ri = parent[ri];
                        while (parent[rj] != rj) rj = parent[rj];

                        if (ri != rj) {
                            if (rank_arr[ri] < rank_arr[rj]) {
                                parent[ri] = rj;
                            } else if (rank_arr[ri] > rank_arr[rj]) {
                                parent[rj] = ri;
                            } else {
                                parent[rj] = ri;
                                rank_arr[ri]++;
                            }
                        }
                    }
                }
            }

            // Count components
            int count = 0;
            for (int i = 0; i < n_neighbors; i++) {
                if (parent[i] == i) count++;
            }

            out[r * FEATURES_PER_RADIUS + s] = (float)count;
            out[r * FEATURES_PER_RADIUS + 4 + s] = 0.0f; // Betti-1 simplified
        }

        // Simplified persistence and directional features
        out[r * FEATURES_PER_RADIUS + TOTAL_PERSISTENCE] = 0.0f;
        out[r * FEATURES_PER_RADIUS + MAX_PERSISTENCE] = 0.0f;
        out[r * FEATURES_PER_RADIUS + PERSISTENCE_ENTROPY] = 0.0f;
        out[r * FEATURES_PER_RADIUS + SIGNIFICANT_FEATURES] = 0.0f;

        int plus_x = 0, plus_y = 0, plus_z = 0;
        for (int i = 0; i < n_neighbors; i++) {
            if (local_coords[i*3+0] > cx) plus_x++;
            if (local_coords[i*3+1] > cy) plus_y++;
            if (local_coords[i*3+2] > cz) plus_z++;
        }

        const float n_f = (float)n_neighbors;
        out[r * FEATURES_PER_RADIUS + DIR_PLUS_X] = (float)plus_x / n_f;
        out[r * FEATURES_PER_RADIUS + DIR_PLUS_Y] = (float)plus_y / n_f;
        out[r * FEATURES_PER_RADIUS + DIR_PLUS_Z] = (float)plus_z / n_f;
        out[r * FEATURES_PER_RADIUS + ANISOTROPY] = 0.0f;
    }
}
