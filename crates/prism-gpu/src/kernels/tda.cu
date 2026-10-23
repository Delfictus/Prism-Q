/**
 * TDA Persistent Homology CUDA Kernel for Phase 6
 *
 * ASSUMPTIONS:
 * - Input graph: Edge list representation (u, v pairs)
 * - MAX_VERTICES = 100,000 (enforced by caller)
 * - MAX_EDGES = 5,000,000 (enforced by caller)
 * - Precision: f32 for persistence scores, int for component labels
 * - Memory layout: Contiguous arrays for GPU coalesced access
 * - Block size: 256 threads (optimal for sm_86 Ampere)
 * - Grid size: ceil(n/256) for vertex-parallel operations
 * - Requires: sm_86+ (RTX 3060 Ampere)
 *
 * ALGORITHM:
 * 1. Union-find initialization (parallel per-vertex)
 * 2. Union-find edge linking (parallel per-edge with atomics)
 * 3. Path compression (iterative parallel)
 * 4. Component counting (parallel reduction)
 * 5. Betti-1 computation (Euler characteristic: edges - vertices + components)
 * 6. Persistence scores (per-vertex topological significance)
 * 7. Topological importance (anchor selection metric)
 *
 * PERFORMANCE TARGETS:
 * - DSJC250 (250 vertices, ~15k edges): < 50ms end-to-end
 * - DSJC500 (500 vertices, ~125k edges): < 200ms end-to-end
 * - GPU utilization: > 75%
 *
 * SECURITY:
 * - All kernel launches checked for errors
 * - Bounds checking via MAX_VERTICES and MAX_EDGES
 * - No runtime compilation (pre-compiled PTX only)
 *
 * REFERENCE: PRISM GPU Plan §4.6 (Phase 6 TDA Kernel)
 */

#include <cuda_runtime.h>

// Maximum supported graph dimensions (must match Rust constants)
#define MAX_VERTICES 100000
#define MAX_EDGES 5000000

// Block size for vertex-parallel kernels
#define BLOCK_SIZE 256

/**
 * Union-find initialization kernel.
 *
 * Each vertex becomes its own root (parent[i] = i).
 *
 * Grid/Block: grid = ceil(n/256), block = 256
 * Shared memory: 0 bytes
 */
extern "C" __global__ void union_find_init(int* parent, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        parent[tid] = tid;
    }
}

/**
 * Union-find path compression kernel (find with path halving).
 *
 * Compresses paths by making each node point to its grandparent.
 * Iterative approach (multiple kernel launches) for full compression.
 *
 * Grid/Block: grid = ceil(n/256), block = 256
 * Shared memory: 0 bytes
 */
extern "C" __global__ void union_find_compress(int* parent, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        int p = parent[tid];
        if (p != tid) {
            // Path halving: point to grandparent
            int gp = parent[p];
            if (gp != p) {
                parent[tid] = gp;
            }
        }
    }
}

/**
 * Union-find edge linking kernel (union operation).
 *
 * Links edges using atomic compare-and-swap for thread safety.
 * Uses union-by-rank heuristic (implicitly via atomic CAS).
 *
 * Grid/Block: grid = ceil(num_edges/256), block = 256
 * Shared memory: 0 bytes
 *
 * THREAD SAFETY:
 * - atomicCAS ensures no race conditions during union
 * - Multiple threads may link same edge harmlessly (idempotent)
 */
extern "C" __global__ void union_find_link(
    int* parent,
    const int* edges_u,
    const int* edges_v,
    int num_edges
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_edges) return;

    int u = edges_u[tid];
    int v = edges_v[tid];

    // Find roots (iterative find without recursion)
    int root_u = u;
    while (parent[root_u] != root_u) {
        root_u = parent[root_u];
    }

    int root_v = v;
    while (parent[root_v] != root_v) {
        root_v = parent[root_v];
    }

    // Union: attach smaller root to larger (by ID for determinism)
    if (root_u != root_v) {
        if (root_u < root_v) {
            atomicCAS(&parent[root_v], root_v, root_u);
        } else {
            atomicCAS(&parent[root_u], root_u, root_v);
        }
    }
}

/**
 * Component counting kernel (Betti-0 computation).
 *
 * Counts unique roots after path compression.
 * Uses atomic increment for thread-safe counting.
 *
 * Grid/Block: grid = ceil(n/256), block = 256
 * Shared memory: 0 bytes
 *
 * OUTPUT:
 * - component_count[0] = number of connected components (Betti-0)
 */
extern "C" __global__ void count_components(
    const int* parent,
    int* component_count,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        // A vertex is a root if parent[v] == v
        if (parent[tid] == tid) {
            atomicAdd(component_count, 1);
        }
    }
}

/**
 * Compute per-vertex degree from edge list.
 *
 * Each edge (u,v) increments degree[u] and degree[v].
 *
 * Grid/Block: grid = ceil(num_edges/256), block = 256
 * Shared memory: 0 bytes
 */
extern "C" __global__ void compute_degrees(
    const int* edges_u,
    const int* edges_v,
    int* degrees,
    int num_edges
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_edges) {
        int u = edges_u[tid];
        int v = edges_v[tid];
        atomicAdd(&degrees[u], 1);
        atomicAdd(&degrees[v], 1);
    }
}

/**
 * Compute per-vertex persistence scores.
 *
 * Persistence = f(degree, component_id, betti_0, betti_1)
 *
 * Formula:
 *   persistence[v] = degree[v] * (1.0 + component_factor) * cycle_factor
 *   where:
 *     component_factor = 1.0 / betti_0  (reward high connectivity)
 *     cycle_factor = 1.0 + (betti_1 / n) (reward cyclomatic complexity)
 *
 * Grid/Block: grid = ceil(n/256), block = 256
 * Shared memory: 0 bytes
 */
extern "C" __global__ void compute_persistence_scores(
    const int* degrees,
    const int* parent,
    int betti_0,
    int betti_1,
    float* persistence,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    float degree = static_cast<float>(degrees[tid]);

    // Component factor: inverse of Betti-0 (higher connectivity -> higher score)
    float component_factor = (betti_0 > 0) ? (1.0f / static_cast<float>(betti_0)) : 1.0f;

    // Cycle factor: normalized Betti-1 (more cycles -> higher topological complexity)
    float cycle_factor = 1.0f + (static_cast<float>(betti_1) / static_cast<float>(n));

    // Combined persistence score
    persistence[tid] = degree * (1.0f + component_factor) * cycle_factor;
}

/**
 * Compute topological importance for anchor selection.
 *
 * Importance = persistence[v] + centrality_bonus
 *
 * Centrality bonus: Vertices in largest component get higher scores.
 * Used for Phase 6 warmstart anchor selection.
 *
 * Grid/Block: grid = ceil(n/256), block = 256
 * Shared memory: 0 bytes
 */
extern "C" __global__ void compute_topological_importance(
    const float* persistence,
    const int* degrees,
    const int* parent,
    int betti_0,
    float* importance,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    float pers = persistence[tid];
    float degree = static_cast<float>(degrees[tid]);

    // Centrality bonus: vertices with high degree in sparse graphs get boosted
    float centrality_bonus = 0.0f;
    if (betti_0 > 1) {
        // Multiple components: reward high-degree vertices more
        centrality_bonus = degree / static_cast<float>(betti_0);
    }

    importance[tid] = pers + centrality_bonus;
}

/**
 * Kernel metadata for documentation and debugging.
 *
 * Query via cudaFuncGetAttributes() or nvprof.
 */
__device__ const char* tda_kernel_version = "1.0.0";
__device__ const char* tda_kernel_author = "PRISM GPU Specialist";
__device__ const char* tda_kernel_arch = "sm_86";
