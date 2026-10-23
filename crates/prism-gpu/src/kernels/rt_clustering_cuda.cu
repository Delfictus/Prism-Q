// RT Clustering CUDA Kernels v2 - Union-Find on Flat Neighbor Buffer
//
// This file contains regular CUDA kernels for the clustering pipeline.
// The OptiX raygen/anyhit shaders are in rt_clustering.cu (compiled to .optixir)
//
// v2 Pipeline (single-pass):
// 1. find neighbors (single OptiX launch with anyhit)
// 2. init_union_find
// 3. union_neighbors_flat (works on fixed-size buffer, no CSR)
// 4. flatten_clusters_full (path compression)
// 5. propagate_cluster_ids
//
// Removed from v1: compute_neighbor_offsets (no longer needed — no CSR)

extern "C" {

// ============================================================================
// Phase 1: Initialize union-find parent array
// ============================================================================

__global__ void init_union_find(
    int* __restrict__ parent,
    unsigned int num_events
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_events) return;
    parent[idx] = idx;
}

// ============================================================================
// Device function: Find root with path splitting
// ============================================================================

__device__ int find_root(int* parent, int i) {
    while (parent[i] != i) {
        int p = parent[i];
        int gp = parent[p];
        parent[i] = gp;  // Path splitting: point to grandparent
        i = gp;
    }
    return i;
}

// ============================================================================
// Phase 2: Union neighbors from FLAT buffer (new v2 kernel)
//
// Works with fixed-size per-event neighbor buffer:
//   neighbor_list[event * max_neighbors + i] = neighbor_event_id
//   neighbor_count[event] = actual number of neighbors found
//
// No CSR offsets needed — direct index arithmetic.
// ============================================================================

__global__ void union_neighbors_flat(
    int* __restrict__ parent,
    const unsigned int* __restrict__ neighbor_list,
    const unsigned int* __restrict__ neighbor_count,
    unsigned int num_events,
    unsigned int max_neighbors
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_events) return;

    unsigned int count = neighbor_count[idx];
    if (count == 0) return;
    if (count > max_neighbors) count = max_neighbors;  // Cap at buffer size

    unsigned int base = idx * max_neighbors;

    for (unsigned int i = 0; i < count; i++) {
        unsigned int neighbor = neighbor_list[base + i];
        if (neighbor >= num_events) continue;  // Safety: skip invalid indices

        int root_a = find_root(parent, idx);
        int root_b = find_root(parent, neighbor);

        if (root_a != root_b) {
            // Union by min: always point larger root to smaller
            int min_root = min(root_a, root_b);
            int max_root = max(root_a, root_b);
            atomicCAS(&parent[max_root], max_root, min_root);
        }
    }
}

// ============================================================================
// Phase 3: Flatten clusters (full path compression)
// ============================================================================

__global__ void flatten_clusters_full(
    int* __restrict__ parent,
    unsigned int num_events
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_events) return;

    // Chase pointers to root
    int root = idx;
    int prev = root;
    while (parent[root] != root) {
        prev = root;
        root = parent[root];
        if (root < 0 || root >= (int)num_events) {
            root = prev;  // Safety check
            break;
        }
    }

    // Point directly to root (full path compression)
    parent[idx] = root;
}

// ============================================================================
// Phase 4: Propagate final cluster IDs
// ============================================================================

__global__ void propagate_cluster_ids(
    const int* __restrict__ parent,
    int* __restrict__ cluster_ids,
    unsigned int num_events,
    unsigned int* __restrict__ num_clusters
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_events) return;

    int root = parent[idx];
    cluster_ids[idx] = root;

    // Count unique roots (only roots count themselves)
    if (root == (int)idx) {
        atomicAdd(num_clusters, 1);
    }
}

// ============================================================================
// Post-processing: Count cluster sizes and filter small clusters
// ============================================================================

__global__ void count_cluster_sizes(
    const int* __restrict__ cluster_ids,
    unsigned int* __restrict__ cluster_sizes,
    unsigned int num_events,
    unsigned int max_clusters
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_events) return;

    int cluster = cluster_ids[idx];
    if (cluster >= 0 && (unsigned int)cluster < max_clusters) {
        atomicAdd(&cluster_sizes[cluster], 1);
    }
}

__global__ void filter_small_clusters(
    int* __restrict__ cluster_ids,
    const unsigned int* __restrict__ cluster_sizes,
    unsigned int num_events,
    unsigned int min_cluster_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_events) return;

    int cluster = cluster_ids[idx];
    if (cluster >= 0) {
        if (cluster_sizes[cluster] < min_cluster_size) {
            cluster_ids[idx] = -1;
        }
    }
}

// ============================================================================
// LEGACY KERNELS (kept for backward compatibility)
// ============================================================================

// Sequential prefix sum (v1 CSR pipeline)
__global__ void compute_neighbor_offsets(
    const unsigned int* __restrict__ neighbor_counts,
    unsigned int* __restrict__ neighbor_offsets,
    unsigned int num_events,
    unsigned int* __restrict__ total_neighbors
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    unsigned int offset = 0;
    for (unsigned int i = 0; i < num_events; i++) {
        neighbor_offsets[i] = offset;
        offset += neighbor_counts[i];
    }
    *total_neighbors = offset;
}

// CSR-based union neighbors (v1 pipeline)
__global__ void union_neighbors(
    int* __restrict__ parent,
    const unsigned int* __restrict__ neighbor_counts,
    const unsigned int* __restrict__ neighbor_offsets,
    const unsigned int* __restrict__ neighbor_indices,
    unsigned int num_events
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_events) return;
    unsigned int count = neighbor_counts[idx];
    if (count == 0) return;
    unsigned int offset = neighbor_offsets[idx];
    for (unsigned int i = 0; i < count; i++) {
        unsigned int neighbor_idx = neighbor_indices[offset + i];
        if (neighbor_idx >= num_events) continue;
        int root_a = find_root(parent, idx);
        int root_b = find_root(parent, neighbor_idx);
        if (root_a != root_b) {
            int min_root = min(root_a, root_b);
            int max_root = max(root_a, root_b);
            atomicCAS(&parent[max_root], max_root, min_root);
        }
    }
}

// Pointer-jumping flatten (v1)
__global__ void flatten_clusters(
    int* __restrict__ parent,
    unsigned int num_events
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_events) return;
    int p = parent[idx];
    if (p != (int)idx && p >= 0) {
        int gp = parent[p];
        if (gp != p && gp >= 0) {
            parent[idx] = gp;
        }
    }
}

} // extern "C"
