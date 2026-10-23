// [RT-KNN] OptiX RT-Core Accelerated k-Nearest Neighbor Distances
//
// Hardware-accelerated k-NN distance computation using RTX RT cores.
// Used for adaptive epsilon selection in clustering.
//
// Architecture:
// 1. Build BVH from all positions (spheres)
// 2. For each query point, trace rays to find neighbors within search radius
// 3. Collect distances, sort on GPU
// 4. Return k-th nearest distance for each query point
//
// Target: NVIDIA RTX 5080 (84 RT cores, sm_120)

#include <optix.h>
#include <cuda_runtime.h>

// ═══════════════════════════════════════════════════════════════════════════════
// Launch Parameters
// ═══════════════════════════════════════════════════════════════════════════════

struct RtKnnParams {
    // BVH for all positions
    OptixTraversableHandle traversable;

    // Input: All positions (for BVH)
    float3* all_positions;          // [num_total] all spike positions
    unsigned int num_total;

    // Input: Query positions (subset to find k-NN for)
    unsigned int* query_indices;    // [num_queries] indices into all_positions
    unsigned int num_queries;

    // k-NN parameters
    unsigned int k;                 // Number of neighbors to find
    float search_radius;            // Maximum search radius (Å)
    unsigned int rays_per_query;    // Rays to cast per query point

    // Output: Distances
    float* distances;               // [num_queries * max_neighbors] candidate distances
    unsigned int* neighbor_counts;  // [num_queries] actual neighbor count per query
    float* kth_distances;           // [num_queries] final k-th nearest distance

    // Working memory
    unsigned int max_neighbors;     // Max neighbors to store per query
};

extern "C" {
    __constant__ RtKnnParams knn_params;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Utility: Sphere Direction Generation (Fibonacci lattice)
// ═══════════════════════════════════════════════════════════════════════════════

__device__ __forceinline__ float3 fibonacci_sphere_direction_knn(unsigned int idx, unsigned int total) {
    const float golden_ratio = 1.618033988749895f;
    const float pi = 3.14159265358979f;

    float theta = 2.0f * pi * idx / golden_ratio;
    float phi = acosf(1.0f - 2.0f * (idx + 0.5f) / total);

    float sin_phi = sinf(phi);
    return make_float3(
        sin_phi * cosf(theta),
        sin_phi * sinf(theta),
        cosf(phi)
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// RAYGEN: Find Neighbors and Store Distances
// ═══════════════════════════════════════════════════════════════════════════════

extern "C" __global__ void __raygen__knn_find_neighbors() {
    const uint3 idx = optixGetLaunchIndex();
    unsigned int query_id = idx.x;
    unsigned int ray_id = idx.y;

    if (query_id >= knn_params.num_queries) return;
    if (ray_id >= knn_params.rays_per_query) return;

    // Get query position from index
    unsigned int pos_idx = knn_params.query_indices[query_id];
    float3 origin = knn_params.all_positions[pos_idx];

    // Generate ray direction
    float3 direction = fibonacci_sphere_direction_knn(ray_id, knn_params.rays_per_query);

    // Trace ray with search_radius as max distance
    // Payload: p0 = query_id, p1 = pos_idx (for self-exclusion)
    unsigned int p0 = query_id;
    unsigned int p1 = pos_idx;

    optixTrace(
        knn_params.traversable,
        origin,
        direction,
        0.001f,                         // tmin (avoid self-intersection)
        knn_params.search_radius,       // tmax = search radius
        0.0f,                           // rayTime
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,            // Want all hits, not just closest
        0,                              // SBT offset
        1,                              // SBT stride
        0,                              // missSBTIndex
        p0, p1
    );
}

extern "C" __global__ void __closesthit__knn_find_neighbors() {
    unsigned int hit_idx = optixGetPrimitiveIndex();
    unsigned int query_id = optixGetPayload_0();
    unsigned int self_idx = optixGetPayload_1();

    // Skip self-hit
    if (hit_idx == self_idx) return;

    // Get distance from ray t-value
    float t = optixGetRayTmax();  // Distance to hit

    // Actually we need the actual intersection distance
    // The sphere intersection gives us the t-value
    float distance = t;

    // Atomically claim a slot for this distance
    unsigned int slot = atomicAdd(&knn_params.neighbor_counts[query_id], 1);

    // Store distance if we have room
    if (slot < knn_params.max_neighbors) {
        unsigned int dist_idx = query_id * knn_params.max_neighbors + slot;
        knn_params.distances[dist_idx] = distance;
    }
}

extern "C" __global__ void __miss__knn_find_neighbors() {
    // No action needed for miss
}

// ═══════════════════════════════════════════════════════════════════════════════
// CUDA KERNEL: Sort distances and extract k-th
// ═══════════════════════════════════════════════════════════════════════════════

// Simple insertion sort for small k (k < 32)
__device__ void insertion_sort(float* arr, int n) {
    for (int i = 1; i < n; i++) {
        float key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

// Partial sort: find k smallest elements (more efficient for large neighbor counts)
__device__ void partial_sort_k(float* arr, int n, int k) {
    // Use a simple approach: sort first k, then insert-sort remaining
    // For small k, this is O(n*k) which beats O(n log n) for small k

    if (n <= k) {
        insertion_sort(arr, n);
        return;
    }

    // Sort first k elements
    insertion_sort(arr, k);

    // For remaining elements, insert if smaller than arr[k-1]
    for (int i = k; i < n; i++) {
        if (arr[i] < arr[k-1]) {
            // Insert arr[i] into sorted portion
            float val = arr[i];
            int j = k - 2;
            while (j >= 0 && arr[j] > val) {
                arr[j + 1] = arr[j];
                j--;
            }
            arr[j + 1] = val;
        }
    }
}

extern "C" __global__ void extract_kth_distance(
    float* distances,           // [num_queries * max_neighbors]
    const unsigned int* neighbor_counts,
    float* kth_distances,       // [num_queries] output
    unsigned int num_queries,
    unsigned int max_neighbors,
    unsigned int k
) {
    unsigned int query_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_id >= num_queries) return;

    unsigned int count = neighbor_counts[query_id];

    // If we don't have k neighbors, return a large value
    if (count < k) {
        kth_distances[query_id] = 1e10f;  // Sentinel for "not enough neighbors"
        return;
    }

    // Copy distances to local array for sorting
    // Use shared memory for better performance
    float* query_distances = &distances[query_id * max_neighbors];

    // Limit to actual count
    int n = min(count, max_neighbors);

    // Partial sort to find k-th smallest
    partial_sort_k(query_distances, n, k);

    // k-th distance is at index k-1 (0-indexed)
    kth_distances[query_id] = query_distances[k - 1];
}

// ═══════════════════════════════════════════════════════════════════════════════
// Alternative: Bitonic sort for power-of-2 sizes (more parallel)
// ═══════════════════════════════════════════════════════════════════════════════

__device__ void bitonic_sort_step(float* arr, int i, int j, bool ascending) {
    if (ascending == (arr[i] > arr[j])) {
        float tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

// Warp-level bitonic sort for up to 32 elements
__device__ void warp_bitonic_sort(float* arr, int n) {
    // Pad to power of 2
    int size = 1;
    while (size < n) size <<= 1;

    for (int k = 2; k <= size; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = 0; i < size; i++) {
                int ij = i ^ j;
                if (ij > i && i < n && ij < n) {
                    bool ascending = ((i & k) == 0);
                    bitonic_sort_step(arr, i, ij, ascending);
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Compute percentiles from k-th distances
// ═══════════════════════════════════════════════════════════════════════════════

extern "C" __global__ void compute_percentiles(
    const float* kth_distances,
    float* percentiles,         // [4] output: p25, p50, p75, p90
    unsigned int num_queries
) {
    // Single thread computes percentiles
    // (Could parallelize with parallel reduction, but num_queries is small ~1000)
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Copy to local array for sorting
    // Note: For production, use thrust::sort or CUB
    extern __shared__ float shared_distances[];

    int valid_count = 0;
    for (unsigned int i = 0; i < num_queries && i < 2048; i++) {
        float d = kth_distances[i];
        if (d < 1e9f) {  // Valid distance
            shared_distances[valid_count++] = d;
        }
    }

    if (valid_count < 4) {
        // Not enough data, use defaults
        percentiles[0] = 5.0f;
        percentiles[1] = 7.0f;
        percentiles[2] = 10.0f;
        percentiles[3] = 14.0f;
        return;
    }

    // Sort distances
    insertion_sort(shared_distances, valid_count);

    // Extract percentiles with clamping
    int p25_idx = valid_count / 4;
    int p50_idx = valid_count / 2;
    int p75_idx = 3 * valid_count / 4;
    int p90_idx = 9 * valid_count / 10;

    // Clamp to reasonable epsilon ranges
    percentiles[0] = fminf(fmaxf(shared_distances[p25_idx], 3.0f), 8.0f);
    percentiles[1] = fminf(fmaxf(shared_distances[p50_idx], 5.0f), 12.0f);
    percentiles[2] = fminf(fmaxf(shared_distances[p75_idx], 7.0f), 18.0f);
    percentiles[3] = fminf(fmaxf(shared_distances[p90_idx], 10.0f), 25.0f);
}
