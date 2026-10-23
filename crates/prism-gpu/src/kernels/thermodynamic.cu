/**
 * @file thermodynamic.cu
 * @brief GPU-accelerated parallel tempering simulated annealing for graph coloring.
 *
 * Implements §4.2 (Phase 2: Thermodynamic) of the PRISM GPU Plan.
 *
 * Key features:
 * - Parallel tempering across multiple temperature schedules
 * - Metropolis-Hastings acceptance criterion
 * - Replica exchange for enhanced sampling
 * - CSR graph format for memory efficiency
 *
 * KNOWN ISSUE: Race conditions in parallel vertex updates (lines 143-146).
 * All threads modify replica_colors simultaneously without synchronization, causing:
 *   - Data races when reading neighbor colors during updates
 *   - Non-deterministic convergence behavior
 *   - Persistent conflicts even with high iteration counts
 *
 * QUICK FIX: Added device.synchronize() between kernel iterations (thermodynamic.rs:225)
 * PROPER FIX: Requires independent set scheduling or sequential Gibbs sampling rewrite
 *
 * Compilation:
 *   nvcc --ptx -o thermodynamic.ptx thermodynamic.cu -arch=sm_86 --use_fast_math -O3
 */

#include <curand_kernel.h>

/**
 * @brief Count conflicts for a given vertex coloring.
 *
 * @param row_ptr CSR row pointer array
 * @param col_idx CSR column index array
 * @param colors Vertex color assignments
 * @param num_vertices Number of vertices in graph
 * @return Number of conflicting edges
 */
__device__ unsigned int count_conflicts(
    const unsigned int* row_ptr,
    const unsigned int* col_idx,
    const unsigned int* colors,
    unsigned int num_vertices
) {
    unsigned int conflicts = 0;

    for (unsigned int v = 0; v < num_vertices; ++v) {
        unsigned int start = row_ptr[v];
        unsigned int end = row_ptr[v + 1];
        unsigned int color_v = colors[v];

        for (unsigned int e = start; e < end; ++e) {
            unsigned int neighbor = col_idx[e];
            if (colors[neighbor] == color_v) {
                conflicts++;
            }
        }
    }

    // Each edge counted twice, divide by 2
    return conflicts / 2;
}

/**
 * @brief Parallel tempering simulated annealing step with independent set scheduling.
 *
 * Each thread processes one vertex in one temperature replica.
 * Only processes vertices belonging to the current independent set to avoid race conditions.
 * Proposes color changes and accepts/rejects via Metropolis criterion.
 *
 * @param row_ptr CSR row pointer array (size: num_vertices + 1)
 * @param col_idx CSR column index array (size: num_edges)
 * @param replica_colors Color assignments for all replicas (size: num_replicas * num_vertices)
 * @param temperatures Temperature schedule for each replica (size: num_replicas)
 * @param conflicts Output conflict counts per replica (size: num_replicas)
 * @param independent_sets Independent set assignment per vertex (size: num_vertices)
 * @param num_vertices Number of vertices
 * @param num_edges Number of edges
 * @param num_replicas Number of temperature replicas
 * @param iteration Current iteration (for RNG seed)
 * @param current_set_id Only process vertices in this independent set
 */
extern "C" __global__ void parallel_tempering_step(
    const unsigned int* row_ptr,
    const unsigned int* col_idx,
    unsigned int* replica_colors,
    const float* temperatures,
    unsigned int* conflicts,
    const unsigned int* independent_sets,
    unsigned int num_vertices,
    unsigned int num_edges,
    unsigned int num_replicas,
    unsigned int iteration,
    unsigned int current_set_id
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_threads = num_replicas * num_vertices;

    if (tid >= total_threads) return;

    // Decompose thread ID into replica and vertex
    unsigned int replica = tid / num_vertices;
    unsigned int vertex = tid % num_vertices;

    // Only process vertices in the current independent set (race-free guarantee)
    if (independent_sets[vertex] != current_set_id) return;

    // Initialize RNG for this thread
    curandState_t state;
    curand_init(iteration * total_threads + tid, 0, 0, &state);

    // Get current color and temperature
    unsigned int offset = replica * num_vertices;
    unsigned int current_color = replica_colors[offset + vertex];
    float temperature = temperatures[replica];

    // Find maximum color in use (to limit color space)
    unsigned int max_color = 1;
    for (unsigned int v = 0; v < num_vertices; ++v) {
        unsigned int c = replica_colors[offset + v];
        if (c > max_color) max_color = c;
    }

    // Propose new color (random selection from 1 to max_color + 1)
    unsigned int new_color = (curand(&state) % (max_color + 1)) + 1;

    // Skip if same color
    if (new_color == current_color) return;

    // Calculate delta conflicts for this vertex
    int delta_conflicts = 0;
    unsigned int start = row_ptr[vertex];
    unsigned int end = row_ptr[vertex + 1];

    for (unsigned int e = start; e < end; ++e) {
        unsigned int neighbor = col_idx[e];
        unsigned int neighbor_color = replica_colors[offset + neighbor];

        // Old conflicts
        if (neighbor_color == current_color) {
            delta_conflicts--;
        }

        // New conflicts
        if (neighbor_color == new_color) {
            delta_conflicts++;
        }
    }

    // CHEMICAL POTENTIAL: Add penalty for using higher color indices
    // This creates pressure to compress the coloring
    float chemical_potential = 0.75f;  // BALANCED - aligned with quantum phase (was 0.75)
    float color_penalty = chemical_potential * ((float)new_color - (float)current_color);

    // Total energy change includes both conflict and color index changes
    float delta_energy = delta_conflicts + color_penalty;

    // Metropolis-Hastings acceptance with chemical potential
    bool accept = false;
    if (delta_energy <= 0) {
        // Always accept improvements (lower conflicts OR lower colors)
        accept = true;
    } else {
        // Accept with probability exp(-delta_E / T)
        float acceptance_prob = expf(-delta_energy / temperature);
        float rand_val = curand_uniform(&state);
        accept = (rand_val < acceptance_prob);
    }

    // Apply move if accepted
    if (accept) {
        replica_colors[offset + vertex] = new_color;
    }

    // Update conflict count (one thread per replica does this)
    if (vertex == 0) {
        conflicts[replica] = count_conflicts(row_ptr, col_idx, &replica_colors[offset], num_vertices);
    }
}

/**
 * @brief Attempt replica swaps between adjacent temperature levels.
 *
 * Implements parallel tempering exchange moves to enhance sampling.
 * Each thread handles one pair of adjacent replicas.
 *
 * @param replica_colors Color assignments for all replicas (size: num_replicas * num_vertices)
 * @param temperatures Temperature schedule (size: num_replicas)
 * @param conflicts Conflict counts per replica (size: num_replicas)
 * @param num_vertices Number of vertices
 * @param num_replicas Number of temperature replicas
 */
extern "C" __global__ void replica_swap(
    unsigned int* replica_colors,
    const float* temperatures,
    unsigned int* conflicts,
    unsigned int num_vertices,
    unsigned int num_replicas
) {
    unsigned int pair_idx = threadIdx.x;

    // Each thread handles swapping replicas (2*pair_idx) and (2*pair_idx + 1)
    unsigned int r1 = 2 * pair_idx;
    unsigned int r2 = r1 + 1;

    if (r2 >= num_replicas) return;

    // Get temperatures and energies (conflicts)
    float T1 = temperatures[r1];
    float T2 = temperatures[r2];
    unsigned int E1 = conflicts[r1];
    unsigned int E2 = conflicts[r2];

    // Compute swap acceptance probability
    // P_swap = min(1, exp(-(1/T1 - 1/T2) * (E2 - E1)))
    float beta_diff = (1.0f / T1) - (1.0f / T2);
    float energy_diff = (float)(E2 - E1);
    float log_prob = -beta_diff * energy_diff;

    // Initialize RNG
    curandState_t state;
    curand_init(r1, 0, 0, &state);
    float rand_val = curand_uniform(&state);

    bool accept = (log_prob >= 0.0f) || (rand_val < expf(log_prob));

    // Perform swap if accepted
    if (accept) {
        // Swap colors
        unsigned int offset1 = r1 * num_vertices;
        unsigned int offset2 = r2 * num_vertices;

        for (unsigned int v = 0; v < num_vertices; ++v) {
            unsigned int temp_color = replica_colors[offset1 + v];
            replica_colors[offset1 + v] = replica_colors[offset2 + v];
            replica_colors[offset2 + v] = temp_color;
        }

        // Swap conflict counts
        unsigned int temp_conflicts = conflicts[r1];
        conflicts[r1] = conflicts[r2];
        conflicts[r2] = temp_conflicts;
    }
}

/**
 * @brief Color compaction kernel - renumber colors to minimize gaps.
 *
 * Given a valid coloring, renumber colors from 1..k where k is the chromatic number.
 * This is a post-processing step to improve solution quality.
 *
 * @param colors Input/output color array (size: num_vertices)
 * @param num_vertices Number of vertices
 * @param color_map Mapping from old colors to new colors (size: max_colors)
 * @param max_colors Maximum color value in use
 */
extern "C" __global__ void compact_colors(
    unsigned int* colors,
    unsigned int num_vertices,
    const unsigned int* color_map,
    unsigned int max_colors
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= num_vertices) return;

    unsigned int old_color = colors[tid];
    if (old_color <= max_colors && old_color > 0) {
        colors[tid] = color_map[old_color - 1];
    }
}
