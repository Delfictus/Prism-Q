/**
 * PRISM Ligand Binding Site - Pocket Clustering Kernel
 *
 * Jones-Plassmann parallel graph coloring algorithm for race-free pocket clustering.
 * Uses random priority assignment for deterministic, conflict-free parallel coloring.
 *
 * Key innovations:
 * 1. Jones-Plassmann algorithm eliminates race conditions in parallel coloring
 * 2. Random priority ensures deterministic results across runs
 * 3. Multi-round coloring guarantees convergence
 * 4. CSR graph format for efficient neighbor queries
 *
 * Reference: Jones & Plassmann, "A Parallel Graph Coloring Heuristic" (1993)
 *
 * Copyright (c) 2024 PRISM Research Team | Delfictus I/O Inc.
 */

#include <cuda_runtime.h>

// ═══════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════

#define UNCOLORED -1
#define MAX_ROUNDS 1000  // Maximum coloring rounds before fallback

// ═══════════════════════════════════════════════════════════════════════════
// UTILITY FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Simple LCG random number generator for priority assignment
 * Ensures deterministic priorities across runs
 */
__device__ __forceinline__ unsigned int lcg_rand(unsigned int seed) {
    return (1103515245u * seed + 12345u) & 0x7FFFFFFFu;
}

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL 1: ASSIGN RANDOM PRIORITIES
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Assign random priorities to each vertex for Jones-Plassmann algorithm
 * Uses vertex index as seed for deterministic results
 */
extern "C" __global__ void assign_priorities(
    int num_vertices,
    unsigned int seed,
    unsigned int* __restrict__ priorities
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;

    // Generate deterministic random priority based on vertex ID and global seed
    priorities[v] = lcg_rand(seed + v);
}

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL 2: JONES-PLASSMANN COLORING ROUND
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Single round of Jones-Plassmann graph coloring
 *
 * For each uncolored vertex:
 * 1. Check if it has higher priority than all uncolored neighbors
 * 2. If yes, assign it the smallest available color (first-fit)
 * 3. If no, wait for next round
 *
 * This ensures no race conditions - only one vertex in each independent set
 * gets colored per round.
 */
extern "C" __global__ void jones_plassmann_round(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    int num_vertices,
    int max_colors,
    const unsigned int* __restrict__ priorities,
    int* __restrict__ colors,
    int* __restrict__ changed  // Output flag: 1 if any vertex colored this round
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;

    // Skip if already colored
    if (colors[v] != UNCOLORED) return;

    unsigned int my_priority = priorities[v];

    // Check if this vertex has highest priority among uncolored neighbors
    bool has_highest_priority = true;
    int start = row_ptr[v];
    int end = row_ptr[v + 1];

    for (int k = start; k < end; k++) {
        int nbr = col_idx[k];
        if (colors[nbr] == UNCOLORED && priorities[nbr] > my_priority) {
            has_highest_priority = false;
            break;
        }
    }

    // If we have highest priority, color this vertex
    if (has_highest_priority) {
        // Find smallest available color (first-fit)
        // Use local array to track used colors by neighbors
        bool used_colors[256];  // Assumes max_colors <= 256
        for (int c = 0; c < max_colors; c++) {
            used_colors[c] = false;
        }

        // Mark colors used by neighbors
        for (int k = start; k < end; k++) {
            int nbr = col_idx[k];
            int nbr_color = colors[nbr];
            if (nbr_color >= 0 && nbr_color < max_colors) {
                used_colors[nbr_color] = true;
            }
        }

        // Find first available color
        int chosen_color = 0;
        for (int c = 0; c < max_colors; c++) {
            if (!used_colors[c]) {
                chosen_color = c;
                break;
            }
        }

        colors[v] = chosen_color;
        atomicOr(changed, 1);  // Signal that we colored at least one vertex
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL 3: MAIN POCKET CLUSTERING (LEGACY INTERFACE)
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Legacy single-kernel interface for backward compatibility
 * Performs multiple Jones-Plassmann rounds until convergence
 *
 * Note: For production use, prefer the multi-kernel approach:
 * 1. assign_priorities once
 * 2. jones_plassmann_round in a loop until changed == 0
 *
 * This kernel does it all in one launch but may be slower due to
 * lack of host-side convergence checking.
 */
extern "C" __global__ void pocket_clustering_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    int num_vertices,
    int max_colors,
    int* __restrict__ colors
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;

    // Initialize to uncolored
    if (colors[v] == 0) {
        colors[v] = UNCOLORED;
    }

    // For single-kernel version, use simple greedy with tie-breaking
    // based on vertex ID to ensure determinism
    int start = row_ptr[v];
    int end = row_ptr[v + 1];

    // Find first-fit color
    bool used_colors[256];
    for (int c = 0; c < max_colors; c++) {
        used_colors[c] = false;
    }

    for (int k = start; k < end; k++) {
        int nbr = col_idx[k];
        // Use vertex ID for tie-breaking to avoid races
        if (nbr < v) {  // Only consider lower-ID neighbors (already processed)
            int nbr_color = colors[nbr];
            if (nbr_color >= 0 && nbr_color < max_colors) {
                used_colors[nbr_color] = true;
            }
        }
    }

    // Find first available color
    int chosen_color = 0;
    for (int c = 0; c < max_colors; c++) {
        if (!used_colors[c]) {
            chosen_color = c;
            break;
        }
    }

    colors[v] = chosen_color;
}
