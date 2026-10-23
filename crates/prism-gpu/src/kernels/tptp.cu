/**
 * TPTP: Topological Phase Transition Prediction
 *
 * Live persistent homology computation for detecting phase boundaries
 * in the optimization landscape.
 *
 * Copyright (c) 2024 PRISM Research Team | Delfictus I/O Inc.
 * Los Angeles, CA 90013
 * All Rights Reserved.
 */

#include <cuda_runtime.h>
#include <math_constants.h>
#include "runtime_config.cuh"

#define BLOCK_SIZE 256
#define MAX_SIMPLICES 4096
#define MAX_PERSISTENCE_PAIRS 1024
#define MAX_FILTRATION_STEPS 64
#define MAX_VERTICES_LOCAL 512

// Simplex types
#define SIMPLEX_VERTEX 0
#define SIMPLEX_EDGE 1
#define SIMPLEX_TRIANGLE 2

/**
 * Simplex structure for persistent homology
 */
struct Simplex {
    int type;               // 0=vertex, 1=edge, 2=triangle
    int vertices[3];        // Vertex indices (up to 3 for triangle)
    float filtration_value; // When simplex appears in filtration
    int boundary[3];        // Indices of boundary simplices
    int num_boundary;       // Number of boundary simplices
    int paired_with;        // Index of destroying/creating simplex (-1 if unpaired)
    float birth_time;       // Birth time in filtration
    float death_time;       // Death time in filtration
};

/**
 * Persistence pair (birth-death pair)
 */
struct PersistencePair {
    float birth;
    float death;
    int dimension;          // 0, 1, or 2
    int creator_simplex;    // Simplex index that creates this feature
    int destroyer_simplex;  // Simplex index that destroys this feature
    float persistence;      // death - birth
};

/**
 * TPTP State - Persistent homology state
 */
struct TPTPState {
    // Simplicial complex
    Simplex simplices[MAX_SIMPLICES];
    int num_simplices;

    // Persistence pairs
    PersistencePair pairs[MAX_PERSISTENCE_PAIRS];
    int num_pairs;

    // Betti numbers (β0, β1, β2)
    float betti[3];

    // Historical Betti numbers for derivative computation
    float betti_history[MAX_FILTRATION_STEPS][3];
    int history_idx;
    int history_filled;

    // Persistence statistics
    float max_persistence;
    float avg_persistence;
    float persistence_entropy;
    float total_persistence;

    // Phase transition detection
    int phase_transition_detected;
    float transition_strength;
    float betti_0_derivative;
    float betti_1_derivative;
    float betti_2_derivative;

    // Stability metrics
    float stability_score;
    int stable_iterations;
    int unstable_iterations;

    // Filtration parameters
    float current_filtration;
    float filtration_step;
    int filtration_index;
};

// ═══════════════════════════════════════════════════════════════════════════
// DEVICE HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Compare two simplices by filtration value (for sorting)
 */
__device__ int compare_simplices(const Simplex* a, const Simplex* b) {
    if (a->filtration_value < b->filtration_value) return -1;
    if (a->filtration_value > b->filtration_value) return 1;

    // Tie-break by dimension (lower dimension first)
    if (a->type < b->type) return -1;
    if (a->type > b->type) return 1;

    return 0;
}

/**
 * Check if two vertices are connected in the graph
 */
__device__ bool are_connected(
    int v1, int v2,
    const int* graph_row_ptr,
    const int* graph_col_idx
) {
    int start = graph_row_ptr[v1];
    int end = graph_row_ptr[v1 + 1];

    for (int e = start; e < end; e++) {
        if (graph_col_idx[e] == v2) {
            return true;
        }
    }

    return false;
}

/**
 * Check if three vertices form a triangle in the graph
 */
__device__ bool forms_triangle(
    int v0, int v1, int v2,
    const int* graph_row_ptr,
    const int* graph_col_idx
) {
    return are_connected(v0, v1, graph_row_ptr, graph_col_idx) &&
           are_connected(v1, v2, graph_row_ptr, graph_col_idx) &&
           are_connected(v2, v0, graph_row_ptr, graph_col_idx);
}

/**
 * Compute entropy of persistence diagram
 */
__device__ float compute_persistence_entropy(
    const PersistencePair* pairs,
    int num_pairs
) {
    if (num_pairs == 0) return 0.0f;

    // Normalize persistence values to probabilities
    float total = 0.0f;
    for (int i = 0; i < num_pairs; i++) {
        total += pairs[i].persistence;
    }

    if (total < 1e-8f) return 0.0f;

    // Compute Shannon entropy
    float entropy = 0.0f;
    for (int i = 0; i < num_pairs; i++) {
        float p = pairs[i].persistence / total;
        if (p > 1e-8f) {
            entropy -= p * logf(p);
        }
    }

    return entropy;
}

/**
 * Merge two components in union-find structure
 */
__device__ void union_components(
    int* parent,
    int* rank,
    int v1, int v2
) {
    // Find roots with path compression
    int root1 = v1;
    while (parent[root1] != root1) {
        int next = parent[root1];
        parent[root1] = parent[next];
        root1 = next;
    }

    int root2 = v2;
    while (parent[root2] != root2) {
        int next = parent[root2];
        parent[root2] = parent[next];
        root2 = next;
    }

    if (root1 == root2) return;

    // Union by rank
    if (rank[root1] < rank[root2]) {
        parent[root1] = root2;
    } else if (rank[root1] > rank[root2]) {
        parent[root2] = root1;
    } else {
        parent[root2] = root1;
        rank[root1]++;
    }
}

/**
 * Find root of component in union-find
 */
__device__ int find_root(int* parent, int v) {
    int root = v;
    while (parent[root] != root) {
        int next = parent[root];
        parent[root] = parent[next];
        root = next;
    }
    return root;
}

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Build Vietoris-Rips complex from conflict graph
 *
 * This kernel constructs a simplicial complex where:
 * - 0-simplices (vertices) appear at filtration value = conflict_signal[v]
 * - 1-simplices (edges) appear when both endpoints are in the complex
 * - 2-simplices (triangles) appear when all edges are in the complex
 */
extern "C" __global__ void tptp_build_complex(
    const int* __restrict__ graph_row_ptr,
    const int* __restrict__ graph_col_idx,
    const float* __restrict__ conflict_signal,
    TPTPState* __restrict__ state,
    const RuntimeConfig* __restrict__ config,
    int num_vertices
) {
    __shared__ int simplex_count;
    __shared__ int edge_count;
    __shared__ int triangle_count;

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize counters
    if (tid == 0) {
        simplex_count = 0;
        edge_count = 0;
        triangle_count = 0;
    }
    __syncthreads();

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 1: Add vertices as 0-simplices
    // ═══════════════════════════════════════════════════════════════════

    if (gid < num_vertices) {
        int idx = atomicAdd(&simplex_count, 1);
        if (idx < MAX_SIMPLICES) {
            Simplex* s = &state->simplices[idx];
            s->type = SIMPLEX_VERTEX;
            s->vertices[0] = gid;
            s->vertices[1] = -1;
            s->vertices[2] = -1;
            s->filtration_value = conflict_signal[gid];
            s->num_boundary = 0;
            s->paired_with = -1;
            s->birth_time = conflict_signal[gid];
            s->death_time = CUDART_INF_F;
        }
    }
    __syncthreads();

    int vertex_count = simplex_count;

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 2: Add edges as 1-simplices
    // ═══════════════════════════════════════════════════════════════════

    if (gid < num_vertices) {
        int start = graph_row_ptr[gid];
        int end = graph_row_ptr[gid + 1];

        for (int e = start; e < end; e++) {
            int neighbor = graph_col_idx[e];

            // Avoid duplicates: only add edge if gid < neighbor
            if (neighbor > gid) {
                int idx = atomicAdd(&simplex_count, 1);
                if (idx < MAX_SIMPLICES) {
                    Simplex* s = &state->simplices[idx];
                    s->type = SIMPLEX_EDGE;
                    s->vertices[0] = gid;
                    s->vertices[1] = neighbor;
                    s->vertices[2] = -1;

                    // Filtration value is max of endpoint conflicts
                    float filt = fmaxf(conflict_signal[gid], conflict_signal[neighbor]);
                    s->filtration_value = filt;

                    // Boundary is the two vertices
                    s->num_boundary = 2;
                    s->boundary[0] = gid;          // Index in simplex array
                    s->boundary[1] = neighbor;
                    s->boundary[2] = -1;

                    s->paired_with = -1;
                    s->birth_time = filt;
                    s->death_time = CUDART_INF_F;

                    atomicAdd(&edge_count, 1);
                }
            }
        }
    }
    __syncthreads();

    int edge_start = vertex_count;
    int edge_end = simplex_count;

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 3: Add triangles as 2-simplices
    // ═══════════════════════════════════════════════════════════════════

    // For each edge, check if it forms triangles with other vertices
    for (int edge_idx = edge_start + tid; edge_idx < edge_end; edge_idx += blockDim.x) {
        if (edge_idx >= MAX_SIMPLICES) break;

        Simplex* edge = &state->simplices[edge_idx];
        int v0 = edge->vertices[0];
        int v1 = edge->vertices[1];

        // Check all neighbors of v0 to see if they complete a triangle
        int start = graph_row_ptr[v0];
        int end = graph_row_ptr[v0 + 1];

        for (int e = start; e < end; e++) {
            int v2 = graph_col_idx[e];

            // Only add if v2 > v1 to avoid duplicates
            if (v2 > v1 && forms_triangle(v0, v1, v2, graph_row_ptr, graph_col_idx)) {
                int idx = atomicAdd(&simplex_count, 1);
                if (idx < MAX_SIMPLICES) {
                    Simplex* s = &state->simplices[idx];
                    s->type = SIMPLEX_TRIANGLE;
                    s->vertices[0] = v0;
                    s->vertices[1] = v1;
                    s->vertices[2] = v2;

                    // Filtration value is max of vertex conflicts
                    float filt = fmaxf(fmaxf(conflict_signal[v0], conflict_signal[v1]),
                                       conflict_signal[v2]);
                    s->filtration_value = filt;

                    // Boundary is the three edges (simplified - just store vertices)
                    s->num_boundary = 3;
                    s->boundary[0] = v0;
                    s->boundary[1] = v1;
                    s->boundary[2] = v2;

                    s->paired_with = -1;
                    s->birth_time = filt;
                    s->death_time = CUDART_INF_F;

                    atomicAdd(&triangle_count, 1);
                }
            }
        }
    }
    __syncthreads();

    // Store final counts
    if (tid == 0) {
        state->num_simplices = min(simplex_count, MAX_SIMPLICES);
    }
}

/**
 * Compute persistent homology via simplified matrix reduction
 *
 * This implements a simplified version of the standard persistence algorithm.
 * Full production version would implement optimized matrix reduction.
 */
extern "C" __global__ void tptp_compute_homology(
    TPTPState* __restrict__ state,
    const RuntimeConfig* __restrict__ config
) {
    int tid = threadIdx.x;

    __shared__ int parent[MAX_VERTICES_LOCAL];
    __shared__ int rank[MAX_VERTICES_LOCAL];
    __shared__ float betti_local[3];

    // Initialize for union-find
    for (int i = tid; i < MAX_VERTICES_LOCAL; i += blockDim.x) {
        parent[i] = i;
        rank[i] = 0;
    }

    if (tid == 0) {
        betti_local[0] = 0.0f;
        betti_local[1] = 0.0f;
        betti_local[2] = 0.0f;
    }
    __syncthreads();

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 1: Compute β0 (connected components) via union-find
    // ═══════════════════════════════════════════════════════════════════

    if (tid == 0) {
        int num_vertices = 0;
        int num_edges = 0;

        // Count vertices
        for (int i = 0; i < state->num_simplices; i++) {
            if (state->simplices[i].type == SIMPLEX_VERTEX) {
                num_vertices++;
            }
        }

        // Process edges in filtration order to track component merging
        for (int i = 0; i < state->num_simplices; i++) {
            Simplex* s = &state->simplices[i];

            if (s->type == SIMPLEX_EDGE) {
                int v0 = s->vertices[0];
                int v1 = s->vertices[1];

                if (v0 < MAX_VERTICES_LOCAL && v1 < MAX_VERTICES_LOCAL) {
                    int root0 = find_root(parent, v0);
                    int root1 = find_root(parent, v1);

                    if (root0 != root1) {
                        // Merging components - this edge creates no cycle
                        union_components(parent, rank, v0, v1);
                    } else {
                        // Edge creates a cycle (1-dimensional feature)
                        num_edges++;
                    }
                }
            }
        }

        // Count connected components
        int components = 0;
        for (int i = 0; i < num_vertices && i < MAX_VERTICES_LOCAL; i++) {
            if (parent[i] == i) {
                components++;
            }
        }

        betti_local[0] = (float)components;
        betti_local[1] = (float)num_edges;
        betti_local[2] = 0.0f; // Simplified - would need full homology computation
    }
    __syncthreads();

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 2: Store Betti numbers and history
    // ═══════════════════════════════════════════════════════════════════

    if (tid == 0) {
        state->betti[0] = betti_local[0];
        state->betti[1] = betti_local[1];
        state->betti[2] = betti_local[2];

        // Store in history
        int idx = state->history_idx;
        state->betti_history[idx][0] = betti_local[0];
        state->betti_history[idx][1] = betti_local[1];
        state->betti_history[idx][2] = betti_local[2];

        state->history_idx = (idx + 1) % MAX_FILTRATION_STEPS;
        if (state->history_filled < MAX_FILTRATION_STEPS) {
            state->history_filled++;
        }
    }
    __syncthreads();

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 3: Compute persistence pairs (simplified)
    // ═══════════════════════════════════════════════════════════════════

    if (tid == 0) {
        int pair_count = 0;
        float total_pers = 0.0f;
        float max_pers = 0.0f;

        // Create persistence pairs from simplices
        for (int i = 0; i < state->num_simplices && pair_count < MAX_PERSISTENCE_PAIRS; i++) {
            Simplex* s = &state->simplices[i];

            if (s->type == SIMPLEX_EDGE && s->birth_time < CUDART_INF_F) {
                PersistencePair* pair = &state->pairs[pair_count++];
                pair->birth = s->birth_time;
                pair->death = s->death_time;
                pair->dimension = 1;
                pair->creator_simplex = i;
                pair->destroyer_simplex = -1;

                float pers = (s->death_time < CUDART_INF_F) ?
                            (s->death_time - s->birth_time) : 0.0f;
                pair->persistence = pers;

                total_pers += pers;
                if (pers > max_pers) max_pers = pers;
            }
        }

        state->num_pairs = pair_count;
        state->max_persistence = max_pers;
        state->avg_persistence = (pair_count > 0) ? (total_pers / pair_count) : 0.0f;
        state->total_persistence = total_pers;

        // Compute persistence entropy
        state->persistence_entropy = compute_persistence_entropy(
            state->pairs,
            state->num_pairs
        );
    }
}

/**
 * Detect phase transitions from Betti number dynamics
 *
 * Phase transitions manifest as:
 * 1. Sudden changes in Betti numbers
 * 2. Spikes in persistence
 * 3. Changes in topological stability
 */
extern "C" __global__ void tptp_detect_transition(
    TPTPState* __restrict__ state,
    const RuntimeConfig* __restrict__ config
) {
    int tid = threadIdx.x;

    if (tid == 0) {
        // Need at least 2 history entries to compute derivatives
        if (state->history_filled < 2) {
            state->phase_transition_detected = 0;
            state->transition_strength = 0.0f;
            return;
        }

        // ═══════════════════════════════════════════════════════════════
        // PHASE 1: Compute Betti number derivatives
        // ═══════════════════════════════════════════════════════════════

        int curr_idx = (state->history_idx - 1 + MAX_FILTRATION_STEPS) % MAX_FILTRATION_STEPS;
        int prev_idx = (state->history_idx - 2 + MAX_FILTRATION_STEPS) % MAX_FILTRATION_STEPS;

        float db0 = state->betti_history[curr_idx][0] - state->betti_history[prev_idx][0];
        float db1 = state->betti_history[curr_idx][1] - state->betti_history[prev_idx][1];
        float db2 = state->betti_history[curr_idx][2] - state->betti_history[prev_idx][2];

        state->betti_0_derivative = db0;
        state->betti_1_derivative = db1;
        state->betti_2_derivative = db2;

        // ═══════════════════════════════════════════════════════════════
        // PHASE 2: Compute transition score
        // ═══════════════════════════════════════════════════════════════

        float transition_score = 0.0f;

        // Indicator 1: Sudden drop in β0 (components merging rapidly)
        if (db0 < -config->betti_0_threshold) {
            transition_score += fabsf(db0) * 2.0f;
        }

        // Indicator 2: Spike in β1 (cycles forming/breaking)
        if (fabsf(db1) > config->betti_1_threshold) {
            transition_score += fabsf(db1) * 3.0f;
        }

        // Indicator 3: Change in β2 (voids appearing/disappearing)
        if (fabsf(db2) > config->betti_2_threshold) {
            transition_score += fabsf(db2) * 1.5f;
        }

        // Indicator 4: High maximum persistence
        if (state->max_persistence > config->persistence_threshold) {
            transition_score += state->max_persistence;
        }

        // Indicator 5: Sudden change in persistence entropy
        if (state->history_filled >= 2) {
            // Could compute entropy derivative here
            float entropy_weight = state->persistence_entropy * 0.1f;
            transition_score += entropy_weight;
        }

        // ═══════════════════════════════════════════════════════════════
        // PHASE 3: Update stability metrics
        // ═══════════════════════════════════════════════════════════════

        // Stability score based on variance of recent Betti numbers
        float variance = 0.0f;
        int window = min(state->history_filled, config->stability_window);

        if (window > 1) {
            float mean_b1 = 0.0f;
            for (int i = 0; i < window; i++) {
                int idx = (state->history_idx - i - 1 + MAX_FILTRATION_STEPS) % MAX_FILTRATION_STEPS;
                mean_b1 += state->betti_history[idx][1];
            }
            mean_b1 /= window;

            for (int i = 0; i < window; i++) {
                int idx = (state->history_idx - i - 1 + MAX_FILTRATION_STEPS) % MAX_FILTRATION_STEPS;
                float diff = state->betti_history[idx][1] - mean_b1;
                variance += diff * diff;
            }
            variance /= window;
        }

        state->stability_score = 1.0f / (1.0f + variance);

        // Track stable/unstable iterations
        if (variance < 0.1f) {
            state->stable_iterations++;
            state->unstable_iterations = 0;
        } else {
            state->unstable_iterations++;
            state->stable_iterations = 0;
        }

        // ═══════════════════════════════════════════════════════════════
        // PHASE 4: Make transition detection decision
        // ═══════════════════════════════════════════════════════════════

        state->transition_strength = transition_score;
        state->phase_transition_detected =
            (transition_score > config->transition_sensitivity) ? 1 : 0;
    }
}

/**
 * Combined TPTP update kernel - runs all phases sequentially
 */
extern "C" __global__ void tptp_full_update(
    const int* __restrict__ graph_row_ptr,
    const int* __restrict__ graph_col_idx,
    const float* __restrict__ conflict_signal,
    TPTPState* __restrict__ state,
    const RuntimeConfig* __restrict__ config,
    int num_vertices
) {
    // In production, this would orchestrate the full TPTP pipeline
    // For now, the phases should be called as separate kernels for better occupancy

    // Phase 1: Build complex (done in tptp_build_complex)
    // Phase 2: Compute homology (done in tptp_compute_homology)
    // Phase 3: Detect transitions (done in tptp_detect_transition)

    // This kernel serves as a placeholder for future fusion optimization
}

/**
 * Reset TPTP state for new optimization run
 */
extern "C" __global__ void tptp_reset_state(
    TPTPState* __restrict__ state
) {
    int tid = threadIdx.x;

    if (tid == 0) {
        state->num_simplices = 0;
        state->num_pairs = 0;
        state->betti[0] = 0.0f;
        state->betti[1] = 0.0f;
        state->betti[2] = 0.0f;
        state->history_idx = 0;
        state->history_filled = 0;
        state->max_persistence = 0.0f;
        state->avg_persistence = 0.0f;
        state->persistence_entropy = 0.0f;
        state->total_persistence = 0.0f;
        state->phase_transition_detected = 0;
        state->transition_strength = 0.0f;
        state->betti_0_derivative = 0.0f;
        state->betti_1_derivative = 0.0f;
        state->betti_2_derivative = 0.0f;
        state->stability_score = 1.0f;
        state->stable_iterations = 0;
        state->unstable_iterations = 0;
        state->current_filtration = 0.0f;
        state->filtration_step = 0.1f;
        state->filtration_index = 0;
    }

    // Clear arrays in parallel
    for (int i = tid; i < MAX_SIMPLICES; i += blockDim.x) {
        state->simplices[i].type = -1;
        state->simplices[i].paired_with = -1;
    }

    for (int i = tid; i < MAX_PERSISTENCE_PAIRS; i += blockDim.x) {
        state->pairs[i].birth = 0.0f;
        state->pairs[i].death = 0.0f;
        state->pairs[i].persistence = 0.0f;
    }

    for (int i = tid; i < MAX_FILTRATION_STEPS; i += blockDim.x) {
        state->betti_history[i][0] = 0.0f;
        state->betti_history[i][1] = 0.0f;
        state->betti_history[i][2] = 0.0f;
    }
}

/**
 * Extract topological features for FluxNet RL state
 */
extern "C" __global__ void tptp_extract_features(
    const TPTPState* __restrict__ state,
    float* __restrict__ features,
    int num_features
) {
    int tid = threadIdx.x;

    if (tid == 0 && num_features >= 16) {
        // Feature vector for RL state
        features[0] = state->betti[0];                  // β0
        features[1] = state->betti[1];                  // β1
        features[2] = state->betti[2];                  // β2
        features[3] = state->betti_0_derivative;        // dβ0/dt
        features[4] = state->betti_1_derivative;        // dβ1/dt
        features[5] = state->betti_2_derivative;        // dβ2/dt
        features[6] = state->max_persistence;           // Max persistence
        features[7] = state->avg_persistence;           // Avg persistence
        features[8] = state->persistence_entropy;       // Entropy
        features[9] = state->stability_score;           // Stability
        features[10] = (float)state->phase_transition_detected;
        features[11] = state->transition_strength;
        features[12] = (float)state->stable_iterations / 100.0f;
        features[13] = (float)state->unstable_iterations / 100.0f;
        features[14] = (float)state->num_pairs / 100.0f;
        features[15] = state->total_persistence;
    }
}
