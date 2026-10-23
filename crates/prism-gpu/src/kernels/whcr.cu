/**
 * PRISM Wavelet-Hierarchical Conflict Repair (WHCR) - CUDA Kernels
 * 
 * Mixed-precision GPU acceleration for multiresolution conflict repair:
 * - f32 (float) for coarse levels: fast exploration, 2x memory bandwidth
 * - f64 (double) for fine levels: precise refinement near solution
 * 
 * Key innovations:
 * 1. Wavelet-decomposed conflict signals on GPU
 * 2. Hierarchical coarsening with spectral preservation
 * 3. Geometry-coupled move evaluation using TDA/geodesic metrics
 * 4. Precision-stratified computation based on resolution level
 */

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <math.h>

namespace cg = cooperative_groups;

// ═══════════════════════════════════════════════════════════════════════════
// CONSTANTS AND CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define MAX_COLORS 256  // Fixed: increased to match Rust implementation
#define MAX_DEGREE 128

// Precision thresholds
#define COARSE_THRESHOLD 0.5f   // Use f32 when conflict density > threshold
#define FINE_THRESHOLD 0.1f     // Use f64 when conflict density < threshold

// ═══════════════════════════════════════════════════════════════════════════
// GEOMETRY WEIGHTS (Constant Memory for Configuration)
// ═══════════════════════════════════════════════════════════════════════════

// Geometry weights configurable from host
// These are set via cudaMemcpyToSymbol from Rust before kernel launch
__constant__ float c_stress_weight = 0.25f;
__constant__ float c_persistence_weight = 0.1f;
__constant__ float c_belief_weight = 0.3f;
__constant__ float c_hotspot_multiplier = 1.2f;

// ═══════════════════════════════════════════════════════════════════════════
// DATA STRUCTURES
// ═══════════════════════════════════════════════════════════════════════════

/**
 * CSR (Compressed Sparse Row) representation of graph adjacency
 * Used for both fine and coarse levels
 */
struct GraphCSR {
    int num_vertices;
    int num_edges;
    int* row_ptr;      // [num_vertices + 1] - start index of each vertex's neighbors
    int* col_idx;      // [num_edges] - neighbor indices
    float* edge_weights;  // [num_edges] - optional edge weights
};

/**
 * Multiresolution hierarchy for conflict repair
 */
struct MRAHierarchy {
    int num_levels;
    GraphCSR* levels;           // [num_levels] - coarsened graphs
    int** projections;          // [num_levels][fine_vertices] - fine → coarse mapping
    int** lifting_ptr;          // [num_levels][coarse_vertices+1] - coarse → fine CSR
    int** lifting_idx;          // [num_levels][fine_vertices] - coarse → fine indices
    float* approximations;      // Concatenated approximation coefficients
    double* details;            // Concatenated detail coefficients (f64 for precision)
};

/**
 * Geometry coupling from prior PRISM phases
 */
struct GeometryData {
    double* stress_scores;       // [num_vertices] from Phase 4 geodesic
    double* persistence_scores;  // [num_vertices] from Phase 6 TDA
    int* hotspot_mask;           // [num_vertices] 1 if hotspot, 0 otherwise
    double* belief_distribution; // [num_vertices * num_colors] from Phase 1
    int num_vertices;
    int num_colors;
};

// ═══════════════════════════════════════════════════════════════════════════
// UTILITY KERNELS
// ═══════════════════════════════════════════════════════════════════════════

// Wrap all kernels in extern "C" to prevent C++ name mangling
extern "C" {

/**
 * Count conflicts per vertex (f32 version for coarse computation)
 */
__global__ void count_conflicts_f32(
    const int* __restrict__ coloring,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    float* __restrict__ conflict_counts,
    int num_vertices
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_vertices) {
        int my_color = coloring[tid];
        int start = row_ptr[tid];
        int end = row_ptr[tid + 1];
        
        float count = 0.0f;
        for (int i = start; i < end; i++) {
            int neighbor = col_idx[i];
            if (coloring[neighbor] == my_color) {
                count += 1.0f;
            }
        }
        conflict_counts[tid] = count;
    }
}

/**
 * Count conflicts per vertex (f64 version for fine computation)
 */
__global__ void count_conflicts_f64(
    const int* __restrict__ coloring,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const double* __restrict__ stress_scores,
    const int* __restrict__ hotspot_mask,
    double* __restrict__ conflict_counts,
    int num_vertices
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_vertices) {
        int my_color = coloring[tid];
        int start = row_ptr[tid];
        int end = row_ptr[tid + 1];
        
        double count = 0.0;
        for (int i = start; i < end; i++) {
            int neighbor = col_idx[i];
            if (coloring[neighbor] == my_color) {
                // Weight by geometry: high-stress conflicts count more
                double weight = 1.0 + stress_scores[neighbor] * 0.5;
                count += weight;
            }
        }
        
        // Hotspots get boosted conflict visibility
        if (hotspot_mask[tid]) {
            count *= 1.2;
        }
        
        conflict_counts[tid] = count;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// WAVELET DECOMPOSITION KERNELS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Compute wavelet detail coefficients (high-frequency local conflicts)
 * Detail = fine_signal - interpolated(coarse_signal)
 */
__global__ void compute_wavelet_details(
    const float* __restrict__ fine_signal,
    const float* __restrict__ coarse_signal,
    const int* __restrict__ projection,
    double* __restrict__ details,
    int num_fine_vertices
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_fine_vertices) {
        int coarse_v = projection[tid];
        // Detail captures local variation not explained by coarse structure
        details[tid] = (double)fine_signal[tid] - (double)coarse_signal[coarse_v];
    }
}

/**
 * Reconstruct conflict signal from wavelet coefficients
 * Used for wavelet-guided vertex prioritization
 */
__global__ void reconstruct_from_wavelets(
    const float* __restrict__ approximation,
    const double* __restrict__ details,
    const int* __restrict__ projection,
    float* __restrict__ reconstructed,
    int num_fine_vertices
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_fine_vertices) {
        int coarse_v = projection[tid];
        reconstructed[tid] = approximation[coarse_v] + (float)details[tid];
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// COARSENING KERNELS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Compute matching scores for heavy-edge matching with conflict bias
 * Uses f32 for speed since this is exploratory
 */
__global__ void compute_matching_scores(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const float* __restrict__ vertex_weights,
    const float* __restrict__ conflict_mass,
    float* __restrict__ edge_scores,
    int num_vertices
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_vertices) {
        int start = row_ptr[tid];
        int end = row_ptr[tid + 1];
        
        for (int i = start; i < end; i++) {
            int neighbor = col_idx[i];
            if (neighbor > tid) {  // Only compute once per edge
                // Score combines weight sum and conflict similarity
                float weight_sum = vertex_weights[tid] + vertex_weights[neighbor];
                float conflict_diff = fabsf(conflict_mass[tid] - conflict_mass[neighbor]);
                float conflict_similarity = 1.0f / (1.0f + conflict_diff);
                
                edge_scores[i] = weight_sum * conflict_similarity;
            }
        }
    }
}

/**
 * Aggregate fine vertices to coarse level
 * Computes coarse weights and conflict mass
 */
__global__ void aggregate_to_coarse(
    const float* __restrict__ fine_weights,
    const float* __restrict__ fine_conflicts,
    const int* __restrict__ lifting_ptr,
    const int* __restrict__ lifting_idx,
    float* __restrict__ coarse_weights,
    float* __restrict__ coarse_conflicts,
    int num_coarse_vertices
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_coarse_vertices) {
        int start = lifting_ptr[tid];
        int end = lifting_ptr[tid + 1];
        
        float weight_sum = 0.0f;
        float conflict_sum = 0.0f;
        
        for (int i = start; i < end; i++) {
            int fine_v = lifting_idx[i];
            weight_sum += fine_weights[fine_v];
            conflict_sum += fine_conflicts[fine_v];
        }
        
        coarse_weights[tid] = weight_sum;
        coarse_conflicts[tid] = conflict_sum;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MOVE EVALUATION KERNELS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Evaluate all possible color moves for conflicting vertices (f32 coarse)
 * Fast exploration at coarse resolution levels
 */
__global__ void evaluate_moves_f32(
    const int* __restrict__ coloring,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const int* __restrict__ conflict_vertices,  // Only vertices with conflicts
    int num_conflict_vertices,
    int num_colors,
    float* __restrict__ move_deltas,  // [num_conflict_vertices * num_colors]
    int* __restrict__ best_colors,    // [num_conflict_vertices]
    const float* __restrict__ reservoir_priorities // optional, can be null-equivalent (zeroed buffer)
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_conflict_vertices) {
        int vertex = conflict_vertices[tid];
        int current_color = coloring[vertex];
        int start = row_ptr[vertex];
        int end = row_ptr[vertex + 1];

        // Fixed: Validate num_colors to prevent buffer overflow
        if (num_colors > MAX_COLORS) return;
        
        // Count neighbor colors
        int neighbor_colors[MAX_COLORS];
        for (int c = 0; c < num_colors; c++) {
            neighbor_colors[c] = 0;
        }
        
        for (int i = start; i < end; i++) {
            int neighbor = col_idx[i];
            int n_color = coloring[neighbor];
            if (n_color < num_colors) {
                neighbor_colors[n_color]++;
            }
        }
        
        // Evaluate each possible new color
        int base = tid * num_colors;
        int current_conf = neighbor_colors[current_color];
        move_deltas[base + current_color] = 0.0f;

        float best_delta = 0.0f;
        int best_color = current_color;
        bool found_improving = false;

        for (int new_color = 0; new_color < num_colors; new_color++) {
            int delta_int = neighbor_colors[new_color] - current_conf;
            float delta = (float)delta_int;

            // Reservoir priority bias: encourage moves away from high-priority conflicted vertices
            if (reservoir_priorities != nullptr) {
                float rp = reservoir_priorities[vertex];
                delta -= rp * 0.25f;
            }

            move_deltas[base + new_color] = delta;

            if (delta < best_delta) {
                best_delta = delta;
                best_color = new_color;
                found_improving = true;
            }
        }

        // Tie-break/annealing: if no improving move, pick an equal-conflict color and nudge delta negative
        if (!found_improving) {
            for (int new_color = 0; new_color < num_colors; new_color++) {
                if (neighbor_colors[new_color] == current_conf && new_color != current_color) {
                    best_color = new_color;
                    move_deltas[base + new_color] = -0.01f; // allow application on ties
                    break;
                }
            }

            // If still no escape move selected, add a small randomized nudge
            if (best_color == current_color) {
                int r = vertex * 1664525 + 1013904223; // simple LCG
                int cand = r % num_colors;
                if (cand == current_color) {
                    cand = (cand + 1) % num_colors;
                }
                float jitter = -0.05f - 0.001f * (float)(r & 0xF);
                best_color = cand;
                move_deltas[base + cand] = jitter;
            }
        }

        best_colors[tid] = best_color;
    }
}

/**
 * Evaluate moves with geometry coupling (f64 fine) - GEOMETRY WEIGHTS VERSION
 *
 * This version has exactly 12 parameters but now includes configurable weights.
 * Weight parameters are packed into a single float4 to stay within the limit.
 *
 * PARAMETERS (12 total):
 * 1. coloring - current vertex colors
 * 2. row_ptr - CSR row pointers
 * 3. col_idx - CSR column indices
 * 4. conflict_vertices - list of conflicting vertices
 * 5. num_conflict_vertices - count of conflicting vertices
 * 6. num_colors - number of colors in use
 * 7. stress_scores - Phase 4 geodesic stress
 * 8. persistence_scores - Phase 6 TDA persistence
 * 9. belief_distribution - Phase 1 active inference beliefs
 * 10. total_vertices - total graph size
 * 11. move_deltas - output delta scores [num_conflict_vertices * num_colors]
 * 12. best_colors - output best color per vertex [num_conflict_vertices]
 *
 * NOTE: Weights are now passed inline as constants in this kernel.
 * To make them configurable, we use kernel parameters during launch from Rust.
 */
__global__ void evaluate_moves_f64(
    const int* __restrict__ coloring,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const int* __restrict__ conflict_vertices,
    int num_conflict_vertices,
    int num_colors,
    const double* __restrict__ stress_scores,
    const double* __restrict__ persistence_scores,
    const double* __restrict__ belief_distribution,  // [num_vertices * num_colors]
    int total_vertices,
    double* __restrict__ move_deltas,
    int* __restrict__ best_colors
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Fixed: Validate num_colors to prevent buffer overflow
    if (num_colors > MAX_COLORS) return;

    if (tid < num_conflict_vertices) {
        int vertex = conflict_vertices[tid];
        int current_color = coloring[vertex];
        int start = row_ptr[vertex];
        int end = row_ptr[vertex + 1];

        // Geometry weights for this vertex
        double my_stress = stress_scores[vertex];
        double my_persistence = persistence_scores[vertex];

        // Derive hotspot status from high stress (top 20% = hotspot)
        bool is_hotspot = my_stress > 0.8;

        // Count weighted neighbor colors
        double neighbor_weights[MAX_COLORS];
        for (int c = 0; c < num_colors; c++) {
            neighbor_weights[c] = 0.0;
        }

        for (int i = start; i < end; i++) {
            int neighbor = col_idx[i];
            int n_color = coloring[neighbor];
            if (n_color < num_colors) {
                // Resolved TODO(GPU-WHCR-3): Use configurable geometry weight
                double n_stress = stress_scores[neighbor];
                double weight = 1.0 + (my_stress + n_stress) * c_stress_weight;
                neighbor_weights[n_color] += weight;
            }
        }

        // Evaluate each possible new color with belief guidance
        double best_delta = 0.0;
        int best_color = current_color;

        for (int new_color = 0; new_color < num_colors; new_color++) {
            if (new_color == current_color) {
                // Explicitly define delta for current color
                move_deltas[tid * num_colors + new_color] = 0.0;
                continue;
            }

            // Base delta: weighted conflict change
            double delta = neighbor_weights[new_color] - neighbor_weights[current_color];

            // Resolved TODO(GPU-WHCR-3): Use configurable belief weight
            if (belief_distribution != nullptr) {
                double belief_current = belief_distribution[vertex * num_colors + current_color];
                double belief_new = belief_distribution[vertex * num_colors + new_color];
                delta -= (belief_new - belief_current) * c_belief_weight;
            }

            // Resolved TODO(GPU-WHCR-3): Use configurable hotspot multiplier
            if (is_hotspot && delta < 0.0) {
                delta *= c_hotspot_multiplier;
            }

            // Resolved TODO(GPU-WHCR-3): Use configurable persistence weight
            delta += my_persistence * c_persistence_weight;

            move_deltas[tid * num_colors + new_color] = delta;

            if (delta < best_delta) {
                best_delta = delta;
                best_color = new_color;
            }
        }

        // Randomized escape: if nothing improves, allow a small negative nudge to a random color
        if (best_delta >= 0.0) {
            int r = vertex * 1664525 + 1013904223; // simple LCG
            int cand = r % num_colors;
            if (cand == current_color) {
                cand = (cand + 1) % num_colors;
            }
            double jitter = -0.05 - 0.001 * (double)(r & 0xF);
            best_color = cand;
            best_delta = jitter;
            move_deltas[tid * num_colors + cand] = jitter;
        }

        best_colors[tid] = best_color;
    }
}

/**
 * Evaluate moves with geometry coupling (f64 fine) - FULL GEOMETRY VERSION
 *
 * This version has 13 parameters and cannot be called via LaunchAsync.
 * Kept for future use when we have a better parameter passing strategy.
 * TODO(GPU-WHCR): Implement struct-based parameter passing for 13+ params
 */
__global__ void evaluate_moves_f64_geometry(
    const int* __restrict__ coloring,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const int* __restrict__ conflict_vertices,
    int num_conflict_vertices,
    int num_colors,
    const double* __restrict__ stress_scores,
    const double* __restrict__ persistence_scores,
    const int* __restrict__ hotspot_mask,
    const double* __restrict__ belief_distribution,  // [num_vertices * num_colors]
    int total_vertices,
    double* __restrict__ move_deltas,
    int* __restrict__ best_colors
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Fixed: Validate num_colors to prevent buffer overflow
    if (num_colors > MAX_COLORS) return;

    if (tid < num_conflict_vertices) {
        int vertex = conflict_vertices[tid];
        int current_color = coloring[vertex];
        int start = row_ptr[vertex];
        int end = row_ptr[vertex + 1];

        // Geometry weights for this vertex
        double my_stress = stress_scores[vertex];
        double my_persistence = persistence_scores[vertex];
        bool is_hotspot = hotspot_mask[vertex] != 0;

        // Count weighted neighbor colors
        double neighbor_weights[MAX_COLORS];
        for (int c = 0; c < num_colors; c++) {
            neighbor_weights[c] = 0.0;
        }

        for (int i = start; i < end; i++) {
            int neighbor = col_idx[i];
            int n_color = coloring[neighbor];
            if (n_color < num_colors) {
                // Resolved TODO(GPU-WHCR-3): Use configurable geometry weight
                double n_stress = stress_scores[neighbor];
                double weight = 1.0 + (my_stress + n_stress) * c_stress_weight;
                neighbor_weights[n_color] += weight;
            }
        }

        // Evaluate each possible new color with belief guidance
        double best_delta = 0.0;
        int best_color = current_color;

        for (int new_color = 0; new_color < num_colors; new_color++) {
            if (new_color == current_color) continue;

            // Base delta: weighted conflict change
            double delta = neighbor_weights[new_color] - neighbor_weights[current_color];

            // Resolved TODO(GPU-WHCR-3): Use configurable belief weight
            if (belief_distribution != nullptr) {
                double belief_current = belief_distribution[vertex * num_colors + current_color];
                double belief_new = belief_distribution[vertex * num_colors + new_color];
                delta -= (belief_new - belief_current) * c_belief_weight;
            }

            // Resolved TODO(GPU-WHCR-3): Use configurable hotspot multiplier
            if (is_hotspot && delta < 0.0) {
                delta *= c_hotspot_multiplier;
            }

            // Resolved TODO(GPU-WHCR-3): Use configurable persistence weight
            delta += my_persistence * c_persistence_weight;

            move_deltas[tid * num_colors + new_color] = delta;

            if (delta < best_delta) {
                best_delta = delta;
                best_color = new_color;
            }
        }

        best_colors[tid] = best_color;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MOVE APPLICATION KERNEL
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Apply best moves in parallel with conflict detection
 * Uses vertex-level locking to avoid simultaneous conflicting updates
 */
__global__ void apply_moves_with_locking(
    int* __restrict__ coloring,
    const int* __restrict__ conflict_vertices,
    const int* __restrict__ best_colors,
    const float* __restrict__ move_deltas,  // Best delta per vertex
    int num_conflict_vertices,
    int num_colors,
    int* __restrict__ locks,  // Per-vertex locks
    int* __restrict__ num_moves_applied,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_conflict_vertices) {
        int vertex = conflict_vertices[tid];
        int new_color = best_colors[tid];
        float delta = move_deltas[tid * num_colors + new_color];

        if (new_color == coloring[vertex]) return;

        // Recompute local conflicts against current coloring to avoid stale moves
        int start = row_ptr[vertex];
        int end   = row_ptr[vertex + 1];
        int cur_color = coloring[vertex];
        int cur_conf = 0;
        int new_conf = 0;
        for (int i = start; i < end; ++i) {
            int n = col_idx[i];
            int n_color = coloring[n];
            cur_conf += (n_color == cur_color);
            new_conf += (n_color == new_color);
        }

        float delta_conf = (float)new_conf - (float)cur_conf;
        bool accept = (delta_conf <= 0.5f && delta < -1e-3f) ||   // allow ties/slight improvement
                      (delta_conf <= 3.0f && delta < -1e-2f);     // allow small worsening only with strong preference

        // Apply when conflicts do not worsen (or only slightly) and delta shows preference
        if (accept) {
            // Try to acquire lock on vertex
            if (atomicCAS(&locks[vertex], 0, 1) == 0) {
                // Also lock neighbors to avoid simultaneous adjacent moves
                int start = row_ptr[vertex];
                int end   = row_ptr[vertex + 1];
                bool neighbor_locked = true;
                for (int i = start; i < end; ++i) {
                    int n = col_idx[i];
                    if (atomicCAS(&locks[n], 0, 1) != 0) {
                        neighbor_locked = false;
                        // release any neighbor locks acquired so far
                        for (int j = start; j < i; ++j) {
                            int rel = col_idx[j];
                            atomicExch(&locks[rel], 0);
                        }
                        break;
                    }
                }

                if (neighbor_locked) {
                    coloring[vertex] = new_color;
                    atomicAdd(num_moves_applied, 1);
                    for (int i = start; i < end; ++i) {
                        int rel = col_idx[i];
                        atomicExch(&locks[rel], 0);
                    }
                }

                atomicExch(&locks[vertex], 0);
            }
        }
    }
}

/**
 * Apply best moves with parallel locking (f64 precision version)
 * Uses double precision for move deltas when fine-level precision is needed
 */
extern "C" __global__ void apply_moves_with_locking_f64(
    int* __restrict__ coloring,
    const int* __restrict__ conflict_vertices,
    const int* __restrict__ best_colors,
    const double* __restrict__ move_deltas,  // f64 instead of f32
    int num_conflict_vertices,
    int num_colors,
    int* __restrict__ locks,  // Per-vertex locks
    int* __restrict__ num_moves_applied,
    // CSR graph (needed for on-the-fly validation)
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    // Geometry for weighted conflict metric (must match count_conflicts_f64)
    const double* __restrict__ stress_scores,
    const int* __restrict__ hotspot_mask
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_conflict_vertices) {
        int vertex = conflict_vertices[tid];
        int new_color = best_colors[tid];
        double delta = move_deltas[tid * num_colors + new_color];  // double precision

        if (new_color == coloring[vertex]) return;

        // Recompute local conflict delta against current coloring to avoid stale decisions
        int start = row_ptr[vertex];
        int end   = row_ptr[vertex + 1];
        int cur_color = coloring[vertex];
        double cur_conf = 0.0;
        double new_conf = 0.0;
        for (int i = start; i < end; ++i) {
            int n = col_idx[i];
            int n_color = coloring[n];
            double weight = 1.0 + stress_scores[n] * 0.5;
            if (hotspot_mask[vertex]) {
                weight *= 1.2;
            }
            if (n_color == cur_color) cur_conf += weight;
            if (n_color == new_color) new_conf += weight;
        }

        // Allow non-worsening moves when delta prefers the change
        double delta_conf = new_conf - cur_conf;
        if (delta_conf <= 0.0 && delta < -1e-3) {
            // Try to acquire lock on vertex
            if (atomicCAS(&locks[vertex], 0, 1) == 0) {
                // Optional: also lock neighbors to avoid simultaneous adjacent moves
                bool neighbor_locked = true;
                for (int i = start; i < end; ++i) {
                    int n = col_idx[i];
                    if (atomicCAS(&locks[n], 0, 1) != 0) {
                        neighbor_locked = false;
                        // release any neighbor locks acquired so far
                        for (int j = start; j < i; ++j) {
                            int rel = col_idx[j];
                            atomicExch(&locks[rel], 0);
                        }
                        break;
                    }
                }

                if (neighbor_locked) {
                    coloring[vertex] = new_color;
                    atomicAdd(num_moves_applied, 1);

                    // release neighbor locks
                    for (int i = start; i < end; ++i) {
                        int rel = col_idx[i];
                        atomicExch(&locks[rel], 0);
                    }
                }

                // release vertex lock
                atomicExch(&locks[vertex], 0);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// WAVELET-GUIDED PRIORITIZATION
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Compute priority scores based on wavelet detail coefficients
 * High |detail| = local conflict not explained by global structure = high priority
 */
__global__ void compute_wavelet_priorities(
    const double* __restrict__ details,
    const float* __restrict__ conflict_counts,
    const double* __restrict__ stress_scores,
    const int* __restrict__ hotspot_mask,
    float* __restrict__ priorities,
    int num_vertices
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_vertices) {
        // Only prioritize conflicting vertices
        if (conflict_counts[tid] < 0.5f) {
            priorities[tid] = -1.0f;  // Not conflicting
            return;
        }
        
        // Priority combines wavelet detail, conflict count, and geometry
        double detail_magnitude = fabs(details[tid]);
        float conflict_weight = conflict_counts[tid];
        double stress_weight = stress_scores[tid];
        float hotspot_bonus = hotspot_mask[tid] ? 2.0f : 0.0f;
        
        // Wavelet detail is primary driver: local anomalies need attention
        float priority = (float)(detail_magnitude * 3.0) 
                       + conflict_weight 
                       + (float)(stress_weight * 0.5)
                       + hotspot_bonus;
        
        priorities[tid] = priority;
    }
}

} // extern "C" - End of kernel definitions to prevent C++ name mangling

// ═══════════════════════════════════════════════════════════════════════════
// MAIN REPAIR ITERATION KERNEL
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Single iteration of wavelet-hierarchical repair at specified level
 * Dispatches to f32 or f64 kernels based on precision level
 */
extern "C" __global__ void whcr_iteration(
    int* coloring,
    const GraphCSR graph,
    const GeometryData geometry,
    float* workspace_f32,
    double* workspace_f64,
    int* conflict_vertices,
    int* num_conflicts,
    int precision_level,  // 0=coarse(f32), 1=mixed, 2=fine(f64)
    int num_colors,
    int* iteration_result
) {
    // This is a device-side orchestration kernel
    // Actual implementation would dispatch to appropriate sub-kernels
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Cooperative groups for synchronization
    cg::grid_group grid = cg::this_grid();
    
    // Phase 1: Count conflicts (precision-dependent)
    if (precision_level == 0) {
        // f32 path - fast
        if (tid < graph.num_vertices) {
            int my_color = coloring[tid];
            int start = graph.row_ptr[tid];
            int end = graph.row_ptr[tid + 1];
            
            float count = 0.0f;
            for (int i = start; i < end; i++) {
                if (coloring[graph.col_idx[i]] == my_color) {
                    count += 1.0f;
                }
            }
            workspace_f32[tid] = count;
        }
    } else {
        // f64 path - precise with geometry
        if (tid < graph.num_vertices) {
            int my_color = coloring[tid];
            int start = graph.row_ptr[tid];
            int end = graph.row_ptr[tid + 1];
            
            double count = 0.0;
            for (int i = start; i < end; i++) {
                int neighbor = graph.col_idx[i];
                if (coloring[neighbor] == my_color) {
                    double weight = 1.0 + geometry.stress_scores[neighbor] * 0.5;
                    count += weight;
                }
            }
            workspace_f64[tid] = count;
        }
    }
    
    grid.sync();
    
    // Phase 2: Find best moves and apply
    // (Simplified - full implementation would use sub-kernels)
    
    if (tid == 0) {
        *iteration_result = 1;  // Success flag
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// HOST-SIDE HELPER FUNCTIONS (declared extern "C" for Rust FFI)
// ═══════════════════════════════════════════════════════════════════════════

extern "C" {

/**
 * Launch f32 conflict counting kernel
 */
void launch_count_conflicts_f32(
    const int* coloring,
    const int* row_ptr,
    const int* col_idx,
    float* conflict_counts,
    int num_vertices,
    cudaStream_t stream
) {
    int blocks = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
    count_conflicts_f32<<<blocks, BLOCK_SIZE, 0, stream>>>(
        coloring, row_ptr, col_idx, conflict_counts, num_vertices
    );
}

/**
 * Launch f64 conflict counting kernel with geometry
 */
void launch_count_conflicts_f64(
    const int* coloring,
    const int* row_ptr,
    const int* col_idx,
    const double* stress_scores,
    const int* hotspot_mask,
    double* conflict_counts,
    int num_vertices,
    cudaStream_t stream
) {
    int blocks = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
    count_conflicts_f64<<<blocks, BLOCK_SIZE, 0, stream>>>(
        coloring, row_ptr, col_idx, stress_scores, hotspot_mask, 
        conflict_counts, num_vertices
    );
}

/**
 * Launch wavelet detail computation
 */
void launch_compute_wavelet_details(
    const float* fine_signal,
    const float* coarse_signal,
    const int* projection,
    double* details,
    int num_fine_vertices,
    cudaStream_t stream
) {
    int blocks = (num_fine_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_wavelet_details<<<blocks, BLOCK_SIZE, 0, stream>>>(
        fine_signal, coarse_signal, projection, details, num_fine_vertices
    );
}

/**
 * Launch f32 move evaluation (coarse levels)
 */
void launch_evaluate_moves_f32(
    const int* coloring,
    const int* row_ptr,
    const int* col_idx,
    const int* conflict_vertices,
    int num_conflict_vertices,
    int num_colors,
    float* move_deltas,
    int* best_colors,
    const float* reservoir_priorities,
    cudaStream_t stream
) {
    int blocks = (num_conflict_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
    evaluate_moves_f32<<<blocks, BLOCK_SIZE, 0, stream>>>(
        coloring, row_ptr, col_idx, conflict_vertices,
        num_conflict_vertices, num_colors, move_deltas, best_colors,
        reservoir_priorities
    );
}

/**
 * Launch f64 move evaluation (fine levels) - 12 parameter version
 */
void launch_evaluate_moves_f64(
    const int* coloring,
    const int* row_ptr,
    const int* col_idx,
    const int* conflict_vertices,
    int num_conflict_vertices,
    int num_colors,
    const double* stress_scores,
    const double* persistence_scores,
    const double* belief_distribution,
    int total_vertices,
    double* move_deltas,
    int* best_colors,
    cudaStream_t stream
) {
    int blocks = (num_conflict_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
    evaluate_moves_f64<<<blocks, BLOCK_SIZE, 0, stream>>>(
        coloring, row_ptr, col_idx, conflict_vertices,
        num_conflict_vertices, num_colors,
        stress_scores, persistence_scores, belief_distribution,
        total_vertices, move_deltas, best_colors
    );
}

/**
 * Launch f64 move evaluation with full geometry (fine levels) - 13 parameter version
 * NOTE: This launcher exists but cannot be called from Rust via cudarc due to 12-parameter limit
 * TODO(GPU-WHCR): Implement struct-based parameter passing
 */
void launch_evaluate_moves_f64_geometry(
    const int* coloring,
    const int* row_ptr,
    const int* col_idx,
    const int* conflict_vertices,
    int num_conflict_vertices,
    int num_colors,
    const double* stress_scores,
    const double* persistence_scores,
    const int* hotspot_mask,
    const double* belief_distribution,
    int total_vertices,
    double* move_deltas,
    int* best_colors,
    cudaStream_t stream
) {
    int blocks = (num_conflict_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
    evaluate_moves_f64_geometry<<<blocks, BLOCK_SIZE, 0, stream>>>(
        coloring, row_ptr, col_idx, conflict_vertices,
        num_conflict_vertices, num_colors,
        stress_scores, persistence_scores, hotspot_mask, belief_distribution,
        total_vertices, move_deltas, best_colors
    );
}

/**
 * Launch wavelet priority computation
 */
void launch_compute_wavelet_priorities(
    const double* details,
    const float* conflict_counts,
    const double* stress_scores,
    const int* hotspot_mask,
    float* priorities,
    int num_vertices,
    cudaStream_t stream
) {
    int blocks = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_wavelet_priorities<<<blocks, BLOCK_SIZE, 0, stream>>>(
        details, conflict_counts, stress_scores, hotspot_mask,
        priorities, num_vertices
    );
}

} // extern "C"
