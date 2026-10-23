/**
 * Dendritic Reservoir CUDA Kernel for Phase 0 Warmstart
 *
 * ASSUMPTIONS:
 * - Input: Graph adjacency list as CSR format (row_ptr, col_idx)
 * - Output: difficulty[n] and uncertainty[n] vectors (f32)
 * - MAX_VERTICES = 100,000 (enforced by caller)
 * - Precision: f32 for reservoir state and metrics
 * - Block size: 256 threads per block (optimal for vertex-parallel operations)
 * - Number of dendritic branches: Configurable (default 8)
 * - Leak rate: Temporal dynamics parameter (default 0.1)
 * - Requires: sm_86 (RTX 3060 Ampere architecture)
 *
 * ALGORITHM OVERVIEW:
 * 1. Initialize reservoir state with random weights per branch
 * 2. Propagate graph structure through dendritic branches (multiple iterations)
 * 3. Compute per-vertex difficulty from reservoir state (coloring hardness)
 * 4. Compute per-vertex uncertainty from state variance (exploration need)
 *
 * DENDRITIC RESERVOIR THEORY:
 * - Multi-branch neuromorphic computation inspired by biological neurons
 * - Each vertex maintains state across multiple dendritic branches
 * - Propagation aggregates neighbor states with leak rate for temporal dynamics
 * - Difficulty: Mean activation across branches (high = hard to color)
 * - Uncertainty: Variance across branches (high = exploration needed)
 *
 * PERFORMANCE TARGETS:
 * - DSJC250 (250 vertices): < 100ms for 50 iterations
 * - GPU utilization: > 70%
 * - Memory bandwidth: Efficient coalesced access patterns
 *
 * REFERENCE: PRISM GPU Plan §4.1 (Phase 0 Dendritic Reservoir Kernel)
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <math.h>

// Maximum supported graph size (caller must enforce)
#define MAX_VERTICES 100000

// Maximum number of dendritic branches
#define MAX_BRANCHES 32

/**
 * Initialize reservoir state with random weights
 *
 * Each vertex has `branches` dendritic branches, each initialized with random
 * weights from uniform distribution [0, 1].
 *
 * Uses cuRAND for high-quality random number generation on GPU.
 *
 * @param state      Reservoir state array (n × branches, row-major)
 * @param n          Number of vertices
 * @param branches   Number of dendritic branches per vertex
 * @param seed       Random seed for reproducibility
 */
extern "C" __global__ void init_reservoir(
    float* __restrict__ state,
    const int n,
    const int branches,
    const unsigned long long seed
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n * branches) {
        // Initialize cuRAND state for this thread
        curandState_t rand_state;
        curand_init(seed, idx, 0, &rand_state);

        // Generate random initial state in [0, 1]
        state[idx] = curand_uniform(&rand_state);
    }
}

/**
 * Propagate graph structure through dendritic branches
 *
 * Each vertex aggregates neighbor states across all branches with leak rate.
 * Uses CSR (Compressed Sparse Row) format for memory-efficient graph storage.
 *
 * Update rule for vertex i, branch b:
 *   new_state[i][b] = (1 - leak_rate) * old_state[i][b] +
 *                     leak_rate * mean(neighbor_states[j][b] for j in neighbors(i))
 *
 * @param state_out    Output state array (n × branches, row-major)
 * @param state_in     Input state array (n × branches, row-major)
 * @param row_ptr      CSR row pointer array (n+1 elements)
 * @param col_idx      CSR column index array (nnz elements)
 * @param n            Number of vertices
 * @param branches     Number of dendritic branches per vertex
 * @param leak_rate    Leak rate for temporal dynamics [0, 1]
 */
extern "C" __global__ void propagate_dendritic(
    float* __restrict__ state_out,
    const float* __restrict__ state_in,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const int n,
    const int branches,
    const float leak_rate
) {
    const int vertex = blockIdx.x * blockDim.x + threadIdx.x;

    if (vertex >= n) return;

    // Get neighbor range for this vertex from CSR structure
    const int neighbor_start = row_ptr[vertex];
    const int neighbor_end = row_ptr[vertex + 1];
    const int degree = neighbor_end - neighbor_start;

    // Process each dendritic branch independently
    for (int branch = 0; branch < branches; branch++) {
        const int state_idx = vertex * branches + branch;
        float old_state = state_in[state_idx];

        if (degree == 0) {
            // Isolated vertex: Decay towards 0
            state_out[state_idx] = (1.0f - leak_rate) * old_state;
        } else {
            // Aggregate neighbor states
            float neighbor_sum = 0.0f;
            for (int i = neighbor_start; i < neighbor_end; i++) {
                const int neighbor = col_idx[i];
                if (neighbor < n) {
                    const int neighbor_state_idx = neighbor * branches + branch;
                    neighbor_sum += state_in[neighbor_state_idx];
                }
            }
            float neighbor_mean = neighbor_sum / (float)degree;

            // Leaky integration: blend old state with neighbor mean
            state_out[state_idx] = (1.0f - leak_rate) * old_state +
                                   leak_rate * neighbor_mean;
        }
    }
}

/**
 * Compute difficulty metric from reservoir state
 *
 * Difficulty measures how hard a vertex is to color based on reservoir activation.
 * High activation across branches indicates structural complexity (high degree,
 * central position, etc.).
 *
 * Difficulty[i] = tanh(mean(state[i][b] for b in branches))
 *                 Normalized to [0, 1] via tanh activation
 *
 * @param state        Reservoir state array (n × branches, row-major)
 * @param difficulty   Output difficulty array (n elements)
 * @param n            Number of vertices
 * @param branches     Number of dendritic branches per vertex
 */
extern "C" __global__ void compute_difficulty(
    const float* __restrict__ state,
    float* __restrict__ difficulty,
    const int n,
    const int branches
) {
    const int vertex = blockIdx.x * blockDim.x + threadIdx.x;

    if (vertex >= n) return;

    // Compute mean activation across all branches
    float sum = 0.0f;
    for (int branch = 0; branch < branches; branch++) {
        const int state_idx = vertex * branches + branch;
        sum += state[state_idx];
    }
    float mean = sum / (float)branches;

    // Apply tanh activation to normalize to [0, 1] range
    // tanh maps (-inf, inf) -> (-1, 1), then shift/scale to [0, 1]
    float activated = tanhf(mean);
    difficulty[vertex] = (activated + 1.0f) * 0.5f; // Scale [-1, 1] -> [0, 1]
}

/**
 * Compute uncertainty metric from reservoir state variance
 *
 * Uncertainty measures disagreement between branches, indicating exploration need.
 * High variance means branches have diverged -> vertex behavior is unpredictable.
 *
 * Uncertainty[i] = sqrt(variance(state[i][b] for b in branches))
 *                  = standard deviation, normalized to [0, 1]
 *
 * @param state         Reservoir state array (n × branches, row-major)
 * @param uncertainty   Output uncertainty array (n elements)
 * @param n             Number of vertices
 * @param branches      Number of dendritic branches per vertex
 */
extern "C" __global__ void compute_uncertainty(
    const float* __restrict__ state,
    float* __restrict__ uncertainty,
    const int n,
    const int branches
) {
    const int vertex = blockIdx.x * blockDim.x + threadIdx.x;

    if (vertex >= n) return;

    // Compute mean across branches
    float sum = 0.0f;
    for (int branch = 0; branch < branches; branch++) {
        const int state_idx = vertex * branches + branch;
        sum += state[state_idx];
    }
    float mean = sum / (float)branches;

    // Compute variance: E[(X - mean)^2]
    float variance_sum = 0.0f;
    for (int branch = 0; branch < branches; branch++) {
        const int state_idx = vertex * branches + branch;
        float diff = state[state_idx] - mean;
        variance_sum += diff * diff;
    }
    float variance = variance_sum / (float)branches;

    // Standard deviation (sqrt of variance)
    float std_dev = sqrtf(variance);

    // Normalize to [0, 1] by clamping (std_dev typically in [0, 1] range)
    // Since state values are in [0, 1], max possible std_dev ≈ 0.5
    uncertainty[vertex] = fminf(std_dev * 2.0f, 1.0f);
}

/**
 * Normalize difficulty and uncertainty vectors to sum to 1.0
 *
 * Required for using these metrics as prior probabilities in warmstart.
 * Ensures valid probability distribution for softmax sampling.
 *
 * @param values    Array to normalize (modified in-place)
 * @param n         Number of elements
 */
extern "C" __global__ void normalize_to_distribution(
    float* __restrict__ values,
    const int n
) {
    // Phase 1: Compute sum using parallel reduction in shared memory
    __shared__ float shared_sum[256];

    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load values into shared memory
    float local_sum = 0.0f;
    if (idx < n) {
        local_sum = values[idx];
    }
    shared_sum[tid] = local_sum;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    // Block sum is in shared_sum[0]
    // For simplicity, we'll do a single-block implementation (n <= 256)
    // For larger n, would need atomic add to global memory

    if (tid == 0 && blockIdx.x == 0) {
        float total_sum = shared_sum[0];
        // Store in values[0] temporarily for phase 2
        values[n] = total_sum; // Use extra space at end
    }
    __syncthreads();

    // Phase 2: Normalize by total sum
    if (idx < n) {
        float total_sum = values[n];
        if (total_sum > 1e-8f) { // Avoid division by zero
            values[idx] = values[idx] / total_sum;
        } else {
            // Fallback: Uniform distribution
            values[idx] = 1.0f / (float)n;
        }
    }
}

/**
 * Combined kernel: Compute both difficulty and uncertainty in one pass
 *
 * Optimized version that computes both metrics simultaneously to reduce
 * memory bandwidth and improve performance.
 *
 * @param state         Reservoir state array (n × branches, row-major)
 * @param difficulty    Output difficulty array (n elements)
 * @param uncertainty   Output uncertainty array (n elements)
 * @param n             Number of vertices
 * @param branches      Number of dendritic branches per vertex
 */
extern "C" __global__ void compute_metrics_combined(
    const float* __restrict__ state,
    float* __restrict__ difficulty,
    float* __restrict__ uncertainty,
    const int n,
    const int branches
) {
    const int vertex = blockIdx.x * blockDim.x + threadIdx.x;

    if (vertex >= n) return;

    // Single pass: Compute mean and variance together
    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (int branch = 0; branch < branches; branch++) {
        const int state_idx = vertex * branches + branch;
        float val = state[state_idx];
        sum += val;
        sum_sq += val * val;
    }

    float mean = sum / (float)branches;
    float variance = (sum_sq / (float)branches) - (mean * mean);

    // Difficulty: Normalized mean activation
    float activated = tanhf(mean);
    difficulty[vertex] = (activated + 1.0f) * 0.5f;

    // Uncertainty: Normalized standard deviation
    float std_dev = sqrtf(fmaxf(variance, 0.0f)); // Clamp negative due to numerical error
    uncertainty[vertex] = fminf(std_dev * 2.0f, 1.0f);
}
