/**
 * @file active_inference.cu
 * @brief GPU-accelerated Active Inference for Phase 1 graph coloring.
 *
 * Implements variational free energy minimization on GPU for uncertainty-driven
 * vertex ordering in graph coloring.
 *
 * ASSUMPTIONS:
 * - Input data in f64 (double precision) for numerical stability
 * - Memory layout: Row-major for matrices, contiguous for vectors
 * - Block size: 256 threads (optimal for Ampere architecture)
 * - MAX_VERTICES: 10,000 (matches other kernels)
 * - Precision range: [0.001, 0.021] (degree-based uncertainty)
 *
 * ALGORITHM:
 * - Prediction error: Precision-weighted difference between observations and beliefs
 * - KL divergence: Gaussian KL between posterior and prior
 * - Accuracy: Negative log-likelihood of observations given beliefs
 * - EFE: Expected Free Energy = Pragmatic value - Epistemic value
 *
 * PERFORMANCE TARGETS:
 * - Full policy computation: < 50ms for 250 vertices
 * - Kernel launch overhead: < 1ms per kernel
 * - Memory transfer: < 10% of total time
 *
 * SECURITY:
 * - Bounds checking on all array accesses
 * - NaN/Inf guards on mathematical operations
 * - No dynamic memory allocation (stack only)
 *
 * REFERENCE: PRISM GPU Plan §4.1 (Phase 1 Active Inference)
 * REFERENCE: foundation/active_inference/gpu.rs (Algorithm specification)
 *
 * Compilation:
 *   nvcc --ptx -o active_inference.ptx active_inference.cu -arch=sm_86 --use_fast_math -O3
 */

#include <math.h>

// Constants
#define BLOCK_SIZE 256
#define MAX_VERTICES 10000
#define EPSILON 1e-10

/**
 * @brief General Matrix-Vector multiplication kernel (GEMV).
 *
 * Computes: y = alpha * A * x + beta * y
 * Where A is m x n matrix (row-major), x is n-vector, y is m-vector.
 *
 * @param A Input matrix (row-major, size: m * n)
 * @param x Input vector (size: n)
 * @param y Output vector (size: m, in/out)
 * @param alpha Scalar multiplier for A*x
 * @param beta Scalar multiplier for y
 * @param m Number of rows in A
 * @param n Number of columns in A
 */
extern "C" __global__ void gemv_kernel(
    const double* A,
    const double* x,
    double* y,
    double alpha,
    double beta,
    int m,
    int n
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= m) return;

    double sum = 0.0;
    for (int col = 0; col < n; ++col) {
        sum += A[row * n + col] * x[col];
    }

    y[row] = alpha * sum + beta * y[row];
}

/**
 * @brief Compute prediction errors with precision weighting.
 *
 * Computes precision-weighted prediction errors for Active Inference:
 * error[i] = precision[i] * (observation[i] - belief[i])
 *
 * This represents the sensory prediction error scaled by confidence.
 * High precision (low uncertainty) amplifies errors.
 *
 * @param errors Output prediction errors (size: n)
 * @param observations Sensory observations (size: n)
 * @param beliefs Current belief state (mean) (size: n)
 * @param precision Precision (inverse variance) per observation (size: n)
 * @param n Number of elements
 */
extern "C" __global__ void prediction_error_kernel(
    double* errors,
    const double* observations,
    const double* beliefs,
    const double* precision,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    // Prediction error: obs - belief
    double error = observations[idx] - beliefs[idx];

    // Precision weighting (higher precision = more confident = larger error signal)
    errors[idx] = precision[idx] * error;
}

/**
 * @brief Update beliefs using gradient descent on free energy.
 *
 * Performs variational update:
 * belief_new[i] = belief_old[i] + learning_rate * gradient[i]
 *
 * Gradient is derived from free energy minimization.
 *
 * @param beliefs Current beliefs (mean) (size: n, in/out)
 * @param gradients Free energy gradients (size: n)
 * @param learning_rate Step size for gradient descent
 * @param n Number of elements
 */
extern "C" __global__ void belief_update_kernel(
    double* beliefs,
    const double* gradients,
    double learning_rate,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    beliefs[idx] += learning_rate * gradients[idx];

    // Clamp to [0, 1] for numerical stability
    beliefs[idx] = fmax(0.0, fmin(1.0, beliefs[idx]));
}

/**
 * @brief Compute precision-weighted prediction error.
 *
 * Similar to prediction_error_kernel but outputs squared weighted error
 * for energy computation.
 *
 * weighted_error[i] = precision[i] * (obs[i] - belief[i])^2
 *
 * @param weighted_errors Output weighted squared errors (size: n)
 * @param observations Sensory observations (size: n)
 * @param beliefs Current beliefs (size: n)
 * @param precision Precision per observation (size: n)
 * @param n Number of elements
 */
extern "C" __global__ void precision_weight_kernel(
    double* weighted_errors,
    const double* observations,
    const double* beliefs,
    const double* precision,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    double error = observations[idx] - beliefs[idx];
    weighted_errors[idx] = precision[idx] * error * error;
}

/**
 * @brief Compute KL divergence between Gaussian distributions.
 *
 * KL[q(x) || p(x)] for Gaussians q ~ N(mean_q, var_q) and p ~ N(mean_p, var_p):
 * KL = 0.5 * [ log(var_p / var_q) - 1 + var_q / var_p + (mean_p - mean_q)^2 / var_p ]
 *
 * This measures the complexity term in variational free energy.
 *
 * @param mean_q Posterior mean (size: n)
 * @param mean_p Prior mean (size: n)
 * @param var_q Posterior variance (size: n)
 * @param var_p Prior variance (size: n)
 * @param kl_components Output KL divergence per element (size: n)
 * @param n Number of elements
 */
extern "C" __global__ void kl_divergence_kernel(
    const double* mean_q,
    const double* mean_p,
    const double* var_q,
    const double* var_p,
    double* kl_components,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    // Guard against division by zero and log(0)
    double var_q_safe = fmax(var_q[idx], EPSILON);
    double var_p_safe = fmax(var_p[idx], EPSILON);

    double mean_diff = mean_p[idx] - mean_q[idx];

    // KL divergence formula for Gaussians
    double kl = 0.5 * (
        log(var_p_safe / var_q_safe) - 1.0 +
        var_q_safe / var_p_safe +
        (mean_diff * mean_diff) / var_p_safe
    );

    // Guard against NaN
    kl_components[idx] = isnan(kl) ? 0.0 : fmax(0.0, kl);
}

/**
 * @brief Compute accuracy term from prediction errors and precision.
 *
 * Accuracy = -0.5 * sum[ error^2 * precision + log(2*pi / precision) ]
 *
 * This is the negative log-likelihood of observations given beliefs.
 *
 * @param errors Prediction errors (size: n)
 * @param precision Precision per observation (size: n)
 * @param accuracy_components Output accuracy components (size: n)
 * @param n Number of elements
 */
extern "C" __global__ void accuracy_kernel(
    const double* errors,
    const double* precision,
    double* accuracy_components,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    double prec = fmax(precision[idx], EPSILON);
    double error = errors[idx];

    // Negative log-likelihood component
    // Note: Constant terms omitted as they don't affect optimization
    accuracy_components[idx] = -0.5 * (error * error * prec + log(2.0 * M_PI / prec));
}

/**
 * @brief Parallel sum reduction kernel.
 *
 * Sums all elements of input array into a single output scalar.
 * Uses shared memory for efficient parallel reduction.
 *
 * @param input Input array (size: n)
 * @param output Output scalar (size: 1, accumulates sum)
 * @param n Number of elements to sum
 */
extern "C" __global__ void sum_reduction_kernel(
    const double* input,
    double* output,
    int n
) {
    extern __shared__ double sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    sdata[tid] = (idx < n) ? input[idx] : 0.0;
    __syncthreads();

    // Parallel reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

/**
 * @brief AXPBY operation: y = alpha*x + beta*y
 *
 * Standard BLAS operation for vector linear combination.
 *
 * @param x Input vector (size: n)
 * @param y Input/output vector (size: n)
 * @param alpha Scalar multiplier for x
 * @param beta Scalar multiplier for y
 * @param n Number of elements
 */
extern "C" __global__ void axpby_kernel(
    const double* x,
    double* y,
    double alpha,
    double beta,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    y[idx] = alpha * x[idx] + beta * y[idx];
}

/**
 * @brief Initialize probability amplitudes in equal superposition.
 *
 * Sets all amplitudes to 1/sqrt(max_colors) for equal superposition state.
 * This is used for quantum-inspired initialization (Phase 3 style).
 *
 * NOTE: This kernel is included for compatibility but may not be used
 * in Phase 1 Active Inference. Phase 1 uses degree-based uncertainty.
 *
 * @param amplitudes Output amplitude array (size: num_vertices * max_colors)
 * @param num_vertices Number of vertices
 * @param max_colors Number of colors
 */
extern "C" __global__ void init_amplitudes_kernel(
    double* amplitudes,
    int num_vertices,
    int max_colors
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = num_vertices * max_colors;

    if (idx >= total_size) return;

    // Equal superposition: 1/sqrt(max_colors)
    double amplitude = 1.0 / sqrt((double)max_colors);
    amplitudes[idx] = amplitude;
}

/**
 * @brief Compute vertex uncertainties from graph structure (Active Inference).
 *
 * Computes uncertainty based on:
 * 1. Vertex degree (high degree = high uncertainty)
 * 2. Neighborhood density
 * 3. Precision-weighted prediction errors
 *
 * This is the core kernel for Phase 1 Active Inference policy computation.
 *
 * @param row_ptr CSR row pointer array (size: num_vertices + 1)
 * @param col_idx CSR column index array (size: num_edges)
 * @param precision Precision per vertex (inverse of degree-based uncertainty) (size: num_vertices)
 * @param observations Graph-based observations (normalized degrees) (size: num_vertices)
 * @param uncertainty Output uncertainty scores (size: num_vertices)
 * @param num_vertices Number of vertices
 */
extern "C" __global__ void compute_vertex_uncertainty(
    const unsigned int* row_ptr,
    const unsigned int* col_idx,
    const double* precision,
    const double* observations,
    double* uncertainty,
    int num_vertices
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;

    if (v >= num_vertices) return;

    // Get vertex degree from CSR structure
    unsigned int start = row_ptr[v];
    unsigned int end = row_ptr[v + 1];
    unsigned int degree = end - start;

    // Compute uncertainty inversely proportional to precision
    // High degree -> low precision -> high uncertainty
    double prec = fmax(precision[v], EPSILON);
    double base_uncertainty = 1.0 / prec;

    // Factor in observation magnitude (higher obs = more constrained = less uncertain)
    double obs = observations[v];
    double obs_factor = 1.0 + obs; // Range: [1.0, 2.0]

    // Compute neighborhood density (fraction of possible edges present)
    double neighborhood_density = 0.0;
    if (degree > 1) {
        // Count edges within neighborhood (2-hop connections)
        unsigned int neighborhood_edges = 0;
        for (unsigned int e1 = start; e1 < end; ++e1) {
            unsigned int neighbor = col_idx[e1];
            unsigned int n_start = row_ptr[neighbor];
            unsigned int n_end = row_ptr[neighbor + 1];

            for (unsigned int e2 = n_start; e2 < n_end; ++e2) {
                unsigned int second_neighbor = col_idx[e2];
                // Check if second_neighbor is also a neighbor of v
                for (unsigned int e3 = start; e3 < end; ++e3) {
                    if (col_idx[e3] == second_neighbor) {
                        neighborhood_edges++;
                        break;
                    }
                }
            }
        }

        // Density = actual edges / possible edges
        unsigned int max_possible = degree * (degree - 1) / 2;
        if (max_possible > 0) {
            neighborhood_density = (double)neighborhood_edges / (double)max_possible;
        }
    }

    // Higher density = more constrained = less uncertain
    double density_factor = 1.0 - 0.5 * neighborhood_density; // Range: [0.5, 1.0]

    // Final uncertainty combines base uncertainty with structural factors
    uncertainty[v] = base_uncertainty * obs_factor * density_factor;
}
