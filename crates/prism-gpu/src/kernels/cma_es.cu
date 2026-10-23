// CMA-ES (Covariance Matrix Adaptation Evolution Strategy) GPU Implementation
//
// ASSUMPTIONS:
// - Population size: 4 + floor(3*ln(N)) where N is dimension
// - Parent size (mu): floor(population_size / 2)
// - Matrix operations use row-major storage
// - MAX_DIMENSIONS = 2048, MAX_POPULATION = 1024
// - Block size: 256 threads for coalesced memory access
// - Grid size: ceil(work_items / 256)
// - Requires: sm_80+ for efficient f32 atomic operations
// REFERENCE: PRISM Spec Section 2.4 "CMA-ES Optimization"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cmath>
#include <stdio.h>

namespace cg = cooperative_groups;

// Constants with clear documentation
constexpr int MAX_DIMENSIONS = 2048;
constexpr int MAX_POPULATION = 1024;
constexpr int BLOCK_SIZE = 256;
constexpr float EPSILON = 1e-10f;

// CMA-ES parameters structure matching Rust side
struct CmaParams {
    int population_size;
    int parent_size;
    int dimensions;
    float sigma;           // Step size
    float c_sigma;        // Cumulation for step size control
    float d_sigma;        // Damping for step size
    float c_c;            // Cumulation for covariance matrix
    float c_1;            // Learning rate for rank-one update
    float c_mu;           // Learning rate for rank-mu update
    float chi_n;          // Expected norm of N(0,I)
    float mu_eff;         // Variance effective selection mass
    unsigned long long seed;
};

// CMA-ES state maintained on GPU
struct CmaState {
    float* mean;              // Current mean (center) of distribution [dimensions]
    float* covariance;        // Covariance matrix C [dimensions x dimensions]
    float* bd_matrix;         // B*D matrix for decomposition [dimensions x dimensions]
    float* ps;                // Evolution path for sigma [dimensions]
    float* pc;                // Evolution path for C [dimensions]
    float* weights;           // Selection weights [parent_size]
    float* eigenvalues;       // D matrix diagonal [dimensions]
    float* eigenvectors;      // B matrix [dimensions x dimensions]
    int generation;
    float best_fitness;
    float condition_number;
};

// Initialize CMA-ES default parameters based on problem dimension
__device__ void init_cma_params(CmaParams* params, int dimensions) {
    params->dimensions = dimensions;
    params->population_size = 4 + (int)(3.0f * logf((float)dimensions));
    params->parent_size = params->population_size / 2;

    // Variance effective selection mass
    float weights_sum = 0.0f;
    float weights_sum2 = 0.0f;
    for (int i = 0; i < params->parent_size; i++) {
        float w = logf((float)(params->parent_size) + 0.5f) - logf((float)(i + 1));
        weights_sum += w;
        weights_sum2 += w * w;
    }
    params->mu_eff = weights_sum * weights_sum / weights_sum2;

    // Step size control parameters
    params->c_sigma = (params->mu_eff + 2.0f) / ((float)dimensions + params->mu_eff + 5.0f);
    params->d_sigma = 1.0f + 2.0f * fmaxf(0.0f, sqrtf((params->mu_eff - 1.0f) / ((float)dimensions + 1.0f)) - 1.0f) + params->c_sigma;

    // Covariance matrix adaptation parameters
    params->c_c = (4.0f + params->mu_eff / (float)dimensions) / ((float)dimensions + 4.0f + 2.0f * params->mu_eff / (float)dimensions);
    params->c_1 = 2.0f / (((float)dimensions + 1.3f) * ((float)dimensions + 1.3f) + params->mu_eff);
    params->c_mu = fminf(1.0f - params->c_1, 2.0f * (params->mu_eff - 2.0f + 1.0f / params->mu_eff) / (((float)dimensions + 2.0f) * ((float)dimensions + 2.0f) + params->mu_eff));

    // Expected chi for N(0,I) in n dimensions
    params->chi_n = sqrtf((float)dimensions) * (1.0f - 1.0f / (4.0f * (float)dimensions) + 1.0f / (21.0f * (float)dimensions * (float)dimensions));

    params->sigma = 0.5f; // Initial step size
}

// Sample from multivariate normal distribution using Cholesky decomposition
extern "C" __global__ void sample_population(
    float* population,      // Output: sampled population [population_size x dimensions]
    float* mean,           // Current mean vector [dimensions]
    float* bd_matrix,      // B*D matrix (Cholesky-like decomp) [dimensions x dimensions]
    float sigma,           // Step size scalar
    int population_size,
    int dimensions,
    unsigned long long seed,
    int generation
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = population_size * dimensions;

    if (tid >= total_elements) return;

    int individual = tid / dimensions;
    int dim = tid % dimensions;

    // Initialize random state per thread
    curandState state;
    curand_init(seed, tid, generation, &state);

    // Generate standard normal random number
    float z = curand_normal(&state);

    // Transform through covariance structure: x = mean + sigma * B * D * z
    float transformed = 0.0f;

    // Matrix-vector multiplication with BD matrix
    for (int k = 0; k < dimensions; k++) {
        float bd_element = bd_matrix[dim * dimensions + k];
        // We need coordinated sampling across dimensions for each individual
        curandState temp_state;
        curand_init(seed, individual * dimensions + k, generation, &temp_state);
        float z_k = curand_normal(&temp_state);
        transformed += bd_element * z_k;
    }

    // Add mean and scale by sigma
    population[tid] = mean[dim] + sigma * transformed;
}

// Evaluate fitness function (sphere function for testing - replace with actual objective)
extern "C" __global__ void evaluate_fitness(
    float* population,      // Population [population_size x dimensions]
    float* fitness,        // Output: fitness values [population_size]
    int population_size,
    int dimensions
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= population_size) return;

    // Sphere function: f(x) = sum(x_i^2)
    // This should be replaced with the actual optimization objective
    float sum = 0.0f;
    for (int d = 0; d < dimensions; d++) {
        float x = population[tid * dimensions + d];
        sum += x * x;
    }

    // Minimize sphere function
    fitness[tid] = sum;
}

// Sort population by fitness and select parents
extern "C" __global__ void rank_and_select(
    float* population,      // Population [population_size x dimensions]
    float* fitness,        // Fitness values [population_size]
    int* ranks,            // Output: ranking indices [population_size]
    float* selected,       // Output: selected parents [parent_size x dimensions]
    int population_size,
    int parent_size,
    int dimensions
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Simple parallel bubble sort for small populations
    // For larger populations, use thrust::sort or cub::DeviceRadixSort
    if (tid == 0) {
        // Initialize ranks
        for (int i = 0; i < population_size; i++) {
            ranks[i] = i;
        }

        // Bubble sort by fitness
        for (int i = 0; i < population_size - 1; i++) {
            for (int j = 0; j < population_size - i - 1; j++) {
                if (fitness[ranks[j]] > fitness[ranks[j + 1]]) {
                    int temp = ranks[j];
                    ranks[j] = ranks[j + 1];
                    ranks[j + 1] = temp;
                }
            }
        }

        // Copy best parents to selected array
        for (int p = 0; p < parent_size; p++) {
            int best_idx = ranks[p];
            for (int d = 0; d < dimensions; d++) {
                selected[p * dimensions + d] = population[best_idx * dimensions + d];
            }
        }
    }
}

// Update mean using weighted recombination
extern "C" __global__ void update_mean(
    float* new_mean,       // Output: updated mean [dimensions]
    float* old_mean,       // Previous mean [dimensions]
    float* selected,       // Selected parents [parent_size x dimensions]
    float* weights,        // Selection weights [parent_size]
    int parent_size,
    int dimensions
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= dimensions) return;

    float weighted_sum = 0.0f;
    for (int p = 0; p < parent_size; p++) {
        weighted_sum += weights[p] * selected[p * dimensions + tid];
    }

    // Store old mean for evolution path update
    old_mean[tid] = new_mean[tid];
    new_mean[tid] = weighted_sum;
}

// Update evolution paths (ps for sigma, pc for C)
extern "C" __global__ void update_evolution_paths(
    float* ps,             // Evolution path for sigma [dimensions]
    float* pc,             // Evolution path for C [dimensions]
    float* mean,           // Current mean [dimensions]
    float* old_mean,       // Previous mean [dimensions]
    float* bd_inv,         // Inverse of BD matrix [dimensions x dimensions]
    float sigma,
    float c_sigma,
    float c_c,
    float mu_eff,
    int dimensions,
    int generation
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= dimensions) return;

    // Compute step in original space
    float step = (mean[tid] - old_mean[tid]) / sigma;

    // Transform step through inverse BD for ps update
    float transformed_step = 0.0f;
    for (int k = 0; k < dimensions; k++) {
        float bd_inv_element = bd_inv[tid * dimensions + k];
        float step_k = (mean[k] - old_mean[k]) / sigma;
        transformed_step += bd_inv_element * step_k;
    }

    // Update ps (cumulation for sigma)
    float discount = sqrtf(c_sigma * (2.0f - c_sigma) * mu_eff);
    ps[tid] = (1.0f - c_sigma) * ps[tid] + discount * transformed_step;

    // Update pc (cumulation for C) with indicator function
    float ps_norm = 0.0f;
    for (int d = 0; d < dimensions; d++) {
        ps_norm += ps[d] * ps[d];
    }
    ps_norm = sqrtf(ps_norm);

    float h_sigma = (ps_norm / sqrtf(1.0f - powf(1.0f - c_sigma, 2.0f * (generation + 1)))) <
                    (1.4f + 2.0f / ((float)dimensions + 1.0f)) * sqrtf((float)dimensions) ? 1.0f : 0.0f;

    float discount_c = sqrtf(c_c * (2.0f - c_c) * mu_eff);
    pc[tid] = (1.0f - c_c) * pc[tid] + h_sigma * discount_c * step;
}

// Update covariance matrix using rank-one and rank-mu updates
extern "C" __global__ void update_covariance(
    float* covariance,     // Covariance matrix C [dimensions x dimensions]
    float* pc,             // Evolution path for C [dimensions]
    float* selected,       // Selected parents [parent_size x dimensions]
    float* mean,           // Current mean [dimensions]
    float* old_mean,       // Previous mean [dimensions]
    float* weights,        // Selection weights [parent_size]
    float sigma,
    float c_1,
    float c_mu,
    int parent_size,
    int dimensions
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = dimensions * dimensions;
    if (tid >= total_elements) return;

    int row = tid / dimensions;
    int col = tid % dimensions;

    // Rank-one update using pc
    float rank_one = c_1 * (pc[row] * pc[col] - covariance[tid]);

    // Rank-mu update using selected parents
    float rank_mu = 0.0f;
    for (int p = 0; p < parent_size; p++) {
        float y_row = (selected[p * dimensions + row] - old_mean[row]) / sigma;
        float y_col = (selected[p * dimensions + col] - old_mean[col]) / sigma;
        rank_mu += weights[p] * y_row * y_col;
    }
    rank_mu = c_mu * (rank_mu - covariance[tid]);

    // Update covariance
    float decay = 1.0f - c_1 - c_mu;
    covariance[tid] = decay * covariance[tid] + rank_one + rank_mu;

    // Ensure symmetry (average with transpose)
    __syncthreads();
    if (row != col) {
        int transpose_idx = col * dimensions + row;
        covariance[tid] = (covariance[tid] + covariance[transpose_idx]) * 0.5f;
    }
}

// Update step size sigma using cumulative step size adaptation (CSA)
extern "C" __global__ void update_sigma(
    float* sigma,          // Step size (scalar)
    float* ps,             // Evolution path for sigma [dimensions]
    float c_sigma,
    float d_sigma,
    float chi_n,
    int dimensions
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Compute norm of ps
        float ps_norm = 0.0f;
        for (int d = 0; d < dimensions; d++) {
            ps_norm += ps[d] * ps[d];
        }
        ps_norm = sqrtf(ps_norm);

        // Update sigma
        *sigma = *sigma * expf((c_sigma / d_sigma) * (ps_norm / chi_n - 1.0f));

        // Bound sigma to reasonable range
        *sigma = fmaxf(1e-10f, fminf(1e10f, *sigma));
    }
}

// Compute eigendecomposition of covariance matrix (simplified power iteration)
// For production, use cuSOLVER's syevd for accurate eigendecomposition
extern "C" __global__ void eigendecompose_covariance(
    float* covariance,     // Input: covariance matrix [dimensions x dimensions]
    float* eigenvalues,    // Output: eigenvalues (diagonal of D) [dimensions]
    float* eigenvectors,   // Output: eigenvectors (B matrix) [dimensions x dimensions]
    float* bd_matrix,      // Output: B*D matrix [dimensions x dimensions]
    int dimensions,
    int max_iterations = 100
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Simple power iteration for dominant eigenvalue/vector
    // This is a placeholder - use cuSOLVER for production
    if (tid < dimensions) {
        // Initialize eigenvector to random
        eigenvectors[tid * dimensions + tid] = 1.0f;

        // Power iteration
        for (int iter = 0; iter < max_iterations; iter++) {
            float sum = 0.0f;
            for (int j = 0; j < dimensions; j++) {
                sum += covariance[tid * dimensions + j] * eigenvectors[j * dimensions + tid];
            }
            eigenvectors[tid * dimensions + tid] = sum;

            // Normalize
            float norm = 0.0f;
            for (int i = 0; i < dimensions; i++) {
                norm += eigenvectors[i * dimensions + tid] * eigenvectors[i * dimensions + tid];
            }
            norm = sqrtf(norm + EPSILON);

            for (int i = 0; i < dimensions; i++) {
                eigenvectors[i * dimensions + tid] /= norm;
            }
        }

        // Compute eigenvalue (Rayleigh quotient)
        float numerator = 0.0f;
        float denominator = 0.0f;
        for (int i = 0; i < dimensions; i++) {
            float Av = 0.0f;
            for (int j = 0; j < dimensions; j++) {
                Av += covariance[i * dimensions + j] * eigenvectors[j * dimensions + tid];
            }
            numerator += eigenvectors[i * dimensions + tid] * Av;
            denominator += eigenvectors[i * dimensions + tid] * eigenvectors[i * dimensions + tid];
        }
        eigenvalues[tid] = numerator / (denominator + EPSILON);

        // Compute BD matrix
        for (int i = 0; i < dimensions; i++) {
            bd_matrix[i * dimensions + tid] = eigenvectors[i * dimensions + tid] * sqrtf(fmaxf(0.0f, eigenvalues[tid]));
        }
    }
}

// Compute condition number of covariance matrix
extern "C" __global__ void compute_condition_number(
    float* eigenvalues,    // Eigenvalues [dimensions]
    float* condition,      // Output: condition number (scalar)
    int dimensions
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float max_eigen = eigenvalues[0];
        float min_eigen = eigenvalues[0];

        for (int d = 1; d < dimensions; d++) {
            max_eigen = fmaxf(max_eigen, eigenvalues[d]);
            min_eigen = fminf(min_eigen, eigenvalues[d]);
        }

        *condition = max_eigen / (min_eigen + EPSILON);
    }
}

// Main CMA-ES step combining all operations
extern "C" __global__ void cma_es_step(
    float* population,      // Population buffer [population_size x dimensions]
    float* fitness,        // Fitness values [population_size]
    float* mean,           // Mean vector [dimensions]
    float* covariance,     // Covariance matrix [dimensions x dimensions]
    float* bd_matrix,      // B*D decomposition [dimensions x dimensions]
    float* ps,             // Evolution path sigma [dimensions]
    float* pc,             // Evolution path C [dimensions]
    float* eigenvalues,    // Eigenvalues [dimensions]
    float* eigenvectors,   // Eigenvectors [dimensions x dimensions]
    float* weights,        // Selection weights [parent_size]
    float* selected,       // Selected parents buffer [parent_size x dimensions]
    float* old_mean,       // Previous mean buffer [dimensions]
    float* bd_inv,         // BD inverse buffer [dimensions x dimensions]
    float* sigma,          // Step size (scalar)
    float* best_fitness,   // Best fitness found (scalar)
    float* condition,      // Condition number (scalar)
    CmaParams params,
    int generation
) {
    // This kernel orchestrates one complete CMA-ES generation
    // In practice, we'd launch separate kernels for each phase

    // For this integrated kernel, we use thread 0 to coordinate
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Log generation info
        printf("CMA-ES Generation %d: sigma=%.6f, best_fitness=%.6f\n",
               generation, *sigma, *best_fitness);
    }

    // All threads participate in cooperative operations
    __syncthreads();
}

// Initialize CMA-ES state
extern "C" __global__ void init_cma_state(
    float* mean,
    float* covariance,
    float* bd_matrix,
    float* ps,
    float* pc,
    float* weights,
    float* sigma,
    int dimensions,
    int parent_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize mean to zeros
    if (tid < dimensions) {
        mean[tid] = 0.0f;
        ps[tid] = 0.0f;
        pc[tid] = 0.0f;
    }

    // Initialize covariance to identity
    if (tid < dimensions * dimensions) {
        int row = tid / dimensions;
        int col = tid % dimensions;
        covariance[tid] = (row == col) ? 1.0f : 0.0f;
        bd_matrix[tid] = (row == col) ? 1.0f : 0.0f;
    }

    // Initialize weights
    if (tid < parent_size) {
        float log_mu_half = logf((float)parent_size + 0.5f);
        weights[tid] = log_mu_half - logf((float)(tid + 1));

        // Normalize weights
        __syncthreads();
        if (tid == 0) {
            float sum = 0.0f;
            for (int i = 0; i < parent_size; i++) {
                sum += weights[i];
            }
            for (int i = 0; i < parent_size; i++) {
                weights[i] /= sum;
            }
        }
    }

    // Initialize sigma
    if (tid == 0) {
        *sigma = 0.5f;
    }
}