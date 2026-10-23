// Ensemble Exchange kernel for CMA-ES replica management
//
// ASSUMPTIONS:
// - Population stored as contiguous f32 arrays
// - MAX_POPULATION = 1024 (CMA-ES population size)
// - MAX_DIMENSIONS = 2048 (problem dimensionality)
// - MAX_REPLICAS = 64 (parallel CMA-ES instances)
// - Precision: f32 for efficiency, f64 for high-precision mode
// - Block size: 256 threads (optimal for memory bandwidth)
// - Grid size: Variable based on population and replicas
// - Requires: sm_80+ for efficient matrix operations
// REFERENCE: PRISM Spec Section 2.4 "CMA-ES Ensemble Exchange"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cmath>

// Configuration constants
constexpr int MAX_POPULATION = 1024;
constexpr int MAX_DIMENSIONS = 2048;
constexpr int MAX_REPLICAS = 64;
constexpr int THREADS_PER_BLOCK = 256;
constexpr int WARP_SIZE = 32;
constexpr float EPSILON = 1e-8f;

// CMA-ES ensemble parameters
struct CmaEnsembleParams {
    int num_replicas;         // Number of parallel CMA-ES instances
    int population_size;      // Population per replica (λ)
    int parent_size;          // Parents selected (μ)
    int dimensions;           // Problem dimensionality (n)
    float sigma;              // Step size
    float c_sigma;            // Cumulation constant for sigma
    float d_sigma;            // Damping for sigma
    float c_c;                // Cumulation constant for C
    float c_1;                // Rank-one update weight
    float c_mu;               // Rank-μ update weight
    float chi_n;              // Expected norm of N(0,I)
    unsigned long seed;       // Random seed
    int exchange_interval;    // Steps between exchanges
    float exchange_rate;      // Fraction of population to exchange
};

// Device function: Matrix-vector multiplication
__device__ void matvec_mul(
    const float* __restrict__ mat, // [n x n]
    const float* __restrict__ vec, // [n]
    float* __restrict__ result,    // [n]
    int n
) {
    for (int i = 0; i < n; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < n; ++j) {
            sum += mat[i * n + j] * vec[j];
        }
        result[i] = sum;
    }
}

// Device function: Outer product update
__device__ void rank_one_update(
    float* __restrict__ mat,     // [n x n]
    const float* __restrict__ vec, // [n]
    float weight,
    int n
) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            mat[i * n + j] += weight * vec[i] * vec[j];
        }
    }
}

// Device function: Evaluate fitness (placeholder - would be problem-specific)
__device__ float evaluate_fitness(
    const float* __restrict__ solution,
    int dimensions
) {
    // Sphere function as placeholder
    float sum = 0.0f;
    for (int i = 0; i < dimensions; ++i) {
        sum += solution[i] * solution[i];
    }
    return sum;
}

// Main CMA-ES evolution kernel for single replica
extern "C" __global__ void cma_evolution_kernel(
    float* __restrict__ populations,      // [num_replicas][population_size][dimensions]
    float* __restrict__ fitness_values,   // [num_replicas][population_size]
    float* __restrict__ mean_vectors,     // [num_replicas][dimensions]
    float* __restrict__ covariance_matrices, // [num_replicas][dimensions][dimensions]
    float* __restrict__ evolution_paths_sigma, // [num_replicas][dimensions]
    float* __restrict__ evolution_paths_c,     // [num_replicas][dimensions]
    float* __restrict__ sigmas,           // [num_replicas] step sizes
    int* __restrict__ generations,        // [num_replicas] generation counters
    CmaEnsembleParams params
) {
    // Each block handles one replica
    int replica_id = blockIdx.x;
    if (replica_id >= params.num_replicas) return;

    // Thread ID within block
    int tid = threadIdx.x;

    // Shared memory for collaborative operations
    extern __shared__ float shared_mem[];
    float* shared_fitness = &shared_mem[0];
    int* shared_indices = (int*)&shared_mem[params.population_size];

    // Initialize random state
    curandState rand_state;
    curand_init(params.seed + replica_id * 1000 + tid, 0, 0, &rand_state);

    // Pointers to replica-specific data
    int replica_offset = replica_id * params.population_size * params.dimensions;
    float* replica_pop = &populations[replica_offset];
    float* replica_fitness = &fitness_values[replica_id * params.population_size];
    float* replica_mean = &mean_vectors[replica_id * params.dimensions];
    float* replica_cov = &covariance_matrices[replica_id * params.dimensions * params.dimensions];
    float* replica_ps = &evolution_paths_sigma[replica_id * params.dimensions];
    float* replica_pc = &evolution_paths_c[replica_id * params.dimensions];
    float& replica_sigma = sigmas[replica_id];

    // Step 1: Sample new population
    if (tid < params.population_size) {
        float* individual = &replica_pop[tid * params.dimensions];

        // Generate standard normal vector
        float z_vector[MAX_DIMENSIONS];
        for (int i = 0; i < params.dimensions; ++i) {
            z_vector[i] = curand_normal(&rand_state);
        }

        // Transform: x = m + σ * C^(1/2) * z
        // Simplified: using diagonal approximation for now
        for (int i = 0; i < params.dimensions; ++i) {
            float cov_sqrt = sqrtf(replica_cov[i * params.dimensions + i] + EPSILON);
            individual[i] = replica_mean[i] + replica_sigma * cov_sqrt * z_vector[i];
        }

        // Evaluate fitness
        replica_fitness[tid] = evaluate_fitness(individual, params.dimensions);
        shared_fitness[tid] = replica_fitness[tid];
        shared_indices[tid] = tid;
    }
    __syncthreads();

    // Step 2: Sort by fitness (simple parallel bubble sort for small populations)
    for (int phase = 0; phase < params.population_size; ++phase) {
        if (tid < params.population_size - 1) {
            if ((phase % 2 == 0 && tid % 2 == 0) ||
                (phase % 2 == 1 && tid % 2 == 1)) {
                if (shared_fitness[tid] > shared_fitness[tid + 1]) {
                    // Swap fitness
                    float temp_fit = shared_fitness[tid];
                    shared_fitness[tid] = shared_fitness[tid + 1];
                    shared_fitness[tid + 1] = temp_fit;

                    // Swap indices
                    int temp_idx = shared_indices[tid];
                    shared_indices[tid] = shared_indices[tid + 1];
                    shared_indices[tid + 1] = temp_idx;
                }
            }
        }
        __syncthreads();
    }

    // Step 3: Update mean (only thread 0)
    if (tid == 0) {
        // Compute weights for recombination
        float weights[MAX_POPULATION];
        float weight_sum = 0.0f;

        for (int i = 0; i < params.parent_size; ++i) {
            weights[i] = logf(params.parent_size + 0.5f) - logf(i + 1.0f);
            weight_sum += weights[i];
        }

        // Normalize weights
        for (int i = 0; i < params.parent_size; ++i) {
            weights[i] /= weight_sum;
        }

        // Update mean: m = Σ w_i * x_i
        float new_mean[MAX_DIMENSIONS];
        for (int d = 0; d < params.dimensions; ++d) {
            new_mean[d] = 0.0f;
            for (int i = 0; i < params.parent_size; ++i) {
                int idx = shared_indices[i];
                new_mean[d] += weights[i] * replica_pop[idx * params.dimensions + d];
            }
        }

        // Step 4: Update evolution paths
        float displacement[MAX_DIMENSIONS];
        for (int d = 0; d < params.dimensions; ++d) {
            displacement[d] = (new_mean[d] - replica_mean[d]) / replica_sigma;
        }

        // Update p_sigma
        float ps_decay = 1.0f - params.c_sigma;
        float ps_update = sqrtf(params.c_sigma * (2.0f - params.c_sigma));
        for (int d = 0; d < params.dimensions; ++d) {
            replica_ps[d] = ps_decay * replica_ps[d] + ps_update * displacement[d];
        }

        // Update sigma
        float ps_norm = 0.0f;
        for (int d = 0; d < params.dimensions; ++d) {
            ps_norm += replica_ps[d] * replica_ps[d];
        }
        ps_norm = sqrtf(ps_norm);

        replica_sigma *= expf(params.c_sigma / params.d_sigma *
                             (ps_norm / params.chi_n - 1.0f));

        // Update p_c
        float pc_decay = 1.0f - params.c_c;
        float pc_update = sqrtf(params.c_c * (2.0f - params.c_c));
        for (int d = 0; d < params.dimensions; ++d) {
            replica_pc[d] = pc_decay * replica_pc[d] + pc_update * displacement[d];
        }

        // Step 5: Update covariance (simplified - rank-1 update only)
        rank_one_update(replica_cov, replica_pc, params.c_1, params.dimensions);

        // Store new mean
        for (int d = 0; d < params.dimensions; ++d) {
            replica_mean[d] = new_mean[d];
        }

        // Increment generation counter
        atomicAdd(&generations[replica_id], 1);
    }
}

// Replica exchange kernel
extern "C" __global__ void replica_exchange_kernel(
    float* __restrict__ populations,
    float* __restrict__ fitness_values,
    float* __restrict__ mean_vectors,
    float* __restrict__ sigmas,
    float* __restrict__ exchange_matrix, // [num_replicas][num_replicas] exchange rates
    int* __restrict__ exchange_counts,
    CmaEnsembleParams params
) {
    // Each thread handles one replica pair
    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_pairs = params.num_replicas * (params.num_replicas - 1) / 2;
    if (pair_idx >= num_pairs) return;

    // Map pair index to replica indices
    int replica1 = 0, replica2 = 0;
    int count = 0;
    for (int i = 0; i < params.num_replicas; ++i) {
        for (int j = i + 1; j < params.num_replicas; ++j) {
            if (count == pair_idx) {
                replica1 = i;
                replica2 = j;
                break;
            }
            count++;
        }
        if (replica1 != replica2) break;
    }

    // Initialize random state
    curandState rand_state;
    curand_init(params.seed + pair_idx * 2000, 0, 0, &rand_state);

    // Get best fitness for each replica
    float best_fitness1 = fitness_values[replica1 * params.population_size];
    float best_fitness2 = fitness_values[replica2 * params.population_size];

    // Compute exchange probability based on fitness difference
    float fitness_diff = fabsf(best_fitness1 - best_fitness2);
    float exchange_prob = expf(-fitness_diff / (sigmas[replica1] + sigmas[replica2]));

    // Store exchange rate for monitoring
    exchange_matrix[replica1 * params.num_replicas + replica2] = exchange_prob;
    exchange_matrix[replica2 * params.num_replicas + replica1] = exchange_prob;

    // Perform exchange with probability
    if (curand_uniform(&rand_state) < exchange_prob * params.exchange_rate) {
        // Exchange best individuals
        int offset1 = replica1 * params.population_size * params.dimensions;
        int offset2 = replica2 * params.population_size * params.dimensions;

        for (int i = 0; i < params.dimensions; ++i) {
            float temp = populations[offset1 + i];
            populations[offset1 + i] = populations[offset2 + i];
            populations[offset2 + i] = temp;
        }

        // Optionally exchange means
        int mean_offset1 = replica1 * params.dimensions;
        int mean_offset2 = replica2 * params.dimensions;

        for (int i = 0; i < params.dimensions; ++i) {
            float temp = mean_vectors[mean_offset1 + i];
            mean_vectors[mean_offset1 + i] = mean_vectors[mean_offset2 + i];
            mean_vectors[mean_offset2 + i] = temp;
        }

        // Update exchange counter
        atomicAdd(&exchange_counts[pair_idx], 1);
    }
}

// Migration kernel for island model
extern "C" __global__ void island_migration_kernel(
    float* __restrict__ populations,
    float* __restrict__ fitness_values,
    int* __restrict__ topology_matrix, // [num_replicas][num_replicas] connectivity
    float* __restrict__ migration_buffer,
    CmaEnsembleParams params,
    int migration_size
) {
    int replica_id = blockIdx.x;
    if (replica_id >= params.num_replicas) return;

    int tid = threadIdx.x;
    if (tid >= migration_size) return;

    // Initialize random state
    curandState rand_state;
    curand_init(params.seed + replica_id * 3000 + tid, 0, 0, &rand_state);

    // Select individuals for migration (best individuals)
    if (tid < migration_size) {
        int individual_idx = tid; // Migrate top individuals
        int offset = replica_id * params.population_size * params.dimensions +
                    individual_idx * params.dimensions;

        // Copy to migration buffer
        int buffer_offset = replica_id * migration_size * params.dimensions +
                          tid * params.dimensions;
        for (int i = 0; i < params.dimensions; ++i) {
            migration_buffer[buffer_offset + i] = populations[offset + i];
        }
    }
    __syncthreads();

    // Receive migrants from connected replicas
    if (tid == 0) {
        for (int source = 0; source < params.num_replicas; ++source) {
            if (source == replica_id) continue;

            // Check if connected in topology
            if (topology_matrix[source * params.num_replicas + replica_id] > 0) {
                // Replace worst individuals with migrants
                for (int m = 0; m < migration_size; ++m) {
                    int worst_idx = params.population_size - 1 - m;
                    int pop_offset = replica_id * params.population_size * params.dimensions +
                                    worst_idx * params.dimensions;
                    int buffer_offset = source * migration_size * params.dimensions +
                                      m * params.dimensions;

                    for (int i = 0; i < params.dimensions; ++i) {
                        populations[pop_offset + i] = migration_buffer[buffer_offset + i];
                    }
                }
            }
        }
    }
}

// Diversity maintenance kernel
extern "C" __global__ void diversity_kernel(
    float* __restrict__ populations,
    float* __restrict__ diversity_metrics, // [num_replicas] diversity scores
    float* __restrict__ crowding_distances, // [num_replicas][population_size]
    CmaEnsembleParams params
) {
    int replica_id = blockIdx.x;
    if (replica_id >= params.num_replicas) return;

    int tid = threadIdx.x;

    extern __shared__ float shared_distances[];

    // Compute pairwise distances for diversity
    if (tid < params.population_size) {
        float min_distance = 1e10f;

        int my_offset = replica_id * params.population_size * params.dimensions +
                       tid * params.dimensions;

        for (int j = 0; j < params.population_size; ++j) {
            if (j == tid) continue;

            int other_offset = replica_id * params.population_size * params.dimensions +
                             j * params.dimensions;

            float dist = 0.0f;
            for (int d = 0; d < params.dimensions; ++d) {
                float diff = populations[my_offset + d] - populations[other_offset + d];
                dist += diff * diff;
            }
            dist = sqrtf(dist);

            min_distance = fminf(min_distance, dist);
        }

        shared_distances[tid] = min_distance;
        crowding_distances[replica_id * params.population_size + tid] = min_distance;
    }
    __syncthreads();

    // Compute average diversity (thread 0 only)
    if (tid == 0) {
        float total_diversity = 0.0f;
        for (int i = 0; i < params.population_size; ++i) {
            total_diversity += shared_distances[i];
        }
        diversity_metrics[replica_id] = total_diversity / params.population_size;
    }
}

// Adaptive parameter control kernel
extern "C" __global__ void adaptive_parameters_kernel(
    float* __restrict__ sigmas,
    float* __restrict__ fitness_values,
    float* __restrict__ diversity_metrics,
    float* __restrict__ success_rates, // [num_replicas] fraction of improving offspring
    CmaEnsembleParams* params_array, // Mutable parameters per replica
    int generation
) {
    int replica_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (replica_id >= params_array[0].num_replicas) return;

    // Get current performance metrics
    float diversity = diversity_metrics[replica_id];
    float success_rate = success_rates[replica_id];
    float current_sigma = sigmas[replica_id];

    // Adaptive sigma control based on success rate
    float target_success_rate = 0.2f; // 1/5 success rule variant

    if (success_rate > target_success_rate) {
        // Increase step size (exploration)
        sigmas[replica_id] = current_sigma * 1.1f;
    } else {
        // Decrease step size (exploitation)
        sigmas[replica_id] = current_sigma * 0.9f;
    }

    // Adjust parameters based on diversity
    if (diversity < 0.01f) {
        // Low diversity - increase mutation
        params_array[replica_id].c_1 *= 0.95f;
        params_array[replica_id].c_mu *= 0.95f;
    } else if (diversity > 1.0f) {
        // High diversity - increase selection pressure
        params_array[replica_id].c_1 *= 1.05f;
        params_array[replica_id].c_mu *= 1.05f;
    }

    // Clamp parameters to reasonable ranges
    sigmas[replica_id] = fmaxf(1e-6f, fminf(1e3f, sigmas[replica_id]));
    params_array[replica_id].c_1 = fmaxf(0.0f, fminf(1.0f, params_array[replica_id].c_1));
    params_array[replica_id].c_mu = fmaxf(0.0f, fminf(1.0f, params_array[replica_id].c_mu));
}

// Convergence detection kernel
extern "C" __global__ void convergence_detection_kernel(
    float* __restrict__ fitness_values,
    float* __restrict__ diversity_metrics,
    float* __restrict__ sigmas,
    int* __restrict__ convergence_flags, // [num_replicas] 1 if converged
    float* __restrict__ convergence_metrics, // [num_replicas] convergence score
    CmaEnsembleParams params,
    float fitness_tolerance,
    float diversity_threshold
) {
    int replica_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (replica_id >= params.num_replicas) return;

    // Check multiple convergence criteria
    float best_fitness = fitness_values[replica_id * params.population_size];
    float worst_fitness = fitness_values[replica_id * params.population_size +
                                       params.population_size - 1];
    float fitness_range = worst_fitness - best_fitness;

    float diversity = diversity_metrics[replica_id];
    float sigma = sigmas[replica_id];

    // Convergence criteria
    bool fitness_converged = fitness_range < fitness_tolerance;
    bool diversity_converged = diversity < diversity_threshold;
    bool sigma_converged = sigma < 1e-5f;

    // Combined convergence score
    float convergence_score = 0.0f;
    if (fitness_converged) convergence_score += 0.4f;
    if (diversity_converged) convergence_score += 0.3f;
    if (sigma_converged) convergence_score += 0.3f;

    convergence_metrics[replica_id] = convergence_score;
    convergence_flags[replica_id] = (convergence_score > 0.8f) ? 1 : 0;
}

// Performance monitoring kernel
extern "C" __global__ void cma_performance_metrics(
    const float* __restrict__ fitness_values,
    const float* __restrict__ diversity_metrics,
    const int* __restrict__ exchange_counts,
    const int* __restrict__ generations,
    float* __restrict__ metrics, // [best_fitness, avg_diversity, exchange_rate, convergence_speed]
    CmaEnsembleParams params
) {
    // Single thread computes aggregate metrics
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Find global best fitness
        float global_best = 1e10f;
        for (int r = 0; r < params.num_replicas; ++r) {
            float replica_best = fitness_values[r * params.population_size];
            global_best = fminf(global_best, replica_best);
        }

        // Average diversity
        float avg_diversity = 0.0f;
        for (int r = 0; r < params.num_replicas; ++r) {
            avg_diversity += diversity_metrics[r];
        }
        avg_diversity /= params.num_replicas;

        // Exchange success rate
        int total_exchanges = 0;
        int num_pairs = params.num_replicas * (params.num_replicas - 1) / 2;
        for (int p = 0; p < num_pairs; ++p) {
            total_exchanges += exchange_counts[p];
        }
        float exchange_rate = (float)total_exchanges /
                             (num_pairs * params.exchange_interval);

        // Average generation (convergence speed indicator)
        float avg_generation = 0.0f;
        for (int r = 0; r < params.num_replicas; ++r) {
            avg_generation += generations[r];
        }
        avg_generation /= params.num_replicas;

        metrics[0] = global_best;
        metrics[1] = avg_diversity;
        metrics[2] = exchange_rate;
        metrics[3] = 1.0f / (avg_generation + 1.0f); // Inverse for speed metric
    }
}