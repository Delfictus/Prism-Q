// Path Integral Monte Carlo (PIMC) kernel for quantum annealing
//
// ASSUMPTIONS:
// - Input states stored as contiguous f32 arrays
// - MAX_REPLICAS = 512 (multiple Trotter slices per thread block)
// - MAX_DIMENSIONS = 1024 (system size limit)
// - Precision: f32 for efficiency, f64 available via template
// - Block size: 256 threads (optimal for A100/RTX3060 coalesced access)
// - Grid size: ceil(num_replicas / threads_per_block)
// - Requires: sm_70+ for efficient atomic operations
// REFERENCE: PRISM Spec Section 3.3 "Quantum Annealing via PIMC"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>

// Configuration constants
constexpr int MAX_REPLICAS = 512;
constexpr int MAX_DIMENSIONS = 1024;
constexpr int THREADS_PER_BLOCK = 256;
constexpr int WARP_SIZE = 32;

// PIMC parameters structure
struct PimcParams {
    int num_replicas;      // Number of Trotter slices
    int dimensions;        // State space dimensionality
    float beta;            // Inverse temperature (1/kT)
    float delta_tau;       // Imaginary time step
    float transverse_field; // Quantum tunneling strength (Γ)
    float coupling_strength; // Inter-replica coupling (J)
    int mc_steps;          // Monte Carlo steps per kernel call
    unsigned long seed;    // Random seed for reproducibility
};

// Device function: Compute local energy for a configuration
__device__ float compute_local_energy(
    const float* __restrict__ state,
    const float* __restrict__ coupling_matrix,
    int dimensions,
    int idx
) {
    float energy = 0.0f;

    // Classical Ising energy: E = -Σ J_ij * s_i * s_j
    for (int j = 0; j < dimensions; ++j) {
        if (idx != j) {
            int coupling_idx = idx * dimensions + j;
            energy -= coupling_matrix[coupling_idx] * state[idx] * state[j];
        }
    }

    return energy;
}

// Device function: Compute quantum kinetic energy between replicas
__device__ float compute_kinetic_energy(
    const float* __restrict__ replica1,
    const float* __restrict__ replica2,
    int dimensions,
    float coupling_strength
) {
    float kinetic = 0.0f;

    // Quantum kinetic term: -J * Σ s_i^(k) * s_i^(k+1)
    for (int i = 0; i < dimensions; ++i) {
        kinetic -= coupling_strength * replica1[i] * replica2[i];
    }

    return kinetic;
}

// Device function: Metropolis-Hastings acceptance
__device__ bool metropolis_accept(
    float delta_energy,
    float beta,
    curandState* rand_state
) {
    if (delta_energy <= 0.0f) {
        return true;
    }

    float prob = expf(-beta * delta_energy);
    float rand = curand_uniform(rand_state);
    return rand < prob;
}

// Main PIMC evolution kernel
extern "C" __global__ void pimc_evolution_kernel(
    float* __restrict__ replicas,        // [num_replicas][dimensions]
    const float* __restrict__ coupling_matrix, // [dimensions][dimensions]
    float* __restrict__ energies,        // [num_replicas] energy per replica
    float* __restrict__ magnetizations,  // [num_replicas] order parameter
    float* __restrict__ acceptance_rates, // [num_replicas] MC acceptance
    PimcParams params
) {
    // Global thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= params.num_replicas) return;

    // Initialize random state
    curandState rand_state;
    curand_init(params.seed, tid, 0, &rand_state);

    // Shared memory for replica caching
    extern __shared__ float shared_mem[];
    float* local_replica = &shared_mem[threadIdx.x * params.dimensions];

    // Load replica into shared memory
    int replica_offset = tid * params.dimensions;
    for (int i = 0; i < params.dimensions; ++i) {
        local_replica[i] = replicas[replica_offset + i];
    }
    __syncthreads();

    // Monte Carlo evolution
    int accepted = 0;
    float total_energy = 0.0f;

    for (int step = 0; step < params.mc_steps; ++step) {
        // Choose random spin to flip
        int spin_idx = curand(&rand_state) % params.dimensions;

        // Compute energy change for spin flip
        float old_spin = local_replica[spin_idx];
        float new_spin = -old_spin; // Flip spin

        // Classical energy change
        float delta_classical = 0.0f;
        for (int j = 0; j < params.dimensions; ++j) {
            if (j != spin_idx) {
                int coupling_idx = spin_idx * params.dimensions + j;
                delta_classical += 2.0f * coupling_matrix[coupling_idx] *
                                  old_spin * local_replica[j];
            }
        }

        // Quantum tunneling term (transverse field)
        float delta_quantum = -2.0f * params.transverse_field * old_spin;

        // Inter-replica coupling (periodic boundary conditions)
        float delta_kinetic = 0.0f;
        if (params.num_replicas > 1) {
            int prev_replica = (tid - 1 + params.num_replicas) % params.num_replicas;
            int next_replica = (tid + 1) % params.num_replicas;

            // Contribution from neighboring replicas
            float prev_spin = replicas[prev_replica * params.dimensions + spin_idx];
            float next_spin = replicas[next_replica * params.dimensions + spin_idx];
            delta_kinetic = 2.0f * params.coupling_strength * old_spin *
                           (prev_spin + next_spin);
        }

        // Total energy change
        float delta_energy = delta_classical + delta_quantum + delta_kinetic;

        // Metropolis acceptance
        if (metropolis_accept(delta_energy, params.beta, &rand_state)) {
            local_replica[spin_idx] = new_spin;
            accepted++;
        }

        // Accumulate energy
        if (step % 10 == 0) { // Sample every 10 steps to reduce correlation
            total_energy += compute_local_energy(
                local_replica, coupling_matrix, params.dimensions, tid
            );
        }
    }

    // Write back updated replica
    for (int i = 0; i < params.dimensions; ++i) {
        replicas[replica_offset + i] = local_replica[i];
    }

    // Compute observables
    float magnetization = 0.0f;
    for (int i = 0; i < params.dimensions; ++i) {
        magnetization += local_replica[i];
    }
    magnetization /= params.dimensions;

    // Store results
    energies[tid] = total_energy / (params.mc_steps / 10);
    magnetizations[tid] = fabsf(magnetization);
    acceptance_rates[tid] = (float)accepted / params.mc_steps;
}

// Parallel tempering exchange kernel
extern "C" __global__ void replica_exchange_kernel(
    float* __restrict__ replicas,
    float* __restrict__ temperatures,
    float* __restrict__ energies,
    int* __restrict__ exchange_counts,
    PimcParams params
) {
    // Each thread handles one replica pair
    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_pairs = params.num_replicas / 2;
    if (pair_idx >= num_pairs) return;

    // Initialize random state
    curandState rand_state;
    curand_init(params.seed + 1000, pair_idx, 0, &rand_state);

    // Determine replica indices (even-odd pairing)
    int replica1 = pair_idx * 2;
    int replica2 = replica1 + 1;

    // Get temperatures and energies
    float T1 = temperatures[replica1];
    float T2 = temperatures[replica2];
    float E1 = energies[replica1];
    float E2 = energies[replica2];

    // Compute exchange probability (Metropolis criterion)
    float delta = (1.0f/T1 - 1.0f/T2) * (E2 - E1);

    if (metropolis_accept(-delta, 1.0f, &rand_state)) {
        // Exchange replicas
        int offset1 = replica1 * params.dimensions;
        int offset2 = replica2 * params.dimensions;

        for (int i = 0; i < params.dimensions; ++i) {
            float temp = replicas[offset1 + i];
            replicas[offset1 + i] = replicas[offset2 + i];
            replicas[offset2 + i] = temp;
        }

        // Update exchange counter
        atomicAdd(&exchange_counts[pair_idx], 1);
    }
}

// Quantum annealing schedule kernel
extern "C" __global__ void update_annealing_schedule(
    float* __restrict__ transverse_fields,
    float* __restrict__ temperatures,
    float annealing_time,
    float total_time,
    int num_replicas
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_replicas) return;

    // Linear annealing schedule: Γ(t) = Γ_0 * (1 - t/T)
    float s = annealing_time / total_time;
    float gamma = 1.0f * (1.0f - s); // Initial transverse field = 1.0

    transverse_fields[tid] = gamma;

    // Temperature schedule (geometric spacing)
    float T_min = 0.01f;
    float T_max = 10.0f;
    float alpha = powf(T_max / T_min, 1.0f / (num_replicas - 1));
    temperatures[tid] = T_min * powf(alpha, tid);
}

// Ensemble statistics kernel (for CMA-ES integration)
extern "C" __global__ void compute_ensemble_statistics(
    const float* __restrict__ replicas,
    const float* __restrict__ energies,
    float* __restrict__ mean_state,
    float* __restrict__ covariance,
    float* __restrict__ entropy,
    PimcParams params
) {
    // Cooperative groups would be ideal here for efficient reduction
    // Using block-level reduction for now

    extern __shared__ float shared_data[];

    int tid = threadIdx.x;
    int dim_idx = blockIdx.x;
    if (dim_idx >= params.dimensions) return;

    // Compute mean for this dimension
    float sum = 0.0f;
    for (int r = tid; r < params.num_replicas; r += blockDim.x) {
        sum += replicas[r * params.dimensions + dim_idx];
    }

    // Block reduction
    shared_data[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        mean_state[dim_idx] = shared_data[0] / params.num_replicas;
    }

    // Second pass for variance (simplified - full covariance needs more work)
    __syncthreads();

    float mean = mean_state[dim_idx];
    float var_sum = 0.0f;

    for (int r = tid; r < params.num_replicas; r += blockDim.x) {
        float diff = replicas[r * params.dimensions + dim_idx] - mean;
        var_sum += diff * diff;
    }

    shared_data[tid] = var_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        // Store diagonal of covariance matrix
        covariance[dim_idx * params.dimensions + dim_idx] =
            shared_data[0] / (params.num_replicas - 1);
    }

    // Compute ensemble entropy (simplified - based on energy distribution)
    if (blockIdx.x == 0 && tid == 0) {
        float Z = 0.0f; // Partition function
        float E_mean = 0.0f;

        for (int r = 0; r < params.num_replicas; ++r) {
            float weight = expf(-params.beta * energies[r]);
            Z += weight;
            E_mean += energies[r] * weight;
        }

        E_mean /= Z;
        float S = params.beta * E_mean + logf(Z);
        *entropy = S;
    }
}

// Performance monitoring kernel
extern "C" __global__ void pimc_performance_metrics(
    const float* __restrict__ acceptance_rates,
    const float* __restrict__ exchange_counts,
    float* __restrict__ metrics, // [efficiency, tunneling_rate, equilibration]
    int num_replicas,
    int total_exchanges
) {
    // Single thread computes aggregate metrics
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Average acceptance rate
        float avg_acceptance = 0.0f;
        for (int i = 0; i < num_replicas; ++i) {
            avg_acceptance += acceptance_rates[i];
        }
        avg_acceptance /= num_replicas;

        // Exchange efficiency
        float exchange_rate = 0.0f;
        int num_pairs = num_replicas / 2;
        for (int i = 0; i < num_pairs; ++i) {
            exchange_rate += (float)exchange_counts[i] / total_exchanges;
        }
        exchange_rate /= num_pairs;

        // Store metrics
        metrics[0] = avg_acceptance;  // MC efficiency
        metrics[1] = exchange_rate;   // Tunneling/exchange rate
        metrics[2] = 1.0f - fabsf(avg_acceptance - 0.5f) * 2.0f; // Equilibration measure
    }
}

// Helper kernel for initializing random spin configurations
extern "C" __global__ void initialize_random_spins(
    float* replicas,
    int num_replicas,
    int dimensions,
    unsigned long seed
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_spins = num_replicas * dimensions;
    if (tid >= total_spins) return;

    curandState rand_state;
    curand_init(seed, tid, 0, &rand_state);

    // Random +1 or -1 spin
    replicas[tid] = (curand_uniform(&rand_state) < 0.5f) ? -1.0f : 1.0f;
}