/**
 * Simple Evolution Strategy (OpenAI-ES style) for High-Dimensional Optimization
 *
 * Designed for evolving Neural Network weights (35k+ params) on GPU.
 * Uses diagonal covariance (isotropic noise) to avoid O(N^2) memory scaling.
 *
 * Architecture:
 * 1. perturb_weights: W_i = W_mean + sigma * noise_i
 * 2. update_weights: W_mean += alpha * sum(fitness_i * noise_i)
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>

extern "C" __global__ void
init_rng_states(
    curandState* states,
    unsigned long long seed,
    int n_threads
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_threads) {
        curand_init(seed, tid, 0, &states[tid]);
    }
}

/**
 * Generate perturbed weights for a population
 *
 * @param mean_weights    [n_params] Current mean weight vector
 * @param population      [pop_size * n_params] Output population weights
 * @param noise_buffer    [pop_size * n_params] Buffer to store generated noise (for update)
 * @param rng_states      [pop_size * n_params] Random states
 * @param sigma           Noise standard deviation
 * @param n_params        Number of parameters per individual
 * @param pop_size        Population size
 */
extern "C" __global__ void __launch_bounds__(256)
perturb_weights(
    const float* __restrict__ mean_weights,
    float* __restrict__ population,
    float* __restrict__ noise_buffer,
    curandState* states,
    float sigma,
    int n_params,
    int pop_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = n_params * pop_size;

    if (tid < total_elements) {
        int param_idx = tid % n_params;
        
        // Generate Gaussian noise
        float noise = curand_normal(&states[tid]);
        
        // Store noise for gradient approximation
        noise_buffer[tid] = noise;
        
        // Apply perturbation
        population[tid] = mean_weights[param_idx] + sigma * noise;
    }
}

/**
 * Update mean weights based on fitness scores
 *
 * W_new = W_old + learning_rate / (pop_size * sigma) * sum(fitness_i * noise_i)
 *
 * @param mean_weights    [n_params] Input/Output mean weights
 * @param noise_buffer    [pop_size * n_params] Stored noise vectors
 * @param fitness_scores  [pop_size] Fitness scores for each individual (centered/ranked)
 * @param learning_rate   Optimization step size
 * @param sigma           Noise standard deviation
 * @param n_params        Number of parameters
 * @param pop_size        Population size
 */
extern "C" __global__ void __launch_bounds__(256)
update_weights(
    float* __restrict__ mean_weights,
    const float* __restrict__ noise_buffer,
    const float* __restrict__ fitness_scores,
    float learning_rate,
    float sigma,
    int n_params,
    int pop_size
) {
    int param_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (param_idx < n_params) {
        float gradient_est = 0.0f;
        
        // Accumulate weighted noise across population
        // This loop is O(pop_size), which is small (e.g., 64-256)
        for (int i = 0; i < pop_size; i++) {
            float noise = noise_buffer[i * n_params + param_idx];
            float score = fitness_scores[i];
            gradient_est += score * noise;
        }
        
        // Update weight
        float scale = learning_rate / ((float)pop_size * sigma);
        mean_weights[param_idx] += scale * gradient_est;
    }
}
