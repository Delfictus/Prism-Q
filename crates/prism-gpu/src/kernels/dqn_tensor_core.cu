#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Dimensions (MUST MATCH Rust Configuration)
#define INPUT_DIM 140
#define HIDDEN1_DIM 256
#define HIDDEN2_DIM 128
#define HIDDEN3_DIM 64
#define VALUE_DIM 1
#define ADVANTAGE_DIM 4
#define WARP_SIZE 32

// Tensor Core tile dimensions (M16N8K16 for FP16 - legacy placeholder)
#define TILE_M 16
#define TILE_N 8
#define TILE_K 16

/**
 * Packed DQN weight structure for efficient GPU access
 * Uses float (FP32) for compatibility with Evolution Strategy
 */
struct DQNWeights {
    float weights_fc1[INPUT_DIM * HIDDEN1_DIM];
    float bias_fc1[HIDDEN1_DIM];
    float weights_fc2[HIDDEN1_DIM * HIDDEN2_DIM];
    float bias_fc2[HIDDEN2_DIM];
    float weights_fc3[HIDDEN2_DIM * HIDDEN3_DIM];
    float bias_fc3[HIDDEN3_DIM];
    float weights_value[HIDDEN3_DIM];
    float bias_value[VALUE_DIM];
    float weights_advantage[HIDDEN3_DIM * ADVANTAGE_DIM];
    float bias_advantage[ADVANTAGE_DIM];
};

// Forward declarations
__device__ void tensor_core_gemv_140_256(const float* input, const float* weights, const float* bias, float* output, int tid);
__device__ void tensor_core_gemv_256_128(const float* input, const float* weights, const float* bias, float* output, int tid);
__device__ void tensor_core_gemv_128_64(const float* input, const float* weights, const float* bias, float* output, int tid);

/**
 * Dueling DQN forward pass using Tensor Cores (Emulated in FP32)
 */
extern "C" __global__ void __launch_bounds__(256, 2)
dqn_forward_pass(
    const float* __restrict__ input_features,
    const float* __restrict__ weights_fc1,
    const float* __restrict__ bias_fc1,
    const float* __restrict__ weights_fc2,
    const float* __restrict__ bias_fc2,
    const float* __restrict__ weights_fc3,
    const float* __restrict__ bias_fc3,
    const float* __restrict__ weights_value,
    const float* __restrict__ bias_value,
    const float* __restrict__ weights_advantage,
    const float* __restrict__ bias_advantage,
    float* __restrict__ q_values,
    const int batch_size
) {
    // Shared memory for intermediate activations
    __shared__ float activations_fc1[HIDDEN1_DIM];
    __shared__ float activations_fc2[HIDDEN2_DIM];
    __shared__ float activations_fc3[HIDDEN3_DIM];

    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    // Input pointer for this batch
    const float* input = input_features + batch_idx * INPUT_DIM;

    // Layer 1: 140 -> 256
    tensor_core_gemv_140_256(input, weights_fc1, bias_fc1, activations_fc1, tid);
    __syncthreads();

    // ReLU activation
    for (int i = tid; i < HIDDEN1_DIM; i += blockDim.x) {
        activations_fc1[i] = fmaxf(0.0f, activations_fc1[i]);
    }
    __syncthreads();

    // Layer 2: 256 -> 128
    tensor_core_gemv_256_128(activations_fc1, weights_fc2, bias_fc2, activations_fc2, tid);
    __syncthreads();

    // ReLU activation
    for (int i = tid; i < HIDDEN2_DIM; i += blockDim.x) {
        activations_fc2[i] = fmaxf(0.0f, activations_fc2[i]);
    }
    __syncthreads();

    // Layer 3: 128 -> 64
    tensor_core_gemv_128_64(activations_fc2, weights_fc3, bias_fc3, activations_fc3, tid);
    __syncthreads();

    // ReLU activation
    for (int i = tid; i < HIDDEN3_DIM; i += blockDim.x) {
        activations_fc3[i] = fmaxf(0.0f, activations_fc3[i]);
    }
    __syncthreads();

    // Value head: 64 -> 1
    float value = 0.0f;
    for (int i = tid; i < HIDDEN3_DIM; i += blockDim.x) {
        value += activations_fc3[i] * weights_value[i];
    }

    // Warp reduction for value
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xFFFFFFFF, value, offset);
    }

    __shared__ float shared_value;
    if (tid % WARP_SIZE == 0) {
        atomicAdd(&shared_value, value);
    }
    __syncthreads();

    if (tid == 0) {
        shared_value += bias_value[0];
    }
    __syncthreads();

    // Advantage head: 64 -> 4
    __shared__ float advantages[ADVANTAGE_DIM];
    if (tid < ADVANTAGE_DIM) {
        float adv = bias_advantage[tid];
        for (int i = 0; i < HIDDEN3_DIM; i++) {
            adv += activations_fc3[i] * weights_advantage[i * ADVANTAGE_DIM + tid];
        }
        advantages[tid] = adv;
    }
    __syncthreads();

    // Compute mean advantage
    float mean_advantage = 0.0f;
    if (tid < ADVANTAGE_DIM) {
        mean_advantage = (advantages[0] + advantages[1] + advantages[2] + advantages[3]) / 4.0f;
    }

    __shared__ float shared_mean;
    if (tid == 0) shared_mean = mean_advantage;
    __syncthreads();

    // Dueling combination: Q(s,a) = V(s) + (A(s,a) - mean(A))
    float* output = q_values + batch_idx * 4;
    if (tid < ADVANTAGE_DIM) {
        output[tid] = shared_value + (advantages[tid] - shared_mean);
    }
}

/**
 * Optimized tensor core GEMV: 140 -> 256
 */
__device__ void tensor_core_gemv_140_256(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    int tid
) {
    for (int out_idx = tid; out_idx < HIDDEN1_DIM; out_idx += blockDim.x) {
        float sum = bias[out_idx];
        for (int in_idx = 0; in_idx < INPUT_DIM; in_idx++) {
            // Scale input by 0.01 to normalize raw features (e.g. SASA)
            float feat_val = input[in_idx] * 0.01f;
            sum += feat_val * weights[in_idx * HIDDEN1_DIM + out_idx];
        }
        output[out_idx] = sum;
    }
}

/**
 * Optimized tensor core GEMV: 256 -> 128
 */
__device__ void tensor_core_gemv_256_128(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    int tid
) {
    for (int out_idx = tid; out_idx < HIDDEN2_DIM; out_idx += blockDim.x) {
        float sum = bias[out_idx];
        for (int in_idx = 0; in_idx < HIDDEN1_DIM; in_idx++) {
            sum += input[in_idx] * weights[in_idx * HIDDEN2_DIM + out_idx];
        }
        output[out_idx] = sum;
    }
}

/**
 * Optimized tensor core GEMV: 128 -> 64
 */
__device__ void tensor_core_gemv_128_64(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    int tid
) {
    for (int out_idx = tid; out_idx < HIDDEN3_DIM; out_idx += blockDim.x) {
        float sum = bias[out_idx];
        for (int in_idx = 0; in_idx < HIDDEN2_DIM; in_idx++) {
            sum += input[in_idx] * weights[in_idx * HIDDEN3_DIM + out_idx];
        }
        output[out_idx] = sum;
    }
}

/**
 * Batch DQN inference for multiple residues
 */
extern "C" __global__ void __launch_bounds__(256, 2)
dqn_batch_inference(
    const float* __restrict__ feature_vectors,
    float* __restrict__ q_value_output,
    const int total_residues,
    const DQNWeights* __restrict__ weights_ptr
) {
    const int residue_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (residue_idx >= total_residues) return;

    // Single-residue inference using the full DQN
    const float* features = feature_vectors + residue_idx * INPUT_DIM;
    float* q_out = q_value_output + residue_idx * 4;

    // Local storage for activations
    float hidden1[HIDDEN1_DIM];
    float hidden2[HIDDEN2_DIM];
    float hidden3[HIDDEN3_DIM];

    // Layer 1: 140 -> 256
    for (int i = 0; i < HIDDEN1_DIM; i++) {
        float sum = weights_ptr->bias_fc1[i];
        for (int j = 0; j < INPUT_DIM; j++) {
            // SCALE INPUT HERE TOO
            float val = features[j] * 0.01f;
            sum += val * weights_ptr->weights_fc1[j * HIDDEN1_DIM + i];
        }
        hidden1[i] = fmaxf(0.0f, sum); // ReLU
    }

    // Layer 2: 256 -> 128
    for (int i = 0; i < HIDDEN2_DIM; i++) {
        float sum = weights_ptr->bias_fc2[i];
        for (int j = 0; j < HIDDEN1_DIM; j++) {
            sum += hidden1[j] * weights_ptr->weights_fc2[j * HIDDEN1_DIM + i]; // FIXED INDEXING
        }
        hidden2[i] = fmaxf(0.0f, sum); // ReLU
    }

    // Layer 3: 128 -> 64
    for (int i = 0; i < HIDDEN3_DIM; i++) {
        float sum = weights_ptr->bias_fc3[i];
        for (int j = 0; j < HIDDEN2_DIM; j++) {
            sum += hidden2[j] * weights_ptr->weights_fc3[j * HIDDEN2_DIM + i]; // FIXED INDEXING
        }
        hidden3[i] = fmaxf(0.0f, sum); // ReLU
    }

    // Value head
    float value = weights_ptr->bias_value[0];
    for (int i = 0; i < HIDDEN3_DIM; i++) {
        value += hidden3[i] * weights_ptr->weights_value[i];
    }

    // Advantage head
    float advantages[4];
    for (int a = 0; a < 4; a++) {
        advantages[a] = weights_ptr->bias_advantage[a];
        for (int i = 0; i < HIDDEN3_DIM; i++) {
            advantages[a] += hidden3[i] * weights_ptr->weights_advantage[i * 4 + a];
        }
    }

    // Compute mean advantage
    float mean_adv = (advantages[0] + advantages[1] + advantages[2] + advantages[3]) / 4.0f;

    // Dueling Q-values
    for (int a = 0; a < 4; a++) {
        q_out[a] = value + (advantages[a] - mean_adv);
    }
}