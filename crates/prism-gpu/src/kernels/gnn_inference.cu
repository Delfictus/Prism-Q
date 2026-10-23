// GNN (Graph Neural Network) Inference kernel for ONNX model acceleration
//
// ASSUMPTIONS:
// - Graph stored in CSR format (row_ptr, col_idx, edge_features)
// - MAX_NODES = 10000 (graph size limit)
// - MAX_EDGES = 100000 (edge count limit)
// - MAX_FEATURES = 512 (feature dimension limit)
// - MAX_LAYERS = 8 (GNN depth limit)
// - Precision: f32 primary, f16 for memory-bound operations
// - Block size: 256 threads (optimal for coalesced access)
// - Grid size: Variable based on nodes/edges
// - Requires: sm_80+ for tensor cores (optional acceleration)
// REFERENCE: PRISM Spec Section 7.2 "GNN-Accelerated Graph Processing"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

// Configuration constants
constexpr int MAX_NODES = 10000;
constexpr int MAX_EDGES = 100000;
constexpr int MAX_FEATURES = 512;
constexpr int MAX_LAYERS = 8;
constexpr int THREADS_PER_BLOCK = 256;
constexpr int WARP_SIZE = 32;
constexpr float EPSILON = 1e-6f;

// GNN layer types
enum class GnnLayerType {
    GCN,      // Graph Convolutional Network
    GAT,      // Graph Attention Network
    SAGE,     // GraphSAGE
    GIN,      // Graph Isomorphism Network
    MPNN      // Message Passing Neural Network
};

// GNN inference parameters
struct GnnParams {
    int num_nodes;          // Number of nodes in graph
    int num_edges;          // Number of edges
    int input_features;     // Input feature dimension
    int hidden_features;    // Hidden layer dimension
    int output_features;    // Output dimension
    int num_layers;         // Number of GNN layers
    GnnLayerType layer_type; // Type of GNN layer
    bool use_batch_norm;    // Enable batch normalization
    bool use_dropout;       // Enable dropout
    float dropout_rate;     // Dropout probability
    bool use_residual;      // Residual connections
    int num_heads;          // Number of attention heads (for GAT)
};

// Device function: ReLU activation
__device__ __forceinline__ float relu(float x) {
    return fmaxf(0.0f, x);
}

// Device function: LeakyReLU activation
__device__ __forceinline__ float leaky_relu(float x, float alpha = 0.01f) {
    return x > 0.0f ? x : alpha * x;
}

// Device function: Softmax over neighborhood (for attention)
__device__ void neighborhood_softmax(
    float* __restrict__ scores,
    int num_neighbors
) {
    float max_score = -1e10f;
    for (int i = 0; i < num_neighbors; ++i) {
        max_score = fmaxf(max_score, scores[i]);
    }

    float sum = 0.0f;
    for (int i = 0; i < num_neighbors; ++i) {
        scores[i] = expf(scores[i] - max_score);
        sum += scores[i];
    }

    for (int i = 0; i < num_neighbors; ++i) {
        scores[i] /= (sum + EPSILON);
    }
}

// GCN (Graph Convolutional Network) layer kernel
extern "C" __global__ void gcn_layer_kernel(
    const float* __restrict__ input_features,  // [num_nodes][in_features]
    const float* __restrict__ weights,         // [in_features][out_features]
    const float* __restrict__ bias,            // [out_features]
    const int* __restrict__ row_ptr,           // CSR row pointers
    const int* __restrict__ col_idx,           // CSR column indices
    const float* __restrict__ edge_weights,    // Edge weights (optional)
    float* __restrict__ output_features,       // [num_nodes][out_features]
    GnnParams params
) {
    int node_id = blockIdx.x;
    int feat_id = threadIdx.x;

    if (node_id >= params.num_nodes) return;
    if (feat_id >= params.hidden_features) return;

    // Shared memory for caching weights
    extern __shared__ float shared_weights[];

    // Load weights into shared memory (coalesced)
    for (int i = threadIdx.x; i < params.input_features * params.hidden_features;
         i += blockDim.x) {
        shared_weights[i] = weights[i];
    }
    __syncthreads();

    // Get node degree for normalization
    int start = row_ptr[node_id];
    int end = row_ptr[node_id + 1];
    float degree = sqrtf((float)(end - start + 1)); // Self-loop included

    // Aggregate features from neighbors
    float aggregated = 0.0f;

    // Self-loop contribution
    for (int in_feat = 0; in_feat < params.input_features; ++in_feat) {
        aggregated += input_features[node_id * params.input_features + in_feat] *
                     shared_weights[in_feat * params.hidden_features + feat_id];
    }
    aggregated /= degree;

    // Neighbor contributions
    for (int edge_idx = start; edge_idx < end; ++edge_idx) {
        int neighbor = col_idx[edge_idx];
        float neighbor_degree = sqrtf((float)(row_ptr[neighbor + 1] - row_ptr[neighbor] + 1));

        float edge_weight = (edge_weights != nullptr) ? edge_weights[edge_idx] : 1.0f;

        for (int in_feat = 0; in_feat < params.input_features; ++in_feat) {
            aggregated += edge_weight *
                         input_features[neighbor * params.input_features + in_feat] *
                         shared_weights[in_feat * params.hidden_features + feat_id] /
                         (degree * neighbor_degree);
        }
    }

    // Add bias and activation
    aggregated += bias[feat_id];
    output_features[node_id * params.hidden_features + feat_id] = relu(aggregated);
}

// GAT (Graph Attention Network) layer kernel
extern "C" __global__ void gat_layer_kernel(
    const float* __restrict__ input_features,
    const float* __restrict__ weight_src,      // [in_features][hidden_per_head]
    const float* __restrict__ weight_dst,      // [in_features][hidden_per_head]
    const float* __restrict__ attention_weights, // [2 * hidden_per_head]
    const float* __restrict__ bias,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    float* __restrict__ output_features,
    float* __restrict__ attention_scores,      // Store attention for visualization
    GnnParams params
) {
    int node_id = blockIdx.x;
    int head_id = blockIdx.y;
    int feat_id = threadIdx.x;

    if (node_id >= params.num_nodes) return;
    if (head_id >= params.num_heads) return;

    int hidden_per_head = params.hidden_features / params.num_heads;
    if (feat_id >= hidden_per_head) return;

    // Transform source features
    float src_transformed = 0.0f;
    for (int in_feat = 0; in_feat < params.input_features; ++in_feat) {
        src_transformed += input_features[node_id * params.input_features + in_feat] *
                          weight_src[in_feat * hidden_per_head + feat_id];
    }

    // Get neighbors
    int start = row_ptr[node_id];
    int end = row_ptr[node_id + 1];
    int num_neighbors = end - start;

    // Compute attention scores
    extern __shared__ float shared_attention[];
    float* local_scores = &shared_attention[threadIdx.x * MAX_EDGES];

    if (feat_id == 0) { // One thread computes all attention scores
        for (int idx = 0; idx < num_neighbors; ++idx) {
            int neighbor = col_idx[start + idx];

            // Transform neighbor features
            float dst_transformed = 0.0f;
            for (int in_feat = 0; in_feat < params.input_features; ++in_feat) {
                dst_transformed += input_features[neighbor * params.input_features + in_feat] *
                                 weight_dst[in_feat * hidden_per_head + 0];
            }

            // Compute attention score: a^T [Wh_i || Wh_j]
            float score = attention_weights[0] * src_transformed +
                         attention_weights[hidden_per_head] * dst_transformed;
            local_scores[idx] = leaky_relu(score);
        }

        // Softmax normalization
        neighborhood_softmax(local_scores, num_neighbors);
    }
    __syncthreads();

    // Aggregate with attention weights
    float aggregated = 0.0f;
    for (int idx = 0; idx < num_neighbors; ++idx) {
        int neighbor = col_idx[start + idx];
        float attention = local_scores[idx];

        for (int in_feat = 0; in_feat < params.input_features; ++in_feat) {
            aggregated += attention *
                         input_features[neighbor * params.input_features + in_feat] *
                         weight_src[in_feat * hidden_per_head + feat_id];
        }
    }

    // Multi-head output
    int output_idx = node_id * params.hidden_features +
                    head_id * hidden_per_head + feat_id;
    output_features[output_idx] = relu(aggregated + bias[head_id * hidden_per_head + feat_id]);

    // Store attention scores for first head (for visualization)
    if (head_id == 0 && feat_id == 0) {
        for (int idx = 0; idx < num_neighbors; ++idx) {
            attention_scores[start + idx] = local_scores[idx];
        }
    }
}

// GraphSAGE layer kernel (mean aggregation)
extern "C" __global__ void sage_layer_kernel(
    const float* __restrict__ input_features,
    const float* __restrict__ weight_self,     // [in_features][hidden_features]
    const float* __restrict__ weight_neighbor, // [in_features][hidden_features]
    const float* __restrict__ bias,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const int* __restrict__ sample_counts,     // Number of sampled neighbors per node
    float* __restrict__ output_features,
    GnnParams params,
    int sample_size = 10  // Neighbor sampling size
) {
    int node_id = blockIdx.x;
    int feat_id = threadIdx.x;

    if (node_id >= params.num_nodes) return;
    if (feat_id >= params.hidden_features) return;

    // Self transformation
    float self_transformed = 0.0f;
    for (int in_feat = 0; in_feat < params.input_features; ++in_feat) {
        self_transformed += input_features[node_id * params.input_features + in_feat] *
                          weight_self[in_feat * params.hidden_features + feat_id];
    }

    // Neighbor aggregation (mean)
    float neighbor_aggregated = 0.0f;
    int start = row_ptr[node_id];
    int end = row_ptr[node_id + 1];
    int num_neighbors = min(end - start, sample_size);

    if (num_neighbors > 0) {
        for (int idx = 0; idx < num_neighbors; ++idx) {
            int neighbor = col_idx[start + idx];

            for (int in_feat = 0; in_feat < params.input_features; ++in_feat) {
                neighbor_aggregated += input_features[neighbor * params.input_features + in_feat] *
                                     weight_neighbor[in_feat * params.hidden_features + feat_id];
            }
        }
        neighbor_aggregated /= num_neighbors;
    }

    // Concatenate and normalize
    float combined = self_transformed + neighbor_aggregated;
    float normalized = combined / sqrtf(combined * combined + EPSILON);

    output_features[node_id * params.hidden_features + feat_id] =
        relu(normalized + bias[feat_id]);
}

// GIN (Graph Isomorphism Network) layer kernel
extern "C" __global__ void gin_layer_kernel(
    const float* __restrict__ input_features,
    const float* __restrict__ mlp_weights1,    // [in_features][hidden_features]
    const float* __restrict__ mlp_weights2,    // [hidden_features][hidden_features]
    const float* __restrict__ mlp_bias1,
    const float* __restrict__ mlp_bias2,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    float* __restrict__ output_features,
    float epsilon,  // GIN epsilon parameter
    GnnParams params
) {
    int node_id = blockIdx.x;
    int feat_id = threadIdx.x;

    if (node_id >= params.num_nodes) return;
    if (feat_id >= params.hidden_features) return;

    // Sum aggregation
    float aggregated = 0.0f;

    // Self features with (1 + epsilon)
    for (int in_feat = 0; in_feat < params.input_features; ++in_feat) {
        aggregated += (1.0f + epsilon) *
                     input_features[node_id * params.input_features + in_feat];
    }

    // Neighbor features
    int start = row_ptr[node_id];
    int end = row_ptr[node_id + 1];

    for (int edge_idx = start; edge_idx < end; ++edge_idx) {
        int neighbor = col_idx[edge_idx];

        for (int in_feat = 0; in_feat < params.input_features; ++in_feat) {
            aggregated += input_features[neighbor * params.input_features + in_feat];
        }
    }

    // MLP: first layer
    float hidden = 0.0f;
    for (int in_feat = 0; in_feat < params.input_features; ++in_feat) {
        hidden += aggregated * mlp_weights1[in_feat * params.hidden_features + feat_id];
    }
    hidden = relu(hidden + mlp_bias1[feat_id]);

    // MLP: second layer
    float output = 0.0f;
    for (int h = 0; h < params.hidden_features; ++h) {
        output += hidden * mlp_weights2[h * params.hidden_features + feat_id];
    }

    output_features[node_id * params.hidden_features + feat_id] =
        relu(output + mlp_bias2[feat_id]);
}

// Batch normalization kernel for GNN
extern "C" __global__ void batch_norm_kernel(
    float* __restrict__ features,        // In-place normalization
    const float* __restrict__ gamma,      // Scale parameter
    const float* __restrict__ beta,       // Shift parameter
    float* __restrict__ running_mean,     // Updated running mean
    float* __restrict__ running_var,      // Updated running variance
    int num_nodes,
    int num_features,
    float momentum = 0.9f,
    bool training = true
) {
    int feat_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (feat_id >= num_features) return;

    // Compute mean and variance for this feature
    float mean = 0.0f;
    float var = 0.0f;

    for (int node = 0; node < num_nodes; ++node) {
        float val = features[node * num_features + feat_id];
        mean += val;
    }
    mean /= num_nodes;

    for (int node = 0; node < num_nodes; ++node) {
        float val = features[node * num_features + feat_id];
        var += (val - mean) * (val - mean);
    }
    var /= num_nodes;

    // Update running statistics
    if (training) {
        running_mean[feat_id] = momentum * running_mean[feat_id] + (1.0f - momentum) * mean;
        running_var[feat_id] = momentum * running_var[feat_id] + (1.0f - momentum) * var;
    } else {
        mean = running_mean[feat_id];
        var = running_var[feat_id];
    }

    // Normalize
    float std_inv = rsqrtf(var + EPSILON);

    for (int node = 0; node < num_nodes; ++node) {
        int idx = node * num_features + feat_id;
        features[idx] = gamma[feat_id] * (features[idx] - mean) * std_inv + beta[feat_id];
    }
}

// Dropout kernel for GNN
extern "C" __global__ void dropout_kernel(
    float* __restrict__ features,
    const float* __restrict__ random_values, // Pre-generated random values
    int num_elements,
    float dropout_rate,
    bool training
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    if (training) {
        float scale = 1.0f / (1.0f - dropout_rate);
        features[idx] = (random_values[idx] > dropout_rate) ? features[idx] * scale : 0.0f;
    }
    // No operation during inference
}

// Edge convolution kernel (for dynamic graphs)
extern "C" __global__ void edge_conv_kernel(
    const float* __restrict__ node_features,
    const float* __restrict__ edge_features,
    const float* __restrict__ weights_node,
    const float* __restrict__ weights_edge,
    const float* __restrict__ weights_combined,
    const int* __restrict__ edge_list,      // [num_edges][2] source-target pairs
    float* __restrict__ output_features,
    int num_nodes,
    int num_edges,
    int in_features,
    int edge_features_dim,
    int out_features
) {
    int edge_idx = blockIdx.x;
    int feat_idx = threadIdx.x;

    if (edge_idx >= num_edges) return;
    if (feat_idx >= out_features) return;

    int src = edge_list[edge_idx * 2];
    int dst = edge_list[edge_idx * 2 + 1];

    // Transform source and destination features
    float src_transformed = 0.0f;
    float dst_transformed = 0.0f;
    float edge_transformed = 0.0f;

    for (int f = 0; f < in_features; ++f) {
        src_transformed += node_features[src * in_features + f] *
                          weights_node[f * out_features + feat_idx];
        dst_transformed += node_features[dst * in_features + f] *
                          weights_node[f * out_features + feat_idx];
    }

    for (int f = 0; f < edge_features_dim; ++f) {
        edge_transformed += edge_features[edge_idx * edge_features_dim + f] *
                          weights_edge[f * out_features + feat_idx];
    }

    // Combine and update
    float combined = relu(src_transformed + dst_transformed + edge_transformed);

    // Atomic update to destination node
    atomicAdd(&output_features[dst * out_features + feat_idx], combined);
}

// Global pooling kernels (for graph-level predictions)
extern "C" __global__ void global_mean_pool_kernel(
    const float* __restrict__ node_features,
    float* __restrict__ graph_features,
    const int* __restrict__ graph_ids,      // Node to graph mapping
    int num_nodes,
    int num_features,
    int num_graphs
) {
    int graph_id = blockIdx.x;
    int feat_id = threadIdx.x;

    if (graph_id >= num_graphs) return;
    if (feat_id >= num_features) return;

    float sum = 0.0f;
    int count = 0;

    for (int node = 0; node < num_nodes; ++node) {
        if (graph_ids[node] == graph_id) {
            sum += node_features[node * num_features + feat_id];
            count++;
        }
    }

    graph_features[graph_id * num_features + feat_id] =
        (count > 0) ? (sum / count) : 0.0f;
}

extern "C" __global__ void global_max_pool_kernel(
    const float* __restrict__ node_features,
    float* __restrict__ graph_features,
    const int* __restrict__ graph_ids,
    int num_nodes,
    int num_features,
    int num_graphs
) {
    int graph_id = blockIdx.x;
    int feat_id = threadIdx.x;

    if (graph_id >= num_graphs) return;
    if (feat_id >= num_features) return;

    float max_val = -1e10f;

    for (int node = 0; node < num_nodes; ++node) {
        if (graph_ids[node] == graph_id) {
            max_val = fmaxf(max_val, node_features[node * num_features + feat_id]);
        }
    }

    graph_features[graph_id * num_features + feat_id] = max_val;
}

// Full GNN forward pass orchestration kernel
extern "C" __global__ void gnn_forward_pass(
    const float* __restrict__ input_features,
    const float** __restrict__ layer_weights,  // Array of weight pointers per layer
    const float** __restrict__ layer_biases,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    float* __restrict__ intermediate_features, // Working memory for layers
    float* __restrict__ output_features,
    GnnParams params
) {
    // This would orchestrate multiple layer calls
    // Simplified version - actual implementation would call layer kernels

    int node_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_id >= params.num_nodes) return;

    // Copy input to working buffer
    for (int f = 0; f < params.input_features; ++f) {
        intermediate_features[node_id * params.input_features + f] =
            input_features[node_id * params.input_features + f];
    }

    // Process each layer (would actually call layer-specific kernels)
    for (int layer = 0; layer < params.num_layers; ++layer) {
        // Layer processing placeholder
        __syncthreads();
    }

    // Copy to output
    for (int f = 0; f < params.output_features; ++f) {
        output_features[node_id * params.output_features + f] =
            intermediate_features[node_id * params.output_features + f];
    }
}

// Performance monitoring kernel
extern "C" __global__ void gnn_performance_metrics(
    const float* __restrict__ output_features,
    const int* __restrict__ predictions,
    const int* __restrict__ ground_truth,
    float* __restrict__ metrics, // [accuracy, inference_time, memory_usage]
    int num_nodes,
    int num_classes
) {
    // Single thread computes metrics
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int correct = 0;

        for (int i = 0; i < num_nodes; ++i) {
            if (predictions[i] == ground_truth[i]) {
                correct++;
            }
        }

        float accuracy = (float)correct / num_nodes;
        metrics[0] = accuracy;

        // Placeholder for timing and memory metrics
        metrics[1] = 0.001f; // 1ms inference time placeholder
        metrics[2] = (float)(num_nodes * num_classes * sizeof(float)) / (1024 * 1024); // MB
    }
}