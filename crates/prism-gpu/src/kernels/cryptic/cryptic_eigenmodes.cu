/**
 * PRISM Cryptic Site Detection - GPU Eigenmode Computation Kernel
 *
 * Implements parallel power iteration / Lanczos algorithm for computing
 * low-frequency normal modes from the ANM Hessian matrix.
 *
 * Copyright (c) 2024 PRISM Research Team | Delfictus I/O Inc.
 */

#include <cuda_runtime.h>
#include <cmath>

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define MAX_ITERATIONS 100
#define CONVERGENCE_TOLERANCE 1e-6f

/**
 * Matrix-vector multiplication: y = A * x
 * For symmetric Hessian matrix stored as dense.
 */
extern "C" __global__ void hessian_matvec(
    const float* __restrict__ hessian,  // [dim, dim] symmetric matrix
    const float* __restrict__ x,         // [dim] input vector
    float* __restrict__ y,               // [dim] output vector
    const int dim,
    const float shift                    // Diagonal shift for conditioning
) {
    __shared__ float s_partial[BLOCK_SIZE];

    const int row = blockIdx.x;
    if (row >= dim) return;

    // Each block computes one row of the result
    float sum = 0.0f;
    for (int col = threadIdx.x; col < dim; col += blockDim.x) {
        float h_val = hessian[row * dim + col];
        if (row == col) {
            h_val += shift;  // Add shift to diagonal
        }
        sum += h_val * x[col];
    }

    // Warp reduction
    s_partial[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_partial[threadIdx.x] += s_partial[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        y[row] = s_partial[0];
    }
}

/**
 * Vector dot product: result = sum(a * b)
 */
extern "C" __global__ void vector_dot(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ result,
    const int n
) {
    __shared__ float s_partial[BLOCK_SIZE];

    float sum = 0.0f;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
        sum += a[i] * b[i];
    }

    s_partial[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_partial[threadIdx.x] += s_partial[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(result, s_partial[0]);
    }
}

/**
 * Vector normalization: v = v / ||v||
 */
extern "C" __global__ void vector_normalize(
    float* __restrict__ v,
    const float* __restrict__ norm_sq,
    const int n
) {
    float norm = sqrtf(*norm_sq);
    if (norm < 1e-10f) norm = 1.0f;

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        v[idx] /= norm;
    }
}

/**
 * Vector saxpy: y = alpha * x + y
 */
extern "C" __global__ void vector_axpy(
    const float alpha,
    const float* __restrict__ x,
    float* __restrict__ y,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = alpha * x[idx] + y[idx];
    }
}

/**
 * Deflate matrix: A = A - lambda * v * v^T
 * For removing contribution of found eigenvector
 */
extern "C" __global__ void deflate_matrix(
    float* __restrict__ matrix,
    const float* __restrict__ eigenvector,
    const float eigenvalue,
    const int dim
) {
    const int row = blockIdx.x;
    const int col = threadIdx.x + blockIdx.y * blockDim.x;

    if (row < dim && col < dim) {
        float deflation = eigenvalue * eigenvector[row] * eigenvector[col];
        atomicAdd(&matrix[row * dim + col], -deflation);
    }
}

/**
 * Compute per-residue mobility from eigenmodes
 *
 * For each residue i, mobility = sum over modes of (displacement_i² / eigenvalue)
 */
extern "C" __global__ void compute_residue_mobility(
    const float* __restrict__ eigenvectors,  // [num_modes, dim]
    const float* __restrict__ eigenvalues,   // [num_modes]
    float* __restrict__ mobility,            // [n_residues]
    const int n_residues,
    const int num_modes,
    const int dim                            // = 3 * n_residues
) {
    const int residue_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (residue_idx >= n_residues) return;

    float total_mobility = 0.0f;

    for (int mode = 0; mode < num_modes; mode++) {
        float eigenvalue = eigenvalues[mode];
        if (eigenvalue <= 0.0f) continue;

        // Get displacement components for this residue
        const float* mode_vec = &eigenvectors[mode * dim];
        float dx = mode_vec[3 * residue_idx + 0];
        float dy = mode_vec[3 * residue_idx + 1];
        float dz = mode_vec[3 * residue_idx + 2];

        float disp_sq = dx * dx + dy * dy + dz * dz;

        // Mean squared displacement contribution: disp² / eigenvalue
        total_mobility += disp_sq / eigenvalue;
    }

    mobility[residue_idx] = total_mobility;
}

/**
 * Normalize mobility scores to [0, 1] range
 */
extern "C" __global__ void normalize_mobility(
    float* __restrict__ mobility,
    const float* __restrict__ max_mobility,
    const int n_residues
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_residues) return;

    float max_val = *max_mobility;
    if (max_val > 0.0f) {
        mobility[idx] /= max_val;
    }
}

/**
 * Find maximum value in array (parallel reduction)
 */
extern "C" __global__ void find_max(
    const float* __restrict__ values,
    float* __restrict__ max_out,
    const int n
) {
    __shared__ float s_max[BLOCK_SIZE];

    float local_max = -1e30f;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
        local_max = fmaxf(local_max, values[i]);
    }

    s_max[threadIdx.x] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_max[threadIdx.x] = fmaxf(s_max[threadIdx.x], s_max[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicMax((int*)max_out, __float_as_int(s_max[0]));
    }
}

/**
 * Initialize random vector for power iteration
 * Using deterministic pseudo-random based on index
 */
extern "C" __global__ void init_random_vector(
    float* __restrict__ v,
    const int n,
    const unsigned int seed
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Simple LCG-based pseudo-random
    unsigned int state = seed + idx * 1103515245u + 12345u;
    state = state * 1103515245u + 12345u;

    // Convert to float in [-0.5, 0.5]
    v[idx] = ((float)(state & 0x7FFFFFFF) / (float)0x7FFFFFFF) - 0.5f;
}

/**
 * Power iteration step for finding smallest eigenvalue
 * Combines: y = A*x, eigenvalue = x·y, normalize y
 *
 * This is the main iterative kernel - called repeatedly until convergence
 */
extern "C" __global__ void power_iteration_step(
    const float* __restrict__ hessian,
    float* __restrict__ v,              // Current eigenvector estimate
    float* __restrict__ Av,             // Workspace for A*v
    float* __restrict__ eigenvalue,     // Rayleigh quotient estimate
    float* __restrict__ converged,      // Convergence flag
    const int dim,
    const float shift,
    const float prev_eigenvalue
) {
    // This kernel orchestrates a single power iteration step
    // The actual computation is done by calling sub-kernels

    // For efficiency, this is structured as a device-side launch
    // In practice, we'll orchestrate from the host using multiple kernel calls
}

/**
 * Compute Rayleigh quotient: lambda = (v · A · v) / (v · v)
 */
extern "C" __global__ void rayleigh_quotient(
    const float* __restrict__ v,
    const float* __restrict__ Av,
    float* __restrict__ eigenvalue,
    float* __restrict__ v_norm_sq,
    const int n
) {
    __shared__ float s_vAv[BLOCK_SIZE];
    __shared__ float s_vv[BLOCK_SIZE];

    float local_vAv = 0.0f;
    float local_vv = 0.0f;

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
        float vi = v[i];
        local_vAv += vi * Av[i];
        local_vv += vi * vi;
    }

    s_vAv[threadIdx.x] = local_vAv;
    s_vv[threadIdx.x] = local_vv;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_vAv[threadIdx.x] += s_vAv[threadIdx.x + s];
            s_vv[threadIdx.x] += s_vv[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(eigenvalue, s_vAv[0]);
        atomicAdd(v_norm_sq, s_vv[0]);
    }
}

/**
 * Copy and scale Av to v for next iteration: v = Av / ||Av||
 */
extern "C" __global__ void copy_and_normalize(
    const float* __restrict__ Av,
    float* __restrict__ v,
    const float* __restrict__ Av_norm_sq,
    const int n
) {
    float norm = sqrtf(*Av_norm_sq);
    if (norm < 1e-10f) norm = 1.0f;

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        v[idx] = Av[idx] / norm;
    }
}
