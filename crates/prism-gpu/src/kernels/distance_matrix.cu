//=============================================================================
// PRISM-LBS DISTANCE MATRIX KERNEL
// Session 10E: Multi-Pass Architecture - Pass 1
// Computes global pairwise Cα distances for all residue pairs
//=============================================================================

#include <cuda_runtime.h>

//=============================================================================
// KERNEL: Compute Distance Matrix (Upper Triangle)
//=============================================================================
// Each thread computes one distance
// Grid: 2D (covers i,j pairs)
// Output: Symmetric matrix, store only i < j to save memory
//=============================================================================

extern "C" __global__ void compute_distance_matrix_kernel(
    const float* __restrict__ atoms,        // [n_atoms * 3] XYZ coordinates
    const int* __restrict__ ca_indices,     // [n_residues] Cα atom index per residue
    const int n_residues,                   // Total residue count
    float* __restrict__ distance_matrix,    // [n_residues * n_residues] output
    const float cutoff                      // Distance cutoff (e.g., 15.0 Å)
) {
    // 2D thread indexing
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Bounds check
    if (i >= n_residues || j >= n_residues) return;

    // Diagonal
    if (i == j) {
        distance_matrix[i * n_residues + j] = 0.0f;
        return;
    }

    // Get Cα indices
    int ca_i = ca_indices[i];
    int ca_j = ca_indices[j];

    // Handle invalid indices
    if (ca_i < 0 || ca_j < 0) {
        distance_matrix[i * n_residues + j] = 999.0f;  // Sentinel for missing data
        return;
    }

    // Load Cα positions
    float3 pos_i, pos_j;
    pos_i.x = atoms[ca_i * 3 + 0];
    pos_i.y = atoms[ca_i * 3 + 1];
    pos_i.z = atoms[ca_i * 3 + 2];

    pos_j.x = atoms[ca_j * 3 + 0];
    pos_j.y = atoms[ca_j * 3 + 1];
    pos_j.z = atoms[ca_j * 3 + 2];

    // Compute Euclidean distance
    float dx = pos_j.x - pos_i.x;
    float dy = pos_j.y - pos_i.y;
    float dz = pos_j.z - pos_i.z;
    float dist = sqrtf(dx*dx + dy*dy + dz*dz);

    // Store distance (use 999.0 for beyond-cutoff to save memory in sparse ops)
    distance_matrix[i * n_residues + j] = (dist < cutoff) ? dist : 999.0f;
}

//=============================================================================
// HELPER: Sparse Upper Triangle Version (Memory Optimized)
//=============================================================================
// For very large proteins, store only upper triangle
// Access: dist(i,j) = matrix[i < j ? i*n + j : j*n + i]
//=============================================================================

extern "C" __global__ void compute_distance_matrix_upper_triangle(
    const float* __restrict__ atoms,
    const int* __restrict__ ca_indices,
    const int n_residues,
    float* __restrict__ distance_upper,     // [n_residues * (n_residues+1) / 2]
    const float cutoff
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= n_residues || j >= n_residues || i > j) return;

    if (i == j) {
        // Diagonal
        int idx = i * (n_residues + n_residues - i + 1) / 2;  // Upper triangle index
        distance_upper[idx] = 0.0f;
        return;
    }

    int ca_i = ca_indices[i];
    int ca_j = ca_indices[j];

    if (ca_i < 0 || ca_j < 0) {
        int idx = i * n_residues - i * (i + 1) / 2 + j;
        distance_upper[idx] = 999.0f;
        return;
    }

    float3 pos_i, pos_j;
    pos_i.x = atoms[ca_i * 3 + 0];
    pos_i.y = atoms[ca_i * 3 + 1];
    pos_i.z = atoms[ca_i * 3 + 2];

    pos_j.x = atoms[ca_j * 3 + 0];
    pos_j.y = atoms[ca_j * 3 + 1];
    pos_j.z = atoms[ca_j * 3 + 2];

    float dx = pos_j.x - pos_i.x;
    float dy = pos_j.y - pos_i.y;
    float dz = pos_j.z - pos_i.z;
    float dist = sqrtf(dx*dx + dy*dy + dz*dz);

    // Store in upper triangle
    int idx = i * n_residues - i * (i + 1) / 2 + j;
    distance_upper[idx] = (dist < cutoff) ? dist : 999.0f;
}
