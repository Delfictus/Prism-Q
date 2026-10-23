//=============================================================================
// PRISM-4D GPU LCPO SASA (Linear Combination of Pairwise Overlaps)
//
// High-performance solvent accessible surface area calculation using the
// LCPO analytical approximation with shared memory tiling for O(N) wall-clock.
//
// Reference: Weiser, Shenkin, Still (1999) J. Comput. Chem. 20:217-230
//
// Key optimizations:
// - Shared memory tiling: reduces global memory bandwidth
// - Per-pair early exit: 9A cutoff skips ~90% of pairs
// - Multi-frame batching: process entire trajectory in one launch
// - Constant memory: LCPO parameters stored in fast constant memory
//
// Performance target (RTX 3060):
// - 6LU7 (4,730 atoms, 100 frames): < 100ms
// - 6M0J (12,510 atoms, 100 frames): < 250ms
//=============================================================================

#include <cuda_runtime.h>
#include "reduction_primitives.cuh"

//-----------------------------------------------------------------------------
// Constants
//-----------------------------------------------------------------------------

#define TILE_SIZE 128           // Atoms per tile (fits in shared memory)
#define LCPO_CUTOFF 9.0f        // Angstrom cutoff for LCPO overlaps
#define LCPO_CUTOFF_SQ 81.0f    // Squared cutoff
#define PI 3.14159265358979f
#define PROBE_RADIUS 1.4f       // Solvent probe radius (water)

// Maximum number of atom types for LCPO parameters
#define N_ATOM_TYPES 8

//-----------------------------------------------------------------------------
// LCPO Parameters (stored in constant memory for fast access)
//
// LCPO formula: SASA_i = P1*S_i + P2*sum(A_ij) + P3*sum(A_ij*A_ik) + P4*S_i*sum(A_ij^2)
//
// Where:
//   S_i = 4*pi*(R_i + probe)^2  (unoccluded sphere surface)
//   A_ij = pi*(R_i + probe)*(overlap_area)  (pairwise overlap term)
//
// Parameters are indexed by atom type (0-7):
//   0: Carbon (sp3)
//   1: Carbon (sp2)
//   2: Nitrogen
//   3: Oxygen
//   4: Sulfur
//   5: Phosphorus
//   6: Hydrogen
//   7: Other (default)
//-----------------------------------------------------------------------------

__constant__ float LCPO_P1[N_ATOM_TYPES] = {
    0.7887f,   // C sp3
    0.7887f,   // C sp2
    0.7887f,   // N
    0.7887f,   // O
    0.7887f,   // S
    0.7887f,   // P
    0.7887f,   // H
    0.7887f    // Other
};

__constant__ float LCPO_P2[N_ATOM_TYPES] = {
    -0.3556f,  // C sp3
    -0.3556f,  // C sp2
    -0.3556f,  // N
    -0.3556f,  // O
    -0.3556f,  // S
    -0.3556f,  // P
    -0.3556f,  // H
    -0.3556f   // Other
};

__constant__ float LCPO_P3[N_ATOM_TYPES] = {
    -0.0018f,  // C sp3
    -0.0018f,  // C sp2
    -0.0018f,  // N
    -0.0018f,  // O
    -0.0018f,  // S
    -0.0018f,  // P
    -0.0018f,  // H
    -0.0018f   // Other
};

__constant__ float LCPO_P4[N_ATOM_TYPES] = {
    0.0052f,   // C sp3
    0.0052f,   // C sp2
    0.0052f,   // N
    0.0052f,   // O
    0.0052f,   // S
    0.0052f,   // P
    0.0052f,   // H
    0.0052f    // Other
};

// VDW radii for each atom type (Angstrom)
__constant__ float VDW_RADII[N_ATOM_TYPES] = {
    1.70f,     // C sp3
    1.70f,     // C sp2
    1.55f,     // N
    1.52f,     // O
    1.80f,     // S
    1.80f,     // P
    1.20f,     // H
    1.70f      // Other (default C radius)
};

//-----------------------------------------------------------------------------
// Device helper: Compute overlap term A_ij between atoms i and j
//-----------------------------------------------------------------------------
__device__ __forceinline__ float compute_overlap_term(
    float ri,           // Radius of atom i + probe
    float rj,           // Radius of atom j + probe
    float dist          // Distance between atoms
) {
    // LCPO overlap term based on sphere-sphere intersection
    // A_ij = pi * ri * (ri - dist/2 - (ri^2 - rj^2)/(2*dist))

    if (dist < 0.001f) return 0.0f;  // Avoid division by zero

    float ri_sq = ri * ri;
    float rj_sq = rj * rj;

    // Check if spheres actually overlap
    if (dist >= ri + rj) return 0.0f;

    // Overlap area calculation
    float term1 = ri - dist * 0.5f;
    float term2 = (ri_sq - rj_sq) / (2.0f * dist);
    float A_ij = PI * ri * (term1 - term2);

    return fmaxf(A_ij, 0.0f);  // Clamp to non-negative
}

//-----------------------------------------------------------------------------
// KERNEL: Compute LCPO SASA for single frame
//
// Grid: (n_atoms + TILE_SIZE - 1) / TILE_SIZE blocks
// Block: TILE_SIZE threads
//
// Each thread computes SASA for one atom, iterating over all other atoms
// using shared memory tiling to reduce global memory bandwidth.
//-----------------------------------------------------------------------------
extern "C" __global__ void lcpo_sasa_kernel(
    const float* __restrict__ positions,    // [n_atoms * 3] XYZ coordinates
    const int* __restrict__ atom_types,     // [n_atoms] atom type indices (0-7)
    const float* __restrict__ radii,        // [n_atoms] VDW radii (or NULL to use defaults)
    const int n_atoms,                      // Total number of atoms
    float* __restrict__ sasa_out            // [n_atoms] per-atom SASA output
) {
    // Shared memory for tiled atom data
    __shared__ float4 tile_data[TILE_SIZE];  // x, y, z, radius+probe
    __shared__ int tile_types[TILE_SIZE];     // atom types

    // Global atom index for this thread
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load this atom's data
    float3 pos_i;
    float ri;       // radius + probe
    int type_i;
    float Si;       // Unoccluded surface area

    if (i < n_atoms) {
        pos_i.x = positions[i * 3 + 0];
        pos_i.y = positions[i * 3 + 1];
        pos_i.z = positions[i * 3 + 2];

        type_i = atom_types[i];
        if (type_i < 0 || type_i >= N_ATOM_TYPES) type_i = 7;  // Default to "Other"

        // Get radius (from array or constant memory default)
        float vdw_r = (radii != nullptr) ? radii[i] : VDW_RADII[type_i];
        ri = vdw_r + PROBE_RADIUS;

        // Unoccluded sphere surface area
        Si = 4.0f * PI * ri * ri;
    } else {
        pos_i = make_float3(0.0f, 0.0f, 0.0f);
        ri = 0.0f;
        type_i = 7;
        Si = 0.0f;
    }

    // Accumulate LCPO terms
    float sum_Aij = 0.0f;           // P2 term: sum of overlaps
    float sum_Aij_Aik = 0.0f;       // P3 term: product of overlaps (simplified)
    float sum_Aij_sq = 0.0f;        // P4 term: sum of squared overlaps

    // Iterate over all atoms using tiled approach
    const int n_tiles = (n_atoms + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < n_tiles; tile++) {
        // Cooperative load: each thread loads one atom into shared memory
        const int j_load = tile * TILE_SIZE + threadIdx.x;

        if (j_load < n_atoms) {
            float vdw_r = (radii != nullptr) ? radii[j_load] : VDW_RADII[atom_types[j_load]];
            tile_data[threadIdx.x] = make_float4(
                positions[j_load * 3 + 0],
                positions[j_load * 3 + 1],
                positions[j_load * 3 + 2],
                vdw_r + PROBE_RADIUS
            );
            tile_types[threadIdx.x] = atom_types[j_load];
        } else {
            tile_data[threadIdx.x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            tile_types[threadIdx.x] = 7;
        }

        __syncthreads();

        // Now compute overlaps with all atoms in this tile
        if (i < n_atoms) {
            #pragma unroll 8
            for (int k = 0; k < TILE_SIZE; k++) {
                const int j = tile * TILE_SIZE + k;

                // Skip self-interaction and out-of-bounds
                if (j >= n_atoms || j == i) continue;

                // Load neighbor data from shared memory
                float4 data_j = tile_data[k];
                float3 pos_j = make_float3(data_j.x, data_j.y, data_j.z);
                float rj = data_j.w;

                // Compute distance
                float dx = pos_j.x - pos_i.x;
                float dy = pos_j.y - pos_i.y;
                float dz = pos_j.z - pos_i.z;
                float dist_sq = dx * dx + dy * dy + dz * dz;

                // Early exit: skip if beyond cutoff
                if (dist_sq > LCPO_CUTOFF_SQ) continue;

                float dist = sqrtf(dist_sq);

                // Compute overlap term
                float Aij = compute_overlap_term(ri, rj, dist);

                if (Aij > 0.0f) {
                    sum_Aij += Aij;
                    sum_Aij_sq += Aij * Aij;
                    // P3 term approximation: use average Aij for cross-term
                    sum_Aij_Aik += Aij;  // Will be multiplied by average later
                }
            }
        }

        __syncthreads();
    }

    // Compute final SASA using LCPO formula
    if (i < n_atoms) {
        float P1 = LCPO_P1[type_i];
        float P2 = LCPO_P2[type_i];
        float P3 = LCPO_P3[type_i];
        float P4 = LCPO_P4[type_i];

        // P3 term: approximate as (sum_Aij)^2 / n_neighbors
        // This is a simplification that avoids O(N^3) triple product
        float n_neighbors = sum_Aij / (PI * ri * 0.5f + 0.001f);  // Estimate neighbor count
        float Aij_Aik_term = (n_neighbors > 1.0f) ?
            sum_Aij_Aik * sum_Aij_Aik / n_neighbors : 0.0f;

        float sasa = P1 * Si
                   + P2 * sum_Aij
                   + P3 * Aij_Aik_term
                   + P4 * Si * sum_Aij_sq;

        // Clamp to valid range [0, Si]
        sasa_out[i] = fmaxf(0.0f, fminf(sasa, Si));
    }
}

//-----------------------------------------------------------------------------
// KERNEL: Batched LCPO SASA for multiple frames (trajectory processing)
//
// Grid: (n_atoms, n_frames) - each block processes atoms for one frame
// Block: TILE_SIZE threads
//
// This kernel processes multiple conformations in parallel, ideal for
// cryptic site detection which needs SASA for 50-200 frames.
//-----------------------------------------------------------------------------
extern "C" __global__ void lcpo_sasa_batched_kernel(
    const float* __restrict__ positions,    // [n_frames * n_atoms * 3] XYZ coordinates
    const int* __restrict__ atom_types,     // [n_atoms] atom type indices (shared across frames)
    const float* __restrict__ radii,        // [n_atoms] VDW radii (or NULL)
    const int n_atoms,                      // Atoms per frame
    const int n_frames,                     // Number of frames
    float* __restrict__ sasa_out            // [n_frames * n_atoms] per-atom SASA output
) {
    // Shared memory for tiled atom data
    __shared__ float4 tile_data[TILE_SIZE];

    // Get frame and atom indices
    const int frame = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (frame >= n_frames) return;

    // Offset for this frame's data
    const int pos_offset = frame * n_atoms * 3;
    const int sasa_offset = frame * n_atoms;

    // Load this atom's data
    float3 pos_i;
    float ri;
    int type_i;
    float Si;

    if (i < n_atoms) {
        pos_i.x = positions[pos_offset + i * 3 + 0];
        pos_i.y = positions[pos_offset + i * 3 + 1];
        pos_i.z = positions[pos_offset + i * 3 + 2];

        type_i = atom_types[i];
        if (type_i < 0 || type_i >= N_ATOM_TYPES) type_i = 7;

        float vdw_r = (radii != nullptr) ? radii[i] : VDW_RADII[type_i];
        ri = vdw_r + PROBE_RADIUS;
        Si = 4.0f * PI * ri * ri;
    } else {
        pos_i = make_float3(0.0f, 0.0f, 0.0f);
        ri = 0.0f;
        type_i = 7;
        Si = 0.0f;
    }

    // Accumulate LCPO terms
    float sum_Aij = 0.0f;
    float sum_Aij_Aik = 0.0f;
    float sum_Aij_sq = 0.0f;

    const int n_tiles = (n_atoms + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < n_tiles; tile++) {
        const int j_load = tile * TILE_SIZE + threadIdx.x;

        if (j_load < n_atoms) {
            float vdw_r = (radii != nullptr) ? radii[j_load] : VDW_RADII[atom_types[j_load]];
            tile_data[threadIdx.x] = make_float4(
                positions[pos_offset + j_load * 3 + 0],
                positions[pos_offset + j_load * 3 + 1],
                positions[pos_offset + j_load * 3 + 2],
                vdw_r + PROBE_RADIUS
            );
        } else {
            tile_data[threadIdx.x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        __syncthreads();

        if (i < n_atoms) {
            #pragma unroll 8
            for (int k = 0; k < TILE_SIZE; k++) {
                const int j = tile * TILE_SIZE + k;
                if (j >= n_atoms || j == i) continue;

                float4 data_j = tile_data[k];
                float dx = data_j.x - pos_i.x;
                float dy = data_j.y - pos_i.y;
                float dz = data_j.z - pos_i.z;
                float dist_sq = dx * dx + dy * dy + dz * dz;

                if (dist_sq > LCPO_CUTOFF_SQ) continue;

                float dist = sqrtf(dist_sq);
                float rj = data_j.w;
                float Aij = compute_overlap_term(ri, rj, dist);

                if (Aij > 0.0f) {
                    sum_Aij += Aij;
                    sum_Aij_sq += Aij * Aij;
                    sum_Aij_Aik += Aij;
                }
            }
        }

        __syncthreads();
    }

    // Final SASA calculation
    if (i < n_atoms) {
        float P1 = LCPO_P1[type_i];
        float P2 = LCPO_P2[type_i];
        float P3 = LCPO_P3[type_i];
        float P4 = LCPO_P4[type_i];

        float n_neighbors = sum_Aij / (PI * ri * 0.5f + 0.001f);
        float Aij_Aik_term = (n_neighbors > 1.0f) ?
            sum_Aij_Aik * sum_Aij_Aik / n_neighbors : 0.0f;

        float sasa = P1 * Si + P2 * sum_Aij + P3 * Aij_Aik_term + P4 * Si * sum_Aij_sq;
        sasa_out[sasa_offset + i] = fmaxf(0.0f, fminf(sasa, Si));
    }
}

//-----------------------------------------------------------------------------
// KERNEL: Sum per-atom SASA to get total SASA per frame
//
// Uses block reduction for efficient parallel sum.
//-----------------------------------------------------------------------------
extern "C" __global__ void sum_sasa_kernel(
    const float* __restrict__ per_atom_sasa,  // [n_frames * n_atoms]
    const int n_atoms,
    const int n_frames,
    float* __restrict__ total_sasa            // [n_frames] total SASA per frame
) {
    __shared__ float smem[32];  // For block reduction

    const int frame = blockIdx.x;
    if (frame >= n_frames) return;

    const int offset = frame * n_atoms;

    // Each thread sums a strided subset of atoms
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < n_atoms; i += blockDim.x) {
        local_sum += per_atom_sasa[offset + i];
    }

    // Block reduction
    float total = block_reduce_sum_f32(local_sum, smem);

    if (threadIdx.x == 0) {
        total_sasa[frame] = total;
    }
}

//-----------------------------------------------------------------------------
// KERNEL: Compute per-residue SASA by summing atoms in each residue
//-----------------------------------------------------------------------------
extern "C" __global__ void residue_sasa_kernel(
    const float* __restrict__ per_atom_sasa,  // [n_atoms]
    const int* __restrict__ residue_map,       // [n_atoms] -> residue index
    const int n_atoms,
    const int n_residues,
    float* __restrict__ residue_sasa           // [n_residues]
) {
    const int res = blockIdx.x * blockDim.x + threadIdx.x;
    if (res >= n_residues) return;

    float sum = 0.0f;
    for (int i = 0; i < n_atoms; i++) {
        if (residue_map[i] == res) {
            sum += per_atom_sasa[i];
        }
    }

    residue_sasa[res] = sum;
}

//-----------------------------------------------------------------------------
// KERNEL: Optimized per-residue SASA using atomic add
// More efficient when atoms per residue is small
//-----------------------------------------------------------------------------
extern "C" __global__ void residue_sasa_atomic_kernel(
    const float* __restrict__ per_atom_sasa,  // [n_atoms]
    const int* __restrict__ residue_map,       // [n_atoms] -> residue index
    const int n_atoms,
    float* __restrict__ residue_sasa           // [n_residues] (must be zeroed first)
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_atoms) return;

    const int res = residue_map[i];
    if (res >= 0) {
        atomicAdd(&residue_sasa[res], per_atom_sasa[i]);
    }
}
