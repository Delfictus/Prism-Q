/**
 * ARCHITECT DIRECTIVE: PHASE 1 - GPU Glycan Shield (Stage 0)
 *
 * Implements N-X-S/T sequon detection and 10Å sphere masking entirely on GPU.
 * Uses proper CUDA float3 math with norm3df intrinsic for distance calculations.
 * Optimized with __shfl_sync to broadcast sequon coordinates.
 *
 * Copyright (c) 2024 PRISM Research Team | Delfictus I/O Inc.
 */

#include <stdint.h>

#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define GLYCAN_RADIUS 10.0f
#define MAX_SEQUONS_PER_STRUCTURE 64

// Amino acid character encoding
#define AA_N 'N'
#define AA_S 'S'
#define AA_T 'T'
#define AA_P 'P'

/**
 * Device function: Check if position starts an N-X-S/T sequon (X != P)
 */
__device__ __forceinline__ bool is_sequon(const char* sequence, int pos, int n_residues) {
    if (pos + 2 >= n_residues) return false;

    char aa1 = sequence[pos];
    char aa2 = sequence[pos + 1];
    char aa3 = sequence[pos + 2];

    return (aa1 == AA_N) && (aa2 != AA_P) && (aa3 == AA_S || aa3 == AA_T);
}

/**
 * ARCHITECT SPECIFIED KERNEL: GPU Glycan Masking
 *
 * @param sequence    [n_residues] Amino acid sequence (single-letter codes)
 * @param coords      [n_residues] CA atom coordinates as float3
 * @param n_residues  Number of residues in the structure
 * @param mask        [n_residues] Output mask (1 = Shielded, 0 = Exposed)
 */
extern "C" __global__ void __launch_bounds__(256, 4)
glycan_mask_kernel(
    const char* __restrict__ sequence,
    const float3* __restrict__ coords,
    const int n_residues,
    uint8_t* __restrict__ mask
) {
    const int tid = threadIdx.x;
    const int residue_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Shared memory for sequon coordinates (max 64 sequons per structure)
    __shared__ float3 sequon_coords[MAX_SEQUONS_PER_STRUCTURE];
    __shared__ int num_sequons;

    // Initialize shared memory
    if (tid == 0) {
        num_sequons = 0;
    }
    __syncthreads();

    // PHASE 1: Thread per residue - Check for sequons
    if (residue_idx < n_residues) {
        if (is_sequon(sequence, residue_idx, n_residues)) {
            // Atomically claim a slot in shared memory for this sequon
            int sequon_slot = atomicAdd(&num_sequons, 1);
            if (sequon_slot < MAX_SEQUONS_PER_STRUCTURE) {
                sequon_coords[sequon_slot] = coords[residue_idx];
            }
        }
    }
    __syncthreads();

    // PHASE 2: Check all residues against all sequons using __shfl_sync optimization
    if (residue_idx < n_residues) {
        uint8_t is_masked = 0;
        float3 my_coord = coords[residue_idx];

        // Process sequons in warp-sized chunks for __shfl_sync broadcasting
        for (int sequon_base = 0; sequon_base < num_sequons; sequon_base += WARP_SIZE) {
            int sequon_idx = sequon_base + (tid % WARP_SIZE);

            // Load sequon coordinate (or dummy if out of bounds)
            float3 sequon_coord;
            if (sequon_idx < num_sequons) {
                sequon_coord = sequon_coords[sequon_idx];
            } else {
                // Dummy coordinate far away
                sequon_coord = make_float3(1000.0f, 1000.0f, 1000.0f);
            }

            // Broadcast sequon coordinates across warp using __shfl_sync
            for (int lane = 0; lane < WARP_SIZE && (sequon_base + lane) < num_sequons; lane++) {
                float sequon_x = __shfl_sync(0xFFFFFFFF, sequon_coord.x, lane);
                float sequon_y = __shfl_sync(0xFFFFFFFF, sequon_coord.y, lane);
                float sequon_z = __shfl_sync(0xFFFFFFFF, sequon_coord.z, lane);

                // Calculate distance using CUDA intrinsic norm3df (no operator overloading)
                float distance = norm3df(
                    my_coord.x - sequon_x,
                    my_coord.y - sequon_y,
                    my_coord.z - sequon_z
                );

                // Check if within glycan shield radius
                if (distance < GLYCAN_RADIUS) {
                    is_masked = 1;
                    break; // No need to check more sequons
                }
            }

            if (is_masked) break; // Early exit if already masked
        }

        // Write result to global memory
        mask[residue_idx] = is_masked;
    }
}

/**
 * Utility kernel: Count masked residues for validation
 */
extern "C" __global__ void
count_masked_residues(
    const uint8_t* __restrict__ mask,
    int* __restrict__ count,
    const int n_residues
) {
    __shared__ int local_count;
    const int tid = threadIdx.x;

    if (tid == 0) local_count = 0;
    __syncthreads();

    // Count masked residues in parallel
    for (int idx = tid; idx < n_residues; idx += blockDim.x) {
        if (mask[idx] == 1) {
            atomicAdd(&local_count, 1);
        }
    }
    __syncthreads();

    if (tid == 0) {
        atomicAdd(count, local_count);
    }
}

/**
 * Debug kernel: Extract sequon positions for validation
 */
extern "C" __global__ void
extract_sequon_positions(
    const char* __restrict__ sequence,
    int* __restrict__ sequon_positions,
    int* __restrict__ num_sequons_found,
    const int n_residues
) {
    const int tid = threadIdx.x;
    const int residue_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (residue_idx < n_residues) {
        if (is_sequon(sequence, residue_idx, n_residues)) {
            int slot = atomicAdd(num_sequons_found, 1);
            if (slot < MAX_SEQUONS_PER_STRUCTURE) {
                sequon_positions[slot] = residue_idx;
            }
        }
    }
}