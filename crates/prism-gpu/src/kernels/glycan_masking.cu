/**
 * GPU-Side Glycan Masking Kernel (Stage 0)
 *
 * Eliminates CPU preprocessing by performing N-X-S/T sequon detection and
 * 10Å sphere masking entirely on the GPU using parallel pattern matching
 * and shared memory optimization.
 *
 * This kernel replaces CPU-based glycan shield modeling with a high-performance
 * GPU implementation that keeps all data in VRAM.
 *
 * Copyright (c) 2024 PRISM Research Team | Delfictus I/O Inc.
 */

#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define GLYCAN_RADIUS 10.0f
#define GLYCAN_RADIUS_SQ (GLYCAN_RADIUS * GLYCAN_RADIUS)
#define MAX_RESIDUES_PER_STRUCTURE 2048

// Amino acid encoding for sequon pattern matching
#define AA_N 14  // Asparagine
#define AA_S 19  // Serine
#define AA_T 20  // Threonine
#define AA_P 16  // Proline (excluded in X position)

/**
 * GPU-optimized sequon pattern detector
 * Finds N-X-S/T patterns where X != P using parallel string matching
 */
__device__ bool detect_sequon_at_position(
    const unsigned char* sequence,
    int pos,
    int sequence_length
) {
    // Check bounds for triplet
    if (pos + 2 >= sequence_length) return false;

    unsigned char aa1 = sequence[pos];
    unsigned char aa2 = sequence[pos + 1];
    unsigned char aa3 = sequence[pos + 2];

    // N-X-S/T pattern where X != P
    return (aa1 == AA_N) &&
           (aa2 != AA_P) &&
           (aa3 == AA_S || aa3 == AA_T);
}

/**
 * Compute 3D distance squared between two CA atoms
 */
__device__ __forceinline__ float distance_squared_3d(
    float x1, float y1, float z1,
    float x2, float y2, float z2
) {
    float dx = x1 - x2;
    float dy = y1 - y2;
    float dz = z1 - z2;
    return dx*dx + dy*dy + dz*dz;
}

/**
 * Main glycan masking kernel
 *
 * Each thread block processes one protein structure.
 * Each thread within the block processes a chunk of residues.
 */
extern "C" __global__ void __launch_bounds__(256, 4)
glycan_masking_kernel(
    const unsigned char* __restrict__ sequences,     // [batch_size, max_seq_len] amino acid sequences
    const float* __restrict__ ca_coords,             // [batch_size, max_residues, 3] CA coordinates
    const int* __restrict__ sequence_lengths,        // [batch_size] actual sequence lengths
    const int* __restrict__ residue_counts,          // [batch_size] actual residue counts
    const int* __restrict__ structure_offsets,       // [batch_size] offsets into coordinate arrays
    unsigned int* __restrict__ glycan_bitmasks,      // [batch_size, (max_residues+31)/32] output bitmasks
    const int batch_size,
    const int max_seq_len,
    const int max_residues
) {
    const int structure_idx = blockIdx.x;
    if (structure_idx >= batch_size) return;

    const int tid = threadIdx.x;
    const int sequence_length = sequence_lengths[structure_idx];
    const int n_residues = residue_counts[structure_idx];
    const int coord_offset = structure_offsets[structure_idx];

    // Shared memory for detected sequon positions
    __shared__ int sequon_positions[64]; // Max 64 sequons per structure
    __shared__ int num_sequons;

    if (tid == 0) {
        num_sequons = 0;
    }
    __syncthreads();

    // Phase 1: Parallel sequon detection
    // Each thread scans a portion of the sequence
    const unsigned char* seq = sequences + structure_idx * max_seq_len;

    for (int pos = tid; pos < sequence_length - 2; pos += blockDim.x) {
        if (detect_sequon_at_position(seq, pos, sequence_length)) {
            // Atomically add sequon position to shared list
            int sequon_idx = atomicAdd(&num_sequons, 1);
            if (sequon_idx < 64) {
                sequon_positions[sequon_idx] = pos;
            }
        }
    }
    __syncthreads();

    // Phase 2: Parallel glycan sphere masking
    // Each thread processes a chunk of residues
    const float* coords = ca_coords + coord_offset * 3;
    unsigned int* bitmask = glycan_bitmasks + structure_idx * ((max_residues + 31) / 32);

    for (int res_idx = tid; res_idx < n_residues; res_idx += blockDim.x) {
        bool is_shielded = false;

        // Get this residue's CA coordinates
        float res_x = coords[res_idx * 3];
        float res_y = coords[res_idx * 3 + 1];
        float res_z = coords[res_idx * 3 + 2];

        // Check distance to all detected sequon Asn residues
        for (int s = 0; s < num_sequons && s < 64; s++) {
            int sequon_pos = sequon_positions[s];

            // Bounds check
            if (sequon_pos < n_residues) {
                float sequon_x = coords[sequon_pos * 3];
                float sequon_y = coords[sequon_pos * 3 + 1];
                float sequon_z = coords[sequon_pos * 3 + 2];

                float dist_sq = distance_squared_3d(
                    res_x, res_y, res_z,
                    sequon_x, sequon_y, sequon_z
                );

                if (dist_sq < GLYCAN_RADIUS_SQ) {
                    is_shielded = true;
                    break;
                }
            }
        }

        // Set bit in bitmask if residue is glycan-shielded
        if (is_shielded) {
            int word_idx = res_idx / 32;
            int bit_idx = res_idx % 32;
            atomicOr(&bitmask[word_idx], 1U << bit_idx);
        }
    }
}

/**
 * Utility kernel to apply glycan masking to SASA values
 * Multiplies shielded residues' SASA by occlusion factor (0.3)
 */
extern "C" __global__ void __launch_bounds__(256, 4)
apply_glycan_sasa_masking(
    const float* __restrict__ raw_sasa,              // [total_residues] input SASA
    const unsigned int* __restrict__ glycan_bitmasks, // [total_residues/32] glycan shielding
    float* __restrict__ effective_sasa,              // [total_residues] output masked SASA
    const int total_residues,
    const float occlusion_factor                     // 0.3 for 70% occlusion
) {
    const int res_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (res_idx >= total_residues) return;

    // Check if this residue is glycan-shielded
    int word_idx = res_idx / 32;
    int bit_idx = res_idx % 32;
    bool is_shielded = (glycan_bitmasks[word_idx] >> bit_idx) & 1;

    // Apply occlusion factor if shielded
    float sasa = raw_sasa[res_idx];
    effective_sasa[res_idx] = is_shielded ? (sasa * occlusion_factor) : sasa;
}

/**
 * Kernel to check cryptic site validity based on glycan masking
 * Excludes glycan-shielded residues from cryptic site candidates
 */
extern "C" __global__ void __launch_bounds__(256, 4)
validate_cryptic_candidates(
    const unsigned int* __restrict__ glycan_bitmasks, // [total_residues/32] glycan shielding
    const float* __restrict__ cryptic_scores,         // [total_residues] raw cryptic scores
    float* __restrict__ valid_cryptic_scores,         // [total_residues] filtered scores
    const int total_residues
) {
    const int res_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (res_idx >= total_residues) return;

    // Check if this residue is glycan-shielded
    int word_idx = res_idx / 32;
    int bit_idx = res_idx % 32;
    bool is_shielded = (glycan_bitmasks[word_idx] >> bit_idx) & 1;

    // Zero out cryptic scores for glycan-shielded residues
    float score = cryptic_scores[res_idx];
    valid_cryptic_scores[res_idx] = is_shielded ? 0.0f : score;
}

/**
 * Debug kernel to count glycan-shielded residues
 */
extern "C" __global__ void
count_shielded_residues(
    const unsigned int* __restrict__ glycan_bitmasks,
    int* __restrict__ shielded_counts,
    const int* __restrict__ residue_counts,
    const int batch_size,
    const int max_residues
) {
    const int structure_idx = blockIdx.x;
    if (structure_idx >= batch_size) return;

    const int tid = threadIdx.x;
    const int n_residues = residue_counts[structure_idx];
    const unsigned int* bitmask = glycan_bitmasks + structure_idx * ((max_residues + 31) / 32);

    __shared__ int local_count;
    if (tid == 0) local_count = 0;
    __syncthreads();

    // Count bits in parallel
    for (int res_idx = tid; res_idx < n_residues; res_idx += blockDim.x) {
        int word_idx = res_idx / 32;
        int bit_idx = res_idx % 32;
        if ((bitmask[word_idx] >> bit_idx) & 1) {
            atomicAdd(&local_count, 1);
        }
    }
    __syncthreads();

    if (tid == 0) {
        shielded_counts[structure_idx] = local_count;
    }
}