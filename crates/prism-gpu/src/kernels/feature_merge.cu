//! Feature Merge "Zipper" Kernel
//!
//! ARCHITECT DIRECTIVE: PHASE 1.5 - FEATURE MERGE KERNEL
//!
//! Combines 136-dimensional main features with 4-dimensional cryptic features
//! into a single 140-dimensional output for DQN consumption.
//!
//! Memory Layout:
//! - Input main_features:    [136 floats per residue] (from mega_fused_batch)
//! - Input cryptic_features: [4 floats per residue]   (from cryptic pipeline)
//! - Output combined:        [140 floats per residue] (for DQN)
//!
//! Performance: Memory-bound kernel optimized for high occupancy with many warps.

#include <cuda_runtime.h>

/// Feature Merge "Zipper" Kernel
///
/// Merges main features (136-dim) + cryptic features (4-dim) -> combined (140-dim)
///
/// @param main_features    Input main features [n_residues * 136]
/// @param cryptic_features Input cryptic features [n_residues * 4]
/// @param combined_features Output combined features [n_residues * 140]
/// @param n_residues       Number of residues to process
extern "C" __global__ void __launch_bounds__(128)
feature_merge_kernel(
    const float* __restrict__ main_features,
    const float* __restrict__ cryptic_features,
    float* __restrict__ combined_features,
    const int n_residues
) {
    // Calculate global residue index
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check - early exit if beyond residue count
    if (idx >= n_residues) return;

    // Calculate base pointers for current residue
    const float* src_main = main_features + idx * 136;
    const float* src_cryptic = cryptic_features + idx * 4;
    float* dst = combined_features + idx * 140;

    // Copy 136 main features to positions [0..135]
    #pragma unroll 8
    for (int i = 0; i < 136; ++i) {
        dst[i] = src_main[i];
    }

    // Copy 4 cryptic features to positions [136..139]
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        dst[136 + i] = src_cryptic[i];
    }
}

/// Feature Merge kernel launcher configuration
///
/// @param n_residues Number of residues to process
/// @return dim3 values for grid and block dimensions
extern "C" __host__ __device__
void get_feature_merge_config(int n_residues, dim3* grid, dim3* block) {
    // Use 128 threads per block for high occupancy
    block->x = 128;
    block->y = 1;
    block->z = 1;

    // Calculate grid size to cover all residues
    grid->x = (n_residues + block->x - 1) / block->x;
    grid->y = 1;
    grid->z = 1;
}

/// Validate kernel configuration for given parameters
///
/// @param n_residues Number of residues
/// @return 0 if valid, error code otherwise
extern "C" __host__
int validate_feature_merge_config(int n_residues) {
    if (n_residues <= 0) return -1;
    if (n_residues > (1 << 20)) return -2; // Max ~1M residues
    return 0;
}

/*
ARCHITECT'S TECHNICAL NOTES:

1. MEMORY ACCESS PATTERN:
   - Sequential access to both input arrays (coalesced)
   - Sequential write to output array (coalesced)
   - Each thread processes exactly one residue

2. OCCUPANCY OPTIMIZATION:
   - __launch_bounds__(128): Ensures many warps per SM
   - Memory-bound workload benefits from high warp count
   - 128 threads/block provides good balance

3. PERFORMANCE CHARACTERISTICS:
   - Memory bandwidth limited (not compute limited)
   - ~350 GB/s theoretical on RTX 3060
   - Each residue: (136+4)*4 = 560 bytes read + 140*4 = 560 bytes write
   - Total: 1120 bytes per residue

4. UNROLL PRAGMAS:
   - Main loop unrolled by 8 for better instruction level parallelism
   - Cryptic loop fully unrolled (only 4 iterations)

5. VALIDATION TARGETS:
   - combined[135] should equal main_features[135] (last main feature)
   - combined[136] should equal cryptic_features[0] (first cryptic feature)
   - combined[139] should equal cryptic_features[3] (last cryptic feature)
*/