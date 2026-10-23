/**
 * NHS Hydrophobic Exclusion Mapping (HEM) CUDA Kernel
 *
 * Computes the 3D exclusion field from classified atoms using the "holographic negative"
 * principle: hydrophobic atoms create exclusion zones where water CANNOT exist.
 *
 * PHYSICS:
 *   - Hydrophobic atoms: Large exclusion radius (4.5Å) - water expelled
 *   - Polar atoms: Small exclusion radius (2.5Å) - can hydrogen bond
 *   - Charged atoms: Minimal exclusion (2.0Å) - strong hydration shell
 *   - Aromatic atoms: Special handling (UV bias targets)
 *
 * OUTPUT:
 *   exclusion_field[grid_dim³]: 0.0 = water allowed, 1.0 = water excluded
 *   pocket_probability[grid_dim³]: Likelihood of cryptic pocket (temporal variance)
 *
 * PERFORMANCE TARGET:
 *   <0.5ms for 10,000 atoms on RTX 3060
 *   O(N_atoms * grid_cells) but parallelized over grid cells
 *
 * REFERENCE: NHS Architecture - Hydrophobic Exclusion Mapping
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

// ============================================================================
// ATOM CLASSIFICATION CONSTANTS
// ============================================================================

// Atom type encoding (matches Rust enum AtomType)
#define ATOM_HYDROPHOBIC     0
#define ATOM_POLAR           1
#define ATOM_CHARGED_POS     2
#define ATOM_CHARGED_NEG     3
#define ATOM_AROMATIC        4
#define ATOM_BACKBONE        5
#define ATOM_WATER           6

// Exclusion radii in Angstroms (physics-based)
#define EXCLUSION_HYDROPHOBIC   4.5f    // Large - water expelled from hydrophobic region
#define EXCLUSION_POLAR         2.5f    // Moderate - can H-bond but not fully solvate
#define EXCLUSION_CHARGED       2.0f    // Small - strong hydration shell retained
#define EXCLUSION_AROMATIC      3.5f    // Medium - ring stacking, pi-water weak
#define EXCLUSION_BACKBONE      2.2f    // Small - exposed backbone prefers H-bond

// Gaussian softness for exclusion (smooth transition)
#define EXCLUSION_SOFTNESS      0.8f    // σ for Gaussian decay at boundary

// Maximum atoms per voxel interaction (for memory bounds)
#define MAX_ATOMS_PER_BLOCK     1024

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Get exclusion radius for atom type
 */
__device__ __forceinline__ float get_exclusion_radius(int atom_type) {
    switch (atom_type) {
        case ATOM_HYDROPHOBIC:  return EXCLUSION_HYDROPHOBIC;
        case ATOM_POLAR:        return EXCLUSION_POLAR;
        case ATOM_CHARGED_POS:  return EXCLUSION_CHARGED;
        case ATOM_CHARGED_NEG:  return EXCLUSION_CHARGED;
        case ATOM_AROMATIC:     return EXCLUSION_AROMATIC;
        case ATOM_BACKBONE:     return EXCLUSION_BACKBONE;
        case ATOM_WATER:        return 0.0f;  // Water doesn't exclude water
        default:                return 2.5f;  // Conservative default
    }
}

/**
 * Compute Gaussian exclusion contribution
 * Returns value in [0, 1] where 1 = fully excluded
 */
__device__ __forceinline__ float gaussian_exclusion(float distance, float radius) {
    if (distance >= radius) return 0.0f;

    // Soft Gaussian boundary
    float x = (distance - radius) / (EXCLUSION_SOFTNESS * radius);
    return expf(-0.5f * x * x);
}

/**
 * Convert 3D grid coordinates to linear index
 */
__device__ __forceinline__ int grid_to_linear(int x, int y, int z, int dim) {
    return z * dim * dim + y * dim + x;
}

// ============================================================================
// MAIN EXCLUSION KERNEL
// ============================================================================

/**
 * Compute exclusion field from classified atoms
 *
 * PARALLELISM: One thread per grid voxel
 * Each thread computes contributions from ALL atoms (brute force for correctness)
 *
 * @param atom_positions    Atom coordinates [n_atoms * 3] as (x,y,z) triples
 * @param atom_types        Atom type classification [n_atoms]
 * @param atom_charges      Partial charges [n_atoms] (for charge-weighted exclusion)
 * @param exclusion_field   OUTPUT: Exclusion values [grid_dim³]
 * @param n_atoms           Number of atoms
 * @param grid_dim          Grid dimension (cube: dim × dim × dim)
 * @param grid_origin_x     Grid origin X coordinate
 * @param grid_origin_y     Grid origin Y coordinate
 * @param grid_origin_z     Grid origin Z coordinate
 * @param grid_spacing      Grid spacing in Angstroms
 */
extern "C" __global__ void compute_exclusion_field(
    const float* __restrict__ atom_positions,
    const int* __restrict__ atom_types,
    const float* __restrict__ atom_charges,
    float* __restrict__ exclusion_field,
    const int n_atoms,
    const int grid_dim,
    const float grid_origin_x,
    const float grid_origin_y,
    const float grid_origin_z,
    const float grid_spacing
) {
    // 3D thread indexing for grid voxels
    const int gx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gy = blockIdx.y * blockDim.y + threadIdx.y;
    const int gz = blockIdx.z * blockDim.z + threadIdx.z;

    if (gx >= grid_dim || gy >= grid_dim || gz >= grid_dim) return;

    // Compute voxel center position
    const float voxel_x = grid_origin_x + (gx + 0.5f) * grid_spacing;
    const float voxel_y = grid_origin_y + (gy + 0.5f) * grid_spacing;
    const float voxel_z = grid_origin_z + (gz + 0.5f) * grid_spacing;

    // Accumulate exclusion from all atoms
    // Using max() for overlapping contributions (not sum - that would exceed 1.0)
    float max_exclusion = 0.0f;

    for (int a = 0; a < n_atoms; a++) {
        // Load atom data
        const float ax = atom_positions[a * 3 + 0];
        const float ay = atom_positions[a * 3 + 1];
        const float az = atom_positions[a * 3 + 2];
        const int atype = atom_types[a];

        // Skip water atoms (they don't exclude)
        if (atype == ATOM_WATER) continue;

        // Compute distance to voxel center
        const float dx = voxel_x - ax;
        const float dy = voxel_y - ay;
        const float dz = voxel_z - az;
        const float dist = sqrtf(dx * dx + dy * dy + dz * dz);

        // Get exclusion radius for this atom type
        const float radius = get_exclusion_radius(atype);

        // Early skip if clearly outside
        if (dist > radius * 2.0f) continue;

        // Compute Gaussian exclusion contribution
        float contribution = gaussian_exclusion(dist, radius);

        // Charge modulation: charged atoms have smaller effective exclusion
        // (water is attracted to charges, reducing exclusion)
        const float charge = atom_charges[a];
        if (fabsf(charge) > 0.3f) {
            contribution *= (1.0f - 0.3f * fabsf(charge));
        }

        max_exclusion = fmaxf(max_exclusion, contribution);
    }

    // Write to output grid
    const int linear_idx = grid_to_linear(gx, gy, gz, grid_dim);
    exclusion_field[linear_idx] = max_exclusion;
}

// ============================================================================
// OPTIMIZED EXCLUSION KERNEL WITH CELL LIST
// ============================================================================

/**
 * Compute exclusion field using cell-list acceleration
 *
 * PERFORMANCE: O(N_atoms/cells * grid_cells) instead of O(N_atoms * grid_cells)
 *
 * @param atom_positions    Atom coordinates [n_atoms * 3]
 * @param atom_types        Atom type classification [n_atoms]
 * @param atom_charges      Partial charges [n_atoms]
 * @param cell_start        Start index in sorted atoms for each cell [n_cells]
 * @param cell_end          End index in sorted atoms for each cell [n_cells]
 * @param sorted_atom_ids   Atom indices sorted by cell [n_atoms]
 * @param exclusion_field   OUTPUT: Exclusion values [grid_dim³]
 * @param n_atoms           Number of atoms
 * @param grid_dim          Grid dimension
 * @param cell_dim          Cell grid dimension (for neighbor lookup)
 * @param grid_origin_x/y/z Grid origin coordinates
 * @param grid_spacing      Grid spacing in Angstroms
 * @param cell_size         Cell size in Angstroms (>= max exclusion radius)
 */
extern "C" __global__ void compute_exclusion_field_cell_list(
    const float* __restrict__ atom_positions,
    const int* __restrict__ atom_types,
    const float* __restrict__ atom_charges,
    const int* __restrict__ cell_start,
    const int* __restrict__ cell_end,
    const int* __restrict__ sorted_atom_ids,
    float* __restrict__ exclusion_field,
    const int n_atoms,
    const int grid_dim,
    const int cell_dim,
    const float grid_origin_x,
    const float grid_origin_y,
    const float grid_origin_z,
    const float grid_spacing,
    const float cell_size
) {
    // 3D thread indexing for grid voxels
    const int gx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gy = blockIdx.y * blockDim.y + threadIdx.y;
    const int gz = blockIdx.z * blockDim.z + threadIdx.z;

    if (gx >= grid_dim || gy >= grid_dim || gz >= grid_dim) return;

    // Compute voxel center position
    const float voxel_x = grid_origin_x + (gx + 0.5f) * grid_spacing;
    const float voxel_y = grid_origin_y + (gy + 0.5f) * grid_spacing;
    const float voxel_z = grid_origin_z + (gz + 0.5f) * grid_spacing;

    // Determine which cell this voxel belongs to
    const int cell_x = (int)((voxel_x - grid_origin_x) / cell_size);
    const int cell_y = (int)((voxel_y - grid_origin_y) / cell_size);
    const int cell_z = (int)((voxel_z - grid_origin_z) / cell_size);

    float max_exclusion = 0.0f;

    // Iterate over neighboring cells (3x3x3 stencil)
    for (int dcz = -1; dcz <= 1; dcz++) {
        for (int dcy = -1; dcy <= 1; dcy++) {
            for (int dcx = -1; dcx <= 1; dcx++) {
                const int ncx = cell_x + dcx;
                const int ncy = cell_y + dcy;
                const int ncz = cell_z + dcz;

                // Bounds check
                if (ncx < 0 || ncx >= cell_dim ||
                    ncy < 0 || ncy >= cell_dim ||
                    ncz < 0 || ncz >= cell_dim) continue;

                // Linear cell index
                const int cell_idx = ncz * cell_dim * cell_dim + ncy * cell_dim + ncx;

                // Iterate over atoms in this cell
                const int start = cell_start[cell_idx];
                const int end = cell_end[cell_idx];

                for (int i = start; i < end; i++) {
                    const int a = sorted_atom_ids[i];

                    // Load atom data
                    const float ax = atom_positions[a * 3 + 0];
                    const float ay = atom_positions[a * 3 + 1];
                    const float az = atom_positions[a * 3 + 2];
                    const int atype = atom_types[a];

                    if (atype == ATOM_WATER) continue;

                    // Distance
                    const float dx = voxel_x - ax;
                    const float dy = voxel_y - ay;
                    const float dz = voxel_z - az;
                    const float dist = sqrtf(dx * dx + dy * dy + dz * dz);

                    const float radius = get_exclusion_radius(atype);
                    if (dist > radius * 2.0f) continue;

                    float contribution = gaussian_exclusion(dist, radius);

                    const float charge = atom_charges[a];
                    if (fabsf(charge) > 0.3f) {
                        contribution *= (1.0f - 0.3f * fabsf(charge));
                    }

                    max_exclusion = fmaxf(max_exclusion, contribution);
                }
            }
        }
    }

    const int linear_idx = grid_to_linear(gx, gy, gz, grid_dim);
    exclusion_field[linear_idx] = max_exclusion;
}

// ============================================================================
// WATER INFERENCE KERNEL
// ============================================================================

/**
 * Infer water density from exclusion field (holographic negative)
 *
 * PHYSICS: water_density = 1.0 - exclusion_field (with smoothing)
 *
 * Also computes local gradient for dewetting detection
 *
 * @param exclusion_field   Input: Exclusion values [grid_dim³]
 * @param water_density     OUTPUT: Inferred water density [grid_dim³]
 * @param water_gradient    OUTPUT: Gradient magnitude for dewetting detection [grid_dim³]
 * @param grid_dim          Grid dimension
 * @param grid_spacing      Grid spacing in Angstroms
 */
extern "C" __global__ void infer_water_density(
    const float* __restrict__ exclusion_field,
    float* __restrict__ water_density,
    float* __restrict__ water_gradient,
    const int grid_dim,
    const float grid_spacing
) {
    const int gx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gy = blockIdx.y * blockDim.y + threadIdx.y;
    const int gz = blockIdx.z * blockDim.z + threadIdx.z;

    if (gx >= grid_dim || gy >= grid_dim || gz >= grid_dim) return;

    const int idx = grid_to_linear(gx, gy, gz, grid_dim);

    // Water density is the "holographic negative" of exclusion
    const float exclusion = exclusion_field[idx];
    water_density[idx] = 1.0f - exclusion;

    // Compute gradient magnitude using central differences
    // For detecting dewetting boundaries (cryptic site edges)
    float grad_x = 0.0f, grad_y = 0.0f, grad_z = 0.0f;

    if (gx > 0 && gx < grid_dim - 1) {
        const int idx_m = grid_to_linear(gx - 1, gy, gz, grid_dim);
        const int idx_p = grid_to_linear(gx + 1, gy, gz, grid_dim);
        grad_x = (exclusion_field[idx_p] - exclusion_field[idx_m]) / (2.0f * grid_spacing);
    }

    if (gy > 0 && gy < grid_dim - 1) {
        const int idx_m = grid_to_linear(gx, gy - 1, gz, grid_dim);
        const int idx_p = grid_to_linear(gx, gy + 1, gz, grid_dim);
        grad_y = (exclusion_field[idx_p] - exclusion_field[idx_m]) / (2.0f * grid_spacing);
    }

    if (gz > 0 && gz < grid_dim - 1) {
        const int idx_m = grid_to_linear(gx, gy, gz - 1, grid_dim);
        const int idx_p = grid_to_linear(gx, gy, gz + 1, grid_dim);
        grad_z = (exclusion_field[idx_p] - exclusion_field[idx_m]) / (2.0f * grid_spacing);
    }

    water_gradient[idx] = sqrtf(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z);
}

// ============================================================================
// POCKET PROBABILITY ACCUMULATOR
// ============================================================================

/**
 * Accumulate pocket probability from temporal variance
 *
 * PHYSICS: Cryptic pockets show HIGH temporal variance in water density
 *          (opening/closing events) compared to stable regions
 *
 * Uses Welford's online algorithm for stable variance computation
 *
 * @param water_density         Current frame water density [grid_dim³]
 * @param pocket_mean           Running mean [grid_dim³]
 * @param pocket_m2             Running M2 for variance [grid_dim³]
 * @param frame_count           Number of frames processed
 * @param grid_dim              Grid dimension
 */
extern "C" __global__ void accumulate_pocket_probability(
    const float* __restrict__ water_density,
    float* __restrict__ pocket_mean,
    float* __restrict__ pocket_m2,
    const int frame_count,
    const int grid_dim
) {
    const int gx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gy = blockIdx.y * blockDim.y + threadIdx.y;
    const int gz = blockIdx.z * blockDim.z + threadIdx.z;

    if (gx >= grid_dim || gy >= grid_dim || gz >= grid_dim) return;

    const int idx = grid_to_linear(gx, gy, gz, grid_dim);

    // Welford's online algorithm
    const float density = water_density[idx];
    const float old_mean = pocket_mean[idx];
    const float delta = density - old_mean;
    const float new_mean = old_mean + delta / (float)(frame_count + 1);
    const float delta2 = density - new_mean;

    pocket_mean[idx] = new_mean;
    pocket_m2[idx] += delta * delta2;
}

/**
 * Finalize pocket probability from accumulated variance
 *
 * @param pocket_mean           Final mean (not used, for debug)
 * @param pocket_m2             Accumulated M2
 * @param pocket_probability    OUTPUT: Probability [0,1] that voxel is cryptic pocket
 * @param frame_count           Total frames processed
 * @param grid_dim              Grid dimension
 * @param variance_threshold    Variance threshold for pocket classification
 */
extern "C" __global__ void finalize_pocket_probability(
    const float* __restrict__ pocket_mean,
    const float* __restrict__ pocket_m2,
    float* __restrict__ pocket_probability,
    const int frame_count,
    const int grid_dim,
    const float variance_threshold
) {
    const int gx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gy = blockIdx.y * blockDim.y + threadIdx.y;
    const int gz = blockIdx.z * blockDim.z + threadIdx.z;

    if (gx >= grid_dim || gy >= grid_dim || gz >= grid_dim) return;

    const int idx = grid_to_linear(gx, gy, gz, grid_dim);

    // Compute variance
    float variance = 0.0f;
    if (frame_count > 1) {
        variance = pocket_m2[idx] / (float)(frame_count - 1);
    }

    // Convert variance to probability using sigmoid
    // High variance → high pocket probability
    float z = (variance - variance_threshold) / (variance_threshold * 0.5f);
    pocket_probability[idx] = 1.0f / (1.0f + expf(-z));
}

// ============================================================================
// AROMATIC ATOM DETECTION (FOR UV BIAS)
// ============================================================================

/**
 * Mark aromatic atoms for UV bias targeting
 *
 * PHYSICS: Aromatic rings (Trp, Tyr, Phe) absorb 280nm UV
 *          Water is transparent at 280nm
 *          This enables pump-probe causal detection of cryptic pockets
 *
 * @param atom_types            Atom classifications [n_atoms]
 * @param atom_residues         Residue index for each atom [n_atoms]
 * @param aromatic_mask         OUTPUT: 1 if aromatic, 0 otherwise [n_atoms]
 * @param uv_target_intensity   OUTPUT: UV absorption intensity [n_atoms]
 * @param n_atoms               Number of atoms
 */
extern "C" __global__ void detect_aromatic_targets(
    const int* __restrict__ atom_types,
    const int* __restrict__ atom_residues,
    int* __restrict__ aromatic_mask,
    float* __restrict__ uv_target_intensity,
    const int n_atoms
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_atoms) return;

    const int atype = atom_types[idx];

    if (atype == ATOM_AROMATIC) {
        aromatic_mask[idx] = 1;
        // Intensity varies by ring type (Trp > Tyr > Phe)
        // For now, use uniform intensity - can be refined with residue type
        uv_target_intensity[idx] = 1.0f;
    } else {
        aromatic_mask[idx] = 0;
        uv_target_intensity[idx] = 0.0f;
    }
}

// ============================================================================
// GRID UTILITIES
// ============================================================================

/**
 * Reset grid to zeros
 */
extern "C" __global__ void reset_grid(
    float* __restrict__ grid,
    const int total_elements
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        grid[idx] = 0.0f;
    }
}

/**
 * Reset integer grid to zeros
 */
extern "C" __global__ void reset_grid_int(
    int* __restrict__ grid,
    const int total_elements
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        grid[idx] = 0;
    }
}
