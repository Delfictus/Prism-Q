/**
 * PRISM Ligand Binding Site - Surface Accessibility (SASA) Kernel
 *
 * Optimized SASA computation with uniform spatial grid for O(N×27) neighbor lookup.
 * Implements Fibonacci sphere sampling for uniform surface point distribution.
 *
 * Performance target: <20ms for 10K atoms (100× improvement over naive O(N²))
 *
 * Key innovations:
 * 1. Uniform spatial grid eliminates O(N²) all-pairs checks → O(N×27)
 * 2. Fibonacci sphere for uniformly distributed surface samples
 * 3. Coalesced memory access patterns
 * 4. Atomic-free parallel SASA accumulation
 *
 * Copyright (c) 2024 PRISM Research Team | Delfictus I/O Inc.
 */

#include <cuda_runtime.h>

#define PI 3.14159265358979323846f
#define GOLDEN_RATIO 1.618033988749895f

// ═══════════════════════════════════════════════════════════════════════════
// UTILITY FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Compute grid cell index from 3D position
 */
__device__ __forceinline__ int get_cell_index(
    float px, float py, float pz,
    float origin_x, float origin_y, float origin_z,
    float cell_size,
    int grid_dim_x, int grid_dim_y, int grid_dim_z
) {
    int ix = min(max((int)floorf((px - origin_x) / cell_size), 0), grid_dim_x - 1);
    int iy = min(max((int)floorf((py - origin_y) / cell_size), 0), grid_dim_y - 1);
    int iz = min(max((int)floorf((pz - origin_z) / cell_size), 0), grid_dim_z - 1);
    return ix + iy * grid_dim_x + iz * grid_dim_x * grid_dim_y;
}

/**
 * Generate Fibonacci sphere point at index k out of n samples
 * Provides uniform distribution on sphere surface
 */
__device__ __forceinline__ void fibonacci_sphere_point(
    int k, int n, float* px, float* py, float* pz
) {
    float y = 1.0f - (2.0f * k) / (n - 1.0f);
    float radius = sqrtf(1.0f - y * y);
    float theta = 2.0f * PI * k / GOLDEN_RATIO;
    *px = cosf(theta) * radius;
    *py = y;
    *pz = sinf(theta) * radius;
}

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL 1: BUILD SPATIAL GRID
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Count atoms per grid cell (first pass for CSR construction)
 */
extern "C" __global__ void count_atoms_per_cell(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ z,
    int num_atoms,
    float origin_x,
    float origin_y,
    float origin_z,
    float cell_size,
    int grid_dim_x,
    int grid_dim_y,
    int grid_dim_z,
    int* __restrict__ cell_counts  // [num_cells] - output
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_atoms) return;

    int cell_idx = get_cell_index(
        x[idx], y[idx], z[idx],
        origin_x, origin_y, origin_z,
        cell_size, grid_dim_x, grid_dim_y, grid_dim_z
    );

    atomicAdd(&cell_counts[cell_idx], 1);
}

/**
 * Fill grid cells with atom indices (second pass after prefix sum)
 */
extern "C" __global__ void fill_grid_cells(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ z,
    int num_atoms,
    float origin_x,
    float origin_y,
    float origin_z,
    float cell_size,
    int grid_dim_x,
    int grid_dim_y,
    int grid_dim_z,
    const int* __restrict__ cell_start,  // [num_cells + 1] - after prefix sum
    int* __restrict__ atom_indices,      // [num_atoms] - output
    int* __restrict__ cell_offsets       // [num_cells] - working space
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_atoms) return;

    int cell_idx = get_cell_index(
        x[idx], y[idx], z[idx],
        origin_x, origin_y, origin_z,
        cell_size, grid_dim_x, grid_dim_y, grid_dim_z
    );

    int offset = atomicAdd(&cell_offsets[cell_idx], 1);
    int position = cell_start[cell_idx] + offset;
    atom_indices[position] = idx;
}

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL 2: COMPUTE SASA WITH SPATIAL GRID
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Compute SASA for each atom using spatial grid for neighbor lookup
 * Achieves O(N×27) complexity by checking only neighboring grid cells
 *
 * Algorithm:
 * 1. For each atom, generate Fibonacci sphere points on its surface
 * 2. For each surface point, check 27 neighboring grid cells (3×3×3)
 * 3. Count accessible points (not occluded by neighbors)
 * 4. SASA = (accessible_points / total_points) × 4πr²
 */
extern "C" __global__ void surface_accessibility_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ z,
    const float* __restrict__ radii,
    int num_atoms,
    int samples,
    float probe_radius,
    // Spatial grid parameters
    float origin_x,
    float origin_y,
    float origin_z,
    float cell_size,
    int grid_dim_x,
    int grid_dim_y,
    int grid_dim_z,
    const int* __restrict__ cell_start,
    const int* __restrict__ atom_indices,
    // Output
    float* __restrict__ out_sasa,
    unsigned char* __restrict__ out_surface
) {
    int atom_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom_idx >= num_atoms) return;

    float atom_x = x[atom_idx];
    float atom_y = y[atom_idx];
    float atom_z = z[atom_idx];
    float atom_radius = radii[atom_idx] + probe_radius;

    // Get grid cell for this atom
    int cell_idx = get_cell_index(
        atom_x, atom_y, atom_z,
        origin_x, origin_y, origin_z,
        cell_size, grid_dim_x, grid_dim_y, grid_dim_z
    );

    int cell_x = cell_idx % grid_dim_x;
    int cell_y = (cell_idx / grid_dim_x) % grid_dim_y;
    int cell_z = cell_idx / (grid_dim_x * grid_dim_y);

    // Count accessible surface points
    int accessible_count = 0;

    // Generate Fibonacci sphere points and test accessibility
    for (int s = 0; s < samples; s++) {
        // Generate surface point
        float px, py, pz;
        fibonacci_sphere_point(s, samples, &px, &py, &pz);

        float point_x = atom_x + px * atom_radius;
        float point_y = atom_y + py * atom_radius;
        float point_z = atom_z + pz * atom_radius;

        bool accessible = true;

        // Check 27 neighboring cells (3×3×3 cube)
        for (int dz = -1; dz <= 1 && accessible; dz++) {
            for (int dy = -1; dy <= 1 && accessible; dy++) {
                for (int dx = -1; dx <= 1 && accessible; dx++) {
                    int neighbor_x = cell_x + dx;
                    int neighbor_y = cell_y + dy;
                    int neighbor_z = cell_z + dz;

                    // Boundary check
                    if (neighbor_x < 0 || neighbor_x >= grid_dim_x ||
                        neighbor_y < 0 || neighbor_y >= grid_dim_y ||
                        neighbor_z < 0 || neighbor_z >= grid_dim_z) {
                        continue;
                    }

                    int neighbor_cell = neighbor_x +
                                       neighbor_y * grid_dim_x +
                                       neighbor_z * grid_dim_x * grid_dim_y;

                    int start = cell_start[neighbor_cell];
                    int end = cell_start[neighbor_cell + 1];

                    // Check all atoms in this cell
                    for (int i = start; i < end; i++) {
                        int other_idx = atom_indices[i];
                        if (other_idx == atom_idx) continue;

                        float other_x = x[other_idx];
                        float other_y = y[other_idx];
                        float other_z = z[other_idx];
                        float other_radius = radii[other_idx] + probe_radius;

                        // Check if surface point is inside other atom
                        float dx_val = point_x - other_x;
                        float dy_val = point_y - other_y;
                        float dz_val = point_z - other_z;
                        float dist_sq = dx_val * dx_val + dy_val * dy_val + dz_val * dz_val;

                        if (dist_sq < other_radius * other_radius) {
                            accessible = false;
                            break;
                        }
                    }
                }
            }
        }

        if (accessible) {
            accessible_count++;
        }
    }

    // Compute SASA: (accessible_fraction) × 4πr²
    float accessibility_fraction = (float)accessible_count / (float)samples;
    float surface_area = 4.0f * PI * atom_radius * atom_radius;
    out_sasa[atom_idx] = accessibility_fraction * surface_area;

    // Mark as surface atom if >10% accessible (configurable threshold)
    if (out_surface) {
        out_surface[atom_idx] = (unsigned char)(accessibility_fraction > 0.1f);
    }
}
