/**
 * PRISM Cryptic Site Detection - GPU Probe Scoring Kernel
 *
 * Implements FTMap-style probe scoring and clustering for identifying
 * binding hot spots. This is the most compute-intensive part of cryptic
 * site detection - scoring millions of probe positions against protein atoms.
 *
 * Copyright (c) 2024 PRISM Research Team | Delfictus I/O Inc.
 */

#include <cuda_runtime.h>
#include <cmath>

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define TILE_SIZE 32

// Physical constants
#define LJ_EPSILON 0.1f          // Lennard-Jones well depth (kcal/mol)
#define ELECTROSTATIC_K 332.0f   // Coulomb constant (kcal·Å/mol·e²)
#define DIELECTRIC 4.0f          // Effective dielectric constant
#define PROBE_RADIUS 1.7f        // Probe radius (Angstroms)

// Van der Waals radii for common elements
#define VDW_C 1.70f
#define VDW_N 1.55f
#define VDW_O 1.52f
#define VDW_S 1.80f
#define VDW_H 1.20f
#define VDW_DEFAULT 1.50f

// Energy thresholds
#define FAVORABLE_ENERGY -0.5f
#define CLASH_DISTANCE 2.0f

/**
 * Get VDW radius for an atom type (encoded as integer)
 */
__device__ __forceinline__ float get_vdw_radius(int atom_type) {
    switch (atom_type) {
        case 0: return VDW_C;  // Carbon
        case 1: return VDW_N;  // Nitrogen
        case 2: return VDW_O;  // Oxygen
        case 3: return VDW_S;  // Sulfur
        case 4: return VDW_H;  // Hydrogen
        default: return VDW_DEFAULT;
    }
}

/**
 * Score all probe positions against protein atoms
 *
 * Uses simplified Lennard-Jones potential + electrostatics:
 * E_total = E_vdw + E_elec + E_desolv
 *
 * Each thread handles one probe position.
 */
extern "C" __global__ void score_probes(
    const float* __restrict__ probe_positions,  // [n_probes, 3]
    const float* __restrict__ atom_coords,      // [n_atoms, 3]
    const int* __restrict__ atom_types,         // [n_atoms] element type 0-4
    const float* __restrict__ atom_charges,     // [n_atoms] partial charges
    const int* __restrict__ atom_residues,      // [n_atoms] residue indices
    float* __restrict__ probe_energies,         // [n_probes] total energy
    float* __restrict__ probe_vdw,              // [n_probes] VDW contribution
    float* __restrict__ probe_elec,             // [n_probes] electrostatic contribution
    int* __restrict__ probe_valid,              // [n_probes] 1 if valid, 0 if clashing
    const int n_probes,
    const int n_atoms,
    const float probe_charge                    // Probe partial charge
) {
    // Shared memory for atom tile
    __shared__ float s_atom_x[TILE_SIZE];
    __shared__ float s_atom_y[TILE_SIZE];
    __shared__ float s_atom_z[TILE_SIZE];
    __shared__ int s_atom_type[TILE_SIZE];
    __shared__ float s_atom_charge[TILE_SIZE];

    const int probe_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (probe_idx >= n_probes) return;

    // Load probe position
    float px = probe_positions[probe_idx * 3 + 0];
    float py = probe_positions[probe_idx * 3 + 1];
    float pz = probe_positions[probe_idx * 3 + 2];

    float total_vdw = 0.0f;
    float total_elec = 0.0f;
    int is_valid = 1;

    // Process atoms in tiles for coalesced memory access
    for (int tile_start = 0; tile_start < n_atoms; tile_start += TILE_SIZE) {
        // Cooperative loading of atom tile into shared memory
        int tile_idx = tile_start + threadIdx.x % TILE_SIZE;
        if (tile_idx < n_atoms && threadIdx.x < TILE_SIZE) {
            s_atom_x[threadIdx.x] = atom_coords[tile_idx * 3 + 0];
            s_atom_y[threadIdx.x] = atom_coords[tile_idx * 3 + 1];
            s_atom_z[threadIdx.x] = atom_coords[tile_idx * 3 + 2];
            s_atom_type[threadIdx.x] = atom_types[tile_idx];
            s_atom_charge[threadIdx.x] = atom_charges[tile_idx];
        }
        __syncthreads();

        // Process atoms in this tile
        int tile_end = min(TILE_SIZE, n_atoms - tile_start);
        for (int t = 0; t < tile_end; t++) {
            float ax = s_atom_x[t];
            float ay = s_atom_y[t];
            float az = s_atom_z[t];

            float dx = px - ax;
            float dy = py - ay;
            float dz = pz - az;
            float dist_sq = dx * dx + dy * dy + dz * dz;
            float dist = sqrtf(dist_sq);

            // Clash detection
            if (dist < CLASH_DISTANCE) {
                is_valid = 0;
            }

            if (dist > 0.1f && dist < 10.0f) {
                // Lennard-Jones 6-12 potential (simplified)
                float atom_vdw = get_vdw_radius(s_atom_type[t]);
                float sigma = (PROBE_RADIUS + atom_vdw) * 0.5f;
                float sigma_r = sigma / dist;
                float sigma_r6 = sigma_r * sigma_r * sigma_r * sigma_r * sigma_r * sigma_r;
                float sigma_r12 = sigma_r6 * sigma_r6;

                float e_lj = 4.0f * LJ_EPSILON * (sigma_r12 - sigma_r6);
                total_vdw += e_lj;

                // Electrostatic potential
                float e_elec = (ELECTROSTATIC_K * probe_charge * s_atom_charge[t]) / (DIELECTRIC * dist);
                total_elec += e_elec;
            }
        }
        __syncthreads();
    }

    // Desolvation penalty (simplified - based on burial)
    float desolv = 0.0f;  // Would need surface area calculation

    float total_energy = total_vdw + total_elec + desolv;

    // Write results
    probe_energies[probe_idx] = total_energy;
    probe_vdw[probe_idx] = total_vdw;
    probe_elec[probe_idx] = total_elec;
    probe_valid[probe_idx] = is_valid;
}

/**
 * Filter favorable probes (energy < threshold)
 */
extern "C" __global__ void filter_favorable_probes(
    const float* __restrict__ probe_energies,
    const int* __restrict__ probe_valid,
    int* __restrict__ favorable_mask,
    int* __restrict__ favorable_count,
    const int n_probes,
    const float energy_threshold
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_probes) return;

    int is_favorable = (probe_valid[idx] && probe_energies[idx] < energy_threshold) ? 1 : 0;
    favorable_mask[idx] = is_favorable;

    if (is_favorable) {
        atomicAdd(favorable_count, 1);
    }
}

/**
 * Compact favorable probes using prefix sum (stream compaction)
 */
extern "C" __global__ void compact_favorable_probes(
    const float* __restrict__ probe_positions,
    const float* __restrict__ probe_energies,
    const int* __restrict__ favorable_mask,
    const int* __restrict__ prefix_sum,
    float* __restrict__ out_positions,
    float* __restrict__ out_energies,
    const int n_probes
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_probes) return;

    if (favorable_mask[idx]) {
        int out_idx = prefix_sum[idx];
        out_positions[out_idx * 3 + 0] = probe_positions[idx * 3 + 0];
        out_positions[out_idx * 3 + 1] = probe_positions[idx * 3 + 1];
        out_positions[out_idx * 3 + 2] = probe_positions[idx * 3 + 2];
        out_energies[out_idx] = probe_energies[idx];
    }
}

/**
 * Compute pairwise distances between favorable probes for clustering
 */
extern "C" __global__ void probe_pairwise_distances(
    const float* __restrict__ positions,  // [n, 3]
    float* __restrict__ distances,        // [n, n] or just upper triangle
    const int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= n || j >= n || j <= i) return;

    float dx = positions[i * 3 + 0] - positions[j * 3 + 0];
    float dy = positions[i * 3 + 1] - positions[j * 3 + 1];
    float dz = positions[i * 3 + 2] - positions[j * 3 + 2];
    float dist = sqrtf(dx * dx + dy * dy + dz * dz);

    distances[i * n + j] = dist;
    distances[j * n + i] = dist;
}

/**
 * DBSCAN-style clustering step: find neighbors within epsilon
 */
extern "C" __global__ void find_cluster_neighbors(
    const float* __restrict__ positions,
    int* __restrict__ neighbor_counts,
    int* __restrict__ neighbor_lists,   // [n, max_neighbors]
    const int n,
    const int max_neighbors,
    const float epsilon
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float px = positions[idx * 3 + 0];
    float py = positions[idx * 3 + 1];
    float pz = positions[idx * 3 + 2];

    int count = 0;
    float eps_sq = epsilon * epsilon;

    for (int j = 0; j < n && count < max_neighbors; j++) {
        if (j == idx) continue;

        float dx = px - positions[j * 3 + 0];
        float dy = py - positions[j * 3 + 1];
        float dz = pz - positions[j * 3 + 2];

        if (dx * dx + dy * dy + dz * dz < eps_sq) {
            neighbor_lists[idx * max_neighbors + count] = j;
            count++;
        }
    }

    neighbor_counts[idx] = count;
}

/**
 * Assign probes to clusters (iterative label propagation)
 */
extern "C" __global__ void propagate_cluster_labels(
    const int* __restrict__ neighbor_counts,
    const int* __restrict__ neighbor_lists,
    int* __restrict__ labels,
    int* __restrict__ changed,
    const int n,
    const int max_neighbors,
    const int min_pts               // Minimum points for core
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int my_label = labels[idx];
    int my_count = neighbor_counts[idx];

    // Only core points propagate labels
    if (my_count < min_pts) return;

    // Propagate to neighbors
    for (int i = 0; i < my_count; i++) {
        int neighbor = neighbor_lists[idx * max_neighbors + i];
        int neighbor_label = labels[neighbor];

        // Take minimum label (union-find style)
        if (my_label < neighbor_label) {
            atomicMin(&labels[neighbor], my_label);
            *changed = 1;
        }
    }
}

/**
 * Compute per-residue binding propensity from clustered probes
 */
extern "C" __global__ void compute_residue_binding_score(
    const float* __restrict__ probe_positions,
    const float* __restrict__ probe_energies,
    const int* __restrict__ cluster_labels,
    const float* __restrict__ atom_coords,
    const int* __restrict__ atom_residues,
    float* __restrict__ residue_probe_count,
    float* __restrict__ residue_energy_sum,
    const int n_probes,
    const int n_atoms,
    const int n_residues,
    const float contact_distance
) {
    const int probe_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (probe_idx >= n_probes) return;

    // Skip noise points (label = -1 or unassigned)
    if (cluster_labels[probe_idx] < 0) return;

    float px = probe_positions[probe_idx * 3 + 0];
    float py = probe_positions[probe_idx * 3 + 1];
    float pz = probe_positions[probe_idx * 3 + 2];
    float energy = probe_energies[probe_idx];
    float contact_sq = contact_distance * contact_distance;

    // Find contacting residues
    for (int a = 0; a < n_atoms; a++) {
        float dx = px - atom_coords[a * 3 + 0];
        float dy = py - atom_coords[a * 3 + 1];
        float dz = pz - atom_coords[a * 3 + 2];

        if (dx * dx + dy * dy + dz * dz < contact_sq) {
            int res_idx = atom_residues[a];
            if (res_idx >= 0 && res_idx < n_residues) {
                atomicAdd(&residue_probe_count[res_idx], 1.0f);
                atomicAdd(&residue_energy_sum[res_idx], energy);
            }
        }
    }
}

/**
 * Normalize residue binding scores to [0, 1]
 */
extern "C" __global__ void normalize_binding_scores(
    const float* __restrict__ probe_counts,
    const float* __restrict__ energy_sums,
    float* __restrict__ binding_scores,
    const int n_residues,
    const float max_count,
    const float energy_scale
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_residues) return;

    float count = probe_counts[idx];
    float energy = energy_sums[idx];

    if (count > 0) {
        float mean_energy = energy / count;
        float count_score = fminf(count / max_count, 1.0f);
        float energy_score = fminf(-mean_energy * energy_scale, 1.0f);
        binding_scores[idx] = 0.6f * count_score + 0.4f * fmaxf(energy_score, 0.0f);
    } else {
        binding_scores[idx] = 0.0f;
    }
}

/**
 * Generate probe grid positions around protein
 *
 * Creates a 3D grid of candidate probe positions within the binding shell
 * (between min_dist and max_dist from protein surface)
 */
extern "C" __global__ void generate_probe_grid(
    const float* __restrict__ atom_coords,
    const int n_atoms,
    float* __restrict__ probe_positions,
    int* __restrict__ probe_count,
    const float min_x, const float max_x,
    const float min_y, const float max_y,
    const float min_z, const float max_z,
    const float grid_spacing,
    const float min_dist,
    const float max_dist,
    const int max_probes
) {
    // Calculate grid dimensions
    int nx = (int)((max_x - min_x) / grid_spacing) + 1;
    int ny = (int)((max_y - min_y) / grid_spacing) + 1;
    int nz = (int)((max_z - min_z) / grid_spacing) + 1;

    int grid_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_grid = nx * ny * nz;
    if (grid_idx >= total_grid) return;

    // Convert linear index to 3D coordinates
    int iz = grid_idx / (nx * ny);
    int iy = (grid_idx - iz * nx * ny) / nx;
    int ix = grid_idx - iz * nx * ny - iy * nx;

    float px = min_x + ix * grid_spacing;
    float py = min_y + iy * grid_spacing;
    float pz = min_z + iz * grid_spacing;

    // Find minimum distance to any atom
    float min_atom_dist_sq = 1e30f;
    float min_dist_sq = min_dist * min_dist;
    float max_dist_sq = max_dist * max_dist;

    for (int a = 0; a < n_atoms; a++) {
        float dx = px - atom_coords[a * 3 + 0];
        float dy = py - atom_coords[a * 3 + 1];
        float dz = pz - atom_coords[a * 3 + 2];
        float d_sq = dx * dx + dy * dy + dz * dz;
        min_atom_dist_sq = fminf(min_atom_dist_sq, d_sq);
    }

    // Check if within binding shell
    if (min_atom_dist_sq >= min_dist_sq && min_atom_dist_sq <= max_dist_sq) {
        int out_idx = atomicAdd(probe_count, 1);
        if (out_idx < max_probes) {
            probe_positions[out_idx * 3 + 0] = px;
            probe_positions[out_idx * 3 + 1] = py;
            probe_positions[out_idx * 3 + 2] = pz;
        }
    }
}
