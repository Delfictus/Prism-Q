/**
 * Bio-Chemistry Feature Extraction CUDA Kernel
 *
 * GPU-accelerated computation of the three bio-chemistry features:
 * 1. Hydrophobic Exposure Delta ("Grease Signal")
 * 2. Local Displacement Anisotropy ("Hinge Signal")
 * 3. Electrostatic Frustration ("Spring Signal")
 *
 * DESIGN PRINCIPLES:
 * - Exact numerical equivalence with CPU implementation
 * - FP32 precision throughout (no shortcuts)
 * - Parallel reduction for aggregates
 * - Coalesced memory access patterns
 *
 * Author: PRISMdevTeam + Claude Opus 4.5
 * Date: 2026-01-09
 */

#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// CONSTANTS (matching CPU implementation exactly)
// ============================================================================

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define MAX_ATOMS 32768
#define MAX_RESIDUES 4096
#define MAX_CA_ATOMS 4096
#define MAX_CHARGED_ATOMS 2048

// Epsilon for numerical stability
#define EPSILON 1e-6f

// ============================================================================
// DEVICE HELPER FUNCTIONS
// ============================================================================

/**
 * Warp-level reduction for sum
 */
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * Warp-level reduction for max
 */
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

/**
 * Block-level reduction for sum using shared memory
 */
__device__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    // Warp-level reduction
    val = warp_reduce_sum(val);

    // Write warp results to shared memory
    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    // Final reduction in first warp
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : 0.0f;

    if (warp_id == 0) {
        val = warp_reduce_sum(val);
    }

    return val;
}

/**
 * Block-level reduction for max using shared memory
 */
__device__ float block_reduce_max(float val, float* shared) {
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    // Warp-level reduction
    val = warp_reduce_max(val);

    // Write warp results to shared memory
    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    // Final reduction in first warp
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : -1e30f;

    if (warp_id == 0) {
        val = warp_reduce_max(val);
    }

    return val;
}

// ============================================================================
// KERNEL 1: HYDROPHOBIC EXPOSURE DELTA
// ============================================================================

/**
 * Count neighbors for atoms in a residue (parallel over atoms)
 *
 * Each thread handles one atom and counts neighbors within cutoff.
 * Results are summed across threads for the residue total.
 */
__global__ void kernel_count_neighbors(
    const float* __restrict__ positions,      // [n_atoms * 3]
    const int* __restrict__ atom_to_residue,  // [n_atoms]
    const int* __restrict__ residue_start,    // [n_residues] - first atom index
    const int* __restrict__ residue_end,      // [n_residues] - last atom index + 1
    float* __restrict__ neighbor_counts,      // [n_residues] output
    int n_atoms,
    int n_residues,
    float cutoff_sq
) {
    extern __shared__ float smem[];

    int res_idx = blockIdx.x;
    if (res_idx >= n_residues) return;

    int atom_start = residue_start[res_idx];
    int atom_end = residue_end[res_idx];
    int n_res_atoms = atom_end - atom_start;

    float local_count = 0.0f;

    // Each thread handles atoms strided by blockDim.x
    for (int local_idx = threadIdx.x; local_idx < n_res_atoms; local_idx += blockDim.x) {
        int atom_idx = atom_start + local_idx;

        float ax = positions[atom_idx * 3];
        float ay = positions[atom_idx * 3 + 1];
        float az = positions[atom_idx * 3 + 2];

        // Count neighbors (all other atoms within cutoff)
        for (int j = 0; j < n_atoms; j++) {
            if (j == atom_idx) continue;

            float bx = positions[j * 3];
            float by = positions[j * 3 + 1];
            float bz = positions[j * 3 + 2];

            float dx = ax - bx;
            float dy = ay - by;
            float dz = az - bz;
            float dist_sq = dx * dx + dy * dy + dz * dz;

            if (dist_sq < cutoff_sq) {
                local_count += 1.0f;
            }
        }
    }

    // Block-level reduction
    float total = block_reduce_sum(local_count, smem);

    if (threadIdx.x == 0) {
        neighbor_counts[res_idx] = total;
    }
}

/**
 * Compute hydrophobic exposure delta for target residues
 *
 * Formula: sum((initial_neighbors - current_neighbors) * hydrophobicity) / n_targets
 */
__global__ void kernel_hydrophobic_exposure(
    const float* __restrict__ initial_counts,   // [n_residues]
    const float* __restrict__ current_counts,   // [n_residues]
    const float* __restrict__ hydrophobicity,   // [n_residues]
    const int* __restrict__ target_residues,    // [n_targets]
    float* __restrict__ result,                 // [1] output
    int n_targets
) {
    extern __shared__ float smem[];

    float local_sum = 0.0f;

    for (int i = threadIdx.x; i < n_targets; i += blockDim.x) {
        int res_idx = target_residues[i];
        float exposure_delta = initial_counts[res_idx] - current_counts[res_idx];
        float hydro = hydrophobicity[res_idx];
        local_sum += exposure_delta * hydro;
    }

    float total = block_reduce_sum(local_sum, smem);

    if (threadIdx.x == 0) {
        result[0] = (n_targets > 0) ? total / (float)n_targets : 0.0f;
    }
}

// ============================================================================
// KERNEL 2: DISPLACEMENT ANISOTROPY (HINGE DETECTION)
// ============================================================================

/**
 * Compute Cα displacement magnitudes
 */
__global__ void kernel_ca_displacements(
    const float* __restrict__ positions,         // [n_atoms * 3]
    const float* __restrict__ initial_positions, // [n_atoms * 3]
    const int* __restrict__ ca_indices,          // [n_ca]
    float* __restrict__ displacements,           // [n_ca] output
    int n_ca
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_ca) return;

    int atom_idx = ca_indices[idx];
    int base = atom_idx * 3;

    float dx = positions[base] - initial_positions[base];
    float dy = positions[base + 1] - initial_positions[base + 1];
    float dz = positions[base + 2] - initial_positions[base + 2];

    displacements[idx] = sqrtf(dx * dx + dy * dy + dz * dz);
}

/**
 * Compute anisotropy for each internal Cα
 *
 * Anisotropy = |displacement[i] - avg(displacement[i-1], displacement[i+1])|
 */
__global__ void kernel_anisotropy(
    const float* __restrict__ displacements,  // [n_ca]
    float* __restrict__ anisotropies,         // [n_ca] output
    float* __restrict__ max_anisotropy,       // [1] output
    float* __restrict__ mean_anisotropy,      // [1] output
    int n_ca
) {
    extern __shared__ float smem[];

    float local_max = 0.0f;
    float local_sum = 0.0f;
    int local_count = 0;

    // Each thread handles multiple Cα atoms
    for (int i = 1 + threadIdx.x; i < n_ca - 1; i += blockDim.x) {
        float d_prev = displacements[i - 1];
        float d_curr = displacements[i];
        float d_next = displacements[i + 1];

        float neighbor_avg = (d_prev + d_next) * 0.5f;
        float aniso = fabsf(d_curr - neighbor_avg);

        anisotropies[i] = aniso;
        local_max = fmaxf(local_max, aniso);
        local_sum += aniso;
        local_count++;
    }

    // Reduce max
    float global_max = block_reduce_max(local_max, smem);
    __syncthreads();

    // Reduce sum and count
    float global_sum = block_reduce_sum(local_sum, smem);
    __syncthreads();

    // Reduce count (reuse shared memory)
    smem[threadIdx.x] = (float)local_count;
    __syncthreads();
    float global_count = block_reduce_sum(smem[threadIdx.x], smem);

    if (threadIdx.x == 0) {
        max_anisotropy[0] = global_max;
        mean_anisotropy[0] = (global_count > 0) ? global_sum / global_count : 0.0f;
    }
}

// ============================================================================
// KERNEL 3: ELECTROSTATIC FRUSTRATION
// ============================================================================

/**
 * Compute electrostatic frustration (Coulombic repulsion of like charges)
 *
 * Formula: sum(q_i * q_j / r_ij) for all charged pairs within cutoff
 * where q_i * q_j > 0 (like charges)
 */
__global__ void kernel_electrostatic_frustration(
    const float* __restrict__ positions,        // [n_atoms * 3]
    const float* __restrict__ charges,          // [n_atoms]
    const int* __restrict__ charged_indices,    // [n_charged]
    float* __restrict__ result,                 // [1] output
    int n_charged,
    float cutoff_sq
) {
    extern __shared__ float smem[];

    float local_frustration = 0.0f;

    // Each thread handles a subset of pairs
    // Total pairs = n_charged * (n_charged - 1) / 2
    int total_pairs = n_charged * (n_charged - 1) / 2;

    for (int pair_idx = threadIdx.x + blockIdx.x * blockDim.x;
         pair_idx < total_pairs;
         pair_idx += blockDim.x * gridDim.x) {

        // Convert linear pair index to (i, j) where i < j
        // Using quadratic formula: i = floor((2n-1 - sqrt((2n-1)^2 - 8*pair_idx)) / 2)
        int n = n_charged;
        float disc = (2.0f * n - 1.0f) * (2.0f * n - 1.0f) - 8.0f * pair_idx;
        int i = (int)floorf((2.0f * n - 1.0f - sqrtf(disc)) * 0.5f);
        int j = pair_idx - i * (2 * n - i - 1) / 2 + i + 1;

        if (i >= n_charged || j >= n_charged || i >= j) continue;

        int idx_i = charged_indices[i];
        int idx_j = charged_indices[j];

        float q_i = charges[idx_i];
        float q_j = charges[idx_j];

        // Skip if either charge is negligible
        if (fabsf(q_i) < 0.01f || fabsf(q_j) < 0.01f) continue;

        // Only count like-charge repulsion (frustration)
        if (q_i * q_j <= 0.0f) continue;

        int base_i = idx_i * 3;
        int base_j = idx_j * 3;

        float dx = positions[base_i] - positions[base_j];
        float dy = positions[base_i + 1] - positions[base_j + 1];
        float dz = positions[base_i + 2] - positions[base_j + 2];
        float dist_sq = dx * dx + dy * dy + dz * dz;

        if (dist_sq < cutoff_sq && dist_sq > EPSILON) {
            float r = sqrtf(dist_sq);
            // Coulombic energy (positive for like charges = frustration)
            local_frustration += q_i * q_j / r;
        }
    }

    // Block-level reduction
    float total = block_reduce_sum(local_frustration, smem);

    // Atomic add across blocks
    if (threadIdx.x == 0 && total != 0.0f) {
        atomicAdd(result, total);
    }
}

// ============================================================================
// UNIFIED KERNEL: ALL THREE FEATURES IN ONE PASS
// ============================================================================

/**
 * Compute all three bio-chemistry features in a single kernel launch
 *
 * This is more efficient than separate kernels when data fits in L2 cache.
 *
 * Output:
 *   result[0] = hydrophobic_exposure_delta (normalized to [0,1])
 *   result[1] = max_anisotropy (normalized to [0,1])
 *   result[2] = electrostatic_frustration (normalized to [0,1])
 */
extern "C" __global__ void kernel_bio_chemistry_unified(
    // Position data
    const float* __restrict__ positions,
    const float* __restrict__ initial_positions,
    int n_atoms,

    // Residue metadata
    const float* __restrict__ hydrophobicity,  // per residue
    const int* __restrict__ atom_to_residue,
    int n_residues,

    // Target residues for hydrophobic calc
    const int* __restrict__ target_residues,
    int n_targets,

    // Cα indices for anisotropy
    const int* __restrict__ ca_indices,
    int n_ca,

    // Charge data for frustration
    const float* __restrict__ charges,
    const int* __restrict__ charged_indices,
    int n_charged,

    // Parameters
    float neighbor_cutoff_sq,

    // Output [3]
    float* __restrict__ result
) {
    // This unified kernel is launched with a single block
    // For production, separate kernels may be more efficient for large proteins

    extern __shared__ float smem[];

    // Partition shared memory
    float* hydro_smem = smem;
    float* aniso_smem = smem + BLOCK_SIZE;
    float* frust_smem = smem + 2 * BLOCK_SIZE;

    float hydro_sum = 0.0f;
    float max_aniso = 0.0f;
    float aniso_sum = 0.0f;
    int aniso_count = 0;
    float frustration = 0.0f;

    // === HYDROPHOBIC EXPOSURE ===
    // Each thread handles subset of target residues
    for (int t = threadIdx.x; t < n_targets; t += blockDim.x) {
        int res_idx = target_residues[t];
        float hydro = hydrophobicity[res_idx];

        // Count neighbors for this residue (simplified - count atoms in residue)
        float initial_neighbors = 0.0f;
        float current_neighbors = 0.0f;

        for (int a = 0; a < n_atoms; a++) {
            if (atom_to_residue[a] != res_idx) continue;

            float ax = positions[a * 3];
            float ay = positions[a * 3 + 1];
            float az = positions[a * 3 + 2];

            float ax0 = initial_positions[a * 3];
            float ay0 = initial_positions[a * 3 + 1];
            float az0 = initial_positions[a * 3 + 2];

            // Count current neighbors
            for (int b = 0; b < n_atoms; b++) {
                if (a == b) continue;
                float bx = positions[b * 3];
                float by = positions[b * 3 + 1];
                float bz = positions[b * 3 + 2];
                float dist_sq = (ax-bx)*(ax-bx) + (ay-by)*(ay-by) + (az-bz)*(az-bz);
                if (dist_sq < neighbor_cutoff_sq) current_neighbors += 1.0f;
            }

            // Count initial neighbors
            for (int b = 0; b < n_atoms; b++) {
                if (a == b) continue;
                float bx = initial_positions[b * 3];
                float by = initial_positions[b * 3 + 1];
                float bz = initial_positions[b * 3 + 2];
                float dist_sq = (ax0-bx)*(ax0-bx) + (ay0-by)*(ay0-by) + (az0-bz)*(az0-bz);
                if (dist_sq < neighbor_cutoff_sq) initial_neighbors += 1.0f;
            }
        }

        float exposure_delta = initial_neighbors - current_neighbors;
        hydro_sum += exposure_delta * hydro;
    }

    // === ANISOTROPY ===
    // Compute Cα displacements first (store in shared memory if small enough)
    for (int i = 1 + threadIdx.x; i < n_ca - 1; i += blockDim.x) {
        int ca_prev = ca_indices[i - 1];
        int ca_curr = ca_indices[i];
        int ca_next = ca_indices[i + 1];

        // Displacement magnitudes
        auto disp = [&](int idx) {
            int base = idx * 3;
            float dx = positions[base] - initial_positions[base];
            float dy = positions[base + 1] - initial_positions[base + 1];
            float dz = positions[base + 2] - initial_positions[base + 2];
            return sqrtf(dx*dx + dy*dy + dz*dz);
        };

        float d_prev = disp(ca_prev);
        float d_curr = disp(ca_curr);
        float d_next = disp(ca_next);

        float neighbor_avg = (d_prev + d_next) * 0.5f;
        float aniso = fabsf(d_curr - neighbor_avg);

        max_aniso = fmaxf(max_aniso, aniso);
        aniso_sum += aniso;
        aniso_count++;
    }

    // === ELECTROSTATIC FRUSTRATION ===
    int total_pairs = n_charged * (n_charged - 1) / 2;
    for (int pair = threadIdx.x; pair < total_pairs; pair += blockDim.x) {
        // Convert linear index to (i, j)
        int n = n_charged;
        float disc = (2.0f*n - 1.0f)*(2.0f*n - 1.0f) - 8.0f*pair;
        int i = (int)floorf((2.0f*n - 1.0f - sqrtf(disc)) * 0.5f);
        int j = pair - i*(2*n - i - 1)/2 + i + 1;

        if (i >= n_charged || j >= n_charged) continue;

        int idx_i = charged_indices[i];
        int idx_j = charged_indices[j];

        float q_i = charges[idx_i];
        float q_j = charges[idx_j];

        if (fabsf(q_i) < 0.01f || fabsf(q_j) < 0.01f) continue;
        if (q_i * q_j <= 0.0f) continue;  // Only like charges

        float dx = positions[idx_i*3] - positions[idx_j*3];
        float dy = positions[idx_i*3+1] - positions[idx_j*3+1];
        float dz = positions[idx_i*3+2] - positions[idx_j*3+2];
        float dist_sq = dx*dx + dy*dy + dz*dz;

        if (dist_sq < neighbor_cutoff_sq && dist_sq > EPSILON) {
            frustration += q_i * q_j / sqrtf(dist_sq);
        }
    }

    // === REDUCTIONS ===
    float total_hydro = block_reduce_sum(hydro_sum, hydro_smem);
    __syncthreads();

    float global_max_aniso = block_reduce_max(max_aniso, aniso_smem);
    __syncthreads();

    float total_frustration = block_reduce_sum(frustration, frust_smem);

    // === OUTPUT (normalized) ===
    if (threadIdx.x == 0) {
        // Hydrophobic: normalize to [0,1], typical range [-10, 10]
        float hydro_norm = (n_targets > 0) ? total_hydro / n_targets : 0.0f;
        result[0] = fminf(fmaxf((hydro_norm / 10.0f) + 0.5f, 0.0f), 1.0f);

        // Anisotropy: normalize, typical hinge motion 0-5 Angstroms
        result[1] = fminf(global_max_aniso / 5.0f, 1.0f);

        // Frustration: normalize, typical range 0-10
        result[2] = fminf(total_frustration / 10.0f, 1.0f);
    }
}

// ============================================================================
// C-STYLE WRAPPER FOR RUST FFI
// ============================================================================

extern "C" {

/**
 * Launch the unified bio-chemistry feature kernel
 *
 * Returns 0 on success, non-zero on error.
 */
int launch_bio_chemistry_features(
    cudaStream_t stream,

    // Position data
    const float* d_positions,
    const float* d_initial_positions,
    int n_atoms,

    // Residue metadata
    const float* d_hydrophobicity,
    const int* d_atom_to_residue,
    int n_residues,

    // Target residues
    const int* d_target_residues,
    int n_targets,

    // Cα indices
    const int* d_ca_indices,
    int n_ca,

    // Charge data
    const float* d_charges,
    const int* d_charged_indices,
    int n_charged,

    // Parameters
    float neighbor_cutoff,

    // Output
    float* d_result  // [3]
) {
    // Clear result
    cudaMemsetAsync(d_result, 0, 3 * sizeof(float), stream);

    float cutoff_sq = neighbor_cutoff * neighbor_cutoff;

    // Launch unified kernel with enough shared memory
    int shared_mem_size = 3 * BLOCK_SIZE * sizeof(float);

    kernel_bio_chemistry_unified<<<1, BLOCK_SIZE, shared_mem_size, stream>>>(
        d_positions,
        d_initial_positions,
        n_atoms,
        d_hydrophobicity,
        d_atom_to_residue,
        n_residues,
        d_target_residues,
        n_targets,
        d_ca_indices,
        n_ca,
        d_charges,
        d_charged_indices,
        n_charged,
        cutoff_sq,
        d_result
    );

    return cudaGetLastError();
}

}  // extern "C"
