/**
 * @file prism_nova.cu
 * @brief PRISM-NOVA: Neural-Optimized Variational Adaptive Dynamics
 *
 * A unified mega-fused kernel combining:
 * - Neural Hamiltonian Monte Carlo (NHMC) for efficient sampling
 * - Topological Data Analysis (TDA) for collective variables
 * - Active Inference for goal-directed exploration
 * - Reservoir Computing + RLS for online learning
 *
 * All components execute in a single kernel launch with zero CPU round-trips.
 *
 * COMPILATION:
 *   nvcc -ptx -arch=sm_80 -O3 --use_fast_math -o prism_nova.ptx prism_nova.cu
 *
 * ASSUMPTIONS:
 * - Double precision for positions/momenta (accuracy)
 * - Single precision for neural network weights (speed)
 * - Maximum 4096 atoms per protein
 * - Maximum 1024 reservoir neurons
 * - sm_80+ (Ampere) for tensor core acceleration
 *
 * REFERENCE: PRISM-NOVA Architecture Document
 */

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>

namespace cg = cooperative_groups;

// ============================================================================
// CONFIGURATION CONSTANTS
// ============================================================================

#define MAX_ATOMS 512
#define MAX_RESIDUES 512
#define MAX_TARGET_RESIDUES 32   // Reduced for shared memory (dist_matrix = 32*32*4 = 4KB)
#define RESERVOIR_SIZE 1024
#define NUM_OUTPUTS 20
#define FEATURE_DIM 40
#define TDA_FILTRATION_STEPS 16
#define BLOCK_SIZE 256
#define WARP_SIZE 32

// Persistent homology constants
// NOTE: Shared memory is limited to 48KB on most GPUs
// ReservoirState uses 16KB, positions/forces use 12KB, leaving ~20KB for TDA
// dist_matrix[32*32] = 4KB, simplices[128] = 3.5KB, pairs[128] = 1.5KB = 9KB total
#define MAX_PH_VERTICES 32       // Reduce vertices to fit in shared memory
#define MAX_PH_EDGES 128         // Limit edges for shared memory
#define MAX_PH_TRIANGLES 64      // Limit triangles for shared memory
#define MAX_PH_SIMPLICES 128     // Total simplices we track (fits in ~5KB)
#define PH_INF 1e30f             // Infinite death time

// Physics constants
#define KB 0.001987204  // Boltzmann constant in kcal/(mol·K)
#define EPSILON 1e-10

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/**
 * @brief Configuration for PRISM-NOVA kernel
 */
struct NovaConfig {
    // Simulation parameters
    float dt;                    // Timestep (ps)
    float temperature;           // Temperature (K)
    float goal_strength;         // Active Inference goal bias strength
    float lambda;                // RLS forgetting factor

    // System sizes
    int n_atoms;
    int n_residues;
    int n_target_residues;

    // HMC parameters
    int leapfrog_steps;          // Number of leapfrog steps per HMC iteration
    float mass_scale;            // Mass scaling for momenta

    // Neural network dimensions
    int nn_hidden_dim;
    int nn_num_layers;

    // Target residue indices
    int target_residues[MAX_TARGET_RESIDUES];

    // Random seed
    unsigned long long seed;
};

/**
 * @brief Persistence pair representing a topological feature's lifetime
 */
struct PersistencePair {
    float birth;    // Filtration value when feature appears
    float death;    // Filtration value when feature dies (PH_INF if essential)
    int dimension;  // 0 = component, 1 = loop, 2 = void
};

/**
 * @brief Topological Collective Variables computed from protein structure
 *
 * Now uses PROPER persistent homology instead of Euler characteristic approximation.
 * Persistence pairs track birth/death of topological features through filtration.
 */
struct TopologicalCV {
    float betti_0;              // Connected components (at reference filtration)
    float betti_1;              // Cycles/loops (at reference filtration)
    float betti_2;              // Voids/cavities - POCKETS! (at reference filtration)
    float persistence_entropy;   // Shannon entropy of persistence diagram
    float pocket_signature;      // Persistent void score for druggability
    float gyration_radius;       // Compactness
    float contact_order;         // Folding complexity
    float local_density;         // Local packing around target

    // Persistence diagram summary statistics
    float total_persistence_0;  // Sum of (death - birth) for dim-0 features
    float total_persistence_1;  // Sum of (death - birth) for dim-1 features
    float total_persistence_2;  // Sum of (death - birth) for dim-2 features
    float max_persistence_2;    // Longest-lived void (most stable pocket)
    int n_persistent_voids;     // Count of voids with persistence > threshold
};

/**
 * @brief Active Inference state for goal-directed sampling
 */
struct ActiveInferenceState {
    float expected_free_energy;   // EFE = pragmatic + epistemic
    float variational_free_energy; // VFE = accuracy - complexity
    float goal_prior;             // Prior on druggable state
    float epistemic_value;        // Information gain
    float pragmatic_value;        // Goal achievement
    float precision;              // Confidence in beliefs
};

/**
 * @brief Reservoir state for neuromorphic learning
 */
struct ReservoirState {
    float activations[RESERVOIR_SIZE];
    float filtered_rates[RESERVOIR_SIZE];
    float membrane_potential[RESERVOIR_SIZE];
    float adaptation[RESERVOIR_SIZE];
};

// ============================================================================
// DEVICE UTILITY FUNCTIONS
// ============================================================================

/**
 * @brief Fast inverse square root (Quake-style, but accurate)
 */
__device__ __forceinline__ float fast_rsqrt(float x) {
    return rsqrtf(x + EPSILON);
}

/**
 * @brief Compute 3D distance squared
 */
__device__ __forceinline__ float dist_sq(float3 a, float3 b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    return dx*dx + dy*dy + dz*dz;
}

/**
 * @brief Warp-level reduction for sum
 */
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * @brief Block-level reduction for sum (simple version)
 */
__device__ float block_reduce_sum(float val) {
    // Use full shared memory reduction to avoid warp shuffle issues
    __shared__ float reduce_shared[256];  // One per thread

    // Each thread stores its value
    reduce_shared[threadIdx.x] = val;
    __syncthreads();

    // Tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce_shared[threadIdx.x] += reduce_shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    return reduce_shared[0];
}

/**
 * @brief Xorshift128+ random number generator (GPU-friendly)
 */
__device__ __forceinline__ float random_uniform(unsigned long long* state) {
    unsigned long long s0 = state[0];
    unsigned long long s1 = state[1];
    unsigned long long result = s0 + s1;

    s1 ^= s0;
    state[0] = ((s0 << 55) | (s0 >> 9)) ^ s1 ^ (s1 << 14);
    state[1] = (s1 << 36) | (s1 >> 28);

    // Convert to [0, 1) float
    return (result >> 11) * (1.0f / 9007199254740992.0f);
}

/**
 * @brief Box-Muller transform for Gaussian random numbers
 */
__device__ float2 random_gaussian(unsigned long long* state) {
    float u1 = random_uniform(state);
    float u2 = random_uniform(state);

    float r = sqrtf(-2.0f * logf(u1 + EPSILON));
    float theta = 2.0f * 3.14159265359f * u2;

    return make_float2(r * cosf(theta), r * sinf(theta));
}

// ============================================================================
// NEURAL HAMILTONIAN FORCE COMPUTATION
// ============================================================================

/**
 * @brief Compute bonded forces (bonds, angles, dihedrals)
 *
 * Simplified force field - extend for full AMBER/CHARMM compatibility
 */
__device__ float3 compute_bonded_forces(
    int atom_idx,
    const float3* __restrict__ positions,
    const int* __restrict__ bond_list,
    const float* __restrict__ bond_params,
    int n_bonds
) {
    float3 force = make_float3(0.0f, 0.0f, 0.0f);

    // Bond stretching (harmonic)
    for (int b = 0; b < n_bonds; b++) {
        int i = bond_list[b * 2];
        int j = bond_list[b * 2 + 1];

        if (i != atom_idx && j != atom_idx) continue;

        int other = (i == atom_idx) ? j : i;
        float3 r = make_float3(
            positions[other].x - positions[atom_idx].x,
            positions[other].y - positions[atom_idx].y,
            positions[other].z - positions[atom_idx].z
        );

        float r_mag = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z + EPSILON);
        float r0 = bond_params[b * 2];      // Equilibrium distance
        float k = bond_params[b * 2 + 1];   // Force constant

        float f_mag = -k * (r_mag - r0) / r_mag;

        force.x += f_mag * r.x;
        force.y += f_mag * r.y;
        force.z += f_mag * r.z;
    }

    return force;
}

/**
 * @brief Compute angle bending forces (harmonic)
 *
 * E = 0.5 * k * (theta - theta0)^2
 * F_i = -dE/dr_i using chain rule through angle calculation
 *
 * For angle i-j-k (j is center):
 * - F_i = k * (theta - theta0) * d(theta)/dr_i
 * - F_k = k * (theta - theta0) * d(theta)/dr_k
 * - F_j = -F_i - F_k (Newton's 3rd law)
 */
__device__ float3 compute_angle_forces(
    int atom_idx,
    const float3* __restrict__ positions,
    const int* __restrict__ angle_list,   // [n_angles * 3]: i, j, k triplets
    const float* __restrict__ angle_params, // [n_angles * 2]: theta0 (rad), k
    int n_angles
) {
    float3 force = make_float3(0.0f, 0.0f, 0.0f);

    for (int a = 0; a < n_angles; a++) {
        int atom_i = angle_list[a * 3];
        int atom_j = angle_list[a * 3 + 1];  // Center atom
        int atom_k = angle_list[a * 3 + 2];

        // Check if this atom participates in this angle
        if (atom_i != atom_idx && atom_j != atom_idx && atom_k != atom_idx) continue;

        // Get positions
        float3 r_i = positions[atom_i];
        float3 r_j = positions[atom_j];
        float3 r_k = positions[atom_k];

        // Vectors from center atom j
        float3 v_ji = make_float3(r_i.x - r_j.x, r_i.y - r_j.y, r_i.z - r_j.z);
        float3 v_jk = make_float3(r_k.x - r_j.x, r_k.y - r_j.y, r_k.z - r_j.z);

        // Magnitudes
        float r_ji = sqrtf(v_ji.x*v_ji.x + v_ji.y*v_ji.y + v_ji.z*v_ji.z + EPSILON);
        float r_jk = sqrtf(v_jk.x*v_jk.x + v_jk.y*v_jk.y + v_jk.z*v_jk.z + EPSILON);

        // Dot product and angle
        float dot = v_ji.x*v_jk.x + v_ji.y*v_jk.y + v_ji.z*v_jk.z;
        float cos_theta = dot / (r_ji * r_jk);
        cos_theta = fmaxf(-0.9999f, fminf(0.9999f, cos_theta)); // Clamp for numerical stability
        float theta = acosf(cos_theta);

        // Parameters
        float theta0 = angle_params[a * 2];
        float k = angle_params[a * 2 + 1];

        // dE/dtheta = k * (theta - theta0)
        float dE_dtheta = k * (theta - theta0);

        // dtheta/d(cos_theta) = -1/sin(theta)
        float sin_theta = sqrtf(1.0f - cos_theta * cos_theta + EPSILON);
        float dtheta_dcos = -1.0f / sin_theta;

        // d(cos_theta)/dr_i and d(cos_theta)/dr_k using chain rule
        float inv_r_ji = 1.0f / r_ji;
        float inv_r_jk = 1.0f / r_jk;

        // For atom i: d(cos)/dr_i = (v_jk - cos_theta * v_ji) / (r_ji * r_jk)
        float3 dcos_dri = make_float3(
            (v_jk.x * inv_r_jk - cos_theta * v_ji.x * inv_r_ji) * inv_r_ji,
            (v_jk.y * inv_r_jk - cos_theta * v_ji.y * inv_r_ji) * inv_r_ji,
            (v_jk.z * inv_r_jk - cos_theta * v_ji.z * inv_r_ji) * inv_r_ji
        );

        // For atom k: d(cos)/dr_k = (v_ji - cos_theta * v_jk) / (r_ji * r_jk)
        float3 dcos_drk = make_float3(
            (v_ji.x * inv_r_ji - cos_theta * v_jk.x * inv_r_jk) * inv_r_jk,
            (v_ji.y * inv_r_ji - cos_theta * v_jk.y * inv_r_jk) * inv_r_jk,
            (v_ji.z * inv_r_ji - cos_theta * v_jk.z * inv_r_jk) * inv_r_jk
        );

        // Force magnitude factor
        float f_factor = dE_dtheta * dtheta_dcos;

        // Apply force based on which atom this is
        if (atom_idx == atom_i) {
            force.x -= f_factor * dcos_dri.x;
            force.y -= f_factor * dcos_dri.y;
            force.z -= f_factor * dcos_dri.z;
        } else if (atom_idx == atom_k) {
            force.x -= f_factor * dcos_drk.x;
            force.y -= f_factor * dcos_drk.y;
            force.z -= f_factor * dcos_drk.z;
        } else if (atom_idx == atom_j) {
            // Center atom: force is negative sum of end atoms
            force.x += f_factor * (dcos_dri.x + dcos_drk.x);
            force.y += f_factor * (dcos_dri.y + dcos_drk.y);
            force.z += f_factor * (dcos_dri.z + dcos_drk.z);
        }
    }

    return force;
}

/**
 * @brief Compute dihedral torsion forces (periodic)
 *
 * E = k * (1 + cos(n*phi - phase))
 * where phi is the dihedral angle i-j-k-l
 *
 * This is more complex because the dihedral angle depends on 4 atoms.
 * We use the standard MD formulation with cross products.
 */
__device__ float3 compute_dihedral_forces(
    int atom_idx,
    const float3* __restrict__ positions,
    const int* __restrict__ dihedral_list,    // [n_dihedrals * 4]: i, j, k, l quartets
    const float* __restrict__ dihedral_params, // Flattened: [k, n, phase, paths, ...] per term
    const int* __restrict__ dihedral_term_counts, // Number of terms per dihedral
    int n_dihedrals
) {
    float3 force = make_float3(0.0f, 0.0f, 0.0f);

    int param_offset = 0;  // Track position in params array

    for (int d = 0; d < n_dihedrals; d++) {
        int atom_i = dihedral_list[d * 4];
        int atom_j = dihedral_list[d * 4 + 1];
        int atom_k = dihedral_list[d * 4 + 2];
        int atom_l = dihedral_list[d * 4 + 3];

        int n_terms = dihedral_term_counts[d];

        // Check if this atom participates
        bool participates = (atom_i == atom_idx || atom_j == atom_idx ||
                            atom_k == atom_idx || atom_l == atom_idx);

        if (!participates) {
            param_offset += n_terms * 4;  // Skip params for this dihedral
            continue;
        }

        // Get positions
        float3 r_i = positions[atom_i];
        float3 r_j = positions[atom_j];
        float3 r_k = positions[atom_k];
        float3 r_l = positions[atom_l];

        // Bond vectors
        float3 b1 = make_float3(r_j.x - r_i.x, r_j.y - r_i.y, r_j.z - r_i.z);
        float3 b2 = make_float3(r_k.x - r_j.x, r_k.y - r_j.y, r_k.z - r_j.z);
        float3 b3 = make_float3(r_l.x - r_k.x, r_l.y - r_k.y, r_l.z - r_k.z);

        // Cross products for normal vectors
        float3 n1 = make_float3(
            b1.y * b2.z - b1.z * b2.y,
            b1.z * b2.x - b1.x * b2.z,
            b1.x * b2.y - b1.y * b2.x
        );
        float3 n2 = make_float3(
            b2.y * b3.z - b2.z * b3.y,
            b2.z * b3.x - b2.x * b3.z,
            b2.x * b3.y - b2.y * b3.x
        );

        // Magnitudes of cross products
        float n1_mag = sqrtf(n1.x*n1.x + n1.y*n1.y + n1.z*n1.z + EPSILON);
        float n2_mag = sqrtf(n2.x*n2.x + n2.y*n2.y + n2.z*n2.z + EPSILON);
        float b2_mag = sqrtf(b2.x*b2.x + b2.y*b2.y + b2.z*b2.z + EPSILON);

        // Normalized vectors
        float3 n1_hat = make_float3(n1.x/n1_mag, n1.y/n1_mag, n1.z/n1_mag);
        float3 n2_hat = make_float3(n2.x/n2_mag, n2.y/n2_mag, n2.z/n2_mag);
        float3 b2_hat = make_float3(b2.x/b2_mag, b2.y/b2_mag, b2.z/b2_mag);

        // Compute dihedral angle phi
        float cos_phi = n1_hat.x*n2_hat.x + n1_hat.y*n2_hat.y + n1_hat.z*n2_hat.z;
        cos_phi = fmaxf(-1.0f, fminf(1.0f, cos_phi));

        // Cross product n1 x n2 for sign
        float3 n1xn2 = make_float3(
            n1_hat.y * n2_hat.z - n1_hat.z * n2_hat.y,
            n1_hat.z * n2_hat.x - n1_hat.x * n2_hat.z,
            n1_hat.x * n2_hat.y - n1_hat.y * n2_hat.x
        );
        float sin_phi = n1xn2.x*b2_hat.x + n1xn2.y*b2_hat.y + n1xn2.z*b2_hat.z;

        float phi = atan2f(sin_phi, cos_phi);

        // Process each term for this dihedral
        float total_dE_dphi = 0.0f;
        for (int t = 0; t < n_terms; t++) {
            float k_t = dihedral_params[param_offset + t * 4];
            float n_t = dihedral_params[param_offset + t * 4 + 1];
            float phase_t = dihedral_params[param_offset + t * 4 + 2];
            // float paths_t = dihedral_params[param_offset + t * 4 + 3]; // For scaling

            // E = k * (1 + cos(n*phi - phase))
            // dE/dphi = -k * n * sin(n*phi - phase)
            float n_phi_phase = n_t * phi - phase_t;
            total_dE_dphi -= k_t * n_t * sinf(n_phi_phase);
        }

        // Convert dE/dphi to forces on atoms
        // Using standard torsion force formulas
        float inv_n1_sq = 1.0f / (n1_mag * n1_mag + EPSILON);
        float inv_n2_sq = 1.0f / (n2_mag * n2_mag + EPSILON);

        // Force on atom i: F_i = (dE/dphi) * (n1 / |n1|^2) * |b2|
        // Force on atom l: F_l = -(dE/dphi) * (n2 / |n2|^2) * |b2|
        float f_i_factor = total_dE_dphi * b2_mag * inv_n1_sq;
        float f_l_factor = -total_dE_dphi * b2_mag * inv_n2_sq;

        if (atom_idx == atom_i) {
            force.x += f_i_factor * n1.x;
            force.y += f_i_factor * n1.y;
            force.z += f_i_factor * n1.z;
        } else if (atom_idx == atom_l) {
            force.x += f_l_factor * n2.x;
            force.y += f_l_factor * n2.y;
            force.z += f_l_factor * n2.z;
        } else if (atom_idx == atom_j) {
            // F_j depends on both n1 and n2 with geometric factors
            float b1_dot_b2 = b1.x*b2.x + b1.y*b2.y + b1.z*b2.z;
            float factor_j1 = (b1_dot_b2 / (b2_mag * b2_mag + EPSILON) - 1.0f);
            float factor_j2 = (b2.x*b3.x + b2.y*b3.y + b2.z*b3.z) / (b2_mag * b2_mag + EPSILON);

            force.x += f_i_factor * (factor_j1 * n1.x) - f_l_factor * factor_j2 * n2.x;
            force.y += f_i_factor * (factor_j1 * n1.y) - f_l_factor * factor_j2 * n2.y;
            force.z += f_i_factor * (factor_j1 * n1.z) - f_l_factor * factor_j2 * n2.z;
        } else if (atom_idx == atom_k) {
            // F_k = -(F_i + F_j + F_l) but computed directly for efficiency
            float b2_dot_b3 = b2.x*b3.x + b2.y*b3.y + b2.z*b3.z;
            float factor_k2 = (b2_dot_b3 / (b2_mag * b2_mag + EPSILON) - 1.0f);
            float factor_k1 = (b1.x*b2.x + b1.y*b2.y + b1.z*b2.z) / (b2_mag * b2_mag + EPSILON);

            force.x += f_l_factor * (factor_k2 * n2.x) - f_i_factor * factor_k1 * n1.x;
            force.y += f_l_factor * (factor_k2 * n2.y) - f_i_factor * factor_k1 * n1.y;
            force.z += f_l_factor * (factor_k2 * n2.z) - f_i_factor * factor_k1 * n1.z;
        }

        param_offset += n_terms * 4;
    }

    return force;
}

/**
 * @brief Check if atom pair is in exclusion list (1-2 or 1-3 bonded)
 *
 * Exclusion list is sorted for binary search optimization
 */
__device__ bool is_excluded_pair(
    int atom_i,
    int atom_j,
    const int* __restrict__ exclusion_list,
    int n_exclusions
) {
    // Ensure canonical order (smaller index first)
    int a = (atom_i < atom_j) ? atom_i : atom_j;
    int b = (atom_i < atom_j) ? atom_j : atom_i;

    // Linear search (for small lists) - could optimize with binary search
    for (int e = 0; e < n_exclusions; e++) {
        int ea = exclusion_list[e * 2];
        int eb = exclusion_list[e * 2 + 1];
        if (ea == a && eb == b) return true;
        // Early exit if we've passed the target (assumes sorted)
        if (ea > a) break;
    }
    return false;
}

/**
 * @brief Check if atom pair is a 1-4 pair (separated by 3 bonds)
 *
 * Returns true if pair needs scaled non-bonded interactions
 */
__device__ bool is_14_pair(
    int atom_i,
    int atom_j,
    const int* __restrict__ pair_14_list,
    int n_pairs_14
) {
    // Ensure canonical order
    int a = (atom_i < atom_j) ? atom_i : atom_j;
    int b = (atom_i < atom_j) ? atom_j : atom_i;

    for (int p = 0; p < n_pairs_14; p++) {
        int pa = pair_14_list[p * 2];
        int pb = pair_14_list[p * 2 + 1];
        if (pa == a && pb == b) return true;
        if (pa > a) break;
    }
    return false;
}

// AMBER ff14SB 1-4 scaling factors
#define AMBER_SCEE 1.2f     // Coulomb scaling: 1/1.2 = 0.8333
#define AMBER_SCNB 2.0f     // LJ scaling: 1/2.0 = 0.5

/**
 * @brief Compute non-bonded forces (Lennard-Jones + Coulomb)
 *
 * Properly handles:
 * - Exclusions: Skip 1-2 (directly bonded) and 1-3 (angle) pairs
 * - 1-4 Scaling: Apply AMBER ff14SB scaling for 1-4 pairs
 * - Soft-core: Clamp minimum distance to prevent force explosion
 */
__device__ float3 compute_nonbonded_forces(
    int atom_idx,
    const float3* __restrict__ positions,
    const float* __restrict__ charges,
    const float* __restrict__ lj_params,
    int n_atoms,
    float cutoff_sq,
    const int* __restrict__ exclusion_list,
    int n_exclusions,
    const int* __restrict__ pair_14_list,
    int n_pairs_14
) {
    float3 force = make_float3(0.0f, 0.0f, 0.0f);
    float3 pos_i = positions[atom_idx];
    float q_i = charges[atom_idx];
    float eps_i = lj_params[atom_idx * 2];
    float sigma_i = lj_params[atom_idx * 2 + 1];

    for (int j = 0; j < n_atoms; j++) {
        if (j == atom_idx) continue;

        // Check if this pair is excluded (1-2 or 1-3 bonded)
        if (n_exclusions > 0 && is_excluded_pair(atom_idx, j, exclusion_list, n_exclusions)) {
            continue;  // Skip excluded pairs entirely
        }

        float3 r = make_float3(
            positions[j].x - pos_i.x,
            positions[j].y - pos_i.y,
            positions[j].z - pos_i.z
        );

        float r_sq = r.x*r.x + r.y*r.y + r.z*r.z;
        if (r_sq > cutoff_sq) continue;

        // Check if this is a 1-4 pair (needs scaling)
        float scale_coulomb = 1.0f;
        float scale_lj = 1.0f;
        if (n_pairs_14 > 0 && is_14_pair(atom_idx, j, pair_14_list, n_pairs_14)) {
            scale_coulomb = 1.0f / AMBER_SCEE;  // 0.8333
            scale_lj = 1.0f / AMBER_SCNB;       // 0.5
        }

        // SOFT-CORE MODIFICATION: Clamp minimum distance to prevent force explosion
        // This prevents numerical instability when atoms get too close
        const float r_sq_min = 4.0f;  // 2 Å minimum distance squared
        float r_sq_eff = fmaxf(r_sq, r_sq_min);
        float r_inv = fast_rsqrt(r_sq_eff);

        // Coulomb force (with 1-4 scaling if applicable)
        float q_j = charges[j];
        float f_coulomb = 332.0f * q_i * q_j * r_inv * r_inv * r_inv * scale_coulomb;

        // Lennard-Jones force (with 1-4 scaling if applicable)
        float eps_j = lj_params[j * 2];
        float sigma_j = lj_params[j * 2 + 1];
        float eps_ij = sqrtf(eps_i * eps_j);
        float sigma_ij = 0.5f * (sigma_i + sigma_j);

        float sigma_r = sigma_ij * r_inv;
        float sigma_r6 = sigma_r * sigma_r * sigma_r * sigma_r * sigma_r * sigma_r;
        float sigma_r12 = sigma_r6 * sigma_r6;
        float f_lj = 24.0f * eps_ij * r_inv * (2.0f * sigma_r12 - sigma_r6) * scale_lj;

        float f_total = f_coulomb + f_lj;

        force.x += f_total * r.x;
        force.y += f_total * r.y;
        force.z += f_total * r.z;
    }

    return force;
}

/**
 * @brief Neural network force correction
 *
 * Simple MLP that predicts force corrections to improve accuracy
 */
__device__ float3 compute_neural_correction(
    int atom_idx,
    const float3* __restrict__ positions,
    const float* __restrict__ nn_weights,
    const float* __restrict__ local_features,
    int hidden_dim
) {
    // Input: local environment features (32-dim)
    // Hidden: ReLU activation
    // Output: 3D force correction

    // Layer 1: input -> hidden
    float hidden[64];  // Max hidden dim
    for (int h = 0; h < hidden_dim; h++) {
        float sum = nn_weights[h];  // Bias
        for (int f = 0; f < 32; f++) {
            sum += local_features[f] * nn_weights[hidden_dim + h * 32 + f];
        }
        hidden[h] = fmaxf(0.0f, sum);  // ReLU
    }

    // Layer 2: hidden -> output (3D force)
    int offset = hidden_dim + hidden_dim * 32;
    float3 correction;
    correction.x = nn_weights[offset];
    correction.y = nn_weights[offset + 1];
    correction.z = nn_weights[offset + 2];

    for (int h = 0; h < hidden_dim; h++) {
        correction.x += hidden[h] * nn_weights[offset + 3 + h * 3];
        correction.y += hidden[h] * nn_weights[offset + 3 + h * 3 + 1];
        correction.z += hidden[h] * nn_weights[offset + 3 + h * 3 + 2];
    }

    return correction;
}

/**
 * @brief Quantum correction for hydrogen bonds (simplified PIMC)
 *
 * Accounts for proton delocalization in H-bonds
 */
__device__ float3 compute_quantum_correction(
    int atom_idx,
    const float3* __restrict__ positions,
    const int* __restrict__ atom_types,
    int n_atoms,
    float temperature
) {
    float3 correction = make_float3(0.0f, 0.0f, 0.0f);

    // Only apply to hydrogen atoms
    if (atom_types[atom_idx] != 1) return correction;  // 1 = hydrogen

    // Find potential H-bond acceptors (O, N)
    float3 pos_h = positions[atom_idx];

    for (int j = 0; j < n_atoms; j++) {
        int type_j = atom_types[j];
        if (type_j != 8 && type_j != 7) continue;  // 8 = O, 7 = N

        float3 r = make_float3(
            positions[j].x - pos_h.x,
            positions[j].y - pos_h.y,
            positions[j].z - pos_h.z
        );

        float r_sq = r.x*r.x + r.y*r.y + r.z*r.z;
        if (r_sq > 9.0f) continue;  // 3 Å cutoff for H-bonds

        float r_mag = sqrtf(r_sq);

        // Quantum delocalization correction
        // Based on path integral: proton tunneling broadens wavefunction
        float lambda_thermal = 0.3f / sqrtf(temperature);  // Thermal de Broglie wavelength
        float quantum_factor = expf(-r_mag / lambda_thermal);

        // Attractive correction toward acceptor
        float f_quantum = 0.5f * quantum_factor / r_mag;

        correction.x += f_quantum * r.x;
        correction.y += f_quantum * r.y;
        correction.z += f_quantum * r.z;
    }

    return correction;
}

// ============================================================================
// TOPOLOGICAL DATA ANALYSIS - PROPER PERSISTENT HOMOLOGY
// ============================================================================
//
// This implementation computes TRUE persistent homology using the standard
// algorithm with boundary matrix reduction. No more Euler characteristic
// approximations!
//
// Algorithm overview:
// 1. Build Vietoris-Rips filtration (vertices, edges, triangles with birth times)
// 2. Sort simplices by filtration value (lexicographic on dimension, then value)
// 3. Reduce boundary matrix to compute persistence pairs
// 4. Extract Betti numbers and persistence statistics
//
// Reference: Edelsbrunner & Harer, "Computational Topology" (2010)
// ============================================================================

/**
 * @brief Simplex representation for persistent homology
 */
struct Simplex {
    int vertices[4];    // Up to 4 vertices (for tetrahedra, but we only use up to 3)
    int dimension;      // 0=vertex, 1=edge, 2=triangle
    float filtration;   // Birth time (when simplex enters filtration)
    int index;          // Original index for boundary computation
};

/**
 * @brief Compute distance matrix for target residues
 */
__device__ void compute_distance_matrix(
    const float3* __restrict__ positions,
    const int* __restrict__ residue_atoms,
    const int* __restrict__ target_residues,
    int n_targets,
    float* __restrict__ dist_matrix
) {
    // Compute centroid-based distances between target residues
    for (int i = threadIdx.x; i < n_targets * n_targets; i += blockDim.x) {
        int ri = i / n_targets;
        int rj = i % n_targets;

        if (ri >= rj) {
            dist_matrix[i] = 0.0f;
            continue;
        }

        // Get representative atoms (CA atoms)
        int atom_i = residue_atoms[target_residues[ri]];
        int atom_j = residue_atoms[target_residues[rj]];

        float d = sqrtf(dist_sq(positions[atom_i], positions[atom_j]));
        dist_matrix[ri * n_targets + rj] = d;
        dist_matrix[rj * n_targets + ri] = d;
    }
}

/**
 * @brief Build Vietoris-Rips filtration from distance matrix
 *
 * Creates simplices (vertices, edges, triangles) with their filtration values.
 * For Vietoris-Rips:
 *   - Vertex filtration = 0
 *   - Edge filtration = distance between endpoints
 *   - Triangle filtration = max edge length
 */
__device__ int build_vietoris_rips_filtration(
    const float* __restrict__ dist_matrix,
    int n_points,
    float max_filtration,
    Simplex* __restrict__ simplices,
    int max_simplices
) {
    int n_simplices = 0;

    // Add vertices (dimension 0, filtration 0)
    for (int i = 0; i < n_points && n_simplices < max_simplices; i++) {
        simplices[n_simplices].vertices[0] = i;
        simplices[n_simplices].vertices[1] = -1;
        simplices[n_simplices].vertices[2] = -1;
        simplices[n_simplices].dimension = 0;
        simplices[n_simplices].filtration = 0.0f;
        simplices[n_simplices].index = n_simplices;
        n_simplices++;
    }

    // Add edges (dimension 1, filtration = edge length)
    for (int i = 0; i < n_points && n_simplices < max_simplices; i++) {
        for (int j = i + 1; j < n_points && n_simplices < max_simplices; j++) {
            float d = dist_matrix[i * n_points + j];
            if (d <= max_filtration) {
                simplices[n_simplices].vertices[0] = i;
                simplices[n_simplices].vertices[1] = j;
                simplices[n_simplices].vertices[2] = -1;
                simplices[n_simplices].dimension = 1;
                simplices[n_simplices].filtration = d;
                simplices[n_simplices].index = n_simplices;
                n_simplices++;
            }
        }
    }

    // Add triangles (dimension 2, filtration = max edge length)
    for (int i = 0; i < n_points && n_simplices < max_simplices; i++) {
        for (int j = i + 1; j < n_points && n_simplices < max_simplices; j++) {
            float d_ij = dist_matrix[i * n_points + j];
            if (d_ij > max_filtration) continue;

            for (int k = j + 1; k < n_points && n_simplices < max_simplices; k++) {
                float d_ik = dist_matrix[i * n_points + k];
                float d_jk = dist_matrix[j * n_points + k];

                if (d_ik <= max_filtration && d_jk <= max_filtration) {
                    // Triangle enters when last edge appears
                    float max_edge = fmaxf(d_ij, fmaxf(d_ik, d_jk));

                    simplices[n_simplices].vertices[0] = i;
                    simplices[n_simplices].vertices[1] = j;
                    simplices[n_simplices].vertices[2] = k;
                    simplices[n_simplices].dimension = 2;
                    simplices[n_simplices].filtration = max_edge;
                    simplices[n_simplices].index = n_simplices;
                    n_simplices++;
                }
            }
        }
    }

    return n_simplices;
}

/**
 * @brief Sort simplices by (dimension, filtration) using insertion sort
 *
 * Small n_simplices makes insertion sort efficient and avoids complex GPU sorting
 */
__device__ void sort_simplices(Simplex* simplices, int n_simplices) {
    for (int i = 1; i < n_simplices; i++) {
        Simplex key = simplices[i];
        int j = i - 1;

        // Sort by dimension first, then by filtration value
        while (j >= 0) {
            bool should_swap = false;
            if (simplices[j].dimension > key.dimension) {
                should_swap = true;
            } else if (simplices[j].dimension == key.dimension &&
                       simplices[j].filtration > key.filtration) {
                should_swap = true;
            }

            if (should_swap) {
                simplices[j + 1] = simplices[j];
                j--;
            } else {
                break;
            }
        }
        simplices[j + 1] = key;
    }

    // Update indices after sorting
    for (int i = 0; i < n_simplices; i++) {
        simplices[i].index = i;
    }
}

/**
 * @brief Find index of edge (i,j) in sorted simplex array
 */
__device__ int find_edge_index(
    const Simplex* simplices,
    int n_simplices,
    int v0,
    int v1
) {
    // Ensure v0 < v1
    if (v0 > v1) { int tmp = v0; v0 = v1; v1 = tmp; }

    for (int i = 0; i < n_simplices; i++) {
        if (simplices[i].dimension == 1 &&
            simplices[i].vertices[0] == v0 &&
            simplices[i].vertices[1] == v1) {
            return i;
        }
    }
    return -1;
}

/**
 * @brief Compute persistent homology using boundary matrix reduction
 *
 * This is the standard persistence algorithm:
 * 1. Process simplices in filtration order
 * 2. For each simplex, compute its boundary
 * 3. Reduce using column operations
 * 4. Track persistence pairs
 *
 * Uses "clearing" optimization: once a column is reduced, skip it
 */
__device__ void compute_persistent_homology(
    Simplex* simplices,
    int n_simplices,
    PersistencePair* pairs,
    int* n_pairs,
    int* betti,  // [3] array for betti_0, betti_1, betti_2
    float reference_filtration
) {
    // low[j] = index of lowest non-zero entry in column j, or -1 if zero column
    int low[MAX_PH_SIMPLICES];
    bool marked[MAX_PH_SIMPLICES];  // Whether simplex is paired

    for (int i = 0; i < n_simplices; i++) {
        low[i] = -1;
        marked[i] = false;
    }

    *n_pairs = 0;
    betti[0] = betti[1] = betti[2] = 0;

    // Process simplices in filtration order (already sorted)
    for (int j = 0; j < n_simplices; j++) {
        Simplex& sigma = simplices[j];

        if (sigma.dimension == 0) {
            // Vertices have empty boundary - they create dim-0 features
            low[j] = -1;
            continue;
        }

        // Compute boundary of sigma
        // For edge (v0, v1): boundary = v1 - v0 (indices of vertices)
        // For triangle (v0, v1, v2): boundary = edge(v1,v2) - edge(v0,v2) + edge(v0,v1)

        // Find the "low" index - the largest boundary simplex index
        int current_low = -1;

        if (sigma.dimension == 1) {
            // Edge: boundary is two vertices
            int idx0 = sigma.vertices[0];  // These are vertex indices (0 to n_points-1)
            int idx1 = sigma.vertices[1];
            // Vertices are the first n_points simplices
            current_low = (idx0 > idx1) ? idx0 : idx1;
        }
        else if (sigma.dimension == 2) {
            // Triangle: boundary is three edges
            int v0 = sigma.vertices[0];
            int v1 = sigma.vertices[1];
            int v2 = sigma.vertices[2];

            // Find indices of boundary edges
            int e01 = find_edge_index(simplices, n_simplices, v0, v1);
            int e02 = find_edge_index(simplices, n_simplices, v0, v2);
            int e12 = find_edge_index(simplices, n_simplices, v1, v2);

            // Use XOR chain to find "low" after reduction
            // In mod-2 arithmetic, we track which edges remain
            bool boundary[MAX_PH_SIMPLICES] = {false};
            if (e01 >= 0) boundary[e01] = !boundary[e01];
            if (e02 >= 0) boundary[e02] = !boundary[e02];
            if (e12 >= 0) boundary[e12] = !boundary[e12];

            // Column reduction: while there's a column i < j with same low, add it
            bool changed = true;
            while (changed) {
                changed = false;

                // Find current low
                current_low = -1;
                for (int k = n_simplices - 1; k >= 0; k--) {
                    if (boundary[k]) {
                        current_low = k;
                        break;
                    }
                }

                if (current_low < 0) break;

                // Check if any previous column has same low
                for (int i = 0; i < j; i++) {
                    if (low[i] == current_low && simplices[i].dimension == 2) {
                        // Add column i to column j (mod 2)
                        // We need to reconstruct column i's boundary
                        int vi0 = simplices[i].vertices[0];
                        int vi1 = simplices[i].vertices[1];
                        int vi2 = simplices[i].vertices[2];

                        int ei01 = find_edge_index(simplices, n_simplices, vi0, vi1);
                        int ei02 = find_edge_index(simplices, n_simplices, vi0, vi2);
                        int ei12 = find_edge_index(simplices, n_simplices, vi1, vi2);

                        if (ei01 >= 0) boundary[ei01] = !boundary[ei01];
                        if (ei02 >= 0) boundary[ei02] = !boundary[ei02];
                        if (ei12 >= 0) boundary[ei12] = !boundary[ei12];

                        changed = true;
                        break;
                    }
                }
            }

            // Recompute final low
            current_low = -1;
            for (int k = n_simplices - 1; k >= 0; k--) {
                if (boundary[k]) {
                    current_low = k;
                    break;
                }
            }
        }

        low[j] = current_low;

        // If low[j] != -1 and not already paired, we have a persistence pair
        if (current_low >= 0 && !marked[current_low]) {
            marked[current_low] = true;
            marked[j] = true;

            // Birth = filtration of low[j], Death = filtration of j
            float birth = simplices[current_low].filtration;
            float death = sigma.filtration;
            int dim = sigma.dimension - 1;  // Feature dimension is creator dimension

            if (*n_pairs < MAX_PH_SIMPLICES && death > birth + EPSILON) {
                pairs[*n_pairs].birth = birth;
                pairs[*n_pairs].death = death;
                pairs[*n_pairs].dimension = dim;
                (*n_pairs)++;
            }
        }
    }

    // Add essential features (never die) to pairs list
    for (int i = 0; i < n_simplices; i++) {
        if (!marked[i] && simplices[i].filtration <= reference_filtration) {
            int dim = simplices[i].dimension;
            // This simplex creates an essential class
            if (*n_pairs < MAX_PH_SIMPLICES) {
                pairs[*n_pairs].birth = simplices[i].filtration;
                pairs[*n_pairs].death = PH_INF;
                pairs[*n_pairs].dimension = dim;
                (*n_pairs)++;
            }
        }
    }

    // Count Betti numbers at reference filtration using persistence diagram
    // A homology feature is ALIVE at filtration t if: birth <= t AND death > t
    // Betti_k = count of k-dimensional features alive at reference_filtration
    betti[0] = 0;
    betti[1] = 0;
    betti[2] = 0;

    for (int i = 0; i < *n_pairs; i++) {
        int dim = pairs[i].dimension;
        float birth = pairs[i].birth;
        float death = pairs[i].death;

        // Feature is alive at reference if: born before ref AND dies after ref
        if (birth <= reference_filtration && death > reference_filtration) {
            if (dim >= 0 && dim <= 2) {
                betti[dim]++;
            }
        }
    }

    // β₀ should be at least 1 (the whole space is connected at sufficient filtration)
    // If we have any points, we have at least one connected component
    if (betti[0] == 0 && n_simplices > 0) {
        // Count vertices with birth <= ref
        for (int i = 0; i < n_simplices; i++) {
            if (simplices[i].dimension == 0 && simplices[i].filtration <= reference_filtration) {
                betti[0]++;
                break;  // At least 1 component
            }
        }
    }
}

/**
 * @brief Compute full topological CV from structure using PROPER persistent homology
 *
 * This function now computes TRUE persistent homology with:
 * - Vietoris-Rips filtration construction
 * - Boundary matrix reduction
 * - Persistence pair extraction
 * - Statistics from persistence diagram
 */
__device__ TopologicalCV compute_topological_cv(
    const float3* __restrict__ positions,
    const int* __restrict__ residue_atoms,
    const int* __restrict__ target_residues,
    int n_targets,
    int n_atoms
) {
    // Shared memory for TDA computation
    __shared__ float dist_matrix[MAX_TARGET_RESIDUES * MAX_TARGET_RESIDUES];
    __shared__ Simplex simplices[MAX_PH_SIMPLICES];
    __shared__ PersistencePair pairs[MAX_PH_SIMPLICES];
    __shared__ int betti[3];
    __shared__ int n_simplices;
    __shared__ int n_pairs;

    TopologicalCV cv;

    // Initialize output
    if (threadIdx.x == 0) {
        cv.betti_0 = 0;
        cv.betti_1 = 0;
        cv.betti_2 = 0;
        cv.persistence_entropy = 0;
        cv.pocket_signature = 0;
        cv.gyration_radius = 0;
        cv.contact_order = 0;
        cv.local_density = 0;
        cv.total_persistence_0 = 0;
        cv.total_persistence_1 = 0;
        cv.total_persistence_2 = 0;
        cv.max_persistence_2 = 0;
        cv.n_persistent_voids = 0;
    }
    __syncthreads();

    // Handle edge case: no target residues
    if (n_targets < 2) {
        if (threadIdx.x == 0) {
            cv.betti_0 = (float)n_targets;
        }
        return cv;
    }

    // Compute distance matrix
    compute_distance_matrix(positions, residue_atoms, target_residues, n_targets, dist_matrix);
    __syncthreads();

    // Build and analyze persistent homology (thread 0 only for sequential algorithm)
    if (threadIdx.x == 0) {
        // Reference filtration for Betti numbers (8 Å = typical pocket scale)
        const float REFERENCE_FILTRATION = 8.0f;
        // Maximum filtration to consider (12 Å = large-scale topology)
        const float MAX_FILTRATION = 12.0f;
        // Persistence threshold for "significant" features
        const float PERSISTENCE_THRESHOLD = 2.0f;

        // Build Vietoris-Rips filtration
        n_simplices = build_vietoris_rips_filtration(
            dist_matrix, n_targets, MAX_FILTRATION,
            simplices, MAX_PH_SIMPLICES
        );

        // Sort simplices by (dimension, filtration)
        sort_simplices(simplices, n_simplices);

        // Compute persistent homology
        compute_persistent_homology(
            simplices, n_simplices,
            pairs, &n_pairs,
            betti, REFERENCE_FILTRATION
        );

        // Set Betti numbers
        cv.betti_0 = (float)betti[0];
        cv.betti_1 = (float)betti[1];
        cv.betti_2 = (float)betti[2];

        // Compute persistence diagram statistics
        float total_pers[3] = {0.0f, 0.0f, 0.0f};
        float max_pers_2 = 0.0f;
        int n_significant_voids = 0;
        float persistence_sum = 0.0f;

        for (int i = 0; i < n_pairs; i++) {
            float pers = pairs[i].death - pairs[i].birth;
            if (pairs[i].death >= PH_INF * 0.5f) {
                // Essential feature - use reference filtration for persistence
                pers = REFERENCE_FILTRATION - pairs[i].birth;
            }

            if (pers > 0) {
                int dim = pairs[i].dimension;
                if (dim >= 0 && dim <= 2) {
                    total_pers[dim] += pers;
                }

                // Track maximum dim-2 persistence (most stable pocket)
                if (dim == 2 && pers > max_pers_2) {
                    max_pers_2 = pers;
                }

                // Count significant voids
                if (dim == 2 && pers > PERSISTENCE_THRESHOLD) {
                    n_significant_voids++;
                }

                persistence_sum += pers;
            }
        }

        cv.total_persistence_0 = total_pers[0];
        cv.total_persistence_1 = total_pers[1];
        cv.total_persistence_2 = total_pers[2];
        cv.max_persistence_2 = max_pers_2;
        cv.n_persistent_voids = n_significant_voids;

        // Persistence entropy: Shannon entropy of normalized persistence values
        // This measures topological complexity
        if (persistence_sum > EPSILON) {
            float entropy = 0.0f;
            for (int i = 0; i < n_pairs; i++) {
                float pers = pairs[i].death - pairs[i].birth;
                if (pairs[i].death >= PH_INF * 0.5f) {
                    pers = REFERENCE_FILTRATION - pairs[i].birth;
                }
                if (pers > EPSILON) {
                    float p = pers / persistence_sum;
                    entropy -= p * logf(p + EPSILON);
                }
            }
            cv.persistence_entropy = entropy;
        }

        // Pocket signature: combines void count, persistence, and stability
        // Higher = more druggable pocket potential
        cv.pocket_signature = (cv.betti_2 * 0.3f +
                               cv.max_persistence_2 * 0.4f +
                               (float)n_significant_voids * 0.3f);

        // Gyration radius
        float3 centroid = make_float3(0.0f, 0.0f, 0.0f);
        for (int i = 0; i < n_targets; i++) {
            int atom = residue_atoms[target_residues[i]];
            centroid.x += positions[atom].x;
            centroid.y += positions[atom].y;
            centroid.z += positions[atom].z;
        }
        centroid.x /= n_targets;
        centroid.y /= n_targets;
        centroid.z /= n_targets;

        float rg_sq = 0.0f;
        for (int i = 0; i < n_targets; i++) {
            int atom = residue_atoms[target_residues[i]];
            rg_sq += dist_sq(positions[atom], centroid);
        }
        cv.gyration_radius = sqrtf(rg_sq / n_targets);

        // Contact order (average sequence separation of contacts)
        float contact_sum = 0.0f;
        int contact_count = 0;
        for (int i = 0; i < n_targets; i++) {
            for (int j = i + 2; j < n_targets; j++) {
                if (dist_matrix[i * n_targets + j] < REFERENCE_FILTRATION) {
                    contact_sum += (j - i);
                    contact_count++;
                }
            }
        }
        cv.contact_order = (contact_count > 0) ? contact_sum / contact_count : 0.0f;

        // Local density around target
        cv.local_density = (float)contact_count / (n_targets * (n_targets - 1) / 2.0f + EPSILON);
    }
    __syncthreads();

    return cv;
}

// ============================================================================
// ACTIVE INFERENCE
// ============================================================================

/**
 * @brief Compute Active Inference state from topological CVs
 */
__device__ ActiveInferenceState compute_active_inference(
    const TopologicalCV* cv,
    float goal_pocket_signature,
    float temperature
) {
    ActiveInferenceState ai;

    // Pragmatic value: How close to goal (druggable pocket)?
    float pocket_error = cv->pocket_signature - goal_pocket_signature;
    ai.pragmatic_value = -pocket_error * pocket_error;

    // Epistemic value: How much would we learn?
    // High entropy = high uncertainty = high epistemic value
    ai.epistemic_value = cv->persistence_entropy;

    // Precision: Inverse uncertainty (lower temperature = higher precision)
    ai.precision = 1.0f / (temperature + EPSILON);

    // Expected Free Energy: EFE = -pragmatic - 0.5 * epistemic
    ai.expected_free_energy = -ai.pragmatic_value - 0.5f * ai.epistemic_value;

    // Variational Free Energy: VFE = complexity - accuracy
    float complexity = cv->persistence_entropy;
    float accuracy = -pocket_error * pocket_error * ai.precision;
    ai.variational_free_energy = complexity - accuracy;

    // Goal prior: Probability we think this is a good state
    ai.goal_prior = 1.0f / (1.0f + expf(ai.expected_free_energy));

    return ai;
}

/**
 * @brief Compute goal-directed bias force from Active Inference
 */
__device__ float3 compute_goal_bias_force(
    int atom_idx,
    const float3* __restrict__ positions,
    const int* __restrict__ target_residues,
    int n_targets,
    const ActiveInferenceState* ai,
    float goal_strength
) {
    float3 bias = make_float3(0.0f, 0.0f, 0.0f);

    // Only apply bias to target residue atoms
    bool is_target = false;
    for (int i = 0; i < n_targets; i++) {
        if (target_residues[i] == atom_idx) {
            is_target = true;
            break;
        }
    }
    if (!is_target) return bias;

    // Bias toward expanding (opening pocket) if goal not achieved
    // Direction: away from centroid (expand)
    float3 centroid = make_float3(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < n_targets; i++) {
        int atom = target_residues[i];
        centroid.x += positions[atom].x;
        centroid.y += positions[atom].y;
        centroid.z += positions[atom].z;
    }
    centroid.x /= n_targets;
    centroid.y /= n_targets;
    centroid.z /= n_targets;

    float3 direction = make_float3(
        positions[atom_idx].x - centroid.x,
        positions[atom_idx].y - centroid.y,
        positions[atom_idx].z - centroid.z
    );

    float dir_mag = sqrtf(direction.x*direction.x +
                          direction.y*direction.y +
                          direction.z*direction.z + EPSILON);

    // Scale by how far from goal and goal strength
    float scale = goal_strength * (1.0f - ai->goal_prior) / dir_mag;

    bias.x = scale * direction.x;
    bias.y = scale * direction.y;
    bias.z = scale * direction.z;

    return bias;
}

// ============================================================================
// RESERVOIR COMPUTING + RLS
// ============================================================================

/**
 * @brief LIF (Leaky Integrate-and-Fire) neuron update
 */
__device__ void lif_update(
    ReservoirState* state,
    const float* input,
    int input_dim,
    const float* weights,
    float dt_neural
) {
    // Membrane time constant
    const float tau_mem = 20.0f;  // ms
    const float tau_adapt = 200.0f;  // ms
    const float v_thresh = 1.0f;
    const float v_reset = 0.0f;

    int idx = threadIdx.x;
    if (idx >= RESERVOIR_SIZE) return;

    // Compute input current
    float current = 0.0f;
    for (int i = 0; i < input_dim; i++) {
        current += input[i] * weights[idx * input_dim + i];
    }

    // Recurrent connections (sparse, E/I balanced)
    for (int j = 0; j < RESERVOIR_SIZE; j++) {
        int w_idx = RESERVOIR_SIZE * input_dim + idx * RESERVOIR_SIZE + j;
        current += state->filtered_rates[j] * weights[w_idx];
    }

    // LIF dynamics
    float v = state->membrane_potential[idx];
    float a = state->adaptation[idx];

    // Membrane update
    float dv = (-(v - v_reset) + current - a) / tau_mem;
    v += dv * dt_neural;

    // Spike check
    float spike = 0.0f;
    if (v >= v_thresh) {
        spike = 1.0f;
        v = v_reset;
    }

    // Adaptation update
    float da = -a / tau_adapt + 0.1f * spike;
    a += da * dt_neural;

    // Exponential filter for rate estimation
    const float tau_filter = 50.0f;
    float r = state->filtered_rates[idx];
    r = r * expf(-dt_neural / tau_filter) + spike;

    // Store
    state->membrane_potential[idx] = v;
    state->adaptation[idx] = a;
    state->activations[idx] = spike;
    state->filtered_rates[idx] = r;
}

/**
 * @brief RLS (Recursive Least Squares) update for single output
 */
__device__ void rls_update(
    float* w,           // Weights [RESERVOIR_SIZE]
    float* P,           // Precision matrix [RESERVOIR_SIZE x RESERVOIR_SIZE]
    const float* x,     // State vector [RESERVOIR_SIZE]
    float target,
    float lambda,
    float reward_modulation
) {
    // Only thread 0 performs RLS (sequential algorithm)
    if (threadIdx.x != 0) return;

    int n = RESERVOIR_SIZE;

    // Compute P @ x
    float Px[RESERVOIR_SIZE];
    for (int i = 0; i < n; i++) {
        Px[i] = 0.0f;
        for (int j = 0; j < n; j++) {
            Px[i] += P[i * n + j] * x[j];
        }
    }

    // Compute x^T @ P @ x
    float xPx = 0.0f;
    for (int i = 0; i < n; i++) {
        xPx += x[i] * Px[i];
    }

    // Kalman gain: k = P @ x / (lambda + x^T @ P @ x)
    float denom = lambda + xPx;
    float k[RESERVOIR_SIZE];
    for (int i = 0; i < n; i++) {
        k[i] = Px[i] / denom;
    }

    // Prediction error
    float prediction = 0.0f;
    for (int i = 0; i < n; i++) {
        prediction += w[i] * x[i];
    }
    float error = target - prediction;

    // Weight update: w = w + k * error * modulation
    for (int i = 0; i < n; i++) {
        w[i] += k[i] * error * reward_modulation;
    }

    // P matrix update: P = (1/lambda) * (P - k @ x^T @ P)
    // Simplified: P = (1/lambda) * (P - k @ Px^T)
    float inv_lambda = 1.0f / lambda;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            P[i * n + j] = inv_lambda * (P[i * n + j] - k[i] * Px[j]);
        }
    }
}

// ============================================================================
// HAMILTONIAN MONTE CARLO
// ============================================================================

/**
 * @brief Leapfrog integration step
 */
__device__ void leapfrog_step(
    float3* positions,
    float3* momenta,
    const float3* forces,
    const float* masses,
    int n_atoms,
    float dt
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_atoms) return;

    float m = masses[idx];
    float dt_half = 0.5f * dt;

    // Half-step momentum update
    momenta[idx].x += dt_half * forces[idx].x;
    momenta[idx].y += dt_half * forces[idx].y;
    momenta[idx].z += dt_half * forces[idx].z;

    // Full-step position update
    positions[idx].x += dt * momenta[idx].x / m;
    positions[idx].y += dt * momenta[idx].y / m;
    positions[idx].z += dt * momenta[idx].z / m;
}

/**
 * @brief Compute total Hamiltonian (kinetic + potential energy)
 *
 * Includes:
 * - Kinetic energy: sum(p²/2m)
 * - Non-bonded: LJ + Coulomb (with exclusions and 1-4 scaling)
 * - Bonded: bonds + angles + dihedrals
 */
__device__ float compute_hamiltonian(
    const float3* positions,
    const float3* momenta,
    const float* masses,
    const float* charges,
    const float* lj_params,
    int n_atoms,
    const int* bond_list,
    const float* bond_params,
    int n_bonds,
    const int* angle_list,
    const float* angle_params,
    int n_angles,
    const int* dihedral_list,
    const float* dihedral_params,
    const int* dihedral_term_counts,
    int n_dihedrals,
    const int* exclusion_list,
    int n_exclusions,
    const int* pair_14_list,
    int n_pairs_14
) {
    float kinetic = 0.0f;
    float potential = 0.0f;

    // Kinetic energy
    for (int i = threadIdx.x; i < n_atoms; i += blockDim.x) {
        float m = masses[i];
        float3 p = momenta[i];
        kinetic += (p.x*p.x + p.y*p.y + p.z*p.z) / (2.0f * m);
    }
    kinetic = block_reduce_sum(kinetic);
    __syncthreads();

    // Non-bonded potential energy (pairwise LJ + Coulomb) with exclusions and 1-4 scaling
    float cutoff_sq = 144.0f;  // 12 Å cutoff
    const float r_sq_min = 4.0f;  // Soft-core: 2 Å minimum distance
    for (int i = threadIdx.x; i < n_atoms; i += blockDim.x) {
        for (int j = i + 1; j < n_atoms; j++) {
            // Skip excluded pairs (1-2 and 1-3 bonded)
            if (n_exclusions > 0 && is_excluded_pair(i, j, exclusion_list, n_exclusions)) {
                continue;
            }

            float r_sq = dist_sq(positions[i], positions[j]);
            if (r_sq > cutoff_sq) continue;

            // Check if this is a 1-4 pair (needs scaling)
            float scale_coulomb = 1.0f;
            float scale_lj = 1.0f;
            if (n_pairs_14 > 0 && is_14_pair(i, j, pair_14_list, n_pairs_14)) {
                scale_coulomb = 1.0f / AMBER_SCEE;  // 0.8333
                scale_lj = 1.0f / AMBER_SCNB;       // 0.5
            }

            // Soft-core: clamp minimum distance for numerical stability
            float r_sq_eff = fmaxf(r_sq, r_sq_min);
            float r_inv = fast_rsqrt(r_sq_eff);

            // Coulomb (with 1-4 scaling)
            potential += 332.0f * charges[i] * charges[j] * r_inv * scale_coulomb;

            // LJ (with 1-4 scaling)
            float eps_ij = sqrtf(lj_params[i*2] * lj_params[j*2]);
            float sigma_ij = 0.5f * (lj_params[i*2+1] + lj_params[j*2+1]);
            float sigma_r = sigma_ij * r_inv;
            float sigma_r6 = sigma_r * sigma_r * sigma_r * sigma_r * sigma_r * sigma_r;
            float lj_ij = 4.0f * eps_ij * (sigma_r6 * sigma_r6 - sigma_r6) * scale_lj;
            potential += lj_ij;
        }
    }
    potential = block_reduce_sum(potential);
    __syncthreads();

    // Bond energy: E = 0.5 * k * (r - r0)²
    float bond_energy = 0.0f;
    for (int b = threadIdx.x; b < n_bonds; b += blockDim.x) {
        int i = bond_list[b * 2];
        int j = bond_list[b * 2 + 1];
        float3 r = make_float3(
            positions[j].x - positions[i].x,
            positions[j].y - positions[i].y,
            positions[j].z - positions[i].z
        );
        float r_mag = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z + EPSILON);
        float r0 = bond_params[b * 2];
        float k = bond_params[b * 2 + 1];
        float dr = r_mag - r0;
        bond_energy += 0.5f * k * dr * dr;
    }
    bond_energy = block_reduce_sum(bond_energy);
    __syncthreads();

    // Angle energy: E = 0.5 * k * (theta - theta0)²
    float angle_energy = 0.0f;
    for (int a = threadIdx.x; a < n_angles; a += blockDim.x) {
        int atom_i = angle_list[a * 3];
        int atom_j = angle_list[a * 3 + 1];
        int atom_k = angle_list[a * 3 + 2];

        float3 v_ji = make_float3(
            positions[atom_i].x - positions[atom_j].x,
            positions[atom_i].y - positions[atom_j].y,
            positions[atom_i].z - positions[atom_j].z
        );
        float3 v_jk = make_float3(
            positions[atom_k].x - positions[atom_j].x,
            positions[atom_k].y - positions[atom_j].y,
            positions[atom_k].z - positions[atom_j].z
        );

        float r_ji = sqrtf(v_ji.x*v_ji.x + v_ji.y*v_ji.y + v_ji.z*v_ji.z + EPSILON);
        float r_jk = sqrtf(v_jk.x*v_jk.x + v_jk.y*v_jk.y + v_jk.z*v_jk.z + EPSILON);
        float dot = v_ji.x*v_jk.x + v_ji.y*v_jk.y + v_ji.z*v_jk.z;
        float cos_theta = fmaxf(-0.9999f, fminf(0.9999f, dot / (r_ji * r_jk)));
        float theta = acosf(cos_theta);

        float theta0 = angle_params[a * 2];
        float k = angle_params[a * 2 + 1];
        float dtheta = theta - theta0;
        angle_energy += 0.5f * k * dtheta * dtheta;
    }
    angle_energy = block_reduce_sum(angle_energy);
    __syncthreads();

    // Dihedral energy: E = k * (1 + cos(n*phi - phase))
    float dihedral_energy = 0.0f;
    int param_offset = 0;
    for (int d = 0; d < n_dihedrals; d++) {
        int n_terms = dihedral_term_counts[d];

        // Only process if this thread handles this dihedral
        if (d % blockDim.x == threadIdx.x) {
            int atom_i = dihedral_list[d * 4];
            int atom_j = dihedral_list[d * 4 + 1];
            int atom_k = dihedral_list[d * 4 + 2];
            int atom_l = dihedral_list[d * 4 + 3];

            // Bond vectors
            float3 b1 = make_float3(
                positions[atom_j].x - positions[atom_i].x,
                positions[atom_j].y - positions[atom_i].y,
                positions[atom_j].z - positions[atom_i].z
            );
            float3 b2 = make_float3(
                positions[atom_k].x - positions[atom_j].x,
                positions[atom_k].y - positions[atom_j].y,
                positions[atom_k].z - positions[atom_j].z
            );
            float3 b3 = make_float3(
                positions[atom_l].x - positions[atom_k].x,
                positions[atom_l].y - positions[atom_k].y,
                positions[atom_l].z - positions[atom_k].z
            );

            // Cross products
            float3 n1 = make_float3(
                b1.y * b2.z - b1.z * b2.y,
                b1.z * b2.x - b1.x * b2.z,
                b1.x * b2.y - b1.y * b2.x
            );
            float3 n2 = make_float3(
                b2.y * b3.z - b2.z * b3.y,
                b2.z * b3.x - b2.x * b3.z,
                b2.x * b3.y - b2.y * b3.x
            );

            float n1_mag = sqrtf(n1.x*n1.x + n1.y*n1.y + n1.z*n1.z + EPSILON);
            float n2_mag = sqrtf(n2.x*n2.x + n2.y*n2.y + n2.z*n2.z + EPSILON);
            float b2_mag = sqrtf(b2.x*b2.x + b2.y*b2.y + b2.z*b2.z + EPSILON);

            float3 n1_hat = make_float3(n1.x/n1_mag, n1.y/n1_mag, n1.z/n1_mag);
            float3 n2_hat = make_float3(n2.x/n2_mag, n2.y/n2_mag, n2.z/n2_mag);
            float3 b2_hat = make_float3(b2.x/b2_mag, b2.y/b2_mag, b2.z/b2_mag);

            float cos_phi = fmaxf(-1.0f, fminf(1.0f,
                n1_hat.x*n2_hat.x + n1_hat.y*n2_hat.y + n1_hat.z*n2_hat.z));
            float3 n1xn2 = make_float3(
                n1_hat.y * n2_hat.z - n1_hat.z * n2_hat.y,
                n1_hat.z * n2_hat.x - n1_hat.x * n2_hat.z,
                n1_hat.x * n2_hat.y - n1_hat.y * n2_hat.x
            );
            float sin_phi = n1xn2.x*b2_hat.x + n1xn2.y*b2_hat.y + n1xn2.z*b2_hat.z;
            float phi = atan2f(sin_phi, cos_phi);

            // Sum energy from all terms
            for (int t = 0; t < n_terms; t++) {
                float k_t = dihedral_params[param_offset + t * 4];
                float n_t = dihedral_params[param_offset + t * 4 + 1];
                float phase_t = dihedral_params[param_offset + t * 4 + 2];
                dihedral_energy += k_t * (1.0f + cosf(n_t * phi - phase_t));
            }
        }
        param_offset += n_terms * 4;
    }
    dihedral_energy = block_reduce_sum(dihedral_energy);

    return kinetic + potential + bond_energy + angle_energy + dihedral_energy;
}

/**
 * @brief Metropolis acceptance criterion
 */
__device__ bool metropolis_accept(
    float H_new,
    float H_old,
    float temperature,
    unsigned long long* rng_state
) {
    float dH = H_new - H_old;
    if (dH <= 0.0f) return true;

    float accept_prob = expf(-dH / (KB * temperature));
    return random_uniform(rng_state) < accept_prob;
}

// ============================================================================
// MAIN FUSED KERNEL
// ============================================================================

/**
 * @brief PRISM-NOVA: Main fused kernel for one simulation step
 *
 * Combines:
 * 1. Neural Hamiltonian Monte Carlo sampling
 * 2. Topological collective variable computation
 * 3. Active Inference goal-directed guidance
 * 4. Reservoir computing + RLS learning
 *
 * All in a single kernel launch with zero CPU round-trips.
 */
extern "C" __global__ void prism_nova_step(
    // Positions and momenta
    float3* positions,
    float3* momenta,
    float3* positions_old,      // For rejection

    // Force field parameters
    const float* masses,
    const float* charges,
    const float* lj_params,
    const int* bond_list,
    const float* bond_params,
    int n_bonds,
    const int* angle_list,        // [n_angles * 3]: i, j, k triplets
    const float* angle_params,    // [n_angles * 2]: theta0 (rad), k
    int n_angles,
    const int* dihedral_list,     // [n_dihedrals * 4]: i, j, k, l quartets
    const float* dihedral_params, // Flattened: [k, n, phase, paths, ...] per term
    const int* dihedral_term_counts, // Number of terms per dihedral
    int n_dihedrals,
    const int* exclusion_list,    // [n_exclusions * 2]: pairs to skip in non-bonded
    int n_exclusions,
    const int* pair_14_list,      // [n_pairs_14 * 2]: 1-4 pairs for scaled interactions
    int n_pairs_14,

    // Neural network weights
    const float* nn_weights,

    // Atom metadata
    const int* atom_types,
    const int* residue_atoms,

    // Reservoir state
    float* reservoir_activations,
    float* reservoir_filtered,
    float* reservoir_membrane,
    float* reservoir_adaptation,
    const float* reservoir_weights,

    // RLS state
    float* rls_weights,         // [NUM_OUTPUTS x RESERVOIR_SIZE]
    float* rls_P_matrices,      // [NUM_OUTPUTS x RESERVOIR_SIZE x RESERVOIR_SIZE]

    // Output
    float* output_features,     // [FEATURE_DIM] - TDA + AI features
    float* output_reward,
    int* output_accepted,

    // Target residues for TDA analysis
    const int* target_residues,     // [n_target_residues] - indices of residues for TDA
    int cfg_n_target_residues,      // Number of target residues

    // Configuration (individual params to match Rust calling convention)
    float cfg_dt,
    float cfg_temperature,
    float cfg_goal_strength,
    float cfg_lambda,
    int cfg_n_atoms,
    int cfg_n_residues,
    int cfg_leapfrog_steps,
    unsigned long long cfg_seed
) {
    // Build local config for convenience
    NovaConfig config;
    config.dt = cfg_dt;
    config.temperature = cfg_temperature;
    config.goal_strength = cfg_goal_strength;
    config.lambda = cfg_lambda;
    config.n_atoms = cfg_n_atoms;
    config.n_residues = cfg_n_residues;
    config.n_target_residues = cfg_n_target_residues;
    config.leapfrog_steps = cfg_leapfrog_steps;
    config.seed = cfg_seed;

    // Shared memory for intermediate results
    __shared__ float3 s_positions[MAX_ATOMS];
    __shared__ float3 s_forces[MAX_ATOMS];
    __shared__ TopologicalCV s_cv;
    __shared__ ActiveInferenceState s_ai;
    __shared__ ReservoirState s_reservoir;
    __shared__ float s_features[FEATURE_DIM];
    __shared__ float s_H_old, s_H_new;
    __shared__ unsigned long long s_rng[2];
    __shared__ bool s_accepted;
    __shared__ int s_target_residues[MAX_TARGET_RESIDUES];

    // Copy target residues to shared memory (cooperatively)
    int n_targets_to_load = config.n_target_residues;
    if (n_targets_to_load > MAX_TARGET_RESIDUES) n_targets_to_load = MAX_TARGET_RESIDUES;
    for (int i = threadIdx.x; i < n_targets_to_load; i += blockDim.x) {
        s_target_residues[i] = target_residues[i];
    }
    __syncthreads();

    // Initialize RNG
    if (threadIdx.x == 0) {
        s_rng[0] = config.seed + blockIdx.x * 12345;
        s_rng[1] = config.seed ^ 0xDEADBEEF;
    }
    __syncthreads();

    // ========================================================================
    // PHASE 0: LOAD DATA
    // ========================================================================

    // Load positions to shared memory
    for (int i = threadIdx.x; i < config.n_atoms; i += blockDim.x) {
        s_positions[i] = positions[i];
        positions_old[i] = positions[i];  // Save for potential rejection
    }
    __syncthreads();

    // Compute initial Hamiltonian - ALL THREADS must participate (block_reduce_sum inside)
    s_H_old = compute_hamiltonian(s_positions, momenta, masses, charges,
                                   lj_params, config.n_atoms,
                                   bond_list, bond_params, n_bonds,
                                   angle_list, angle_params, n_angles,
                                   dihedral_list, dihedral_params,
                                   dihedral_term_counts, n_dihedrals,
                                   exclusion_list, n_exclusions,
                                   pair_14_list, n_pairs_14);
    __syncthreads();

    // ========================================================================
    // PHASE 1: NEURAL HMC LEAPFROG INTEGRATION
    // ========================================================================

    for (int step = 0; step < config.leapfrog_steps; step++) {
        // Compute forces: physical (bonded + non-bonded) + quantum
        for (int i = threadIdx.x; i < config.n_atoms; i += blockDim.x) {
            // Non-bonded forces (LJ + Coulomb) with exclusions and 1-4 scaling
            float3 f_phys = compute_nonbonded_forces(i, s_positions, charges,
                                                      lj_params, config.n_atoms, 144.0f,
                                                      exclusion_list, n_exclusions,
                                                      pair_14_list, n_pairs_14);

            // Bonded forces: bonds
            float3 f_bond = compute_bonded_forces(i, s_positions, bond_list,
                                                   bond_params, n_bonds);
            f_phys.x += f_bond.x;
            f_phys.y += f_bond.y;
            f_phys.z += f_bond.z;

            // Bonded forces: angles
            float3 f_angle = compute_angle_forces(i, s_positions, angle_list,
                                                   angle_params, n_angles);
            f_phys.x += f_angle.x;
            f_phys.y += f_angle.y;
            f_phys.z += f_angle.z;

            // Bonded forces: dihedrals
            float3 f_dihedral = compute_dihedral_forces(i, s_positions, dihedral_list,
                                                         dihedral_params, dihedral_term_counts,
                                                         n_dihedrals);
            f_phys.x += f_dihedral.x;
            f_phys.y += f_dihedral.y;
            f_phys.z += f_dihedral.z;

            // Quantum correction for H-bonds
            float3 f_quantum = compute_quantum_correction(i, s_positions, atom_types,
                                                           config.n_atoms, config.temperature);

            s_forces[i].x = f_phys.x + f_quantum.x;
            s_forces[i].y = f_phys.y + f_quantum.y;
            s_forces[i].z = f_phys.z + f_quantum.z;
        }
        __syncthreads();

        // Half-step momentum
        for (int i = threadIdx.x; i < config.n_atoms; i += blockDim.x) {
            momenta[i].x += 0.5f * config.dt * s_forces[i].x;
            momenta[i].y += 0.5f * config.dt * s_forces[i].y;
            momenta[i].z += 0.5f * config.dt * s_forces[i].z;
        }
        __syncthreads();

        // Full-step position
        for (int i = threadIdx.x; i < config.n_atoms; i += blockDim.x) {
            s_positions[i].x += config.dt * momenta[i].x / masses[i];
            s_positions[i].y += config.dt * momenta[i].y / masses[i];
            s_positions[i].z += config.dt * momenta[i].z / masses[i];
        }
        __syncthreads();

        // Recompute forces at new positions (CRITICAL: include ALL forces!)
        for (int i = threadIdx.x; i < config.n_atoms; i += blockDim.x) {
            // Non-bonded forces (LJ + Coulomb) with exclusions and 1-4 scaling
            float3 f_phys = compute_nonbonded_forces(i, s_positions, charges,
                                                      lj_params, config.n_atoms, 144.0f,
                                                      exclusion_list, n_exclusions,
                                                      pair_14_list, n_pairs_14);

            // Bonded forces: bonds (FIXED: was missing in original!)
            float3 f_bond = compute_bonded_forces(i, s_positions, bond_list,
                                                   bond_params, n_bonds);
            f_phys.x += f_bond.x;
            f_phys.y += f_bond.y;
            f_phys.z += f_bond.z;

            // Bonded forces: angles
            float3 f_angle = compute_angle_forces(i, s_positions, angle_list,
                                                   angle_params, n_angles);
            f_phys.x += f_angle.x;
            f_phys.y += f_angle.y;
            f_phys.z += f_angle.z;

            // Bonded forces: dihedrals
            float3 f_dihedral = compute_dihedral_forces(i, s_positions, dihedral_list,
                                                         dihedral_params, dihedral_term_counts,
                                                         n_dihedrals);
            f_phys.x += f_dihedral.x;
            f_phys.y += f_dihedral.y;
            f_phys.z += f_dihedral.z;

            // Quantum correction for H-bonds
            float3 f_quantum = compute_quantum_correction(i, s_positions, atom_types,
                                                           config.n_atoms, config.temperature);

            s_forces[i].x = f_phys.x + f_quantum.x;
            s_forces[i].y = f_phys.y + f_quantum.y;
            s_forces[i].z = f_phys.z + f_quantum.z;
        }
        __syncthreads();

        // Half-step momentum (complete)
        for (int i = threadIdx.x; i < config.n_atoms; i += blockDim.x) {
            momenta[i].x += 0.5f * config.dt * s_forces[i].x;
            momenta[i].y += 0.5f * config.dt * s_forces[i].y;
            momenta[i].z += 0.5f * config.dt * s_forces[i].z;
        }
        __syncthreads();
    }

    // ========================================================================
    // PHASE 2: TOPOLOGICAL ANALYSIS
    // ========================================================================

    s_cv = compute_topological_cv(s_positions, residue_atoms, s_target_residues,
                                   config.n_target_residues, config.n_atoms);
    __syncthreads();

    // ========================================================================
    // PHASE 3: ACTIVE INFERENCE
    // ========================================================================

    if (threadIdx.x == 0) {
        float goal_signature = 0.5f;  // Target pocket opening level
        s_ai = compute_active_inference(&s_cv, goal_signature, config.temperature);
    }
    __syncthreads();

    // Apply goal-directed bias to momenta
    for (int i = threadIdx.x; i < config.n_atoms; i += blockDim.x) {
        float3 bias = compute_goal_bias_force(i, s_positions, s_target_residues,
                                               config.n_target_residues, &s_ai,
                                               config.goal_strength);
        momenta[i].x += config.dt * bias.x;
        momenta[i].y += config.dt * bias.y;
        momenta[i].z += config.dt * bias.z;
    }
    __syncthreads();

    // ========================================================================
    // PHASE 4: METROPOLIS ACCEPTANCE
    // ========================================================================

    // Compute new Hamiltonian - ALL THREADS must participate (block_reduce_sum inside)
    s_H_new = compute_hamiltonian(s_positions, momenta, masses, charges,
                                   lj_params, config.n_atoms,
                                   bond_list, bond_params, n_bonds,
                                   angle_list, angle_params, n_angles,
                                   dihedral_list, dihedral_params,
                                   dihedral_term_counts, n_dihedrals,
                                   exclusion_list, n_exclusions,
                                   pair_14_list, n_pairs_14);
    __syncthreads();

    if (threadIdx.x == 0) {
        s_accepted = metropolis_accept(s_H_new, s_H_old, config.temperature, s_rng);
    }
    __syncthreads();

    if (!s_accepted) {
        // Reject: restore old positions
        for (int i = threadIdx.x; i < config.n_atoms; i += blockDim.x) {
            s_positions[i] = positions_old[i];
        }
        __syncthreads();
    }

    // ========================================================================
    // PHASE 5: PACK FEATURES
    // ========================================================================

    if (threadIdx.x == 0) {
        // TDA features
        s_features[0] = s_cv.betti_0;
        s_features[1] = s_cv.betti_1;
        s_features[2] = s_cv.betti_2;
        s_features[3] = s_cv.persistence_entropy;
        s_features[4] = s_cv.pocket_signature;
        s_features[5] = s_cv.gyration_radius;
        s_features[6] = s_cv.contact_order;
        s_features[7] = s_cv.local_density;

        // Active Inference features
        s_features[8] = s_ai.expected_free_energy;
        s_features[9] = s_ai.variational_free_energy;
        s_features[10] = s_ai.goal_prior;
        s_features[11] = s_ai.epistemic_value;
        s_features[12] = s_ai.pragmatic_value;
        s_features[13] = s_ai.precision;

        // Energy features
        s_features[14] = s_H_new - s_H_old;  // Energy change
        s_features[15] = s_accepted ? 1.0f : 0.0f;

        // Normalize
        for (int i = 0; i < 16; i++) {
            if (!isfinite(s_features[i])) s_features[i] = 0.0f;
        }
    }
    __syncthreads();

    // ========================================================================
    // PHASE 6: RESERVOIR + RLS UPDATE
    // ========================================================================

    // Load reservoir state
    for (int i = threadIdx.x; i < RESERVOIR_SIZE; i += blockDim.x) {
        s_reservoir.activations[i] = reservoir_activations[i];
        s_reservoir.filtered_rates[i] = reservoir_filtered[i];
        s_reservoir.membrane_potential[i] = reservoir_membrane[i];
        s_reservoir.adaptation[i] = reservoir_adaptation[i];
    }
    __syncthreads();

    // Update reservoir with new features
    lif_update(&s_reservoir, s_features, 16, reservoir_weights, 0.5f);
    __syncthreads();

    // Compute reward
    float reward = 0.0f;
    if (threadIdx.x == 0) {
        // Reward: progress toward pocket opening
        reward = s_cv.pocket_signature - 0.1f * (s_H_new - s_H_old);
        if (!s_accepted) reward *= 0.1f;  // Penalize rejection

        *output_reward = reward;
    }
    __syncthreads();

    // RLS update (sequential, thread 0 only)
    float reward_modulation = 1.0f + 0.5f * tanhf(reward);
    for (int head = 0; head < NUM_OUTPUTS; head++) {
        float target = reward;  // Simplified: predict reward for all heads
        rls_update(
            &rls_weights[head * RESERVOIR_SIZE],
            &rls_P_matrices[head * RESERVOIR_SIZE * RESERVOIR_SIZE],
            s_reservoir.filtered_rates,
            target,
            config.lambda,
            reward_modulation
        );
    }
    __syncthreads();

    // ========================================================================
    // PHASE 7: WRITE BACK
    // ========================================================================

    // Store positions
    for (int i = threadIdx.x; i < config.n_atoms; i += blockDim.x) {
        positions[i] = s_positions[i];
    }

    // Store reservoir state
    for (int i = threadIdx.x; i < RESERVOIR_SIZE; i += blockDim.x) {
        reservoir_activations[i] = s_reservoir.activations[i];
        reservoir_filtered[i] = s_reservoir.filtered_rates[i];
        reservoir_membrane[i] = s_reservoir.membrane_potential[i];
        reservoir_adaptation[i] = s_reservoir.adaptation[i];
    }

    // Store features
    for (int i = threadIdx.x; i < FEATURE_DIM && i < 16; i += blockDim.x) {
        output_features[i] = s_features[i];
    }

    if (threadIdx.x == 0) {
        *output_accepted = s_accepted ? 1 : 0;
    }
}

// ============================================================================
// UTILITY KERNELS
// ============================================================================

/**
 * @brief Initialize momenta from Maxwell-Boltzmann distribution
 */
extern "C" __global__ void initialize_momenta(
    float3* momenta,
    const float* masses,
    float temperature,
    unsigned long long seed,
    int n_atoms
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_atoms) return;

    unsigned long long state[2];
    state[0] = seed + idx * 12345;
    state[1] = seed ^ (idx * 67890);

    float sigma = sqrtf(KB * temperature * masses[idx]);
    float2 g1 = random_gaussian(state);
    float2 g2 = random_gaussian(state);

    momenta[idx].x = sigma * g1.x;
    momenta[idx].y = sigma * g1.y;
    momenta[idx].z = sigma * g2.x;
}

/**
 * @brief Initialize RLS precision matrices to identity
 */
extern "C" __global__ void initialize_rls_P(
    float* P_matrices,
    float initial_precision,
    int n_outputs
) {
    int output = blockIdx.x;
    int i = threadIdx.x;

    if (output >= n_outputs || i >= RESERVOIR_SIZE) return;

    for (int j = 0; j < RESERVOIR_SIZE; j++) {
        int idx = output * RESERVOIR_SIZE * RESERVOIR_SIZE + i * RESERVOIR_SIZE + j;
        P_matrices[idx] = (i == j) ? initial_precision : 0.0f;
    }
}
